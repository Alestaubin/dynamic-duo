#!/usr/bin/env python3
"""
compare_calibrators.py
=======================
End-to-end comparison of JointProxyWeighted (filtered-proxy soft weighting,
Sections 2-5) against the fixed_ts / oracle_ts / coca_ts baselines, all on
the same sample subset / seed / adaptation mode so only the calibrator
differs. Pure Python driver (no SLURM/subprocess): runs every config in one
process and logs each to Weights & Biases under a shared `group`, so they
show up together in the W&B UI (Runs table -> Group by -> group) for
side-by-side comparison, plus a final summary run with one Table holding
every config's average metrics.

Each config reloads the large/small models FRESH from get_model() before
building its calibrator — both_duo mode permanently adapts BN/LN parameters
across corruptions within one evaluate_dynamic_duo call (only reset at each
corruption boundary, not after the last one), so reusing the same model
objects across configs without reloading would silently start config N+1
from wherever config N's adaptation left off, not from pretrained weights.

Run configs live in JSON files under cfgs/compare_runs/ (a list of dicts,
same shape as the old hardcoded RUN_CONFIGS) rather than in this file, so
new comparisons can be added without editing code — see --configs_file.
cfgs/compare_runs/default.json holds the identity-calibration configs;
cfgs/compare_runs/calibrated.json holds linear/isotonic variants (see
their docstring-equivalent comments in each file for why: an end-to-end run
with the identity configs found the gate for ac_mc/nuclear_norm NEVER once
favored the small model, even in corruptions where it was actually more
accurate — a systematic architecture-confidence-scale bias that Section 3
calibration exists to remove, and that sweep_proxies.py's plain sel_acc
metric couldn't see because the large model is better on average across
most corruptions. See sweep_proxies.py's module docstring and
bal_sel_acc/gap_bias metrics for the diagnostic, and cfgs/compare_runs/
calibrated.json for the fix under test).

Usage
-----
    python scripts/compare_calibrators.py --config cfgs/dynamic_duo_config.yaml \
        --num_samples 10000 --seed 0 --mode no_adapt

    # test whether calibration fixes the identity-config bias:
    python scripts/compare_calibrators.py --config cfgs/dynamic_duo_config.yaml \
        --configs_file cfgs/compare_runs/calibrated.json \
        --num_samples 10000 --seed 0 --mode no_adapt
    
    # Check with the oracle accuracy proxy (no source-data fitting needed) that the proxy-weighted approach works:
    python scripts/compare_calibrators.py --config cfgs/dynamic_duo_config_2.yaml \
        --configs_file cfgs/compare_runs/oracle.json \
        --num_samples 10000 --seed 0 --mode no_adapt

    # only re-run a subset while iterating:
    python scripts/compare_calibrators.py --config cfgs/dynamic_duo_config.yaml \\
        --only proxy_weighted_ac_mc coca_ts

    # cache large/small logits so repeat --mode no_adapt comparisons (e.g.
    # default.json then calibrated.json then oracle.json, same --num_samples
    # --seed) skip the model forward pass entirely after the first one:
    python scripts/compare_calibrators.py --config cfgs/dynamic_duo_config.yaml \\
        --configs_file cfgs/compare_runs/calibrated.json \\
        --num_samples 10000 --seed 0 --mode no_adapt \\
        --logits_cache_dir cache/compare_logits
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_coca import JointCoca
from src.reliability.setup import build_proxy_weighted_calibrator, fit_beta

DEFAULT_CONFIGS_FILE = "cfgs/compare_runs/default.json"


class _CachedModel(nn.Module):
    """Wraps a FROZEN model so forward() replays cached logits instead of
    recomputing them, once a cache file exists for the current (model_name,
    corruption, severity, num_samples, seed, batch_size). On a cache miss,
    forwards normally and records outputs to save once the stream ends.

    Only valid for --mode no_adapt: configure_model_frozen puts the model in
    train() (TENT uses batch statistics even when frozen — see tent.py), so
    BatchNorm outputs depend on batch COMPOSITION, not just which samples
    are used. The cache is safe because load_imagenetC's DataLoader shuffle
    is seeded deterministically: calling it again with the identical
    (corruption, severity, num_samples, seed, batch_size) reproduces the
    exact same batch sequence, so replaying logits in forward() call order
    lines up with the correct samples without needing to reimplement any
    sampling/shuffling logic here.

    Attribute lookups that miss on the wrapper (e.g. FeatureExtractor's
    search for a final Linear classifier, for the prototype proxy) fall
    through to the wrapped model transparently.
    """

    def __init__(self, model: nn.Module, cache_dir: str, model_name: str,
                 num_samples: int | None, seed: int | None, batch_size: int):
        super().__init__()
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.num_samples = num_samples
        self.seed = seed
        self.batch_size = batch_size
        self._cached: torch.Tensor | None = None
        self._pos = 0
        self._recording: list[torch.Tensor] = []
        self._current_path: Path | None = None

    def _path_for(self, corruption: str, severity: int) -> Path:
        key = f"{corruption}_{severity}_n{self.num_samples}_s{self.seed}_bs{self.batch_size}"
        return self.cache_dir / self.model_name / f"{key}.pt"

    def set_stream(self, corruption: str, severity: int) -> None:
        """Call once per (corruption, severity), before that stream's batches
        start arriving — flushes the previous stream's recording (if any)
        and loads/prepares the new stream's cache slot."""
        self._flush()
        self._current_path = self._path_for(corruption, severity)
        if self._current_path.exists():
            print(f"[logits cache] hit  {self.model_name}/{corruption}_{severity}")
            self._cached = torch.load(self._current_path, map_location="cpu", weights_only=True)
        else:
            print(f"[logits cache] miss {self.model_name}/{corruption}_{severity} — will cache this run")
            self._cached = None
        self._pos = 0

    def _flush(self) -> None:
        if self._cached is None and self._recording and self._current_path is not None:
            self._current_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.cat(self._recording), self._current_path)
            print(f"[logits cache] saved {self._current_path}")
        self._recording = []

    def finish(self) -> None:
        """Call after the last stream to flush any still-pending recording."""
        self._flush()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cached is not None:
            n = x.shape[0]
            out = self._cached[self._pos: self._pos + n].to(x.device)
            self._pos += n
            return out
        out = self.model(x)
        if self._current_path is not None:  # only record once inside a real eval stream
            self._recording.append(out.detach().cpu())
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _load_run_configs(path: str) -> list[dict]:
    with open(path) as f:
        run_configs = json.load(f)
    if not isinstance(run_configs, list) or not all(isinstance(c, dict) and "name" in c for c in run_configs):
        raise ValueError(f"{path} must contain a JSON list of objects, each with a 'name' key")
    return run_configs


# Curated columns for the console table (the wandb summary Table gets everything).
_SUMMARY_DISPLAY_COLS = [
    "duo/accuracy", "large/accuracy", "small/accuracy",
    "duo/nll", "duo/ece",
]


def _build_calibrator(
    run_cfg: dict, config: dict,
    large_model, large_preprocess, small_model, small_preprocess,
    device: torch.device, num_samples: int | None, seed: int | None,
    csv_path: str,
):
    mode = run_cfg["calibration_mode"]
    if mode == "fixed_ts":
        return JointFixedTS.load(run_cfg["fixed_ts_config"])
    if mode == "oracle_ts":
        return JointFixedTS()
    if mode == "coca":
        return JointCoca(num_steps=10, lr=5e-2, chunk_size=run_cfg.get("coca_bs"))
    if mode == "proxy_weighted":
        base_ts = JointFixedTS.load(run_cfg["fixed_ts_config"]) if run_cfg.get("fixed_ts_config") else None
        calibrator = build_proxy_weighted_calibrator(
            proxy_kind=run_cfg["proxy_kind"],
            proxy_cache=run_cfg.get("proxy_cache"),
            calib_map=run_cfg.get("calib_map"),
            calib_method=run_cfg.get("calib_method", "isotonic"),
            filter_kind=run_cfg.get("filter_kind", "none"),
            filter_kwargs=run_cfg.get("filter_kwargs"),
            beta=run_cfg.get("beta", 4.0),
            pool=run_cfg.get("pool", "linear"),
            prior_l=run_cfg.get("prior_l", 0.5),
            prior_s=run_cfg.get("prior_s", 0.5),
            base_ts=base_ts,
            csv_path=csv_path,
            config=config,
            large_model=large_model, large_preprocess=large_preprocess,
            small_model=small_model, small_preprocess=small_preprocess,
            device=device,
            num_samples=num_samples, seed=seed,
            proto_metric=run_cfg.get("proto_metric", "cosine"),
            proxy_batch_size=run_cfg.get("proxy_batch_size", 1),
        )
        if run_cfg.get("fit_beta"):
            fit_beta(
                calibrator, large_model, large_preprocess, small_model, small_preprocess,
                config, device, num_samples=num_samples, seed=seed,
            )
        return calibrator
    raise ValueError(f"Unknown calibration_mode '{mode}'")


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        print("No runs completed.")
        return
    print("\n" + "=" * 90)
    header = f"{'name':<28}" + "".join(f"{c.split('/')[-1]:>12}" for c in _SUMMARY_DISPLAY_COLS)
    print(header)
    print("-" * 90)
    for r in rows:
        line = f"{r['name']:<28}"
        for c in _SUMMARY_DISPLAY_COLS:
            v = r.get(c, float("nan"))
            line += f"{v:>12.4f}" if v == v else f"{'nan':>12}"
        print(line)


def _log_summary_to_wandb(rows: list[dict], project: str, group: str) -> None:
    if not rows:
        return
    run = wandb.init(project=project, group=group, name=f"{group}_summary", job_type="summary")
    cols = ["name"] + [c for c in rows[0].keys() if c != "name"]
    table = wandb.Table(columns=cols)
    for r in rows:
        table.add_data(*[r.get(c, float("nan")) for c in cols])
    run.log({"comparison_summary": table})
    run.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Run JointProxyWeighted against the fixed_ts/oracle_ts/coca_ts baselines "
                    "and log all of them, grouped, to Weights & Biases for easy comparison."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both_duo")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="dynamic-duos")
    parser.add_argument("--group", type=str, default=None,
                        help="Shared wandb group tag for all runs in this comparison. "
                             "Defaults to a timestamp so repeated invocations don't collide.")
    parser.add_argument("--out_dir", type=str, default="out/compare")
    parser.add_argument("--configs_file", type=str, default=DEFAULT_CONFIGS_FILE,
                        help="JSON file with a list of run configs (see cfgs/compare_runs/).")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Restrict to these run names (see --configs_file) instead of all of them.")
    parser.add_argument("--logits_cache_dir", type=str, default=None,
                        help="If given (and --mode no_adapt), cache each model's per-"
                             "corruption logits here so repeat comparisons on the same "
                             "duo/--num_samples/--seed skip the model forward pass "
                             "entirely after the first run_cfg populates the cache. "
                             "Ignored (with a warning) for any other --mode, where "
                             "logits are calibrator-dependent and can't be shared.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.logits_cache_dir and args.mode != "no_adapt":
        print(f"WARNING: --logits_cache_dir has no effect with --mode {args.mode!r} "
              f"(only 'no_adapt' has calibrator-independent logits to cache); ignoring.")

    # Duo identity is always prefixed onto the group, even a user-supplied
    # one — so grouping by `group` in the W&B UI can never accidentally mix
    # runs from two different duos (e.g. re-using the same --group label
    # "calibration_test" across a vit_b_16+resnet50 experiment and a later
    # efficientnet_b0+resnet50 one still yields two distinct group values).
    duo_tag = f"{config['LARGE']['NAME']}+{config['SMALL']['NAME']}"
    group = f"{duo_tag}__{args.group or datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Using device: {device}  |  wandb group: {group}")

    all_run_configs = _load_run_configs(args.configs_file)
    print(f"Loaded {len(all_run_configs)} run config(s) from {args.configs_file}")
    run_configs = all_run_configs
    if args.only is not None:
        wanted = set(args.only)
        run_configs = [c for c in all_run_configs if c["name"] in wanted]
        missing = wanted - {c["name"] for c in run_configs}
        if missing:
            parser.error(f"Unknown run name(s) in --only: {sorted(missing)}")

    summary_rows = []
    for run_cfg in run_configs:
        print(f"\n{'=' * 70}\n=== {run_cfg['name']} ===\n{'=' * 70}")

        # Fresh models every config: both_duo adaptation permanently mutates
        # BN/LN params across corruptions, so reusing model objects across
        # configs would leak config N's adapted state into config N+1.
        large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
        small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
        large_model = large_model.to(device)
        small_model = small_model.to(device)

        cached_large = cached_small = None
        if args.logits_cache_dir and args.mode == "no_adapt" and run_cfg.get("proxy_kind") == "prototype":
            print(f"[{run_cfg['name']}] proxy_kind=prototype needs a live forward pass every "
                  f"batch (penultimate features via a hook that a cache hit would never "
                  f"trigger) — skipping the logits cache for this config only.")
        elif args.logits_cache_dir and args.mode == "no_adapt":
            cached_large = _CachedModel(
                large_model, args.logits_cache_dir, config["LARGE"]["NAME"],
                args.num_samples, args.seed, config["BS"],
            )
            cached_small = _CachedModel(
                small_model, args.logits_cache_dir, config["SMALL"]["NAME"],
                args.num_samples, args.seed, config["BS"],
            )
            large_model, small_model = cached_large, cached_small

        csv_path = str(Path(args.out_dir) / run_cfg["name"])
        calibrator = _build_calibrator(
            run_cfg, config, large_model, large_preprocess, small_model, small_preprocess,
            device, args.num_samples, args.seed, csv_path,
        )

        duo = setup_duo(
            large=large_model, large_preprocess=large_preprocess,
            small=small_model, small_preprocess=small_preprocess,
            mode=args.mode, joint_calibrator=calibrator,
            calibration_mode=run_cfg["calibration_mode"],
            cfg=config, steps=args.steps,
        )

        def _on_corruption_start(corruption, severity):
            if cached_large is not None:
                cached_large.set_stream(corruption, severity)
                cached_small.set_stream(corruption, severity)

        results_rows = evaluate_dynamic_duo(
            duo, config, wandb_project=args.wandb_project,
            num_samples=args.num_samples, seed=args.seed,
            use_wandb=True, group=group, run_name=run_cfg["name"],
            on_corruption_start=_on_corruption_start,
        )
        if cached_large is not None:
            cached_large.finish()
            cached_small.finish()

        avg_row = next((r for r in results_rows if r.get("corruption") == "average"), None)
        if avg_row is not None:
            summary_rows.append({"name": run_cfg["name"], **avg_row})

    _print_summary(summary_rows)
    _log_summary_to_wandb(summary_rows, args.wandb_project, group)
    print(f"\nAll runs grouped under wandb group='{group}' in project '{args.wandb_project}'.")


if __name__ == "__main__":
    main()
