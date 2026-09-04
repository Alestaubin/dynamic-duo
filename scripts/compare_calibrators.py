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

    # cache large/small logits+features so repeat --mode no_adapt comparisons
    # (e.g. default.json then calibrated.json then oracle.json, same
    # --num_samples --seed) skip the model forward pass entirely after the
    # first one populates the cache. Directory is automatic (per duo — see
    # src.utils.stream_cache), no path to pick:
    python scripts/compare_calibrators.py --config cfgs/dynamic_duo_config.yaml \
        --configs_file cfgs/compare_runs/calibrated.json \
        --num_samples 10000 --seed 0 --mode no_adapt \
        --use_cache --verbose

    python scripts/compare_calibrators.py --config cfgs/dynamic_duo_config.yaml \
        --configs_file cfgs/compare_runs/calibrated.json \
        --num_samples 10000 --seed 0 --mode no_adapt \
        --use_cache --verbose


"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import calibration as cal
import torch
import torch.nn.functional as F
import wandb

from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.tta.tent import softmax_entropy
from src.utils.model import get_model
from src.utils.data import load_config, load_imagenetC
from src.utils.stream_cache import DuoStreamCache, duo_cache_dir
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_coca import JointCoca
from src.calibrators.joint_optimal_w_oracle import JointOptimalWOracle
from src.reliability.setup import build_proxy_weighted_calibrator, fit_beta
from scripts._cli import add_duo_config_arg, add_num_samples_arg, add_seed_arg, add_cache_toggle_args

DEFAULT_CONFIGS_FILE = "cfgs/compare_runs/default.json"


def _load_run_configs(path: str) -> list[dict]:
    with open(path) as f:
        run_configs = json.load(f)
    if not isinstance(run_configs, list) or not all(isinstance(c, dict) and "name" in c for c in run_configs):
        raise ValueError(f"{path} must contain a JSON list of objects, each with a 'name' key")
    return run_configs


# evaluate_dynamic_duo's results_rows now only carry accuracy/ece/nll/entropy
# per model (duo/large/small) — this is just the console table's column
# order/subset, the wandb summary Table (see _log_summary_to_wandb) logs
# every column that's there.
_SUMMARY_DISPLAY_COLS = [
    "duo/accuracy", "large/accuracy", "small/accuracy",
    "duo/ece", "duo/nll", "duo/entropy",
]


def _build_calibrator(
    run_cfg: dict, config: dict,
    large_model, large_preprocess, small_model, small_preprocess,
    device: torch.device, num_samples: int | None, seed: int | None,
    csv_path: str, verbose: bool = False,
):
    mode = run_cfg["calibration_mode"]
    if mode == "fixed_ts":
        return JointFixedTS.load(run_cfg["fixed_ts_config"])
    if mode == "oracle_ts":
        return JointFixedTS()
    if mode == "coca":
        return JointCoca(num_steps=10, lr=5e-2, chunk_size=run_cfg.get("coca_bs"))
    if mode == "optimal_w_oracle":
        base_ts = JointFixedTS.load(run_cfg["fixed_ts_config"]) if run_cfg.get("fixed_ts_config") else None
        return JointOptimalWOracle(
            base_ts=base_ts, proxy_batch_size=run_cfg.get("proxy_batch_size", 1), verbose=verbose,
        )
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
            # Stay quiet for fit_beta's own dev-corruption pass (many
            # batches, purely for the beta grid search) regardless of the
            # final verbose setting -- only respect it for the actual eval
            # loop below.
            calibrator.verbose = False
            fit_beta(
                calibrator, large_model, large_preprocess, small_model, small_preprocess,
                config, device, num_samples=num_samples, seed=seed,
            )
        calibrator.verbose = verbose
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
    add_duo_config_arg(parser, required=True)
    parser.add_argument("--mode", type=str, default="both_duo")
    parser.add_argument("--steps", type=int, default=1)
    add_num_samples_arg(parser)
    add_seed_arg(parser)
    parser.add_argument("--wandb_project", type=str, default="proxy-weighted-duo-calibration",
                        help="Dedicated W&B project for these filtered-proxy soft-weighting "
                             "comparisons, separate from other dynamic-duos experiments.")
    parser.add_argument("--group", type=str, default=None,
                        help="Shared wandb group tag for all runs in this comparison. "
                             "Defaults to a timestamp so repeated invocations don't collide.")
    parser.add_argument("--out_dir", type=str, default="out/compare")
    parser.add_argument("--configs_file", type=str, default=DEFAULT_CONFIGS_FILE,
                        help="JSON file with a list of run configs (see cfgs/compare_runs/).")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Restrict to these run names (see --configs_file) instead of all of them.")
    add_cache_toggle_args(parser, use_cache_help=(
        "If set (and --mode no_adapt), cache each (corruption, severity) stream's logits AND "
        "penultimate features (see src.utils.stream_cache) under an automatic, duo-specific "
        "directory, so repeat comparisons on the same duo/--num_samples/--seed skip the model "
        "forward pass entirely after the first run_cfg populates the cache — including "
        "proxy_kind='prototype', which needs the features too. Ignored (with a warning) for "
        "any other --mode, where logits are calibrator-dependent (adaptation mutates the "
        "models) and can't be shared across run_cfgs."
    ))
    parser.add_argument("--verbose", action="store_true",
                        help="Print one line per batch: the current run_cfg's calibrator "
                             "accuracy/NLL (already tracked in duo._diag, free) alongside a "
                             "fixed_ts reference (--fixed_ts_reference) recomputed fresh on "
                             "the SAME batch's z_large/z_small every time (a plain temperature "
                             "scale + combine — super cheap, no adaptation) — so you can watch "
                             "in real time whether the technique under test is actually beating "
                             "the simplest baseline batch by batch, not just in the final "
                             "corruption average.")
    parser.add_argument("--fixed_ts_reference", type=str, default="checkpoints/fixed_ts/default",
                        help="JointFixedTS checkpoint used ONLY for the --verbose per-batch "
                             "reference comparison — independent of whatever fixed_ts_config "
                             "(if any) a given run_cfg uses for its own calibrator, so every "
                             "run_cfg (including calibration_mode='fixed_ts' itself) is "
                             "compared against the SAME fixed baseline.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.use_cache and args.mode != "no_adapt":
        print(f"WARNING: --use_cache has no effect with --mode {args.mode!r} "
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

    # Loaded ONCE, reused as the fixed_ts reference (both --verbose's
    # per-batch lines and every run_cfg's per-corruption summary) for every
    # run_cfg — frozen (a plain temperature scale + combine, no state to
    # mutate), so unlike large_model/small_model it's safe to share. Tolerant
    # of a missing checkpoint (e.g. a duo other than the one --fixed_ts_
    # reference's default was fit for) — degrades to reporting the method's
    # own accuracy alone rather than hard-failing every run.
    try:
        ts_reference = JointFixedTS.load(args.fixed_ts_reference)
        for p in ts_reference.parameters():
            p.requires_grad_(False)
    except FileNotFoundError:
        print(f"WARNING: --fixed_ts_reference={args.fixed_ts_reference!r} not found; "
              f"skipping the fixed_ts comparison in --verbose and the per-corruption summary.")
        ts_reference = None

    summary_rows = []
    # One row per (run_cfg, corruption) with BOTH the method's own metrics
    # AND the fixed_ts reference's, side by side — logged to wandb instead
    # of summary_rows' per-config AVERAGE-only view, which can't show where
    # a method actually beats/loses to the cheap baseline. See
    # _on_corruption_end below (where each row is built) and the
    # per-run_cfg average rows appended after the loop.
    comparison_rows: list[dict] = []
    for run_cfg in run_configs:
        print(f"\n{'=' * 70}\n=== {run_cfg['name']} ===\n{'=' * 70}")

        # Fresh models every config: both_duo adaptation permanently mutates
        # BN/LN params across corruptions, so reusing model objects across
        # configs would leak config N's adapted state into config N+1.
        large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
        small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
        large_model = large_model.to(device)
        small_model = small_model.to(device)

        stream_cache_ctrl = None
        if args.use_cache and args.mode == "no_adapt":
            stream_cache_ctrl = DuoStreamCache(
                large_model, config["LARGE"]["NAME"], large_preprocess,
                small_model, config["SMALL"]["NAME"], small_preprocess,
                device=device,
                cache_dir=duo_cache_dir(config["LARGE"]["NAME"], config["SMALL"]["NAME"]),
                num_samples=args.num_samples, seed=args.seed,
                use_cache=True, overwrite_cache=args.overwrite_cache,
            )
            large_model, small_model = stream_cache_ctrl.large, stream_cache_ctrl.small

        csv_path = str(Path(args.out_dir) / run_cfg["name"])
        calibrator = _build_calibrator(
            run_cfg, config, large_model, large_preprocess, small_model, small_preprocess,
            device, args.num_samples, args.seed, csv_path, verbose=args.verbose,
        )

        duo = setup_duo(
            large=large_model, large_preprocess=large_preprocess,
            small=small_model, small_preprocess=small_preprocess,
            mode=args.mode, joint_calibrator=calibrator,
            calibration_mode=run_cfg["calibration_mode"],
            cfg=config, steps=args.steps,
        )

        if stream_cache_ctrl is not None:
            # register_hooks (proxy_kind="prototype") ran inside setup_duo, if
            # at all — feed its extractor(s) too, so a replayed (cached) batch
            # still gets valid features pushed into whatever the calibrator
            # itself reads from (see DuoStreamCache/CachedModel docstrings).
            stream_cache_ctrl.attach_extra_feature_extractors(
                getattr(calibrator, "_ext_l", None), getattr(calibrator, "_ext_s", None),
            )

        # Per-corruption running fixed_ts accuracy/NLL/ECE, reset at each
        # corruption boundary (_on_corruption_start) and flushed at the next
        # boundary (_on_corruption_end) — a plain dict rather than closure
        # variables reassigned with `nonlocal`, since both callbacks only
        # ever mutate it in place. probs/labels are accumulated in full
        # (not just a running sum) because ECE bins over the WHOLE
        # corruption's distribution — it can't be computed batch-by-batch
        # like a sum (same reason get_metrics_dict/calibrate_gate_oracle.py
        # only ever compute it once per full stream).
        fts_stats = {"correct": 0, "total": 0, "nll_sum": 0.0, "probs": [], "labels": []}

        def _on_corruption_start(corruption, severity):
            fts_stats["correct"] = 0
            fts_stats["total"] = 0
            fts_stats["nll_sum"] = 0.0
            fts_stats["probs"] = []
            fts_stats["labels"] = []
            if stream_cache_ctrl is not None:
                def _loader_factory(corruption=corruption, severity=severity):
                    return load_imagenetC(
                        config["TEST_DIR"], severities=severity, corruption_types=[corruption],
                        device=device, batch_size=config["BS"], num_workers=config["WORKERS"],
                        num_samples=args.num_samples, seed=args.seed,
                    )
                stream_cache_ctrl.set_stream(f"{corruption}_s{severity}", _loader_factory)

        def _on_batch(batch_idx, prefix, duo, outputs, z_large, z_small, labels):
            d = duo._diag["duo"]
            ref_acc = ref_nll = ref_ent = None
            if ts_reference is not None:
                with torch.no_grad():
                    z_ref = ts_reference.calibrate(z_large, z_small)
                    labels_ref = labels.to(z_ref.device)
                    correct = int((z_ref.argmax(1) == labels_ref).sum())
                    batch_nll_sum = float(F.cross_entropy(z_ref, labels_ref, reduction="sum"))
                fts_stats["correct"] += correct
                fts_stats["total"] += labels_ref.shape[0]
                fts_stats["nll_sum"] += batch_nll_sum
                fts_stats["probs"].append(F.softmax(z_ref, dim=1).detach().cpu())
                fts_stats["labels"].append(labels_ref.detach().cpu())
                if args.verbose:
                    ref_acc = correct / labels_ref.shape[0]
                    ref_nll = batch_nll_sum / labels_ref.shape[0]
                    ref_ent = float(softmax_entropy(z_ref).mean())
            if not args.verbose:
                return
            # last_w_l/last_nll only exist on calibrators that solve/gate a
            # per-batch mixing weight (JointOptimalWOracle, JointProxyWeighted
            # via its own console line) — getattr rather than isinstance so
            # this stays agnostic to which calibrator is under test.
            w_l = getattr(duo.joint_calibrator, "last_w_l", None)
            w_l_str = f" w_l={w_l:.4f}" if w_l is not None else ""
            ref_str = (
                f"  |  fixed_ts ref: acc={ref_acc:.4f} nll={ref_nll:.4f} ent={ref_ent:.4f}"
                f"  |  Δacc={d['acc_last'] - ref_acc:+.4f} Δnll={d['nll_last'] - ref_nll:+.4f}"
                if ref_acc is not None else ""
            )
            print(
                f"[{prefix}batch {batch_idx}] {run_cfg['name']}:{w_l_str} "
                f"acc={d['acc_last']:.4f} nll={d['nll_last']:.4f} ent={d['ent_last']:.4f}"
                f"{ref_str}"
            )

        def _on_corruption_end(corruption, severity, metrics_by_model):
            method_acc = metrics_by_model["duo"]["accuracy"]
            method_nll = metrics_by_model["duo"]["nll"]
            method_ece = metrics_by_model["duo"]["ece"]
            if fts_stats["total"] > 0:
                fts_acc = fts_stats["correct"] / fts_stats["total"]
                fts_nll = fts_stats["nll_sum"] / fts_stats["total"]
                fts_probs = torch.cat(fts_stats["probs"], dim=0).numpy()
                fts_labels = torch.cat(fts_stats["labels"], dim=0).numpy()
                fts_ece = cal.get_ece(fts_probs, fts_labels, num_bins=15)
                print(
                    f"[{corruption}/s{severity}] {run_cfg['name']} summary: "
                    f"acc={method_acc:.4f} nll={method_nll:.4f} ece={method_ece:.4f}  |  "
                    f"fixed_ts: acc={fts_acc:.4f} nll={fts_nll:.4f} ece={fts_ece:.4f}  |  "
                    f"Δacc={method_acc - fts_acc:+.4f} Δnll={method_nll - fts_nll:+.4f} "
                    f"Δece={method_ece - fts_ece:+.4f}"
                )
            else:
                fts_acc = fts_nll = fts_ece = float("nan")
                print(
                    f"[{corruption}/s{severity}] {run_cfg['name']} summary: "
                    f"acc={method_acc:.4f} nll={method_nll:.4f} ece={method_ece:.4f}  |  "
                    f"fixed_ts reference unavailable"
                )
            comparison_rows.append({
                "name": run_cfg["name"], "corruption": corruption, "severity": severity,
                "proxy_batch_size": getattr(calibrator, "proxy_batch_size", None),
                "adaptation_batch_size": config["BS"],
                "method_acc": method_acc, "method_nll": method_nll, "method_ece": method_ece,
                "fixed_ts_acc": fts_acc, "fixed_ts_nll": fts_nll, "fixed_ts_ece": fts_ece,
                "delta_acc": method_acc - fts_acc, "delta_nll": method_nll - fts_nll,
                "delta_ece": method_ece - fts_ece,
            })

        # pbs/abs baked into the run name too (not just evaluate_dynamic_duo's
        # config, which run_name= below overrides) -- visible in the wandb
        # Runs list with no clicking, matching the config fields it also sets.
        _pbs = getattr(calibrator, "proxy_batch_size", None)
        _pbs_str = f"__pbs{_pbs}_abs{config['BS']}" if _pbs is not None else ""
        results_rows = evaluate_dynamic_duo(
            duo, config, wandb_project=args.wandb_project,
            num_samples=args.num_samples, seed=args.seed,
            use_wandb=True, group=group, run_name=f"{run_cfg['name']}__{args.mode}{_pbs_str}",
            on_corruption_start=_on_corruption_start,
            # Always on now (not just --verbose): _on_batch needs to run every
            # batch to accumulate the per-corruption fixed_ts running total
            # that _on_corruption_end reports; --verbose only gates whether it
            # ALSO prints a line per batch (see _on_batch's early return).
            on_batch=_on_batch,
            on_corruption_end=_on_corruption_end,
        )
        if stream_cache_ctrl is not None:
            stream_cache_ctrl.finish()

        avg_row = next((r for r in results_rows if r.get("corruption") == "average"), None)
        if avg_row is not None:
            summary_rows.append({"name": run_cfg["name"], **avg_row})

        # One extra "average" row per run_cfg, macro-averaged over that
        # config's own per-corruption comparison_rows just appended above —
        # a quick overall glance in the SAME table as the per-corruption
        # breakdown, same convention as evaluate_dynamic_duo's own
        # per-corruption results_rows + trailing average row.
        cfg_rows = [r for r in comparison_rows if r["name"] == run_cfg["name"]]
        if cfg_rows:
            _numeric_cols = [
                "method_acc", "method_nll", "method_ece",
                "fixed_ts_acc", "fixed_ts_nll", "fixed_ts_ece",
                "delta_acc", "delta_nll", "delta_ece",
            ]
            comparison_rows.append({
                "name": run_cfg["name"], "corruption": "average", "severity": 0,
                # constant across every corruption for this run_cfg -- copied
                # from the first row rather than averaged.
                "proxy_batch_size": cfg_rows[0]["proxy_batch_size"],
                "adaptation_batch_size": cfg_rows[0]["adaptation_batch_size"],
                **{c: sum(r[c] for r in cfg_rows) / len(cfg_rows) for c in _numeric_cols},
            })

    _print_summary(summary_rows)
    _log_summary_to_wandb(comparison_rows, args.wandb_project, group)
    print(f"\nAll runs grouped under wandb group='{group}' in project '{args.wandb_project}'.")


if __name__ == "__main__":
    main()
