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

The proxy_weighted configs below are the top candidates from the proxy
sweep (scripts/sweep_proxies.py) on vit_b_16 + resnet50: ac_mc and
nuclear_norm both did BEST with no calibration at all (identity beat
linear/isotonic), while prototype needed calibration just to be usable
(identity was near coin-flip) and won there once calibrated. See
out/proxy_sweep.csv for the full comparison table.

Usage
-----
    python scripts/compare_calibrators.py --config cfgs/dynamic_duo_config.yaml \
        --num_samples 10000 --seed 0 --mode no_adapt

    # only re-run a subset while iterating:
    python scripts/compare_calibrators.py --config cfgs/dynamic_duo_config.yaml \\
        --only proxy_weighted_ac_mc coca_ts
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import wandb

from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_coca import JointCoca
from src.reliability.setup import build_proxy_weighted_calibrator, fit_beta

FIXED_TS_CKPT = "checkpoints/fixed_ts/default"  # T_l/T_s fit on the 4 CALIBRATOR corruptions

RUN_CONFIGS = [
    {
        "name": "fixed_ts",
        "calibration_mode": "fixed_ts",
        "fixed_ts_config": FIXED_TS_CKPT,
    },
    {
        "name": "oracle_ts",
        "calibration_mode": "oracle_ts",
    },
    {
        "name": "coca_ts",
        "calibration_mode": "coca",
        "coca_bs": 128,
    },
    {
        "name": "proxy_weighted_ac_mc",
        "calibration_mode": "proxy_weighted",
        "proxy_kind": "ac_mc",
        "calib_method": "identity",
        "proxy_batch_size": 512,
        "filter_kind": "ema",
        "filter_kwargs": {"alpha": 0.5},
        # "filter_kind": "kalman",
        # "filter_kwargs": {"q": 1e-3, "r": 1e-1},
        "pool": "linear",
        "fit_beta": True,
        "fixed_ts_config": FIXED_TS_CKPT,
    },
    {
        "name": "proxy_weighted_nuclear_norm",
        "calibration_mode": "proxy_weighted",
        "proxy_kind": "nuclear_norm",
        "calib_method": "identity",
        "proxy_batch_size": 512,
        # "filter_kind": "kalman",
        # "filter_kwargs": {"q": 1e-3, "r": 1e-1},
        "filter_kind": "ema",
        "filter_kwargs": {"alpha": 0.5},
        "pool": "linear",
        "fit_beta": True,
        "fixed_ts_config": FIXED_TS_CKPT,
    },
    # {
    #     "name": "proxy_weighted_prototype",
    #     "calibration_mode": "proxy_weighted",
    #     "proxy_kind": "prototype",
    #     "proto_metric": "cosine",
    #     "calib_map": "resnet50_vitb16_prototype_dev",
    #     "calib_method": "linear",
    #     "proxy_batch_size": 512,
    #     "filter_kind": "kalman",
    #     "filter_kwargs": {"q": 1e-3, "r": 1e-1},
    #     "pool": "linear",
    #     "fit_beta": True,
    #     "fixed_ts_config": FIXED_TS_CKPT,
    # },
]

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
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Restrict to these run names (see RUN_CONFIGS) instead of all of them.")
    args = parser.parse_args()

    group = args.group or f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  wandb group: {group}")

    config = load_config(args.config)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    run_configs = RUN_CONFIGS
    if args.only is not None:
        wanted = set(args.only)
        run_configs = [c for c in RUN_CONFIGS if c["name"] in wanted]
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
        results_rows = evaluate_dynamic_duo(
            duo, config, wandb_project=args.wandb_project,
            num_samples=args.num_samples, seed=args.seed,
            use_wandb=True, group=group, run_name=run_cfg["name"],
        )
        avg_row = next((r for r in results_rows if r.get("corruption") == "average"), None)
        if avg_row is not None:
            summary_rows.append({"name": run_cfg["name"], **avg_row})

    _print_summary(summary_rows)
    _log_summary_to_wandb(summary_rows, args.wandb_project, group)
    print(f"\nAll runs grouped under wandb group='{group}' in project '{args.wandb_project}'.")


if __name__ == "__main__":
    main()
