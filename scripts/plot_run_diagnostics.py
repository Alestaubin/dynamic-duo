#!/usr/bin/env python3
"""
scripts/plot_run_diagnostics.py
================================
Single-config exploratory run: pick a duo config, a set of corruptions, an
adaptation mode, and a calibration framework (fixed_ts or filtered-proxy
soft weighting), run it once end-to-end, and plot the resulting per-batch /
running diagnostics -- accuracy, NLL, entropy for large/small/duo, plus (for
calibration_mode=proxy_weighted) the raw proxy signal and gate weight over
time. Everything is also logged to Weights & Biases (on by default; --no_wandb
to disable): evaluate_dynamic_duo's own per-batch/per-corruption/summary
logging, PLUS this script's own additions (gate weight and raw proxy signal
merged onto the SAME step as each batch's accuracy/NLL/entropy, so they line
up on one x-axis in the wandb UI; the two PNG figures as Images; the full
proxy log as a Table) into one shared run -- see _make_wandb_run.

The calibrator (calibration_mode + all its knobs -- proxy_kind, calib_method,
filter_kind, proxy_batch_size, beta, ...) is specified via --calib_config, a
JSON file holding ONE run_cfg dict -- the exact same shape as one entry in
cfgs/compare_runs/*.json (compare_calibrators.py's --configs_file), so a
config already written for a comparison run works here unmodified, and vice
versa. See cfgs/calib_configs/ for ready-made ones. This keeps "which
calibrator, with which knobs" as a versioned, diffable file rather than a
long CLI invocation that's easy to fat-finger or forget to record.

This is deliberately a THIN driver: model/calibrator construction reuses
compare_calibrators._build_calibrator (same run_cfg dict shape as
cfgs/compare_runs/*.json) so this script can never drift from how a real
comparison run builds a calibrator, and per-batch accuracy/NLL/entropy are
read straight out of DynamicDuo._diag (the same accumulator run_duo already
maintains for wandb logging) rather than recomputed here.

Proxy diagnostics (r_l, r_s, a_l, a_s, x_l, x_s, w_l, and the per-proxy-batch
ground-truth acc_l/acc_s/duo_acc) come from JointProxyWeighted's own CSV
logging (csv_path=... at construction, see joint_proxy_weighted.py's
_CSV_FIELDS) -- this script just points that at out_dir and reads it back
for plotting/wandb, rather than re-deriving the same numbers a second way.

Usage
-----
    # fixed_ts baseline, no adaptation, 3 corruptions
    python scripts/plot_run_diagnostics.py --config cfgs/dynamic_duo_config.yaml \
        --calib_config cfgs/calib_configs/fixed_ts_default.json \
        --mode no_adapt --corruptions gaussian_noise fog brightness --severities 5 \
        --num_samples 10000

    # filtered-proxy soft weighting, both models adapting jointly
    python scripts/plot_run_diagnostics.py --config cfgs/dynamic_duo_config.yaml \
        --calib_config cfgs/calib_configs/nuclear_norm_identity_pbs128.json \
        --mode both_indep --corruptions gaussian_noise fog brightness --severities 5 \
        --num_samples 10000

    # same, but skip wandb entirely (quick local iteration)
    python scripts/plot_run_diagnostics.py --config cfgs/dynamic_duo_config.yaml \
        --calib_config cfgs/calib_configs/nuclear_norm_identity_pbs128.json \
        --no_wandb --num_samples 500
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import wandb

from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo, _MODES, _CALIB_MODES
from src.utils.data import load_config
from src.utils.model import get_model
from src.reliability.proxies.stats import PROXY_KINDS
from src.calibrators.joint_proxy_weighted import JointProxyWeighted
from src.utils.diagnostics_plots import (
    plot_batch_diagnostics, plot_proxy_diagnostics, plot_per_corruption_proxy_vs_accuracy,
)
from scripts.compare_calibrators import _build_calibrator

# Private JointProxyWeighted attributes holding the LATEST cached gate
# internals (refreshed at proxy-batch flushes, reused between them) -- no
# public accessor exists for these beyond last_w_l, so this script reads
# them the same way scripts/calibrate_gate_oracle.py reaches into
# calibrator._forward directly: an accepted pattern in this codebase for a
# driver script that needs diagnostic internals, not a public API surface.
_GATE_INTERNALS = [
    ("_cached_r_l", "gate/r_l"), ("_cached_r_s", "gate/r_s"),
    ("_cached_a_l", "gate/a_l"), ("_cached_a_s", "gate/a_s"),
    ("_cached_x_l", "gate/x_l"), ("_cached_x_s", "gate/x_s"),
]

_CALIB_METHODS = {"identity", "linear", "platt", "beta", "isotonic"}
_FILTER_KINDS = {"none", "running_mean", "ema", "kalman"}


def _load_calib_config(path: str) -> dict:
    """Load and validate a single run_cfg dict (calibration_mode + its knobs)
    from a JSON file -- the same shape as one entry in cfgs/compare_runs/*.json
    (see compare_calibrators._load_run_configs), just not wrapped in a list
    since this script only ever runs one config at a time."""
    with open(path) as f:
        run_cfg = json.load(f)
    if not isinstance(run_cfg, dict):
        raise ValueError(f"{path} must contain a single JSON object (a run_cfg dict), "
                          f"not a {type(run_cfg).__name__} -- see cfgs/calib_configs/ for examples.")
    if run_cfg.get("calibration_mode") not in _CALIB_MODES:
        raise ValueError(f"{path}: 'calibration_mode' must be one of {sorted(_CALIB_MODES)}, "
                          f"got {run_cfg.get('calibration_mode')!r}")
    if run_cfg["calibration_mode"] == "proxy_weighted":
        if run_cfg.get("proxy_kind") not in PROXY_KINDS:
            raise ValueError(f"{path}: 'proxy_kind' must be one of {sorted(PROXY_KINDS)}, "
                              f"got {run_cfg.get('proxy_kind')!r}")
        calib_method = run_cfg.get("calib_method", "identity")
        if calib_method not in _CALIB_METHODS:
            raise ValueError(f"{path}: 'calib_method' must be one of {sorted(_CALIB_METHODS)}, "
                              f"got {calib_method!r}")
        filter_kind = run_cfg.get("filter_kind", "none")
        if filter_kind not in _FILTER_KINDS:
            raise ValueError(f"{path}: 'filter_kind' must be one of {sorted(_FILTER_KINDS)}, "
                              f"got {filter_kind!r}")
    run_cfg.setdefault("name", Path(path).stem)
    return run_cfg


def _resolve_fixed_ts_config(path: str | None) -> str | None:
    """None if the checkpoint doesn't exist, so _build_calibrator's
    JointFixedTS.load(...) call is never handed a path it will raise on --
    prints a warning and falls back to T_l=T_s=1.0 instead."""
    if path is None:
        return None
    if not (Path(path) / "config.json").exists():
        print(f"WARNING: fixed_ts_config={path!r} not found; combining at T_l=T_s=1.0 "
              f"(or, for proxy_weighted, gating with no base_ts prior).")
        return None
    return path


def _default_calib_map(cfg: dict, proxy_kind: str, calib_method: str) -> str:
    return f"{cfg['LARGE']['NAME']}_{cfg['SMALL']['NAME']}_{proxy_kind}_{calib_method}"


def _wandb_config(args: argparse.Namespace, run_cfg: dict, cfg: dict) -> dict:
    config = {
        "mode": args.mode,
        "steps": args.steps,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "batch_size": cfg["BS"],
        "eval/corruptions": cfg["EVAL"]["CORRUPTIONS"],
        "eval/severities": cfg["EVAL"]["SEVERITIES"],
        "large/name": cfg["LARGE"]["NAME"],
        "small/name": cfg["SMALL"]["NAME"],
    }
    # Whatever the calib_config file actually declared, verbatim -- avoids
    # hand-maintaining a per-calibration_mode field list here that would
    # silently drift out of sync with cfgs/calib_configs/*.json.
    config.update({f"calib/{k}": v for k, v in run_cfg.items() if k != "name"})
    return config


def _make_wandb_run(args: argparse.Namespace, run_cfg: dict, cfg: dict, run_name: str):
    if not args.use_wandb:
        return None
    duo_tag = f"{cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}"
    group = f"{duo_tag}__{args.wandb_group}" if args.wandb_group else None
    return wandb.init(
        project=args.wandb_project, name=run_name, group=group,
        tags=[cfg["LARGE"]["NAME"], cfg["SMALL"]["NAME"], run_cfg["calibration_mode"], args.mode],
        config=_wandb_config(args, run_cfg, cfg),
    )


def _run(
    args: argparse.Namespace, run_cfg: dict, cfg: dict, out_dir: Path, run_name: str, wandb_run,
) -> tuple[list[dict], list[dict], list[dict], list]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  duo: {cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}  |  "
          f"calib_config: {run_cfg['name']!r} ({run_cfg['calibration_mode']})  |  out_dir: {out_dir}")

    large_model, large_preprocess = get_model(cfg["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(cfg["SMALL"]["NAME"])
    large_model, small_model = large_model.to(device), small_model.to(device)

    calibrator = _build_calibrator(
        run_cfg, cfg, large_model, large_preprocess, small_model, small_preprocess,
        device, args.num_samples, args.seed, csv_path=str(out_dir / "proxy_log"), verbose=True,
    )

    duo = setup_duo(
        large=large_model, large_preprocess=large_preprocess,
        small=small_model, small_preprocess=small_preprocess,
        mode=args.mode, joint_calibrator=calibrator, calibration_mode=run_cfg["calibration_mode"],
        cfg=cfg, steps=args.steps,
    )

    batch_records: list[dict] = []
    corruption_boundaries: list[dict] = []

    def _on_corruption_start(corruption, severity):
        corruption_boundaries.append({"idx": len(batch_records), "label": f"{corruption}/s{severity}"})

    def _on_batch(batch_idx, prefix, duo, outputs, z_large, z_small, labels):
        row = {"global_idx": len(batch_records), "corruption": prefix.rstrip("/")}
        for name in ("large", "small", "duo"):
            d = duo._diag[name]
            row[f"{name}_acc"] = d["acc_last"]
            row[f"{name}_nll"] = d["nll_last"]
            row[f"{name}_ent"] = d["ent_last"]
            row[f"{name}_acc_run"] = d["acc_sum"] / d["n"] if d["n"] > 0 else float("nan")
            row[f"{name}_nll_run"] = d["nll_sum"] / d["n"] if d["n"] > 0 else float("nan")
            row[f"{name}_ent_run"] = d["ent_sum"] / d["n"] if d["n"] > 0 else float("nan")
        w_l = getattr(duo.joint_calibrator, "last_w_l", None)
        if w_l is not None:
            row["w_l"] = w_l
        batch_records.append(row)

        if wandb_run is not None:
            # commit=False: buffered into the SAME step evaluate_dynamic_duo's
            # own (unconditional, every batch) wandb_run.log() call flushes
            # right after this returns -- so gate/proxy internals land on
            # exactly the same x-axis step as that batch's accuracy/NLL/
            # entropy, instead of splitting into two adjacent steps.
            log_dict = {}
            if w_l is not None:
                log_dict["gate/w_l"] = w_l
            if run_cfg["calibration_mode"] == "proxy_weighted":
                for attr, key in _GATE_INTERNALS:
                    val = getattr(duo.joint_calibrator, attr, None)
                    if val is not None:
                        log_dict[key] = val
            if log_dict:
                wandb_run.log(log_dict, commit=False)

    results_rows = evaluate_dynamic_duo(
        duo, cfg, num_samples=args.num_samples, seed=args.seed,
        wandb_run=wandb_run, run_name=run_name,
        on_corruption_start=_on_corruption_start, on_batch=_on_batch,
    )

    if batch_records:
        with (out_dir / "batch_diagnostics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(batch_records[0].keys()))
            writer.writeheader()
            writer.writerows(batch_records)
        print(f"Wrote {len(batch_records)} rows to {out_dir / 'batch_diagnostics.csv'}")

    proxy_rows: list[dict] = []
    proxy_csv_path = getattr(calibrator, "_csv_path", None)
    if proxy_csv_path is not None and Path(proxy_csv_path).exists():
        with Path(proxy_csv_path).open() as f:
            proxy_rows = list(csv.DictReader(f))

    return batch_records, corruption_boundaries, proxy_rows, results_rows


def _log_wandb_artifacts(
    wandb_run, out_dir: Path, has_batch_plot: bool, has_proxy_plot: bool,
    proxy_rows: list[dict], results_rows: list[dict],
) -> None:
    """Everything that can only be logged AFTER the run finishes and the PNGs
    exist: the figures as Images, the full proxy log as a Table, and the
    final per-corruption average as run.summary (for quick scanning in the
    wandb Runs table without opening the run)."""
    media = {}
    if has_batch_plot:
        media["plots/batch_diagnostics"] = wandb.Image(str(out_dir / "batch_diagnostics.png"))
    if has_proxy_plot:
        media["plots/proxy_diagnostics"] = wandb.Image(str(out_dir / "proxy_diagnostics.png"))
    if media:
        wandb_run.log(media)

    if proxy_rows:
        # proxy_rows came back from csv.DictReader -- every value is still a
        # str. "corruption" is the only genuinely textual column; cast
        # everything else back to int/float so the wandb Table's columns are
        # numeric (sortable, plottable) rather than opaque text.
        _INT_FIELDS = {"n_refreshes", "n"}
        table = wandb.Table(columns=JointProxyWeighted._CSV_FIELDS)
        for row in proxy_rows:
            values = []
            for c in JointProxyWeighted._CSV_FIELDS:
                if c == "corruption":
                    values.append(row[c])
                elif c in _INT_FIELDS:
                    values.append(int(row[c]))
                else:
                    values.append(float(row[c]))
            table.add_data(*values)
        wandb_run.log({"proxy_diagnostics_table": table})

    avg_row = next((r for r in results_rows if r.get("corruption") == "average"), None)
    if avg_row is not None:
        for k, v in avg_row.items():
            if isinstance(v, (int, float)):
                wandb_run.summary[k] = v


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default="cfgs/dynamic_duo_config.yaml",
                    help="Duo config -- picks the duo (LARGE/SMALL model names).")
    p.add_argument("--calib_config", type=str, required=True,
                    help="Path to a JSON file holding ONE run_cfg dict -- calibration_mode plus "
                         "all its knobs (proxy_kind, calib_method, filter_kind, proxy_batch_size, "
                         "beta, fixed_ts_config, ...). Same shape as one entry in "
                         "cfgs/compare_runs/*.json -- see cfgs/calib_configs/ for ready-made ones.")
    p.add_argument("--mode", type=str, default="no_adapt", choices=sorted(_MODES))
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--corruptions", type=str, nargs="+", default=None,
                    help="Overrides cfg['EVAL']['CORRUPTIONS']. Default: use the config's own list.")
    p.add_argument("--severities", type=int, nargs="+", default=None,
                    help="Overrides cfg['EVAL']['SEVERITIES'].")
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=None, help="Overrides cfg['BS'].")

    p.add_argument("--out_dir", type=str, default="out/run_diagnostics")
    p.add_argument("--run_name", type=str, default=None,
                    help="Subdirectory name under --out_dir, and the wandb run name. Default: "
                         "auto-generated from the calib_config name/mode/timestamp.")

    wandb_group_args = p.add_argument_group("wandb options")
    wandb_group_args.add_argument("--use_wandb", dest="use_wandb", action="store_true", default=True,
                                   help="Log everything to Weights & Biases (default: on).")
    wandb_group_args.add_argument("--no_wandb", dest="use_wandb", action="store_false",
                                   help="Disable wandb logging entirely (quick local iteration).")
    wandb_group_args.add_argument("--wandb_project", type=str, default="proxy-weighted-duo-calibration")
    wandb_group_args.add_argument("--wandb_group", type=str, default=None,
                                   help="Optional shared group tag (e.g. to cluster several manual "
                                        "invocations in the W&B UI). Always prefixed with the duo's "
                                        "model names. Default: ungrouped (a standalone run).")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.corruptions:
        cfg["EVAL"]["CORRUPTIONS"] = args.corruptions
    if args.severities:
        cfg["EVAL"]["SEVERITIES"] = args.severities
    if args.batch_size:
        cfg["BS"] = args.batch_size

    run_cfg = _load_calib_config(args.calib_config)
    if "fixed_ts_config" in run_cfg:
        run_cfg["fixed_ts_config"] = _resolve_fixed_ts_config(run_cfg["fixed_ts_config"])
    if (run_cfg["calibration_mode"] == "proxy_weighted"
            and run_cfg.get("calib_map") is None
            and run_cfg.get("calib_method", "identity") != "identity"):
        run_cfg["calib_map"] = _default_calib_map(cfg, run_cfg["proxy_kind"], run_cfg["calib_method"])
        print(f"No 'calib_map' in {args.calib_config} with calib_method={run_cfg['calib_method']!r}; "
              f"auto-naming it {run_cfg['calib_map']!r} (fit fresh if not already cached).")

    run_name = args.run_name or (
        f"{run_cfg['name']}__{args.mode}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _make_wandb_run(args, run_cfg, cfg, run_name)
    if wandb_run is not None:
        print(f"wandb run: {wandb_run.url}")

    batch_records, boundaries, proxy_rows, results_rows = _run(args, run_cfg, cfg, out_dir, run_name, wandb_run)

    has_batch_plot = False
    if batch_records:
        plot_batch_diagnostics(batch_records, boundaries, out_dir / "batch_diagnostics.png")
        has_batch_plot = True
    else:
        print("No batches were recorded -- nothing to plot.")

    has_proxy_plot = plot_proxy_diagnostics(proxy_rows, out_dir / "proxy_diagnostics.png")

    # One figure (+ aligned CSV) per corruption (saved locally only, not sent
    # to wandb): running-average accuracy for large/small/duo (bold, right
    # axis) with the raw proxy scores r_l/r_s (light, left axis) overlaid.
    per_corruption_dir = out_dir / "per_corruption"
    per_corruption_dir.mkdir(parents=True, exist_ok=True)
    plot_per_corruption_proxy_vs_accuracy(batch_records, proxy_rows, per_corruption_dir)

    if wandb_run is not None:
        _log_wandb_artifacts(wandb_run, out_dir, has_batch_plot, has_proxy_plot, proxy_rows, results_rows)
        wandb_run.finish()

    avg_row = next((r for r in results_rows if r.get("corruption") == "average"), None)
    if avg_row is not None:
        print(f"\nFinal average: duo={avg_row['duo/accuracy']:.4f}  "
              f"large={avg_row['large/accuracy']:.4f}  small={avg_row['small/accuracy']:.4f}")
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
