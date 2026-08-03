#!/usr/bin/env python3
"""
sweep_proxies.py
================
Sweeps combinations of (proxy_kind, calib_method, proxy_batch_size) and
compares them on the same metrics as test_proxies.py — selection accuracy
and score<->accuracy correlation — in one table.

Efficient by construction: the models are only run twice total (one forward
pass over the CALIBRATOR corruptions to collect dev data for fitting
calibration maps, one over the EVAL corruptions to collect the data the
sweep is actually scored on), never once per combination. Every proxy kind's
raw score, every calib_method's fit, and every proxy_batch_size's chunking
are then cheap in-memory recomputations reusing those two passes — a
100-combination sweep costs about the same as a single test_proxies.py run.

"identity" is one of --calib_methods rather than a separate code path (it's
already a legitimate rung of the Section-3 ladder, see
src.reliability.calibration.identity), so it's directly comparable to the
other methods as the "no calibration" reference row.

Why sel_acc alone is misleading (and what bal_sel_acc / gap_bias fix)
----------------------------------------------------------------------
Plain `sel_acc` (does sign(score_l - score_s) match sign(acc_l - acc_s)?)
looks good for a proxy that's simply, structurally biased toward whichever
model is better ON AVERAGE across the eval set, even if it never actually
tracks BATCH-level swaps in relative reliability — because that global
favorite is right most of the time anyway. Concretely: ac_mc/nuclear_norm
with identity calibration scored ~0.94-0.95 sel_acc in an earlier sweep, but
an end-to-end run showed their gate weight w_l NEVER dropped below 0.5 in
ANY corruption, including ones where the small model was actually more
accurate — the raw confidence scores of a ViT and a ResNet just live on
different scales, exactly the problem Section 3 calibration exists to fix,
and sel_acc's sign-only check can't see it because the larger model wins
often enough overall to mask it.

Two additions catch this:
  - bal_sel_acc: sel_acc computed separately on "large is actually better"
    and "small is actually better" chunks, then averaged (like balanced
    accuracy under class imbalance). A proxy that's just biased toward one
    model scores near 1.0 on its favorite's chunks and near 0.0 on the
    other's; a genuinely batch-reactive proxy scores well on both.
  - gap_bias / gap_corr: fit score_gap = slope*acc_gap + gap_bias (both
    SIGNED, continuous — not just their signs) across chunks. gap_bias is
    the score gap the calibration predicts when the two models are equally
    accurate — should be ~0; a large nonzero value is the same "always
    favors one model" signature caught more directly (in the units the
    Section-5 gate actually consumes) than bal_sel_acc's binary view.
    gap_corr is the Pearson correlation of the two signed gaps — whether
    the score's MAGNITUDE (not just sign) tracks the true relative
    advantage, which is what a smooth sigmoid gate needs, unlike sel_acc.

Usage
-----
    python scripts/sweep_proxies.py --config cfgs/dynamic_duo_config.yaml \
        --proxy_kinds nuclear_norm atc prototype ac_mc cot \
        --calib_methods identity linear isotonic \
        --proxy_batch_sizes 32 128 512 \
        --csv_path out/proxy_sweep.csv
"""

from __future__ import annotations

import argparse
import csv as csv_module
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.utils.data import load_config, load_imagenetC
from src.utils.model import get_model
from src.utils.stream_cache import duo_cache_dir, stream_key, collect_stream, load_or_collect_stream
from src.reliability.proxies.stats import FeatureExtractor, build_proxy_stats, ProxyStats
from src.reliability.calibration.maps import make_record, fit_calibration_maps, CalibrationMaps

_ALL_PROXY_KINDS = ["nuclear_norm", "atc", "prototype", "ac_mc", "cot"]
_ALL_CALIB_METHODS = ["identity", "linear", "platt", "beta", "isotonic"]
_SOURCE_FIT_KINDS = {"atc", "prototype", "cot"}

_TABLE_COLUMNS = [
    "proxy_kind", "calib_method", "proxy_batch_size", "n",
    "sel_acc", "sel_correct", "sel_total",
    "bal_sel_acc", "sel_acc_large_better", "sel_acc_small_better",
    "n_large_better", "n_small_better",
    "gap_bias", "gap_slope", "gap_corr",
    "l_r2", "l_pearson_r", "l_spearman_rho",
    "s_r2", "s_pearson_r", "s_spearman_rho",
]


def _corr_stats(xs: list[float], ys: list[float]) -> dict:
    """R^2, Pearson r, Spearman rho between xs and ys. nan when n < 3."""
    n = len(xs)
    nan_result = {"r2": float("nan"), "pearson_r": float("nan"), "spearman_rho": float("nan"), "n": n}
    if n < 3:
        return nan_result
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return nan_result
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    return {"r2": float(pr ** 2), "pearson_r": float(pr), "spearman_rho": float(sr), "n": n}


def _selection_accuracy(score_l, score_s, acc_l, acc_s) -> tuple[float, int, int]:
    """Fraction of non-tie chunks where sign(score_l - score_s) matches
    sign(acc_l - acc_s). Returns (accuracy, n_correct, n_total_non_tie).

    Misleading on its own when one model is better on AVERAGE across most
    chunks: a proxy that just always favors that model scores well here
    without ever tracking batch-level swaps — see _balanced_selection_accuracy
    and _gap_stats, which catch that failure mode (see module docstring).
    """
    correct = total = 0
    for sl, ss, al, as_ in zip(score_l, score_s, acc_l, acc_s):
        if abs(al - as_) < 1e-9:
            continue
        total += 1
        if (al > as_) == (sl > ss):
            correct += 1
    acc = correct / total if total > 0 else float("nan")
    return acc, correct, total


def _balanced_selection_accuracy(score_l, score_s, acc_l, acc_s) -> dict:
    """sel_acc computed separately on chunks where large is actually better
    vs where small is actually better, then averaged — like balanced
    accuracy under class imbalance. A proxy that's simply biased toward one
    model (rather than genuinely batch-reactive) scores near 1.0 on that
    model's chunks and near 0.0 on the other's, which the plain (pooled)
    sel_acc can't see if one regime dominates the eval set.
    """
    correct_l = total_l = 0
    correct_s = total_s = 0
    for sl, ss, al, as_ in zip(score_l, score_s, acc_l, acc_s):
        if abs(al - as_) < 1e-9:
            continue
        pred_large_better = sl > ss
        if al > as_:
            total_l += 1
            correct_l += int(pred_large_better)
        else:
            total_s += 1
            correct_s += int(not pred_large_better)
    acc_l_better = correct_l / total_l if total_l > 0 else float("nan")
    acc_s_better = correct_s / total_s if total_s > 0 else float("nan")
    balanced = (
        (acc_l_better + acc_s_better) / 2
        if total_l > 0 and total_s > 0 else float("nan")
    )
    return {
        "bal_sel_acc": balanced,
        "sel_acc_large_better": acc_l_better,
        "sel_acc_small_better": acc_s_better,
        "n_large_better": total_l,
        "n_small_better": total_s,
    }


def _gap_stats(cal_l, cal_s, acc_l, acc_s) -> dict:
    """Fit score_gap = slope*acc_gap + gap_bias across chunks (both signed,
    continuous, not just their signs) plus their Pearson correlation.

    gap_bias is the score gap the calibration predicts when the two models
    are EQUALLY accurate — should be ~0; a large nonzero value means the
    score structurally favors one model regardless of who's actually
    better, exactly the failure a smooth sigmoid gate is vulnerable to (it
    consumes the gap's magnitude, not sel_acc's sign-only view of it).
    gap_corr is how well the gap's MAGNITUDE tracks the true advantage.
    """
    nan_result = {"gap_bias": float("nan"), "gap_slope": float("nan"), "gap_corr": float("nan")}
    if len(cal_l) < 3:
        return nan_result
    score_gap = np.asarray(cal_l, dtype=np.float64) - np.asarray(cal_s, dtype=np.float64)
    acc_gap = np.asarray(acc_l, dtype=np.float64) - np.asarray(acc_s, dtype=np.float64)
    if acc_gap.std() < 1e-8 or score_gap.std() < 1e-8:
        return nan_result
    slope, bias = np.polyfit(acc_gap, score_gap, 1)
    corr = float(np.corrcoef(score_gap, acc_gap)[0, 1])
    return {"gap_bias": float(bias), "gap_slope": float(slope), "gap_corr": corr}


def _chunks(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _records_from_raw(cfg_l, cfg_s, z_l, z_s, f_l, f_s, labels, batch_size, corruption, severity):
    """Rebuild BatchRecords from cached raw (z, f, labels), one per original
    batch_size-sized chunk (matching the DataLoader batches make_record would
    have seen live — see _chunks). Always rescored fresh against the CURRENT
    cfg_l/cfg_s: a cached record must never be reused across runs that fit
    different proxies, or a newly-requested proxy_kind would be silently
    missing from a stale record's raw_l/raw_s dict."""
    n = z_l.shape[0]
    return [
        make_record(
            cfg_l, cfg_s,
            z_l[sl], z_s[sl], f_l[sl], f_s[sl], labels[sl],
            corruption, severity,
        )
        for sl in _chunks(n, batch_size)
    ]


def _fmt(v: float, w: int = 6, d: int = 3) -> str:
    return f"{v:{w}.{d}f}" if v == v else f"{'nan':>{w}}"


def main():
    parser = argparse.ArgumentParser(
        description="Sweep (proxy_kind, calib_method, proxy_batch_size) combinations "
                    "and compare them on selection accuracy and score<->accuracy correlation."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--proxy_kinds", type=str, nargs="+", default=_ALL_PROXY_KINDS,
                        choices=_ALL_PROXY_KINDS)
    parser.add_argument("--calib_methods", type=str, nargs="+", default=_ALL_CALIB_METHODS,
                        choices=_ALL_CALIB_METHODS)
    parser.add_argument("--proxy_batch_sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--proto_metric", type=str, default="cosine", choices=["cosine", "mahalanobis"])
    parser.add_argument("--calib_corruptions", type=str, nargs="+", default=None,
                        help="Defaults to the config's CALIBRATOR.CORRUPTIONS.")
    parser.add_argument("--calib_severities", type=int, nargs="+", default=None,
                        help="Defaults to the config's CALIBRATOR.SEVERITIES.")
    parser.add_argument("--eval_corruptions", type=str, nargs="+", default=None,
                        help="Defaults to the config's EVAL.CORRUPTIONS.")
    parser.add_argument("--eval_severities", type=int, nargs="+", default=None,
                        help="Defaults to the config's EVAL.SEVERITIES.")
    parser.add_argument("--calib_num_samples", type=int, default=None,
                        help="Cap on dev samples used to fit calibration maps.")
    parser.add_argument("--eval_num_samples", type=int, default=None,
                        help="Cap on samples per eval (corruption, severity) stream.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_cache", action="store_true",
                        help="Cache the (z_l, z_s, f_l, f_s, labels) collected for the "
                             "calibration stream and each eval (corruption, severity) stream "
                             "under cache/stream_cache/<large>+<small>/ (see src.utils."
                             "stream_cache — automatic, duo-specific, no directory to pick), "
                             "keyed by corruptions/severities/num_samples/seed — a repeat sweep "
                             "on the same duo/data skips both model-forward passes entirely. "
                             "Caches penultimate features too, so it's never skipped for "
                             "proxy_kind='prototype'. Proxy scoring itself always re-runs fresh "
                             "from the cached tensors (never cached), so a cache built with a "
                             "different --proxy_kinds selection is still safe to reuse.")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="With --use_cache, always recompute and overwrite any existing "
                             "cache entries instead of reusing them.")
    parser.add_argument("--sort_by", type=str, default="bal_sel_acc", choices=_TABLE_COLUMNS,
                        help="Defaults to bal_sel_acc rather than sel_acc, since sel_acc "
                             "alone can be fooled by a proxy that's just biased toward "
                             "whichever model is better on average (see module docstring).")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Where to write the full comparison table. Defaults to "
                             "out/proxy_sweep_<large>_<small>_<timestamp>.csv.")
    parser.add_argument("--log_wandb", action="store_true",
                        help="Log the full sweep as one wandb.Table (all rows, every "
                             "_TABLE_COLUMNS field) — wandb's table UI lets you click any "
                             "column header (bal_sel_acc, gap_corr, sel_acc, ...) to sort "
                             "interactively, so you aren't limited to --sort_by's one ranking "
                             "or the console's two fixed tables.")
    parser.add_argument("--wandb_project", type=str, default="proxy-weighted-duo-calibration")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Defaults to a timestamp. Always prefixed with the duo's model "
                             "names (see duo_tag below) so two duos' sweeps can never mix in "
                             "the same wandb group.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device).eval()
    small_model = small_model.to(device).eval()

    if args.csv_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.csv_path = f"out/proxy_sweep_{config['LARGE']['NAME']}_{config['SMALL']['NAME']}_{timestamp}.csv"

    # One source-fit pass covering every requested proxy kind at once (fit_source
    # is a no-op for stateless proxies) — never repeated per proxy_kind, and
    # cached to checkpoints/proxy_stats/<large>+<small>/ so re-running the
    # sweep (or a different sweep) against the same duo skips the source pass
    # entirely instead of re-fitting atc/prototype/cot from scratch every time.
    needs_source_fit = any(pk in _SOURCE_FIT_KINDS for pk in args.proxy_kinds)
    if needs_source_fit:
        duo_dir = Path("checkpoints/proxy_stats") / f"{config['LARGE']['NAME']}+{config['SMALL']['NAME']}"
        # proto_metric is baked into the cache name (not just the duo) since a
        # cached prototype proxy is metric-specific — a different --proto_metric
        # needs its own cache entry rather than silently reusing the wrong one.
        cache_name = f"proto_{args.proto_metric}"
        cache_file = duo_dir / f"{cache_name}.proxystats.pt"

        if cache_file.exists():
            print(
                f"\n{'=' * 70}\n"
                f"[proxy stats] LOADING CACHED source-fit stats for "
                f"{config['LARGE']['NAME']}+{config['SMALL']['NAME']} "
                f"(proto_metric={args.proto_metric})\n"
                f"  <- {cache_file}\n"
                f"{'=' * 70}\n"
            )
            src_loader = None  # unused on a cache hit — build_proxy_stats returns before touching it
        else:
            from torch.utils.data import DataLoader
            from torchvision import datasets
            from src.utils.data import _pil_collate_fn
            print(f"No cached source-fit stats at {cache_file} — fitting "
                  f"atc/prototype/cot from VAL_DIR (will cache for next time)...")
            src_ds = datasets.ImageFolder(config["VAL_DIR"])
            src_loader = DataLoader(
                src_ds, batch_size=config["BS"], shuffle=False,
                num_workers=config["WORKERS"], pin_memory=(device.type == "cuda"),
                collate_fn=_pil_collate_fn,
            )

        cfg_l, cfg_s = build_proxy_stats(
            large_model, large_preprocess, config["LARGE"]["NAME"],
            small_model, small_preprocess, config["SMALL"]["NAME"],
            src_loader, device, proto_metric=args.proto_metric,
            cache_path=cache_name, cache_dir=duo_dir,
        )
    else:
        cfg_l = ProxyStats(name=config["LARGE"]["NAME"], num_classes=1000)
        cfg_s = ProxyStats(name=config["SMALL"]["NAME"], num_classes=1000)

    calib_corruptions = args.calib_corruptions or config["CALIBRATOR"]["CORRUPTIONS"]
    calib_severities = args.calib_severities or config["CALIBRATOR"]["SEVERITIES"]
    eval_corruptions = args.eval_corruptions or config["EVAL"]["CORRUPTIONS"]
    eval_severities = args.eval_severities or config["EVAL"]["SEVERITIES"]

    ext_l = FeatureExtractor(large_model, cfg_l.name)
    ext_s = FeatureExtractor(small_model, cfg_s.name)

    stream_cache_dir = duo_cache_dir(config["LARGE"]["NAME"], config["SMALL"]["NAME"]) if args.use_cache else None

    try:
        # --- Phase B: one pass over calibration (dev-shift) data, fit every
        # (proxy_kind, calib_method) map from the SAME collected records. ---
        print(f"Collecting calibration records over {len(calib_corruptions)} corruptions x {len(calib_severities)} severities, over {args.calib_num_samples} samples each...")
        calib_tag = (
            "calib_" + "-".join(sorted(calib_corruptions)) +
            f"_sev{'-'.join(str(s) for s in sorted(calib_severities))}"
        )
        calib_key = stream_key(calib_tag, args.calib_num_samples, args.seed)

        def _collect_calib():
            calib_loader = load_imagenetC(
                config["TEST_DIR"], severities=calib_severities, corruption_types=calib_corruptions,
                device=device, batch_size=config["BS"], num_workers=config["WORKERS"],
                num_samples=args.calib_num_samples, seed=args.seed,
            )
            return collect_stream(calib_loader, large_preprocess, small_preprocess, ext_l, ext_s, device)

        z_l, z_s, f_l, f_s, labels = load_or_collect_stream(
            stream_cache_dir, calib_key, _collect_calib,
            use_cache=args.use_cache, overwrite_cache=args.overwrite_cache,
        )
        records = _records_from_raw(cfg_l, cfg_s, z_l, z_s, f_l, f_s, labels, config["BS"], "mixed", 0)
        print(f"Collected {len(records)} calibration records.")

        fitted_maps: dict[tuple[str, str], CalibrationMaps] = {}
        for pk in args.proxy_kinds:
            for cm in args.calib_methods:
                fitted_maps[(pk, cm)] = fit_calibration_maps(
                    records, pk, cfg_l.name, cfg_s.name, method=cm,
                )

        # --- Phase C: one pass over eval data per (corruption, severity),
        # cached at sample granularity so every proxy_batch_size can re-chunk
        # it in memory without re-running the models. ---
        eval_data: dict[tuple[str, int], tuple] = {}
        for severity in eval_severities:
            for corruption in eval_corruptions:
                eval_key = stream_key(f"{corruption}_s{severity}", args.eval_num_samples, args.seed)

                def _collect_eval(corruption=corruption, severity=severity):
                    loader = load_imagenetC(
                        config["TEST_DIR"], severities=severity, corruption_types=[corruption],
                        device=device, batch_size=config["BS"], num_workers=config["WORKERS"],
                        num_samples=args.eval_num_samples, seed=args.seed,
                    )
                    return collect_stream(loader, large_preprocess, small_preprocess, ext_l, ext_s, device)

                print(f"Collecting eval stream {corruption}/s{severity}...")
                eval_data[(corruption, severity)] = load_or_collect_stream(
                    stream_cache_dir, eval_key, _collect_eval,
                    use_cache=args.use_cache, overwrite_cache=args.overwrite_cache,
                )
    finally:
        ext_l.remove()
        ext_s.remove()

    # --- Sweep: cheap in-memory recomputation from the two cached passes. ---
    rows = []
    for pk in args.proxy_kinds:
        for pbs in args.proxy_batch_sizes:
            raw = []  # (r_l, r_s, acc_l, acc_s) per proxy-batch chunk, across all eval streams
            for (zl, zs, fl, fs, labels) in eval_data.values():
                n = zl.shape[0]
                for sl in _chunks(n, pbs):
                    z_l_c, z_s_c, f_l_c, f_s_c, labels_c = zl[sl], zs[sl], fl[sl], fs[sl], labels[sl]
                    r_l = cfg_l.score(pk, z_l_c, f_l_c)
                    r_s = cfg_s.score(pk, z_s_c, f_s_c)
                    acc_l = float((z_l_c.argmax(1) == labels_c).float().mean())
                    acc_s = float((z_s_c.argmax(1) == labels_c).float().mean())
                    raw.append((r_l, r_s, acc_l, acc_s))

            for cm in args.calib_methods:
                maps = fitted_maps[(pk, cm)]
                cal_l = [maps.predict_l(pk, r_l) for r_l, r_s, acc_l, acc_s in raw]
                cal_s = [maps.predict_s(pk, r_s) for r_l, r_s, acc_l, acc_s in raw]
                acc_l_list = [acc_l for r_l, r_s, acc_l, acc_s in raw]
                acc_s_list = [acc_s for r_l, r_s, acc_l, acc_s in raw]

                sel_acc, sel_correct, sel_total = _selection_accuracy(cal_l, cal_s, acc_l_list, acc_s_list)
                bal_stats = _balanced_selection_accuracy(cal_l, cal_s, acc_l_list, acc_s_list)
                gap_stats = _gap_stats(cal_l, cal_s, acc_l_list, acc_s_list)
                l_stats = _corr_stats(cal_l, acc_l_list)
                s_stats = _corr_stats(cal_s, acc_s_list)
                rows.append({
                    "proxy_kind": pk, "calib_method": cm, "proxy_batch_size": pbs, "n": len(raw),
                    "sel_acc": sel_acc, "sel_correct": sel_correct, "sel_total": sel_total,
                    **bal_stats,
                    **gap_stats,
                    "l_r2": l_stats["r2"], "l_pearson_r": l_stats["pearson_r"], "l_spearman_rho": l_stats["spearman_rho"],
                    "s_r2": s_stats["r2"], "s_pearson_r": s_stats["pearson_r"], "s_spearman_rho": s_stats["spearman_rho"],
                })

    rows.sort(key=lambda r: (r[args.sort_by] if r[args.sort_by] == r[args.sort_by] else -1), reverse=True)
    _print_table(rows, title=f"Ranked by {args.sort_by}")

    # gap_corr (does the signed, continuous score gap track the signed true
    # accuracy gap?) is the metric closest to what the Section-5 sigmoid gate
    # actually consumes — printed as its own ranking alongside --sort_by's,
    # since the two don't always agree (see the gap_bias gotcha in CLAUDE.md).
    if args.sort_by != "gap_corr":
        gap_corr_rows = sorted(
            rows, key=lambda r: (r["gap_corr"] if r["gap_corr"] == r["gap_corr"] else -1), reverse=True,
        )
        _print_table(gap_corr_rows, title="Ranked by gap_corr")

    if args.csv_path:
        path = Path(args.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=_TABLE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {path}")

    if args.log_wandb:
        import wandb
        duo_tag = f"{config['LARGE']['NAME']}+{config['SMALL']['NAME']}"
        group = f"{duo_tag}__{args.wandb_group or datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = wandb.init(
            project=args.wandb_project, group=group, name=f"proxy_sweep_{duo_tag}",
            job_type="proxy_sweep", tags=[config["LARGE"]["NAME"], config["SMALL"]["NAME"]],
        )
        table = wandb.Table(columns=_TABLE_COLUMNS)
        for r in rows:
            table.add_data(*[r[c] for c in _TABLE_COLUMNS])
        run.log({"proxy_sweep": table})
        run.finish()
        print(f"\nLogged {len(rows)} rows to wandb project '{args.wandb_project}' "
              f"(group='{group}') — click any column header in the table UI to sort by it.")


def _print_table(rows: list[dict], title: str | None = None) -> None:
    header = (f"{'proxy_kind':<12} {'calib_method':<10} {'pbs':>6} {'n':>5}  "
              f"{'sel_acc':>8}  {'bal_sel':>8} ({'L':>6}/{'S':>6}) ({'nL':>4}/{'nS':>4})  "
              f"{'gap_bias':>9} {'gap_corr':>9}   "
              f"{'l_R2':>6} {'l_r':>6} {'l_rho':>6}   {'s_R2':>6} {'s_r':>6} {'s_rho':>6}")
    if title:
        print(f"\n=== {title} ===")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['proxy_kind']:<12} {r['calib_method']:<10} {r['proxy_batch_size']:>6} {r['n']:>5}  "
            f"{_fmt(r['sel_acc'], 8)}  {_fmt(r['bal_sel_acc'], 8)} "
            f"({_fmt(r['sel_acc_large_better'], 6)}/{_fmt(r['sel_acc_small_better'], 6)}) "
            f"({r['n_large_better']:>4}/{r['n_small_better']:>4})  "
            f"{_fmt(r['gap_bias'], 9)} {_fmt(r['gap_corr'], 9)}   "
            f"{_fmt(r['l_r2'])} {_fmt(r['l_pearson_r'])} {_fmt(r['l_spearman_rho'])}   "
            f"{_fmt(r['s_r2'])} {_fmt(r['s_pearson_r'])} {_fmt(r['s_spearman_rho'])}"
        )
    print(
        "\nsel_acc = pooled selection accuracy (can be misleading, see module docstring).  "
        "bal_sel = balanced selection accuracy, averaged over (L=large-actually-better, "
        "S=small-actually-better) chunks — the more trustworthy target. (nL/nS) are the chunk "
        "COUNTS behind L/S — bal_sel is nan (not 0) when one side has zero chunks (e.g. a large "
        "proxy_batch_size can average away every chance the small model had to actually win a "
        "chunk), which is an honest 'no data' rather than a computed score of 0.  "
        "gap_bias = predicted score gap when models are equally accurate (want ~0; large "
        "nonzero = structurally favors one model).  gap_corr = correlation of the signed "
        "score gap with the signed true accuracy gap (want high — this is what the "
        "Section-5 sigmoid gate actually consumes)."
    )


if __name__ == "__main__":
    main()
