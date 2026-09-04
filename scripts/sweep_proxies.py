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
  - gap_bias / gap_slope / gap_corr / gap_spearman: fit
    score_gap = slope*acc_gap + gap_bias (both SIGNED, continuous — not
    just their signs) across chunks. gap_bias is the score gap the
    calibration predicts when the two models are equally accurate — should
    be ~0; a large nonzero value is the same "always favors one model"
    signature caught more directly (in the units the Section-5 gate
    actually consumes) than bal_sel_acc's binary view. gap_slope is the
    fitted sensitivity — how much score_gap moves per unit of acc_gap; near
    0 means the score barely reacts to real accuracy swings regardless of
    how consistent that (weak) reaction is. gap_corr is the Pearson
    correlation of the two signed gaps — whether the score's MAGNITUDE (not
    just sign) tracks the true relative advantage LINEARLY, which is what a
    smooth sigmoid gate needs, unlike sel_acc. gap_spearman is the same
    check but rank-based instead of linear, so it still credits a proxy
    that tracks the accuracy gap monotonically through a nonlinear
    calibration map even when gap_corr undersells it.

Per-corruption proxy rankings (held-out-shift proxy SELECTION, not just
scoring)
----------------------------------------------------------------------------
The sweep table above pools every eval corruption into ONE row per (proxy_
kind, calib_method, proxy_batch_size) — the right view for "how good is this
proxy on average", but useless for "if I picked this proxy using only ONE
held-out shift, would that choice have generalized to the others". A SEPARATE
table (_RANKING_TABLE_COLUMNS, logged as wandb's "proxy_rankings" and written
to --rankings_csv_path) answers that: the EXACT SAME columns and per-row
stats as the sweep table above (same _selection_accuracy/
_balanced_selection_accuracy/_gap_stats/_corr_stats calls), just NOT pooled
across streams first — one row per (proxy_kind, calib_method,
proxy_batch_size, corruption) instead of one row per (proxy_kind,
calib_method, proxy_batch_size). With E eval corruptions (+ "clean", see
below), this table has (E+1)x as many rows as the pooled one. Pivot it in
Excel (corruption as columns, bal_sel_acc as values, one proxy_kind/
calib_method/proxy_batch_size combo per row) and CORREL() any two corruption
columns: a high correlation means picking the best proxy on a held-out shift
transfers to the others; a low one means it doesn't, and per-corruption
re-sweeping (see the module docstring's transferability warning) is
unavoidable.

The "clean" corruption value scores every proxy against the UNCORRUPTED
ImageNet validation set (config['VAL_DIR']) — collected once, same as every
other stream, cached under --use_cache like the rest. Included because
clean-data access is usually free at deployment time (unlike a held-out
CORRUPTION, which assumes you already know something about the shift you'll
face) — if ranking proxies on clean data alone turns out to correlate well
with ranking them on real corruptions, that's a genuinely deployable
proxy-selection strategy; if it doesn't, that's worth knowing before relying
on it.

Adaptation mode (--mode)
------------------------
By default (--mode no_adapt) the models are frozen (configure_model_frozen:
train() mode for live batch statistics, no gradient updates — see
src.tta.tent) — this is what made the "collect once, resweep every combo"
design above safe: a frozen model's output is a pure function of its own
batch, never of call history.

--mode also accepts any of src.tta.dynamic_duo's other modes
(large_indep/small_indep/both_indep, large_duo/small_duo/both_duo). Two
different cases:
  - indep-signal modes (*_indep): each model adapts by minimizing entropy on
    its OWN logits — this does NOT depend on which proxy_kind/calib_method/
    proxy_batch_size is under test, so the adaptation trajectory (and hence
    the collected z_l/z_s/f_l/f_s stream) is identical across the whole
    sweep grid. The one-collection-pass-per-stream design is therefore still
    exactly safe and cheap here, just with TENT stepping turned on for
    whichever side(s) the mode adapts.
  - duo-signal modes (*_duo): real end-to-end adaptation is driven by the
    JOINT CALIBRATED output — i.e. by the exact (proxy_kind, calib_method,
    proxy_batch_size) combo under test, which is a genuine circular
    dependency (the sweep can't know which combo "wins" until it's already
    adapted the models with it). Rather than pay for one full model reload +
    adaptation run per combo (what scripts/compare_calibrators.py does, and
    the only fully faithful option), this script adapts the models ONCE per
    stream using a frozen --fixed_ts_reference calibrator (plain per-model
    temperature scaling, not proxy-weighted) to produce the entropy signal,
    then sweeps proxy_kind/calib_method/proxy_batch_size cheaply on top of
    that one adapted trajectory, same as every other mode. This measures
    "how good is this combo at combining logits from a duo that's already
    adapting under a generic reference calibrator" — NOT "how would the
    model actually adapt if this exact combo were driving it". If you need
    the fully faithful (and fully expensive) version, use
    compare_calibrators.py with calibration_mode="proxy_weighted" instead.

--steps mirrors DynamicDuo's steps (gradient steps per batch); default 1.
Cache keys (see --use_cache) always fold in --mode and --steps, since they
change what gets collected.

Usage
-----
    python scripts/sweep_proxies.py --config cfgs/dynamic_duo_config.yaml \
        --proxy_kinds nuclear_norm atc prototype ac_mc cot \
        --calib_methods identity linear isotonic \
        --proxy_batch_sizes 32 128 512 \
        --csv_path out/proxy_sweep.csv

    # sweep under both_duo adaptation (driven by a fixed reference
    # calibrator — see "Adaptation mode" above), instead of no_adapt:
    python scripts/sweep_proxies.py --config cfgs/dynamic_duo_config.yaml \
        --mode both_duo --steps 1 \
        --proxy_kinds nuclear_norm atc ac_mc \
        --calib_methods identity isotonic \
        --csv_path out/proxy_sweep_both_duo.csv
"""

from __future__ import annotations

import argparse
import csv as csv_module
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from src.utils.data import load_config, load_imagenetC
from src.utils.model import get_model
from src.utils.stream_cache import duo_cache_dir, stream_key, load_or_collect_stream
from src.reliability.proxies.stats import FeatureExtractor, build_proxy_stats, ProxyStats
from src.reliability.calibration.maps import make_record, fit_calibration_maps, CalibrationMaps
from src.tta.dynamic_duo import setup_duo
from src.calibrators.joint_fixed_TS import JointFixedTS
from scripts._cli import (
    add_duo_config_arg, add_seed_arg, add_cache_toggle_args,
    add_proto_metric_arg, add_wandb_project_group_args,
)

_ALL_PROXY_KINDS = ["nuclear_norm", "atc", "prototype", "ac_mc", "cot"]
_ALL_CALIB_METHODS = ["identity", "linear", "platt", "beta", "isotonic"]
_SOURCE_FIT_KINDS = {"atc", "prototype", "cot"}

_TABLE_COLUMNS = [
    "proxy_kind", "calib_method", "proxy_batch_size", "n",
    "sel_acc", "sel_correct", "sel_total",
    "bal_sel_acc", "sel_acc_large_better", "sel_acc_small_better",
    "n_large_better", "n_small_better",
    "gap_bias", "gap_slope", "gap_pearson", "gap_spearman",
    "l_r2", "l_pearson_r", "l_spearman_rho",
    "s_r2", "s_pearson_r", "s_spearman_rho",
]

# Same fields as _TABLE_COLUMNS (every stat computed exactly the same way,
# _selection_accuracy/_balanced_selection_accuracy/_gap_stats/_corr_stats),
# just NOT pooled across eval streams first — one row per (proxy_kind,
# calib_method, proxy_batch_size, corruption) instead of one row per
# (proxy_kind, calib_method, proxy_batch_size) pooling every corruption
# together. "corruption" is every eval corruption (as "<name>_s<severity>")
# plus "clean" — see module docstring's ranking-table section.
_RANKING_TABLE_COLUMNS = ["proxy_kind", "calib_method", "proxy_batch_size", "corruption"] + [
    c for c in _TABLE_COLUMNS if c not in ("proxy_kind", "calib_method", "proxy_batch_size")
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
    continuous, not just their signs) plus their Pearson and Spearman
    correlations.

    gap_bias is the score gap the calibration predicts when the two models
    are EQUALLY accurate — should be ~0; a large nonzero value means the
    score structurally favors one model regardless of who's actually
    better, exactly the failure a smooth sigmoid gate is vulnerable to (it
    consumes the gap's magnitude, not sel_acc's sign-only view of it).
    gap_slope is the fitted sensitivity of score_gap to acc_gap.
    gap_corr is how well the gap's MAGNITUDE tracks the true advantage
    LINEARLY; gap_spearman is the same but rank-based, so it still credits
    a monotonic-but-nonlinear relationship that gap_corr would undersell.
    """
    nan_result = {
        "gap_bias": float("nan"), "gap_slope": float("nan"),
        "gap_pearson": float("nan"), "gap_spearman": float("nan"),
    }
    if len(cal_l) < 3:
        return nan_result
    score_gap = np.asarray(cal_l, dtype=np.float64) - np.asarray(cal_s, dtype=np.float64)
    acc_gap = np.asarray(acc_l, dtype=np.float64) - np.asarray(acc_s, dtype=np.float64)
    if acc_gap.std() < 1e-8 or score_gap.std() < 1e-8:
        return nan_result
    slope, bias = np.polyfit(acc_gap, score_gap, 1)
    pearson_corr, _ = pearsonr(score_gap, acc_gap)
    spearman_corr, _ = spearmanr(score_gap, acc_gap)
    return {
        "gap_bias": float(bias), "gap_slope": float(slope),
        "gap_pearson": float(pearson_corr), "gap_spearman": float(spearman_corr),
    }


def _chunks(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def collect_stream_via_duo(duo, ext_l, ext_s, loader):
    """Like src.utils.stream_cache.collect_stream, but drives batches through
    a (possibly adapting) DynamicDuo instead of two frozen models directly —
    see the module docstring's "Adaptation mode" section.

    ext_l/ext_s are plain FeatureExtractor hooks registered on duo.large/
    duo.small purely for their forward-PRE-hook side effect: they capture
    the penultimate feature on every call, including the ones INSIDE
    duo.forward -> forward_and_adapt. They are never called directly here
    (ext_l(x) forces its own @torch.no_grad() forward pass — see
    FeatureExtractor.__call__ — which would both duplicate the forward pass
    and break the TENT gradient path duo.forward relies on).

    Caller must duo.reset() before this if the stream should start from the
    pre-adaptation model state (every call site below does).
    """
    zl_all, zs_all, fl_all, fs_all, labels_all = [], [], [], [], []
    for imgs, labels in tqdm(loader, desc="collecting", leave=False):
        outputs, z_large, z_small = duo(imgs, labels=None)
        zl_all.append(z_large.detach().cpu()); zs_all.append(z_small.detach().cpu())
        fl_all.append(ext_l._feats.detach().cpu()); fs_all.append(ext_s._feats.detach().cpu())
        labels_all.append(labels.cpu())
    return (torch.cat(zl_all), torch.cat(zs_all), torch.cat(fl_all), torch.cat(fs_all), torch.cat(labels_all))


def _records_from_raw(cfg_l, cfg_s, z_l, z_s, f_l, f_s, labels, batch_size, corruption, severity):
    """Rebuild BatchRecords from cached raw (z, f, labels), one per original
    batch_size-sized chunk (matching the DataLoader batches make_record would
    have seen live — see _chunks). Always rescored fresh against the CURRENT
    cfg_l/cfg_s: a cached record must never be reused across runs that fit
    different proxies, or a newly-requested proxy_kind would be silently
    missing from a stale record's raw_l/raw_s dict.

    This is the actually-expensive step, NOT the sklearn fits in
    fit_calibration_maps: make_record -> raw_proxies() scores EVERY fitted
    proxy on cfg_l/cfg_s (see ProxyStats.proxies = build_all()), regardless of
    which proxy_kinds the caller asked for — so if cot/atc/prototype were
    source-fitted, every chunk here pays cot's O(batch_size^3) Hungarian
    assignment and nuclear_norm's SVD. tqdm makes that visible instead of a
    silent multi-minute list comprehension.
    """
    n = z_l.shape[0]
    chunks = _chunks(n, batch_size)
    return [
        make_record(
            cfg_l, cfg_s,
            z_l[sl], z_s[sl], f_l[sl], f_s[sl], labels[sl],
            corruption, severity,
        )
        for sl in tqdm(chunks, desc=f"Scoring calibration records ({corruption})")
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Sweep (proxy_kind, calib_method, proxy_batch_size) combinations "
                    "and compare them on selection accuracy and score<->accuracy correlation."
    )
    add_duo_config_arg(parser, required=True)
    parser.add_argument("--proxy_kinds", type=str, nargs="+", default=_ALL_PROXY_KINDS,
                        choices=_ALL_PROXY_KINDS)
    parser.add_argument("--calib_methods", type=str, nargs="+", default=_ALL_CALIB_METHODS,
                        choices=_ALL_CALIB_METHODS)
    parser.add_argument("--proxy_batch_sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--mode", type=str, default="no_adapt",
                        help="Any src.tta.dynamic_duo mode: no_adapt (default, frozen "
                             "models, batch-norm stats), large_indep/small_indep/both_indep "
                             "(each model adapts on its own entropy — combo-independent, "
                             "still cheap), large_duo/small_duo/both_duo (adaptation driven "
                             "by --fixed_ts_reference instead of the combo under test — see "
                             "module docstring's 'Adaptation mode' section for why). Invalid "
                             "values raise from setup_duo's own assertion.")
    parser.add_argument("--steps", type=int, default=1,
                        help="Gradient steps per batch, passed straight to setup_duo "
                             "(DynamicDuo.steps). No effect when --mode no_adapt.")
    parser.add_argument("--fixed_ts_reference", type=str, default="checkpoints/fixed_ts/default",
                        help="JointFixedTS checkpoint used as the joint_calibrator that drives "
                             "TENT adaptation for *_duo modes (frozen — its own temperatures "
                             "are never adapted). Ignored substantively by no_adapt/*_indep "
                             "modes (no adaptation depends on the joint-calibrated output "
                             "there), but still loaded since DynamicDuo/setup_duo always "
                             "require a joint_calibrator.")
    add_proto_metric_arg(parser)
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
    add_seed_arg(parser, default=None)
    add_cache_toggle_args(parser, use_cache_help=(
        "Cache the (z_l, z_s, f_l, f_s, labels) collected for the calibration stream and each "
        "eval (corruption, severity) stream under cache/stream_cache/<large>+<small>/ (see "
        "src.utils.stream_cache — automatic, duo-specific, no directory to pick), keyed by "
        "corruptions/severities/num_samples/seed/--mode/--steps — a repeat sweep on the same "
        "duo/data/mode skips both model-forward passes entirely. Caches penultimate features "
        "too, so it's never skipped for proxy_kind='prototype'. Proxy scoring itself always "
        "re-runs fresh from the cached tensors (never cached), so a cache built with a "
        "different --proxy_kinds selection is still safe to reuse."
    ))
    parser.add_argument("--sort_by", type=str, default="bal_sel_acc", choices=_TABLE_COLUMNS,
                        help="Defaults to bal_sel_acc rather than sel_acc, since sel_acc "
                             "alone can be fooled by a proxy that's just biased toward "
                             "whichever model is better on average (see module docstring).")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Where to write the full comparison table. Defaults to "
                             "out/proxy_sweep_<large>_<small>_<timestamp>.csv.")
    parser.add_argument("--rankings_csv_path", type=str, default=None,
                        help="Where to write the per-corruption proxy-ranking table (see "
                             "module docstring) — the same columns/stats as --csv_path's sweep "
                             "table plus a 'corruption' column, one row per (proxy_kind, "
                             "calib_method, proxy_batch_size, corruption) instead of pooling "
                             "every corruption into one row — (E+1)x as many rows as the sweep "
                             "table for E eval corruptions (+1 for 'clean'). Pivot in Excel "
                             "(corruption as columns) and CORREL() any two to check whether a "
                             "held-out shift (or clean data) would have picked the same proxy "
                             "as the real corruptions. Defaults to --csv_path with '_rankings' "
                             "inserted before the extension.")
    parser.add_argument("--log_wandb", action="store_true",
                        help="Log the full sweep as one wandb.Table (all rows, every "
                             "_TABLE_COLUMNS field) — wandb's table UI lets you click any "
                             "column header (bal_sel_acc, gap_corr, sel_acc, ...) to sort "
                             "interactively, so you aren't limited to --sort_by's one ranking. "
                             "Also logs the per-corruption ranking table (see "
                             "--rankings_csv_path) as a second wandb.Table, 'proxy_rankings'.")
    add_wandb_project_group_args(
        parser,
        group_help="Defaults to a timestamp. Always prefixed with the duo's model names (see "
                    "duo_tag below) so two duos' sweeps can never mix in the same wandb group.",
    )
    parser.add_argument("--device", type=str, default=None,
                        help="torch device for model loading, source-fitting, AND the Sweep")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device).eval()
    small_model = small_model.to(device).eval()

    if args.csv_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.csv_path = f"out/proxy_sweep_{config['LARGE']['NAME']}_{config['SMALL']['NAME']}_{timestamp}.csv"
    if args.rankings_csv_path is None:
        p = Path(args.csv_path)
        args.rankings_csv_path = str(p.with_name(f"{p.stem}_rankings{p.suffix}"))

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

    # ref_calibrator only actually drives adaptation for *_duo modes (see
    # module docstring's "Adaptation mode" section) — for no_adapt/*_indep it
    # is loaded purely because DynamicDuo/setup_duo require a
    # joint_calibrator, and is never consulted for those modes' loss.
    # calibration_mode="fixed_ts" freezes it (setup_duo/DynamicDuo never
    # tune it), so it's safe regardless of --mode.
    ref_calibrator = JointFixedTS.load(args.fixed_ts_reference)
    for p in ref_calibrator.parameters():
        p.requires_grad_(False)
    # Built ONCE and reused for every stream below (calib/eval/clean) —
    # duo.reset() (called at the top of each _collect_* closure) restores the
    # pre-adaptation state each time, matching evaluate_dynamic_duo's own
    # per-corruption reset. Source-fitting above ran BEFORE this on the
    # plain .eval() models; setup_duo now reconfigures them (train() mode,
    # live batch stats, TENT params where the mode adapts) for collection.
    duo = setup_duo(
        large=large_model, large_preprocess=large_preprocess,
        small=small_model, small_preprocess=small_preprocess,
        joint_calibrator=ref_calibrator, calibration_mode="fixed_ts",
        mode=args.mode, cfg=config, steps=args.steps,
    )

    ext_l = FeatureExtractor(large_model, cfg_l.name)
    ext_s = FeatureExtractor(small_model, cfg_s.name)

    stream_cache_dir = duo_cache_dir(config["LARGE"]["NAME"], config["SMALL"]["NAME"]) if args.use_cache else None
    # --mode/--steps folded into every stream's cache tag below: a frozen
    # no_adapt stream and a both_duo-adapted stream for the same corruption
    # are entirely different logits and must never collide in the cache.
    mode_tag = f"mode{args.mode}_steps{args.steps}"

    try:
        # --- Phase B: one pass over calibration (dev-shift) data, fit every
        # (proxy_kind, calib_method) map from the SAME collected records. ---
        print(f"Collecting calibration records over {len(calib_corruptions)} corruptions x {len(calib_severities)} severities, over {args.calib_num_samples} samples each...")
        calib_tag = (
            f"{mode_tag}_calib_" + "-".join(sorted(calib_corruptions)) +
            f"_sev{'-'.join(str(s) for s in sorted(calib_severities))}"
        )
        calib_key = stream_key(calib_tag, args.calib_num_samples, args.seed)

        def _collect_calib():
            duo.reset()
            calib_loader = load_imagenetC(
                config["TEST_DIR"], severities=calib_severities, corruption_types=calib_corruptions,
                device=device, batch_size=config["BS"], num_workers=config["WORKERS"],
                num_samples=args.calib_num_samples, seed=args.seed,
            )
            return collect_stream_via_duo(duo, ext_l, ext_s, calib_loader)

        z_l, z_s, f_l, f_s, labels = load_or_collect_stream(
            stream_cache_dir, calib_key, _collect_calib,
            use_cache=args.use_cache, overwrite_cache=args.overwrite_cache,
        )
        # load_or_collect_stream always hands back CPU tensors regardless of
        # --device (see the eval_data move below) — move once here too, so
        # _records_from_raw's raw_proxies() calls (nuclear_norm SVD,
        # prototype matmuls) run on-device instead of CPU.
        if device.type != "cpu":
            z_l, z_s, f_l, f_s, labels = (t.to(device) for t in (z_l, z_s, f_l, f_s, labels))
        t0 = time.time()
        records = _records_from_raw(cfg_l, cfg_s, z_l, z_s, f_l, f_s, labels, config["BS"], "mixed", 0)
        print(f"Collected {len(records)} calibration records in {time.time() - t0:.1f}s.")

        fitted_maps: dict[tuple[str, str], CalibrationMaps] = {}
        map_combos = [(pk, cm) for pk in args.proxy_kinds for cm in args.calib_methods]
        t0 = time.time()
        for pk, cm in tqdm(map_combos, desc="Fitting calibration maps (proxy_kind, calib_method)"):
            fitted_maps[(pk, cm)] = fit_calibration_maps(
                records, pk, cfg_l.name, cfg_s.name, method=cm,
            )
        print(f"Fitted {len(map_combos)} calibration maps in {time.time() - t0:.1f}s.")

        # --- Phase C: one pass over eval data per (corruption, severity),
        # cached at sample granularity so every proxy_batch_size can re-chunk
        # it in memory without re-running the models. ---
        eval_data: dict[tuple[str, int], tuple] = {}
        for severity in eval_severities:
            for corruption in eval_corruptions:
                eval_key = stream_key(f"{mode_tag}_{corruption}_s{severity}", args.eval_num_samples, args.seed)

                def _collect_eval(corruption=corruption, severity=severity):
                    duo.reset()
                    loader = load_imagenetC(
                        config["TEST_DIR"], severities=severity, corruption_types=[corruption],
                        device=device, batch_size=config["BS"], num_workers=config["WORKERS"],
                        num_samples=args.eval_num_samples, seed=args.seed,
                    )
                    return collect_stream_via_duo(duo, ext_l, ext_s, loader)

                print(f"Collecting eval stream {corruption}/s{severity}...")
                eval_data[(corruption, severity)] = load_or_collect_stream(
                    stream_cache_dir, eval_key, _collect_eval,
                    use_cache=args.use_cache, overwrite_cache=args.overwrite_cache,
                )

        # --- Phase C.5: the UNCORRUPTED validation set, for the ranking
        # table's "clean" column (see module docstring) -- clean-data access
        # is usually free at deployment time, unlike a held-out corruption,
        # so this checks whether ranking proxies on clean data alone would
        # have picked the same proxy as the real corruptions. Collected and
        # cached exactly like every other stream (plain ImageFolder over
        # VAL_DIR, same pattern as the source-fit pass above).
        clean_key = stream_key(f"{mode_tag}_clean_val", args.eval_num_samples, args.seed)

        def _collect_clean():
            from torch.utils.data import DataLoader, Subset
            from torchvision import datasets
            from src.utils.data import _pil_collate_fn

            duo.reset()
            clean_ds = datasets.ImageFolder(config["VAL_DIR"])
            if args.eval_num_samples is not None:
                n = min(args.eval_num_samples, len(clean_ds))
                gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
                indices = torch.randperm(len(clean_ds), generator=gen)[:n].tolist()
                clean_ds = Subset(clean_ds, indices)
            clean_loader = DataLoader(
                clean_ds, batch_size=config["BS"], shuffle=False,
                num_workers=config["WORKERS"], pin_memory=(device.type == "cuda"),
                collate_fn=_pil_collate_fn,
            )
            return collect_stream_via_duo(duo, ext_l, ext_s, clean_loader)

        print("Collecting clean validation stream...")
        clean_stream = load_or_collect_stream(
            stream_cache_dir, clean_key, _collect_clean,
            use_cache=args.use_cache, overwrite_cache=args.overwrite_cache,
        )
    finally:
        ext_l.remove()
        ext_s.remove()

    # collect_stream_via_duo/load_or_collect_stream always hand back CPU
    # tensors (see src/utils/stream_cache.py — cached to disk that way regardless of
    # --device), so move each stream onto the target device ONCE here rather
    # than per chunk inside the (proxy_kind, proxy_batch_size) sweep below —
    # every score() call in the Sweep phase then runs on-device for free via
    # plain slicing, no repeated host<->device transfer per chunk per combo.
    if device.type != "cpu":
        eval_data = {
            key: tuple(t.to(device) for t in tensors)
            for key, tensors in eval_data.items()
        }
        clean_stream = tuple(t.to(device) for t in clean_stream)

    # Named (not tuple-keyed) view of every stream the RANKING table scores
    # against: every eval corruption plus "clean" — see module docstring.
    # Kept separate from eval_data/the pooled Sweep below, which stays
    # corruption-only and unchanged, so this is purely additive.
    ranking_streams: dict[str, tuple] = {
        f"{corruption}_s{severity}": eval_data[(corruption, severity)]
        for (corruption, severity) in eval_data
    }
    ranking_streams["clean"] = clean_stream
    ranking_stream_names = list(ranking_streams.keys())

    # --- Sweep: cheap in-memory recomputation from the two cached passes,
    # EXCEPT proxy_kind='cot' (Hungarian assignment, O(proxy_batch_size^3)
    # per chunk, scipy on CPU) and 'nuclear_norm' (SVD per chunk, also CPU),
    # whose per-chunk score() cost grows sharply with proxy_batch_size — a
    # tqdm + per-combo timing over (proxy_kind, proxy_batch_size), not over
    # every chunk, keeps this visible without flooding the log.
    #
    # raw scores are computed ONCE per stream (raw_by_stream), then reused
    # for BOTH the pooled row (rows, corruption-only — "clean" excluded so
    # the existing pooled semantics are unchanged) and the per-stream
    # ranking row (ranking_rows, includes "clean") — no duplicate scoring. ---
    rows = []
    ranking_rows = []
    combos = [(pk, pbs) for pk in args.proxy_kinds for pbs in args.proxy_batch_sizes]
    for pk, pbs in tqdm(combos, desc="Sweep (proxy_kind, proxy_batch_size)"):
        t0 = time.time()
        raw_by_stream: dict[str, list] = {}
        for stream_name, (zl, zs, fl, fs, labels) in ranking_streams.items():
            stream_raw = []
            n = zl.shape[0]
            for sl in _chunks(n, pbs):
                z_l_c, z_s_c, f_l_c, f_s_c, labels_c = zl[sl], zs[sl], fl[sl], fs[sl], labels[sl]
                r_l = cfg_l.score(pk, z_l_c, f_l_c)
                r_s = cfg_s.score(pk, z_s_c, f_s_c)
                acc_l = float((z_l_c.argmax(1) == labels_c).float().mean())
                acc_s = float((z_s_c.argmax(1) == labels_c).float().mean())
                stream_raw.append((r_l, r_s, acc_l, acc_s))
            raw_by_stream[stream_name] = stream_raw
        raw = [item for name, items in raw_by_stream.items() if name != "clean" for item in items]
        print(f"  [{pk}, pbs={pbs}] raw scoring over {len(raw)} chunks "
              f"(+ clean, {len(raw_by_stream['clean'])} chunks) took {time.time() - t0:.1f}s")

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

            # Same stats as the pooled row above, computed separately per
            # stream (not pooled) — one row per (pk, cm, pbs, corruption),
            # matching _TABLE_COLUMNS' schema exactly plus "corruption".
            for stream_name in ranking_stream_names:
                stream_raw = raw_by_stream[stream_name]
                s_cal_l = [maps.predict_l(pk, r_l) for r_l, r_s, acc_l, acc_s in stream_raw]
                s_cal_s = [maps.predict_s(pk, r_s) for r_l, r_s, acc_l, acc_s in stream_raw]
                s_acc_l = [acc_l for r_l, r_s, acc_l, acc_s in stream_raw]
                s_acc_s = [acc_s for r_l, r_s, acc_l, acc_s in stream_raw]

                s_sel_acc, s_sel_correct, s_sel_total = _selection_accuracy(s_cal_l, s_cal_s, s_acc_l, s_acc_s)
                s_bal_stats = _balanced_selection_accuracy(s_cal_l, s_cal_s, s_acc_l, s_acc_s)
                s_gap_stats = _gap_stats(s_cal_l, s_cal_s, s_acc_l, s_acc_s)
                s_l_stats = _corr_stats(s_cal_l, s_acc_l)
                s_s_stats = _corr_stats(s_cal_s, s_acc_s)
                ranking_rows.append({
                    "proxy_kind": pk, "calib_method": cm, "proxy_batch_size": pbs,
                    "corruption": stream_name, "n": len(stream_raw),
                    "sel_acc": s_sel_acc, "sel_correct": s_sel_correct, "sel_total": s_sel_total,
                    **s_bal_stats,
                    **s_gap_stats,
                    "l_r2": s_l_stats["r2"], "l_pearson_r": s_l_stats["pearson_r"],
                    "l_spearman_rho": s_l_stats["spearman_rho"],
                    "s_r2": s_s_stats["r2"], "s_pearson_r": s_s_stats["pearson_r"],
                    "s_spearman_rho": s_s_stats["spearman_rho"],
                })

    rows.sort(key=lambda r: (r[args.sort_by] if r[args.sort_by] == r[args.sort_by] else -1), reverse=True)
    ranking_rows.sort(key=lambda r: (r["proxy_kind"], r["calib_method"], r["proxy_batch_size"], r["corruption"]))

    if args.csv_path:
        path = Path(args.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=_TABLE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {path}")

    if args.rankings_csv_path:
        path = Path(args.rankings_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=_RANKING_TABLE_COLUMNS)
            writer.writeheader()
            writer.writerows(ranking_rows)
        print(f"Wrote {len(ranking_rows)} rows to {path} "
              f"({len(ranking_stream_names)} corruptions incl. clean, "
              f"{len(rows)} combos each) — pivot in Excel (corruption as columns) and "
              f"CORREL() any two to check proxy-selection transferability across "
              f"corruptions (see module docstring).")

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

        ranking_table = wandb.Table(columns=_RANKING_TABLE_COLUMNS)
        for r in ranking_rows:
            ranking_table.add_data(*[r[c] for c in _RANKING_TABLE_COLUMNS])
        run.log({"proxy_rankings": ranking_table})

        run.finish()
        print(f"\nLogged {len(rows)} rows to wandb project '{args.wandb_project}' "
              f"(group='{group}') — click any column header in the table UI to sort by it.\n"
              f"Logged {len(ranking_rows)} rows to the 'proxy_rankings' table in the same run.")


if __name__ == "__main__":
    main()
