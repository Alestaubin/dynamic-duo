#!/usr/bin/env python3
"""
scripts/diagnose_ema_filter.py
================================
Why does "no filter" (NoFilter) consistently beat EMA on the filtered-proxy
gate, even though the per-batch proxy score is visibly noisy? This script
investigates WITHOUT any new model forward passes: it replays the real
JointProxyWeighted pipeline (the actual production class, not a
reimplementation) against ALREADY-CACHED (z_l, z_s, labels) from a past
`plot_run_diagnostics.py --cache_logits` run, under several filter_kind
variants built from the SAME proxy/calib/beta/proxy_batch_size, and compares
them batch-by-batch on an identical stream.

Two competing explanations for "EMA always loses" are tested directly:

  (H1) COLD-START-ON-RESET: `set_corruption()` resets EMA to a fixed,
       uninformative prior (logit(0.5) by default) at every corruption
       boundary (see joint_proxy_weighted.py / filters/ema.py's docstrings
       -- intentional, matching the model's own TENT reset there). NoFilter
       has no state to reset, so it is exactly as good on batch 1 of a
       corruption as on batch 1000. If EMA's deficit is concentrated in the
       first few proxy-batches after every reset and vanishes later, the
       REPEATED per-corruption cold start is what's actually costing
       accuracy, not noise-smoothing being a bad idea per se.

  (H2) INTRINSIC LAG UNDER DRIFT: even fully converged (deep into a
       corruption, far from any reset), low-alpha EMA may still lag a
       genuinely drifting reliability gap (TENT continues adapting every
       batch, so the "true" better-model boundary can move over time, not
       just jitter around a fixed mean). Smoothing only pays for itself when
       the quantity being smoothed is noise around a CONSTANT mean; against
       real drift it trades variance for a bias that a memoryless estimator
       doesn't pay. This is tested by comparing EMA's cross-correlation lag
       (vs. the true accuracy gap) against NoFilter's, and by measuring
       within-corruption drift directly in the true accuracy gap.

A third run, "ema_noreset", is EMA with H1 surgically removed (its internal
state is carried across the corruption boundary instead of reset -- see
_carry_filter_state) while everything else about it (buffering, gate, pool)
is untouched. If ema_noreset closes most of the gap to NoFilter, that is
strong evidence for H1 over H2 (a REAL bug / design flaw in the reset
policy, not an inherent property of smoothing). If ema_noreset is barely
different from ema, the deficit is intrinsic to smoothing under drift (H2),
and no code bug is implicated.

Expensive proxy scoring (e.g. nuclear_norm's SVD) is computed ONCE per
corruption and replayed for every filter variant (see
_mk_recording_proxy_scores / _mk_cached_proxy_scores) -- only the filter/
gate/combine step, which is what's actually under test, differs between
variants. Still, a full multi-corruption replay on CPU-only SVD can run
long on a loaded shared node; `per_batch_rows.csv` is flushed to disk after
EVERY corruption (not just at the end), so a run killed partway still
leaves a usable partial result. Add `-u` (or `PYTHONUNBUFFERED=1`) to see
per-corruption progress lines live instead of only at exit, if stdout is
redirected to a file.

Usage
-----
    PYTHONPATH=. python -u scripts/diagnose_ema_filter.py \\
        --config cfgs/dynamic_duo_config_vitb_resnet.yaml \\
        --cache_dir "out/run_diagnostics/vit_b_16+resnet50__nuclear_norm_identity_pbs128__both_indep__20260916_174559/logits_cache" \\
        --out_dir out/diagnose_ema_filter

Requires a logits_cache/ directory from a prior `--cache_logits` run (one
<corruption>_s<severity>.pt file per corruption, each holding z_l, z_s,
labels in original batch-stream order -- see plot_run_diagnostics.py).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.data import load_config
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_proxy_weighted import JointProxyWeighted
from src.reliability.setup import build_proxy_weighted_calibrator
from scripts._cli import add_duo_config_arg

try:
    from scipy.stats import pearsonr
except ImportError:  # pragma: no cover
    pearsonr = None


# ─────────────────────────────────────────────────────────────────────── #
# Filter-variant construction
# ─────────────────────────────────────────────────────────────────────── #

def _build_variant(
    name: str, filter_kind: str, filter_kwargs: dict,
    proxy_kind: str, calib_method: str, calib_map: str | None,
    beta: float, prior_l: float, prior_s: float, proxy_batch_size: int, pool: str,
    base_ts: JointFixedTS | None, config: dict,
) -> JointProxyWeighted:
    """Build one JointProxyWeighted via the real production factory
    (build_proxy_weighted_calibrator) so this script can never drift from
    how an actual run constructs the calibrator. proxy_kind='nuclear_norm'
    (and 'ac_mc') are stateless -- _build_proxy_stats never touches the
    model/preprocess args for them -- so None placeholders are safe here and
    no model needs to be loaded for this diagnostic."""
    calibrator = build_proxy_weighted_calibrator(
        proxy_kind=proxy_kind,
        proxy_cache=None,
        calib_map=calib_map,
        calib_method=calib_method,
        filter_kind=filter_kind,
        filter_kwargs=filter_kwargs,
        beta=beta,
        prior_l=prior_l,
        prior_s=prior_s,
        base_ts=base_ts,
        csv_path=None,
        config=config,
        large_model=None, large_preprocess=None,
        small_model=None, small_preprocess=None,
        device=torch.device("cpu"),
        proxy_batch_size=proxy_batch_size,
        pool=pool,
    )
    calibrator.verbose = False
    calibrator._variant_name = name  # type: ignore[attr-defined]
    return calibrator


def _filter_state(calibrator: JointProxyWeighted) -> tuple[dict, dict]:
    """Duck-typed state getter: EMA exposes `_x`, Kalman exposes `_mean`/
    `_var`. NoFilter/RunningMean have no meaningful state to carry."""
    def _one(f):
        if hasattr(f, "_x"):
            return {"_x": f._x}
        if hasattr(f, "_mean"):
            return {"_mean": f._mean, "_var": f._var}
        return {}
    return _one(calibrator._filter_l), _one(calibrator._filter_s)


def _restore_filter_state(calibrator: JointProxyWeighted, state_l: dict, state_s: dict) -> None:
    for attr, val in state_l.items():
        setattr(calibrator._filter_l, attr, val)
    for attr, val in state_s.items():
        setattr(calibrator._filter_s, attr, val)


# ─────────────────────────────────────────────────────────────────────── #
# Replay
# ─────────────────────────────────────────────────────────────────────── #

def _mk_cached_proxy_scores(precomputed: list[tuple[float, float]]):
    """Replaces a JointProxyWeighted instance's (expensive) `_proxy_scores`
    with a simple pop-from-list lookup. The raw proxy score (e.g.
    nuclear_norm's SVD) depends only on proxy_kind/calib/proxy_batch_size/bs
    -- identical across every filter_kind variant -- so computing it once
    (on the FIRST variant replayed, via `_record_proxy_scores` below) and
    replaying the recorded sequence for every other variant skips the
    expensive part entirely without changing what's actually under test
    (the filter + gate + combine step). Assigning a plain function directly
    onto the instance (not the class) means Python calls it as-is, with no
    implicit `self` -- matching `_proxy_scores`'s own (z_l, z_s, f_l, f_s,
    labels) signature minus self.
    """
    it = iter(precomputed)

    def _cached(z_l, z_s, f_l, f_s, labels):
        return next(it)

    return _cached


def _mk_recording_proxy_scores(real_proxy_scores, sink: list[tuple[float, float]]):
    """Wraps a calibrator's real (bound) `_proxy_scores` to additionally
    append every (r_l, r_s) it computes to `sink`, in flush order -- used on
    exactly one ("recorder") variant per corruption so every other variant
    can replay via `_mk_cached_proxy_scores` instead of recomputing."""
    def _recording(z_l, z_s, f_l, f_s, labels):
        r = real_proxy_scores(z_l, z_s, f_l, f_s, labels)
        sink.append(r)
        return r

    return _recording


def _replay_corruption(
    calibrator: JointProxyWeighted, label: str,
    z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor, bs: int,
    no_reset: bool, carry_l: dict | None, carry_s: dict | None,
) -> tuple[list[dict], dict, dict]:
    """Feed one corruption's cached logits through `calibrator` batch by
    batch (adaptation_batch_size = bs, matching the cached run), returning a
    per-ADAPTATION-BATCH record (dense, uniformly spaced -- unlike the
    sparser per-proxy-batch flushes) plus the filter state at the end, for a
    no_reset variant's next corruption to carry forward. `calibrator` is
    expected to already have its `_proxy_scores` set up (real, recording, or
    cached-replay -- see the two helpers above) by the caller.
    """
    n = z_l.shape[0]
    calibrator.set_corruption(label, total_samples=n)
    if no_reset and carry_l is not None:
        # Surgically undo set_corruption's reset-to-prior (H1's ablation):
        # everything else (buffer clear, cached_w_l reset) stays identical,
        # only the filters' own internal memory survives the boundary.
        _restore_filter_state(calibrator, carry_l, carry_s)
        # The cached gate weight (used for any partial batch before the
        # first flush) must reflect the carried state too, or the very
        # first batch of the corruption would still gate on the neutral
        # prior despite the filters themselves not having reset.
        x_l, x_s = calibrator._filter_l._x if hasattr(calibrator._filter_l, "_x") else calibrator._filter_l._mean, \
                   calibrator._filter_s._x if hasattr(calibrator._filter_s, "_x") else calibrator._filter_s._mean
        calibrator._cached_x_l, calibrator._cached_x_s = x_l, x_s
        calibrator._cached_w_l = float(torch.sigmoid(torch.tensor(calibrator.beta * (x_l - x_s))))

    rows = []
    start = 0
    batch_idx = 0
    while start < n:
        end = min(start + bs, n)
        zl_b, zs_b, y_b = z_l[start:end], z_s[start:end], labels[start:end]
        calibrator.set_labels(y_b)
        z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l = calibrator._forward(zl_b, zs_b)

        acc_l = float((zl_b.argmax(1) == y_b).float().mean())
        acc_s = float((zs_b.argmax(1) == y_b).float().mean())
        duo_acc = float((z_duo.argmax(1) == y_b).float().mean())

        rows.append({
            "corruption": label, "batch_idx": batch_idx, "n_refreshes": calibrator.n_refreshes,
            "r_l": r_l, "r_s": r_s, "a_l": a_l, "a_s": a_s, "x_l": x_l, "x_s": x_s,
            "w_l": w_l, "acc_l": acc_l, "acc_s": acc_s, "duo_acc": duo_acc,
            "true_gap": acc_l - acc_s, "gate_gap": x_l - x_s,
            "best_is_large": acc_l > acc_s, "oracle_acc": max(acc_l, acc_s),
            "regret": max(acc_l, acc_s) - duo_acc,
        })
        start = end
        batch_idx += 1

    carry_l_out, carry_s_out = _filter_state(calibrator)
    return rows, carry_l_out, carry_s_out


# ─────────────────────────────────────────────────────────────────────── #
# Analysis
# ─────────────────────────────────────────────────────────────────────── #

def _cold_start_breakdown(rows: list[dict], warmup_batches: int) -> dict:
    """Mean duo accuracy / regret split into 'first `warmup_batches` after a
    reset' vs 'the rest' -- the direct test of H1. n_refreshes resets to 0 at
    every set_corruption() call, so (per corruption) batch_idx < some small
    count after the *adaptation*-batch boundary isn't quite n_refreshes==0..k;
    instead we key off batch_idx directly, which is already per-corruption."""
    by_corr: dict[str, list[dict]] = {}
    for r in rows:
        by_corr.setdefault(r["corruption"], []).append(r)
    warm_acc, warm_reg, late_acc, late_reg = [], [], [], []
    for corr_rows in by_corr.values():
        for r in corr_rows:
            if r["batch_idx"] < warmup_batches:
                warm_acc.append(r["duo_acc"]); warm_reg.append(r["regret"])
            else:
                late_acc.append(r["duo_acc"]); late_reg.append(r["regret"])
    return {
        "warmup_mean_duo_acc": float(np.mean(warm_acc)) if warm_acc else float("nan"),
        "warmup_mean_regret": float(np.mean(warm_reg)) if warm_reg else float("nan"),
        "late_mean_duo_acc": float(np.mean(late_acc)) if late_acc else float("nan"),
        "late_mean_regret": float(np.mean(late_reg)) if late_reg else float("nan"),
        "n_warmup": len(warm_acc), "n_late": len(late_acc),
    }


def _lag_correlation(rows: list[dict], max_lag: int = 8) -> dict:
    """Pearson corr between gate_gap[t] and true_gap[t+k] for k=0..max_lag,
    per corruption then averaged -- the lag k maximizing |corr| is how many
    batches "behind" this filter's signal effectively sits relative to the
    ground truth it's trying to track. NoFilter's gate_gap is just a_l-a_s
    at that same instant, so its own best lag is the baseline to compare
    against (should come out near 0 if proxy noise, not lag, were the
    limiting factor)."""
    by_corr: dict[str, list[dict]] = {}
    for r in rows:
        by_corr.setdefault(r["corruption"], []).append(r)

    best_lags = []
    corr_at_0 = []
    for corr_rows in by_corr.values():
        gate = np.array([r["gate_gap"] for r in corr_rows])
        true = np.array([r["true_gap"] for r in corr_rows])
        if len(gate) < max_lag + 10 or pearsonr is None:
            continue
        best_k, best_abs_r = 0, -1.0
        r0 = float("nan")
        for k in range(0, max_lag + 1):
            if k == 0:
                g, t = gate, true
            else:
                g, t = gate[:-k], true[k:]
            if g.std() < 1e-9 or t.std() < 1e-9:
                continue
            r, _ = pearsonr(g, t)
            if k == 0:
                r0 = r
            if abs(r) > best_abs_r:
                best_abs_r, best_k = abs(r), k
        best_lags.append(best_k)
        corr_at_0.append(r0)
    return {
        "mean_best_lag": float(np.mean(best_lags)) if best_lags else float("nan"),
        "mean_corr_at_lag0": float(np.nanmean(corr_at_0)) if corr_at_0 else float("nan"),
        "n_corruptions": len(best_lags),
    }


def _drift_within_corruption(rows: list[dict]) -> dict:
    """How non-stationary is the TRUE accuracy gap within one corruption?
    (H2's premise.) Fits a linear trend of true_gap vs batch_idx per
    corruption and reports the mean |slope| * n_batches (total drift over
    the corruption) against the mean batch-to-batch std (noise floor) -- if
    total drift >> noise, smoothing trades away real signal, not just noise."""
    by_corr: dict[str, list[dict]] = {}
    for r in rows:
        by_corr.setdefault(r["corruption"], []).append(r)
    drifts, noises = [], []
    for corr_rows in by_corr.values():
        x = np.array([r["batch_idx"] for r in corr_rows], dtype=np.float64)
        y = np.array([r["true_gap"] for r in corr_rows], dtype=np.float64)
        if len(x) < 10:
            continue
        slope, _ = np.polyfit(x, y, 1)
        drifts.append(abs(slope) * (x.max() - x.min()))
        noises.append(float(np.std(np.diff(y))))
    return {
        "mean_total_drift": float(np.mean(drifts)) if drifts else float("nan"),
        "mean_batch_noise_std": float(np.mean(noises)) if noises else float("nan"),
    }


def _overall_accuracy(rows: list[dict]) -> float:
    return float(np.mean([r["duo_acc"] for r in rows])) if rows else float("nan")


# ─────────────────────────────────────────────────────────────────────── #
# Plotting
# ─────────────────────────────────────────────────────────────────────── #

def _plot_summary(results: dict[str, list[dict]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    accs = [_overall_accuracy(results[n]) for n in names]

    fig, ax = plt.subplots(figsize=(1.2 * max(4, len(names)), 4))
    bars = ax.bar(names, accs, color="#4C72B0")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a, f"{a:.3f}", ha="center", va="bottom")
    ax.set_ylabel("overall duo accuracy")
    ax.set_title("Overall duo accuracy by filter variant (all corruptions)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_accuracy.png", dpi=150)
    plt.close(fig)


def _plot_timeseries(results: dict[str, list[dict]], corruption: str, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for name, rows in results.items():
        corr_rows = [r for r in rows if r["corruption"] == corruption]
        if not corr_rows:
            continue
        xs = [r["batch_idx"] for r in corr_rows]
        ax1.plot(xs, [r["w_l"] for r in corr_rows], label=name, linewidth=1.2)
        ax2.plot(xs, [r["duo_acc"] for r in corr_rows], label=name, alpha=0.6, linewidth=0.8)

    # Ground truth: which model is actually better, as a step function at 0/1.
    any_rows = next(iter(results.values()))
    truth_rows = [r for r in any_rows if r["corruption"] == corruption]
    ax1.plot([r["batch_idx"] for r in truth_rows],
              [1.0 if r["best_is_large"] else 0.0 for r in truth_rows],
              color="black", alpha=0.25, linewidth=2.0, label="truth: large is better (1/0)")

    ax1.set_ylabel("w_l (gate weight on large model)")
    ax1.set_title(f"Gate weight vs. ground truth — {corruption}")
    ax1.legend(fontsize=7, ncol=3)
    ax2.set_ylabel("per-batch duo accuracy")
    ax2.set_xlabel("adaptation batch index (within corruption)")
    ax2.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / f"timeseries_{corruption}.png", dpi=150)
    plt.close(fig)


def _plot_cold_start(cold_start: dict[str, dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(cold_start.keys())
    warm = [cold_start[n]["warmup_mean_duo_acc"] for n in names]
    late = [cold_start[n]["late_mean_duo_acc"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.4 * max(4, len(names)), 4))
    ax.bar(x - width / 2, warm, width, label="warm-up batches (just after reset)")
    ax.bar(x + width / 2, late, width, label="steady-state batches")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("mean duo accuracy")
    ax.set_title("H1 test: cold-start-after-reset vs. steady-state accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cold_start_breakdown.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────── #
# Main
# ─────────────────────────────────────────────────────────────────────── #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_duo_config_arg(p, default="cfgs/dynamic_duo_config_vitb_resnet.yaml")
    p.add_argument("--cache_dir", type=str, required=True,
                    help="A past run's out_dir/logits_cache/ directory (needs --cache_logits "
                         "to have been used on that run) -- one <corruption>_s<sev>.pt per "
                         "corruption, holding z_l/z_s/labels in original stream order.")
    p.add_argument("--corruptions", type=str, nargs="*", default=None,
                    help="Subset of cache files (by stem, e.g. 'brightness_s5') to use. "
                         "Default: every .pt file found in --cache_dir.")
    p.add_argument("--bs", type=int, default=64,
                    help="Adaptation batch size the cache was generated with (see the "
                         "source run's cfgs/<duo>.yaml BS -- NOT re-derivable from the cache "
                         "itself, pass it explicitly).")
    p.add_argument("--proxy_kind", type=str, default="nuclear_norm")
    p.add_argument("--calib_method", type=str, default="identity")
    p.add_argument("--calib_map", type=str, default=None)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--proxy_batch_size", type=int, default=128)
    p.add_argument("--pool", type=str, default="log", choices=["log", "linear"])
    p.add_argument("--prior_l", type=float, default=0.5)
    p.add_argument("--prior_s", type=float, default=0.5)
    p.add_argument("--fixed_ts_config", type=str, default="checkpoints/fixed_ts/default")
    p.add_argument("--alphas", type=float, nargs="*", default=[0.05, 0.1, 0.3, 0.5, 0.9999],
                    help="EMA alphas to test, one variant each, plus a no-reset ablation "
                         "of the middle one.")
    p.add_argument("--warmup_batches", type=int, default=10,
                    help="How many adaptation batches after a reset count as 'warm-up' "
                         "for the H1 cold-start breakdown.")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Truncate each corruption's cached stream to this many samples "
                         "(from the start) -- for fast iteration on the script itself "
                         "before committing to a full replay. Default: use everything cached.")
    p.add_argument("--out_dir", type=str, default="out/diagnose_ema_filter")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    base_ts = JointFixedTS.load(args.fixed_ts_config) if args.fixed_ts_config else None

    cache_dir = Path(args.cache_dir)
    cache_files = sorted(cache_dir.glob("*.pt"))
    if args.corruptions:
        wanted = set(args.corruptions)
        cache_files = [f for f in cache_files if f.stem in wanted]
    if not cache_files:
        raise SystemExit(f"No cache files found in {cache_dir} (looked for *.pt)")
    print(f"[diagnose_ema_filter] replaying {len(cache_files)} corruption(s) from {cache_dir}")

    common = dict(
        proxy_kind=args.proxy_kind, calib_method=args.calib_method, calib_map=args.calib_map,
        beta=args.beta, prior_l=args.prior_l, prior_s=args.prior_s,
        proxy_batch_size=args.proxy_batch_size, pool=args.pool, base_ts=base_ts, config=config,
    )

    variant_specs: list[tuple[str, str, dict, bool]] = [("none", "none", {}, False)]
    for a in args.alphas:
        variant_specs.append((f"ema_alpha={a}", "ema", {"alpha": a}, False))
    # No-reset ablation on the MIDDLE (most heavily-smoothing-but-plausible)
    # alpha supplied, to directly isolate H1 from H2 on a variant that
    # actually smooths meaningfully (alpha near 1 has nothing to carry).
    mid_alpha = sorted(args.alphas)[len(args.alphas) // 2]
    variant_specs.append((f"ema_alpha={mid_alpha}_noreset", "ema", {"alpha": mid_alpha}, True))

    # One persistent calibrator per variant, built ONCE (not per corruption)
    # so a no_reset variant's filter state survives across corruption
    # boundaries the way _replay_corruption's carry_state mechanism expects.
    calibrators = {name: _build_variant(name, fk, fkw, **common) for name, fk, fkw, _ in variant_specs}
    # Captured BEFORE any corruption's recording wrapper overwrites the
    # instance attribute, so each corruption's recorder always wraps the
    # true original (no nesting buildup across corruptions).
    orig_proxy_scores = {name: calibrator._proxy_scores for name, calibrator in calibrators.items()}
    results: dict[str, list[dict]] = {name: [] for name, *_ in variant_specs}
    carry_state: dict[str, tuple[dict, dict]] = {name: ({}, {}) for name, *_ in variant_specs}

    # Written incrementally (one corruption's rows at a time, flushed to
    # disk immediately) rather than only at the very end -- on a shared/
    # contended node a multi-corruption replay can run far longer than a
    # quick local test would suggest (see this script's own run notes), so
    # a run interrupted partway still leaves a usable, inspectable partial
    # CSV instead of nothing (matching plot_run_diagnostics.py's own
    # per-corruption incremental-write convention).
    import csv
    all_rows_path = out_dir / "per_batch_rows.csv"
    _row_fieldnames = [
        "variant", "corruption", "batch_idx", "n_refreshes", "r_l", "r_s", "a_l", "a_s",
        "x_l", "x_s", "w_l", "acc_l", "acc_s", "duo_acc", "true_gap", "gate_gap",
        "best_is_large", "oracle_acc", "regret",
    ]
    csv_fh = all_rows_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=_row_fieldnames)
    csv_writer.writeheader()

    for f in cache_files:
        d = torch.load(f, map_location="cpu")
        z_l, z_s, labels = d["z_l"], d["z_s"], d["labels"]
        if args.max_samples:
            z_l, z_s, labels = z_l[: args.max_samples], z_s[: args.max_samples], labels[: args.max_samples]
        label = f.stem

        # The raw proxy score (e.g. nuclear_norm's SVD) is identical across
        # every variant here -- proxy_kind/calib/proxy_batch_size/bs never
        # vary, only filter_kind/no_reset do. Pay for it ONCE per
        # corruption (on the first variant processed) and replay the
        # recorded sequence for every other variant -- a ~len(variant_specs)x
        # speedup on what's otherwise the dominant cost (see this script's
        # own benchmarking: ~100ms/SVD call on a CPU-only node).
        precomputed: list[tuple[float, float]] = []
        for i, (name, filter_kind, filter_kwargs, no_reset) in enumerate(variant_specs):
            calibrator = calibrators[name]
            if i == 0:
                calibrator._proxy_scores = _mk_recording_proxy_scores(orig_proxy_scores[name], precomputed)
            else:
                calibrator._proxy_scores = _mk_cached_proxy_scores(precomputed)

            carry_l, carry_s = carry_state[name]
            corr_rows, carry_l, carry_s = _replay_corruption(
                calibrator, label, z_l, z_s, labels, args.bs,
                no_reset=no_reset, carry_l=(carry_l if no_reset else None), carry_s=(carry_s if no_reset else None),
            )
            carry_state[name] = (carry_l, carry_s)
            results[name].extend(corr_rows)
            for r in corr_rows:
                csv_writer.writerow({"variant": name, **r})
        csv_fh.flush()
        print(f"  [{label}] done ({len(precomputed)} proxy-batch flushes, shared across variants)", flush=True)

    csv_fh.close()
    print(f"[diagnose_ema_filter] wrote {all_rows_path}", flush=True)

    for name in results:
        rows = results[name]
        print(f"  [{name}] overall duo acc = {_overall_accuracy(rows):.4f}  "
              f"(n_batches={len(rows)})", flush=True)

    # ── H1: cold-start-after-reset breakdown ──────────────────────────── #
    cold_start = {name: _cold_start_breakdown(rows, args.warmup_batches) for name, rows in results.items()}
    print("\n=== H1: cold-start-after-reset breakdown ===")
    for name, cs in cold_start.items():
        print(f"  [{name}] warm-up(n={cs['n_warmup']}) acc={cs['warmup_mean_duo_acc']:.4f} "
              f"regret={cs['warmup_mean_regret']:.4f}  |  "
              f"steady(n={cs['n_late']}) acc={cs['late_mean_duo_acc']:.4f} "
              f"regret={cs['late_mean_regret']:.4f}")

    # ── H2: lag correlation + within-corruption drift ─────────────────── #
    print("\n=== H2: lag correlation (gate_gap[t] vs true_gap[t+lag]) ===")
    lag_results = {}
    for name, rows in results.items():
        lag = _lag_correlation(rows)
        lag_results[name] = lag
        print(f"  [{name}] best_lag={lag['mean_best_lag']:.2f} batches  "
              f"corr_at_lag0={lag['mean_corr_at_lag0']:.3f}  (n_corr={lag['n_corruptions']})")

    drift = _drift_within_corruption(next(iter(results.values())))
    print(f"\n=== H2 premise: true accuracy-gap drift within a corruption ===")
    print(f"  mean total drift over a corruption: {drift['mean_total_drift']:.4f}")
    print(f"  mean batch-to-batch noise std:      {drift['mean_batch_noise_std']:.4f}")
    print(f"  ratio (drift / noise):              "
          f"{drift['mean_total_drift'] / max(drift['mean_batch_noise_std'], 1e-9):.2f}")

    # ── Verdict ─────────────────────────────────────────────────────────#
    none_acc = _overall_accuracy(results["none"])
    best_ema_name = max((n for n in results if n.startswith("ema_") and "noreset" not in n),
                         key=lambda n: _overall_accuracy(results[n]))
    best_ema_acc = _overall_accuracy(results[best_ema_name])
    noreset_name = f"ema_alpha={mid_alpha}_noreset"
    noreset_acc = _overall_accuracy(results[noreset_name])
    reset_variant_name = f"ema_alpha={mid_alpha}"
    reset_acc = _overall_accuracy(results[reset_variant_name])
    none_vs_ema_gap = none_acc - reset_acc  # positive iff EMA actually underperforms none here
    gap_closed = (
        (noreset_acc - reset_acc) / none_vs_ema_gap
        if none_vs_ema_gap > 1e-4 else float("nan")
    )

    verdict = {
        "none_acc": none_acc,
        "best_ema_variant": best_ema_name, "best_ema_acc": best_ema_acc,
        "ema_with_reset_acc": reset_acc, "ema_no_reset_acc": noreset_acc,
        "none_vs_ema_gap": none_vs_ema_gap,
        "fraction_of_none_vs_ema_gap_closed_by_removing_reset": gap_closed,
    }
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    if none_vs_ema_gap <= 1e-4:
        print(
            f"-> On this data, filter_kind='none' did NOT actually beat "
            f"ema_alpha={mid_alpha} (none={none_acc:.4f} vs ema={reset_acc:.4f}) -- there's no "
            f"gap here to attribute to H1 vs H2. This is expected on a small/unrepresentative "
            f"sample (e.g. --max_samples smoke tests); re-run on the full cached stream (omit "
            f"--max_samples) and/or more corruptions before drawing a conclusion."
        )
    elif gap_closed > 0.5:
        print(
            "-> Most of EMA's deficit disappears when the per-corruption reset-to-prior "
            "is removed: H1 (repeated cold start) is the dominant explanation. This points "
            "at the filter reset POLICY (reset to an uninformative prior every corruption "
            "boundary), not the EMA math itself, as what to fix/reconsider."
        )
    else:
        print(
            "-> Removing the reset barely helps: H1 is not the main driver. The deficit is "
            "likely intrinsic lag under real within-corruption drift (H2) -- check the "
            "drift/noise ratio above; if it's >> 1, smoothing is trading away real signal, "
            "which is a genuine bias-variance property of EMA here, not a bug."
        )

    with (out_dir / "verdict.json").open("w") as fh:
        json.dump({
            "verdict": verdict, "cold_start": cold_start, "lag": lag_results, "drift": drift,
        }, fh, indent=2)

    # ── Plots ───────────────────────────────────────────────────────────#
    _plot_summary(results, out_dir)
    _plot_cold_start(cold_start, out_dir)
    first_corruption = cache_files[0].stem
    _plot_timeseries(results, first_corruption, out_dir)
    print(f"\n[diagnose_ema_filter] plots + verdict.json written to {out_dir}")


if __name__ == "__main__":
    main()
