#!/usr/bin/env python3
"""
scripts/proxy_vs_optimal_temperature.py
========================================
Does a label-free reliability proxy actually track how much a model NEEDS
re-calibrating, batch by batch?

The duo is normally wrapped in two temperatures T_l, T_s fit once on clean
validation data by minimizing NLL (JointFixedTS.tune, checkpoints/fixed_ts/*)
-- that's a single global calibration for the whole run. This script instead
re-fits that SAME NLL-minimizing joint temperature-scale fresh on every
single adaptation batch (using that batch's own ground-truth labels -- an
oracle, cheating measurement, never used to drive the duo, only to ask "what
WOULD the ideal per-batch calibration have been"), and, independently,
computes each proxy_kind's raw label-free score (Section 2) on that same
batch. It then correlates the two: does r_l track T_l*, does r_s track T_s*,
and does the raw score GAP (r_l - r_s) track the optimal temperature RATIO
(log(T_l*/T_s*))?

The two are DELIBERATELY decoupled:
  - --calib_config picks the ACTUAL joint_calibrator driving the duo (any of
    _CALIB_MODES: fixed_ts/oracle_ts/coca/proxy_weighted) via
    compare_calibrators._build_calibrator -- exactly as plot_run_diagnostics.py
    does. This shapes the batches' logits (especially under an adapting
    --mode), but the correlation study works the same regardless of which one
    is chosen.
  - The proxy_kind used for the r_l/r_s measurement comes from --calib_config
    too when it's a proxy_weighted config (the "usual" mechanism, same as
    plot_run_diagnostics.py) -- but can also be set explicitly via
    --proxy_kind, which is REQUIRED for any other calibration_mode (fixed_ts/
    oracle_ts/coca configs have no proxy_kind field of their own).

Caveat: a per-batch temperature fit sees only that batch's samples (typically
~100-500), so T_l*/T_s* are noisier estimates than the whole-val-set fit --
expected, not a bug; this is exactly the "done per batch instead" the
correlation is measuring against.

Usage
-----
    # fixed_ts baseline drives the duo; nuclear_norm proxy measured throughout
    python scripts/proxy_vs_optimal_temperature.py --config cfgs/dynamic_duo_config.yaml \
        --calib_config cfgs/calib_configs/fixed_ts_default.json \
        --mode no_adapt --proxy_kind nuclear_norm --num_samples 5000

    # proxy_weighted itself drives the duo; proxy_kind picked up from the calib_config
    python scripts/proxy_vs_optimal_temperature.py --config cfgs/dynamic_duo_config.yaml \
        --calib_config cfgs/calib_configs/nuclear_norm_identity_pbs128.json \
        --mode both_indep --num_samples 5000
"""

from __future__ import annotations

import argparse
import csv
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo, _MODES
from src.utils.data import load_config
from src.utils.model import get_model
from src.utils.logit_transforms import combine_logits
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.reliability.proxies.stats import PROXY_KINDS, FeatureExtractor
from src.reliability.setup import _build_proxy_stats
from src.utils.diagnostics_plots import C_LARGE, C_SMALL, C_MUTED, C_INK, C_GRID, C_SURFACE
from scripts.compare_calibrators import _build_calibrator
from scripts.plot_run_diagnostics import _load_calib_config, _resolve_fixed_ts_config, _default_calib_map
from scripts._cli import (
    add_duo_config_arg, add_num_samples_arg, add_seed_arg,
    add_proto_metric_arg, add_out_dir_run_name_args,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK,
    "text.color": C_INK, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "grid.color": C_GRID, "font.size": 10,
})

_BATCH_CSV_FIELDS = [
    "corruption", "batch_idx", "n",
    "T_l_opt", "T_s_opt", "log_T_ratio_opt", "opt_nll",
    "T_l_base", "T_s_base", "base_nll",
    "r_l", "r_s", "gap_r",
    "acc_l", "acc_s",
]


def _extract_base_ts(calibrator) -> tuple[float, float]:
    """(T_l, T_s) of whatever whole-validation-set-fit temperatures the duo's
    ACTUAL joint_calibrator is currently using, for reference alongside the
    per-batch optimum -- JointFixedTS itself (fixed_ts/oracle_ts) exposes
    Tl/Ts directly; JointProxyWeighted/JointOptimalWOracle expose them via
    .base_ts (possibly None); coca has no such concept. (nan, nan) if none
    of those apply, rather than crashing -- this script must run for ANY
    calibration_mode.
    """
    ts = calibrator if hasattr(calibrator, "Tl") else getattr(calibrator, "base_ts", None)
    if ts is not None and hasattr(ts, "Tl") and hasattr(ts, "Ts"):
        return float(ts.Tl.item()), float(ts.Ts.item())
    return float("nan"), float("nan")


def _corr(xs: list[float], ys: list[float]) -> dict:
    """Pearson r and Spearman rho between xs and ys. nan when n < 3 or either
    side is constant (undefined correlation, not zero)."""
    from scipy.stats import pearsonr, spearmanr

    n = len(xs)
    nan_result = {"n": n, "pearson_r": float("nan"), "spearman_rho": float("nan")}
    if n < 3:
        return nan_result
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return nan_result
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    return {"n": n, "pearson_r": float(pr), "spearman_rho": float(sr)}


def _corr_row(label: str, rows: list[dict]) -> dict:
    """One correlation-summary row: proxy-vs-optimal-T for large, small, and
    the score-gap-vs-log-temperature-ratio (the paper's Section-5 gate acts
    on exactly this gap -- see JointProxyWeighted -- so it's the single most
    relevant number here, mirroring CLAUDE.md's gap_bias/gap_corr diagnostic
    for accuracy)."""
    l_c = _corr([r["T_l_opt"] for r in rows], [r["r_l"] for r in rows])
    s_c = _corr([r["T_s_opt"] for r in rows], [r["r_s"] for r in rows])
    g_c = _corr([r["log_T_ratio_opt"] for r in rows], [r["gap_r"] for r in rows])
    return {
        "corruption": label, "n": len(rows),
        "l_pearson_r": l_c["pearson_r"], "l_spearman_rho": l_c["spearman_rho"],
        "s_pearson_r": s_c["pearson_r"], "s_spearman_rho": s_c["spearman_rho"],
        "gap_pearson_r": g_c["pearson_r"], "gap_spearman_rho": g_c["spearman_rho"],
    }


def _print_corr_table(corr_rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'corruption':<28}{'n':>6}   "
          f"{'large (T_l* vs r_l)':^24}   {'small (T_s* vs r_s)':^24}   {'gap (logT* vs r_l-r_s)':^24}")
    print(f"{'':<28}{'':>6}   {'pearson':>11}{'spearman':>13}   "
          f"{'pearson':>11}{'spearman':>13}   {'pearson':>11}{'spearman':>13}")
    print("-" * 100)
    for r in corr_rows:
        print(f"{r['corruption']:<28}{r['n']:>6}   "
              f"{r['l_pearson_r']:>11.3f}{r['l_spearman_rho']:>13.3f}   "
              f"{r['s_pearson_r']:>11.3f}{r['s_spearman_rho']:>13.3f}   "
              f"{r['gap_pearson_r']:>11.3f}{r['gap_spearman_rho']:>13.3f}")


def _plot(rows: list[dict], corr_rows: list[dict], out_path: Path) -> None:
    corruptions = sorted({r["corruption"] for r in rows})
    cmap = plt.get_cmap("tab20", max(len(corruptions), 1))
    color_by_corr = {c: cmap(i) for i, c in enumerate(corruptions)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specs = [
        (axes[0], "T_l_opt", "r_l", "T_l* (per-batch optimal)", "r_l (raw proxy)", "large"),
        (axes[1], "T_s_opt", "r_s", "T_s* (per-batch optimal)", "r_s (raw proxy)", "small"),
        (axes[2], "log_T_ratio_opt", "gap_r", "log(T_l*/T_s*)", "r_l - r_s", "gap"),
    ]
    overall = next(r for r in corr_rows if r["corruption"] == "ALL (pooled, ignoring corruption)")
    for ax, xk, yk, xlabel, ylabel, tag in specs:
        for c in corruptions:
            xs = [r[xk] for r in rows if r["corruption"] == c]
            ys = [r[yk] for r in rows if r["corruption"] == c]
            ax.scatter(xs, ys, s=10, alpha=0.5, color=color_by_corr[c], label=c, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.4, lw=0.5)
        key = "l" if tag == "large" else ("s" if tag == "small" else "gap")
        ax.set_title(f"pooled pearson r={overall[f'{key}_pearson_r']:.3f}  "
                      f"spearman ρ={overall[f'{key}_spearman_rho']:.3f}", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncols=min(len(corruptions), 8), fontsize=7.5)
    fig.suptitle("Per-batch optimal temperature vs. raw proxy score", fontsize=11, y=1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_duo_config_arg(p)
    p.add_argument("--calib_config", type=str, required=True,
                    help="Path to a JSON run_cfg dict (same shape/mechanism as "
                         "plot_run_diagnostics.py's --calib_config, see cfgs/calib_configs/) "
                         "-- picks calibration_mode + knobs for the duo's ACTUAL joint "
                         "calibrator (any of fixed_ts/oracle_ts/coca/proxy_weighted). If "
                         "calibration_mode is proxy_weighted, its proxy_kind/proto_metric "
                         "are also used as the DEFAULT for --proxy_kind/--proto_metric below.")
    p.add_argument("--mode", type=str, default="no_adapt", choices=sorted(_MODES),
                    help="Duo adaptation mode -- any of _MODES (no_adapt/both_duo/large_duo/"
                         "small_duo/*_indep). Shapes the logits the per-batch optimal-T fit "
                         "and the proxy are both measured on.")
    p.add_argument("--steps", type=int, default=1)
    add_num_samples_arg(p)
    add_seed_arg(p)
    p.add_argument("--batch_size", type=int, default=None, help="Overrides cfg['BS'].")

    p.add_argument("--proxy_kind", type=str, default=None, choices=sorted(PROXY_KINDS),
                    help="Which reliability proxy to measure r_l/r_s with. Defaults to "
                         "--calib_config's own proxy_kind (proxy_weighted configs only); "
                         "REQUIRED for any other calibration_mode.")
    p.add_argument("--proto_metric", type=str, default=None, choices=["cosine", "mahalanobis"],
                    help="Only used when the resolved --proxy_kind is 'prototype'. Defaults "
                         "to --calib_config's proto_metric, else 'cosine'.")
    p.add_argument("--proxy_cache", type=str, default=None,
                    help="Optional cache name for the proxy's source-fitted state (atc/"
                         "prototype/cot only -- see src.reliability.proxies.stats). Skips "
                         "re-running the source pass on repeat invocations.")

    p.add_argument("--grid_steps", type=int, default=15,
                    help="Per-batch oracle temperature fit: t_range resolution for "
                         "JointFixedTS.tune's grid-search init (grid_steps^2 combinations "
                         "evaluated per batch before LBFGS refinement). Lower than "
                         "JointFixedTS's own default (25) since this runs once per batch, "
                         "not once per run.")
    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=50.0)

    add_out_dir_run_name_args(
        p, out_dir_default="out/proxy_vs_temperature",
        run_name_help="Subdirectory name under --out_dir. Default: auto-generated from "
                       "calib_config/mode/proxy_kind/timestamp.",
    )

    args = p.parse_args()

    cfg = load_config(args.config)
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

    proxy_kind = args.proxy_kind or run_cfg.get("proxy_kind")
    if proxy_kind is None:
        p.error(f"--calib_config {args.calib_config!r} has calibration_mode="
                 f"{run_cfg['calibration_mode']!r} (no proxy_kind of its own) -- pass "
                 f"--proxy_kind explicitly to pick which proxy to measure.")
    proto_metric = args.proto_metric or run_cfg.get("proto_metric", "cosine")

    run_name = args.run_name or (
        f"{run_cfg['name']}__{args.mode}__{proxy_kind}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  duo: {cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}  |  "
          f"calib_config: {run_cfg['name']!r} ({run_cfg['calibration_mode']})  |  "
          f"proxy_kind: {proxy_kind}  |  mode: {args.mode}  |  out_dir: {out_dir}")

    large_model, large_preprocess = get_model(cfg["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(cfg["SMALL"]["NAME"])
    large_model, small_model = large_model.to(device), small_model.to(device)

    # The ACTUAL joint calibrator driving the duo -- any calibration_mode.
    calibrator = _build_calibrator(
        run_cfg, cfg, large_model, large_preprocess, small_model, small_preprocess,
        device, args.num_samples, args.seed, csv_path=None, verbose=False,
    )

    # The STANDALONE proxy instrument -- completely independent of `calibrator`
    # above, so this works the same regardless of calibration_mode. Only
    # atc/prototype/cot need a source (clean VAL_DIR) fitting pass first.
    cfg_l, cfg_s = _build_proxy_stats(
        proxy_kind, cfg, large_model, large_preprocess, small_model, small_preprocess,
        device, cache_path=args.proxy_cache, proto_metric=proto_metric,
    )
    ext_l = ext_s = None
    if proxy_kind == "prototype":
        ext_l = FeatureExtractor(large_model, cfg_l.name)
        ext_s = FeatureExtractor(small_model, cfg_s.name)

    duo = setup_duo(
        large=large_model, large_preprocess=large_preprocess,
        small=small_model, small_preprocess=small_preprocess,
        mode=args.mode, joint_calibrator=calibrator, calibration_mode=run_cfg["calibration_mode"],
        cfg=cfg, steps=args.steps,
    )

    batch_rows: list[dict] = []

    def _on_batch(batch_idx, prefix, duo, outputs, z_large, z_small, labels):
        z_l, z_s = z_large.detach(), z_small.detach()
        labels_d = labels.to(z_l.device).long()

        # Section 2's raw label-free score, straight from the fitted
        # ProxyStats -- decoupled from `calibrator` (whatever combines the
        # duo's own output) so this measurement is identical no matter which
        # calibration_mode drives the run.
        f_l = ext_l._feats if ext_l is not None else None
        f_s = ext_s._feats if ext_s is not None else None
        # labels_d is passed through even though every proxy but "oracle"
        # ignores it -- OracleProxy.score() asserts on a missing labels
        # kwarg (see src/reliability/proxies/oracle.py), and it's a legit
        # --proxy_kind choice here (an upper-bound reference, same role as
        # elsewhere in this codebase).
        r_l = cfg_l.score(proxy_kind, z_l, f_l, labels=labels_d)
        r_s = cfg_s.score(proxy_kind, z_s, f_s, labels=labels_d)

        # This batch's own NLL-minimizing joint temperatures -- the oracle,
        # per-batch analogue of the whole-val-set JointFixedTS.tune fit.
        ts_opt = JointFixedTS(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # boundary-pinned T*s are common under collapse; expected here
            ts_opt.tune(
                z_l, z_s, labels_d,
                t_min=args.t_min, t_max=args.t_max, grid_steps=args.grid_steps,
            )
        T_l_opt, T_s_opt = ts_opt.Tl.item(), ts_opt.Ts.item()
        opt_nll = F.cross_entropy(
            combine_logits(z_l=z_l, z_s=z_s, tau_l=T_l_opt, tau_s=T_s_opt), labels_d
        ).item()

        T_l_base, T_s_base = _extract_base_ts(calibrator)
        base_nll = float("nan")
        if T_l_base == T_l_base:  # not nan
            base_nll = F.cross_entropy(
                combine_logits(z_l=z_l, z_s=z_s, tau_l=T_l_base, tau_s=T_s_base), labels_d
            ).item()

        batch_rows.append({
            "corruption": prefix.rstrip("/"), "batch_idx": len(batch_rows), "n": z_l.shape[0],
            "T_l_opt": T_l_opt, "T_s_opt": T_s_opt,
            "log_T_ratio_opt": float(np.log(T_l_opt / T_s_opt)), "opt_nll": opt_nll,
            "T_l_base": T_l_base, "T_s_base": T_s_base, "base_nll": base_nll,
            "r_l": r_l, "r_s": r_s, "gap_r": r_l - r_s,
            "acc_l": float((z_l.argmax(1) == labels_d).float().mean()),
            "acc_s": float((z_s.argmax(1) == labels_d).float().mean()),
        })

    try:
        evaluate_dynamic_duo(
            duo, cfg, num_samples=args.num_samples, seed=args.seed,
            on_batch=_on_batch,
        )
    finally:
        if ext_l is not None:
            ext_l.remove()
        if ext_s is not None:
            ext_s.remove()

    if not batch_rows:
        print("No batches were recorded -- nothing to analyze.")
        return

    with (out_dir / "batch_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BATCH_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(batch_rows)
    print(f"\nWrote {len(batch_rows)} rows to {out_dir / 'batch_diagnostics.csv'}")

    # (1) WITHIN each corruption: does a batch-to-batch swing in the proxy
    # track a batch-to-batch swing in the optimal temperature, within one
    # otherwise-constant corruption/severity stream?
    corruptions = []
    seen = set()
    for r in batch_rows:
        if r["corruption"] not in seen:
            seen.add(r["corruption"]); corruptions.append(r["corruption"])
    corr_rows = [
        _corr_row(c, [r for r in batch_rows if r["corruption"] == c]) for c in corruptions
    ]
    # (2) ACROSS corruptions, pooled: every batch from every corruption
    # treated as one sample -- dominated by whichever effect is larger,
    # within-corruption dynamics or between-corruption differences.
    corr_rows.append(_corr_row("ALL (pooled, ignoring corruption)", batch_rows))
    # (3) ACROSS corruptions, on per-corruption MEANS only: does a proxy that
    # merely tracks "which corruptions are harder overall" (the sel_acc
    # pitfall from CLAUDE.md's Gotcha -- a proxy that's structurally biased
    # toward whichever model is better ON AVERAGE looks good here even if it
    # never once tracks a real batch-level swap) show up as strong here even
    # if weak within-corruption -- comparing this row against the
    # per-corruption rows above is exactly how to catch that.
    per_corr_means = [
        {k: float(np.mean([r[k] for r in batch_rows if r["corruption"] == c]))
         for k in ("T_l_opt", "T_s_opt", "log_T_ratio_opt", "r_l", "r_s", "gap_r")}
        for c in corruptions
    ]
    corr_rows.append(_corr_row("ACROSS corruptions (per-corruption means)", per_corr_means))

    with (out_dir / "correlations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"Wrote {out_dir / 'correlations.csv'}")

    _print_corr_table(corr_rows)
    _plot(batch_rows, corr_rows, out_dir / "proxy_vs_temperature.png")

    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
