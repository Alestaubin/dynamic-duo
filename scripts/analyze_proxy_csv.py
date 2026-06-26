#!/usr/bin/env python3
"""
Reads a CSV produced by JointProxyAnchorCoca and prints per-corruption and
aggregate metrics:

  - R², Pearson r, Spearman ρ  between proxy score  and GT accuracy (large & small)
  - R², Pearson r, Spearman ρ  between predicted acc and GT accuracy (large & small)
  - Selection accuracy (fraction of non-tie batches where anchor == true_better)

Usage:
    python scripts/analyze_proxy_csv.py out/run_20260626_143022.csv
    python scripts/analyze_proxy_csv.py out/run_*.csv --sort sel_acc
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


# ── helpers ──────────────────────────────────────────────────────────────── #

def _corr(xs, ys):
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(x) < 3 or x.std() < 1e-8 or y.std() < 1e-8:
        return float("nan"), float("nan"), float("nan")
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    return float(pr ** 2), float(pr), float(sr)


def _f(v, w=6, d=3):
    return f"{v:{w}.{d}f}" if v == v else f"{'nan':>{w}}"


def _compute(rows):
    """Compute metrics dict from a list of CSV row dicts."""
    r_l    = [float(r["r_l"])    for r in rows]
    r_s    = [float(r["r_s"])    for r in rows]
    acc_l  = [float(r["acc_l"])  for r in rows]
    acc_s  = [float(r["acc_s"])  for r in rows]

    has_pred = "pred_l" in rows[0] and rows[0]["pred_l"] != ""
    pred_l = [float(r["pred_l"]) for r in rows] if has_pred else []
    pred_s = [float(r["pred_s"]) for r in rows] if has_pred else []

    # selection accuracy: only non-tie batches
    non_tie = [r for r in rows if r["true_better"] != "tie"]
    sel_acc = (sum(1 for r in non_tie if r["anchor_correct"] in ("True", "1", "true"))
               / len(non_tie)) if non_tie else float("nan")

    r2_proxy_l, pr_proxy_l, sp_proxy_l = _corr(r_l, acc_l)
    r2_proxy_s, pr_proxy_s, sp_proxy_s = _corr(r_s, acc_s)

    if has_pred:
        r2_pred_l, pr_pred_l, sp_pred_l = _corr(pred_l, acc_l)
        r2_pred_s, pr_pred_s, sp_pred_s = _corr(pred_s, acc_s)
    else:
        r2_pred_l = pr_pred_l = sp_pred_l = float("nan")
        r2_pred_s = pr_pred_s = sp_pred_s = float("nan")

    return {
        "n": len(rows),
        "n_nontie": len(non_tie),
        "sel_acc": sel_acc,
        "r2_proxy_l": r2_proxy_l, "r_proxy_l": pr_proxy_l, "rho_proxy_l": sp_proxy_l,
        "r2_proxy_s": r2_proxy_s, "r_proxy_s": pr_proxy_s, "rho_proxy_s": sp_proxy_s,
        "r2_pred_l":  r2_pred_l,  "r_pred_l":  pr_pred_l,  "rho_pred_l":  sp_pred_l,
        "r2_pred_s":  r2_pred_s,  "r_pred_s":  pr_pred_s,  "rho_pred_s":  sp_pred_s,
    }


# ── printing ─────────────────────────────────────────────────────────────── #

def _header():
    cols = [
        f"{'corruption':<28}",
        f"{'n':>4}",
        f"{'sel_acc':>7}",
        # proxy
        f"{'R²(prx,L)':>10}", f"{'r(prx,L)':>9}", f"{'ρ(prx,L)':>9}",
        f"{'R²(prx,S)':>10}", f"{'r(prx,S)':>9}", f"{'ρ(prx,S)':>9}",
        # predicted acc
        f"{'R²(prd,L)':>10}", f"{'r(prd,L)':>9}", f"{'ρ(prd,L)':>9}",
        f"{'R²(prd,S)':>10}", f"{'r(prd,S)':>9}", f"{'ρ(prd,S)':>9}",
    ]
    line = " ".join(cols)
    print(line)
    print("-" * len(line))


def _row(label, m):
    parts = [
        f"{label:<28}",
        f"{m['n']:>4}",
        f"{_f(m['sel_acc'], 7, 3)}",
        f"{_f(m['r2_proxy_l'], 10, 3)}", f"{_f(m['r_proxy_l'], 9, 3)}", f"{_f(m['rho_proxy_l'], 9, 3)}",
        f"{_f(m['r2_proxy_s'], 10, 3)}", f"{_f(m['r_proxy_s'], 9, 3)}", f"{_f(m['rho_proxy_s'], 9, 3)}",
        f"{_f(m['r2_pred_l'],  10, 3)}", f"{_f(m['r_pred_l'],  9, 3)}", f"{_f(m['rho_pred_l'],  9, 3)}",
        f"{_f(m['r2_pred_s'],  10, 3)}", f"{_f(m['r_pred_s'],  9, 3)}", f"{_f(m['rho_pred_s'],  9, 3)}",
    ]
    print(" ".join(parts))


# ── main ─────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="CSV file(s) produced by JointProxyAnchorCoca")
    ap.add_argument("--sort", default="corruption",
                    choices=["corruption", "sel_acc", "r2_proxy_l", "r2_pred_l"],
                    help="Column to sort per-corruption rows by")
    args = ap.parse_args()

    for csv_path in args.csv:
        rows = []
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            print(f"{csv_path}: empty", file=sys.stderr)
            continue

        print(f"\n=== {Path(csv_path).name} ===\n")

        # group by corruption
        by_corr: dict[str, list] = {}
        for r in rows:
            by_corr.setdefault(r["corruption"], []).append(r)

        per_corr = {c: _compute(rs) for c, rs in by_corr.items()}

        # sort
        if args.sort == "corruption":
            order = sorted(per_corr)
        else:
            order = sorted(per_corr, key=lambda c: per_corr[c][args.sort], reverse=True)

        _header()
        for corr in order:
            _row(corr, per_corr[corr])

        metric_keys = [k for k in next(iter(per_corr.values())) if k not in ("n", "n_nontie")]

        # macro-average across corruptions (avoids Simpson's paradox from pooling)
        macro = {"n": sum(m["n"] for m in per_corr.values()),
                 "n_nontie": sum(m["n_nontie"] for m in per_corr.values())}
        for k in metric_keys:
            vals = [m[k] for m in per_corr.values() if m[k] == m[k]]  # skip nan
            macro[k] = float(np.mean(vals)) if vals else float("nan")

        # pooled across all rows (inflated by between-corruption variation)
        pooled = _compute(rows)

        print("-" * 140)
        _row("MACRO-AVG", macro)
        _row("POOLED", pooled)
        print()


if __name__ == "__main__":
    main()
