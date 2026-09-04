#!/usr/bin/env python3
"""
scripts/screen_duo_candidates.py
=================================
Cheap, frozen (no adaptation) sanity check for a candidate model pair BEFORE
investing in a full duo experiment: is there any complementarity to exploit
at all?

For each --configs entry (a normal cfgs/dynamic_duo_config*.yaml -- LARGE/
SMALL model names + EVAL corruptions/severities), both models are run frozen
(no TENT adaptation; batch statistics only, same as calibration_mode
irrelevant here -- there is no joint calibrator, just two independent
forward passes) over every EVAL corruption/severity, then chunked into
consecutive windows of cfg['BS'] samples (b=128 by default, reset at every
corruption/severity boundary so no chunk straddles two streams) to report,
per chunk and pooled:

  1. win-rate split: of the chunks where large and small aren't exactly
     tied, what fraction does each model actually win (by accuracy)? A
     healthy complementary pair should be close to 50/50 -- if one model
     wins ~100% of chunks, the "duo" is just carrying one model and a joint
     calibrator has nothing to gate between.
  2. oracle Δacc: per chunk, (oracle accuracy -- fraction where EITHER model
     is correct) minus (the better single model's accuracy in that same
     chunk), averaged in percentage points. This is the headroom a perfect
     per-chunk selector could capture beyond just always using the stronger
     model -- the ceiling any joint calibrator/gate is chasing.
  3. disagreement rate: fraction of samples where the two models' top-1
     predictions differ (agreement is a NECESSARY but not sufficient
     condition for the oracle Δacc to be nonzero -- can't disagree, can't
     complement).

Reuses get_model_logits' per-(model, corruption, severity) cache (the same
one oracle_ts and compare_calibrators --use_cache rely on) so re-running
this against a different --batch_size / --num_samples on the same duo is
fast after the first pass, and running two duos that happen to share a
model (e.g. two configs both using deit_small) shares that model's cache
entries too.

Usage
-----
    python scripts/screen_duo_candidates.py \\
        --configs cfgs/dynamic_duo_config_convnext_swin.yaml \\
                  cfgs/dynamic_duo_config_convnext_deit.yaml \\
                  cfgs/dynamic_duo_config_resnet_deit.yaml \\
        --num_samples 10000
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.utils.data import load_config
from src.utils.logits import get_model_logits
from scripts._cli import add_num_samples_arg, add_seed_arg, add_out_dir_run_name_args

_SUMMARY_ROW_FIELDS = [
    "duo", "corruption", "severity", "n_samples", "n_chunks",
    "win_rate_large", "win_rate_small", "n_large_win", "n_small_win", "n_tie",
    "oracle_delta_acc_mean_pp", "oracle_delta_acc_median_pp",
    "disagree_rate", "acc_large", "acc_small",
]


def _chunks(n: int, size: int) -> list[slice]:
    """Consecutive windows of `size` samples, trailing remainder kept as a
    smaller final chunk (same convention as scripts/sweep_proxies.py's
    _chunks and JointProxyWeighted's proxy-batch buffering)."""
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _chunk_stats(z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor, batch_size: int) -> dict:
    """Win-rate split / oracle Δacc / disagreement rate for one (corruption,
    severity) stream, chunked at `batch_size`. Returns the aggregate dict
    PLUS the raw per-chunk delta list (needed to pool an exact mean/median
    across corruptions later, not just an average-of-averages)."""
    pred_l, pred_s = z_l.argmax(1), z_s.argmax(1)
    correct_l, correct_s = (pred_l == labels), (pred_s == labels)
    n = z_l.shape[0]

    n_large_win = n_small_win = n_tie = 0
    deltas = []
    for sl in _chunks(n, batch_size):
        acc_l = float(correct_l[sl].float().mean())
        acc_s = float(correct_s[sl].float().mean())
        oracle_acc = float((correct_l[sl] | correct_s[sl]).float().mean())
        deltas.append(oracle_acc - max(acc_l, acc_s))
        if abs(acc_l - acc_s) < 1e-9:
            n_tie += 1
        elif acc_l > acc_s:
            n_large_win += 1
        else:
            n_small_win += 1

    n_decided = n_large_win + n_small_win
    return {
        "n_samples": n, "n_chunks": len(deltas),
        "n_large_win": n_large_win, "n_small_win": n_small_win, "n_tie": n_tie,
        "win_rate_large": n_large_win / n_decided if n_decided else float("nan"),
        "win_rate_small": n_small_win / n_decided if n_decided else float("nan"),
        "oracle_delta_acc_mean_pp": 100.0 * float(np.mean(deltas)) if deltas else float("nan"),
        "oracle_delta_acc_median_pp": 100.0 * float(np.median(deltas)) if deltas else float("nan"),
        "disagree_rate": float((pred_l != pred_s).float().mean()),
        "acc_large": float(correct_l.float().mean()), "acc_small": float(correct_s.float().mean()),
        "_deltas": deltas,  # pooled across corruptions by the caller, then dropped
    }


def _pool(rows: list[dict]) -> dict:
    """Combine several _chunk_stats dicts (one per corruption/severity) into
    one overall row -- exact (not an average of averages): counts sum
    directly, rates/accuracies are n-weighted, and the oracle Δacc mean/
    median are recomputed over every individual chunk's raw delta, pooled."""
    n_total = sum(r["n_samples"] for r in rows)
    all_deltas = [d for r in rows for d in r["_deltas"]]
    n_large_win = sum(r["n_large_win"] for r in rows)
    n_small_win = sum(r["n_small_win"] for r in rows)
    n_tie = sum(r["n_tie"] for r in rows)
    n_decided = n_large_win + n_small_win

    def _wavg(key: str) -> float:
        return sum(r[key] * r["n_samples"] for r in rows) / n_total if n_total else float("nan")

    return {
        "n_samples": n_total, "n_chunks": sum(r["n_chunks"] for r in rows),
        "n_large_win": n_large_win, "n_small_win": n_small_win, "n_tie": n_tie,
        "win_rate_large": n_large_win / n_decided if n_decided else float("nan"),
        "win_rate_small": n_small_win / n_decided if n_decided else float("nan"),
        "oracle_delta_acc_mean_pp": 100.0 * float(np.mean(all_deltas)) if all_deltas else float("nan"),
        "oracle_delta_acc_median_pp": 100.0 * float(np.median(all_deltas)) if all_deltas else float("nan"),
        "disagree_rate": _wavg("disagree_rate"),
        "acc_large": _wavg("acc_large"), "acc_small": _wavg("acc_small"),
    }


def _screen_duo(config_path: str, args, device: torch.device) -> list[dict]:
    cfg = load_config(config_path)
    batch_size = args.batch_size or cfg["BS"]
    duo_name = f"{cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}"
    print(f"\n{'#' * 78}\n# {duo_name}  (b={batch_size})\n{'#' * 78}")

    rows: list[dict] = []
    for severity in cfg["EVAL"]["SEVERITIES"]:
        for corruption in cfg["EVAL"]["CORRUPTIONS"]:
            common = dict(
                val_dir=cfg["VAL_DIR"], test_dir=cfg["TEST_DIR"], cache_dir=args.cache_dir,
                batch_size=cfg["BS"], num_workers=cfg["WORKERS"],
                corruption=corruption, severity=severity, device=device,
                tent_mode=True, seed=args.seed, num_samples=args.num_samples, verbose=args.verbose,
            )
            z_l, labels_l = get_model_logits(model_name=cfg["LARGE"]["NAME"], norm_type=cfg["LARGE"]["NORM"], **common)
            z_s, labels_s = get_model_logits(model_name=cfg["SMALL"]["NAME"], norm_type=cfg["SMALL"]["NORM"], **common)
            assert torch.equal(labels_l, labels_s), \
                f"logit cache desync for {duo_name} {corruption}/s{severity}: large/small labels differ"

            stats = _chunk_stats(z_l, z_s, labels_l, batch_size)
            print(
                f"  {corruption}/s{severity}: win_rate(L/S)={stats['win_rate_large']:.2f}/"
                f"{stats['win_rate_small']:.2f}  oracle_Δacc={stats['oracle_delta_acc_mean_pp']:+.2f}pp  "
                f"disagree={stats['disagree_rate']:.3f}  acc(L/S)={stats['acc_large']:.3f}/{stats['acc_small']:.3f}"
            )
            row = {"duo": duo_name, "corruption": corruption, "severity": severity, **stats}
            rows.append(row)

    overall = {"duo": duo_name, "corruption": "ALL", "severity": "-", **_pool(rows)}
    print(
        f"  {'-' * 74}\n  OVERALL: win_rate(L/S)={overall['win_rate_large']:.2f}/"
        f"{overall['win_rate_small']:.2f}  oracle_Δacc={overall['oracle_delta_acc_mean_pp']:+.2f}pp "
        f"(median {overall['oracle_delta_acc_median_pp']:+.2f}pp)  disagree={overall['disagree_rate']:.3f}  "
        f"acc(L/S)={overall['acc_large']:.3f}/{overall['acc_small']:.3f}"
    )
    for r in rows:
        r.pop("_deltas", None)
    rows.append(overall)
    return rows


def _print_comparison(all_rows: list[dict]) -> None:
    overall_rows = [r for r in all_rows if r["corruption"] == "ALL"]
    if not overall_rows:
        return
    print("\n" + "=" * 100)
    print(f"{'duo':<30}{'win_rate L/S':>16}{'oracle Δacc (pp)':>20}{'disagree_rate':>16}{'acc L/S':>18}")
    print("-" * 100)
    for r in overall_rows:
        print(
            f"{r['duo']:<30}"
            f"{r['win_rate_large']:>7.2f}/{r['win_rate_small']:<7.2f} "
            f"{r['oracle_delta_acc_mean_pp']:>+10.2f}pp"
            f"{'':<6}{r['disagree_rate']:>10.3f}      "
            f"{r['acc_large']:.3f}/{r['acc_small']:.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--configs", type=str, nargs="+", required=True,
                    help="One or more duo config YAMLs (cfgs/dynamic_duo_config*.yaml) -- "
                         "each is screened independently and compared side by side at the end.")
    add_num_samples_arg(p, default=10000)
    add_seed_arg(p)
    p.add_argument("--batch_size", type=int, default=None,
                    help="Chunk size b for the win-rate/oracle-Δacc computation. Defaults to "
                         "each config's own cfg['BS'] (128 in every existing config).")
    p.add_argument("--cache_dir", type=str, default="cache/logits",
                    help="Per-(model, corruption, severity) logit cache -- same directory "
                         "get_model_logits uses elsewhere (oracle_ts, compare_calibrators "
                         "--use_cache), so a model shared across --configs is only run once.")
    p.add_argument("--verbose", action="store_true",
                    help="Print get_model_logits' own cache hit/miss + progress bar lines.")
    add_out_dir_run_name_args(p, out_dir_default="out/duo_screening")
    args = p.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  {len(args.configs)} candidate duo(s)  |  out_dir: {out_dir}")

    all_rows: list[dict] = []
    for config_path in args.configs:
        all_rows.extend(_screen_duo(config_path, args, device))

    with (out_dir / "duo_screening.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_ROW_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_dir / 'duo_screening.csv'}")

    _print_comparison(all_rows)
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
