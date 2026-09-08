#!/usr/bin/env python3
"""
scripts/screen_duo_candidates.py
=================================
Cheap, frozen (no adaptation) sanity check for a candidate model pair BEFORE
investing in a full duo experiment: is there any complementarity to exploit
at all?

For each --configs entry (a normal cfgs/dynamic_duo_config*.yaml -- only
LARGE/SMALL model names + paths/BS are read from it), both models are run
frozen (no TENT adaptation; batch statistics only, calibration_mode
irrelevant here -- there is no joint calibrator, just two independent
forward passes) over every corruption/severity in the _CORRUPTIONS/
_SEVERITIES constants below (hardcoded here, deliberately NOT each config's
own EVAL.CORRUPTIONS/EVAL.SEVERITIES, so that candidate duos are always
screened against exactly the same data regardless of what any one config's
EVAL block happens to list), then chunked into
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

A fixed JointFixedTS is fit once per duo on clean val logits (or loaded from
--fixed_ts_dir if already fit) to supply T_l/T_s -- every accuracy column
below combines T-scaled logits w*(z_l/T_l) + (1-w)*(z_s/T_s), so they form a
ladder of increasingly optimistic (oracle, in-sample) uses of the SAME base
calibration:

  acc_fixed     -- w=0.5 always (exactly JointFixedTS.calibrate()). The
                   baseline every other column below adds a mixing weight on
                   top of.
  acc_gate      -- best single w per chunk (grid search over w in [0,1],
                   maximizing that chunk's own accuracy): the per-batch
                   ceiling a proxy-driven gate like JointProxyWeighted is
                   chasing.
  acc_persample -- best w per INDIVIDUAL SAMPLE (any grid w that gets that
                   sample right counts): the granularity ceiling -- how much
                   headroom is left if the gate could react per-sample
                   instead of per-batch.
  acc_hard      -- best single MODEL per chunk (no mixing, w in {0, 1}):
                   what a hard per-batch selector (no soft combination) could
                   achieve; the gap to acc_gate is what soft weighting buys
                   over hard switching.
  acc_either    -- fraction of samples where at least one model is correct
                   (descriptive only, not w-reachable in general -- e.g. a
                   sample only the small model gets right can be lost by any
                   w > 0 if the large model's wrong logit dominates it).

Reuses get_model_logits' per-(model, corruption, severity) cache (the same
one oracle_ts and compare_calibrators --use_cache rely on) so re-running
this against a different --batch_size / --num_samples on the same duo is
fast after the first pass, and running two duos that happen to share a
model (e.g. two configs both using deit_small) shares that model's cache
entries too.

Usage
-----
    python scripts/screen_duo_candidates.py \
        --configs cfgs/dynamic_duo_config_convnext_swin.yaml \
                  cfgs/dynamic_duo_config_convnext_deit.yaml \
                  cfgs/dynamic_duo_config_resnet_deit.yaml \
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
from src.calibrators.joint_fixed_TS import JointFixedTS
from scripts._cli import add_num_samples_arg, add_seed_arg, add_out_dir_run_name_args

# Hardcoded (not read from each config's own EVAL.CORRUPTIONS/SEVERITIES) so
# every --configs entry is screened against exactly the same data -- this is
# the 15-corruption, severity-5 EVAL set already shared by every screening
# candidate config (dynamic_duo_config_{vitb_resnet,convnext_*,resnet_*}.yaml).
_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur",
]
_SEVERITIES = [5]

_SUMMARY_ROW_FIELDS = [
    "duo", "corruption", "severity", "n_samples", "n_chunks",
    "win_rate_large", "win_rate_small", "n_large_win", "n_small_win", "n_tie",
    "oracle_delta_acc_mean_pp", "oracle_delta_acc_median_pp",
    "disagree_rate", "acc_large", "acc_small",
    "acc_fixed", "acc_gate", "acc_persample", "acc_hard", "acc_either",
]


def _chunks(n: int, size: int) -> list[slice]:
    """Consecutive windows of `size` samples, trailing remainder kept as a
    smaller final chunk (same convention as scripts/sweep_proxies.py's
    _chunks and JointProxyWeighted's proxy-batch buffering)."""
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _w_grid(steps: int) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, steps)


def _chunk_stats(
    z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor, batch_size: int, w_grid: torch.Tensor,
) -> dict:
    """Win-rate split / oracle Δacc / disagreement rate / acc_* ceilings for
    one (corruption, severity) stream, chunked at `batch_size`. Returns the
    aggregate dict PLUS raw per-chunk lists (needed to pool an exact mean/
    median across corruptions later, not just an average-of-averages).

    z_l, z_s must already be scaled by the fixed_ts temperatures (T_l, T_s)
    -- see _fit_or_load_fixed_ts -- so that w=0.5 below reproduces
    JointFixedTS.calibrate() exactly and the w-grid search only ever layers
    a mixing weight on top of that fixed calibration, never re-fitting
    temperature and weight jointly (unidentifiable together)."""
    pred_l, pred_s = z_l.argmax(1), z_s.argmax(1)
    correct_l, correct_s = (pred_l == labels), (pred_s == labels)
    n = z_l.shape[0]

    acc_fixed = float((((z_l + z_s) / 2.0).argmax(1) == labels).float().mean())
    acc_either = float((correct_l | correct_s).float().mean())

    # correct_grid[k] = which samples grid point w_grid[k] gets right. Reused
    # for acc_persample (OR across the grid -- a different w per SAMPLE) and,
    # chunked below, acc_gate (MAX across the grid within one chunk -- a
    # different w per BATCH).
    correct_grid = torch.stack([
        ((w * z_l + (1.0 - w) * z_s).argmax(1) == labels) for w in w_grid
    ])  # (K, n)
    acc_persample = float(correct_grid.any(dim=0).float().mean())

    n_large_win = n_small_win = n_tie = 0
    deltas, gate_deltas, hard_deltas = [], [], []
    for sl in _chunks(n, batch_size):
        acc_l = float(correct_l[sl].float().mean())
        acc_s = float(correct_s[sl].float().mean())
        oracle_acc = float((correct_l[sl] | correct_s[sl]).float().mean())
        deltas.append(oracle_acc - max(acc_l, acc_s))
        hard_deltas.append(max(acc_l, acc_s))
        gate_deltas.append(float(correct_grid[:, sl].float().mean(dim=1).max()))
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
        "acc_fixed": acc_fixed,
        "acc_gate": float(np.mean(gate_deltas)) if gate_deltas else float("nan"),
        "acc_persample": acc_persample,
        "acc_hard": float(np.mean(hard_deltas)) if hard_deltas else float("nan"),
        "acc_either": acc_either,
        # pooled across corruptions by the caller, then dropped
        "_deltas": deltas, "_gate_deltas": gate_deltas, "_hard_deltas": hard_deltas,
    }


def _pool(rows: list[dict]) -> dict:
    """Combine several _chunk_stats dicts (one per corruption/severity) into
    one overall row -- exact (not an average of averages): counts sum
    directly, rates/accuracies are n-weighted, and the oracle Δacc mean/
    median are recomputed over every individual chunk's raw delta, pooled."""
    n_total = sum(r["n_samples"] for r in rows)
    all_deltas = [d for r in rows for d in r["_deltas"]]
    all_gate_deltas = [d for r in rows for d in r["_gate_deltas"]]
    all_hard_deltas = [d for r in rows for d in r["_hard_deltas"]]
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
        "acc_fixed": _wavg("acc_fixed"),
        "acc_gate": float(np.mean(all_gate_deltas)) if all_gate_deltas else float("nan"),
        "acc_persample": _wavg("acc_persample"),
        "acc_hard": float(np.mean(all_hard_deltas)) if all_hard_deltas else float("nan"),
        "acc_either": _wavg("acc_either"),
    }


def _fit_or_load_fixed_ts(cfg: dict, duo_name: str, args, device: torch.device) -> JointFixedTS:
    """Load a previously-fit JointFixedTS for this duo from --fixed_ts_dir if
    present; otherwise fit fresh temperatures on clean val logits (same
    approach as scripts/fit_fixed_ts.py's --clean_only mode) and, if
    --fixed_ts_dir was given, save it there for reuse across runs/duos."""
    save_dir = Path(args.fixed_ts_dir) / duo_name if args.fixed_ts_dir else None
    if save_dir is not None and (save_dir / "config.json").exists():
        return JointFixedTS.load(str(save_dir))

    common = dict(
        val_dir=cfg["VAL_DIR"], test_dir=cfg["TEST_DIR"], cache_dir=args.cache_dir,
        batch_size=cfg["BS"], num_workers=cfg["WORKERS"], device=device,
        tent_mode=True, seed=args.seed, verbose=args.verbose,
    )
    z_l, y_l = get_model_logits(model_name=cfg["LARGE"]["NAME"], norm_type=cfg["LARGE"]["NORM"], **common)
    z_s, y_s = get_model_logits(model_name=cfg["SMALL"]["NAME"], norm_type=cfg["SMALL"]["NORM"], **common)
    assert torch.equal(y_l, y_s), f"logit cache desync fitting fixed_ts for {duo_name}: large/small val labels differ"

    fixed_ts = JointFixedTS(verbose=args.verbose)
    fixed_ts.tune(logits_l=z_l, logits_s=z_s, labels=y_l)
    print(f"  fixed_ts fit on {len(y_l):,} clean val samples: Tl={fixed_ts.Tl.item():.4f} Ts={fixed_ts.Ts.item():.4f}")

    if save_dir is not None:
        fixed_ts.save(str(save_dir), trained_on={
            "large_model": cfg["LARGE"]["NAME"], "small_model": cfg["SMALL"]["NAME"], "clean_val": True,
        })
    return fixed_ts


def _screen_duo(config_path: str, args, device: torch.device, w_grid: torch.Tensor) -> list[dict]:
    cfg = load_config(config_path)
    batch_size = args.batch_size or cfg["BS"]
    duo_name = f"{cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}"
    print(f"\n{'#' * 78}\n# {duo_name}  (b={batch_size})\n{'#' * 78}")

    fixed_ts = _fit_or_load_fixed_ts(cfg, duo_name, args, device)
    Tl, Ts = fixed_ts.Tl.item(), fixed_ts.Ts.item()

    rows: list[dict] = []
    for severity in _SEVERITIES:
        for corruption in _CORRUPTIONS:
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

            # Scaling by T_l/T_s doesn't change either model's own argmax
            # (positive scalar), so win-rate/oracle/disagree above are
            # unaffected -- only the acc_fixed/gate/persample/hard combos
            # below need the T-scaled logits (see _chunk_stats docstring).
            stats = _chunk_stats(z_l / Tl, z_s / Ts, labels_l, batch_size, w_grid)
            print(
                f"  {corruption}/s{severity}: win_rate(L/S)={stats['win_rate_large']:.2f}/"
                f"{stats['win_rate_small']:.2f}  oracle_Δacc={stats['oracle_delta_acc_mean_pp']:+.2f}pp  "
                f"disagree={stats['disagree_rate']:.3f}  acc(L/S)={stats['acc_large']:.3f}/{stats['acc_small']:.3f}\n"
                f"    acc_fixed={stats['acc_fixed']:.3f}  acc_gate={stats['acc_gate']:.3f}  "
                f"acc_persample={stats['acc_persample']:.3f}  acc_hard={stats['acc_hard']:.3f}  "
                f"acc_either={stats['acc_either']:.3f}"
            )
            row = {"duo": duo_name, "corruption": corruption, "severity": severity, **stats}
            rows.append(row)

    overall = {"duo": duo_name, "corruption": "ALL", "severity": "-", **_pool(rows)}
    print(
        f"  {'-' * 74}\n  OVERALL: win_rate(L/S)={overall['win_rate_large']:.2f}/"
        f"{overall['win_rate_small']:.2f}  oracle_Δacc={overall['oracle_delta_acc_mean_pp']:+.2f}pp "
        f"(median {overall['oracle_delta_acc_median_pp']:+.2f}pp)  disagree={overall['disagree_rate']:.3f}  "
        f"acc(L/S)={overall['acc_large']:.3f}/{overall['acc_small']:.3f}\n"
        f"  acc_fixed={overall['acc_fixed']:.3f}  acc_gate={overall['acc_gate']:.3f}  "
        f"acc_persample={overall['acc_persample']:.3f}  acc_hard={overall['acc_hard']:.3f}  "
        f"acc_either={overall['acc_either']:.3f}"
    )
    for r in rows:
        r.pop("_deltas", None)
        r.pop("_gate_deltas", None)
        r.pop("_hard_deltas", None)
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

    print("\n" + "=" * 100)
    print(f"{'duo':<30}{'acc_fixed':>11}{'acc_gate':>11}{'acc_persample':>15}{'acc_hard':>11}{'acc_either':>13}")
    print("-" * 100)
    for r in overall_rows:
        print(
            f"{r['duo']:<30}{r['acc_fixed']:>11.3f}{r['acc_gate']:>11.3f}"
            f"{r['acc_persample']:>15.3f}{r['acc_hard']:>11.3f}{r['acc_either']:>13.3f}"
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
    p.add_argument("--fixed_ts_dir", type=str, default=None,
                    help="Directory of per-duo JointFixedTS checkpoints (subfolder named after "
                         "the duo, e.g. '<fixed_ts_dir>/<large>+<small>/config.json'). If a "
                         "checkpoint exists there it's loaded; otherwise one is fit fresh on "
                         "clean val logits (scripts/fit_fixed_ts.py's --clean_only recipe) and, "
                         "if this flag is given, saved there for reuse. Omit to always fit fresh "
                         "without saving.")
    p.add_argument("--w_grid_steps", type=int, default=51,
                    help="Number of w in [0,1] grid points searched for acc_gate/acc_persample "
                         "(default 51 -> 0.02 resolution).")
    p.add_argument("--verbose", action="store_true",
                    help="Print get_model_logits' own cache hit/miss + progress bar lines.")
    add_out_dir_run_name_args(p, out_dir_default="out/duo_screening")
    args = p.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  {len(args.configs)} candidate duo(s)  |  out_dir: {out_dir}")

    w_grid = _w_grid(args.w_grid_steps)
    all_rows: list[dict] = []
    for config_path in args.configs:
        all_rows.extend(_screen_duo(config_path, args, device, w_grid))

    with (out_dir / "duo_screening.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_ROW_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_dir / 'duo_screening.csv'}")

    _print_comparison(all_rows)
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
