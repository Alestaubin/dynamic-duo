#!/usr/bin/env python3
"""
TEMPORARY (deleted in plan.md Phase 9 together with the old engine it imports).

Does the new experiment engine (src/experiments) reproduce the old one (src/tta/dynamic_duo.py)?
Runs BOTH on the same ImageNet-C streams with real models, independent Tent adaptation, and compares each
corruption's accuracy and ECE for the large member, the small member and the proxy-weighted duo:

    old  setup_duo(... "proxy_weighted", "both_indep") + evaluate_dynamic_duo
    new  evaluate(RunSpec, ...): adapt the members once, cache their logits (centered fp16), replay the gate

The gate is configured as the existing paper tables were produced (EMA alpha=0.9999, beta=0.5, b_t=128, log
pool, T_L/T_S from --fixed_ts), so a match here means the new pipeline reproduces the existing numbers.

    sbatch slurm/job.sh scripts/parity_old_vs_new.py --corruptions brightness fog --num_samples 2000
    sbatch slurm/job.sh scripts/parity_old_vs_new.py --num_samples 50000   # all 15: the Phase 5 parity gate
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.calibrators.joint_fixed_TS import JointFixedTS
from src.experiments.cli import pin_threads
from src.experiments.runner import Context, ensure_members, replay
from src.experiments.specs import GateSpec, RunSpec
from src.reliability.setup import build_proxy_weighted_calibrator
from src.tta.dynamic_duo import evaluate_dynamic_duo, setup_duo
from src.utils.config import load_global
from src.utils.model import get_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", default="vitb16_rn50")
    ap.add_argument("--corruptions", nargs="+", default=None, help="default: brightness fog")
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed_ts", default="checkpoints/fixed_ts/default")
    ap.add_argument("--tol", type=float, default=None, help="max |accuracy diff|; default 3 samples per stream")
    args = ap.parse_args()
    pin_threads()

    g = load_global()
    corruptions = args.corruptions or ["brightness", "fog"]
    ctx = Context.create(args.duo, g)
    cfg = copy.deepcopy(ctx.duo)
    cfg["EVAL"]["CORRUPTIONS"] = corruptions
    tol = args.tol if args.tol is not None else 3 / args.num_samples
    ts_path = args.fixed_ts

    # ------------------------------------------------------------------ old engine
    t0 = time.time()
    (large, lp), (small, sp) = [get_model(cfg[s]["NAME"], verbose=False) for s in ("LARGE", "SMALL")]
    large, small = large.to(ctx.device), small.to(ctx.device)
    calibrator = build_proxy_weighted_calibrator(
        proxy_kind="nuclear_norm", proxy_cache=None, calib_map=None, calib_method="identity",
        filter_kind="ema", filter_kwargs={"alpha": 0.9999}, beta=0.5, prior_l=0.5, prior_s=0.5,
        base_ts=JointFixedTS.load(ts_path), csv_path=None, config=cfg,
        large_model=large, large_preprocess=lp, small_model=small, small_preprocess=sp, device=ctx.device,
        proxy_batch_size=128, pool="log",
    )
    calibrator.verbose = False
    duo = setup_duo(large, lp, small, sp, calibrator, "proxy_weighted", "both_indep", cfg, steps=1)
    old_rows = {r["corruption"]: r for r in evaluate_dynamic_duo(duo, cfg, num_samples=args.num_samples, seed=args.seed)}
    t_old = time.time() - t0
    del duo, large, small
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ new engine
    t0 = time.time()
    spec = RunSpec(args.duo, tta="tent", seed=args.seed, num_samples=args.num_samples,
                   protocol_args={"corruptions": corruptions})
    gates = [GateSpec("ours", fixed_ts=ts_path, beta=0.5, proxy_batch_size=128, filter="ema",
                      filter_kwargs={"alpha": 0.9999})]
    ensure_members(spec, ctx)
    t_members = time.time() - t0
    res = replay(spec, ctx, gates)
    t_new = time.time() - t0
    new = {(r["segment"].split("/")[0], r["series"]): r for r in res.rows}

    # ------------------------------------------------------------------ compare
    print(f"\n{'corruption':<18}{'series':<7}{'old acc':>9}{'new acc':>9}{'diff':>8}   {'old ece':>8}{'new ece':>8}")
    worst, worst_ece = 0.0, 0.0
    for c in corruptions:
        for series, key in (("large", "large"), ("small", "small"), ("ours", "duo")):
            oa, oe = old_rows[c][f"{key}/accuracy"], old_rows[c][f"{key}/ece"]
            n = new[(c, series)]
            d = abs(oa - n["accuracy"])
            worst, worst_ece = max(worst, d), max(worst_ece, abs(oe - n["ece"]))
            print(f"{c:<18}{series:<7}{100 * oa:9.2f}{100 * n['accuracy']:9.2f}{100 * (n['accuracy'] - oa):8.2f}   "
                  f"{100 * oe:8.2f}{100 * n['ece']:8.2f}")
    print(f"\nworst |accuracy diff| = {100 * worst:.3f} points (tolerance {100 * tol:.3f}), "
          f"worst |ECE diff| = {100 * worst_ece:.3f} points")
    print(f"time: old engine {t_old:.0f}s | new engine {t_new:.0f}s (members {t_members:.0f}s + replay {t_new - t_members:.0f}s)")
    ok = worst <= tol and worst_ece <= 0.005
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
