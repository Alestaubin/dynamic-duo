#!/usr/bin/env python3
"""
Paired comparisons of live reset runs: the mean difference in whole-visit accuracy of two conditions on the SAME images, its standard error over the
visits (paired: the same visit in both runs) and the number of visits in which the first condition wins. Visits are not independent (a collapse
carries over into the next visit, and the two held-out rotations hold the same images in another order), so the standard error is optimistic; it is
the noise floor below which a difference should not be read, not a confidence interval.

    held-out: the visits of rotations 0 and 2 are pooled (32 visits; each rotation's visits are paired with the same rotation's)
    test:     the visits of seeds 0 1 2 are averaged per visit first (the same 15 shifts, loaders seeded differently), then paired (15 visits)

    python scripts/41_paired_comparisons.py --stream heldout --pairs gap_fast_vs_slow:fixed1000 gap_fast_vs_slow:boundary
    python scripts/41_paired_comparisons.py --stream test --lrs 1.4e-4 5e-4 --beta 2 --pairs ...

Series: `gate` (Vanilla DD, the gated duo), `ts` (the fixed-TS duo), `large`, `small`. A condition may be `frozen`. Writes <out>/results/reset_live/<duo>/paired__<stream>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.experiments import continual as RC
from src.reset import live_report as LV
from src.reset import tuning as TU
from src.utils.config import load_global


def visits_of(folder: Path, cond: str, *, stream: str, rots, seeds, lrs, beta: float, series: str, order: int | None = None) -> np.ndarray:
    per_unit = []
    for rot in rots:
        for seed in seeds:
            stem = LV.run_stem(cond, stream=stream, rot=rot, seed=seed, lrs=lrs, proxy="nnn", b_t=256, beta=beta, order=order)
            run = RC.load_run(folder / f"{stem}.npz")
            per_unit.append(TU.visit_accuracy(run["ts" if cond == "frozen" and series == "gate" else series], run["part_n"]))
    if stream == "test":
        return np.mean(per_unit, axis=0)[None, :].ravel()               # seeds averaged per visit
    return np.concatenate(per_unit)                                      # rotations pooled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--stream", choices=("heldout", "test"), required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B, the difference A - B")
    ap.add_argument("--lrs", type=float, nargs=2, default=[1.4e-4, 5e-4])
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--series", default="gate")
    ap.add_argument("--rots", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--order", type=int, default=None)
    args = ap.parse_args()
    g = load_global()
    folder = g.path("out") / "results" / "reset_live" / f"{args.duo}__tent_bn"
    rots = args.rots if args.stream == "heldout" else [0]
    seeds = args.seeds or ([0] if args.stream == "heldout" else [0, 1, 2])
    out = {}
    print(f"{args.stream}: series {args.series}, rates {args.lrs}, beta {args.beta:g}; rotations {rots}, seeds {seeds}")
    for pair in args.pairs:
        a, b = pair.split(":")
        va, vb = (visits_of(folder, c, stream=args.stream, rots=rots, seeds=seeds, lrs=tuple(args.lrs), beta=args.beta, series=args.series, order=args.order) for c in (a, b))
        d = 100 * (va - vb)
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        out[pair] = {"mean": float(d.mean()), "se": se, "wins": int((d > 0).sum()), "visits": len(d), "a": float(100 * va.mean()), "b": float(100 * vb.mean())}
        print(f"  {a} - {b}: {d.mean():+.2f} +- {se:.2f}  ({(d > 0).sum()}/{len(d)} visits)   [{100 * va.mean():.1f} vs {100 * vb.mean():.1f}]")
    (folder / f"paired__{args.stream}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
