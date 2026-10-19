#!/usr/bin/env python3
"""
Run COCA, the coupled small/large duo method, on a duo's streams (Phase 3, Stage D; src/experiments/coca_run.py).

COCA's prediction depends on a temperature fitted batch after batch, so it cannot be replayed from stored logits: it is scored online
and its results are written as small tables instead of going to the logit store. The official code is used in place from
third_party/COCA (it has no license, so nothing is copied). Same learning-rate grids as the other algorithms (SGD, momentum 0.9):
run i uses the i-th value of each model's grid, here the two together because COCA couples the members, so the best pair is read off
the runs themselves.

    --streams heldout|test|clean|<natural dataset>      (default heldout; 1000-class streams only)
    --grid [i ...]                                      grid points (default: all five); or --lr_large LR --lr_small LR
    python scripts/06_run_coca.py --duo vitb16_rn50v1 --streams heldout --grid --seeds 0 --shard auto   # a SLURM array

Writes <out>/coca__<duo>__seed<seed>__lrL<lr>_lrS<lr>.rows.csv (accuracy, ECE, NLL of the combination and of each member, per stream)
and .tau.json (the temperature after every batch). Launch arrays only with the author's approval (plan.md, Phase 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import cli
from src.experiments.coca_run import run_coca
from src.experiments.grids import lr_grid
from src.experiments.logit_runs import stream_segments


def main() -> int:
    ap = cli.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
                             tta=False)
    ap.add_argument("--streams", nargs="+", default=["heldout"], metavar="NAME")
    ap.add_argument("--grid", nargs="*", type=int, default=None, metavar="I")
    ap.add_argument("--lr_large", type=float, default=None)
    ap.add_argument("--lr_small", type=float, default=None)
    args = ap.parse_args()
    if args.grid is not None and (args.lr_large is not None or args.lr_small is not None):
        ap.error("--grid and --lr_large/--lr_small are alternatives")
    if (args.lr_large is None) != (args.lr_small is None):
        ap.error("--lr_large and --lr_small go together")
    if args.grid is None and args.lr_large is None:
        args.grid = []                                   # default: the whole grid

    ctx, seeds, out = cli.setup(args, "run_coca")
    segments = stream_segments(args.streams, ctx.g)
    l_grid, s_grid = lr_grid(ctx.duo["LARGE"]["NAME"]), lr_grid(ctx.duo["SMALL"]["NAME"])
    jobs = []
    for seed in seeds:
        if args.grid is not None:
            for i in (args.grid or range(len(l_grid))):
                if not 0 <= i < len(l_grid):
                    ap.error(f"--grid index {i} outside 0..{len(l_grid) - 1}")
                jobs.append((seed, f"seed{seed}/grid{i}", l_grid[i], s_grid[i]))
        else:
            jobs.append((seed, f"seed{seed}/lr{args.lr_large:g}+{args.lr_small:g}", args.lr_large, args.lr_small))
    jobs = cli.select_jobs(jobs, lambda j: j[1], args.shard, args.only)
    print(f"COCA on {args.duo}: {len(segments)} streams x {len(jobs)} runs, {args.num_samples or 'all'} images per stream")
    for seed, label, lr_l, lr_s in jobs:
        name = f"coca__{args.duo}__seed{seed}__lrL{lr_l:g}_lrS{lr_s:g}"
        if (out / f"{name}.rows.csv").is_file() and not args.num_samples:
            print(f"{label}: done already ({name}), skipped")
            continue
        res = run_coca(ctx, segments, lr_large=lr_l, lr_small=lr_s, seed=seed, num_samples=args.num_samples)
        df = pd.DataFrame(res["rows"])
        df.to_csv(out / f"{name}.rows.csv", index=False)
        (out / f"{name}.tau.json").write_text(json.dumps(res["tau"]))
        piv = df.pivot(index="segment", columns="series", values="accuracy")
        print(f"\n== {label}: large lr {lr_l:g}, small lr {lr_s:g}\n" + (100 * piv[["coca", "large", "small"]]).round(2).to_string()
              + f"\nmacro: " + ", ".join(f"{s} {100 * piv[s].mean():.2f}" for s in ("coca", "large", "small")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
