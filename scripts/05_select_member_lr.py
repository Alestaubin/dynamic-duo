#!/usr/bin/env python3
"""
Choose each member's learning rate from stored sweep logits, offline (Phase 3, Stage D; src/experiments/member_selection.py).

Reads the logits that `scripts/04_cache_logits.py --grid` stored for an algorithm and picks, for the large and the small member
separately, the learning rate with the best STANDALONE mean accuracy over the held-out streams (ties: the smaller one), reporting the
worst held-out stream next to it and warning when the best value is the first or last one tried (the optimum may lie outside the grid).

    python scripts/05_select_member_lr.py --duo vitb16_rn50v1 --tta sar --seeds 0
    python scripts/05_select_member_lr.py --duo vitb16_rn50v1 --tta sar --lrs_large 5e-5 1e-4 2.5e-4 5e-4 1e-3 2.5e-3   # an extended grid
    ... --write_tuned      also record the winners in cfgs/duo_lrs/<duo>__indep.yaml (every duo that uses the model), which every later
                           run of that algorithm reads

Writes <out>/select_member_lr__<duo>__<tta>__seed<seed>.json with every number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import cli
from src.experiments.algorithms import ALGORITHMS
from src.experiments.grids import lr_grid
from src.experiments.logit_runs import stream_segments
from src.experiments.logit_store import LogitStore
from src.experiments.member_selection import select, tuned_record, write_tuned
from src.experiments.specs import RunSpec


def main() -> int:
    ap = cli.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--streams", nargs="+", default=["heldout"], metavar="NAME")
    ap.add_argument("--lrs_large", nargs="+", type=float, default=None, help="default: the model's grid")
    ap.add_argument("--lrs_small", nargs="+", type=float, default=None, help="default: the model's grid")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--write_tuned", action="store_true", help="record the winners in cfgs/duo_lrs/ (every duo that uses the model)")
    args = ap.parse_args()
    if args.tta not in ALGORITHMS:
        ap.error(f"--tta must be one of {ALGORITHMS}")

    ctx, seeds, out = cli.setup(args, "select_member_lr")
    segments = stream_segments(args.streams, ctx.g)
    stems = [s.file_stem for s in segments]
    store = LogitStore.from_context(ctx, None)
    lrs = {"large": args.lrs_large or lr_grid(ctx.duo["LARGE"]["NAME"]), "small": args.lrs_small or lr_grid(ctx.duo["SMALL"]["NAME"])}
    status = 0
    for seed in seeds:
        spec = RunSpec(args.duo, args.tta, args.tta_kwargs or {}, "both_indep", "episodic", seed=seed, num_samples=args.num_samples)
        try:
            res = select(store, ctx, spec, lrs, stems, args.momentum)
        except FileNotFoundError as e:
            print(f"seed {seed}: {e}")
            status = 1
            continue
        for side, r in res.items():
            print(f"\n== {args.tta}, seed {seed}, {side} member ({ctx.duo[side.upper()]['NAME']}), standalone accuracy on {stems}")
            print(f"{'lr':>9} {'macro':>8} {'worst':>8}  worst stream")
            for row in r["rows"]:
                mark = "  <- best" if row["lr"] == r["best_lr"] else ""
                print(f"{row['lr']:9.2e} {100 * row['macro']:8.2f} {100 * row['worst']:8.2f}  {row['worst_stream']}{mark}")
            if r["edge"]:
                print(f"WARNING: the best learning rate ({r['best_lr']:g}) is the {r['edge_side']} one tried: the optimum may lie "
                      f"outside the grid (add a point with --lrs_{side})")
        name = f"select_member_lr__{args.duo}__{args.tta}__seed{seed}"
        (out / f"{name}.json").write_text(json.dumps({"duo": args.duo, "tta": args.tta, "seed": seed, "streams": stems, "result": res},
                                                     indent=1))
        if args.write_tuned:
            for side, r in res.items():
                cfg = ctx.duo[side.upper()]
                paths = write_tuned(ctx.g.path("duo_lrs"),
                                    tuned_record(r, model=cfg["NAME"], algorithm=args.tta, norm=cfg["NORM"], batch_size=ctx.batch_size,
                                                 stems=stems, seed=seed, momentum=args.momentum))
                for path in paths:
                    print(f"wrote {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
