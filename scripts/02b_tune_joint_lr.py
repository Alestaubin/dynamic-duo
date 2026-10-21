#!/usr/bin/env python3
"""
Tune the learning rates of BOTH members of the duo jointly, for the Joint mode of script 03 (both members minimize the
entropy of the static duo's combined prediction), on the 4 held-out corruptions (speckle noise, Gaussian blur, spatter,
saturate; severity 5; none is a test corruption).

Script 02 tunes one model alone, on its own entropy; with the duo's entropy as the signal adaptation should be steadier
and tolerate larger rates, and the two rates interact. So the grid is over PAIRS (lr_large, lr_small): by default 1, 2, 4,
8 and 16 times each member's CURRENT rate (the tuned one of cfgs/duo_lrs/<duo>__indep.yaml, else its default), 25 pairs. Only the learning
rate changes; the rest of each member's OPTIM stays. --mults, or --lrs_large / --lrs_small for explicit values, change it.

A grid point is one Joint run over the 4 full streams (batch size as in the protocol; both members reset to source
weights before every stream). Score = the mean over the 4 streams of the static duo's accuracy (z_L / T_L + z_S / T_S) / 2
(--seeds, default 0 as in script 02, is the stream-order seed; with several, their scores are averaged). The worst stream and each member's own accuracy are reported next to it. The frozen duo and Independent both at the
current rates are run as references, so the report says what the Joint signal buys at equal rates and what larger rates
buy on top. A warning says when the best rate of a member is the smallest or the largest tried. A run whose logits go
non-finite (rates far too large) is recorded as diverged and ranks last.

T_L and T_S are fixed for the whole grid: script 03's Fixed TS (fitted on these 4 corruptions, on independently adapted
members at the current rates; or --temps_checkpoint DIR). Run --temps_only once before an array, so that no two tasks fit them.
Every (point, stream) is cached under a key holding both rates and the temperatures, so a killed job resumes and an array
splits the grid, one point per task:

    sbatch slurm/job.sh scripts/02b_tune_joint_lr.py --duo vitb16_rn50v1 --bn --temps_only
    sbatch -t 00:40:00 --array=0-24 slurm/job.sh scripts/02b_tune_joint_lr.py --duo vitb16_rn50v1 --bn --shard auto
    sbatch slurm/job_cpu.sh scripts/02b_tune_joint_lr.py --duo vitb16_rn50v1 --bn --report_only     # the ranking of THIS grid (no model runs)
    sbatch slurm/job_cpu.sh scripts/02b_tune_joint_lr.py --duo vitb16_rn50v1 --bn --rank_cache      # rank EVERY cached Joint run, no grid flags
    sbatch -t 00:20:00 slurm/job.sh scripts/02b_tune_joint_lr.py --duo vitb16_rn50v1 --bn --num_samples 2000 --mults 1 4  # smoke test

(--bn matters only for the frozen reference: a frozen BatchNorm model then runs on the test batch's statistics; the Joint
and Independent runs adapt both members, which always use batch statistics.)

Unlike 02 and 05 this does NOT touch the indep file of cfgs/duo_lrs/ (those rates also set every cache key of the independent
modes), whereas these rates belong to the Joint mode alone: the ranking run records the best pair itself in
cfgs/duo_lrs/<duo>__joint.yaml (what script 03 and the reset scripts read), replacing the pair that was there for that TTA method
and any `pending:` note, and prints what it replaced. So the file always holds the winner of the LAST ranking run: a report with a
smaller grid than before can record a different winner. (`python -m src.utils.duo_lrs --import_joint <out>/results.json` does the
same by hand.) The last line printed is the script-03 command that scores the pair on the 15 test corruptions (Joint only, with
the temperatures used here). Writes <out>/ranking.csv (one row per run, fractions) and <out>/results.json, with <out> =
<paths.out>/results/tune_joint_lr/<duo>__<tta>[_bn]. A capped run (--num_samples) goes to its own folder; it records nothing and
prints no command: short streams favour larger rates than the 780 updates of a full stream do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments import joint_lr_tuning as JL
from src.experiments.cli import add_common_args, bn_tag, select_jobs, setup
from src.experiments.results import save_rows


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    JL.add_args(ap)
    args = ap.parse_args()
    tag = f"{args.duo}__{args.tta}{bn_tag(args)}" + (f"__n{args.num_samples}" if args.num_samples else "")
    if args.record and not args.rank_cache:
        ap.error("--record is for --rank_cache (a full --report_only run records the best pair anyway)")
    if args.rank_cache:
        args.report_only = True                        # nothing is computed: the cache is read as it is
    ctx, seeds, out = setup(args, f"tune_joint_lr/{tag}", default_seeds=[0])

    current = {side: float(ctx.duo[side]["OPTIM"]["LR"]) for side in ("LARGE", "SMALL")}
    base = tuple(float(f"{current[side]:.6g}") for side in ("LARGE", "SMALL"))
    for side in ("LARGE", "SMALL"):
        grid = "" if args.rank_cache else "; grid " + str([f"{v:g}" for v in JL.lr_axis(
            current[side], args.mults, args.lrs_large if side == "LARGE" else args.lrs_small)])
        print(f"{side.lower()} {ctx.duo[side]['NAME']} ({ctx.duo[side]['NORM']}): {JL.optimizer_text(ctx.duo[side]['OPTIM'])}, "
              f"current learning rate {current[side]:g}{grid}")

    if args.rank_cache:
        tuner, jobs, info = JL.rank_cache_setup(ctx, args.tta, args.tta_kwargs, args.num_samples, seeds, args.temps_checkpoint)
        n_pairs = sum(j.kind == "joint" for j in jobs)
        if tuner is None or not n_pairs:
            print(f"--rank_cache: no complete Joint run of {args.duo} / {args.tta} for seeds {seeds} in the cache "
                  f"(batch size {ctx.batch_size}, {args.num_samples or 'all'} images per stream)")
            for t_l, t_s in info["no_checkpoint"]:
                print(f"  runs exist for T_L={t_l}, T_S={t_s}, but no checkpoint under checkpoints/fixed_ts/ holds that pair: "
                      f"pass --temps_checkpoint")
            return 1
        temps = tuner.temps
        print(f"--rank_cache: {n_pairs} complete Joint runs found in the cache (batch size {ctx.batch_size}, "
              f"{args.num_samples or 'all'} images per stream, seeds {seeds}), scored with T_L={temps.T_l}, T_S={temps.T_s} "
              f"({temps.folder.name}), the temperatures they were made with")
        if len(info["temperatures"]) > 1:
            print("  the cache holds Joint runs for several temperature pairs (runs valid under each): "
                  + ", ".join(f"T_L={a}, T_S={b}: {n}" for (a, b), n in info["temperatures"])
                  + "; the pair with most runs is used, --temps_checkpoint DIR chooses another")
        if info["left_out"]:
            print(f"  {info['left_out']} other Joint runs of this duo are cached with other settings (another input pipeline, "
                  f"temperatures, batch size, seed, --num_samples or optimizer) and are left out")
        for label in info["incomplete"]:
            print(f"  left out, streams missing: {label}")
        for label in info["no_reference"]:
            print(f"  reference not cached (it depends on the current independent rates), its comparison is skipped: {label}")
        lrs_large = sorted({j.lr_large for j in jobs if j.kind == "joint"})
        lrs_small = sorted({j.lr_small for j in jobs if j.kind == "joint"})
    else:
        temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, args.tta, args.tta_kwargs, args.num_samples),
                                        args.temps_checkpoint, report_only=args.report_only)
        print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}  ({temps.folder})")
        if args.temps_only:
            return 0
        tuner = JL.Tuner(ctx, args.tta, args.tta_kwargs, args.num_samples, temps)
        lrs_large = JL.lr_axis(current["LARGE"], args.mults, args.lrs_large)
        lrs_small = JL.lr_axis(current["SMALL"], args.mults, args.lrs_small)
        jobs = JL.build_jobs(lrs_large, lrs_small, seeds)
        print(f"{len(jobs)} runs: {len(jobs) - 2 * len(seeds)} grid points + frozen and Independent both (references), "
              f"batch size {ctx.batch_size}, {args.num_samples or 'all'} images per stream, seeds {seeds}")
    if not args.report_only:
        tuner.compute(select_jobs(jobs, lambda j: j.label, args.shard, args.only))
    gaps = tuner.missing(jobs)
    if gaps:
        for label, streams in gaps.items():
            print(f"{label}: member logits missing for {len(streams)} streams (e.g. {streams[:3]}); "
                  f"run without --report_only to compute them")
        return 0 if args.shard or args.only else 1       # a shard only fills the cache; the full run ranks
    if args.shard:
        return 0

    summaries = tuner.score(jobs)
    points = JL.joint_points(summaries)
    ranked = JL.rank(points)
    if not ranked:
        print("every grid point diverged (non-finite logits): the learning rates are too large; use smaller --mults")
        return 1
    best = ranked[0]

    print(f"\nduo accuracy (%), macro over the 4 held-out streams; rows: large member's rate, columns: small member's; "
          f"* = best\n{JL.grid_table(points, lrs_large, lrs_small, 'duo', best)}")
    print(f"\naccuracy (%) of the worst stream\n{JL.grid_table(points, lrs_large, lrs_small, 'worst', best)}")
    print(f"\n{JL.ranking_table(ranked)}")
    print(f"\nbest: large {best['lr_large']:g}, small {best['lr_small']:g}: duo {100 * best['duo']:.2f}, worst stream "
          f"{100 * best['worst']:.2f} ({best['worst_stream']}), large alone {100 * best['large']:.2f}, "
          f"small alone {100 * best['small']:.2f}")
    for line in JL.comparisons(summaries, best, base):
        print(line)
    frozen = summaries.get(("frozen", None, None))
    if frozen and best["duo"] <= frozen["duo"]:
        print("WARNING: no pair beats the frozen duo; try smaller rates (--mults 0.1 0.25 0.5 1)")
    for w in JL.edge_warnings(best, lrs_large, lrs_small):
        print(f"WARNING: {w}")
    n_div = sum(p["diverged"] for p in points)
    if n_div:
        print(f"{n_div} of {len(points)} grid points diverged (non-finite logits)")

    rows = JL.result_rows(summaries, ranked)
    save_rows(out / "ranking.csv", pd.DataFrame(rows).astype({"rank": "Int64"}))     # the references have no rank
    command = None if args.num_samples else JL.evaluate_command(      # no --joint_lr_*: script 03 reads the pair recorded below
        duo=args.duo, tta=args.tta, tta_kwargs=args.tta_kwargs, bn=args.bn, temps_folder=temps.folder)
    (out / "results.json").write_text(json.dumps({
        "duo": args.duo, "tta": args.tta, "tta_kwargs": args.tta_kwargs, "bn": args.bn, "seeds": seeds,
        "num_samples": args.num_samples, "batch_size": ctx.batch_size, "streams": sorted(best["streams"]),
        "temperatures": {"T_l": temps.T_l, "T_s": temps.T_s, "checkpoint": str(temps.folder)},
        "optim": {side.lower(): ctx.duo[side]["OPTIM"] for side in ("LARGE", "SMALL")},
        "current_lrs": {"large": base[0], "small": base[1]}, "lrs_large": lrs_large, "lrs_small": lrs_small,
        "best": {"lr_large": best["lr_large"], "lr_small": best["lr_small"], "duo": best["duo"], "worst": best["worst"]},
        "evaluate_command": command,
        "runs": [{k: None if isinstance(v, float) and v != v else v for k, v in r.items()} for r in rows]}, indent=1))
    print(f"wrote {out / 'ranking.csv'} and {out / 'results.json'}")
    if args.num_samples:
        print("capped run (--num_samples): the best pair is not recorded, no follow-up command")
        return 0
    if args.rank_cache and not args.record:
        print(f"ranked what is cached; the best pair is not recorded: add --record, or run "
              f"python -m src.utils.duo_lrs --import_joint {out / 'results.json'}")
    else:
        print(JL.record_best(out / "results.json", ctx.g.path("duo_lrs")))
    print(f"\nscore the best pair on the 15 test corruptions:\n{command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
