#!/usr/bin/env python3
"""
Different ways of adapting the duo: Joint, Large / Small adapt, Independent large / small / both.

Every row is scored by the same static duo (z_L / T_L + z_S / T_S) / 2, so the rows differ only in how the members'
weights were adapted:

    Joint               both members minimize the entropy of the duo's combined prediction
    Large adapt         only the large member adapts, on the duo entropy; the small one is frozen (constant logits)
    Small adapt         only the small member adapts, on the duo entropy; the large one is frozen
    Independent large   Tent on the large member's own entropy; the small member is not adapted
    Independent small   Tent on the small member's own entropy; the large member is not adapted
    Independent both    both members run Tent on their own entropy
    (No adaptation)     frozen weights (see --bn), as the reference row

The temperatures T_L, T_S are the paper's Fixed TS, shared by every mode: fitted once on the 4 held-out corruptions
pooled, on the logits of independently adapted members (the "Held-out TS" of 01_tab_temperatures.py, whose cache
this reuses), and stored under paths.checkpoints. --temps_checkpoint DIR uses another JointFixedTS checkpoint.
One live run per (mode, seed) over the 15 test corruptions (severity 5) and the clean validation images; each stream is
cached, so a repeat only replays (and a cache from before the clean stream existed only computes that stream).

    sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --temps_only                  # once: fit T_L, T_S
    sbatch --array=0-20 slurm/job.sh scripts/03_tab_adaptation_modes.py --shard auto      # one job per (mode, seed)
    python scripts/03_tab_adaptation_modes.py --report_only                               # tables from the cache
    python scripts/03_tab_adaptation_modes.py --modes joint indep_duo --seeds 0 --num_samples 2000   # smoke test

Learning rates. The two kinds of mode adapt on different signals and each runs with its own rates (cfgs/duo_lrs/<duo>__<regime>.yaml):
the modes that adapt on a member's OWN entropy (Independent large / small / both) with the indep regime's rates (tuned member by member,
scripts/02_tune_tta.py), the modes that adapt on the DUO entropy (Joint, Large adapt, Small adapt) with the joint regime's pair (tuned
by scripts/02b_tune_joint_lr.py, which records its winner there itself; Large / Small adapt use
their adapting member's rate of that pair). A duo whose joint pair is not tuned yet makes the duo-entropy modes an ERROR
(NotTuned), never a silent run at the indep rates; --joint_lr_large LR --joint_lr_small LR run them with rates given by hand (both are
needed to skip the file; either one alone replaces just that member's). --lr_large / --lr_small replace the own-entropy modes' rates
the same way (the rest of each member's OPTIM stays). Independent large / small (one model adapting alone) run with the rates tuned for
that model ALONE (the single regime); Independent both with the independent duo's pair (the indep regime). Both tables' captions list every
set of rates and say which modes use which.
The rates are part of every cache key, so nothing is shared between two settings, and the temperatures (one pair for every mode, fitted
on members adapted at the own-entropy rates) are fitted again for other own-entropy rates (--temps_only first, as above;
--temps_checkpoint DIR keeps another pair instead). With any of the four flags the CSV and both tables are named
<tag>_lrL<large>_S<small>[_jlrL<large>_S<small>]_<YYYYmmdd-HHMMSS> (the time of the rendering run; a shard never renders), so they
never replace the tuned run's files in the paper folder nor each other.

Writes <paths.paper>/tab_adaptation_modes__<tag>.tex (accuracy per corruption: the duo under each mode, and each
member alone adapting on its own entropy / frozen; a Clean column that is not part of Avg.),
tab_adaptation_modes_summary__<tag>.tex (members' accuracy under each mode, duo accuracy / ECE / NLL, averaged over
the corruptions) and <paths.out>/results/tab_adaptation_modes/<tag>.csv (every stream, clean included). <tag> is the TTA method (plus the duo name
unless it is the default duo). A capped run (--num_samples) writes its tables to the results folder instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments.cli import add_common_args, select_jobs, setup
from src.experiments.results import per_seed_average, save_rows
from src.utils.tables import write_table


def print_rates(what: str, ctx) -> None:
    print(f"{what}: " + ", ".join(f"{side.lower()} {ctx.duo[side]['NAME']} {ctx.duo[side]['OPTIM']['LR']:g}"
                                  for side in ("LARGE", "SMALL")))


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    AM.add_args(ap)
    args = ap.parse_args()
    ctx, seeds, out = setup(args, "tab_adaptation_modes")
    ctx = AM.with_learning_rates(ctx, args.lr_large, args.lr_small)       # the own-entropy modes' rates (indep regime)
    print_rates("own-entropy learning rates", ctx)

    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, args.tta, args.tta_kwargs, args.num_samples),
                                    args.temps_checkpoint, report_only=args.report_only)
    print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}  ({temps.folder})")
    if args.temps_only:
        return 0

    duo_ctx = None            # the duo-entropy modes' rates (joint regime); only needed when one of those modes runs
    if any(AM.MODE_BY_KEY[key].duo_entropy for key in args.modes):
        duo_ctx = AM.joint_context(ctx, args.tta, args.joint_lr_large, args.joint_lr_small)
        print_rates("duo-entropy learning rates", duo_ctx)
    single_ctx = None         # Independent large / small: one model adapting alone, at the rates tuned for that model alone (single regime)
    if any(key in AM.SINGLE_MODES for key in args.modes):
        single_ctx = AM.single_context(ctx, args.tta)
        print_rates("single-model learning rates (one model adapting alone)", single_ctx)

    jobs = AM.mode_jobs(args.duo, args.tta, args.tta_kwargs, seeds, args.num_samples, temps, args.modes)
    if not args.report_only:
        AM.compute_members(ctx, select_jobs(jobs, AM.job_label, args.shard, args.only), duo_ctx=duo_ctx, single_ctx=single_ctx)
    gaps = AM.missing(ctx, jobs, duo_ctx, single_ctx)
    if gaps:
        for label, streams in gaps.items():
            print(f"{label}: member logits missing for {len(streams)} streams (e.g. {streams[:3]}); "
                  f"run without --report_only to compute them")
        return 0 if args.shard or args.only else 1      # a shard only fills the cache; the full run renders
    if args.shard:
        return 0        # (a shard that finds the cache complete must not render too: with --lr_* the names carry a stamp)

    df = pd.DataFrame(AM.duo_rows(jobs, ctx, temps, duo_ctx=duo_ctx, single_ctx=single_ctx))
    tag = AM.output_tag(args, ap.get_default("duo"), stamp=time.strftime("%Y%m%d-%H%M%S"))
    save_rows(out / f"{tag}.csv", df)
    corruptions, clean = AM.split_clean(df)
    mean_acc = lambda d: (per_seed_average(d.assign(series=d["mode"] + "/" + d["series"]))     # noqa: E731
                          .groupby("series")["accuracy"].mean().mul(100).round(2))
    print(pd.DataFrame({"avg": mean_acc(corruptions), "clean": mean_acc(clean)}).to_string())

    dest = out if args.num_samples else ctx.g.path("paper")       # a capped run (smoke test) must not overwrite the paper's
    source = "scripts/03_tab_adaptation_modes.py"
    write_table(dest / f"tab_adaptation_modes__{tag}.tex", AM.render_corruption_table(
        df, ctx, args.tta, temps, label=f"tab:adaptation_modes:{tag}", source=source, duo_ctx=duo_ctx,
        single_ctx=single_ctx))
    write_table(dest / f"tab_adaptation_modes_summary__{tag}.tex", AM.render_summary_table(
        df, ctx, args.tta, temps, label=f"tab:adaptation_modes_summary:{tag}", source=source, duo_ctx=duo_ctx,
        single_ctx=single_ctx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
