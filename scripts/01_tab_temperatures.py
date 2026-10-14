#!/usr/bin/env python3
"""
Optimal temperatures per stream, what each way of choosing them is worth, and the oracles that bound them.

For clean validation, the 4 held-out corruptions and the 15 ImageNet-C test corruptions (severity 5), one row:

    T_L*  T_S*     the NLL-optimal temperatures of the duo on that stream (labels used)
    Large  Small   each member's own accuracy
    Clean TS       accuracy with the temperatures fitted on clean data
    Held-out TS    accuracy with the temperatures fitted on the 4 held-out corruptions pooled ("Fixed TS")
    Corr.-wise     accuracy at the stream's own optimum T_L*, T_S* (oracle: labels, in-sample)
    Batch-wise     (T_L, T_S) refitted on every consecutive batch of B samples, one column per --batch_sizes
    Per-sample     (T_L, T_S) fitted on each sample by gradient descent: the ceiling

The last row averages the 15 test corruptions. Duo = (z_L / T_L + z_S / T_S) / 2. The members' logits come from
one adaptation run per seed; every fit then runs on those logits, on all 20 streams (no blank cells).

    sbatch slurm/job.sh scripts/01_tab_temperatures.py                       # members adapt with Tent, seed 0
    sbatch slurm/job.sh scripts/01_tab_temperatures.py --mode no_adapt       # frozen members (BatchNorm: source statistics)
    sbatch slurm/job.sh scripts/01_tab_temperatures.py --mode no_adapt --bn  # frozen members on the test batch's statistics
    python scripts/01_tab_temperatures.py --report_only                      # table from the cache, no model
    python scripts/01_tab_temperatures.py --mode no_adapt --batch_sizes 16 64 128

Writes <paths.paper>/tab_temperatures__<tag>.tex and <paths.out>/results/tab_temperatures/<tag>.csv (accuracy
and NLL of every column). <tag> is the TTA method or `frozen`, plus `_bn` with --bn, `_trainslice` with
--clean_source train_slice, and `__<duo>` for a duo other than the default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments.cli import add_common_args, setup
from src.experiments.results import annotate, save_rows
from src.experiments.temperature_study import (CacheSource, StudyLabels, add_study_args, all_cached, clean_text,
                                               compute_members, model_names, regime_text, render_table, study_rows,
                                               study_specs, study_tag)
from src.utils.tables import write_table


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    add_study_args(ap)
    args = ap.parse_args()
    ctx, seeds, out = setup(args, "tab_temperatures", default_seeds=[0])
    specs = study_specs(args, ctx, seeds)
    batch_sizes = sorted(set(args.batch_sizes))

    compute_members(args, ctx, specs)
    if not all_cached(ctx, specs):
        return 0 if args.shard or args.only else 1       # a shard only fills the cache; the full run renders

    labels = StudyLabels.from_config(ctx.g, args.clean_source)
    rows = []
    for spec in specs:
        rows += annotate(study_rows(CacheSource(spec, ctx), labels, batch_sizes=batch_sizes, steps=args.steps), spec)
    tag = study_tag(args, ap.get_default("duo"))
    save_rows(out / f"{tag}.csv", rows)
    tex = render_table(
        pd.DataFrame(rows), labels, batch_sizes=batch_sizes, models=model_names(ctx), regime=regime_text(args),
        frozen=args.mode == "no_adapt", severity=ctx.g.severity, clean_desc=clean_text(args),
        table_labels=[f"tab:temperatures:{tag}", f"tab:oracles:{tag}"], source="scripts/01_tab_temperatures.py")
    # a capped run (--num_samples, a smoke test) must not overwrite the final tables
    write_table((out if args.num_samples else ctx.g.path("paper")) / f"tab_temperatures__{tag}.tex", tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
