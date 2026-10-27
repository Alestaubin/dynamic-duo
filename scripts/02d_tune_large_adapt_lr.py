#!/usr/bin/env python3
"""
Tune the learning rate of the LARGE member for the Large adapt mode of script 03: only the large member adapts, on the entropy of the static
duo's combined prediction (z_L / T_L + z_S / T_S) / 2, while the small member is frozen (its logits are a constant of the loss). The grid
is over ONE rate, scored on the 4 held-out corruptions (speckle noise, Gaussian blur, spatter, saturate; severity 5; none is a test corruption).

Why a tuner of its own. Script 03 used to run Large adapt at the large member's rate of the JOINT pair (02b), tuned with both members moving.
With the small member frozen the duo entropy is another signal, so its best rate is another number: it is tuned here, for this mode alone,
and recorded in the regime file cfgs/duo_lrs/<duo>__large_duo.yaml (the large_duo regime), which script 03's Large adapt reads. The joint pair is
never used for it, and a duo with no recorded rate makes the mode an ERROR (NotTuned) in script 03, not a silent run at other rates.

The grid starts at the large member's SINGLE-regime rate (cfgs/duo_lrs/<duo>__single.yaml: the rate tuned for it alone, the hp protocol's
"individual rates first, duos from them") and goes both ways: by default 0.5, 1, 2, 4, 8, 16, 32, 64 times it (--mults), or --lrs for
explicit values; momentum and the rest of the OPTIM stay. A grid point is one Large adapt run over the 4 full streams (batch size as in the
protocol; the members reset to source weights before every stream): the very run script 03 makes on the test corruptions. Score = the
mean over the 4 streams of the static duo's accuracy; the worst stream, the late accuracy (the last 20% of each stream) and the large
member's own accuracy are reported next to it. References: the frozen duo, and Independent large (the same member alone on its OWN entropy, at its
single rate), so the report says what the duo signal buys over the member's own. A point whose late accuracy is below the frozen duo's is
ruled out (drift); a run whose logits go non-finite is recorded as diverged and ranks nowhere. The optimum must be INTERIOR to the grid: a
winner on its edge prints a warning (extend the grid; finished points are reused) and --record refuses.

T_L and T_S are fixed for the whole grid: the Fixed TS fitted on the FROZEN models (never refitted per rate; --temps_checkpoint DIR names
another). Run --temps_only once before an array, so that no two tasks fit them. Every (point, stream) is cached, so a killed job resumes
and an array splits the grid, one point per task (the two references wrap around onto the first two tasks; the launcher
is at QOS high, and GPU jobs keep --exclude=kn117,kn101, the nodes with the ECC error):

    sbatch --exclude=kn117,kn101 slurm/job.sh scripts/02d_tune_large_adapt_lr.py --duo vitb16_rn50v1 --bn --temps_only
    sbatch --exclude=kn117,kn101 -t 00:45:00 --array=0-7 slurm/job.sh scripts/02d_tune_large_adapt_lr.py --duo vitb16_rn50v1 --bn --shard auto
    python scripts/02d_tune_large_adapt_lr.py --duo vitb16_rn50v1 --bn --report_only             # the ranking (no model runs, no GPU)
    python scripts/02d_tune_large_adapt_lr.py --duo vitb16_rn50v1 --bn --report_only --record    # ... and record the winner (refused while a warning is open)
    sbatch --exclude=kn117,kn101 -t 00:20:00 slurm/job.sh scripts/02d_tune_large_adapt_lr.py --duo vitb16_rn50v1 --bn --num_samples 2000 --mults 1 4   # smoke test

(Time per task: about 15 minutes for a point, scaled from script 03's measured 60 minutes for the 16 streams of this mode; not
measured for this script.) --bn matters here: the small member is frozen, and a frozen BatchNorm model then runs on the test batch's
statistics, as in script 03 (use the same flag).

Procedure (as 02b and 02c): the default ladder on seed 0, then a sqrt-2 refinement around the winner (--lrs), then the finalists on
seeds 1 and 2 (--lrs <the finalists> --seeds 1 2, sharded the same way), then the final choice on all three seeds:
    python scripts/02d_tune_large_adapt_lr.py --duo D --bn --lrs <the finalists> --seeds 0 1 2 --report_only --record
--record writes the best rate to cfgs/duo_lrs/<duo>__large_duo.yaml (`python -m src.utils.duo_lrs --import_anchor <results.json>` does the same
by hand) and replaces the one that was there. It never touches the indep, single or joint file, nor any cache key of another mode. A capped
run (--num_samples) goes to its own folder, records nothing and prints no follow-up: short streams favour larger rates than the 780 updates
of a full stream do. The last line printed is the script-03 command that scores the rate on the 15 test corruptions (Large adapt only, with the
temperatures used here). Writes <paths.out>/results/tune_large_adapt_lr/<duo>__<tta>[_bn]/ranking.csv (one row per run, fractions) and
results.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import anchor_lr_tuning as AL


if __name__ == "__main__":
    sys.exit(AL.run("large", __doc__))
