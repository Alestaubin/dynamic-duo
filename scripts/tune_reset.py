#!/usr/bin/env python3
"""
Tune a duo with resetting (`own_fast_vs_slow`): learning rates and detector settings, chosen on the held-out shifts only, on ONE rotation of the held-out stream
(protocol: docs/reset_tuning.md). One command, any duo, either signal, called again after every wave:

    python scripts/tune_reset.py --duo convnextb_deits                       # what is done, and the next step (nothing is launched)
    python scripts/tune_reset.py --duo convnextb_deits --submit              # do it: submit the next wave of Slurm jobs, or run the next CPU step
    python scripts/tune_reset.py --duo convnextb_deits --signal joint        # the same for the joint signal (its own rates, folders and search)

--signal: the loss the members adapt on. `independent` (default): each member on its own entropy (Vanilla DD, the gate outside the loop); the members decouple, so each
is tuned alone: its detector setting and its learning rate. `joint`: both members on the gated duo's entropy (Joint DD, the gate inside the loss); the members are
coupled, so the search is over the pair of rates, the gate's beta (--betas) and each member's own detector setting, scored on the duo's accuracy.

The stages, in order (src/reset/pipeline.py): the duo's frozen temperatures, the continual-rate ladder (each member's conservative and stress rate), the
calibration streams (stress S, healthy H), the offline screens (the shared detector setting and each member's candidates), the search (waves of live runs on the
tuning rotation until the choice is made), and the confirmation: the chosen configuration run once on that rotation.
A job is submitted once: one already queued under its name is skipped, so calling twice is safe (`--force` ignores the queue).

    --stress LR_L LR_S   the stress pair of the calibration streams, if the continual stage's stress rates do not collapse both members (the screen then reports
                         that a member never collapses); --stress_up K: K rungs up its ladder; --healthy LR_L LR_S; --mults X ...: the continual ladder's multiples
                         of the episodic rates (extend it when its report says it does not bracket the frontier). They are saved in the duo's config.json.
    --betas B ...        the joint signal's gate grid (default 0 0.25 0.5 0.75 1; extended upward if the choice sits on the top). 0 is the fixed-TS duo: no gate.
    --rotations K ...    the held-out rotation(s) of the search (default: 0; two, e.g. 0 2, pool their visits).
    --chain              with --submit: after a wave, also submit a small CPU job that calls this script again once the wave has finished OK (a failed job cancels
                         the chain), so the tuning runs itself until it is done or needs a decision (a blocked stage, a failed job: look at its log).
    --baseline COND      the same machinery for a comparison method that has only a rate to tune: none, fixed500, fixed1000, own_fast_vs_slow@<delta> (independent only).

Slurm runs this same script for one unit of work (`--job <spec>`, written by the driver under <out>/results/reset_tune[/<duo>]/jobs/): see src/reset/workers.py.
Setup that this script does not do: the duo's yaml (cfgs/duos/<duo>.yaml) and its episodic rates for the signal (cfgs/duo_lrs/<duo>__indep.yaml, scripts/02c;
cfgs/duo_lrs/<duo>__joint.yaml, scripts/02b).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", default="vitb16_rn50v1", help="cfgs/duos/<name>.yaml (default: the main duo)")
    ap.add_argument("--signal", choices=("independent", "joint"), default="independent", help="the loss the members adapt on (default %(default)s)")
    ap.add_argument("--submit", action="store_true", help="do the next step (submit the jobs / run the CPU step); without it only print it")
    ap.add_argument("--chain", action="store_true", help="with --submit: after a wave, queue a CPU job that calls this script again when the wave has finished")
    ap.add_argument("--force", action="store_true", help="submit jobs even if one of the same name is already queued")
    ap.add_argument("--baseline", default=None, metavar="COND", help="tune a comparison method's rates instead: none, fixed<N>, own_fast_vs_slow@<delta>")
    ap.add_argument("--stress", type=float, nargs=2, default=None, metavar=("LR_L", "LR_S"))
    ap.add_argument("--stress_up", type=int, default=None, metavar="K")
    ap.add_argument("--healthy", type=float, nargs=2, default=None, metavar=("LR_L", "LR_S"))
    ap.add_argument("--mults", type=float, nargs="+", default=None, metavar="X")
    ap.add_argument("--betas", type=float, nargs="+", default=None, metavar="B", help="joint signal: the gate's beta grid")
    ap.add_argument("--rotations", type=int, nargs="+", default=None, metavar="K", help="the held-out rotation(s) of the search (default: 0)")
    ap.add_argument("--job", default=None, metavar="SPEC", help="(what Slurm runs) execute the job of this spec file")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"], help="with --job")
    args = ap.parse_args()

    if args.job:
        from src.reset.workers import run_job
        return run_job(args.job, args.device)
    from src.reset.pipeline import drive
    overrides = {"stress": args.stress, "stress_up": args.stress_up, "healthy": args.healthy, "mults": args.mults, "betas": args.betas,
                 "tuning_rotations": args.rotations}
    return drive(args.duo, submit=args.submit, force=args.force, baseline=args.baseline, overrides=overrides, signal="joint" if args.signal == "joint" else "indep",
                 chain=args.chain)


if __name__ == "__main__":
    sys.exit(main())
