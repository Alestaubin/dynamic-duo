#!/usr/bin/env python3
"""
The multi-gate runs of the gate study: the frozen configuration of a method (`out/results/reset_tune/final_configs.json`: same rule, rates, seed) run again on a stream, with
more gates scored on the same members' logits (a live run's `extra_gates`, src/reset/workers.py; gates in src/reset/gate_variants.py). The default is the BETA LADDER
(0, 0.25, 1, 2, 4, 8 besides the run's own 0.5) for own_fast_vs_slow. The members' logits are not stored, so a gate that was not scored in the run cannot be scored
afterwards. One run is about 65 minutes on one GPU (the extra gates cost a few percent: they reuse the main gate's proxy values).

    python scripts/46_gate_routing_runs.py                          # held-out rotations 0-3: prints the sbatch commands (nothing is submitted); skips runs whose file exists
    python scripts/46_gate_routing_runs.py --submit                 # submits them (slurm/job.sh: QOS high)
    python scripts/46_gate_routing_runs.py --stream test --submit   # the five test orderings (see below)
    python scripts/45_gate_ablation.py --multigate out/results/reset_gates/vitb16_rn50v1__tent_bn --stream heldout --methods own_fast_vs_slow     # the tables

Which stream: a beta is a selection, and the protocol (plan.md, hp-tuning) selects on the four held-out shifts only: run the ladder on `--stream heldout` (4 runs, one per
rotation), choose on that, and only then report the chosen beta on the test stream, whose final-test table is not touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reset import gate_variants as GV
from src.reset import jobs as J
from src.reset import live_report as LV
from src.reset.layout import Paths
from src.utils.config import load_global


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--methods", nargs="+", default=["own_fast_vs_slow"], help="names in final_configs.json (default %(default)s)")
    ap.add_argument("--stream", choices=("heldout", "test"), default="heldout")
    ap.add_argument("--rotations", type=int, nargs="+", default=[0, 1, 2, 3], help="held-out stream: the rotations (default %(default)s)")
    ap.add_argument("--orders", nargs="+", default=["std", "0", "1", "2", "3"], help="test stream: the orderings (std = CoTTA's standard order)")
    ap.add_argument("--extra_gates", nargs="+", default=list(GV.LADDER), help="gate names (default: %(default)s)")
    ap.add_argument("--beta", type=float, default=0.5, help="the main gate's beta (the final test's)")
    ap.add_argument("--time", default="01:50:00", help="the final test's limit (its runs took about 64 minutes)")
    ap.add_argument("--submit", action="store_true", help="run sbatch (default: only print the commands)")
    args = ap.parse_args()
    g = load_global()
    cfg = json.loads((g.path("out") / "results" / "reset_tune" / "final_configs.json").read_text())
    by_name = {m["name"]: m for m in cfg["methods"]}
    unknown = [n for n in args.methods if n not in by_name]
    if unknown:
        raise SystemExit(f"not in final_configs.json: {unknown}; it has {sorted(by_name)}")
    for n in args.extra_gates:
        GV.parse(n)                                                   # a bad name now, not after the queue
    folder = g.path("out") / "results" / "reset_gates" / f"{args.duo}__tent_bn"
    todo = 0
    for name in args.methods:
        m = by_name[name]
        for key in (args.rotations if args.stream == "heldout" else args.orders):
            if args.stream == "heldout":
                stem = LV.run_stem(m["cond"], stream="heldout", rot=key, seed=0, lrs=m["lrs"], proxy="nnn", b_t=256, beta=args.beta)
                order, tag = None, f"h{key}"
            else:
                order = None if key == "std" else int(key)
                stem = LV.run_stem(m["cond"], stream="test", rot=0, seed=0, lrs=m["lrs"], proxy="nnn", b_t=256, beta=args.beta, order=order)
                tag = f"t{key}"
            if (folder / f"{stem}.npz").is_file():
                print(f"# done already: {name}, {args.stream} {key}")
                continue
            job = J.live_job(args.duo, m["cond"], stream=args.stream, lrs=m["lrs"], rotation=key if args.stream == "heldout" else 0, order=order, seed=0, beta=args.beta,
                             extra_gates=args.extra_gates, label=f"{tag}{name}")
            cmd = J.live_command(Paths(g, args.duo), job, time=args.time, write=args.submit)
            todo += 1
            print(" ".join(cmd))
            if args.submit:
                print("  ->", subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip())
    print(f"# {todo} run(s) {'submitted' if args.submit else 'to submit (--submit)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
