#!/usr/bin/env python3
"""
The CoTTA setting for the duo: Vanilla DD, Joint DD, the fixed-TS duo and the frozen duo on ONE long stream of ImageNet-C that
repeats the 15 shifts (CoTTA's standard order, or --order K for one of its random ones) for R rounds, with no resets (src/experiments/recurring.py). Tent, nuclear-norm proxy.

    indep      members adapt on their own entropy (indep regime's learning rates): the Fixed TS duo and, with the gate outside the
               loop, Vanilla DD
    joint_dd   members adapt on the entropy of the gated prediction (joint regime's learning rates): Joint DD
    frozen     no adaptation: the frozen duo

Defaults: 10 rounds x 15 corruptions x 5,000 images (the CoTTA protocol's per-corruption subset; every round draws its own images)
= 750,000 images, gate at proxy batch size 256, beta 0.5. The three runs are independent live jobs over identical batches:

    sbatch -t 01:50:00 slurm/job.sh scripts/03c_recurring_duos.py --duo vitb16_rn50v1 --bn --runs indep    --shard 0/1
    sbatch -t 02:00:00 slurm/job.sh scripts/03c_recurring_duos.py --duo vitb16_rn50v1 --bn --runs joint_dd --shard 0/1
    sbatch -t 00:50:00 slurm/job.sh scripts/03c_recurring_duos.py --duo vitb16_rn50v1 --bn --runs frozen   --shard 0/1
    sbatch slurm/job_cpu.sh scripts/03c_recurring_duos.py --duo vitb16_rn50v1 --bn --report_only       # figure and tables

(a shard only fills the files and never reports; --report_only reads them). Longer streams: --images 50000 (the full corruption)
and/or --rounds; a run is one job, 10x longer per 10x the images. A finished run is never recomputed (its file is the record).
Files, in <paths.out>/results/recurring/: see src/experiments/recurring.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import adaptation_modes as AM
from src.experiments import recurring as RC
from src.experiments.cli import add_common_args, select_jobs, setup
from src.experiments.joint_dd import PureProxyGate


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
                         tta=False)
    ap.add_argument("--rounds", type=int, default=10, help="how many times the 15 corruptions are repeated")
    ap.add_argument("--images", type=int, default=5000, help="images per corruption and round (CoTTA: 5,000; full: 50,000)")
    ap.add_argument("--order", type=int, default=None, choices=range(10), metavar="K",
                    help="one of CoTTA's 10 random corruption orders (cfgs/protocols.yaml); default: the standard order")
    ap.add_argument("--runs", nargs="+", choices=list(RC.RUNS), default=list(RC.RUNS))
    ap.add_argument("--proxy", default=None, help="the gate's proxy (pure.PROXIES key; default: the paper's, nuclear_norm = nnn)")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--b_t", type=int, default=256, help="proxy batch size")
    ap.add_argument("--temps_checkpoint", default=None, metavar="DIR", help="as in script 03 (default: the held-out fit it uses)")
    args = ap.parse_args()
    tta = "tent"            # Joint DD exists for Tent only

    ctx, seeds, out = setup(args, "recurring", default_seeds=[0])
    proxy = args.proxy or ctx.g.gate["proxy"]
    proxy = {"nuclear_norm": "nnn", "ac_mc": "msp"}.get(proxy, proxy)       # the key scripts/09 and PureProxyGate use
    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, tta, None, None), args.temps_checkpoint, report_only=True)
    duo_ctx = AM.joint_context(ctx, tta)
    print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}  ({temps.folder})")
    for what, c in (("own-entropy", ctx), ("duo-entropy", duo_ctx)):
        print(f"{what} learning rates: " + ", ".join(f"{s.lower()} {c.duo[s]['NAME']} {c.duo[s]['OPTIM']['LR']:g}" for s in ("LARGE", "SMALL")))
    print(f"stream: CoTTA {'standard order' if args.order is None else f'order {args.order}'}, {args.rounds} rounds x 15 corruptions x {args.images} images; gate {proxy}, "
          f"proxy batch size {args.b_t}, beta {args.beta:g}")

    def path_of(run: str, seed: int) -> Path:
        tag = RC.stream_tag(args.duo, tta, args.bn, args.rounds, args.images, seed, args.order)
        return RC.run_path(out, tag, run, proxy, args.b_t, args.beta)

    jobs = [(run, seed) for seed in seeds for run in args.runs]
    if not args.report_only:
        for run, seed in select_jobs(jobs, lambda j: f"{j[0]}/seed{j[1]}", args.shard, args.only):
            path = path_of(run, seed)
            if path.is_file():
                print(f"== {run}/seed{seed}: done already ({path.name})")
                continue
            spec = RC.RUNS[run]
            rctx = duo_ctx if spec.duo_entropy else ctx
            print(f"== {run}/seed{seed}: {spec.engine}")
            gate = (PureProxyGate(proxy, beta=args.beta, b_t=args.b_t, T_l=temps.T_l, T_s=temps.T_s, num_classes=1000)
                    if spec.gated else None)
            runner, inside = RC.make_runner(rctx, spec, gate, tta)
            parts = RC.stream_parts(rctx, rounds=args.rounds, seed=seed, order=args.order)
            t0 = time.time()
            res = RC.run_recurring(runner, parts, images=args.images, batch_size=rctx.batch_size, workers=rctx.g.workers,
                                   T_l=temps.T_l, T_s=temps.T_s, gate=gate, inside=inside)
            meta = {"run": run, "engine": spec.engine, "duo": args.duo, "tta": tta, "bn": args.bn, "rounds": args.rounds,
                    "images": args.images, "order": args.order, "seed": seed, "proxy": proxy, "b_t": args.b_t, "beta": args.beta,
                    "T_l": temps.T_l, "T_s": temps.T_s, "batch_size": rctx.batch_size,
                    "lr": {s.lower(): rctx.duo[s]["OPTIM"]["LR"] for s in ("LARGE", "SMALL")},
                    "optim": {s.lower(): rctx.duo[s]["OPTIM"] for s in ("LARGE", "SMALL")},
                    "models": {s.lower(): rctx.duo[s]["NAME"] for s in ("LARGE", "SMALL")}, "seconds": time.time() - t0}
            print(f"wrote {RC.save_run(path, res, parts, meta)} ({sum(res['part_n'])} images, {meta['seconds'] / 60:.0f} min)")
    if args.shard:
        return 0                # a shard only fills the files; the full run reports

    absent = [str(path_of(r, s)) for r, s in jobs if not path_of(r, s).is_file()]
    if absent:
        print("runs missing (run them first, see --help):\n  " + "\n  ".join(absent))
        return 1
    from src.experiments import recurring_report as RR
    return RR.report(args, ctx, duo_ctx, temps, proxy, seeds, path_of, out)


if __name__ == "__main__":
    sys.exit(main())
