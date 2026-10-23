#!/usr/bin/env python3
"""
Tune Tent's learning rates for the CONTINUAL setting, on held-out shifts only (src/experiments/continual_tuning.py has the protocol and the
criteria): each member alone, on a stream of the 4 held-out shifts concatenated 4 times (round r rotated by r; 16 visits, 800,000 images, as
long as the test stream), with no reset. Names, for each member, a CONSERVATIVE rate (stable, with a margin below the frontier) and a STRESS
rate (a complete collapse), for scripts/04c_gate_stress.py. Nothing is written to cfgs/duo_lrs: these are rates of one experiment.

The members never see each other (Vanilla DD, the gate outside the loop), so run i adapts the large member at rate a_i and the small one at
b_i and gives one rung of each ladder; the ladders are `--mults` times each member's episodic rate (the one scripts/02c recorded: half-octaves,
3 below to 2 above), or `--lrs_large` / `--lrs_small`, and must have the same length. A run stops once both members have died. The frozen run
over the same batches is the reference of every criterion.

    sbatch --array=0-11 -t 01:35:00 slurm/job.sh scripts/04b_tune_continual_lr.py --duo vitb16_rn50v1 --bn --seeds 0 --shard auto
    python scripts/04b_tune_continual_lr.py --duo vitb16_rn50v1 --bn --seeds 0 --report_only      # needs the runs; selects

(12 jobs for 11 rungs and the frozen run, per seed). A finished run is never recomputed. --report_only prints each member's ladder, the
frontier / conservative / stress rates, and every warning (a ladder that does not bracket must be extended: run again with other --mults or
--lrs_*); it writes <paths.out>/results/tune_continual_lr/<duo>__tent_bn/results.json and ranking.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments import continual as RC
from src.experiments import continual_tuning as CT
from src.experiments import joint_lr_tuning as JL
from src.experiments.cli import add_common_args, bn_tag, select_jobs, setup
from src.experiments.grids import sgd_optim
from src.experiments.logit_runs import with_optim
from src.utils import provenance

SIDES = ("large", "small")


def run_path(out: Path, seed: int, lrs: tuple[float, float] | None) -> Path:
    """The file of one run: the frozen reference (`lrs` None) or the diagonal run at (lr_large, lr_small)."""
    return out / "runs" / (f"seed{seed}__frozen.npz" if lrs is None else f"seed{seed}__L{lrs[0]:g}_S{lrs[1]:g}.npz")


def collect(out: Path, seeds, rungs, delta: float) -> tuple[list[dict], list[Path]]:
    """One row per (side, rung, seed) from the stored runs, and the files that are missing."""
    rows, missing = [], []
    for seed in seeds:
        paths = {None: run_path(out, seed, None), **{r: run_path(out, seed, r) for r in rungs}}
        missing += [p for p in paths.values() if not p.is_file()]
        if paths[None].is_file():
            frozen = RC.load_run(paths[None])
        for r in rungs:
            if not (paths[r].is_file() and paths[None].is_file()):
                continue
            run = RC.load_run(paths[r])
            n = len(run["part_n"])
            if list(run["part_label"]) != list(frozen["part_label"][:n]) or list(run["part_n"][:-1]) != list(frozen["part_n"][:n - 1]):
                raise ValueError(f"{paths[r].name} saw other visits than the frozen run: the runs must share one stream")
            for side, lr in zip(SIDES, r):
                rows.append({"side": side, "lr": lr, "seed": seed, "lrs": list(r),
                             **CT.summarize(run[side], frozen[side][:len(run[side])], run["part_n"], list(run["part_label"]),
                                            delta=delta)})
    return rows, missing


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), tta=False)
    ap.add_argument("--mults", type=float, nargs="+", default=list(CT.DEFAULT_MULTS), metavar="X",
                    help="each member's ladder: these multiples of its episodic rate. Default: %(default)s")
    ap.add_argument("--lrs_large", type=float, nargs="+", default=None, metavar="LR", help="explicit rates of the large member")
    ap.add_argument("--lrs_small", type=float, nargs="+", default=None, metavar="LR", help="explicit rates of the small member")
    ap.add_argument("--rounds", type=int, default=4, help="times the 4 held-out shifts are visited, each round rotated (default %(default)s)")
    ap.add_argument("--images", type=int, default=50000, help="images per visit (default: all 50,000)")
    ap.add_argument("--delta", type=float, default=CT.DELTA, help="points a visit may end below the frozen member (default %(default)s)")
    ap.add_argument("--margin", type=int, default=CT.MARGIN, help="rungs between the frontier and the conservative rate (default %(default)s)")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--temps_checkpoint", default=None, metavar="DIR", help="as in script 03 (default: the held-out fit it uses)")
    args = ap.parse_args()
    tta = "tent"

    tag = f"{args.duo}__{tta}{bn_tag(args)}"
    ctx, seeds, out = setup(args, f"tune_continual_lr/{tag}", default_seeds=[0])
    episodic = {side: float(ctx.duo[side.upper()]["OPTIM"]["LR"]) for side in SIDES}
    axes = {"large": JL.lr_axis(episodic["large"], args.mults, args.lrs_large), "small": JL.lr_axis(episodic["small"], args.mults, args.lrs_small)}
    if len(axes["large"]) != len(axes["small"]):
        ap.error(f"the two ladders must have the same length (a run pairs the i-th rungs), got {len(axes['large'])} and {len(axes['small'])}")
    rungs = list(zip(axes["large"], axes["small"]))
    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, tta, None, None), args.temps_checkpoint, report_only=True)
    print(provenance.footer(ctx.duo, tta, args.bn))
    for side in SIDES:
        print(f"{side} {ctx.duo[side.upper()]['NAME']}: episodic rate {episodic[side]:g}, ladder {[f'{v:g}' for v in axes[side]]}")
    print(f"stream: the 4 held-out shifts x {args.rounds} rounds (round r rotated by r) x {args.images} images = "
          f"{4 * args.rounds * args.images} images, severity {ctx.g.severity}; stable = every visit ends within {args.delta:g} points of the frozen member")

    if not args.report_only:
        jobs = [(seed, None) for seed in seeds] + [(seed, r) for seed in seeds for r in rungs]
        label = lambda j: f"seed{j[0]}/" + ("frozen" if j[1] is None else f"L{j[1][0]:g}_S{j[1][1]:g}")
        for seed, r in select_jobs(jobs, label, args.shard, args.only):
            path = run_path(out, seed, r)
            if path.is_file():
                print(f"== {label((seed, r))}: done already ({path.name})")
                continue
            print(f"== {label((seed, r))}")
            rctx = ctx if r is None else with_optim(ctx, large=sgd_optim(r[0], args.momentum), small=sgd_optim(r[1], args.momentum))
            runner, _ = RC.make_runner(rctx, RC.RUNS["frozen" if r is None else "indep"], None, tta)
            parts = CT.tuning_parts(rctx, rounds=args.rounds, seed=seed)
            stop = None if r is None else CT.DeadStop(SIDES, "all", 0)
            t0 = time.time()
            res = CT.run_until(runner, parts, images=args.images, batch_size=rctx.batch_size, workers=rctx.g.workers,
                               T_l=temps.T_l, T_s=temps.T_s, stop=stop)
            meta = {"run": "frozen" if r is None else "indep", "duo": args.duo, "tta": tta, "bn": args.bn, "seed": seed, "rounds": args.rounds,
                    "images": args.images, "stream": "heldout, round r rotated by r", "lrs": r, "momentum": args.momentum,
                    "optim": {s.lower(): rctx.duo[s]["OPTIM"] for s in ("LARGE", "SMALL")}, "T_l": temps.T_l, "T_s": temps.T_s,
                    "batch_size": rctx.batch_size, "stopped": res["stopped"], "dead_at": res["dead_at"], "seen": res["seen"],
                    "seconds": time.time() - t0, "models": {s.lower(): rctx.duo[s]["NAME"] for s in ("LARGE", "SMALL")}}
            print(f"wrote {RC.save_run(path, res, parts[:len(res['part_n'])], meta)} ({res['seen']} images, {meta['seconds'] / 60:.0f} min)")
    if args.shard:
        return 0                # a shard only fills the files; the full run reports

    rows, missing = collect(out, seeds, rungs, args.delta)
    if missing:
        print("runs missing (run them first, see --help):\n  " + "\n  ".join(str(p) for p in missing))
        return 1
    selection = {side: CT.select_side([r for r in rows if r["side"] == side], args.margin) for side in SIDES}
    names = {side: provenance.model_name(ctx.duo[side.upper()]["NAME"]) for side in SIDES}
    warnings = []
    for side in SIDES:
        sel = selection[side]
        print(f"\n== {side}: {names[side]} (episodic rate {episodic[side]:g})")
        print(f"  {'rate':>9}  stable  worst late gain (visit)            dead at   mean acc   images")
        for r in sorted((r for r in rows if r["side"] == side), key=lambda r: (r["lr"], r["seed"])):
            dead = "-" if r["dead_at"] is None else f"{r['dead_at'] // 1000}k"
            print(f"  {r['lr']:9.3g}  {'yes' if r['stable'] else 'NO':>6}  {r['min_gain']:+6.1f}  {r['worst_visit']:<30} {dead:>8}  {r['acc']:8.2f}  {r['images']:7d}"
                  + (f"  seed {r['seed']}" if len(seeds) > 1 else ""))
        for k in ("frontier", "conservative", "stress"):
            print(f"  {k:<13}: {sel[k]:g}" if sel[k] is not None else f"  {k:<13}: none")
        warnings += [f"{side}: {w}" for w in sel["warnings"]]
    for w in warnings:
        print("WARNING " + w)
    save = {"duo": args.duo, "tta": tta, "bn": args.bn, "seeds": seeds, "episodic": episodic, "ladders": axes,
            "stream": {"held_out": ctx.g.tuning_corruptions, "rounds": args.rounds, "images": args.images, "severity": ctx.g.severity,
                       "rotation": "round r starts at the r-th shift"},
            "criteria": {"delta": args.delta, "margin": args.margin, "late_frac": CT.LATE_FRAC,
                         "dead": {"window": CT.DEAD_WINDOW, "acc": CT.DEAD_ACC, "windows": CT.DEAD_WINDOWS}},
            "temperatures": {"T_l": temps.T_l, "T_s": temps.T_s, "checkpoint": str(temps.folder)},
            "momentum": args.momentum, "sides": {s: {k: v for k, v in selection[s].items()} for s in SIDES},
            "rows": rows, "warnings": warnings}
    (out / "results.json").write_text(json.dumps(save, indent=1))
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("gain", "lrs")} for r in rows]).to_csv(out / "ranking.csv", index=False)
    print(f"\nwrote {out / 'results.json'}" + ("" if not warnings else f" (with {len(warnings)} warning(s): extend the ladder before using it)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
