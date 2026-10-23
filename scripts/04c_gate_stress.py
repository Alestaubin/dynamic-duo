#!/usr/bin/env python3
"""
The gate on a member that collapses (src/experiments/gate_stress.py): the independent duo (Vanilla DD, the gate outside the loop) on the test
continual stream (the 15 corruptions in CoTTA's standard order, severity 5, no reset), one member at a CONSERVATIVE rate and the other at a
STRESS rate, both from scripts/04b_tune_continual_lr.py (held-out shifts only):

    A   ViT-B/16 conservative, ResNet-50 stress      the small member should collapse; the gate moves to the large one
    B   ViT-B/16 stress, ResNet-50 conservative      the large member should collapse; the gate moves to the small one

Each experiment is one job: the adaptive run, stopped `--after` images (default 40,000) after the stressed member has collapsed completely
(or at the end of the stream, if it never does), then the frozen run over exactly the same images. `--stream heldout` runs it on the stream
the rates were tuned on (same batches as the 04b run at those rates, with the gate added). Files, in
<paths.out>/results/gate_stress/<duo>__tent_bn/: exp-<name>__L<lr>_S<lr>__seed<s>.npz (+ .gate.json) and the same with __frozen.

    sbatch --array=0-1 -t 02:15:00 slurm/job.sh scripts/04c_gate_stress.py --duo vitb16_rn50v1 --bn --seeds 0 --shard auto     # A and B
    python scripts/04c_gate_stress.py --duo vitb16_rn50v1 --bn --seeds 0 --report_only          # figures and numbers

    --stress_up K        the stress rates K rungs higher on the ladders (when a stress rate does not collapse on the test stream: the
                         held-out stream decided it, the test stream can differ; report both)
    --name X --lr_large L --lr_small S     one experiment at explicit rates (stops after the first member to die)

A tuning result with warnings (a ladder that did not bracket) is refused unless --allow_warnings. A finished experiment is never recomputed.
--report_only writes gate_stress.png/.pdf (all experiments side by side), gate_stress_<name>.png/.pdf, and summary.json (with `_heldout`
after the stem for --stream heldout).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import adaptation_modes as AM
from src.experiments import continual as RC
from src.experiments import continual_report as RR
from src.experiments import gate_stress as GS
from src.experiments.cli import add_common_args, bn_tag, select_jobs, setup
from src.utils import provenance


def resolve_rate(tuning: dict, side: str, role: str, up: int) -> float:
    """The rate of `role` for `side` from the tuning result; the stress rate may be moved `up` rungs up the ladder."""
    lr = tuning["sides"][side][role]
    if lr is None:
        raise SystemExit(f"the tuning names no {role} rate for the {side} member ({tuning['sides'][side]['warnings']}): extend its ladder")
    if role == "stress" and up:
        ladder = tuning["ladders"][side]
        k = ladder.index(lr) + up
        if k >= len(ladder):
            raise SystemExit(f"--stress_up {up}: the {side} ladder ends at {ladder[-1]:g}")
        lr = ladder[k]
    return float(lr)


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), tta=False)
    ap.add_argument("--tuning", default=None, metavar="JSON", help="results.json of 04b (default: the one of this duo under paths.out)")
    ap.add_argument("--experiments", nargs="+", choices=list(GS.EXPERIMENTS), default=list(GS.EXPERIMENTS))
    ap.add_argument("--stress_up", type=int, default=0, metavar="K")
    ap.add_argument("--name", default=None, help="an experiment at explicit rates (needs --lr_large and --lr_small)")
    ap.add_argument("--lr_large", type=float, default=None)
    ap.add_argument("--lr_small", type=float, default=None)
    ap.add_argument("--allow_warnings", action="store_true", help="use a tuning result that has warnings")
    ap.add_argument("--after", type=int, default=40000, help="images to run after the stressed member has collapsed completely (default %(default)s)")
    ap.add_argument("--stream", choices=("test", "heldout"), default="test",
                    help="the test continual stream (default), or the held-out one the rates were tuned on (04b: the 4 held-out shifts x --rounds, "
                         "round r rotated by r): there the stress rates collapse by construction, and no test shift is touched")
    ap.add_argument("--rounds", type=int, default=4, help="held-out stream: rounds of the 4 shifts (default %(default)s, as in 04b)")
    ap.add_argument("--images", type=int, default=50000, help="images per corruption (default: all 50,000 of each)")
    ap.add_argument("--order", type=int, default=None, metavar="K", help="one of CoTTA's 10 random corruption orders (default: the standard order)")
    ap.add_argument("--proxy", default=None, help="the gate's proxy (default: the paper's, nuclear_norm = nnn)")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--b_t", type=int, default=256, help="proxy batch size")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--temps_checkpoint", default=None, metavar="DIR", help="as in script 03 (default: the held-out fit it uses)")
    args = ap.parse_args()
    tta = "tent"
    if (args.name is None) != (args.lr_large is None) or (args.lr_large is None) != (args.lr_small is None):
        ap.error("--name, --lr_large and --lr_small go together")
    if args.order is not None and not 0 <= args.order < 10:
        ap.error(f"--order must be 0..9, got {args.order}")

    tag = f"{args.duo}__{tta}{bn_tag(args)}"
    ctx, seeds, out = setup(args, f"gate_stress/{tag}", default_seeds=[0])
    proxy = args.proxy or ctx.g.gate["proxy"]
    proxy = {"nuclear_norm": "nnn", "ac_mc": "msp"}.get(proxy, proxy)
    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, tta, None, None), args.temps_checkpoint, report_only=True)
    names = {s: provenance.model_name(ctx.duo[s.upper()]["NAME"]).split(" (")[0] for s in GS.SIDES}

    # the experiments: name -> (rates, the members whose complete collapse ends the run, a title)
    experiments = {}
    tuning_path = None
    if args.name:
        experiments[args.name] = ((args.lr_large, args.lr_small), GS.SIDES, f"({args.name}) {names['large']} at {args.lr_large:g}, {names['small']} at {args.lr_small:g}")
    else:
        tuning_path = Path(args.tuning) if args.tuning else ctx.g.path("out") / "results" / "tune_continual_lr" / tag / "results.json"
        tuning = json.loads(tuning_path.read_text())
        if tuning["warnings"] and not args.allow_warnings:
            raise SystemExit(f"{tuning_path} has warnings (extend the ladders, or --allow_warnings):\n  " + "\n  ".join(tuning["warnings"]))
        if (tuning["duo"], tuning["bn"]) != (args.duo, args.bn):
            raise SystemExit(f"{tuning_path} is for {tuning['duo']} (bn {tuning['bn']}), not {args.duo} (bn {args.bn})")
        for name in args.experiments:
            roles = dict(zip(GS.SIDES, GS.EXPERIMENTS[name]))
            lrs = tuple(resolve_rate(tuning, s, roles[s], args.stress_up) for s in GS.SIDES)
            watch = tuple(s for s in GS.SIDES if roles[s] == "stress")
            title = f"({name}) " + ", ".join(f"{names[s]} at {lr:g} ({roles[s]})" for s, lr in zip(GS.SIDES, lrs))
            experiments[name] = (lrs, watch, title)
    print(provenance.footer(ctx.duo, tta, args.bn))
    print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}; gate {proxy}, proxy batch size {args.b_t}, beta {args.beta:g}")
    for name, (lrs, watch, title) in experiments.items():
        print(f"experiment {title}; stops {args.after} images after {'/'.join(watch)} collapses completely")

    def path_of(name: str, seed: int, frozen: bool = False) -> Path:
        lrs = experiments[name][0]
        return out / f"exp-{name}{'__heldout' if args.stream == 'heldout' else ''}__L{lrs[0]:g}_S{lrs[1]:g}__seed{seed}{'__frozen' if frozen else ''}.npz"

    jobs = [(name, seed) for seed in seeds for name in experiments]
    if not args.report_only:
        for name, seed in select_jobs(jobs, lambda j: f"{j[0]}/seed{j[1]}", args.shard, args.only):
            if path_of(name, seed).is_file() and path_of(name, seed, True).is_file():
                print(f"== {name}/seed{seed}: done already ({path_of(name, seed).name})")
                continue
            lrs, watch, title = experiments[name]
            print(f"== {title}, seed {seed}")
            t0 = time.time()
            res, frozen, parts, meta = GS.run_experiment(ctx, lrs=lrs, watch=watch, after=args.after, seed=seed, order=args.order, images=args.images,
                                                         proxy=proxy, beta=args.beta, b_t=args.b_t, T_l=temps.T_l, T_s=temps.T_s,
                                                         momentum=args.momentum, tta=tta, stream=args.stream, rounds=args.rounds)
            meta.update(name=name, title=title, tuning=str(tuning_path) if tuning_path else None, seconds=time.time() - t0)
            RC.save_run(path_of(name, seed, True), frozen, parts, {**meta, "run": "frozen"})
            print(f"wrote {RC.save_run(path_of(name, seed), res, parts, {**meta, 'run': 'indep'})} ({res['seen']} images, {meta['seconds'] / 60:.0f} min)")
    if args.shard:
        return 0

    absent = [str(path_of(n, s, f)) for n, s in jobs for f in (False, True) if not path_of(n, s, f).is_file()]
    if absent:
        print("runs missing (run them first, see --help):\n  " + "\n  ".join(absent))
        return 1
    summary, items = {}, {}
    for seed in seeds:
        for name, (lrs, watch, title) in experiments.items():
            run, frozen = RC.load_run(path_of(name, seed)), RC.load_run(path_of(name, seed, True))
            records = RR.load_gate_records(path_of(name, seed))
            an = GS.analyze(run, frozen, records, args.b_t)
            print(f"\n== {title}, seed {seed}")
            print("\n".join("  " + l for l in GS.describe(an, {**names, "large": names["large"], "small": names["small"]})))
            summary[f"{name}/seed{seed}"] = {"title": title, "lrs": list(lrs), "meta": run["meta"], "analysis": an}
            if seed == seeds[0]:
                items[name] = {"title": title, "run": run, "frozen": frozen, "records": records, "b_t": args.b_t, "analysis": an, "names": names}
    where = (f"held-out stream (4 shifts x {args.rounds} rounds, the tuning stream)" if args.stream == "heldout" else
             f"test stream, {'CoTTA standard order' if args.order is None else f'order {args.order}'}")
    footer = provenance.footer(ctx.duo, tta, args.bn, f"{where}, seed {seeds[0]}; "
                               f"gate {proxy}, b_t {args.b_t}, beta {args.beta:g}; T_L = {temps.T_l}, T_S = {temps.T_s}; "
                               f"stopped {args.after} images after the stressed member's complete collapse")
    suffix = "_heldout" if args.stream == "heldout" else ""
    written = GS.plot_stress(list(items.values()), out / f"gate_stress{suffix}.png", footer=footer)
    for name, item in items.items():
        written += GS.plot_stress([item], out / f"gate_stress{suffix}_{name}.png", footer=footer)
    (out / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("\nwrote " + ", ".join(str(p) for p in written) + f" and {out / f'summary{suffix}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
