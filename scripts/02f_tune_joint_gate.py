#!/usr/bin/env python3
"""
Tune Joint DD's learning rates TOGETHER with its gate: the pair (lr_large, lr_small) and the gate's beta, on the 4 held-out streams (speckle noise, Gaussian blur,
spatter, saturate; severity 5; none is a test corruption), episodic, one live run per cell.

Script 02b tunes the pair for the static fixed-TS duo (every gate weight 0.5, beta = 0) and script 09 then picks beta at those rates. But the gate is part of the
loss: beta changes what each member learns, so the best rates can depend on beta (a gate that holds a collapsing member down may allow larger rates) and the best
beta on the rates. Here the gate (nnn, proxy batch 256 by default) is inside the loss in EVERY cell, with the Fixed TS of the FROZEN models (T_L, T_S as everywhere),
so a cell is scored by the duo's whole-stream accuracy of the gated prediction, and beta = 0 is exactly 02b's fixed-TS duo (a live run reproduces its cached one).

The search (src/experiments/joint_gate_tuning.py; rules of the reset tuning's joint signal, src/reset/tuning.py, without the detectors): at EVERY beta of the grid
(0, 0.25, 0.5, 0.75, 1; extended to 2 and 4 while the best is the sharpest tried) the pair of rungs is searched on the sqrt-2 grid anchored at the recorded joint pair
(cfgs/duo_lrs/<duo>__joint.yaml; pinned in the tuning folder's config.json at the first call): start diagonal 0.25x, 0.5x, 1x, 2x both members together, the argmax's
neighbourhood, the choice among the best and its neighbours (within one paired SE over the streams, then the lowest rungs). The betas are then compared at their OWN best
pairs (within one paired SE, ties to the lowest beta: 0 = "the gate does not help the joint loss even with its rates retuned"). Guards: a member dead on a stream, or a late
accuracy below the frozen duo's, rules a cell out. The last stage, report only, runs the recorded pair, the best no-gate pair and the chosen pair with and without the gate on
seeds 0 1 2: does tuning with the gate beat the pair tuned without it?

The script is stateless: call it again after each wave. It finds the first stage that is not done and prints, or with --submit submits, the next wave (one GPU job per cell,
all the betas in parallel); with --submit --chain a small CPU job calls it again when the wave has finished OK, until the search and the confirmation are done.

    python scripts/02f_tune_joint_gate.py --duo vitb16_rn50v1 --bn                        # what is done, the next wave (nothing is launched)
    python scripts/02f_tune_joint_gate.py --duo vitb16_rn50v1 --bn --submit --chain       # do it, and keep going
    python scripts/02f_tune_joint_gate.py --duo vitb16_rn50v1 --bn --run 0.0032:0.004:0.5:0 ...   # what a job runs: the cells LR_LARGE:LR_SMALL:BETA:SEED
    python scripts/02f_tune_joint_gate.py --duo vitb16_rn50v1 --bn --record                # after the confirmation: write the chosen pair into cfgs/duo_lrs/<duo>__joint.yaml

--bn as in every script (it picks the frozen models' temperatures); a --num_samples run is a smoke test with its own folder and records nothing. The recorded pair replaces
the joint regime's pair, which script 03's Joint row, script 09 and the reset tuning's joint signal read (their caches are keyed by it): the chosen beta goes with it
(`gate:` in the file) and is NOT a default of scripts 03b / 09, which take --beta / --betas. Files: <paths.out>/results/tune_joint_gate/<duo>__tent_bn__<proxy>_bt<b_t>/
(runs/<cell>.json, config.json, results.json, report.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.experiments import adaptation_modes as AM
from src.experiments import criteria as C
from src.experiments import cli
from src.experiments import joint_gate_tuning as JG
from src.reliability.proxies import pure
from src.reset import tuning as TU
from src.utils import duo_lrs

CONFIG_KEYS = ("anchor", "betas", "start_rungs", "search_seeds", "confirm_seeds")


def parse() -> argparse.Namespace:
    ap = cli.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), tta=False, seeds=False)
    ap.add_argument("--proxy", default="nnn", help="the gate's proxy (a stateless key of src/reliability/proxies/pure.PROXIES; default nnn)")
    ap.add_argument("--b_t", type=int, default=256, help="the gate's proxy batch size (default 256, as scripts 03b / 09 and the reset tuning)")
    ap.add_argument("--betas", type=float, nargs="+", default=None, help=f"the gate grid (default {' '.join(f'{b:g}' for b in JG.BETAS)}, then 2 and 4 while the best is the sharpest)")
    ap.add_argument("--start_rungs", type=int, nargs="+", default=None, help=f"the start diagonal, in sqrt-2 rungs of both members (default {' '.join(map(str, JG.START_RUNGS))})")
    ap.add_argument("--search_seeds", type=int, nargs="+", default=None, help="stream-order seeds the search pools: one (default 0) or two")
    ap.add_argument("--confirm_seeds", type=int, nargs="+", default=None, help="seeds of the confirmation (default 0 1 2)")
    ap.add_argument("--run", nargs="+", default=None, metavar="LR_L:LR_S:BETA:SEED", help="worker: run these cells (what a Slurm job does)")
    ap.add_argument("--submit", action="store_true", help="submit the next wave to Slurm (one GPU job per cell) instead of only printing it")
    ap.add_argument("--chain", action="store_true", help="with --submit: a CPU job calls this script again when the wave has finished OK")
    ap.add_argument("--time", default="00:40:00", help="time limit of one cell's job (default %(default)s)")
    ap.add_argument("--record", action="store_true", help="after the confirmation: record the chosen pair in cfgs/duo_lrs/<duo>__joint.yaml (under the criterion's key)")
    C.add_tol_arg(ap)
    return ap.parse_args()


def load_config(out: Path, args: argparse.Namespace, ctx) -> dict:
    """The tuning's settings, pinned in the folder at the first call (explicit flags replace them, with a note): the anchor must not move when the joint file is
    updated by --record, or the rungs of every stored cell would change meaning."""
    path = out / "config.json"
    cfg = json.loads(path.read_text()) if path.is_file() else {}
    given = {"betas": args.betas, "start_rungs": args.start_rungs, "search_seeds": args.search_seeds, "confirm_seeds": args.confirm_seeds}
    defaults = {"betas": list(JG.BETAS), "start_rungs": list(JG.START_RUNGS), "search_seeds": [0], "confirm_seeds": [0, 1, 2]}
    new = dict(cfg)
    for k, v in given.items():
        if v is not None and v != cfg.get(k):
            if k in cfg:
                print(f"config: {k} {cfg[k]} -> {v}")
            new[k] = list(v)
        new.setdefault(k, defaults[k])
    if "anchor" not in new:
        joint = AM.joint_context(ctx, "tent")
        new["anchor"] = [float(joint.duo[side]["OPTIM"]["LR"]) for side in ("LARGE", "SMALL")]
    if len(new["search_seeds"]) > 2:
        raise SystemExit("--search_seeds: one or two seeds (plan_pair pools at most two)")
    if new != cfg:
        path.write_text(json.dumps(new, indent=1))
    return new


def frozen_floor(ctx, temps, num_samples, out: Path):
    """The frozen duo's late accuracy (the drift guard's floor): from the frozen run the temperatures were fitted on (cached), stored in the folder."""
    from src.experiments import joint_lr_tuning as JL

    key = {"T_l": temps.T_l, "T_s": temps.T_s, "num_samples": num_samples, "batch_size": ctx.batch_size}
    path = out / "frozen_floor.json"
    if path.is_file() and json.loads(path.read_text())["key"] == key:
        return json.loads(path.read_text())["late_duo"]
    tuner = JL.Tuner(ctx, "tent", None, num_samples, temps)
    job = JL.Job("frozen", 0)
    if tuner.missing([job]):
        print("WARNING: the frozen run (the temperatures' fit) is not cached: the late-stream guard is OFF")
        return None
    late = float(tuner.score([job])[job.point]["late_duo"])
    path.write_text(json.dumps({"key": key, "late_duo": late}))
    return late


def reference_check(ctx, temps, store: JG.Store, run: JG.Run, num_samples) -> str | None:
    """beta = 0 is 02b's fixed-TS joint duo: compare the live run with 02b's cached one at the same pair and seed (None if that one is not cached)."""
    from src.experiments import joint_lr_tuning as JL

    if num_samples or not store.has(run) or run.beta != 0:
        return None
    tuner = JL.Tuner(AM.with_learning_rates(ctx, run.lr_large, run.lr_small), "tent", None, None, temps)
    job = JL.Job("joint", run.seed, run.lr_large, run.lr_small)
    if tuner.missing([job]):
        return None
    cached = float(tuner.score([job])[job.point]["duo"])
    live = JG.summarize(store.read(run))["duo"]
    return (f"check: beta 0 at ({run.lr_large:g}, {run.lr_small:g}) seed {run.seed}: live {100 * live:.2f} vs 02b's cached fixed-TS run {100 * cached:.2f} "
            f"({100 * (live - cached):+.2f}; the gate code path at beta 0 must give the same duo up to fp16 / BatchNorm batch noise)")


def main() -> int:
    args = parse()
    if args.proxy not in pure.PROXIES or pure.PROXIES[args.proxy].stateful:
        raise SystemExit(f"--proxy must be a stateless key of {sorted(k for k, v in pure.PROXIES.items() if not v.stateful)} (a stateful proxy needs script 09's --eval)")
    crit = C.Criterion.from_args(args)              # how the pair and beta are chosen (mean unless --criterion worst_shift); the runs are shared between criteria
    tag_mean = f"{args.duo}__tent{cli.bn_tag(args)}__{args.proxy}_bt{args.b_t}" + (f"__n{args.num_samples}" if args.num_samples else "")
    ctx, _, out = cli.setup(args, f"tune_joint_gate/{tag_mean}{C.folder_tag(crit)}", criterion="mean")
    shared = out if (not crit.robust or args.out) else out.parent / tag_mean         # the stored runs (one live run per cell) are the mean study's folder: a cell is a cell whatever chose it
    if crit.robust and shared != out and (shared / "config.json").is_file() and not (out / "config.json").is_file():
        (out / "config.json").write_text((shared / "config.json").read_text())       # the same anchor and start: the same rungs, so the same cells
        print(f"criterion {crit.describe()}: config.json copied from {shared}")
    if crit.robust:
        print(f"criterion: {crit.describe()}; the chosen pair is recorded under {crit.key('tent')}; stored runs are shared with {shared}")
    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, "tent", None, None), None, report_only=True)       # the full frozen fit, also under a --num_samples cap
    print(f"{args.duo}: frozen Fixed TS T_L = {temps.T_l}, T_S = {temps.T_s}  ({temps.folder.name}); gate {args.proxy}, proxy batch {args.b_t}; folder {out}")

    if args.run:                                                                  # ---- worker
        from src.experiments.logit_runs import stream_segments

        segments = stream_segments(["heldout"], ctx.g)
        store = JG.Store(shared)
        for text in args.run:
            JG.execute(ctx, segments, temps, JG.Run.parse(text), store=store, proxy=args.proxy, b_t=args.b_t, num_samples=args.num_samples)
        return 0

    cfg = load_config(out, args, ctx)
    grid = TU.Grid({"large": cfg["anchor"][0], "small": cfg["anchor"][1]})
    rates = lambda pair: (grid.rate("large", pair[0]), grid.rate("small", pair[1]))                      # noqa: E731
    floor = frozen_floor(ctx, temps, args.num_samples, out)
    store = JG.Store(shared, floor)
    run_of = lambda key, seed: JG.Run(*rates(key[0]), key[1], seed)                                  # noqa: E731
    print(f"anchor (the recorded joint pair, rung 0): large {cfg['anchor'][0]:g}, small {cfg['anchor'][1]:g}; rungs are sqrt-2 steps; start diagonal {cfg['start_rungs']}; betas {cfg['betas']}; "
          f"search seeds {cfg['search_seeds']}; late-stream floor (frozen duo) " + ("off" if floor is None else f"{100 * floor:.2f}"))

    pl = JG.plan(lambda key, rot: store.cell(run_of(key, rot)), start=[(k, k) for k in cfg["start_rungs"]], betas=cfg["betas"], rots=cfg["search_seeds"], criterion=crit)
    seen = [r for r in store.runs() if r.seed in cfg["search_seeds"]]
    print(f"\n{len(seen)} search runs stored; the search:")
    for n in pl.notes:
        print(f"  {n}")
    if seen:
        print("\n" + JG.runs_table(store, sorted(seen, key=lambda r: (r.beta, r.lr_large, r.lr_small, r.seed)),
                                   mark=[run_of(pl.final, s) for s in cfg["search_seeds"]] if pl.final else []))
    wave, stage = [], "search"
    if pl.requests:
        wave = list(dict.fromkeys(run_of(k, rot) for k, rot in pl.requests if not store.has(run_of(k, rot))))
    elif pl.final is None:
        print("\nBLOCKED: no cell survives the guards; see the notes above (try --start_rungs lower, or inspect the runs)")
        return 1
    else:
        stage = "confirmation"
        pair, beta = pl.final
        by_beta = {r["beta"]: r["pair"] for r in pl.table}
        old = rates((0, 0))
        zero = rates(by_beta[0.0]) if 0.0 in by_beta else old
        conf = JG.confirm_runs(old, zero, rates(pair), beta, cfg["confirm_seeds"])
        wave = [r for r in conf if not store.has(r)]
        print(f"\nsearch done: gate beta {beta:g}, rates large {rates(pair)[0]:g}, small {rates(pair)[1]:g} (rungs {pair}); per beta:")
        for r in pl.table:
            print(f"  beta {r['beta']:<5g} best pair {rates(r['pair'])}  duo {100 * r['acc']:.2f}{'  <- chosen' if r['chosen'] else ('  (tied)' if r['tied'] else '')}")
        if not wave:
            return finish(args, ctx, temps, store, cfg, pl, conf, out, floor, rates, record=args.record)
    print(f"\n{stage}: {len(wave)} run(s) to do")
    common = (["--bn"] if args.bn else []) + ["--proxy", args.proxy, "--b_t", str(args.b_t)] + (["--num_samples", str(args.num_samples)] if args.num_samples else []) \
        + (["--global", args.global_path] if args.global_path else []) + (["--out", str(args.out)] if args.out else [])
    names = JG.submit(args.duo, wave, common, args.time, go=args.submit, cwd=Path.cwd())
    if args.submit and args.chain:
        JG.chain(out, args.duo, names, sys.argv[1:], go=True, cwd=Path.cwd())
    elif args.chain:
        JG.chain(out, args.duo, names, sys.argv[1:], go=False)
    if not args.submit:
        print("\n(nothing launched: add --submit; --chain makes it run itself)")
    return 0


def finish(args, ctx, temps, store: JG.Store, cfg, pl, conf, out: Path, floor, rates, record: bool) -> int:
    """The search and the confirmation are complete: the report, results.json (the shape of 02b's, so duo_lrs.import_joint reads it), and --record."""
    pair, beta = pl.final
    seeds = cfg["confirm_seeds"]
    new_pair = rates(pair)
    text = [f"Joint DD, {args.duo}: rates tuned with the gate, held-out streams, gate {args.proxy} (proxy batch {args.b_t}), frozen Fixed TS T_L {temps.T_l} / T_S {temps.T_s}",
            f"chosen: gate beta {beta:g}, large {new_pair[0]:g}, small {new_pair[1]:g} (recorded pair: large {rates((0, 0))[0]:g}, small {rates((0, 0))[1]:g})", "",
            "search notes:"] + [f"  {n}" for n in pl.notes]
    text += ["", "the search (seed(s) " + str(cfg["search_seeds"]) + "):", JG.runs_table(store, sorted((r for r in store.runs() if r.seed in cfg["search_seeds"]),
                                                                                  key=lambda r: (r.beta, r.lr_large, r.lr_small, r.seed)),
                                                                                mark=[JG.Run(*new_pair, beta, s) for s in cfg["search_seeds"]])]
    text += ["", f"confirmation, duo accuracy (%) macro over the 4 held-out streams, seeds {seeds} (first row: the recorded pair, no gate = script 03's Joint row):",
             JG.confirm_table(store, conf, seeds)]
    check = reference_check(ctx, temps, store, JG.Run(*rates((0, 0)), 0.0, 0), args.num_samples)
    if check:
        text += ["", check]
    chosen = [JG.summarize(store.read(JG.Run(*new_pair, beta, s))) for s in seeds]
    text += ["", f"chosen configuration per seed: " + ", ".join(f"seed {s}: duo {100 * c['duo']:.2f} (worst {100 * c['worst']:.2f}, late {100 * c['late_duo']:.2f}, large {100 * c['large']:.2f}, small {100 * c['small']:.2f}, "
                                                              f"mean gate weight {c['w_mean']:.3f})" for s, c in zip(seeds, chosen))]
    report = "\n".join(text)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")
    best = {"lr_large": new_pair[0], "lr_small": new_pair[1], "beta": beta, "duo": float(np.mean([c["duo"] for c in chosen])), "worst": float(np.mean([c["worst"] for c in chosen])),
            "late_duo": float(np.mean([c["late_duo"] for c in chosen]))}
    joint = AM.joint_context(ctx, "tent")                                           # the OPTIM blocks (momentum, ...) the joint file already has
    seen_l = sorted({r.lr_large for r in store.runs()})
    seen_s = sorted({r.lr_small for r in store.runs()})
    results = {"duo": args.duo, "tta": "tent", "tta_kwargs": None, "bn": args.bn, "seeds": seeds, "num_samples": args.num_samples, "batch_size": ctx.batch_size,
               "streams": sorted(store.read(conf[0])["streams"]), "temperatures": {"T_l": temps.T_l, "T_s": temps.T_s, "checkpoint": str(temps.folder)},
               "optim": {"large": joint.duo["LARGE"]["OPTIM"], "small": joint.duo["SMALL"]["OPTIM"]}, "current_lrs": {"large": cfg["anchor"][0], "small": cfg["anchor"][1]},
               "lrs_large": seen_l, "lrs_small": seen_s, "best": best, "gate": {"proxy": args.proxy, "b_t": args.b_t, "beta": beta, "tuned_with_the_gate": True},
               "late_floor": floor, "search": {"table": pl.table, "notes": pl.notes}, "criterion": crit.to_json()}
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {out / 'results.json'} and {out / 'report.txt'}")
    if args.num_samples:
        print("capped run (--num_samples): nothing is recorded")
        return 0
    if record:
        print(f"recorded in {duo_lrs.import_joint(out / 'results.json')} (the joint regime's pair: script 03's Joint row, script 09 and the reset tuning's joint signal now read it; "
              f"the gate beta {beta:g} is stored with it, scripts 03b / 09 take --beta / --betas)")
    else:
        print(f"not recorded: add --record (or python -m src.utils.duo_lrs --import_joint {out / 'results.json'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
