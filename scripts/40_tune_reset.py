#!/usr/bin/env python3
"""
Status, selection and launch commands of the reset tuning (plan.md, "Reset tuning and evaluation protocol"; the rules are src/reset/tuning.py).
It reads the stored live runs of scripts/38_run_reset_live.py on the held-out stream (gate beta 0.5), applies the pre-registered rules and prints what is
chosen and what still has to be run. `--submit` submits those runs (one array per condition and rotation, QOS high from slurm/job.sh); without it the
sbatch commands are only printed.

    python scripts/40_tune_reset.py rates --cond fixed1000                       # decoupled: the members' rates, one ladder per member
    python scripts/40_tune_reset.py rates --cond own_fast_vs_slow@0.7
    python scripts/40_tune_reset.py rates --cond gap_fast_vs_slow@1 --coupled    # the pair of rates, scored by the gated duo
    python scripts/40_tune_reset.py baseline2                                    # the no-reset baseline: the 04b ladder plus rotation 2
    python scripts/40_tune_reset.py threshold --family own_fast_vs_slow --base 0.7   # the threshold ladder at the chosen rates (Stage 3)
    python scripts/40_tune_reset.py rates --cond own_fast_vs_slow@0.5 --around 1.4e-4 5e-4   # recheck the rates one rung either side (Stage 3)
    python scripts/40_tune_reset.py confirm --cond fixed1000                     # the chosen pair on all four rotations (rotations 1, 3 validate)

A condition is what scripts/38 takes: noreset is `none`, a schedule `fixed<N>`, a rule `family@threshold` (the threshold replaces the screened one).
Repeat the command after each wave finishes: it prints the next wave until `chosen` appears. Writes <out>/results/reset_tune/<cond>__<command>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.reset import tuning as TU
from src.utils.config import load_global

BETA = 0.5
TIME = "01:50:00"


def rate_g(x: float) -> str:
    return f"{x:g}"


def runs_for(needs: dict[str, list[tuple[int, int]]], rate_of) -> dict[int, list[tuple[float, float]]]:
    """Pair the two members' requested rungs into runs, per rotation: `needs[side]` = [(rung, rotation)]; a member without a partner request gets the
    diagonal partner (a decoupled method does not care which)."""
    out: dict[int, list[tuple[float, float]]] = {}
    for rot in sorted({r for side in TU.SIDES for _, r in needs[side]}):
        a = [k for k, r in needs["large"] if r == rot]
        b = [k for k, r in needs["small"] if r == rot]
        for i in range(max(len(a), len(b))):
            kv = a[i] if i < len(a) else b[i] + 2
            ks = b[i] if i < len(b) else kv - 2
            out.setdefault(rot, []).append((rate_of("large", kv), rate_of("small", ks)))
    return {r: list(dict.fromkeys(p)) for r, p in out.items()}


def launch(args, duo: str, cond: str, by_rot: dict[int, list[tuple[float, float]]], tag: str) -> list[str]:
    cmds = []
    for rot, pairs in sorted(by_rot.items()):
        lrs = " ".join(f"--lrs {rate_g(l)} {rate_g(s)}" for l, s in pairs)
        cmds.append(f"sbatch --array=0-{len(pairs) - 1} -t {TIME} -J {tag}_r{rot} --exclude=kn117,kn101 slurm/job.sh scripts/38_run_reset_live.py --duo {duo} --bn "
                    f"--stream heldout --rotations {rot} --conditions {cond} --beta {BETA:g} {lrs} --shard auto")
    for c in cmds:
        print(("SUBMIT  " if args.submit else "to run  ") + c)
        if args.submit:
            print("   ->", subprocess.run(c, shell=True, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip())
    return cmds


def show_member(plan: TU.Plan, side: str, rate_of) -> None:
    print(f"  {side}: " + "; ".join(plan.notes))
    for r in plan.table:
        print(f"    rung {r['rung']:3d}  {rate_of(side, r['rung']):.3g}  acc(rot0) {100 * r['acc_rot0']:6.2f}  acc(0+2) " +
              (f"{100 * r['acc']:6.2f}" if r["acc"] is not None else "     -") + f"  resets/1000 " + (f"{r['resets']:6.2f}" if r["resets"] is not None else "     -") +
              ("  tied" if r["tied"] else "") + ("  <- chosen" if r["chosen"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("rates", "baseline2", "confirm", "threshold"))
    ap.add_argument("--cond", default=None)
    ap.add_argument("--coupled", action="store_true", help="the members are coupled (the gap rule): tune the pair on the gated duo")
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--rates", type=float, nargs=2, default=None, metavar=("LR_L", "LR_S"), help="confirm / threshold: the pair (default: the chosen one of the rates run)")
    ap.add_argument("--around", type=float, nargs=2, default=None, metavar=("LR_L", "LR_S"), help="rates: start the ladders one rung either side of this pair instead of the x2 ladder")
    ap.add_argument("--family", default=None, help="threshold: the rule family, e.g. own_fast_vs_slow")
    ap.add_argument("--base", type=float, default=None, help="threshold: the screened threshold the sqrt-2 grid is anchored at")
    ap.add_argument("--rung", type=int, nargs="+", default=[-2, -1, 0, 1], help="threshold: the starting rungs of the ladder (default %(default)s: thresholds base * 2**(k/2))")
    args = ap.parse_args()
    g = load_global()
    folder = g.path("out") / "results" / "reset_live" / f"{args.duo}__tent_bn"
    outdir = g.path("out") / "results" / "reset_tune"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.what == "baseline2":
        d = json.loads((g.path("out") / "results" / "tune_continual_lr" / f"{args.duo}__tent_bn" / "results.json").read_text())
        ladders = d["ladders"]
        n = len(ladders["large"])
        store = TU.Store(folder, gate=f"nnn_bt256_beta{BETA:g}")
        b04 = g.path("out") / "results" / "tune_continual_lr" / f"{args.duo}__tent_bn" / "runs"
        cache: dict[int, TU.Cell] = {}

        def cell_of(side):
            def f(i: int, rot: int):
                if not 0 <= i < n:
                    return None
                if rot == 0:                                       # the 04b run of this rung (pairs the i-th rungs of the two ladders)
                    if i not in cache:
                        cache[i] = TU.load_cell(b04 / f"seed0__L{rate_g(ladders['large'][i])}_S{rate_g(ladders['small'][i])}.npz")
                    return cache[i]
                return store.cell("none", rot, **{side: ladders[side][i]})
            return f

        needs, plans = {}, {}
        for side in TU.SIDES:
            plans[side] = TU.plan_1d(start=tuple(range(n)), neighbours=lambda i: (i - 1, i + 1), cell_of=cell_of(side), series=side, side=side)
            needs[side] = plans[side].requests
            print(f"{side}: " + "; ".join(plans[side].notes))
            for r in plans[side].table:
                print(f"    idx {r['rung']:2d}  {ladders[side][r['rung']]:.3g}  acc(rot0) {100 * r['acc_rot0']:6.2f}  acc(0+2) " + (f"{100 * r['acc']:6.2f}" if r["acc"] is not None else "     -") +
                      ("  tied" if r["tied"] else "") + ("  <- chosen" if r["chosen"] else ""))
        by_rot = runs_for(needs, lambda side, i: ladders[side][i])
        if by_rot:
            launch(args, args.duo, "none", by_rot, "tune_none")
        else:
            chosen = {s: ladders[s][plans[s].chosen] for s in TU.SIDES}
            print(f"\nchosen (accuracy-tuned, no reset): ViT {chosen['large']:g}, ResNet {chosen['small']:g}; the 04b conservative pair is 1.26e-05 / 2.2e-05")
            (outdir / "none__baseline2.json").write_text(json.dumps({"chosen": chosen, "tables": {s: plans[s].table for s in TU.SIDES}}, default=float, indent=1))
        return 0

    if not args.cond and args.what != "threshold":
        ap.error("--cond is required")
    store = TU.Store(folder, gate=f"nnn_bt256_beta{BETA:g}")
    cond = args.cond
    state = outdir / f"{cond}__rates.json"

    if args.what == "rates" and args.coupled:
        plan = TU.plan_pair(cell_of=lambda c, rot: store.cell(cond, rot, large=TU.rate("large", c[0]), small=TU.rate("small", c[1])))
        print(f"{cond} (coupled): " + "; ".join(plan.notes))
        for r in plan.table:
            print(f"    cell {r['cell']}  ViT {TU.rate('large', r['cell'][0]):.3g} ResNet {TU.rate('small', r['cell'][1]):.3g}  acc(0+2) {100 * r['acc']:6.2f}  resets/1000 {r['resets']:6.2f}"
                  + ("  tied" if r["tied"] else "") + ("  <- chosen" if r["chosen"] else ""))
        by_rot: dict[int, list[tuple[float, float]]] = {}
        for (kv, ks), rot in plan.requests:
            by_rot.setdefault(rot, []).append((TU.rate("large", kv), TU.rate("small", ks)))
        if by_rot:
            launch(args, args.duo, cond, by_rot, f"tune_{cond.split('@')[0]}")
        elif plan.chosen is not None:
            kv, ks = plan.chosen
            print(f"\nchosen: ViT {TU.rate('large', kv):g}, ResNet {TU.rate('small', ks):g}")
            state.write_text(json.dumps({"cond": cond, "coupled": True, "chosen": [TU.rate("large", kv), TU.rate("small", ks)], "table": plan.table}, default=float, indent=1))
        return 0

    if args.what == "rates":
        needs, plans = {}, {}
        for side in TU.SIDES:
            k0 = TU.rung(side, args.around[TU.SIDES.index(side)]) if args.around else None
            plans[side] = TU.plan_1d(start=TU.X2[side] if k0 is None else (k0 - 1, k0, k0 + 1), neighbours=lambda k: (k - 1, k + 1), series=side, side=side,
                                     cell_of=lambda k, rot, side=side: store.cell(cond, rot, **{side: TU.rate(side, k)}))
            needs[side] = plans[side].requests
            show_member(plans[side], side, TU.rate)
        by_rot = runs_for(needs, TU.rate)
        if by_rot:
            launch(args, args.duo, cond, by_rot, f"tune_{cond.split('@')[0]}")
        elif all(plans[s].chosen is not None for s in TU.SIDES):
            chosen = [TU.rate(s, plans[s].chosen) for s in TU.SIDES]
            print(f"\nchosen: ViT {chosen[0]:g}, ResNet {chosen[1]:g} (each member on its own accuracy); the pair is confirmed with `confirm`")
            state.write_text(json.dumps({"cond": cond, "coupled": False, "chosen": chosen, "tables": {s: plans[s].table for s in TU.SIDES}}, default=float, indent=1))
        return 0

    if args.what == "threshold":
        if not (args.family and args.base):
            ap.error("threshold needs --family and --base")
        pair = args.rates or json.loads((outdir / f"{args.family}@{args.base:g}__rates.json").read_text())["chosen"]
        cond_of = lambda k: f"{args.family}@{TU.threshold(args.base, k):g}"
        plan = TU.plan_1d(start=args.rung, neighbours=lambda k: (k - 1, k + 1), series="gate", side=None, tie_order=lambda k: -k,
                          cell_of=lambda k, rot: store.cell(cond_of(k), rot, large=pair[0], small=pair[1]))
        print(f"{args.family} thresholds at ViT {pair[0]:g}, ResNet {pair[1]:g}: " + "; ".join(plan.notes))
        for r in plan.table:
            print(f"    rung {r['rung']:3d}  threshold {TU.threshold(args.base, r['rung']):g}  acc(rot0) {100 * r['acc_rot0']:6.2f}  acc(0+2) " + (f"{100 * r['acc']:6.2f}" if r["acc"] is not None else "     -")
                  + f"  resets/1000 " + (f"{r['resets']:6.2f}" if r["resets"] is not None else "     -") + ("  tied" if r["tied"] else "") + ("  <- chosen" if r["chosen"] else ""))
        by_rot: dict[int, list[str]] = {}
        for k, rot in plan.requests:
            by_rot.setdefault(rot, []).append(cond_of(k))
        for rot, conds in sorted(by_rot.items()):
            c = (f"sbatch --array=0-{len(conds) - 1} -t {TIME} -J thr_{args.family}_r{rot} --exclude=kn117,kn101 slurm/job.sh scripts/38_run_reset_live.py --duo {args.duo} --bn "
                 f"--stream heldout --rotations {rot} --conditions {' '.join(conds)} --beta {BETA:g} --lrs {rate_g(pair[0])} {rate_g(pair[1])} --shard auto")
            print(("SUBMIT  " if args.submit else "to run  ") + c)
            if args.submit:
                print("   ->", subprocess.run(c, shell=True, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip())
        if plan.chosen is not None and not by_rot:
            thr = TU.threshold(args.base, plan.chosen)
            print(f"\nchosen threshold: {thr:g} (rung {plan.chosen})")
            (outdir / f"{args.family}__threshold.json").write_text(json.dumps({"family": args.family, "base": args.base, "chosen_rung": plan.chosen, "threshold": thr, "pair": pair, "table": plan.table}, default=float, indent=1))
        return 0

    # confirm: the chosen pair on all four rotations; 1 and 3 are validation only
    pair = args.rates or json.loads(state.read_text())["chosen"]
    missing = {r: [(pair[0], pair[1])] for r in (0, 1, 2, 3) if store.cell(cond, r, large=pair[0], small=pair[1]) is None}
    if missing:
        launch(args, args.duo, cond, missing, f"conf_{cond.split('@')[0]}")
        return 0
    print(f"{cond} at ViT {pair[0]:g}, ResNet {pair[1]:g}: whole-stream accuracy (%) per rotation (rotations 1 and 3 are validation)")
    rows = {}
    for r in (0, 1, 2, 3):
        c = store.cell(cond, r, large=pair[0], small=pair[1])
        rows[r] = {s: c.mean(s) * 100 for s in ("gate", "ts", "large", "small")} | {"resets_per_1000": c.resets, "dead": c.dead}
        print(f"  rot {r}: duo {rows[r]['gate']:.1f}  fixed-TS {rows[r]['ts']:.1f}  ViT {rows[r]['large']:.1f}  ResNet {rows[r]['small']:.1f}  resets/1000 ViT {c.resets['large']:.2f} ResNet {c.resets['small']:.2f}")
    (outdir / f"{cond}__confirm.json").write_text(json.dumps({"cond": cond, "pair": pair, "rotations": rows}, default=float, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
