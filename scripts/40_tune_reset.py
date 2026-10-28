#!/usr/bin/env python3
"""
Status, selection and launch commands of the reset tuning (docs/reset_tuning.md; the rules are src/reset/tuning.py).
It reads the stored live runs of scripts/38_run_reset_live.py on the held-out stream (gate beta 0.5), applies the pre-registered rules and prints what is
chosen and what still has to be run. `--submit` submits those runs (one array per condition and rotation, QOS high from slurm/job.sh); without it the
sbatch commands are only printed. Repeat the command after each wave finishes: it prints the next wave until the final choice appears.

    python scripts/40_tune_reset.py member                                       # own_fast_vs_slow: each member's detector setting AND rate (the pipeline)
    python scripts/40_tune_reset.py confirm --cond <cond> --rates LR_L LR_S      # the chosen configuration on all four rotations (1 and 3 only validate)

The comparison baselines have only the rate to tune (one ladder per member):

    python scripts/40_tune_reset.py rates --cond fixed1000                       # also fixed500, and own_fast_vs_slow@0.7 (the shared setting)
    python scripts/40_tune_reset.py rates --cond own_fast_vs_slow@0.5 --around 1.4e-4 5e-4   # recheck the rates one rung either side of a pair
    python scripts/40_tune_reset.py baseline2                                    # the no-reset baseline: the 04b ladder plus rotation 2

A condition is what scripts/38 takes: no reset is `none`, a schedule `fixed<N>`, the rule `own_fast_vs_slow@<threshold>` (shared setting) or
`own_fast_vs_slow@pmL..._S...` (per member). Writes <out>/results/reset_tune/<cond>__<command>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
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


FILLER_RUNG = {"large": 6, "small": -5}                      # a pairing filler: the shared setting at its tuned rate (5.6e-4, 8.84e-5): a known cell, only needed when one member has fewer requests


def member_main(args, g, folder: Path, outdir: Path) -> int:
    """Per-member tuning (plan.md, amendment of 2026-10-28): each member's own detector setting AND rate, by `TU.plan_member`; a run pairs one cell request of each member."""
    from src.reset import member_rules as MR
    shared = MR.Setting(args.shared[0], args.shared[1], int(args.shared[2]), args.shared[3])
    cdir = Path(args.candidates) if args.candidates else g.path("out") / "results" / "screen_resets_stress"
    cands = {}
    for side in TU.SIDES:
        d = json.loads((cdir / f"member_detector__{side}.json").read_text())
        cands[side] = [MR.Setting(c["alpha_fast"], c["alpha_slow"], int(c["W0"]), c["delta"]) for c in d["candidates"]]
    store = TU.MemberStore(folder, f"nnn_bt256_beta{BETA:g}", shared, args.shared_cond)
    plans = {s: TU.plan_member(s, lambda st, k, rot, s=s: store.member_cell(s, st, rot, TU.rate(s, k)), shared, cands[s]) for s in TU.SIDES}
    for side, pl in plans.items():
        print(f"{side} ({'ViT' if side == 'large' else 'ResNet'}): candidates " + ", ".join("shared" if c is shared else str(tuple(c)) for c in [shared, *cands[side]]))
        print("   " + "; ".join(pl.notes))
        for r in pl.table:
            print(f"     {str(r['setting']):36s} rate rung {r['rate_rung']:3d}  acc(0+2) {100 * r['acc']:6.2f}  resets/1000 {r['resets']:5.2f}" + ("  tied" if r["tied"] else "") + ("  <- best" if r["chosen"] else ""))
    reqs = {s: list(dict.fromkeys(pl.requests)) for s, pl in plans.items()}
    if any(reqs.values()):
        runs: dict[int, list[tuple]] = {}
        for rot in sorted({r for s in TU.SIDES for _, _, r in reqs[s]}):
            lst = {s: [(st, k) for st, k, r in reqs[s] if r == rot] for s in TU.SIDES}
            for i in range(max(len(lst["large"]), len(lst["small"]))):
                a = lst["large"][i] if i < len(lst["large"]) else (shared, FILLER_RUNG["large"])
                b = lst["small"][i] if i < len(lst["small"]) else (shared, FILLER_RUNG["small"])
                runs.setdefault(rot, []).append((MR.condition(a[0], b[0]), TU.rate("large", a[1]), TU.rate("small", b[1])))
        n = 0
        lock = outdir / "member_submit.lock"                 # a wave takes about 70 minutes: a second submission within 15 minutes is a double trigger (the monitor and a manual call), not a new wave
        if args.submit and lock.exists() and time.time() - lock.stat().st_mtime < 900 and not args.force:
            print(f"NOT SUBMITTED: a wave was submitted {(time.time() - lock.stat().st_mtime) / 60:.0f} minutes ago (lock {lock.name}); use --force to override")
            return 0
        if args.submit:
            lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        for rot, lst in sorted(runs.items()):
            for cond, rl, rs in dict.fromkeys(lst):
                c = (f"sbatch --array=0-0 -t {TIME} -J mem_r{rot}_{n} --exclude=kn117,kn101 slurm/job.sh scripts/38_run_reset_live.py --duo {args.duo} --bn --stream heldout "
                     f"--rotations {rot} --conditions {cond} --beta {BETA:g} --lrs {rate_g(rl)} {rate_g(rs)} --shard auto")
                print(("SUBMIT  " if args.submit else "to run  ") + c)
                if args.submit:
                    print("   ->", subprocess.run(c, shell=True, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip())
                n += 1
        print(f"{n} run(s) requested")
        return 0
    final = {s: plans[s].final for s in TU.SIDES}
    if any(f is None for f in final.values()):
        print("no final choice and no requests: see the notes above")
        return 1
    cond = MR.condition(final["large"][0], final["small"][0])
    rates = [TU.rate("large", final["large"][1]), TU.rate("small", final["small"][1])]
    print(f"\nfinal: {cond} at ViT {rates[0]:g}, ResNet {rates[1]:g}; confirm with `confirm --cond {cond} --rates {rates[0]:g} {rates[1]:g}`")
    (outdir / "member__rates.json").write_text(json.dumps({"cond": cond, "chosen": rates, "settings": {s: list(final[s][0]) for s in TU.SIDES}, "tables": {s: plans[s].table for s in TU.SIDES}}, default=float, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("member", "rates", "baseline2", "confirm"))
    ap.add_argument("--cond", default=None)
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--force", action="store_true", help="member: submit although a wave was submitted less than 15 minutes ago")
    ap.add_argument("--rates", type=float, nargs=2, default=None, metavar=("LR_L", "LR_S"), help="confirm: the pair (default: the chosen one of the rates run)")
    ap.add_argument("--around", type=float, nargs=2, default=None, metavar=("LR_L", "LR_S"), help="rates: start the ladders one rung either side of this pair instead of the x2 ladder")
    ap.add_argument("--shared", type=float, nargs=4, default=[0.05, 0.005, 20, 0.7], metavar=("AF", "AS", "W0", "DELTA"), help="member: the shared setting (a candidate of both members)")
    ap.add_argument("--shared_cond", default="own_fast_vs_slow@0.7", help="member: the plain condition whose runs are the shared setting's cells")
    ap.add_argument("--candidates", default=None, help="member: folder of member_detector__<side>.json (default <out>/results/screen_resets_stress)")
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
            plans[side] = TU.plan_1d(side=side, start=tuple(range(n)), neighbours=lambda i: (i - 1, i + 1), cell_of=cell_of(side))
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

    if args.what == "member":
        return member_main(args, g, folder, outdir)

    if not args.cond:
        ap.error("--cond is required")
    store = TU.Store(folder, gate=f"nnn_bt256_beta{BETA:g}")
    cond = args.cond
    state = outdir / f"{cond}__rates.json"

    if args.what == "rates":
        needs, plans = {}, {}
        for side in TU.SIDES:
            k0 = TU.rung(side, args.around[TU.SIDES.index(side)]) if args.around else None
            plans[side] = TU.plan_1d(side=side, start=TU.X2[side] if k0 is None else (k0 - 1, k0, k0 + 1), neighbours=lambda k: (k - 1, k + 1),
                                     cell_of=lambda k, rot, side=side: store.cell(cond, rot, **{side: TU.rate(side, k)}))
            needs[side] = plans[side].requests
            show_member(plans[side], side, TU.rate)
        by_rot = runs_for(needs, TU.rate)
        if by_rot:
            launch(args, args.duo, cond, by_rot, f"tune_{cond.split('@')[0]}")
        elif all(plans[s].chosen is not None for s in TU.SIDES):
            chosen = [TU.rate(s, plans[s].chosen) for s in TU.SIDES]
            print(f"\nchosen: ViT {chosen[0]:g}, ResNet {chosen[1]:g} (each member on its own accuracy); the pair is confirmed with `confirm`")
            state.write_text(json.dumps({"cond": cond, "chosen": chosen, "tables": {s: plans[s].table for s in TU.SIDES}}, default=float, indent=1))
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
