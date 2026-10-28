#!/usr/bin/env python3
"""
Live reset runs (Phase 4, R-E): Vanilla DD (both members adapt on their own entropy, the gate outside the loop) over one long continual stream
with a TRIGGER deciding when a member goes back to source (src/reset/live_runs.py). Both members run at FIXED STRESS RATES (default ViT 1.4e-4,
ResNet 5e-4: both collapse without a reset), so the comparison is between the ways of deciding a reset. Conditions (`--conditions`):

    none            no reset
    own_fast_vs_slow       the rule: the (configuration, threshold) that scripts/37b_screen_resets_stress.py chose at `--budget` false alarms per
                    1,000 proxy batches (read from `--screen_json`; label-free), ONE setting for both members
    own_fast_vs_slow@<d>   the same with its threshold delta replaced, every other parameter as screened
    own_fast_vs_slow@pmL<af>-<as>-<W0>-<d>_S<af>-<as>-<W0>-<d>   PER-MEMBER settings (averaging speeds, warm-up, threshold of the large and the
                    small member; src/reset/member_rules.py): the default of the tuning (scripts/40_tune_reset.py member)
    fixed<N>        both members every N adaptation steps (fixed1000 = RDumb's schedule; label-free and blind)
    random~<cond>~d<D>   the control for detection: each member is reset as often as in the run of condition <cond> (same stream, rates, seed; it must exist),
                    at random steps (draw D), at least 80 steps apart
    boundary        both members, and the gate, at every shift boundary: the episodic protocol as one stream (a reference: it is told the shifts)
    frozen          no adaptation, the same images (the label reference; cheap)
(The other rules, the staggered schedule and the oracle trigger were dropped with their code: git tag `pre-reset-cleanup`.)

Streams: `--stream heldout` (the 4 held-out shifts x 4 rounds, `--rotations K ...` picks the opening shift) or `--stream test` (the 15 corruptions
once, CoTTA's standard order; `--order K` one of its random orders). `--lrs LR_L LR_S` may be repeated: one run per rate pair (the ladders of the tuning).
Every run is the whole stream, no early stop. Files, in <paths.out>/results/reset_live/<duo>__tent_bn/:
live-<condition>[__heldout][__rot<K>]__L<lr>_S<lr>__<proxy>_bt<b_t>_beta<beta>__seed<s>.npz (+ .gate.json), in the format of continual.save_run
(per-sample correctness of ts / gate / large / small; the resets, with their steps, are in its `meta`), and live-frozen[...]__seed<s>.npz.

    sbatch --array=0-23 -t 01:50:00 slurm/job.sh scripts/38_run_reset_live.py --duo vitb16_rn50v1 --bn --stream heldout --rotations 0 2 \\
        --conditions none boundary fixed250 fixed500 fixed1000 fixed2000 own_fast_vs_slow ... --shard auto
    python scripts/38_run_reset_live.py --duo vitb16_rn50v1 --bn --stream heldout --rotations 0 2 --conditions ... --report_only    # the table

`--max_images N` stops each run after N images (smoke tests, parity checks); a run that stopped early is saved under a name that says so.

`--extra_gates NAME ...` (or `default`) scores more gates on the same members' logits in the same run (src/reset/gate_variants.py: other betas, routing after a reset,
another proxy), outside the loop, so they change nothing in the run (self-tested); the members' logits are not stored, which is why it has to happen in the run. Their
correctness is stored as the series `gate__<name>` (`continual.load_run(path, extra=True)`), in <paths.out>/results/reset_gates/<duo>__tent_bn/ (the same file names as
the stored runs, hence a folder of their own), read by `scripts/45_gate_ablation.py --multigate`. One call is one (condition, rates, order); slurm/gate_routing.sh submits
the set of the final test's rules:

    sbatch -t 01:50:00 slurm/job.sh scripts/38_run_reset_live.py --duo vitb16_rn50v1 --bn --stream test --beta 0.5 --conditions own_cusum@0.25 --lrs 0.00028 0.001 \\
        --order 0 --extra_gates default
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.experiments import adaptation_modes as AM
from src.experiments import continual as RC
from src.experiments import continual_tuning as CT
from src.experiments.cli import add_common_args, bn_tag, select_jobs, setup
from src.experiments.grids import sgd_optim
from src.experiments.joint_dd import PureProxyGate
from src.experiments.logit_runs import with_optim
from src.reset import live_report as LV
from src.reset import gate_variants as GV
from src.reset import live_runs as LR
from src.reset import member_rules as MR
from src.reset.controller import RuleSpec
from src.utils import provenance

SIDES = ("large", "small")
DEFAULT_SCREEN = "{out}/results/screen_resets_stress/tent__{duo}_bn__SHT__nnn_bt256__n10.json"


def rate_str(lr: float) -> str:
    mant, exp = f"{lr:.2e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"


def with_threshold(spec: RuleSpec, delta: float) -> RuleSpec:
    """`spec` with the detection threshold delta replaced (nothing else changes)."""
    j = spec.to_json()
    j["parts"]["own"]["delta"] = float(delta)
    return RuleSpec.from_json(j)


def split_condition(name: str) -> tuple[str, float | None]:
    """'own_fast_vs_slow@0.7' -> ('own_fast_vs_slow', 0.7); 'own_fast_vs_slow' -> ('own_fast_vs_slow', None)."""
    base, _, thr = name.partition("@")
    return base, (float(thr) if thr else None)


def make_trigger(name: str, rules: dict[str, RuleSpec]) -> LR.Trigger:
    if name == "none":
        return LR.NoTrigger()
    if name == "boundary":
        return LR.BoundaryTrigger()
    if (m := re.fullmatch(r"fixed(\d+)", name)):
        return LR.ScheduleTrigger(int(m.group(1)))
    if MR.parse_condition(name) is not None:                      # per-member settings of own_fast_vs_slow (src/reset/member_rules.py)
        return LR.RuleTrigger(MR.specs_of(name, rules[MR.FAMILY]), name=name)
    base, thr = split_condition(name)
    if base in rules:
        return LR.RuleTrigger(rules[base] if thr is None else with_threshold(rules[base], thr), name=name)
    raise ValueError(f"unknown condition {name!r}: none, boundary, frozen, fixed<N>, random~<cond>~d<D> or {MR.FAMILY}[@threshold]")


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), tta=False)
    ap.add_argument("--conditions", nargs="+", required=True, help="see above")
    ap.add_argument("--stream", choices=("test", "heldout"), default="test")
    ap.add_argument("--rounds", type=int, default=4, help="held-out stream: rounds of the 4 shifts (default %(default)s)")
    ap.add_argument("--rotations", type=int, nargs="+", default=[0], metavar="K", help="held-out stream: which shift opens it, 0..3")
    ap.add_argument("--order", type=int, default=None, metavar="K", help="test stream: one of CoTTA's 10 random orders (default: the standard order)")
    ap.add_argument("--lrs", type=float, nargs=2, action="append", default=None, metavar=("LR_L", "LR_S"),
                    help="the members' fixed rates; repeat for a ladder, one run per pair (default 1.4e-4 5e-4)")
    ap.add_argument("--screen_json", default=None, help="the screening result the rules are read from (default: the SHT screen of this duo)")
    ap.add_argument("--budget", type=float, default=1.0, help="false-alarm budget (per 1,000 proxy batches) whose selected rules are used")
    ap.add_argument("--images", type=int, default=50000, help="images per corruption (default: all)")
    ap.add_argument("--max_images", type=int, default=None, help="stop each run after this many images (smoke tests)")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--b_t", type=int, default=256)
    ap.add_argument("--extra_gates", nargs="+", default=[], metavar="GATE", help="more gates scored on the same logits (see above); `default` = " + " ".join(GV.DEFAULT_SET))
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--temps_checkpoint", default=None, metavar="DIR")
    args = ap.parse_args()
    tta = "tent"
    if any(not 0 <= k < 4 for k in args.rotations) or (args.stream == "test" and args.rotations != [0]):
        ap.error("--rotations are 0..3 and only for --stream heldout")

    args.extra_gates = list(dict.fromkeys(n for g in args.extra_gates for n in (GV.DEFAULT_SET if g == "default" else (g,))))
    for n in args.extra_gates:
        try:
            GV.parse(n)
        except ValueError as e:
            ap.error(str(e))
    if args.extra_gates and any(c.startswith("random~") for c in args.conditions):
        ap.error("--extra_gates cannot go with a random~ condition (its reference run is in the other folder)")
    tag = f"{args.duo}__{tta}{bn_tag(args)}"
    ctx, seeds, out = setup(args, f"{'reset_gates' if args.extra_gates else 'reset_live'}/{tag}", default_seeds=[0])
    proxy = args.proxy or ctx.g.gate["proxy"]
    proxy = {"nuclear_norm": "nnn", "ac_mc": "msp"}.get(proxy, proxy)
    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, tta, None, None), args.temps_checkpoint, report_only=True)

    screen_path = Path(args.screen_json) if args.screen_json else Path(DEFAULT_SCREEN.format(out=ctx.g.path("out"), duo=args.duo))
    rules: dict[str, RuleSpec] = {}
    rule_names = sorted({(MR.FAMILY if MR.parse_condition(c) else split_condition(c)[0]) for c in args.conditions
                         if c not in ("none", "boundary", "frozen") and not re.fullmatch(r"fixed\d+", c) and not c.startswith("random~")} & {MR.FAMILY})
    if rule_names:
        if not screen_path.is_file():
            raise SystemExit(f"{screen_path} is missing: run scripts/37b_screen_resets_stress.py first (or pass --screen_json)")
        screen = json.loads(screen_path.read_text())
        chosen = screen["selected"].get(str(float(args.budget)))
        if chosen is None:
            raise SystemExit(f"{screen_path} has no budget {args.budget}: {sorted(screen['selected'])}")
        for name in rule_names:
            if chosen.get(name) is None:
                raise SystemExit(f"no rule {name!r} within budget {args.budget} in {screen_path}: {sorted(k for k, v in chosen.items() if v)}")
            rules[name] = RuleSpec.from_json(chosen[name]["spec"])
    for c in args.conditions:                                                     # fail on a bad name now, not after hours in the queue
        if c.startswith("random~"):
            if not re.fullmatch(r"random~[^~]+~d\d+", c):
                raise SystemExit(f"{c!r}: write random~<reference condition>~d<draw>")
        elif c != "frozen":
            make_trigger(c, rules)

    pairs = list(dict.fromkeys(tuple(p) for p in (args.lrs or [[1.4e-4, 5e-4]])))

    def stem(cond: str, seed: int, rot: int, lrs: tuple) -> str:
        return LV.run_stem(cond, stream=args.stream, rot=rot, seed=seed, lrs=lrs, proxy=proxy, b_t=args.b_t, beta=args.beta, max_images=args.max_images,
                           order=args.order)

    def path_of(cond: str, seed: int, rot: int, lrs: tuple) -> Path:
        return out / f"{stem(cond, seed, rot, lrs)}.npz"

    def parts_of(rctx, seed: int, rot: int):
        if args.stream == "heldout":
            return CT.tuning_parts(rctx, rounds=args.rounds, seed=seed, rotation=rot)
        return RC.stream_parts(rctx, rounds=1, seed=seed, order=args.order, stream="test")

    names = {s: provenance.model_name(ctx.duo[s.upper()]["NAME"]).split(" (")[0] for s in SIDES}
    print(provenance.footer(ctx.duo, tta, args.bn))
    print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}; gate {proxy}, b_t {args.b_t}, beta {args.beta:g}; members at "
          + "; ".join(f"{names['large']} {rate_str(l)}, {names['small']} {rate_str(s_)}" for l, s_ in pairs) + f"; stream {args.stream}; conditions {', '.join(args.conditions)}")
    # the frozen run does not depend on the rates: one job, whatever the number of pairs
    jobs = [(cond, seed, rot, lrs) for rot in args.rotations for seed in seeds for lrs in pairs for cond in args.conditions if cond != "frozen" or lrs == pairs[0]]

    if not args.report_only:
        for cond, seed, rot, lrs in select_jobs(jobs, lambda j: f"{j[0]}/seed{j[1]}/rot{j[2]}/L{rate_str(j[3][0])}_S{rate_str(j[3][1])}", args.shard, args.only):
            path = path_of(cond, seed, rot, lrs)
            if path.is_file():
                print(f"== {cond}/seed{seed}/rot{rot}/L{rate_str(lrs[0])}_S{rate_str(lrs[1])}: done already ({path.name})")
                continue
            print(f"== {cond}, seed {seed}" + (f", rotation {rot}" if args.stream == "heldout" else "") + f" -> {path.name}", flush=True)
            t0 = time.time()
            rctx = with_optim(ctx, large=sgd_optim(lrs[0], args.momentum), small=sgd_optim(lrs[1], args.momentum))
            parts = parts_of(rctx, seed, rot)
            common = dict(images=args.images, batch_size=rctx.batch_size, workers=rctx.g.workers, T_l=temps.T_l, T_s=temps.T_s)
            meta = {"duo": ctx.duo["NAME"], "tta": tta, "bn": ctx.bn, "seed": seed, "stream": args.stream, "rounds": args.rounds if args.stream == "heldout" else 1,
                    "rotation": rot if args.stream == "heldout" else None, "order": args.order, "images": args.images, "max_images": args.max_images,
                    "condition": cond, "lrs": list(lrs), "momentum": args.momentum, "proxy": proxy, "beta": args.beta, "b_t": args.b_t,
                    "T_l": temps.T_l, "T_s": temps.T_s, "batch_size": rctx.batch_size, "budget": args.budget, "extra_gates": args.extra_gates,
                    "screen_json": str(screen_path) if rules else None,
                    "optim": {s.lower(): rctx.duo[s]["OPTIM"] for s in ("LARGE", "SMALL")}, "models": {s.lower(): rctx.duo[s]["NAME"] for s in ("LARGE", "SMALL")}}
            if cond == "frozen":
                runner, _ = RC.make_runner(ctx, RC.RUNS["frozen"], None, tta)
                res = CT.run_until(runner, parts, max_images=args.max_images, **common)
                res["resets"] = []
            else:
                gate = PureProxyGate(proxy, beta=args.beta, b_t=args.b_t, T_l=temps.T_l, T_s=temps.T_s, num_classes=1000)
                extra = {n: GV.make_gate(n, gate, T_l=temps.T_l, T_s=temps.T_s, b_t=args.b_t) for n in args.extra_gates}
                runner, _ = RC.make_runner(rctx, RC.RUNS["indep"], gate, tta)
                if cond.startswith("random~"):
                    _, ref, draw = cond.split("~")
                    ref_path = path_of(ref, seed, rot, lrs)
                    if not ref_path.is_file():
                        raise SystemExit(f"{cond}: the reference run {ref_path.name} does not exist yet")
                    ref_meta = RC.load_run(ref_path)["meta"]
                    trigger = LR.RandomTrigger(ref_meta["n_resets"], ref_meta["seen"] // rctx.batch_size, draw=int(draw[1:]), name=cond)
                else:
                    trigger = make_trigger(cond, rules)
                res = LR.run_live(runner, parts, trigger=trigger, gate=gate, max_images=args.max_images, extra_gates=extra, **common)
                meta["trigger"] = res["trigger"]
            meta.update(seconds=time.time() - t0, seen=res["seen"], stopped=res["stopped"], resets=res["resets"],
                        n_resets={s: sum(r["member"] == s for r in res["resets"]) for s in SIDES})
            visited = parts[:len(res["part_n"])]
            RC.save_run(path, res, visited, {**meta, "run": "frozen" if cond == "frozen" else "indep"})
            acc = {s: float(v.mean() * 100) for s, v in res["correct"].items()}
            print(f"wrote {path.name}: {res['seen']} images, {meta['seconds'] / 60:.0f} min; accuracy "
                  + ", ".join(f"{s} {a:.1f}" for s, a in acc.items() if not s.startswith("frozen")) + f"; resets {meta['n_resets']}", flush=True)
    if args.shard:
        return 0

    # ---- the table: whole-stream accuracy of every condition and the resets it made
    absent = [str(path_of(c, s, r, lr)) for c, s, r, lr in jobs if not path_of(c, s, r, lr).is_file()]
    if absent:
        print("runs missing (run them first, see --help):\n  " + "\n  ".join(absent))
        return 1
    summary = {}
    for rot in args.rotations:
        for seed in seeds:
            for lrs in pairs:
                where = f"L{rate_str(lrs[0])}_S{rate_str(lrs[1])}"
                print(f"\n== {args.stream}{f' rotation {rot}' if args.stream == 'heldout' else ''}, seed {seed}, rates {where}: mean accuracy over the whole stream (%), resets per member")
                print(f"{'condition':22s} {'duo (gate)':>10s} {'fixed TS':>9s} {'ViT':>6s} {'ResNet':>7s}   resets (ViT / ResNet)")
                for cond in args.conditions:
                    run = RC.load_run(path_of(cond, seed, rot, lrs), extra=bool(args.extra_gates))
                    m = run["meta"]
                    key = f"{cond}/seed{seed}/rot{rot}" + (f"/{where}" if len(pairs) > 1 else "")
                    if cond == "frozen":
                        print(f"{cond:22s} {'':>10s} {100 * run['ts'].mean():9.1f} {100 * run['large'].mean():6.1f} {100 * run['small'].mean():7.1f}")
                        summary[key] = {"ts": float(run["ts"].mean() * 100), "large": float(run["large"].mean() * 100), "small": float(run["small"].mean() * 100)}
                        continue
                    print(f"{cond:22s} {100 * run['gate'].mean():10.1f} {100 * run['ts'].mean():9.1f} {100 * run['large'].mean():6.1f} {100 * run['small'].mean():7.1f}"
                          f"   {m['n_resets']['large']:4d} / {m['n_resets']['small']:4d}")
                    summary[key] = {s: float(run[s].mean() * 100) for s in ("gate", "ts", "large", "small")} | {"n_resets": m["n_resets"]}
                    for n in args.extra_gates:
                        summary[key][f"gate__{n}"] = float(run[f"gate__{n}"].mean() * 100)
                        print(f"  {'gate ' + n:20s} {summary[key][f'gate__{n}']:10.2f} {summary[key][f'gate__{n}'] - summary[key]['ts']:+9.2f} vs the fixed-TS duo")
    (out / f"summary__{args.stream}.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out / f'summary__{args.stream}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
