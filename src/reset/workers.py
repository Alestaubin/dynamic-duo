"""
workers.py
==========
What a Slurm job of the reset tuning runs (`scripts/tune_reset.py --job <spec>`; the specs are src/reset/jobs.py): one run, for any duo, written to the file
the driver looks for (src/reset/layout.py). These are the run bodies of the former scripts 04b (continual rates), 04c (calibration streams) and 38 (live reset
runs), moved here unchanged in what they compute.

    temperatures   the duo's frozen-model fit of T_L, T_S (cached; every other job needs it)
    continual      one rung of the continual-rate ladder, or its frozen reference: both members adapt on the held-out stream (4 shifts x 4 rounds, round r rotated
                   by r, 800,000 images) at (lr_large, lr_small), no reset, stopped once both are dead; the members never see each other (Vanilla DD)

The SIGNAL of a job is the loss the members adapt on: "indep" (the default: each on its own entropy, the gate outside the loop) or "joint" (`"signal": "joint"` in the
spec: both on the gated duo's entropy, the gate inside the loss, its beta part of the run; the duo's prediction is the one the step used). The rates are the signal's own.
    calibration    one calibration stream of the screens: both members at a stress (S) or a healthy (H) pair of rates, the gate attached, then the frozen run over
                   the same images
    live           one live reset run: a condition on one stream at two rates (conditions: `none`, `boundary`, `fixed<N>`, `own_fast_vs_slow[@<delta>]`,
                   `own_fast_vs_slow@pmL...-..._S...`, `random~<condition>~d<draw>`, `frozen`); per-sample correctness of the fixed-TS duo, the gated duo and
                   both members, and every reset
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from src.experiments import adaptation_modes as AM
from src.experiments import continual as RC
from src.experiments import continual_tuning as CT
from src.experiments import gate_stress as GS
from src.experiments.cli import pin_threads
from src.experiments.grids import sgd_optim
from src.experiments.joint_dd import PureProxyGate
from src.experiments.logit_runs import with_optim
from src.experiments.runner import Context
from src.reset import gate_variants as GV
from src.reset import live_runs as LR
from src.reset import member_rules as MR
from src.reset.controller import RuleSpec
from src.reset.layout import TTA, Paths
from src.utils import provenance
from src.utils.config import GlobalConfig, load_global

SIDES = ("large", "small")


def _context(g: GlobalConfig, duo: str, device: str | None, joint: bool = False) -> Context:
    return Context.create(duo, g, device, tta=TTA, bn=True, regime="joint" if joint else "indep")


def _temps(ctx: Context, duo: str, *, compute: bool = False):
    return AM.resolve_temperatures(ctx, AM.temperature_spec(duo, TTA, None, None), None, report_only=not compute)


def rate_str(lr: float) -> str:
    mant, exp = f"{lr:.2e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"


# ------------------------------------------------------------------------------------------------------------------ temperatures

def temperatures(g: GlobalConfig, duo: str, a: dict, device: str | None) -> int:
    ctx = _context(g, duo, device)
    t = _temps(ctx, duo, compute=True)
    print(f"{duo}: T_L = {t.T_l}, T_S = {t.T_s} ({t.folder})")
    return 0


# ------------------------------------------------------------------------------------------------------------------ continual

def continual(g: GlobalConfig, duo: str, a: dict, device: str | None) -> int:
    joint = a.get("signal") == "joint"
    P = Paths(g, duo, "joint" if joint else "indep")
    lrs = None if a["lrs"] is None else (float(a["lrs"][0]), float(a["lrs"][1]))
    path = P.continual_run(a["seed"], lrs)
    if path.is_file():
        print(f"done already ({path.name})")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx = _context(g, duo, device, joint)
    temps = _temps(ctx, duo)
    rctx = ctx if lrs is None else with_optim(ctx, large=sgd_optim(lrs[0], a["momentum"]), small=sgd_optim(lrs[1], a["momentum"]))
    gate, inside = None, False
    if joint and lrs is not None:                                       # the joint signal needs the gate inside the loss (the frozen reference has none)
        gate = PureProxyGate("nnn", beta=a["beta"], b_t=256, T_l=temps.T_l, T_s=temps.T_s, num_classes=1000)
    runner, inside = RC.make_runner(rctx, RC.RUNS["frozen" if lrs is None else ("joint_dd" if joint else "indep")], gate, TTA)
    parts = CT.tuning_parts(rctx, rounds=a["rounds"], seed=a["seed"])
    stop = None if lrs is None else CT.DeadStop(SIDES, "all", 0)
    print(f"== {duo} {'joint ' if joint else ''}seed {a['seed']}, " + ("frozen" if lrs is None else f"L{lrs[0]:g}_S{lrs[1]:g}" + (f", gate beta {a['beta']:g}" if joint else "")))
    t0 = time.time()
    res = CT.run_until(runner, parts, images=a["images"], batch_size=rctx.batch_size, workers=rctx.g.workers, T_l=temps.T_l, T_s=temps.T_s, stop=stop, gate=gate, inside=inside)
    meta = {"run": "frozen" if lrs is None else ("joint_dd" if joint else "indep"), "signal": "joint" if joint else "indep", "beta": a.get("beta"), "duo": duo, "tta": TTA, "bn": True, "seed": a["seed"], "rounds": a["rounds"], "images": a["images"],
            "stream": "heldout, round r rotated by r", "lrs": lrs, "momentum": a["momentum"], "optim": {s.lower(): rctx.duo[s]["OPTIM"] for s in ("LARGE", "SMALL")},
            "T_l": temps.T_l, "T_s": temps.T_s, "batch_size": rctx.batch_size, "stopped": res["stopped"], "dead_at": res["dead_at"], "seen": res["seen"],
            "seconds": time.time() - t0, "models": {s.lower(): rctx.duo[s]["NAME"] for s in ("LARGE", "SMALL")}}
    print(f"wrote {RC.save_run(path, res, parts[:len(res['part_n'])], meta)} ({res['seen']} images, {meta['seconds'] / 60:.0f} min)")
    return 0


# ------------------------------------------------------------------------------------------------------------------ calibration

def calibration(g: GlobalConfig, duo: str, a: dict, device: str | None) -> int:
    joint = a.get("signal") == "joint"
    P = Paths(g, duo, "joint" if joint else "indep")
    lrs = (float(a["lrs"][0]), float(a["lrs"][1]))
    key = dict(lrs=lrs, seed=a["seed"], rotation=a["rotation"], beta=a["beta"], b_t=a["b_t"], proxy=a["proxy"])
    path, frozen_path = P.calibration_run(a["name"], **key), P.calibration_run(a["name"], **key, frozen=True)
    if path.is_file() and frozen_path.is_file():
        print(f"done already ({path.name})")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx = _context(g, duo, device, joint)
    temps = _temps(ctx, duo)
    names = {s: provenance.model_name(ctx.duo[s.upper()]["NAME"]).split(" (")[0] for s in SIDES}
    title = f"({a['name']}{', joint' if joint else ''}) " + ", ".join(f"{names[s]} at {GS.rate_str(lr)}" for s, lr in zip(SIDES, lrs))
    print(f"== {title}, seed {a['seed']}, rotation {a['rotation']}; gate {a['proxy']}, b_t {a['b_t']}, beta {a['beta']:g}; stops {a['after']} images after "
          f"{'every' if a['stop_mode'] == 'all' else 'the first'} member has collapsed completely")
    t0 = time.time()
    res, frozen, parts, meta = GS.run_experiment(ctx, lrs=lrs, watch=SIDES, after=a["after"], seed=a["seed"], order=None, images=a["images"], proxy=a["proxy"],
                                                 beta=a["beta"], b_t=a["b_t"], T_l=temps.T_l, T_s=temps.T_s, momentum=a["momentum"], tta=TTA, stream="heldout",
                                                 rounds=a["rounds"], rotation=a["rotation"], stop_mode=a["stop_mode"], joint=joint)
    meta.update(name=a["name"], title=title, tuning=None, seconds=time.time() - t0)
    RC.save_run(frozen_path, frozen, parts, {**meta, "run": "frozen"})
    print(f"wrote {RC.save_run(path, res, parts, {**meta, 'run': 'joint_dd' if joint else 'indep'})} ({res['seen']} images, {meta['seconds'] / 60:.0f} min)")
    return 0


# ------------------------------------------------------------------------------------------------------------------ live

def live(g: GlobalConfig, duo: str, a: dict, device: str | None) -> int:
    cond, stream, rot, seed, order = a["cond"], a["stream"], a["rotation"], a["seed"], a["order"]
    joint = a.get("signal") == "joint"
    lrs = (float(a["lrs"][0]), float(a["lrs"][1]))
    extra_gates = list(dict.fromkeys(n for x in a["extra_gates"] for n in (GV.DEFAULT_SET if x == "default" else (x,))))
    for n in extra_gates:
        GV.parse(n)                                                             # a bad gate name is an error now, not after the run
    if extra_gates and cond.startswith("random~"):
        raise ValueError("extra gates cannot go with a random~ condition (its reference run is in the other folder)")
    if extra_gates and joint:
        raise ValueError("the joint signal has its gate inside the loss: extra gates are not scored")
    if cond.startswith("random~") and not re.fullmatch(r"random~[^~]+~d\d+", cond):
        raise ValueError(f"{cond!r}: write random~<reference condition>~d<draw>")
    P = Paths(g, duo, "joint" if joint else "indep")
    ctx = _context(g, duo, device, joint)
    proxy = a["proxy"] or ctx.g.gate["proxy"]
    proxy = {"nuclear_norm": "nnn", "ac_mc": "msp"}.get(proxy, proxy)
    b_t, beta = a["b_t"], a["beta"]
    path_of = lambda c: P.live_run(c, stream=stream, rot=rot, seed=seed, lrs=lrs, proxy=proxy, b_t=b_t, beta=beta, order=order, max_images=a["max_images"],
                                   gates=bool(extra_gates))
    path = path_of(cond)
    if path.is_file():
        print(f"== {cond}/seed{seed}/rot{rot}/L{rate_str(lrs[0])}_S{rate_str(lrs[1])}: done already ({path.name})")
        return 0
    temps = _temps(ctx, duo)

    # the rule: the screened one (the shared screen's budget-1 choice), whose threshold and per-member settings the condition replaces
    rules: dict[str, RuleSpec] = {}
    # a `random~<reference>~d<k>` control has no rule of its own (its reference run's reset counts are read below); `split_condition` would choke on its name
    family = None if cond.startswith("random~") else (MR.FAMILY if MR.parse_condition(cond) else LR.split_condition(cond)[0])
    if cond not in ("none", "boundary", "frozen") and not re.fullmatch(r"fixed\d+", cond) and not cond.startswith("random~") and family == MR.FAMILY:
        screen_path = Path(a["screen_json"]) if a.get("screen_json") else P.shared_screen
        if not screen_path.is_file():
            raise SystemExit(f"{screen_path} is missing: the shared detector screen has not been run for {duo} (python scripts/tune_reset.py --duo {duo})")
        chosen = json.loads(screen_path.read_text())["selected"].get(str(float(a["budget"])))
        if chosen is None or chosen.get(family) is None:
            raise SystemExit(f"no rule {family!r} within budget {a['budget']} in {screen_path}")
        rules[family] = RuleSpec.from_json(chosen[family]["spec"])
    if not cond.startswith("random~") and cond != "frozen":
        LR.make_trigger(cond, rules)                                            # fail on a bad name now, not after hours in the queue

    rctx = with_optim(ctx, large=sgd_optim(lrs[0], a["momentum"]), small=sgd_optim(lrs[1], a["momentum"]))
    parts = CT.tuning_parts(rctx, rounds=a["rounds"], seed=seed, rotation=rot) if stream == "heldout" else RC.stream_parts(rctx, rounds=1, seed=seed, order=order, stream="test")
    names = {s: provenance.model_name(ctx.duo[s.upper()]["NAME"]).split(" (")[0] for s in SIDES}
    print(provenance.footer(ctx.duo, TTA, True))
    print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}; gate {proxy}, b_t {b_t}, beta {beta:g}{' (inside the loss: the joint signal)' if joint else ''}; members at "
          f"{names['large']} {rate_str(lrs[0])}, {names['small']} {rate_str(lrs[1])}; stream {stream}; condition {cond}")
    print(f"== {cond}, seed {seed}" + (f", rotation {rot}" if stream == "heldout" else "") + f" -> {path.name}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    common = dict(images=a["images"], batch_size=rctx.batch_size, workers=rctx.g.workers, T_l=temps.T_l, T_s=temps.T_s)
    meta = {"duo": ctx.duo["NAME"], "signal": "joint" if joint else "indep", "tta": TTA, "bn": True, "seed": seed, "stream": stream, "rounds": a["rounds"] if stream == "heldout" else 1,
            "rotation": rot if stream == "heldout" else None, "order": order, "images": a["images"], "max_images": a["max_images"], "condition": cond, "lrs": list(lrs),
            "momentum": a["momentum"], "proxy": proxy, "beta": beta, "b_t": b_t, "T_l": temps.T_l, "T_s": temps.T_s, "batch_size": rctx.batch_size, "budget": a["budget"],
            "extra_gates": extra_gates, "screen_json": str(P.shared_screen) if rules else None,
            "optim": {s.lower(): rctx.duo[s]["OPTIM"] for s in ("LARGE", "SMALL")}, "models": {s.lower(): rctx.duo[s]["NAME"] for s in ("LARGE", "SMALL")}}
    if cond == "frozen":
        runner, _ = RC.make_runner(ctx, RC.RUNS["frozen"], None, TTA)
        res = CT.run_until(runner, parts, max_images=a["max_images"], **common)
        res["resets"] = []
    else:
        gate = PureProxyGate(proxy, beta=beta, b_t=b_t, T_l=temps.T_l, T_s=temps.T_s, num_classes=1000)
        extra = {n: GV.make_gate(n, gate, T_l=temps.T_l, T_s=temps.T_s, b_t=b_t) for n in extra_gates}
        runner, inside = RC.make_runner(rctx, RC.RUNS["joint_dd" if joint else "indep"], gate, TTA)
        if cond.startswith("random~"):
            _, ref, draw = cond.split("~")
            ref_path = path_of(ref)
            if not ref_path.is_file():
                raise SystemExit(f"{cond}: the reference run {ref_path.name} does not exist yet")
            ref_meta = RC.load_run(ref_path)["meta"]
            trigger = LR.RandomTrigger(ref_meta["n_resets"], ref_meta["seen"] // rctx.batch_size, draw=int(draw[1:]), name=cond)
        else:
            trigger = LR.make_trigger(cond, rules)
        res = LR.run_live(runner, parts, trigger=trigger, gate=gate, max_images=a["max_images"], extra_gates=extra, inside=inside, **common)
        meta["trigger"] = res["trigger"]
    meta.update(seconds=time.time() - t0, seen=res["seen"], stopped=res["stopped"], resets=res["resets"], n_resets={s: sum(r["member"] == s for r in res["resets"]) for s in SIDES})
    visited = parts[:len(res["part_n"])]
    RC.save_run(path, res, visited, {**meta, "run": "frozen" if cond == "frozen" else ("joint_dd" if joint else "indep")})
    acc = {s: float(v.mean() * 100) for s, v in res["correct"].items()}
    print(f"wrote {path.name}: {res['seen']} images, {meta['seconds'] / 60:.0f} min; accuracy " + ", ".join(f"{s} {x:.1f}" for s, x in acc.items()) + f"; resets {meta['n_resets']}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------ dispatch

WORKERS = {"temperatures": temperatures, "continual": continual, "calibration": calibration, "live": live}


def run_job(spec_path: str | Path, device: str | None = None) -> int:
    """Execute the job a spec file describes (see src/reset/jobs.py)."""
    spec = json.loads(Path(spec_path).read_text())
    if spec["kind"] not in WORKERS:
        raise ValueError(f"unknown job kind {spec['kind']!r} in {spec_path}")
    pin_threads()
    return WORKERS[spec["kind"]](load_global(), spec["duo"], spec["args"], device)


if __name__ == "__main__":
    from src.reset import jobs as J

    assert set(WORKERS) == set(J.KINDS), "every job kind has a worker"
    assert rate_str(1.4e-4) == "1.4e-4" and rate_str(6.25e-5) == "6.25e-5" and rate_str(5e-4) == "5e-4"
    print("workers self-test passed (the run bodies are exercised by a smoke job: scripts/tune_reset.py --job)")
