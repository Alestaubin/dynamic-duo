"""
pipeline.py
===========
The tuning of ANY duo with resetting (`own_fast_vs_slow`), as ONE resumable procedure: `drive(duo)` looks at the files of the duo, finds the first stage that is not
done, and either prints the work (jobs to submit, or a CPU step to run) or does it (`submit=True`). Call it again after a wave finishes. The protocol and its
rules: docs/reset_tuning.md and src/reset/tuning.py.

    stage          what it needs                                        what it gives
    prerequisites  the duo's yaml and its episodic rates                the frozen-model temperatures T_L, T_S (a job, once per duo)
    continual      the temperatures                                     the continual-rate ladder (no reset; the former 04b): each member's CONSERVATIVE and
                                                                        STRESS rate -> tune_continual_lr/<duo>__tent_bn/results.json
    calibration    those rates                                          the calibration streams (the former 04c): stress S (both members collapse) and healthy H
    screens        S and H                                              the offline replay of the detector (src/reset/screens.py): the SHARED setting and each
                                                                        member's two CANDIDATE settings (CPU, inline)
    tune           the screens                                          the search (src/reset/tuning.py): waves of live runs on the tuning rotation (rotation 0) until
                                                                        the choice is made -> member__rates.json (independent) / joint__choice.json (joint)
    confirm        the choice                                           the chosen configuration run once on the tuning rotation

The SIGNAL is the loss the members adapt on (`signal`, `--signal`). INDEPENDENT (the default): each member on its own entropy, the gate outside the loop; the members
decouple, so each is searched alone: its detector setting and its learning rate (`plan_member`). JOINT: both on the gated duo's entropy, the gate inside the loss; the
members are coupled, so the search is over cells scored on the duo's accuracy (`plan_joint`): the pair of rates, the gate's beta, and each member's own detector setting,
in blocks, repeated. The joint signal has its own rates (cfgs/duo_lrs/<duo>__joint.yaml), folders and files; every stage runs per signal.

Everything duo-specific is derived: the rate grid is anchored at the duo's episodic rates, the stress and conservative rates come from the continual stage, the
shared setting from the screen, the candidates' start rungs from the shared ladder. `config.json` in the duo's tune folder overrides any of them (written by the
flags `--stress`, `--stress_up`, `--healthy`, `--mults`, `--betas`, `--rotations`, or by hand: also `ladders`, explicit continual rates per member, `confirm_rotations`,
`max_passes`); the main duo's file records the choices its earlier runs were made with.
`baseline=<condition>` runs the same machinery for a comparison method with only a rate to tune: `none`, `fixed500`, `fixed1000`, `own_fast_vs_slow@<delta>` (independent only).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments import continual as RC
from src.experiments import continual_tuning as CT
from src.experiments import joint_lr_tuning as JL
from src.experiments.runner import Context
from src.reset import jobs as J
from src.reset import member_rules as MR
from src.reset import screens as SCR
from src.reset import tuning as TU
from src.reset.layout import LEGACY_DUO, SIDES, TTA, Paths
from src.utils import duo_lrs, provenance
from src.utils.config import GlobalConfig, load_duo, load_global

BETA = 0.5                          # the gate of the tuning runs (an evaluation choice: members and proxy values do not depend on it)
B_T, PROXY = 256, "nnn"
GATE = f"{PROXY}_bt{B_T}_beta{BETA:g}"
SEED = 0
CALIBRATION = {"S": dict(rotations=(0, 1, 2, 3), seeds=(0, 1), stop_mode="all", after=20000),      # stress: both members collapse (the events)
               "H": dict(rotations=(0, 2), seeds=(0,), stop_mode="any", after=40000)}              # healthy: full length (the false-alarm budget)


def rate_g(x: float) -> str:
    return f"{x:g}"


# ------------------------------------------------------------------------------------------------------------------ one duo

@dataclass
class Setup:
    """What every stage needs of the duo, derived once."""
    g: GlobalConfig
    duo: str
    P: Paths
    cfg: dict
    names: dict[str, str]
    episodic: dict[str, float]
    grid: TU.Grid
    duo_cfg: dict = field(repr=False, default_factory=dict)
    signal: str = "indep"

    @classmethod
    def load(cls, duo: str, g: GlobalConfig | None = None, overrides: dict | None = None, log=print, signal: str = "indep") -> "Setup":
        g = g or load_global()
        P = Paths(g, duo, signal)
        try:
            duo_cfg = load_duo(duo, g, tta=TTA, regime=signal)
        except FileNotFoundError:
            raise SystemExit(f"cfgs/duos/{duo}.yaml does not exist: the duo must be defined there (its two models and their norm layers)")
        except duo_lrs.NotTuned as e:
            raise SystemExit(f"{duo}: no tuned {signal} learning rates: {e} (scripts/{'02b_tune_joint_lr' if signal == 'joint' else '02c_tune_indep_pair'}.py --record)")
        tuned = duo_lrs.load(duo, signal)["methods"].get(TTA, {})
        if any(side not in tuned for side in SIDES):                    # the rate grid is anchored at the episodic rates: a placeholder default is no anchor
            raise SystemExit(f"{duo}: no tuned episodic learning rate for the {[x for x in SIDES if x not in tuned]} member in cfgs/duo_lrs/{duo}__{signal}.yaml "
                             f"(a placeholder default is not an anchor): run scripts/{'02b_tune_joint_lr' if signal == 'joint' else '02c_tune_indep_pair'}.py --record first")
        cfg = json.loads(P.config_path.read_text()) if P.config_path.is_file() else {}
        changed = {k: v for k, v in (overrides or {}).items() if v is not None and cfg.get(k) != v}
        if changed:
            cfg.update(changed)
            P.tune_dir.mkdir(parents=True, exist_ok=True)
            P.config_path.write_text(json.dumps(cfg, indent=1))
            log(f"config {P.config_path}: set " + ", ".join(f"{k} = {v}" for k, v in changed.items()))
        episodic = {s: float(duo_cfg[s.upper()]["OPTIM"]["LR"]) for s in SIDES}
        names = {s: provenance.model_name(duo_cfg[s.upper()]["NAME"]).split(" (")[0] for s in SIDES}
        return cls(g, duo, P, cfg, names, episodic, TU.Grid(episodic), duo_cfg, signal)

    @property
    def joint(self) -> bool:
        return self.signal == "joint"

    @property
    def store(self) -> TU.MemberStore:
        """The independent signal's runs (a member's cell is any run with that member at that setting and rate)."""
        return TU.MemberStore(self.P.live_dir(), GATE, self.shared, f"{MR.FAMILY}@{self.shared.delta:g}", SEED)

    @property
    def jstore(self) -> TU.JointStore:
        """The joint signal's runs (a cell is a run at both rates, both settings and the gate's beta)."""
        return TU.JointStore(self.P.live_dir(), SEED, PROXY, B_T)

    @property
    def rots(self) -> tuple[int, ...]:
        """The held-out rotation(s) the search runs on: `config.json` `tuning_rotations`, else one (rotation 0)."""
        return tuple(int(r) for r in (self.cfg.get("tuning_rotations") or TU.TUNING_ROTS))

    @property
    def confirm_rots(self) -> tuple[int, ...]:
        return tuple(int(r) for r in (self.cfg.get("confirm_rotations") or self.rots[:1]))

    @property
    def betas(self) -> tuple[float, ...]:
        return tuple(float(b) for b in (self.cfg.get("betas") or TU.BETAS))

    def live_job(self, cond: str, rotation: int, lrs, *, beta: float = BETA) -> J.Job:
        """A live run of this duo and signal on the held-out stream."""
        return J.live_job(self.duo, cond, stream="heldout", lrs=tuple(lrs), rotation=rotation, seed=SEED, beta=beta, b_t=B_T, proxy=PROXY, signal=self.signal)

    @property
    def shared(self) -> MR.Setting:
        """The shared candidate: `config.json`, else the shared screen's budget-1 choice."""
        if self.cfg.get("shared"):
            af, as_, w0, delta = self.cfg["shared"]
            return MR.Setting(float(af), float(as_), int(w0), float(delta))
        return SCR.shared_setting(json.loads(self.P.shared_screen.read_text()))

    def mults(self) -> list[float]:
        return list(self.cfg.get("mults") or CT.DEFAULT_MULTS)

    def ladders(self) -> dict[str, list[float]]:
        """Each member's continual ladder: explicit rates in `config.json` (`ladders`), else the multiples `mults` of its episodic rate."""
        explicit = self.cfg.get("ladders") or {}
        axes = {s: JL.lr_axis(self.episodic[s], self.mults(), explicit.get(s)) for s in SIDES}
        if len(axes["large"]) != len(axes["small"]):
            raise ValueError("the two continual ladders must have the same length (a run pairs the i-th rungs)")
        return axes

    def beta_cal(self) -> float:
        return float(self.cfg.get("calibration_beta", BETA))


@dataclass
class Step:
    """The state of one stage: done / jobs (GPU runs to submit) / inline (a CPU step) / blocked (needs a decision)."""
    stage: str
    state: str
    notes: list[str] = field(default_factory=list)
    jobs: list[J.Job] = field(default_factory=list)
    run: Callable[[], None] | None = None


# ------------------------------------------------------------------------------------------------------------------ prerequisites, continual

def stage_prerequisites(S: Setup) -> Step:
    ctx = Context(S.g, S.duo_cfg, "cpu", bn=True)
    try:
        t = AM.resolve_temperatures(ctx, AM.temperature_spec(S.duo, TTA, None, None), None, report_only=True)
    except FileNotFoundError:
        return Step("prerequisites", "jobs", ["the frozen-model temperatures T_L, T_S are not fitted yet (one run on the four held-out shifts)"], [J.temperatures_job(S.duo)])
    return Step("prerequisites", "done", [f"{S.names['large']} {S.episodic['large']:g} + {S.names['small']} {S.episodic['small']:g} (episodic {S.signal} rates); T_L {t.T_l}, T_S {t.T_s}"])


def continual_jobs(S: Setup) -> list[J.Job]:
    axes = S.ladders()
    rungs = list(zip(axes["large"], axes["small"]))
    return ([J.continual_job(S.duo, seed=SEED, lrs=None, signal=S.signal)] +
            [J.continual_job(S.duo, seed=SEED, lrs=r, signal=S.signal, beta=S.beta_cal()) for r in rungs])


def _continual_path(S: Setup, j: J.Job) -> Path:
    return S.P.continual_run(j.args["seed"], None if j.args["lrs"] is None else tuple(j.args["lrs"]))


def continual_report(S: Setup, log=print) -> dict:
    """The former 04b report: each member's ladder with its stability, the frontier / conservative / stress rates, the warnings; writes results.json and ranking.csv."""
    axes, delta, margin = S.ladders(), CT.DELTA, CT.MARGIN
    rungs = list(zip(axes["large"], axes["small"]))
    rows = []
    frozen = RC.load_run(S.P.continual_run(SEED, None))
    for r in rungs:
        run = RC.load_run(S.P.continual_run(SEED, r))
        n = len(run["part_n"])
        if list(run["part_label"]) != list(frozen["part_label"][:n]) or list(run["part_n"][:-1]) != list(frozen["part_n"][:n - 1]):
            raise ValueError(f"the run at {r} saw other visits than the frozen run: the runs must share one stream")
        for side, lr in zip(SIDES, r):
            rows.append({"side": side, "lr": lr, "seed": SEED, "lrs": list(r),
                         **CT.summarize(run[side], frozen[side][:len(run[side])], run["part_n"], list(run["part_label"]), delta=delta)})
    selection = {side: CT.select_side([r for r in rows if r["side"] == side], margin) for side in SIDES}
    warnings = []
    for side in SIDES:
        sel = selection[side]
        log(f"\n== {side}: {S.names[side]} (episodic rate {S.episodic[side]:g})")
        log(f"  {'rate':>9}  stable  worst late gain (visit)            dead at   mean acc   images")
        for r in sorted((r for r in rows if r["side"] == side), key=lambda r: (r["lr"], r["seed"])):
            dead = "-" if r["dead_at"] is None else f"{r['dead_at'] // 1000}k"
            log(f"  {r['lr']:9.3g}  {'yes' if r['stable'] else 'NO':>6}  {r['min_gain']:+6.1f}  {r['worst_visit']:<30} {dead:>8}  {r['acc']:8.2f}  {r['images']:7d}")
        for k in ("frontier", "conservative", "stress"):
            log(f"  {k:<13}: {sel[k]:g}" if sel[k] is not None else f"  {k:<13}: none")
        if sel["lagging"]:
            log("  lagging      : " + ", ".join(f"{x:g}" for x in sel["lagging"]) + "  (below the frontier: too slow to beat the frozen member, not a collapse)")
        warnings += [f"{side}: {w}" for w in sel["warnings"]]
    for w in warnings:
        log("WARNING " + w)
    ctx = Context(S.g, S.duo_cfg, "cpu", bn=True)
    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(S.duo, TTA, None, None), None, report_only=True)
    save = {"duo": S.duo, "signal": S.signal, "tta": TTA, "bn": True, "seeds": [SEED], "episodic": S.episodic, "ladders": axes,
            "stream": {"held_out": S.g.tuning_corruptions, "rounds": 4, "images": 50000, "severity": S.g.severity, "rotation": "round r starts at the r-th shift"},
            "criteria": {"delta": delta, "margin": margin, "late_frac": CT.LATE_FRAC, "dead": {"window": CT.DEAD_WINDOW, "acc": CT.DEAD_ACC, "windows": CT.DEAD_WINDOWS}},
            "temperatures": {"T_l": temps.T_l, "T_s": temps.T_s, "checkpoint": str(temps.folder)}, "momentum": 0.9,
            "sides": {s: dict(selection[s]) for s in SIDES}, "rows": rows, "warnings": warnings}
    S.P.continual_dir.mkdir(parents=True, exist_ok=True)
    (S.P.continual_dir / "results.json").write_text(json.dumps(save, indent=1))
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("gain", "lrs")} for r in rows]).to_csv(S.P.continual_dir / "ranking.csv", index=False)
    log(f"\nwrote {S.P.continual_dir / 'results.json'}" + ("" if not warnings else f" (with {len(warnings)} warning(s): extend the ladder before using it)"))
    return save


def continual_results(S: Setup) -> dict | None:
    """results.json if it was made from the current ladders."""
    p = S.P.continual_dir / "results.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    axes = S.ladders()
    same = all(len(d["ladders"][s]) == len(axes[s]) and np.allclose(d["ladders"][s], axes[s], rtol=1e-9) for s in SIDES)   # an extended ladder is a different shape
    return d if same else None


def stage_continual(S: Setup) -> Step:
    missing = [j for j in continual_jobs(S) if not _continual_path(S, j).is_file()]
    if missing:
        how = "both members adapting jointly (the gate inside the loss)" if S.joint else "both members adapting on their own entropy"
        return Step("continual", "jobs", [f"the continual-rate ladder: {len(missing)} run(s) missing of {len(continual_jobs(S))} (the frozen reference and "
                                          f"{len(continual_jobs(S)) - 1} rungs, {how}, without a reset, on the held-out stream x 4 rounds)"], missing)
    res = continual_results(S)
    if res is None:
        return Step("continual", "inline", ["all runs are there: select the conservative and stress rates (CPU, seconds)"], run=lambda: continual_report(S))
    if res["warnings"]:
        return Step("continual", "blocked", ["the ladder does not bracket the frontier:"] + [f"  {w}" for w in res["warnings"]] +
                    ["extend it: `--mults X X X ...` (multiples of the episodic rates; the new rungs are requested), then call again"])
    sel = res["sides"]
    return Step("continual", "done", [f"{s}: conservative {sel[s]['conservative']:g}, stress {sel[s]['stress']:g}" for s in SIDES])


# ------------------------------------------------------------------------------------------------------------------ calibration, screens

def calibration_rates(S: Setup) -> tuple[tuple[float, float], tuple[float, float]]:
    """(stress pair, healthy pair): config.json, else the continual stage's rates (the stress rate may be moved `stress_up` rungs up its ladder)."""
    sel = continual_results(S)["sides"]
    ladders = S.ladders()
    up = int(S.cfg.get("stress_up", 0))

    def stress(side: str) -> float:
        lr = sel[side]["stress"]
        k = ladders[side].index(lr) + up
        if k >= len(ladders[side]):
            raise ValueError(f"stress_up {up}: the {side} ladder ends at {ladders[side][-1]:g}")
        return float(ladders[side][k])

    s = tuple(S.cfg["stress"]) if S.cfg.get("stress") else tuple(stress(side) for side in SIDES)
    h = tuple(S.cfg["healthy"]) if S.cfg.get("healthy") else tuple(float(sel[side]["conservative"]) for side in SIDES)
    return s, h


def calibration_jobs(S: Setup) -> list[J.Job]:
    stress, healthy = calibration_rates(S)
    out = []
    for name, lrs in (("S", stress), ("H", healthy)):
        c = CALIBRATION[name]
        out += [J.calibration_job(S.duo, name=name, lrs=lrs, seed=seed, rotation=rot, stop_mode=c["stop_mode"], after=c["after"], beta=S.beta_cal(), b_t=B_T, proxy=PROXY,
                                  signal=S.signal) for rot in c["rotations"] for seed in c["seeds"]]
    return out


def _calibration_done(S: Setup, j: J.Job) -> bool:
    a = j.args
    key = dict(lrs=tuple(a["lrs"]), seed=a["seed"], rotation=a["rotation"], beta=a["beta"], b_t=a["b_t"], proxy=a["proxy"])
    return S.P.calibration_run(a["name"], **key).is_file() and S.P.calibration_run(a["name"], **key, frozen=True).is_file()


def stage_calibration(S: Setup) -> Step:
    res = continual_results(S)
    if any(res["sides"][s][k] is None for s in SIDES for k in ("stress", "conservative")):
        return Step("calibration", "blocked", ["the continual stage names no stress or conservative rate: extend the ladder (`--mults`)"])
    jobs = calibration_jobs(S)
    missing = [j for j in jobs if not _calibration_done(S, j)]
    stress, healthy = calibration_rates(S)
    notes = [f"stress S at {S.names['large']} {stress[0]:g} / {S.names['small']} {stress[1]:g} (both members collapse), healthy H at {healthy[0]:g} / {healthy[1]:g}; {len(jobs)} streams"]
    if missing:
        return Step("calibration", "jobs", notes + [f"{len(missing)} run(s) missing"], missing)
    return Step("calibration", "done", notes)


def stage_screens(S: Setup) -> Step:
    todo = [p for p in [S.P.shared_screen, *[S.P.member_screen(s) for s in SIDES]] if not p.is_file()]
    if not todo:
        return Step("screens", "done", [f"shared setting {tuple(S.shared)}; candidates: " +
                                        "; ".join(f"{s}: " + ", ".join(str(tuple(c)) for c in SCR.candidates_of(S.P, s)) for s in SIDES)])

    def run() -> None:
        beta = S.beta_cal()
        if not S.P.shared_screen.is_file():
            SCR.shared_screen(S.P, beta=beta, b_t=B_T)
        for s in SIDES:
            if not S.P.member_screen(s).is_file():
                SCR.member_screen(S.P, s, S.shared, beta=beta, b_t=B_T)

    return Step("screens", "inline", [f"replay the detector on the calibration streams: {len(todo)} screen(s) to make (CPU, about a minute each)"], run=run)


# ------------------------------------------------------------------------------------------------------------------ the per-member search

def _pair_jobs(S: Setup, reqs: dict[str, list[tuple]], filler: dict[str, tuple]) -> list[J.Job]:
    """One live run per (rotation, i): the i-th request of each member; a member with fewer requests gets its filler cell (a run measures both members)."""
    runs: dict[int, list[tuple]] = {}
    for rot in sorted({r for s in SIDES for _, _, r in reqs[s]}):
        lst = {s: [(st, k) for st, k, r in reqs[s] if r == rot] for s in SIDES}
        for i in range(max(len(lst["large"]), len(lst["small"]))):
            a = lst["large"][i] if i < len(lst["large"]) else filler["large"]
            b = lst["small"][i] if i < len(lst["small"]) else filler["small"]
            runs.setdefault(rot, []).append((MR.condition(a[0], b[0]), S.grid.rate("large", a[1]), S.grid.rate("small", b[1])))
    return [J.live_job(S.duo, cond, stream="heldout", lrs=(rl, rs), rotation=rot, seed=SEED, beta=BETA, b_t=B_T, proxy=PROXY)
            for rot, lst in sorted(runs.items()) for cond, rl, rs in dict.fromkeys(lst)]


def member_plans(S: Setup) -> dict[str, TU.MemberPlan]:
    shared, store = S.shared, S.store
    start = S.cfg.get("candidate_start") or {}
    return {s: TU.plan_member(s, lambda st, k, rot, s=s: store.member_cell(s, st, rot, S.grid.rate(s, k)), shared, SCR.candidates_of(S.P, s),
                              candidate_start=start.get(s), rots=S.rots) for s in SIDES}


def _rots_label(S: Setup) -> str:
    return "+".join(str(r) for r in S.rots)


def stage_tune(S: Setup, log=print) -> Step:
    if S.joint:
        return stage_tune_joint(S)
    shared, plans = S.shared, member_plans(S)
    cands = {s: SCR.candidates_of(S.P, s) for s in SIDES}
    notes = []
    for side, pl in plans.items():
        notes.append(f"{side} ({S.names[side]}): candidates " + ", ".join("shared" if c == shared else str(tuple(c)) for c in [shared, *cands[side]]))
        notes.append("   " + "; ".join(pl.notes))
        for r in pl.table:
            notes.append(f"     {str(r['setting']):36s} rate rung {r['rate_rung']:3d} ({S.grid.rate(side, r['rate_rung']):.3g})  acc({_rots_label(S)}) {100 * r['acc']:6.2f}  "
                         f"resets/1000 {r['resets']:5.2f}" + ("  tied" if r["tied"] else "") + ("  <- best" if r["chosen"] else ""))
    reqs = {s: list(dict.fromkeys(pl.requests)) for s, pl in plans.items()}
    if any(reqs.values()):
        filler = {s: (shared, plans[s].shared_rung if plans[s].shared_rung is not None else TU.X2[s][1]) for s in SIDES}
        jobs = _pair_jobs(S, reqs, filler)
        return Step("tune", "jobs", notes + [f"{len(jobs)} run(s) requested"], jobs)
    final = {s: plans[s].final for s in SIDES}
    if any(f is None for f in final.values()):
        return Step("tune", "blocked", notes + ["no final choice and no requests: see the notes above"])
    cond = MR.condition(final["large"][0], final["small"][0])
    rates = [S.grid.rate(s, final[s][1]) for s in SIDES]
    state = {"cond": cond, "chosen": rates, "settings": {s: list(final[s][0]) for s in SIDES}, "tables": {s: plans[s].table for s in SIDES}}
    S.P.tune_dir.mkdir(parents=True, exist_ok=True)
    (S.P.tune_dir / "member__rates.json").write_text(json.dumps(state, default=float, indent=1))
    return Step("tune", "done", notes + [f"final: {cond} at {S.names['large']} {rates[0]:g}, {S.names['small']} {rates[1]:g}"])


def _joint_lines(S: Setup, jp: TU.JointPlan) -> list[str]:
    out = []
    for r in jp.table:
        flag = ("  tied" if r["tied"] else "") + ("  <- chosen" if r["chosen"] else "")
        if r["block"] == "rates":
            (kv, ks) = r["cell"]
            what = f"rates  {S.names['large']} {S.grid.rate('large', kv):.3g} (rung {kv}), {S.names['small']} {S.grid.rate('small', ks):.3g} (rung {ks})"
        elif r["block"] == "beta":
            what = f"gate beta {r['beta']:g}"
        else:
            what = f"{r['block']}  {tuple(r['setting'])}"
        out.append(f"     {what:78s} duo acc {100 * r['acc']:6.2f}  resets/1000 {r['resets']:6.2f}{flag}")
    return out


def stage_tune_joint(S: Setup) -> Step:
    """The search of the joint signal (`TU.plan_joint`): the pair of rates, the gate's beta, each member's detector, repeated; every cell scored on the duo's accuracy."""
    shared = S.shared
    cands = {s: SCR.candidates_of(S.P, s) for s in SIDES}
    store = S.jstore

    def run_args(k):
        (kv, ks), beta, sl, ss = k
        return MR.condition(sl, ss), (S.grid.rate("large", kv), S.grid.rate("small", ks)), beta

    def cell_of(k, rot):
        cond, (rl, rs), beta = run_args(k)
        return store.cell(cond, rot, large=rl, small=rs, beta=beta)

    jp = TU.plan_joint(cell_of, shared, cands, betas=S.betas, rots=S.rots, max_passes=int(S.cfg.get("max_passes", 2)))
    notes = [f"shared detector setting {tuple(shared)}; candidates: " + "; ".join(f"{s}: " + ", ".join(str(tuple(c)) for c in cands[s]) for s in SIDES)]
    notes += [f"   {n}" for n in jp.notes] + _joint_lines(S, jp)
    if jp.requests:
        jobs = []
        for k, rot in dict.fromkeys(jp.requests):
            cond, lrs, beta = run_args(k)
            jobs.append(S.live_job(cond, rot, lrs, beta=beta))
        return Step("tune", "jobs", notes + [f"{len(jobs)} run(s) requested"], jobs)
    if jp.final is None:
        return Step("tune", "blocked", notes + ["no final choice and no requests: see the notes above"])
    cond, rates, beta = run_args(jp.final)
    (kv, ks), _, sl, ss = jp.final
    state = {"cond": cond, "chosen": list(rates), "rungs": [kv, ks], "beta": beta, "settings": {"large": list(sl), "small": list(ss)}, "converged": jp.converged,
             "notes": jp.notes, "table": jp.table}
    S.P.tune_dir.mkdir(parents=True, exist_ok=True)
    (S.P.tune_dir / "joint__choice.json").write_text(json.dumps(state, default=float, indent=1))
    return Step("tune", "done", notes + [f"final: {cond} at {S.names['large']} {rates[0]:g}, {S.names['small']} {rates[1]:g}, gate beta {beta:g}" +
                                         ("" if jp.converged else " (not converged: the last pass still moved)")])


# ------------------------------------------------------------------------------------------------------------------ confirm

def _cell_fn(S: Setup, pair, beta: float):
    """cell(cond, rotation): the stored run of a condition at the chosen rates (and, for the joint signal, the chosen gate)."""
    if S.joint:
        return lambda cond, rot: S.jstore.cell(cond, rot, large=pair[0], small=pair[1], beta=beta)
    store = S.store
    return lambda cond, rot: store.cell(cond, rot, large=pair[0], small=pair[1])


def _confirm_table(S: Setup, cond: str, pair, cell, log=print) -> dict:
    rows = {}
    log(f"{cond} at {S.names['large']} {pair[0]:g}, {S.names['small']} {pair[1]:g}: whole-stream accuracy (%) per rotation")
    for r in S.confirm_rots:
        c = cell(cond, r)
        rows[r] = {s: c.mean(s) * 100 for s in ("gate", "ts", "large", "small")} | {"resets_per_1000": c.resets, "dead": c.dead}
        log(f"  rot {r}: duo {rows[r]['gate']:.1f}  fixed-TS {rows[r]['ts']:.1f}  {S.names['large']} {rows[r]['large']:.1f}  {S.names['small']} {rows[r]['small']:.1f}  "
            f"resets/1000 {c.resets['large']:.2f} / {c.resets['small']:.2f}")
    return rows


def stage_confirm(S: Setup, log=print) -> Step:
    """The chosen configuration on the confirmation rotation(s) (default: the tuning rotation; the independent search never ran this exact pair, the joint search did)."""
    name = "joint__choice.json" if S.joint else "member__rates.json"
    state = json.loads((S.P.tune_dir / name).read_text())
    cond, pair, beta = state["cond"], state["chosen"], float(state.get("beta", BETA))
    cell = _cell_fn(S, pair, beta)
    missing = [r for r in S.confirm_rots if cell(cond, r) is None]
    if missing:
        return Step("confirm", "jobs", [f"the chosen configuration on rotation(s) {missing}"], [S.live_job(cond, r, pair, beta=beta) for r in missing])
    rows = _confirm_table(S, cond, pair, cell, log)
    out = "joint__confirm.json" if S.joint else "member__confirm.json"
    (S.P.tune_dir / out).write_text(json.dumps({"cond": cond, "pair": pair, "beta": beta if S.joint else None, "rotations": rows}, default=float, indent=1))
    return Step("confirm", "done", [f"{cond}: run on rotation(s) {list(S.confirm_rots)} ({out})"])


# ------------------------------------------------------------------------------------------------------------------ comparison baselines

def _runs_for(needs: dict[str, list[tuple[int, int]]], rate_of) -> dict[int, list[tuple[float, float]]]:
    """Pair the two members' requested rungs into runs, per rotation; a member without a partner request gets the diagonal partner (a decoupled method does not care)."""
    out: dict[int, list[tuple[float, float]]] = {}
    for rot in sorted({r for side in SIDES for _, r in needs[side]}):
        a = [k for k, r in needs["large"] if r == rot]
        b = [k for k, r in needs["small"] if r == rot]
        for i in range(max(len(a), len(b))):
            kv = a[i] if i < len(a) else b[i] + 2
            ks = b[i] if i < len(b) else kv - 2
            out.setdefault(rot, []).append((rate_of("large", kv), rate_of("small", ks)))
    return {r: list(dict.fromkeys(p)) for r, p in out.items()}


def _show_ladder(S: Setup, plan: TU.Plan, side: str, rate_of) -> list[str]:
    lines = [f"  {side}: " + "; ".join(plan.notes)]
    for r in plan.table:
        lines.append(f"    rung {r['rung']:3d}  {rate_of(side, r['rung']):.3g}  acc(rot0) {100 * r['acc_rot0']:6.2f}  acc({_rots_label(S)}) " + (f"{100 * r['acc']:6.2f}" if r["acc"] is not None else "     -")
                     + "  resets/1000 " + (f"{r['resets']:6.2f}" if r["resets"] is not None else "     -") + ("  tied" if r["tied"] else "") + ("  <- chosen" if r["chosen"] else ""))
    return lines


def stage_baseline_rates(S: Setup, cond: str) -> Step:
    """The rate of a comparison method (one ladder per member, scored on the member's own accuracy). `none` uses the continual ladder as rotation 0 (the former baseline2)."""
    store = TU.Store(S.P.live_dir(), GATE)
    state_path = S.P.tune_dir / ("none__baseline2.json" if cond == "none" else f"{cond}__rates.json")
    if cond == "none":
        lad = continual_results(S)["ladders"]
        n = len(lad["large"])
        cache: dict[int, TU.Cell] = {}

        def cell_of(side):
            def f(i: int, rot: int):
                if not 0 <= i < n:
                    return None
                if rot == 0:
                    if i not in cache:
                        cache[i] = TU.load_cell(S.P.continual_run(SEED, (lad["large"][i], lad["small"][i])))
                    return cache[i]
                return store.cell("none", rot, **{side: lad[side][i]})
            return f
        plans = {side: TU.plan_1d(side=side, start=tuple(range(n)), neighbours=lambda i: (i - 1, i + 1), cell_of=cell_of(side), rots=S.rots) for side in SIDES}
        rate_of = lambda side, i: lad[side][i]
    else:
        plans = {side: TU.plan_1d(side=side, start=TU.X2[side], neighbours=lambda k: (k - 1, k + 1), rots=S.rots,
                                  cell_of=lambda k, rot, side=side: store.cell(cond, rot, **{side: S.grid.rate(side, k)})) for side in SIDES}
        rate_of = S.grid.rate
    notes = [line for side in SIDES for line in _show_ladder(S, plans[side], side, rate_of)]
    needs = {s: plans[s].requests for s in SIDES}
    by_rot = _runs_for(needs, rate_of)
    if by_rot:
        jobs = [J.live_job(S.duo, cond, stream="heldout", lrs=lrs, rotation=rot, seed=SEED, beta=BETA, b_t=B_T, proxy=PROXY) for rot, pairs in by_rot.items() for lrs in pairs]
        return Step("rates", "jobs", notes, jobs)
    if any(plans[s].chosen is None for s in SIDES):
        return Step("rates", "blocked", notes + ["no rung is alive: see the notes above"])
    chosen = [rate_of(s, plans[s].chosen) for s in SIDES]
    tables = {s: plans[s].table for s in SIDES}
    # the formats the report scripts read: the no-reset baseline stores {side: rate}, the others a [large, small] list
    state = {"chosen": dict(zip(SIDES, chosen)), "tables": tables} if cond == "none" else {"cond": cond, "chosen": chosen, "tables": tables}
    notes.append(f"chosen: {S.names['large']} {chosen[0]:g}, {S.names['small']} {chosen[1]:g} (each member on its own accuracy)")
    if state_path.is_file():                                          # the frozen record of the comparison method: never overwritten
        old = json.loads(state_path.read_text())["chosen"]
        old = [old[s] for s in SIDES] if isinstance(old, dict) else old
        notes.append(f"{state_path.name} exists and is kept" + ("" if np.allclose(old, chosen, rtol=1e-6) else f" (it holds {old[0]:g} / {old[1]:g}: delete it to record the new choice)"))
    else:
        S.P.tune_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, default=float, indent=1))
    return Step("rates", "done", notes)


def stage_baseline_confirm(S: Setup, cond: str) -> Step:
    store = TU.Store(S.P.live_dir(), GATE)
    state = json.loads((S.P.tune_dir / ("none__baseline2.json" if cond == "none" else f"{cond}__rates.json")).read_text())
    pair = [state["chosen"][s] for s in SIDES] if isinstance(state["chosen"], dict) else state["chosen"]
    missing = [r for r in S.confirm_rots if store.cell(cond, r, large=pair[0], small=pair[1]) is None]
    if missing:
        return Step("confirm", "jobs", [f"the chosen pair on the rotations still missing: {missing}"],
                    [J.live_job(S.duo, cond, stream="heldout", lrs=tuple(pair), rotation=r, seed=SEED, beta=BETA, b_t=B_T, proxy=PROXY) for r in missing])
    rows = {}
    for r in S.confirm_rots:
        c = store.cell(cond, r, large=pair[0], small=pair[1])
        rows[r] = {s: c.mean(s) * 100 for s in ("gate", "ts", "large", "small")} | {"resets_per_1000": c.resets, "dead": c.dead}
    out = S.P.tune_dir / f"{cond}__confirm.json"
    if not out.is_file():                                             # written once, like the rates it confirms
        out.write_text(json.dumps({"cond": cond, "pair": pair, "rotations": rows}, default=float, indent=1))
    return Step("confirm", "done", [f"{cond} at {pair[0]:g} / {pair[1]:g}: " + "  ".join(f"rot {r}: duo {rows[r]['gate']:.1f}" for r in S.confirm_rots)])


# ------------------------------------------------------------------------------------------------------------------ the driver

MAIN_STAGES = (stage_prerequisites, stage_continual, stage_calibration, stage_screens, stage_tune, stage_confirm)


def legacy_wave_in_queue() -> bool:
    """Transition guard: waves submitted by the former scripts/40 are named mem_*; the driver does not know their runs, so it waits for them."""
    return any(n.startswith("mem_") for n in J.queued_names())


def drive(duo: str, *, submit: bool = False, force: bool = False, baseline: str | None = None, overrides: dict | None = None, log=print, signal: str = "indep",
          chain: bool = False) -> int:
    """Find the first stage of `duo` (for the `signal`'s runs) that is not done and print / do its work. Returns 0, or 1 where a decision is needed.
    With `chain`, a wave of jobs is followed by one CPU job that calls this again when the wave has finished (jobs.chain)."""
    if baseline and signal == "joint":
        log("the comparison methods (none, fixed schedules) are tuned for the independent signal only: the joint signal's members are coupled, which needs a pair search per method")
        return 1
    S = Setup.load(duo, overrides=overrides, log=log, signal=signal)
    log(f"duo {duo}: {S.names['large']} + {S.names['small']}; {'JOINT signal (both members on the gated duo entropy, the gate inside the loss)' if S.joint else 'independent signal (each member on its own entropy)'}"
        f"; tuning rotation(s) {list(S.rots)}" + (f"; comparison method {baseline}" if baseline else ""))
    if baseline:
        stages = [stage_prerequisites] + ([stage_continual] if baseline == "none" else []) + [lambda S: stage_baseline_rates(S, baseline), lambda S: stage_baseline_confirm(S, baseline)]
    else:
        stages = list(MAIN_STAGES)
    for fn in stages:
        for _ in range(3):                                          # an inline step is followed by a second look at the same stage
            step = fn(S)
            log(f"\n[{step.stage}] {step.state}")
            for line in step.notes:
                log(f"  {line}" if not line.startswith(" ") else line)
            if step.state == "done":
                break
            if step.state == "blocked":
                return 1
            if step.state == "inline":
                if not submit:
                    log("  next: this CPU step runs in place (add --submit)")
                    return 0
                try:
                    step.run()
                except SCR.ScreenError as e:
                    log(f"  STOP: {e}")
                    return 1
                continue
            if step.state == "jobs":
                if not baseline and not S.joint and step.stage == "tune" and legacy_wave_in_queue():
                    log("  NOT SUBMITTED: a wave of the former scripts/40 (jobs named mem_*) is still queued; call again when it has finished")
                    return 0
                n = J.submit(S.P, step.jobs, go=submit, force=force, cwd=Path(__file__).resolve().parents[2])
                log(f"  {len(step.jobs)} job(s)" + (f", {n} submitted" if submit else ": add --submit to launch them"))
                if chain:
                    again = ["--duo", duo, *(["--signal", "joint"] if S.joint else []), *(["--baseline", baseline] if baseline else []), "--submit", "--chain"]
                    if not J.chain(S.P, step.jobs, again, go=submit, tag=baseline or "", cwd=Path(__file__).resolve().parents[2]) and submit:
                        log("  no chain job: call this again by hand when the wave has finished")
                return 0
        else:
            raise RuntimeError(f"stage {step.stage} did not settle")
    log("\nall stages are done")
    return 0


if __name__ == "__main__":
    # the pairing of requests into runs: a member with fewer requests gets its filler cell
    S0 = MR.Setting(0.05, 0.005, 20, 0.7)
    C1 = MR.Setting(0.1, 0.01, 50, 1.4)
    grid = TU.Grid({"large": 7e-5, "small": 5e-4})
    fake = type("S", (), {"duo": "toy", "grid": grid})()
    reqs = {"large": [(S0, 4, 0), (C1, 6, 0), (C1, 8, 0)], "small": [(S0, -5, 0)]}
    jobs = _pair_jobs(fake, reqs, {"large": (S0, 6), "small": (S0, -5)})
    assert len(jobs) == 3 and all(j.args["rotation"] == 0 for j in jobs)
    assert jobs[0].args["cond"] == MR.condition(S0, S0) and jobs[0].args["lrs"] == [grid.rate("large", 4), grid.rate("small", -5)]
    assert jobs[1].args["cond"] == MR.condition(C1, S0) and jobs[1].args["lrs"] == [grid.rate("large", 6), grid.rate("small", -5)], "the small member's filler cell"
    # decoupled pairing of two ladders: unequal requests get the diagonal partner
    needs = {"large": [(0, 0), (2, 0), (3, 2)], "small": [(-4, 0)]}
    pairs = _runs_for(needs, grid.rate)
    assert pairs[0][0] == (grid.rate("large", 0), grid.rate("small", -4)) and len(pairs[0]) == 2 and len(pairs[2]) == 1 and pairs[2][0][0] == grid.rate("large", 3)
    # a saved continual report made from a shorter ladder is stale (None), not an error: extending the ladder with --mults must reach the report again
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _t:
        (Path(_t) / "results.json").write_text(json.dumps({"ladders": {"large": [1.0, 2.0], "small": [1.0, 2.0]}, "warnings": []}))
        stub = lambda axes: type("S", (), {"P": type("P", (), {"continual_dir": Path(_t)})(), "ladders": lambda self: axes})()
        assert continual_results(stub({"large": [1.0, 2.0], "small": [1.0, 2.0]})) is not None
        assert continual_results(stub({"large": [0.5, 1.0, 2.0], "small": [0.5, 1.0, 2.0]})) is None
    assert CALIBRATION["S"]["rotations"] == (0, 1, 2, 3) and len(CALIBRATION["S"]["seeds"]) * 4 + 2 == 10, "ten calibration streams: 8 stress + 2 healthy"
    # ---- the flow of a duo: a scratch output folder, the real config of convnextb_deits, synthetic calibration streams ----------------------------------------
    import os
    import re
    import tempfile

    quiet = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as tmp:
        text = (Path(__file__).resolve().parents[2] / "cfgs" / "global.yaml").read_text()
        yaml = Path(tmp) / "global.yaml"
        yaml.write_text(re.sub(r"(?m)^  out: out .*$", f"  out: {tmp}/out", text))
        os.environ["DUO_GLOBAL_CONFIG"] = str(yaml)
        g = load_global(str(yaml))
        S = Setup.load("convnextb_deits", g=g, log=quiet)
        lad = S.ladders()
        assert S.grid.rate("large", 0) == 1.25e-5 and S.grid.rate("small", 0) == 8e-4, "the grid is anchored at the duo's own episodic rates"
        assert len(lad["large"]) == 11 and lad["large"][4] == 6.25e-6 and lad["small"][4] == 4e-4 and lad["large"][10] == 5e-5
        cj = continual_jobs(S)
        assert len(cj) == 12 and cj[0].args["lrs"] is None and cj[1].args["lrs"] == [lad["large"][0], lad["small"][0]]
        assert stage_continual(S).state == "jobs" and len(stage_continual(S).jobs) == 12, "nothing is run yet: the whole ladder is requested"
        # what the continual report writes: the rates the calibration streams are made at (stress 8 rungs up, conservative 3)
        S.P.continual_dir.mkdir(parents=True)
        sides = {side: {"conservative": lad[side][3], "stress": lad[side][8], "frontier": lad[side][4], "warnings": []} for side in SIDES}
        (S.P.continual_dir / "results.json").write_text(json.dumps({"ladders": lad, "warnings": [], "sides": sides}))
        assert continual_results(S) is not None
        stress, healthy = calibration_rates(S)
        assert stress == (lad["large"][8], lad["small"][8]) and healthy == (lad["large"][3], lad["small"][3]), "derived from the continual stage"
        up = Setup.load("convnextb_deits", g=g, overrides={"stress_up": 1}, log=quiet)
        assert calibration_rates(up)[0] == (lad["large"][9], lad["small"][9]) and S.P.config_path.is_file(), "stress_up moves the stress rates up their ladders and is saved"
        up = Setup.load("convnextb_deits", g=g, overrides={"stress": [1e-4, 5e-3], "stress_up": 0}, log=quiet)
        assert calibration_rates(up)[0] == (1e-4, 5e-3), "explicit stress rates win"
        cfg = json.loads(S.P.config_path.read_text())
        S.P.config_path.write_text(json.dumps({k: v for k, v in cfg.items() if k not in ("stress", "stress_up")}))          # back to the derived rates
        S = Setup.load("convnextb_deits", g=g, log=quiet)
        jobs = calibration_jobs(S)
        assert len(jobs) == 10 and sum(j.args["name"] == "S" for j in jobs) == 8 and sum(j.args["name"] == "H" for j in jobs) == 2
        assert {j.args["rotation"] for j in jobs if j.args["name"] == "S"} == {0, 1, 2, 3} and all(j.args["stop_mode"] == "all" for j in jobs if j.args["name"] == "S")
        assert stage_calibration(S).state == "jobs" and len(stage_calibration(S).jobs) == 10
        # the calibration jobs' files are what the screens read: write them (collapsing both members in S, none in H)
        from src.reset.screens import synthetic_stream
        S.P.gate_stress_dir.mkdir(parents=True)
        for j in jobs:
            a = j.args
            synthetic_stream(S.P.gate_stress_dir, a["name"], None if a["name"] == "H" else 150 + 20 * a["seed"], a["seed"], lrs=a["lrs"], rotation=a["rotation"], beta=a["beta"],
                             collapse=("large", "small"))
            if j is jobs[0]:
                assert _calibration_done(S, j) and stage_calibration(S).jobs and len(stage_calibration(S).jobs) == 9, "one file written: one job less"
        assert stage_calibration(S).state == "done"
        # the screens run in place; the shared setting is the screen's, the candidates are other speeds
        st = stage_screens(S)
        assert st.state == "inline"
        st.run()
        assert stage_screens(S).state == "done"
        sh = S.shared
        assert isinstance(sh, MR.Setting) and all((c.alpha_fast, c.alpha_slow) != (sh.alpha_fast, sh.alpha_slow) for s in SIDES for c in SCR.candidates_of(S.P, s))
        # the first wave of the search: only the shared setting's X2 ladder, both members paired, rotation 0 (the candidates start at its chosen rung)
        tune = stage_tune(S)
        assert tune.state == "jobs" and len(tune.jobs) == 4 and {j.args["rotation"] for j in tune.jobs} == {0}, [j.args for j in tune.jobs]
        assert [j.args["lrs"] for j in tune.jobs] == [[S.grid.rate("large", a), S.grid.rate("small", b)] for a, b in zip(TU.X2["large"], TU.X2["small"])]
        assert all(j.args["cond"] == MR.condition(sh, sh) for j in tune.jobs)
        # a member that never collapses stops the driver with its name, instead of a traceback; every earlier stage is done, so nothing is submitted
        for j in cj:                                                         # (the continual stage looks for its run files before it reads results.json)
            _continual_path(S, j).parent.mkdir(parents=True, exist_ok=True)
            _continual_path(S, j).touch()
        assert stage_continual(S).state == "done"
        for f in list(S.P.gate_stress_dir.glob("exp-*")) + list(S.P.screens_dir.glob("*")):
            f.unlink()
        for j in jobs:
            a = j.args
            synthetic_stream(S.P.gate_stress_dir, a["name"], None if a["name"] == "H" else 150 + 20 * a["seed"], a["seed"], lrs=a["lrs"], rotation=a["rotation"], beta=a["beta"])
        real_submit, J.submit = J.submit, lambda *a, **k: (_ for _ in ()).throw(AssertionError("a self-test must not submit"))
        try:
            lines: list[str] = []
            assert drive("convnextb_deits", submit=True, log=lambda *a, **k: lines.append(" ".join(map(str, a)))) == 1
            assert any("STOP" in line and "large" in line and "never collapses" in line for line in lines), lines[-4:]
            assert not S.P.member_screen("large").is_file() and S.P.shared_screen.is_file(), "the shared screen was made, the large member's was refused"
        finally:
            J.submit = real_submit
        # ---- the JOINT signal: its own folders, anchors at the joint rates, jobs that carry the signal, the search to a known optimum ---------------------------------------
        import math

        SJ = Setup.load("convnextb_deits", g=g, log=quiet, signal="joint")
        assert SJ.joint and SJ.P.tag.endswith("_joint") and SJ.P.live_dir() != S.P.live_dir() and SJ.P.tune_dir != S.P.tune_dir
        assert SJ.grid.rate("large", 0) == 1.4e-4 and SJ.grid.rate("small", 0) == 1.1e-3, "anchored at the joint pair of cfgs/duo_lrs/convnextb_deits__joint.yaml"
        cjj = continual_jobs(SJ)
        assert len(cjj) == 12 and all(j.args.get("signal") == "joint" for j in cjj) and "beta" not in cjj[0].args and all(j.args["beta"] == 0.5 for j in cjj[1:])
        assert SJ.rots == (0,) and SJ.confirm_rots == (0,) and SJ.betas == (0.0, 0.25, 0.5, 0.75, 1.0), "the defaults: one rotation, the user's beta grid"
        assert drive("convnextb_deits", baseline="fixed500", signal="joint", log=quiet) == 1, "the comparison methods are independent-only"
        # the screens' outputs the search reads: the shared setting from config, the candidates from the member screens
        S0, C1, C2 = MR.Setting(0.05, 0.005, 20, 0.7), MR.Setting(0.1, 0.01, 50, 1.4), MR.Setting(0.1, 0.025, 50, 1.4)
        SJ.P.screens_dir.mkdir(parents=True, exist_ok=True)
        for side, cs in (("large", [C1, C2]), ("small", [C2, C1])):
            SJ.P.member_screen(side).write_text(json.dumps({"candidates": [dict(alpha_fast=c.alpha_fast, alpha_slow=c.alpha_slow, W0=c.W0, delta=c.delta) for c in cs]}))
        SJ.P.tune_dir.mkdir(parents=True, exist_ok=True)
        SJ.P.config_path.write_text(json.dumps({"shared": list(S0)}))
        SJ = Setup.load("convnextb_deits", g=g, log=quiet, signal="joint")
        made: dict = {}
        real_cell = TU.JointStore.cell
        TU.JointStore.cell = lambda self, cond, rot, *, large, small, beta: made.get((cond, rot, large, small, beta))
        TRUE = {"large": TU.delta_setting(C1, 1), "small": C2}

        def synthetic(a) -> TU.Cell:
            """The accuracy of a requested run: a known surface over (rate rungs, beta, both settings), plus noise."""
            inner = a["cond"].split("~")[1] if a["cond"].startswith("random~") else a["cond"]
            sets = MR.parse_condition(inner)
            kv, ks = SJ.grid.rung("large", a["lrs"][0]), SJ.grid.rung("small", a["lrs"][1])

            def dpen(st, true):
                return 0.01 if (st.alpha_fast, st.alpha_slow, st.W0) != (true.alpha_fast, true.alpha_slow, true.W0) else 0.004 * (math.log2(st.delta / true.delta) * 2) ** 2
            acc = (0.6 - 0.01 * (kv - 1) ** 2 - 0.01 * (ks + 1) ** 2 - 0.004 * ((a["beta"] - 0.75) / 0.25) ** 2 - dpen(sets["large"], TRUE["large"]) - dpen(sets["small"], TRUE["small"])
                   - (0.02 if a["cond"].startswith("random~") else 0.0))
            acc = acc + np.random.default_rng(abs(hash((a["cond"], a["rotation"], a["lrs"][0], a["beta"]))) % 2**32).normal(0, 0.0015, 16)
            return TU.Cell({"gate": acc, "ts": acc, "large": acc, "small": acc}, {"large": 5.0, "small": 5.0}, {"large": False, "small": False}, 800000)

        try:
            waves = 0
            for _ in range(60):
                st = stage_tune(SJ)
                if st.state != "jobs":
                    break
                waves += 1
                for j in st.jobs:
                    a = j.args
                    assert a["signal"] == "joint" and a["rotation"] == 0 and a["stream"] == "heldout", a
                    made[(a["cond"], a["rotation"], a["lrs"][0], a["lrs"][1], a["beta"])] = synthetic(a)
            assert st.state == "done", st.notes[-3:]
            choice = json.loads((SJ.P.tune_dir / "joint__choice.json").read_text())
            assert choice["beta"] == 0.75 and choice["rungs"] == [1, -1] and choice["converged"] and choice["cond"] == MR.condition(TRUE["large"], TRUE["small"]), choice
            assert choice["chosen"] == [SJ.grid.rate("large", 1), SJ.grid.rate("small", -1)]
            print(f"  joint flow: the known optimum (rungs (1, -1), gate beta 0.75, one detector per member) in {waves} waves, {len(made)} runs")
            # the confirmation of the joint signal: the chosen cell exists already (the search ran it), so there is nothing left to run
            cf = stage_confirm(SJ)
            conf = json.loads((SJ.P.tune_dir / "joint__confirm.json").read_text())
            assert cf.state == "done" and conf["beta"] == 0.75 and list(conf["rotations"]) == ["0"], cf.notes
        finally:
            TU.JointStore.cell = real_cell
    os.environ.pop("DUO_GLOBAL_CONFIG", None)
    print("pipeline self-test passed (a synthetic duo through every stage but the live waves; the main duo's stored runs are checked by scripts/tune_reset.py)")
