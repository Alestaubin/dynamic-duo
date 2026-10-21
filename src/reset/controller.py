"""
controller.py
=============
The reset rules of the duo's two members, as a function of the proxy stream only (Phase 4).

    ctl = ResetController(RuleSpec.of(gap=dict(kind="cusum", nu=0.5, h=8.0), own=dict(kind="cusum", nu=0.5, h=8.0), W0=20),
                          c=0.0)                                   # the combined rule: gap_cusum OR own_cusum, on both members
    events = ctl.update(r_l, r_s, position=n_samples_seen)         # once per proxy batch; r = the selected proxy of each member
    for e in events:                                                # each e.member must be reset (actions.reset_member)
        ...

Each member has its OWN rule (`RuleSpec` per member; one spec for both is the usual case). A member's rule watches
  own   s_t^(m) = logit(clip(r_t^(m), eps, 1 - eps))
  gap   g_t^(m) = s_t^(m) - s_t^(m') - c   for the large member (m' = small; c = the gate's centering constant, 0 if none),
        and its mirror  s_small - s_large + c  for the small one. The brief writes "s^(m) - s^(m') - c" for both; with one c
        that is not antisymmetric (a constant offset d between the members could not be removed from both gaps at once), so
        c is the offset of s_large - s_small, the way the gate uses it (w_L = sigmoid(beta (s_L - s_S - c)): g^(small) = -g^(large)).
        With c = 0, today's selection, the two readings coincide.
and fires when any of its detectors alarms (OR; every detector is updated on every batch, none is skipped).

**Decisions are independent**: every detector of both members is updated first, the members to reset are then read off, and only
then are the actions applied, so a member is never flagged because of what the partner's action did on the same batch. Both can
fire on one batch.

**The action** (the live loop does the model/optimizer part, actions.py): for each flagged member m the controller
  (3) restarts all detectors of m, and the `gap` detector of the partner (the partner's gap is built from m's proxy, which jumps
      when m is reset; the partner's `own` detector and the partner itself are untouched), and
  (4) starts a cooldown of K proxy batches for m, during which m cannot be flagged again (its detectors keep running).
`restart_partner_gap=False` exists only to document what the restart prevents (tests below).

Every flag is logged as a ResetEvent: proxy-batch index, stream position, member, rule (the detectors that fired), the value of the
(first) firing detector's statistic, and both s values.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping

from src.reset.detectors import Detector, make_detector

MEMBERS = ("large", "small")
PARTNER = {"large": "small", "small": "large"}
QUANTITIES = ("gap", "own")


def logit_score(r: float, eps: float) -> float:
    """s = logit(clip(r, eps, 1 - eps))."""
    r = min(max(float(r), eps), 1.0 - eps)
    return math.log(r / (1.0 - r))


@dataclass
class RuleSpec:
    """One member's rule: which detectors watch which quantity (`parts`: {"own": {"kind": ..., **params}, "gap": {...}}), and the
    parameters shared by its detectors: warm-up W0, cooldown K (default W0), standardizer rate alpha_var, standardize."""
    parts: dict[str, dict]
    W0: int = 20
    K: int | None = None
    alpha_var: float = 0.01
    standardize: bool = True

    def __post_init__(self):
        if not self.parts or set(self.parts) - set(QUANTITIES):
            raise ValueError(f"parts must be a non-empty subset of {QUANTITIES}, got {sorted(self.parts)}")
        if self.cooldown < 0:
            raise ValueError("the cooldown K must be >= 0")

    @classmethod
    def of(cls, own: dict | None = None, gap: dict | None = None, **common) -> "RuleSpec":
        return cls({q: dict(p) for q, p in (("gap", gap), ("own", own)) if p is not None}, **common)

    @property
    def cooldown(self) -> int:
        return self.W0 if self.K is None else int(self.K)

    @property
    def name(self) -> str:
        return " OR ".join(f"{q}_{self.parts[q]['kind']}" for q in QUANTITIES if q in self.parts)

    def build(self) -> dict[str, Detector]:
        out = {}
        for q in QUANTITIES:
            if q not in self.parts:
                continue
            p = dict(self.parts[q])
            kind = p.pop("kind")
            if kind == "cusum" and "mu" not in p:
                p["mu"] = "warmup" if q == "own" else 0.0        # own: the frozen warm-up mean; gap: 0
            for k, v in (("W0", self.W0), ("alpha_var", self.alpha_var), ("standardize", self.standardize)):
                p.setdefault(k, v)
            out[q] = make_detector(kind, **p)
        return out

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping) -> "RuleSpec":
        return cls(**d)


@dataclass
class ResetEvent:
    t: int                       # proxy batch (0-based, counted by the controller)
    position: int | None         # stream position given by the caller (samples seen when the proxy batch completed)
    member: str
    rule: str                    # the detectors that fired, e.g. "gap_cusum" or "gap_cusum+own_cusum"
    statistic: float             # the value of the first firing detector's statistic
    s_large: float
    s_small: float

    def to_json(self) -> dict:
        return asdict(self)


class ResetController:
    def __init__(self, specs: RuleSpec | Mapping[str, RuleSpec], *, c: float = 0.0, eps: float = 1e-6,
                 restart_partner_gap: bool = True):
        self.specs = {m: specs for m in MEMBERS} if isinstance(specs, RuleSpec) else dict(specs)
        if set(self.specs) != set(MEMBERS):
            raise ValueError(f"one rule per member {MEMBERS}, got {sorted(self.specs)}")
        self.c, self.eps, self.restart_partner_gap = float(c), float(eps), bool(restart_partner_gap)
        self.det: dict[str, dict[str, Detector]] = {m: self.specs[m].build() for m in MEMBERS}
        self.cooldown = {m: 0 for m in MEMBERS}
        self.t = 0
        self.events: list[ResetEvent] = []

    def quantities(self, s_l: float, s_s: float) -> dict[str, dict[str, float]]:
        return {"large": {"own": s_l, "gap": s_l - s_s - self.c},
                "small": {"own": s_s, "gap": s_s - s_l + self.c}}

    def update(self, r_l: float, r_s: float, position: int | None = None) -> list[ResetEvent]:
        """One proxy batch (the two members' proxy values). Returns the flags (members to reset), possibly both."""
        s_l, s_s = logit_score(r_l, self.eps), logit_score(r_s, self.eps)
        q = self.quantities(s_l, s_s)
        alarms: dict[str, list[tuple[str, Detector]]] = {m: [] for m in MEMBERS}
        for m in MEMBERS:
            for name, d in self.det[m].items():              # every detector is updated, whatever the others say
                if d.update(q[m][name]):
                    alarms[m].append((name, d))
        flagged = []
        for m in MEMBERS:
            blocked = self.cooldown[m] > 0
            if blocked:
                self.cooldown[m] -= 1
            if alarms[m] and not blocked:
                flagged.append(m)
        events = [ResetEvent(self.t, position, m, "+".join(f"{n}_{d.kind}" for n, d in alarms[m]),
                             float(alarms[m][0][1].statistic), s_l, s_s) for m in flagged]
        for m in flagged:                                    # the actions come after ALL the decisions of this batch
            for d in self.det[m].values():
                d.restart()
            if self.restart_partner_gap and "gap" in self.det[PARTNER[m]]:
                self.det[PARTNER[m]]["gap"].restart()
            self.cooldown[m] = self.specs[m].cooldown
        self.t += 1
        self.events += events
        return events

    # ---- state ----------------------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"t": self.t, "cooldown": dict(self.cooldown), "events": [e.to_json() for e in self.events],
                "detectors": {m: {n: d.state_dict() for n, d in ds.items()} for m, ds in self.det.items()}}

    def load_state_dict(self, s: dict) -> None:
        self.t, self.cooldown = s["t"], dict(s["cooldown"])
        self.events = [ResetEvent(**e) for e in s["events"]]
        for m, ds in s["detectors"].items():
            for n, st in ds.items():
                self.det[m][n].load_state_dict(st)

    def describe(self) -> dict:
        return {"rules": {m: self.specs[m].to_json() for m in MEMBERS}, "c": self.c, "eps": self.eps,
                "restart_partner_gap": self.restart_partner_gap}


# ---------------------------------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    import json

    import numpy as np

    sig = lambda s: 1.0 / (1.0 + np.exp(-np.asarray(s, dtype=float)))                      # noqa: E731
    cus = dict(kind="cusum", nu=0.5, h=8.0)

    def drive(ctl, s_l, s_s):
        for t, (a, b) in enumerate(zip(s_l, s_s)):
            ctl.update(sig(a), sig(b), position=64 * (t + 1))
        return ctl.events

    # ---- logit_score, specs ----------------------------------------------------------------------------------------------
    assert abs(logit_score(0.5, 1e-6)) < 1e-12 and abs(logit_score(0.0, 1e-6) + logit_score(1.0, 1e-6)) < 1e-9
    assert abs(logit_score(0.0, 1e-6) - math.log(1e-6 / (1 - 1e-6))) < 1e-9
    assert abs(logit_score(0.0, 1e-3) - math.log(1e-3 / (1 - 1e-3))) < 1e-9, "the clip is the eps given"
    spec = RuleSpec.of(gap=cus, own=cus)
    assert spec.name == "gap_cusum OR own_cusum" and spec.cooldown == 20 and RuleSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec
    ds = spec.build()
    assert ds["gap"].mu == 0.0 and ds["own"].mu == "warmup", "CUSUM reference: 0 for the gap, the warm-up mean for own"
    for bad in ({}, {"level": {"kind": "cusum"}}):
        try:
            RuleSpec(bad)
            raise AssertionError("bad parts must be refused")
        except ValueError:
            pass
    ctl = ResetController(spec, c=2.0)
    q = ctl.quantities(5.0, 1.0)
    assert q["large"]["gap"] == 5.0 - 1.0 - 2.0 and q["small"]["gap"] == -q["large"]["gap"], "the small member's gap is the mirror"

    rng = np.random.default_rng(1)
    T, t0 = 500, 250

    # ---- unit test: equal drop in both members -> gap silent for both, own fires for each member separately -------------
    s_l, s_s = rng.normal(0, 1, T), rng.normal(0, 1, T)
    s_l[t0:] -= 5.0
    s_s[t0:] -= 5.0
    assert drive(ResetController(RuleSpec.of(gap=cus, W0=50)), s_l, s_s) == [], "the gap does not move when both drop together"
    ev = drive(ResetController(RuleSpec.of(own=cus, W0=50)), s_l, s_s)
    assert sorted(e.member for e in ev) == ["large", "small"] and all(t0 <= e.t <= t0 + 4 for e in ev), ev
    assert {e.rule for e in ev} == {"own_cusum"}
    ev = drive(ResetController(spec.__class__.of(gap=cus, own=cus, W0=50)), s_l, s_s)
    assert sorted(e.member for e in ev) == ["large", "small"] and {e.rule for e in ev} == {"own_cusum"}, "combined: the own part fires"
    print(f"  equal drop of 5 sigma at batch {t0}: gap silent, own fires for large at +{ev[0].t - t0}, for small at +{ev[1].t - t0}")

    # ---- one member alone collapses: the gap catches it (and own does too); the other is not flagged ----------------------
    s_l, s_s = rng.normal(0, 1, T), rng.normal(0, 1, T)
    s_l[t0:] -= 5.0
    ev = drive(ResetController(RuleSpec.of(gap=cus, W0=50)), s_l, s_s)
    assert ev and ev[0].member == "large" and ev[0].t - t0 <= 5 and all(e.member == "large" or e.t > t0 for e in ev)
    s_l, s_s = rng.normal(0, 1, T), rng.normal(0, 1, T)
    s_s[t0:] -= 5.0
    ev = drive(ResetController(RuleSpec.of(gap=cus, W0=50)), s_l, s_s)
    assert ev and ev[0].member == "small" and ev[0].t - t0 <= 5, "the mirror gap catches the small member's collapse"

    # ---- unit test: a member that is just weaker (constant offset): gap with c = the offset stays silent ------------------
    s_l, s_s = rng.normal(0, 1, T), rng.normal(-3.0, 1, T)             # s_S is 3 below s_L all the time
    assert drive(ResetController(RuleSpec.of(gap=cus, W0=50), c=3.0), s_l, s_s) == [], "centred on the offset: silent for both"
    ev = drive(ResetController(RuleSpec.of(gap=cus, W0=50), c=0.0), s_l, s_s)
    assert ev and {e.member for e in ev} == {"small"}, "without the centring the weaker member keeps being flagged"
    print(f"  constant offset of 3: c = 3 -> 0 events; c = 0 -> {len(ev)} flags of the small member")
    s_l, s_s = rng.normal(-3.0, 1, T), rng.normal(0, 1, T)             # the large member is the weaker one: s_L - s_S = -3
    assert drive(ResetController(RuleSpec.of(gap=cus, W0=50), c=-3.0), s_l, s_s) == [], "c is the offset of s_L - s_S, signed"
    ev = drive(ResetController(RuleSpec.of(gap=cus, W0=50), c=0.0), s_l, s_s)
    assert ev and {e.member for e in ev} == {"large"}

    # ---- cooldown, restarts, the log -----------------------------------------------------------------------------------------
    s_l, s_s = np.concatenate([rng.normal(0, 1, 100), np.full(100, -9.0)]), rng.normal(0, 1, 200)
    ctl = ResetController(RuleSpec.of(own=dict(kind="level", tau=-3.0), W0=10, K=25))      # a level rule keeps firing while low
    ev = drive(ctl, s_l, s_s)
    larges = [e.t for e in ev if e.member == "large"]
    assert larges[0] == 100 and all(b - a >= 25 + 1 + 10 - 1 for a, b in zip(larges, larges[1:])), larges
    # after a flag of `large`: its detectors restart, the partner's gap detector restarts, the partner's own one does not
    ctl = ResetController(RuleSpec.of(own=cus, gap=cus, W0=10, K=15))
    for _ in range(40):
        ctl.update(sig(rng.normal(0, 1)), sig(rng.normal(0, 1)))
    n_small_own = ctl.det["small"]["own"].n
    assert ctl.det["small"]["gap"].n == ctl.det["large"]["own"].n == 40
    ev = []
    while not ev:
        ev = ctl.update(sig(-9.0), sig(rng.normal(0, 1)))
    assert [e.member for e in ev] == ["large"]
    assert ctl.det["large"]["own"].n == ctl.det["large"]["gap"].n == 0 and ctl.det["small"]["gap"].n == 0, "restarted"
    assert ctl.det["small"]["own"].n == ctl.t, "the partner's own detector is untouched"
    assert ctl.cooldown == {"large": 15, "small": 0}
    e = ev[0]
    assert e.statistic > 8.0 and e.member == "large" and e.rule in ("own_cusum", "gap_cusum", "gap_cusum+own_cusum") and e.s_large < -8
    assert json.loads(json.dumps(e.to_json()))["t"] == ctl.t - 1

    # ---- decisions are independent, and both members can fire on one batch -------------------------------------------------
    base = rng.normal(0, 1, T)
    base[t0:] -= 6.0                                                    # identical sequences: both rules must fire on ONE batch
    fwd = drive(ResetController(RuleSpec.of(own=cus, W0=50)), base, base)
    assert [(e.t, e.member) for e in fwd[:2]] == [(fwd[0].t, "large"), (fwd[0].t, "small")] and t0 <= fwd[0].t <= t0 + 3, fwd
    s_l, s_s = rng.normal(0, 1, T), rng.normal(0, 1, T)
    s_l[t0:] -= 6.0
    s_s[t0:] -= 6.0
    fwd = drive(ResetController(RuleSpec.of(own=cus, W0=50)), s_l, s_s)
    swp = drive(ResetController(RuleSpec.of(own=cus, W0=50)), s_s, s_l)
    assert {(e.t, e.member) for e in fwd} == {(e.t, {"large": "small", "small": "large"}[e.member]) for e in swp}, "no order effect"
    # ... and a rule of one member never reads the other's detectors: a member with a silent rule is never reset
    sil = ResetController({"large": RuleSpec.of(own=cus, W0=50), "small": RuleSpec.of(own=dict(kind="cusum", nu=0.5, h=1e9), W0=50)})
    ev = drive(sil, s_l, s_s)
    assert {e.member for e in ev} == {"large"}, "each member has its own rule"

    # ---- unit test: one member reset, partner unchanged (closed loop) ----------------------------------------------------
    # The large member collapses at batch 300 and is reset by its own (deliberately slow) rule; the reset restores it from the next
    # batch on. The small member never changes, but its gap = s_S - s_L rises while L is collapsed and falls back when L recovers.
    def closed_loop(partner_gap: dict, restart_partner_gap: bool, standardize: bool, seed: int = 0, depth: float = 10.0):
        r = np.random.default_rng(seed)
        ctl = ResetController({"large": RuleSpec.of(own=dict(kind="cusum", nu=0.5, h=100.0), W0=20),
                               "small": RuleSpec.of(gap=partner_gap, W0=50, K=50, standardize=standardize)},
                              restart_partner_gap=restart_partner_gap)
        collapsed = False
        for t in range(700):
            collapsed = collapsed or t == 300
            for e in ctl.update(sig(r.normal(0, 1) - (depth if collapsed else 0.0)), sig(r.normal(0, 1))):
                if e.member == "large":
                    collapsed = False                                  # the reset restores it
        return ctl.events

    # thresholds in noise units (standardized) and in raw s units (not): the same distance from the noise floor either way
    procs = {True: {"imm_vs_ema": dict(kind="imm_vs_ema", alpha=0.2, delta=4.5),
                    "fast_vs_slow": dict(kind="fast_vs_slow", alpha_fast=0.2, alpha_slow=0.05, delta=1.5),
                    "slope": dict(kind="slope", W=16, gamma=0.3, k=1), "cusum": cus, "level": dict(kind="level", tau=-3.5)},
             False: {"imm_vs_ema": dict(kind="imm_vs_ema", alpha=0.2, delta=6.0),
                     "fast_vs_slow": dict(kind="fast_vs_slow", alpha_fast=0.2, alpha_slow=0.05, delta=3.0),
                     "slope": dict(kind="slope", W=16, gamma=0.4, k=1), "cusum": dict(kind="cusum", nu=0.7, h=11.0),
                     "level": dict(kind="level", tau=-5.0)}}
    print("  one member reset (large), partner (small) unchanged: flags of the SMALL member, with / without the partner-gap restart")
    for std, table in procs.items():
        for name, p in table.items():
            with_r = [e for e in closed_loop(p, True, std) if e.member == "small"]
            without = [e for e in closed_loop(p, False, std) if e.member == "small"]
            large = [e for e in closed_loop(p, True, std) if e.member == "large"]
            assert len(large) == 1 and 300 <= large[0].t <= 420, (name, large)           # the reset itself happens once
            print(f"    {'standardized' if std else 'raw units  '} {name:13s} with restart: {len(with_r)}   without: {len(without)}")
            assert not with_r, (std, name, with_r)
            if not std and name in ("imm_vs_ema", "fast_vs_slow", "slope"):
                assert without, f"{name}: the recovery of the reset member looks like a drop of the partner's gap"
            if name in ("cusum", "level"):
                assert not without, f"{name} compares with 0 / the warm-up mean of the healthy past, so L's recovery is no drop"
    # (standardized: the running std grows 2-3x within a few batches of the jump, which absorbs most of the spurious drop; the
    #  counts above are reported, not asserted, because they depend on how long L stays collapsed before its rule fires)

    # ---- state round trip ---------------------------------------------------------------------------------------------------
    s_l, s_s = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    s_l[300:] -= 5.0
    a = ResetController(RuleSpec.of(gap=cus, own=cus, W0=30))
    b = ResetController(RuleSpec.of(gap=cus, own=cus, W0=30))
    for x, y in zip(s_l[:200], s_s[:200]):
        a.update(sig(x), sig(y))
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    ra = [a.update(sig(x), sig(y)) for x, y in zip(s_l[200:], s_s[200:])]
    rb = [b.update(sig(x), sig(y)) for x, y in zip(s_l[200:], s_s[200:])]
    assert ra == rb and json.dumps(a.state_dict()) == json.dumps(b.state_dict()) and a.events      # (NaN != NaN: compare the JSON)
    print("controller self-test passed")
