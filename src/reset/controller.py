"""
controller.py
=============
The reset rule of the duo's two members, as a function of the proxy stream only (Phase 4).

    ctl = ResetController(RuleSpec.of(own=dict(kind="fast_vs_slow", alpha_fast=0.05, alpha_slow=0.005, delta=0.7), W0=20))
    events = ctl.update(r_l, r_s, position=n_samples_seen)         # once per proxy batch; r = the gate's proxy of each member
    for e in events:                                                # each e.member must be reset (actions.reset_member)
        ...

Each member has its OWN rule (a `RuleSpec` per member; one spec for both is the simple case, per-member settings are src/reset/member_rules.py).
A member's rule watches only its own score s_t = logit(clip(r_t, eps, 1 - eps)): a member is never reset because of what its partner does,
and the partner's detector is untouched when a member is reset.

**Decisions are independent**: both members' detectors are updated first, the members to reset are then read off, and only then are the actions
applied. Both can fire on one batch.

**The action** (the live loop does the model/optimizer part, actions.py): for each flagged member the controller (3) restarts its detector (a new
warm-up) and (4) starts a cooldown of K proxy batches (default W0), during which it cannot be flagged again (its detector keeps running).

Every flag is logged as a ResetEvent: proxy-batch index, stream position, member, rule, the detector's statistic, and both s values.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

from src.reset.detectors import FastVsSlow, make_detector

MEMBERS = ("large", "small")
QUANTITIES = ("own",)          # the only quantity a rule watches; `RuleSpec.parts` keeps the stored JSON shape {"own": {"kind": "fast_vs_slow", ...}}


def logit_score(r: float, eps: float) -> float:
    """s = logit(clip(r, eps, 1 - eps))."""
    r = min(max(float(r), eps), 1.0 - eps)
    return math.log(r / (1.0 - r))


@dataclass
class RuleSpec:
    """One member's rule: the detector's parameters (`parts["own"]`: {"kind": "fast_vs_slow", alpha_fast, alpha_slow, delta}) and the parameters it
    shares with the controller: warm-up W0, cooldown K (default W0), standardizer rate alpha_var, standardize."""
    parts: dict[str, dict]
    W0: int = 20
    K: int | None = None
    alpha_var: float = 0.01
    standardize: bool = True

    def __post_init__(self):
        if set(self.parts) != set(QUANTITIES):
            raise ValueError(f"parts must be {QUANTITIES} (the gap rules were dropped), got {sorted(self.parts)}")
        if self.cooldown < 0:
            raise ValueError("the cooldown K must be >= 0")

    @classmethod
    def of(cls, own: dict, **common) -> "RuleSpec":
        return cls({"own": dict(own)}, **common)

    @property
    def cooldown(self) -> int:
        return self.W0 if self.K is None else int(self.K)

    @property
    def name(self) -> str:
        return f"own_{self.parts['own']['kind']}"

    def build(self) -> FastVsSlow:
        p = dict(self.parts["own"])
        kind = p.pop("kind")
        for k, v in (("W0", self.W0), ("alpha_var", self.alpha_var), ("standardize", self.standardize)):
            p.setdefault(k, v)
        return make_detector(kind, **p)

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
    rule: str                    # "own_fast_vs_slow"
    statistic: float             # the detector's statistic when it fired
    s_large: float
    s_small: float

    def to_json(self) -> dict:
        return asdict(self)


class ResetController:
    def __init__(self, specs: RuleSpec | Mapping[str, RuleSpec], *, eps: float = 1e-6):
        self.specs = {m: specs for m in MEMBERS} if isinstance(specs, RuleSpec) else dict(specs)
        if set(self.specs) != set(MEMBERS):
            raise ValueError(f"one rule per member {MEMBERS}, got {sorted(self.specs)}")
        self.eps = float(eps)
        self.det: dict[str, FastVsSlow] = {m: self.specs[m].build() for m in MEMBERS}
        self.cooldown = {m: 0 for m in MEMBERS}
        self.t = 0
        self.events: list[ResetEvent] = []

    def update(self, r_l: float, r_s: float, position: int | None = None) -> list[ResetEvent]:
        """One proxy batch (the two members' proxy values). Returns the flags (members to reset), possibly both."""
        s = {"large": logit_score(r_l, self.eps), "small": logit_score(r_s, self.eps)}
        alarm = {m: self.det[m].update(s[m]) for m in MEMBERS}            # both detectors are updated, whatever the other says
        flagged = []
        for m in MEMBERS:
            blocked = self.cooldown[m] > 0
            if blocked:
                self.cooldown[m] -= 1
            if alarm[m] and not blocked:
                flagged.append(m)
        events = [ResetEvent(self.t, position, m, self.specs[m].name, float(self.det[m].statistic), s["large"], s["small"]) for m in flagged]
        for m in flagged:                                                   # the actions come after ALL the decisions of this batch
            self.det[m].restart()
            self.cooldown[m] = self.specs[m].cooldown
        self.t += 1
        self.events += events
        return events

    # ---- state ----------------------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"t": self.t, "cooldown": dict(self.cooldown), "events": [e.to_json() for e in self.events],
                "detectors": {m: d.state_dict() for m, d in self.det.items()}}

    def load_state_dict(self, s: dict) -> None:
        self.t, self.cooldown = s["t"], dict(s["cooldown"])
        self.events = [ResetEvent(**e) for e in s["events"]]
        for m, st in s["detectors"].items():
            self.det[m].load_state_dict(st)


# ---------------------------------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    import json

    import numpy as np

    sig = lambda s: 1.0 / (1.0 + np.exp(-np.asarray(s, dtype=float)))                      # noqa: E731
    fvs = dict(kind="fast_vs_slow", alpha_fast=0.2, alpha_slow=0.02, delta=2.0)

    def drive(ctl, s_l, s_s):
        for t, (a, b) in enumerate(zip(s_l, s_s)):
            ctl.update(sig(a), sig(b), position=64 * (t + 1))
        return ctl.events

    # ---- logit_score, specs ----------------------------------------------------------------------------------------------
    assert abs(logit_score(0.5, 1e-6)) < 1e-12 and abs(logit_score(0.0, 1e-6) + logit_score(1.0, 1e-6)) < 1e-9
    assert abs(logit_score(0.0, 1e-6) - math.log(1e-6 / (1 - 1e-6))) < 1e-9
    assert abs(logit_score(0.0, 1e-3) - math.log(1e-3 / (1 - 1e-3))) < 1e-9, "the clip is the eps given"
    spec = RuleSpec.of(own=fvs)
    assert spec.name == "own_fast_vs_slow" and spec.cooldown == 20 and RuleSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec
    d = spec.build()
    assert (d.alpha_fast, d.alpha_slow, d.delta, d.W0) == (0.2, 0.02, 2.0, 20)
    for bad in ({}, {"gap": fvs}, {"own": fvs, "gap": fvs}):
        try:
            RuleSpec(bad)
            raise AssertionError("bad parts must be refused")
        except ValueError:
            pass
    try:
        RuleSpec.of(own=dict(kind="cusum", nu=0.5, h=8.0)).build()
        raise AssertionError("a dropped procedure must be refused")
    except ValueError:
        pass

    rng = np.random.default_rng(1)
    T, t0 = 500, 250

    # ---- unit test: equal drop in both members -> each member fires separately, on its own signal -----------------------
    s_l, s_s = rng.normal(0, 1, T), rng.normal(0, 1, T)
    s_l[t0:] -= 5.0
    s_s[t0:] -= 5.0
    ev = drive(ResetController(RuleSpec.of(own=fvs, W0=50)), s_l, s_s)
    assert sorted(e.member for e in ev) == ["large", "small"] and all(t0 <= e.t <= t0 + 15 for e in ev) and {e.rule for e in ev} == {"own_fast_vs_slow"}, ev
    print(f"  equal drop of 5 sigma at batch {t0}: large fires at +{[e.t for e in ev if e.member == 'large'][0] - t0}, small at +{[e.t for e in ev if e.member == 'small'][0] - t0}")

    # ---- one member alone collapses: only that member is flagged -------------------------------------------------------------
    for collapsing, other in (("large", "small"), ("small", "large")):
        s = {"large": rng.normal(0, 1, T), "small": rng.normal(0, 1, T)}
        s[collapsing][t0:] -= 5.0
        ev = drive(ResetController(RuleSpec.of(own=fvs, W0=50)), s["large"], s["small"])
        assert ev and ev[0].member == collapsing and ev[0].t - t0 <= 15 and all(e.member == collapsing or e.t < t0 for e in ev), (collapsing, ev)

    # ---- cooldown, restarts, the log -----------------------------------------------------------------------------------------
    # a rule that fires on every batch after its warm-up (delta far below any statistic, raw units): flags are max(W0, K) + 1 batches apart
    always = dict(kind="fast_vs_slow", alpha_fast=0.2, alpha_slow=0.02, delta=-1e9)
    for W0, K in ((10, 25), (10, 5)):
        ctl = ResetController(RuleSpec.of(own=always, W0=W0, K=K, standardize=False))
        larges = [e.t for e in drive(ctl, rng.normal(0, 1, 200), rng.normal(0, 1, 200)) if e.member == "large"]
        assert larges[0] == W0 and all(b - a == max(W0, K) + 1 for a, b in zip(larges, larges[1:])), (W0, K, larges)
    # after a flag of `large`: its detector restarts and its cooldown starts, the partner's detector is untouched
    ctl = ResetController(RuleSpec.of(own=fvs, W0=10, K=15))
    for _ in range(40):
        ctl.update(sig(rng.normal(0, 1)), sig(rng.normal(0, 1)))
    assert ctl.det["small"].n == ctl.det["large"].n == 40
    ev = []
    while not ev:
        ev = ctl.update(sig(-9.0), sig(rng.normal(0, 1)))
    assert [e.member for e in ev] == ["large"]
    assert ctl.det["large"].n == 0, "restarted"
    assert ctl.det["small"].n == ctl.t, "the partner's detector is untouched"
    assert ctl.cooldown == {"large": 15, "small": 0}
    e = ev[0]
    assert e.statistic > 2.0 and e.rule == "own_fast_vs_slow" and e.s_large < -8
    assert json.loads(json.dumps(e.to_json()))["t"] == ctl.t - 1

    # ---- decisions are independent, and both members can fire on one batch -------------------------------------------------
    base = rng.normal(0, 1, T)
    base[t0:] -= 6.0                                                    # identical sequences: both rules fire on ONE batch
    fwd = drive(ResetController(RuleSpec.of(own=fvs, W0=50)), base, base)
    assert [(e.t, e.member) for e in fwd[:2]] == [(fwd[0].t, "large"), (fwd[0].t, "small")] and t0 <= fwd[0].t <= t0 + 15, fwd
    s_l, s_s = rng.normal(0, 1, T), rng.normal(0, 1, T)
    s_l[t0:] -= 6.0
    s_s[t0:] -= 6.0
    fwd = drive(ResetController(RuleSpec.of(own=fvs, W0=50)), s_l, s_s)
    swp = drive(ResetController(RuleSpec.of(own=fvs, W0=50)), s_s, s_l)
    assert {(e.t, e.member) for e in fwd} == {(e.t, {"large": "small", "small": "large"}[e.member]) for e in swp}, "no order effect"
    # ... and a member with a silent rule is never reset, whatever the other does
    silent = dict(fvs, delta=1e9)
    sil = ResetController({"large": RuleSpec.of(own=fvs, W0=50), "small": RuleSpec.of(own=silent, W0=50)})
    assert {e.member for e in drive(sil, s_l, s_s)} == {"large"}, "each member has its own rule"

    # ---- state round trip ---------------------------------------------------------------------------------------------------
    s_l, s_s = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    s_l[300:] -= 5.0
    a = ResetController(RuleSpec.of(own=fvs, W0=30))
    b = ResetController(RuleSpec.of(own=fvs, W0=30))
    for x, y in zip(s_l[:200], s_s[:200]):
        a.update(sig(x), sig(y))
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    ra = [a.update(sig(x), sig(y)) for x, y in zip(s_l[200:], s_s[200:])]
    rb = [b.update(sig(x), sig(y)) for x, y in zip(s_l[200:], s_s[200:])]
    assert ra == rb and json.dumps(a.state_dict()) == json.dumps(b.state_dict()) and a.events      # (NaN != NaN: compare the JSON)
    print("controller self-test passed")
