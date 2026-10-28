"""
member_rules.py
===============
PER-MEMBER detector settings of the default reset rule, `own_fast_vs_slow`: each member of a duo gets its OWN `fast_vs_slow` parameters -- the two averaging speeds
(alpha_fast, alpha_slow), the warm-up W0 (cooldown K = W0) and the threshold delta -- besides its own learning rate, because different architectures collapse
differently (plan.md, "Per-member detectors"). A condition of scripts/38_run_reset_live.py names the two settings in the open, so a run's file name says what it ran:

    own_fast_vs_slow@pmL<af>-<as>-<W0>-<delta>_S<af>-<as>-<W0>-<delta>        L = the large member (ViT), S = the small member (ResNet)
    e.g.  own_fast_vs_slow@pmL0.05-0.005-20-0.7_S0.02-0.005-100-0.7

Only own-signal rules are per-member (the members' decisions then do not depend on each other, so each member's trajectory depends on its own rate and its own
setting alone: one run measures both members, and the two are tuned independently). The other detector parameters (alpha_var, standardize, K) stay as the screen
selected them (`base`).
"""

from __future__ import annotations

import re
from typing import NamedTuple

from src.reset.controller import RuleSpec

FAMILY = "own_fast_vs_slow"
_NUM = r"[0-9]+(?:\.[0-9]+)?"
_SET = rf"{_NUM}-{_NUM}-[0-9]+-{_NUM}"
_COND = re.compile(rf"^(?P<fam>{FAMILY})@pmL(?P<L>{_SET})_S(?P<S>{_SET})$")


class Setting(NamedTuple):
    alpha_fast: float
    alpha_slow: float
    W0: int
    delta: float

    def tag(self) -> str:
        return f"{self.alpha_fast:g}-{self.alpha_slow:g}-{self.W0:d}-{self.delta:g}"

    @classmethod
    def parse(cls, text: str) -> "Setting":
        af, as_, w0, d = text.split("-")
        return cls(float(af), float(as_), int(w0), float(d))

    def same(self, other: "Setting") -> bool:
        return self.W0 == other.W0 and all(abs(a - b) <= 1e-9 * max(abs(a), 1e-12) for a, b in zip(self[:2] + self[3:], other[:2] + other[3:]))


def condition(large: Setting, small: Setting) -> str:
    return f"{FAMILY}@pmL{large.tag()}_S{small.tag()}"


def parse_condition(name: str) -> dict[str, Setting] | None:
    """{'large': Setting, 'small': Setting} of a per-member condition, None if `name` is not one."""
    m = _COND.match(name)
    return None if m is None else {"large": Setting.parse(m["L"]), "small": Setting.parse(m["S"])}


def spec_of(setting: Setting, base: RuleSpec) -> RuleSpec:
    """`base` (the screened own_fast_vs_slow rule) with this setting's speeds, warm-up and threshold."""
    return RuleSpec.of(own=dict(kind="fast_vs_slow", delta=setting.delta, alpha_fast=setting.alpha_fast, alpha_slow=setting.alpha_slow),
                       W0=int(setting.W0), K=base.K, alpha_var=base.alpha_var, standardize=base.standardize)


def specs_of(name: str, base: RuleSpec) -> dict[str, RuleSpec] | None:
    """The per-member rules of a per-member condition name (None if `name` is not one)."""
    s = parse_condition(name)
    return None if s is None else {m: spec_of(v, base) for m, v in s.items()}


def setting_of_spec(spec: RuleSpec) -> Setting:
    """The Setting of a screened own_fast_vs_slow RuleSpec (for the shared reference)."""
    p = spec.parts["own"]
    return Setting(float(p["alpha_fast"]), float(p["alpha_slow"]), int(spec.W0), float(p["delta"]))


if __name__ == "__main__":
    a, b = Setting(0.05, 0.005, 20, 0.7), Setting(0.02, 0.005, 100, 0.35)
    name = condition(a, b)
    assert name == "own_fast_vs_slow@pmL0.05-0.005-20-0.7_S0.02-0.005-100-0.35", name
    assert parse_condition(name) == {"large": a, "small": b}
    assert parse_condition("own_fast_vs_slow@0.7") is None and parse_condition("fixed500") is None and parse_condition("own_cusum@pmL1-2-3-4_S1-2-3-4") is None
    base = RuleSpec.of(own=dict(kind="fast_vs_slow", delta=1.0, alpha_fast=0.05, alpha_slow=0.005), W0=20)
    sp = specs_of(name, base)
    assert sp["small"].W0 == 100 and sp["small"].parts["own"]["delta"] == 0.35 and sp["small"].parts["own"]["alpha_fast"] == 0.02 and sp["large"].parts["own"]["delta"] == 0.7
    assert setting_of_spec(sp["small"]) == b and a.same(Setting(0.05, 0.005, 20, 0.7)) and not a.same(b)
    for s in sp.values():
        s.build()                                           # every spec builds its detector
    assert "__" not in name and "/" not in name
    print("member_rules self-test passed")
