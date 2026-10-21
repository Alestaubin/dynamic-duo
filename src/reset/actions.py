"""
actions.py
==========
The reset action on the models: put ONE member back to its source state, exactly, and leave the other member alone (Phase 4,
step 1 and 2 of the action of controller.py; steps 3 and 4 are the controller's).

    reset_member(runner, "large")           # the large member's adapted parameters -> source, its optimizer state -> empty
    reset_members(runner, ["large", "small"])   # each exactly as if it had been reset alone (the two never interact)

What "source" is. `TentMethod._setup_side` snapshots each adapting member's model state dict and optimizer state dict (a deepcopy)
BEFORE the first update, so the snapshot is the pretrained weights and an empty optimizer (SGD creates its momentum buffer at the
first step, Adam its moments; an empty state IS zero buffers, and the next update is a fresh run's first update). Restoring is
`load_state_dict` on that member's own module and its own optimizer, so the other member's parameters, optimizer state and
gradients are never read or written. Nothing else is touched: not the data loader, not the gate's buffer (the stream goes on), not
the controller's detectors (the controller restarts those).

Runners: `MemberRunner` (Tent, the engine's own path) through `TentMethod._reset_side`, which is per side already; a `PairRunner`
(ETA / EATA / SAR) through `members[side].reset()`, which is per member already. A member that does not adapt in the runner's mode
has nothing to reset and is refused. No existing file is edited (design note, D8).
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch

SIDES = ("large", "small")


def reset_member(runner, side: str) -> None:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if hasattr(runner, "members"):                          # PairRunner: one independent member object per side
        member = runner.members[side]
        if not getattr(member, "adapt", True):
            raise ValueError(f"the {side} member does not adapt: nothing to reset")
        member.reset()
        return
    tta = runner.tta                                        # MemberRunner: the TTA method keeps per-side snapshots
    if side not in tta.adapting:
        raise ValueError(f"the {side} member does not adapt in mode {runner.mode!r}: nothing to reset")
    tta._reset_side(side)


def reset_members(runner, sides: Iterable[str]) -> None:
    for side in sorted(set(sides), key=SIDES.index):         # a fixed order; the result does not depend on it (tested)
        reset_member(runner, side)


def member_state(runner, side: str) -> dict:
    """A deep copy of one member's model state and optimizer state (what the equivalence tests compare)."""
    tta = runner.tta
    return {"model": {k: v.detach().clone() for k, v in tta.models[side].state_dict().items()},
            "optimizer": copy.deepcopy(tta._optimizers[side].state_dict())}


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def states_equal(a: dict, b: dict) -> bool:
    """Bitwise equality of two `member_state`s."""
    return _same(a, b)


def optimizer_is_fresh(state: dict) -> bool:
    """No momentum / moment buffers, or only zero ones."""
    return all(not isinstance(v, torch.Tensor) or not v.any() for slot in state["state"].values() for v in slot.values()
               if isinstance(v, torch.Tensor) and v.dim() > 0)


# ---------------------------------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    from src.experiments._testing import DUO, FakeStream, tiny_models
    from src.experiments.members import MemberRunner
    from src.experiments.runner import loop_calibrator
    from src.experiments.specs import RunSpec

    import logging

    logging.disable(logging.INFO)                          # the TTA method logs every setup
    torch.set_num_threads(2)
    ds = FakeStream(64, seed=1)
    base = tiny_models(0)

    def new_runner(mode: str) -> MemberRunner:
        cal = loop_calibrator(RunSpec("tiny", mode=mode, loop_ts=(1.4, 0.9) if mode.endswith("_duo") else None), torch.device("cpu"))
        return MemberRunner(DUO, "tent", {}, mode, "cpu", models=copy.deepcopy(base), loop_calibrator=cal, bn=False)

    def batch(seed: int):
        g = torch.Generator().manual_seed(seed)
        return torch.rand(32, 3, 8, 8, generator=g), torch.rand(32, 3, 8, 8, generator=g)

    def advance(r: MemberRunner, n: int = 6, first: int = 0):
        out = None
        for i in range(n):
            out = r.step_tensors(*batch(first + i), ds)
        return out

    for mode in ("both_indep", "both_duo"):
        print(f"  mode {mode}")
        fresh = new_runner(mode)
        source = {s: member_state(fresh, s) for s in SIDES}
        assert all(optimizer_is_fresh(source[s]["optimizer"]) for s in SIDES) and not source["large"]["optimizer"]["state"]
        runner = new_runner(mode)
        advance(runner)
        moved = {s: member_state(runner, s) for s in SIDES}
        for s in SIDES:
            assert not states_equal(moved[s], source[s]), f"{s} must have adapted"
            assert not optimizer_is_fresh(moved[s]["optimizer"]), f"{s}: the optimizer must hold moments after adapting"

        # ---- test 2: after a reset the member's parameters equal the source exactly, its optimizer buffers are zero ----------
        a = copy.deepcopy(runner)
        reset_member(a, "large")
        now = member_state(a, "large")
        assert states_equal(now["model"], source["large"]["model"]), "parameters must equal the source bit for bit"
        assert optimizer_is_fresh(now["optimizer"]) and not now["optimizer"]["state"], "optimizer buffers must be gone"
        # ... and the member behaves as a fresh one: its next step is a fresh run's first step (logits and updated parameters)
        nxt = new_runner(mode)
        if mode == "both_indep":              # in the joint mode the partner's logits enter the loss: compare after matching them
            za, zb = a.step_tensors(*batch(99), ds), nxt.step_tensors(*batch(99), ds)
            assert za[0].equal(zb[0]), "a reset member's logits on the next batch are a fresh member's"
            assert states_equal(member_state(a, "large"), member_state(nxt, "large")), "and so is its first update"
        else:
            b = copy.deepcopy(runner)
            reset_members(b, SIDES)
            za, zb = b.step_tensors(*batch(99), ds), nxt.step_tensors(*batch(99), ds)
            assert za[0].equal(zb[0]) and za[1].equal(zb[1]), "both members reset: the joint step is a fresh run's first step"
            assert all(states_equal(member_state(b, s), member_state(nxt, s)) for s in SIDES)

        # ---- test 4: resetting one member leaves the other's parameters and optimizer state untouched -----------------------
        for who, other in (("large", "small"), ("small", "large")):
            a = copy.deepcopy(runner)
            before = member_state(a, other)
            grads = [None if p.grad is None else p.grad.clone() for p in a.tta.models[other].parameters()]
            reset_member(a, who)
            assert states_equal(member_state(a, other), before), f"resetting {who} changed {other}"
            assert states_equal(member_state(a, who), source[who]) and not states_equal(before, source[other])
            assert all((g is None) == (p.grad is None) for g, p in zip(grads, a.tta.models[other].parameters()))
            # the partner goes on exactly as if nothing had happened to the other one (indep mode: its step is its own)
            if mode == "both_indep":
                ref = copy.deepcopy(runner)
                zr, za2 = ref.step_tensors(*batch(77), ds), a.step_tensors(*batch(77), ds)
                i = SIDES.index(other)
                assert zr[i].equal(za2[i]) and states_equal(member_state(ref, other), member_state(a, other))

        # ---- test 5: both members reset on the same proxy batch: each ends as after its own reset alone -------------------
        only_l, only_s = copy.deepcopy(runner), copy.deepcopy(runner)
        reset_member(only_l, "large")
        reset_member(only_s, "small")
        for order in (("large", "small"), ("small", "large")):
            both = copy.deepcopy(runner)
            for s in order:
                reset_member(both, s)
            assert states_equal(member_state(both, "large"), member_state(only_l, "large")), order
            assert states_equal(member_state(both, "small"), member_state(only_s, "small")), order
        both = copy.deepcopy(runner)
        reset_members(both, ["small", "large"])
        assert all(states_equal(member_state(both, s), source[s]) for s in SIDES)

    # ---- a member that does not adapt cannot be reset; a PairRunner member is reset through its own reset() ------------------
    frozen_small = MemberRunner(DUO, "tent", {}, "large_indep", "cpu", models=copy.deepcopy(base), bn=False)
    advance(frozen_small, 2)
    try:
        reset_member(frozen_small, "small")
        raise AssertionError("a frozen member has nothing to reset")
    except ValueError:
        pass
    reset_member(frozen_small, "large")

    class FakeMember:
        def __init__(self, adapt=True):
            self.adapt, self.resets = adapt, 0

        def reset(self):
            self.resets += 1

    class FakePair:
        members = {"large": FakeMember(), "small": FakeMember()}

    reset_member(FakePair, "small")
    assert FakePair.members["small"].resets == 1 and FakePair.members["large"].resets == 0
    try:
        reset_member(frozen_small, "middle")
        raise AssertionError("unknown side")
    except ValueError:
        pass
    print("actions self-test passed")
