"""
live.py
=======
The stream loop of the reset experiments (Phase 4): the two members adapt over ONE long stream without episodic resets, a gate
turns their logits into the duo's prediction, and a controller may reset members between adaptation batches. One loop for

  Vanilla DD   runner mode both_indep (each member on its own entropy); the gate sits OUTSIDE the loop and is called here
  Joint DD     runner mode both_duo with the gate as the runner's `loop_calibrator` (a RecordingCalibrator around it): the gate is
               inside the adaptation loss and has already run when the step returns

    runner = MemberRunner(duo, "tent", {}, "both_indep", device, models=models)
    gate   = PureProxyGate("nnn", beta=0.5, b_t=128, T_l=0.82, T_s=1.04)
    ctl    = ResetController(RuleSpec.of(own=dict(kind="fast_vs_slow", alpha_fast=0.05, alpha_slow=0.005, delta=0.7)))
    res    = run_stream(runner, parts, batch_size=64, workers=8, gate=gate, controller=ctl)

Per adaptation batch: step both members (the returned logits are the PRE-update ones), score the duo, collect the proxy batches the
gate completed during this step (`gate.records`), hand each to the controller, and apply the flagged resets (actions.reset_member)
BEFORE the next step. A proxy batch that completes in the middle of an adaptation batch (the 16-sample tail of a 50,000-image
stream shifts the alignment) therefore acts at most one batch late: the gate's own rule (dd_eval.plan_buckets), not a new one.

The gate is told where streams START (`set_corruption`, once) and, between parts, only the label (`mark_segment`): it keeps its
buffer, so a proxy batch may straddle two corruptions, and nothing marks a shift. Members are reset only by the controller, by
`reset_at_boundaries` (the key equivalence test: forcing a reset at every boundary must reproduce the episodic run), or by
`scripted` (step -> members; a fixed schedule, tests).

Not here (stage R-C): the frozen copies, the per-proxy-batch summaries, checkpoints. `keep_logits` is for short test streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import torch

from src.experiments.protocols import Segment
from src.reset.actions import SIDES, reset_member
from src.reset.controller import ResetController, ResetEvent


@dataclass
class StreamPart:
    segment: Segment
    seed: int                    # the loader seed of this part (a recurring stream reshuffles every round)

    @property
    def label(self) -> str:
        return self.segment.label


@dataclass
class StreamResult:
    events: list[ResetEvent] = field(default_factory=list)       # what the controller flagged
    resets: list[dict] = field(default_factory=list)             # every reset applied: step, position, member, why
    gate_records: list[dict] = field(default_factory=list)
    n_steps: int = 0
    n_samples: int = 0
    z_l: list = field(default_factory=list)                      # keep_logits only (CPU tensors, one per step)
    z_s: list = field(default_factory=list)
    z_duo: list = field(default_factory=list)
    labels: list = field(default_factory=list)


def _gate_of(runner, gate):
    """(the gate object whose `records` we read, whether it runs inside the adaptation loss)."""
    if runner.loop_calibrator is not None:
        inner = getattr(runner.loop_calibrator, "inner", runner.loop_calibrator)
        if gate is not None and gate is not inner:
            raise ValueError("in a *_duo mode the gate is the runner's loop_calibrator; pass that one (or none)")
        return inner, True
    return gate, False


def run_stream(runner, parts: Sequence[StreamPart], *, batch_size: int, workers: int = 0, num_samples: int | None = None,
               gate=None, controller: ResetController | None = None, reset_at_boundaries: Iterable[str] = (),
               scripted: Mapping[int, Iterable[str]] | None = None,
               boundary_hook: Callable[[int, StreamPart], None] | None = None, keep_logits: bool = False) -> StreamResult:
    if not hasattr(runner, "iter_batches"):
        raise TypeError("run_stream drives a MemberRunner (Tent); the ported algorithms need a loader of their own")
    gate_obj, inside = _gate_of(runner, gate)
    has_records = gate_obj is not None and hasattr(gate_obj, "records")
    if controller is not None and not has_records:
        raise ValueError("a controller needs a gate that records its proxy batches (PureProxyGate)")
    boundary_sides, scripted = tuple(reset_at_boundaries), {int(k): tuple(v) for k, v in (scripted or {}).items()}
    res = StreamResult()
    seen_records = 0
    if has_records:
        gate_obj.records = []
        total = sum(min(num_samples, len(p.segment.dataset)) if num_samples else len(p.segment.dataset) for p in parts)
        gate_obj.set_corruption(parts[0].label, total_samples=total)

    def apply(sides: Iterable[str], why: str) -> None:
        for side in sorted(set(sides), key=SIDES.index):
            reset_member(runner, side)
            res.resets.append({"step": res.n_steps, "position": res.n_samples, "member": side, "why": why})

    for i, part in enumerate(parts):
        if i > 0:
            apply(boundary_sides, "boundary")                       # before the part's first step: the loader is built lazily
            if has_records:
                gate_obj.mark_segment(part.label)
        if boundary_hook is not None:
            boundary_hook(i, part)
        for z_l, z_s, y in runner.iter_batches(part.segment.dataset, batch_size=batch_size, workers=workers, seed=part.seed,
                                               num_samples=num_samples):
            res.n_steps += 1
            res.n_samples += len(y)
            if gate_obj is not None and not inside:
                with torch.no_grad():
                    z_duo = gate_obj.calibrate(z_l, z_s)
            else:
                z_duo = getattr(runner.loop_calibrator, "last", None) if inside else None
            if keep_logits:
                res.z_l.append(z_l.cpu()); res.z_s.append(z_s.cpu()); res.labels.append(y)
                res.z_duo.append(None if z_duo is None else z_duo.detach().cpu())
            if controller is not None:
                new = gate_obj.records[seen_records:]
                seen_records = len(gate_obj.records)
                for rec in new:
                    for e in controller.update(rec["r_l"], rec["r_s"], position=res.n_samples):
                        res.events.append(e)
                        apply([e.member], e.rule)
            elif has_records:
                seen_records = len(gate_obj.records)
            if res.n_steps - 1 in scripted:
                apply(scripted[res.n_steps - 1], "scripted")
    if has_records:
        res.gate_records = list(gate_obj.records)
    return res


# ---------------------------------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    import copy

    from src.calibrators.joint_fixed_TS import JointFixedTS
    from src.experiments._testing import DUO, FakeStream, tiny_models
    from src.experiments.joint_dd import PureProxyGate, RecordingCalibrator
    from src.experiments.members import MemberRunner
    from src.reset.actions import member_state, states_equal
    from src.reset.controller import RuleSpec

    import logging

    logging.disable(logging.INFO)                          # the TTA method logs every setup
    torch.set_num_threads(2)
    torch.manual_seed(0)
    base = tiny_models(0)
    BS, B_T, SEED = 32, 64, 3
    segments = [Segment(f"c{i}/s5", FakeStream(150, seed=sd), i == 0) for i, sd in enumerate((1, 2, 3))]
    N = 150
    T_L, T_S = 1.4, 0.9
    parts = [StreamPart(s, SEED) for s in segments]

    def make(mode: str):
        """(runner, gate): Vanilla DD (gate outside) or Joint DD (the gate inside, as the loop calibrator)."""
        gate = PureProxyGate("nnn", beta=2.0, b_t=B_T, T_l=T_L, T_s=T_S, num_classes=1000)
        if mode == "vanilla":
            return MemberRunner(DUO, "tent", {}, "both_indep", "cpu", models=copy.deepcopy(base), bn=False), gate
        rec = RecordingCalibrator(gate)
        return MemberRunner(DUO, "tent", {}, "both_duo", "cpu", models=copy.deepcopy(base), loop_calibrator=rec, bn=False), gate

    def same(a: StreamResult, b: StreamResult, duo: bool = True) -> bool:
        ok = len(a.z_l) == len(b.z_l) > 0 and all(x.equal(y) for x, y in zip(a.z_l, b.z_l)) and all(x.equal(y) for x, y in zip(a.z_s, b.z_s))
        return ok and (not duo or all(x.equal(y) for x, y in zip(a.z_duo, b.z_duo)))

    # the weak rule that fires on every batch after its warm-up: a threshold far below any statistic, on the raw s of each member
    always = RuleSpec.of(own=dict(kind="fast_vs_slow", alpha_fast=0.2, alpha_slow=0.02, delta=-1e9), W0=2, K=1, standardize=False)
    never = RuleSpec.of(own=dict(kind="fast_vs_slow", alpha_fast=0.2, alpha_slow=0.02, delta=float("inf")), W0=2)

    for mode in ("vanilla", "joint"):
        print(f"  {mode} DD")
        # ---- test 1: a rule that never fires reproduces the no-reset run exactly -------------------------------------------
        runner, gate = make(mode)
        plain = run_stream(runner, parts, batch_size=BS, gate=gate, keep_logits=True)
        runner, gate = make(mode)
        ctl = ResetController(never)
        with_ctl = run_stream(runner, parts, batch_size=BS, gate=gate, controller=ctl, keep_logits=True)
        assert same(plain, with_ctl) and with_ctl.events == [] and with_ctl.resets == []
        n_rec = len(with_ctl.gate_records)
        assert n_rec == ctl.t == -(-3 * N // B_T) and n_rec > 4, "the controller must have seen every proxy batch"
        assert with_ctl.n_steps == 3 * len(range(0, N, BS)) and with_ctl.n_samples == 3 * N
        # ... and the loop itself adds nothing: the same logits as MemberRunner.iter_batches over the three streams
        runner, gate = make(mode)
        raw = [(a.clone(), b.clone()) for seg in segments for a, b, _ in runner.iter_batches(seg.dataset, batch_size=BS, workers=0, seed=SEED, num_samples=None)]
        assert all(a.equal(x) and b.equal(y) for (a, b), x, y in zip(raw, plain.z_l, plain.z_s)) and len(raw) == len(plain.z_l)
        assert any(abs(r["w_l"] - 0.5) > 1e-3 for r in plain.gate_records), "the gate must actually move"

        # ---- test 3 (key): a forced reset of each member at every boundary of a continual stream = the episodic run ---------
        runner_e, gate_e = make(mode)
        epi = StreamResult()
        gate_e.records = []
        for seg in segments:                                                    # the episodic loop of runner.ensure_members / run_joint_dd
            runner_e.reset()
            gate_e.set_corruption(seg.label, total_samples=N)
            for z_l, z_s, y in runner_e.iter_batches(seg.dataset, batch_size=BS, workers=0, seed=SEED, num_samples=None):
                z_duo = runner_e.loop_calibrator.last if mode == "joint" else gate_e.calibrate(z_l, z_s)
                epi.z_l.append(z_l.cpu()); epi.z_s.append(z_s.cpu()); epi.z_duo.append(z_duo.detach().cpu())
        runner_c, gate_c = make(mode)
        cont = run_stream(runner_c, [StreamPart(Segment(s.label, s.dataset, False), SEED) for s in segments], batch_size=BS, gate=gate_c,
                          reset_at_boundaries=SIDES, keep_logits=True,
                          boundary_hook=lambda i, p: gate_c.set_corruption(p.label, total_samples=N))      # the episodic run's per-stream gate
        assert same(epi, cont), "forced resets at the boundaries must reproduce the episodic run (members AND the duo's prediction)"
        assert [(r["step"], r["member"], r["why"]) for r in cont.resets] == [(s, m, "boundary") for s in (5, 10) for m in SIDES]
        # without the forced resets (continual) the run is NOT the episodic one: the test has teeth
        runner_n, gate_n = make(mode)
        nores = run_stream(runner_n, [StreamPart(Segment(s.label, s.dataset, False), SEED) for s in segments], batch_size=BS, gate=gate_n,
                           keep_logits=True)
        assert not same(epi, nores, duo=False)
        k = len(range(0, N, BS))
        assert all(a.equal(b) for a, b in zip(epi.z_l[:k], nores.z_l[:k])), "the first segment starts from source either way"
        # resetting only the large member at the boundaries is not the episodic run either (the small member carries over)
        runner_h, gate_h = make(mode)
        half = run_stream(runner_h, [StreamPart(Segment(s.label, s.dataset, False), SEED) for s in segments], batch_size=BS, gate=gate_h,
                          reset_at_boundaries=("large",), keep_logits=True)
        assert not same(epi, half, duo=False)
        print("    forced boundary resets reproduce the episodic run bit for bit; one member only or none does not")

        # ---- a controller-triggered reset is exactly the scripted reset at the same step ----------------------------------------
        runner, gate = make(mode)
        ctl = ResetController({"large": always, "small": never})
        trig = run_stream(runner, parts, batch_size=BS, gate=gate, controller=ctl, keep_logits=True)
        assert trig.events and {e.member for e in trig.events} == {"large"} and all(r["why"] == "own_fast_vs_slow" for r in trig.resets)
        assert [e.position for e in trig.events] == sorted(e.position for e in trig.events) and trig.events[0].position > 0
        sched = {}
        for r in trig.resets:
            sched.setdefault(r["step"] - 1, []).append(r["member"])     # a flag is applied after step r["step"], i.e. index step - 1
        runner, gate = make(mode)
        scr = run_stream(runner, parts, batch_size=BS, gate=gate, scripted=sched, keep_logits=True)
        assert same(trig, scr), "the controller's resets are exactly the scripted ones"
        assert [(r["step"], r["member"]) for r in trig.resets] == [(r["step"], r["member"]) for r in scr.resets]
        assert not same(trig, plain, duo=False), "and they changed the trajectory"
        # both members flagged on the same batch -> both reset, each as if alone (the actions test proves the independence)
        runner, gate = make(mode)
        ctl = ResetController(always)
        both = run_stream(runner, parts, batch_size=BS, gate=gate, controller=ctl, keep_logits=True)
        firsts = [e for e in both.events if e.t == both.events[0].t]
        assert {e.member for e in firsts} == set(SIDES), "both members fire on the same proxy batch"

    # ---- guards ---------------------------------------------------------------------------------------------------------------
    runner, gate = make("joint")
    try:
        run_stream(runner, parts, batch_size=BS, gate=PureProxyGate("nnn", beta=1.0, b_t=B_T, T_l=1.0, T_s=1.0))
        raise AssertionError("a gate other than the loop calibrator must be refused")
    except ValueError:
        pass
    static = MemberRunner(DUO, "tent", {}, "both_duo", "cpu", models=copy.deepcopy(base),
                          loop_calibrator=JointFixedTS(Tl=T_L, Ts=T_S, verbose=False), bn=False)
    try:
        run_stream(static, parts, batch_size=BS, controller=ResetController(never))
        raise AssertionError("a controller needs proxy batches")
    except ValueError:
        pass
    print("live self-test passed")
