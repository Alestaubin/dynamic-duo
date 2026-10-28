"""
live_runs.py
============
The live reset runs of scripts/38_run_reset_live.py (Phase 4, stage R-E): Vanilla DD (both members adapt on their own entropy, the gate outside the
loop) over one long stream, a TRIGGER that decides when a member goes back to source, and per-sample correctness stored in the format of
continual.save_run (so the continual report and the gate-stress plots read it). Same loop as `src.reset.live.run_stream`, but it keeps the
per-sample correctness of the four series (ts = the fixed-TS duo, gate = Vanilla DD, large, small) instead of logits.

    runner, gate = ..., PureProxyGate(...)
    res = run_live(runner, parts, trigger=RuleTrigger(spec), images=50000, batch_size=64, workers=8, T_l=0.65, T_s=1.01, gate=gate)
    res["correct"]  {series: uint8 array}    res["resets"]  [{step, position, member, why}]

Triggers (what decides a reset; every reset is the full reset of actions.reset_member, the stream goes on):
  NoTrigger        never: the no-reset run
  RuleTrigger      a ResetController (src/reset/controller.py) fed with the gate's proxy batches: `own_fast_vs_slow` (label-free)
  ScheduleTrigger  both members every `period` adaptation steps (RDumb's fixed schedule uses 1,000); label-free and blind
  RandomTrigger    the control for detection: the reset counts of a reference run, at random steps
  BoundaryTrigger  both members at every shift boundary, the gate too (set_corruption): the episodic protocol run as one stream. It is told where
                   the shifts are, which no rule is: a reference, not a method
(The staggered schedule and the oracle trigger, controls of rules that were dropped, are in git at the tag `pre-reset-cleanup`.)
A flag is applied after the adaptation batch that completed the proxy batch, before the next one (the gate's own rule). The gate is not reset
when a member is (design note, 1.2).

`extra_gates` ({name: gate}, src/reset/gate_variants.py) score the same logits with other gates, outside the loop like the main one: their predictions are stored as the
series `gate__<name>`, and a gate with a `notify_reset(member, position)` is told of every reset. They change nothing in the run.
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np
import torch

from src.experiments.members import PairCollate
from src.reset.actions import SIDES, reset_member
from src.reset.controller import ResetController, RuleSpec


# ------------------------------------------------------------------------------------------------------------------ triggers

class Trigger:
    """What decides a reset. `start_part(i)`: flags before part i's first step (i >= 1); `after_batch`: flags after an adaptation batch, given the
    gate's NEW proxy batches (`records`), the stream position (samples seen) and the batch's per-sample correctness (uint8 arrays: ts, large,
    small, gate). A flag is (member, why)."""
    name = "none"
    new_gate_per_part = False          # the gate starts afresh at every part (the episodic protocol)

    def start_part(self, i: int) -> list[tuple[str, str]]:
        return []

    def after_batch(self, *, step: int, position: int, records: Sequence[Mapping], correct: Mapping[str, np.ndarray]) -> list[tuple[str, str]]:
        return []

    def summary(self) -> dict:
        return {"trigger": self.name}


class NoTrigger(Trigger):
    name = "none"


class RuleTrigger(Trigger):
    def __init__(self, spec: RuleSpec | Mapping[str, RuleSpec], name: str = "rule"):
        """`spec`: one rule for both members, or {"large": RuleSpec, "small": RuleSpec} (src/reset/member_rules.py: per-member settings)."""
        self.spec, self.name, self.ctl = spec, name, ResetController(spec)

    def after_batch(self, *, step, position, records, correct):
        out = []
        for rec in records:
            out += [(e.member, e.rule) for e in self.ctl.update(rec["r_l"], rec["r_s"], position=position)]
        return out

    def summary(self) -> dict:
        spec = {m: sp.to_json() for m, sp in self.spec.items()} if isinstance(self.spec, Mapping) else self.spec.to_json()
        return {"trigger": self.name, "spec": spec, "events": [e.to_json() for e in self.ctl.events]}


class ScheduleTrigger(Trigger):
    def __init__(self, period: int):
        if period < 1:
            raise ValueError(f"period must be at least 1, got {period}")
        self.period, self.name = int(period), f"fixed{int(period)}"

    def after_batch(self, *, step, position, records, correct):
        return [(s, self.name) for s in SIDES] if step % self.period == 0 else []

    def summary(self) -> dict:
        return {"trigger": self.name, "period": self.period}


class RandomTrigger(Trigger):
    """The control for detection: each member is reset the same number of times as in a reference run (`counts`, resets per member) but at RANDOM
    adaptation steps, uniform over the stream, at least `min_gap` steps apart (a rule's cooldown is K proxy batches) and none in the first `min_gap`
    steps (the warm-up). `draw` seeds the choice. If it does not beat the rule it copies, the rule provides reset frequency, not detection."""

    def __init__(self, counts: Mapping[str, int], total_steps: int, *, draw: int = 0, min_gap: int = 80, name: str | None = None):
        self.counts, self.total_steps, self.draw, self.min_gap = dict(counts), int(total_steps), int(draw), int(min_gap)
        self.name = name or f"random{draw}"
        rng = np.random.default_rng(10_007 + self.draw)
        self.at: dict[str, set[int]] = {}
        for side in SIDES:
            n = int(self.counts.get(side, 0))
            span = self.total_steps - self.min_gap * (n + 1)
            if n and span < 0:
                raise ValueError(f"{n} resets of the {side} member with a gap of {self.min_gap} do not fit in {self.total_steps} steps")
            # n sorted draws from a shrunken range, then spread by the gap: they are distinct and at least min_gap apart (the standard trick for spaced samples)
            raw = np.sort(rng.integers(0, span + 1, size=n)) if n else np.zeros(0, dtype=int)
            self.at[side] = {int(x) + self.min_gap * (i + 1) for i, x in enumerate(raw)}

    def after_batch(self, *, step, position, records, correct):
        return [(s, self.name) for s in SIDES if step in self.at[s]]

    def summary(self) -> dict:
        return {"trigger": self.name, "counts": self.counts, "draw": self.draw, "min_gap": self.min_gap, "steps": {s: sorted(v) for s, v in self.at.items()}}


class BoundaryTrigger(Trigger):
    name, new_gate_per_part = "boundary", True

    def start_part(self, i):
        return [(s, "boundary") for s in SIDES]


# ------------------------------------------------------------------------------------------------------------------ the loop

def _batches(runner, dataset, *, batch_size: int, workers: int, seed: int, num_samples: int | None):
    """What MemberRunner.iter_batches loads, without stepping: (x_large, x_small, labels)."""
    return dataset.loader(batch_size=batch_size, num_samples=num_samples, seed=seed, workers=workers, pin_memory=runner.device.type == "cuda",
                          collate_fn=PairCollate(dataset.input_transform(runner.large_preprocess), dataset.input_transform(runner.small_preprocess)))


def run_live(runner, parts, *, trigger: Trigger, images: int, batch_size: int, workers: int, T_l: float, T_s: float, gate,
             max_images: int | None = None, log_every: int = 1000, log=print, extra_gates: Mapping[str, object] | None = None) -> dict:
    """Adapt over `parts` (continual.Part: segment, round, seed) with `trigger` deciding the resets.
    `extra_gates`: more gates to score on the same logits (module text); their series are `gate__<name>`; they must be called after `gate` (followers read its cache).
    Returns {"correct": {series: uint8 array}, "part_n", "gate_records", "seen", "stopped", "resets", "trigger"}."""
    if not hasattr(gate, "records"):
        raise ValueError("a live run needs a gate that records its proxy batches (PureProxyGate)")
    extra = dict(extra_gates or {})
    names = ["ts", "large", "small", "gate"] + [f"gate__{k}" for k in extra]
    correct = {s: [] for s in names}
    resets, part_n, seen, step, seen_rec, stopped = [], [], 0, 0, 0, None
    n_total = sum(min(images, len(p.segment.dataset)) for p in parts)
    gate.records = []
    # the gate is told where the stream starts and how long it is; the episodic protocol (boundary trigger) gives every part its own gate, so the first one too
    first_total = min(images, len(parts[0].segment.dataset)) if trigger.new_gate_per_part else n_total
    for g in (gate, *extra.values()):
        g.set_corruption(parts[0].label, total_samples=first_total)

    def apply(flags) -> None:
        done = set()
        for side, why in flags:
            if side in done:
                continue
            done.add(side)
            reset_member(runner, side)
            resets.append({"step": step, "position": seen, "member": side, "why": why})
            for g in extra.values():
                if hasattr(g, "notify_reset"):
                    g.notify_reset(side, seen)

    t0 = time.time()
    for i, part in enumerate(parts):
        ds = part.segment.dataset
        if i:
            for g in (gate, *extra.values()):
                if trigger.new_gate_per_part:
                    g.set_corruption(part.label, total_samples=min(images, len(ds)))
                else:
                    g.mark_segment(part.label)                   # between parts only the label: the buffer stays
            apply(trigger.start_part(i))
        n = 0
        for x_l, x_s, y in _batches(runner, ds, batch_size=batch_size, workers=workers, seed=part.seed, num_samples=images):
            z_l, z_s = runner.step_tensors(x_l, x_s, ds)         # adapts; the logits are the PRE-update ones
            y = y.to(z_l.device)
            with torch.no_grad():
                preds = {"ts": (z_l / T_l + z_s / T_s).argmax(1), "large": z_l.argmax(1), "small": z_s.argmax(1), "gate": gate.calibrate(z_l, z_s).argmax(1)}
                preds.update({f"gate__{k}": g.calibrate(z_l, z_s).argmax(1) for k, g in extra.items()})
            batch = {s: (p == y).to(torch.uint8).cpu() for s, p in preds.items()}
            for s, c in batch.items():
                correct[s].append(c)
            step += 1
            n += len(y)
            seen += len(y)
            new, seen_rec = gate.records[seen_rec:], len(gate.records)
            apply(trigger.after_batch(step=step, position=seen, records=new, correct={s: c.numpy() for s, c in batch.items()}))
            if log_every and step % log_every == 0:
                log(f"  step {step}: {seen} images ({part.label}, visit {i + 1}/{len(parts)}), {seen / (time.time() - t0):.0f} img/s, {len(resets)} resets", flush=True)
            if max_images is not None and seen >= max_images:
                stopped = f"max_images {max_images}"
                break
        part_n.append(n)
        if stopped:
            break
    return {"correct": {s: torch.cat(v).numpy() for s, v in correct.items()}, "part_n": part_n, "gate_records": list(gate.records), "seen": seen,
            "stopped": stopped, "resets": resets, "trigger": trigger.summary()}


# ------------------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import copy
    import logging

    from src.experiments import continual_tuning as CT
    from src.experiments._testing import DUO, FakeStream, tiny_models
    from src.experiments.continual import Part
    from src.experiments.joint_dd import PureProxyGate
    from src.experiments.members import MemberRunner
    from src.experiments.protocols import Segment
    from src.reset.live import StreamPart, run_stream

    logging.disable(logging.INFO)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    base = tiny_models(0)
    BS, B_T, SEED, N = 32, 64, 3, 150
    T_L, T_S = 1.4, 0.9
    segments = [Segment(f"c{i}/s5", FakeStream(N, seed=sd), i == 0) for i, sd in enumerate((1, 2, 3))]
    parts = [Part(s, 0, SEED) for s in segments]
    lparts = [StreamPart(s, SEED) for s in segments]
    ALWAYS = RuleSpec.of(own=dict(kind="fast_vs_slow", alpha_fast=0.2, alpha_slow=0.02, delta=-1e9), W0=2, K=1, standardize=False)    # fires after every warm-up
    NEVER = RuleSpec.of(own=dict(kind="fast_vs_slow", alpha_fast=0.2, alpha_slow=0.02, delta=float("inf")), W0=2)

    def make(mode: str = "no_adapt"):
        gate = PureProxyGate("nnn", beta=2.0, b_t=B_T, T_l=T_L, T_s=T_S, num_classes=1000)
        return MemberRunner(DUO, "tent", {}, "both_indep", "cpu", models=copy.deepcopy(base), bn=False), gate

    def live(trigger, **kw):
        runner, gate = make()
        return run_live(runner, parts, trigger=trigger, images=N, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, gate=gate, log_every=0, **kw)

    def reference(**kw):
        """run_stream's correctness of the four series (logits -> argmax == label), the self-tested loop this one mirrors."""
        runner, gate = make()
        res = run_stream(runner, lparts, batch_size=BS, gate=gate, keep_logits=True, num_samples=N, **kw)
        z_l, z_s, z_d, y = (torch.cat(v) for v in (res.z_l, res.z_s, res.z_duo, res.labels))
        return res, {"ts": ((z_l / T_L + z_s / T_S).argmax(1) == y).numpy(), "large": (z_l.argmax(1) == y).numpy(), "small": (z_s.argmax(1) == y).numpy(),
                     "gate": (z_d.argmax(1) == y).numpy()}

    def proxies(records) -> list:
        return [(r["r_l"], r["r_s"], r["w_l"]) for r in records]

    def same(res, ref, ref_res) -> bool:
        """Predictions AND the gate's proxy values (a random tiny model is almost never right, so correctness alone says little; the proxy values
        follow the logits exactly)."""
        return all(np.array_equal(res["correct"][s], ref[s].astype(np.uint8)) for s in ref) and proxies(res["gate_records"]) == proxies(ref_res.gate_records)

    k = len(range(0, N, BS))                                     # steps per part

    # ---- no trigger = run_until (the 04c loop) and run_stream: the loop adds and drops nothing ------------------------------------------
    plain = live(NoTrigger())
    runner, gate = make()
    until = CT.run_until(runner, parts, images=N, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, gate=gate, log_every=0)
    assert all(np.array_equal(plain["correct"][s], until["correct"][s]) for s in until["correct"]) and plain["part_n"] == until["part_n"] == [N] * 3
    rs_plain, ref_plain = reference()
    assert same(plain, ref_plain, rs_plain) and plain["resets"] == [] and plain["seen"] == 3 * N
    assert len(plain["gate_records"]) == -(-3 * N // B_T)
    print("  no trigger: the same predictions as run_until and run_stream")

    # ---- a rule that never fires, and one that always does, against run_stream's controller ----------------------------------------------
    assert same(live(RuleTrigger(NEVER)), ref_plain, rs_plain)
    rs, ref_rule = reference(controller=ResetController(ALWAYS))
    lv = live(RuleTrigger(ALWAYS))
    assert same(lv, ref_rule, rs) and lv["resets"] and [(r["step"], r["member"]) for r in lv["resets"]] == [(r["step"], r["member"]) for r in rs.resets]
    assert not same(lv, ref_plain, rs_plain), "the resets changed the trajectory"
    print(f"  rule trigger: {len(lv['resets'])} resets at the controller's steps, the same predictions as run_stream")

    # ---- the fixed schedule = run_stream's scripted resets; the boundary trigger = the episodic protocol ------------------------------------
    period = 3
    sched = live(ScheduleTrigger(period))
    rs_sched, ref_sched = reference(scripted={s - 1: SIDES for s in range(period, 3 * k + 1, period)})
    assert same(sched, ref_sched, rs_sched) and [r["step"] for r in sched["resets"]] == [s for s in range(period, 3 * k + 1, period) for _ in SIDES]
    assert {r["why"] for r in sched["resets"]} == {"fixed3"}
    bound = live(BoundaryTrigger())
    runner, gate = make()
    cont = run_stream(runner, lparts, batch_size=BS, gate=gate, reset_at_boundaries=SIDES, keep_logits=True, num_samples=N,
                      boundary_hook=lambda i, p: gate.set_corruption(p.label, total_samples=N))
    z_l, z_s, z_d, y = (torch.cat(v) for v in (cont.z_l, cont.z_s, cont.z_duo, cont.labels))
    ref_bound = {"ts": ((z_l / T_L + z_s / T_S).argmax(1) == y).numpy(), "large": (z_l.argmax(1) == y).numpy(), "small": (z_s.argmax(1) == y).numpy(), "gate": (z_d.argmax(1) == y).numpy()}
    assert same(bound, ref_bound, cont) and [(r["step"], r["member"]) for r in bound["resets"]] == [(s, m) for s in (k, 2 * k) for m in SIDES]
    print("  fixed schedule and boundary resets: the same predictions as run_stream's scripted and boundary resets (episodic protocol)")

    # ---- extra gates: scored on the same logits, change nothing, are told every reset at the stream position it acts from ------------------------------
    from src.reset import gate_variants as GV

    class Probe:
        """A stand-in gate: counts the samples it is given and writes down what notify_reset tells it."""
        def __init__(self):
            self.n, self.told, self.corruptions = 0, [], []

        def set_corruption(self, label, total_samples=None):
            self.corruptions.append(label)

        mark_segment = set_corruption

        def calibrate(self, z_l, z_s):
            self.n += z_l.shape[0]
            return z_l

        def notify_reset(self, member, position):
            self.told.append((member, position, self.n))

    def live_extra(trigger, names, **kw):
        runner, gate = make()
        extra = {n: GV.make_gate(n, gate, T_l=T_L, T_s=T_S, b_t=B_T, num_classes=1000) for n in names}
        probe = Probe()
        extra["probe"] = probe
        return run_live(runner, parts, trigger=trigger, images=N, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, gate=gate, extra_gates=extra, log_every=0, **kw), probe

    plain_s = live(ScheduleTrigger(4))
    both, probe = live_extra(ScheduleTrigger(4), ("beta0", "beta2", "beta8", "route0.2w96", "agree_duo@1"))
    assert all(np.array_equal(both["correct"][s], plain_s["correct"][s]) for s in plain_s["correct"]) and both["resets"] == plain_s["resets"], "extra gates must change nothing"
    assert set(both["correct"]) == set(plain_s["correct"]) | {f"gate__{n}" for n in ("beta0", "beta2", "beta8", "route0.2w96", "agree_duo@1", "probe")}
    assert np.array_equal(both["correct"]["gate__beta0"], both["correct"]["ts"]), "beta 0 is the fixed-TS duo"
    assert np.array_equal(both["correct"]["gate__beta2"], both["correct"]["gate"]), "the follower at the leader's beta (2) is the leader"
    assert both["resets"] and [(m, p) for m, p, _ in probe.told] == [(r["member"], r["position"]) for r in both["resets"]] and all(p == n for _, p, n in probe.told), probe.told
    assert probe.n == both["seen"] and probe.corruptions == [p.label for p in parts]
    print(f"  extra gates: the run is unchanged, beta0 = fixed TS, the follower at the leader's beta = the leader, {len(probe.told)} resets told at the position they act from")

    # ---- max_images ------------------------------------------------------------------------------------------------------------------------
    cut = live(NoTrigger(), max_images=70)
    assert cut["seen"] == 96 and cut["stopped"] == "max_images 70" and np.array_equal(cut["correct"]["gate"], plain["correct"]["gate"][:96])

    # ---- the control for detection: random resets with the counts of a reference run ------------------------------------------------------------
    rnd = RandomTrigger({"large": 7, "small": 12}, 12_000, draw=3, min_gap=80)
    for side, n in (("large", 7), ("small", 12)):
        ts_ = sorted(rnd.at[side])
        assert len(ts_) == n and ts_[0] >= 80 and ts_[-1] <= 12_000 and all(b - a >= 80 for a, b in zip(ts_, ts_[1:])), (side, ts_)
    assert RandomTrigger({"large": 7, "small": 12}, 12_000, draw=3).at == rnd.at and RandomTrigger({"large": 7, "small": 12}, 12_000, draw=4).at != rnd.at
    hits = [(st, f) for st in range(1, 12_001) for f in rnd.after_batch(step=st, position=0, records=[], correct={})]
    assert len(hits) == 19 and {w for _, (_, w) in hits} == {"random3"}
    try:
        RandomTrigger({"large": 200}, 1000, min_gap=80)
        raise AssertionError("200 resets cannot fit")
    except ValueError:
        pass
    print("  random-matched trigger: counts, spacing, reproducibility")
    print("live_runs self-test passed")
