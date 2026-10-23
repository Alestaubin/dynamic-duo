"""
continual_tuning.py
===================
Tent's learning rates for the CONTINUAL setting, chosen on held-out shifts only (never on the 15 test shifts), and the loop that runs a
duo over a long stream and stops once a member has completely collapsed. Used by scripts/04b_tune_continual_lr.py (the tuning) and
scripts/04c_gate_stress.py (the gate on a collapsing member).

Why a second tuning. A rate chosen by whole-stream accuracy on 50,000-image episodic streams (scripts/02c) is the largest one that has not
collapsed yet, so it sits at the edge of a cliff: the ResNet-50's 5e-4 survives every held-out stream and dies at the 4th corruption of the
test stream. The continual setting needs its own rates, found on a stream that is as long and as continual as the one it is used on.

The stream (`tuning_parts`). The 4 held-out shifts (speckle noise, gaussian blur, spatter, saturate), severity 5, 50,000 images each,
concatenated and repeated `rounds` times with no reset; round r starts at the r-th shift of the list, so every shift is visited after
every other one. Four rounds are 16 visits = 800,000 images, as long as the test stream (15 x 50,000). The members start from source
weights once. Each round has its own loader seed (seed + 1000 * round), the convention of continual.py.

Criteria, per member and rate (labels are used: these are held-out shifts). The FROZEN member, run over the same batches, is the reference.
  late gain   of a visit: accuracy on the last 20% of the visit minus the frozen member's on the same images (points)
  degraded    some visit has a late gain below -delta (default 2 points: the paired standard error of a 10,000-image window is
              about 0.5). The repo's late-accuracy guard (02c), per visit instead of macro: an adaptation that ends a shift worse than not
              adapting is drift, whatever its mean
  stable      not degraded, in every seed
  dead        a complete collapse: accuracy below 5% for 4 consecutive windows of 5,120 images (chance is 0.1%; members that died in the
              continual runs sit at 0.1 to 0.6%; the hardest shift of a healthy member is above 10%)

Selection (`select_side`; the ladder of each member is half-octave spaced around the rate scripts/02c recorded, and must be extended until
it is bracketed). With the rates ascending:
  first degraded   the lowest rung that is not stable
  frontier         the rung just below it: the largest rate that is stable, which is where an accuracy-maximising rule would stop
  conservative     `margin` rungs below the frontier (default 1): the margin is what the episodic tuning lacked
  stress           the lowest rung from the first degraded one up that is dead: a rate that collapses completely
A warning is raised, and no rate named, where the ladder does not bracket (no rung degraded; the lowest one already is; no rung dead; a
stable rung above a degraded one, i.e. noise).

`run_until` is `continual.run_continual` with two additions: a `stop` object consulted after every batch (`DeadStop`) and `max_images`
(stop after exactly that many images, so that the frozen run can cover the same prefix). A stopped run stores only the visits it made.
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np
import torch

from src.experiments.continual import Part
from src.experiments.joint_dd import PureProxyGate
from src.experiments.members import MemberRunner
from src.experiments.protocols import Segment
from src.experiments.runner import Context
from src.utils.datasets.imagenet_c import ImageNetC

DEFAULT_MULTS = (0.125, 0.18, 0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 2.8, 4)     # of the episodic rate: half-octaves, 3 below to 2 above
DELTA = 2.0                    # points: a visit ending this far below the frozen member is degraded
MARGIN = 1                     # rungs below the frontier
LATE_FRAC = 0.2
DEAD_WINDOW, DEAD_ACC, DEAD_WINDOWS = 5120, 0.05, 4        # 80 batches of 64; below 5% for 4 windows in a row = completely collapsed


# ------------------------------------------------------------------------------------------------------------ the stream

def tuning_parts(ctx: Context, *, rounds: int, seed: int) -> list[Part]:
    """The held-out shifts concatenated `rounds` times, round r rotated left by r (see the module docstring)."""
    g = ctx.g
    held = g.tuning_corruptions
    segments = {c: Segment(f"{c}/s{g.severity}", ImageNetC.from_config(g, c, g.severity), i == 0) for i, c in enumerate(held)}
    parts = []
    for r in range(rounds):
        k = r % len(held)
        parts += [Part(segments[c], r, seed + 1000 * r) for c in held[k:] + held[:k]]
    return parts


# ------------------------------------------------------------------------------------------------------------ collapse

def dead_onset(correct: np.ndarray, window: int = DEAD_WINDOW, acc: float = DEAD_ACC, windows: int = DEAD_WINDOWS) -> int | None:
    """The sample index where the first run of `windows` consecutive non-overlapping windows of `window` samples with accuracy below
    `acc` starts (a complete collapse), None if there is none. Only whole windows count."""
    n = len(correct) // window
    if n < windows:
        return None
    a = correct[:n * window].reshape(n, window).mean(1)
    low = a < acc
    run = np.convolve(low.astype(int), np.ones(windows, dtype=int), mode="valid") == windows
    return int(np.flatnonzero(run)[0]) * window if run.any() else None


class DeadStop:
    """Consulted after every batch: True once the watched members have collapsed completely (`dead_onset`) and `after` more images
    have gone by. `mode` "all": every watched member; "any": the first one that does. `update` takes the batch's per-sample correctness
    of each watched member (a mapping side -> 0/1 array); windows are checked whenever a whole one has been completed."""

    def __init__(self, watch: Sequence[str] = ("large", "small"), mode: str = "all", after: int = 0, *, window: int = DEAD_WINDOW,
                 acc: float = DEAD_ACC, windows: int = DEAD_WINDOWS):
        if mode not in ("all", "any") or not watch:
            raise ValueError(f"mode must be 'all' or 'any' and something must be watched, got {mode!r}, {watch!r}")
        self.watch, self.mode, self.after = tuple(watch), mode, after
        self.window, self.acc, self.windows = window, acc, windows
        self.buf = {s: [] for s in self.watch}
        self.seen, self.next_check = 0, window
        self.dead_at: dict[str, int] = {}          # side -> where its dead streak starts
        self.known_at: dict[str, int] = {}         # side -> the sample count at which the streak was complete
        self.stop_at: int | None = None

    def update(self, correct: Mapping[str, np.ndarray]) -> bool:
        for s in self.watch:
            self.buf[s].append(np.asarray(correct[s], dtype=np.uint8))
        self.seen += len(next(iter(correct.values())))
        if self.seen >= self.next_check:
            self.next_check = (self.seen // self.window + 1) * self.window
            for s in self.watch:
                if s not in self.dead_at:
                    onset = dead_onset(np.concatenate(self.buf[s]), self.window, self.acc, self.windows)
                    if onset is not None:
                        self.dead_at[s], self.known_at[s] = onset, onset + self.windows * self.window
            if self.stop_at is None and self.dead_at:
                if self.mode == "any" or len(self.dead_at) == len(self.watch):
                    pick = min if self.mode == "any" else max
                    self.stop_at = pick(self.known_at.values()) + self.after
        return self.stop_at is not None and self.seen >= self.stop_at


# ------------------------------------------------------------------------------------------------------------ the loop

def run_until(runner: MemberRunner, parts: Sequence[Part], *, images: int, batch_size: int, workers: int, T_l: float, T_s: float,
              gate: PureProxyGate | None = None, stop: DeadStop | None = None, max_images: int | None = None, log_every: int = 4) -> dict:
    """Adapt over `parts` (no resets) like `continual.run_continual` (gate OUTSIDE the loop) and return {"correct": {series: uint8 array},
    "part_n": [images of each visit made, the last possibly partial], "gate_records": [...], "seen": images, "stopped": why or None,
    "dead_at": {side: sample index}}. Stops after the batch on which `stop` says so or `max_images` is reached."""
    correct = {s: [] for s in ("ts", "large", "small") + (("gate",) if gate is not None else ())}
    part_n, seen, stopped = [], 0, None
    n_total = sum(min(images, len(p.segment.dataset)) for p in parts)
    if gate is not None:
        gate.records = []
        gate.set_corruption(parts[0].label, total_samples=n_total)
    t0 = time.time()
    for i, part in enumerate(parts):
        if i and gate is not None:
            gate.mark_segment(part.label)
        n = 0
        for z_l, z_s, y in runner.iter_batches(part.segment.dataset, batch_size=batch_size, workers=workers, seed=part.seed,
                                               num_samples=images):
            y = y.to(z_l.device)
            with torch.no_grad():
                preds = {"ts": (z_l / T_l + z_s / T_s).argmax(1), "large": z_l.argmax(1), "small": z_s.argmax(1)}
                if gate is not None:
                    preds["gate"] = gate.calibrate(z_l, z_s).argmax(1)
            batch = {s: (p == y).to(torch.uint8).cpu() for s, p in preds.items()}
            for s, c in batch.items():
                correct[s].append(c)
            n += len(y)
            seen += len(y)
            if stop is not None and stop.update({s: batch[s].numpy() for s in stop.watch}):
                stopped = f"collapsed completely: {stop.dead_at}"
            elif max_images is not None and seen >= max_images:
                stopped = f"max_images {max_images}"
            if stopped:
                break
        part_n.append(n)
        if log_every and (i + 1) % log_every == 0 or stopped:
            print(f"  {i + 1}/{len(parts)} visits ({part.label}, round {part.round}): {seen} images, "
                  f"{seen / (time.time() - t0):.0f} img/s" + (f"; stopped, {stopped}" if stopped else ""), flush=True)
        if stopped:
            break
    return {"correct": {s: torch.cat(v).numpy() for s, v in correct.items()}, "part_n": part_n,
            "gate_records": list(gate.records) if gate is not None else [], "seen": seen, "stopped": stopped,
            "dead_at": dict(stop.dead_at) if stop is not None else {}}


# ------------------------------------------------------------------------------------------------------------ criteria

def visit_stats(correct: np.ndarray, frozen: np.ndarray, part_n: Sequence[int], late_frac: float = LATE_FRAC) -> dict[str, np.ndarray]:
    """Per visit of a run: accuracy and late accuracy of the member and of the frozen member on the same images (%), and the late gain
    (points). `frozen` may be longer than `correct` (a stopped run): only the images the run saw are compared."""
    edges = np.r_[0, np.cumsum(part_n)]
    if len(correct) != edges[-1] or len(frozen) < edges[-1]:
        raise ValueError(f"the run has {len(correct)} samples for visits of {list(part_n)}, the frozen run {len(frozen)}")
    out = {k: [] for k in ("acc", "late", "frozen_acc", "frozen_late")}
    for a, b in zip(edges[:-1], edges[1:]):
        tail = max(int(round(late_frac * (b - a))), 1)
        out["acc"].append(100 * correct[a:b].mean())
        out["late"].append(100 * correct[b - tail:b].mean())
        out["frozen_acc"].append(100 * frozen[a:b].mean())
        out["frozen_late"].append(100 * frozen[b - tail:b].mean())
    out = {k: np.asarray(v) for k, v in out.items()}
    out["gain"] = out["late"] - out["frozen_late"]
    return out


def summarize(correct: np.ndarray, frozen: np.ndarray, part_n: Sequence[int], labels: Sequence[str], *, delta: float = DELTA,
              window: int = DEAD_WINDOW, acc: float = DEAD_ACC, windows: int = DEAD_WINDOWS) -> dict:
    """One member of one run: whether it is stable / dead and the numbers behind it."""
    st = visit_stats(correct, frozen, part_n)
    worst = int(np.argmin(st["gain"]))
    onset = dead_onset(correct, window, acc, windows)
    return {"stable": bool(st["gain"].min() >= -delta), "dead": onset is not None, "dead_at": onset, "min_gain": float(st["gain"].min()),
            "worst_visit": f"{worst + 1}: {labels[worst]}", "acc": float(correct.mean() * 100), "images": int(len(correct)),
            "visits": len(part_n), "gain": [round(float(v), 2) for v in st["gain"]]}


def select_side(rows: Sequence[Mapping], margin: int = MARGIN) -> dict:
    """Frontier, conservative and stress rate of ONE member from its rungs (see the module docstring). `rows`: dicts with `lr`, `seed`,
    `stable`, `dead`, `acc`, `min_gain`; several seeds of a rate are combined (stable in all, dead in all, mean accuracy, worst gain).
    Returns {"table": [one dict per rate], "frontier", "conservative", "stress" (rates or None), "warnings": [...]}."""
    by = {}
    for r in rows:
        by.setdefault(float(r["lr"]), []).append(r)
    lrs = sorted(by)
    table = [{"lr": lr, "stable": all(r["stable"] for r in by[lr]), "dead": all(r["dead"] for r in by[lr]),
              "any_dead": any(r["dead"] for r in by[lr]), "acc": float(np.mean([r["acc"] for r in by[lr]])),
              "min_gain": float(min(r["min_gain"] for r in by[lr])), "seeds": len(by[lr])} for lr in lrs]
    out = {"table": table, "frontier": None, "conservative": None, "stress": None, "warnings": []}
    bad = [i for i, t in enumerate(table) if not t["stable"]]
    if not bad:
        out["warnings"].append("no rung is degraded: extend the ladder upward")
    elif bad[0] == 0:
        out["warnings"].append("the lowest rung is already degraded: extend the ladder downward")
    else:
        first = bad[0]
        out["frontier"] = lrs[first - 1]
        if any(table[i]["stable"] for i in range(first + 1, len(table))):
            out["warnings"].append(f"a stable rung above the first degraded one ({lrs[first]:g}): not monotone, check the seeds")
        if first - 1 - margin >= 0:
            out["conservative"] = lrs[first - 1 - margin]
        else:
            out["warnings"].append(f"no rung {margin} below the frontier: extend the ladder downward")
        dead = [i for i in range(first, len(table)) if table[i]["dead"]]
        if dead:
            out["stress"] = lrs[dead[0]]
        else:
            out["warnings"].append("no rung above the frontier dies completely: extend the ladder upward")
    return out


# ------------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import logging
    import tempfile
    from pathlib import Path

    from src.experiments import continual as RC
    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.utils.config import load_global

    # ---- criteria on synthetic correctness
    rng = np.random.default_rng(0)
    w = 100
    healthy = (rng.random(20 * w) < 0.5).astype(np.uint8)
    assert dead_onset(healthy, w, 0.05, 4) is None
    broken = healthy.copy()
    broken[7 * w:] = (rng.random(13 * w) < 0.01).astype(np.uint8)
    assert dead_onset(broken, w, 0.05, 4) == 7 * w, dead_onset(broken, w, 0.05, 4)
    assert dead_onset(broken[:10 * w], w, 0.05, 4) is None, "three dead windows are not four"
    assert dead_onset(broken[:11 * w + 50], w, 0.05, 4) == 7 * w, "a trailing partial window is ignored"
    flicker = healthy.copy()
    flicker[3 * w:5 * w] = 0
    flicker[8 * w:10 * w] = 0
    assert dead_onset(flicker, w, 0.05, 4) is None, "dead windows must be consecutive"

    frozen = (rng.random(4 * 1000) < 0.4).astype(np.uint8)
    good = (rng.random(4 * 1000) < 0.55).astype(np.uint8)
    labels = ["a/s5", "b/s5", "c/s5", "d/s5"]
    s = summarize(good, frozen, [1000] * 4, labels)
    assert s["stable"] and not s["dead"] and s["min_gain"] > 5 and s["visits"] == 4 and abs(s["acc"] - 55) < 3
    drift = good.copy()
    drift[2 * 1000:3 * 1000] = (rng.random(1000) < 0.2).astype(np.uint8)
    s = summarize(drift, frozen, [1000] * 4, labels)
    assert not s["stable"] and not s["dead"] and s["worst_visit"] == "3: c/s5" and s["min_gain"] < -10, s
    s = summarize(good[:2500], frozen, [1000, 1000, 500], labels)       # a stopped run is compared on the images it saw
    assert s["visits"] == 3 and s["images"] == 2500
    try:
        visit_stats(good[:2500], frozen, [1000] * 4)
        raise AssertionError("part_n and correctness must agree")
    except ValueError:
        pass

    # ---- selection: a ladder that brackets, then every warning
    def ladder(spec):
        return [{"lr": lr, "seed": 0, "stable": st, "dead": dd, "acc": acc_, "min_gain": 0.0 if st else -9.0} for lr, st, dd, acc_ in spec]

    ok = ladder([(1, True, False, 50), (2, True, False, 52), (3, True, False, 54), (4, True, False, 55), (5, False, False, 40),
                 (6, False, True, 5), (7, False, True, 1)])
    sel = select_side(ok)
    assert (sel["frontier"], sel["conservative"], sel["stress"], sel["warnings"]) == (4.0, 3.0, 6.0, []), sel
    assert select_side(ok, margin=0)["conservative"] == 4.0 and select_side(ok, margin=2)["conservative"] == 2.0
    assert select_side(ladder([(1, True, False, 50), (2, True, False, 52)]))["warnings"] == ["no rung is degraded: extend the ladder upward"]
    assert "lowest rung is already degraded" in select_side(ladder([(1, False, True, 1), (2, False, True, 1)]))["warnings"][0]
    assert select_side(ok[:5])["stress"] is None and "dies completely" in select_side(ok[:5])["warnings"][0]
    assert select_side(ok[:2] + ok[3:], margin=3)["conservative"] is None
    noisy = ladder([(1, True, False, 50), (2, False, False, 40), (3, True, False, 45), (4, False, True, 3)])
    assert "not monotone" in select_side(noisy)["warnings"][0]
    two = ok + [{**r, "seed": 1} for r in ok[:3]] + [{**ok[3], "seed": 1, "stable": False}]
    assert select_side(two)["frontier"] == 3.0, "a rate that is unstable in one seed is unstable"
    assert [t["seeds"] for t in select_side(two)["table"]][:4] == [2, 2, 2, 2]

    # ---- the stream and the loop, on tiny models
    logging.disable(logging.INFO)
    torch.set_num_threads(2)
    g = load_global()
    base = tiny_models(0)
    BS, IMAGES = 32, 100
    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_context(Path(tmp), batch_size=BS, base=base)
        parts = tuning_parts(ctx, rounds=4, seed=7)
        held = g.tuning_corruptions
        labels = [p.label.split("/")[0] for p in parts]
        assert len(parts) == 16 and labels[:4] == held and labels[4:8] == held[1:] + held[:1] and labels[12:] == held[3:] + held[:3]
        assert [p.round for p in parts] == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 and [p.seed for p in parts[::4]] == [7, 1007, 2007, 3007]
        assert all(sum(l == c for l in labels) == 4 for c in held)
        assert {labels[i] for i in range(0, 16, 4)} == set(held), "every shift opens a round once"

        segs = [Segment(f"{c}/s5", FakeStream(150, seed=i + 1), i == 0) for i, c in enumerate(held[:3])]
        small = [Part(s_, r, 3 + 1000 * r) for r in range(2) for s_ in segs]
        T_L, T_S = 1.4, 0.9

        def fresh(mode_key="indep", gated=True):
            gate = PureProxyGate("nnn", beta=2.0, b_t=64, T_l=T_L, T_s=T_S, num_classes=1000) if gated else None
            return RC.make_runner(ctx, RC.RUNS[mode_key], gate)[0], gate

        runner, gate = fresh()
        ref = RC.run_continual(runner, small, images=IMAGES, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, gate=gate, log_every=0)
        runner, gate = fresh()
        full = run_until(runner, small, images=IMAGES, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, gate=gate, log_every=0)
        assert full["stopped"] is None and full["part_n"] == ref["part_n"] and full["seen"] == IMAGES * len(small)
        assert all(np.array_equal(full["correct"][k], ref["correct"][k]) for k in ref["correct"]), "no stop: identical to run_continual"
        assert len(full["gate_records"]) == len(ref["gate_records"])

        cut = 2 * IMAGES + 40                                              # mid-visit, on a batch boundary of the third visit
        runner, gate = fresh()
        pre = run_until(runner, small, images=IMAGES, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, gate=gate, max_images=cut, log_every=0)
        assert pre["seen"] >= cut and pre["seen"] - cut < BS and pre["part_n"][:2] == [IMAGES, IMAGES] and len(pre["part_n"]) == 3
        assert pre["stopped"].startswith("max_images")
        assert all(np.array_equal(pre["correct"][k], ref["correct"][k][:pre["seen"]]) for k in ref["correct"]), "a stopped run is a prefix"

        # a random tiny model is almost never right: both members are 'dead' at once and the run stops after the grace period
        for after, mode in ((0, "all"), (64, "all"), (0, "any")):
            runner, _ = fresh(gated=False)
            stop = DeadStop(("large", "small"), mode, after, window=64, acc=0.5, windows=2)
            res = run_until(runner, small, images=IMAGES, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, stop=stop, log_every=0)
            assert res["stopped"] and res["seen"] < IMAGES * len(small), "the run must stop early"
            if mode == "all":
                assert set(res["dead_at"]) == {"large", "small"} and stop.stop_at == max(stop.known_at.values()) + after
            else:
                assert res["dead_at"] and stop.stop_at == min(stop.known_at.values()) + after
            assert res["seen"] >= stop.stop_at and res["seen"] - stop.stop_at < BS, (res["seen"], stop.stop_at)
            assert all(len(v) == res["seen"] for v in res["correct"].values()) and sum(res["part_n"]) == res["seen"]
        assert DeadStop(("small",), "all", 0).watch == ("small",)
        try:
            DeadStop((), "all")
            raise AssertionError("watching nothing must be refused")
        except ValueError:
            pass

        # a member that never dies never stops the run
        stop = DeadStop(("large",), "all", 0, window=64, acc=0.0, windows=2)             # accuracy below 0% is impossible
        runner, _ = fresh(gated=False)
        res = run_until(runner, small, images=IMAGES, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, stop=stop, log_every=0)
        assert res["stopped"] is None and res["seen"] == IMAGES * len(small)
    print("continual_tuning self-test passed")
