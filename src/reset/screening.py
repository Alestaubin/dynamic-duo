"""
screening.py
============
Offline screening of the reset rules (Phase 4, stage R-D): every rule is REPLAYED over the stored no-reset proxy sequences of the
continual runs (src/experiments/continual.py: the gate's `.gate.json`, one record per proxy batch, and the per-sample correctness),
to find which rule detects a member's collapse soon and rarely cries wolf, before any live run.

    streams = [ScreenStream.load(run_npz, frozen_npz, b_t=256, name="heldout0/indep"), ...]
    row = evaluate(spec, streams)                    # delay, miss rate, false alarms per 1,000 proxy batches of one rule
    rows = sweep(streams)                            # the grid of `families()` crossed with their threshold sweeps
    best = select(rows, budget=1.0)                  # per family: the lowest delay at <= 1 false alarm per 1,000 batches

**Degradation labels** (evaluation only; they never reach a rule). Member m is DEGRADED at proxy batch t if its accuracy minus its
FROZEN accuracy on the same samples, smoothed over a centred window of M batches, is below -margin (M = 8, margin = 1 point). A
degraded STRETCH lasting at least L batches (16) is a collapse; its first batch is the ONSET. `label_sensitivity` repeats a rule's score
for other (M, margin, L).

**Replay protocol.** The controller of src/reset/controller.py is fed the proxy sequence; on each alarm it restarts the detectors and
starts the cooldown exactly as the live rule does, but the TRAJECTORY IS NOT CHANGED: the reset model cannot be simulated offline, so
after an alarm the stored (collapsed) sequence goes on. A documented approximation: a rule's later alarms in the same stretch are
neither delays nor false alarms.

**Metrics**, pooled over the (stream, member) pairs: per stretch the DELAY is the batches from its onset to the first alarm inside it
(an alarm up to `tolerance` = 8 batches BEFORE the onset counts as a detection with delay 0); a stretch with no alarm is a MISS and is
charged its whole length as delay (`delay`; `delay_detected` averages the detected ones only); FALSE ALARMS are the alarms outside
every stretch (less the tolerance before each onset), per 1,000 proxy batches.

**Matched budget.** `families()` lists the own / gap / both (gap OR own with the same parameters) versions of the five procedures, each with
at most 12 configurations of its non-threshold parameters, and a threshold sweep. `select` takes, per family and false-alarm budget,
the (configuration, threshold) with the lowest delay (then fewest misses, then fewest false alarms); the caller picks on the tuning
(held-out) streams, freezes it and runs `evaluate` on the test streams.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from src.reset.controller import MEMBERS, ResetController, RuleSpec

LABEL = (8, 0.01, 16)            # M, margin (accuracy, 0.01 = 1 point), L
TOLERANCE = 8                    # batches before an onset in which an alarm is neither false nor late
BUDGETS = (0.5, 1.0, 2.0)        # false alarms per 1,000 proxy batches


def per_batch(correct: np.ndarray, b_t: int, n_batches: int) -> np.ndarray:
    """Accuracy over each proxy batch (b_t consecutive samples from the stream's start; the last may be shorter)."""
    edges = np.minimum(np.arange(n_batches + 1) * b_t, len(correct))
    cs = np.concatenate([[0], np.cumsum(correct, dtype=np.int64)])
    return (cs[edges[1:]] - cs[edges[:-1]]) / np.maximum(np.diff(edges), 1)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over `window` values (a shorter window at the ends)."""
    k = np.ones(window)
    return np.convolve(x, k, "same") / np.convolve(np.ones_like(x), k, "same")


def degraded_stretches(acc: np.ndarray, frozen: np.ndarray, M: int = LABEL[0], margin: float = LABEL[1], L: int = LABEL[2]) -> list[tuple[int, int]]:
    """[(onset, end)) of the stretches of at least L proxy batches in which smoothed (acc - frozen) < -margin."""
    low = np.concatenate([[False], smooth(acc - frozen, M) < -margin, [False]])
    change = np.flatnonzero(low[1:] != low[:-1])
    return [(int(a), int(b)) for a, b in zip(change[::2], change[1::2]) if b - a >= L]


@dataclass
class ScreenStream:
    """One no-reset run: the proxy values of its proxy batches and each member's adapted and frozen accuracy over the same batches."""
    name: str
    r_l: np.ndarray
    r_s: np.ndarray
    acc: dict[str, np.ndarray]
    frozen: dict[str, np.ndarray]
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.r_l)

    @classmethod
    def load(cls, run: Path, frozen: Path, b_t: int, name: str) -> "ScreenStream":
        """`run`: a continual run's .npz (its .gate.json beside it); `frozen`: the frozen run of the same stream."""
        records = json.loads(Path(run).with_suffix(".gate.json").read_text())
        d, f = np.load(run, allow_pickle=False), np.load(frozen, allow_pickle=False)
        n = len(records)
        return cls(name, np.array([r["r_l"] for r in records]), np.array([r["r_s"] for r in records]),
                   {m: per_batch(d[k], b_t, n) for m, k in zip(MEMBERS, ("large", "small"))},
                   {m: per_batch(f[k], b_t, n) for m, k in zip(MEMBERS, ("large", "small"))})

    def stretches(self, member: str, M: int = LABEL[0], margin: float = LABEL[1], L: int = LABEL[2]) -> list[tuple[int, int]]:
        key = (member, M, margin, L)
        if key not in self._cache:
            self._cache[key] = degraded_stretches(self.acc[member], self.frozen[member], M, margin, L)
        return self._cache[key]


def replay(spec: RuleSpec, stream: ScreenStream, c: float = 0.0) -> dict[str, list[int]]:
    """The proxy batches at which `spec` (one rule for both members) flags each member, trajectory unchanged (see the module text)."""
    ctl = ResetController(spec, c=c)
    for r_l, r_s in zip(stream.r_l, stream.r_s):
        ctl.update(float(r_l), float(r_s))
    alarms = {m: [] for m in MEMBERS}
    for e in ctl.events:
        alarms[e.member].append(e.t)
    return alarms


def score_member(alarms: Sequence[int], stretches: Sequence[tuple[int, int]], n: int, tolerance: int = TOLERANCE) -> dict:
    """Delays (a miss is charged its stretch's length), misses and false alarms of one member's alarms against its stretches."""
    delays, detected, misses = [], [], 0
    allowed = np.zeros(n, dtype=bool)
    for a, b in stretches:
        allowed[max(a - tolerance, 0):b] = True
        hit = [t for t in alarms if a - tolerance <= t < b]
        if hit:
            detected.append(max(hit[0] - a, 0))
            delays.append(detected[-1])
        else:
            misses += 1
            delays.append(b - a)
    return {"events": len(stretches), "misses": misses, "delays": delays, "detected": detected,
            "false": int(sum(not allowed[t] for t in alarms))}


def evaluate(spec: RuleSpec, streams: Sequence[ScreenStream], label: tuple = LABEL, tolerance: int = TOLERANCE) -> dict:
    """The pooled score of one rule over `streams`: delay, delay_detected (batches), miss_rate, fa (per 1,000 proxy batches)."""
    events = misses = false = batches = 0
    delays, detected = [], []
    for st in streams:
        alarms = replay(spec, st)
        for m in MEMBERS:
            s = score_member(alarms[m], st.stretches(m, *label), st.n, tolerance)
            events, misses, false, batches = events + s["events"], misses + s["misses"], false + s["false"], batches + st.n
            delays += s["delays"]
            detected += s["detected"]
    nan = float("nan")
    return {"events": events, "misses": misses, "false": false, "batches": batches, "fa": 1000.0 * false / max(batches, 1),
            "miss_rate": misses / events if events else nan, "delay": float(np.mean(delays)) if delays else nan,
            "delay_detected": float(np.mean(detected)) if detected else nan}


# ----------------------------------------------------------------------------------------------------------- the grid

# kind -> (non-threshold parameter grid, the threshold's name and sweep). The five procedures of src/reset/detectors.py; W0 and K = W0 are
# the controller's (RuleSpec). "level" alarms below tau (negative), the others above their threshold in noise units.
W0S = (20, 50)
KINDS = {
    "level": ({}, "tau", (-1.0, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0)),
    "imm_vs_ema": ({"alpha": (0.01, 0.05, 0.1)}, "delta", (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)),
    "fast_vs_slow": ({"speed": ((0.05, 0.005), (0.1, 0.01), (0.2, 0.02))}, "delta", (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)),
    "slope": ({"W": (16, 32, 64), "k": (1, 4)}, "gamma", (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)),
    "cusum": ({"nu": (0.25, 0.5, 1.0)}, "h", (2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)),
}
QUANTITIES = {"own": ("own",), "gap": ("gap",), "both": ("gap", "own")}      # both: gap OR own, same parameters


def part_params(kind: str, config: dict, threshold_name: str, threshold: float) -> dict:
    p = {"kind": kind, threshold_name: threshold}
    for k, v in config.items():
        if k == "speed":
            p["alpha_fast"], p["alpha_slow"] = v
        else:
            p[k] = v
    return p


def configs(kind: str) -> list[dict]:
    """Every configuration of `kind`'s non-threshold parameters, crossed with the warm-up W0 (at most 12)."""
    grid = KINDS[kind][0]
    out = [dict(zip(grid, combo), W0=w0) for w0 in W0S for combo in itertools.product(*grid.values())]
    assert len(out) <= 12, (kind, len(out))
    return out


def families() -> list[str]:
    return [f"{q}_{k}" for q in QUANTITIES for k in KINDS]


def candidates(family: str) -> Iterator[tuple[dict, float, RuleSpec]]:
    """(configuration, threshold, rule) for every point of `family`'s grid."""
    quantity, kind = family.split("_", 1)
    _, thr_name, thresholds = KINDS[kind]
    for config in configs(kind):
        config = dict(config)
        w0 = config.pop("W0")
        for thr in thresholds:
            parts = {q: part_params(kind, config, thr_name, thr) for q in QUANTITIES[quantity]}
            yield dict(config, W0=w0), thr, RuleSpec.of(**parts, W0=w0)


def sweep(streams: Sequence[ScreenStream], fams: Sequence[str] | None = None, label: tuple = LABEL) -> list[dict]:
    """One row per grid point: family, configuration, threshold, the rule (json) and its pooled score."""
    rows = []
    for family in fams or families():
        for config, thr, spec in candidates(family):
            rows.append({"family": family, "config": config, "threshold": thr, "spec": spec.to_json(), **evaluate(spec, streams, label)})
    return rows


def _key(row: dict) -> tuple:
    nan = float("inf")
    return (row["delay"] if row["delay"] == row["delay"] else nan, row["miss_rate"] if row["miss_rate"] == row["miss_rate"] else nan, row["fa"])


def select(rows: Sequence[dict], budget: float) -> dict[str, dict]:
    """Per family, the row with the lowest delay among those with fa <= budget (None where no point meets the budget)."""
    out: dict[str, dict | None] = {}
    for family in sorted({r["family"] for r in rows}):
        ok = [r for r in rows if r["family"] == family and r["fa"] <= budget and r["events"] > 0]
        out[family] = min(ok, key=_key) if ok else None
    return out


def curve(rows: Sequence[dict], family: str) -> list[tuple[float, float]]:
    """The delay-against-false-alarm frontier of a family: (fa, lowest delay at or below that fa), over all configurations and thresholds."""
    pts = sorted((r["fa"], r["delay"]) for r in rows if r["family"] == family and r["delay"] == r["delay"])
    best, out = float("inf"), []
    for fa, d in pts:
        if d < best:
            best = d
            out.append((fa, d))
    return out


def label_sensitivity(spec: RuleSpec, streams: Sequence[ScreenStream],
                      settings: Sequence[tuple] = ((4, 0.01, 16), (16, 0.01, 16), (8, 0.005, 16), (8, 0.02, 16), (8, 0.01, 8), (8, 0.01, 32))) -> list[dict]:
    """The rule's score under other degradation labels (M, margin, L), one at a time around the default."""
    return [{"M": M, "margin": mg, "L": L, **evaluate(spec, streams, (M, mg, L))} for M, mg, L in (LABEL, *settings)]


if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)

    # per_batch: the last proxy batch may be shorter
    c = np.array([1, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
    assert np.allclose(per_batch(c, 3, 3), [2 / 3, 2 / 3, 0.0])
    # smooth keeps a constant constant, and a step is centred
    assert np.allclose(smooth(np.full(10, 0.3), 4), 0.3) and np.allclose(smooth(np.r_[np.zeros(5), np.ones(5)], 3)[3:7], [0, 1 / 3, 2 / 3, 1])
    # degraded stretches: a drop of the adapted accuracy below the frozen one for 40 batches, and a 6-batch dip that is too short
    n = 400
    frozen = np.full(n, 0.5)
    acc = frozen + 0.01 * rng.standard_normal(n) * 0.1
    acc[200:240] = 0.05
    acc[100:106] = 0.05
    st = degraded_stretches(acc, frozen)
    assert len(st) == 1 and 195 <= st[0][0] <= 205 and 236 <= st[0][1] <= 245, st
    assert len(degraded_stretches(acc, frozen, L=4)) == 2, "a shorter L sees the dip too"
    assert degraded_stretches(frozen, frozen) == []

    # score_member: a detection, a late one, a miss charged its length, false alarms outside the tolerance
    s = score_member([195, 300], [(200, 240)], 400)
    assert s["detected"] == [0] and s["false"] == 1 and s["misses"] == 0, s          # 195 is within 8 batches before the onset
    s = score_member([212, 220], [(200, 240)], 400)
    assert s["detected"] == [12] and s["false"] == 0
    s = score_member([], [(200, 240), (300, 320)], 400)
    assert s["misses"] == 2 and s["delays"] == [40, 20] and s["detected"] == []
    assert score_member([10, 11], [], 400)["false"] == 2

    # a synthetic stream: the large member is healthy, the small collapses at batch 200 (accuracy to 0.05, its proxy to 0.2)
    def stream(name: str, collapse_at: int | None, seed: int) -> ScreenStream:
        g = np.random.default_rng(seed)
        r_l = np.clip(0.7 + 0.03 * g.standard_normal(n), 0.01, 0.99)
        r_s = np.clip(0.6 + 0.03 * g.standard_normal(n), 0.01, 0.99)
        acc_s = np.clip(0.4 + 0.02 * g.standard_normal(n), 0, 1)
        if collapse_at is not None:
            ramp = np.clip((np.arange(n) - collapse_at) / 10, 0, 1)
            r_s = r_s * (1 - ramp) + 0.2 * ramp
            acc_s = acc_s * (1 - ramp) + 0.05 * ramp
        fz = {"large": np.full(n, 0.5), "small": np.full(n, 0.4)}
        return ScreenStream(name, r_l, r_s, {"large": np.clip(0.5 + 0.01 * g.standard_normal(n), 0, 1), "small": acc_s}, fz)

    streams = [stream("a", 200, 1), stream("b", 150, 2), stream("calm", None, 3)]
    assert [len(x.stretches("small")) for x in streams] == [1, 1, 0] and all(len(x.stretches("large")) == 0 for x in streams)
    cusum = RuleSpec.of(own=dict(kind="cusum", nu=0.5, h=8.0), W0=20)
    row = evaluate(cusum, streams)
    assert row["events"] == 2 and row["misses"] == 0 and row["fa"] <= 1.0 and 0 <= row["delay"] <= 15, row
    assert evaluate(RuleSpec.of(own=dict(kind="cusum", nu=0.5, h=1e9), W0=20), streams)["miss_rate"] == 1.0, "an unreachable threshold misses"
    flood = evaluate(RuleSpec.of(own=dict(kind="level", tau=0.0), W0=20), streams)
    assert flood["fa"] > 20, "a level threshold at the warm-up mean alarms as often as warm-up and cooldown (about 20 batches each) allow"

    # the grid: five procedures x own / gap / both, at most 12 configurations each, and every rule builds
    assert len(families()) == 15
    for fam in families():
        pts = list(candidates(fam))
        kind = fam.split("_", 1)[1]
        assert len(pts) == len(configs(kind)) * len(KINDS[kind][2]) and len(configs(kind)) <= 12
        pts[0][2].build()
    assert len(configs("slope")) == 12 and len(configs("level")) == 2
    rows = sweep(streams, ["own_cusum", "gap_cusum", "both_cusum", "own_level"])
    assert {r["family"] for r in rows} == {"own_cusum", "gap_cusum", "both_cusum", "own_level"}
    best = select(rows, 1.0)
    assert best["own_cusum"] is not None and best["own_cusum"]["fa"] <= 1.0
    assert best["own_cusum"]["delay"] == min(r["delay"] for r in rows if r["family"] == "own_cusum" and r["fa"] <= 1.0)
    assert select(rows, -1.0)["own_cusum"] is None, "no point meets an impossible budget"
    fr = curve(rows, "own_cusum")
    assert all(fr[i][0] < fr[i + 1][0] and fr[i][1] > fr[i + 1][1] for i in range(len(fr) - 1)), "a frontier strictly improves with more false alarms"
    sens = label_sensitivity(cusum, streams)
    assert len(sens) == 7 and sens[0]["M"] == 8
    json.dumps(rows[:3])                                    # the rows are plain JSON

    # a stored run round-trips: ScreenStream.load reads the .npz and .gate.json the continual runs write
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        b_t, nb = 4, 6
        correct = rng.integers(0, 2, nb * b_t).astype(np.uint8)
        np.savez(tmp / "run.npz", large=correct, small=correct, ts=correct, gate=correct)
        np.savez(tmp / "fz.npz", large=correct, small=1 - correct)
        (tmp / "run.gate.json").write_text(json.dumps([{"r_l": 0.5 + 0.01 * i, "r_s": 0.4} for i in range(nb)]))
        ls = ScreenStream.load(tmp / "run.npz", tmp / "fz.npz", b_t, "t")
        assert ls.n == nb and np.allclose(ls.acc["large"], ls.frozen["large"]) and np.allclose(ls.acc["small"] + ls.frozen["small"], 1.0)
    print("screening self-test passed")
