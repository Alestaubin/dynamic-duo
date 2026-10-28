"""
tuning.py
=========
The selection rules of the reset tuning of `own_fast_vs_slow`, as code, so that no choice is made by eye (the protocol in prose: docs/reset_tuning.md).
Every rule is fixed BEFORE the runs and is applied to the stored live runs of scripts/38_run_reset_live.py on the held-out stream.

The unit is a MEMBER. Under `own_fast_vs_slow` (and the no-reset baseline and the fixed schedules) a member's trajectory depends only on its own
learning rate and its own reset times, so one run measures BOTH members at their rates, whatever the partner's rate or setting is, and each member is
tuned alone, on its own accuracy.

What is tuned, per member: the learning rate and the detector setting (alpha_fast, alpha_slow, W0, delta; src/reset/member_rules.py).
What is scored: the whole-stream accuracy of the member, averaged over the tuning rotations (0 and 2; 1 and 3 only validate), from the per-visit
accuracies (16 visits of 50,000 images per rotation).

Rungs: the sqrt-2 grid of the 04b tuner, rate(side, k) = anchor * 2**(k / 2) rounded to 3 digits (ViT anchor 7e-5, ResNet 5e-4). The first ladder
uses every second rung (X2): ViT 3.5e-5 .. 2.8e-4, ResNet 1.25e-4 .. 1e-3. The rules of ONE ladder (`plan_1d`; the threshold uses the same rules on
its own sqrt-2 grid):

  1. Start: the X2 ladder on rotation 0.
  2. Refine: the argmax of rotation 0 must have BOTH its sqrt-2 neighbours tried (they are run if not); the argmax is then recomputed, until
     it has both (so it is a local maximum of what was tried: "interior").
  3. Rotation 2: the argmax and its two neighbours are run on rotation 2.
  4. Choose among those three by the mean over rotations 0 and 2: the best, then every candidate within ONE PAIRED STANDARD ERROR of it
     (paired per-visit differences, visits of both rotations pooled), then the one with the fewest resets per 1,000 proxy batches, then the lowest rate.
  5. Interior: the chosen rung must have a tried rung on each side (strictly inside the range tried on rotation 0); if the tie-break pushed it to an
     end, the rung beyond is run on rotations 0 and 2 and the choice is made again. At most MAX_EXTEND such extensions per side: a plateau that never
     ends (a rate or threshold that does not matter) is reported as such, not chased.
  6. Guard: a configuration in which the member is DEAD (four consecutive 5,120-image windows below 5%, CT.dead_onset) on a tuning rotation is excluded.

`plan_member` is the whole search of one member: a rate ladder for each candidate detector setting, the best candidate, then its threshold and its
rate alternately until neither moves (the rules above at every step). The comparison baselines (no reset, fixed schedules) have only the rate:
`plan_1d` alone.

`plan_*` return a plan with the runs still to run, the choice once it is made, a table and notes: run the requests, call again.
"""

from __future__ import annotations

import glob
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np

SIDES = ("large", "small")
ANCHOR = {"large": 7e-5, "small": 5e-4}
X2 = {"large": (-2, 0, 2, 4), "small": (-4, -2, 0, 2)}          # ViT 3.5e-5, 7e-5, 1.4e-4, 2.8e-4; ResNet 1.25e-4, 2.5e-4, 5e-4, 1e-3
START_MEMBER = {"large": (2, 4, 6, 8), "small": (-6, -4, -2, 0)}      # start rungs of a candidate setting's rate ladder: around the shared optima (ViT 5.6e-4, ResNet 8.84e-5)
TUNING_ROTS, VALIDATION_ROTS = (0, 2), (1, 3)
MAX_EXTEND = 2
DEAD_WINDOW, DEAD_ACC, DEAD_WINDOWS = 5120, 0.05, 4
PROXY_BATCH = 256


def rate(side: str, k: int) -> float:
    return float(f"{ANCHOR[side] * 2 ** (k / 2):.3g}")


def rung(side: str, lr: float) -> int:
    k = round(2 * math.log2(lr / ANCHOR[side]))
    if abs(rate(side, k) / lr - 1) > 0.005:
        raise ValueError(f"{lr:g} is not on the sqrt-2 grid of the {side} member (nearest rung {k}: {rate(side, k):g})")
    return k


def threshold(base: float, k: int) -> float:
    """The sqrt-2 grid of a detection threshold, anchored at the screened value `base`, rounded to a readable step (0.05 below 2, 0.1 below 5, else 0.5)."""
    x = base * 2 ** (k / 2)
    step = 0.05 if x < 2 else 0.1 if x < 5 else 0.5
    return round(round(x / step) * step, 4)


# ------------------------------------------------------------------------------------------------------------ one run

@dataclass
class Cell:
    """What the selection needs of one stored run: per-visit accuracy of each series (fractions), resets per 1,000 proxy batches per member, dead flags."""
    acc: dict[str, np.ndarray]
    resets: dict[str, float]
    dead: dict[str, bool]
    seen: int = 0

    def mean(self, series: str) -> float:
        return float(self.acc[series].mean())


def visit_accuracy(correct: np.ndarray, part_n: Sequence[int]) -> np.ndarray:
    edges = np.cumsum(part_n)
    if edges[-1] != len(correct):
        raise ValueError(f"the visits add up to {edges[-1]} samples, the series has {len(correct)}")
    return np.asarray([c.mean() for c in np.split(np.asarray(correct, dtype=np.float64), edges[:-1])])


def is_dead(correct: np.ndarray) -> bool:
    n = len(correct) // DEAD_WINDOW
    if n < DEAD_WINDOWS:
        return False
    low = correct[:n * DEAD_WINDOW].reshape(n, DEAD_WINDOW).mean(1) < DEAD_ACC
    return bool((np.convolve(low.astype(int), np.ones(DEAD_WINDOWS, dtype=int), mode="valid") == DEAD_WINDOWS).any())


def cell_from_arrays(series: Mapping[str, np.ndarray], part_n: Sequence[int], resets: Sequence[Mapping], seen: int) -> Cell:
    acc = {s: visit_accuracy(v, part_n) for s, v in series.items() if s in ("gate", "ts", "large", "small")}
    n_batches = max(seen / PROXY_BATCH, 1.0)
    per_1000 = {m: 1000.0 * sum(r["member"] == m for r in resets) / n_batches for m in SIDES}
    return Cell(acc, per_1000, {m: is_dead(np.asarray(series[m])) for m in SIDES}, seen)


def load_cell(path: Path) -> Cell:
    from src.experiments import continual as RC          # not at import time: this module's self-test needs no model code
    run = RC.load_run(path)
    meta = run["meta"]
    return cell_from_arrays(run, run["part_n"], meta.get("resets", []), meta.get("seen", int(sum(run["part_n"]))))


# ------------------------------------------------------------------------------------------------------------ the store of runs

_STEM = re.compile(r"^live-(?P<cond>.+?)__heldout(?:__rot(?P<rot>\d))?__L(?P<l>[0-9.e+-]+)_S(?P<s>[0-9.e+-]+)__(?P<gate>.+)__seed(?P<seed>\d+)$")


class Store:
    """The live runs of one duo on the held-out stream at one gate (proxy, b_t, beta): finds them by condition, rotation and rates."""

    def __init__(self, folder: Path, gate: str = "nnn_bt256_beta0.5", seed: int = 0):
        self.folder, self.gate, self.seed = Path(folder), gate, seed
        self._cells: dict[Path, Cell] = {}

    def paths(self, cond: str) -> list[tuple[int, float, float, Path]]:
        out = []
        for f in glob.glob(str(self.folder / f"live-{glob.escape(cond)}__heldout*__{self.gate}__seed{self.seed}.npz")):
            m = _STEM.match(Path(f).stem)
            if m and m["cond"] == cond:
                out.append((int(m["rot"] or 0), float(m["l"]), float(m["s"]), Path(f)))
        return out

    def cell(self, cond: str, rot: int, *, large: float | None = None, small: float | None = None) -> Cell | None:
        """The run of `cond` on rotation `rot` in which the ViT is at `large` and/or the ResNet at `small` (None = any): the partner's rate does not
        matter for a decoupled method, so any such run serves."""
        for r, rl, rs, path in sorted(self.paths(cond)):
            if r == rot and (large is None or math.isclose(rl, large, rel_tol=1e-6)) and (small is None or math.isclose(rs, small, rel_tol=1e-6)):
                if path not in self._cells:
                    self._cells[path] = load_cell(path)
                return self._cells[path]
        return None


class MemberStore(Store):
    """The runs of `own_fast_vs_slow` with per-member settings (condition `own_fast_vs_slow@pmL...-..._S...`): a member's cell for (setting, rate) is ANY run with that
    member at that setting and rate, whatever the partner does (own rule, gate outside the loop). The SHARED setting's cells are the plain `own_fast_vs_slow@<delta>` runs
    (one setting for both members) and every per-member run in which the member happens to have the shared setting."""

    def __init__(self, folder: Path, gate: str, shared, shared_cond: str, seed: int = 0):
        super().__init__(folder, gate, seed)
        self.shared, self.shared_cond = shared, shared_cond

    def member_cell(self, side: str, setting, rot: int, rate: float) -> Cell | None:
        from src.reset import member_rules as MR
        if setting.same(self.shared):
            c = self.cell(self.shared_cond, rot, **{side: rate})
            if c is not None:
                return c
        if tuple(setting[:3]) == tuple(self.shared[:3]):                      # the shared speeds and warm-up at another threshold: the plain `<family>@<delta>` runs (stage 3)
            c = self.cell(f"{MR.FAMILY}@{setting.delta:g}", rot, **{side: rate})
            if c is not None:
                return c
        for f in glob.glob(str(self.folder / f"live-{MR.FAMILY}@pm*__heldout*__{self.gate}__seed{self.seed}.npz")):
            m = _STEM.match(Path(f).stem)
            parsed = MR.parse_condition(m["cond"]) if m else None
            if parsed is None or int(m["rot"] or 0) != rot or not parsed[side].same(setting):
                continue
            if math.isclose(float(m["l" if side == "large" else "s"]), rate, rel_tol=1e-6):
                path = Path(f)
                if path not in self._cells:
                    self._cells[path] = load_cell(path)
                return self._cells[path]
        return None


# ------------------------------------------------------------------------------------------------------------ the rules

def paired_se(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    return float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")


def choose(cands: Mapping[Hashable, tuple[np.ndarray, float]], tie_order: Callable[[Hashable], float]) -> tuple[Hashable, list[Hashable]]:
    """Rule 4. `cands`: key -> (per-visit accuracies pooled over the tuning rotations, resets per 1,000 proxy batches). Returns the chosen key and
    the keys within one paired standard error of the best."""
    best = max(cands, key=lambda k: float(cands[k][0].mean()))
    top = float(cands[best][0].mean())
    tied = [k for k in cands if k == best or top - float(cands[k][0].mean()) <= paired_se(cands[best][0], cands[k][0])]
    return min(tied, key=lambda k: (round(cands[k][1], 6), tie_order(k))), tied


@dataclass
class Plan:
    requests: list[tuple[Hashable, int]] = field(default_factory=list)
    chosen: Hashable | None = None
    table: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.chosen is not None and not self.requests


def plan_1d(*, side: str, start: Sequence[int], neighbours: Callable[[int], tuple[int, int]], cell_of: Callable[[int, int], Cell | None],
            tie_order: Callable[[int], float] = float, rots: Sequence[int] = TUNING_ROTS, extend_cap: int = MAX_EXTEND) -> Plan:
    """Rules 1 to 6 over one ordered ladder of integer rungs, scored by the accuracy of member `side` (whose dead flag is the guard).
    `cell_of(k, rot)`: the stored run (None if not run yet); `neighbours(k)`: (k below, k above) on the finer grid; `tie_order(k)`: the last tie-break."""
    plan = Plan()
    first, second = rots[0], rots[1]

    def alive(c: Cell) -> bool:
        return not c.dead[side]

    def need(k: int, rot: int) -> bool:
        if cell_of(k, rot) is None:
            if (k, rot) not in plan.requests:
                plan.requests.append((k, rot))
            return True
        return False

    if any([need(k, first) for k in start]):
        plan.notes.append("the starting ladder is not complete on rotation 0")
        return plan
    for _ in range(40):                                                          # rule 2: refine until the argmax has both neighbours tried
        tried = sorted({k for k in _all_keys(cell_of, start, neighbours, first)})
        ok = [k for k in tried if (c := cell_of(k, first)) is not None and alive(c)]
        if not ok:
            plan.notes.append("every rung is dead on rotation 0")
            return plan
        best = max(ok, key=lambda k: cell_of(k, first).mean(side))
        lo, hi = neighbours(best)
        # the extension cap of rule 5 bounds this refinement too: a ladder that keeps creeping up by hundredths per rung is a plateau, not chased
        # (the cap was only enforced after a choice until 2026-10-27, when own_cusum's threshold ladder crept below the cap)
        wanted = [k for k in (lo, hi) if min(start) - 2 * extend_cap <= k <= max(start) + 2 * extend_cap]
        if any([need(k, first) for k in wanted]):
            plan.notes.append(f"argmax of rotation {first} is rung {best}; its neighbours {lo} and {hi} are run first")
            return plan
        if max(ok, key=lambda k: cell_of(k, first).mean(side)) == best:
            break
    cand = [k for k in (lo, best, hi) if (c := cell_of(k, first)) is not None and alive(c)]
    if any([need(k, second) for k in cand]):
        plan.notes.append(f"the argmax {best} and its neighbours are run on rotation {second}")
        return plan
    cand = [k for k in cand if alive(cell_of(k, second))]
    pooled = {k: (np.concatenate([cell_of(k, r).acc[side] for r in rots]), float(np.mean([cell_of(k, r).resets[side] for r in rots]))) for k in cand}
    chosen, tied = choose(pooled, tie_order)
    tried = sorted({k for k in _all_keys(cell_of, start, neighbours, first)})
    plan.table = [{"rung": k, "acc": float(pooled[k][0].mean()) if k in pooled else None, "acc_rot%d" % first: cell_of(k, first).mean(side),
                   "resets": pooled[k][1] if k in pooled else None, "tied": k in tied, "chosen": k == chosen} for k in tried if cell_of(k, first) is not None]
    ends = (tried[0], tried[-1])
    if chosen in ends:                                                          # rule 5
        direction = -1 if chosen == ends[0] else 1
        beyond = neighbours(chosen)[0 if direction < 0 else 1]
        n_ext = len([k for k in tried if k < min(start)]) if direction < 0 else len([k for k in tried if k > max(start)])
        if n_ext >= extend_cap * 2:                                              # an extension is one x2 step = two sqrt-2 rungs
            plan.notes.append(f"the choice stays at the {'low' if direction < 0 else 'high'} end after {extend_cap} extensions: a plateau, reported as such")
            plan.chosen = chosen
            return plan
        for r in rots:
            need(beyond, r)
        plan.notes.append(f"chosen rung {chosen} is the {'lowest' if direction < 0 else 'highest'} tried: rung {beyond} is run")
        return plan
    plan.chosen = chosen
    plan.notes.append(f"chosen rung {chosen}; within one paired SE of the best: {tied}")
    return plan


def _all_keys(cell_of, start, neighbours, rot) -> list[int]:
    """Every rung with a stored run on `rot`: the start ladder, then outwards through the neighbours."""
    seen, frontier = set(), list(start)
    while frontier:
        k = frontier.pop()
        if k in seen or cell_of(k, rot) is None:
            continue
        seen.add(k)
        frontier += [n for n in neighbours(k)]
    return sorted(seen)


# ------------------------------------------------------------------------------------------------------------ the search of one member

@dataclass
class MemberPlan:
    requests: list[tuple[object, int, int]] = field(default_factory=list)     # (Setting, rate rung, rotation) still to run
    final: tuple[object, int] | None = None                                   # (Setting, rate rung)
    notes: list[str] = field(default_factory=list)
    table: list[dict] = field(default_factory=list)


def delta_setting(setting, j: int):
    """`setting` with its threshold moved j sqrt-2 rungs (the readable grid of `threshold`)."""
    return setting._replace(delta=threshold(setting.delta, j))


def plan_member(side: str, cell_of: Callable[[object, int, int], Cell | None], shared, candidates: Sequence, *, rots: Sequence[int] = TUNING_ROTS, max_rounds: int = 3) -> MemberPlan:
    """One member's search over (detector setting, learning rate), by the rules of `plan_1d` (scored by the member's own accuracy; `cell_of(setting, rung, rotation)`).
    A. a rate ladder for the shared setting and for every candidate (`START_MEMBER` start rungs); B. the candidate with the best pooled accuracy at its chosen rate
    (`choose`: ties by fewest resets, then the SHARED setting first, then the order of `candidates`); C. its threshold on the sqrt-2 grid at that rate, then, if the
    threshold moved, the rate one rung either side at the new threshold, and again until neither moves (at most `max_rounds`)."""
    plan = MemberPlan()
    settings = [shared, *candidates]

    def add(reqs, setting):
        plan.requests += [(setting, k, r) for k, r in reqs if (setting, k, r) not in plan.requests]

    ladders = {}
    for c in settings:
        start = X2[side] if c is shared else START_MEMBER[side]
        pl = plan_1d(side=side, start=start, neighbours=lambda k: (k - 1, k + 1), rots=rots, cell_of=lambda k, rot, c=c: cell_of(c, k, rot))
        ladders[c] = pl
        add(pl.requests, c)
    if plan.requests:
        plan.notes.append("rate ladders of the candidates are not complete")
        return plan
    dead = [c for c in settings if ladders[c].chosen is None]
    if dead:
        plan.notes.append(f"every rung of {[tuple(c) for c in dead]} is dead on a tuning rotation: dropped")
        settings = [c for c in settings if c not in dead]
    pooled = {c: (np.concatenate([cell_of(c, ladders[c].chosen, r).acc[side] for r in rots]),
                  float(np.mean([cell_of(c, ladders[c].chosen, r).resets[side] for r in rots]))) for c in settings}
    best, tied = choose(pooled, lambda c: settings.index(c))
    plan.table = [{"setting": tuple(c), "rate_rung": ladders[c].chosen, "acc": float(pooled[c][0].mean()), "resets": pooled[c][1], "tied": c in tied, "chosen": c == best} for c in settings]
    plan.notes.append(f"best candidate {tuple(best)} at rung {ladders[best].chosen}; within one paired SE of the best: {[tuple(c) for c in tied]}")
    k = ladders[best].chosen
    for _ in range(max_rounds):
        pd = plan_1d(side=side, start=(-1, 0, 1), neighbours=lambda j: (j - 1, j + 1), rots=rots, tie_order=lambda j: -j,
                     cell_of=lambda j, rot, k=k: cell_of(delta_setting(best, j), k, rot))
        for j, r in pd.requests:
            if (delta_setting(best, j), k, r) not in plan.requests:
                plan.requests.append((delta_setting(best, j), k, r))
        if plan.requests:
            plan.notes.append("threshold ladder not complete")
            return plan
        chosen = delta_setting(best, pd.chosen)
        pr = plan_1d(side=side, start=(k - 1, k, k + 1), neighbours=lambda q: (q - 1, q + 1), rots=rots, cell_of=lambda q, rot, c=chosen: cell_of(c, q, rot))
        add(pr.requests, chosen)
        if plan.requests:
            plan.notes.append("rate recheck at the chosen threshold not complete")
            return plan
        if pr.chosen == k:                                                      # neither moved: the alternating search has converged
            break
        k = pr.chosen                                                           # the rate moved: the threshold is checked again at the new rate
    plan.final = (chosen, pr.chosen)
    plan.notes.append(f"final: setting {tuple(chosen)} at rate rung {pr.chosen}")
    return plan


# ------------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    assert rate("large", -2) == 3.5e-5 and rate("large", 4) == 2.8e-4 and rate("small", -4) == 1.25e-4 and rate("small", 2) == 1e-3 and rate("small", 0) == 5e-4
    assert rung("large", 7e-5) == 0 and rung("small", 1.25e-4) == -4 and threshold(0.7, -2) == 0.35 and threshold(2.0, -2) == 1.0 and [threshold(0.7, k) for k in (-1, 0, 1, 2)] == [0.5, 0.7, 1.0, 1.4] and [threshold(2.0, k) for k in (-1, 1, 3)] == [1.4, 2.8, 5.5]
    try:
        rung("large", 1e-4)
        raise AssertionError("1e-4 is not on the grid")
    except ValueError:
        pass
    # visit accuracies
    c = np.concatenate([np.ones(10), np.zeros(10), np.ones(20)])
    assert np.allclose(visit_accuracy(c, [10, 10, 20]), [1, 0, 1])
    assert is_dead(np.zeros(DEAD_WINDOW * 5)) and not is_dead(np.ones(DEAD_WINDOW * 5)) and not is_dead(np.zeros(100))

    class Ladder:
        """A synthetic ladder: accuracy 0.6 - 0.02 * (k - peak)^2 on every rotation, with visit noise."""
        def __init__(self, peak: int, reset_slope: float = 0.0, noise: float = 0.002):
            self.peak, self.slope, self.noise, self.have = peak, reset_slope, noise, {}

        def run(self, k: int, rot: int) -> None:
            base = 0.6 - 0.02 * (k - self.peak) ** 2
            acc = base + np.random.default_rng(1000 * rot + k + 50).normal(0, self.noise, 16)
            self.have[(k, rot)] = Cell({"large": acc, "gate": acc}, {"large": max(0.0, 5 + self.slope * k), "small": 1.0}, {"large": False, "small": False}, 800000)

        def get(self, k, rot):
            return self.have.get((k, rot))

    def drive(lad: Ladder, **kw) -> Plan:
        for _ in range(30):
            p = plan_1d(side="large", start=X2["large"], neighbours=lambda k: (k - 1, k + 1), cell_of=lad.get, **kw)
            if not p.requests:
                return p
            for k, r in p.requests:
                lad.run(k, r)
        raise AssertionError("did not converge")

    p = drive(Ladder(peak=1))                                                     # a peak between two X2 rungs: needs refinement, ends interior
    assert p.chosen == 1 and p.done, (p.chosen, p.notes)
    p = drive(Ladder(peak=6))                                                     # the peak is above the start ladder: the ladder must extend
    assert p.chosen == 6, (p.chosen, p.notes)
    p = drive(Ladder(peak=-5))
    assert p.chosen == -5, (p.chosen, p.notes)
    # a plateau: accuracy flat above 0 (noise only), fewer resets at a lower rung -> the tie-break takes the rung with fewer resets, and it is interior
    class Flat(Ladder):
        def run(self, k, rot):
            acc = 0.6 + (0 if k >= 0 else -0.05 * (-k)) + np.random.default_rng(1000 * rot + k + 50).normal(0, 0.003, 16)
            self.have[(k, rot)] = Cell({"large": acc, "gate": acc}, {"large": 2.0 + k, "small": 1.0}, {"large": False, "small": False}, 800000)
    p = drive(Flat(0))
    assert p.chosen is not None and p.chosen >= 0, p.notes
    # a ladder that keeps rising towards lower rungs: refinement stops at the extension cap (4 rungs below the start ladder) and a plateau is reported
    class Creep(Ladder):
        def run(self, k, rot):
            acc = 0.6 - 0.001 * k + np.random.default_rng(1000 * rot + k + 50).normal(0, 0.0005, 16)
            self.have[(k, rot)] = Cell({"large": acc, "gate": acc}, {"large": 5.0, "small": 1.0}, {"large": False, "small": False}, 800000)
    lad = Creep(0)
    p = drive(lad)
    assert min(k for k, _ in lad.have) == min(X2["large"]) - 2 * MAX_EXTEND and p.chosen == min(X2["large"]) - 2 * MAX_EXTEND and any("plateau" in n for n in p.notes), (p.chosen, p.notes)
    # dead rungs are excluded
    class Dead(Ladder):
        def run(self, k, rot):
            super().run(k, rot)
            if k >= 3:
                self.have[(k, rot)].dead["large"] = True
    p = drive(Dead(peak=3))
    assert p.chosen is not None and p.chosen <= 2, (p.chosen, p.notes)
    # the per-member search: the shared setting and two candidates; candidate 1 is the best family, its best threshold is one rung above its own and its best rate rung is 5
    from src.reset.member_rules import Setting
    S0, C1, C2 = Setting(0.05, 0.005, 20, 0.7), Setting(0.1, 0.01, 50, 1.4), Setting(0.1, 0.025, 50, 1.4)
    best_delta = delta_setting(C1, 1)
    fam = lambda st: (st.alpha_fast, st.alpha_slow, st.W0)
    base_acc = {fam(S0): 0.540, fam(C1): 0.560, fam(C2): 0.552}
    peak = {fam(S0): 4, fam(C1): 5, fam(C2): 3}
    made: dict = {}

    def member_cell(st, k, rot):
        return made.get((st, k, rot))

    def run_member(st, k, rot):
        dpen = 0.004 * (math.log2(st.delta / best_delta.delta) * 2) ** 2 if fam(st) == fam(C1) else 0.0
        acc = base_acc[fam(st)] - 0.02 * (k - peak[fam(st)]) ** 2 - dpen + np.random.default_rng(10000 + 100 * rot + 7 * k + int(10 * st.delta)).normal(0, 0.0015, 16)
        made[(st, k, rot)] = Cell({"large": acc}, {"large": 5.0, "small": 1.0}, {"large": False, "small": False}, 800000)

    for sh_k in X2["large"] + (5, 6):                      # the shared setting's ladders exist already (the stage-2 runs)
        for rot in (0, 2):
            run_member(S0, sh_k, rot)
    for _ in range(40):
        mp = plan_member("large", member_cell, S0, [C1, C2])
        for st, k, rot in mp.requests:
            assert st in (S0, C1, C2) or fam(st) == fam(C1), f"a request for a setting that is neither a candidate nor a threshold variant of the best: {st}"
            run_member(st, k, rot)
        if not mp.requests:
            break
    else:
        raise AssertionError("plan_member did not converge")
    assert mp.final is not None and fam(mp.final[0]) == fam(C1) and mp.final[0].delta == best_delta.delta and mp.final[1] == 5, (mp.final, mp.notes)
    print("tuning self-test passed")
