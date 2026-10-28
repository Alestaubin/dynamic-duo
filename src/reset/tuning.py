"""
tuning.py
=========
The selection rules of the reset tuning of `own_fast_vs_slow`, as code, so that no choice is made by eye (the protocol in prose: docs/reset_tuning.md).
Every rule is fixed BEFORE the runs and is applied to the stored live runs (src/reset/workers.py) on the held-out stream.

The unit is a MEMBER. Under `own_fast_vs_slow` (and the no-reset baseline and the fixed schedules) a member's trajectory depends only on its own
learning rate and its own reset times, so one run measures BOTH members at their rates, whatever the partner's rate or setting is, and each member is
tuned alone, on its own accuracy.

What is tuned, per member: the learning rate and the detector setting (alpha_fast, alpha_slow, W0, delta; src/reset/member_rules.py).
What is scored: the whole-stream accuracy of the member on the tuning rotation (rotation 0), from the per-visit accuracies (16 visits of 50,000 images). The
search runs on ONE rotation: on the main duo's stored ladders it picked the rung the pooled rotations 0 and 2 picked in 5 of 6 cases (the miss cost 0.22 points), and the
four rotations of a chosen configuration differ by under 0.1 points (they hold the same images in another order). Two rotations can still be pooled (`rots`).

Rungs: the sqrt-2 grid of the 04b tuner, rate(side, k) = anchor * 2**(k / 2) rounded to 3 digits, the anchor being the member's episodic rate
(`Grid`; the main duo: ViT 7e-5, ResNet 5e-4). The first ladder uses every second rung (X2: 0.5x .. 4x the anchor for the large member, 0.25x .. 2x for the
small one; ViT 3.5e-5 .. 2.8e-4, ResNet 1.25e-4 .. 1e-3). The rules of ONE ladder (`plan_1d`; the threshold uses the same rules on
its own sqrt-2 grid):

  1. Start: the X2 ladder on rotation 0.
  2. Refine: the argmax of rotation 0 must have BOTH its sqrt-2 neighbours tried (they are run if not); the argmax is then recomputed, until
     it has both (so it is a local maximum of what was tried: "interior").
  3. (Only with two tuning rotations: the argmax and its two neighbours are run on the second.)
  4. Choose among the argmax and its two neighbours by the mean over the tuning rotation(s): the best, then every candidate within ONE PAIRED STANDARD ERROR of it
     (paired per-visit differences), then the one with the fewest resets per 1,000 proxy batches, then the lowest rate.
  5. Interior: the chosen rung must have a tried rung on each side (strictly inside the range tried); if the tie-break pushed it to an
     end, the rung beyond is run and the choice is made again. At most MAX_EXTEND such extensions per side: a plateau that never
     ends (a rate or threshold that does not matter) is reported as such, not chased.
  6. Guard: a configuration in which the member is DEAD (four consecutive 5,120-image windows below 5%, CT.dead_onset) on the tuning rotation is excluded.

`plan_member` is the whole search of one member: a rate ladder for each candidate detector setting (the shared setting's first, the candidates' start
rungs are then centred on its chosen rung), the best candidate, then its threshold and its rate alternately until neither moves (the rules above at
every step). The comparison baselines (no reset, fixed schedules) have only the rate:
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
TUNING_ROTS = (0,)                  # the held-out rotation(s) the search runs on: one by default; two (0, 2) pool their visits (the main duo's earlier protocol)
MAX_EXTEND = 2
DEAD_WINDOW, DEAD_ACC, DEAD_WINDOWS = 5120, 0.05, 4
PROXY_BATCH = 256


@dataclass(frozen=True)
class Grid:
    """The learning-rate rungs of a duo: rung k of a member is anchor * 2**(k / 2), rounded to 3 digits; the anchor is the member's episodic rate."""
    anchor: Mapping[str, float]

    def rate(self, side: str, k: int) -> float:
        return float(f"{self.anchor[side] * 2 ** (k / 2):.3g}")

    def rung(self, side: str, lr: float) -> int:
        k = round(2 * math.log2(lr / self.anchor[side]))
        if abs(self.rate(side, k) / lr - 1) > 0.005:
            raise ValueError(f"{lr:g} is not on the sqrt-2 grid of the {side} member (nearest rung {k}: {self.rate(side, k):g})")
        return k


def rate(side: str, k: int) -> float:
    """The main duo's rungs (the report scripts of the main duo use these; any other duo has its own `Grid`)."""
    return Grid(ANCHOR).rate(side, k)


def rung(side: str, lr: float) -> int:
    return Grid(ANCHOR).rung(side, lr)


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


class JointStore:
    """The live runs of the JOINT signal: the gate sits inside the loss, so its beta changes what the members learn and is part of a run's file name. A cell is the
    run of condition `cond` at the rates (large, small) and the gate beta; the condition carries both members' detector settings (src/reset/member_rules.py)."""

    def __init__(self, folder: Path, seed: int = 0, proxy: str = "nnn", b_t: int = 256):
        self.folder, self.seed, self.proxy, self.b_t = Path(folder), seed, proxy, b_t
        self._stores: dict[float, Store] = {}

    def cell(self, cond: str, rot: int, *, large: float, small: float, beta: float) -> Cell | None:
        if beta not in self._stores:
            self._stores[beta] = Store(self.folder, f"{self.proxy}_bt{self.b_t}_beta{beta:g}", self.seed)
        return self._stores[beta].cell(cond, rot, large=large, small=small)


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
            tie_order: Callable[[int], float] = float, rots: Sequence[int] = TUNING_ROTS, extend_cap: int = MAX_EXTEND,
            series: str | None = None, guard: Sequence[str] | None = None) -> Plan:
    """Rules 1 to 6 over one ordered ladder of integer rungs, scored by the accuracy of member `side` (`series`: another series, e.g. "gate" for the duo's
    prediction; `guard`: the members whose dead flag excludes a cell, default `side`). `cell_of(k, rot)`: the stored run (None if not run yet); `neighbours(k)`:
    (k below, k above) on the finer grid; `tie_order(k)`: the last tie-break. `rots`: one rotation, or two whose visits are pooled (rule 3 then runs the argmax
    and its neighbours on the second)."""
    plan = Plan()
    first, second = rots[0], (rots[1] if len(rots) > 1 else None)
    series, guard = series or side, tuple(guard or (side,))

    def alive(c: Cell) -> bool:
        return not any(c.dead[m] for m in guard)

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
        best = max(ok, key=lambda k: cell_of(k, first).mean(series))
        lo, hi = neighbours(best)
        # the extension cap of rule 5 bounds this refinement too: a ladder that keeps creeping up by hundredths per rung is a plateau, not chased
        # (the cap was only enforced after a choice until 2026-10-27, when own_cusum's threshold ladder crept below the cap)
        wanted = [k for k in (lo, hi) if min(start) - 2 * extend_cap <= k <= max(start) + 2 * extend_cap]
        if any([need(k, first) for k in wanted]):
            plan.notes.append(f"argmax of rotation {first} is rung {best}; its neighbours {lo} and {hi} are run first")
            return plan
        if max(ok, key=lambda k: cell_of(k, first).mean(series)) == best:
            break
    cand = [k for k in (lo, best, hi) if (c := cell_of(k, first)) is not None and alive(c)]
    if second is not None:                                                       # rule 3
        if any([need(k, second) for k in cand]):
            plan.notes.append(f"the argmax {best} and its neighbours are run on rotation {second}")
            return plan
        cand = [k for k in cand if alive(cell_of(k, second))]
    pooled = {k: (np.concatenate([cell_of(k, r).acc[series] for r in rots]), float(np.mean([cell_of(k, r).resets[side] for r in rots]))) for k in cand}
    chosen, tied = choose(pooled, tie_order)
    tried = sorted({k for k in _all_keys(cell_of, start, neighbours, first)})
    plan.table = [{"rung": k, "acc": float(pooled[k][0].mean()) if k in pooled else None, "acc_rot%d" % first: cell_of(k, first).mean(series),
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
    shared_rung: int | None = None                                            # the chosen rate rung of the shared setting, once its ladder is complete


def delta_setting(setting, j: int):
    """`setting` with its threshold moved j sqrt-2 rungs (the readable grid of `threshold`)."""
    return setting._replace(delta=threshold(setting.delta, j))


def plan_member(side: str, cell_of: Callable[[object, int, int], Cell | None], shared, candidates: Sequence, *, candidate_start: Sequence[int] | None = None,
                rots: Sequence[int] = TUNING_ROTS, max_rounds: int = 3) -> MemberPlan:
    """One member's search over (detector setting, learning rate), by the rules of `plan_1d` (scored by the member's own accuracy; `cell_of(setting, rung, rotation)`).
    A. a rate ladder for the shared setting (start `X2`) and for every candidate (start `candidate_start`; by default every second rung from 4 below to 2 above the
    shared setting's chosen rung, so the candidates are only requested once the shared ladder is complete); B. the candidate with the best pooled accuracy at its
    chosen rate (`choose`: ties by fewest resets, then the SHARED setting first, then the order of `candidates`); C. its threshold on the sqrt-2 grid at that rate, then,
    if the threshold moved, the rate one rung either side at the new threshold, and again until neither moves (at most `max_rounds`)."""
    plan = MemberPlan()
    settings = [shared, *candidates]

    def add(reqs, setting):
        plan.requests += [(setting, k, r) for k, r in reqs if (setting, k, r) not in plan.requests]

    def ladder(c, start):
        pl = plan_1d(side=side, start=start, neighbours=lambda k: (k - 1, k + 1), rots=rots, cell_of=lambda k, rot: cell_of(c, k, rot))
        add(pl.requests, c)
        return pl

    ladders = {shared: ladder(shared, X2[side])}
    plan.shared_rung = ladders[shared].chosen
    if candidate_start is None:
        if plan.requests:
            plan.notes.append("the shared setting's rate ladder is not complete: the candidates' ladders start at its chosen rung")
            return plan
        k0 = ladders[shared].chosen
        candidate_start = X2[side] if k0 is None else (k0 - 4, k0 - 2, k0, k0 + 2)
    for c in candidates:
        ladders[c] = ladder(c, candidate_start)
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


# ------------------------------------------------------------------------------------------------------------ coupled members: the joint signal

def pair_neighbourhood(kv: int, ks: int) -> list[tuple[int, int]]:
    """The cells that must be tried around a pair of rate rungs: the cross (one member's rung up or down, the other's fixed) and the two cells that scale both
    members together (the direction of the starting ladder). The two off-diagonal corners of the full 3 x 3 are not run."""
    return [(kv - 1, ks), (kv + 1, ks), (kv, ks - 1), (kv, ks + 1), (kv - 1, ks - 1), (kv + 1, ks + 1)]


def plan_pair(*, cell_of: Callable[[tuple[int, int], int], Cell | None], series: str = "gate", rots: Sequence[int] = TUNING_ROTS,
              start: Sequence[tuple[int, int]] | None = None) -> Plan:
    """The rules of `plan_1d` for two members that cannot be tuned separately (the joint signal: each trains on the duo's entropy, so a member's trajectory depends on its
    partner's rate): a pair of rungs (large k, small k') is one cell, scored by `series` (the duo's prediction). Start: the diagonal of the X2 ladders (or `start`); refine
    until the argmax has all of `pair_neighbourhood` tried; the argmax and its two best neighbours are the candidates (run on the second rotation too, if there are two);
    choose by `choose` (ties: fewest resets of both members, then the lowest rungs); the chosen cell must have its neighbourhood tried too (run if not)."""
    plan = Plan()
    first, second = rots[0], (rots[1] if len(rots) > 1 else None)
    start = list(start or zip(X2["large"], X2["small"]))

    def need(cell, rot) -> bool:
        if cell_of(cell, rot) is None:
            if (cell, rot) not in plan.requests:
                plan.requests.append((cell, rot))
            return True
        return False

    if any([need(c, first) for c in start]):
        plan.notes.append("the starting cells are not complete on rotation 0")
        return plan

    def tried() -> list[tuple[int, int]]:
        seen, frontier = set(), list(start)
        while frontier:
            c = frontier.pop()
            if c in seen or cell_of(c, first) is None:
                continue
            seen.add(c)
            frontier += pair_neighbourhood(*c)
        return sorted(seen)

    for _ in range(40):
        ok = [c for c in tried() if not any(cell_of(c, first).dead.values())]
        if not ok:
            plan.notes.append("every cell is dead on the tuning rotation")
            return plan
        best = max(ok, key=lambda c: cell_of(c, first).mean(series))
        if any([need(c, first) for c in pair_neighbourhood(*best)]):
            plan.notes.append(f"argmax is {best}: its neighbourhood is run first")
            return plan
        if max(ok, key=lambda c: cell_of(c, first).mean(series)) == best:
            break
    near = sorted((c for c in pair_neighbourhood(*best) if not any(cell_of(c, first).dead.values())), key=lambda c: -cell_of(c, first).mean(series))[:2]
    cand = [best] + near
    if second is not None:
        if any([need(c, second) for c in cand]):
            plan.notes.append(f"the argmax {best} and its two best neighbours are run on rotation {second}")
            return plan
        cand = [c for c in cand if not any(cell_of(c, second).dead.values())]
    pooled = {c: (np.concatenate([cell_of(c, r).acc[series] for r in rots]), float(np.mean([sum(cell_of(c, r).resets.values()) for r in rots]))) for c in cand}
    chosen, tied = choose(pooled, lambda c: float(c[0] + c[1]))
    plan.table = [{"cell": c, "acc": float(pooled[c][0].mean()), "resets": pooled[c][1], "tied": c in tied, "chosen": c == chosen} for c in cand]
    if any([need(c, first) for c in pair_neighbourhood(*chosen)]):
        plan.notes.append(f"chosen cell {chosen}: its neighbourhood is run first")
        return plan
    plan.chosen = chosen
    plan.notes.append(f"chosen cell {chosen}; within one paired SE of the best: {tied}")
    return plan


BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)       # the gate's sharpness inside the loss; 0 is the fixed-TS duo (no gate)
BETA_EXTEND = (2.0, 4.0)                  # tried above the grid, one at a time, while the choice sits on its top (at most MAX_EXTEND of them)
START_BETA = 0.5                          # the gate of the rate-pair search's first pass


@dataclass
class JointPlan:
    requests: list[tuple[tuple, int]] = field(default_factory=list)        # ((rate rungs, beta, large setting, small setting), rotation) still to run
    final: tuple | None = None
    notes: list[str] = field(default_factory=list)
    table: list[dict] = field(default_factory=list)                        # one row per cell of each block's last plan
    converged: bool = False


def plan_joint(cell_of: Callable[[tuple, int], Cell | None], shared, candidates: Mapping[str, Sequence], *, betas: Sequence[float] = BETAS,
               rots: Sequence[int] = TUNING_ROTS, max_passes: int = 2) -> JointPlan:
    """The search of the JOINT signal: both members adapt on the gated duo's entropy, so nothing is measured per member. A cell is (rate rungs (large, small), gate beta,
    large setting, small setting); every cell is scored by the duo's accuracy (series "gate"), a cell with a dead member is excluded. The blocks, in order, each by the
    rules of `plan_1d` / `choose` with the other blocks held fixed, repeated as a pass until nothing moves (at most `max_passes`):

      rates      the pair of rungs: `plan_pair` (the first pass from the diagonal of the X2 ladders at beta START_BETA and the shared setting for both members,
                 later passes from the previous pair)
      beta       every value of `betas`, then one of BETA_EXTEND at a time while the choice sits on the top of what was tried (ties: the lowest beta: 0 means the gate does
                 not help)
      detector   per member, one at a time (large first): its candidate settings (the shared one and the screen's), the best, then its threshold on the sqrt-2 grid
                 (the same rules; ties: fewest resets, then the shared setting first)

    `cell_of(key, rot)`: the stored run of a cell (None if not run yet). Returns a JointPlan: the runs still needed (the plan stops at the first block that needs any),
    or the final cell. Stateless, like `plan_member`: every call replays the decisions from the stored runs."""
    plan = JointPlan()
    pair, beta, sets = None, START_BETA, {"large": shared, "small": shared}
    SIDES_ = ("large", "small")

    def key(pr, b, st):
        return (pr, b, st["large"], st["small"])

    def stop(requests, note) -> JointPlan:
        plan.requests, plan.notes = requests, plan.notes + [note]
        return plan

    for p in range(max_passes):
        moved = False
        # ---- the rate pair
        sub = plan_pair(cell_of=lambda pr, rot: cell_of(key(pr, beta, sets), rot), rots=rots, start=None if pair is None else [pair])
        if sub.requests:
            return stop([(key(pr, beta, sets), rot) for pr, rot in sub.requests], f"pass {p + 1}, rates: " + "; ".join(sub.notes))
        if sub.chosen is None:
            return stop([], f"pass {p + 1}, rates: " + "; ".join(sub.notes))
        moved |= sub.chosen != pair
        pair = sub.chosen
        plan.table = [{"block": "rates", **r} for r in sub.table]
        plan.notes.append(f"pass {p + 1}: rates {pair}")
        # ---- the gate's beta
        tried = list(betas)
        for ext in range(MAX_EXTEND + 1):
            cells = {b: [cell_of(key(pair, b, sets), r) for r in rots] for b in tried}
            missing = [(key(pair, b, sets), r) for b in tried for r, c in zip(rots, cells[b]) if c is None]
            if missing:
                return stop(missing, f"pass {p + 1}, beta: the gate values {tried} are not complete at the chosen rates")
            alive = {b: cs for b, cs in cells.items() if not any(any(c.dead.values()) for c in cs)}
            if not alive:
                return stop([], f"pass {p + 1}, beta: every gate value kills a member")
            pooled = {b: (np.concatenate([c.acc["gate"] for c in cs]), float(np.mean([sum(c.resets.values()) for c in cs]))) for b, cs in alive.items()}
            new_beta, tied = choose(pooled, float)
            if new_beta == max(tried) and ext < MAX_EXTEND:                       # the best gate is the sharpest tried: try a sharper one
                tried.append(BETA_EXTEND[ext])
                continue
            break
        moved |= new_beta != beta
        beta = new_beta
        plan.table += [{"block": "beta", "beta": b, "acc": float(pooled[b][0].mean()), "resets": pooled[b][1], "tied": b in tied, "chosen": b == beta} for b in pooled]
        plan.notes.append(f"pass {p + 1}: gate beta {beta:g}" + (" (the gate does not help: beta 0 is the fixed-TS duo)" if beta == 0 else "") +
                          (f" (the sharpest gate tried, {max(tried):g}: a plateau, reported as such)" if beta == max(tried) and beta > max(betas) else ""))
        # ---- each member's detector, one at a time
        for side in SIDES_:
            options = [shared, *candidates[side]]

            def at(setting, rot, side=side):
                return cell_of(key(pair, beta, {**sets, side: setting}), rot)

            missing = [(key(pair, beta, {**sets, side: o}), r) for o in options for r in rots if at(o, r) is None]
            if missing:
                return stop(missing, f"pass {p + 1}, {side} detector: its candidate settings are not complete at the chosen rates and gate")
            ok = [o for o in options if not any(any(at(o, r).dead.values()) for r in rots)]
            if not ok:
                return stop([], f"pass {p + 1}, {side} detector: every candidate kills a member")
            pooled = {o: (np.concatenate([at(o, r).acc["gate"] for r in rots]), float(np.mean([at(o, r).resets[side] for r in rots]))) for o in ok}
            best, tied = choose(pooled, options.index)
            ladder = plan_1d(side=side, start=(-1, 0, 1), neighbours=lambda j: (j - 1, j + 1), rots=rots, tie_order=lambda j: -j, series="gate", guard=SIDES_,
                             cell_of=lambda j, r, best=best: at(delta_setting(best, j), r))
            if ladder.requests:
                return stop([(key(pair, beta, {**sets, side: delta_setting(best, j)}), r) for j, r in ladder.requests],
                            f"pass {p + 1}, {side} detector: best candidate {tuple(best)}; " + "; ".join(ladder.notes))
            if ladder.chosen is None:
                return stop([], f"pass {p + 1}, {side} detector: " + "; ".join(ladder.notes))
            new = delta_setting(best, ladder.chosen)
            moved |= new != sets[side]
            sets[side] = new
            plan.table += [{"block": f"{side} detector", "setting": tuple(o), "acc": float(pooled[o][0].mean()), "resets": pooled[o][1], "tied": o in tied, "chosen": o == best} for o in ok]
            plan.notes.append(f"pass {p + 1}: {side} detector {tuple(new)}")
        plan.final = key(pair, beta, sets)
        if p > 0 and not moved:
            plan.converged = True
            break
    plan.final = key(pair, beta, sets)
    plan.notes.append("converged: a second pass changed nothing" if plan.converged else f"not converged after {max_passes} passes: the last choice is kept")
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
        mp = plan_member("large", member_cell, S0, [C1, C2], candidate_start=(2, 4, 6, 8))
        for st, k, rot in mp.requests:
            assert st in (S0, C1, C2) or fam(st) == fam(C1), f"a request for a setting that is neither a candidate nor a threshold variant of the best: {st}"
            run_member(st, k, rot)
        if not mp.requests:
            break
    else:
        raise AssertionError("plan_member did not converge")
    assert mp.final is not None and fam(mp.final[0]) == fam(C1) and mp.final[0].delta == best_delta.delta and mp.final[1] == 5, (mp.final, mp.notes)
    # the candidates' start rungs derived from the shared ladder: nothing is requested for them until the shared ladder is complete, then every second rung
    # from 4 below to 2 above its chosen rung (the main duo's ViT: shared rung 6 -> 2, 4, 6, 8)
    made.clear()
    mp = plan_member("large", member_cell, S0, [C1, C2])
    assert mp.requests and all(st == S0 for st, _, _ in mp.requests) and mp.shared_rung is None and "not complete" in mp.notes[0], mp.notes
    for _ in range(20):                                    # run the shared ladder's waves until it is complete
        mp = plan_member("large", member_cell, S0, [C1, C2])
        if not mp.requests or any(st != S0 for st, _, _ in mp.requests):
            break
        for st, k, rot in mp.requests:
            run_member(st, k, rot)
    assert mp.shared_rung == 4 and sorted({k for st, k, _ in mp.requests if st == C1}) == [0, 2, 4, 6], (mp.shared_rung, mp.requests)
    assert sorted({k for st, k, _ in mp.requests if st == C2}) == [0, 2, 4, 6], "both candidates start at the same rungs"
    # grids: another duo has its own anchors
    gr = Grid({"large": 1.25e-5, "small": 8e-4})
    assert gr.rate("large", 0) == 1.25e-5 and gr.rate("small", -2) == 4e-4 and gr.rung("small", 4e-4) == -2 and gr.rate("large", 2) == 2.5e-5
    try:
        gr.rung("large", 2e-5)
        raise AssertionError("2e-5 is not on this grid")
    except ValueError:
        pass
    # ---- one tuning rotation (the default) and two pooled: the same ladders, the same choices
    for rots in ((0,), (0, 2)):
        assert drive(Ladder(peak=1), rots=rots).chosen == 1 and drive(Ladder(peak=6), rots=rots).chosen == 6 and drive(Ladder(peak=-5), rots=rots).chosen == -5
    lad = Ladder(peak=1)
    drive(lad)
    assert {r for _, r in lad.have} == {0}, "one tuning rotation: nothing is run on a second"
    lad = Ladder(peak=1)
    drive(lad, rots=(0, 2))
    assert {r for _, r in lad.have} == {0, 2}
    # ---- the pair search (coupled members), on one rotation and on two: a 2-D surface with its peak off the diagonal
    for rots in ((0,), (0, 2)):
        have2: dict = {}

        def run2(c, rot):
            kv, ks = c
            acc = 0.6 - 0.01 * (kv - 1) ** 2 - 0.01 * (ks + 1) ** 2 + np.random.default_rng(7 * rot + 31 * kv + 101 * ks + 9999).normal(0, 0.002, 16)
            have2[(c, rot)] = Cell({"gate": acc, "large": acc, "small": acc}, {"large": 2.0, "small": 2.0}, {"large": False, "small": False}, 800000)

        for _ in range(40):
            p2 = plan_pair(cell_of=lambda c, r: have2.get((c, r)), rots=rots)
            if not p2.requests:
                break
            for c, r in p2.requests:
                run2(c, r)
        assert p2.chosen == (1, -1) and p2.done, (rots, p2.chosen, p2.notes)
    # ---- the joint search on a synthetic surface with a known optimum: rates (1, -1), beta 0.75, the large member at C1 (threshold one rung up), the small at C2
    TRUE = {"pair": (1, -1), "beta": 0.75, "large": delta_setting(C1, 1), "small": C2}

    def surface(beta_best, k):
        (kv, ks), b, sl, ss = k
        def dpen(st, true):
            if (st.alpha_fast, st.alpha_slow, st.W0) != (true.alpha_fast, true.alpha_slow, true.W0):
                return 0.01
            return 0.004 * (math.log2(st.delta / true.delta) * 2) ** 2
        return 0.6 - 0.01 * (kv - 1) ** 2 - 0.01 * (ks + 1) ** 2 - 0.004 * ((b - beta_best) / 0.25) ** 2 - dpen(sl, TRUE["large"]) - dpen(ss, TRUE["small"])

    def drive_joint(beta_best, rots=(0,), **kw):
        made_j, waves = {}, []
        def cell_j(k, rot):
            return made_j.get((k, rot))
        for _ in range(80):
            jp = plan_joint(cell_j, S0, {"large": [C1, C2], "small": [C2, C1]}, rots=rots, **kw)
            if not jp.requests:
                return jp, made_j, waves
            waves.append(len(jp.requests))
            for k, rot in jp.requests:
                acc = surface(beta_best, k) + np.random.default_rng(abs(hash((k, rot))) % 2**32).normal(0, 0.0015, 16)
                made_j[(k, rot)] = Cell({"gate": acc, "large": acc, "small": acc}, {"large": 5.0, "small": 5.0}, {"large": False, "small": False}, 800000)
        raise AssertionError("plan_joint did not converge")

    first = plan_joint(lambda k, r: None, S0, {"large": [C1, C2], "small": [C2, C1]})
    assert [k for k, _ in first.requests] == [(c, START_BETA, S0, S0) for c in zip(X2["large"], X2["small"])], "the first wave: the diagonal at beta 0.5, the shared setting for both"
    jp, made_j, waves = drive_joint(0.75)
    assert jp.final == (TRUE["pair"], TRUE["beta"], TRUE["large"], TRUE["small"]) and jp.converged, (jp.final, jp.notes)
    assert all(r == 0 for _, r in made_j), "one tuning rotation"
    print(f"  joint search: found the optimum in {len(waves)} waves, {len(made_j)} runs ({waves})")
    jp2, _, _ = drive_joint(0.75, rots=(0, 2))
    assert jp2.final == jp.final, "two pooled rotations choose the same cell"
    # the gate is best at its sharpest value: the grid is extended upward, one step at a time; at beta 0 the gate is reported as not helping
    jp, _, _ = drive_joint(2.0)
    assert jp.final[1] == 2.0 and jp.converged, (jp.final, jp.notes)
    jp, _, _ = drive_joint(0.0)
    assert jp.final[1] == 0.0 and any("does not help" in n for n in jp.notes), jp.notes
    assert BETAS == (0.0, 0.25, 0.5, 0.75, 1.0)
    # a member that dies at a cell excludes the cell
    made_d, dead_cell = {}, ((1, -1), START_BETA, S0, S0)

    def cell_d(k, rot):
        if (k, rot) not in made_d:
            return None
        return made_d[(k, rot)]
    for _ in range(80):
        jp = plan_joint(cell_d, S0, {"large": [C1, C2], "small": [C2, C1]})
        if not jp.requests:
            break
        for k, rot in jp.requests:
            acc = surface(0.75, k) + np.random.default_rng(abs(hash((k, rot))) % 2**32).normal(0, 0.0015, 16)
            made_d[(k, rot)] = Cell({"gate": acc, "large": acc, "small": acc}, {"large": 5.0, "small": 5.0},
                                    {"large": k == dead_cell, "small": False}, 800000)
    assert jp.final is not None and jp.final != dead_cell and all(not c.dead["large"] for (k, _), c in made_d.items() if k == jp.final), "a dead cell is never chosen"
    print("tuning self-test passed")
