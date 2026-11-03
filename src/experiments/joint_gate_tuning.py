"""
joint_gate_tuning.py
====================
The logic behind scripts/02f_tune_joint_gate.py: choose the two members' learning rates AND the gate's beta TOGETHER for Joint DD (both members adapt on the
entropy of the GATED prediction, the gate sitting inside the loss: src/experiments/joint_dd.py).

Why a tuner of its own. scripts/02b_tune_joint_lr.py tunes the pair for the static fixed-TS duo (every gate weight 0.5, i.e. beta = 0), and script 09 then
picks beta at those rates. But the gate is part of the loss: beta changes what each member learns, so the best rates may depend on beta (a gate that holds a
collapsing member down can allow larger rates) and the best beta on the rates. Here a CELL is (lr_large, lr_small, beta) and every cell is one live
`run_joint_dd` over the 4 held-out streams, each from source weights, one stream-order seed (a gated run cannot be replayed from stored logits).
Everything is on the held-out streams; nothing is chosen on a test stream.

The selection is the rules of the reset tuning's joint signal (src/reset/tuning.py: `plan_pair`, `choose`; protocol: docs/reset_tuning.md), minus the detectors,
with a held-out STREAM as the unit that stands for a visit and the duo's whole-stream accuracy as the score. The search is NESTED, not a coordinate descent: the
reset tuning fixes the pair at one beta and then looks at beta, which cannot follow a ridge where the pair and beta must move together (the very interaction this
tuner is for), so here

  rates   at EVERY beta of the grid (in parallel, one wave) the pair of rungs on the sqrt-2 grid anchored at the duo's recorded joint pair
          (cfgs/duo_lrs/<duo>__joint.yaml): the diagonal of the x2 ladders, the argmax's whole neighbourhood, the best of it and its two best neighbours, the choice
          among them (the best, then everything within one paired standard error over the streams, then the lowest rungs), and the chosen cell's own neighbourhood
  beta    the betas are compared at their OWN best pairs: the best, then everything within one paired standard error, then the LOWEST beta, so a choice of 0 means the
          gate does not help the joint loss even with its rates retuned (the fixed-TS duo); if the best is the sharpest tried, one of BETA_EXTEND at a time

Guards, as in 02b: a cell in which a member is DEAD (four consecutive windows of 5,120 images below 5% on a stream) is excluded, and so is a cell whose late
accuracy (the last 20% of each stream, macro) is below the frozen duo's (an adaptation that ends worse than not adapting is drift, however good its mean).

The CONFIRMATION (report only, it changes nothing), on the confirmation seeds: the recorded pair and the best pair at beta 0 (no gate) at beta 0, and the chosen pair at
beta 0 and at the chosen beta (plus the best no-gate pair under the chosen gate). It answers the question this tuner exists for: does tuning the rates together with
the gate beat the pair tuned without it, and does the gate help at either?

Everything is stateless and resumable: `plan` replays the decisions from the stored runs (`Store`), and says which runs are still needed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.experiments import criteria as C
from src.reset import tuning as TU

SERIES = ("duo", "large", "small")
LATE_FRAC = 0.2
BETAS = TU.BETAS
START_BETA = TU.START_BETA
START_RUNGS = (-4, -2, 0, 2)               # the diagonal of the first x2 ladder: 0.25x, 0.5x, 1x, 2x the recorded pair, both members together
SIDES = TU.SIDES


# ------------------------------------------------------------------------------------------------------------ curves

def late_of_curve(batch_n: Sequence[int], batch_correct: Sequence[int], frac: float = LATE_FRAC) -> float:
    """Accuracy over the last `frac` of a stream (whole batches, counted from the end), as 02b's `late_accuracy`."""
    n, k = np.asarray(batch_n), np.asarray(batch_correct)
    need, got, i = frac * n.sum(), 0, len(n)
    while i > 0 and got < need:
        i -= 1
        got += n[i]
    return float(k[i:].sum() / n[i:].sum())


def curve_dead(batch_n: Sequence[int], batch_correct: Sequence[int]) -> bool:
    """A member is dead on a stream: DEAD_WINDOWS consecutive windows of about DEAD_WINDOW images below DEAD_ACC (the reset tuning's `is_dead`, on per-batch counts)."""
    n, k = np.asarray(batch_n), np.asarray(batch_correct)
    per = max(1, int(round(TU.DEAD_WINDOW / float(np.median(n)))))
    w = len(n) // per
    if w < TU.DEAD_WINDOWS:
        return False
    acc = k[:w * per].reshape(w, per).sum(1) / n[:w * per].reshape(w, per).sum(1)
    low = (acc < TU.DEAD_ACC).astype(int)
    return bool((np.convolve(low, np.ones(TU.DEAD_WINDOWS, dtype=int), mode="valid") == TU.DEAD_WINDOWS).any())


# ------------------------------------------------------------------------------------------------------------ one run

@dataclass(frozen=True)
class Run:
    """One live Joint DD run: both members' rates, the gate's beta, one stream-order seed."""
    lr_large: float
    lr_small: float
    beta: float
    seed: int = 0

    @property
    def stem(self) -> str:
        return f"L{self.lr_large:g}_S{self.lr_small:g}__beta{self.beta:g}__seed{self.seed}"

    @property
    def arg(self) -> str:                                   # what `--run` takes
        return f"{self.lr_large:g}:{self.lr_small:g}:{self.beta:g}:{self.seed}"

    @classmethod
    def parse(cls, text: str) -> "Run":
        try:
            lr_l, lr_s, beta, seed = text.split(":")
            run = cls(float(lr_l), float(lr_s), float(beta), int(seed))
        except ValueError:
            raise ValueError(f"a run is LR_LARGE:LR_SMALL:BETA:SEED (e.g. 0.0032:0.004:0.5:0), got {text!r}") from None
        if run.lr_large <= 0 or run.lr_small <= 0 or run.beta < 0:
            raise ValueError(f"rates must be positive and beta non-negative: {text!r}")
        return run


def summarize(payload: dict) -> dict:
    """The numbers of a stored run (fractions): macro means over its streams, the worst stream, the late duo accuracy, the gate's mean weight."""
    s = payload["streams"]
    out = {f"{k}": float(np.mean([s[lab][k]["accuracy"] for lab in s])) for k in SERIES}
    duo = {lab: s[lab]["duo"]["accuracy"] for lab in s}
    out["worst"], out["worst_stream"] = min(duo.values()), min(duo, key=duo.get)
    out["late_duo"] = float(np.mean([late_of_curve(**payload["curves"][lab]["duo"]) for lab in s]))
    out["nll"] = float(np.mean([s[lab]["duo"]["nll"] for lab in s]))
    w = np.concatenate([np.asarray(v, dtype=float) for v in payload["gate_w"].values()]) if payload.get("gate_w") else np.array([math.nan])
    out["w_mean"], out["w_std"] = float(np.nanmean(w)), float(np.nanstd(w))
    out["streams"] = duo
    return out


class Store:
    """The stored runs of one tuning (one duo, gate and sample cap): `runs/<stem>.json`, written whole and atomically. `floor` is the frozen duo's late accuracy,
    which the drift guard compares each cell with (None: the guard is off)."""

    def __init__(self, folder: Path, floor: float | None = None):
        self.folder, self.floor = Path(folder), floor
        self._cells: dict[str, TU.Cell] = {}

    def path(self, run: Run) -> Path:
        return self.folder / "runs" / f"{run.stem}.json"

    def has(self, run: Run) -> bool:
        return self.path(run).is_file()

    def write(self, run: Run, payload: dict) -> Path:
        p = self.path(run)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, p)
        self._cells.pop(run.stem, None)
        return p

    def read(self, run: Run) -> dict | None:
        return json.loads(self.path(run).read_text()) if self.has(run) else None

    def cell(self, run: Run) -> TU.Cell | None:
        """The run as the selection rules see it: per-stream accuracies ("gate" = the duo's prediction), no resets, dead flags (a member dead on any stream) and
        `drift` (the late accuracy is below the frozen duo's)."""
        if run.stem not in self._cells:
            payload = self.read(run)
            if payload is None:
                return None
            s, curves = payload["streams"], payload["curves"]
            acc = {"gate": np.asarray([s[lab]["duo"]["accuracy"] for lab in s]), "large": np.asarray([s[lab]["large"]["accuracy"] for lab in s]),
                   "small": np.asarray([s[lab]["small"]["accuracy"] for lab in s])}
            dead = {m: any(curve_dead(**curves[lab][m]) for lab in s) for m in SIDES}
            dead["drift"] = self.floor is not None and summarize(payload)["late_duo"] < self.floor
            self._cells[run.stem] = TU.Cell(acc, {m: 0.0 for m in SIDES}, dead, int(sum(s[lab]["duo"]["n"] for lab in s)))
        return self._cells[run.stem]

    def runs(self) -> list[Run]:
        out = []
        for f in sorted((self.folder / "runs").glob("*.json")) if (self.folder / "runs").is_dir() else []:
            m = re.match(r"^L(?P<l>[^_]+)_S(?P<s>[^_]+)__beta(?P<b>[^_]+)__seed(?P<seed>\d+)$", f.stem)
            if m:
                out.append(Run(float(m["l"]), float(m["s"]), float(m["b"]), int(m["seed"])))
        return out


def execute(ctx, segments, temps, run: Run, *, store: Store, proxy: str, b_t: int, num_samples: int | None = None) -> Path | None:
    """Run one cell (a GPU job) and store it; a run already stored is not repeated (returns None)."""
    from src.experiments import adaptation_modes as AM
    from src.experiments.joint_dd import run_joint_dd

    if store.has(run):
        print(f"{run.stem}: stored already, skipped")
        return None
    t0 = time.time()
    res = run_joint_dd(AM.with_learning_rates(ctx, run.lr_large, run.lr_small), segments,
                       gate_kwargs=dict(proxy=proxy, beta=run.beta, b_t=b_t, T_l=temps.T_l, T_s=temps.T_s), seed=run.seed, num_samples=num_samples)
    streams: dict = {}
    for r in res["rows"]:
        streams.setdefault(r["segment"], {})[r["series"]] = {k: r[k] for k in ("n", "accuracy", "ece", "nll", "entropy")}
    payload = {"run": {"lr_large": run.lr_large, "lr_small": run.lr_small, "beta": run.beta, "seed": run.seed},
               "gate": {"proxy": proxy, "b_t": b_t}, "num_samples": num_samples, "batch_size": ctx.batch_size,
               "temperatures": {"T_l": temps.T_l, "T_s": temps.T_s}, "seconds": round(time.time() - t0, 1),
               "streams": streams, "curves": res["curves"],
               "gate_w": {lab: [rec["w_l"] for rec in recs] for lab, recs in res["gate_records"].items()}}
    p = store.write(run, payload)
    s = summarize(payload)
    print(f"{run.stem}: duo {100 * s['duo']:.2f} (worst {100 * s['worst']:.2f} on {s['worst_stream']}, late {100 * s['late_duo']:.2f}), large {100 * s['large']:.2f}, "
          f"small {100 * s['small']:.2f}, gate weight {s['w_mean']:.3f} +- {s['w_std']:.3f}  [{payload['seconds']:.0f} s]")
    return p


# ------------------------------------------------------------------------------------------------------------ the search

def plan(cell_of: Callable[[tuple, int], TU.Cell | None], *, start: Sequence[tuple[int, int]] | None = None, betas: Sequence[float] = BETAS,
         rots: Sequence[int] = (0,), criterion: C.Criterion = C.MEAN) -> TU.JointPlan:
    """The nested rates-and-beta search (see the module docstring). `cell_of((rung pair, beta), seed)`: the stored run of that cell (None if not run yet); `rots` are
    the stream-order seeds the search pools (one, or two). Returns a `TU.JointPlan`: the runs still needed as ((rung pair, beta), seed) (the pair searches of all
    the betas that are not finished, together), or `final` = (rung pair, beta) and `table`, one row per beta with its own best pair. `criterion` chooses the pair at every
    beta and then the beta (src/experiments/criteria.py; under worst_shift the unit is the held-out STREAM: the worst stream among the cells within the tolerance of the best mean)."""
    out = TU.JointPlan()
    start = list(start or [(k, k) for k in START_RUNGS])
    tried, best_at = list(betas), {}
    for ext in range(TU.MAX_EXTEND + 1):
        requests = []
        for b in tried:
            if b in best_at:
                continue
            sub = TU.plan_pair(cell_of=lambda pr, rot, b=b: cell_of((pr, b), rot), rots=rots, start=start, criterion=criterion)
            if sub.requests:
                requests += [((pr, b), rot) for pr, rot in sub.requests]
                out.notes.append(f"beta {b:g}: " + "; ".join(sub.notes))
            else:
                best_at[b] = sub.chosen                                          # None: every cell at this beta kills a member or drifts
                out.notes.append(f"beta {b:g}: " + ("; ".join(sub.notes) if sub.chosen is None else f"rates {sub.chosen}"))
        if requests:
            out.requests = requests
            return out
        alive = {b: pair for b, pair in best_at.items() if pair is not None}
        if not alive:
            out.notes.append("every gate value kills a member or drifts below the frozen duo")
            return out
        pooled = {b: (np.concatenate([cell_of((pair, b), r).acc["gate"] for r in rots]), 0.0) for b, pair in alive.items()}
        beta, tied = TU.choose(pooled, float, criterion)
        if beta == max(tried) and ext < TU.MAX_EXTEND:                         # the sharpest gate tried is the best: try a sharper one
            tried.append(TU.BETA_EXTEND[ext])
            continue
        break
    out.table = [{"beta": b, "pair": alive[b], "acc": float(pooled[b][0].mean()), "tied": b in tied, "chosen": b == beta} for b in sorted(alive)]
    out.final, out.converged = (alive[beta], beta), True
    out.notes.append(f"gate beta {beta:g} at rates {alive[beta]}; " + (f"within {100 * criterion.tol:g} points of the best mean: " if criterion.robust else "within one paired SE of the best: ") + f"{sorted(tied)}" +
                     (" (the gate does not help: beta 0 is the fixed-TS duo)" if beta == 0 else "") +
                     (f" (the sharpest gate tried, {max(tried):g}: a plateau, reported as such)" if beta == max(tried) and beta > max(betas) else ""))
    return out


def confirm_runs(old_pair: tuple[float, float], zero_pair: tuple[float, float], final_pair: tuple[float, float], beta: float, seeds: Sequence[int]) -> list[Run]:
    """The confirmation, without duplicates, the recorded pair at beta 0 first (script 03's Joint row): (recorded pair, 0), (best pair without a gate, 0),
    (chosen pair, 0), (chosen pair, chosen beta), (best pair without a gate, chosen beta), each on every seed."""
    configs = list(dict.fromkeys([(old_pair, 0.0), (zero_pair, 0.0), (final_pair, 0.0), (final_pair, beta), (zero_pair, beta)]))
    return [Run(pair[0], pair[1], b, seed) for seed in seeds for pair, b in configs]


# ------------------------------------------------------------------------------------------------------------ the report

def _pct(x: float) -> str:
    return f"{100 * x:6.2f}"


def runs_table(store: Store, runs: Sequence[Run], mark: Sequence[Run] = ()) -> str:
    head = f"{'lr_large':>10} {'lr_small':>10} {'beta':>5} {'seed':>4}  {'duo':>6} {'worst':>6} {'late':>6} {'large':>6} {'small':>6} {'gate w':>7}  flags"
    lines = [head]
    for r in runs:
        payload = store.read(r)
        if payload is None:
            lines.append(f"{r.lr_large:>10g} {r.lr_small:>10g} {r.beta:>5g} {r.seed:>4}  (not run)")
            continue
        s, c = summarize(payload), store.cell(r)
        flags = [k for k, v in c.dead.items() if v]
        lines.append(f"{r.lr_large:>10g} {r.lr_small:>10g} {r.beta:>5g} {r.seed:>4}  {_pct(s['duo'])} {_pct(s['worst'])} {_pct(s['late_duo'])} {_pct(s['large'])} "
                     f"{_pct(s['small'])} {s['w_mean']:>7.3f}  {','.join(flags)}{' *' if r in mark else ''}")
    return "\n".join(lines)


def confirm_table(store: Store, runs: Sequence[Run], seeds: Sequence[int]) -> str:
    """One row per configuration (pair, beta): the duo's macro accuracy per seed and its mean, and the paired difference to the first row over the seeds."""
    configs = list(dict.fromkeys((r.lr_large, r.lr_small, r.beta) for r in runs))
    acc = {c: [summarize(store.read(Run(*c, s)))["duo"] if store.has(Run(*c, s)) else math.nan for s in seeds] for c in configs}
    base = np.asarray(acc[configs[0]])
    lines = [f"{'lr_large':>10} {'lr_small':>10} {'beta':>5}  " + " ".join(f"seed{s:<3}" for s in seeds) + f"  {'mean':>6}  vs first row (paired over seeds)"]
    for c in configs:
        a = np.asarray(acc[c])
        d = a - base
        diff = "" if c == configs[0] else f"{100 * np.nanmean(d):+6.2f}" + (f" +- {100 * np.nanstd(d, ddof=1) / math.sqrt(np.isfinite(d).sum()):.2f}" if np.isfinite(d).sum() > 1 else "")
        lines.append(f"{c[0]:>10g} {c[1]:>10g} {c[2]:>5g}  " + " ".join(f"{_pct(x):>7}" for x in a) + f"  {_pct(np.nanmean(a))}  {diff}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------ submission

JOB_PREFIX, CHAIN_PREFIX = "jg_", "jg_next_"
EXCLUDE = "kn117,kn101,kn104,kn029"                                # nodes with an uncorrectable GPU ECC error (plan.md)
SCRIPT = "scripts/02f_tune_joint_gate.py"


def job_name(duo: str, run: Run) -> str:
    return JOB_PREFIX + re.sub(r"[^A-Za-z0-9.]+", "", f"{duo}_L{run.lr_large:g}S{run.lr_small:g}b{run.beta:g}s{run.seed}")


def sbatch_argv(duo: str, run: Run, common: Sequence[str], time_limit: str) -> list[str]:
    """One job per cell (a GPU job): `common` are the flags that make the worker's context the driver's (--bn, --proxy, --b_t, ...)."""
    return ["sbatch", "-t", time_limit, "-J", job_name(duo, run), f"--exclude={EXCLUDE}", "slurm/job.sh", SCRIPT, "--duo", duo, *common, "--run", run.arg]


def queued_jobs() -> dict[str, list[str]]:
    from src.reset.jobs import queued_jobs as q
    return q()


def submit(duo: str, runs: Sequence[Run], common: Sequence[str], time_limit: str, *, go: bool, cwd: Path | None = None) -> list[str]:
    """Print the sbatch command of every run; with `go`, submit those not already queued (by name). Returns the job names that are now in the queue."""
    queued = queued_jobs() if go else {}
    names = []
    for run in runs:
        name = job_name(duo, run)
        if name in queued:
            print(f"queued   {name}")
            names.append(name)
            continue
        argv = sbatch_argv(duo, run, common, time_limit)
        print(("SUBMIT  " if go else "to run  ") + " ".join(argv))
        if go:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
            print("   ->", (r.stdout or r.stderr).strip())
            if r.returncode == 0:
                names.append(name)
    return names


def chain(folder: Path, duo: str, names: Sequence[str], driver_args: Sequence[str], *, go: bool, cwd: Path | None = None) -> bool:
    """Submit one small CPU job that calls this driver again once every job of the wave has finished OK (`afterok`: a failed job cancels the chain, so a broken
    wave is not retried forever). One chain job per duo and folder; one already queued is left alone; called from a chain job itself it refuses to chain a wave
    identical to the last one it submitted (the jobs ran and changed nothing). Returns whether a chain job is now waiting."""
    name = CHAIN_PREFIX + re.sub(r"[^A-Za-z0-9.]+", "", duo) + "_" + hashlib.sha1(str(folder).encode()).hexdigest()[:6]
    state = Path(folder) / "chain_state.json"
    wave = sorted(names)
    if os.environ.get("SLURM_JOB_NAME", "").startswith(CHAIN_PREFIX) and state.is_file() and json.loads(state.read_text()).get("jobs") == wave:
        print(f"  CHAIN STOPPED: this wave is the one the last chain call submitted ({len(wave)} job(s) ran and changed nothing); see their logs")
        return False
    running = queued_jobs() if go else {}
    me = os.environ.get("SLURM_JOB_ID")
    if any(i != me for i in running.get(name, [])):
        print(f"queued   {name}")
        return True
    ids = sorted({i for n in wave for i in running.get(n, [])}) if go else ["<wave>"]
    if not ids:
        print("  CHAIN NOT SUBMITTED: none of the wave's jobs is in the queue to wait for")
        return False
    argv = ["sbatch", "-t", "00:30:00", "-J", name, f"--dependency=afterok:{':'.join(ids)}", "--kill-on-invalid-dep=yes", "slurm/job_cpu.sh", SCRIPT, *driver_args]
    print(("SUBMIT  " if go else "to run  ") + " ".join(argv))
    if not go:
        return True
    r = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
    print("   ->", (r.stdout or r.stderr).strip())
    if r.returncode == 0:
        Path(folder).mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"jobs": wave, "after": ids}))
    return r.returncode == 0


# ------------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)

    # ---- curves: late accuracy and the dead flag ---------------------------------------------------------------------------------
    assert late_of_curve([10] * 6, [9, 9, 9, 1, 1, 1], 0.2) == 0.1 and late_of_curve([10, 10, 4, 10, 10, 6], [9, 5, 1, 2, 2, 6], 0.2) == 8 / 16
    assert curve_dead([64] * 400, [0] * 400) and not curve_dead([64] * 400, [40] * 400) and not curve_dead([64] * 100, [0] * 100), "4 windows of 80 batches"
    assert not curve_dead([64] * 400, ([40] * 100 + [0] * 150 + [40] * 150)), "a dead stretch shorter than four windows is not dead"
    assert curve_dead([64] * 400, ([40] * 50 + [0] * 350)) and not curve_dead([64] * 400, ([0] * 79 + [40] * 321)), "a dead start that recovers within the first window"

    # ---- a run's name round-trips, and a bad one is refused -------------------------------------------------------------------------
    r = Run(0.0032, 0.004, 0.5, 1)
    assert r.stem == "L0.0032_S0.004__beta0.5__seed1" and Run.parse(r.arg) == r and Run.parse("1e-05:0.00226:0:0") == Run(1e-5, 0.00226, 0.0, 0)
    for bad in ("1:2:3", "a:b:c:d", "-1:1:0:0", "1:1:-1:0"):
        try:
            Run.parse(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert job_name("vitb16_rn50v1", r) == "jg_vitb16rn50v1L0.0032S0.004b0.5s1" and len(job_name("convnextb_deits", Run(0.000282, 0.0016, 0.75, 2))) < 60

    # ---- a live run on tiny models: stored, read back as a selection cell, not repeated; a diverging rate does not crash -------------
    import torch

    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.experiments.protocols import Segment
    from src.experiments.adaptation_modes import Temps

    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = make_context(tmp, batch_size=32, base=tiny_models(0))
        segs = [Segment(f"s{i}/s5", FakeStream(150, seed=sd), True) for i, sd in enumerate((1, 2, 3))]
        temps = Temps(tmp, 1.4, 0.9, "test")
        store = Store(tmp / "tune", floor=None)
        a = Run(1e-2, 1e-2, 0.0, 0)
        p = execute(ctx, segs, temps, a, store=store, proxy="nnn", b_t=64)
        assert p is not None and store.has(a) and store.runs() == [a]
        mtime = p.stat().st_mtime_ns
        assert execute(ctx, segs, temps, a, store=store, proxy="nnn", b_t=64) is None and p.stat().st_mtime_ns == mtime, "a stored run is not repeated"
        payload = store.read(a)
        assert list(payload["streams"]) == ["s0/s5", "s1/s5", "s2/s5"] and all(set(v) == set(SERIES) for v in payload["streams"].values())
        assert all(w == 0.5 for ws in payload["gate_w"].values() for w in ws), "beta 0: the gate weight is exactly 0.5"
        c = store.cell(a)
        assert c.acc["gate"].shape == (3,) and np.allclose(c.acc["gate"], [payload["streams"][k]["duo"]["accuracy"] for k in payload["streams"]]) and not any(c.dead.values())
        b = Run(1e-2, 1e-2, 4.0, 0)
        execute(ctx, segs, temps, b, store=store, proxy="msp", b_t=64)
        assert not all(w == 0.5 for ws in store.read(b)["gate_w"].values() for w in ws), "beta > 0: the weight moves"
        assert Store(tmp / "tune", floor=2.0).cell(a).dead["drift"] and not store.cell(a).dead["drift"], "the drift guard: late accuracy below the frozen duo's floor"
        assert "not run" in runs_table(store, [a, Run(1, 1, 1, 0)]) and "drift" in runs_table(Store(tmp / "tune", floor=2.0), [a])
        boom = Run(1e30, 1e30, 0.5, 0)
        execute(ctx, segs, temps, boom, store=store, proxy="nnn", b_t=64)
        assert store.has(boom), "a diverging rate is stored (and then read as a bad cell), not a crash"

    # ---- the search on synthetic surfaces ----------------------------------------------------------------------------------------
    def make_surface(best_beta: float, pair_of_beta: Callable[[float], tuple[float, float]], kill_above: int | None = None, noise: float = 4e-4):
        """duo accuracy over (rungs, beta): a bowl around a pair that moves with beta, and a bowl in beta; rungs above `kill_above` kill a member."""
        def surface(key, rot):
            (kl, ks), beta = key
            ol, os_ = pair_of_beta(beta)
            base = 0.6 - 0.004 * (kl - ol) ** 2 - 0.003 * (ks - os_) ** 2 - 0.01 * (beta - best_beta) ** 2
            acc = base + np.random.default_rng(abs(hash((kl, ks, beta, rot))) % 2 ** 32).normal(0, noise, 4)
            dead = {"large": kill_above is not None and kl > kill_above, "small": False, "drift": False}
            return TU.Cell({"gate": acc, "large": acc, "small": acc}, {"large": 0.0, "small": 0.0}, dead, 200000)
        return surface

    def drive(surface, **kw):
        have = {}
        for _ in range(80):
            pl = plan(lambda key, rot: have.get((key, rot)), **kw)
            if not pl.requests:
                return pl, have
            for key, rot in pl.requests:
                have[(key, rot)] = surface(key, rot)
        raise AssertionError("did not finish")

    # (1) the pair optimum does not depend on beta: found at every beta, in a few waves, and the choice is the best beta
    pl, have = drive(make_surface(0.5, lambda b: (1, -1)))
    assert pl.final == ((1, -1), 0.5) and pl.converged and {k[1] for k, _ in have} == set(BETAS), (pl.final, pl.notes)
    assert [r["beta"] for r in pl.table] == sorted(BETAS) and sum(r["chosen"] for r in pl.table) == 1 and all(r["pair"] == (1, -1) for r in pl.table)
    # (2) the gate lets the rates go higher: the optimum pair moves with beta along a RIDGE; the pair is searched at every beta, so it is found
    #     (a pair fixed at one beta and then a beta chosen would sit at beta 0.5 here: beta 1 only pays off together with its own pair)
    pl, have = drive(make_surface(1.0, lambda b: (int(round(2 * b)), int(round(2 * b)) - 2)))
    assert pl.final == ((2, 0), 1.0) and pl.converged, (pl.final, pl.notes)
    assert {r["beta"]: r["pair"] for r in pl.table} == {0.0: (0, -2), 0.25: (0, -2), 0.5: (1, -1), 0.75: (2, 0), 1.0: (2, 0), 2.0: (4, 2)}, pl.table      # beta 2 was tried: 1 was the sharpest
    # (3) the sharpest gate is the best: beta is extended above the grid, one value at a time, and a plateau past the cap is reported
    pl, have = drive(make_surface(3.0, lambda b: (0, 0)))
    assert pl.final[1] in (2.0, 4.0) and {k[1] for k, _ in have} >= {2.0}, (pl.final, pl.notes)
    pl, have = drive(make_surface(9.0, lambda b: (0, 0)))
    assert pl.final[1] == 4.0 and any("plateau" in n for n in pl.notes) and {k[1] for k, _ in have} == set(BETAS) | {2.0, 4.0}, (pl.final, pl.notes)
    # (4) the gate does not help: beta 0 wins, reported and not extended
    pl, have = drive(make_surface(0.0, lambda b: (0, 0)))
    assert pl.final == ((0, 0), 0.0) and any("does not help" in n for n in pl.notes) and {k[1] for k, _ in have} == set(BETAS), pl.notes
    # (5) a rung that kills a member is never chosen, however good the rest of its surface looks
    pl, have = drive(make_surface(0.5, lambda b: (4, 0), kill_above=2))
    assert pl.final[0][0] <= 2, pl.final
    # (6) a beta at which every cell is dead is left out of the choice; nothing is requested twice, and a finished search asks for nothing
    def dead_at_one(key, rot):
        c = make_surface(0.5, lambda b: (0, 0))(key, rot)
        if key[1] == 1.0:
            c.dead["large"] = True
        return c
    pl, have = drive(dead_at_one)
    assert pl.final[1] != 1.0 and 1.0 not in [r["beta"] for r in pl.table], pl.notes
    pl2 = plan(lambda key, rot: have.get((key, rot)))
    assert pl2.requests == [] and pl2.final == pl.final
    # (7) two seeds: the argmax and its neighbours are also run on the second, at every beta
    pl, have = drive(make_surface(0.5, lambda b: (0, 0)), rots=(0, 1))
    assert pl.final == ((0, 0), 0.5) and any(rot == 1 for _, rot in have) and sum(rot == 1 for _, rot in have) < sum(rot == 0 for _, rot in have)
    # (8) the first wave is the start diagonal of every beta: a whole wave in parallel
    first = plan(lambda key, rot: None)
    assert sorted(first.requests) == sorted((((k, k), b), 0) for b in BETAS for k in START_RUNGS), first.requests

    # (9) the worst_shift criterion: a stream that collapses as the rates rise; the search finishes, and its pair is not above the mean criterion's
    WS = C.Criterion.parse("worst_shift", 1.0)
    base_surface = make_surface(0.5, lambda b: (2, 0))

    def collapsing(key, rot):
        c = base_surface(key, rot)
        (kl, ks), beta = key
        d = 0.05 * max(0, kl + ks + 1)
        acc = c.acc["gate"].copy()
        acc[0] -= d
        acc[1:] += d / 3                                                    # the mean is kept, one stream pays for it
        return TU.Cell({"gate": acc, "large": acc, "small": acc}, c.resets, c.dead, c.seen)
    pl_mean, _ = drive(collapsing)
    pl_ws, have_ws = drive(collapsing, criterion=WS)
    assert pl_ws.final is not None and pl_ws.converged and sum(pl_ws.final[0]) <= sum(pl_mean.final[0]), (pl_mean.final, pl_ws.final, pl_ws.notes)
    assert any("points of the best mean" in n for n in pl_ws.notes), pl_ws.notes
    assert plan(lambda key, rot: have_ws.get((key, rot)), criterion=WS).requests == [], "a finished search asks for nothing"

    # ---- the confirmation, the submission --------------------------------------------------------------------------------------------
    cr = confirm_runs((3.2e-3, 4e-3), (2.26e-3, 4e-3), (4.53e-3, 4e-3), 0.5, [0, 1, 2])
    assert len(cr) == 15 and len(set(cr)) == 15 and cr[0] == Run(3.2e-3, 4e-3, 0.0, 0) and Run(4.53e-3, 4e-3, 0.5, 2) in cr
    assert len(confirm_runs((1e-3, 1e-3), (1e-3, 1e-3), (1e-3, 1e-3), 0.0, [0, 1])) == 2, "nothing changed: one configuration"
    assert len(confirm_runs((1e-3, 1e-3), (1e-3, 1e-3), (1e-3, 1e-3), 0.5, [0])) == 2, "the pair did not move: the gate on and off"
    argv = sbatch_argv("vitb16_rn50v1", r, ["--bn"], "01:00:00")
    assert argv[:2] == ["sbatch", "-t"] and argv[-2:] == ["--run", r.arg] and "--bn" in argv and argv[argv.index("-J") + 1] == job_name("vitb16_rn50v1", r)
    assert submit("vitb16_rn50v1", [r], ["--bn"], "01:00:00", go=False) == []
    assert chain(Path("/nonexistent"), "vitb16_rn50v1", [job_name("vitb16_rn50v1", r)], ["--duo", "vitb16_rn50v1"], go=False) is True
    print("joint_gate_tuning self-test passed")
