"""
joint_lr_tuning.py
==================
The logic behind scripts/02b_tune_joint_lr.py: choose the two members' learning rates TOGETHER for the Joint mode of
src/experiments/adaptation_modes.py (both members minimize the entropy of the static duo's combined prediction).

Why a tuner of its own. scripts/02_tune_tta.py tunes ONE model on ITS OWN entropy, and what it records in cfgs/duo_lrs/<duo>__indep.yaml is
what the independent modes run with. In the Joint mode the loss is the duo's entropy, a steadier signal than either
member's own (a member that starts to collapse is held up by its partner), so the best rates may be larger. They also
interact: how far one member can move depends on how far the other does. So the grid is over PAIRS (lr_large, lr_small),
around each member's current (tuned) rate; by default 1, 2, 4, 8 and 16 times it.

A grid point is one Joint run (RunSpec mode both_duo, protocol "tuning": the 4 held-out corruptions, each from source
weights) at that pair of rates. Its score is the MACRO accuracy (the mean over the 4 streams, so no stream weighs more
because it has more images) of the static duo (z_L / T_L + z_S / T_S) / 2, the prediction script 03's Joint row scores,
with the worst stream and each member's own accuracy next to it. Two reference runs ride along: the frozen models, and
Independent both at the current rates (the run the temperatures' fit needs anyway, so it is cached). The report then
says what the Joint signal buys at the SAME rates (grid point x1,x1 against Independent both) and what larger rates buy
on top of it (the best point against x1,x1).

Temperatures. The duo entropy depends on T_L and T_S, so they are fixed before the grid and shared by every point (a
pair per point would change the objective being compared): the Fixed TS fitted on the 4 held-out corruptions on the FROZEN
models' logits (adaptation_modes.temperature_spec; the author's decision of 2026-10-22), or any JointFixedTS checkpoint. The
same pair scores every point.

Criterion. The score is the whole-stream online accuracy (the metric the paper reports, on pre-update predictions). Next to it
every point carries its LATE accuracy (the last 20% of each stream); a point whose late duo accuracy is below the frozen
duo's is ruled out by `rank(late_floor=...)`: an adaptation that ends worse than not adapting is drift, however good its mean.

A learning rate that is too large can make the logits non-finite, which the member cache refuses to store. Such a run is
recorded as DIVERGED (a marker next to its cache), counts as complete, and ranks below every finite one.

Everything is resumable and shardable: the member logits of each (point, stream) are cached under a key that holds both
learning rates and the temperatures, so a killed job resumes at its first missing stream and a SLURM array takes one
point per task.
"""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments.cache import MemberCache
from src.experiments.protocols import Segment, build_segments
from src.experiments.runner import Context, ensure_members, evaluate, run_dir
from src.experiments.specs import GateSpec, RunSpec

ENGINE_MODE = {"joint": "both_duo", "indep": "both_indep", "frozen": "no_adapt"}
DIVERGED = "diverged.json"            # written next to a run's cache when its logits went non-finite
DEFAULT_MULTS = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------------------------------------- grid

def lr_axis(base_lr: float, mults: Sequence[float], explicit: Sequence[float] | None = None) -> list[float]:
    """The learning rates tried for one member, ascending and without duplicates: `explicit` if given, else `mults`
    times the member's current rate. Rounded to 6 significant digits, so that 5e-5 * 4 is 0.0002 and not 0.00019999."""
    values = list(explicit) if explicit else [base_lr * m for m in mults]
    axis = sorted({float(f"{v:.6g}") for v in values})
    if not axis or axis[0] <= 0:
        raise ValueError(f"learning rates must be positive and there must be at least one, got {values}")
    return axis


@dataclass(frozen=True)
class Job:
    """One run: a Joint run at (lr_large, lr_small), or a reference ("indep": Independent both at the current rates,
    "frozen": no adaptation), for one stream-order seed."""
    kind: str
    seed: int
    lr_large: float | None = None
    lr_small: float | None = None

    def __post_init__(self):
        if self.kind not in ENGINE_MODE or (self.kind == "joint") != (self.lr_large is not None and self.lr_small is not None):
            raise ValueError(f"bad job {self}: only a joint job has learning rates, and it needs both")

    @property
    def label(self) -> str:                  # the --only / --shard label
        head = f"joint/L{self.lr_large:g}_S{self.lr_small:g}" if self.kind == "joint" else self.kind
        return f"{head}/seed{self.seed}"

    @property
    def point(self) -> tuple:                # what the seeds of one grid point share
        return (self.kind, self.lr_large, self.lr_small)


def build_jobs(lrs_large: Sequence[float], lrs_small: Sequence[float], seeds: Sequence[int]) -> list[Job]:
    """The grid points first, so that an array with one task per point gives each its own task, then the references
    (which wrap around onto the first tasks: Independent both is cached already, the frozen run is forward-only)."""
    jobs = [Job("joint", s, lr_l, lr_s) for s in seeds for lr_l in lrs_large for lr_s in lrs_small]
    return jobs + [Job(kind, s) for s in seeds for kind in ("frozen", "indep")]


_LABEL = re.compile(r"^joint/L(?P<l>[^_/]+)_S(?P<s>[^/]+)/seed(?P<seed>\d+)$")


def _job_of_dir(d: Path) -> Job | None:
    """The Joint job whose run lives in cache folder `d`, read back from what the run left there (a divergence marker names its
    job; a finished or partial run has a manifest with the spec and the optimizers it used); None for any other kind of run."""
    marker, manifest = d / DIVERGED, d / "manifest.json"
    try:
        if marker.is_file():
            m = _LABEL.match(json.loads(marker.read_text())["job"])
            return Job("joint", int(m["seed"]), float(m["l"]), float(m["s"])) if m else None
        if manifest.is_file():
            info = json.loads(manifest.read_text())
            if info["spec"]["mode"] != "both_duo":
                return None
            return Job("joint", int(info["spec"]["seed"]), float(info["duo"]["LARGE"]["OPTIM"]["LR"]),
                       float(info["duo"]["SMALL"]["OPTIM"]["LR"]))
    except (KeyError, ValueError, TypeError):
        return None
    return None


def discover_jobs(tuner: "Tuner", seeds: Sequence[int]) -> tuple[list[Job], dict]:
    """The runs in the member cache that this tuner could have made (`--rank_cache`): every Joint run of the duo and TTA method whose
    cache folder is exactly the one the tuner would use for its pair of learning rates (so the input pipeline, temperatures, batch
    size, seed, number of samples and optimizer settings all match), whatever grid flags it was run with. Returns (the complete
    ones, grid points first and then the references, and a description of what was left out): {"left_out": runs of other settings,
    "incomplete": labels of runs with missing streams, "no_reference": labels of the references that are not there}."""
    ctx = tuner.ctx
    root = ctx.g.path("cache") / "members" / ctx.duo["NAME"] / tuner.tta
    info = {"left_out": 0, "incomplete": [], "no_reference": []}
    points: dict[Job, Path] = {}
    for d in sorted(root.glob("independent_seed*_*")) if root.is_dir() else []:
        job = _job_of_dir(d)
        if job is None or job.seed not in seeds:
            continue
        if run_dir(tuner.spec(job), tuner.context(job)) != d:
            info["left_out"] += 1
            continue
        points[job] = d
    jobs = []
    for job in sorted(points, key=lambda j: (j.seed, j.lr_large, j.lr_small)):
        if tuner.missing([job]):
            info["incomplete"].append(job.label)
        else:
            jobs.append(job)
    for job in (Job(kind, s) for s in seeds for kind in ("frozen", "indep")):
        if tuner.missing([job]):
            info["no_reference"].append(job.label)
        else:
            jobs.append(job)
    return jobs, info


def cached_temperatures(ctx: Context, tta: str, num_samples: int | None, seeds: Sequence[int]) -> list[tuple[float, float]]:
    """The static duos (T_L, T_S) that the Joint runs in the cache were made with (the `loop_ts` of each run's manifest), the
    pair with the most runs first. They depend on the independent rates the temperatures were once fitted at, so after a change
    of those rates the current members no longer give the pair the grid was run with."""
    root = ctx.g.path("cache") / "members" / ctx.duo["NAME"] / tta
    counts: Counter = Counter()
    for d in sorted(root.glob("independent_seed*_*")) if root.is_dir() else []:
        try:
            sp = json.loads((d / "manifest.json").read_text())["spec"]
        except (OSError, KeyError, ValueError):
            continue
        if sp["mode"] == "both_duo" and sp["tta"] == tta and sp["seed"] in seeds and sp.get("num_samples") == num_samples and sp.get("loop_ts"):
            counts[tuple(float(x) for x in sp["loop_ts"])] += 1
    return [pair for pair, _ in counts.most_common()]


def find_temps_checkpoint(ctx: Context, tta: str, t_l: float, t_s: float) -> Path | None:
    """The held-out JointFixedTS checkpoint of this duo (fitted on the frozen models: it does not depend on the TTA method, `tta`
    is kept for the callers) that holds the pair (T_L, T_S), if there is one."""
    for folder in sorted((ctx.g.path("checkpoints") / "fixed_ts").glob(f"heldout_{ctx.duo['NAME']}_frozen_*")):
        try:
            tl, ts = AM.load_temperatures(folder)
        except (OSError, KeyError, ValueError):
            continue
        if abs(tl - t_l) < 1e-9 and abs(ts - t_s) < 1e-9:
            return folder
    return None


def rank_cache_setup(ctx: Context, tta: str, tta_kwargs: dict | None, num_samples: int | None, seeds: Sequence[int],
                     temps_checkpoint: str | None = None, segments: list[Segment] | None = None):
    """Everything `--rank_cache` needs, from the cache alone: (tuner, jobs, info); the tuner is None if the cache holds no Joint run
    of this duo and TTA method whose temperatures are known (no run, or no checkpoint with its pair). The
    temperatures are those the cached runs were made with: with several pairs in the cache, the one under which the most complete
    runs are valid; `temps_checkpoint` names the pair instead. `info` is discover_jobs's, plus "temperatures": the pairs found and
    the runs valid under each."""
    if temps_checkpoint:
        spec = AM.temperature_spec(ctx.duo["NAME"], tta, tta_kwargs, num_samples)
        options = [AM.resolve_temperatures(ctx, spec, temps_checkpoint, report_only=True)]
        skipped = []
    else:
        options, skipped = [], []
        for t_l, t_s in cached_temperatures(ctx, tta, num_samples, seeds):
            folder = find_temps_checkpoint(ctx, tta, t_l, t_s)
            if folder is None:
                skipped.append((t_l, t_s))
            else:
                options.append(AM.Temps(folder, t_l, t_s, f"taken from the cached runs, which were made with the checkpoint \\texttt{{{folder.name}}}"))
    best, valid = None, []
    for temps in options:
        tuner = Tuner(ctx, tta, tta_kwargs, num_samples, temps, segments=segments)
        jobs, info = discover_jobs(tuner, seeds)
        n = sum(j.kind == "joint" for j in jobs)
        valid.append(((temps.T_l, temps.T_s), n))
        if best is None or n > best[0]:
            best = (n, tuner, jobs, info)
    if best is None:
        return None, [], {"left_out": 0, "incomplete": [], "no_reference": [], "temperatures": [], "no_checkpoint": skipped}
    _, tuner, jobs, info = best
    return tuner, jobs, {**info, "temperatures": valid, "no_checkpoint": skipped}


# ------------------------------------------------------------------------------------------------------ running

@dataclass
class Tuner:
    ctx: Context                         # at the CURRENT (tuned) learning rates: what the references run with
    tta: str
    tta_kwargs: dict | None
    num_samples: int | None
    temps: AM.Temps                      # T_L, T_S of the static duo: the loop's and the scoring one
    segments: list[Segment] | None = None   # None = the protocol's 4 held-out corruptions (tests pass fake streams)

    def spec(self, job: Job) -> RunSpec:
        return RunSpec(self.ctx.duo["NAME"], tta=self.tta, tta_kwargs=self.tta_kwargs or {}, mode=ENGINE_MODE[job.kind],
                       protocol="tuning", seed=job.seed, num_samples=self.num_samples,
                       loop_ts=(self.temps.T_l, self.temps.T_s) if job.kind == "joint" else None)

    def context(self, job: Job) -> Context:
        return AM.with_learning_rates(self.ctx, job.lr_large, job.lr_small) if job.kind == "joint" else self.ctx

    def _marker(self, job: Job) -> Path:
        return run_dir(self.spec(job), self.context(job)) / DIVERGED

    def compute(self, jobs: Sequence[Job]) -> None:
        """Adapt the members of every job (a GPU job); streams already cached are skipped."""
        for job in jobs:
            if self._marker(job).is_file():
                print(f"== {job.label}: recorded as diverged, skipped")
                continue
            print(f"== {job.label}")
            try:
                ensure_members(self.spec(job), self.context(job), self.segments)
            except ValueError as e:
                if "non-finite" not in str(e):
                    raise
                marker = self._marker(job)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"job": job.label, "error": str(e)}))
                print(f"{job.label}: the logits went non-finite (learning rates too large): recorded as diverged")

    def missing(self, jobs: Sequence[Job]) -> dict[str, list[str]]:
        """{job label: streams not cached yet}, leaving out complete and diverged jobs."""
        out = {}
        for job in jobs:
            if self._marker(job).is_file():
                continue
            cache = MemberCache(run_dir(self.spec(job), self.context(job)))
            segments = self.segments or build_segments(self.spec(job), self.ctx.g)
            if gaps := [s.label for s in segments if not cache.has(s.file_stem)]:
                out[job.label] = gaps
        return out

    def score(self, jobs: Sequence[Job]) -> dict[tuple, dict]:
        """{job.point: summary} from the cached logits, scored by the static duo; the seeds of a point are pooled. Needs
        every job complete (FileNotFoundError naming the gaps otherwise)."""
        gate = GateSpec("duo", kind="fixed_ts", fixed_ts=str(self.temps.folder))
        rows: dict[tuple, list[dict]] = defaultdict(list)
        late: dict[tuple, list[dict]] = defaultdict(list)
        diverged = set()
        for job in jobs:
            if self._marker(job).is_file():
                diverged.add(job.point)
                continue
            res = evaluate(self.spec(job), self.context(job), [gate], report_only=True, segments=self.segments)
            rows[job.point] += res.rows
            late[job.point].append({series: late_accuracy(res, series) for series in ("duo", "large", "small")})
        return {**{p: summarize(r, late[p]) for p, r in rows.items() if p not in diverged},
                **{p: DIVERGED_SUMMARY for p in diverged}}


# ------------------------------------------------------------------------------------------------------ scoring

NAN = float("nan")
LATE_FRAC = 0.2           # the last fraction of each stream that the late accuracy covers
DIVERGED_SUMMARY = {"diverged": True, "n_seeds": 0, "duo": NAN, "large": NAN, "small": NAN, "duo_ece": NAN, "duo_nll": NAN,
                    "worst": NAN, "worst_stream": None, "streams": {}, "late_duo": NAN, "late_large": NAN, "late_small": NAN}


def late_accuracy(res, series: str, frac: float = LATE_FRAC) -> dict[str, float]:
    """{segment: accuracy over the last `frac` of the segment's samples (whole batches, from the end), in the order the
    stream delivered them} for `series` of an `evaluate` result (its per-batch curves)."""
    curve = res.curves[series]
    n, k = np.asarray(curve["batch_n"]), np.asarray(curve["batch_correct"])
    starts = [b for _, b in res.segment_batches] + [len(n)]
    out = {}
    for (label, _), a, b in zip(res.segment_batches, starts[:-1], starts[1:]):
        nn, kk = n[a:b], k[a:b]
        need, got, i = frac * nn.sum(), 0, len(nn)
        while i > 0 and got < need:
            i -= 1
            got += nn[i]
        out[label] = float(kk[i:].sum() / nn[i:].sum())
    return out


def summarize(rows: Sequence[dict], late: Sequence[dict] | None = None) -> dict:
    """One grid point from the rows of `evaluate` (series "duo", "large", "small"; one row per stream and seed).
    Macro means: over the streams first, then over the seeds. `late`: one {series: late_accuracy(...)} per seed (the
    `late_*` fields are the same macro means of the last 20% of each stream; NaN without it)."""
    df = pd.DataFrame(rows)
    macro = df.groupby(["seed", "series"])[["accuracy", "ece", "nll"]].mean().groupby("series").mean()
    duo_streams = df[df["series"] == "duo"].groupby("segment")["accuracy"].mean()
    out = {"diverged": False, "n_seeds": int(df["seed"].nunique()), "duo": float(macro.loc["duo", "accuracy"]), "large": float(macro.loc["large", "accuracy"]),
           "small": float(macro.loc["small", "accuracy"]), "duo_ece": float(macro.loc["duo", "ece"]),
           "duo_nll": float(macro.loc["duo", "nll"]), "worst": float(duo_streams.min()),
           "worst_stream": str(duo_streams.idxmin()), "streams": {k: float(v) for k, v in duo_streams.items()}}
    for series in ("duo", "large", "small"):
        out[f"late_{series}"] = float(np.mean([np.mean(list(per[series].values())) for per in late])) if late else NAN
    return out


def joint_points(summaries: dict[tuple, dict]) -> list[dict]:
    """The grid points as flat dicts (learning rates + summary), in grid order."""
    return [{"lr_large": p[1], "lr_small": p[2], **s} for p, s in summaries.items() if p[0] == "joint"]


def rank(points: Sequence[dict], late_floor: float | None = None) -> list[dict]:
    """Finite points, best duo accuracy first. A tie goes to the smaller pair of rates (the smaller product, then the
    smaller large-member rate), the repo's convention of preferring the more conservative setting. With `late_floor` (the frozen
    duo's late accuracy) a point whose late duo accuracy is below it is ruled out (`guard_excluded` lists them); if that
    would leave nothing, the guard is dropped (the caller says so)."""
    finite = [p for p in points if not p["diverged"]]
    if late_floor is not None and (kept := [p for p in finite if not p.get("late_duo", NAN) < late_floor]):
        finite = kept
    return sorted(finite, key=lambda p: (-round(p["duo"], 12), p["lr_large"] * p["lr_small"], p["lr_large"]))


def guard_excluded(points: Sequence[dict], late_floor: float | None) -> list[dict]:
    """The finite points that `rank(late_floor=...)` rules out."""
    if late_floor is None:
        return []
    return [p for p in points if not p["diverged"] and p.get("late_duo", NAN) < late_floor]


def edge_warnings(best: dict, lrs_large: Sequence[float], lrs_small: Sequence[float]) -> list[str]:
    """The winner sits at the end of a member's range: the optimum may lie outside the grid."""
    out = []
    for side, lrs, flag in (("large", lrs_large, "--lrs_large"), ("small", lrs_small, "--lrs_small")):
        lr = best[f"lr_{side}"]
        if len(lrs) > 1 and lr in (min(lrs), max(lrs)):
            end = "smallest" if lr == min(lrs) else "largest"
            out.append(f"the best {side}-member learning rate ({lr:g}) is the {end} one tried ({min(lrs):g}..{max(lrs):g}): "
                       f"the optimum may lie beyond it; add values with {flag} (or --mults); finished points are reused")
    return out


# ------------------------------------------------------------------------------------------------------ reporting

def _pct(x: float) -> str:
    return "  div." if x != x else f"{100 * x:5.2f}"


def grid_table(points: Sequence[dict], lrs_large: Sequence[float], lrs_small: Sequence[float], key: str = "duo",
               mark: dict | None = None) -> str:
    """Rows: the large member's rate; columns: the small member's. `key` is a summary field (a fraction, printed as
    %). `mark` (a point) gets a trailing `*`. A diverged cell prints as `div.`, a pair that was not run as `-`."""
    cell = {(p["lr_large"], p["lr_small"]): p[key] for p in points}
    corner = "large \\ small"
    head = f"{corner:>14}" + "".join(f"{s:>10g}" for s in lrs_small)
    lines = [head]
    for lr_l in lrs_large:
        cells = []
        for lr_s in lrs_small:
            star = "*" if mark and (lr_l, lr_s) == (mark["lr_large"], mark["lr_small"]) else " "
            cells.append(f"{(_pct(cell[lr_l, lr_s]) if (lr_l, lr_s) in cell else '-'):>9}{star}")
        lines.append(f"{lr_l:>14g}" + "".join(cells))
    return "\n".join(lines)


def ranking_table(ranked: Sequence[dict], top: int = 8) -> str:
    lines = [f"{'rank':>4} {'lr_large':>10} {'lr_small':>10} {'duo':>7} {'worst':>7} {'large':>7} {'small':>7} {'duo nll':>8}"]
    for i, p in enumerate(ranked[:top], 1):
        lines.append(f"{i:>4} {p['lr_large']:>10g} {p['lr_small']:>10g} {_pct(p['duo']):>7} {_pct(p['worst']):>7} "
                     f"{_pct(p['large']):>7} {_pct(p['small']):>7} {p['duo_nll']:>8.3f}")
    return "\n".join(lines)


def comparisons(summaries: dict[tuple, dict], best: dict, base: tuple[float, float]) -> list[str]:
    """The lines that answer the question: against the frozen duo, against Independent both at the current rates, and
    (when the current rates are on the grid) against the Joint run at them."""
    d = lambda a, b: f"{100 * (a - b):+.2f}"                                                      # noqa: E731
    out = []
    frozen, indep = summaries.get(("frozen", None, None)), summaries.get(("indep", None, None))
    at_base = summaries.get(("joint", *base))
    if frozen:
        out.append(f"best joint {_pct(best['duo']).strip()} vs frozen duo {_pct(frozen['duo']).strip()} "
                   f"({d(best['duo'], frozen['duo'])} points)")
    if indep:
        out.append(f"best joint {_pct(best['duo']).strip()} vs Independent both at the current rates "
                   f"{_pct(indep['duo']).strip()} ({d(best['duo'], indep['duo'])} points)")
    if at_base and not at_base["diverged"]:
        out.append(f"Joint at the current rates (L {base[0]:g}, S {base[1]:g}): {_pct(at_base['duo']).strip()}"
                   + (f", {d(at_base['duo'], indep['duo'])} vs Independent both: what the duo signal buys at equal rates"
                      if indep else "")
                   + f"; the best pair adds {d(best['duo'], at_base['duo'])} on top: what the larger rates buy")
    return out


def result_rows(summaries: dict[tuple, dict], ranked: Sequence[dict]) -> list[dict]:
    """One flat row per run (references and diverged points included), ranked joint points first; fractions."""
    place = {(p["lr_large"], p["lr_small"]): i for i, p in enumerate(ranked, 1)}
    rows = []
    for (kind, lr_l, lr_s), s in summaries.items():
        rows.append({"rank": place.get((lr_l, lr_s)) if kind == "joint" else None, "kind": kind, "lr_large": lr_l,
                     "lr_small": lr_s, "diverged": s["diverged"], "duo_acc": s["duo"], "worst_acc": s["worst"],
                     "worst_stream": s["worst_stream"], "large_acc": s["large"], "small_acc": s["small"],
                     "duo_ece": s["duo_ece"], "duo_nll": s["duo_nll"], "late_duo": s.get("late_duo", NAN),
                     "late_large": s.get("late_large", NAN), "late_small": s.get("late_small", NAN),
                     **{f"acc/{k}": v for k, v in s["streams"].items()}})
    return sorted(rows, key=lambda r: (r["kind"] != "joint", r["rank"] is None, r["rank"] or 0, r["lr_large"] or 0,
                                       r["lr_small"] or 0, r["kind"]))


def evaluate_command(*, duo: str, tta: str, tta_kwargs: dict | None, bn: bool, temps_folder: Path,
                     lr_large: float | None = None, lr_small: float | None = None) -> str:
    """The script-03 command that scores the chosen pair on the 15 test corruptions: the Joint mode only, with the SAME
    temperatures as the tuning (03 fits its own on the indep regime's members otherwise). Script 03 reads the pair from
    cfgs/duo_lrs/<duo>__joint.yaml, where `record_best` has put it; `lr_large` / `lr_small` add `--joint_lr_*` to give it by
    hand instead (that run's files then carry a rates tag and a time stamp)."""
    parts = ["python", "scripts/03_tab_adaptation_modes.py", "--duo", duo, "--tta", tta]
    if tta_kwargs:
        parts += ["--tta_kwargs", json.dumps(tta_kwargs)]
    parts += (["--bn"] if bn else []) + ["--modes", "joint"]
    for flag, lr in (("--joint_lr_large", lr_large), ("--joint_lr_small", lr_small)):
        if lr is not None:
            parts += [flag, f"{lr:g}"]
    return shlex.join(parts + ["--temps_checkpoint", str(temps_folder)])


def record_best(results_json: Path, root: Path | None = None, duos_dir: Path | None = None) -> str:
    """Record the best pair of a finished full run (its results.json) as the joint learning rates of the duo for that TTA
    method, in cfgs/duo_lrs/<duo>__joint.yaml (`root`: that folder): what script 03 and the reset scripts read. It replaces
    the pair already there, and a `pending:` note. Returns the line to print."""
    from src.utils import duo_lrs

    r = json.loads(Path(results_json).read_text())
    duo, tta = r["duo"], r["tta"]
    old = None
    try:
        entry = duo_lrs.load(duo, "joint", root)["methods"].get(tta)
        old = (float(entry["large"]["OPTIM"]["LR"]), float(entry["small"]["OPTIM"]["LR"])) if entry else None
    except FileNotFoundError:
        pass
    path = duo_lrs.import_joint(results_json, root=root, duos_dir=duos_dir)
    new = (float(r["best"]["lr_large"]), float(r["best"]["lr_small"]))
    note = ("" if old is None else " (unchanged)" if old == new else f" (replacing large {old[0]:g}, small {old[1]:g})")
    return (f"recorded the best pair, large {new[0]:g} and small {new[1]:g}, as the joint learning rates of "
            f"{duo} / {tta} in {path}{note}")


def optimizer_text(optim: dict) -> str:
    extra = f"momentum {optim['MOMENTUM']:g}" if optim["METHOD"] == "SGD" else f"beta1 {optim['BETA']:g}"
    return f"{optim['METHOD']} ({extra}, weight decay {optim['WD']:g}, {int(optim.get('STEPS', 1))} step per batch)"


# ----------------------------------------------------------------------------------------------- script plumbing

def add_args(ap) -> None:
    ap.add_argument("--mults", type=float, nargs="+", default=list(DEFAULT_MULTS), metavar="X",
                    help="the grid of each member: these multiples of its CURRENT learning rate (cfgs/duo_lrs/<duo>__indep.yaml, "
                         "else its default). Default: %(default)s")
    ap.add_argument("--lrs_large", type=float, nargs="+", default=None, metavar="LR",
                    help="explicit learning rates of the large member (replaces --mults for it)")
    ap.add_argument("--lrs_small", type=float, nargs="+", default=None, metavar="LR",
                    help="explicit learning rates of the small member (replaces --mults for it)")
    ap.add_argument("--rank_cache", action="store_true",
                    help="rank EVERY Joint run in the member cache that matches the other settings (duo, --tta, --seeds, --bn, "
                         "temperatures), whatever grid it was run with; --mults / --lrs_large / --lrs_small are ignored and "
                         "nothing is computed (no GPU). The best pair is not recorded unless --record is given")
    ap.add_argument("--record", action="store_true",
                    help="with --rank_cache: record the best pair in cfgs/duo_lrs/<duo>__joint.yaml (a full --report_only run does)")
    ap.add_argument("--min_seeds", type=int, default=1, metavar="N",
                    help="only grid points run with at least N of the --seeds compete (the final choice among the finalists, which are the "
                         "only points run on every seed)")
    ap.add_argument("--temps_checkpoint", default=None, metavar="DIR",
                    help="JointFixedTS checkpoint (a folder with config.json) for the duo's temperatures; default: "
                         "the Fixed TS fitted on the FROZEN models (never refitted per pair)")
    ap.add_argument("--temps_only", action="store_true",
                    help="fit and store the temperatures, then stop (run this once before a --shard array so that "
                         "no two tasks fit them at the same time)")


if __name__ == "__main__":
    import math
    import tempfile

    import torch

    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.utils.config import load_global

    torch.set_num_threads(2)
    g = load_global()

    # ---- the grid ------------------------------------------------------------------------------------------------
    assert lr_axis(5e-5, (1, 2, 4)) == [5e-5, 1e-4, 2e-4], "5e-5 * 4 must be 2e-4, not 0.00019999999999999998"
    assert lr_axis(5e-5, (4, 1, 2, 2)) == [5e-5, 1e-4, 2e-4], "sorted and without duplicates"
    assert lr_axis(5e-5, (1, 2), explicit=[3e-4, 1e-4]) == [1e-4, 3e-4], "explicit rates replace the multiples"
    for bad in (dict(base_lr=1e-3, mults=()), dict(base_lr=1e-3, mults=(0.0,)), dict(base_lr=1e-3, mults=(1,), explicit=[-1e-3])):
        try:
            lr_axis(**bad)
            raise AssertionError(f"must raise: {bad}")
        except ValueError:
            pass
    jobs = build_jobs([1e-3, 2e-3], [1e-4, 2e-4, 4e-4], [0, 1])
    assert len(jobs) == 2 * (6 + 2) and len({j.label for j in jobs}) == len(jobs)
    assert [j.kind for j in jobs[:6]] == ["joint"] * 6 and {j.kind for j in jobs[-4:]} == {"frozen", "indep"}, \
        "points first, references last"
    assert jobs[0].label == "joint/L0.001_S0.0001/seed0" and jobs[-1].label == "indep/seed1"
    for bad in (dict(kind="joint", seed=0), dict(kind="indep", seed=0, lr_large=1e-3, lr_small=1e-3), dict(kind="x", seed=0)):
        try:
            Job(**bad)
            raise AssertionError(f"must raise: {bad}")
        except ValueError:
            pass

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tiny_models(0)
        ctx = make_context(tmp, batch_size=32, base=base)
        ctx.g.raw["paths"]["checkpoints"] = str(tmp / "checkpoints")
        ctx.g.raw["runtime"]["workers"] = 0              # tiny in-memory streams: no DataLoader processes needed
        held =[Segment(f"{c}/s5", FakeStream(150, seed=10 + i), True) for i, c in enumerate(g.tuning_corruptions)]
        tspec = AM.temperature_spec("tiny", "tent", None, None)
        temps = AM.resolve_temperatures(ctx, tspec, segments=held)
        tuner = Tuner(ctx, "tent", None, None, temps, segments=held)
        base_lr = ctx.duo["LARGE"]["OPTIM"]["LR"]
        lrs_l, lrs_s = lr_axis(base_lr, (1, 4)), lr_axis(base_lr, (0.5, 1))
        grid = build_jobs(lrs_l, lrs_s, [0])
        base_pair = (float(f"{base_lr:.6g}"),) * 2

        # ---- what each job is: mode, protocol, temperatures only where the loop needs them -----------------------
        for job in grid:
            s = tuner.spec(job)
            assert s.protocol == "tuning" and s.mode == ENGINE_MODE[job.kind] and s.seed == 0
            assert (s.loop_ts == (temps.T_l, temps.T_s)) == (job.kind == "joint")
        assert len({tuner.spec(j).fingerprint(tuner.context(j).duo, 32, "masked") for j in grid}) == len(grid), \
            "every job has its own cache: each learning rate and each mode is part of the key"
        # the tuning point IS script 03's Joint run: same spec up to the stream list, so the same cache key space
        j = Job("joint", 0, base_pair[0], base_pair[1])
        mine = tuner.spec(j).fingerprint(tuner.context(j).duo, 32, "masked")
        theirs = AM.mode_spec(AM.MODE_BY_KEY["joint"], "tiny", "tent", None, 0, None, temps).fingerprint(ctx.duo, 32, "masked")
        assert mine == theirs, "the tuner's Joint run must be script 03's Joint run"
        assert tuner.spec(Job("frozen", 0)).fingerprint(ctx.duo, 32, "masked") == tspec.fingerprint(ctx.duo, 32, "masked") and \
            run_dir(tuner.spec(Job("frozen", 0)), ctx) == run_dir(tspec, ctx), \
            "the frozen reference is the run the temperatures were fitted on (already cached)"

        # ---- running: nothing is cached, then everything; a repeat recomputes nothing --------------------------------
        assert set(tuner.missing(grid)) == {j.label for j in grid if j.kind != "frozen"}, "the temperatures' fit cached 'frozen'"
        try:
            tuner.score(grid)
            raise AssertionError("scoring an incomplete grid must fail, not guess")
        except FileNotFoundError as e:
            assert "missing" in str(e)
        tuner.compute(grid)
        assert tuner.missing(grid) == {}
        files = {p: p.stat().st_mtime_ns for j in grid for p in MemberCache(run_dir(tuner.spec(j), tuner.context(j))).dir.glob("*.pt")}
        assert len(files) == len(grid) * len(held)
        tuner.compute(grid)
        assert files == {p: p.stat().st_mtime_ns for p in files}, "a finished job is not recomputed"

        # ---- the score is the static duo's macro accuracy, read off the cached logits ---------------------------------
        summaries = tuner.score(grid)
        assert len(summaries) == len(lrs_l) * len(lrs_s) + 2
        for job in grid:
            cache = MemberCache(run_dir(tuner.spec(job), tuner.context(job)))
            acc = {"duo": [], "large": [], "small": []}
            for seg in held:
                d = cache.read(seg.file_stem)
                duo = 0.5 * d["z_l"] / temps.T_l + 0.5 * d["z_s"] / temps.T_s
                for name, z in (("duo", duo), ("large", d["z_l"]), ("small", d["z_s"])):
                    acc[name].append(float((z.argmax(1) == d["labels"]).float().mean()))
            s = summaries[job.point]
            assert not s["diverged"] and set(s["streams"]) == {x.label for x in held}
            for name in acc:
                assert abs(s[name] - sum(acc[name]) / 4) < 1e-9, (job.label, name)
            assert abs(s["worst"] - min(acc["duo"])) < 1e-9 and s["streams"][s["worst_stream"]] == s["worst"]
        pts = [j for j in grid if j.kind == "joint"]
        a, b = (MemberCache(run_dir(tuner.spec(j), tuner.context(j))).read(held[0].file_stem) for j in (pts[0], pts[-1]))
        assert not a["z_l"].equal(b["z_l"]), "other learning rates, other logits"
        indep = MemberCache(run_dir(tuner.spec(Job("indep", 0)), ctx)).read(held[0].file_stem)
        same = MemberCache(run_dir(tuner.spec(Job("joint", 0, *base_pair)), ctx)).read(held[0].file_stem)
        assert not indep["z_l"].equal(same["z_l"]), "the duo's entropy is not the member's own: Joint differs from Independent"

        # ---- a diverging run is recorded, counts as complete, ranks last, and is never recomputed ---------------------
        boom = Job("joint", 0, 1e30, base_pair[1])
        assert tuner.missing([boom]) != {}
        tuner.compute([boom])
        assert tuner.missing([boom]) == {} and tuner._marker(boom).is_file(), "non-finite logits: diverged, not a crash"
        sc = tuner.score(grid + [boom])
        assert sc[boom.point]["diverged"] and math.isnan(sc[boom.point]["duo"])
        pts_all = joint_points(sc)
        ranked = rank(pts_all)
        assert len(ranked) == len(pts) and all(not p["diverged"] for p in ranked), "diverged runs are not ranked"
        mtime = tuner._marker(boom).stat().st_mtime_ns
        tuner.compute([boom])
        assert tuner._marker(boom).stat().st_mtime_ns == mtime
        assert "div." in grid_table(pts_all, lrs_l + [1e30], lrs_s) and "div." in ranking_table(ranked + [pts_all[-1]])
        try:
            tuner.compute([Job("joint", 0, -1.0, 1e-3)])
            raise AssertionError("a negative learning rate must raise, not be recorded as divergence")
        except ValueError as e:
            assert "positive" in str(e)

        # ---- --rank_cache: the runs are found in the cache without the grid; what does not belong is left out -----------
        found, info = discover_jobs(tuner, [0])
        want = {j.point for j in grid} | {boom.point}
        assert {j.point for j in found} == want, "every complete Joint run, the diverged one too, and both references"
        assert [j.kind for j in found[-2:]] == ["frozen", "indep"] and info == {"left_out": 0, "incomplete": [], "no_reference": []}
        assert all(j.lr_large == float(f"{j.lr_large:.6g}") for j in found if j.kind == "joint"), "the rates read back are the run's"
        assert not [j for j in discover_jobs(tuner, [1])[0] if j.kind == "joint"], "another seed: none of the runs belongs to it"
        other = AM.Temps(temps.folder, temps.T_l * 1.5, temps.T_s, "test")
        f2, i2 = discover_jobs(Tuner(ctx, "tent", None, None, other, segments=held), [0])
        assert not [j for j in f2 if j.kind == "joint"] and i2["left_out"] == len(pts) + 1, \
            "runs made with other temperatures are not valid for these (and are counted, not guessed at)"
        victim = MemberCache(run_dir(tuner.spec(pts[0]), tuner.context(pts[0]))).dir
        next(victim.glob("*.pt")).unlink()
        f3, i3 = discover_jobs(tuner, [0])
        assert pts[0].label in i3["incomplete"] and pts[0].point not in {j.point for j in f3}, "an incomplete run is reported, not ranked"
        tuner.compute([pts[0]])
        assert pts[0].point in {j.point for j in discover_jobs(tuner, [0])[0]}, "and is found again once it is complete"
        assert "-" in grid_table(pts_all, lrs_l + [3e-3], lrs_s).splitlines()[-1], "a pair that was not run prints as '-'"

        # ---- the temperatures come from the cache too: no flag, and no dependence on the CURRENT independent rates ------
        got, gjobs, ginfo = rank_cache_setup(ctx, "tent", None, None, [0], segments=held)
        assert got.temps.folder == temps.folder and {j.point for j in gjobs} == want and ginfo["temperatures"] == [((temps.T_l, temps.T_s), len(pts) + 1)]
        moved = AM.with_learning_rates(ctx, base_lr * 2, base_lr * 2)                  # the situation after the Tent rates were changed
        got2, jobs2, info2 = rank_cache_setup(moved, "tent", None, None, [0], segments=held)
        assert {j.point for j in jobs2 if j.kind == "joint"} == {j.point for j in found if j.kind == "joint"}, "the same grid is found"
        assert got2.temps.folder == temps.folder and info2["no_reference"] == ["indep/seed0"], \
            "only the Independent reference depends on the current rates, so only it is missing"
        assert rank_cache_setup(ctx, "eta", None, None, [0], segments=held)[0] is None, "no run of that TTA method: nothing to rank"
        assert rank_cache_setup(ctx, "tent", None, None, [0], str(temps.folder), held)[0].temps.folder == temps.folder, "--temps_checkpoint"
        assert find_temps_checkpoint(ctx, "tent", temps.T_l, temps.T_s) == temps.folder and find_temps_checkpoint(ctx, "tent", 9.9, 9.9) is None

    # ---- the late-stream accuracy: the last 20% of each stream, whole batches from the end ---------------------------------
    class _Res:
        pass
    res = _Res()
    res.curves = {"duo": {"batch_n": [10] * 6, "batch_correct": [9, 9, 9, 1, 1, 1]}}
    res.segment_batches = [("a", 0), ("b", 3)]
    assert late_accuracy(res, "duo", 0.2) == {"a": 0.9, "b": 0.1}, "20% of 30 samples is the last batch"
    res.curves = {"duo": {"batch_n": [10] * 6, "batch_correct": [9, 5, 1, 2, 2, 2]}}
    assert late_accuracy(res, "duo", 0.7) == {"a": 0.5, "b": 0.2} and late_accuracy(res, "duo", 1.0)["a"] == 0.5
    res.curves = {"duo": {"batch_n": [10, 10, 4, 10, 10, 6], "batch_correct": [9, 5, 1, 2, 2, 6]}}
    la = late_accuracy(res, "duo", 0.2)
    assert abs(la["a"] - 6 / 14) < 1e-12 and la["b"] == 1.0, "20% of 24 samples is 4.8: the 4-sample last batch is not enough, a second one is added"

    # ---- the guard: an adaptation that ends worse than the frozen duo is ruled out, however good its mean ------------------
    gp = [dict(lr_large=1e-4, lr_small=1e-3, diverged=False, duo=.60, late_duo=.40),
          dict(lr_large=2e-4, lr_small=1e-3, diverged=False, duo=.58, late_duo=.55),
          dict(lr_large=4e-4, lr_small=1e-3, diverged=True, duo=NAN, late_duo=NAN)]
    assert rank(gp)[0]["lr_large"] == 1e-4 and rank(gp, late_floor=0.5)[0]["lr_large"] == 2e-4
    assert [p["lr_large"] for p in guard_excluded(gp, 0.5)] == [1e-4] and guard_excluded(gp, None) == []
    assert rank(gp, late_floor=0.9)[0]["lr_large"] == 1e-4, "a guard that would rule out everything is dropped (the caller says so)"

    # ---- ranking, ties, edges, tables on synthetic summaries -----------------------------------------------------
    def fake(duo, **kw):
        return {**DIVERGED_SUMMARY, "diverged": False, "duo": duo, "large": duo - .05, "small": duo - .1, "worst": duo - .02,
                "worst_stream": "a", "duo_ece": .05, "duo_nll": 1.5, "streams": {"a": duo - .02, "b": duo + .02}, **kw}
    L, S = [1e-4, 2e-4, 4e-4], [1e-3, 2e-3]
    table = {("joint", l, s): fake(acc) for (l, s), acc in zip(((l, s) for l in L for s in S), (.50, .52, .55, .53, .55, .51))}
    table[("frozen", None, None)], table[("indep", None, None)] = fake(.40), fake(.49)
    pts = joint_points(table)
    ranked = rank(pts)
    assert [(p["lr_large"], p["lr_small"]) for p in ranked[:2]] == [(2e-4, 1e-3), (4e-4, 1e-3)], \
        "tie at .55: the smaller product of rates wins"
    assert [p["duo"] for p in ranked] == [.55, .55, .53, .52, .51, .50]
    edge = edge_warnings(ranked[0], L, S)
    assert len(edge) == 1 and "small-member learning rate (0.001) is the smallest" in edge[0] and "--lrs_small" in edge[0]
    assert edge_warnings({"lr_large": 2e-4, "lr_small": 1e-3}, L, [5e-4, 1e-3, 2e-3]) == [], "an interior optimum: no warning"
    both = edge_warnings({"lr_large": 4e-4, "lr_small": 2e-3}, L, S)
    assert len(both) == 2 and "large-member" in both[0] and "largest" in both[0]
    assert edge_warnings({"lr_large": 1e-4, "lr_small": 1e-3}, [1e-4], [1e-3]) == [], "one value tried: no edge"
    gt = grid_table(pts, L, S, mark=ranked[0]).splitlines()
    assert len(gt) == 1 + len(L) and gt[2].count("*") == 1 and "55.00*" in gt[2] and "0.002" in gt[0]
    cmp = comparisons(table, ranked[0], (1e-4, 1e-3))
    assert len(cmp) == 3 and "+15.00 points" in cmp[0] and "+6.00 points" in cmp[1]
    assert "+1.00 vs Independent both" in cmp[2] and "adds +5.00 on top" in cmp[2], cmp
    assert len(comparisons(table, ranked[0], (9e-9, 9e-9))) == 2, "no Joint run at the current rates on the grid: no third line"
    rows = result_rows(table, ranked)
    assert [r["kind"] for r in rows[-2:]] == ["frozen", "indep"] and rows[0]["rank"] == 1 and rows[0]["acc/a"] == ranked[0]["duo"] - .02
    assert sorted(r["rank"] for r in rows if r["kind"] == "joint") == list(range(1, 7))
    json.dumps(rows)

    # ---- seeds: macro over the streams first, then over the seeds --------------------------------------------------
    mk = lambda seed, seg, series, acc: {"seed": seed, "segment": seg, "series": series, "accuracy": acc, "ece": .1, "nll": 1.}   # noqa: E731
    rr = [mk(sd, sg, sr, a) for sd, vals in ((0, {"s1": (.6, .5, .4), "s2": (.2, .3, .4)}), (1, {"s1": (.8, .5, .4), "s2": (.4, .3, .4)}))
          for sg, accs in vals.items() for sr, a in zip(("duo", "large", "small"), accs)]
    sm = summarize(rr)
    assert abs(sm["duo"] - (((.6 + .2) / 2 + (.8 + .4) / 2) / 2)) < 1e-12 and sm["worst_stream"] == "s2"
    assert abs(sm["worst"] - .3) < 1e-12 and abs(sm["small"] - .4) < 1e-12 and sm["n_seeds"] == 2, "the seeds a point was run on are counted"

    # ---- the follow-up command: Joint only, the tuning's own temperatures ---------------------------------------------
    cmd = evaluate_command(duo="vitb16_rn50v1", tta="tent", tta_kwargs=None, bn=True, lr_large=2e-4, lr_small=5e-4,
                           temps_folder=Path("/x/heldout y"))
    assert cmd == ("python scripts/03_tab_adaptation_modes.py --duo vitb16_rn50v1 --tta tent --bn --modes joint "
                   "--joint_lr_large 0.0002 --joint_lr_small 0.0005 --temps_checkpoint '/x/heldout y'"), cmd
    assert evaluate_command(duo="d", tta="tent", tta_kwargs=None, bn=True, temps_folder=Path("t")) == \
        "python scripts/03_tab_adaptation_modes.py --duo d --tta tent --bn --modes joint --temps_checkpoint t", \
        "without rates: script 03 reads the recorded pair"
    plain = evaluate_command(duo="d", tta="eata", tta_kwargs={"d_margin": 0.1}, bn=False, lr_large=1e-3, lr_small=1e-3,
                             temps_folder=Path("t"))
    assert "--tta_kwargs '{\"d_margin\": 0.1}'" in plain and "--bn" not in plain, plain
    assert optimizer_text({"METHOD": "SGD", "MOMENTUM": 0.9, "WD": 0.0, "STEPS": 1}) == \
        "SGD (momentum 0.9, weight decay 0, 1 step per batch)"

    # ---- the best pair is recorded in the duo's joint file, replacing the previous one and the pending note ---------------
    import tempfile

    import yaml

    from src.utils import duo_lrs
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "duos").mkdir()
        (tmp / "duos" / "tiny.yaml").write_text(yaml.safe_dump({"NAME": "tiny", "LARGE": {"NAME": "tiny_ln", "NORM": "LN"},
                                                                 "SMALL": {"NAME": "tiny_bn", "NORM": "BN"}}))
        lrs = tmp / "duo_lrs"
        duo_lrs.init_joint("tiny", lrs)
        duo_lrs.set_pending("tiny", "tent", "02b is running", lrs)
        sgd = {"METHOD": "SGD", "STEPS": 1, "LR": 5e-5, "MOMENTUM": 0.9, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}

        def results(lr_large, lr_small, **extra):
            f = tmp / "results.json"
            f.write_text(json.dumps({"duo": "tiny", "tta": "tent", "bn": True, "seeds": [0], "num_samples": None,
                                     "streams": ["a__s5", "b__s5"], "temperatures": {"T_l": 0.8, "T_s": 1.0, "checkpoint": "x"},
                                     "optim": {"large": sgd, "small": sgd}, "lrs_large": [5e-5, 1e-3], "lrs_small": [5e-5, 1e-3],
                                     "best": {"lr_large": lr_large, "lr_small": lr_small, "duo": 0.6, "worst": 0.4}, **extra}))
            return f

        msg = record_best(results(1e-3, 5e-5), lrs, tmp / "duos")
        assert "large 0.001 and small 5e-05" in msg and "replacing" not in msg and "unchanged" not in msg, msg
        doc = duo_lrs.load("tiny", "joint", lrs)
        assert doc["pending"] == {} and doc["methods"]["tent"]["large"]["OPTIM"] == {**sgd, "LR": 1e-3}
        assert duo_lrs.resolve({"NAME": "tiny", "LARGE": {"NAME": "tiny_ln", "NORM": "LN"}, "SMALL": {"NAME": "tiny_bn", "NORM": "BN"}},
                               "joint", "tent", root=lrs)["SMALL"]["LR"] == 5e-5, "script 03 and load_duo can now hand it out"
        assert "(unchanged)" in record_best(results(1e-3, 5e-5), lrs, tmp / "duos")
        assert "(replacing large 0.001, small 5e-05)" in record_best(results(5e-5, 1e-3), lrs, tmp / "duos")
        assert duo_lrs.load("tiny", "joint", lrs)["methods"]["tent"]["small"]["OPTIM"]["LR"] == 1e-3
        try:
            record_best(results(1e-3, 1e-3, num_samples=2000), lrs, tmp / "duos")
            raise AssertionError("a capped run must not be recorded")
        except ValueError:
            pass
        assert duo_lrs.load("tiny", "joint", lrs)["methods"]["tent"]["large"]["OPTIM"]["LR"] == 5e-5, "the capped run changed nothing"
    print("joint_lr_tuning self-test passed")
