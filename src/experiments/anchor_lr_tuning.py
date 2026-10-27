"""
anchor_lr_tuning.py
===================
The logic behind scripts/02d_tune_large_adapt_lr.py and scripts/02e_tune_small_adapt_lr.py: choose the learning rate of the ONE member
that adapts in script 03's Large adapt / Small adapt mode (engine modes large_duo / small_duo). That member minimizes the entropy of the
static duo's combined prediction (z_L / T_L + z_S / T_S) / 2 while the other member is frozen: its logits are a constant of the loss.

Why a tuner of its own. 02b tunes the PAIR of the Joint mode, where both members move and hold each other up; script 03 used to run
Large / Small adapt at that pair's rates. With the partner frozen the duo entropy is another signal (the frozen member's confidence
is fixed, so the adapting one cannot lean on a partner that adapts with it), so its best rate is another number, and a rate tuned
for two moving members says nothing about it. So each mode is tuned for its own member: a 1-D grid over that member's rate, in a
regime file of its own (cfgs/duo_lrs/<duo>__large_duo.yaml / __small_duo.yaml; src/utils/duo_lrs.py), never the joint pair's.

A grid point is one run of the mode (RunSpec mode large_duo / small_duo, protocol "tuning": the 4 held-out corruptions, each from
source weights) at that rate, with the SAME run script 03 makes on the test corruptions (the fingerprints are equal, asserted by the
self-test). Its score is the MACRO accuracy of the static duo over the 4 streams, the prediction script 03's "Duo, duo entropy: Large /
Small adapts" row scores, with the worst stream and the adapting member's own accuracy next to it. Two references ride along: the
frozen duo, and the SAME member adapting alone on its OWN entropy at its single-regime rate (script 03's Independent large / small:
the rate the grid starts from), so the report says what the duo signal buys over the member's own.

The grid starts at that single rate (cfgs/duo_lrs/<duo>__single.yaml: each member tuned alone; hp-tuning-protocol: "individual rates
first, duos upward from them") and goes both ways: by default 0.5, 1, 2, 4, 8, 16, 32 and 64 times it. The optimum must be INTERIOR
to the grid (the author's rule), and a point whose late accuracy (the last 20% of each stream) is below the frozen duo's is ruled out
as drift, exactly as in 02b/02c. Only the learning rate changes; momentum and the rest of the OPTIM stay.

Temperatures are fixed for the whole grid: the Fixed TS fitted on the FROZEN models (adaptation_modes.temperature_spec), because the
duo entropy depends on T_L and T_S, and a pair per point would change the objective being compared.

The machinery (cache keys, divergence markers, resuming, sharding, scoring) is 02b's `joint_lr_tuning.Tuner`; this module only says
what a job is (one rate instead of a pair) and how a 1-D grid is ranked, tabled, warned about and recorded.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments import joint_lr_tuning as JL
from src.experiments.cli import add_common_args, bn_tag, select_jobs, setup
from src.experiments.results import save_rows
from src.experiments.runner import Context
from src.experiments.specs import RunSpec
from src.utils import provenance

DEFAULT_MULTS = (0.5, 1, 2, 4, 8, 16, 32, 64)
KINDS = ("adapt", "own", "frozen")
NAN = JL.NAN


@dataclass(frozen=True)
class Side:
    side: str            # "large" | "small": the member that adapts
    mode: str            # the engine mode of the duo-entropy run
    own_mode: str        # the engine mode of the same member alone on its own entropy (the reference)
    key: str             # script 03's mode key
    label: str           # script 03's row name
    own_label: str
    script: str

    @property
    def SIDE(self) -> str:
        return self.side.upper()

    @property
    def regime(self) -> str:
        return self.mode


SIDES = {
    "large": Side("large", "large_duo", "large_indep", "large_anchor", "Large adapt", "Independent large", "scripts/02d_tune_large_adapt_lr.py"),
    "small": Side("small", "small_duo", "small_indep", "small_anchor", "Small adapt", "Independent small", "scripts/02e_tune_small_adapt_lr.py"),
}


# ---------------------------------------------------------------------------------------------------------- jobs

@dataclass(frozen=True)
class Job:
    """One run: the duo-entropy mode at `lr` ("adapt"), or a reference for one stream-order seed: "own" (the same member alone, on its own
    entropy, at its single-regime rate) or "frozen" (no adaptation)."""
    kind: str
    seed: int
    lr: float | None = None

    def __post_init__(self):
        if self.kind not in KINDS or (self.kind == "adapt") != (self.lr is not None):
            raise ValueError(f"bad job {self}: only an 'adapt' job has a learning rate, and it needs one")

    @property
    def label(self) -> str:                  # the --only / --shard label
        return f"adapt/lr{self.lr:g}/seed{self.seed}" if self.kind == "adapt" else f"{self.kind}/seed{self.seed}"

    @property
    def point(self) -> tuple:                # what the seeds of one grid point share
        return (self.kind, self.lr)


def build_jobs(lrs: Sequence[float], seeds: Sequence[int]) -> list[Job]:
    """The grid points first, so that an array with one task per point gives each its own task, then the references (which wrap around
    onto the first tasks: the frozen run is forward-only and cached by the temperatures' fit for seed 0)."""
    return [Job("adapt", s, lr) for s in seeds for lr in lrs] + [Job(kind, s) for s in seeds for kind in ("frozen", "own")]


@dataclass
class Tuner(JL.Tuner):
    """02b's Tuner with one rate instead of a pair. `ctx` is the SINGLE-regime context (adaptation_modes.single_context): its member
    rates are those each member was tuned for alone, which the "own" reference runs at and the grid's non-LR optimizer fields come from."""
    side: str = "large"

    @property
    def S(self) -> Side:
        return SIDES[self.side]

    def spec(self, job: Job) -> RunSpec:
        mode = {"adapt": self.S.mode, "own": self.S.own_mode, "frozen": "no_adapt"}[job.kind]
        return RunSpec(self.ctx.duo["NAME"], tta=self.tta, tta_kwargs=self.tta_kwargs or {}, mode=mode, protocol="tuning", seed=job.seed,
                       num_samples=self.num_samples, loop_ts=(self.temps.T_l, self.temps.T_s) if job.kind == "adapt" else None)

    def context(self, job: Job) -> Context:
        if job.kind != "adapt":
            return self.ctx
        return AM.with_learning_rates(self.ctx, job.lr if self.side == "large" else None, job.lr if self.side == "small" else None)


# ------------------------------------------------------------------------------------------------------ ranking

def adapt_points(summaries: dict[tuple, dict]) -> list[dict]:
    """The grid points as flat dicts (learning rate + summary), by ascending rate."""
    return sorted(({"lr": p[1], **s} for p, s in summaries.items() if p[0] == "adapt"), key=lambda p: p["lr"])


def rank(points: Sequence[dict], late_floor: float | None = None) -> list[dict]:
    """Finite points, best duo accuracy first. A tie goes to the smaller rate (the repo's convention of preferring the more conservative
    setting). With `late_floor` (the frozen duo's late accuracy) a point whose late duo accuracy is below it is ruled out
    (`JL.guard_excluded` lists them); if that would leave nothing, the guard is dropped (the caller says so)."""
    finite = [p for p in points if not p["diverged"]]
    if late_floor is not None and (kept := [p for p in finite if not p.get("late_duo", NAN) < late_floor]):
        finite = kept
    return sorted(finite, key=lambda p: (-round(p["duo"], 12), p["lr"]))


def edge_warning(best: dict, lrs: Sequence[float], side: str) -> str | None:
    """The winner sits at an end of the grid: the optimum may lie outside it (a grid of fewer than three rates has no interior)."""
    lr = best["lr"]
    if len(lrs) < 3:
        return f"only {len(lrs)} learning rate(s) tried: the optimum cannot be shown to be interior to the grid (use at least three)"
    if lr in (min(lrs), max(lrs)):
        end = "smallest" if lr == min(lrs) else "largest"
        return (f"the best {side}-member learning rate ({lr:g}) is the {end} one tried ({min(lrs):g}..{max(lrs):g}): the optimum may lie "
                f"beyond it; add values with --lrs (or --mults); finished points are reused")
    return None


# ------------------------------------------------------------------------------------------------------ reporting

def _pct(x: float) -> str:
    return JL._pct(x)


def lr_table(points: Sequence[dict], side: str, base_lr: float, best: dict | None, refs: dict[str, dict | None]) -> str:
    """One row per rate, ascending (the curve), `*` on the best; the references under it. Columns: the multiple of the single rate,
    the duo's macro accuracy, its worst stream, its late accuracy, the adapting member alone, its NLL."""
    head = f"{'lr':>10} {'x single':>9} {'duo':>8} {'worst':>7} {'late duo':>8} {side + ' alone':>11} {'duo nll':>8}"
    lines = [head]
    for p in points:
        star = "*" if best is not None and p["lr"] == best["lr"] else " "
        lines.append(f"{p['lr']:>10g} {p['lr'] / base_lr:>9.3g} {_pct(p['duo']):>7}{star} {_pct(p['worst']):>7} {_pct(p['late_duo']):>8} "
                     f"{_pct(p[side]):>11} {p['duo_nll']:>8.3f}")
    for name, s in refs.items():
        if s is not None:
            lines.append(f"{name:>20} {_pct(s['duo']):>7}  {_pct(s['worst']):>7} {_pct(s['late_duo']):>8} {_pct(s[side]):>11} {s['duo_nll']:>8.3f}")
    return "\n".join(lines)


def comparisons(summaries: dict[tuple, dict], best: dict, base_lr: float, S: Side) -> list[str]:
    """The lines that answer the question: against the frozen duo, against the member alone on its OWN entropy (Independent large /
    small, at the single rate), and, when the single rate is on the grid, against the duo entropy at that same rate."""
    d = lambda a, b: f"{100 * (a - b):+.2f}"                                                      # noqa: E731
    out = []
    frozen, own = summaries.get(("frozen", None)), summaries.get(("own", None))
    at_base = summaries.get(("adapt", base_lr))
    if frozen:
        out.append(f"best {S.label} {_pct(best['duo']).strip()} vs frozen duo {_pct(frozen['duo']).strip()} ({d(best['duo'], frozen['duo'])} points)")
    if own:
        out.append(f"best {S.label} {_pct(best['duo']).strip()} vs {S.own_label} at its single rate {base_lr:g} "
                   f"{_pct(own['duo']).strip()} ({d(best['duo'], own['duo'])} points): what the duo entropy buys over the member's own")
    if at_base and not at_base["diverged"]:
        out.append(f"{S.label} at the single rate {base_lr:g}: {_pct(at_base['duo']).strip()}"
                   + (f", {d(at_base['duo'], own['duo'])} vs {S.own_label}: the duo signal at an equal rate" if own else "")
                   + f"; the best rate adds {d(best['duo'], at_base['duo'])} on top: what moving the rate buys")
    return out


def result_rows(summaries: dict[tuple, dict], ranked: Sequence[dict]) -> list[dict]:
    """One flat row per run (references and diverged points included), ranked grid points first; fractions."""
    place = {p["lr"]: i for i, p in enumerate(ranked, 1)}
    rows = []
    for (kind, lr), s in summaries.items():
        rows.append({"rank": place.get(lr) if kind == "adapt" else None, "kind": kind, "lr": lr, "diverged": s["diverged"],
                     "duo_acc": s["duo"], "worst_acc": s["worst"], "worst_stream": s["worst_stream"], "large_acc": s["large"],
                     "small_acc": s["small"], "duo_ece": s["duo_ece"], "duo_nll": s["duo_nll"], "late_duo": s.get("late_duo", NAN),
                     "late_large": s.get("late_large", NAN), "late_small": s.get("late_small", NAN), "n_seeds": s["n_seeds"],
                     **{f"acc/{k}": v for k, v in s["streams"].items()}})
    return sorted(rows, key=lambda r: (r["kind"] != "adapt", r["rank"] is None, r["rank"] or 0, r["lr"] or 0, r["kind"]))


def evaluate_command(*, duo: str, tta: str, tta_kwargs: dict | None, bn: bool, temps_folder: Path, S: Side,
                     lr: float | None = None) -> str:
    """The script-03 command that scores the rate on the 15 test corruptions: this mode only, with the SAME temperatures as the
    tuning. Without `lr` script 03 reads the rate from cfgs/duo_lrs/<duo>__<regime>.yaml, where `record_best` has put it; with it
    (`--anchor_lr_<side>`) the rate is given by hand (that run's files then carry a rates tag and a time stamp)."""
    parts = ["python", "scripts/03_tab_adaptation_modes.py", "--duo", duo, "--tta", tta]
    if tta_kwargs:
        parts += ["--tta_kwargs", json.dumps(tta_kwargs)]
    parts += (["--bn"] if bn else []) + ["--modes", S.key]
    if lr is not None:
        parts += [f"--anchor_lr_{S.side}", f"{lr:g}"]
    return shlex.join(parts + ["--temps_checkpoint", str(temps_folder)])


def record_best(results_json: Path, root: Path | None = None, duos_dir: Path | None = None) -> str:
    """Record the best rate of a finished full run (its results.json) in cfgs/duo_lrs/<duo>__<side>_duo.yaml (`root`: that folder), the
    rate script 03's Large / Small adapt reads. It replaces the entry already there for that TTA method. Returns the line to print."""
    from src.utils import duo_lrs

    r = json.loads(Path(results_json).read_text())
    duo, tta, side = r["duo"], r["tta"], r["side"]
    old = None
    try:
        entry = duo_lrs.load(duo, f"{side}_duo", root)["methods"].get(tta)
        old = float(entry[side]["OPTIM"]["LR"]) if entry else None
    except FileNotFoundError:
        pass
    path = duo_lrs.import_anchor(results_json, root=root, duos_dir=duos_dir)
    new = float(r["best"]["lr"])
    note = "" if old is None else " (unchanged)" if old == new else f" (replacing {old:g})"
    return f"recorded the best rate, {new:g}, as the {side} member's learning rate for adapting on the duo entropy of {duo} / {tta} in {path}{note}"


# ------------------------------------------------------------------------------------------------ script plumbing

def add_args(ap: argparse.ArgumentParser, side: str) -> None:
    S = SIDES[side]
    ap.add_argument("--mults", type=float, nargs="+", default=list(DEFAULT_MULTS), metavar="X",
                    help=f"the grid: these multiples of the {side} member's SINGLE-regime rate (cfgs/duo_lrs/<duo>__single.yaml: the rate tuned "
                         f"for it alone, else its indep entry). Default: %(default)s")
    ap.add_argument("--lrs", type=float, nargs="+", default=None, metavar="LR", help="explicit learning rates (replaces --mults)")
    ap.add_argument("--temps_checkpoint", default=None, metavar="DIR",
                    help="JointFixedTS checkpoint (a folder with config.json) for the duo's temperatures; default: the Fixed TS fitted on the "
                         "FROZEN models (never refitted per rate)")
    ap.add_argument("--temps_only", action="store_true",
                    help="fit and store the temperatures, then stop (run this once before a --shard array so that no two tasks fit them at "
                         "the same time)")
    ap.add_argument("--record", action="store_true",
                    help=f"after the ranking, record the best rate in cfgs/duo_lrs/<duo>__{S.regime}.yaml (what script 03's {S.label} reads). "
                         f"Refused while a warning is open (a winner on the edge of the grid, fewer than three rates, no gain over the frozen "
                         f"duo), for a capped --num_samples run, and for an incomplete grid")


def run(side: str, doc: str, argv: Sequence[str] | None = None, segments: list | None = None) -> int:
    """The whole of script 02d (side "large") / 02e ("small"). `argv`, `segments`: for the self-test (the command line, and fake streams
    instead of the protocol's 4 held-out corruptions)."""
    S = SIDES[side]
    ap = add_common_args(argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter))
    add_args(ap, side)
    args = ap.parse_args(argv)
    tag = f"{args.duo}__{args.tta}{bn_tag(args)}" + (f"__n{args.num_samples}" if args.num_samples else "")
    ctx, seeds, out = setup(args, f"tune_{side}_adapt_lr/{tag}", default_seeds=[0])
    single = AM.single_context(ctx, args.tta)

    print(provenance.footer(ctx.duo, args.tta, args.bn))
    base_lr = float(f"{single.duo[S.SIDE]['OPTIM']['LR']:.6g}")
    lrs = JL.lr_axis(base_lr, args.mults, args.lrs)
    print(f"{side} {single.duo[S.SIDE]['NAME']} ({single.duo[S.SIDE]['NORM']}): {JL.optimizer_text(single.duo[S.SIDE]['OPTIM'])}; single-regime rate "
          f"{base_lr:g}; grid {[f'{v:g}' for v in lrs]}")
    other = "small" if side == "large" else "large"
    frozen_bn = single.duo[other.upper()]["NORM"] == "BN"
    print(f"{S.label}: the {side} member adapts on the duo entropy, the {other} member ({single.duo[other.upper()]['NAME']}) is frozen"
          + ("" if not frozen_bn else " (BatchNorm on the test batch's statistics)" if args.bn else
             " (BatchNorm on the SOURCE statistics: pass --bn as script 03 does)"))

    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, args.tta, args.tta_kwargs, args.num_samples),
                                    args.temps_checkpoint, report_only=args.report_only, segments=segments)
    print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}  ({temps.folder})")
    if args.temps_only:
        return 0

    tuner = Tuner(single, args.tta, args.tta_kwargs, args.num_samples, temps, segments=segments, side=side)
    jobs = build_jobs(lrs, seeds)
    print(f"{len(jobs)} runs: {len(lrs) * len(seeds)} grid points + frozen and {S.own_label} (references), batch size {ctx.batch_size}, "
          f"{args.num_samples or 'all'} images per stream, seeds {seeds}")
    if not args.report_only:
        tuner.compute(select_jobs(jobs, lambda j: j.label, args.shard, args.only))
    gaps = tuner.missing(jobs)
    if gaps:
        for label, streams in gaps.items():
            print(f"{label}: member logits missing for {len(streams)} streams (e.g. {streams[:3]}); run without --report_only to compute them")
        return 0 if args.shard or args.only else 1       # a shard only fills the cache; the full run ranks
    if args.shard:
        return 0

    summaries = tuner.score(jobs)
    points = adapt_points(summaries)
    frozen, own = summaries.get(("frozen", None)), summaries.get(("own", None))
    floor = None if frozen is None else frozen["late_duo"]           # the guard: ending worse than the frozen duo is drift
    ranked = rank(points, late_floor=floor)
    if not ranked:
        print("every grid point diverged (non-finite logits): the learning rates are too large; use smaller --mults")
        return 1
    best = ranked[0]
    excluded = JL.guard_excluded(points, floor)
    guard_dropped = floor is not None and bool(excluded) and len(excluded) == len([p for p in points if not p["diverged"]])
    if floor is not None:
        print(f"late-stream guard: the frozen duo's accuracy on the last {int(100 * JL.LATE_FRAC)}% of each stream is {100 * floor:.2f}; "
              f"{len(excluded)} of {len(points)} grid points end below it and are ruled out"
              + (": " + ", ".join(f"{p['lr']:g}" for p in excluded) if excluded else ""))
        if guard_dropped:
            print("WARNING: every finite point ends below the frozen duo, so the guard was dropped")

    refs = {"frozen duo": frozen, f"{S.own_label} (own entropy, single rate {base_lr:g})": own}
    print(f"\n{S.label}: duo accuracy (%), macro over the 4 held-out streams, per learning rate of the {side} member; * = best\n"
          f"{lr_table(points, side, base_lr, best, refs)}")
    print(f"\nbest: {side} {best['lr']:g} ({best['lr'] / base_lr:.3g} x single): duo {100 * best['duo']:.2f}, worst stream "
          f"{100 * best['worst']:.2f} ({best['worst_stream']}), late {100 * best['late_duo']:.2f}, {side} alone {100 * best[side]:.2f}")
    for line in comparisons(summaries, best, base_lr, S):
        print(line)
    warnings = []
    if (w := edge_warning(best, lrs, side)):
        warnings.append(w)
    if frozen and best["duo"] <= frozen["duo"]:
        warnings.append("no rate beats the frozen duo; try smaller rates (--mults 0.01 0.03 0.1 0.3 1)")
    if guard_dropped:
        warnings.append("every finite rate ends below the frozen duo (late-stream guard)")
    for w in warnings:
        print(f"WARNING: {w}")
    n_div = sum(p["diverged"] for p in points)
    if n_div:
        print(f"{n_div} of {len(points)} grid points diverged (non-finite logits)")

    rows = result_rows(summaries, ranked)
    save_rows(out / "ranking.csv", pd.DataFrame(rows).astype({"rank": "Int64"}))     # the references have no rank
    command = None if args.num_samples else evaluate_command(duo=args.duo, tta=args.tta, tta_kwargs=args.tta_kwargs, bn=args.bn,
                                                             temps_folder=temps.folder, S=S, lr=None if args.record else best["lr"])
    (out / "results.json").write_text(json.dumps({
        "duo": args.duo, "tta": args.tta, "tta_kwargs": args.tta_kwargs, "side": side, "bn": args.bn, "seeds": seeds,
        "num_samples": args.num_samples, "batch_size": ctx.batch_size, "streams": sorted(best["streams"]),
        "temperatures": {"T_l": temps.T_l, "T_s": temps.T_s, "checkpoint": str(temps.folder)},
        "optim": {side: single.duo[S.SIDE]["OPTIM"]}, "single_lr": base_lr, "lrs": lrs,
        "best": {"lr": best["lr"], "duo": best["duo"], "worst": best["worst"], "worst_stream": best["worst_stream"], "late_duo": best["late_duo"]},
        "frozen_duo": None if frozen is None else frozen["duo"], "own_duo": None if own is None else own["duo"],
        "late_floor": floor, "guard_excluded": [p["lr"] for p in excluded], "warnings": warnings, "evaluate_command": command,
        "runs": [{k: None if isinstance(v, float) and v != v else v for k, v in r.items()} for r in rows]}, indent=1))
    print(f"wrote {out / 'ranking.csv'} and {out / 'results.json'}")

    if args.num_samples:
        print("capped run (--num_samples): the best rate is not recorded, no follow-up command"
              + (" (--record ignored)" if args.record else ""))
        return 1 if args.record else 0
    if args.record:
        if warnings:
            print("NOT RECORDED (a warning is open): " + "; ".join(warnings))
            return 1
        print(record_best(out / "results.json", ctx.g.path("duo_lrs")))
        if len(seeds) < 3:
            print(f"note: recorded from {len(seeds)} seed(s); the protocol confirms the finalists on seeds 0 1 2 "
                  f"(--lrs <the finalists> --seeds 0 1 2 --record)")
    else:
        print(f"not recorded: add --record (it writes {best['lr']:g} to {ctx.g.path('duo_lrs')}/{args.duo}__{S.regime}.yaml; "
              f"python -m src.utils.duo_lrs --import_anchor {out / 'results.json'} does the same by hand)")
    print(f"\nscore the {'recorded' if args.record else 'best'} rate on the 15 test corruptions:\n{command}")
    return 0


if __name__ == "__main__":
    import contextlib
    import io
    import math
    import tempfile

    import torch
    import yaml

    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.experiments.cache import MemberCache
    from src.experiments.protocols import Segment
    from src.experiments.runner import run_dir
    from src.utils import config as C, duo_lrs
    from src.utils.config import load_global

    torch.set_num_threads(2)
    g = load_global()

    # ---- the grid and the jobs ---------------------------------------------------------------------------------------
    assert JL.lr_axis(7e-5, DEFAULT_MULTS) == [3.5e-5, 7e-5, 1.4e-4, 2.8e-4, 5.6e-4, 1.12e-3, 2.24e-3, 4.48e-3], "from the single rate, both ways"
    jobs = build_jobs([1e-3, 2e-3, 4e-3], [0, 1])
    assert len(jobs) == 2 * (3 + 2) and len({j.label for j in jobs}) == len(jobs)
    assert [j.kind for j in jobs[:6]] == ["adapt"] * 6 and {j.kind for j in jobs[-4:]} == {"frozen", "own"}, "points first, references last"
    assert jobs[0].label == "adapt/lr0.001/seed0" and jobs[-1].label == "own/seed1" and jobs[0].point == ("adapt", 1e-3)
    for bad in (dict(kind="adapt", seed=0), dict(kind="own", seed=0, lr=1e-3), dict(kind="joint", seed=0)):
        try:
            Job(**bad)
            raise AssertionError(f"must raise: {bad}")
        except ValueError:
            pass
    assert set(SIDES) == {"large", "small"} and {S.key for S in SIDES.values()} == set(AM.ANCHOR_SIDE), "the two sides are script 03's two anchor modes"
    assert all(AM.MODE_BY_KEY[S.key].engine == S.mode and S.regime == f"{S.side}_duo" for S in SIDES.values())

    # ---- ranking, the guard, the edge rule, tables, comparisons: synthetic summaries --------------------------------------------
    def fake(duo, **kw):
        return {**JL.DIVERGED_SUMMARY, "diverged": False, "duo": duo, "large": duo - .05, "small": duo - .1, "worst": duo - .02, "worst_stream": "a",
                "duo_ece": .05, "duo_nll": 1.5, "late_duo": duo, "n_seeds": 1, "streams": {"a": duo - .02, "b": duo + .02}, **kw}
    LRS = [1e-4, 2e-4, 4e-4, 8e-4]
    table = {("adapt", lr): fake(acc) for lr, acc in zip(LRS, (.50, .55, .55, .53))}
    table[("frozen", None)], table[("own", None)] = fake(.40), fake(.49)
    pts = adapt_points(table)
    ranked = rank(pts)
    assert [p["lr"] for p in ranked] == [2e-4, 4e-4, 8e-4, 1e-4], "tie at .55: the smaller rate wins"
    assert edge_warning(ranked[0], LRS, "large") is None, "an interior optimum: no warning"
    assert "smallest" in edge_warning(rank(pts[:1] + pts[3:])[0] | {"lr": 1e-4}, LRS, "large") and "--lrs" in edge_warning({"lr": 8e-4}, LRS, "large")
    assert "at least three" in edge_warning({"lr": 1e-4}, [1e-4, 2e-4], "small"), "no interior on a grid of two"
    late = [dict(lr=1e-4, diverged=False, duo=.60, late_duo=.40), dict(lr=2e-4, diverged=False, duo=.58, late_duo=.55),
            dict(lr=4e-4, diverged=True, duo=NAN, late_duo=NAN)]
    assert rank(late)[0]["lr"] == 1e-4 and rank(late, late_floor=.5)[0]["lr"] == 2e-4 and [p["lr"] for p in JL.guard_excluded(late, .5)] == [1e-4], \
        "an adaptation that ends worse than the frozen duo is ruled out, however good its mean"
    assert rank(late, late_floor=.9)[0]["lr"] == 1e-4, "a guard that would rule out everything is dropped (the caller says so)"
    txt = lr_table(pts, "large", 2e-4, ranked[0], {"frozen duo": table[("frozen", None)], "own": table[("own", None)]}).splitlines()
    assert len(txt) == 1 + len(LRS) + 2 and txt[2].count("*") == 1 and "55.00*" in txt[2] and " 1" in txt[2].split()[1] + " 1", txt
    assert "div." in lr_table(adapt_points({**table, ("adapt", 1.6e-3): JL.DIVERGED_SUMMARY}), "large", 2e-4, ranked[0], {})
    cmp = comparisons(table, ranked[0], 2e-4, SIDES["large"])
    assert len(cmp) == 3 and "+15.00 points" in cmp[0] and "+6.00 points" in cmp[1] and "Independent large" in cmp[1], cmp
    assert "+6.00 vs Independent large" in cmp[2] and "adds +0.00 on top" in cmp[2], cmp
    assert len(comparisons(table, ranked[0], 9e-9, SIDES["small"])) == 2, "the single rate is not on the grid: no third line"
    rows = result_rows(table, ranked)
    assert [r["kind"] for r in rows[-2:]] == ["frozen", "own"] and rows[0]["rank"] == 1 and rows[0]["lr"] == 2e-4 and rows[0]["acc/a"] == .55 - .02
    json.dumps(rows)
    assert evaluate_command(duo="d", tta="tent", tta_kwargs=None, bn=True, temps_folder=Path("/x/t y"), S=SIDES["small"]) == \
        "python scripts/03_tab_adaptation_modes.py --duo d --tta tent --bn --modes small_anchor --temps_checkpoint '/x/t y'"
    assert evaluate_command(duo="d", tta="eata", tta_kwargs={"d_margin": 0.1}, bn=False, temps_folder=Path("t"), S=SIDES["large"], lr=2e-3) == \
        "python scripts/03_tab_adaptation_modes.py --duo d --tta eata --tta_kwargs '{\"d_margin\": 0.1}' --modes large_anchor --anchor_lr_large 0.002 " \
        "--temps_checkpoint t"

    # ---- tiny models and fake streams: the real engine, end to end ----------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = make_context(tmp, batch_size=32, base=tiny_models(0))
        ctx.g.raw["paths"]["checkpoints"] = str(tmp / "checkpoints")
        ctx.g.raw["paths"]["duo_lrs"] = str(tmp / "duo_lrs")
        ctx.g.raw["paths"]["out"] = str(tmp / "out")
        ctx.g.raw["runtime"]["workers"] = 0              # tiny in-memory streams: no DataLoader processes needed
        held = [Segment(f"{c}/s5", FakeStream(150, seed=10 + i), True) for i, c in enumerate(g.tuning_corruptions)]
        tspec = AM.temperature_spec("tiny", "tent", None, None)
        temps = AM.resolve_temperatures(ctx, tspec, segments=held)
        base_lr = ctx.duo["LARGE"]["OPTIM"]["LR"]
        lrs = JL.lr_axis(base_lr, (0.5, 1, 4))

        for side in ("large", "small"):
            S = SIDES[side]
            tuner = Tuner(ctx, "tent", None, None, temps, segments=held, side=side)
            grid = build_jobs(lrs, [0])
            for job in grid:                                                      # what each job is
                sp = tuner.spec(job)
                assert sp.protocol == "tuning" and sp.mode == {"adapt": S.mode, "own": S.own_mode, "frozen": "no_adapt"}[job.kind]
                assert (sp.loop_ts == (temps.T_l, temps.T_s)) == (job.kind == "adapt"), "only the duo-entropy run needs the temperatures"
            assert len({tuner.spec(j).fingerprint(tuner.context(j).duo, 32, "masked") for j in grid}) == len(grid), "every job has its own cache"
            # the tuning point IS script 03's anchor run: same spec up to the stream list, so the same cache key space
            at = Job("adapt", 0, lrs[1])
            theirs_ctx = AM.anchor_context(ctx, "tent", S.key, lr=lrs[1])
            assert run_dir(tuner.spec(at), tuner.context(at)) == run_dir(AM.mode_spec(AM.MODE_BY_KEY[S.key], "tiny", "tent", None, 0, None, temps), theirs_ctx), \
                "the tuner's run must be script 03's Large / Small adapt run"
            own = Job("own", 0)
            assert run_dir(tuner.spec(own), tuner.context(own)) == run_dir(AM.mode_spec(AM.MODE_BY_KEY[f"indep_{side}"], "tiny", "tent", None, 0, None, temps), ctx), \
                "the reference is script 03's Independent large / small run at the single rate"
            assert run_dir(tuner.spec(Job("frozen", 0)), ctx) == run_dir(tspec, ctx), "the frozen reference is the run the temperatures were fitted on"

            assert set(tuner.missing(grid)) == {j.label for j in grid if j.kind != "frozen"}, "the temperatures' fit cached 'frozen'"
            tuner.compute(grid)
            assert tuner.missing(grid) == {}
            files = {p: p.stat().st_mtime_ns for j in grid for p in MemberCache(run_dir(tuner.spec(j), tuner.context(j))).dir.glob("*.pt")}
            tuner.compute(grid)
            assert files == {p: p.stat().st_mtime_ns for p in files}, "a finished job is not recomputed"

            # the score is the static duo's macro accuracy read off the cached logits; the member that does not adapt is the frozen one
            summaries = tuner.score(grid)
            frozen_d = MemberCache(run_dir(tspec, ctx)).read(held[0].file_stem)
            for job in grid:
                cache = MemberCache(run_dir(tuner.spec(job), tuner.context(job)))
                accs = []
                for seg in held:
                    d = cache.read(seg.file_stem)
                    accs.append(float(((0.5 * d["z_l"] / temps.T_l + 0.5 * d["z_s"] / temps.T_s).argmax(1) == d["labels"]).float().mean()))
                assert abs(summaries[job.point]["duo"] - sum(accs) / len(accs)) < 1e-9, job.label
                if job.kind == "adapt":
                    d = cache.read(held[0].file_stem)
                    moving, still = (("z_l", "z_s") if side == "large" else ("z_s", "z_l"))
                    assert d[still].equal(frozen_d[still]) and not d[moving].equal(frozen_d[moving]), f"{S.label}: only the {side} member moves"
            a, b = (MemberCache(run_dir(tuner.spec(j), tuner.context(j))).read(held[0].file_stem) for j in (grid[0], grid[2]))
            assert not a["z_l" if side == "large" else "z_s"].equal(b["z_l" if side == "large" else "z_s"]), "another rate, other logits"
            own_d = MemberCache(run_dir(tuner.spec(own), ctx)).read(held[0].file_stem)
            same = MemberCache(run_dir(tuner.spec(Job("adapt", 0, lrs[1])), tuner.context(Job("adapt", 0, lrs[1])))).read(held[0].file_stem)
            m = "z_l" if side == "large" else "z_s"
            assert not own_d[m].equal(same[m]), "the duo's entropy is not the member's own: Large / Small adapt differs from Independent large / small"

            # a diverging rate is recorded, counts as complete, ranks nowhere and is never recomputed
            boom = Job("adapt", 0, 1e30)
            tuner.compute([boom])
            assert tuner.missing([boom]) == {} and tuner._marker(boom).is_file()
            sc = tuner.score(grid + [boom])
            assert sc[boom.point]["diverged"] and math.isnan(sc[boom.point]["duo"]) and len(rank(adapt_points(sc))) == len(lrs)
            try:
                tuner.compute([Job("adapt", 0, -1.0)])
                raise AssertionError("a negative learning rate must raise, not be recorded as divergence")
            except ValueError as e:
                assert "positive" in str(e)

        # ---- the whole script (run), on the same tiny models: flags, caches, ranking, files, --record --------------------------------
        (tmp / "duos").mkdir()
        (tmp / "duos" / "tiny.yaml").write_text(yaml.safe_dump({"NAME": "tiny", "LARGE": {"NAME": "tiny_ln", "NORM": "LN"},
                                                                 "SMALL": {"NAME": "tiny_bn", "NORM": "BN"}}))
        global setup
        real_setup, real_single, real_dir, real_lr_dir = setup, AM.single_context, C.DUO_DIR, duo_lrs.DUO_DIR

        def fake_setup(args, script, default_seeds=None):
            out = tmp / "out" / "results" / script
            out.mkdir(parents=True, exist_ok=True)
            return ctx, list(args.seeds or default_seeds), out

        setup, AM.single_context, C.DUO_DIR, duo_lrs.DUO_DIR = fake_setup, (lambda c, tta: c), tmp / "duos", tmp / "duos"
        try:
            duo_lrs.init_indep("tiny", tmp / "duo_lrs", tmp / "duos", defaults={"large": ctx.duo["LARGE"]["OPTIM"], "small": ctx.duo["SMALL"]["OPTIM"]})

            def go(side, *flags):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = run(side, "doc", ["--duo", "tiny", *flags], segments=held)
                return rc, buf.getvalue()

            grid_flags = ["--lrs", *[f"{v:g}" for v in lrs], "--seeds", "0"]
            for side in ("large", "small"):
                S = SIDES[side]
                rc, text = go(side, *grid_flags, "--temps_only")
                assert rc == 0 and "static duo: T_L" in text and "adapt/" not in text, "--temps_only fits and stops"
                rc, text = go(side, *grid_flags, "--shard", "0/2")
                assert rc == 0 and "best:" not in text, "a shard fills the cache and does not rank"
                rc, text = go(side, *grid_flags, "--shard", "1/2")
                rc, text = go(side, *grid_flags, "--report_only")
                assert rc == 0 and "best:" in text and "late-stream guard" in text and f"{S.label}: duo accuracy" in text, text
                assert "not recorded: add --record" in text and f"--anchor_lr_{side} " in text.split("score the best rate")[1], "the follow-up carries the rate by hand"
                out = tmp / "out" / "results" / f"tune_{side}_adapt_lr" / "tiny__tent"
                res = json.loads((out / "results.json").read_text())
                assert res["side"] == side and res["lrs"] == lrs and res["best"]["lr"] in lrs and set(res["optim"]) == {side} and res["frozen_duo"] is not None
                assert len(pd.read_csv(out / "ranking.csv")) == len(lrs) + 2 and not (tmp / "duo_lrs" / f"tiny__{S.regime}.yaml").exists(), "nothing recorded"
                rc2, text2 = go(side, *grid_flags, "--report_only", "--record")
                assert rc2 == 1 and "NOT RECORDED" in text2 and not (tmp / "duo_lrs" / f"tiny__{S.regime}.yaml").exists(), \
                    "the random tiny streams carry no signal: no rate beats the frozen duo, so nothing is recorded"

            # a controlled ranking: an interior winner that beats the frozen duo is recorded (once; again it is unchanged), an edge winner
            # is refused and the file keeps the rate it had
            real_score = Tuner.score
            ok = {("adapt", lrs[0]): fake(.50), ("adapt", lrs[1]): fake(.55), ("adapt", lrs[2]): fake(.52), ("frozen", None): fake(.40), ("own", None): fake(.49)}
            edge = {**ok, ("adapt", lrs[0]): fake(.60)}
            try:
                for side in ("large", "small"):
                    S = SIDES[side]
                    f = tmp / "duo_lrs" / f"tiny__{S.regime}.yaml"
                    Tuner.score = lambda self, jobs: ok
                    rc, text = go(side, *grid_flags, "--report_only", "--record")
                    assert rc == 0 and f.is_file() and f"recorded the best rate, {lrs[1]:g}, as the {side} member" in text and "WARNING" not in text, text
                    assert "note: recorded from 1 seed(s)" in text and "score the recorded rate" in text and "--anchor_lr_" not in text.split("score the recorded")[1]
                    ac = AM.anchor_context(ctx, "tent", S.key)
                    assert ac.duo[S.SIDE]["OPTIM"]["LR"] == lrs[1] == duo_lrs.resolve(
                        yaml.safe_load((tmp / "duos" / "tiny.yaml").read_text()), S.regime, "tent", tmp / "duo_lrs")[S.SIDE]["LR"]
                    tn = Tuner(ctx, "tent", None, None, temps, segments=held, side=side)
                    bj = Job("adapt", 0, lrs[1])
                    assert run_dir(AM.mode_spec(AM.MODE_BY_KEY[S.key], "tiny", "tent", None, 0, None, temps), ac) == run_dir(tn.spec(bj), tn.context(bj)), \
                        "what was recorded is exactly the run that was tuned: script 03 reads the tuning's cache key for it"
                    res = json.loads((tmp / "out" / "results" / f"tune_{side}_adapt_lr" / "tiny__tent" / "results.json").read_text())
                    assert res["best"]["lr"] == lrs[1] and res["frozen_duo"] == .40 and res["own_duo"] == .49 and res["warnings"] == []
                    assert "(unchanged)" in go(side, *grid_flags, "--report_only", "--record")[1]
                    Tuner.score = lambda self, jobs: edge
                    rc, text = go(side, *grid_flags, "--report_only", "--record")
                    assert rc == 1 and "NOT RECORDED" in text and "smallest one tried" in text
                    assert duo_lrs.resolve(yaml.safe_load((tmp / "duos" / "tiny.yaml").read_text()), S.regime, "tent", tmp / "duo_lrs")[S.SIDE]["LR"] == lrs[1], \
                        "a refused ranking leaves the recorded rate alone"
                    Tuner.score = real_score
            finally:
                Tuner.score = real_score
            rc, text = go("large", "--lrs", "0.01", "--seeds", "0", "--report_only", "--record")
            assert rc == 1 and "NOT RECORDED" in text and "at least three" in text, "a grid of one rate has no interior: refused"
            try:
                go("large", *grid_flags, "--num_samples", "64", "--report_only")
                raise AssertionError("a capped run has its own cache: nothing is there")
            except (FileNotFoundError, SystemExit):
                pass
        finally:
            setup, AM.single_context, C.DUO_DIR, duo_lrs.DUO_DIR = real_setup, real_single, real_dir, real_lr_dir

        # ---- record_best: replaces, says so, never takes a capped run ----------------------------------------------------------
        (tmp / "duos2").mkdir()
        (tmp / "duos2" / "tiny.yaml").write_text((tmp / "duos" / "tiny.yaml").read_text())
        sgd = {"METHOD": "SGD", "STEPS": 1, "LR": 5e-5, "MOMENTUM": 0.9, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}

        def results(lr, **extra):
            f = tmp / "r.json"
            f.write_text(json.dumps({"duo": "tiny", "tta": "tent", "side": "small", "bn": True, "seeds": [0], "num_samples": None, "batch_size": 64,
                                     "streams": ["a__s5"], "temperatures": {"T_l": .8, "T_s": 1., "checkpoint": "/x/ts"}, "optim": {"small": sgd},
                                     "lrs": [1e-3, 2e-3, 4e-3], "best": {"lr": lr, "duo": .6, "worst": .4, "late_duo": .6}, "frozen_duo": .5, **extra}))
            return f
        lrs2 = tmp / "lrs2"
        msg = record_best(results(2e-3), lrs2, tmp / "duos2")
        assert "recorded the best rate, 0.002, as the small member's learning rate" in msg and "replacing" not in msg and "unchanged" not in msg, msg
        assert "(unchanged)" in record_best(results(2e-3), lrs2, tmp / "duos2") and "(replacing 0.002)" in record_best(results(4e-3), lrs2, tmp / "duos2")
        assert duo_lrs.load("tiny", "small_duo", lrs2)["methods"]["tent"]["small"]["OPTIM"] == {**sgd, "LR": 4e-3}
        try:
            record_best(results(1e-3, num_samples=2000), lrs2, tmp / "duos2")
            raise AssertionError("a capped run must not be recorded")
        except ValueError:
            pass
        assert duo_lrs.load("tiny", "small_duo", lrs2)["methods"]["tent"]["small"]["OPTIM"]["LR"] == 4e-3, "the capped run changed nothing"
    print("anchor_lr_tuning self-test passed")
