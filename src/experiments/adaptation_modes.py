"""
adaptation_modes.py
===================
The logic behind scripts/03_tab_adaptation_modes.py: HOW the duo is adapted, with the prediction held fixed.

Every mode is scored by the same static duo, (z_L / T_L + z_S / T_S) / 2 with fixed temperatures, so the rows of
the table differ only in how the members' weights were adapted. The modes (the model named is the one that adapts):

  Joint               both members minimize the entropy of the duo's combined prediction        both_duo
  Large adapt         only the large member adapts on the duo entropy; the small one is frozen    large_duo
  Small adapt         only the small member adapts on the duo entropy; the large one is frozen    small_duo
  Independent large   Tent on the large member's own entropy; the small member is not adapted     large_indep
  Independent small   Tent on the small member's own entropy; the large member is not adapted     small_indep
  Independent both    both members run Tent on their own entropy                                  both_indep
  No adaptation       frozen weights with test-batch statistics (reference row, not in the paper's list)  no_adapt

The right-hand names are the engine's modes (src/experiments/members.py). One live run per (mode, seed) caches both
members' logits (src/experiments/cache.py); the duo's accuracy is a replay of the fixed combination on them.

The per-corruption table has one row per (mode, prediction) pair (ROWS), in four sections named by what is scored and
which entropy the adapting models minimize:

  Frozen                              Large, Small, Duo                 from the No adaptation run
  Single model, own entropy (Tent)    Large / Small adapts              the large member of the Independent large run,
                                                                        the small member of the Independent small run
  Duo, own entropy                    Large / Small / Both adapt        the duo of Independent large / small / both
  Duo, duo entropy                    Large / Small / Both adapt        the duo of Large / Small adapt / Joint

The summary table instead has one row per mode, named by MODES. A "clean" stream (the ImageNet validation images the
corruptions are rendered from) is run like any corruption and shown in its own Clean column of the per-corruption
table; it is never part of Avg. nor of the summary table's averages.

Temperatures. The duo entropy of the first three modes depends on T_L and T_S, so they must exist before the run.
All modes share one pair, the paper's Fixed TS: fitted on the 4 held-out corruptions pooled, on the logits of the
independently adapted members (the same fit as the "Held-out TS" column of scripts/01_tab_temperatures.py, and it
reads the same cache). The pair is fitted once and stored as a JointFixedTS checkpoint under paths.checkpoints
(rounded to 4 decimals, so a refit can never change the cache keys, which include the temperatures); concurrent
SLURM shards that race to write it all end up with the first writer's values.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.experiments import oracles as O
from src.experiments.cli import bn_tag
from src.experiments.protocols import Segment, build_segments
from src.experiments.results import mean_std_over_seeds, per_seed_average
from src.experiments.runner import Context, EvalResult, ensure_members, evaluate
from src.experiments.specs import DUO_MODES, GateSpec, RunSpec
from src.experiments.temperature_study import CacheSource, MODEL_NAMES, missing_streams, pool
from src.utils.tables import corruption_table, fmt_cell, latex_escape, latex_table


@dataclass(frozen=True)
class Mode:
    key: str          # series prefix in the CSV and the --modes flag
    label: str        # row name in the tables
    engine: str       # MemberRunner mode
    group: str        # row group in the per-corruption table


MODES = (
    Mode("joint", "Joint", "both_duo", "Duo entropy"),
    Mode("large_anchor", "Large adapt", "large_duo", "Duo entropy"),
    Mode("small_anchor", "Small adapt", "small_duo", "Duo entropy"),
    Mode("indep_large", "Independent large", "large_indep", "Own entropy"),
    Mode("indep_small", "Independent small", "small_indep", "Own entropy"),
    Mode("indep_duo", "Independent both", "both_indep", "Own entropy"),
    Mode("frozen", "No adaptation", "no_adapt", "Reference"),
)
MODE_KEYS = [m.key for m in MODES]
MODE_BY_KEY = {m.key: m for m in MODES}
assert {m.engine for m in MODES if m.group == "Duo entropy"} == set(DUO_MODES)


@dataclass(frozen=True)
class Row:
    """One row of the per-corruption table: the prediction `series` of the run of mode `mode`, listed under the
    section `group` as `label`. A section is the pair (what is scored, which entropy is minimized), so one mode can
    feed rows of two sections (Independent large: its large member alone, and the duo that combines the two)."""
    mode: str
    series: str       # "duo": the static duo's prediction; "large" / "small": that member's own
    group: str
    label: str

    @property
    def key(self) -> str:                       # the series of _with_mode_series
        return f"{self.mode}/{self.series}"


FROZEN, SINGLE, OWN, DUO = "Frozen", "Single model, own entropy ({method})", "Duo, own entropy", "Duo, duo entropy"
ROWS = (
    Row("frozen", "large", FROZEN, "Large"), Row("frozen", "small", FROZEN, "Small"), Row("frozen", "duo", FROZEN, "Duo"),
    Row("indep_large", "large", SINGLE, "Large adapts"), Row("indep_small", "small", SINGLE, "Small adapts"),
    Row("indep_large", "duo", OWN, "Large adapts"), Row("indep_small", "duo", OWN, "Small adapts"),
    Row("indep_duo", "duo", OWN, "Both adapt (Independent)"),
    Row("large_anchor", "duo", DUO, "Large adapts"), Row("small_anchor", "duo", DUO, "Small adapts"),
    Row("joint", "duo", DUO, "Both adapt (Joint)"),
)
_SECTIONS = [r.group for r in ROWS]
assert all(r.mode in MODE_BY_KEY for r in ROWS) and len({r.key for r in ROWS}) == len(ROWS)
assert _SECTIONS == sorted(_SECTIONS, key=_SECTIONS.index), "a section's rows are contiguous"

CLEAN = "clean"      # label of the clean-validation stream (protocols.py); shown as the Clean column, never averaged


# ------------------------------------------------------------------------------------------------ temperatures

@dataclass(frozen=True)
class Temps:
    folder: Path          # a JointFixedTS checkpoint: what the replayed gate loads
    T_l: float
    T_s: float
    origin: str           # for the caption, completing "... with T_L = a, T_S = b, <origin>"


def load_temperatures(folder: Path) -> tuple[float, float]:
    cfg = json.loads((folder / "config.json").read_text())
    if cfg.get("class_name") != "JointFixedTS":
        raise ValueError(f"{folder}/config.json is not a JointFixedTS checkpoint")
    return float(cfg["T_l"]), float(cfg["T_s"])


def store_temperatures(folder: Path, T_l: float, T_s: float, trained_on: dict) -> tuple[float, float]:
    """Write the pair as a JointFixedTS checkpoint unless another job got there first; return what is on disk
    (the first writer wins), so every shard of a job array ends up with the same temperatures."""
    folder.mkdir(parents=True, exist_ok=True)
    final, tmp = folder / "config.json", folder / f".config.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"class_name": "JointFixedTS", "T_l": T_l, "T_s": T_s, "trained_on": trained_on}, indent=2))
    try:
        os.link(tmp, final)                 # atomic and exclusive: fails if the file exists
    except FileExistsError:
        pass
    except OSError:                         # a filesystem without hard links: plain atomic rename
        if not final.exists():
            os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    return load_temperatures(folder)


def temperature_spec(duo: str, tta: str, tta_kwargs: dict | None, num_samples: int | None) -> RunSpec:
    """The run the temperatures are fitted on: independently adapted members on the 4 held-out corruptions. Its
    cache is the one the held-out rows of scripts/01_tab_temperatures.py already fill (same fingerprint)."""
    return RunSpec(duo, tta=tta, tta_kwargs=tta_kwargs or {}, mode="both_indep", protocol="tuning", seed=0,
                   num_samples=num_samples)


def resolve_temperatures(ctx: Context, spec: RunSpec, checkpoint: str | None = None, *, report_only: bool = False,
                         segments: list[Segment] | None = None) -> Temps:
    """The duo's temperatures: a given JointFixedTS checkpoint, else the fit on the held-out corruptions described
    above (computed and stored on first use; with `report_only` a missing fit is an error, never a computation)."""
    if checkpoint:
        folder = Path(checkpoint).expanduser().resolve()
        tl, ts = load_temperatures(folder)
        return Temps(folder, tl, ts, f"taken from the checkpoint \\texttt{{{latex_escape(folder.name)}}}")
    fp = spec.fingerprint(ctx.duo, ctx.batch_size, ctx.objective_logits, ctx.bn)
    folder = ctx.g.path("checkpoints") / "fixed_ts" / f"heldout_{spec.duo}_{spec.tta}_{fp}"
    origin = "fitted on the four held-out corruptions pooled"
    if (folder / "config.json").is_file():
        return Temps(folder, *load_temperatures(folder), origin)
    if report_only:
        raise FileNotFoundError(f"temperatures not fitted yet ({folder}); run without --report_only (or with --temps_only)")
    segments = segments or build_segments(spec, ctx.g)
    ensure_members(spec, ctx, segments)
    src = CacheSource(spec, ctx)
    held = pool([src.get(s.label) for s in segments])
    tl, ts, _ = O.fit_temperatures(held.z_l, held.z_s, held.y)
    tl, ts = store_temperatures(folder, round(float(tl), 4), round(float(ts), 4),
                                {"streams": [s.label for s in segments], "regime": f"{spec.tta}, both_indep",
                                 "duo": spec.duo, "num_samples": spec.num_samples})
    return Temps(folder, tl, ts, origin)


# ------------------------------------------------------------------------------------------------------ jobs

def mode_spec(mode: Mode, duo: str, tta: str, tta_kwargs: dict | None, seed: int, num_samples: int | None,
              temps: Temps) -> RunSpec:
    # protocol_args is not part of the cache key of an episodic run, so every corruption stream cached before the
    # clean stream was added is still valid: only "clean" is missing.
    return RunSpec(duo, tta=tta, tta_kwargs=tta_kwargs or {}, mode=mode.engine, protocol="episodic", seed=seed,
                   num_samples=num_samples, protocol_args={"clean": "val"},
                   loop_ts=(temps.T_l, temps.T_s) if mode.engine in DUO_MODES else None)


def mode_jobs(duo: str, tta: str, tta_kwargs: dict | None, seeds: Sequence[int], num_samples: int | None,
              temps: Temps, keys: Sequence[str] = MODE_KEYS) -> list[tuple[Mode, RunSpec]]:
    unknown = set(keys) - set(MODE_KEYS)
    if unknown:
        raise ValueError(f"unknown modes {sorted(unknown)}; known: {MODE_KEYS}")
    return [(m, mode_spec(m, duo, tta, tta_kwargs, s, num_samples, temps))
            for s in seeds for m in MODES if m.key in keys]


def job_label(job: tuple[Mode, RunSpec]) -> str:
    return f"{job[0].key}/seed{job[1].seed}"


def compute_members(ctx: Context, jobs: Sequence[tuple[Mode, RunSpec]], segments: list[Segment] | None = None) -> None:
    """Adapt the members of every job (a GPU job); streams already cached are skipped."""
    for mode, spec in jobs:
        print(f"== {job_label((mode, spec))}: {mode.label} ({spec.mode})")
        ensure_members(spec, ctx, segments)


def missing(ctx: Context, jobs: Sequence[tuple[Mode, RunSpec]]) -> dict[str, list[str]]:
    return {job_label(j): m for j in jobs if (m := missing_streams(j[1], ctx))}


def duo_rows(jobs: Sequence[tuple[Mode, RunSpec]], ctx: Context, temps: Temps,
             segments: list[Segment] | None = None) -> list[dict]:
    """One row per (mode, seed, stream, series) from the cached logits: series "duo" is the static duo's
    prediction, "large"/"small" are the members' own predictions under that mode."""
    gate = GateSpec("duo", kind="fixed_ts", fixed_ts=str(temps.folder))
    rows = []
    for mode, spec in jobs:
        res: EvalResult = evaluate(spec, ctx, [gate], report_only=True, segments=segments)
        rows += [{**r, "mode": mode.key, "engine_mode": mode.engine} for r in res.rows]
    return rows


# ---------------------------------------------------------------------------------------------------- rendering

def _with_mode_series(df: pd.DataFrame) -> pd.DataFrame:
    """Series "<mode>/<duo|large|small>", so every (mode, prediction) is one row of a table."""
    df = df.copy()
    df["series"] = df["mode"] + "/" + df["series"]
    return df


def split_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the corruption rows, the clean-stream rows) of `duo_rows`: the clean stream is reported next to the
    corruptions but is never part of an average over them."""
    is_clean = df["segment"] == CLEAN
    return df[~is_clean], df[is_clean]


def _modes_present(df: pd.DataFrame) -> list[Mode]:
    return [m for m in MODES if m.key in set(df["mode"])]


def _method_name(tta: str) -> str:
    return "Tent" if tta == "tent" else tta.upper()


def _has_bn(ctx: Context) -> bool:
    return any(ctx.duo[side].get("NORM") == "BN" for side in ("LARGE", "SMALL"))


def _frozen_text(ctx: Context) -> str:
    """How a model that does not adapt is run (the --bn flag), for the summary's caption: said only if a member has
    BatchNorm."""
    if not _has_bn(ctx):
        return "frozen weights"
    return ("frozen weights, BatchNorm on the test batch's statistics" if ctx.bn
            else "frozen weights in eval() mode, BatchNorm on the source statistics")


def _frozen_sentence(ctx: Context) -> str:
    """The same for the per-corruption table's caption, as a sentence."""
    kept = "Any model that does not adapt is frozen: it keeps its weights"
    if not _has_bn(ctx):
        return kept + "."
    return kept + (" and uses the test batch's normalization statistics." if ctx.bn
                   else ", in eval() mode, with the source normalization statistics.")


def _math_num(x: float) -> str:
    """A number as math-mode LaTeX: 0.9 -> '0.9', 0.00025 -> '2.5 \\times 10^{-4}', 0.001 -> '10^{-3}'."""
    if x == 0 or abs(x) >= 1e-2:
        return f"{x:g}"
    mant, exp = f"{x:.6e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"10^{{{int(exp)}}}" if mant == "1" else f"{mant} \\times 10^{{{int(exp)}}}"


def _tex_num(x: float) -> str:
    """The same for running text: a plain number as it is, a power of ten wrapped in $...$."""
    m = _math_num(x)
    return f"${m}$" if "^" in m else m


def _optim_text(optim: dict) -> str:
    """The optimizer of one adapting member (an OPTIM block of the duo config, after cfgs/tuned is applied), with every
    parameter src/tta/tent.py::setup_optimizer passes on; STEPS (gradient steps per batch) only when it is not 1."""
    num = lambda k: _tex_num(float(optim[k]))               # noqa: E731
    if optim["METHOD"] == "Adam":
        parts = [f"learning rate {num('LR')}", f"$\\beta_1 = {_math_num(float(optim['BETA']))}$", "$\\beta_2 = 0.999$",
                 f"weight decay {num('WD')}"]
    elif optim["METHOD"] == "SGD":
        parts = [f"learning rate {num('LR')}", f"momentum {num('MOMENTUM')}", f"dampening {num('DAMPENING')}",
                 f"weight decay {num('WD')}", "Nesterov" if optim["NESTEROV"] else "no Nesterov"]
    else:
        raise ValueError(f"no caption text for optimizer {optim['METHOD']!r}; add it to _optim_text")
    if int(optim.get("STEPS", 1)) != 1:
        parts.append(f"{int(optim['STEPS'])} steps per batch")
    return f"{optim['METHOD']} ({', '.join(parts)})"


def _model_names(ctx: Context) -> tuple[str, str]:
    return tuple(latex_escape(MODEL_NAMES.get(n, n)) for n in (ctx.duo["LARGE"]["NAME"], ctx.duo["SMALL"]["NAME"]))


def _models_text(ctx: Context, tta: str, severity: int) -> str:
    large, small = _model_names(ctx)
    return f"Large / small: {large} / {small}, adapted with {tta.upper()}, severity {severity}."


def _hyperparameters(ctx: Context) -> str:
    large, small = _model_names(ctx)
    large_opt, small_opt = (_optim_text(ctx.duo[side]["OPTIM"]) for side in ("LARGE", "SMALL"))
    return (f"Hyperparameters of the adapting models: batch size {ctx.batch_size}. {large}: {large_opt}. "
            f"{small}: {small_opt}.")


def _setting(ctx: Context, tta: str, temps: Temps, severity: int) -> str:
    """The summary table's caption sentences (it has one row per mode, named as in MODES)."""
    return (f"{_models_text(ctx, tta, severity)} {_hyperparameters(ctx)} "
            f"The duo's prediction in every mode is the same static duo $(z_L / T_L + z_S / T_S) / 2$ with "
            f"$T_L = {temps.T_l:.2f}$, $T_S = {temps.T_s:.2f}$, {temps.origin}. "
            f"\\emph{{Duo entropy}}: the adapting model(s) minimize the entropy of the duo's combined prediction, "
            f"both at once (Joint) or one while the other is frozen (Large / Small adapt: the named model adapts). "
            f"\\emph{{Own entropy}}: each adapting model minimizes its own entropy (Tent) and the duo only combines "
            f"them; Independent both adapts both, Independent large / small only the named one. "
            f"\\emph{{Reference}}: {_frozen_text(ctx)}; a model that does not adapt in another row is frozen in the "
            f"same way.")


def _max_std_points(df: pd.DataFrame, keys: Sequence[str]) -> float | None:
    """The largest std over seeds (in accuracy points) of any cell of the per-corruption table: a per-corruption
    or clean cell, or an Avg. (the std of the per-seed averages). None with a single seed."""
    df = df[df["series"].isin(list(keys))]
    corr, _ = split_clean(df)
    stds = pd.concat([df.groupby(["series", "segment"])["accuracy"].std(ddof=1),
                      per_seed_average(corr).groupby("series")["accuracy"].std(ddof=1)]).dropna()
    return float(stds.max()) * 100 if len(stds) else None


def render_corruption_table(df: pd.DataFrame, ctx: Context, tta: str, temps: Temps, *, label: str,
                            source: str) -> str:
    """Top-1 accuracy per corruption, one row per (mode, prediction) of ROWS (ImageNet-C paper layout), in sections:
    frozen, a single model adapting on its own entropy, the duo with members adapting on their own entropy, the duo
    with members adapting on the duo's. The clean stream is its own column. Means over seeds, no std in the cells."""
    modes = set(df["mode"])
    df = _with_mode_series(df)
    rows = [r for r in ROWS if r.key in set(df["series"])]
    method = _method_name(tta)
    groups = [(g.format(method=method), [(r.key, r.label) for r in rows if r.group == g])
              for g in dict.fromkeys(r.group for r in rows)]
    has_clean = CLEAN in set(df["segment"])
    n_seeds, n_corr = df["seed"].nunique(), df.loc[df["segment"] != CLEAN, "segment"].nunique()
    max_std = _max_std_points(df, [r.key for r in rows])
    caption = ("Top-1 accuracy (\\%) of each member alone and of the static duo, under each way of adapting. "
               + _models_text(ctx, tta, ctx.g.severity)
               + f" Each section header gives what is scored, a single model or the duo, and the entropy that the "
                 f"adapting models minimize: their \\emph{{own}} entropy, as in {method}, or the entropy of the "
                 f"\\emph{{duo}}'s combined prediction. Each row gives which models adapt. {_frozen_sentence(ctx)} "
                 f"The duo's prediction is always $(z_L / T_L + z_S / T_S) / 2$ with $T_L = {temps.T_l:.2f}$, "
                 f"$T_S = {temps.T_s:.2f}$, {temps.origin}."
               + (" With both models adapting, the own-entropy and duo-entropy rows are the Independent and Joint "
                  "modes of Section~\\ref{sec:duo_tent}." if {"indep_duo", "joint"} <= modes else "")
               + f" {_hyperparameters(ctx)}"
               + (f" \\emph{{Clean}}: the ImageNet validation images the corruptions are rendered from, adapted to like "
                  f"any other stream. It is not part of Avg., the mean over the {n_corr} corruptions." if has_clean else "")
               + (f" Mean over {n_seeds} seeds." + (f" Standard deviations are at most {max_std:.1f} points."
                                                    if max_std is not None else "") if n_seeds > 1 else ""))
    return corruption_table(df, groups, ctx.g.corruption_families, value="accuracy", caption=caption, label=label,
                            star=True, source=source, std=False, clean=CLEAN if has_clean else None,
                            group_rows=True, tabcolsep=3.5)


def render_summary_table(df: pd.DataFrame, ctx: Context, tta: str, temps: Temps, *, label: str, source: str) -> str:
    """Per mode, averaged over the corruptions in `df` (the clean stream is left out): each member's accuracy under
    that mode, then the duo's accuracy, ECE and NLL."""
    df, _ = split_clean(df)
    modes = _modes_present(df)
    stats = mean_std_over_seeds(per_seed_average(_with_mode_series(df)), ["series"]).set_index("series")

    def cell(series: str, col: str, decimals: int, pct: bool) -> str:
        r = stats.loc[series]
        return fmt_cell(float(r[f"{col}_mean"]), float(r[f"{col}_std"]), decimals, pct)

    header = ["Mode", "Large acc.", "Small acc.", "Duo acc.", "Duo ECE", "Duo NLL"]
    body, mid = [], []
    for i, m in enumerate(modes):
        if i and m.group != modes[i - 1].group:
            mid.append(i)
        body.append([latex_escape(m.label), cell(f"{m.key}/large", "accuracy", 1, True),
                     cell(f"{m.key}/small", "accuracy", 1, True), cell(f"{m.key}/duo", "accuracy", 1, True),
                     cell(f"{m.key}/duo", "ece", 1, True), cell(f"{m.key}/duo", "nll", 2, False)])
    n_corr, n_seeds = df["segment"].nunique(), df["seed"].nunique()
    caption = (f"Accuracy (\\%) of each member under each way of adapting, and accuracy, ECE (\\%) and NLL of the duo, "
               f"averaged over the {n_corr} corruptions. " + _setting(ctx, tta, temps, ctx.g.severity)
               + (f" Mean $\\pm$ std over {n_seeds} seeds." if n_seeds > 1 else ""))
    return latex_table(header, body, caption=caption, label=label, midrule_before=mid, star=False, resize=False,
                       source=source)


# ----------------------------------------------------------------------------------------------- script plumbing

def add_args(ap) -> None:
    ap.add_argument("--modes", nargs="+", default=MODE_KEYS, choices=MODE_KEYS, metavar="MODE",
                    help=f"adaptation modes to run (default: all of {MODE_KEYS})")
    ap.add_argument("--temps_checkpoint", default=None, metavar="DIR",
                    help="JointFixedTS checkpoint (a folder with config.json) for the duo's temperatures; "
                         "default: fitted on the 4 held-out corruptions, see adaptation_modes.py")
    ap.add_argument("--temps_only", action="store_true",
                    help="fit and store the temperatures, then stop (run this once before a --shard array so "
                         "that no two shards fit them at the same time)")
    ap.add_argument("--lr_large", type=float, default=None, metavar="LR",
                    help="replace the large member's learning rate (the rest of its tuned OPTIM stays). The learning "
                         "rates are part of every cache key and of the temperatures' checkpoint name, so a run with "
                         "other rates fits its own temperatures and shares no cache with the tuned ones")
    ap.add_argument("--lr_small", type=float, default=None, metavar="LR", help="the same for the small member")


def with_learning_rates(ctx: Context, lr_large: float | None, lr_small: float | None) -> Context:
    """A copy of `ctx` whose members use the given learning rates (None keeps the tuned one)."""
    duo = copy.deepcopy(ctx.duo)
    for side, lr in (("LARGE", lr_large), ("SMALL", lr_small)):
        if lr is not None:
            if lr <= 0:
                raise ValueError(f"learning rate of the {side.lower()} member must be positive, got {lr}")
            duo[side]["OPTIM"] = {**duo[side]["OPTIM"], "LR": lr}       # a new dict: the two sides may share one
    return replace(ctx, duo=duo)


def _lr_text(lr: float) -> str:
    return f"{lr:g}".replace(".", "p")          # no dot: the tag ends up in file names and LaTeX labels


def output_tag(args, default_duo: str, stamp: str | None = None) -> str:
    """File-name tag: the TTA method, `__<duo>` unless it is the default duo, `_bn` with --bn (the frozen members'
    BatchNorm statistics change the numbers, so the two settings must not overwrite each other). With --lr_large /
    --lr_small it adds `_lrL<large>_S<small>` (the rates given) and `_<stamp>`, so that a run with other learning
    rates never overwrites a table of the tuned ones, nor an earlier run of its own."""
    tag = args.tta + ("" if args.duo == default_duo else f"__{args.duo}") + bn_tag(args)
    lrs = [(k, v) for k, v in (("L", getattr(args, "lr_large", None)), ("S", getattr(args, "lr_small", None)))
           if v is not None]
    if not lrs:
        return tag
    tag += "_lr" + "_".join(f"{k}{_lr_text(v)}" for k, v in lrs)
    return tag + (f"_{stamp}" if stamp else "")


if __name__ == "__main__":
    import shutil
    import subprocess
    import tempfile

    import torch

    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.experiments.cache import MemberCache
    from src.experiments.runner import run_dir
    from src.utils.config import load_global

    torch.set_num_threads(2)
    g = load_global()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = make_context(tmp, batch_size=32, base=tiny_models(0))
        ctx.g.raw["paths"]["checkpoints"] = str(tmp / "checkpoints")
        held = [Segment(f"{c}/s5", FakeStream(150, seed=10 + i), True) for i, c in enumerate(g.tuning_corruptions)]
        test = [Segment(f"{c}/s5", FakeStream(150, seed=i), True) for i, c in enumerate(g.test_corruptions[:3])]
        test.append(Segment(CLEAN, FakeStream(150, seed=99), True))          # the clean stream rides along, as in the real run
        tspec = temperature_spec("tiny", "tent", None, None)
        assert tspec.fingerprint(ctx.duo, 32, "masked") == RunSpec("tiny", mode="both_indep").fingerprint(ctx.duo, 32, "masked"), \
            "the temperature run shares its cache with scripts/01 (same fingerprint as an episodic both_indep run)"

        # ---- temperatures: fitted once from the held-out streams, stored, then read back without recomputing ----
        try:
            resolve_temperatures(ctx, tspec, report_only=True, segments=held)
            raise AssertionError("--report_only must never fit")
        except FileNotFoundError as e:
            assert "temperatures not fitted" in str(e)
        temps = resolve_temperatures(ctx, tspec, segments=held)
        assert temps.T_l == round(temps.T_l, 4) and temps.T_s == round(temps.T_s, 4) and 0.05 <= temps.T_l <= 50
        stream_files = {p: p.stat().st_mtime_ns for p in MemberCache(run_dir(tspec, ctx)).dir.glob("*.pt")}
        again = resolve_temperatures(ctx, tspec, report_only=True)
        assert (again.T_l, again.T_s) == (temps.T_l, temps.T_s) and stream_files
        # the fit is the held-out fit of scripts/01: pooled NLL optimum of the cached (centered fp16) logits
        src = CacheSource(tspec, ctx)
        h = pool([src.get(s.label) for s in held])
        want = O.fit_temperatures(h.z_l, h.z_s, h.y)
        assert abs(temps.T_l - float(want[0])) <= 5e-5 and abs(temps.T_s - float(want[1])) <= 5e-5
        # first writer wins: a second job's different values do not overwrite the stored pair
        assert store_temperatures(temps.folder, 9.0, 9.0, {}) == (temps.T_l, temps.T_s)
        assert not list(temps.folder.glob(".config.*")), "no temp files are left behind"
        explicit = resolve_temperatures(ctx, tspec, checkpoint=str(temps.folder))
        assert (explicit.T_l, explicit.T_s) == (temps.T_l, temps.T_s) and explicit.folder == temps.folder.resolve()
        assert explicit.origin.startswith("taken from the checkpoint \\texttt{heldout\\_tiny\\_tent\\_")

        # ---- jobs: duo-entropy modes carry the temperatures in their spec, the others do not -------------------
        jobs = mode_jobs("tiny", "tent", None, [0, 1], None, temps)
        assert len(jobs) == 2 * len(MODES) and [job_label(j) for j in jobs[:2]] == ["joint/seed0", "large_anchor/seed0"]
        for m, s in jobs:
            assert (s.loop_ts == (temps.T_l, temps.T_s)) == (m.engine in DUO_MODES) and s.mode == m.engine
        assert len({s.fingerprint(ctx.duo, 32, "masked") for _, s in jobs}) == len(jobs), "every job has its own cache"
        for m, s in jobs:       # the clean stream is requested without changing any cache key: the corruptions stay cached
            assert s.protocol_args == {"clean": "val"} and s.fingerprint(ctx.duo, 32, "masked") == \
                replace(s, protocol_args={}).fingerprint(ctx.duo, 32, "masked")
        try:
            mode_jobs("tiny", "tent", None, [0], None, temps, keys=["nope"])
            raise AssertionError("unknown mode key must raise")
        except ValueError:
            pass

        # ---- run (tiny models, fake streams), then the rows ----------------------------------------------------
        assert set(missing(ctx, jobs)) == {job_label(j) for j in jobs}
        assert all(m[0] == CLEAN for m in missing(ctx, jobs).values()), "the clean stream is a stream of every job"
        compute_members(ctx, jobs, test)
        # `missing` asks the protocol for its (real) segment list, so on the fake streams use the cache directly
        for m, s in jobs:
            assert all(MemberCache(run_dir(s, ctx)).has(x.file_stem) for x in test)
        rows = duo_rows(jobs, ctx, temps, test)
        df = pd.DataFrame(rows)
        assert set(df["series"]) == {"duo", "large", "small"} and set(df["mode"]) == set(MODE_KEYS)
        assert len(df) == len(jobs) * len(test) * 3 and df["accuracy"].between(0, 1).all()
        # with nothing adapting the members are the frozen ones; the duo row of a mode is the fixed combination of its logits
        by = df.set_index(["mode", "seed", "segment", "series"])["accuracy"]
        for seg in (s.label for s in test):
            assert by[("large_anchor", 0, seg, "small")] == by[("frozen", 0, seg, "small")], "an anchor stays frozen"
            assert by[("small_anchor", 0, seg, "large")] == by[("frozen", 0, seg, "large")]
            assert by[("indep_large", 0, seg, "small")] == by[("frozen", 0, seg, "small")]

        # ---- tables ---------------------------------------------------------------------------------------------
        kw = dict(ctx=ctx, tta="tent", temps=temps, source="test")
        t1 = render_corruption_table(df, label="tab:adaptation_modes", **kw)
        t2 = render_summary_table(df, label="tab:adaptation_modes_summary", **kw)
        lines1 = [l.strip() for l in t1.splitlines()]
        body1 = [l for l in lines1 if l.endswith("\\\\") and "&" in l]
        names = ["Joint", "Large adapt", "Small adapt", "Independent large", "Independent small", "Independent both", "No adaptation"]
        assert len(body1) == 2 + len(ROWS)
        # the layout asked for: no Group column, group rows, members indented, Clean first and apart from the average
        assert "\\multirow{" not in t1 and "Group" not in t1 and sum(l.startswith("\\quad ") for l in lines1) == len(ROWS)
        assert "\\begin{tabular}{l|c|ccc|c}" in t1 and "\\setlength{\\tabcolsep}{3.5pt}" in t1
        assert "Method & Clean & Gauss. & Shot & Impulse & Avg. \\\\" in lines1
        assert all(l.count("&") == 5 for l in lines1 if l.startswith("\\quad ")) and "--" not in t1 and "pm" not in t1.split("\\begin{tabular}")[1]
        # the sections, in order, each with its rows: (what is scored, which entropy), then which models adapt
        shown, section = [], None
        for l in lines1:
            if l.startswith("\\multicolumn{6}{l}{\\textit{"):
                section = l[len("\\multicolumn{6}{l}{\\textit{"):-len("}} \\\\")]
            elif l.startswith("\\quad "):
                label, *cells = [c.strip() for c in l.rstrip(" \\").split("&")]
                shown.append((section, label[len("\\quad "):], cells))
        assert [(g, lab) for g, lab, _ in shown] == [(r.group.format(method="Tent"), r.label) for r in ROWS]
        assert list(dict.fromkeys(g for g, _, _ in shown)) == [
            "Frozen", "Single model, own entropy (Tent)", "Duo, own entropy", "Duo, duo entropy"]
        assert [lab for g, lab, _ in shown if g == "Duo, duo entropy"] == ["Large adapts", "Small adapts", "Both adapt (Joint)"]
        # every row reads its own (mode, prediction) of the data: a member alone comes from the run that adapts it
        # (Large adapts, single model: the large member of indep_large), a duo row from the duo of its mode's run
        for r, (_, _, cells) in zip(ROWS, shown):
            d = df[(df["mode"] == r.mode) & (df["series"] == r.series)]
            want_clean = d[d["segment"] == CLEAN]["accuracy"].mean() * 100
            want_avg = d[d["segment"] != CLEAN].groupby("seed")["accuracy"].mean().mean() * 100
            assert abs(float(cells[0]) - want_clean) <= 0.051 and abs(float(cells[-1]) - want_avg) <= 0.051, r
        assert t1.startswith("% Generated by test on 20") and t1.splitlines()[0].endswith(". Requires booktabs, graphicx.")
        assert f"$T_L = {temps.T_l:.2f}$" in t1 and "Mean over 2 seeds. Standard deviations are at most" in t1
        assert "It is not part of Avg., the mean over the 3 corruptions" in t1
        body2 = [l.strip() for l in t2.splitlines() if l.strip().endswith("\\\\") and "&" in l]
        assert len(body2) == 1 + len(names) and body2[0].startswith("Mode & Large acc.") and t2.count("\\midrule") == 3
        assert all(l.count("&") == 5 for l in body2) and "--" not in t2
        assert "averaged over the 3 corruptions" in t2 and "Independent duo" not in t1 + t2 and "anchor" not in t1 + t2
        # without a clean stream in the rows there is no Clean column (nor its caption sentence)
        no_clean = render_corruption_table(df[df["segment"] != CLEAN], label="x", **kw)
        assert "Clean" not in no_clean and "\\begin{tabular}{l|ccc|c}" in no_clean
        # the caption lists each member's optimizer and the batch size (the values the cached logits were made with)
        assert _tex_num(0.001) == "$10^{-3}$" and _tex_num(0.00025) == "$2.5 \\times 10^{-4}$" and _tex_num(0.9) == "0.9"
        assert _tex_num(0) == "0" and _tex_num(0.01) == "0.01" and _tex_num(5e-4) == "$5 \\times 10^{-4}$"
        sgd = {"METHOD": "SGD", "STEPS": 1, "LR": 0.00025, "MOMENTUM": 0.9, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}
        assert _optim_text(sgd) == ("SGD (learning rate $2.5 \\times 10^{-4}$, momentum 0.9, dampening 0, weight decay 0, "
                                    "no Nesterov)")
        assert _optim_text({**sgd, "NESTEROV": True, "WD": 1e-4, "STEPS": 3}) == (
            "SGD (learning rate $2.5 \\times 10^{-4}$, momentum 0.9, dampening 0, weight decay $10^{-4}$, Nesterov, "
            "3 steps per batch)")
        adam = {"METHOD": "Adam", "STEPS": 1, "LR": 1e-4, "BETA": 0.9, "WD": 0.0}
        assert _optim_text(adam) == "Adam (learning rate $10^{-4}$, $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, weight decay 0)"
        try:
            _optim_text({"METHOD": "RMSprop"})
            raise AssertionError("an optimizer without caption text must raise, not print a wrong caption")
        except ValueError:
            pass
        for side in ("LARGE", "SMALL"):
            shown = latex_escape(MODEL_NAMES.get(ctx.duo[side]["NAME"], ctx.duo[side]["NAME"]))
            assert f"{shown}: {_optim_text(ctx.duo[side]['OPTIM'])}" in t1 + t2
        assert f"Hyperparameters of the adapting models: batch size {ctx.batch_size}. " in t1 and "batch size 32" in t2
        # the caption has to compile: no bare power of ten outside math mode
        caption = t1.split("\\caption{")[1].split("\n")[0]
        assert caption.count("$") % 2 == 0

        # the caption says how the frozen BatchNorm model is run (--bn); the tag keeps the two settings apart
        assert "with the source normalization statistics" in t1 and "test batch" not in t1.split("\\begin{tabular}")[0]
        bn_ctx = make_context(tmp, batch_size=32, base=tiny_models(0), bn=True)
        bn_t1 = render_corruption_table(df, label="x", **{**kw, "ctx": bn_ctx})
        assert "it keeps its weights and uses the test batch's normalization statistics." in bn_t1
        assert "BatchNorm on the source statistics" in t2 and "BatchNorm on the test batch's statistics" in \
            render_summary_table(df, label="x", **{**kw, "ctx": bn_ctx})
        from types import SimpleNamespace as NS
        assert output_tag(NS(tta="tent", duo="d", bn=False), "d") == "tent" and output_tag(NS(tta="tent", duo="d", bn=True), "d") == "tent_bn"
        assert output_tag(NS(tta="tent", duo="e", bn=True), "d") == "tent__e_bn"
        # learning-rate overrides: tagged with the rates and the stamp; absent flags leave every name as it was
        lr = dict(tta="tent", duo="d", bn=True)
        assert output_tag(NS(**lr, lr_large=None, lr_small=None), "d", "S") == "tent_bn", "no override: no stamp either"
        assert output_tag(NS(**lr, lr_large=5e-4, lr_small=1e-3), "d", "20261019-153012") == \
            "tent_bn_lrL0p0005_S0p001_20261019-153012"
        assert output_tag(NS(**lr, lr_large=None, lr_small=2.5e-3), "d") == "tent_bn_lrS0p0025"
        assert "." not in output_tag(NS(**lr, lr_large=2.5e-4, lr_small=1e-5), "d", "x"), "the tag goes into file names"
        tuned_ctx = with_learning_rates(ctx, None, None)
        assert tuned_ctx.duo == ctx.duo and tuned_ctx.duo is not ctx.duo
        fast = with_learning_rates(ctx, 5e-4, None)
        assert fast.duo["LARGE"]["OPTIM"]["LR"] == 5e-4 and fast.duo["SMALL"]["OPTIM"] == ctx.duo["SMALL"]["OPTIM"]
        assert {k: v for k, v in fast.duo["LARGE"]["OPTIM"].items() if k != "LR"} == \
            {k: v for k, v in ctx.duo["LARGE"]["OPTIM"].items() if k != "LR"}, "only the rate changes"
        assert ctx.duo["LARGE"]["OPTIM"]["LR"] != 5e-4, "the original context is untouched"
        try:
            with_learning_rates(ctx, 0.0, None)
            raise AssertionError("a non-positive learning rate must raise")
        except ValueError:
            pass
        # the rates are part of the cache key (so of the temperatures' checkpoint too); the other side's alone is enough
        fp = lambda c, mode: RunSpec("tiny", mode=mode, loop_ts=(1.0, 1.0) if mode in DUO_MODES else None     # noqa: E731
                                     ).fingerprint(c.duo, 32, "masked")
        both = with_learning_rates(ctx, 5e-4, 1e-3)
        assert all(len({fp(ctx, m), fp(fast, m), fp(both, m), fp(with_learning_rates(ctx, None, 1e-3), m)}) == 4
                   for m in ("both_duo", "both_indep", "no_adapt")), "a changed rate must change every run's key"
        assert "learning rate $5 \\times 10^{-4}$" in render_summary_table(df, label="x", **{**kw, "ctx": fast}), \
            "the caption states the rates the run used"
        sub = render_summary_table(df[df["mode"].isin(["joint", "indep_duo"])], label="x", **kw)
        assert "Large adapt" not in sub and "Joint" in sub, "only the modes present get a row"
        only_joint = render_corruption_table(df[df["mode"] == "joint"], label="x", **kw)
        assert "Both adapt (Joint)" in only_joint and "Frozen" not in only_joint and "Section~" not in only_joint, \
            "only the rows (and sections) present are listed, and the caption names no mode that is absent"
        assert "Section~\\ref{sec:duo_tent}" in t1.split("\\begin{tabular}")[0]
        eata = render_corruption_table(df, label="x", **{**kw, "tta": "eata"})
        assert "Single model, own entropy (EATA)" in eata and "as in EATA" in eata and "Tent" not in eata.split("\\begin{tabular}")[0]
        if shutil.which("pdflatex"):
            doc = ("\\documentclass{article}\\usepackage{booktabs,multirow,graphicx}\\usepackage[landscape]{geometry}"
                   "\\begin{document}\n" + t1 + t2 + "\\end{document}\n")
            with tempfile.TemporaryDirectory() as latex_dir:
                (Path(latex_dir) / "t.tex").write_text(doc)
                r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"], cwd=latex_dir,
                                   capture_output=True, text=True, timeout=120)
                assert r.returncode == 0, r.stdout[-1500:]
            print("adaptation_modes: generated LaTeX compiles with pdflatex")
    print("adaptation_modes self-test passed")
