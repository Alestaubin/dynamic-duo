"""
temperature_study.py
====================
The data plumbing, row builder and table renderer behind scripts/01_tab_temperatures.py.

The study looks at 20 streams of one duo, adapted once and cached (src/experiments/cache.py): clean validation,
the 4 held-out corruptions, the 15 test corruptions. A `LogitSource` hands out the cached member logits of one
stream at a time; `study_rows` turns those into one dict of metrics per stream and knows nothing about models,
datasets or the cache, so it is tested on synthetic logits.

Vocabulary (the paper's). The duo is (z_L / T_L + z_S / T_S) / 2 throughout.
  Clean TS            (T_L, T_S) fitted on clean images
  Held-out TS         (T_L, T_S) fitted on the 4 held-out corruptions pooled. This is "Fixed TS" throughout.
  T_L*, T_S*          (T_L, T_S) fitted on the stream being scored (labels used; in-sample)
Oracles (labels used; in-sample; upper bounds), finest last:
  Corruption-wise TS  the accuracy at T_L*, T_S*
  Batch-wise TS       (T_L, T_S) refitted on every consecutive batch of B samples, for each B asked for
                      (one column each, the largest B first, so that the oracles get finer left to right)
  Per-sample oracle   (T_L, T_S) per sample, by gradient descent on that sample's NLL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import pandas as pd
import torch
import torch.nn.functional as F

from src.experiments import oracles as O
from src.experiments.cache import MemberCache
from src.experiments.protocols import build_segments
from src.experiments.cli import bn_tag
from src.experiments.runner import Context, run_dir
from src.experiments.specs import RunSpec
from src.utils.config import GlobalConfig
from src.utils.tables import fmt_cell, latex_escape, pretty_name, table_env


@dataclass
class Stream:
    label: str
    z_l: torch.Tensor      # (N, 1000) float32, on the compute device
    z_s: torch.Tensor
    y: torch.Tensor        # (N,) int64


@dataclass
class StudyLabels:
    clean: str
    heldout: list[str]
    test: list[str]

    @classmethod
    def from_config(cls, g: GlobalConfig, clean: str = "val") -> "StudyLabels":
        sev = g.severity
        return cls("clean" if clean == "val" else "clean_train_slice",
                   [f"{c}/s{sev}" for c in g.tuning_corruptions], [f"{c}/s{sev}" for c in g.test_corruptions])

    @property
    def all(self) -> list[str]:
        return [self.clean, *self.heldout, *self.test]


class LogitSource(Protocol):
    def get(self, label: str) -> Stream: ...


class CacheSource:
    """The cached member logits of one run (centered fp16 on disk), one stream at a time."""

    def __init__(self, spec: RunSpec, ctx: Context):
        self.cache = MemberCache(run_dir(spec, ctx))
        self.device = ctx.device

    def get(self, label: str) -> Stream:
        d = self.cache.read(label.replace("/", "__"))
        return Stream(label, d["z_l"].to(self.device), d["z_s"].to(self.device), d["labels"].to(self.device))


class DictSource:
    def __init__(self, streams: dict[str, Stream]):
        self.streams = streams

    def get(self, label: str) -> Stream:
        return self.streams[label]


def study_spec(duo: str, g: GlobalConfig, *, tta: str = "tent", mode: str = "both_indep", seed: int = 0,
               num_samples: int | None = None, clean: str = "val", tta_kwargs: dict | None = None) -> RunSpec:
    labels = StudyLabels.from_config(g, clean)
    corruptions = [c for c in g.test_corruptions + g.tuning_corruptions]
    del labels
    return RunSpec(duo, tta=tta, tta_kwargs=tta_kwargs or {}, mode=mode, protocol="episodic", seed=seed,
                   num_samples=num_samples, protocol_args={"corruptions": corruptions, "clean": clean})


def missing_streams(spec: RunSpec, ctx: Context) -> list[str]:
    cache = MemberCache(run_dir(spec, ctx))
    return [s.label for s in build_segments(spec, ctx.g) if not cache.has(s.file_stem)]


def pool(streams: Sequence[Stream]) -> Stream:
    return Stream("pooled", torch.cat([s.z_l for s in streams]), torch.cat([s.z_s for s in streams]),
                  torch.cat([s.y for s in streams]))


def _single(z: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    return float((z.argmax(1) == y).float().mean()), float(F.cross_entropy(z, y))


@torch.no_grad()
def reference_temperatures(src: LogitSource, labels: StudyLabels) -> dict[str, tuple[float, float]]:
    """{"clean": (T_L, T_S) fitted on clean validation, "heldout": fitted on the held-out corruptions pooled}."""
    c = src.get(labels.clean)
    tl, ts, _ = O.fit_temperatures(c.z_l, c.z_s, c.y)
    out = {"clean": (float(tl), float(ts))}
    del c
    h = pool([src.get(l) for l in labels.heldout])
    tl, ts, _ = O.fit_temperatures(h.z_l, h.z_s, h.y)
    out["heldout"] = (float(tl), float(ts))
    return out


def _group(labels: StudyLabels, label: str) -> str:
    return "clean" if label == labels.clean else "heldout" if label in labels.heldout else "test"


@torch.no_grad()
def study_rows(src: LogitSource, labels: StudyLabels, *, batch_sizes: Sequence[int] = (16, 128), steps: int = 200
               ) -> list[dict]:
    """One row per stream (clean, held-out, test), every column of the table. Per-sample oracles start from the
    Held-out-TS temperatures (so they are never worse than Held-out TS)."""
    ref = reference_temperatures(src, labels)
    rows = []
    for label in labels.all:
        s = src.get(label)
        row = {"segment": label, "group": _group(labels, label), "n": len(s.y),
               "clean_T_l": ref["clean"][0], "clean_T_s": ref["clean"][1],
               "heldout_T_l": ref["heldout"][0], "heldout_T_s": ref["heldout"][1]}
        row["acc_large"], row["nll_large"] = _single(s.z_l, s.y)
        row["acc_small"], row["nll_small"] = _single(s.z_s, s.y)
        for name, T in (("clean_ts", ref["clean"]), ("heldout_ts", ref["heldout"])):
            row[f"acc_{name}"], row[f"nll_{name}"] = O.acc_nll(s.z_l, s.z_s, s.y, *T)

        tl, ts, _ = O.fit_temperatures(s.z_l, s.z_s, s.y)
        row["T_l_opt"], row["T_s_opt"] = float(tl), float(ts)
        row["acc_corruption"], row["nll_corruption"] = O.acc_nll(s.z_l, s.z_s, s.y, float(tl), float(ts))

        for b in batch_sizes:
            btl, bts = O.batchwise_temperatures(s.z_l, s.z_s, s.y, b)
            row[f"acc_batch{b}"], row[f"nll_batch{b}"] = O.acc_nll(s.z_l, s.z_s, s.y, btl, bts)

        ptl, pts = O.per_sample_temperatures(s.z_l, s.z_s, s.y, init=ref["heldout"], steps=steps)
        row["acc_per_sample"], row["nll_per_sample"] = O.acc_nll(s.z_l, s.z_s, s.y, ptl, pts)
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ rendering

def _per_seed_mean(df: pd.DataFrame, labels: Sequence[str], cols: Sequence[str]) -> pd.DataFrame:
    """The Average row: mean over `labels` per seed, as a pseudo-stream named 'average'."""
    sub = df[df["segment"].isin(list(labels))].groupby("seed")[list(cols)].mean().reset_index()
    sub["segment"], sub["group"] = "average", "average"
    return sub


def _cell(values: pd.Series, *, pct: bool = True, decimals: int = 1, std: bool = True) -> str:
    v = values.dropna()
    if not len(v):
        return "--"
    return fmt_cell(float(v.mean()), float(v.std(ddof=1)) if std and len(v) > 1 else None, decimals, pct)


def render_table(df: pd.DataFrame, labels: StudyLabels, *, batch_sizes: Sequence[int], models: tuple[str, str],
                 regime: str, frozen: bool, severity: int, clean_desc: str, table_labels: Sequence[str],
                 source: str) -> str:
    """The merged table: per stream T_L*, T_S*, the members, the two fixed-temperature fits and the oracles
    (corruption-wise, batch-wise for each B in `batch_sizes`, per-sample). `table_labels[0]` is the label,
    the others are extra \\label{}s for the same table. The batch-wise columns run from the largest B to the
    smallest, whatever order `batch_sizes` is given in."""
    batch_sizes = sorted(set(batch_sizes), reverse=True)
    nb = len(batch_sizes)
    ref = {k: tuple(float(df[f"{k}_T_{m}"].mean()) for m in "ls") for k in ("clean", "heldout")}
    num = [c for c in df.columns if c.startswith(("acc_", "nll_", "T_"))]
    df = pd.concat([df, _per_seed_mean(df, labels.test, num)], ignore_index=True)
    cols = [("T_l_opt", False, 2), ("T_s_opt", False, 2), ("acc_large", True, 1), ("acc_small", True, 1),
            ("acc_clean_ts", True, 1), ("acc_heldout_ts", True, 1), ("acc_corruption", True, 1),
            *[(f"acc_batch{b}", True, 1) for b in batch_sizes], ("acc_per_sample", True, 1)]

    # header: groups / the batch-wise columns' common title / one name per column
    first_batch = 9                                                # 1-based: after Stream, 2 T*, 2 members, 2 fixed, Corr.-wise
    head = [
        "& \\multicolumn{2}{c|}{Optimal temps.} & \\multicolumn{2}{c|}{Members} & \\multicolumn{2}{c|}{Fixed temperatures} "
        f"& \\multicolumn{{{nb + 2}}}{{c}}{{Oracles (labels used)}} \\\\",
        " & ".join([""] * 8 + [f"\\multicolumn{{{nb}}}{{c}}{{Batch-wise TS}}", ""]) + " \\\\",
        f"\\cmidrule(lr){{{first_batch}-{first_batch + nb - 1}}}",
        " & ".join(["Stream", "$T_L^*$", "$T_S^*$", "Large", "Small", "Clean TS", "Held-out TS", "Corr.-wise",
                    *[f"$B{{=}}{b}$" for b in batch_sizes], "Per-sample"]) + " \\\\",
        "\\midrule",
    ]
    body = list(head)
    blocks = [[labels.clean], labels.heldout, labels.test, ["average"]]
    for i, labs in enumerate(blocks):
        if i:
            body.append("\\midrule")
        for lab in labs:
            sub = df[df["segment"] == lab]
            name = "Average (15 test)" if lab == "average" else pretty_name(lab) + ("$^\\dagger$" if lab in labels.heldout else "")
            cells = [name] + [_cell(sub[c], pct=p, decimals=d, std=p) for c, p, d in cols]
            body.append(" & ".join(cells) + " \\\\")

    n_seeds = df["seed"].nunique()
    batches = (f"on every batch of {batch_sizes[0]} consecutive samples" if nb == 1 else
               f"on every batch of $B$ consecutive samples, $B \\in \\{{{', '.join(map(str, batch_sizes))}\\}}$")
    caption = (
        f"Temperatures and top-1 accuracy (\\%) of {'the frozen duo' if frozen else 'the duo'} at severity {severity}. "
        f"Large / Small: {models[0]} / {models[1]} {regime}. $T_L^*, T_S^*$: the temperatures minimizing the NLL of the duo "
        f"on each stream (labels used), which define the corruption-wise oracle. \\emph{{Fixed temperatures}}, fitted without "
        f"test labels: Clean TS on {clean_desc} ($T_L = {ref['clean'][0]:.2f}$, $T_S = {ref['clean'][1]:.2f}$), and Held-out TS "
        f"on the four held-out corruptions ($^\\dagger$) pooled ($T_L = {ref['heldout'][0]:.2f}$, $T_S = {ref['heldout'][1]:.2f}$, "
        f"in-sample on those four rows). \\emph{{Oracles}}, all using labels and evaluated in-sample: corruption-wise TS fits the "
        f"two temperatures on each stream, batch-wise TS refits them {batches}, and the per-sample oracle fits them on each "
        f"sample by gradient descent. The last row averages the 15 test corruptions."
        + (f" Mean $\\pm$ std over {n_seeds} seeds." if n_seeds > 1 else "")
    )
    return table_env(body, "l|cc|cc|cc|" + "c" * (nb + 2), caption, table_labels[0], star=True, resize=False,
                     source=source, tabcolsep=4.5, extra_labels=table_labels[1:])


# ------------------------------------------------------------------------------------------ script plumbing

MODEL_NAMES = {"vit_b_16": "ViT-B/16", "vit_l_16": "ViT-L/16", "vit_s_16": "ViT-S/16", "resnet50": "ResNet-50",
               "resnet50_v1": "ResNet-50 (V1)", "vit_b_16_augreg": "ViT-B/16 (AugReg)",
               "vit_b_16_augreg2": "ViT-B/16 (AugReg2)"}


def add_study_args(ap) -> None:
    ap.add_argument("--mode", default="both_indep", choices=["both_indep", "no_adapt"],
                    help="both_indep: the members adapt independently with --tta (the paper's setting); "
                         "no_adapt: frozen weights (BatchNorm on the source statistics, or on the test batch's with --bn)")
    ap.add_argument("--clean_source", default="val", choices=["val", "train_slice"],
                    help="images that define 'clean' (default: the 50,000 validation images)")
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 128], metavar="B",
                    help="batch sizes of the batch-wise TS oracle: one column each, the largest first (default: 16 128)")
    ap.add_argument("--steps", type=int, default=200, help="gradient steps of the per-sample oracle")


def study_tag(args, default_duo: str | None = None) -> str:
    """File-name tag naming the regime: 'frozen' or the TTA method; `_bn` with --bn (in every mode: only frozen
    BatchNorm members change, but the name must not depend on that); '_trainslice' if clean comes from train;
    `__<duo>` if `default_duo` is given and this is another duo (so a second duo never overwrites the default
    duo's table)."""
    tag = ("frozen" if args.mode == "no_adapt" else args.tta) + bn_tag(args)
    tag += "_trainslice" if args.clean_source == "train_slice" else ""
    return tag + (f"__{args.duo}" if default_duo is not None and args.duo != default_duo else "")


def regime_text(args) -> str:
    if args.mode != "no_adapt":
        return f"adapted independently with {args.tta.upper()}"
    return ("with frozen weights, their BatchNorm layers using test-batch statistics" if args.bn
            else "with frozen weights in eval() mode, their BatchNorm layers using the source statistics")


def clean_text(args) -> str:
    return "clean ImageNet validation data" if args.clean_source == "val" else "a clean slice of the ImageNet training set"


def study_specs(args, ctx: Context, seeds: Sequence[int]) -> list[RunSpec]:
    return [study_spec(args.duo, ctx.g, tta=args.tta, mode=args.mode, seed=s, num_samples=args.num_samples,
                       clean=args.clean_source, tta_kwargs=args.tta_kwargs) for s in seeds]


def compute_members(args, ctx: Context, specs: Sequence[RunSpec]) -> None:
    """Adapt the members for this shard's seeds (a GPU job); a no-op with --report_only."""
    from src.experiments.cli import select_jobs
    from src.experiments.runner import ensure_members

    if args.report_only:
        return
    for spec in select_jobs(list(specs), lambda sp: f"seed{sp.seed}", args.shard, args.only):
        ensure_members(spec, ctx)


def all_cached(ctx: Context, specs: Sequence[RunSpec]) -> bool:
    ok = True
    for spec in specs:
        miss = missing_streams(spec, ctx)
        if miss:
            ok = False
            print(f"seed {spec.seed}: member logits missing for {len(miss)}/20 streams (e.g. {miss[:3]}); "
                  f"run without --report_only to compute them")
    return ok


def model_names(ctx: Context) -> tuple[str, str]:
    a, b = ctx.duo["LARGE"]["NAME"], ctx.duo["SMALL"]["NAME"]
    return MODEL_NAMES.get(a) or latex_escape(a), MODEL_NAMES.get(b) or latex_escape(b)      # raw names go into a caption


if __name__ == "__main__":
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from src.utils.config import load_global

    torch.manual_seed(0)
    g = load_global()
    labels = StudyLabels.from_config(g)
    assert labels.clean == "clean" and len(labels.heldout) == 4 and len(labels.test) == 15 and len(labels.all) == 20

    def synth(label, n, sharp_l, sharp_s, acc_l, acc_s, K=60):
        y = torch.randint(0, K, (n,))
        zs = []
        for sharp, acc in ((sharp_l, acc_l), (sharp_s, acc_s)):
            z = torch.randn(n, K)
            pred = torch.where(torch.rand(n) < acc, y, torch.randint(0, K, (n,)))
            z[torch.arange(n), pred] += sharp
            zs.append(z)
        return Stream(label, zs[0], zs[1], y)

    streams = {"clean": synth("clean", 1500, 7, 5, 0.85, 0.8)}
    for i, l in enumerate(labels.heldout):
        streams[l] = synth(l, 1000, 6, 4, 0.5 + 0.05 * i, 0.55)
    for i, l in enumerate(labels.test):
        streams[l] = synth(l, 600, 6, 4, 0.2 + 0.04 * i, 0.5 - 0.02 * i)     # the large model degrades, the small one less
    src = DictSource(streams)

    # ---- rows: every stream has every column (no blanks), and the hierarchy of definitions holds ----------------
    rows = study_rows(src, labels, batch_sizes=(16, 128), steps=120)
    assert [r["segment"] for r in rows] == labels.all
    eps = 1e-5
    for r in rows:
        assert all(v is not None and v == v for v in r.values()), r                # no NaN anywhere: the table has no "--"
        assert 0.05 <= r["T_l_opt"] <= 50 and 0.05 <= r["T_s_opt"] <= 50
        assert r["nll_corruption"] <= r["nll_clean_ts"] + eps and r["nll_corruption"] <= r["nll_heldout_ts"] + eps
        assert r["nll_batch128"] <= r["nll_corruption"] + eps, r
        assert r["nll_batch16"] <= r["nll_batch128"] + eps, "16 divides 128, so its batches nest inside the 128-batches"
        assert r["nll_per_sample"] <= r["nll_heldout_ts"] + eps, "per-sample starts at the Held-out-TS temperatures"
    clean_row = rows[0]
    assert clean_row["acc_clean_ts"] == clean_row["acc_corruption"] and clean_row["nll_clean_ts"] == clean_row["nll_corruption"], \
        "on the clean row Clean TS IS the optimum"
    # the fit on the pooled held-out streams is in-sample for them: it beats the clean temperatures there, on average
    held = [r for r in rows if r["group"] == "heldout"]
    assert sum(r["nll_heldout_ts"] for r in held) <= sum(r["nll_clean_ts"] for r in held) + eps
    only_b = study_rows(src, labels, batch_sizes=(64,), steps=5)                     # any batch size, any number of them
    assert "acc_batch64" in only_b[0] and "acc_batch16" not in only_b[0]

    # ---- rendering, with two seeds so the +- path is exercised ---------------------------------------------------
    def with_seed(rs, seed, jitter):
        return [{**r, "seed": seed, **{k: v + jitter for k, v in r.items() if k.startswith("acc_")}} for r in rs]
    df = pd.DataFrame(with_seed(rows, 0, 0.0) + with_seed(rows, 1, 0.004))
    kw = dict(models=("ViT-B/16", "ResNet-50"), regime="adapted independently with TENT", frozen=False, severity=5,
              clean_desc="clean ImageNet validation data", table_labels=["tab:temperatures:tent", "tab:oracles:tent"],
              source="test")
    t = render_table(df, labels, batch_sizes=(16, 128), **kw)
    body = [l.strip() for l in t.splitlines() if l.strip().endswith("\\\\") and "&" in l]
    assert len(body) == 3 + 1 + 4 + 15 + 1, len(body)                      # 3 header rows (groups, batch title, names)
    assert body[0].count("&") == 4 and body[1].count("&") == 9 and all(l.count("&") == 10 for l in body[2:])   # 11 columns
    assert body[2].startswith("Stream") and "$B{=}128$ & $B{=}16$ & Per-sample" in body[2], "largest B first"
    assert body[3].startswith("Clean (val)") and body[4].startswith("Speckle noise$^\\dagger$") and "pm" in body[-1]
    assert body[-1].startswith("Average (15 test)") and "--" not in t
    assert t.count("\\midrule") == 4 and "\\cmidrule(lr){9-10}" in t and "\\multicolumn{2}{c}{Batch-wise TS}" in t
    assert "\\label{tab:temperatures:tent}" in t and "\\label{tab:oracles:tent}" in t and "{tabular}{l|cc|cc|cc|cccc}" in t
    assert "$B \\in \\{128, 16\\}$" in t and "\\begin{table*}[t]" in t and "tabcolsep}{4.5pt}" in t
    t1 = render_table(pd.DataFrame(with_seed(rows, 0, 0.0)), labels, batch_sizes=(16,), **{**kw, "frozen": True})
    body1 = [l.strip() for l in t1.splitlines() if l.strip().endswith("\\\\") and "&" in l]
    assert body1[0].count("&") == 4 and all(l.count("&") == 9 for l in body1[1:]) and "every batch of 16 consecutive samples" in t1
    assert "{tabular}{l|cc|cc|cc|ccc}" in t1 and "the frozen duo" in t1 and "\\multicolumn{1}{c}{Batch-wise TS}" in t1

    # the frozen regime is named by --bn: the two settings give different numbers, so different files and captions
    from types import SimpleNamespace as NS
    a = lambda **k: NS(**{"mode": "no_adapt", "bn": False, "tta": "tent", "clean_source": "val", **k})   # noqa: E731
    assert study_tag(a()) == "frozen" and study_tag(a(bn=True)) == "frozen_bn" and study_tag(a(clean_source="train_slice")) == "frozen_trainslice"
    assert study_tag(a(mode="both_indep")) == "tent" and study_tag(a(mode="both_indep", bn=True)) == "tent_bn", \
        "--bn renames the files in every mode, even where it changes no number"
    assert study_tag(a(bn=True, clean_source="train_slice")) == "frozen_bn_trainslice" and study_tag(a(duo="e", bn=True), "d") == "frozen_bn__e"
    assert study_tag(a(duo="d"), "d") == "frozen" and study_tag(a(duo="e"), "d") == "frozen__e", "another duo has its own files"
    assert study_tag(a(duo="e")) == "frozen", "no default given: the old behaviour"
    assert "source statistics" in regime_text(a()) and "test-batch statistics" in regime_text(a(bn=True))
    assert regime_text(a(mode="both_indep")) == "adapted independently with TENT"

    if shutil.which("pdflatex"):
        with tempfile.TemporaryDirectory() as tmp:
            doc = ("\\documentclass{article}\\usepackage{booktabs,multirow,graphicx}\\usepackage[landscape]{geometry}"
                   "\\begin{document}\n" + t + t1 + "\\end{document}\n")
            (Path(tmp) / "t.tex").write_text(doc)
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"], cwd=tmp,
                               capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, r.stdout[-1500:]
        print("temperature_study: generated LaTeX compiles with pdflatex")
    print("temperature_study self-test passed")
