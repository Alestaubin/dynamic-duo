"""
gated_duos.py
=============
The logic behind scripts/03b_tab_gated_duos.py: how Dynamic Duos' reliability gate compares with the fixed combination, in
the two ways of adapting the members (scripts/03's regimes), on one set of streams. One gate setting (proxy, proxy batch size
b_t, beta) is evaluated twice:

  Vanilla DD   the gate sits OUTSIDE the adaptation loop: the members adapt on their own entropy (Independent both, the indep
               regime's learning rates) and never see it, so it is replayed on the cached logits of that run (no model).
  Joint DD     the gate sits INSIDE the loop: both members adapt on the entropy of the gated prediction (the weight is a
               constant of each step), at the joint regime's learning rates. It cannot be replayed: each seed is one live run
               of scripts/09_run_joint_dd.py (`--streams test clean`), whose .rows.csv this module reads.

Both are put beside their fixed-temperature baselines (the Independent both and Joint rows of script 03, the same static duo
(z_L / T_L + z_S / T_S) / 2, read from script 03's caches) and the frozen members, so the table shows what the gate adds in each
regime. Every row is a (method, prediction) pair of ROWS, in the series naming of script 03 ("<method>/<duo|large|small>").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments.protocols import Segment
from src.experiments.results import mean_std_over_seeds, per_seed_average
from src.experiments.runner import Context, replay
from src.experiments.specs import GateSpec, RunSpec
from src.utils import provenance
from src.utils.tables import corruption_table, latex_escape

VANILLA, JOINT_DD = "vanilla_dd", "joint_dd"
# (section, [(series, row label)]): the sections are those of script 03's per-corruption table
FROZEN, OWN, DUO = "Frozen", "Duo, own entropy", "Duo, duo entropy"
ROWS = (
    (FROZEN, [("frozen/large", "Large"), ("frozen/small", "Small"), ("frozen/duo", "Duo (fixed TS)")]),
    (OWN, [("indep_duo/duo", "Independent, fixed TS"), (f"{VANILLA}/duo", "Vanilla DD")]),
    (DUO, [("joint/duo", "Joint, fixed TS"), (f"{JOINT_DD}/duo", "Joint DD")]),
)
MODES = ("frozen", "indep_duo", "joint")           # script 03's runs the fixed-TS rows are read from


def vanilla_gate(g, temps: AM.Temps, *, proxy: str, beta: float, b_t: int) -> GateSpec:
    """The gate under test, as the replay builds it (identity calibration, log pooling, no filter, the paper's defaults
    of cfgs/global.yaml `gate:` except the three given)."""
    return GateSpec.from_global(g, "duo", proxy=proxy, beta=beta, proxy_batch_size=b_t, fixed_ts=str(temps.folder))


def gate_mode(kind: str, beta: float) -> str:
    """The method name of one gated run when several betas share a table: "vanilla_dd@0.25", "joint_dd@0.5"."""
    return f"{kind}@{beta:g}"


def vanilla_rows(spec: RunSpec, ctx: Context, gate: GateSpec, segments: list[Segment] | None = None,
                 mode: str = VANILLA) -> list[dict]:
    """Vanilla DD: `gate` replayed on the cached logits of the Independent both run `spec` (a GPU-free step). Rows of the
    gated prediction only, as method `mode` ("vanilla_dd"; `gate_mode` names one per beta)."""
    res = replay(spec, ctx, [gate], segments)
    return [{**r, "mode": mode} for r in res.rows if r["series"] == gate.name]


def joint_dd_path(directory: Path, duo: str, proxy: str, b_t: int, beta: float, seed: int) -> Path:
    """Where scripts/09_run_joint_dd.py writes the rows of one run."""
    return Path(directory) / f"joint_dd__{duo}__{proxy}_bt{b_t}__beta{beta:g}__seed{seed}.rows.csv"


def joint_dd_rows(directory: Path, duo: str, tta: str, protocol_tag: str, proxy: str, b_t: int, beta: float,
                  seeds: Sequence[int], mode: str = JOINT_DD) -> list[dict]:
    """Joint DD: the rows of script 09's runs, one file per seed, as method `mode` ("joint_dd"; `gate_mode` names one per beta;
    series duo / large / small)."""
    files = {s: joint_dd_path(directory, duo, proxy, b_t, beta, s) for s in seeds}
    absent = [str(f) for f in files.values() if not f.is_file()]
    if absent:
        raise FileNotFoundError(f"Joint DD rows missing: {absent}; run scripts/09_run_joint_dd.py --streams test clean "
                                f"--cells {proxy}:{b_t} --betas {beta:g} --seeds {' '.join(map(str, seeds))}")
    rows = []
    for seed, f in files.items():
        for r in pd.read_csv(f).to_dict("records"):
            rows.append({"duo": duo, "tta": tta, "protocol_tag": protocol_tag, "seed": seed, **r, "mode": mode})
    return rows


def joint_dd_rates(directory: Path, duo: str, proxy: str, b_t: int, beta: float, seeds: Sequence[int]) -> tuple[float, float]:
    """(ViT / large rate, small rate) the Joint DD runs of `beta` used, from the .meta.json script 09 writes beside each. The file name
    holds no rate, so a table must read them from here: a run without a meta file (older than this check), or seeds that disagree, is an
    error rather than a caption that guesses."""
    pairs = {}
    for s in seeds:
        f = joint_dd_path(directory, duo, proxy, b_t, beta, s).with_suffix("").with_suffix(".meta.json")
        if not f.is_file():
            raise FileNotFoundError(f"{f} missing: rerun scripts/09_run_joint_dd.py (it records the learning rates) or use another folder")
        m = json.loads(f.read_text())
        pairs[s] = (float(m["lr_large"]), float(m["lr_small"]))
    if len(set(pairs.values())) != 1:
        raise ValueError(f"Joint DD runs of beta {beta:g} in {directory} used different learning rates: {pairs}")
    return next(iter(pairs.values()))


def gated_row_defs(betas: Sequence[float], *, indep: bool = True, joint: bool = True, joint_betas: Sequence[float] | None = None
                   ) -> tuple[list[tuple[str, AM.Row]], dict[str, str]]:
    """(extra rows, relabel) for `AM.render_corruption_table`: each beta of `betas` adds a Vanilla DD row after Independent both and a
    Joint DD row after Joint (series "<gate_mode>/duo"; only for the betas of `joint_betas` when given: a beta without Joint DD runs has no row),
    and those two ROWS rows are renamed to say they are the fixed combination."""
    extra, relabel = [], {}
    for run, base, section, kind, name in ((indep, "indep_duo", AM.OWN, VANILLA, "Independent"), (joint, "joint", AM.DUO, JOINT_DD, "Joint")):
        if not run:
            continue
        relabel[f"{base}/duo"] = f"Both adapt ({name}), Fixed TS"
        for beta in betas:
            if kind == JOINT_DD and joint_betas is not None and beta not in joint_betas:
                continue
            extra.append((f"{base}/duo", AM.Row(gate_mode(kind, beta), "duo", section,
                                                f"Both adapt ({name}), {'Vanilla' if kind == VANILLA else 'Joint'} DD, beta {beta:g}")))
    return extra, relabel


def gate_caption(ctx: Context, proxy: str, b_t: int, betas: Sequence[float],
                 rates: dict[float, tuple[float, float]], missing_joint: Sequence[float] = (),
                 joint_rates: tuple[float, float] | None = None) -> str:
    """The sentence(s) a table with gated rows adds to its caption: what the gate is, where it sits in each kind of run, and the Joint DD
    learning rates (`rates[beta]`: those of its runs, so the optimizer is `duo_ctx`'s with the rate replaced). A beta whose rates are the
    Joint row's own (`joint_rates`, the pair tuned WITHOUT the gate) is said to be not retuned for the gate; the others were tuned with it."""
    large, small = AM._model_names(ctx)
    text = (f"The rows with a gate weight the members with $w_L = \\sigma(\\beta (s_L - s_S))$, $s$ the logit of the {latex_escape(proxy)} proxy "
            f"(identity calibration, no temporal filter) computed on proxy batches of {b_t} samples, and pool the logits as "
            f"$w_L z_L / T_L + w_S z_S / T_S$ with $w_S = 1 - w_L$; Fixed TS is $\\beta = 0$, and $\\beta$ is "
            f"{', '.join(f'{b:g}' for b in betas[:-1])}{' or ' if len(betas) > 1 else ''}{betas[-1]:g} in the rows. "
            f"In Vanilla DD the gate sits outside the adaptation loop (the members adapt on their own entropy, as in Independent, and never see "
            f"it); in Joint DD it sits inside, both members adapt on the entropy of the gated prediction, and the weight is a constant of each step.")
    same = [b for b in betas if b in rates and joint_rates is not None
            and all(abs(r - j) <= 1e-9 * abs(j) for r, j in zip(rates[b], joint_rates))]       # the Joint row's own rates: the gate was not retuned
    tuned = [b for b in betas if b in rates and b not in same]
    beta_list = lambda bs: "$\\beta = " + ", ".join(f"{b:g}" for b in bs) + "$"                  # noqa: E731
    at = beta_list(tuned) if same else "each $\\beta$"
    if tuned:
        text += (f" Joint DD uses, at {at}, the learning rates tuned together with the gate at that "
                 "$\\beta$ on the four held-out corruptions (same optimizer as Joint otherwise): "
                 + "; ".join(f"$\\beta = {b:g}$: {large} {AM._tex_num(rates[b][0])}, {small} {AM._tex_num(rates[b][1])}" for b in tuned)
                 + ".")
    if same:
        text += (f" Joint DD at {beta_list(same)} uses the learning rates of the Joint row ({large} {AM._tex_num(rates[same[0]][0])}, "
                 f"{small} {AM._tex_num(rates[same[0]][1])}), which were tuned on the four held-out corruptions without the gate: they were not retuned for it.")
    if missing_joint:
        text += (f" There is no Joint DD row for $\\beta = {', '.join(f'{b:g}' for b in missing_joint)}$: it needs live runs of the gate inside the loss at "
                 f"learning rates tuned together with the gate, and none exist for this duo yet.")
    return text


# ---------------------------------------------------------------------------------------------------- rendering

def _gate_text(proxy: str, beta: float, b_t: int) -> str:
    return (f"Vanilla DD and Joint DD weight the members with $w_L = \\sigma(\\beta (s_L - s_S))$, $s$ the logit of the "
            f"{latex_escape(proxy)} proxy (identity calibration, no temporal filter) computed on proxy batches of {b_t} "
            f"samples, $\\beta = {beta:g}$, and pool the logits as $w_L z_L / T_L + w_S z_S / T_S$ with $w_S = 1 - w_L$. "
            f"In Vanilla DD the gate sits outside the adaptation loop (the members adapt on their own entropy and never "
            f"see it); in Joint DD it sits inside, and both members adapt on the entropy of the gated prediction.")


def render_table(df: pd.DataFrame, ctx: Context, duo_ctx: Context, tta: str, temps: AM.Temps, *, proxy: str, beta: float,
                 b_t: int, label: str, source: str) -> str:
    """Top-1 accuracy per corruption (script 03's layout and its Clean column) of ROWS. Means over seeds."""
    present = set(df["mode"] + "/" + df["series"])
    groups = [(g, [(k, name) for k, name in rows if k in present]) for g, rows in ROWS]
    groups = [(g, rows) for g, rows in groups if rows]
    d = df.assign(series=df["mode"] + "/" + df["series"])
    has_clean = AM.CLEAN in set(d["segment"])
    n_seeds, n_corr = d["seed"].nunique(), d.loc[d["segment"] != AM.CLEAN, "segment"].nunique()
    caption = ("Top-1 accuracy (\\%) of the gated duos against the static duo, in each way of adapting. "
               + AM._models_text(ctx, tta, ctx.g.severity)
               + " " + _gate_text(proxy, beta, b_t)
               + f" The static duo is $(z_L / T_L + z_S / T_S) / 2$ with $T_L = {temps.T_l:.2f}$, $T_S = {temps.T_s:.2f}$, "
                 f"{temps.origin}, and the gates use the same $T_L$, $T_S$. {AM._frozen_sentence(ctx)} "
                 f"{AM._hyperparameters(ctx, duo_ctx)}"
               + (" The own-entropy rows (Independent, Vanilla DD) use the first set of learning rates, the duo-entropy "
                  "rows (Joint, Joint DD) the second." if AM._optims(ctx) != AM._optims(duo_ctx) else "")
               + (f" \\emph{{Clean}}: the ImageNet validation images, not part of Avg., the mean over the {n_corr} "
                  f"corruptions." if has_clean else "")
               + (f" Mean over {n_seeds} seeds." if n_seeds > 1 else ""))
    return corruption_table(d, groups, ctx.g.corruption_families, value="accuracy", caption=caption, label=label, star=True,
                            source=source, std=False, clean=AM.CLEAN if has_clean else None, group_rows=True, tabcolsep=3.5,
                            models=provenance.run_label(ctx.duo, tta, ctx.bn))


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per row of ROWS, averaged over the corruptions (the clean stream left out) and the seeds: accuracy, ECE and NLL (%, %,
    nats), with the std over seeds when there are several."""
    corr, _ = AM.split_clean(df)
    d = corr.assign(series=corr["mode"] + "/" + corr["series"])
    stats = mean_std_over_seeds(per_seed_average(d), ["series"]).set_index("series")
    keys = [k for _, rows in ROWS for k, _ in rows if k in stats.index]
    names = {k: name for _, rows in ROWS for k, name in rows}
    out = pd.DataFrame(index=[names[k] for k in keys])
    for col, scale, nd in (("accuracy", 100, 2), ("ece", 100, 2), ("nll", 1, 3)):
        out[col] = [round(stats.loc[k, f"{col}_mean"] * scale, nd) for k in keys]
        if stats["n_seeds"].max() > 1:
            out[f"{col}_std"] = [round(stats.loc[k, f"{col}_std"] * scale, nd) for k in keys]
    return out


if __name__ == "__main__":
    import tempfile

    import torch

    from src.calibrators.joint_fixed_TS import JointFixedTS
    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.experiments.runner import ensure_members
    from src.utils.config import load_global

    torch.set_num_threads(2)
    g = load_global()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = make_context(tmp, batch_size=32, base=tiny_models(0))
        ctx.g.raw["paths"]["checkpoints"] = str(tmp / "checkpoints")
        duo_ctx = AM.with_learning_rates(ctx, 5e-4, 1e-3)
        test = [Segment(f"{c}/s5", FakeStream(150, seed=i), True) for i, c in enumerate(g.test_corruptions[:3])]
        test.append(Segment(AM.CLEAN, FakeStream(150, seed=99), True))
        ts = tmp / "ts"
        JointFixedTS(Tl=1.4, Ts=0.9, verbose=False).save(str(ts))
        temps = AM.Temps(ts, 1.4, 0.9, "fitted on the four held-out corruptions pooled")
        seeds = [0, 1]

        # ---- the fixed-TS rows come from script 03's runs, Vanilla DD is a replay of the Independent both run ---------
        jobs = AM.mode_jobs("tiny", "tent", None, seeds, None, temps, list(MODES))
        AM.compute_members(ctx, jobs, test, duo_ctx=duo_ctx)
        rows = AM.duo_rows(jobs, ctx, temps, test, duo_ctx=duo_ctx)
        gate = vanilla_gate(ctx.g, temps, proxy="nnn", beta=0.5, b_t=64)
        assert gate.proxy_batch_size == 64 and gate.beta == 0.5 and gate.proxy == "nnn" and gate.fixed_ts == str(ts)
        assert (gate.pool, gate.filter, gate.calib_method) == ("log", "none", "identity")
        spec_of = {(m.key, s.seed): s for m, s in jobs}
        for seed in seeds:
            rows += vanilla_rows(spec_of[("indep_duo", seed)], ctx, gate, test)
        vanilla = [r for r in rows if r["mode"] == VANILLA]
        assert len(vanilla) == len(seeds) * len(test) and {r["series"] for r in vanilla} == {"duo"}
        tag = rows[0]["protocol_tag"]

        # a gate with beta = 0 weighs the members equally: it is the static duo of the same logits, exactly
        flat = vanilla_gate(ctx.g, temps, proxy="nnn", beta=0.0, b_t=64)
        for r0 in vanilla_rows(spec_of[("indep_duo", 0)], ctx, flat, test):
            ref = next(r for r in rows if (r["mode"], r["seed"], r["segment"], r["series"]) == ("indep_duo", 0, r0["segment"], "duo"))
            assert abs(r0["accuracy"] - ref["accuracy"]) < 1e-9, r0["segment"]

        # ---- Joint DD: script 09's files, one per seed --------------------------------------------------------------
        d = tmp / "jdd"
        d.mkdir()
        try:
            joint_dd_rows(d, "tiny", "tent", tag, "nnn", 64, 0.5, seeds)
            raise AssertionError("missing runs must be named, not skipped")
        except FileNotFoundError as e:
            assert "scripts/09_run_joint_dd.py" in str(e) and "seed0" in str(e)
        for seed in seeds:
            pd.DataFrame([{"segment": s.label, "series": ser, "n": 150, "accuracy": 0.3 + 0.1 * seed + 0.01 * i, "ece": 0.1,
                           "nll": 2.0, "entropy": 1.0} for i, s in enumerate(test) for ser in ("duo", "large", "small")]
                         ).to_csv(joint_dd_path(d, "tiny", "nnn", 64, 0.5, seed), index=False)
        jd = joint_dd_rows(d, "tiny", "tent", tag, "nnn", 64, 0.5, seeds)
        assert len(jd) == len(seeds) * len(test) * 3 and {r["mode"] for r in jd} == {JOINT_DD} and {r["seed"] for r in jd} == {0, 1}
        df = pd.DataFrame(rows + jd)

        # ---- the table and the summary ------------------------------------------------------------------------------
        t = render_table(df, ctx, duo_ctx, "tent", temps, proxy="nnn", beta=0.5, b_t=64, label="tab:x", source="test")
        lines = [l.strip() for l in t.splitlines()]
        shown, section = [], None
        for l in lines:
            if l.startswith("\\multicolumn{") and "\\textit{" in l:
                section = l.split("\\textit{")[1].split("}")[0]
            elif l.startswith("\\quad "):
                shown.append((section, l[len("\\quad "):].split("&")[0].strip()))
        assert shown == [(g_, name) for g_, rows_ in ROWS for _, name in rows_], shown
        cap = t.split("\\caption{")[1].split("\n")[0]
        assert cap.count("$") % 2 == 0 and "proxy batches of 64 samples" in cap and "$\\beta = 0.5$" in cap
        assert "outside the adaptation loop" in cap and "Mean over 2 seeds" in cap and "the second." in cap
        assert "Models that adapt on their own entropy:" in cap and "Models that adapt on the duo entropy:" in cap
        for key, _ in (k for _, rows_ in ROWS for k in rows_):          # every row reads its own series of the data
            sub = df[(df["mode"] + "/" + df["series"] == key) & (df["segment"] != AM.CLEAN)]
            assert not sub.empty, key
        sm = summary(df)
        assert list(sm.index) == [name for _, rows_ in ROWS for _, name in rows_] and "accuracy_std" in sm
        want = 100 * df[(df["mode"] == JOINT_DD) & (df["series"] == "duo") & (df["segment"] != AM.CLEAN)].groupby("seed")["accuracy"].mean().mean()
        assert abs(sm.loc["Joint DD", "accuracy"] - want) < 0.006
        # only the rows present are listed
        only = render_table(df[df["mode"].isin(["frozen", VANILLA])], ctx, duo_ctx, "tent", temps, proxy="nnn", beta=0.5, b_t=64,
                            label="x", source="t")
        body = only.split("\\begin{tabular}")[1]               # (the caption names both gates whichever rows are listed)
        assert "Joint DD" not in body and "Vanilla DD" in body

        # ---- several betas in script 03's table: Vanilla DD replays, Joint DD files with their own rates -----------------
        betas = [0.25, 0.5]
        jdir = tmp / "jdd_gated"
        jdir.mkdir()
        lrs = {0.25: (1.13e-3, 2.83e-3), 0.5: (8e-4, 2.83e-3)}
        for beta in betas:
            for seed in seeds:
                pd.DataFrame([{"segment": s.label, "series": ser, "n": 150, "accuracy": 0.2 + 0.1 * beta + 0.01 * i, "ece": 0.1, "nll": 2.0,
                               "entropy": 1.0} for i, s in enumerate(test) for ser in ("duo", "large", "small")]
                             ).to_csv(joint_dd_path(jdir, "tiny", "nnn", 64, beta, seed), index=False)
        try:
            joint_dd_rates(jdir, "tiny", "nnn", 64, 0.25, seeds)
            raise AssertionError("a run without a meta file must not get a guessed caption")
        except FileNotFoundError as e:
            assert "meta.json" in str(e)
        for beta in betas:
            for seed in seeds:
                m = {"lr_large": lrs[beta][0], "lr_small": lrs[beta][1] * (1.5 if (beta, seed) == (0.5, 1) else 1)}
                joint_dd_path(jdir, "tiny", "nnn", 64, beta, seed).with_suffix("").with_suffix(".meta.json").write_text(json.dumps(m))
        assert joint_dd_rates(jdir, "tiny", "nnn", 64, 0.25, seeds) == lrs[0.25]
        try:
            joint_dd_rates(jdir, "tiny", "nnn", 64, 0.5, seeds)
            raise AssertionError("seeds run at different rates must be refused")
        except ValueError:
            pass
        joint_dd_path(jdir, "tiny", "nnn", 64, 0.5, 1).with_suffix("").with_suffix(".meta.json").write_text(
            json.dumps({"lr_large": lrs[0.5][0], "lr_small": lrs[0.5][1]}))
        rates = {b: joint_dd_rates(jdir, "tiny", "nnn", 64, b, seeds) for b in betas}
        assert rates == lrs

        grows = [r for r in rows if r["mode"] not in (VANILLA,)]
        for beta in betas:
            g_ = vanilla_gate(ctx.g, temps, proxy="nnn", beta=beta, b_t=64)
            for seed in seeds:
                grows += vanilla_rows(spec_of[("indep_duo", seed)], ctx, g_, test, mode=gate_mode(VANILLA, beta))
            grows += joint_dd_rows(jdir, "tiny", "tent", tag, "nnn", 64, beta, seeds, mode=gate_mode(JOINT_DD, beta))
        gdf = pd.DataFrame(grows)
        assert {gate_mode(VANILLA, b) for b in betas} | {gate_mode(JOINT_DD, b) for b in betas} <= set(gdf["mode"])
        assert gate_mode(VANILLA, 0.25) == "vanilla_dd@0.25" and gate_mode(JOINT_DD, 0.5) == "joint_dd@0.5"
        extra, relabel = gated_row_defs(betas)
        text = gate_caption(ctx, "nnn", 64, betas, rates)
        assert text.count("$") % 2 == 0 and "$\\beta$ is 0.25 or 0.5 in the rows" in text and "proxy batches of 64 samples" in text
        assert text.count("$\\beta = 0.25$:") == 1 and "$1.13 \\times 10^{-3}$" in text and "$8 \\times 10^{-4}$" in text
        assert "not retuned" not in text and "at each $\\beta$" in text
        t_mix = gate_caption(ctx, "nnn", 64, betas, rates, joint_rates=rates[0.5])           # beta 0.5 runs at the Joint row's own rates
        assert "at $\\beta = 0.25$, the learning rates tuned together" in t_mix and t_mix.count("not retuned") == 1 and t_mix.count("$") % 2 == 0
        assert "$\\beta = 0.5$: " not in t_mix.split("not retuned")[0].split("tuned together")[1]
        t_all = gate_caption(ctx, "nnn", 64, betas, {b: rates[0.5] for b in betas}, joint_rates=rates[0.5])
        assert "tuned together with the gate" not in t_all and "$\\beta = 0.25, 0.5$ uses the learning rates of the Joint row" in t_all
        assert gate_caption(ctx, "nnn", 64, betas, rates, joint_rates=(0.1, 0.1)) == text     # other rates: the former caption, word for word
        gt = AM.render_corruption_table(gdf, ctx, "tent", temps, label="x", source="t", duo_ctx=duo_ctx, extra_rows=extra, relabel=relabel,
                                        gate_text=text)
        shown, section = [], None
        for l in (l.strip() for l in gt.splitlines()):
            if l.startswith("\\multicolumn{") and "\\textit{" in l:
                section = l.split("\\textit{")[1].split("}")[0]
            elif l.startswith("\\quad "):
                shown.append((section, l[len("\\quad "):].split("&")[0].strip()))
        own = [n for s_, n in shown if s_ == "Duo, own entropy"]
        duo_e = [n for s_, n in shown if s_ == "Duo, duo entropy"]
        assert own[-3:] == ["Both adapt (Independent), Fixed TS", "Both adapt (Independent), Vanilla DD, beta 0.25",
                            "Both adapt (Independent), Vanilla DD, beta 0.5"], own
        assert duo_e[-3:] == ["Both adapt (Joint), Fixed TS", "Both adapt (Joint), Joint DD, beta 0.25", "Both adapt (Joint), Joint DD, beta 0.5"], duo_e
        cap = gt.split("\\caption{")[1].split("\n")[0]
        assert "except in the rows with a gate" in cap and "is always" not in cap and text in cap and cap.count("$") % 2 == 0
        # without gated rows the table is the one of script 03, word for word
        plain = AM.render_corruption_table(df[df["mode"].isin(["frozen", "indep_duo", "joint"])], ctx, "tent", temps, label="x", source="t", duo_ctx=duo_ctx)
        same = AM.render_corruption_table(gdf[gdf["mode"].isin(["frozen", "indep_duo", "joint"])], ctx, "tent", temps, label="x", source="t",
                                          duo_ctx=duo_ctx, extra_rows=extra, relabel={}, gate_text="")
        assert plain == same and "Both adapt (Joint)" in plain and "is always" in plain
        # a mode that was not run adds no row, and no row of the other kind is renamed
        only_v, rl = gated_row_defs(betas, joint=False)
        assert all(r.mode.startswith(VANILLA) for _, r in only_v) and list(rl) == ["indep_duo/duo"]
    print("gated_duos self-test passed")
