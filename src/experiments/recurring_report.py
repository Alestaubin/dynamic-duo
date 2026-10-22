"""
recurring_report.py
===================
The figure and the tables of scripts/03c_recurring_duos.py, from the per-sample correctness the three runs stored
(src/experiments/recurring.py). Four lines, in this order and colors (a validated categorical set; the frozen reference is a neutral dashed line):

    Frozen duo    frozen run, the static duo (z_L / T_L + z_S / T_S) / 2           gray, dashed
    Fixed TS duo  Independent run (members adapt on their own entropy), same duo   aqua
    Vanilla DD    the same members, the gate outside the loop                      blue
    Joint DD      members adapt on the gated entropy, the gate inside the loop     orange

Figure (full text width): the whole stream on top (accuracy per block of images/10 samples, i.e. ten points per corruption visit;
every shift is a faint line, every round start a labeled one), and below it the first and the last round zoomed so that all 15
shifts are labeled with their corruption. Mean over seeds (a band when there are several). Tables, in the layout of CoTTA's
Table 4: one column per corruption in stream order (accuracy averaged over the rounds), then the mean; and the mean per round.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.experiments import adaptation_modes as AM
from src.experiments import gated_duos as GD
from src.experiments import recurring as RC
from src.utils import provenance
from src.utils.tables import latex_escape, timestamp

# (run, series, label, color, linestyle): drawn in this order, the first at the back. Slots 1-3 of the reference palette validate
# all-pairs (CVD dE >= 9); the frozen duo is a neutral reference line, so it carries no hue. Aqua is below 3:1 on white: every
# line is directly labeled, and the tables carry the numbers.
LINES = (
    ("frozen", "ts", "Frozen duo", "#6b6a66", (0, (4, 2))),
    ("indep", "ts", "Fixed TS duo", "#1baf7a", "-"),
    ("indep", "gate", "Vanilla DD", "#2a78d6", "-"),
    ("joint_dd", "gate", "Joint DD", "#eb6834", "-"),
)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
SHORT = {"gaussian_noise": "Gaussian", "shot_noise": "shot", "impulse_noise": "impulse", "defocus_blur": "defocus",
         "glass_blur": "glass", "motion_blur": "motion", "zoom_blur": "zoom", "snow": "snow", "frost": "frost", "fog": "fog",
         "brightness": "brightness", "contrast": "contrast", "elastic_transform": "elastic\\_trans", "pixelate": "pixelate",
         "jpeg_compression": "jpeg"}
PLAIN = {k: v.replace("\\_", "_") for k, v in SHORT.items()}


class Stream:
    """The stored runs of one stream: `acc[label]` is a (seeds, samples) uint8 array of per-sample correctness, and the visit
    structure (`part_n`, `part_label`, `part_round`) every run shares."""

    def __init__(self, acc: dict[str, np.ndarray], part_n: np.ndarray, part_label: list[str], part_round: np.ndarray, meta: dict):
        self.acc, self.part_n, self.part_label, self.part_round, self.meta = acc, part_n, part_label, part_round, meta
        if len(set(part_n.tolist())) != 1:
            raise ValueError(f"every visit must have the same number of images, got {sorted(set(part_n.tolist()))}")
        self.images = int(part_n[0])
        self.rounds = int(part_round.max()) + 1
        self.corruptions = [l.split("/")[0] for l in part_label[: len(part_label) // self.rounds]]
        self.seeds = next(iter(acc.values())).shape[0]

    @property
    def n_visits(self) -> int:
        return len(self.part_n)

    def per_visit(self, label: str) -> np.ndarray:
        """Accuracy (%) per (seed, round, corruption)."""
        a = self.acc[label].reshape(self.seeds, self.rounds, len(self.corruptions), self.images)
        return a.mean(-1) * 100

    def blocks(self, label: str, per_visit: int = 10) -> np.ndarray:
        """Accuracy (%) per block of images/per_visit samples, (seeds, blocks)."""
        size = max(self.images // per_visit, 1)
        n = self.acc[label].shape[1] // size * size
        return self.acc[label][:, :n].reshape(self.seeds, -1, size).mean(-1) * 100


def load_stream(path_of, seeds: Sequence[int], runs: Sequence[tuple[str, str, str]] = tuple(l[:3] for l in LINES)) -> Stream:
    """Read the runs of every seed (`path_of(run, seed)`) into a Stream, checking that they saw the same visits."""
    acc, ref, meta = {}, None, None
    for run, series, label in runs:
        per_seed = []
        for seed in seeds:
            d = RC.load_run(path_of(run, seed))
            if series not in d:
                raise KeyError(f"{path_of(run, seed)} has no series {series!r}")
            key = (d["part_n"].tolist(), list(d["part_label"]), d["part_round"].tolist())
            if ref is not None and key != ref:
                raise ValueError(f"{path_of(run, seed)} saw other visits than the others: the runs must share one stream")
            ref, meta = key, meta or d["meta"]
            per_seed.append(d[series])
        acc[label] = np.stack(per_seed)
    return Stream(acc, np.asarray(ref[0]), ref[1], np.asarray(ref[2]), meta)


# ------------------------------------------------------------------------------------------------------------ figure

def _spread(ys: dict[str, float], gap: float) -> dict[str, float]:
    """Label heights: the lines' end values, pushed apart to at least `gap` (so direct labels never overlap)."""
    order = sorted(ys, key=ys.get)
    out, prev = {}, None
    for k in order:
        out[k] = ys[k] if prev is None else max(ys[k], prev + gap)
        prev = out[k]
    shift = (out[order[-1]] - ys[order[-1]]) / 2 if order else 0
    return {k: v - shift for k, v in out.items()}


def plot_stream(stream: Stream, path: Path, *, width: float = 7.2, height: float = 4.7, dpi: int = 300, footer: str = "") -> list[Path]:
    """Write the figure as `path` (.png) and next to it as .pdf; returns both. `footer`: the line naming the models, the run and the time
    (provenance.footer), drawn on the figure and stored in the files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    labels = [l[2] for l in LINES]
    style = {l[2]: (l[3], l[4]) for l in LINES}
    per_visit, size = 10, max(stream.images // 10, 1)
    x_block = (np.arange(stream.acc[labels[0]].shape[1] // size) + 0.5) * size          # sample index of each block's middle
    n_total = stream.n_visits * stream.images
    allv = np.concatenate([stream.blocks(l, per_visit).ravel() for l in labels])
    lo, hi = np.floor(allv.min() / 5) * 5 - 1, np.ceil(allv.max() / 5) * 5 + 1

    zoom = stream.rounds > 1
    fig = plt.figure(figsize=(width, height), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(2 if zoom else 1, 2, height_ratios=[1.0, 1.0] if zoom else None, hspace=0.62, wspace=0.07,
                          left=0.065, right=0.895, top=0.9 if zoom else 0.74, bottom=0.115)
    top = fig.add_subplot(gs[0, :])
    axes = [(top, 0, n_total)]
    if zoom:
        first = fig.add_subplot(gs[1, 0])
        last = fig.add_subplot(gs[1, 1], sharey=first)
        r_size = len(stream.corruptions) * stream.images
        axes += [(first, 0, r_size), (last, n_total - r_size, n_total)]
        last.tick_params(labelleft=False)

    for ax, a, b in axes:
        ax.set_facecolor("white")
        ax.set_xlim(a, b)
        ax.set_ylim(lo, hi)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    top.set_xlim(0, n_total)

    def label_shifts(ax, a: int) -> None:
        """Every shift of the round that starts at sample `a` as a dashed line, labeled with its corruption."""
        for j in range(len(stream.corruptions)):
            lo_x = a + j * stream.images
            if j:
                ax.axvline(lo_x, color=INK2, linewidth=0.45, linestyle="--", zorder=1)
            ax.text(lo_x + stream.images / 2, 1.015, PLAIN[stream.corruptions[j]], transform=ax.get_xaxis_transform(), rotation=90,
                    ha="center", va="bottom", fontsize=5.6, color=INK2)
        rnd = int(stream.part_round[min(a // stream.images, stream.n_visits - 1)]) + 1
        ax.set_title(f"Round {rnd}", fontsize=6.5, color=INK, loc="left", pad=34)

    if zoom:
        # the whole stream: a faint line at every shift, a stronger labeled one at every round start; the zoomed rounds name the shifts
        for k in range(1, stream.n_visits):
            new_round = stream.part_round[k] != stream.part_round[k - 1]
            top.axvline(k * stream.images, color=INK2 if new_round else GRID, linewidth=0.7 if new_round else 0.35,
                        linestyle="--" if new_round else "-", zorder=1)
        for r in range(stream.rounds):
            mid = (r + 0.5) * len(stream.corruptions) * stream.images
            top.text(mid, 1.015, f"Round {r + 1}", transform=top.get_xaxis_transform(), ha="center", va="bottom", fontsize=6, color=INK2)
        for ax, a, b in axes[1:]:
            label_shifts(ax, a)
    else:
        label_shifts(top, 0)                     # one round: the single panel names every shift

    for label in labels:
        color, ls = style[label]
        y = stream.blocks(label, per_visit)
        mean = y.mean(0)
        for ax, a, b in axes:
            if stream.seeds > 1:
                ax.fill_between(x_block, y.min(0), y.max(0), color=color, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(x_block, mean, color=color, linestyle=ls, linewidth=0.8 if ax is top else 1.0, zorder=3, label=label,
                    solid_capstyle="round")

    # direct labels at the right end of the top panel (heights pushed apart), and a legend
    ends = {l: float(stream.blocks(l, per_visit).mean(0)[-per_visit:].mean()) for l in labels}      # the last visit's level
    for l, yv in _spread(ends, (hi - lo) * 0.075).items():
        top.annotate(l, (n_total, ends[l]), xytext=(n_total * 1.012, yv), textcoords="data", fontsize=6.2, color=INK, va="center",
                     annotation_clip=False, arrowprops=dict(arrowstyle="-", color=style[l][0], linewidth=0.6, shrinkA=0, shrinkB=1))
    top.set_ylabel("Accuracy (%)")
    top.set_xlabel("Observed samples", labelpad=1)
    if zoom:
        axes[1][0].set_ylabel("Accuracy (%)")
        for ax in (axes[1][0], axes[2][0]):
            ax.set_xlabel("Observed samples", labelpad=1)
    h, l_ = top.get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=len(labels), frameon=False, fontsize=6.5, bbox_to_anchor=(0.48, 1.0),
               handlelength=2.2, columnspacing=1.6)
    out = provenance.save_figure(fig, path, footer or "unlabeled figure: no run text given", dpi=dpi)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------- gate reaction

BETAS = (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
# one-hue ramp for beta (light = small beta); the gate is the blue entity of the other figure. Members are neutral lines:
# the two are told apart by dash pattern and direct label, not by hue.
BETA_COLORS = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#104281", "#0d366b")
MEMBER = {"large": ("ViT-B/16", "#3d3c39", (0, (5, 2))), "small": ("ResNet-50", "#9a9992", (0, (1.5, 1.5, 5, 1.5)))}


def load_gate_records(run_file: Path) -> list[dict]:
    import json
    return json.loads(Path(run_file).with_suffix(".gate.json").read_text())


def gate_curves(run: dict, records: Sequence[dict], b_t: int) -> dict[str, np.ndarray]:
    """Per proxy batch of the gate (b_t consecutive samples from the stream's start; the last may be shorter): the members' and the
    duos' TRUE accuracy over the same samples (from the stored per-sample correctness), and the gate's own record (proxy scores r,
    their logits x). `w(beta)` of the result is the gate weight w_L = sigmoid(beta (x_L - x_S)) at any beta, so one run gives every beta."""
    n_samples = len(next(v for k, v in run.items() if k in ("large", "small")))
    n = len(records)
    edges = np.minimum(np.arange(n + 1) * b_t, n_samples)
    if edges[-1] != n_samples:
        raise ValueError(f"{n} proxy batches of {b_t} cover {edges[-1]} samples, the run has {n_samples}")
    width = np.diff(edges)

    def per_batch(c: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0], np.cumsum(c, dtype=np.int64)])
        return (cs[edges[1:]] - cs[edges[:-1]]) / width

    out = {s: per_batch(run[s]) for s in ("large", "small", "ts", "gate") if s in run}
    out.update(x=(edges[:-1] + edges[1:]) / 2, r_l=np.array([r["r_l"] for r in records]), r_s=np.array([r["r_s"] for r in records]),
               x_l=np.array([r["x_l"] for r in records]), x_s=np.array([r["x_s"] for r in records]),
               w_l=np.array([r["w_l"] for r in records]))
    return out


def gate_weight(curves: dict[str, np.ndarray], beta: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-beta * (curves["x_l"] - curves["x_s"])))


def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty(len(y), dtype=float)
    acc = float(y[0])
    for i, v in enumerate(y):
        acc = alpha * float(v) + (1 - alpha) * acc
        out[i] = acc
    return out


def plot_gate_reaction(rows: Sequence[dict], path: Path, *, images: int, n_corruptions: int = 15, betas: Sequence[float] = BETAS,
                       alpha: float = 0.1, width: float = 7.2, row_height: float = 2.15, dpi: int = 300, footer: str = "") -> list[Path]:
    """The gate against the true accuracy, in the layout of the CSSRC draft's Figure 1: one row per (run, window of the stream).
    Left: per-proxy-batch accuracy of each member (dashed) and of the duos (solid), EMA-smoothed over the faint raw values, and the
    two proxy scores (dotted). Right: the gate weight w_L at every beta (a one-hue ramp, beta = the run's own bold) against the
    members' accuracy (dashed). `rows`: dicts {title, curves (gate_curves), window (first, last sample), beta (the run's), duos:
    [(series key, label, color)]}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(len(rows), 2, figsize=(width, row_height * len(rows) + 0.5), dpi=dpi, facecolor="white", squeeze=False,
                             gridspec_kw=dict(left=0.07, right=0.975, top=0.93, bottom=0.155, hspace=0.62, wspace=0.17))
    round_len = n_corruptions * images
    for (axl, axr), row in zip(axes, rows):
        c, (a, b) = row["curves"], row["window"]
        inside = (c["x"] >= a) & (c["x"] <= b)
        x = c["x"][inside]
        for ax in (axl, axr):
            ax.set_facecolor("white")
            ax.set_xlim(a, b)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for k in range(int(a // images) + 1, int(b // images) + 1):                    # every shift; a round start is stronger
                new_round = k % n_corruptions == 0
                ax.axvline(k * images, color=INK2 if new_round else GRID, linewidth=0.6 if new_round else 0.3,
                           linestyle="--" if new_round else "-", zorder=1)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
            ax.set_xlabel("Samples processed", labelpad=1)
        n_rounds_shown = (b - a) / round_len
        if n_rounds_shown > 1.5:                                                           # name the rounds of a long window
            for r in range(int(a // round_len), int(np.ceil(b / round_len))):
                mid = np.clip((r + 0.5) * round_len, a, b)
                axr.text(mid, 1.02, f"R{r + 1}", transform=axr.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.4, color=INK2)
        # left: accuracies and proxy scores
        for key in ("large", "small"):
            name, color, ls = MEMBER[key]
            raw = c[key][inside]
            axl.plot(x, raw, color=color, linewidth=0.4, alpha=0.25, zorder=2)
            axl.plot(x, ema(raw, alpha), color=color, linestyle=ls, linewidth=1.0, zorder=3, label=f"{name} acc")
            axl.plot(x, ema(c["r_" + key[0]][inside], alpha), color=color, linestyle=":", linewidth=0.9, zorder=3, label=f"{name} proxy")
        for key, label, color in row["duos"]:
            raw = c[key][inside]
            axl.plot(x, raw, color=color, linewidth=0.4, alpha=0.25, zorder=2)
            axl.plot(x, ema(raw, alpha), color=color, linewidth=1.3, zorder=4, label=label)
        axl.set_ylabel("Accuracy / proxy score")
        axl.set_title(row["title"], loc="left", fontsize=7, color=INK, pad=11)
        # right: gate weight at every beta against the true accuracies
        for key in ("large", "small"):
            name, color, ls = MEMBER[key]
            axr.plot(x, ema(c[key][inside], alpha), color=color, linestyle=ls, linewidth=1.0, zorder=3, label=f"{name} acc")
        for beta, color in zip(betas, BETA_COLORS):
            own = abs(beta - row["beta"]) < 1e-9
            axr.plot(x, ema(gate_weight(c, beta)[inside], alpha), color=color, linewidth=1.7 if own else 0.9, zorder=5 if own else 4,
                     label=f"$w_L$, $\\beta$ = {beta:g}" + (" (run)" if own else ""))
        axr.axhline(0.5, color=INK2, linewidth=0.4, linestyle=":", zorder=1)
        axr.set_ylabel("Gate weight $w_L$ / accuracy")
    # legends: the left panels' entries (every row's duos included), then the gate weights, one below the other so they never meet
    left: dict[str, object] = {}
    for axl, _ in axes:
        for h, l in zip(*axl.get_legend_handles_labels()):
            left.setdefault(l, h)
    seen = set(left)
    right: dict[str, object] = {}
    for _, axr in axes:
        for h, l in zip(*axr.get_legend_handles_labels()):
            if l not in seen:
                right.setdefault(l, h)
    fig.legend(list(left.values()), list(left), loc="lower left", ncol=4, frameon=False, fontsize=5.8, bbox_to_anchor=(0.06, 0.058),
               handlelength=2.0, columnspacing=1.0)
    fig.legend(list(right.values()), list(right), loc="lower right", ncol=4, frameon=False, fontsize=5.8, bbox_to_anchor=(0.985, 0.004),
               handlelength=1.8, columnspacing=1.0)
    out = provenance.save_figure(fig, path, footer or "unlabeled figure: no run text given", dpi=dpi)
    plt.close(fig)
    return out


def collapse_round(curves: dict[str, np.ndarray], images: int, n_corruptions: int = 15, drop: float = 0.5) -> tuple[list[str], int] | None:
    """(the members, the 0-based round) of the first round in which a member's accuracy is below `drop` of its own round-0 accuracy:
    every member that is below it in that round; None if no member ever is. Rounds are read off the proxy batches' sample positions."""
    rl = n_corruptions * images
    rounds = np.floor(curves["x"] / rl).astype(int)
    first = {}
    for key in ("large", "small"):
        per_round = np.array([curves[key][rounds == r].mean() for r in range(rounds.max() + 1)])
        low = np.flatnonzero(per_round < drop * per_round[0])
        if len(low):
            first[key] = int(low[0])
    if not first:
        return None
    r = min(first.values())
    return [k for k, v in first.items() if v == r], r


def _who(members: list[str]) -> str:
    names = [MEMBER[k][0] for k in members]
    return " and ".join(names) + (" collapse" if len(names) > 1 else " collapses")


def default_rows(runs: dict[str, dict], images: int, b_t: int, beta: float, n_corruptions: int = 15) -> list[dict]:
    """The rows of the gate-reaction figure for the stored runs: round 1 of Vanilla DD; Vanilla DD around the first collapse of one of
    its members (5 rounds after it, 1 before); Joint DD from the start to one round after its first collapse. `runs`: {"indep": curves,
    "joint_dd": curves} (gate_curves); a row is left out when its run has no collapse."""
    rl = n_corruptions * images
    n_rounds = int(runs["indep"]["x"].max() // rl) + 1
    duos_v = [("ts", "Fixed TS duo", "#1baf7a"), ("gate", "Vanilla DD", "#2a78d6")]
    rows = [dict(title="(a) Vanilla DD, round 1", curves=runs["indep"], window=(0, rl), beta=beta, duos=duos_v)]
    hit = collapse_round(runs["indep"], images, n_corruptions)
    if hit:
        members, r = hit
        a, b = max(r - 1, 0), min(r + 5, n_rounds)
        rows.append(dict(title=f"(b) Vanilla DD, rounds {a + 1}-{b}: the {_who(members)}", curves=runs["indep"],
                         window=(a * rl, b * rl), beta=beta, duos=duos_v))
    if "joint_dd" in runs:
        hit = collapse_round(runs["joint_dd"], images, n_corruptions)
        b = min(hit[1] + 2, n_rounds) if hit else n_rounds
        what = f": the {_who(hit[0])} in round {hit[1] + 1}" if hit else ""
        rows.append(dict(title=f"({'c' if len(rows) > 1 else 'b'}) Joint DD, rounds 1-{b}{what}", curves=runs["joint_dd"],
                         window=(0, b * rl), beta=beta, duos=[("ts", "Fixed TS (joint members)", "#1baf7a"), ("gate", "Joint DD", "#eb6834")]))
    return rows


# ------------------------------------------------------------------------------------------------------------ tables

def _bold(vals: Sequence[float], decimals: int = 1) -> list[str]:
    best = max(round(v, decimals) for v in vals)
    return [f"\\textbf{{{v:.{decimals}f}}}" if round(v, decimals) == best else f"{v:.{decimals}f}" for v in vals]


def _setting(stream: Stream, ctx, duo_ctx, tta: str, temps, proxy: str, beta: float, b_t: int) -> str:
    order = stream.meta.get("order")
    order_text = "CoTTA's standard order" if order is None else f"CoTTA order {order}"
    seeds = f" Mean over {stream.seeds} seeds." if stream.seeds > 1 else ""
    return (f"{AM._models_text(ctx, tta, ctx.g.severity)} The stream visits the 15 corruptions in "
            f"{order_text} and repeats them for {stream.rounds} rounds, "
            f"{stream.images} images per corruption and round (every round draws its own images), with no reset: the models "
            f"adapt continually from the source weights. The duos combine the models as $(z_L / T_L + z_S / T_S) / 2$ with "
            f"$T_L = {temps.T_l:.2f}$, $T_S = {temps.T_s:.2f}$, {temps.origin}. \\emph{{Frozen duo}}: no adaptation. "
            f"\\emph{{Fixed TS duo}}: the models adapt on their own entropy. {GD._gate_text(proxy, beta, b_t)} "
            f"{AM._frozen_sentence(ctx)} {AM._hyperparameters(ctx, duo_ctx)} The own-entropy rows (Fixed TS duo, Vanilla DD) use "
            f"the first set of learning rates, Joint DD the second.{seeds}")


def _env(body: Sequence[str], col_spec: str, caption: str, label: str, source: str, models: str = "") -> str:
    lines = [f"% Generated by {source} on {timestamp()}{f' for {models}' if models else ''}. Requires booktabs, graphicx.", "\\begin{table*}[t]", "    \\centering",
             "    \\small", "    \\setlength{\\tabcolsep}{3.2pt}", f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
             f"    \\begin{{tabular}}{{{col_spec}}}", "        \\toprule", *("        " + b for b in body), "        \\bottomrule",
             "    \\end{tabular}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


def render_main_table(stream: Stream, ctx, duo_ctx, tta: str, temps, *, proxy: str, beta: float, b_t: int, label: str, source: str) -> str:
    """CoTTA's Table 4 layout: the corruptions in stream order (accuracy averaged over the rounds and seeds), then the mean."""
    labels = [l[2] for l in LINES]
    n = len(stream.corruptions)
    acc = {l: stream.per_visit(l).mean((0, 1)) for l in labels}                      # (corruptions,)
    cols = {l: list(acc[l]) + [float(acc[l].mean())] for l in labels}
    best = [_bold([cols[l][j] for l in labels]) for j in range(n + 1)]
    names = " & ".join(f"\\rotatebox{{90}}{{{SHORT[c]}}}" for c in stream.corruptions)
    body = [f"Time & \\multicolumn{{{n}}}{{l}}{{$t$\\ \\rightarrowfill}} & \\\\", "\\midrule",
            f"Method & {names} & Mean \\\\", "\\midrule"]
    body += [f"{latex_escape(l)} & " + " & ".join(best[j][i] for j in range(n + 1)) + " \\\\" for i, l in enumerate(labels)]
    caption = (f"Top-1 accuracy (\\%) in the CoTTA setting on ImageNet-C, severity 5, per corruption (mean over the "
               f"{stream.rounds} rounds). " + _setting(stream, ctx, duo_ctx, tta, temps, proxy, beta, b_t))
    return _env(body, "l|" + "c" * n + "|c", caption, label, source, provenance.run_label(ctx.duo, tta, ctx.bn))


def render_round_table(stream: Stream, ctx, duo_ctx, tta: str, temps, *, proxy: str, beta: float, b_t: int, label: str, source: str) -> str:
    """The mean accuracy over the 15 corruptions in each round (and over all rounds): does a method decay when the shifts recur?"""
    labels = [l[2] for l in LINES]
    R = stream.rounds
    per_round = {l: stream.per_visit(l).mean((0, 2)) for l in labels}               # (rounds,)
    cols = {l: list(per_round[l]) + [float(per_round[l].mean())] for l in labels}
    best = [_bold([cols[l][j] for l in labels]) for j in range(R + 1)]
    body = ["Method & " + " & ".join(f"{r + 1}" for r in range(R)) + " & Mean \\\\", "\\midrule"]
    body += [f"{latex_escape(l)} & " + " & ".join(best[j][i] for j in range(R + 1)) + " \\\\" for i, l in enumerate(labels)]
    caption = (f"Mean top-1 accuracy (\\%) over the 15 corruptions in each round of the CoTTA setting. "
               + _setting(stream, ctx, duo_ctx, tta, temps, proxy, beta, b_t))
    return _env(body, "l|" + "c" * R + "|c", caption, label, source, provenance.run_label(ctx.duo, tta, ctx.bn))


def summary_text(stream: Stream) -> str:
    labels = [l[2] for l in LINES]
    rows = []
    for l in labels:
        pr = stream.per_visit(l).mean((0, 2))
        rows.append(f"{l:14s} " + " ".join(f"{v:5.1f}" for v in pr) + f" | mean {pr.mean():5.2f}")
    return "accuracy (%) per round (round 1 .. R) and mean\n" + "\n".join(rows)


def accuracy_rows(stream: Stream) -> list[dict]:
    """One row per (method, seed, round, corruption): the accuracy (%) of that visit."""
    rows = []
    for l in [l[2] for l in LINES]:
        pv = stream.per_visit(l)
        for s in range(stream.seeds):
            for r in range(stream.rounds):
                for j, c in enumerate(stream.corruptions):
                    rows.append({"method": l, "seed": s, "round": r + 1, "position": j + 1, "corruption": c, "accuracy": float(pv[s, r, j])})
    return rows


def report(args, ctx, duo_ctx, temps, proxy: str, seeds: Sequence[int], path_of, out: Path) -> int:
    """The report step of scripts/03c_recurring_duos.py."""
    import pandas as pd

    from src.utils.tables import write_table

    stream = load_stream(path_of, seeds)
    tta = "tent"
    tag = RC.stream_tag(args.duo, tta, args.bn, args.rounds, args.images, seeds[0] if len(seeds) == 1 else "s".join(map(str, seeds)),
                        args.order) + f"__{proxy}_bt{args.b_t}_beta{args.beta:g}"
    print(summary_text(stream))
    pd.DataFrame(accuracy_rows(stream)).to_csv(out / f"{tag}__accuracy.csv", index=False)
    note = provenance.footer(ctx.duo, tta, args.bn, f"CoTTA {'standard order' if args.order is None else f'order {args.order}'}, {args.rounds} rounds x "
                             f"{args.images} images, gate {proxy} b_t={args.b_t} beta={args.beta:g}, T_L = {temps.T_l}, T_S = {temps.T_s}")
    figs = plot_stream(stream, out / f"{tag}__accuracy.png", footer=note)
    dest = out if (args.num_samples or args.out) else ctx.g.path("paper")       # an explicit --out keeps the tables beside the data
    kw = dict(proxy=proxy, beta=args.beta, b_t=args.b_t, source="scripts/03c_recurring_duos.py")
    write_table(dest / f"tab_recurring__{tag}.tex", render_main_table(stream, ctx, duo_ctx, tta, temps, label=f"tab:recurring:{tag}", **kw))
    write_table(dest / f"tab_recurring_rounds__{tag}.tex", render_round_table(stream, ctx, duo_ctx, tta, temps,
                                                                              label=f"tab:recurring_rounds:{tag}", **kw))
    for f in figs:
        print(f"wrote {f}")
    try:
        curves = {}
        for run in ("indep", "joint_dd"):
            curves[run] = gate_curves(RC.load_run(path_of(run, seeds[0])), load_gate_records(path_of(run, seeds[0])), args.b_t)
    except FileNotFoundError as e:
        print(f"no gate figure: {e}")
    else:
        rows = default_rows(curves, stream.images, args.b_t, args.beta)
        for f in plot_gate_reaction(rows, out / f"{tag}__gate_reaction.png", images=stream.images, footer=note):
            print(f"wrote {f}")
    return 0


# ------------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import shutil
    import subprocess
    import tempfile

    from src.experiments._testing import make_context
    from src.experiments.protocols import Segment
    from src.utils.config import load_global

    g = load_global()
    corr = g.test_corruptions
    R, IMAGES, S = 3, 100, 2
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = [RC.Part(Segment(f"{c}/s5", None, i == 0), r, 1000 * r) for r in range(R) for i, c in enumerate(corr)]
        N = len(parts) * IMAGES
        # synthetic runs: the frozen duo is flat, the others have a level that depends on the corruption; Joint DD decays by round
        level = {"frozen": 0.45, "indep": 0.55, "joint_dd": 0.85}
        drift = np.repeat(np.linspace(0, -0.15, R), len(corr) * IMAGES)[None, :]

        def fake(run, series, seed):
            p = level[run] + (0.04 if series == "gate" else 0) + (drift[0] if run == "joint_dd" else 0)
            p = np.clip(p + 0.1 * np.sin(np.arange(N) / 300.0), 0.02, 0.98)
            return (np.random.default_rng(seed * 7 + sum(map(ord, run + series))).random(N) < p).astype(np.uint8)

        for run in RC.RUNS:
            for seed in range(S):
                res = {"correct": {s: fake(run, s, seed) for s in RC.SERIES if s != "gate" or RC.RUNS[run].gated},
                       "part_n": [IMAGES] * len(parts), "gate_records": []}
                RC.save_run(tmp / f"t__{run}__seed{seed}.npz", res, parts, {"run": run, "order": None, "seed": seed})
        path_of = lambda run, seed: tmp / f"t__{run}__seed{seed}.npz"                       # noqa: E731
        st = load_stream(path_of, list(range(S)))
        assert st.seeds == S and st.rounds == R and st.images == IMAGES and st.corruptions == corr and st.n_visits == R * 15
        assert st.per_visit("Joint DD").shape == (S, R, 15) and st.blocks("Joint DD").shape == (S, N // (IMAGES // 10))
        pr = st.per_visit("Joint DD").mean((0, 2))
        assert pr[0] > pr[1] > pr[2], "the synthetic Joint DD decays from round to round and the per-round means see it"
        assert abs(st.per_visit("Frozen duo").mean() - 100 * np.mean([fake("frozen", "ts", s).mean() for s in range(S)])) < 1e-6

        # runs that saw different visits cannot be put on one plot
        bad = RC.load_run(path_of("indep", 0))
        RC.save_run(path_of("indep", 1), {"correct": {k: bad[k] for k in ("ts", "gate", "large", "small")},
                                          "part_n": [IMAGES + 1] * len(parts), "gate_records": []}, parts, {"run": "indep"})
        try:
            load_stream(path_of, [0, 1])
            raise AssertionError("different visits must be refused")
        except ValueError as e:
            assert "share one stream" in str(e)
        RC.save_run(path_of("indep", 1), {"correct": {k: fake("indep", k, 1) for k in ("ts", "gate", "large", "small")},
                                          "part_n": [IMAGES] * len(parts), "gate_records": []}, parts, {"run": "indep"})

        # labels never overlap
        sp = _spread({"a": 50.0, "b": 50.5, "c": 51.0, "d": 70.0}, 3.0)
        assert all(sp[k2] - sp[k1] >= 3.0 - 1e-9 for k1, k2 in (("a", "b"), ("b", "c"))) and sp["d"] == 70.0

        # tables and figure
        ctx = make_context(tmp)
        duo_ctx = AM.with_learning_rates(ctx, 5e-4, 1e-3)
        temps = AM.Temps(tmp, 1.4, 0.9, "fitted on the four held-out corruptions pooled")
        kw = dict(proxy="nnn", beta=0.5, b_t=256, source="test")
        t1 = render_main_table(st, ctx, duo_ctx, "tent", temps, label="tab:x", **kw)
        t2 = render_round_table(st, ctx, duo_ctx, "tent", temps, label="tab:y", **kw)
        rows = [l.strip() for l in t1.splitlines() if l.strip().endswith("\\\\") and "&" in l]
        assert len(rows) == 2 + len(LINES) and rows[1].startswith("Method & \\rotatebox{90}{Gaussian}") and rows[1].endswith("Mean \\\\")
        assert all(r.count("&") == 16 for r in rows[2:]) and "\\rotatebox{90}{elastic\\_trans}" in rows[1]
        assert rows[0].startswith("Time & \\multicolumn{15}{l}{$t$\\ \\rightarrowfill}")
        best_col = [r.split("&")[1].strip() for r in rows[2:]]
        assert sum("\\textbf" in c for c in best_col) == 1 and "\\textbf" in best_col[3], "the best of the first column is bold (Joint DD)"
        rows2 = [l.strip() for l in t2.splitlines() if l.strip().endswith("\\\\") and "&" in l]
        assert rows2[0] == "Method & 1 & 2 & 3 & Mean \\\\" and all(r.count("&") == R + 1 for r in rows2[1:])
        for t in (t1, t2):
            cap = t.split("\\caption{")[1].split("\n")[0]
            assert cap.count("$") % 2 == 0 and "CoTTA's standard order" in cap and f"repeats them for {R} rounds" in cap, cap[:200]
            assert "Mean over 2 seeds" in cap and "Models that adapt on their own entropy:" in cap and "proxy batches of 256 samples" in cap
        figs = plot_stream(st, tmp / "fig.png", footer="ViT-B/16 + ResNet-50 (V1) [tiny], Tent | generated 2026-01-01 00:00:00 +0000")
        assert [f.suffix for f in figs] == [".png", ".pdf"] and all(f.stat().st_size > 5000 for f in figs)
        import PIL.Image
        assert "ViT-B/16 + ResNet-50 (V1) [tiny]" in PIL.Image.open(figs[0]).text["Description"], "the footer is stored in the figure"
        one = Stream({k: v[:1] for k, v in st.acc.items()}, st.part_n, st.part_label, st.part_round, st.meta)
        plot_stream(one, tmp / "fig1.png")                                                   # one seed: no band
        st1 = Stream({k: v[:, : 15 * IMAGES] for k, v in st.acc.items()}, st.part_n[:15], st.part_label[:15], st.part_round[:15], st.meta)
        plot_stream(st1, tmp / "fig_r1.png")                                                  # one round: a single panel
        assert len(accuracy_rows(st)) == len(LINES) * S * R * 15 and "Joint DD" in summary_text(st)

        # gate reaction: per-proxy-batch accuracies from the stored correctness, the gate weight at any beta from the stored logits
        B_T = 64
        nb = -(-N // B_T)
        xl, xs = rng.normal(0.5, 1.0, nb), rng.normal(0.0, 1.0, nb)
        recs = [{"r_l": 0.6, "r_s": 0.4, "x_l": float(a), "x_s": float(b), "w_l": float(1 / (1 + np.exp(-0.5 * (a - b))))} for a, b in zip(xl, xs)]
        run0 = {k: st.acc[l][0] for k, l in (("large", "Frozen duo"), ("small", "Fixed TS duo"), ("ts", "Fixed TS duo"), ("gate", "Vanilla DD"))}
        cv = gate_curves(run0, recs, B_T)
        assert len(cv["x"]) == nb and np.allclose(gate_weight(cv, 0.5), cv["w_l"]) and np.all((0 <= cv["large"]) & (cv["large"] <= 1))
        assert abs(np.average(cv["gate"], weights=np.diff(np.minimum(np.arange(nb + 1) * B_T, N))) - run0["gate"].mean()) < 1e-12
        assert np.all(np.abs(gate_weight(cv, 0.01) - 0.5) < 0.05) and np.all(gate_weight(cv, 0) == 0.5), "beta -> 0 is the equal weighting"
        try:
            gate_curves(run0, recs[:-1], B_T)
            raise AssertionError("records that do not cover the stream must be refused")
        except ValueError:
            pass
        assert np.allclose(ema(np.array([1.0, 0.0, 0.0]), 0.5), [1.0, 0.5, 0.25])
        run1 = {k: np.where(np.arange(N) >= 15 * IMAGES, 0, v).astype(np.uint8) for k, v in run0.items()}   # collapses when round 2 starts
        cv1 = gate_curves(run1, recs, B_T)
        hit = collapse_round(cv1, IMAGES)
        assert hit is not None and hit[1] == 1 and hit[0] == ["large", "small"], hit
        assert _who(hit[0]) == "ViT-B/16 and ResNet-50 collapse" and _who(["small"]) == "ResNet-50 collapses"
        assert collapse_round(cv, IMAGES) is None, "a run whose members do not drop has no collapse round"
        rows_g = default_rows({"indep": cv1, "joint_dd": cv1}, IMAGES, B_T, 0.5)
        assert [r["title"][:3] for r in rows_g] == ["(a)", "(b)", "(c)"] and "ViT-B/16 and ResNet-50 collapse" in rows_g[1]["title"]
        figs_g = plot_gate_reaction(rows_g, tmp / "gate.png", images=IMAGES)
        assert all(f.stat().st_size > 5000 for f in figs_g)
        if shutil.which("pdflatex"):
            doc = ("\\documentclass{article}\\usepackage{booktabs,graphicx,amsmath}\\usepackage[landscape,margin=1cm]{geometry}"
                   "\\begin{document}\n" + t1 + t2 + "\\end{document}\n")
            (tmp / "t.tex").write_text(doc)
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"], cwd=tmp, capture_output=True,
                               text=True, timeout=120)
            assert r.returncode == 0, r.stdout[-1500:]
            print("recurring_report: generated LaTeX compiles with pdflatex")
    print("recurring_report self-test passed")
