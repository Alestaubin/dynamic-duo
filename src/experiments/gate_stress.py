"""
gate_stress.py
==============
The gate on a member that collapses (scripts/04c_gate_stress.py): the independent duo (Vanilla DD: both members adapt on their own entropy,
the gate outside the loop) on the test continual stream, one member at a CONSERVATIVE rate and the other at a STRESS rate, both chosen on
held-out shifts by scripts/04b_tune_continual_lr.py. Two symmetric experiments:

  A   ViT-B/16 conservative, ResNet-50 stress     the small member collapses; the gate should move to the large one
  B   ViT-B/16 stress, ResNet-50 conservative     the large member collapses; the gate should move to the small one

The run stops once the stressed member has collapsed completely (continual_tuning.DeadStop) and `after` more images have passed. The FROZEN
duo is then run over exactly the same images (`max_images`), so every line of the figure covers the same stream. Stored with continual.save_run
(per-sample correctness of ts / gate / large / small, and the gate's proxy batches).

Numbers (`analyze`), for the member that collapses first (the focus):
  onset       first window of 2,560 images from which the member stays below half of the FROZEN member's accuracy on the same images for 5
              windows (the half of continual_report.collapse_visit, at the resolution of a window instead of a visit)
  dead at     the start of its complete collapse (continual_tuning.dead_onset)
  gate delay  images from the onset until the gate's weight on the focus has covered half the way from its mean before the onset to its mean
              at the end of the run (5-refresh trailing mean, held for 3 refreshes); negative: the gate moved first
  regions     accuracy of every series before the onset, from the onset to the complete collapse, and after it
  recovery    after the collapse, (gate - Fixed TS duo) / (surviving member - Fixed TS duo): 1 when the gate is as good as the survivor
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.experiments import continual as RC
from src.experiments import continual_tuning as CT
from src.experiments.grids import sgd_optim
from src.experiments.joint_dd import PureProxyGate
from src.experiments.logit_runs import with_optim
from src.experiments.runner import Context
from src.utils import provenance

EXPERIMENTS = {"A": ("conservative", "stress"), "B": ("stress", "conservative")}        # (role of the large member, of the small one)
WINDOW, DROP, SUSTAIN = 2560, 0.5, 5          # 40 batches of 64; below half of the frozen member for 5 windows
SIDES = ("large", "small")


# ------------------------------------------------------------------------------------------------------------------ the run

def run_experiment(ctx: Context, *, lrs: tuple[float, float], watch: Sequence[str], after: int, seed: int, order: int | None, images: int,
                   proxy: str, beta: float, b_t: int, T_l: float, T_s: float, momentum: float = 0.9, tta: str = "tent",
                   stream: str = "test", rounds: int = 4) -> tuple[dict, dict, list[RC.Part], dict]:
    """The adaptive run at (lr_large, lr_small), stopped `after` images after any member of `watch` is dead (never, if `watch` is empty),
    then the frozen run over the same images. `stream`: "test" (the 15 corruptions once, `order`) or "heldout" (the tuning stream of
    continual_tuning.tuning_parts: the 4 held-out shifts x `rounds`, the stream the rates were chosen on). Returns (adaptive result, frozen
    result, the visits made, meta); results as `run_until`."""
    rctx = with_optim(ctx, large=sgd_optim(lrs[0], momentum), small=sgd_optim(lrs[1], momentum))
    gate = PureProxyGate(proxy, beta=beta, b_t=b_t, T_l=T_l, T_s=T_s, num_classes=1000)
    runner, _ = RC.make_runner(rctx, RC.RUNS["indep"], gate, tta)
    if stream == "test":
        parts = RC.stream_parts(rctx, rounds=1, seed=seed, order=order, stream="test")
    elif stream == "heldout":
        parts = CT.tuning_parts(rctx, rounds=rounds, seed=seed)
    else:
        raise ValueError(f"stream must be 'test' or 'heldout', got {stream!r}")
    common = dict(images=images, batch_size=rctx.batch_size, workers=rctx.g.workers, T_l=T_l, T_s=T_s)
    res = CT.run_until(runner, parts, gate=gate, stop=CT.DeadStop(tuple(watch), "any", after) if watch else None, **common)
    print(f"adaptive run: {res['seen']} images, {'ran to the end' if res['stopped'] is None else res['stopped']}; frozen run over the same images")
    frozen_runner, _ = RC.make_runner(ctx, RC.RUNS["frozen"], None, tta)
    frozen = CT.run_until(frozen_runner, parts, max_images=res["seen"], **common)
    if frozen["part_n"] != res["part_n"]:
        raise RuntimeError(f"the frozen run saw other visits than the adaptive one: {frozen['part_n']} against {res['part_n']}")
    meta = {"duo": ctx.duo["NAME"], "tta": tta, "bn": ctx.bn, "seed": seed, "stream": stream, "rounds": rounds if stream == "heldout" else 1, "order": order, "images": images, "lrs": list(lrs),
            "momentum": momentum, "proxy": proxy, "beta": beta, "b_t": b_t, "T_l": T_l, "T_s": T_s, "batch_size": rctx.batch_size,
            "watch": list(watch), "after": after, "stopped": res["stopped"], "dead_at": res["dead_at"], "seen": res["seen"],
            "optim": {s.lower(): rctx.duo[s]["OPTIM"] for s in ("LARGE", "SMALL")},
            "models": {s.lower(): rctx.duo[s]["NAME"] for s in ("LARGE", "SMALL")}}
    return res, frozen, parts[:len(res["part_n"])], meta


# ------------------------------------------------------------------------------------------------------------------ the numbers

def block_means(correct: np.ndarray, size: int = WINDOW) -> np.ndarray:
    """Accuracy (%) per block of `size` samples (a trailing partial block is dropped)."""
    n = len(correct) // size * size
    return correct[:n].reshape(-1, size).mean(1) * 100


def collapse_onset(member: np.ndarray, frozen: np.ndarray, window: int = WINDOW, drop: float = DROP, sustain: int = SUSTAIN) -> int | None:
    """Where the member first stays below `drop` of the frozen member's accuracy on the same images for `sustain` windows; None if never."""
    n = min(len(member), len(frozen)) // window
    if n < sustain:
        return None
    a = member[:n * window].reshape(n, window).mean(1)
    f = frozen[:n * window].reshape(n, window).mean(1)
    held = np.convolve((a < drop * f).astype(int), np.ones(sustain, dtype=int), mode="valid") == sustain
    return int(np.flatnonzero(held)[0]) * window if held.any() else None


def gate_trace(records: Sequence[dict], b_t: int) -> dict[str, np.ndarray]:
    """The gate's proxy batches: `end` the sample count at which batch k was complete (its weight is used from there on), and the weight
    w_L and the proxy logits of both members."""
    return {"end": (np.arange(len(records)) + 1) * b_t, **{k: np.array([r[k] for r in records]) for k in ("w_l", "x_l", "x_s")}}


def _trailing_mean(y: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(y)])
    i = np.arange(1, len(y) + 1)
    return (c[i] - c[np.maximum(i - k, 0)]) / (i - np.maximum(i - k, 0))


def reaction(trace: dict[str, np.ndarray], focus: str, onset: int, *, last: int = 40, smooth: int = 5, hold: int = 3, lead: int = 8 * WINDOW) -> dict:
    """The gate's weight on the focus member: its mean before the onset (`last` refreshes), its mean at the end (`last` refreshes), and the
    delay after which the smoothed weight has covered half the way (searched from `lead` images before the onset); None if the weight
    moves by less than 0.1."""
    w = trace["w_l"] if focus == "large" else 1.0 - trace["w_l"]
    end = trace["end"]
    before = np.flatnonzero(end <= onset)
    pre = float(w[before[-last:]].mean()) if len(before) else float(w[:smooth].mean())
    post = float(w[-last:].mean())
    out = {"w_focus": w, "w_before": pre, "w_after": post, "half_at": None, "delay": None}
    if abs(post - pre) < 0.1:
        return out
    frac = (_trailing_mean(w, smooth) - pre) / (post - pre)
    reached = np.convolve((frac >= 0.5).astype(int), np.ones(hold, dtype=int), mode="valid") == hold
    reached &= end[:len(reached)] > onset - lead
    if reached.any():
        k = int(np.flatnonzero(reached)[0])
        out["half_at"], out["delay"] = int(end[k]), int(end[k] - onset)
    return out


def analyze(run: dict, frozen: dict, records: Sequence[dict], b_t: int) -> dict:
    """The numbers of the module docstring for one stored experiment (`run`, `frozen`: RC.load_run results of the same visits)."""
    n = len(run["large"])
    onsets = {s: collapse_onset(run[s], frozen[s]) for s in SIDES}
    dead = {s: CT.dead_onset(run[s]) for s in SIDES}
    hit = [s for s in SIDES if onsets[s] is not None]
    focus = min(hit, key=onsets.get) if hit else None
    out = {"images": n, "onset": onsets, "dead_at": dead, "focus": focus,
           "mean": {"Frozen duo": float(frozen["ts"].mean() * 100), "Fixed TS duo": float(run["ts"].mean() * 100),
                    "Vanilla DD": float(run["gate"].mean() * 100), "large": float(run["large"].mean() * 100),
                    "small": float(run["small"].mean() * 100)}}
    if focus is None:
        return out
    survivor = "small" if focus == "large" else "large"
    onset, d = onsets[focus], dead[focus]
    spans = {"before the onset": (0, onset), "onset to complete collapse": (onset, d if d is not None else n),
             **({"after complete collapse": (d, n)} if d is not None else {})}
    series = {"Frozen duo": frozen["ts"], "Fixed TS duo": run["ts"], "Vanilla DD": run["gate"], "large": run["large"], "small": run["small"]}
    out["regions"] = {name: {"images": b - a, **{k: float(v[a:b].mean() * 100) for k, v in series.items()}} for name, (a, b) in spans.items() if b > a}
    out["survivor"] = survivor
    gate = reaction(gate_trace(records, b_t), focus, onset)
    out["gate"] = {k: v for k, v in gate.items() if k != "w_focus"}
    after = out["regions"].get("after complete collapse")
    if after and after[survivor] - after["Fixed TS duo"] > 1.0:
        out["recovery"] = (after["Vanilla DD"] - after["Fixed TS duo"]) / (after[survivor] - after["Fixed TS duo"])
    return out


def describe(an: dict, names: dict[str, str]) -> list[str]:
    """The lines a report prints for one analysis."""
    if an["focus"] is None:
        return [f"no member collapsed in {an['images']} images; means " + ", ".join(f"{k} {v:.1f}" for k, v in an["mean"].items())]
    f, s = an["focus"], an["survivor"]
    dead = an["dead_at"][f]
    lines = [f"{names[f]} collapses: onset at {an['onset'][f]} images" + (f", complete from {dead}" if dead is not None else ", not yet complete")
             + f"; run of {an['images']} images"]
    g = an["gate"]
    lines.append(f"gate weight on {names[f]}: {g['w_before']:.2f} before the onset, {g['w_after']:.2f} at the end; "
                 + (f"half-way {g['delay']:+d} images after the onset" if g["delay"] is not None else "it barely moves"))
    cols = ("Frozen duo", "Fixed TS duo", "Vanilla DD", "large", "small")
    lines.append(f"{'accuracy (%) by region':<30}{'images':>8}" + "".join(f"{(names[c] if c in names else c):>14}" for c in cols))
    for name, r in an["regions"].items():
        lines.append(f"{name:<30}{r['images']:>8}" + "".join(f"{r[c]:>14.1f}" for c in cols))
    if "recovery" in an:
        lines.append(f"recovery after the collapse: the gate closes {an['recovery'] * 100:.0f}% of the way from the Fixed TS duo to {names[s]}")
    return lines


# ------------------------------------------------------------------------------------------------------------------ the figure

def plot_stress(items: Sequence[dict], path: Path, *, footer: str = "", width: float = 7.2, dpi: int = 300) -> list[Path]:
    """One column per item: accuracy of the frozen duo, the Fixed TS duo, the adaptive duo (Vanilla DD) and both members; the gate's weight on
    the large member; the proxy scores it reads. An item: {"title", "run", "frozen", "records", "b_t", "analysis", "names", "lrs"}. Written as
    `path` (.png) and .pdf. Direct labels with one column, a legend always."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from src.experiments.continual_report import GRID, INK, INK2, LINES, MEMBER, PLAIN, _spread

    plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    style = {l[2]: (l[3], l[4]) for l in LINES}
    BLUE = style["Vanilla DD"][0]
    cols = len(items)
    fig = plt.figure(figsize=(width, 5.6 if cols == 1 else 5.9), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(3, cols, height_ratios=[3.0, 1.25, 1.05], hspace=0.14, wspace=0.13, left=0.08 if cols == 1 else 0.075,
                          right=0.835 if cols == 1 else 0.975, top=0.835, bottom=0.105)
    first_axes = None
    for c, item in enumerate(items):
        run, frozen, an, names = item["run"], item["frozen"], item["analysis"], item["names"]
        n = an["images"]
        if n < 2 * WINDOW:
            raise ValueError(f"{item['title']}: {n} images are fewer than two windows of {WINDOW}: nothing to plot")
        edges = np.r_[0, np.cumsum(run["part_n"])]
        ax1 = fig.add_subplot(gs[0, c], sharey=None if first_axes is None else first_axes[0])
        ax2 = fig.add_subplot(gs[1, c], sharex=ax1, sharey=None if first_axes is None else first_axes[1])
        ax3 = fig.add_subplot(gs[2, c], sharex=ax1, sharey=None if first_axes is None else first_axes[2])
        first_axes = first_axes or (ax1, ax2, ax3)
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor("white")
            ax.set_xlim(0, n)
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for e in edges[1:-1]:
                ax.axvline(e, color=INK2, linewidth=0.4, linestyle="--", alpha=0.55, zorder=1)
        for ax in (ax1, ax2):
            ax.tick_params(labelbottom=False)
        ax3.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
        for lo, hi, label in zip(edges[:-1], edges[1:], run["part_label"]):
            ax1.text((lo + hi) / 2, 1.015, PLAIN[label.split("/")[0]], transform=ax1.get_xaxis_transform(), rotation=90, ha="center", va="bottom",
                     fontsize=5.4, color=INK2)

        # accuracy
        x = (np.arange(n // WINDOW) + 0.5) * WINDOW
        lines = [("Frozen duo", block_means(frozen["ts"]), *style["Frozen duo"], 0.9), ("Fixed TS duo", block_means(run["ts"]), *style["Fixed TS duo"], 0.9),
                 ("Vanilla DD", block_means(run["gate"]), *style["Vanilla DD"], 1.5)]
        lines += [(f"{names[s]}, adapting", block_means(run[s]), MEMBER[s][1], MEMBER[s][2], 1.0) for s in SIDES]
        ends = {}
        for label, y, color, ls, lw in lines:
            ax1.plot(x[:len(y)], y, color=color, linestyle=ls, linewidth=lw, zorder=3 if label != "Vanilla DD" else 4, label=label, solid_capstyle="round")
            ends[label] = float(y[-8:].mean())
        top = max(float(l[1].max()) for l in lines)
        ax1.set_ylim(-1, np.ceil(top / 10) * 10 + 2)
        onset = an["onset"].get(an["focus"]) if an["focus"] else None
        if onset is not None:
            ax1.axvline(onset, color=INK, linewidth=0.9, zorder=2)
            ax1.text(onset, 0.97, f" {names[an['focus']]} collapses", transform=ax1.get_xaxis_transform(), fontsize=6, color=INK, ha="left", va="top")
        if cols == 1:
            for label, yv in _spread(ends, (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.07).items():
                color = next(l[2] for l in lines if l[0] == label)
                ax1.annotate(label, (n, ends[label]), xytext=(n * 1.012, yv), textcoords="data", fontsize=6.2, color=INK, va="center", annotation_clip=False,
                             arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6, shrinkA=0, shrinkB=1))
        ax1.set_title(item["title"], fontsize=6.6, color=INK, loc="left", pad=31)
        if c == 0:
            ax1.set_ylabel("Accuracy (%)")

        # the gate's weight on the large member
        tr = gate_trace(item["records"], item["b_t"])
        ax2.axhline(0.5, color=INK2, linewidth=0.5, linestyle=":", zorder=1)
        ax2.plot(tr["end"], tr["w_l"], color=BLUE, linewidth=0.45, alpha=0.35, zorder=2)
        ax2.plot(tr["end"], _trailing_mean(tr["w_l"], 5), color=BLUE, linewidth=1.3, zorder=3)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_yticks([0, 0.5, 1])
        g = an.get("gate") or {}
        if g.get("half_at") is not None:
            w_half = float(_trailing_mean(tr["w_l"], 5)[np.searchsorted(tr["end"], g["half_at"])])
            ax2.plot([g["half_at"]], [w_half], marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor=BLUE, markeredgewidth=1.0, zorder=5)
            rising = an["focus"] == "small"                      # the plotted weight is the large member's: it rises when the small one collapses
            ax2.annotate(f"half-way: {g['delay'] / 1000:+.1f}k images", (g["half_at"], w_half), xytext=(7, -13 if rising else 6),
                         textcoords="offset points", fontsize=5.8, color=INK, ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=0.7))
        if onset is not None:
            ax2.axvline(onset, color=INK, linewidth=0.9, zorder=2)
        if c == 0:
            ax2.set_ylabel(f"Gate weight\non {names['large']}")

        # the proxy scores the gate reads
        for s, key in (("large", "x_l"), ("small", "x_s")):
            ax3.plot(tr["end"], _trailing_mean(tr[key], 5), color=MEMBER[s][1], linestyle=MEMBER[s][2], linewidth=1.0, label=names[s])
        if onset is not None:
            ax3.axvline(onset, color=INK, linewidth=0.9, zorder=2)
        ax3.set_xlabel("Observed samples", labelpad=1)
        if c == 0:
            ax3.set_ylabel("Proxy score\n(logit)")
        else:
            for ax in (ax1, ax2, ax3):
                ax.tick_params(labelleft=False)

    h, l_ = first_axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=5, frameon=False, fontsize=6.3, bbox_to_anchor=(0.46 if cols == 1 else 0.53, 1.0),
               handlelength=2.2, columnspacing=1.2)
    out = provenance.save_figure(fig, path, footer or "unlabeled figure: no run text given", dpi=dpi)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import logging
    import tempfile

    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.experiments.protocols import Segment
    from src.utils.config import load_global

    # ---- the numbers on a synthetic run: the large member collapses at 40k of 120k images, the gate follows 3k later
    rng = np.random.default_rng(1)
    n, b_t, onset_true = 120_000, 256, 40_000
    p_frozen = 0.40
    large = (rng.random(n) < 0.55).astype(np.uint8)
    large[onset_true:] = (rng.random(n - onset_true) < 0.004).astype(np.uint8)
    small = (rng.random(n) < 0.45).astype(np.uint8)
    frozen = {s: (rng.random(n) < p_frozen).astype(np.uint8) for s in ("large", "small", "ts")}
    ts = np.where(np.arange(n) < onset_true, rng.random(n) < 0.6, rng.random(n) < 0.05).astype(np.uint8)
    gate = np.where(np.arange(n) < onset_true + 4000, rng.random(n) < 0.6, rng.random(n) < 0.44).astype(np.uint8)
    run = {"large": large, "small": small, "ts": ts, "gate": gate, "part_n": np.array([50_000, 50_000, 20_000]),
           "part_label": np.array(["gaussian_noise/s5", "shot_noise/s5", "impulse_noise/s5"])}
    k = np.arange(n // b_t)
    w_l = 0.55 - 0.40 / (1 + np.exp(-(k * b_t - (onset_true + 3000)) / 500)) + rng.normal(0, 0.02, len(k))
    records = [{"w_l": float(w), "x_l": float(4 * (w - 0.5)), "x_s": float(-4 * (w - 0.5))} for w in w_l]

    assert collapse_onset(small, frozen["small"]) is None and collapse_onset(frozen["large"], frozen["large"]) is None
    on = collapse_onset(large, frozen["large"])
    assert on is not None and onset_true <= on <= onset_true + 2 * WINDOW, on
    assert collapse_onset(large[:onset_true + 3 * WINDOW], frozen["large"]) is None, "fewer than 5 windows below: no onset"
    an = analyze(run, frozen, records, b_t)
    assert an["focus"] == "large" and an["survivor"] == "small" and an["onset"]["small"] is None and an["images"] == n
    assert an["dead_at"]["large"] is not None and an["dead_at"]["large"] >= an["onset"]["large"]
    g = an["gate"]
    assert abs(g["w_before"] - 0.55) < 0.05 and abs(g["w_after"] - 0.15) < 0.05, g
    assert g["delay"] is not None and 0 <= g["delay"] <= 6000, g
    reg = an["regions"]
    assert {"before the onset", "after complete collapse"} <= set(reg) <= {"before the onset", "onset to complete collapse", "after complete collapse"}
    assert abs(reg["before the onset"]["Vanilla DD"] - 60) < 2 and abs(reg["after complete collapse"]["small"] - 45) < 2
    assert abs(reg["after complete collapse"]["Vanilla DD"] - 44) < 2 and reg["after complete collapse"]["Fixed TS duo"] < 8
    assert 0.9 < an["recovery"] < 1.1, an["recovery"]
    assert sum(r["images"] for r in reg.values()) == n
    steady = analyze({**run, "large": small, "small": small}, frozen, records, b_t)
    assert steady["focus"] is None and "no member collapsed" in describe(steady, {"large": "ViT", "small": "ResNet"})[0]
    flat = reaction(gate_trace([{"w_l": 0.5, "x_l": 0.0, "x_s": 0.0}] * 200, b_t), "large", 10_000)
    assert flat["delay"] is None, "a gate that does not move has no half-way"
    mirror = reaction(gate_trace([{**r, "w_l": 1 - r["w_l"]} for r in records], b_t), "small", an["onset"]["large"])
    assert abs(mirror["w_before"] - g["w_before"]) < 1e-9 and mirror["delay"] == g["delay"], "the small member's weight is the mirror image"
    text = describe(an, {"large": "ViT", "small": "ResNet"})
    assert text[0].startswith("ViT collapses") and any("recovery" in l for l in text)
    assert np.allclose(_trailing_mean(np.array([1.0, 2, 3, 4]), 2), [1, 1.5, 2.5, 3.5])

    with tempfile.TemporaryDirectory() as tmp:
        item = {"title": "(A) test", "run": run, "frozen": frozen, "records": records, "b_t": b_t, "analysis": an,
                "names": {"large": "ViT-B/16", "small": "ResNet-50"}}
        for items in ([item], [item, {**item, "title": "(B) test"}]):
            paths = plot_stress(items, Path(tmp) / f"fig{len(items)}.png", footer="self-test")
            assert [p.suffix for p in paths] == [".png", ".pdf"] and all(p.stat().st_size > 5000 for p in paths)

        # ---- the live run on tiny models: the frozen run covers the adaptive run's images, and `watch` is honoured
        logging.disable(logging.INFO)
        g_ = load_global()
        ctx = make_context(Path(tmp), batch_size=32, base=tiny_models(0))
        segs = [Segment(f"{c}/s5", FakeStream(150, seed=i + 1), i == 0) for i, c in enumerate(g_.test_corruptions[:3])]
        orig, orig_tuning = RC.stream_parts, CT.tuning_parts
        RC.stream_parts = lambda ctx_, **kw: [RC.Part(s, 0, kw["seed"]) for s in segs]
        CT.tuning_parts = lambda ctx_, **kw: [RC.Part(s, r, kw["seed"] + 1000 * r) for r in range(kw["rounds"]) for s in segs[:2]]
        try:
            kw = dict(lrs=(1e-3, 1e-3), after=0, seed=2, order=None, images=100, proxy="nnn", beta=0.5, b_t=64, T_l=1.0, T_s=1.0)
            res, fr, parts, meta = run_experiment(ctx, watch=(), **kw)
            assert res["stopped"] is None and res["seen"] == 300 and sum(fr["part_n"]) == 300 and len(parts) == 3
            assert set(res["correct"]) == {"ts", "large", "small", "gate"} and set(fr["correct"]) == {"ts", "large", "small"}
            assert meta["lrs"] == [1e-3, 1e-3] and meta["seen"] == 300 and meta["stream"] == "test"
            res, fr, parts, meta = run_experiment(ctx, watch=(), stream="heldout", rounds=3, **kw)
            assert meta["stream"] == "heldout" and meta["rounds"] == 3 and res["seen"] == 600 and len(parts) == 6 and sum(fr["part_n"]) == 600
            try:
                run_experiment(ctx, watch=(), stream="other", **kw)
                raise AssertionError("an unknown stream must be refused")
            except ValueError:
                pass
        finally:
            RC.stream_parts, CT.tuning_parts = orig, orig_tuning
    print("gate_stress self-test passed")
