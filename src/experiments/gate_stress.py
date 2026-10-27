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

Numbers (`analyze`). A member's COLLAPSE EPISODES are the stretches where its accuracy stays below half of the FROZEN member's on the same
images for at least 5 windows of 2,560 images (the half of continual_report.collapse_visit, at the resolution of a window instead of a visit).
The ViT's collapse is an episode inside one shift that ends when the next shift starts; the ResNet's can be a decay over several shifts, so
one run can hold several episodes or one that lasts to its end. For each episode:
  weight      the gate's weight on the member (5-refresh trailing mean): before the episode, at its start, its minimum inside, at its end, and just
              after it (None if the run ends first); `delay`: images from the start until the weight has covered half the way from its start value
              to its minimum (None if it falls by less than 0.1)
  accuracy    of every series over the episode's images: the frozen duo, the Fixed TS duo, Vanilla DD (the gate), and both members
The complete collapse (continual_tuning.dead_onset) of each member over the whole run is reported too.
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
                   stream: str = "test", rounds: int = 4, rotation: int = 0, stop_mode: str = "any") -> tuple[dict, dict, list[RC.Part], dict]:
    """The adaptive run at (lr_large, lr_small), stopped `after` images after any member of `watch` is dead (never, if `watch` is empty),
    then the frozen run over the same images. `stream`: "test" (the 15 corruptions once, `order`) or "heldout" (the tuning stream of
    continual_tuning.tuning_parts: the 4 held-out shifts x `rounds`, the stream the rates were chosen on). Returns (adaptive result, frozen
    result, the visits made, meta); results as `run_until`. `rotation`: the held-out stream starts with that shift of the list. `stop_mode`: "any"
    stops after the first member of `watch` has died, "all" after every one of them has."""
    rctx = with_optim(ctx, large=sgd_optim(lrs[0], momentum), small=sgd_optim(lrs[1], momentum))
    gate = PureProxyGate(proxy, beta=beta, b_t=b_t, T_l=T_l, T_s=T_s, num_classes=1000)
    runner, _ = RC.make_runner(rctx, RC.RUNS["indep"], gate, tta)
    if stream == "test":
        parts = RC.stream_parts(rctx, rounds=1, seed=seed, order=order, stream="test")
    elif stream == "heldout":
        parts = CT.tuning_parts(rctx, rounds=rounds, seed=seed, rotation=rotation)
    else:
        raise ValueError(f"stream must be 'test' or 'heldout', got {stream!r}")
    common = dict(images=images, batch_size=rctx.batch_size, workers=rctx.g.workers, T_l=T_l, T_s=T_s)
    res = CT.run_until(runner, parts, gate=gate, stop=CT.DeadStop(tuple(watch), stop_mode, after) if watch else None, **common)
    print(f"adaptive run: {res['seen']} images, {'ran to the end' if res['stopped'] is None else res['stopped']}; frozen run over the same images")
    frozen_runner, _ = RC.make_runner(ctx, RC.RUNS["frozen"], None, tta)
    frozen = CT.run_until(frozen_runner, parts, max_images=res["seen"], **common)
    if frozen["part_n"] != res["part_n"]:
        raise RuntimeError(f"the frozen run saw other visits than the adaptive one: {frozen['part_n']} against {res['part_n']}")
    meta = {"duo": ctx.duo["NAME"], "tta": tta, "bn": ctx.bn, "seed": seed, "stream": stream, "rounds": rounds if stream == "heldout" else 1, "rotation": rotation if stream == "heldout" else None, "stop_mode": stop_mode, "order": order, "images": images, "lrs": list(lrs),
            "momentum": momentum, "proxy": proxy, "beta": beta, "b_t": b_t, "T_l": T_l, "T_s": T_s, "batch_size": rctx.batch_size,
            "watch": list(watch), "after": after, "stopped": res["stopped"], "dead_at": res["dead_at"], "seen": res["seen"],
            "optim": {s.lower(): rctx.duo[s]["OPTIM"] for s in ("LARGE", "SMALL")},
            "models": {s.lower(): rctx.duo[s]["NAME"] for s in ("LARGE", "SMALL")}}
    return res, frozen, parts[:len(res["part_n"])], meta


# ------------------------------------------------------------------------------------------------------------------ the numbers

def rate_str(lr: float) -> str:
    """1.26e-05 -> '1.26e-5', 0.00025 -> '2.5e-4' (a learning rate as a figure title writes it)."""
    mant, exp = f"{lr:.2e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"


def block_means(correct: np.ndarray, size: int = WINDOW) -> np.ndarray:
    """Accuracy (%) per block of `size` samples (a trailing partial block is dropped)."""
    n = len(correct) // size * size
    return correct[:n].reshape(-1, size).mean(1) * 100


def episodes(member: np.ndarray, frozen: np.ndarray, window: int = WINDOW, drop: float = DROP, sustain: int = SUSTAIN) -> list[tuple[int, int]]:
    """The stretches (start, end) in samples where the member stays below `drop` of the frozen member's accuracy on the same images for at
    least `sustain` windows; an episode that lasts to the end of the data ends at the last whole window."""
    n = min(len(member), len(frozen)) // window
    a = member[:n * window].reshape(n, window).mean(1)
    f = frozen[:n * window].reshape(n, window).mean(1)
    below, out, i = a < drop * f, [], 0
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if j - i >= sustain:
                out.append((i * window, j * window))
            i = j
        else:
            i += 1
    return out


def gate_trace(records: Sequence[dict], b_t: int) -> dict[str, np.ndarray]:
    """The gate's proxy batches: `end` the sample count at which batch k was complete (its weight is used from there on), and the weight
    w_L and the proxy logits of both members."""
    return {"end": (np.arange(len(records)) + 1) * b_t, **{k: np.array([r[k] for r in records]) for k in ("w_l", "x_l", "x_s")}}


def weight_on(trace: dict[str, np.ndarray], side: str) -> np.ndarray:
    """The gate's weight on one member: w_L for the large one, 1 - w_L for the small one."""
    return trace["w_l"] if side == "large" else 1.0 - trace["w_l"]


def _trailing_mean(y: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(y)])
    i = np.arange(1, len(y) + 1)
    return (c[i] - c[np.maximum(i - k, 0)]) / (i - np.maximum(i - k, 0))


def episode_reaction(trace: dict[str, np.ndarray], side: str, start: int, end: int, *, last: int = 40, smooth: int = 5, hold: int = 3) -> dict:
    """The gate's weight on `side` around its collapse episode [start, end): the mean of the `last` refreshes before it, the smoothed weight at
    its start and minimum inside it, the mean of its last 10 refreshes, the mean of the 20 refreshes after it (None if fewer than 5 remain),
    and the delay in images from the start until the smoothed weight has covered half the way to its minimum (None if it falls by < 0.1)."""
    w = weight_on(trace, side)
    sm = _trailing_mean(w, smooth)
    e = trace["end"]
    before = np.flatnonzero(e <= start)
    inside = np.flatnonzero((e > start) & (e <= end))
    after = np.flatnonzero(e > end)
    out = {"w_before": float(w[before[-last:]].mean()) if len(before) else float(w[:smooth].mean()), "w_start": None, "w_min": None, "w_end": None,
           "w_after": float(w[after[:20]].mean()) if len(after) >= 5 else None, "delay": None}
    if not len(inside):
        return out
    out.update(w_start=float(sm[inside[0]]), w_min=float(sm[inside].min()), w_end=float(w[inside[-10:]].mean()))
    if out["w_start"] - out["w_min"] >= 0.1:
        half = out["w_start"] - 0.5 * (out["w_start"] - out["w_min"])
        low = np.convolve((sm[inside] <= half).astype(int), np.ones(hold, dtype=int), mode="valid") == hold
        if low.any():
            out["delay"] = int(e[inside[int(np.flatnonzero(low)[0])]] - start)
    return out


def analyze(run: dict, frozen: dict, records: Sequence[dict], b_t: int) -> dict:
    """The numbers of the module docstring for one stored experiment (`run`, `frozen`: RC.load_run results of the same visits)."""
    n = len(run["large"])
    series = {"Frozen duo": frozen["ts"], "Fixed TS duo": run["ts"], "Vanilla DD": run["gate"], "large": run["large"], "small": run["small"]}
    trace = gate_trace(records, b_t)
    out = {"images": n, "dead_at": {s: CT.dead_onset(run[s]) for s in SIDES}, "mean": {k: float(v.mean() * 100) for k, v in series.items()}, "episodes": []}
    for side in SIDES:
        for start, end in episodes(run[side], frozen[side]):
            is_open = end >= n // WINDOW * WINDOW                  # the run ends in it: nothing comes "after"
            out["episodes"].append({"side": side, "start": start, "end": end, "images": end - start, "open": is_open,
                                    "acc": {k: float(v[start:end].mean() * 100) for k, v in series.items()},
                                    **{**episode_reaction(trace, side, start, end), **({"w_after": None} if is_open else {})}})
    out["episodes"].sort(key=lambda ep: ep["start"])
    return out


def describe(an: dict, names: dict[str, str]) -> list[str]:
    """The lines a report prints for one analysis."""
    means = ", ".join(f"{(names.get(k, k))} {v:.1f}" for k, v in an["mean"].items())
    if not an["episodes"]:
        return [f"no member collapsed in {an['images']} images; mean accuracy {means}"]
    lines = [f"{an['images']} images; mean accuracy {means}"]
    for side in SIDES:
        if an["dead_at"][side] is not None:
            lines.append(f"{names[side]} is completely collapsed (below 5% for 20k images) from {an['dead_at'][side]}")
    cols = ("Frozen duo", "Fixed TS duo", "Vanilla DD", "large", "small")
    for i, ep in enumerate(an["episodes"], 1):
        f = lambda v: "-" if v is None else f"{v:.2f}"
        lines.append(f"{names[ep['side']]} collapse episode {i}: images {ep['start']} to {ep['end']}{' (the run ends in it)' if ep['open'] else ''}")
        lines.append(f"  gate weight on it: {f(ep['w_before'])} before, {f(ep['w_start'])} at the start, {f(ep['w_min'])} minimum, {f(ep['w_end'])} at the end, "
                     f"{f(ep['w_after'])} after; "
                     + (f"half-way {ep['delay']} images after the start" if ep["delay"] is not None else "no half-way (it falls by less than 0.1)"))
        lines.append(f"  {'accuracy (%) over the episode':<32}" + "".join(f"{(names[c] if c in names else c):>14}" for c in cols))
        lines.append(f"  {'':<32}" + "".join(f"{ep['acc'][c]:>14.1f}" for c in cols))
    return lines


# ------------------------------------------------------------------------------------------------------------------ the figure

def plot_stress(items: Sequence[dict], path: Path, *, footer: str = "", width: float | None = None, dpi: int = 300) -> list[Path]:
    """One column per item: accuracy of the frozen duo, the Fixed TS duo, the adaptive duo (Vanilla DD) and both members; the gate's weight on
    the large member; the proxy scores it reads. An item: {"title", "run", "frozen", "records", "b_t", "analysis", "names", optionally "beta" and
    "resets" (a live run's resets, [{position, member, ...}]: a triangle per reset at the foot of the accuracy panel and a line in the weight panel,
    in the member's colour)}. Written as `path` (.png) and .pdf. Direct labels with one column, a legend always."""
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
    width = width or (7.2 if cols <= 2 else 3.6 * cols)
    fig = plt.figure(figsize=(width, 5.6 if cols == 1 else 5.9), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(3, cols, height_ratios=[3.0, 1.25, 1.05], hspace=0.14, wspace=0.13, left=0.095 if cols == 1 else 0.075,
                          right=0.835 if cols == 1 else 0.975, top=0.80, bottom=0.105)
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
        resets = item.get("resets") or []
        ax1.set_ylim(-7 if resets else -1, np.ceil(top / 10) * 10 + 2)
        last_label: dict[str, int] = {}
        for ep in an["episodes"]:                                  # a collapse episode: a band on every row, named at the top (once per 60k images)
            for ax in (ax1, ax2, ax3):
                ax.axvspan(ep["start"], ep["end"], color=GRID, alpha=0.7, linewidth=0, zorder=0)
            if ep["start"] - last_label.get(ep["side"], -10 ** 9) >= 60_000:
                ax1.text(ep["start"], 0.975, f" {names[ep['side']]}\n collapses", transform=ax1.get_xaxis_transform(), fontsize=5.8, color=INK, ha="left",
                         va="top", linespacing=1.0)
                last_label[ep["side"]] = ep["start"]
        if cols == 1:
            for label, yv in _spread(ends, (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.07).items():
                color = next(l[2] for l in lines if l[0] == label)
                ax1.annotate(label, (n, ends[label]), xytext=(n * 1.012, yv), textcoords="data", fontsize=6.2, color=INK, va="center", annotation_clip=False,
                             arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6, shrinkA=0, shrinkB=1))
        ax1.set_title(item["title"], fontsize=6.6, color=INK, loc="left", pad=47)
        if c == 0:
            ax1.set_ylabel("Accuracy (%)")

        # the gate's weight on the large member
        tr = gate_trace(item["records"], item["b_t"])
        ax2.axhline(0.5, color=INK2, linewidth=0.5, linestyle=":", zorder=1)
        ax2.plot(tr["end"], tr["w_l"], color=BLUE, linewidth=0.45, alpha=0.35, zorder=2)
        ax2.plot(tr["end"], _trailing_mean(tr["w_l"], 5), color=BLUE, linewidth=1.3, zorder=3)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_yticks([0, 0.5, 1])
        for s in SIDES:                                             # the resets: a triangle at the foot of the accuracy panel, a line through the weight panel
            at = [r["position"] for r in resets if r["member"] == s]
            if at:
                ax1.scatter(at, [ax1.get_ylim()[0] + (1.7 if s == "large" else 4.1)] * len(at), marker="^", s=9, color=MEMBER[s][1], linewidths=0, zorder=6, clip_on=False)
                for a_ in at:
                    ax2.axvline(a_, color=MEMBER[s][1], linewidth=0.5, alpha=0.45, zorder=1)
        if c == 0:
            ax2.set_ylabel(f"Gate weight\non {names['large']}" + (f"\n(beta {item['beta']:g})" if item.get("beta") is not None else ""))

        # the proxy scores the gate reads
        for s, key in (("large", "x_l"), ("small", "x_s")):
            ax3.plot(tr["end"], _trailing_mean(tr[key], 5), color=MEMBER[s][1], linestyle=MEMBER[s][2], linewidth=1.0, label=names[s])
        ax3.set_xlabel("Observed samples", labelpad=1)
        if c == 0:
            ax3.set_ylabel("Proxy score\n(logit)")
        else:
            for ax in (ax1, ax2, ax3):
                ax.tick_params(labelleft=False)

    h, l_ = first_axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    for s in SIDES:
        if any(r["member"] == s for item in items for r in item.get("resets") or []):
            h.append(Line2D([], [], marker="^", linestyle="", color=MEMBER[s][1], markersize=3.5, label=f"{items[0]['names'][s]} reset"))
            l_.append(f"{items[0]['names'][s]} reset")
    fig.legend(h, l_, loc="upper center", ncol=min(len(h), 7), frameon=False, fontsize=6.3, bbox_to_anchor=(0.46 if cols == 1 else 0.53, 1.0),
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

    # ---- the numbers on a synthetic run: the large member collapses at 40k of 120k images for good, the gate follows 3k later
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

    assert episodes(small, frozen["small"]) == [] and episodes(frozen["large"], frozen["large"]) == []
    ep = episodes(large, frozen["large"])
    assert len(ep) == 1 and onset_true <= ep[0][0] <= onset_true + 2 * WINDOW and ep[0][1] == n // WINDOW * WINDOW, ep
    assert episodes(large[:onset_true + 3 * WINDOW], frozen["large"]) == [], "fewer than 5 windows below: no episode"
    an = analyze(run, frozen, records, b_t)
    assert an["images"] == n and len(an["episodes"]) == 1 and an["dead_at"]["small"] is None and an["dead_at"]["large"] is not None
    e = an["episodes"][0]
    assert e["side"] == "large" and e["open"] and (e["start"], e["end"]) == ep[0] and e["images"] == e["end"] - e["start"]
    assert abs(e["w_before"] - 0.55) < 0.05 and e["w_start"] > e["w_min"] and abs(e["w_min"] - 0.15) < 0.06 and e["w_after"] is None, e
    assert e["delay"] is not None and 0 <= e["delay"] <= 8000, e
    assert abs(e["acc"]["Vanilla DD"] - 44) < 3 and abs(e["acc"]["small"] - 45) < 3 and e["acc"]["Fixed TS duo"] < 8 and e["acc"]["large"] < 2
    assert abs(an["mean"]["Frozen duo"] - 40) < 1
    steady = analyze({**run, "large": small, "small": small}, frozen, records, b_t)
    assert steady["episodes"] == [] and "no member collapsed" in describe(steady, {"large": "ViT", "small": "ResNet"})[0]
    text = describe(an, {"large": "ViT", "small": "ResNet"})
    assert any("ViT collapse episode 1" in l for l in text) and any("completely collapsed" in l for l in text)

    # a collapse that ends when the next shift starts (the ViT's): two episodes, the weight comes back
    large2 = (rng.random(n) < 0.55).astype(np.uint8)
    for a_, b_ in ((20_480, 40_960), (71_680, 87_040)):
        large2[a_:b_] = (rng.random(b_ - a_) < 0.01).astype(np.uint8)
    w2 = np.full(n // b_t, 0.58)
    for a_, b_ in ((20_480, 40_960), (71_680, 87_040)):
        w2[a_ // b_t:b_ // b_t] = np.linspace(0.58, 0.2, (b_ - a_) // b_t)
    rec2 = [{"w_l": float(w), "x_l": 0.0, "x_s": 0.0} for w in w2 + rng.normal(0, 0.01, len(w2))]
    an2 = analyze({**run, "large": large2}, frozen, rec2, b_t)
    assert [(x["start"], x["end"]) for x in an2["episodes"]] == [(20_480, 40_960), (71_680, 87_040)], an2["episodes"]
    assert all(x["w_after"] > 0.5 and x["w_min"] < 0.3 and x["w_before"] > 0.5 for x in an2["episodes"]), an2["episodes"]
    assert all(not x["open"] for x in an2["episodes"])
    assert an2["dead_at"]["large"] == 20_480, "the first episode is 4 whole dead windows long: it is a complete collapse, though not a permanent one"

    # the small member's weight is the mirror image of the large one's
    mirror = episode_reaction(gate_trace([{**r, "w_l": 1 - r["w_l"]} for r in records], b_t), "small", e["start"], e["end"])
    same = episode_reaction(gate_trace(records, b_t), "large", e["start"], e["end"])
    assert all((mirror[k_] is None and same[k_] is None) or abs(mirror[k_] - same[k_]) < 1e-9 for k_ in mirror)
    flat = episode_reaction(gate_trace([{"w_l": 0.5, "x_l": 0.0, "x_s": 0.0}] * 200, b_t), "large", 10_000, 30_000)
    assert flat["delay"] is None, "a gate that does not move has no half-way"
    assert np.allclose(_trailing_mean(np.array([1.0, 2, 3, 4]), 2), [1, 1.5, 2.5, 3.5])
    assert [rate_str(v) for v in (1.26e-5, 0.00025, 3.5e-5, 2.2e-5, 7e-5, 1e-3)] == ["1.26e-5", "2.5e-4", "3.5e-5", "2.2e-5", "7e-5", "1e-3"]

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
        CT.tuning_parts = lambda ctx_, **kw: [RC.Part(s, r, kw["seed"] + 1000 * r) for r in range(kw["rounds"]) for s in segs[:2]][kw.get("rotation", 0):]
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
