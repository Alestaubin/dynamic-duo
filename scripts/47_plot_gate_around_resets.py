#!/usr/bin/env python3
"""
Where the gate gains and loses against fixed TS around a reset (own_fast_vs_slow, Vanilla DD). Every reset of a member is a time-zero; the stored live runs
(`out/results/reset_live/<duo>__tent_bn/`, the frozen configuration of `final_configs.json`, the gate at beta 0.5) are cut +- `--half` samples around it and averaged over the
resets of that member (a reset takes effect at the sample `position` of the run's meta: the window before is [position - half, position), the one after [position, position + half)).
Two columns (the ViT's resets, the ResNet's resets), three rows:

    accuracy        the four series of the run (ViT, ResNet, fixed-TS duo, gated duo) around the reset (test stream)
    gate - TS       the gated duo's accuracy minus the fixed-TS duo's on the same samples, in points, +- SE over resets (band: test; dashed: held-out)
    cumulative      the same gap summed over samples from -half: images gained (rising) or lost (falling) per reset

The two duos see the same samples and the same members' logits (the gate is outside the loop in Vanilla DD), so the gap is the gate's effect alone. Windows are not cut at the
shifts of the stream: a reset within `--half` samples of a shift boundary mixes two corruptions (10 of the 54 ViT resets on the test stream).

    python scripts/47_plot_gate_around_resets.py                       # docs/figures/reset_live/gate_around_resets.{png,pdf}
    python scripts/47_plot_gate_around_resets.py --beta 0.5 --half 5120 --bin 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import continual as RC
from src.reset import live_report as LV
from src.utils.config import load_global

SIDES = ("large", "small")
TEST_KEYS, HELDOUT_KEYS = [None, 0, 1, 2, 3], [0, 1, 2, 3]


def load(folder: Path, m: dict, stream: str, beta: float) -> list[dict]:
    runs = []
    for k in TEST_KEYS if stream == "test" else HELDOUT_KEYS:
        kw = dict(seed=0, lrs=m["lrs"], proxy="nnn", b_t=256, beta=beta)
        stem = LV.run_stem(m["cond"], stream="heldout", rot=k, **kw) if stream == "heldout" else LV.run_stem(m["cond"], stream="test", rot=0, order=k, **kw)
        if (folder / f"{stem}.npz").is_file():
            runs.append(RC.load_run(folder / f"{stem}.npz"))
    if not runs:
        raise SystemExit(f"no {stream} run of {m['name']} at beta {beta:g} in {folder}")
    return runs


def around(runs: list[dict], side: str, half: int) -> dict[str, np.ndarray]:
    """{series: (resets, 2 * half) 0/1 correctness of `side`'s resets that have `half` samples on both sides}; row = one reset."""
    out: dict[str, list] = {s: [] for s in ("ts", "gate", "large", "small")}
    out["spans_shift"] = []
    for r in runs:
        edges = np.cumsum(r["part_n"])[:-1]
        for res in r["meta"].get("resets", []):
            pos = int(res["position"])
            if res["member"] == side and half <= pos <= len(r["ts"]) - half:
                for s in ("ts", "gate", "large", "small"):
                    out[s].append(np.asarray(r[s][pos - half:pos + half], dtype=float))
                out["spans_shift"].append(bool(((edges > pos - half) & (edges < pos + half)).any()))
    return {s: np.array(v) for s, v in out.items()}


def inside_outside(runs: list[dict], half: int) -> tuple[float, float, int]:
    """Images the gate gained over fixed TS (sum of gate - ts correctness) inside the +- `half` windows around ANY reset (either member; the union, so overlaps count
    once) and outside them, and the number of images."""
    inside = outside = 0.0
    n_img = 0
    for r in runs:
        d = np.asarray(r["gate"], dtype=float) - np.asarray(r["ts"], dtype=float)
        mask = np.zeros(len(d), dtype=bool)
        for res in r["meta"].get("resets", []):
            pos = int(res["position"])
            mask[max(pos - half, 0):pos + half] = True
        inside += d[mask].sum()
        outside += d[~mask].sum()
        n_img += len(d)
    return inside, outside, n_img


def binned(x: np.ndarray, size: int) -> np.ndarray:
    """(resets, 2 * half) -> (resets, bins): mean per bin."""
    return x.reshape(x.shape[0], -1, size).mean(2)


def mean_se(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over resets (axis 0) and its standard error."""
    return x.mean(0), (x.std(0, ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.full(x.shape[1:], np.nan))


def cumulative(gap: np.ndarray, size: int) -> np.ndarray:
    """(resets, 2 * half) per-sample gap -> (resets, bins + 1): images gained since the window's start, at the bin edges (0 at the start)."""
    c = np.cumsum(gap, axis=1)[:, size - 1::size]
    return np.concatenate([np.zeros((len(gap), 1)), c], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--names", nargs=2, default=["ViT-B/16", "ResNet-50"], metavar=("LARGE", "SMALL"), help="the members' names in the figure")
    ap.add_argument("--method", default="own_fast_vs_slow", help="name in final_configs.json (default %(default)s)")
    ap.add_argument("--beta", type=float, default=0.5, help="the stored gate's beta (the final test's)")
    ap.add_argument("--half", type=int, default=5120, help="samples before and after a reset (default %(default)s = 20 proxy batches)")
    ap.add_argument("--bin", type=int, default=512, help="samples per point of the first two rows (default %(default)s)")
    ap.add_argument("--out", default=None, help="figure path without suffix (default docs/figures/reset_live/gate_around_resets)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    if args.half % args.bin:
        raise SystemExit("--half must be a multiple of --bin")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from src.experiments.continual_report import GRID, INK, INK2, MEMBER
    from src.utils import provenance

    g = load_global()
    cfg = json.loads((g.path("out") / "results" / "reset_tune" / "final_configs.json").read_text())
    m = next((x for x in cfg["methods"] if x["name"] == args.method), None)
    if m is None:
        raise SystemExit(f"not in final_configs.json: {args.method}")
    folder = g.path("out") / "results" / "reset_live" / f"{args.duo}__tent_bn"
    runs = {s: load(folder, m, s, args.beta) for s in ("test", "heldout")}
    half, size = args.half, args.bin
    nb = 2 * half // size
    xb = (np.arange(nb) + 0.5) * size - half                                          # bin centres, samples relative to the reset
    xe = np.arange(nb + 1) * size - half                                              # bin edges
    data = {(s, side): around(runs[s], side, half) for s in runs for side in SIDES}
    for s, (i, o, n) in {s: inside_outside(runs[s], half) for s in runs}.items():
        print(f"{s:8s} whole stream: {i:+,.0f} images inside the +-{half} windows of any reset, {o:+,.0f} outside, net {i + o:+,.0f} = {100 * (i + o) / n:+.3f} pts")
    for (s, side), d in data.items():
        print(f"{s:8s} {side:6s} {len(d['ts']):3d} resets"
              f" ({int(d['spans_shift'].sum())} span a shift)   gap in the {half // 4} samples before: {100 * (d['gate'] - d['ts'])[:, half - half // 4:half].mean():+.2f}   after: {100 * (d['gate'] - d['ts'])[:, half:half + half // 4].mean():+.2f}")

    plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    TS, GATE = "#1baf7a", "#2a78d6"                                                   # the codebase's colours of the fixed-TS duo and of Vanilla DD (continual_report)
    members = {"large": (args.names[0], *MEMBER["large"][1:]), "small": (args.names[1], *MEMBER["small"][1:])}
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.8), dpi=args.dpi, sharex=True, sharey="row", facecolor="white")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.845, bottom=0.075, hspace=0.12, wspace=0.09)
    pct = lambda v: 100 * v

    for col, side in enumerate(SIDES):
        test, held = data[("test", side)], data[("heldout", side)]
        a0, a1, a2 = axes[0, col], axes[1, col], axes[2, col]
        for ax in axes[:, col]:
            ax.set_facecolor("white")
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.axvline(0, color=INK2, linewidth=0.7, linestyle=(0, (4, 2)), zorder=1)
        # row 1: accuracy of the four series (test)
        for key, label, color, ls, lw in (("large", *members["large"], 1.0), ("small", *members["small"], 1.0),
                                          ("ts", "Fixed TS duo", TS, "-", 1.4), ("gate", "Vanilla DD (gate)", GATE, "-", 1.4)):
            a0.plot(xb, pct(binned(test[key], size).mean(0)), color=color, linestyle=ls, linewidth=lw, label=label, zorder=3)
        # rows 2 and 3: gate - fixed TS
        for d, ls, lw, band in ((test, "-", 1.4, True), (held, (0, (3, 1.5)), 1.0, False)):
            gap = binned(d["gate"] - d["ts"], size) * 100
            mu, se = mean_se(gap)
            a1.plot(xb, mu, color=GATE, linestyle=ls, linewidth=lw, zorder=3)
            if band:
                a1.fill_between(xb, mu - se, mu + se, color=GATE, alpha=0.18, linewidth=0, zorder=2)
            cum = cumulative(d["gate"] - d["ts"], size)
            cmu, cse = mean_se(cum)
            a2.plot(xe, cmu, color=GATE, linestyle=ls, linewidth=lw, zorder=3)
            if band:
                a2.fill_between(xe, cmu - cse, cmu + cse, color=GATE, alpha=0.18, linewidth=0, zorder=2)
        a1.axhline(0, color=INK2, linewidth=0.6, zorder=1)
        a2.axhline(0, color=INK2, linewidth=0.6, zorder=1)
        # the headline numbers of the test stream: images gained before the reset, lost after, net
        cum_t = cumulative(test["gate"] - test["ts"], size)
        before, net = cum_t[:, nb // 2].mean(), cum_t[:, -1].mean()
        a2.text(0.02, 0.97, f"test, per reset: {before:+.0f} images before, {net - before:+.0f} after, {net:+.0f} net", transform=a2.transAxes, ha="left", va="top", fontsize=6.2, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=5)
        a0.set_title(f"{members[side][0]} resets   (n = {len(test['ts'])} test, {len(held['ts'])} held-out; {int(test['spans_shift'].sum())} / {int(held['spans_shift'].sum())} span a shift)",
                     fontsize=6.8, color=INK, loc="left", pad=4)
    for ax in axes[0]:
        ax.text(0, 0.985, " reset", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=5.8, color=INK2)
    axes[0, 0].set_ylabel("Accuracy (%)\n(test stream)")
    axes[1, 0].set_ylabel("Gate $-$ fixed TS\n(points)")
    axes[2, 0].set_ylabel("Cumulative gate $-$ fixed TS\n(images per reset)")
    for ax in axes[2]:
        ax.set_xlim(-half, half)
        ticks = np.linspace(-half, half, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v / 1000:+.1f}k" if v else "0" for v in ticks])
        ax.get_xticklabels()[0].set_ha("left")                                            # the outer labels stay inside their panel: no collision between the columns
        ax.get_xticklabels()[-1].set_ha("right")
        ax.set_xlabel("Samples relative to the reset", labelpad=2)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=4, frameon=False, fontsize=6.6, bbox_to_anchor=(0.545, 0.975), handlelength=2.6, columnspacing=1.6)
    axes[1, 0].legend([Line2D([], [], color=GATE, linewidth=1.4), Line2D([], [], color=GATE, linewidth=1.0, linestyle=(0, (3, 1.5)))],
                      [f"Test ({len(runs['test'])} orderings), band: SE", f"Held-out ({len(runs['heldout'])} rotations)"],
                      loc="upper right", frameon=False, fontsize=6.0, handlelength=2.6)
    fig.suptitle(f"Own fast/slow resets, gate (beta {args.beta:g}) against fixed TS: where the gate gains and where it loses", fontsize=7.4, color=INK, x=0.105, ha="left", y=0.995)
    tally = {s: inside_outside(runs[s], half) for s in runs}
    fig.text(0.105, 0.935, f"Whole stream, images gained over fixed TS inside / outside the \u00b1{half:,} samples around any reset (either member):", fontsize=6.2, color=INK2, ha="left", va="top")
    fig.text(0.105, 0.918, "     " + "        ".join(f"{s}: {i:+,.0f} / {o:+,.0f}  (net {i + o:+,.0f} = {100 * (i + o) / n:+.2f} pts)" for s, (i, o, n) in tally.items()),
             fontsize=6.2, color=INK, ha="left", va="top")
    footer = (f"scripts/47_plot_gate_around_resets.py | {args.duo}, {m['name']} (rates {m['lrs'][0]:g} / {m['lrs'][1]:g}), gate beta {args.beta:g}, proxy batch 256 | test: "
              f"{len(runs['test'])} orderings of the 15-shift stream, held-out: {len(runs['heldout'])} rotations | +-{half} samples around each reset, {size}-sample bins; "
              "windows are not cut at the shifts of the stream")
    dest = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "docs" / "figures" / "reset_live" / "gate_around_resets"
    for p in provenance.save_figure(fig, dest, footer, dpi=args.dpi):
        print("wrote", p)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
