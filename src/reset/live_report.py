"""
live_report.py
==============
What scripts/39_tab_reset.py reads out of the live reset runs of scripts/38_run_reset_live.py (src/reset/live_runs.py): one summary per run, the audit
of its resets, the table over conditions, and the overlay figure of every condition on one stream.

    run, frozen = RC.load_run(live_file), RC.load_run(frozen_file)
    s = summarize(run, frozen, b_t=256)         # mean accuracy (%) of the four series, resets per member, the audit
    rows = table_rows({"none": [s1, s2], "own_fast_vs_slow": [s3, s4]})        # one row per condition: mean (and spread) over runs

**Audit of a reset** (labels, evaluation only; never a rule's input). A reset of member m at stream position p is JUSTIFIED if m's accuracy over the
last `M` proxy batches before p was more than `margin` below its FROZEN copy's on the same images (the label definition of screening.degraded_stretches,
read causally), else FALSE: it reset a member that was not below its frozen copy. The false share is the cost side of a rule; the whole-stream accuracy
is its net effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SIDES = ("large", "small")
M, MARGIN = 8, 0.01


def audit_resets(run: Mapping, frozen: Mapping, resets: Sequence[Mapping], b_t: int = 256, window: int = M, margin: float = MARGIN) -> dict:
    """Per member: resets, justified ones, false ones (see the module text). A reset with fewer than one proxy batch behind it counts as false."""
    out = {s: {"resets": 0, "justified": 0, "false": 0} for s in SIDES}
    for r in resets:
        s, p = r["member"], int(r["position"])
        lo = max(p - window * b_t, 0)
        n = min(len(run[s]), len(frozen[s]))
        lo, hi = min(lo, n), min(p, n)
        ok = hi - lo >= b_t and float(run[s][lo:hi].mean() - frozen[s][lo:hi].mean()) < -margin
        out[s]["resets"] += 1
        out[s]["justified" if ok else "false"] += 1
    return out


def summarize(run: Mapping, frozen: Mapping, b_t: int = 256) -> dict:
    """Mean accuracy (%) of Vanilla DD (`gate`), the fixed-TS duo, both members over the run, the frozen duo and members over the same images, and the audit."""
    n = len(run["large"])
    resets = run["meta"].get("resets", [])
    steps = int(run["meta"].get("seen", n))
    out = {"images": n, "mean": {s: float(run[s].mean() * 100) for s in ("gate", "ts", "large", "small") if s in run},
           "frozen": {s: float(frozen[s][:n].mean() * 100) for s in ("ts", "large", "small")},
           "audit": audit_resets(run, frozen, resets, b_t), "resets": [{"member": r["member"], "position": r["position"], "why": r["why"]} for r in resets]}
    out["per_1000_batches"] = {s: 1000.0 * out["audit"][s]["resets"] / max(-(-n // b_t), 1) for s in SIDES}
    return out


def table_rows(by_condition: Mapping[str, Sequence[dict]]) -> list[dict]:
    """One row per condition: the mean over its runs (std over runs when there are several) of each accuracy, the gain over the run's own frozen duo,
    resets per 1,000 proxy batches and per member, and the share of false resets."""
    rows = []
    for cond, runs in by_condition.items():
        def stat(f):
            v = np.array([f(r) for r in runs], dtype=float)
            return float(np.nanmean(v)), float(np.nanstd(v)) if len(v) > 1 else 0.0

        n_res = {s: sum(r["audit"][s]["resets"] for r in runs) for s in SIDES}
        n_false = {s: sum(r["audit"][s]["false"] for r in runs) for s in SIDES}
        row = {"condition": cond, "runs": len(runs)}
        for key in ("gate", "ts", "large", "small"):
            if all(key in r["mean"] for r in runs):
                row[key], row[key + "_sd"] = stat(lambda r, k=key: r["mean"][k])
        if all("gate" in r["mean"] for r in runs):
            row["gain_vs_frozen_duo"], row["gain_sd"] = stat(lambda r: r["mean"]["gate"] - r["frozen"]["ts"])
        row.update({f"resets_per_1000_{s}": stat(lambda r, s=s: r["per_1000_batches"][s])[0] for s in SIDES})
        row.update({f"resets_{s}": n_res[s] for s in SIDES})
        row["false_share"] = (sum(n_false.values()) / sum(n_res.values())) if sum(n_res.values()) else float("nan")
        rows.append(row)
    return rows


def run_stem(cond: str, *, stream: str, rot: int, seed: int, lrs: Sequence[float], proxy: str, b_t: int, beta: float, max_images: int | None = None,
             order: int | None = None) -> str:
    """The file stem of a live run (scripts/38_run_reset_live.py writes it, scripts/39_tab_reset.py reads it); the frozen run does not depend on the rates or the gate.
    `order`: a test stream in one of CoTTA's random orders (None = the standard order, which keeps its old names)."""
    where = ("__heldout" if stream == "heldout" else "") + (f"__rot{rot}" if rot else "") + (f"__order{order}" if order is not None and stream == "test" else "")
    cut = f"__first{max_images}" if max_images else ""
    if cond == "frozen":
        return f"live-frozen{where}{cut}__seed{seed}"
    return f"live-{cond}{where}__L{lrs[0]:g}_S{lrs[1]:g}__{proxy}_bt{b_t}_beta{beta:g}{cut}__seed{seed}"


NAMES = {"none": "No reset", "boundary": "At every shift (episodic)", "frozen": "Frozen"}


def pretty(cond: str) -> str:
    """own_fast_vs_slow -> 'own fast/slow'; fixed1000 -> 'every 1,000 steps'."""
    if cond in NAMES:
        return NAMES[cond]
    if cond.startswith("fixed") and cond[5:].isdigit():
        return f"Every {int(cond[5:]):,} steps"
    q, _, kind = cond.partition("_")
    return f"{q} {kind.replace('_vs_', '/').replace('_', ' ')}"


def latex_table(rows: Sequence[dict], caption: str, label: str, frozen: float | None = None, script: str = "scripts/39_tab_reset.py") -> str:
    """tab:reset_live: whole-stream accuracy of every condition (needs booktabs)."""
    def f(row, k, sd=None):
        if k not in row:
            return "--"
        s = f"{row[k]:.1f}"
        return s + (f" $\\pm$ {row[sd]:.1f}" if sd and row.get(sd, 0) else "")

    lines = [f"% Generated by {script}", "\\begin{table}[t]", "  \\centering\\small", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             "  \\begin{tabular}{lrrrrrr}", "    \\toprule",
             "    condition & Vanilla DD & ViT-B/16 & ResNet-50 & vs frozen duo & resets (ViT / ResNet) & false (\\%) \\\\", "    \\midrule"]
    if frozen is not None:
        lines.append(f"    Frozen duo & {frozen:.1f} & -- & -- & -- & -- & -- \\\\")
    for r in rows:
        gain = f"{r['gain_vs_frozen_duo']:+.1f}" if "gain_vs_frozen_duo" in r else "--"
        false = "--" if r["false_share"] != r["false_share"] else f"{100 * r['false_share']:.0f}"
        lines.append(f"    {pretty(r['condition'])} & {f(r, 'gate', 'gate_sd')} & {f(r, 'large', 'large_sd')} & {f(r, 'small', 'small_sd')} & {gain} & "
                     f"{r['resets_per_1000_large']:.1f} / {r['resets_per_1000_small']:.1f} & {false} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------ the overlay figure

def plot_overlay(runs: Mapping[str, Mapping], frozen: Mapping, path: Path, *, names: Mapping[str, str], title: str, footer: str, window: int = 5120,
                 dpi: int = 300) -> list[Path]:
    """Three rows (Vanilla DD, the large member, the small member): the accuracy over windows of `window` images of every condition in `runs`
    ({condition: load_run result}); the frozen duo / members dashed in grey. Lines of the label-free rules in colour, references (no reset, episodic)
    in greys and black so the eye finds the rules."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from src.experiments.continual_report import GRID, INK, INK2, PLAIN
    from src.utils import provenance

    plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#a05cc4", "#c9a227", "#d6428a", "#3aa7b8", "#7a5230"]
    refs = {"none": ("#0b0b0b", "-", 1.2), "boundary": ("#6b6a66", (0, (5, 2)), 1.0)}
    first = next(iter(runs.values()))
    n = min(len(r["large"]) for r in runs.values())
    window = max(min(window, n // 12), 256)                 # a short (smoke) stream still gets a line
    edges = np.r_[0, np.cumsum(first["part_n"])]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), dpi=dpi, sharex=True, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.76, top=0.86, bottom=0.085, hspace=0.12)
    rows = (("gate", "ts", "Vanilla DD (the gated duo)"), ("large", "large", names["large"]), ("small", "small", names["small"]))
    x = (np.arange(n // window) + 0.5) * window
    k = 0
    for ax, (series, frozen_key, ylabel) in zip(axes, rows):
        ax.set_facecolor("white")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for e in edges[1:-1]:
            ax.axvline(e, color=INK2, linewidth=0.4, linestyle="--", alpha=0.5, zorder=1)
        fy = frozen[frozen_key][:n // window * window].reshape(-1, window).mean(1) * 100
        ax.plot(x, fy, color="#6b6a66", linestyle=(0, (4, 2)), linewidth=0.9, label="Frozen" if series != "gate" else "Frozen duo", zorder=2)
        k = 0
        for cond, run in runs.items():
            y = run[series][:n // window * window].reshape(-1, window).mean(1) * 100
            if cond in refs:
                color, ls, lw = refs[cond]
            else:
                color, ls, lw = palette[k % len(palette)], "-", 1.2
                k += 1
            ax.plot(x, y, color=color, linestyle=ls, linewidth=lw, label=pretty(cond), zorder=3)
        ax.set_ylabel(ylabel + "\naccuracy (%)")
        ax.set_xlim(0, n)
    for lo, hi, label in zip(edges[:-1], edges[1:], first["part_label"]):
        if hi <= n:
            axes[0].text((lo + hi) / 2, 1.015, PLAIN[label.split("/")[0]], transform=axes[0].get_xaxis_transform(), rotation=90, ha="center", va="bottom",
                         fontsize=5.4, color=INK2)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
    axes[-1].set_xlabel("Observed samples", labelpad=1)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="center left", frameon=False, fontsize=6.3, bbox_to_anchor=(0.775, 0.5), handlelength=2.4)
    fig.suptitle(title, fontsize=6.8, color=INK, x=0.08, ha="left", y=0.995)
    out = provenance.save_figure(fig, path, footer, dpi=dpi)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    n, b_t = 20_480, 256
    frozen = {s: (rng.random(n) < 0.5).astype(np.uint8) for s in ("ts", "large", "small")}
    run = {"large": frozen["large"].copy(), "small": frozen["small"].copy(), "ts": frozen["ts"].copy(), "gate": frozen["gate"] if "gate" in frozen else frozen["ts"].copy(),
           "part_n": np.array([10_240, 10_240]), "part_label": np.array(["gaussian_noise/s5", "shot_noise/s5"])}
    run["small"][8_192:12_288] = (rng.random(4_096) < 0.05).astype(np.uint8)             # the small member collapses over 8,192..12,288
    resets = [{"member": "small", "position": 10_240, "why": "own_fast_vs_slow"},               # inside the collapse (2,048 images of it behind it): justified
              {"member": "large", "position": 10_240, "why": "own_fast_vs_slow"},               # the large member is fine: false
              {"member": "small", "position": 16_384, "why": "fixed"},                   # after the collapse: false
              {"member": "large", "position": 100, "why": "fixed"}]                      # less than a proxy batch behind it: false
    run["meta"] = {"resets": resets, "seen": n}
    a = audit_resets(run, frozen, resets, b_t)
    assert a["small"] == {"resets": 2, "justified": 1, "false": 1} and a["large"] == {"resets": 2, "justified": 0, "false": 2}, a
    s = summarize(run, frozen, b_t)
    assert abs(s["mean"]["large"] - s["frozen"]["large"]) < 1e-9 and s["mean"]["small"] < s["frozen"]["small"] - 3
    assert abs(s["per_1000_batches"]["small"] - 1000 * 2 / 80) < 1e-9 and len(s["resets"]) == 4
    quiet = {**run, "meta": {"resets": [], "seen": n}}
    rows = table_rows({"none": [summarize(quiet, frozen, b_t), summarize(quiet, frozen, b_t)], "own_fast_vs_slow": [s]})
    assert rows[0]["resets_small"] == 0 and rows[0]["false_share"] != rows[0]["false_share"] and rows[1]["false_share"] == 0.75 and rows[0]["runs"] == 2
    assert abs(rows[0]["gain_vs_frozen_duo"]) < 1e-9 and rows[0]["gate_sd"] == 0.0
    kw = dict(stream="heldout", rot=2, seed=0, lrs=(1.4e-4, 5e-4), proxy="nnn", b_t=256, beta=2.0)
    assert run_stem("own_fast_vs_slow", **kw) == "live-own_fast_vs_slow__heldout__rot2__L0.00014_S0.0005__nnn_bt256_beta2__seed0"
    assert run_stem("frozen", **{**kw, "rot": 0}) == "live-frozen__heldout__seed0" and run_stem("none", **{**kw, "stream": "test", "rot": 0, "max_images": 100}).endswith("beta2__first100__seed0")
    assert run_stem("none", **{**kw, "stream": "test", "rot": 0, "order": 3}) == "live-none__order3__L0.00014_S0.0005__nnn_bt256_beta2__seed0" and run_stem("none", **{**kw, "stream": "test", "rot": 0}).startswith("live-none__L")
    assert [pretty(c) for c in ("none", "fixed1000", "own_fast_vs_slow")] == ["No reset", "Every 1,000 steps", "own fast/slow"]
    tex = latex_table(rows, "caption", "tab:reset_live", frozen=50.0)
    assert tex.startswith("% Generated by") and "\\label{tab:reset_live}" in tex and "No reset" in tex and "own fast/slow" in tex and tex.count("\\\\") >= 4
    with tempfile.TemporaryDirectory() as tmp:
        written = plot_overlay({"none": quiet, "own_fast_vs_slow": run, "fixed1000": run}, frozen, Path(tmp) / "overlay.png", names={"large": "ViT-B/16", "small": "ResNet-50"},
                               title="self-test", footer="self-test", window=1024)
        assert [p.suffix for p in written] == [".png", ".pdf"] and all(p.stat().st_size > 5000 for p in written)
    print("live_report self-test passed")
