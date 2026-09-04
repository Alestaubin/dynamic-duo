"""Shared per-batch / per-proxy-batch diagnostics plotting.

Used by both scripts/run_dynamic_duo.py and scripts/plot_run_diagnostics.py so
the two never drift into two slightly-different versions of the same plot.

Large/small only -- no duo series. The duo's combined output is what the
joint calibrator under test produces; these plots exist to compare the two
INPUT models to each other and to the proxy signal, so a duo line here would
only ever be a third, differently-scaled series crowding the same axes
without answering that question.

plot_batch_diagnostics: accuracy/NLL/entropy for large/small, per adaptation
batch -- the direct "is one model collapsing" signal (ground truth, ignores
the proxy entirely).

plot_proxy_diagnostics: for calibration_mode=proxy_weighted runs only, the
raw per-model proxy scores (r_l, r_s) and the resulting gate weight (w_l)
plotted against each proxy batch's ACTUAL accuracy -- this is what answers
"does the proxy track a model collapsing", since a useful proxy should dip
for whichever model's real accuracy is dipping.

plot_per_corruption_proxy_vs_accuracy: same question as plot_proxy_diagnostics,
but one figure per corruption instead of one figure spanning the whole run --
EMA-smoothed accuracy (bold SOLID, right axis) vs. EMA-smoothed proxy score
(bold DOTTED, left axis, raw score also shown faint) vs. EMA-smoothed entropy
(DASH-DOT, its own third axis -- nats, not on [0, 1] like the other two) so a
collapse and a proxy dip within a single corruption stream are easy to
eyeball side by side, and so a proxy that's really just tracking entropy (see
e.g. nuclear_norm -- a near-monotone function of confidence) rather than
accuracy is visible directly on the plot. Proxy and accuracy get distinct
line styles (not just distinct axes) deliberately -- once both are EMA-bold
lines in the same per-model color, style is what keeps them from reading as
the same line at a glance.

All three functions plot every series as an EMA (bold) over the raw
per-point values (faint), with a shared `ema_window` hyperparameter (see
_ema below) -- a cumulative running mean never forgets a stale batch from
the start of a corruption, so it lags real collapses; an EMA with a small
window reacts to recent behavior instead. plot_batch_diagnostics and
plot_proxy_diagnostics span the whole run (multiple corruptions
concatenated), so their EMA resets at each corruption boundary --
plot_per_corruption_proxy_vs_accuracy already operates on one corruption's
rows at a time, so its EMA needs no explicit reset.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette (dataviz skill's validated categorical slots 1/2/7 -- blue/orange
# for large/small identity everywhere, violet for the gate weight w_l so it
# never collides with a model color).
C_LARGE = "#2a78d6"
C_SMALL = "#eb6834"
C_GATE = "#4a3aa7"
C_INK = "#0b0b0b"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK,
    "text.color": C_INK, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "grid.color": C_GRID, "font.size": 10,
})

DEFAULT_EMA_WINDOW = 10


def _ema(values: list[float], window: int, reset_idxs: set[int] | None = None) -> list[float]:
    """Exponential moving average, span-style: alpha = 2 / (window + 1).

    reset_idxs (if given) are indices where accumulation restarts from that
    point's raw value instead of blending with the prior EMA state -- used
    to keep a multi-corruption series from smearing signal across a
    corruption boundary, where a genuine, instantaneous regime change is
    expected rather than noise to smooth out.
    """
    alpha = 2.0 / (window + 1.0)
    reset_idxs = reset_idxs or set()
    out = []
    prev = None
    for i, v in enumerate(values):
        if prev is None or i in reset_idxs:
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


def _mark_corruption_boundaries(ax, boundaries: list[dict], n: int) -> None:
    for b in boundaries:
        if b["idx"] == 0:
            continue
        ax.axvline(b["idx"], color=C_MUTED, lw=0.8, ls="--", alpha=0.6, zorder=1)
    ymin, ymax = ax.get_ylim()
    for b in boundaries:
        ax.text(b["idx"], ymax, b["label"], rotation=90, va="top", ha="right",
                 fontsize=7, color=C_MUTED, alpha=0.9)


def _plot_series(ax, x, batch_vals, ema_vals, color, label) -> None:
    ax.plot(x, batch_vals, color=color, lw=0.8, alpha=0.30, zorder=2)
    ax.plot(x, ema_vals, color=color, lw=2.0, alpha=0.95, label=label, zorder=3)


def plot_batch_diagnostics(
    batch_records: list[dict], boundaries: list[dict], out_path: Path,
    ema_window: int = DEFAULT_EMA_WINDOW,
) -> None:
    x = [r["global_idx"] for r in batch_records]
    reset_idxs = {b["idx"] for b in boundaries}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    specs = [
        ("acc", "Accuracy", axes[0]),
        ("nll", "NLL", axes[1]),
        ("ent", "Entropy (nats)", axes[2]),
    ]
    for metric, ylabel, ax in specs:
        for name, color in (("large", C_LARGE), ("small", C_SMALL)):
            batch_vals = [r[f"{name}_{metric}"] for r in batch_records]
            ema_vals = _ema(batch_vals, ema_window, reset_idxs)
            _plot_series(ax, x, batch_vals, ema_vals, color, name)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.5, lw=0.5)
        _mark_corruption_boundaries(ax, boundaries, len(batch_records))
    axes[0].legend(loc="upper right", fontsize=8, ncols=2)
    axes[0].set_title(
        f"Per-batch diagnostics -- faint = single batch, bold = EMA "
        f"(window={ema_window}, resets at each corruption boundary, dashed lines)",
        fontsize=10,
    )
    axes[-1].set_xlabel("adaptation batch (global index across all corruptions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_proxy_diagnostics(
    proxy_rows: list[dict], out_path: Path, ema_window: int = DEFAULT_EMA_WINDOW,
) -> bool:
    if not proxy_rows:
        print("No proxy log rows to plot (calibration_mode != proxy_weighted, or no labeled "
              "batches were seen) -- skipping proxy plot.")
        return False

    x = list(range(len(proxy_rows)))
    r_l = [float(r["r_l"]) for r in proxy_rows]
    r_s = [float(r["r_s"]) for r in proxy_rows]
    w_l = [float(r["w_l"]) for r in proxy_rows]
    acc_l = [float(r["acc_l"]) for r in proxy_rows]
    acc_s = [float(r["acc_s"]) for r in proxy_rows]

    boundaries = []
    last_corr = None
    for i, r in enumerate(proxy_rows):
        if r["corruption"] != last_corr:
            boundaries.append({"idx": i, "label": r["corruption"]})
            last_corr = r["corruption"]
    reset_idxs = {b["idx"] for b in boundaries}

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    for vals, color, label in ((r_l, C_LARGE, "r_l"), (r_s, C_SMALL, "r_s")):
        ax.plot(x, vals, color=color, lw=0.8, alpha=0.30, zorder=2)
        ax.plot(x, _ema(vals, ema_window, reset_idxs), color=color, lw=2.0, alpha=0.95,
                 label=f"{label} (EMA proxy score)", zorder=3)
    ax.set_ylabel("proxy score")
    ax.grid(True, alpha=0.5, lw=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Filtered-proxy soft weighting -- proxy score (top) vs. gate weight against "
        f"ground-truth accuracy (bottom, per proxy batch) -- faint = raw, bold = EMA "
        f"(window={ema_window}, resets at each corruption boundary)",
        fontsize=10,
    )
    _mark_corruption_boundaries(ax, boundaries, len(proxy_rows))

    ax = axes[1]
    ax.plot(x, w_l, color=C_GATE, lw=0.8, alpha=0.30, zorder=2)
    ax.plot(x, _ema(w_l, ema_window, reset_idxs), color=C_GATE, lw=2.0,
             label="w_l (EMA gate weight on large model)", zorder=3)
    ax.axhline(0.5, color=C_MUTED, lw=0.8, ls=":", alpha=0.7, zorder=1)
    for vals, color, ls, label in (
        (acc_l, C_LARGE, "--", "acc_l"), (acc_s, C_SMALL, "--", "acc_s"),
    ):
        ax.plot(x, _ema(vals, ema_window, reset_idxs), color=color, lw=1.2, ls=ls, alpha=0.85,
                 label=f"{label} (EMA, this proxy batch)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("weight / accuracy [0, 1]")
    ax.set_xlabel("proxy batch (n_refreshes, global index across all corruptions)")
    ax.grid(True, alpha=0.5, lw=0.5)
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    _mark_corruption_boundaries(ax, boundaries, len(proxy_rows))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


def plot_per_corruption_proxy_vs_accuracy(
    batch_records: list[dict], proxy_rows: list[dict], out_dir: Path,
    ema_window: int = DEFAULT_EMA_WINDOW,
) -> list[Path]:
    """One figure per corruption: EMA-smoothed accuracy for large/small/
    duo (bold, right axis) with the raw proxy scores r_l/r_s (light, left
    axis, EMA overlaid bold) overlaid.

    batch_records (one row per adaptation batch) and proxy_rows (one row per
    proxy batch) can have different counts within the same corruption --
    proxy_batch_size need not equal the adaptation batch size (config's BS)
    -- so each series' x-axis is its own index normalised to [0, 1] (fraction
    of the way through the corruption's stream) rather than a shared raw
    batch index; this keeps the two curves' SHAPE comparable even when their
    resolutions differ.

    Also writes one CSV per corruption alongside its PNG (corruption_<name>.csv):
    one row per PROXY time point (the coarser, sparser series in typical
    usage -- proxy_batch_size is usually >= the adaptation batch size), with
    each model's EMA-smoothed accuracy AND entropy INTERPOLATED (np.interp)
    onto that same fractional position in the stream, so the raw proxy score
    and the concurrent accuracy/entropy sit side by side in one row --
    skipped for a corruption with no proxy rows (nothing to align).

    Returns the list of PNG paths written (empty if batch_records is empty).
    """
    if not batch_records:
        print("plot_per_corruption_proxy_vs_accuracy: no batch records -- skipping.")
        return []

    corruptions: dict[str, list[dict]] = {}
    for r in batch_records:
        corruptions.setdefault(r["corruption"], []).append(r)
    proxy_by_corruption: dict[str, list[dict]] = {}
    for r in proxy_rows:
        proxy_by_corruption.setdefault(r["corruption"], []).append(r)

    written: list[Path] = []
    for corruption, rows in corruptions.items():
        fig, ax_proxy = plt.subplots(figsize=(10, 5))
        ax_acc = ax_proxy.twinx()
        # Third axis for entropy (nats -- not on [0, 1] like accuracy or a
        # typical proxy score, so it can't share either existing axis
        # without one of the three getting visually squashed). Offset
        # outward past ax_acc's own ticks/label.
        ax_ent = ax_proxy.twinx()
        ax_ent.spines["right"].set_position(("axes", 1.16))
        ax_ent.set_frame_on(True)
        ax_ent.patch.set_visible(False)

        n = len(rows)
        x_acc = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
        acc_series = {name: _ema([r[f"{name}_acc"] for r in rows], ema_window)
                      for name in ("large", "small")}
        ent_series = {name: _ema([r[f"{name}_ent"] for r in rows], ema_window)
                      for name in ("large", "small")}
        for name, color in (("large", C_LARGE), ("small", C_SMALL)):
            ax_acc.plot(x_acc, acc_series[name], color=color, lw=2.2,
                        alpha=0.95, label=f"{name} acc (EMA)", zorder=3)
            ax_ent.plot(x_acc, ent_series[name], color=color, lw=1.4, ls="-.",
                        alpha=0.75, label=f"{name} entropy (EMA)", zorder=2)

        safe_name = corruption.replace("/", "_")
        prows = proxy_by_corruption.get(corruption, [])
        if prows:
            m = len(prows)
            x_proxy = [i / (m - 1) for i in range(m)] if m > 1 else [0.0]
            r_l = [float(r["r_l"]) for r in prows]
            r_s = [float(r["r_s"]) for r in prows]
            r_l_ema = _ema(r_l, ema_window)
            r_s_ema = _ema(r_s, ema_window)
            ax_proxy.plot(x_proxy, r_l, color=C_LARGE, lw=0.8, ls=":", alpha=0.25, zorder=1)
            ax_proxy.plot(x_proxy, r_s, color=C_SMALL, lw=0.8, ls=":", alpha=0.25, zorder=1)
            ax_proxy.plot(x_proxy, r_l_ema, color=C_LARGE, lw=1.8, ls=":",
                          alpha=0.9, label="r_l (proxy, EMA)", zorder=2)
            ax_proxy.plot(x_proxy, r_s_ema, color=C_SMALL, lw=1.8, ls=":",
                          alpha=0.9, label="r_s (proxy, EMA)", zorder=2)

            interp_acc = {name: np.interp(x_proxy, x_acc, acc_series[name]).tolist()
                          for name in ("large", "small")}
            interp_ent = {name: np.interp(x_proxy, x_acc, ent_series[name]).tolist()
                          for name in ("large", "small")}
            csv_path = out_dir / f"corruption_{safe_name}.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["corruption", "t_frac", "r_l", "r_s", "r_l_ema", "r_s_ema",
                                  "large_acc_ema", "small_acc_ema",
                                  "large_ent_ema", "small_ent_ema"])
                for i, t in enumerate(x_proxy):
                    writer.writerow([
                        corruption, t, r_l[i], r_s[i], r_l_ema[i], r_s_ema[i],
                        interp_acc["large"][i], interp_acc["small"][i],
                        interp_ent["large"][i], interp_ent["small"][i],
                    ])
            print(f"wrote {csv_path}")
        else:
            # No proxy CSV at all (calibration_mode != proxy_weighted) or no
            # proxy rows for THIS corruption specifically -- make that visible
            # on the figure itself rather than silently leaving the left axis
            # empty, which reads as a bug rather than "there's no data".
            print(f"[{corruption}] no proxy log rows -- proxy overlay skipped "
                  f"(calibration_mode != proxy_weighted, or the proxy CSV wasn't found).")
            ax_proxy.text(
                0.5, 0.5, "no proxy data\n(calibration_mode != proxy_weighted)",
                transform=ax_proxy.transAxes, ha="center", va="center",
                fontsize=9, color=C_MUTED, alpha=0.8, zorder=1,
            )

        ax_proxy.set_ylabel("proxy score", color=C_MUTED)
        ax_acc.set_ylabel("EMA accuracy")
        ax_acc.set_ylim(-0.02, 1.02)
        ax_ent.set_ylabel("EMA entropy (nats)", color=C_MUTED)
        ax_proxy.set_xlabel("fraction of corruption stream elapsed")
        ax_acc.set_title(
            f"{corruption} -- accuracy (right, solid bold EMA) vs. proxy score (left, dotted -- "
            f"faint raw / bold EMA) vs. entropy (far right, dash-dot EMA) -- window={ema_window}",
            fontsize=10,
        )
        ax_proxy.grid(True, alpha=0.4, lw=0.5)

        # Below the plot (not "lower left" etc.) -- with 6 series (2 proxy +
        # 2 accuracy + 2 entropy) any in-axes corner risks covering real
        # data, and the accuracy lines in particular sit in a narrow band
        # that a corner legend keeps landing on.
        lines_proxy, labels_proxy = ax_proxy.get_legend_handles_labels()
        lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
        lines_ent, labels_ent = ax_ent.get_legend_handles_labels()
        ax_acc.legend(lines_proxy + lines_acc + lines_ent, labels_proxy + labels_acc + labels_ent,
                      loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5, ncols=3)

        fig.subplots_adjust(right=0.80, bottom=0.24)
        out_path = out_dir / f"corruption_{safe_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")
        written.append(out_path)

    return written
