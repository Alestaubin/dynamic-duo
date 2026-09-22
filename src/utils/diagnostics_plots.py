"""Shared per-batch / per-proxy-batch diagnostics plotting.

Used by scripts/run_dynamic_duo.py, scripts/plot_run_diagnostics.py, and
scripts/run_tent.py (single-model) so they never drift into slightly
different versions of the same plot.

Large/small INPUT models only -- no duo series -- by default. The duo's
combined output is what the joint calibrator under test produces; these plots
exist to compare the input models to each other and to the proxy signal, so a
duo line here would only ever be a third, differently-scaled series crowding
the same axes without answering that question. plot_batch_diagnostics and
plot_per_corruption_proxy_vs_accuracy take an optional `series` (and
`proxy_series`) list so a single-model run can pass one entry instead of the
default large/small pair -- see scripts/run_tent.py. plot_proxy_diagnostics
stays large/small-only (its whole point is the two-model gate weight, which
has no single-model analogue); scripts/run_tent.py uses
plot_single_model_proxy_diagnostics instead.

plot_per_corruption_proxy_vs_accuracy also takes an opt-in `extra_series`: one
accuracy-only line (plus an avg-accuracy tag) per named "calibrated duo" --
used by scripts/plot_run_diagnostics.py's --compare_configs to compare this
run's own duo accuracy against a list of OTHER calibrators re-evaluated on the
SAME z_large/z_small every batch (see extra_duo_series_from_batch_records).
This is the one deliberate exception to the "no duo series" rule above: unlike
the input-model comparison every other plot here does, --compare_configs's
whole point IS comparing duo accuracy under different calibrators, so it asks
for this explicitly rather than it being a default.

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
two axes, not three: EMA-smoothed accuracy (bold SOLID) and EMA-smoothed
proxy score (bold DOTTED, raw score also shown faint) share one right axis
and one y-range (see _adaptive_ylim -- computed from the actual data, not a
fixed [0, 1], so a real but small accuracy swing doesn't get visually
flattened by a full-range axis, and the two stay directly comparable);
EMA-smoothed entropy (DASH-DOT, its own left axis -- nats, not on the same
scale as the other two) is the one series that still needs a separate axis.
This makes a collapse and a proxy dip within a single corruption stream easy
to eyeball side by side, and makes a proxy that's really just tracking
entropy (see e.g. nuclear_norm -- a near-monotone function of confidence)
rather than accuracy visible directly on the plot. Proxy and accuracy get
distinct line styles (not just distinct colors) deliberately -- once both
are EMA-bold lines in the same per-model color on the same axis, style is
what keeps them from reading as the same line at a glance. Each series also
gets its plain overall average accuracy for that corruption, tagged bottom-
right in gray.

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
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex

# MATLAB's "parula" colormap, reconstructed from its standard 64-point control
# table (matplotlib has no built-in "parula") -- dark blue/purple -> teal ->
# green -> yellow. large/small/duo each sample one fixed point from it rather
# than an arbitrary hand-picked hex, so the three stay a coherent low/mid/high
# progression along one perceptually-graded map instead of three unrelated
# categorical colors.
_PARULA_STOPS = [
    (0.2081, 0.1663, 0.5292), (0.2116, 0.1898, 0.5777), (0.2123, 0.2138, 0.6270),
    (0.2081, 0.2386, 0.6771), (0.1959, 0.2645, 0.7279), (0.1707, 0.2919, 0.7792),
    (0.1253, 0.3242, 0.8303), (0.0591, 0.3598, 0.8683), (0.0117, 0.3875, 0.8820),
    (0.0060, 0.4086, 0.8828), (0.0165, 0.4266, 0.8786), (0.0329, 0.4430, 0.8720),
    (0.0498, 0.4586, 0.8641), (0.0629, 0.4737, 0.8554), (0.0723, 0.4887, 0.8467),
    (0.0779, 0.5040, 0.8384), (0.0793, 0.5200, 0.8312), (0.0749, 0.5375, 0.8263),
    (0.0641, 0.5570, 0.8240), (0.0488, 0.5772, 0.8228), (0.0343, 0.5966, 0.8199),
    (0.0265, 0.6137, 0.8135), (0.0239, 0.6287, 0.8038), (0.0231, 0.6418, 0.7913),
    (0.0228, 0.6535, 0.7768), (0.0267, 0.6642, 0.7607), (0.0384, 0.6743, 0.7436),
    (0.0590, 0.6838, 0.7254), (0.0843, 0.6928, 0.7062), (0.1133, 0.7015, 0.6859),
    (0.1453, 0.7098, 0.6646), (0.1801, 0.7177, 0.6424), (0.2178, 0.7250, 0.6193),
    (0.2586, 0.7317, 0.5954), (0.3022, 0.7376, 0.5712), (0.3482, 0.7424, 0.5473),
    (0.3953, 0.7459, 0.5244), (0.4420, 0.7481, 0.5033), (0.4871, 0.7491, 0.4840),
    (0.5300, 0.7491, 0.4661), (0.5709, 0.7485, 0.4494), (0.6099, 0.7473, 0.4337),
    (0.6473, 0.7456, 0.4188), (0.6834, 0.7435, 0.4044), (0.7184, 0.7411, 0.3905),
    (0.7525, 0.7384, 0.3768), (0.7858, 0.7356, 0.3633), (0.8185, 0.7327, 0.3498),
    (0.8507, 0.7299, 0.3360), (0.8824, 0.7274, 0.3217), (0.9139, 0.7258, 0.3063),
    (0.9449, 0.7261, 0.2886), (0.9739, 0.7314, 0.2666), (0.9938, 0.7455, 0.2403),
    (0.9990, 0.7653, 0.2164), (0.9955, 0.7861, 0.1967), (0.9880, 0.8066, 0.1794),
    (0.9789, 0.8271, 0.1633), (0.9697, 0.8481, 0.1475), (0.9626, 0.8705, 0.1309),
    (0.9589, 0.8949, 0.1132), (0.9598, 0.9218, 0.0948), (0.9661, 0.9514, 0.0755),
    (0.9763, 0.9831, 0.0538),
]
_PARULA = LinearSegmentedColormap.from_list("parula", _PARULA_STOPS, N=256)

# large = cool/dark end, small = mid (teal-green), duo/gate = warm/bright end
# -- also, deliberately, the plot's most OPAQUE line everywhere it appears
# (see every alpha= below): duo is the thing every one of these plots exists
# to judge, so it should read as the foreground signal, not one of three
# equally-weighted lines.
C_LARGE = to_hex(_PARULA(0.05))
C_SMALL = to_hex(_PARULA(0.50))
C_GATE = to_hex(_PARULA(0.85))  # gold, not pure yellow -- stays legible on the off-white C_SURFACE
# Frozen (non-adapting) baseline overlay -- see scripts/run_tent.py's
# --track_frozen. Deliberately a neutral gray rather than a categorical color:
# it's a reference baseline, not a model identity that needs to stand out.
C_FROZEN = "#6b7280"
C_INK = "#0b0b0b"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"

# Categorical slots 3/4/5/6/8 of the dataviz skill's validated palette
# (references/palette.md) -- slots 1/2/7 are already C_LARGE/C_SMALL/C_GATE
# above. Used by plot_per_corruption_proxy_vs_accuracy's `extra_series` for
# an arbitrary-length list of "calibrated duo" comparison lines (see
# scripts/plot_run_diagnostics.py's --compare_configs), assigned in this
# fixed order rather than cycled, matching the compare-configs file's own
# ordering. Past 5 entries colors repeat -- a comparison with more
# calibrators than that on one figure is already pushing legibility.
EXTRA_SERIES_PALETTE = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#e34948"]

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


def _adaptive_ylim(
    *value_lists: list[float], pad_frac: float = 0.08, min_span: float = 0.05,
) -> tuple[float, float]:
    """Shared y-limits for two or more series meant to be visually compared
    on the same scale (e.g. proxy score vs. true accuracy in
    plot_per_corruption_proxy_vs_accuracy) -- computed from the actual data
    instead of a fixed [0, 1]. A fixed full-range axis makes a real but
    small accuracy swing (a few points) read as a flat line; zooming to the
    data's own span makes it visible while still sharing one scale across
    both axes so the comparison stays apples-to-apples.

    min_span floors the displayed range (e.g. a near-constant corruption
    doesn't collapse to an unreadably tight band around noise); pad_frac
    adds breathing room around the (possibly floored) span so the plotted
    lines never touch the frame.
    """
    vals = [v for lst in value_lists for v in lst]
    if not vals:
        return (-0.02, 1.02)
    lo, hi = min(vals), max(vals)
    if hi - lo < min_span:
        mid = (lo + hi) / 2
        lo, hi = mid - min_span / 2, mid + min_span / 2
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def _mark_corruption_boundaries(ax, boundaries: list[dict], n: int) -> None:
    for b in boundaries:
        if b["idx"] == 0:
            continue
        ax.axvline(b["idx"], color=C_MUTED, lw=0.8, ls="--", alpha=0.6, zorder=1)
    ymin, ymax = ax.get_ylim()
    for b in boundaries:
        ax.text(b["idx"], ymax, b["label"], rotation=90, va="top", ha="right",
                 fontsize=7, color=C_MUTED, alpha=0.9)


def _plot_series(ax, x, batch_vals, ema_vals, color, label, lw_scale: float = 1.0) -> None:
    ax.plot(x, batch_vals, color=color, lw=0.8 * lw_scale, alpha=0.30, zorder=2)
    ax.plot(x, ema_vals, color=color, lw=2.0 * lw_scale, alpha=0.95, label=label, zorder=3)


# Default series for a duo run: (row-key prefix, color, legend label,
# lw_scale). lw_scale multiplies every line width this series is drawn with,
# relative to that plot's own base widths -- 1.0 is the normal bold series;
# a baseline overlay (e.g. scripts/run_tent.py's --track_frozen) passes a
# smaller value for a visibly thinner line without hardcoding an absolute
# width that would need to track each plot function's own base widths. A
# single-model run (see scripts/run_tent.py) passes a one- or two-entry list
# instead -- everything below just loops over however many series it gets.
_DEFAULT_SERIES = [("large", C_LARGE, "large", 1.0), ("small", C_SMALL, "small", 1.0)]


def plot_batch_diagnostics(
    batch_records: list[dict], boundaries: list[dict], out_path: Path,
    ema_window: int = DEFAULT_EMA_WINDOW,
    series: list[tuple[str, str, str, float]] | None = None,
) -> None:
    series = series or _DEFAULT_SERIES
    x = [r["global_idx"] for r in batch_records]
    reset_idxs = {b["idx"] for b in boundaries}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    specs = [
        ("acc", "Accuracy", axes[0]),
        ("nll", "NLL", axes[1]),
        ("ent", "Entropy (nats)", axes[2]),
    ]
    for metric, ylabel, ax in specs:
        for key, color, label, lw_scale in series:
            batch_vals = [r[f"{key}_{metric}"] for r in batch_records]
            ema_vals = _ema(batch_vals, ema_window, reset_idxs)
            _plot_series(ax, x, batch_vals, ema_vals, color, label, lw_scale)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.5, lw=0.5)
        _mark_corruption_boundaries(ax, boundaries, len(batch_records))
    axes[0].legend(loc="upper right", fontsize=8, ncols=2)
    axes[0].set_title(
        f"Per-batch diagnostics"
        # f" -- faint = single batch, bold = EMA "
        # f"(window={ema_window}, resets at each corruption boundary, dashed lines)",
        # fontsize=10,
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
        ax.plot(x, vals, color=color, lw=0.8, alpha=0.18, zorder=2)
        ax.plot(x, _ema(vals, ema_window, reset_idxs), color=color, lw=2.0, alpha=0.6,
                 label=f"{label} (proxy score)", zorder=3)
    ax.set_ylabel("proxy score")
    ax.grid(True, alpha=0.5, lw=0.5)
    ax.legend(loc="upper right", fontsize=8)
    # ax.set_title(
    #     f"Filtered-proxy soft weighting -- proxy score (top) vs. gate weight against "
    #     f"ground-truth accuracy (bottom, per proxy batch) -- faint = raw, bold = EMA "
    #     f"(window={ema_window}, resets at each corruption boundary)",
    #     fontsize=10,
    # )
    _mark_corruption_boundaries(ax, boundaries, len(proxy_rows))

    ax = axes[1]
    ax.plot(x, w_l, color=C_GATE, lw=0.8, alpha=0.35, zorder=2)
    ax.plot(x, _ema(w_l, ema_window, reset_idxs), color=C_GATE, lw=2.2, alpha=1.0,
             label="w_l (gate weight on large model)", zorder=3)
    ax.axhline(0.5, color=C_MUTED, lw=0.8, ls=":", alpha=0.7, zorder=1)
    for vals, color, ls, label in (
        (acc_l, C_LARGE, "--", "acc_l"), (acc_s, C_SMALL, "--", "acc_s"),
    ):
        ax.plot(x, _ema(vals, ema_window, reset_idxs), color=color, lw=1.2, ls=ls, alpha=0.5,
                 label=f"{label} (this proxy batch)")
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


def plot_single_model_proxy_diagnostics(
    proxy_rows: list[dict], out_path: Path, ema_window: int = DEFAULT_EMA_WINDOW,
) -> bool:
    """Single-model counterpart of plot_proxy_diagnostics (see scripts/run_tent.py)
    -- there is no second model to gate against, so this has no w_l/selection
    panel. Instead: raw proxy score r (top) and calibrated predicted accuracy
    a vs. the model's ACTUAL accuracy (bottom, both on [0, 1]) -- the
    single-model version of "does the proxy track this model's accuracy",
    plotted per proxy batch.
    """
    if not proxy_rows:
        print("No proxy log rows to plot -- skipping proxy plot.")
        return False

    x = list(range(len(proxy_rows)))
    r = [float(row["r"]) for row in proxy_rows]
    a = [float(row["a"]) for row in proxy_rows]
    acc = [float(row["acc"]) for row in proxy_rows]

    boundaries = []
    last_corr = None
    for i, row in enumerate(proxy_rows):
        if row["corruption"] != last_corr:
            boundaries.append({"idx": i, "label": row["corruption"]})
            last_corr = row["corruption"]
    reset_idxs = {b["idx"] for b in boundaries}

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    ax.plot(x, r, color=C_LARGE, lw=0.8, alpha=0.30, zorder=2)
    ax.plot(x, _ema(r, ema_window, reset_idxs), color=C_LARGE, lw=2.0, alpha=0.95,
             label="r (raw proxy score)", zorder=3)
    ax.set_ylabel("proxy score")
    ax.grid(True, alpha=0.5, lw=0.5)
    ax.legend(loc="upper right", fontsize=8)
    _mark_corruption_boundaries(ax, boundaries, len(proxy_rows))

    ax = axes[1]
    ax.plot(x, _ema(a, ema_window, reset_idxs), color=C_GATE, lw=2.0, ls="-",
             alpha=0.95, label="a (calibrated predicted acc)", zorder=3)
    ax.plot(x, _ema(acc, ema_window, reset_idxs), color=C_LARGE, lw=1.6, ls="--",
             alpha=0.9, label="acc (actual accuracy)", zorder=3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("accuracy [0, 1]")
    ax.set_xlabel("proxy batch (n_refreshes, global index across all corruptions)")
    ax.grid(True, alpha=0.5, lw=0.5)
    ax.legend(loc="upper right", fontsize=8)
    _mark_corruption_boundaries(ax, boundaries, len(proxy_rows))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return True


# Default proxy overlay for a duo run: (proxy-CSV column, color, legend
# label). A single-model run (see scripts/run_tent.py) passes a single-entry
# list (its proxy log has one column, "r", not "r_l"/"r_s").
_DEFAULT_PROXY_SERIES = [("r_l", C_LARGE, "r_l"), ("r_s", C_SMALL, "r_s")]

_CMP_ACC_RE = re.compile(r"^cmp_(.+)_acc$")


def extra_duo_series_from_batch_records(
    batch_records: list[dict], main_label: str | None = None,
    calib_mode_by_name: dict[str, str] | None = None,
) -> list[tuple[str, str, str, str | None]]:
    """Reconstruct plot_per_corruption_proxy_vs_accuracy's `extra_series` from
    a REPLAYED batch_diagnostics.csv (see scripts/plot_run_diagnostics.py's
    --csv_dir) -- cmp_<name>_acc columns (one per --compare_configs entry,
    written by that script's _on_batch) plus, if main_label is given, this
    run's own "duo_acc" (present in every batch_diagnostics.csv regardless of
    --compare_configs, but only surfaced as a series here since a caller has
    explicitly asked for the duo-comparison view -- see extra_series' own
    docstring for why it's opt-in everywhere else). Column order (not an
    alphabetical resort) drives color order via EXTRA_SERIES_PALETTE, so
    replotting from disk assigns the same colors the live run used.

    calib_mode_by_name (optional): maps each entry's label/name to its
    calibration_mode ("proxy_weighted", "fixed_ts", ...), threaded through as
    each tuple's 4th element -- see extra_series' own docstring for how
    plot_per_corruption_proxy_vs_accuracy uses it. A caller with no run_cfg
    dicts on hand (e.g. a plain --csv_dir replot with no --compare_configs)
    can omit this; every entry's mode is then None, which draws as "not
    proxy_weighted" (thin/faint) -- correct as a default even when wrong,
    since there's no way to recover the original calibration_mode from
    batch_diagnostics.csv's column names alone.
    """
    if not batch_records:
        return []
    calib_mode_by_name = calib_mode_by_name or {}
    out: list[tuple[str, str, str, str | None]] = []
    if main_label is not None and "duo_acc" in batch_records[0]:
        out.append(("duo_acc", C_GATE, main_label, calib_mode_by_name.get(main_label)))
    names = [m.group(1) for k in batch_records[0] if (m := _CMP_ACC_RE.match(k))]
    for i, name in enumerate(names):
        out.append((f"cmp_{name}_acc", EXTRA_SERIES_PALETTE[i % len(EXTRA_SERIES_PALETTE)], name,
                    calib_mode_by_name.get(name)))
    return out


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters in a plain string -- corruption names
    (e.g. 'gaussian_noise') and calib_config/compare_configs names (e.g.
    'nuclear_norm_identity_pbs128') routinely contain underscores, which
    LaTeX would otherwise read as a subscript outside math mode."""
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in s)


# Hendrycks & Dietterich's standard ImageNet-C grouping/abbreviations, plus
# this repo's own "held-out" extras (used as CALIBRATOR.CORRUPTIONS in most
# configs, so they show up as EVAL corruptions in a _heldout config instead).
# base corruption name -> (family, column abbreviation).
_CORRUPTION_FAMILIES: dict[str, tuple[str, str]] = {
    "gaussian_noise": ("Noise", "Gauss."),
    "shot_noise": ("Noise", "Shot"),
    "impulse_noise": ("Noise", "Impulse"),
    "defocus_blur": ("Blur", "Defocus"),
    "glass_blur": ("Blur", "Glass"),
    "motion_blur": ("Blur", "Motion"),
    "zoom_blur": ("Blur", "Zoom"),
    "snow": ("Weather", "Snow"),
    "frost": ("Weather", "Frost"),
    "fog": ("Weather", "Fog"),
    "brightness": ("Weather", "Bright."),
    "contrast": ("Digital", "Contrast"),
    "elastic_transform": ("Digital", "Elastic"),
    "pixelate": ("Digital", "Pixel."),
    "jpeg_compression": ("Digital", "JPEG"),
    "gaussian_blur": ("Extra", "G.Blur"),
    "saturate": ("Extra", "Saturate"),
    "spatter": ("Extra", "Spatter"),
    "speckle_noise": ("Extra", "Speckle"),
}
_FAMILY_ORDER = ["Noise", "Blur", "Weather", "Digital", "Extra", "Other"]


def write_accuracy_latex_table(
    batch_records: list[dict], out_path: Path, duo_label: str = "duo",
) -> bool:
    """Write a standalone LaTeX table* (out_path, e.g. accuracy_table.tex) of
    per-corruption accuracy for every method this run tracked, in the
    ImageNet-C-paper style: corruption columns grouped into Noise/Blur/
    Weather/Digital families (\\multicolumn header + \\cmidrule, standard
    Hendrycks & Dietterich abbreviations -- see _CORRUPTION_FAMILIES; a
    corruption this repo uses as a held-out CALIBRATOR set falls under
    "Extra", anything else under "Other"), a two-level row header
    (\\multirow "Models" for large/small, "Calibrators" for the main duo
    ("duo_acc", included regardless of --hide_duo_line -- that flag only
    declutters the PNG legend, not this table) and every --compare_configs
    alternative), and a trailing Avg. column. Only families/corruptions
    actually present in batch_records get a column -- e.g. a config missing
    zoom_blur just gets a 3-wide Blur group instead of 4.

    Values are percentages (matching the ImageNet-C literature convention),
    each the SAMPLE-WEIGHTED accuracy over that corruption's adaptation
    batches (sum(acc_i * n_i) / sum(n_i), using each row's own "n") -- the
    same arithmetic evaluate_dynamic_duo uses internally (via the
    concatenated probs), just reconstructed from batch_records instead, so a
    corruption whose last batch is smaller than the rest is still weighted
    correctly. The Avg. column/row-tail is the plain mean over corruption
    columns (matching evaluate_dynamic_duo's own "average" row convention,
    not a further sample-weighted global figure). Works identically for a
    live run or a --csv_dir replot/replay -- batch_records always carries
    "n"/"*_acc"/"cmp_*_acc" regardless of which path produced it. If the same
    base corruption appears at more than one severity in this run, its
    column header disambiguates with "(sN)" instead of colliding.

    Requires \\usepackage{graphicx}, \\usepackage{booktabs}, and
    \\usepackage{multirow} wherever this file is \\input{}'d. The caption
    only states the corruption count/severities and sample count actually
    present in this run -- add any paper-specific detail (e.g. exactly how a
    fixed_ts baseline included in --compare_configs was fitted) by hand.

    Returns False (writes nothing) if batch_records is empty.
    """
    if not batch_records:
        print("write_accuracy_latex_table: no batch records -- skipping.")
        return False

    corr_order: list[str] = []
    rows_by_corruption: dict[str, list[dict]] = {}
    for r in batch_records:
        c = r["corruption"]
        if c not in rows_by_corruption:
            corr_order.append(c)
            rows_by_corruption[c] = []
        rows_by_corruption[c].append(r)

    def _split(c: str) -> tuple[str, str | None]:
        base, _, sev = c.partition("/")
        return (base, sev) if sev else (base, None)

    base_names = [_split(c)[0] for c in corr_order]
    base_counts = {b: base_names.count(b) for b in set(base_names)}

    # (family, column label, corruption key), family order fixed
    # (_FAMILY_ORDER), corruptions within a family kept in their
    # first-seen/EVAL.CORRUPTIONS order rather than resorted alphabetically.
    tagged = []
    for c in corr_order:
        base, sev = _split(c)
        family, abbrev = _CORRUPTION_FAMILIES.get(base, ("Other", base.replace("_", " ").title()))
        label = abbrev if base_counts[base] <= 1 or sev is None else f"{abbrev} ({sev})"
        tagged.append((family, label, c))
    families_present = sorted(
        {f for f, _, _ in tagged},
        key=lambda f: _FAMILY_ORDER.index(f) if f in _FAMILY_ORDER else len(_FAMILY_ORDER),
    )
    ordered_cols = [t for fam in families_present for t in tagged if t[0] == fam]
    corruptions = [c for _, _, c in ordered_cols]

    methods: list[tuple[str, str]] = [("large_acc", "Large"), ("small_acc", "Small")]
    duo_methods: list[tuple[str, str]] = []
    if "duo_acc" in batch_records[0]:
        duo_methods.append(("duo_acc", duo_label))
    cmp_names = [m.group(1) for k in batch_records[0] if (m := _CMP_ACC_RE.match(k))]
    duo_methods += [(f"cmp_{name}_acc", name) for name in cmp_names]

    def _weighted_acc(rows: list[dict], key: str) -> float:
        total_n = sum(r["n"] for r in rows)
        return sum(r[key] * r["n"] for r in rows) / total_n if total_n > 0 else float("nan")

    per_corr_acc: dict[str, dict[str, float]] = {
        c: {key: _weighted_acc(rows, key) for key, _ in methods + duo_methods}
        for c, rows in rows_by_corruption.items()
    }

    def _fmt(v: float) -> str:
        return f"{100 * v:.1f}" if v == v else "--"  # v == v is False only for NaN

    def _row_line(group_cell: str, label: str, key: str) -> str:
        vals = [per_corr_acc[c][key] for c in corruptions]
        finite = [v for v in vals if v == v]
        avg = sum(finite) / len(finite) if finite else float("nan")
        cells = " & ".join(_fmt(v) for v in vals)
        return f"        {group_cell}{_latex_escape(label)} & {cells} & {_fmt(avg)} \\\\"

    body_lines: list[str] = []
    for group_label, group_methods in (("Models", methods), ("Calibrators", duo_methods)):
        if not group_methods:
            continue
        for i, (key, label) in enumerate(group_methods):
            group_cell = f"\\multirow{{{len(group_methods)}}}{{*}}{{\\textit{{{group_label}}}}} & " \
                if i == 0 else "& "
            body_lines.append(_row_line(group_cell, label, key))
        body_lines.append("        \\midrule")
    if body_lines and body_lines[-1].strip() == "\\midrule":
        body_lines.pop()

    col_groups = ["c" * sum(1 for f, _, _ in ordered_cols if f == fam) for fam in families_present]
    col_spec = "ll | " + " | ".join(col_groups) + " | c"

    cmidrules, col_cursor = [], 3
    family_header_cells = []
    for fam in families_present:
        n = sum(1 for f, _, _ in ordered_cols if f == fam)
        family_header_cells.append(f"\\multicolumn{{{n}}}{{c}}{{{fam}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_cursor}-{col_cursor + n - 1}}}")
        col_cursor += n

    n_samples = sum(r["n"] for r in rows_by_corruption[corruptions[0]]) if corruptions else 0
    severities = sorted({_split(c)[1] for c in corruptions if _split(c)[1] is not None})
    sev_phrase = (f"severity {severities[0].lstrip('s')}" if len(severities) == 1
                  else f"severities {', '.join(s.lstrip('s') for s in severities)}" if severities
                  else "unlabeled severity")

    lines = [
        "% Auto-generated by plot_run_diagnostics.py -- see",
        "% src/utils/diagnostics_plots.py's write_accuracy_latex_table.",
        "% Requires \\usepackage{graphicx}, \\usepackage{booktabs}, \\usepackage{multirow}.",
        "% \\newpage matches this table's reference style (a full-width table*",
        "% in a two-column document) -- delete it if that doesn't apply here.",
        "\\newpage",
        "\\begin{table*}[t]",
        "    \\centering",
        f"    \\caption{{Top-1 accuracy (\\%) at {sev_phrase} across ImageNet-C "
        f"corruptions ({n_samples} samples per corruption), grouped by family.}}",
        "    \\label{tab:accuracy}",
        "    \\resizebox{\\textwidth}{!}{%",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\toprule",
        "        & & " + " & ".join(family_header_cells) + " & \\\\",
        "        " + "".join(cmidrules),
        "        Method & Models & " + " & ".join(_latex_escape(l) for _, l, _ in ordered_cols) + " & Avg. \\\\",
        "        \\midrule",
    ] + body_lines + [
        "        \\bottomrule",
        "    \\end{tabular}%",
        "    }",
        "\\end{table*}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    return True


def _cumulative_samples(rows: list[dict]) -> list[float] | None:
    """Cumulative sample count AFTER each row (i.e. x[i] = how many samples
    had been processed once row i completed), from each row's own "n" field
    -- None if any row is missing "n" (older batch_diagnostics.csv/proxy_log
    written before "n" was added to every row, or one of the CSV round-trip
    loaders that doesn't happen to carry it through), so callers can fall
    back to a plain fractional-index x-axis instead of a wrong one."""
    if not rows or any("n" not in r for r in rows):
        return None
    out, cum = [], 0.0
    for r in rows:
        cum += float(r["n"])
        out.append(cum)
    return out


def plot_proxy_pbs_comparison(
    batch_records: list[dict], proxy_rows_by_pbs: dict[int, list[dict]], out_dir: Path,
    ema_window: int = DEFAULT_EMA_WINDOW, model_key: str = "model",
) -> list[Path]:
    """One figure per corruption: this model's ACTUAL accuracy (bold, dark,
    EMA) vs. the CALIBRATED predicted accuracy "a" from every tracked
    proxy_batch_size (one line per pbs, color-ramped light -> dark for
    small -> large pbs), sharing ONE [0, 1] y-axis -- both are already on
    the accuracy scale, so no twin axis is needed like
    plot_per_corruption_proxy_vs_accuracy -- and a shared x-axis of
    cumulative samples processed within that corruption (see
    _cumulative_samples). This is the direct "how reactive is each pbs to a
    real accuracy swing" comparison: a small pbs reacts fast but noisy, a
    large pbs is smooth but lags -- eyeball which one tracks the ACTUAL
    accuracy line most closely.

    Written by scripts/run_tent.py's --proxy_batch_sizes when more than one
    pbs is tracked from the same TENT run (a single pbs has nothing to
    compare against and keeps using plot_per_corruption_proxy_vs_accuracy
    instead). Returns the list of PNG paths written.
    """
    if not batch_records or not proxy_rows_by_pbs:
        print("plot_proxy_pbs_comparison: nothing to plot.")
        return []

    corruptions: dict[str, list[dict]] = {}
    for r in batch_records:
        corruptions.setdefault(r["corruption"], []).append(r)

    pbs_sorted = sorted(proxy_rows_by_pbs)
    cmap = plt.get_cmap("viridis")
    color_by_pbs = {
        pbs: cmap(t) for pbs, t in zip(pbs_sorted, np.linspace(0.15, 0.9, len(pbs_sorted)))
    }

    proxy_by_pbs_corruption: dict[int, dict[str, list[dict]]] = {}
    for pbs, rows in proxy_rows_by_pbs.items():
        by_corr: dict[str, list[dict]] = {}
        for r in rows:
            by_corr.setdefault(r["corruption"], []).append(r)
        proxy_by_pbs_corruption[pbs] = by_corr

    written: list[Path] = []
    for corruption, rows in corruptions.items():
        n = len(rows)
        x_acc = _cumulative_samples(rows)
        if x_acc is None:
            x_acc = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
        acc_vals = [r[f"{model_key}_acc"] for r in rows]
        acc_ema = _ema(acc_vals, ema_window)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_acc, acc_vals, color=C_INK, lw=0.6, alpha=0.20, zorder=1)
        ax.plot(x_acc, acc_ema, color=C_INK, lw=2.4, alpha=0.95,
                label="actual accuracy", zorder=5)

        for pbs in pbs_sorted:
            prows = proxy_by_pbs_corruption.get(pbs, {}).get(corruption, [])
            if not prows:
                continue
            m = len(prows)
            x_p = _cumulative_samples(prows)
            if x_p is None:
                x_p = [i / (m - 1) for i in range(m)] if m > 1 else [0.0]
            a_vals = [float(r["a"]) for r in prows]
            a_ema = _ema(a_vals, ema_window)
            color = color_by_pbs[pbs]
            ax.plot(x_p, a_vals, color=color, lw=0.6, alpha=0.15, zorder=2)
            ax.plot(x_p, a_ema, color=color, lw=1.8, alpha=0.9,
                    label=f"pbs={pbs} (a)", zorder=3)

        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("accuracy / calibrated proxy score [0, 1]")
        ax.set_xlabel("samples processed")
        ax.set_title(f"Corruption {corruption} -- proxy_batch_size reactivity")
        ax.grid(True, alpha=0.4, lw=0.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7.5,
                  ncols=min(4, len(pbs_sorted) + 1))
        fig.subplots_adjust(bottom=0.3)

        safe_name = corruption.replace("/", "_")
        out_path = out_dir / f"corruption_{safe_name}_pbs_comparison.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")
        written.append(out_path)

    return written


def plot_per_corruption_gate_weight(
    proxy_rows_by_method: dict[str, list[dict]], out_dir: Path, ema_window: int = DEFAULT_EMA_WINDOW,
) -> list[Path]:
    """One figure per corruption: EVERY proxy_weighted method's gate weight
    w_l overlaid on the SAME axes -- one line per method, distinctly colored
    and labeled -- against a single true-accuracy reference (large/small,
    dashed, thin). The direct "is each method's gate putting weight where
    the accuracy actually is, and how do they compare to EACH OTHER" check,
    zoomed to one corruption instead of plot_proxy_diagnostics' whole-run,
    all-corruptions-concatenated, single-method view.

    proxy_rows_by_method: {method_name: proxy_rows}, one entry per
    proxy_weighted source -- the main duo plus any --compare_configs entry
    that's itself proxy_weighted (see scripts/plot_run_diagnostics.py's
    _load_compare_proxy_logs; entries using another calibration_mode simply
    have no proxy log and never reach here). Insertion order matters: the
    first entry's color is C_GATE (matching the "main duo" convention used
    elsewhere in this module) and it supplies the true-accuracy reference if
    it has rows for a given corruption, falling through to the next entry
    otherwise; the rest cycle through EXTRA_SERIES_PALETTE. Entries with no
    rows at all are dropped up front.

    Reads proxy log rows only (not batch_records): w_l/acc_l/acc_s are
    computed once per PROXY batch (see JointProxyWeighted._flush_bucket and
    the proxy_batch_size gotcha in CLAUDE.md) -- the natural granularity for
    "did the gate move when it should have". Different methods can use
    different proxy_batch_size, so their x-values (cumulative samples
    processed) land at different points along the SAME corruption stream --
    matplotlib overlays them correctly regardless, since each line supplies
    its own x array rather than assuming a shared one.

    Deliberately drops the raw per-model proxy score (r_l/r_s) overlay the
    single-method version of this plot used to have: different methods can
    use different proxy_kind, whose raw scores live on incomparable scales
    (a nuclear_norm score and an ac_mc score mean different things), so
    overlaying them together would compare apples to oranges. See
    plot_proxy_diagnostics for a single method's own raw-score-vs-gate view.

    No-op (returns []) if proxy_rows_by_method is empty or every entry is.
    """
    proxy_rows_by_method = {name: rows for name, rows in proxy_rows_by_method.items() if rows}
    if not proxy_rows_by_method:
        print("plot_per_corruption_gate_weight: no proxy_weighted methods with proxy log "
              "rows -- skipping.")
        return []

    method_names = list(proxy_rows_by_method.keys())
    color_by_method = {
        name: (C_GATE if i == 0 else EXTRA_SERIES_PALETTE[(i - 1) % len(EXTRA_SERIES_PALETTE)])
        for i, name in enumerate(method_names)
    }

    # corruption -> {method_name: that method's rows for this corruption}
    corr_order: list[str] = []
    by_corruption: dict[str, dict[str, list[dict]]] = {}
    for name in method_names:
        for r in proxy_rows_by_method[name]:
            c = r["corruption"]
            if c not in by_corruption:
                corr_order.append(c)
                by_corruption[c] = {}
            by_corruption[c].setdefault(name, []).append(r)

    def _x_axis(rows: list[dict]) -> tuple[list[float], bool]:
        n = len(rows)
        x = _cumulative_samples(rows)
        if x is not None:
            return x, True
        return ([i / (n - 1) for i in range(n)] if n > 1 else [0.0]), False

    written: list[Path] = []
    for corruption in corr_order:
        methods_here = by_corruption[corruption]
        fig, ax = plt.subplots(figsize=(6, 5))

        # True-accuracy reference: from the first method (in the dict's own
        # order) that actually has rows for this corruption -- the ground
        # truth doesn't depend on which method is gating, only its exact
        # proxy-batch chunking might differ slightly across methods with
        # different proxy_batch_size.
        ref_name = next((n for n in method_names if n in methods_here), None)
        x_is_samples = True
        if ref_name is not None:
            ref_rows = methods_here[ref_name]
            x_ref, x_is_samples = _x_axis(ref_rows)
            for key, color, lbl in (("acc_l", C_LARGE, "large"), ("acc_s", C_SMALL, "small")):
                vals = [float(r[key]) for r in ref_rows]
                ax.plot(x_ref, vals, color=color, lw=0.8, alpha=0.15, zorder=1)
                ax.plot(x_ref, _ema(vals, ema_window), color=color, lw=1.4, ls="--", alpha=0.5,
                        label=f"{lbl} acc (this proxy batch)", zorder=2)

        for name in method_names:
            rows = methods_here.get(name)
            if not rows:
                continue
            w_l = [float(r["w_l"]) for r in rows]
            x, _ = _x_axis(rows)
            color = color_by_method[name]
            ax.plot(x, w_l, color=color, lw=0.8, alpha=0.25, zorder=2)
            ax.plot(x, _ema(w_l, ema_window), color=color, lw=2.0, alpha=0.9,
                    label=f"w_l ({name})", zorder=3)

        ax.axhline(0.5, color=C_MUTED, lw=0.8, ls=":", alpha=0.7, zorder=1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("gate weight / accuracy [0, 1]")
        ax.set_xlabel("samples processed" if x_is_samples else "fraction of corruption stream elapsed")
        ax.set_title(f"Corruption {corruption} -- gate weight by method vs. true accuracy")
        ax.grid(True, alpha=0.4, lw=0.5)
        # Fixed at 2 COLUMNS (not a row target) -- this figure is narrow on
        # purpose (to fit a constrained layout, e.g. a paper column), and
        # method names (e.g. "nuclear_norm_identity_pbs128") are long, so
        # width is the scarce resource here, not height: more methods just
        # grow the legend downward instead of widening it.
        n_legend_items = len(method_names) + 2
        ncols = 2
        n_rows = -(-n_legend_items // ncols)  # ceil(n_legend_items / ncols)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14 - 0.07 * n_rows),
                  fontsize=7, ncols=ncols)

        fig.subplots_adjust(bottom=0.2 + 0.07 * n_rows, left=0.12, right=0.95, top=0.92)
        safe_name = corruption.replace("/", "_")
        out_path = out_dir / f"gate_weight_{safe_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")
        written.append(out_path)

    return written


def plot_per_corruption_proxy_vs_accuracy(
    batch_records: list[dict], proxy_rows: list[dict], out_dir: Path,
    ema_window: int = DEFAULT_EMA_WINDOW,
    series: list[tuple[str, str, str, float]] | None = None,
    proxy_series: list[tuple[str, str, str]] | None = None,
    extra_series: list[tuple[str, str, str, str | None]] | None = None,
    show_gate_weight: bool = False,
) -> list[Path]:
    """One figure per corruption, two axes: EMA-smoothed accuracy for
    large/small/duo (bold) and the raw proxy scores r_l/r_s (light, EMA
    overlaid bold) share one right axis and one adaptive y-range (see
    _adaptive_ylim, instead of a fixed [0, 1]) so the two stay on a directly
    comparable scale and small accuracy swings stay visible. EMA-smoothed
    entropy gets the left axis (nats, not on the same scale as the other
    two). Each series' plain overall average accuracy is tagged bottom-right
    in gray.

    show_gate_weight (opt-in, off by default -- see
    scripts/plot_run_diagnostics.py's --show_gate_weight): overlays w_l, the
    gate weight the MAIN duo's own JointProxyWeighted calibrator actually
    assigned the large model (proxy_rows, same source as
    plot_per_corruption_gate_weight, NOT any --compare_configs alternative),
    as a faint gray line on its OWN third y-axis -- unlike everything else on
    this figure, w_l isn't an accuracy or a proxy score, so it doesn't belong
    sharing either existing axis' scale. Silently skipped (no third axis
    drawn) for a corruption with no proxy rows or a non-proxy_weighted run.

    extra_series, if given, is a list of (row_key, color, label,
    calibration_mode) -- one accuracy-only line (no entropy) per named
    "calibrated duo", reading batch_records[row_key] directly (unlike
    `series`, whose entries are a prefix combined with "_acc"/"_ent") -- see
    scripts/plot_run_diagnostics.py's --compare_configs and
    extra_duo_series_from_batch_records. Each also gets an avg-accuracy tag
    appended to the same bottom-right text block as `series`, and is folded
    into the shared adaptive y-range and the per-corruption CSV export
    alongside `series`.

    calibration_mode (the tuple's 4th element, possibly None if unknown --
    see extra_duo_series_from_batch_records) drives that line's own width/
    alpha: "proxy_weighted" draws bold and fully opaque (lw=2.0, alpha=1.0)
    -- the method actually under test -- anything else (a fixed_ts/coca_ts/
    oracle_ts/optimal_w_oracle baseline, or an unknown mode) draws thin and
    faint (lw=1.2, alpha=0.5) as a reference line, not the main subject.

    batch_records (one row per adaptation batch) and proxy_rows (one row per
    proxy batch) can have different counts within the same corruption --
    proxy_batch_size need not equal the adaptation batch size (config's BS)
    -- so each series' x-axis is its own cumulative sample count (see
    _cumulative_samples, from each row's own "n" field) rather than a shared
    raw batch/proxy-batch index; this keeps the two curves aligned by actual
    position in the stream even when their resolutions differ. Falls back to
    a [0, 1] fractional-index x-axis (the old behavior) if batch_records is
    missing "n" (e.g. an older batch_diagnostics.csv re-plotted via
    --csv_dir, from before this field existed).

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
    series = series or _DEFAULT_SERIES
    proxy_series = proxy_series if proxy_series is not None else _DEFAULT_PROXY_SERIES
    extra_series = extra_series or []

    corruptions: dict[str, list[dict]] = {}
    for r in batch_records:
        corruptions.setdefault(r["corruption"], []).append(r)
    proxy_by_corruption: dict[str, list[dict]] = {}
    for r in proxy_rows:
        proxy_by_corruption.setdefault(r["corruption"], []).append(r)

    written: list[Path] = []
    for corruption, rows in corruptions.items():
        # Two DATA axes, not three: proxy score and accuracy now share one
        # scale (see _adaptive_ylim below), so they share one axis (right)
        # instead of each getting its own. Entropy (nats -- not on the same
        # [~0, 1] scale as the other two) keeps the other axis (left). A
        # third, narrow sidebar axis (ax_info, no ticks/spines) holds the
        # avg-accuracy tag OUTSIDE the plotting area -- with the main duo
        # plus every --compare_configs entry now also drawn here (see
        # extra_series), that tag grew to 7+ stacked lines and started
        # covering real data lines when it lived inside ax_data itself.
        fig = plt.figure(figsize=(13, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1], wspace=0.4)
        ax_ent = fig.add_subplot(gs[0, 0])
        ax_data = ax_ent.twinx()
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis("off")

        n = len(rows)
        x_acc = _cumulative_samples(rows)
        x_is_samples = x_acc is not None
        if x_acc is None:
            x_acc = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
        acc_series = {key: _ema([r[f"{key}_acc"] for r in rows], ema_window)
                      for key, _, _, _ in series}
        ent_series = {key: _ema([r[f"{key}_ent"] for r in rows], ema_window)
                      for key, _, _, _ in series}
        avg_acc = {key: float(np.mean([r[f"{key}_acc"] for r in rows])) for key, _, _, _ in series}
        for key, color, label, lw_scale in series:
            ax_data.plot(x_acc, acc_series[key], color=color, lw=1.4 * lw_scale,ls=(0, (5, 5)),
                         alpha=0.75, label=f"{label} acc", zorder=6)
            ax_ent.plot(x_acc, ent_series[key], color=color, lw=1.4 * lw_scale, ls="-.",
                        alpha=0.4, label=f"{label} entropy", zorder=2)

        # extra_series: accuracy-only -- a different KIND of comparison
        # (calibrated duo outputs, not input models) sharing the same axis/
        # scale as `series` and the proxy overlay below. Width/alpha are
        # keyed off each entry's OWN calibration_mode (its 4th tuple element,
        # see extra_series' own docstring), not whether it's "duo_acc" --
        # this run's main --calib_config might not be proxy_weighted, and one
        # of the --compare_configs alternatives might be, so the highlighting
        # follows the METHOD, not which line happens to be the main duo.
        extra_acc_series = {row_key: _ema([r[row_key] for r in rows], ema_window)
                             for row_key, _, _, _ in extra_series}
        for row_key, color, label, calib_mode in extra_series:
            is_proxy_weighted = calib_mode == "proxy_weighted"
            duo_label = f"{label} (ours)" if is_proxy_weighted else label
            ax_data.plot(x_acc, extra_acc_series[row_key], color=color,
                         lw=2.0 if is_proxy_weighted else 1.2,
                         alpha=1.0 if is_proxy_weighted else 0.5,
                         label=f"{duo_label} (duo)", zorder=5 if is_proxy_weighted else 4)

        # Overall average accuracy tag per series (plain mean over this
        # corruption's rows, not the EMA's tail value) -- in the sidebar
        # (ax_info), OUTSIDE the plot's own data area, so it never covers a
        # real line no matter how many series (independent models + main
        # duo + every --compare_configs entry) are stacked into it.
        avg_acc_lines = [f"{label} avg acc: {avg_acc[key]:.3f}" for key, _, label, _ in series]
        avg_acc_lines += [
            f"{label} (duo) avg acc: {float(np.mean([r[row_key] for r in rows])):.3f}"
            for row_key, _, label, _ in extra_series
        ]
        ax_info.text(
            0.02, 0.98, "avg accuracy\n(this corruption)\n\n" + "\n".join(avg_acc_lines),
            transform=ax_info.transAxes, ha="left", va="top", fontsize=8,
            color=C_MUTED, fontweight="bold", wrap=True,
            bbox=dict(boxstyle="round,pad=0.4", fc=C_SURFACE, ec=C_MUTED, alpha=0.85),
            zorder=6,
        )

        safe_name = corruption.replace("/", "_")
        prows = proxy_by_corruption.get(corruption, [])
        ax_gate = None
        lines_gate, labels_gate = [], []
        if prows:
            m = len(prows)
            # Gated on x_is_samples (the ACCURACY series' own units), not
            # recomputed independently -- np.interp below needs x_proxy and
            # x_acc on the SAME scale (both cumulative samples, or both
            # fractional-index), so an older batch_diagnostics.csv missing
            # "n" (pre-dating this field) falls proxy_rows back to
            # fractional too, even though proxy_rows' own "n" (a mandatory
            # field in every proxy-log writer) would otherwise be available.
            x_proxy = (_cumulative_samples(prows) if x_is_samples else None) \
                or ([i / (m - 1) for i in range(m)] if m > 1 else [0.0])
            proxy_vals = {pkey: [float(r[pkey]) for r in prows] for pkey, _, _ in proxy_series}
            proxy_ema = {pkey: _ema(vals, ema_window) for pkey, vals in proxy_vals.items()}

            # Proxy score and accuracy share one y-range (computed from
            # BOTH -- not fixed [0, 1]) so a real but small accuracy swing
            # isn't visually flattened by an axis sized for the full [0, 1]
            # span, and the two are still directly comparable at a glance.
            ax_data.set_ylim(*_adaptive_ylim(
                *acc_series.values(), *proxy_vals.values(), *proxy_ema.values(),
                *extra_acc_series.values(),
            ))

            for pkey, color, label in proxy_series:
                ax_data.plot(x_proxy, proxy_vals[pkey], color=color, lw=0.8, ls=":", alpha=0.15, zorder=1)
                ax_data.plot(x_proxy, proxy_ema[pkey], color=color, lw=1.8, ls=":",
                             alpha=0.5, label=f"{label} (proxy)", zorder=2)

            # --show_gate_weight: w_l on its OWN axis -- a weight isn't an
            # accuracy or a proxy score, so it doesn't belong on ax_data's
            # shared scale, and it isn't in nats like ax_ent's entropy either.
            # A third spine, pushed out past ax_data's own right edge, keeps
            # all three readable without overlapping ticks/labels.
            if show_gate_weight and "w_l" in prows[0]:
                w_l_vals = [float(r["w_l"]) for r in prows]
                w_l_ema = _ema(w_l_vals, ema_window)
                ax_gate = ax_ent.twinx()
                ax_gate.spines["right"].set_position(("axes", 1.12))
                ax_gate.set_frame_on(True)
                ax_gate.patch.set_visible(False)
                for spine in ax_gate.spines.values():
                    spine.set_visible(False)
                ax_gate.spines["right"].set_visible(True)
                ax_gate.spines["right"].set_color(C_MUTED)
                ax_gate.plot(x_proxy, w_l_vals, color=C_MUTED, lw=0.8, alpha=0.2, zorder=1)
                ax_gate.plot(x_proxy, w_l_ema, color=C_MUTED, lw=1.6, alpha=0.55,
                             label="w_l (gate weight, main duo)", zorder=2)
                ax_gate.set_ylim(-0.02, 1.02)
                ax_gate.set_ylabel("gate weight w_l (main duo)", color=C_MUTED)
                ax_gate.tick_params(axis="y", colors=C_MUTED)
                lines_gate, labels_gate = ax_gate.get_legend_handles_labels()

            interp_acc = {key: np.interp(x_proxy, x_acc, acc_series[key]).tolist()
                          for key, _, _, _ in series}
            interp_ent = {key: np.interp(x_proxy, x_acc, ent_series[key]).tolist()
                          for key, _, _, _ in series}
            interp_extra = {row_key: np.interp(x_proxy, x_acc, extra_acc_series[row_key]).tolist()
                             for row_key, _, _, _ in extra_series}
            csv_path = out_dir / f"corruption_{safe_name}.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["corruption", "n_samples" if x_is_samples else "t_frac"]
                    + [pkey for pkey, _, _ in proxy_series]
                    + [f"{pkey}_ema" for pkey, _, _ in proxy_series]
                    + [f"{key}_acc_ema" for key, _, _, _ in series]
                    + [f"{key}_ent_ema" for key, _, _, _ in series]
                    + [f"{label}_duo_acc_ema" for _, _, label, _ in extra_series]
                )
                for i, t in enumerate(x_proxy):
                    writer.writerow(
                        [corruption, t]
                        + [proxy_vals[pkey][i] for pkey, _, _ in proxy_series]
                        + [proxy_ema[pkey][i] for pkey, _, _ in proxy_series]
                        + [interp_acc[key][i] for key, _, _, _ in series]
                        + [interp_ent[key][i] for key, _, _, _ in series]
                        + [interp_extra[row_key][i] for row_key, _, _, _ in extra_series]
                    )
            print(f"wrote {csv_path}")
        else:
            # No proxy CSV at all (calibration_mode != proxy_weighted) or no
            # proxy rows for THIS corruption specifically -- make that visible
            # on the figure itself rather than silently leaving the left axis
            # empty, which reads as a bug rather than "there's no data".
            print(f"[{corruption}] no proxy log rows -- proxy overlay skipped "
                  f"(calibration_mode != proxy_weighted, or the proxy CSV wasn't found).")
            ax_data.text(
                0.5, 0.5, "no proxy data\n(calibration_mode != proxy_weighted)",
                transform=ax_data.transAxes, ha="center", va="center",
                fontsize=9, color=C_MUTED, alpha=0.8, zorder=1,
            )
            # No proxy series to share a scale with -- still adapt to
            # accuracy's own range rather than a fixed [0, 1].
            ax_data.set_ylim(*_adaptive_ylim(*acc_series.values(), *extra_acc_series.values()))

        ax_data.set_ylabel("accuracy / proxy score")
        ax_ent.set_ylabel("EMA entropy (nats)", color=C_MUTED)
        ax_ent.set_xlabel("samples processed" if x_is_samples else "fraction of corruption stream elapsed")
        ax_ent.set_title(f"Corruption {corruption}"
            # f"{corruption} -- accuracy/proxy score (right, solid/dotted bold EMA) vs. "
            # f"entropy (left, dash-dot EMA) -- window={ema_window}",
            # fontsize=10,
        )
        ax_ent.grid(True, alpha=0.4, lw=0.5)

        # Below the plot (not "lower left" etc.) -- with up to 4 series (2
        # proxy + 2 accuracy, before even counting entropy) any in-axes
        # corner legend risks covering real data.
        lines_data, labels_data = ax_data.get_legend_handles_labels()
        lines_ent, labels_ent = ax_ent.get_legend_handles_labels()
        ax_data.legend(lines_data + lines_ent + lines_gate, labels_data + labels_ent + labels_gate,
                        loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5, ncols=3)

        # Narrower right margin when the gate-weight axis pushed a third
        # spine out past ax_data's own -- otherwise its axis label is
        # clipped by the figure edge.
        fig.subplots_adjust(bottom=0.24, left=0.07, right=0.90 if ax_gate is not None else 0.98, top=0.92)
        out_path = out_dir / f"corruption_{safe_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")
        written.append(out_path)

    return written
