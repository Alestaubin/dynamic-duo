#!/usr/bin/env python3
"""
Offline screening of the reset rules (Phase 4, R-D; src/reset/screening.py): which rule flags a collapsing member soon and rarely cries wolf.

The rules are replayed over the proxy sequences of the no-reset CONTINUAL runs. They are CALIBRATED on the held-out streams only (the 4
held-out corruptions concatenated, four rotations: scripts/03c_continual_duos.py --stream heldout --order K, runs indep = Vanilla DD and
joint_dd = Joint DD, with the frozen run of each as the label reference) and EVALUATED on the test stream (the 15 corruptions concatenated).
No parameter is ever chosen on the test stream.

    sbatch -t 00:40:00 slurm/job_cpu.sh scripts/37_screen_resets.py --duo vitb16_rn50v1 --bn

Per rule family (own / gap / both x level, imm_vs_ema, fast_vs_slow, slope, cusum) and false-alarm budget (0.5, 1, 2 per 1,000 proxy
batches) it picks on the held-out streams the configuration and threshold with the lowest delay, freezes it and scores it on the test
streams; it also scores the top three rules under other degradation labels. Writes <paths.out>/results/screen_resets/<tag>.{json,csv} (every
grid point; the selections; the test scores), the delay-against-false-alarm curves and prints the table of budget 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.experiments import continual as RC
from src.experiments.cli import add_common_args, setup
from src.reset import screening as SC
from src.reset.controller import RuleSpec
from src.utils import provenance

KIND_STYLE = {"level": ("#6b6a66", (0, (4, 2))), "imm_vs_ema": ("#2a78d6", "-"), "fast_vs_slow": ("#1baf7a", "-"),
              "slope": ("#8e5bd6", (0, (1, 1))), "cusum": ("#eb6834", "-")}


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), tta=False)
    ap.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2, 3], help="held-out rotations to calibrate on")
    ap.add_argument("--runs", nargs="+", choices=("indep", "joint_dd"), default=["indep", "joint_dd"],
                    help="whose proxy sequences to screen: Vanilla DD (indep), Joint DD (joint_dd)")
    ap.add_argument("--images", type=int, default=50000)
    ap.add_argument("--proxy", default="nnn")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--b_t", type=int, default=256)
    ap.add_argument("--budgets", type=float, nargs="+", default=list(SC.BUDGETS), help="false alarms per 1,000 proxy batches")
    ap.add_argument("--families", nargs="+", default=None, help="default: all 15 (own / gap / both x the five procedures)")
    ap.add_argument("--no_test", action="store_true", help="calibrate only (no evaluation on the test stream)")
    args = ap.parse_args()
    tta = "tent"

    ctx, seeds, out = setup(args, "screen_resets", default_seeds=[0])
    results = out.parent

    def stream_files(directory: Path, stream: str, order: int | None, seed: int, run: str) -> tuple[Path, Path]:
        tag = RC.stream_tag(args.duo, tta, args.bn, 1, args.images, seed, order, stream)
        return (RC.run_path(results / directory, tag, run, args.proxy, args.b_t, args.beta), RC.run_path(results / directory, tag, "frozen", args.proxy, args.b_t, args.beta))

    def load(directory: str, stream: str, orders: list, label: str) -> list[SC.ScreenStream]:
        found, absent = [], []
        for order in orders:
            for run in args.runs:
                f, fz = stream_files(Path(directory), stream, order, seeds[0], run)
                (found if f.is_file() and fz.is_file() and f.with_suffix(".gate.json").is_file() else absent).append((order, run, f, fz))
        if absent:
            print(f"{label}: missing runs (run scripts/03c_continual_duos.py first): " + ", ".join(f"{r}/order{o}" for o, r, *_ in absent))
        return [SC.ScreenStream.load(f, fz, args.b_t, f"{stream}{'' if o is None else o}/{r}") for o, r, f, fz in found], not absent

    tuning, tuning_ok = load("continual_heldout", "heldout", args.orders, "held-out")
    if not tuning_ok:
        return 1
    test, test_ok = ([], True) if args.no_test else load("continual", "test", [None], "test")
    if not test_ok:
        return 1
    for st in tuning + test:
        print(f"{st.name:22s} {st.n} proxy batches; degraded stretches (batches): " +
              "; ".join(f"{m} {[(a, b - a) for a, b in st.stretches(m)]}" for m in ("large", "small")))

    fams = args.families or SC.families()
    rows = SC.sweep(tuning, fams)
    selected = {b: SC.select(rows, b) for b in args.budgets}
    test_scores = {}
    for b, sel in selected.items():
        for fam, row in sel.items():
            if row is not None and test:
                test_scores[(b, fam)] = SC.evaluate(RuleSpec.from_json(row["spec"]), test)

    # the table of the budget closest to 1: the tuning score, the frozen rule's test score
    b1 = min(args.budgets, key=lambda b: abs(b - 1.0))
    table = []
    for fam, row in selected[b1].items():
        if row is None:
            table.append({"rule": fam, "config": "no point within the budget"})
            continue
        t = test_scores.get((b1, fam))
        table.append({"rule": fam, "config": json.dumps({**row["config"], "thr": row["threshold"]}), "tune_delay": row["delay"],
                      "tune_miss": row["miss_rate"], "tune_fa": row["fa"], "test_delay": t and t["delay"], "test_miss": t and t["miss_rate"],
                      "test_fa": t and t["fa"]})
    table = pd.DataFrame(table).sort_values("tune_delay", na_position="last")
    print(f"\nbudget {b1:g} false alarms per 1,000 proxy batches: delay and miss rate on the held-out streams (rule chosen there), "
          f"and the same rule on the test stream\n" + table.drop(columns="config").round(3).to_string(index=False))
    top = [r for r in table["rule"] if selected[b1][r] is not None][:3]
    sensitivity = {r: SC.label_sensitivity(RuleSpec.from_json(selected[b1][r]["spec"]), tuning) for r in top}
    print(f"\ntop three on the held-out streams: {top}")
    for r in top:
        print(f"label sensitivity of {r} (held-out): " + "; ".join(f"M{s['M']}/m{s['margin']:g}/L{s['L']}: delay {s['delay']:.1f} miss {s['miss_rate']:.2f}" for s in sensitivity[r]))

    tag = (f"{tta}__{args.duo}" + ("_bn" if args.bn else "") + f"__{args.proxy}_bt{args.b_t}__ho{''.join(map(str, args.orders))}"
           + ("" if sorted(args.runs) == ["indep", "joint_dd"] else "__" + "_".join(args.runs)) + ("__notest" if args.no_test else ""))
    note = provenance.footer(ctx.duo, tta, args.bn, f"reset-rule screening, calibrated on held-out rotations {args.orders}, test {'not used' if args.no_test else 'continual stream'}, "
                             f"gate {args.proxy} b_t={args.b_t} beta={args.beta:g}")
    (out / f"{tag}.json").write_text(json.dumps({
        "args": vars(args), "label": SC.LABEL, "tolerance": SC.TOLERANCE, "tuning_streams": [s.name for s in tuning], "test_streams": [s.name for s in test],
        "selected": {str(b): sel for b, sel in selected.items()}, "test_scores": {f"{b}|{fam}": s for (b, fam), s in test_scores.items()},
        "sensitivity": sensitivity, "rows": rows}, default=float))
    pd.DataFrame(rows).drop(columns="spec").to_csv(out / f"{tag}.csv", index=False)
    table.to_csv(out / f"{tag}__budget{b1:g}.csv", index=False)
    for f in plot_curves(rows, selected[b1], out / f"{tag}__curves.png", note):
        print(f"wrote {f}")
    print(f"wrote {out / f'{tag}.json'}")
    return 0


def plot_curves(rows: list[dict], chosen: dict, path: Path, footer: str) -> list[Path]:
    """Delay against false alarms per 1,000 proxy batches (held-out streams): one panel per quantity (own, gap, both), one line per procedure,
    the budget's chosen rule marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=300, sharey=True, gridspec_kw=dict(left=0.07, right=0.985, top=0.86, bottom=0.27, wspace=0.08))
    for ax, quantity in zip(axes, SC.QUANTITIES):
        for kind, (color, ls) in KIND_STYLE.items():
            fam = f"{quantity}_{kind}"
            pts = SC.curve(rows, fam)
            if not pts:
                continue
            ax.step([p[0] for p in pts], [p[1] for p in pts], where="post", color=color, linestyle=ls, linewidth=1.1, label=kind)
            row = chosen.get(fam)
            if row is not None and row["delay"] == row["delay"]:
                ax.plot([row["fa"]], [row["delay"]], "o", color=color, markersize=3.5)
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_title(f"{quantity}", loc="left", fontsize=7)
        ax.set_xlabel("False alarms per 1,000 proxy batches", labelpad=1)
        ax.grid(color="#e3e2de", linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Delay (proxy batches; a miss counts its stretch)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.06))
    return provenance.save_figure(fig, path, footer, dpi=300)


if __name__ == "__main__":
    sys.exit(main())
