#!/usr/bin/env python3
"""
The tables of the reset tuning (scripts/40_tune_reset.py, src/reset/tuning.py) for the report, written to docs/tables/ as LaTeX fragments from the STORED runs and the
states the selection saved (out/results/reset_tune/*.json): no number is typed by hand.

    tuning_rates_large.tex / tuning_rates_small.tex   rate ladders of the decoupled methods (the schedules and the own_* rules): one row per rung of the sqrt-2 grid, one
                                                      column per method, the member's OWN whole-stream accuracy on held-out rotation 0 (and rotation 2 where it was run);
                                                      the chosen rung in bold, a dagger where the member died (four 5,120-image windows below 5%)
    tuning_rates_gap.tex                              the coupled rule: one row per (ViT, ResNet) cell tried, the gated duo (beta 0.5) on rotations 0 and 2
    tuning_noreset.tex                                the no-reset baseline's ladder (the 04b tuner's runs, rotation 0) and its rotation-2 runs
    tuning_thresholds.tex                             the threshold ladders (stage 3) at the tuned rates: the gated duo on rotations 0 and 2, resets per 1,000 batches
    tuning_confirm.tex                                the chosen configurations on all four rotations (1 and 3 are validation only)

    python scripts/43_tuning_tables.py                # every table that has data
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.reset import live_report as LV
from src.reset import tuning as TU
from src.utils.config import load_global

DECOUPLED = ("fixed500", "fixed1000", "own_fast_vs_slow@0.7", "own_cusum@2", "own_cusum@0.25")
RULES = {"own_fast_vs_slow": 0.7, "own_cusum": 2.0, "gap_fast_vs_slow": 1.0}


def pretty(cond: str) -> str:
    base, _, thr = cond.partition("@")
    name = LV.pretty(base)
    sym = {"own_fast_vs_slow": r"$\delta$", "own_cusum": "$h$", "gap_fast_vs_slow": r"$\delta$"}.get(base)
    return f"{name} ({sym}={thr})" if thr and sym else name


def esc(s: str) -> str:
    return s.replace("_", r"\_")


def fmt_rate(x: float) -> str:
    m, e = f"{x:.3g}".replace("e-0", "e-").split("e") if "e" in f"{x:.3g}" else (f"{x:.3g}", None)
    if e is None:
        mant, exp = f"{x:.2e}".split("e")
        m, e = mant.rstrip("0").rstrip("."), str(int(exp))
    return f"${m}\\!\\times\\!10^{{{int(e)}}}$"


def state(outdir: Path, name: str) -> dict | None:
    p = outdir / name
    return json.loads(p.read_text()) if p.exists() else None


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--tex_dir", default="docs/tables")
    args = ap.parse_args()
    g = load_global()
    root = Path(__file__).resolve().parents[1]
    folder = g.path("out") / "results" / "reset_live" / f"{args.duo}__tent_bn"
    outdir = g.path("out") / "results" / "reset_tune"
    store = TU.Store(folder, gate=f"nnn_bt256_beta{args.beta:g}")
    tex = root / args.tex_dir

    # ---- member ladders of the decoupled methods
    for side, ks, title in (("large", range(-3, 10), "ViT-B/16"), ("small", range(-8, 6), "ResNet-50")):
        cols, chosen = [], {}
        for cond in DECOUPLED:
            st = state(outdir, f"{cond}__rates.json")
            chosen[cond] = (st["chosen"][TU.SIDES.index(side)] if st else None)
            cols.append(cond)
        rows = []
        for k in ks:
            r = TU.rate(side, k)
            cells = {}
            for cond in cols:
                c0, c2 = (store.cell(cond, rot, **{side: r}) for rot in (0, 2))
                cells[cond] = (c0, c2)
            if all(c0 is None for c0, _ in cells.values()):
                continue
            row = f"    {fmt_rate(r)}"
            for cond in cols:
                c0, c2 = cells[cond]
                if c0 is None:
                    row += " & --"
                    continue
                s = f"{100 * c0.mean(side):.2f}" + (f" / {100 * c2.mean(side):.2f}" if c2 is not None else "") + ("$^\\dagger$" if c0.dead[side] or (c2 is not None and c2.dead[side]) else "")
                if chosen[cond] is not None and math.isclose(chosen[cond], r, rel_tol=1e-3):
                    s = f"\\textbf{{{s}}}"
                row += f" & {s}"
            rows.append(row + " \\\\")
        if not rows:
            continue
        lines = ["\\begin{table}[t]", "  \\centering\\small", "  \\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "    \\toprule",
                 "    Rate & " + " & ".join(pretty(c) for c in cols) + " \\\\", "    \\midrule"] + rows + ["    \\bottomrule", "  \\end{tabular}",
                 f"  \\caption{{{title}'s own accuracy (\\%, whole held-out stream) at each rung of its learning-rate ladder, under each method with the trigger active: "
                 "rotation 0 / rotation 2 (rotation 2 only for the argmax and its neighbours). Bold: the chosen rung (tie-break included); $^\\dagger$: the member died. "
                 "Produced by \\rl{scripts/43\\_tuning\\_tables.py}.}", f"  \\label{{tab:tune_rates_{side}}}", "\\end{table}"]
        write(tex / f"tuning_rates_{side}.tex", lines)

    # ---- the coupled rule: cells
    cond = "gap_fast_vs_slow@1"
    st = state(outdir, f"{cond}__rates.json")
    seen = []
    for rl, rs, path in ((l, s, p) for r0, l, s, p in store.paths(cond) if r0 == 0):
        seen.append((rl, rs))
    if seen:
        rows = []
        for rl, rs in sorted(set(seen)):
            c0, c2 = store.cell(cond, 0, large=rl, small=rs), store.cell(cond, 2, large=rl, small=rs)
            ch = st is not None and all(math.isclose(a, b, rel_tol=1e-3) for a, b in zip(st["chosen"], (rl, rs)))
            s0 = f"{100 * c0.mean('gate'):.2f}" + ("$^\\dagger$" if any(c0.dead.values()) else "")
            s2 = "--" if c2 is None else f"{100 * c2.mean('gate'):.2f}" + ("$^\\dagger$" if any(c2.dead.values()) else "")
            res = f"{c0.resets['large']:.1f} / {c0.resets['small']:.1f}"
            row = f"    {fmt_rate(rl)} & {fmt_rate(rs)} & {s0} & {s2} & {res} \\\\"
            rows.append(row.replace(f"{s2} &", f"\\textbf{{{s2}}} &") if ch and s2 != "--" else row)
        write(tex / "tuning_rates_gap.tex", ["\\begin{table}[t]", "  \\centering\\small", "  \\begin{tabular}{@{}lcccc@{}}", "    \\toprule",
                                                "    ViT rate & ResNet rate & Duo, rotation 0 & Duo, rotation 2 & Resets/1,000 (rot 0) \\\\", "    \\midrule"] + rows +
              ["    \\bottomrule", "  \\end{tabular}",
               "  \\caption{The coupled rule (gap fast/slow, $\\delta=1$): the gated duo's accuracy (\\%, $\\beta=0.5$) at each (ViT, ResNet) pair tried, since its ViT resets "
               "depend on the ResNet's proxy and the members cannot be tuned separately. Bold: the chosen pair; $^\\dagger$: a member died.}", "  \\label{tab:tune_rates_gap}", "\\end{table}"])

    # ---- no-reset baseline (04b ladder + rotation 2)
    nr = state(outdir, "none__baseline2.json")
    p04 = g.path("out") / "results" / "tune_continual_lr" / f"{args.duo}__tent_bn" / "results.json"
    if nr and p04.exists():
        lad = json.loads(p04.read_text())["ladders"]
        rows = []
        for side, name in (("large", "ViT"), ("small", "ResNet")):
            for r in nr["tables"][side]:
                rate = lad[side][r["rung"]]
                s = f"{100 * r['acc_rot0']:.2f}" + (f" / {100 * r['acc']:.2f}" if r["acc"] is not None else "")
                rows.append(f"    {name} & {fmt_rate(rate)} & " + (f"\\textbf{{{s}}}" if r["chosen"] else s) + " \\\\")
        write(tex / "tuning_noreset.tex", ["\\begin{table}[t]", "  \\centering\\small", "  \\begin{tabular}{@{}llc@{}}", "    \\toprule",
                                           "    Member & Rate & Own accuracy, rotation 0 / rotation 2 \\\\", "    \\midrule"] + rows + ["    \\bottomrule", "  \\end{tabular}",
              "  \\caption{No reset: the ladder of the continual-TTA tuner (04b), each member scored on its own accuracy; the argmax and its neighbours were also run on rotation 2. "
              "Bold: the accuracy-tuned rate. The conservative pair of 04b ($1.26\\!\\times\\!10^{-5}$ / $2.2\\!\\times\\!10^{-5}$) is a stability rule, reported beside it.}",
              "  \\label{tab:tune_noreset}", "\\end{table}"])

    # ---- threshold ladders
    blocks = []
    for fam, base in RULES.items():
        st = state(outdir, f"{fam}__threshold.json")
        rs = state(outdir, f"{fam}@{base:g}__rates.json")
        if rs is None:
            continue
        pair = st["pair"] if st else rs["chosen"]
        for k in range(-6, 5):
            thr = TU.threshold(base, k)
            c0, c2 = (store.cell(f"{fam}@{thr:g}", rot, large=pair[0], small=pair[1]) for rot in (0, 2))
            if c0 is None:
                continue
            ch = st is not None and math.isclose(st["threshold"], thr, rel_tol=1e-6)
            s0 = f"{100 * c0.mean('gate'):.2f}"
            s2 = "--" if c2 is None else f"{100 * c2.mean('gate'):.2f}"
            res = f"{c0.resets['large']:.1f} / {c0.resets['small']:.1f}"
            blocks.append(f"    {pretty(fam).split(' (')[0]} & {thr:g} & " + (f"\\textbf{{{s0}}} & \\textbf{{{s2}}}" if ch else f"{s0} & {s2}") + f" & {res} \\\\")
    if blocks:
        write(tex / "tuning_thresholds.tex", ["\\begin{table}[t]", "  \\centering\\small", "  \\begin{tabular}{@{}lcccc@{}}", "    \\toprule",
                                              "    Rule & Threshold & Duo, rotation 0 & Duo, rotation 2 & Resets/1,000 (ViT / ResNet) \\\\", "    \\midrule"] + blocks + ["    \\bottomrule", "  \\end{tabular}",
              "  \\caption{Stage 3: the detection threshold at the tuned rates (gated duo, $\\beta=0.5$, whole held-out stream). Rotation 2 only for the argmax and its neighbours. "
              "Bold: the chosen threshold.}", "  \\label{tab:tune_thresholds}", "\\end{table}"])

    # ---- stage 4: the random-matched control of the best rule, per draw, on validation rotation 1
    rule, rule_pair = "own_fast_vs_slow@0.7", (0.00056, 8.84e-05)
    ref = store.cell(rule, 1, large=rule_pair[0], small=rule_pair[1])
    draws = [(d, store.cell(f"random~{rule}~d{d}", 1, large=rule_pair[0], small=rule_pair[1])) for d in range(10)]
    draws = [(d, c) for d, c in draws if c is not None]
    if ref is not None and draws:
        def diff(a, b, series):
            d = 100 * (a.acc[series] - b.acc[series])
            return d.mean(), d.std(ddof=1) / math.sqrt(len(d)), int((d > 0).sum()), len(d)
        rows = [f"    The rule itself & {100 * ref.mean('gate'):.1f} & {100 * ref.mean('large'):.1f} & {100 * ref.mean('small'):.1f} & {ref.resets['large']:.1f} / {ref.resets['small']:.1f} & -- & -- & -- \\\\", "    \\midrule"]
        for d, c in draws:
            m, se, w, n = diff(ref, c, "gate")
            mv, sv, _, _ = diff(ref, c, "large")
            ms, ss, _, _ = diff(ref, c, "small")
            rows.append(f"    Random, draw {d} & {100 * c.mean('gate'):.1f} & {100 * c.mean('large'):.1f} & {100 * c.mean('small'):.1f} & {c.resets['large']:.1f} / {c.resets['small']:.1f} & {m:+.2f}$\\pm${se:.2f} ({w}/{n}) & {mv:+.1f}$\\pm${sv:.1f} & {ms:+.2f}$\\pm${ss:.2f} \\\\")
        mean_gate = np.mean([100 * c.mean("gate") for _, c in draws])
        rows += ["    \\midrule", f"    Mean of the draws & {mean_gate:.1f} & {np.mean([100 * c.mean('large') for _, c in draws]):.1f} & {np.mean([100 * c.mean('small') for _, c in draws]):.1f} & -- & {100 * ref.mean('gate') - mean_gate:+.2f} & {np.mean([100 * (ref.mean('large') - c.mean('large')) for _, c in draws]):+.1f} & {np.mean([100 * (ref.mean('small') - c.mean('small')) for _, c in draws]):+.2f} \\\\"]
        write(tex / "tuning_control.tex", ["\\begin{table}[t]", "  \\centering\\small\\setlength{\\tabcolsep}{4pt}", "  \\begin{tabular}{@{}lccccccc@{}}", "    \\toprule",
                                           "    & Duo & \\vit{} & \\rn{} & Resets/1,000 & Duo: rule $-$ random & \\vit{} & \\rn{} \\\\", "    \\midrule"] + rows + ["    \\bottomrule", "  \\end{tabular}",
              "  \\caption{Stage 4, held-out validation rotation 1: \\rl{own\\_fast\\_vs\\_slow} ($\\delta=0.7$; \\vit{} $5.6{\\cdot}10^{-4}$, \\rn{} $8.84{\\cdot}10^{-5}$) against a trigger that resets each member as often as the rule did, at random steps (five draws). "
              "Accuracy (\\%) and the paired difference per visit (mean$\\pm$SE over the 16 visits; visits won by the rule). The rule's advantage is almost all in the \\vit{}.}", "  \\label{tab:tune_control}", "\\end{table}"])

    # ---- confirmation of the chosen pairs on all four rotations
    cfg_path = outdir / "final_configs.json"
    methods = json.loads(cfg_path.read_text())["methods"] if cfg_path.exists() else []
    rows = []
    for m in methods:
        cs = [store.cell(m["cond"], r, large=m["lrs"][0], small=m["lrs"][1]) for r in (0, 1, 2, 3)]
        if any(c is None for c in cs):
            continue
        rows.append(f"    {m['label']} & {fmt_rate(m['lrs'][0])} & {fmt_rate(m['lrs'][1])} & " + " & ".join(f"{100 * c.mean('gate'):.1f}" for c in cs) + f" & {100 * np.mean([c.mean('ts') for c in cs]):.1f} \\\\")
    if rows:
        write(tex / "tuning_confirm.tex", ["\\begin{table}[t]", "  \\centering\\small", "  \\begin{tabular}{@{}lccccccc@{}}", "    \\toprule",
                                           "    Method & ViT rate & ResNet rate & Rot 0 & Rot 1$^\\ast$ & Rot 2 & Rot 3$^\\ast$ & Fixed-TS (mean) \\\\", "    \\midrule"] + rows +
              ["    \\bottomrule", "  \\end{tabular}",
               "  \\caption{The tuned configurations on all four rotations of the held-out stream (gated duo, $\\beta=0.5$, whole-stream accuracy in \\%). Rotations 0 and 2 were used for "
               "the tuning; $^\\ast$ rotations 1 and 3 only validate.}", "  \\label{tab:tune_confirm}", "\\end{table}"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
