#!/usr/bin/env python3
"""
Stage 5 of the reset tuning (plan.md, "Reset tuning and evaluation protocol"): the FINAL TEST, run once. Every configuration of the comparison is the one the
held-out tuning chose (rates, thresholds), frozen in a file BEFORE this script is run, and is run on the test stream (the 15 ImageNet-C corruptions once, severity 5)
in the standard order and in CoTTA's random orders, with the gate at beta 0.5. Nothing is selected on the test stream.

    <out>/results/reset_tune/final_configs.json
        {"methods": [{"name": "no_reset_tuned", "cond": "none", "lrs": [1.75e-5, 3.1e-5], "label": "No reset (tuned)", "kind": "baseline"}, ...],
         "control": {"rule": "own_cusum", "draws": [0]}}
        kind: baseline (no reset) / schedule / rule. `cond` is a condition of scripts/38_run_reset_live.py (`own_cusum@2`, `fixed500`, ...).
        `control`: the random-matched trigger of the named rule (it resets each member as often as the rule's run of the same order, at random steps): one run per
        draw and order, submitted with a dependency on the rule's run of that order (it reads that run's reset counts).

    python scripts/42_final_test.py plan                       # what is needed, what is missing (no job submitted)
    python scripts/42_final_test.py submit                     # sbatch one run per missing (method, order); controls wait for their reference run
    python scripts/42_final_test.py report                     # tables, paired comparisons, per-shift table: <out>/results/reset_final/ (+ docs/tables/*.tex)

`--orders std 0 1 2 3`: the standard order and CoTTA's random orders 0..3 (the replicates: the 15 corruptions in another order, the same images within a corruption).
Frozen (the label reference) comes from the stored frozen run and the 03b CSV; the boundary reset (episodic, the ImageNet-C runs of script 03b) is rescored from that CSV
(it does not depend on the order). Paired comparisons are over the 15 corruptions (each one's accuracy averaged over the orders), with a bootstrap over corruptions.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.experiments import continual as RC
from src.reset import live_report as LV
from src.reset import tuning as TU
from src.utils.config import load_global

TIME = "01:50:00"
SERIES = ("gate", "ts", "large", "small")
CSV_MODE = {"frozen": "frozen", "boundary": "indep_duo"}          # rows of the 03b CSV (series duo / large / small, segment 'clean' dropped)


def order_of(tok: str) -> int | None:
    return None if tok == "std" else int(tok)


def stem_of(cond: str, lrs, order: int | None, beta: float) -> str:
    return LV.run_stem(cond, stream="test", rot=0, seed=0, lrs=lrs, proxy="nnn", b_t=256, beta=beta, order=order)


def control_cond(rule_cond: str, draw: int) -> str:
    return f"random~{rule_cond}~d{draw}"


def all_methods(cfg: dict) -> list[dict]:
    """The methods of the file plus the control runs (kind 'control'; `ref` = the rule's method it matches)."""
    methods = [dict(m) for m in cfg["methods"]]
    ctl = cfg.get("control")
    if ctl:
        rule = next(m for m in cfg["methods"] if m["cond"] == ctl["rule"] or m["name"] == ctl["rule"])
        for d in ctl.get("draws", [0]):
            methods.append({"name": f"random_{rule['name']}_d{d}", "cond": control_cond(rule["cond"], d), "lrs": rule["lrs"], "kind": "control",
                            "label": f"Random resets, matched to {rule['label']}" + (f" (draw {d})" if len(ctl.get("draws", [0])) > 1 else ""), "ref": rule["name"]})
    return methods


def sbatch_cmd(duo: str, m: dict, order: int | None, tok: str, beta: float, dep: str | None) -> list[str]:
    cmd = ["sbatch", "--parsable", "--array=0-0", "-t", TIME, "-J", f"final_{m['name']}_o{tok}", "--exclude=kn117,kn101"]
    if dep:
        cmd.append(f"--dependency=afterok:{dep}")
    cmd += ["slurm/job.sh", "scripts/38_run_reset_live.py", "--duo", duo, "--bn", "--stream", "test", "--seeds", "0", "--conditions", m["cond"], "--beta", f"{beta:g}",
            "--lrs", f"{m['lrs'][0]:g}", f"{m['lrs'][1]:g}", "--shard", "auto"]
    if order is not None:
        cmd += ["--order", str(order)]
    return cmd


# ------------------------------------------------------------------------------------------------------------ loading

def load_visits(path: Path) -> dict:
    """{'acc': {series: {corruption: accuracy}}, 'resets': {member: per 1,000 proxy batches}} of one stored test run."""
    run = RC.load_run(path)
    labels = list(run["part_label"])
    cell = TU.cell_from_arrays(run, run["part_n"], run["meta"].get("resets", []), run["meta"].get("seen", int(sum(run["part_n"]))))
    return {"acc": {s: dict(zip(labels, v.tolist())) for s, v in cell.acc.items()}, "resets": cell.resets}


def load_csv_method(csv: Path, mode: str) -> dict:
    """The 03b episodic rows of `mode`, averaged over its seeds, per corruption: gate = the gated duo, large, small."""
    d = pd.read_csv(csv)
    d = d[(d["protocol"] == "episodic") & (d["mode"] == mode) & (d["segment"] != "clean")]
    out = {}
    for s_csv, s in (("duo", "gate"), ("large", "large"), ("small", "small")):
        out[s] = d[d["series"] == s_csv].groupby("segment")["accuracy"].mean().to_dict()
    return {"acc": out, "resets": {}}


# ------------------------------------------------------------------------------------------------------------ statistics

def bootstrap_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_shift(rec: dict, series: str, labels: list[str]) -> np.ndarray:
    return np.asarray([rec[series][c] for c in labels])


# ------------------------------------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("plan", "submit", "report"))
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--configs", default=None, help="default: <out>/results/reset_tune/final_configs.json")
    ap.add_argument("--orders", nargs="+", default=["std", "0", "1", "2", "3"])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--baseline", default="no_reset_tuned", help="report: the method every paired difference of the main table is taken against")
    ap.add_argument("--pairs", nargs="*", default=None, metavar="A:B", help="report: paired differences A - B between method names (default: every rule/schedule against the baseline, the control and the boundary reset)")
    ap.add_argument("--csv", default=None, help="the 03b CSV of the frozen and episodic rows (default: the one of --beta)")
    ap.add_argument("--tex_dir", default="docs/tables")
    ap.add_argument("--out_dir", default=None, help="report: default <out>/results/reset_final")
    args = ap.parse_args()
    g = load_global()
    folder = g.path("out") / "results" / "reset_live" / f"{args.duo}__tent_bn"
    cfg_path = Path(args.configs) if args.configs else g.path("out") / "results" / "reset_tune" / "final_configs.json"
    cfg = json.loads(cfg_path.read_text())
    methods = all_methods(cfg)
    orders = [(tok, order_of(tok)) for tok in args.orders]
    csv = Path(args.csv) if args.csv else g.path("out") / "results" / "tab_gated_duos" / f"tent__{args.duo}_bn_nnn_bt256_beta{args.beta:g}.csv"
    root = Path(__file__).resolve().parents[1]

    if args.what in ("plan", "submit"):
        have = lambda m, o: (folder / f"{stem_of(m['cond'], m['lrs'], o, args.beta)}.npz").exists()
        job_of: dict[tuple[str, str], str] = {}
        n_missing = 0
        for kind_is_control in (False, True):                                       # the rules first: the controls depend on them
            for m in methods:
                if (m["kind"] == "control") != kind_is_control:
                    continue
                for tok, o in orders:
                    if have(m, o):
                        continue
                    n_missing += 1
                    dep = None
                    if kind_is_control:
                        ref = next(x for x in methods if x["name"] == m["ref"])
                        if not have(ref, o):
                            dep = job_of.get((ref["name"], tok))
                            if dep is None:
                                print(f"  {m['name']} order {tok}: its reference run ({ref['name']}) is not run and not submitted: submit the rules first")
                                continue
                    cmd = sbatch_cmd(args.duo, m, o, tok, args.beta, dep)
                    print(("SUBMIT  " if args.what == "submit" else "to run  ") + " ".join(shlex.quote(c) for c in cmd))
                    if args.what == "submit":
                        res = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
                        jid = res.stdout.strip().split(";")[0]
                        print("   ->", jid or res.stderr.strip())
                        job_of[(m["name"], tok)] = jid
        print(f"{n_missing} run(s) {'submitted' if args.what == 'submit' else 'missing'} of {len(methods) * len(orders)} ({len(methods)} methods x {len(orders)} orders)")
        return 0

    # ---- report
    recs: dict[str, dict[str, dict]] = {}                                           # method -> order token -> record
    for m in methods:
        recs[m["name"]] = {}
        for tok, o in orders:
            p = folder / f"{stem_of(m['cond'], m['lrs'], o, args.beta)}.npz"
            if p.exists():
                recs[m["name"]][tok] = load_visits(p)
    rows = []
    out_dir = Path(args.out_dir) if args.out_dir else g.path("out") / "results" / "reset_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = None
    for m in methods:
        r = recs[m["name"]]
        if len(r) == len(orders):
            labels = list(r[orders[0][0]]["acc"]["gate"])
            break
    if labels is None:
        print("no method has a run for every order yet:", {n: sorted(v) for n, v in recs.items()})
        return 1
    # the reference rows: frozen and the boundary reset do not depend on the order
    extra = {}
    fz = folder / "live-frozen__seed0.npz"
    csv_fz = load_csv_method(csv, "frozen")
    extra["frozen"] = {"label": "Frozen (no adaptation)", "kind": "reference", "rec": {"acc": {**csv_fz["acc"], **({"ts": load_visits(fz)["acc"]["ts"]} if fz.exists() else {})}, "resets": {}}}
    extra["boundary"] = {"label": "Reset at every shift (episodic)", "kind": "reference", "rec": load_csv_method(csv, "indep_duo")}

    table = []
    per_shift_gate: dict[str, np.ndarray] = {}                                       # method -> (15,) accuracy averaged over orders
    per_order_gate: dict[str, np.ndarray] = {}
    for m in methods:
        r = recs[m["name"]]
        got = [tok for tok, _ in orders if tok in r]
        if not got:
            continue
        gate = np.asarray([[r[t]["acc"]["gate"][c] for c in labels] for t in got])    # (orders, 15)
        per_shift_gate[m["name"]], per_order_gate[m["name"]] = gate.mean(0), gate.mean(1)
        row = {"name": m["name"], "label": m["label"], "kind": m["kind"], "orders": len(got), "lrs": m["lrs"], "cond": m["cond"], "ref": m.get("ref")}
        for s in SERIES:
            v = np.asarray([[r[t]["acc"][s][c] for c in labels] for t in got]).mean(1) * 100
            row[s] = float(v.mean())
            row[s + "_se"] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
        row["resets_large"] = float(np.mean([r[t]["resets"]["large"] for t in got]))
        row["resets_small"] = float(np.mean([r[t]["resets"]["small"] for t in got]))
        table.append(row)
    for name, e in extra.items():
        acc = e["rec"]["acc"]
        row = {"name": name, "label": e["label"], "kind": e["kind"], "orders": 1, "lrs": None, "cond": name}
        for s in SERIES:
            row[s] = float(np.mean([acc[s][c] for c in labels]) * 100) if s in acc and all(c in acc[s] for c in labels) else float("nan")
            row[s + "_se"] = float("nan")
        row["resets_large"] = row["resets_small"] = float("nan")
        table.append(row)
        if "gate" in acc:
            per_shift_gate[name] = np.asarray([acc["gate"][c] for c in labels])

    base = next((r for r in table if r["name"] == args.baseline), None)
    pairs = args.pairs
    if pairs is None:
        names = {r["name"] for r in table}
        pairs = []
        for r in table:
            if r["kind"] in ("rule", "schedule") and base:
                pairs.append(f"{r['name']}:{args.baseline}")
        for r in table:
            if r["kind"] == "rule":
                pairs += [f"{r['name']}:{c['name']}" for c in table if c.get("ref") == r["name"]]
                pairs += [f"{r['name']}:{x}" for x in ("boundary", "fixed500", "fixed1000") if x in names]
        rules = [r["name"] for r in table if r["kind"] == "rule"]
        pairs += [f"{a}:{b}" for i, a in enumerate(rules) for b in rules[i + 1:]]               # the rules against each other
        if base:
            pairs.append(f"{base['name']}:frozen")
            pairs += [f"{base['name']}:{r['name']}" for r in table if r["name"] == "no_reset_conservative"]
        pairs = list(dict.fromkeys(pairs))
    paired = []
    for pair in pairs:
        a, b = pair.split(":")
        if a not in per_shift_gate or b not in per_shift_gate:
            continue
        d = 100 * (per_shift_gate[a] - per_shift_gate[b])
        lo, hi = bootstrap_ci(d)
        ent = {"pair": pair, "mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))), "ci": [lo, hi], "wins": int((d > 0).sum()), "shifts": len(d)}
        if a in per_order_gate and b in per_order_gate and len(per_order_gate[a]) == len(per_order_gate[b]) > 1:
            do = 100 * (per_order_gate[a] - per_order_gate[b])
            ent["per_order"] = do.tolist()
        paired.append(ent)

    (out_dir / "final_test.json").write_text(json.dumps({"labels": labels, "table": table, "paired": paired, "orders": [t for t, _ in orders], "beta": args.beta,
                                                         "per_shift_gate": {k: (100 * v).tolist() for k, v in per_shift_gate.items()}}, indent=1, default=float))
    pd.DataFrame({k: 100 * v for k, v in per_shift_gate.items()}, index=labels).round(2).to_csv(out_dir / "final_test_per_shift.csv")

    f = lambda x: "--" if x != x else f"{x:.1f}"
    fse = lambda x, se: "--" if x != x else (f"{x:.1f}" + (f"$\\pm${se:.1f}" if se == se else ""))
    lines = ["\\begin{table}[t]", "  \\centering\\small", "  \\begin{tabular}{@{}lccccc@{}}", "    \\toprule",
             "    Method & Vanilla DD & Fixed-TS duo & ViT & ResNet & Resets/1,000 \\\\", "    \\midrule"]
    for kind, title in (("reference", "References"), ("baseline", "No reset"), ("schedule", "Schedules"), ("rule", "Label-free triggers"), ("control", "Control")):
        rs = [r for r in table if r["kind"] == kind]
        if not rs:
            continue
        for r in rs:
            rst = "--" if r["resets_large"] != r["resets_large"] else f"{r['resets_large']:.1f} / {r['resets_small']:.1f}"
            lines.append(f"    {r['label']} & {fse(r['gate'], r['gate_se'])} & {fse(r['ts'], r['ts_se'])} & {fse(r['large'], r['large_se'])} & {fse(r['small'], r['small_se'])} & {rst} \\\\")
        lines.append("    \\midrule")
    lines[-1] = "    \\bottomrule"
    lines += ["  \\end{tabular}",
              f"  \\caption{{Final test: accuracy (\\%, mean over the 15 corruptions) on the ImageNet-C test stream, every configuration as tuned on the held-out stream, the gate at $\\beta={args.beta:g}$. "
              f"Mean$\\pm$SE over {len(orders)} orderings of the corruptions (standard order and CoTTA's random orders); resets per 1,000 proxy batches for the ViT / ResNet. "
              "Frozen and the episodic reset are order-independent (scripts/03b rows). Produced by \\rl{scripts/42\\_final\\_test.py}.}", "  \\label{tab:final_test}", "\\end{table}"]
    (root / args.tex_dir).mkdir(parents=True, exist_ok=True)
    (root / args.tex_dir / "final_test_main.tex").write_text("\n".join(lines) + "\n")

    abbr = {"gaussian_noise": "Gauss", "shot_noise": "Shot", "impulse_noise": "Impul", "defocus_blur": "Defoc", "glass_blur": "Glass", "motion_blur": "Motn", "zoom_blur": "Zoom",
            "snow": "Snow", "frost": "Frost", "fog": "Fog", "brightness": "Brit", "contrast": "Contr", "elastic_transform": "Elast", "pixelate": "Pixel", "jpeg_compression": "JPEG"}
    cols = [abbr.get(c.split("/")[0], c.split("/")[0]) for c in labels]
    rows_ps = [(r["label"], 100 * per_shift_gate[r["name"]]) for r in table if r["name"] in per_shift_gate]
    best = np.max([v for _, v in rows_ps], axis=0)
    lt = ["\\begin{table*}[t]", "  \\centering\\scriptsize\\setlength{\\tabcolsep}{2.2pt}", "  \\resizebox{\\linewidth}{!}{%", "  \\begin{tabular}{@{}l" + "r" * (len(cols) + 1) + "@{}}", "    \\toprule",
          "    Method & " + " & ".join(cols) + " & Avg. \\\\", "    \\midrule"]
    for lab, v in rows_ps:
        cells = [f"\\textbf{{{x:.1f}}}" if abs(x - b) < 1e-9 else f"{x:.1f}" for x, b in zip(v, best)]
        lt.append(f"    {lab} & " + " & ".join(cells) + f" & {v.mean():.1f} \\\\")
    lt += ["    \\bottomrule", "  \\end{tabular}}",
           f"  \\caption{{Final test, per corruption (severity 5; accuracy of Vanilla DD, $\\beta={args.beta:g}$, \\%): each corruption's accuracy averaged over the orderings of the stream (the episodic reset and the frozen duo: scripts/03b rows). Bold: the best method on that corruption.}}",
           "  \\label{tab:final_per_shift}", "\\end{table*}"]
    (root / args.tex_dir).mkdir(parents=True, exist_ok=True)
    (root / args.tex_dir / "final_test_per_shift.tex").write_text("\n".join(lt) + "\n")

    name_of = {r["name"]: r["label"] for r in table}
    lp = ["\\begin{table}[t]", "  \\centering\\small", "  \\begin{tabular}{@{}lrrcc@{}}", "    \\toprule",
          "    Difference (Vanilla DD, points) & Mean & SE & 95\\% CI & Corruptions won \\\\", "    \\midrule"]
    for e in paired:
        a, b = e["pair"].split(":")
        lp.append(f"    {name_of[a]} $-$ {name_of[b]} & {e['mean']:+.2f} & {e['se']:.2f} & [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}] & {e['wins']}/{e['shifts']} \\\\")
    lp += ["    \\bottomrule", "  \\end{tabular}",
           "  \\caption{Paired differences on the test stream: per corruption (its accuracy averaged over the orderings), mean, standard error over the 15 corruptions, "
           "bootstrap 95\\% interval over corruptions (10,000 resamples) and the number of corruptions the first method wins. Corruptions are not independent, so the "
           "interval is a noise floor, not a guarantee.}", "  \\label{tab:final_paired}", "\\end{table}"]
    (root / args.tex_dir / "final_test_paired.tex").write_text("\n".join(lp) + "\n")

    pm = "+-"
    print(f"{'method':46s} {'duo':>10s} {'fixedTS':>8s} {'ViT':>6s} {'ResNet':>7s}  resets/1000 (ViT/ResNet)  orders")
    for r in table:
        print(f"{r['label'][:46]:46s} {fse(r['gate'], r['gate_se']).replace(chr(36) + chr(92) + 'pm' + chr(36), pm):>10s} {f(r['ts']):>8s} {f(r['large']):>6s} {f(r['small']):>7s}  "
              f"{'--' if r['resets_large'] != r['resets_large'] else '%.1f / %.1f' % (r['resets_large'], r['resets_small']):>22s}  {r['orders']}")
    print()
    for e in paired:
        a, b = e["pair"].split(":")
        po = "  per order " + " ".join(f"{x:+.2f}" for x in e["per_order"]) if "per_order" in e else ""
        print(f"  {a} - {b}: {e['mean']:+.2f} +- {e['se']:.2f}  CI [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]  ({e['wins']}/{e['shifts']}){po}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
