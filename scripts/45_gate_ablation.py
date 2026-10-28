#!/usr/bin/env python3
"""
What the gate adds over the fixed-TS duo, from the STORED runs of the final test (scripts/42_final_test.py): every live run scores the gated duo (Vanilla DD, beta 0.5)
and the fixed-TS duo on the same members, so no new run is needed for the comparison itself. Three views, each over the test streams of the frozen configurations
(`out/results/reset_tune/final_configs.json`, the five orderings):

    whole stream   gate - fixed-TS accuracy per method (points; mean over the orderings; paired standard error over the 15 corruptions, each averaged over the orderings)
    after resets   the gap gate - fixed-TS in the W samples after each reset (W = 5,120 = 20 proxy batches), per member that was reset, against the same gap over the
                   whole run: does the gate route to the other member while a freshly reset member is weak?
    by situation   the stream cut into windows of W samples, bucketed by how far the ViT is from the ResNet (adapted accuracies): share of windows and mean gate - fixed-TS
                   in each bucket (ViT worse by more than 5 points / within 5 / better by more than 5): where does routing pay?

Not available from the stored runs: other betas, or other routing rules (the members' logits are not stored). `--multigate DIR` reads the runs that do have them:
the multi-gate runs of `scripts/38_run_reset_live.py --extra_gates` (see scripts/46_gate_routing_runs.py), whose series `gate__<name>` are other gates scored on the same
members in the same run. There every gate (the run's main gate, beta 0.5, included) is compared with the fixed-TS duo of ITS OWN run, so a gate's gap is exact even if a
rerun's trajectory is not the stored run's to the bit (GPU arithmetic); the main gate's accuracy against the stored run's is printed as that check. It also reports the
gap in the samples BEFORE each reset (the member is degrading, the rule has not fired yet) as well as after, for each window length of `--windows`.

Standard errors: over the corruptions for a test stream (each corruption's gap averaged over the orderings, grouped by corruption), over the visits for a held-out stream (the
unit of the tuning, src/reset/tuning.py), after resets over the resets.

    python scripts/45_gate_ablation.py            # prints the tables, writes <out>/results/reset_final/gate_ablation.json
    python scripts/45_gate_ablation.py --multigate out/results/reset_gates/vitb16_rn50v1__tent_bn --methods own_fast_vs_slow            # gate_routing.json (test)
    python scripts/45_gate_ablation.py --multigate ... --stream heldout --rotations 0 1 2 3 --methods own_fast_vs_slow
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.experiments import continual as RC
from src.reset import live_report as LV
from src.utils.config import load_global

W = 5120
BUCKETS = (("ViT worse by > 5 pts", -np.inf, -0.05), ("within 5 pts", -0.05, 0.05), ("ViT better by > 5 pts", 0.05, np.inf))


def stem(cond: str, lrs, key, beta: float, stream: str = "test") -> str:
    """`key`: the ordering (test; None = the standard order) or the rotation (held-out)."""
    if stream == "heldout":
        return LV.run_stem(cond, stream="heldout", rot=key, seed=0, lrs=lrs, proxy="nnn", b_t=256, beta=beta)
    return LV.run_stem(cond, stream="test", rot=0, seed=0, lrs=lrs, proxy="nnn", b_t=256, beta=beta, order=key)


def windows(x: np.ndarray, w: int = W) -> np.ndarray:
    n = len(x) // w
    return x[:n * w].reshape(n, w).mean(1)


def summarize(runs: list[dict], series: str, w: int = W, by: str = "label") -> dict:
    """`series` - fixed TS over the loaded `runs` of one method (see the module text): whole stream, before and after resets, by situation. Points of accuracy; None where
    empty. `by`: the unit of the whole-stream standard error, "label" (each part's gap averaged over the runs, grouped by its corruption) or "visit" (every part of every run)."""
    units, whole = {}, []
    after, before = {"large": [], "small": []}, {"large": [], "small": []}
    bucket = {b[0]: [[], 0] for b in BUCKETS}
    for i, r in enumerate(runs):
        gate, ts = np.asarray(r[series], dtype=float), np.asarray(r["ts"], dtype=float)
        edges = np.cumsum(r["part_n"])
        for j, (label, a, b) in enumerate(zip(r["part_label"], np.split(gate, edges[:-1]), np.split(ts, edges[:-1]))):
            units.setdefault(label if by == "label" else (i, j), []).append(a.mean() - b.mean())
        whole.append((gate.mean(), ts.mean()))
        d = gate - ts
        for res in r["meta"].get("resets", []):
            pos = int(res["position"])
            if pos + w <= len(d):
                after[res["member"]].append(d[pos:pos + w].mean())
            if pos >= w:
                before[res["member"]].append(d[pos - w:pos].mean())
        wd = windows(d, w)
        gap = windows(np.asarray(r["large"], dtype=float), w) - windows(np.asarray(r["small"], dtype=float), w)
        for name, lo, hi in BUCKETS:
            sel = (gap > lo) & (gap <= hi) if lo > -np.inf else (gap <= hi)
            if hi == np.inf:
                sel = gap > lo
            bucket[name][0] += list(wd[sel])
            bucket[name][1] += int(sel.sum())
    per_unit = 100 * np.array([np.mean(v) for v in units.values()])
    gm, tm = 100 * np.mean([x[0] for x in whole]), 100 * np.mean([x[1] for x in whole])
    se = lambda v: float(100 * np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None
    mean = lambda v: 100 * float(np.mean(v)) if v else None
    total = max(sum(b[1] for b in bucket.values()), 1)
    return {"gate": gm, "ts": tm, "gate_minus_ts": gm - tm, "se": float(per_unit.std(ddof=1) / math.sqrt(len(per_unit))), "window": w,
            "after_resets": {k: (mean(v), len(v)) for k, v in after.items()}, "after_resets_se": {k: se(v) for k, v in after.items()},
            "before_resets": {k: (mean(v), len(v)) for k, v in before.items()}, "before_resets_se": {k: se(v) for k, v in before.items()},
            "buckets": {k: {"share": v[1] / total, "mean": mean(v[0])} for k, v in bucket.items()}, "orders": len(whole)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--stream", choices=("test", "heldout"), default="test")
    ap.add_argument("--orders", nargs="+", default=["std", "0", "1", "2", "3"], help="test stream: the orderings")
    ap.add_argument("--rotations", type=int, nargs="+", default=[0, 1, 2, 3], help="held-out stream: the rotations")
    ap.add_argument("--beta", type=float, default=0.5, help="the main gate's beta in the file names (default %(default)s, the final test's)")
    ap.add_argument("--multigate", default=None, metavar="DIR", help="the folder of multi-gate runs: score every gate of it (see above)")
    ap.add_argument("--methods", nargs="+", default=None, help="only these methods of final_configs.json (names)")
    ap.add_argument("--windows", type=int, nargs="+", default=[1280, W], help="multi-gate: window lengths (samples) before and after a reset (default %(default)s = 5 and 20 proxy batches)")
    args = ap.parse_args()
    g = load_global()
    stored = g.path("out") / "results" / "reset_live" / f"{args.duo}__tent_bn"
    folder = Path(args.multigate) if args.multigate else stored
    cfg = json.loads((g.path("out") / "results" / "reset_tune" / "final_configs.json").read_text())
    methods = list(cfg["methods"])
    ctl = cfg.get("control")
    if ctl and not args.multigate:
        rule = next(m for m in methods if m["name"] == ctl["rule"])
        methods.append({"name": f"random_{rule['name']}", "cond": f"random~{rule['cond']}~d{ctl['draws'][0]}", "lrs": rule["lrs"], "label": f"Random resets, matched to {rule['label']}", "kind": "control"})
    if args.methods:
        methods = [m for m in methods if m["name"] in args.methods]
    keys = args.rotations if args.stream == "heldout" else [None if o == "std" else int(o) for o in args.orders]
    dest = g.path("out") / "results" / "reset_final"
    dest.mkdir(parents=True, exist_ok=True)
    return multigate(args, methods, keys, folder, stored, dest) if args.multigate else stored_runs(args, methods, keys, folder, dest)


def load(folder: Path, m: dict, keys, beta: float, extra: bool, stream: str = "test") -> list[dict]:
    return [RC.load_run(p, extra=extra) for k in keys if (p := folder / f"{stem(m['cond'], m['lrs'], k, beta, stream)}.npz").exists()]


def cell(s: dict, when: str, side: str) -> str:
    """'+0.95 +-0.31 n=54' of a summary's before/after-reset gap of one member."""
    mean, n = s[f"{when}_resets"][side]
    err = s[f"{when}_resets_se"][side]
    return "--" if mean is None else f"{mean:+.2f}" + (f" +-{err:.2f}" if err is not None else "") + f" n={n}"


def stored_runs(args, methods, keys, folder: Path, dest: Path) -> int:
    out = {}
    by = "visit" if args.stream == "heldout" else "label"
    print(f"{'method':50s} {'gate':>6s} {'fixedTS':>8s} {'gate-TS':>14s}   ViT resets: before / after                  ResNet resets: before / after")
    for m in methods:
        runs = load(folder, m, keys, args.beta, False, args.stream)
        if not runs:
            continue
        s = summarize(runs, "gate", by=by)
        out[m["name"]] = {"label": m["label"], **s}
        print(f"{m['label'][:50]:50s} {s['gate']:6.1f} {s['ts']:8.1f} {s['gate_minus_ts']:+7.2f} +-{s['se']:4.2f}   "
              f"{cell(s, 'before', 'large'):>17s} / {cell(s, 'after', 'large'):>17s}   {cell(s, 'before', 'small'):>17s} / {cell(s, 'after', 'small'):>17s}")
    print(f"\nby situation: share of {W}-sample windows and mean gate - fixed-TS (points) per bucket of (ViT - ResNet) adapted accuracy")
    print(f"{'method':50s} " + " ".join(f"{b[0]:>26s}" for b in BUCKETS))
    for name, v in out.items():
        print(f"{v['label'][:50]:50s} " + " ".join(f"{100 * v['buckets'][b[0]]['share']:5.0f}% {('--' if v['buckets'][b[0]]['mean'] is None else format(v['buckets'][b[0]]['mean'], '+.2f')):>8s}        " for b in BUCKETS))
    path = dest / ("gate_ablation.json" if args.stream == "test" else f"gate_ablation_{args.stream}.json")
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {path}")
    return 0


def multigate(args, methods, keys, folder: Path, stored: Path, dest: Path) -> int:
    out = {}
    by = "visit" if args.stream == "heldout" else "label"
    unit = "visit" if by == "visit" else "corruption"
    for m in methods:
        runs = load(folder, m, keys, args.beta, True, args.stream)
        if not runs:
            continue
        names = ["gate"] + sorted(k for k in runs[0] if k.startswith("gate__"))
        sto = load(stored, m, keys, args.beta, False, args.stream)
        sm = lambda rs, k: 100 * np.mean([r[k].mean() for r in rs])
        print(f"\n== {m['label']}, {args.stream} stream: {len(runs)} run(s), resets per run (ViT / ResNet) "
              + ", ".join(f"{r['meta']['n_resets']['large']}/{r['meta']['n_resets']['small']}" for r in runs))
        print(f"   fixed-TS duo {sm(runs, 'ts'):.2f}, main gate (beta {args.beta:g}) {sm(runs, 'gate'):.2f}"
              + (f"; the stored runs: fixed-TS {sm(sto, 'ts'):.2f}, gate {sm(sto, 'gate'):.2f} ({len(sto)} run(s); a rerun is the same process, not the same bits)" if sto else ""))
        out[m["name"]] = {"label": m["label"], "stream": args.stream, "runs": len(runs), "ts": sm(runs, "ts"), "gates": {}}
        for w in args.windows:
            print(f"   windows of {w} samples ({w // 256} proxy batches); whole stream +- SE over {unit}s, windows +- SE over resets")
            print(f"   {'gate':14s} {'gate - TS':>14s}   {'before ViT resets':>22s} {'after ViT resets':>22s}   {'before ResNet resets':>22s} {'after ResNet resets':>22s}")
            for k in names:
                s = summarize(runs, k, w=w, by=by)
                name = k.removeprefix("gate__") if k != "gate" else f"main(beta{args.beta:g})"
                out[m["name"]]["gates"].setdefault(name, {})[str(w)] = s
                print(f"   {name:14s} {s['gate_minus_ts']:+7.2f} +-{s['se']:4.2f}   {cell(s, 'before', 'large'):>22s} {cell(s, 'after', 'large'):>22s}   {cell(s, 'before', 'small'):>22s} {cell(s, 'after', 'small'):>22s}")
    path = dest / ("gate_routing.json" if args.stream == "test" else f"gate_routing_{args.stream}.json")
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {path}  (gap = the gate's accuracy minus the fixed-TS duo's of the same run, points; before / after: the window ending / starting at each reset of that member)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
