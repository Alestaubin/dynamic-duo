#!/usr/bin/env python3
"""
Stage E, last step: choose Joint Dynamic Duos' beta from the live held-out runs of scripts/09_run_joint_dd.py, and record it in
results/selection_section45.json under `joint_dd`.

For each (proxy, b_t) cell run, the beta with the best mean held-out accuracy of the gated prediction wins (beta = 0, the joint duo mode with the
fixed combination, is reported as the reference and is not a candidate); cells within 0.01 points of the best are tied and the tie goes to the
SMALLER beta (the closer to the fixed combination). The report says which proxy gives Joint DD its best held-out accuracy and by how much it beats
beta = 0 and the members.

    python scripts/10_select_joint_dd_beta.py --duo vitb16_rn50v1 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments.logit_runs import stream_segments
from src.utils.config import REPO_ROOT, load_global

TIE = 1e-4


def read_runs(run_dir: Path, duo: str, seed: int, heldout: set[str]) -> pd.DataFrame:
    """One row per (proxy, b_t, beta, series): the macro mean accuracy / NLL / ECE over the held-out streams and the worst stream."""
    rows = []
    for f in sorted(glob.glob(str(run_dir / f"joint_dd__{duo}__*__beta*__seed{seed}.rows.csv"))):
        m = re.search(rf"joint_dd__{re.escape(duo)}__(.+)_bt(\d+)__beta([0-9.e+-]+)__seed{seed}\.rows\.csv$", f)
        proxy, b_t, beta = m.group(1), int(m.group(2)), float(m.group(3))
        df = pd.read_csv(f)
        extra = set(df["segment"].str.replace("/", "__")) - heldout
        if extra:
            raise ValueError(f"{f} contains streams that are not held-out: {sorted(extra)}")
        for series, d in df.groupby("series"):
            rows.append({"proxy": proxy, "b_t": b_t, "beta": beta, "series": series, "accuracy": d.accuracy.mean(), "worst": d.accuracy.min(),
                         "nll": d.nll.mean(), "ece": d.ece.mean(), "n_streams": len(d)})
    return pd.DataFrame(rows)


def choose(runs: pd.DataFrame) -> dict:
    out = {"cells": {}}
    duo = runs[runs.series == "duo"]
    zero = duo[duo.beta == 0.0]
    for (proxy, b_t), d in duo.groupby(["proxy", "b_t"]):
        cand = d[d.beta > 0.0]
        if cand.empty:
            continue
        best = cand.accuracy.max()
        tied = cand[cand.accuracy >= best - TIE].sort_values("beta")
        w = tied.iloc[0]
        members = runs[(runs.proxy == proxy) & (runs.b_t == b_t) & (runs.beta == w.beta) & (runs.series != "duo")]
        out["cells"][f"{proxy}_bt{b_t}"] = {
            "proxy": proxy, "b_t": int(b_t), "beta": float(w.beta), "accuracy": float(w.accuracy), "worst_stream_accuracy": float(w.worst),
            "nll": float(w.nll), "ece": float(w.ece), "tie_broken_by_smaller_beta": bool(len(tied) > 1),
            "table": [{"beta": float(r.beta), "accuracy": float(r.accuracy), "worst": float(r.worst), "nll": float(r.nll)}
                      for r in d.sort_values("beta").itertuples()],
            "members_under_the_chosen_run": {r.series: float(r.accuracy) for r in members.itertuples()}}
    if not zero.empty:
        z = zero.sort_values("accuracy", ascending=False).iloc[0]
        out["beta0_reference"] = {"accuracy": float(z.accuracy), "worst_stream_accuracy": float(z.worst), "nll": float(z.nll)}
        for c in out["cells"].values():
            c["gain_over_beta0_points"] = 100 * (c["accuracy"] - z.accuracy)
    if out["cells"]:
        best = max(out["cells"].values(), key=lambda c: (round(c["accuracy"], 12), -c["beta"]))
        out["best_cell"] = f"{best['proxy']}_bt{best['b_t']}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", default=None, help="default: <paths.out>/results/run_joint_dd")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "selection_section45.json"))
    ap.add_argument("--global", dest="global_path", default=None)
    args = ap.parse_args()
    g = load_global(args.global_path)
    run_dir = Path(args.run_dir) if args.run_dir else g.path("out") / "results" / "run_joint_dd"
    runs = read_runs(run_dir, args.duo, args.seed, {s.file_stem for s in stream_segments(["heldout"], g)})
    if runs.empty:
        print(f"no Joint DD runs in {run_dir}")
        return 1
    res = choose(runs)
    for key, c in res["cells"].items():
        print(f"\n== Joint DD, {key}: held-out accuracy of the gated prediction by beta")
        print(pd.DataFrame(c["table"]).assign(accuracy=lambda d: 100 * d.accuracy, worst=lambda d: 100 * d.worst).round(3).to_string(index=False))
        print(f"CHOSEN beta {c['beta']:g}: {100 * c['accuracy']:.2f} (worst stream {100 * c['worst_stream_accuracy']:.2f})"
              + (f", {c['gain_over_beta0_points']:+.2f} over beta = 0" if "gain_over_beta0_points" in c else "")
              + f"; members under that run {c['members_under_the_chosen_run']}")
    if "beta0_reference" in res:
        print(f"\nbeta = 0 reference (the joint duo mode): {100 * res['beta0_reference']['accuracy']:.2f}")
    out = Path(args.out)
    prev = json.loads(out.read_text()) if out.is_file() else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**prev, "joint_dd": res}, indent=1, default=float))
    print(f"\nwrote {out} (joint_dd)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
