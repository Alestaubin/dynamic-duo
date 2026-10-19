#!/usr/bin/env python3
"""
Stage E, offline part: choose the proxy, beta and proxy-batch size b_t of Vanilla Dynamic Duos on the HELD-OUT streams
(Phase 3; src/experiments/proxy_selection.py has the rule), for one or more base algorithms, and write results/selection_section45.json.

Reads what scripts/07_eval_duo_grid.py wrote (<duo>__<tta>__seed<S>__<keys>.rows.csv / .diag.csv / .records.npz / .temps.json in the evaluation
directory). It refuses to read any stream that is not one of the held-out streams. The winner is the Vanilla DD cell (beta > 0) with the
best mean held-out accuracy; cells within 0.01 points tie and the tie goes to the higher ordering accuracy of the proxy at that b_t. With
several algorithms the report says whether the same proxy wins.

    python scripts/08_select_proxy.py --duo vitb16_rn50v1 --algos tent eta --seed 0
    python scripts/08_select_proxy.py ... --result tent=out/results/eval_duo_grid/<name>      # when an algorithm has several evaluations

The JSON keeps a `joint_dd` entry written by scripts/10_select_joint_dd_beta.py, if there is one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments.logit_runs import stream_segments
from src.experiments.proxy_selection import compare_algorithms, load_records, select_algorithm
from src.utils.config import REPO_ROOT, load_global

PROTOCOL = ("Selection on the held-out corruptions only (speckle noise, Gaussian blur, spatter, saturate; severity 5). Candidates: Vanilla DD cells "
            "with beta > 0 (proxy x beta x b_t). Score: top-1 accuracy averaged over the held-out streams. Cells within 0.01 points of the best "
            "are tied; ties go to the higher ordering accuracy of the proxy at that b_t, pooled over the held-out proxy batches (then Spearman, "
            "smaller beta, smaller b_t, name). Proxies run on the raw softmax; temperatures are refitted per base algorithm on the pooled "
            "held-out streams; the clip eps of the gate is 1e-3; the gate semantics are the live gate's (docs/design_section45.md, section 1).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", default="vitb16_rn50v1")
    ap.add_argument("--algos", nargs="+", default=["tent", "eta"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_dir", default=None, help="default: <paths.out>/results/eval_duo_grid")
    ap.add_argument("--result", nargs="*", default=[], metavar="ALGO=PREFIX", help="the evaluation to use for an algorithm")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "selection_section45.json"))
    ap.add_argument("--global", dest="global_path", default=None)
    args = ap.parse_args()

    g = load_global(args.global_path)
    eval_dir = Path(args.eval_dir) if args.eval_dir else g.path("out") / "results" / "eval_duo_grid"
    held = {s.file_stem for s in stream_segments(["heldout"], g)}
    explicit = dict(r.split("=", 1) for r in args.result)
    sel, files = {}, {}
    for algo in args.algos:
        if algo in explicit:
            prefix = Path(explicit[algo])
        else:
            found = sorted(eval_dir.glob(f"{args.duo}__{algo}__seed{args.seed}__*.rows.csv"))
            if len(found) != 1:
                print(f"{algo}: expected exactly one evaluation in {eval_dir} for seed {args.seed}, found {[f.name for f in found]}; "
                      f"name the one to use with --result {algo}=<prefix>")
                return 1
            prefix = Path(str(found[0])[:-len(".rows.csv")])
        rows_df = pd.read_csv(f"{prefix}.rows.csv")
        streams = set(rows_df["stream"])
        if not streams <= held:
            print(f"{algo}: {prefix.name} contains streams that are not held-out: {sorted(streams - held)}; selection never reads them")
            return 1
        meta = json.loads(Path(f"{prefix}.temps.json").read_text())
        rows = rows_df.to_dict("records")
        sel[algo] = select_algorithm(rows, load_records(f"{prefix}.records.npz"), sorted(streams), meta["temps"])
        files[algo] = prefix.name
        c = sel[algo]["chosen"]
        print(f"\n== {algo}  ({prefix.name})  temperatures {meta['temps']}")
        t = pd.DataFrame(sel[algo]["top10"])[["proxy", "beta", "b_t", "accuracy", "nll", "ordering_acc", "spearman", "tied"]]
        t["accuracy"] = 100 * t["accuracy"]
        print("top 10 cells by held-out accuracy:\n" + t.round(3).to_string(index=False))
        b = pd.DataFrame(sel[algo]["best_per_proxy"])
        b["accuracy"] = 100 * b["accuracy"]
        print("best cell of each proxy:\n" + b.round(3).to_string(index=False))
        base = sel[algo]["baselines"]
        print(f"baselines (macro accuracy %): large {100 * base['large']['accuracy']:.2f}, small {100 * base['small']['accuracy']:.2f}, fixed held-out TS "
              f"{100 * base['fixed_heldout_ts']['accuracy']:.2f}, fixed clean TS {100 * base['fixed_clean_ts']['accuracy']:.2f}, oracle selection at "
              f"b_t={c['b_t']} {100 * base['oracle_select_same_b_t']['accuracy']:.2f}, oracle weight {100 * base['oracle_w_same_b_t']['accuracy']:.2f}")
        gap = sel[algo]["share_of_oracle_w_gap_closed"]
        print(f"CHOSEN: proxy {c['proxy']}, beta {c['beta']:g}, b_t {c['b_t']}: accuracy {100 * c['accuracy']:.2f} (worst stream {100 * c['worst_stream_accuracy']:.2f}), "
              f"+{sel[algo]['gain_over_fixed_heldout_ts_points']:.2f} over the fixed combination, {sel[algo]['gain_over_best_member_points']:+.2f} over the best "
              f"member, {'' if gap is None else f'{100 * gap:.0f}% of the oracle-weight gap, '}ordering accuracy {c['ordering_acc']}, Spearman {c['spearman']}"
              f"{'; the tie was broken by ordering accuracy' if sel[algo]['tie_broken_by_ordering'] else ''}"
              f"{'' if sel[algo]['beats_fixed'] else '  ** NO GATE BEATS THE FIXED COMBINATION **'}")
    cmp = compare_algorithms(sel) if len(sel) > 1 else None
    if cmp:
        print(f"\nSame proxy for every algorithm: {cmp['same_proxy_wins']} (winners {cmp['winners']}); same cell: {cmp['same_cell_wins']}; "
              f"top-3 proxies {cmp['top3_proxies']}")
    out = Path(args.out)
    prev = json.loads(out.read_text()) if out.is_file() else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**prev, "duo": args.duo, "seed": args.seed, "protocol": PROTOCOL, "evaluations": files,
                               "vanilla_dd": sel, "comparison": cmp}, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
