#!/usr/bin/env python3
"""
Stage E, live part: Joint Dynamic Duos on the held-out streams, to choose its beta (Phase 3; src/experiments/joint_dd.py).

Joint DD adapts BOTH members on the entropy of the gated prediction (the gate weight is a constant of each step), so it cannot be replayed
from stored logits: each (proxy, b_t, beta) is a live run over the held-out streams. With beta = 0 it IS the joint duo mode with the fixed combination.
The members use one base algorithm's settings (Tent, the learning rates given), the temperatures and the source statistics of the stateful proxies
come from the offline evaluation of that algorithm (its `.temps.json`), and each proxy is run at the b_t that won its Vanilla DD selection.

    --cells proxy:b_t [...]       the proxies to run, each with its b_t (from the Stage E selection)
    --betas B [...]               default 0.1 0.25 0.5 1 2 4; add 0 once for the beta = 0 reference (any cell)
    --eval PREFIX                 the offline evaluation whose temperatures and statistics to use (<dir>/<name>, without .rows.csv)
    python scripts/09_run_joint_dd.py --duo vitb16_rn50v1 --lr_large 5e-5 --lr_small 2.5e-4 --eval out/results/eval_duo_grid/<name> \
        --cells nnn:128 msp:64 --betas 0 0.1 0.25 0.5 1 2 4 --seeds 0 --shard auto

Writes <out>/joint_dd__<duo>__<proxy>_bt<b_t>__beta<beta>__seed<S>.rows.csv (accuracy, ECE, NLL of the gated prediction and of each member, per stream)
and .gate.json (the proxy values and weights of every proxy batch). Launch arrays only with the author's approval (plan.md, Phase 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import cli
from src.experiments.grids import sgd_optim
from src.experiments.joint_dd import run_joint_dd
from src.experiments.logit_runs import stream_segments, with_optim
from src.reliability.proxies.pure import PROXIES


def main() -> int:
    ap = cli.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), tta=False)
    ap.add_argument("--streams", nargs="+", default=["heldout"], metavar="NAME")
    ap.add_argument("--lr_large", type=float, required=True)
    ap.add_argument("--lr_small", type=float, required=True)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eval", required=True, metavar="PREFIX")
    ap.add_argument("--cells", nargs="+", required=True, metavar="PROXY:B_T")
    ap.add_argument("--betas", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    args = ap.parse_args()

    ctx, seeds, out = cli.setup(args, "run_joint_dd")
    ctx = with_optim(ctx, large=sgd_optim(args.lr_large, args.momentum), small=sgd_optim(args.lr_small, args.momentum))
    segments = stream_segments(args.streams, ctx.g)
    meta = json.loads(Path(f"{args.eval}.temps.json").read_text())
    T_l, T_s = meta["temps"]["heldout"]
    stats_l, stats_s = meta["stats"] if meta.get("stats") else (None, None)
    cells = []
    for c in args.cells:
        proxy, _, b_t = c.partition(":")
        if proxy not in PROXIES or not b_t.isdigit():
            ap.error(f"--cells entries are PROXY:B_T with a proxy of {sorted(PROXIES)}, got {c!r}")
        cells.append((proxy, int(b_t)))
    jobs = [(seed, proxy, b_t, beta) for seed in seeds for (proxy, b_t) in cells for beta in args.betas]
    jobs = cli.select_jobs(jobs, lambda j: f"seed{j[0]}/{j[1]}_bt{j[2]}/beta{j[3]:g}", args.shard, args.only)
    print(f"Joint DD on {args.duo}: ViT lr {args.lr_large:g}, ResNet lr {args.lr_small:g}, T_L={T_l:.4f} T_S={T_s:.4f}; {len(segments)} streams x {len(jobs)} runs")
    for seed, proxy, b_t, beta in jobs:
        name = f"joint_dd__{args.duo}__{proxy}_bt{b_t}__beta{beta:g}__seed{seed}"
        if (out / f"{name}.rows.csv").is_file() and not args.num_samples:
            print(f"{name}: done already, skipped")
            continue
        res = run_joint_dd(ctx, segments, gate_kwargs=dict(proxy=proxy, beta=beta, b_t=b_t, T_l=T_l, T_s=T_s, stats_l=stats_l, stats_s=stats_s),
                           seed=seed, num_samples=args.num_samples)
        df = pd.DataFrame(res["rows"])
        df.to_csv(out / f"{name}.rows.csv", index=False)
        (out / f"{name}.gate.json").write_text(json.dumps(res["gate_records"]))
        piv = df.pivot(index="segment", columns="series", values="accuracy")
        print(f"\n== {name}\n" + (100 * piv[["duo", "large", "small"]]).round(2).to_string()
              + "\nmacro: " + ", ".join(f"{s} {100 * piv[s].mean():.2f}" for s in ("duo", "large", "small")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
