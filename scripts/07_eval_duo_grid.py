#!/usr/bin/env python3
"""
Evaluate Vanilla Dynamic Duos offline on a stored pair of members (Phase 3, Stage C; src/experiments/dd_eval.py).

A support script, not a paper table: it reads the members' logits that scripts/cache_logits.py stored, never runs a model, and
writes what proxy selection (Stage E) and the tables are built from: for every proxy x beta x b_t the accuracy, ECE and NLL on
each stream, the members alone, the fixed combinations (clean TS and held-out TS), the oracles, the proxy diagnostics and
the per-bucket records. The temperature sets are fitted here, per pair, on the held-out streams pooled (held-out TS) and on the
clean held-out slice of the TRAINING images (clean TS, and the source statistics of the stateful proxies).

    --lr_large LR --lr_small LR | --grid_large I --grid_small J | (the duo's tuned settings)    which members form the pair
    --test                 also evaluate the 15 test streams (default: the 4 held-out ones only, where selection happens)
    --clean_stream NAME    the clean slice that fits the clean TS and the source statistics (default clean_train_slice;
                           'none' skips the clean TS and the stateful proxies)
    --extra NAME [...]     further stored streams to evaluate (e.g. clean), reported separately
    --proxies / --betas / --b_ts   restrict the grid (default: the brief's: 14 proxies, 7 betas, 6 bucket sizes)

    python scripts/eval_duo_grid.py --duo vitb16_rn50v1 --seeds 0 --grid_large 3 --grid_small 2
    python scripts/eval_duo_grid.py ... --test --extra clean            # with every choice frozen, on the test streams

Output (under <out>, default <paths.out>/results/eval_duo_grid): <run>.rows.csv, .diag.csv, .records.npz, .temps.json.
Needs a GPU for the full grid on 50,000-image streams (a job); the small smoke streams run anywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import cli
from src.experiments.dd_eval import GRID_BETAS, GRID_BT, GridConfig, average_over_streams, pooled_diagnostics, run_pair, save_results
from src.experiments.grids import lr_grid, sgd_optim
from src.experiments.logit_runs import run_member_specs, stream_segments, with_optim
from src.experiments.logit_store import LogitStore, PairView
from src.experiments.specs import RunSpec
from src.reliability.proxies.pure import PROXIES


def main() -> int:
    ap = cli.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--lr_large", type=float, default=None)
    ap.add_argument("--lr_small", type=float, default=None)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--grid_large", type=int, default=None, metavar="I", help="large member = I-th value of its lr grid")
    ap.add_argument("--grid_small", type=int, default=None, metavar="J", help="small member = J-th value of its lr grid")
    ap.add_argument("--test", action="store_true", help="also evaluate the 15 test streams")
    ap.add_argument("--clean_stream", default="clean_train_slice")
    ap.add_argument("--extra", nargs="*", default=[], metavar="NAME")
    ap.add_argument("--proxies", nargs="+", default=list(PROXIES), choices=list(PROXIES))
    ap.add_argument("--betas", nargs="+", type=float, default=list(GRID_BETAS))
    ap.add_argument("--b_ts", nargs="+", type=int, default=list(GRID_BT))
    ap.add_argument("--ts", default="heldout", choices=["heldout", "clean"], help="temperature set of the DD and oracle cells")
    args = ap.parse_args()

    ctx, seeds, out = cli.setup(args, "eval_duo_grid")
    store = LogitStore.from_context(ctx, None)

    def optim(side: str, explicit: float | None, grid: int | None):
        if explicit is not None:
            return sgd_optim(explicit, args.momentum)
        if grid is not None:
            return sgd_optim(lr_grid(ctx.duo[side]["NAME"])[grid], args.momentum)
        return None

    ctx_l = with_optim(ctx, large=optim("LARGE", args.lr_large, args.grid_large))
    ctx_s = with_optim(ctx, small=optim("SMALL", args.lr_small, args.grid_small))
    held = [s.file_stem for s in stream_segments(["heldout"], ctx.g)]
    test = [s.file_stem for s in stream_segments(["test"], ctx.g)] if args.test else []
    extra = [s.file_stem for s in stream_segments(args.extra, ctx.g)] if args.extra else []
    clean = None if args.clean_stream == "none" else stream_segments([args.clean_stream], ctx.g)[0].file_stem
    cfg = GridConfig(proxies=tuple(args.proxies), betas=tuple(args.betas), b_ts=tuple(args.b_ts), ts=args.ts)

    status = 0
    for seed in seeds:
        spec = RunSpec(args.duo, args.tta, args.tta_kwargs or {}, "both_indep", "episodic", seed=seed,
                       num_samples=args.num_samples)
        ml, _ = run_member_specs(spec, ctx_l)
        _, ms = run_member_specs(spec, ctx_s)
        name = f"{args.duo}__{args.tta}__seed{seed}__{ml.key}_{ms.key}"
        pair = PairView(store, ml, ms)
        have = set(pair.stems())
        need = [*held, *test, *extra, *([clean] if clean else [])]
        if missing := [s for s in need if s not in have]:
            print(f"seed {seed}: streams missing from the store for large {ml.key} / small {ms.key}: {missing} "
                  f"(scripts/cache_logits.py fills them)")
            status = 1
            continue
        res = run_pair(pair, held, test, cfg, clean_stem=clean, extra=extra, device=args.device)
        files = save_results(out, name, res)
        print(f"\n== seed {seed}: large {ml.key} (lr {ml.model['OPTIM']['LR']}), small {ms.key} (lr {ms.model['OPTIM']['LR']})")
        print(f"temperatures: " + ", ".join(f"{k} T_L={a:.3f} T_S={b:.3f}" for k, (a, b) in res["temps"].items()))
        avg = average_over_streams(res["rows"], held)
        pd.set_option("display.width", 200)
        show = avg[avg.series.isin(["large", "small", "fixed"])][["series", "ts", "accuracy", "ece", "nll"]]
        print("held-out macro average, members and fixed combinations:\n" + show.round(4).to_string(index=False))
        dd = avg[avg.series == "dd"]
        diag = pooled_diagnostics(res["records"], held)
        best = dd.loc[dd.groupby("proxy").accuracy.idxmax()][["proxy", "beta", "b_t", "accuracy", "nll"]]
        best = best.merge(diag[["proxy", "b_t", "ordering_acc", "spearman"]], on=["proxy", "b_t"], how="left")
        print("best (beta, b_t) per proxy on the held-out streams:\n" + best.sort_values("accuracy", ascending=False).round(4).to_string(index=False))
        for s in ("oracle_select", "oracle_w"):
            o = avg[avg.series == s]
            print(f"{s}: " + ", ".join(f"b_t={int(r.b_t)} acc {r.accuracy:.4f}" for r in o.itertuples()))
        print("files: " + ", ".join(Path(f).name for f in files.values()))
    return status


if __name__ == "__main__":
    sys.exit(main())
