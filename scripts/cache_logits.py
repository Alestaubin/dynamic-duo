#!/usr/bin/env python3
"""
Adapt a duo's members and store their logits per member (Phase 3, Stage B; src/experiments/logit_store.py).

A support script, not a paper result: it fills the cache that the offline evaluator, the proxy selection and the
tables are replayed from. The members never see each other (--mode both_indep), so every member is stored under a key
made of its own settings, and the best large member of one run can be paired with the best small member of another.

    --streams heldout|test|clean|clean_train_slice|<natural dataset>   which streams (default heldout: the 4 held-out
                         corruptions; clean_train_slice: the held-out clean slice of the training images)
    --grid [i ...]       a learning-rate sweep: run i uses the i-th value of EACH model's grid (SGD, momentum 0.9;
                         src/experiments/grids.py); with no index, all five. One job per (seed, grid point).
    --lr_large LR --lr_small LR [--momentum M]   one explicit run (SGD) instead of the tuned settings
    without --grid / --lr_*: the duo's settings as cfgs/duos + cfgs/tuned give them

    python scripts/cache_logits.py --duo vitb16_rn50v1 --streams heldout --grid --seeds 0 --shard auto   # a SLURM array
    python scripts/cache_logits.py --duo vitb16_rn50v1 --streams test clean --seeds 0 1 2 --lr_large 5e-4 --lr_small 2.5e-4
    python scripts/cache_logits.py ... --report_only        # never runs a model: lists what is missing, verifies the rest

Each stream is staged in $SLURM_TMPDIR (when set), copied to the store, its sha256 compared, and only then recorded, so a
job that dies loses at most the stream in flight and a rerun continues. The run manifest (config, git hash, seed, sha256
of every file) goes to <paths.cache>/logits/_runs/. Launch arrays only with the author's approval (plan.md, Phase 3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import cli
from src.experiments.grids import lr_grid, sgd_optim
from src.experiments.logit_runs import ensure_logits, run_id, run_member_specs, stream_segments, with_optim
from src.experiments.logit_store import LogitStore
from src.experiments.specs import MODE_SIDES, DUO_MODES, RunSpec

def main() -> int:
    ap = cli.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--streams", nargs="+", default=["heldout"], metavar="NAME")
    ap.add_argument("--mode", default="both_indep", choices=[m for m in MODE_SIDES if m not in DUO_MODES],
                    help="which members adapt (the joint modes cannot be stored per member)")
    ap.add_argument("--grid", nargs="*", type=int, default=None, metavar="I",
                    help="learning-rate grid points (0-4); no value = all")
    ap.add_argument("--lr_large", type=float, default=None)
    ap.add_argument("--lr_small", type=float, default=None)
    ap.add_argument("--momentum", type=float, default=0.9, help="with --lr_* (default 0.9)")
    ap.add_argument("--staging", default="auto", help="node-local staging dir ('auto' = $SLURM_TMPDIR/logit_stage, "
                                                      "'none' = write straight into the store)")
    args = ap.parse_args()
    if args.grid is not None and (args.lr_large is not None or args.lr_small is not None):
        ap.error("--grid and --lr_large/--lr_small are alternatives")
    if (args.lr_large is None) != (args.lr_small is None) and args.mode == "both_indep":
        ap.error("--lr_large and --lr_small go together")

    ctx, seeds, out = cli.setup(args, "cache_logits")
    segments = stream_segments(args.streams, ctx.g)
    store = LogitStore.from_context(ctx, None if args.staging == "none" else args.staging)

    # jobs: (seed, label, optims)
    jobs = []
    for seed in seeds:
        if args.grid is not None:
            n = len(lr_grid(ctx.duo["LARGE"]["NAME"]))
            for i in (args.grid or range(n)):
                if not 0 <= i < n:
                    ap.error(f"--grid index {i} outside 0..{n - 1}")
                jobs.append((seed, f"seed{seed}/grid{i}",
                             sgd_optim(lr_grid(ctx.duo["LARGE"]["NAME"])[i], args.momentum),
                             sgd_optim(lr_grid(ctx.duo["SMALL"]["NAME"])[i], args.momentum)))
        elif args.lr_large is not None:
            jobs.append((seed, f"seed{seed}/lr{args.lr_large:g}+{args.lr_small:g}",
                         sgd_optim(args.lr_large, args.momentum), sgd_optim(args.lr_small, args.momentum)))
        else:
            jobs.append((seed, f"seed{seed}/tuned", None, None))
    jobs = cli.select_jobs(jobs, lambda j: j[1], args.shard, args.only)
    print(f"{args.duo} {args.tta} {args.mode}: {len(segments)} streams x {len(jobs)} runs, "
          f"{args.num_samples or 'all'} images per stream, store {store.root}")

    status = 0
    for seed, label, opt_l, opt_s in jobs:
        run_ctx = with_optim(ctx, large=opt_l, small=opt_s)
        spec = RunSpec(args.duo, args.tta, args.tta_kwargs or {}, args.mode, "episodic", seed=seed,
                       num_samples=args.num_samples)
        ml, ms = run_member_specs(spec, run_ctx)
        print(f"\n== {label}: large {ml.key} ({ml.model.get('OPTIM', {}).get('LR')}), "
              f"small {ms.key} ({ms.model.get('OPTIM', {}).get('LR')})")
        try:
            man = ensure_logits(spec, run_ctx, segments, store, compute=not args.report_only)
        except FileNotFoundError as e:
            print(f"  MISSING: {e}")
            status = 1
            continue
        for side, m in (("large", ml), ("small", ms)):          # a stored stream must also still match its checksum
            for seg in segments:
                store.verify(m, seg.file_stem)
        total = sum(s[side]["npy_bytes"] for s in man["streams"].values() for side in ("large", "small"))
        print(f"  {len(segments)} streams stored ({total / 1e9:.2f} GB), {man['streams_computed']} computed now "
              f"in {man['seconds']} s, {man['files_written']} files written; manifest {run_id(spec, run_ctx)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
