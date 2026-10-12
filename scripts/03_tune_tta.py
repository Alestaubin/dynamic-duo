#!/usr/bin/env python3
"""
Tune Tent's optimizer settings for ONE model on the 4 held-out corruptions (speckle noise, Gaussian blur, spatter,
saturate; severity 5; none is a test corruption), and write the winner to cfgs/tuned/tent__<model>.yaml, where every
experiment script picks it up (src/utils/config.py::load_duo overlays it on the duo YAML's OPTIM).

    --model NAME    any model src/utils/model.py::get_model knows: resnet50, vit_b_16, convnext_tiny, swin_t, ...
                    The norm layer Tent adapts (BN / LN / GN) is read off the model; --norm overrides it.

A trial is one optimizer setting run over the 4 full streams (batch size and stream order as in the protocol; the
model is reset to its source weights before every stream). Score = the mean of the 4 streams' accuracies. The frozen
model (batch statistics, no updates) is run as the reference; the winner must beat it, and a warning says when the
best learning rate is the smallest or largest tried. Default grid: Adam lr 5e-5..1e-3 (beta 0.9) and SGD lr
1e-4..1e-2 with momentum 0 or 0.9, 15 trials + the reference. Each trial is stored on its own, so a killed job resumes
and a SLURM array splits the grid:

    sbatch slurm/job.sh scripts/03_tune_tta.py --model resnet50                       # whole grid in one job
    sbatch --array=0-15 slurm/job.sh scripts/03_tune_tta.py --model vit_b_16 --shard auto   # one trial per task
    python scripts/03_tune_tta.py --model resnet50 --report_only                      # ranking + tuned file from the stored trials
    python scripts/03_tune_tta.py --model resnet50 --optimizers Adam --adam_lrs 1e-4 3e-4 --num_samples 2000   # smoke test

Writes <paths.out>/results/tune_tta/<model>__tent/{trials/*.json, ranking.csv} and, once every trial is in,
<paths.tuned>/tent__<model>.yaml. A capped run (--num_samples) never writes the tuned file: short streams favour
larger learning rates than the 780 updates of a full stream do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import tuning as T
from src.experiments.cli import add_common_args, select_jobs, setup_global
from src.experiments.results import save_rows
from src.utils.datasets import ImageNetC
from src.utils.device import describe_device, get_device
from src.utils.model import get_model


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
                         tta=False, seeds=False, duo=False)
    ap.add_argument("--model", required=True, help="model name as in src/utils/model.py::get_model")
    ap.add_argument("--tta", default="tent", choices=T.TUNABLE_TTA)
    ap.add_argument("--norm", default=None, choices=["BN", "LN", "GN"], help="default: detected from the model")
    ap.add_argument("--optimizers", nargs="+", default=["Adam", "SGD"], choices=["Adam", "SGD"])
    ap.add_argument("--adam_lrs", type=float, nargs="+", default=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    ap.add_argument("--adam_betas", type=float, nargs="+", default=[0.9], help="Adam beta1 (beta2 is 0.999)")
    ap.add_argument("--sgd_lrs", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    ap.add_argument("--sgd_momentums", type=float, nargs="+", default=[0.0, 0.9])
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay, both optimizers")
    ap.add_argument("--seed", type=int, default=0, help="stream order (subset and shuffle) seed")
    args = ap.parse_args()

    g, _, out = setup_global(args, f"tune_tta/{args.model}__{args.tta}")
    device = get_device(args.device)
    print(f"device: {describe_device(device)}")

    make_model = lambda: get_model(args.model, verbose=False)                      # noqa: E731
    norm = args.norm or T.detect_norm(make_model()[0])
    streams = [(f"{c}/s{g.severity}", ImageNetC.from_config(g, c, g.severity)) for c in g.tuning_corruptions]
    labels = [label for label, _ in streams]
    trials = [T.FROZEN, *T.build_trials(args.optimizers, args.adam_lrs, args.adam_betas, args.sgd_lrs,
                                        args.sgd_momentums, args.wd)]
    print(f"{args.model} ({norm}), {args.tta}: {len(trials) - 1} trials + frozen reference on {labels}, "
          f"batch size {g.batch_size}, {args.num_samples or 'all'} images per stream")

    trial_dir = out / "trials"
    key = lambda t: T.trial_key(t, model=args.model, tta=args.tta, norm=norm, stream_labels=labels,        # noqa: E731
                                batch_size=g.batch_size, seed=args.seed, num_samples=args.num_samples)
    if not args.report_only:
        for t in select_jobs(trials, lambda t: t.name, args.shard, args.only):
            if T.load_result(trial_dir, key(t)) is not None:
                print(f"{t.name}: stored, skipped")
                continue
            res = T.run_trial(t, make_model, norm, streams, device=device, batch_size=g.batch_size, workers=g.workers,
                              seed=args.seed, num_samples=args.num_samples, tta=args.tta, ece_bins=g.ece_bins)
            T.store_result(trial_dir, key(t), res)
            print(f"{t.name}: macro accuracy {100 * res['macro_accuracy']:.2f}  "
                  f"(last quarter {100 * res['macro_final_accuracy']:.2f})")

    results = [r for t in trials if (r := T.load_result(trial_dir, key(t))) is not None]
    if len(results) < len(trials):
        done = {r["trial"] for r in results}
        print(f"{len(trials) - len(results)} of {len(trials)} trials not run yet, e.g. "
              f"{[t.name for t in trials if t.name not in done][:3]}")
        return 0 if args.shard or args.only else 1       # a shard only fills the store; the full run ranks

    rows = T.ranking_rows(results)
    save_rows(out / "ranking.csv", rows)
    show = pd.DataFrame(rows)[["trial", "macro_accuracy", "macro_final_accuracy", "macro_nll"]]
    print(show.assign(macro_accuracy=100 * show.macro_accuracy, macro_final_accuracy=100 * show.macro_final_accuracy)
          .round(3).to_string(index=False))

    frozen = next(r for r in results if r["optim"] is None)
    best = T.rank(results)[0]
    for w in T.edge_warnings(best, trials):
        print(f"WARNING: {w}")
    beats = best["macro_accuracy"] > frozen["macro_accuracy"]
    print(f"best: {best['trial']}  {100 * best['macro_accuracy']:.2f} vs frozen {100 * frozen['macro_accuracy']:.2f}")
    if not beats:
        print("WARNING: no setting beats the frozen model; try smaller learning rates")
    if args.num_samples:
        print("capped run (--num_samples): the tuned file is not written")
        return 0
    if not beats:
        print("not writing a tuned file")
        return 1
    path = T.write_tuned(g.path("tuned") / f"{args.tta}__{args.model}.yaml", T.tuned_record(
        best, frozen, model=args.model, tta=args.tta, norm=norm, batch_size=g.batch_size, stream_labels=labels,
        seed=args.seed))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
