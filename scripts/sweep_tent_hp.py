#!/usr/bin/env python3
"""
scripts/sweep_tent_hp.py
=========================
Hyperparameter search for single-model TENT (src.tta.tent) on the HELD-OUT
corruptions -- --config's CALIBRATOR.CORRUPTIONS / CALIBRATOR.SEVERITIES,
the same set run_tent.py's own --corruptions default EXCLUDES (see its
module docstring) -- never --config's EVAL.CORRUPTIONS. Tuning against the
held-out set rather than the real eval corruptions keeps whatever LR this
sweep settles on from leaking into the numbers a duo/run_tent.py run will
actually be scored on.

Sweeps BOTH TENT optimizer families in one invocation (--optimizers, default
both): Adam's grid (--adam_lrs x --adam_betas) and SGD's grid (--sgd_lrs x
--sgd_momentums), each combination a fully independent TENT trial built via
src.tta.tent.setup_optimizer's own METHOD dispatch. Each trial's headline
metric is macro_accuracy -- the mean of that trial's OWN per-(corruption,
severity) accuracy, not samples pooled together -- so a stream that happens
to get more --num_samples can't dominate the score. This is what "average
accuracy on the held-out corruptions" means throughout this script.

Efficient by keeping each trial cheap, not by sharing work across trials:
the held-out set is deliberately small (4 corruptions in every
cfgs/dynamic_duo_config*.yaml as of this writing), and --num_samples caps
each stream, so a whole 2-optimizer grid costs about (#trials) cheap
single-model TENT passes, not duo-sized ones. Unlike sweep_proxies.py,
model forward passes genuinely CAN'T be shared/cached across trials here --
TENT adaptation permanently mutates a model's norm parameters trial to
trial, exactly why compare_calibrators.py reloads a fresh model per config
(see its module docstring) -- so this script does the same: a fresh
get_model() + setup_tent() per trial, never a reused/reset-in-place model.

Usage
-----
    python scripts/sweep_tent_hp.py --config cfgs/dynamic_duo_config.yaml \\
        --model resnet50 --norm BN --bs 64 \\
        --adam_lrs 0.00005 0.0001 0.00025 0.0005 0.001 \\
        --sgd_lrs 0.0005 0.001 0.005 0.01 --sgd_momentums 0.0 0.9 \\
        --num_samples 2000 --severity 5

    # ViT-B/16, Adam only, small grid, quick smoke test without wandb
    python scripts/sweep_tent_hp.py --config cfgs/dynamic_duo_config.yaml \\
        --model vit_b_16 --norm LN --bs 128 --optimizers Adam \\
        --adam_lrs 0.00005 0.0001 0.0005 --num_samples 500 --no_wandb
"""

from __future__ import annotations

import argparse
import csv
import itertools
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from src.utils.model import get_model, _preprocess_batch
from src.utils.data import load_config, load_imagenetC
from src.utils.metrics import get_metrics_dict
from src.tta.tent import setup_tent
from scripts._cli import (
    add_duo_config_arg, add_num_samples_arg, add_seed_arg, add_wandb_project_group_args,
)


def _build_trials(args: argparse.Namespace) -> list[dict]:
    """Every (METHOD, ...) combo to run this sweep, as a setup_tent-ready
    OPTIM cfg dict (src.tta.tent.setup_optimizer's own expected shape) plus
    a short display name -- Adam's grid and SGD's grid concatenated into one
    list, so both optimizer families are swept together rather than needing
    two separate invocations."""
    trials = []
    if "Adam" in args.optimizers:
        for lr, beta in itertools.product(args.adam_lrs, args.adam_betas):
            trials.append({
                "name": f"Adam_lr{lr:g}_b{beta:g}",
                "cfg": {"METHOD": "Adam", "STEPS": args.steps, "LR": lr, "BETA": beta, "WD": args.wd},
            })
    if "SGD" in args.optimizers:
        for lr, mom in itertools.product(args.sgd_lrs, args.sgd_momentums):
            trials.append({
                "name": f"SGD_lr{lr:g}_m{mom:g}",
                "cfg": {
                    "METHOD": "SGD", "STEPS": args.steps, "LR": lr, "MOMENTUM": mom,
                    "WD": args.wd, "NESTEROV": args.sgd_nesterov, "DAMPENING": args.sgd_dampening,
                },
            })
    return trials


def _run_trial(
    trial: dict, args: argparse.Namespace, cfg: dict, corruptions: list[str],
    severities: list[int], bs: int, device: torch.device, wandb_run,
) -> dict:
    """One fresh TENT adaptation trial across every (corruption, severity) in
    the held-out set, reset between streams exactly like run_tent.py's own
    per-corruption loop (tented_model.reset() -- see src.tta.tent.Tent.reset,
    which restores the model+optimizer to their pre-adaptation state, not
    just re-zeroing gradients). Returns one summary row (per-stream accuracy
    + macro_accuracy); logs the same to `wandb_run` live if given."""
    model, preprocess = get_model(args.model, verbose=False)
    model = model.to(device)
    tented_model = setup_tent(model, norm_type=args.norm, cfg=trial["cfg"])
    tented_model.eval()

    per_stream_acc: dict[str, float] = {}
    for corruption in corruptions:
        for severity in severities:
            tented_model.reset()
            loader = load_imagenetC(
                cfg["TEST_DIR"], severity, [corruption], device=device,
                batch_size=bs, num_workers=cfg.get("WORKERS", 4),
                num_samples=args.num_samples, seed=args.seed,
            )
            all_probs, all_labels = [], []
            for imgs, labels in tqdm(loader, desc=f"{trial['name']} | {corruption}/s{severity}", leave=False):
                x = _preprocess_batch(imgs, preprocess, device)
                z = tented_model.forward(x)
                all_probs.append(F.softmax(z.detach().cpu(), dim=1))
                all_labels.append(labels)
            metrics = get_metrics_dict(torch.cat(all_probs), torch.cat(all_labels))
            stream = f"{corruption}/s{severity}"
            per_stream_acc[stream] = metrics["accuracy"]
            if wandb_run is not None:
                wandb_run.log({f"{stream}/accuracy": metrics["accuracy"]})
            print(f"  [{trial['name']}] {stream}: accuracy={metrics['accuracy']:.4f}")

    macro_accuracy = sum(per_stream_acc.values()) / len(per_stream_acc)
    row = {"trial": trial["name"], "macro_accuracy": macro_accuracy, **trial["cfg"], **per_stream_acc}
    if wandb_run is not None:
        wandb_run.summary["macro_accuracy"] = macro_accuracy
    return row


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_duo_config_arg(
        p, help="Only TEST_DIR/WORKERS/CALIBRATOR.CORRUPTIONS/CALIBRATOR.SEVERITIES are read -- "
                "CALIBRATOR is the HELD-OUT set this sweep is scored on (see module docstring), "
                "never this config's EVAL corruptions."
    )
    p.add_argument("--model", type=str, required=True, help="Model name passed to get_model().")
    p.add_argument("--norm", type=str, required=True, choices=["BN", "LN", "GN"], help="TENT norm type.")
    p.add_argument("--bs", type=int, default=None, help="Adaptation batch size. Defaults to --config's BS, else 64.")
    p.add_argument("--steps", type=int, default=1, help="Adaptation steps per batch, same for every trial.")
    p.add_argument("--wd", type=float, default=0.0, help="Weight decay, shared by both optimizers' grids.")

    p.add_argument("--optimizers", type=str, nargs="+", default=["Adam", "SGD"], choices=["Adam", "SGD"],
                    help="Which optimizer families to include. Default: both.")
    p.add_argument("--adam_lrs", type=float, nargs="+", default=[5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3],
                    help="Adam LR grid.")
    p.add_argument("--adam_betas", type=float, nargs="+", default=[0.9],
                    help="Adam beta1 grid (beta2 fixed at 0.999, per src.tta.tent.setup_optimizer).")
    p.add_argument("--sgd_lrs", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                    help="SGD LR grid.")
    p.add_argument("--sgd_momentums", type=float, nargs="+", default=[0.0, 0.9],
                    help="SGD momentum grid.")
    p.add_argument("--sgd_nesterov", action="store_true", help="Use Nesterov momentum for every SGD trial.")
    p.add_argument("--sgd_dampening", type=float, default=0.0, help="SGD dampening, shared by every SGD trial.")

    p.add_argument("--severities", type=int, nargs="+", default=None,
                    help="Defaults to --config's CALIBRATOR.SEVERITIES.")
    p.add_argument("--corruptions", type=str, nargs="+", default=None,
                    help="Defaults to --config's CALIBRATOR.CORRUPTIONS (the held-out set).")
    add_num_samples_arg(p, default=2000)
    add_seed_arg(p)

    p.add_argument("--csv_path", type=str, default=None,
                    help="Where to write the summary table. Defaults to "
                         "out/tent_hp_sweep_<model>_<timestamp>.csv.")

    wandb_args = p.add_argument_group("wandb options")
    wandb_args.add_argument("--use_wandb", dest="use_wandb", action="store_true", default=True)
    wandb_args.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    add_wandb_project_group_args(
        wandb_args, default_project="dynamic-duos",
        group_help="Optional shared group tag. Always prefixed with --model.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  model: {args.model}")

    cfg = load_config(args.config)
    corruptions = args.corruptions or cfg["CALIBRATOR"]["CORRUPTIONS"]
    severities = args.severities or cfg["CALIBRATOR"]["SEVERITIES"]
    bs = args.bs or cfg.get("BS", 64)

    trials = _build_trials(args)
    if not trials:
        raise ValueError("No trials to run -- check --optimizers/--adam_lrs/--sgd_lrs.")
    print(f"Sweeping {len(trials)} TENT trials for {args.model} on held-out corruptions "
          f"{corruptions} (severities {severities}, bs={bs}, num_samples={args.num_samples})")

    group = f"{args.model}__{args.wandb_group or datetime.now().strftime('%Y%m%d_%H%M%S')}"

    rows: list[dict] = []
    best: dict | None = None
    for trial in tqdm(trials, desc="trials"):
        wandb_run = None
        if args.use_wandb:
            wandb_run = wandb.init(
                project=args.wandb_project, group=group, name=trial["name"], job_type="tent_hp_trial",
                tags=[args.model, args.norm, trial["cfg"]["METHOD"]],
                config={
                    "model": args.model, "norm": args.norm, "bs": bs, "num_samples": args.num_samples,
                    "seed": args.seed, "corruptions": corruptions, "severities": severities, **trial["cfg"],
                },
            )
        row = _run_trial(trial, args, cfg, corruptions, severities, bs, device, wandb_run)
        rows.append(row)
        if wandb_run is not None:
            wandb_run.finish()
        if best is None or row["macro_accuracy"] > best["macro_accuracy"]:
            best = row
        print(f"[{trial['name']}] macro_accuracy={row['macro_accuracy']:.4f}")

    rows.sort(key=lambda r: r["macro_accuracy"], reverse=True)
    print("\n=== Sweep results (best first) ===")
    for r in rows:
        print(f"  {r['trial']:<30s} macro_accuracy={r['macro_accuracy']:.4f}")
    best_cfg = {k: v for k, v in best.items() if k not in ("trial", "macro_accuracy") and "/" not in k}
    print(f"\nBest: {best['trial']}  macro_accuracy={best['macro_accuracy']:.4f}  cfg={best_cfg}")

    csv_path = Path(args.csv_path) if args.csv_path else \
        Path("out") / f"tent_hp_sweep_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(
        {k for row in rows for k in row.keys()},
        key=lambda k: (k != "trial", k != "macro_accuracy", k),
    )
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written to {csv_path}")

    if args.use_wandb:
        summary_run = wandb.init(
            project=args.wandb_project, group=group, name=f"{group}_summary", job_type="summary",
        )
        table = wandb.Table(columns=fieldnames)
        for row in rows:
            table.add_data(*[row.get(c, "") for c in fieldnames])
        summary_run.log({"sweep_results": table})
        summary_run.summary["best_trial"] = best["trial"]
        summary_run.summary["best_macro_accuracy"] = best["macro_accuracy"]
        summary_run.finish()
        print(f"Logged to wandb project '{args.wandb_project}', group '{group}'.")


if __name__ == "__main__":
    main()
