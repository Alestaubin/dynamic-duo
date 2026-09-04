"""
scripts/_cli.py
================
Shared argparse argument groups for scripts/*.py.

Every driver script in this package re-declares the same handful of flags
(--config, --seed, --use_cache/--overwrite_cache, --wandb_project/
--wandb_group, --num_samples, --proto_metric, --out_dir/--run_name) with
defaults that are almost always identical and help text that drifts
slightly each time it's retyped. Each add_*_args(parser, ...) helper below
adds one such group in a single call at the call site; keyword overrides
exist only for the defaults that legitimately differ script to script (a
required vs. optional --config, a different --num_samples cap, ...) — pass
only what actually needs to change, not the whole group.

Flags with genuinely script-specific semantics or long, non-reusable help
text (--mode, --steps, --calib_method, --filter, --gate_beta, ...) are
deliberately NOT here — force-fitting those into a shared helper would lose
real documentation for a cosmetic line-count win. This module only covers
arguments that are actually the same thing everywhere.
"""

from __future__ import annotations

import argparse


def add_duo_config_arg(
    parser: argparse.ArgumentParser, *,
    default: str = "cfgs/dynamic_duo_config.yaml", required: bool = False, help: str | None = None,
) -> None:
    parser.add_argument(
        "--config", type=str, required=required, default=None if required else default,
        help=help or "Duo config -- picks the duo (LARGE/SMALL model names) and the "
             "corruptions/severities to run.",
    )


def add_num_samples_arg(parser: argparse.ArgumentParser, *, default: int | None = 5000) -> None:
    parser.add_argument(
        "--num_samples", type=int, default=default,
        help="Cap on samples per (corruption, severity) stream. Default: all.",
    )


def add_seed_arg(parser: argparse.ArgumentParser, *, default: int | None = 0) -> None:
    parser.add_argument(
        "--seed", type=int, default=default,
        help="Random seed for the ImageNet-C sample subset/ordering.",
    )


def add_cache_toggle_args(parser: argparse.ArgumentParser, *, use_cache_help: str | None = None) -> None:
    """--use_cache/--overwrite_cache. use_cache_help should say exactly WHAT
    gets cached and under what key -- that part is genuinely script-specific
    (raw logits vs. logits+features, keyed by mode/steps or not, ...), so it
    has no generic default; pass it explicitly."""
    parser.add_argument(
        "--use_cache", action="store_true",
        help=use_cache_help or "Cache per-(corruption, severity) results so a repeat run on "
             "the same duo/data skips the model forward pass entirely.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true",
        help="With --use_cache, always recompute and overwrite any existing cache entries "
             "instead of reusing them.",
    )


def add_wandb_project_group_args(
    parser: argparse.ArgumentParser, *,
    default_project: str = "proxy-weighted-duo-calibration", group_help: str | None = None,
) -> None:
    parser.add_argument("--wandb_project", type=str, default=default_project)
    parser.add_argument(
        "--wandb_group", type=str, default=None,
        help=group_help or "Optional shared group tag. Always prefixed with the duo's model "
             "names so two duos' runs can never mix in the same wandb group.",
    )


def add_proto_metric_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--proto_metric", type=str, default="cosine", choices=["cosine", "mahalanobis"],
        help="Only used when the resolved proxy_kind is 'prototype': cosine similarity to "
             "L2-normalised class means, or tied-covariance Mahalanobis to raw class means.",
    )


def add_out_dir_run_name_args(
    parser: argparse.ArgumentParser, *,
    out_dir_default: str, out_dir_help: str | None = None, run_name_help: str | None = None,
) -> None:
    parser.add_argument("--out_dir", type=str, default=out_dir_default, help=out_dir_help)
    parser.add_argument(
        "--run_name", type=str, default=None,
        help=run_name_help or "Subdirectory name under --out_dir. Default: auto-generated.",
    )
