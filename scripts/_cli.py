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
import json

from src.tta.methods import TTA_METHODS, resolve_tta_spec


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


def _json_object(text: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not valid JSON: {e}")
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError("must be a JSON object, e.g. '{\"key\": 1}'")
    return obj


def add_tta_args(parser: argparse.ArgumentParser) -> None:
    """--tta_method/--tta_kwargs. Both default to None = "use the duo YAML's
    `TTA:` block, else tent" (see src.tta.methods.resolve_tta_spec) -- so an
    omitted flag never overrides what the config says."""
    parser.add_argument(
        "--tta_method", type=str, default=None, choices=sorted(TTA_METHODS),
        help="Test-time-adaptation method for the models that adapt (see src/tta/methods/). "
             "Default: the duo config's TTA.METHOD, else 'tent'.",
    )
    parser.add_argument(
        "--tta_kwargs", type=_json_object, default=None, metavar="JSON",
        help="Method hyperparameters as a JSON object, e.g. '{\"e_margin\": 2.0}'; merged over "
             "the duo config's TTA.KWARGS (only applied when that block's METHOD is the one "
             "in use). Not needed for tent (no hyperparameters beyond the LARGE/SMALL OPTIM blocks).",
    )


def tta_tag(cfg: dict, tta_method: str | None = None) -> str:
    """Run-name/cache-tag suffix naming the RESOLVED TTA method (explicit
    argument > cfg's TTA.METHOD > default), empty for tent so every
    pre-existing name, directory and cache key stays exactly as it was."""
    name, _ = resolve_tta_spec(cfg, tta_method)
    return f"__{name}" if name != "tent" else ""
