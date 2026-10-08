"""
cli.py
======
The flags every script in scripts/ takes, and the setup they share.

    --duo NAME          cfgs/duos/<NAME>.yaml                          (default vitb16_rn50)
    --tta METHOD        tent | eata | ...; --tta_kwargs '{"k": v}'      (default tent)
    --seeds 0 1 2       stream-order seeds                              (default: cfgs/global.yaml)
    --num_samples N     cap per stream, for smoke tests                 (default: the whole stream)
    --report_only       never run a model: build the table/figure from what is cached, or fail
                        listing what is missing
    --shard i/N         run only every N-th job, starting at i. "auto" reads $SLURM_ARRAY_TASK_ID and
                        $SLURM_ARRAY_TASK_COUNT, so `sbatch --array=0-8 slurm/job.sh script.py --shard auto`
                        splits the work over the array
    --only TEXT [...]   run only the jobs whose label contains one of these substrings
    --global PATH       alternative cfgs/global.yaml (also $DUO_GLOBAL_CONFIG)
    --out DIR           where this script's CSV/tables go (default: <paths.out>/results/<script>)
    --device cuda|cpu

A script's work is a list of independent jobs (one per run it needs); `select_jobs` applies --shard and
--only to it, so every script splits over a SLURM array the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import torch

from src.experiments.runner import Context
from src.utils.config import load_global

T = TypeVar("T")


def _json_object(text: str) -> dict:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return obj


def add_common_args(p: argparse.ArgumentParser, *, tta: bool = True, seeds: bool = True) -> argparse.ArgumentParser:
    p.add_argument("--duo", default="vitb16_rn50", help="cfgs/duos/<name>.yaml")
    if tta:
        p.add_argument("--tta", default="tent", help="TTA method of the adapting members")
        p.add_argument("--tta_kwargs", type=_json_object, default=None, metavar="JSON")
    if seeds:
        p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: protocol.seeds in global.yaml")
    p.add_argument("--num_samples", type=int, default=None, help="cap per stream (smoke tests); default: all")
    p.add_argument("--report_only", action="store_true", help="never run a model; fail if the cache is incomplete")
    p.add_argument("--shard", default=None, metavar="i/N|auto", help="run only jobs i, i+N, i+2N, ...")
    p.add_argument("--only", nargs="+", default=None, metavar="TEXT", help="run only jobs whose label contains TEXT")
    p.add_argument("--global", dest="global_path", default=None, help="alternative global config")
    p.add_argument("--out", default=None, help="output directory for this script's CSV/tables")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    return p


def parse_shard(text: str | None) -> tuple[int, int]:
    """'i/N' -> (i, N); 'auto' -> from the SLURM array environment; None -> (0, 1)."""
    if text is None:
        return 0, 1
    if text == "auto":
        i, n = os.environ.get("SLURM_ARRAY_TASK_ID"), os.environ.get("SLURM_ARRAY_TASK_COUNT")
        if i is None or n is None:
            raise ValueError("--shard auto needs a SLURM job array ($SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT)")
        return int(i), int(n)
    i, _, n = text.partition("/")
    shard = (int(i), int(n))
    if not (shard[1] >= 1 and 0 <= shard[0] < shard[1]):
        raise ValueError(f"--shard must satisfy 0 <= i < N, got {text!r}")
    return shard


def select_jobs(jobs: Sequence[T], label: Callable[[T], str], shard: str | None = None,
                only: Sequence[str] | None = None) -> list[T]:
    """Apply --only (substring filter on each job's label), then --shard (every N-th of what remains)."""
    jobs = [j for j in jobs if not only or any(o in label(j) for o in only)]
    i, n = parse_shard(shard)
    return [j for k, j in enumerate(jobs) if k % n == i]


def pin_threads() -> None:
    """Match torch's CPU threads to the allocation. Left alone, torch uses every core of the node and a
    128x1000 SVD on a busy node runs thousands of times slower than with 4 threads."""
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    torch.set_num_threads(int(n) if n else min(8, os.cpu_count() or 1))


def setup(args: argparse.Namespace, script: str) -> tuple[Context, list[int], Path]:
    """(context, seeds, output dir) for a script."""
    pin_threads()
    g = load_global(args.global_path)
    ctx = Context.create(args.duo, g, args.device)
    seeds = list(args.seeds) if getattr(args, "seeds", None) else g.seeds
    out = Path(args.out) if args.out else g.path("out") / "results" / script
    out.mkdir(parents=True, exist_ok=True)
    return ctx, seeds, out


if __name__ == "__main__":
    assert parse_shard(None) == (0, 1) and parse_shard("2/9") == (2, 9)
    for bad in ("9/9", "-1/3", "1/0"):
        try:
            parse_shard(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    try:
        parse_shard("auto")
        raise AssertionError("auto outside a SLURM array must raise")
    except ValueError:
        pass
    os.environ.update(SLURM_ARRAY_TASK_ID="3", SLURM_ARRAY_TASK_COUNT="4")
    assert parse_shard("auto") == (3, 4)

    jobs = [f"tent/seed{s}/{c}" for s in range(3) for c in ("fog", "snow", "contrast")]
    got = [select_jobs(jobs, str, shard=f"{i}/4") for i in range(4)]
    assert sorted(sum(got, [])) == sorted(jobs) and all(set(a).isdisjoint(b) for a in got for b in got if a is not b), \
        "shards must partition the jobs"
    assert select_jobs(jobs, str, only=["fog"]) == [j for j in jobs if "fog" in j]
    assert select_jobs(jobs, str, shard="0/2", only=["seed1", "seed2"]) == [j for j in jobs if "seed1" in j or "seed2" in j][::2]

    p = add_common_args(argparse.ArgumentParser())
    a = p.parse_args(["--seeds", "0", "1", "--shard", "1/3", "--tta_kwargs", '{"d_margin": 0.1}'])
    assert a.seeds == [0, 1] and a.tta == "tent" and a.tta_kwargs == {"d_margin": 0.1} and not a.report_only
    print("cli self-test passed")
