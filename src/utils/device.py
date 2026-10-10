"""
device.py
=========
One place that decides which torch device a script runs on.

The old idiom, `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, silently runs on the CPU
when the GPU Slurm allocated is not usable. SLURM job 6012203 did exactly that on kn092 (GPU 2 allocated,
no CUDA device visible to the process): it ran at ~1 img/s (a constant ~68 s per 64-image batch instead
of ~0.3 s) and burned its whole 10 h walltime with nothing in the log to say why.

  * `get_device()`: "cuda" if available. If CUDA is missing but Slurm allocated a GPU to this job, raise
    (set DUO_ALLOW_CPU=1 to run on the CPU on purpose). If CUDA is missing and no GPU was allocated (login
    node, `--gres=gpu:0`, local test), return "cpu" and print one `WARNING` line per process.
  * `describe_device()`: one line for run manifests, e.g. `cuda:0 (NVIDIA L40S) on kn092`.
  * `python -m src.utils.device`: the pre-flight check slurm/job.sh runs; prints the line above and exits 1
    on an allocated-but-unusable GPU. `--selftest` runs this module's tests.
"""

from __future__ import annotations

import os
import socket
import sys

import torch

_warned_cpu = False


def slurm_gpu_allocated() -> bool:
    """True if this process runs inside a Slurm job that was allocated at least one GPU."""
    if not os.environ.get("SLURM_JOB_ID"):
        return False
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS"):
        return True
    try:
        return int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) > 0
    except ValueError:
        return False


def _unusable_gpu_message() -> str:
    host = socket.gethostname()
    return (
        f"no usable CUDA device on {host}, but Slurm allocated a GPU to this job "
        f"(SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')}, "
        f"SLURM_GPUS_ON_NODE={os.environ.get('SLURM_GPUS_ON_NODE')}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}). Running on the CPU would be "
        f"~200x slower and look like a stall. Resubmit with --exclude={host} and report the node, or set "
        f"DUO_ALLOW_CPU=1 to run on the CPU on purpose."
    )


def get_device(requested: str | torch.device | None = None) -> torch.device:
    """The device to run on. `requested` (e.g. a --device flag) is honoured as given, but an explicit
    "cuda" without a usable GPU raises instead of failing later at the first `.to(device)`."""
    global _warned_cpu
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"device {requested!r} requested but torch.cuda.is_available() is False")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if slurm_gpu_allocated() and os.environ.get("DUO_ALLOW_CPU") != "1":
        raise RuntimeError(_unusable_gpu_message())
    if not _warned_cpu:
        _warned_cpu = True
        print(f"WARNING: no CUDA device on {socket.gethostname()}, running on the CPU")
    return torch.device("cpu")


def describe_device(device: str | torch.device | None = None) -> str:
    """One line for a run manifest: `cuda:0 (NVIDIA L40S) on kn092` / `cpu on klogin01`."""
    device = get_device(device)
    host = socket.gethostname()
    if device.type != "cuda":
        return f"{device.type} on {host}"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda:{idx} ({torch.cuda.get_device_name(idx)}) on {host}"


def _selftest() -> None:
    import contextlib
    import io
    from unittest import mock

    global _warned_cpu

    def run(env: dict, cuda: bool, **kw):
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("torch.cuda.is_available", return_value=cuda):
            return get_device(**kw)

    # allocation detection
    for env, want in [
        ({}, False),                                                              # not in a job
        ({"SLURM_JOB_ID": "1"}, False),                                           # CPU job
        ({"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "0"}, False),                # --gres=gpu:0
        ({"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "1"}, True),
        ({"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "2"}, True),                     # job 6012203's env
        ({"SLURM_GPUS_ON_NODE": "1"}, False),                                     # stray var, no job
    ]:
        with mock.patch.dict(os.environ, env, clear=True):
            assert slurm_gpu_allocated() is want, (env, want)

    # CUDA present: always cuda
    assert run({"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "2"}, cuda=True).type == "cuda"
    # allocated GPU but no CUDA: raises, with the escape hatch
    try:
        run({"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "2"}, cuda=False)
        raise AssertionError("allocated GPU without CUDA must raise")
    except RuntimeError as e:
        assert "--exclude=" in str(e) and "DUO_ALLOW_CPU" in str(e)
    # no allocation (or DUO_ALLOW_CPU=1): CPU, with exactly one warning per process
    _warned_cpu = False
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert run({"SLURM_JOB_ID": "1", "SLURM_JOB_GPUS": "2", "DUO_ALLOW_CPU": "1"}, cuda=False).type == "cpu"
        assert run({}, cuda=False).type == "cpu"
        assert run({"SLURM_JOB_ID": "1"}, cuda=False).type == "cpu"
    assert buf.getvalue().count("WARNING") == 1, buf.getvalue()
    # explicit request
    assert run({}, cuda=False, requested="cpu").type == "cpu"
    try:
        run({}, cuda=False, requested="cuda")
        raise AssertionError("explicit cuda without CUDA must raise")
    except RuntimeError:
        pass
    print("src.utils.device self-test OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        try:
            print(f"device: {describe_device()}")
        except RuntimeError as e:
            sys.exit(f"ERROR: {e}")
