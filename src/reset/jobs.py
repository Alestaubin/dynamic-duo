"""
jobs.py
=======
The units of work of the reset tuning, and how they are submitted (no torch: the driver and the report scripts import it cheaply).

A JOB is one run: a kind and its arguments, written to a small JSON spec that `scripts/tune_reset.py --job <spec>` executes inside a Slurm job
(src/reset/workers.py):

    temperatures   the frozen-model fit of the duo's two temperatures (once per duo)
    continual      one rung of the continual-rate ladder, or its frozen reference (what 04b did)
    calibration    one calibration stream of the screens: stress S or healthy H on the held-out stream (what 04c did)
    live           one live reset run: a condition (`none`, `fixed1000`, `own_fast_vs_slow@pm...`, `random~...`, ...) on one stream at two rates (what 38 did)

Submission is idempotent: the job's name carries a hash of its spec, and a job whose name is already in the queue (pending or running) is not
submitted again, so a wave can be requested twice (by hand and by a monitor) without double runs. `live_command` is what the final-test and
gate-study scripts use to submit a live run.

`chain` (`tune_reset.py --submit --chain`) lets a tuning run itself: after a wave it submits one small CPU job that waits for the wave (`afterok`) and calls the
driver again, which submits the next wave and chains again, until the tuning is done or needs a decision. A failed job leaves the chain cancelled, not retrying.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.reset.layout import Paths

SCRIPT = "scripts/tune_reset.py"
TIME = {"temperatures": "01:00:00", "continual": "01:35:00", "calibration": "02:15:00", "live": "01:50:00"}
CHAIN_PREFIX = "rt_next_"                                # the name of the job that calls the driver again after a wave
CHAIN_TIME = "02:00:00"                                  # it may run CPU stages inline (the screens: about a minute each)
EXCLUDE = "kn117,kn101"                                  # nodes with an uncorrectable GPU ECC error (plan.md, rules for every chat)
KINDS = tuple(TIME)


@dataclass(frozen=True)
class Job:
    kind: str
    duo: str
    args: dict = field(default_factory=dict)             # JSON-able: what the worker needs
    label: str = ""                                      # a few readable characters in the job name

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown job kind {self.kind!r}: {KINDS}")

    @property
    def key(self) -> str:
        return hashlib.sha1(json.dumps({"kind": self.kind, "duo": self.duo, "args": self.args}, sort_keys=True).encode()).hexdigest()[:8]

    @property
    def name(self) -> str:
        label = re.sub(r"[^A-Za-z0-9]+", "", self.label)[:14]
        return f"rt_{self.kind[:4]}_{label + '_' if label else ''}{self.key}"

    def spec(self) -> dict:
        return {"kind": self.kind, "duo": self.duo, "args": self.args}


# ------------------------------------------------------------------------------------------------------------------ the jobs

def temperatures_job(duo: str) -> Job:
    return Job("temperatures", duo, {}, "temps")


def _signal_args(signal: str, beta: float | None) -> dict:
    """The joint signal's extra arguments (the independent signal's specs, and so its job names, are unchanged by it)."""
    if signal == "indep":
        return {}
    if signal != "joint":
        raise ValueError(f"signal must be 'indep' or 'joint', got {signal!r}")
    return {"signal": "joint", **({} if beta is None else {"beta": float(beta)})}


def continual_job(duo: str, *, seed: int, lrs: Sequence[float] | None, rounds: int = 4, images: int = 50000, momentum: float = 0.9, signal: str = "indep",
                  beta: float = 0.5) -> Job:
    """One rung of the continual-rate ladder (`lrs` = (lr_large, lr_small)), or the frozen reference (None). Joint: the gate (`beta`) sits inside the loss."""
    args = {"seed": seed, "lrs": None if lrs is None else [float(lrs[0]), float(lrs[1])], "rounds": rounds, "images": images, "momentum": momentum,
            **_signal_args(signal, beta if lrs is not None else None)}
    return Job("continual", duo, args, ("j" if signal == "joint" else "") + ("frozen" if lrs is None else f"c{lrs[0]:.2g}"))


def calibration_job(duo: str, *, name: str, lrs: Sequence[float], seed: int, rotation: int, stop_mode: str, after: int, beta: float, b_t: int = 256,
                    proxy: str = "nnn", rounds: int = 4, images: int = 50000, momentum: float = 0.9, signal: str = "indep") -> Job:
    """One calibration stream on the held-out stream: both members at `lrs`, stopped `after` images once a member is dead (`stop_mode` any / all)."""
    args = {"name": name, "lrs": [float(lrs[0]), float(lrs[1])], "seed": seed, "rotation": rotation, "stop_mode": stop_mode, "after": after,
            "beta": beta, "b_t": b_t, "proxy": proxy, "rounds": rounds, "images": images, "momentum": momentum, **_signal_args(signal, None)}
    return Job("calibration", duo, args, ("j" if signal == "joint" else "") + f"{name}r{rotation}s{seed}")


def live_job(duo: str, cond: str, *, stream: str, lrs: Sequence[float], rotation: int = 0, order: int | None = None, seed: int = 0, beta: float = 0.5,
             b_t: int = 256, proxy: str | None = None, images: int = 50000, max_images: int | None = None, extra_gates: Sequence[str] = (),
             budget: float = 1.0, rounds: int = 4, momentum: float = 0.9, label: str | None = None, signal: str = "indep") -> Job:
    """One live reset run of condition `cond` (scripts' conditions: src/reset/workers.py, `make_trigger`): stream "heldout" (`rotation`) or "test" (`order`)."""
    args = {"cond": cond, "stream": stream, "lrs": [float(lrs[0]), float(lrs[1])], "rotation": rotation, "order": order, "seed": seed, "beta": beta, "b_t": b_t,
            "proxy": proxy, "images": images, "max_images": max_images, "extra_gates": list(extra_gates), "budget": budget, "rounds": rounds, "momentum": momentum,
            **_signal_args(signal, None)}
    default = f"r{rotation}" if stream == "heldout" else f"o{'std' if order is None else order}"
    return Job("live", duo, args, ("j" if signal == "joint" else "") + (label if label is not None else default))


# ------------------------------------------------------------------------------------------------------------------ submission

def sbatch_argv(job: Job, spec_path: Path, extra: Sequence[str] = (), time: str | None = None) -> list[str]:
    return ["sbatch", "-t", time or TIME[job.kind], "-J", job.name, f"--exclude={EXCLUDE}", *extra, "slurm/job.sh", SCRIPT, "--job", str(spec_path)]


def queued_jobs() -> dict[str, list[str]]:
    """This user's pending and running Slurm jobs: name -> job ids (empty, with a warning, where squeue is not available)."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j"], capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        print(f"warning: squeue failed ({e}); jobs already queued cannot be told apart", file=sys.stderr)
        return {}
    jobs: dict[str, list[str]] = {}
    for line in out.splitlines():
        if "|" in line:
            i, name = line.strip().split("|", 1)
            jobs.setdefault(name, []).append(i)
    return jobs


def queued_names() -> set[str]:
    """The names of this user's pending and running Slurm jobs."""
    return set(queued_jobs())


def write_spec(paths: Paths, job: Job) -> Path:
    paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    p = paths.jobs_dir / f"{job.name}.json"
    text = json.dumps(job.spec(), indent=1, sort_keys=True)
    if not p.exists() or p.read_text() != text:
        p.write_text(text)
    return p


def submit(paths: Paths, jobs: Sequence[Job], *, go: bool, force: bool = False, extra: Sequence[str] = (), cwd: Path | None = None) -> int:
    """Print the sbatch command of every job; with `go` submit those that are not already queued. Returns the number submitted."""
    queued = set() if (force or not go) else queued_names()
    n = 0
    for job in jobs:
        if job.name in queued:
            print(f"queued   {job.name}")
            continue
        spec = write_spec(paths, job) if go else paths.jobs_dir / f"{job.name}.json"
        argv = sbatch_argv(job, spec, extra)
        print(("SUBMIT  " if go else "to run  ") + " ".join(argv))
        if go:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
            print("   ->", (r.stdout or r.stderr).strip())
            n += r.returncode == 0
    return n


def chain(paths: Paths, jobs: Sequence[Job], driver_args: Sequence[str], *, go: bool, tag: str = "", cwd: Path | None = None) -> bool:
    """Submit one small CPU job that runs `scripts/tune_reset.py <driver_args>` once every job of this wave has finished OK (`afterok`: a failed job cancels the
    chain, so a broken wave is not retried forever). One chain job per duo, signal and `tag`; one already queued is left alone. Called from a chain job itself,
    it refuses to chain a wave identical to the previous one (the jobs ran and changed nothing). Returns whether a chain job is now waiting."""
    name = CHAIN_PREFIX + paths.name + (f"_{re.sub(r'[^A-Za-z0-9]+', '', tag)}" if tag else "")
    names = sorted(j.name for j in jobs)
    state = paths.jobs_dir / f"chain__{name[len(CHAIN_PREFIX):]}.json"
    from_chain = os.environ.get("SLURM_JOB_NAME", "").startswith(CHAIN_PREFIX)
    if from_chain and state.is_file() and json.loads(state.read_text()).get("jobs") == names:
        print(f"  CHAIN STOPPED: this wave is the one the last chain call submitted ({len(names)} job(s) ran and changed nothing); see their logs")
        return False
    running = queued_jobs() if go else {}
    me = os.environ.get("SLURM_JOB_ID")
    if any(i != me for i in running.get(name, [])):
        print(f"queued   {name}")
        return True
    ids = sorted({i for j in jobs for i in running.get(j.name, [])}) if go else ["<wave>"]
    if not ids:
        print("  CHAIN NOT SUBMITTED: none of the wave's jobs is in the queue to wait for")
        return False
    argv = ["sbatch", "-t", CHAIN_TIME, "-J", name, f"--dependency=afterok:{':'.join(ids)}", "--kill-on-invalid-dep=yes", "slurm/job_cpu.sh", SCRIPT, *driver_args]
    print(("SUBMIT  " if go else "to run  ") + " ".join(argv))
    if not go:
        return True
    r = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
    print("   ->", (r.stdout or r.stderr).strip())
    if r.returncode == 0:
        paths.jobs_dir.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"jobs": names, "after": ids}))
    return r.returncode == 0


def live_command(paths: Paths, job: Job, extra: Sequence[str] = (), time: str | None = None, write: bool = True) -> list[str]:
    """The sbatch argv of a live run (for the final-test and gate-study scripts, which build their own submissions); the spec is written unless `write` is False
    (a dry run: the path it would have is still in the command)."""
    return sbatch_argv(job, write_spec(paths, job) if write else paths.jobs_dir / f"{job.name}.json", extra, time)


if __name__ == "__main__":
    import tempfile

    from src.utils.config import load_global

    a = live_job("convnextb_deits", "own_fast_vs_slow@pmL0.05-0.005-20-0.7_S0.05-0.005-20-0.7", stream="heldout", lrs=(1.25e-5, 8e-4), rotation=2)
    b = live_job("convnextb_deits", "own_fast_vs_slow@pmL0.05-0.005-20-0.7_S0.05-0.005-20-0.7", stream="heldout", lrs=(1.25e-5, 8e-4), rotation=2)
    c = live_job("convnextb_deits", "own_fast_vs_slow@pmL0.05-0.005-20-0.7_S0.05-0.005-20-0.7", stream="heldout", lrs=(1.25e-5, 8e-4), rotation=0)
    assert a.name == b.name and a.name != c.name and a.name.startswith("rt_live_r2_") and len(a.name) < 40, a.name
    assert continual_job("x", seed=0, lrs=None).label == "frozen" and temperatures_job("x").kind == "temperatures"
    # the joint signal: its own specs and names; the independent signal's are exactly what they were
    ji = live_job("x", "none", stream="heldout", lrs=(1e-4, 5e-4), rotation=0)
    assert "signal" not in ji.args and ji.name == live_job("x", "none", stream="heldout", lrs=(1e-4, 5e-4), rotation=0, signal="indep").name
    jj = live_job("x", "none", stream="heldout", lrs=(1e-4, 5e-4), rotation=0, signal="joint", beta=0.75)
    assert jj.args["signal"] == "joint" and jj.name != ji.name and jj.label == "jr0" and "_jr0_" in jj.name
    assert continual_job("x", seed=0, lrs=(1e-4, 5e-4), signal="joint", beta=0.25).args["beta"] == 0.25 and "signal" not in continual_job("x", seed=0, lrs=(1e-4, 5e-4)).args
    assert "beta" not in continual_job("x", seed=0, lrs=None, signal="joint").args, "the frozen reference has no gate"
    assert calibration_job("x", name="S", lrs=(1e-4, 5e-4), seed=0, rotation=1, stop_mode="all", after=1, beta=0.5, signal="joint").args["signal"] == "joint"
    assert calibration_job("x", name="S", lrs=(1e-4, 5e-4), seed=0, rotation=1, stop_mode="all", after=20000, beta=0.5).label == "Sr1s0"
    try:
        Job("nonsense", "x")
        raise AssertionError("an unknown kind must be refused")
    except ValueError:
        pass
    g = load_global()
    with tempfile.TemporaryDirectory() as tmp:
        P = Paths(g, "convnextb_deits")
        P = Paths(type("G", (), {"path": lambda self, k, _t=tmp: Path(_t)})(), "convnextb_deits")          # a scratch output folder
        spec = write_spec(P, a)
        assert json.loads(spec.read_text()) == a.spec() and spec.parent == P.jobs_dir
        argv = live_command(P, a)
        assert argv[:2] == ["sbatch", "-t"] and argv[argv.index("-J") + 1] == a.name and argv[-3:-1] == [SCRIPT, "--job"] and argv[-1] == str(spec)
        assert submit(P, [a, c], go=False) == 0                     # a dry run submits nothing and writes nothing new
        # the chain: a dry run prints its command and writes nothing; called from a chain job it refuses a wave identical to the last one it submitted
        PJ = Paths(P.g, "convnextb_deits", "joint")
        assert chain(PJ, [a], ["--duo", "convnextb_deits", "--submit", "--chain"], go=False) and not list(Path(tmp).rglob("chain__*"))
        PJ.jobs_dir.mkdir(parents=True, exist_ok=True)
        (PJ.jobs_dir / "chain__convnextb_deits__joint.json").write_text(json.dumps({"jobs": [a.name]}))
        old = os.environ.get("SLURM_JOB_NAME")
        os.environ["SLURM_JOB_NAME"] = CHAIN_PREFIX + "convnextb_deits__joint"
        try:
            assert chain(PJ, [a], [], go=False) is False and chain(PJ, [c], [], go=False) is True
        finally:
            os.environ.pop("SLURM_JOB_NAME") if old is None else os.environ.__setitem__("SLURM_JOB_NAME", old)
    print("jobs self-test passed")
