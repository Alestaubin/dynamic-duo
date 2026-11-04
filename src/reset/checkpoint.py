"""
checkpoint.py
=============
Resumable live runs (`live_runs.run_live(checkpoint=...)`): a CCC stream of millions of images is run as a chain of short Slurm jobs, each resuming where the
last one stopped. A checkpoint holds everything the loop's future depends on:

    members   each model's ADAPTED state only (the parameters the method trains and every buffer, e.g. BatchNorm statistics; the rest is the source weights,
              which a fresh runner loads anyway) and the method's optimizers; any other state of the method (ETA's running average of predictions)
    gate      the gate's whole state (its proxy buffer, last weight, the records so far, the double-assimilation guard)
    trigger   the trigger's whole state (a rule's detectors, a schedule's counter, a random control's draws)
    loop      step, images seen, resets so far, the per-sample correctness so far

The method's source snapshots (`_snapshots`, what a reset restores) are NOT saved: a fresh runner takes them from the same source weights at setup.
State is restored IN PLACE into the live objects (a Joint DD runner's loop calibrator holds a reference to the gate, so the gate object must stay the same).
Writes are atomic (a killed job never leaves a half-written checkpoint).
"""

from __future__ import annotations

from pathlib import Path

import torch

_METHOD_SKIP = {"models", "cfg", "adapting", "_optimizers", "_snapshots"}    # set up again by the fresh runner (or, for the optimizers, saved separately)


def _adapted_state(model: torch.nn.Module) -> dict:
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n for n, _ in model.named_buffers()}
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable or k in buffers}


def runner_state(runner) -> dict:
    tta = runner.tta
    return {"models": {"large": _adapted_state(runner.large), "small": _adapted_state(runner.small)},
            "optimizers": {s: o.state_dict() for s, o in getattr(tta, "_optimizers", {}).items()},
            "method": {k: v for k, v in vars(tta).items() if k not in _METHOD_SKIP}}


def load_runner_state(runner, state: dict) -> None:
    for side, model in (("large", runner.large), ("small", runner.small)):
        missing = set(state["models"][side]) - set(model.state_dict())
        if missing:
            raise ValueError(f"checkpoint holds {side} keys the model does not have: {sorted(missing)[:3]}")
        model.load_state_dict(state["models"][side], strict=False)
    for side, opt_state in state["optimizers"].items():
        runner.tta._optimizers[side].load_state_dict(opt_state)
    vars(runner.tta).update(state["method"])


def save(path: Path, *, runner, gate, trigger, loop: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save({"runner": runner_state(runner), "gate": dict(vars(gate)), "trigger": dict(vars(trigger)), "loop": loop}, tmp)
    tmp.replace(path)
    return path


def load(path: Path, *, runner, gate, trigger) -> dict:
    """Restores the runner, the gate and the trigger in place; returns the loop state."""
    ck = torch.load(Path(path), weights_only=False)                  # every tensor back on the device it was saved from (the gate keeps some on the CPU)
    load_runner_state(runner, ck["runner"])
    vars(gate).update(ck["gate"])
    vars(trigger).update(ck["trigger"])
    return ck["loop"]
