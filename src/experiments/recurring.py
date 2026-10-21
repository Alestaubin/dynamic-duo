"""
recurring.py
============
The CoTTA setting for the duo: ONE long stream through the 15 ImageNet-C corruptions in CoTTA's standard order (cfgs/global.yaml's
test corruptions: gaussian, shot, impulse, defocus, glass, motion, zoom, snow, frost, fog, brightness, contrast, elastic, pixelate,
jpeg; or one of its 10 random orders), repeated for R rounds, severity 5, with NO resets (the members start from source weights once
and adapt for the whole stream). Each corruption
is visited with `images` images (the CoTTA protocol's 5,000), and every round draws its own images (the loader seed is
`seed + 1000 * round`, as in docs/design_resets.md D5), so round 0 is the continual stream of that order. The stream length is
R * 15 * images (default 10 * 15 * 5,000 = 750,000 images, 11,730 steps at batch size 64).

Three live runs, each over the identical batches (same loader seeds); every run stores, per sample, whether each prediction was
correct (one byte per sample and series: the plot, the tables and any later window size are sums over it):

  indep      both members adapt on their OWN entropy (Independent both; the indep regime's learning rates). Series
               ts     the static duo (z_L / T_L + z_S / T_S) / 2   = "Fixed TS duo"
               gate   the gated duo, the gate OUTSIDE the loop        = "Vanilla DD"
               large, small   the members alone
  joint_dd   both members adapt on the entropy of the GATED prediction (the gate inside the loop; the joint regime's rates). Series
               gate   the gated duo, pre-update                       = "Joint DD"
               ts, large, small   as above, on this run's members
  frozen     no adaptation (BatchNorm on the test batch's statistics with --bn). Series ts = "Frozen duo", large, small.

The gate is PureProxyGate (the live gate of Joint DD, also what scripts/09 runs): it keeps its buffer across corruptions, so a
proxy batch may straddle a shift and nothing tells it that one happened (src/reset/live.py's convention). `run_recurring` mirrors
`src.reset.live.run_stream` without controller or resets and keeps per-sample correctness instead of logits; the self-test checks
that the two give the same predictions.

File: <dir>/<stream tag>__<run>[__<proxy>_bt<b_t>_beta<beta>].npz (arrays uint8 `ts`, `large`, `small`, `gate`; `part_n`,
`part_round`, `part_label`; `meta`, a JSON string), plus .gate.json (the gate's proxy batches) for the runs that have a gate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.experiments.joint_dd import PureProxyGate, RecordingCalibrator
from src.experiments.members import MemberRunner
from src.experiments.protocols import Segment
from src.experiments.runner import Context
from src.utils.datasets.imagenet_c import ImageNetC

SERIES = ("ts", "gate", "large", "small")


@dataclass(frozen=True)
class Run:
    key: str
    engine: str          # MemberRunner mode
    duo_entropy: bool    # adapts on the duo entropy: runs with the joint regime's learning rates
    gated: bool          # has a gate (the frozen run has none)


RUNS = {r.key: r for r in (Run("indep", "both_indep", False, True), Run("joint_dd", "both_duo", True, True),
                           Run("frozen", "no_adapt", False, False))}


@dataclass(frozen=True)
class Part:
    """One visit of one corruption: `segment` for the images, `round` of the recurrence, `seed` the loader's."""
    segment: Segment
    round: int
    seed: int

    @property
    def label(self) -> str:
        return self.segment.label


def stream_parts(ctx: Context, *, rounds: int, seed: int, order: int | None = None) -> list[Part]:
    """The recurring stream: the 15 corruptions in CoTTA's standard order (`order` None) or in one of its 10 random orders
    (cfgs/protocols.yaml), repeated `rounds` times, one loader seed per round (seed + 1000 * round). Members reset at the start only."""
    g = ctx.g
    corruptions = g.test_corruptions if order is None else g.cotta_orders[order]
    segments = [Segment(f"{c}/s{g.severity}", ImageNetC.from_config(g, c, g.severity), i == 0) for i, c in enumerate(corruptions)]
    return [Part(seg, r, seed + 1000 * r) for r in range(rounds) for seg in segments]


def stream_tag(duo: str, tta: str, bn: bool, rounds: int, images: int, seed: int, order: int | None = None) -> str:
    return f"{duo}__{tta}{'_bn' if bn else ''}__R{rounds}_n{images}_{'std' if order is None else f'order{order}'}_seed{seed}"


def run_path(directory: Path, tag: str, run: str, proxy: str, b_t: int, beta: float) -> Path:
    gate = f"__{proxy}_bt{b_t}_beta{beta:g}" if RUNS[run].gated else ""
    return Path(directory) / f"{tag}__{run}{gate}.npz"


def run_recurring(runner: MemberRunner, parts: Sequence[Part], *, images: int, batch_size: int, workers: int, T_l: float,
                  T_s: float, gate: PureProxyGate | None = None, inside: bool = False, log_every: int = 10,
                  keep_predictions: bool = False) -> dict:
    """Adapt over `parts` (no resets) and return {"correct": {series: uint8 array}, "part_n": [...], "gate_records": [...]}
    (and "predictions": {series: the predicted classes} with `keep_predictions`, for tests: a random tiny model is almost never
    right, so correctness alone cannot tell two runs apart).
    `gate`: None (no gated series), or the gate, outside the loop (`inside` False: it is called here on each batch's logits) or
    the runner's own loop calibrator (`inside` True: the gated prediction is the one the step used, pre-update)."""
    if inside and (gate is None or runner.loop_calibrator is None):
        raise ValueError("a gate inside the loop must be the runner's loop calibrator")
    correct = {s: [] for s in SERIES if s != "gate" or gate is not None}
    predicted = {s: [] for s in correct}
    part_n, n_total = [], sum(min(images, len(p.segment.dataset)) for p in parts)
    if gate is not None:
        gate.records = []
        gate.set_corruption(parts[0].label, total_samples=n_total)       # told where the stream STARTS, once
    t0, seen = time.time(), 0
    for i, part in enumerate(parts):
        if i and gate is not None:
            gate.mark_segment(part.label)                                 # between parts only the label: the buffer stays
        n = 0
        for z_l, z_s, y in runner.iter_batches(part.segment.dataset, batch_size=batch_size, workers=workers, seed=part.seed,
                                               num_samples=images):
            y = y.to(z_l.device)
            with torch.no_grad():
                preds = {"ts": (z_l / T_l + z_s / T_s).argmax(1), "large": z_l.argmax(1), "small": z_s.argmax(1)}
                if gate is not None:
                    z_duo = runner.loop_calibrator.last if inside else gate.calibrate(z_l, z_s)
                    preds["gate"] = z_duo.argmax(1)
            for s, p in preds.items():
                correct[s].append((p == y).to(torch.uint8).cpu())
                if keep_predictions:
                    predicted[s].append(p.cpu())
            n += len(y)
        part_n.append(n)
        seen += n
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(parts)} parts ({part.label}, round {part.round}): {seen} images, "
                  f"{seen / (time.time() - t0):.0f} img/s", flush=True)
    out = {"correct": {s: torch.cat(v).numpy() for s, v in correct.items()}, "part_n": part_n,
           "gate_records": list(gate.records) if gate is not None else []}
    if keep_predictions:
        out["predictions"] = {s: torch.cat(v).numpy() for s, v in predicted.items()}
    return out


def make_runner(ctx: Context, run: Run, gate: PureProxyGate | None, tta: str = "tent") -> tuple[MemberRunner, bool]:
    """(runner, gate inside the loop). Joint DD: the gate is the loop calibrator (through a RecordingCalibrator)."""
    models = ctx.model_factory() if ctx.model_factory else None
    loop = RecordingCalibrator(gate) if run.duo_entropy else None
    return MemberRunner(ctx.duo, tta, {}, run.engine, ctx.device, ctx.objective_logits, models=models,
                        loop_calibrator=loop, bn=ctx.bn), run.duo_entropy


def save_run(path: Path, result: dict, parts: Sequence[Part], meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(tmp, **result["correct"], part_n=np.asarray(result["part_n"], dtype=np.int32),
                        part_round=np.asarray([p.round for p in parts], dtype=np.int32),
                        part_label=np.asarray([p.label for p in parts]), meta=json.dumps(meta))
    tmp.replace(path)                       # atomic: a killed job never leaves a half-written file that looks finished
    if result["gate_records"]:
        path.with_suffix(".gate.json").write_text(json.dumps(result["gate_records"]))
    return path


def load_run(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {s: d[s] for s in SERIES if s in d.files}
    out.update(part_n=d["part_n"], part_round=d["part_round"], part_label=d["part_label"].astype(str),
               meta=json.loads(str(d["meta"])))
    return out


# ------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import copy
    import logging
    import tempfile

    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.reset.live import StreamPart, run_stream
    from src.utils.config import load_global

    logging.disable(logging.INFO)
    torch.set_num_threads(2)
    g = load_global()
    base = tiny_models(0)
    T_L, T_S, BS, B_T, IMAGES = 1.4, 0.9, 32, 64, 100
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = make_context(tmp, batch_size=BS, base=base)
        corruptions = g.cotta_orders[0][:3]
        segs = [Segment(f"{c}/s5", FakeStream(150, seed=i + 1), i == 0) for i, c in enumerate(corruptions)]
        rounds = 2
        parts = [Part(s, r, 3 + 1000 * r) for r in range(rounds) for s in segs]
        assert [p.seed for p in parts] == [3, 3, 3, 1003, 1003, 1003]

        std = stream_parts(ctx, rounds=2, seed=5)
        assert [p.label for p in std[:15]] == [f"{c}/s5" for c in g.test_corruptions] and [p.label for p in std[15:]] == [p.label for p in std[:15]]
        assert [p.round for p in std] == [0] * 15 + [1] * 15 and {p.seed for p in std[:15]} == {5} and {p.seed for p in std[15:]} == {1005}
        assert [p.label.split("/")[0] for p in stream_parts(ctx, rounds=1, seed=0, order=3)] == g.cotta_orders[3]
        assert stream_tag("d", "tent", True, 10, 5000, 0) == "d__tent_bn__R10_n5000_std_seed0"
        assert stream_tag("d", "tent", False, 3, 50000, 1, 4) == "d__tent__R3_n50000_order4_seed1"

        results = {}
        for key, run in RUNS.items():
            gate = PureProxyGate("nnn", beta=2.0, b_t=B_T, T_l=T_L, T_s=T_S, num_classes=1000) if run.gated else None
            runner, inside = make_runner(ctx, run, gate)
            results[key] = run_recurring(runner, parts, images=IMAGES, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, gate=gate,
                                         inside=inside, log_every=0, keep_predictions=True)
            res = results[key]
            n = IMAGES * len(parts)
            assert res["part_n"] == [IMAGES] * len(parts) and set(res["correct"]) == {s for s in SERIES if s != "gate" or run.gated}
            assert all(v.shape == (n,) and v.dtype == np.uint8 and v.max() <= 1 for v in res["correct"].values()), key
            assert bool(res["gate_records"]) == run.gated
        # a frozen member does not move: the same batches give the same predictions in round 0 and in round 1 only if the data
        # are the same, so compare across runs instead: the first batch of every run is seen before any update
        first = {k: v["predictions"]["large"][:BS] for k, v in results.items()}
        assert all(np.array_equal(first["frozen"], first[k]) for k in first), "step 0 predicts before it adapts: all runs agree"
        pred = {k: v["predictions"] for k, v in results.items()}
        assert not np.array_equal(pred["frozen"]["large"], pred["indep"]["large"]), "adaptation changes the large member"
        assert not np.array_equal(pred["frozen"]["small"], pred["indep"]["small"]), "adaptation changes the small member"
        assert not np.array_equal(pred["indep"]["large"], pred["joint_dd"]["large"]), "the two regimes adapt differently"

        # mirror of src.reset.live.run_stream: the same predictions as the library's loop (no controller, no reset)
        for key in ("indep", "joint_dd"):
            run = RUNS[key]
            gate = PureProxyGate("nnn", beta=2.0, b_t=B_T, T_l=T_L, T_s=T_S, num_classes=1000)
            runner, inside = make_runner(ctx, run, gate)
            ref = run_stream(runner, [StreamPart(p.segment, p.seed) for p in parts], batch_size=BS, workers=0, num_samples=IMAGES,
                             gate=None if inside else gate, keep_logits=True)
            y = torch.cat(ref.labels)
            for name, z in (("gate", ref.z_duo), ("large", ref.z_l), ("small", ref.z_s)):
                assert np.array_equal(results[key]["predictions"][name], torch.cat(z).argmax(1).numpy()), (key, name)
                assert np.array_equal(results[key]["correct"][name], (torch.cat(z).argmax(1) == y).numpy().astype(np.uint8))
            want = (torch.cat(ref.z_l) / T_L + torch.cat(ref.z_s) / T_S).argmax(1).numpy()
            assert np.array_equal(results[key]["predictions"]["ts"], want), key
            assert [r["w_l"] for r in ref.gate_records] == [r["w_l"] for r in results[key]["gate_records"]], key

        # beta = 0 inside the loop is the static duo: the gated series equals the ts series, step for step
        gate0 = PureProxyGate("nnn", beta=0.0, b_t=B_T, T_l=T_L, T_s=T_S, num_classes=1000)
        runner, inside = make_runner(ctx, RUNS["joint_dd"], gate0)
        flat = run_recurring(runner, parts, images=IMAGES, batch_size=BS, workers=0, T_l=T_L, T_s=T_S, gate=gate0, inside=True,
                             log_every=0, keep_predictions=True)
        assert np.array_equal(flat["predictions"]["gate"], flat["predictions"]["ts"]), "beta = 0: Joint DD predicts the static duo"
        assert not np.array_equal(results["joint_dd"]["predictions"]["gate"], results["joint_dd"]["predictions"]["ts"]), \
            "beta = 2 weighs the members unequally, so it differs from the static duo somewhere"

        # save / load round trip, and a failure never leaves a finished-looking file
        meta = {"run": "joint_dd", "rounds": rounds}
        p = save_run(run_path(tmp / "o", "tag", "joint_dd", "nnn", B_T, 2.0), results["joint_dd"], parts, meta)
        assert p.name == "tag__joint_dd__nnn_bt64_beta2.npz" and p.with_suffix(".gate.json").is_file()
        assert run_path(tmp / "o", "tag", "frozen", "nnn", B_T, 2.0).name == "tag__frozen.npz", "the frozen run has no gate tag"
        back = load_run(p)
        assert back["meta"] == meta and list(back["part_label"][:3]) == [s.label for s in segs] and list(back["part_round"]) == [0, 0, 0, 1, 1, 1]
        assert all(np.array_equal(back[s], results["joint_dd"]["correct"][s]) for s in ("ts", "gate", "large", "small"))
        assert not list((tmp / "o").glob(".*tmp*")), "no temp file is left behind"
        try:
            run_recurring(MemberRunner(ctx.duo, "tent", {}, "both_indep", "cpu", models=copy.deepcopy(base)), parts, images=IMAGES,
                          batch_size=BS, workers=0, T_l=T_L, T_s=T_S, gate=gate0, inside=True, log_every=0)
            raise AssertionError("a gate inside the loop must be the loop calibrator")
        except ValueError:
            pass
    print("recurring self-test passed")
