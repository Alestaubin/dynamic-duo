"""
tuning.py
=========
The logic behind scripts/02_tune_tta.py: tune a TTA method's optimizer settings for ONE model on the held-out
corruptions (corruptions.tuning in cfgs/global.yaml: speckle noise, Gaussian blur, spatter, saturate; none of them is
a test corruption), then hand the winner to every experiment through cfgs/tuned/<tta>__<model>.yaml
(src/utils/config.py::load_duo overlays it on the duo YAML's OPTIM).

The model is adapted exactly as it is inside the duo's independent modes: the real TTAMethod (src/tta/methods) set
up on the model as its "large" side with the other side a placeholder, one update per batch of cfgs/global.yaml's
adaptation batch size, source weights restored before every stream (episodic), pre-update logits scored. So what is
tuned is the code the experiments run, not a copy of it. Tent's only hyperparameters are the optimizer's.

A trial is one optimizer setting run over the 4 streams; its score is the macro accuracy (the mean of the 4 streams'
accuracies, so no stream weighs more because it has more images). Results are stored one JSON per trial, which makes
the sweep resumable and shardable over a SLURM array; the ranking is built from them. Tuning on full streams matters:
a learning rate that wins on a few hundred images (about 5 updates) is typically too large over the 780 updates of
a 50,000-image stream, so `num_samples` below the stream size is for smoke tests only and never writes cfgs/tuned.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.experiments.evaluate import StreamMetrics
from src.tta.methods import build_tta_method
from src.utils.datasets.base import ShiftDataset

TUNABLE_TTA = ("tent",)


# -------------------------------------------------------------------------------------------------------- grids

@dataclass(frozen=True)
class Trial:
    name: str                 # also the --only / --shard label
    optim: dict | None        # an OPTIM block as src/tta/tent.py::setup_optimizer reads it; None = the frozen reference


FROZEN = Trial("frozen", None)


def build_trials(optimizers: Sequence[str], adam_lrs: Sequence[float], adam_betas: Sequence[float],
                 sgd_lrs: Sequence[float], sgd_momentums: Sequence[float], wd: float = 0.0) -> list[Trial]:
    trials = []
    if "Adam" in optimizers:
        for lr, beta in itertools.product(adam_lrs, adam_betas):
            trials.append(Trial(f"Adam_lr{lr:g}_b{beta:g}",
                                {"METHOD": "Adam", "STEPS": 1, "LR": lr, "BETA": beta, "WD": wd}))
    if "SGD" in optimizers:
        for lr, mom in itertools.product(sgd_lrs, sgd_momentums):
            trials.append(Trial(f"SGD_lr{lr:g}_m{mom:g}",
                                {"METHOD": "SGD", "STEPS": 1, "LR": lr, "MOMENTUM": mom, "DAMPENING": 0.0,
                                 "WD": wd, "NESTEROV": False}))
    if not trials:
        raise ValueError("no trials: check --optimizers and the learning-rate lists")
    if len({t.name for t in trials}) != len(trials):
        raise ValueError("duplicate trials in the grid")
    return trials


def detect_norm(model: nn.Module) -> str:
    """The normalization layer type Tent adapts in `model`: "BN" (BatchNorm2d), "LN" (LayerNorm, which includes
    torchvision's LayerNorm2d) or "GN". Raises if the model has none or several kinds (pass --norm then): a wrong
    choice makes Tent adapt nothing, or adapt the wrong parameters, silently."""
    kinds = {"BN": nn.BatchNorm2d, "LN": nn.LayerNorm, "GN": nn.GroupNorm}
    found = [k for k, cls in kinds.items() if any(isinstance(m, cls) for m in model.modules())]
    if len(found) != 1:
        raise ValueError(f"cannot pick the norm type automatically: the model has {found or 'no BN/LN/GN layers'}; "
                         f"pass it explicitly")
    return found[0]


# ----------------------------------------------------------------------------------------------------- one run

class SingleCollate:
    """list of (PIL, label) -> (preprocessed batch, labels), run inside the DataLoader workers."""

    def __init__(self, preprocess):
        self.pre = preprocess

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        return torch.stack([self.pre(i) for i in imgs]), torch.tensor(labels)


class SingleModel:
    """One model adapted by a TTAMethod: the method is set up on it as the "large" side, the other side being a
    placeholder that is never run. `optim=None` freezes it, no updates: the reference. A frozen BatchNorm model
    uses the test batch's statistics with `bn` (--bn), else the usual eval() mode on the source statistics; an
    adapting model always uses batch statistics."""

    def __init__(self, model: nn.Module, preprocess, norm: str, optim: dict | None, device: torch.device,
                 tta: str = "tent", bn: bool = False):
        self.model, self.preprocess, self.device = model.to(device), preprocess, device
        self.adapt = optim is not None
        side = {"NAME": "model", "NORM": norm, "OPTIM": optim or {}}
        self.tta = build_tta_method(tta)
        self.tta.setup(self.model, nn.Identity(), {"LARGE": side, "SMALL": {**side, "NAME": "placeholder"}},
                       adapt_large=self.adapt, adapt_small=False, bn=bn)

    def reset(self) -> None:
        if self.adapt:
            self.tta.reset()

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Forward one preprocessed batch, adapt, return the PRE-update logits."""
        with torch.set_grad_enabled(self.adapt):
            z = self.model(x.to(self.device, non_blocking=True))
            if self.adapt:
                self.tta.update(self.tta.loss(z, "large"), ["large"])
        return z.detach()


def run_stream(runner: SingleModel, dataset: ShiftDataset, *, batch_size: int, workers: int, seed: int,
               num_samples: int | None, ece_bins: int = 15, desc: str = "") -> dict:
    """Source weights, then one pass over the stream. Metrics of the pre-update logits; `final_accuracy` is the
    accuracy over the last quarter of the batches, where a collapsing run shows."""
    runner.reset()
    metrics = StreamMetrics(ece_bins)
    loader = dataset.loader(batch_size=batch_size, num_samples=num_samples, seed=seed, workers=workers,
                            pin_memory=runner.device.type == "cuda", collate_fn=SingleCollate(runner.preprocess))
    for x, y in tqdm(loader, desc=desc, leave=False):
        metrics.update(runner.step(x), y)
    out = metrics.result()
    tail = max(1, len(metrics.batch_n) // 4)
    out["final_accuracy"] = sum(metrics.batch_correct[-tail:]) / sum(metrics.batch_n[-tail:])
    return out


def run_trial(trial: Trial, make_model: Callable[[], tuple[nn.Module, object]], norm: str,
              streams: Sequence[tuple[str, ShiftDataset]], *, device: torch.device, batch_size: int, workers: int,
              seed: int, num_samples: int | None, tta: str = "tent", ece_bins: int = 15, bn: bool = False) -> dict:
    """A fresh model, then every stream in turn (episodic). Returns the trial's result record."""
    model, preprocess = make_model()
    runner = SingleModel(model, preprocess, norm, trial.optim, device, tta, bn)
    per_stream = {label: run_stream(runner, ds, batch_size=batch_size, workers=workers, seed=seed,
                                    num_samples=num_samples, ece_bins=ece_bins, desc=f"{trial.name} | {label}")
                  for label, ds in streams}
    mean = lambda key: sum(m[key] for m in per_stream.values()) / len(per_stream)     # noqa: E731
    return {"trial": trial.name, "optim": trial.optim, "streams": per_stream,
            "macro_accuracy": mean("accuracy"), "macro_final_accuracy": mean("final_accuracy"),
            "macro_nll": mean("nll")}


# ------------------------------------------------------------------------------------------ storage and ranking

def trial_key(trial: Trial, *, model: str, tta: str, norm: str, stream_labels: Sequence[str], batch_size: int,
              seed: int, num_samples: int | None, bn: bool = False) -> str:
    """File name of a trial's stored result: the trial's name plus a hash of everything that determines it."""
    payload = {"model": model, "tta": tta, "norm": norm, "optim": trial.optim, "streams": list(stream_labels),
               "batch_size": batch_size, "seed": seed, "num_samples": num_samples}
    if trial.optim is None and norm == "BN" and not bn:
        # Only the frozen reference of a BatchNorm model depends on --bn. Stored results from before the flag
        # existed used test-batch statistics (bn=True), so that case keeps its old key.
        payload["frozen_bn"] = "source"
    return f"{trial.name}__{hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]}"


def store_result(directory: Path, key: str, result: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{key}.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(directory / f"{key}.json")


def load_result(directory: Path, key: str) -> dict | None:
    p = directory / f"{key}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def rank(results: Sequence[dict]) -> list[dict]:
    """The adapting trials, best macro accuracy first (ties: the grid order, i.e. the stable sort)."""
    return sorted((r for r in results if r["optim"] is not None), key=lambda r: -r["macro_accuracy"])


def ranking_rows(results: Sequence[dict]) -> list[dict]:
    """One flat row per trial (frozen included), best first; per-stream accuracies as columns."""
    frozen = [r for r in results if r["optim"] is None]
    rows = []
    for r in [*rank(results), *frozen]:
        rows.append({"trial": r["trial"], "macro_accuracy": r["macro_accuracy"],
                     "macro_final_accuracy": r["macro_final_accuracy"], "macro_nll": r["macro_nll"],
                     **{k: v for k, v in (r["optim"] or {}).items() if k != "STEPS"},
                     **{f"acc/{label}": m["accuracy"] for label, m in r["streams"].items()}})
    return rows


def edge_warnings(best: dict, trials: Sequence[Trial]) -> list[str]:
    """The winner's learning rate sits at the end of the range tried for its optimizer: the optimum may be outside."""
    method, lr = best["optim"]["METHOD"], best["optim"]["LR"]
    lrs = sorted({t.optim["LR"] for t in trials if t.optim and t.optim["METHOD"] == method})
    if len(lrs) > 1 and lr in (lrs[0], lrs[-1]):
        side = "smallest" if lr == lrs[0] else "largest"
        return [f"the best {method} learning rate ({lr:g}) is the {side} one tried ({lrs[0]:g}..{lrs[-1]:g}): "
                f"extend --{method.lower()}_lrs in that direction"]
    return []


def tuned_record(best: dict, frozen: dict, *, model: str, tta: str, norm: str, batch_size: int,
                 stream_labels: Sequence[str], seed: int, bn: bool = False) -> dict:
    """The content of cfgs/tuned/<tta>__<model>.yaml; load_duo reads NORM, OPTIM and batch_size, the rest is provenance.
    For a BatchNorm model the frozen score depends on --bn, so the record says which kind of frozen model it is."""
    rec = {"model": model, "tta": tta, "NORM": norm, "batch_size": batch_size, "OPTIM": dict(best["optim"]),
           "tuned_on": {"streams": list(stream_labels), "seed": seed},
           "macro_accuracy": round(best["macro_accuracy"], 5), "frozen_macro_accuracy": round(frozen["macro_accuracy"], 5)}
    if norm == "BN":
        rec["frozen_bn_batch_stats"] = bn
    return {**rec, "written_by": "scripts/02_tune_tta.py"}


def write_tuned(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Written by scripts/02_tune_tta.py on the held-out corruptions; read by src/utils/config.py::load_duo.\n"
                    "# Do not edit by hand: rerun the tuner.\n" + yaml.safe_dump(record, sort_keys=False))
    return path


if __name__ == "__main__":
    import copy
    import tempfile

    from src.experiments._testing import DUO, FakeStream, tiny_models
    from src.experiments.members import MemberRunner
    from src.utils.config import DUO_DIR, GlobalConfig, load_duo, load_global

    torch.set_num_threads(2)
    cpu = torch.device("cpu")
    base = tiny_models(0)
    (large, pre), (small, _) = base

    # ---- norm detection ------------------------------------------------------------------------------------------
    assert detect_norm(large) == "LN" and detect_norm(small) == "BN"
    for odd in (nn.Sequential(nn.Linear(2, 2)), nn.Sequential(nn.BatchNorm2d(2), nn.LayerNorm(2))):
        try:
            detect_norm(odd)
            raise AssertionError("an ambiguous or missing norm must raise")
        except ValueError as e:
            assert "pass it explicitly" in str(e)

    # ---- grids ---------------------------------------------------------------------------------------------------
    full = build_trials(["Adam", "SGD"], [1e-4, 1e-3], [0.9], [1e-3, 1e-2], [0.0, 0.9])
    assert [t.name for t in full] == ["Adam_lr0.0001_b0.9", "Adam_lr0.001_b0.9", "SGD_lr0.001_m0", "SGD_lr0.001_m0.9",
                                      "SGD_lr0.01_m0", "SGD_lr0.01_m0.9"]
    assert build_trials(["SGD"], [], [], [0.1], [0.9])[0].optim == {"METHOD": "SGD", "STEPS": 1, "LR": 0.1, "MOMENTUM": 0.9,
                                                                    "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}
    try:
        build_trials(["Adam"], [], [0.9], [], [])
        raise AssertionError("an empty grid must raise")
    except ValueError:
        pass

    # ---- one trial on tiny models: frozen reference, lr = 0 is frozen, a real lr moves, determinism ----------------
    streams = [(f"s{i}/s5", FakeStream(160, seed=i)) for i in range(2)]
    kw = dict(device=cpu, batch_size=32, workers=0, seed=3, num_samples=None)
    make = lambda: copy.deepcopy((large, pre))                                                              # noqa: E731
    frozen = run_trial(FROZEN, make, "LN", streams, **kw)
    zero = run_trial(Trial("zero", {"METHOD": "Adam", "STEPS": 1, "LR": 0.0, "BETA": 0.9, "WD": 0.0}), make, "LN", streams, **kw)
    big_trial = Trial("big", {"METHOD": "Adam", "STEPS": 1, "LR": 0.05, "BETA": 0.9, "WD": 0.0})
    big, big2 = run_trial(big_trial, make, "LN", streams, **kw), run_trial(big_trial, make, "LN", streams, **kw)
    assert frozen["optim"] is None and set(frozen["streams"]["s0/s5"]) >= {"accuracy", "final_accuracy", "nll", "ece", "entropy", "n"}
    assert frozen["streams"] == zero["streams"], "a learning rate of 0 must reproduce the frozen model"
    assert big["streams"] == big2["streams"], "a trial is deterministic"
    assert big["streams"]["s0/s5"]["entropy"] != frozen["streams"]["s0/s5"]["entropy"], "a real learning rate must adapt"
    assert abs(big["macro_accuracy"] - sum(m["accuracy"] for m in big["streams"].values()) / 2) < 1e-12

    # ---- --bn: only the FROZEN reference of a BatchNorm model changes -----------------------------------------------
    make_bn = lambda: copy.deepcopy((small, pre))                                                           # noqa: E731
    f_src = run_trial(FROZEN, make_bn, "BN", streams, **kw)                                                  # default: eval(), source stats
    f_bn = run_trial(FROZEN, make_bn, "BN", streams, **kw, bn=True)
    zero_bn = Trial("zero", {"METHOD": "Adam", "STEPS": 1, "LR": 0.0, "BETA": 0.9, "WD": 0.0})
    a_src = run_trial(zero_bn, make_bn, "BN", streams, **kw)
    a_bn = run_trial(zero_bn, make_bn, "BN", streams, **kw, bn=True)
    assert f_src["streams"] != f_bn["streams"], "frozen BatchNorm: source vs test-batch statistics differ"
    assert a_src["streams"] == a_bn["streams"], "an adapting model always uses batch statistics: the flag is irrelevant"
    assert a_bn["streams"] == f_bn["streams"], "a model that adapts with lr = 0 is the frozen model on test-batch statistics"
    assert f_src["streams"] == run_trial(FROZEN, make_bn, "BN", streams, **kw)["streams"], "the default is bn=False"
    k = lambda t, norm, bn: trial_key(t, model="m", tta="tent", norm=norm, stream_labels=["a"], batch_size=64,   # noqa: E731
                                      seed=0, num_samples=None, bn=bn)
    assert k(FROZEN, "BN", False) != k(FROZEN, "BN", True), "the frozen BatchNorm reference is keyed by --bn"
    assert k(FROZEN, "LN", False) == k(FROZEN, "LN", True), "a LayerNorm model has no batch statistics: same result"
    assert k(zero_bn, "BN", False) == k(zero_bn, "BN", True), "an adapting trial does not depend on --bn"

    # ---- the single-model path IS the engine's independent adaptation of that member ---------------------------------
    ds = FakeStream(160, seed=0)
    optim = {"METHOD": "Adam", "STEPS": 1, "LR": 1e-2, "BETA": 0.9, "WD": 0.0}
    eng = MemberRunner({**DUO, "LARGE": {**DUO["LARGE"], "OPTIM": optim}}, mode="large_indep", device="cpu",
                       models=copy.deepcopy(base))
    m_eng = StreamMetrics(15)
    for z_l, _, y in eng.iter_batches(ds, batch_size=32, workers=0, seed=3, num_samples=None):
        m_eng.update(z_l, y)
    got = run_trial(Trial("x", optim), make, "LN", [("s", ds)], **kw)["streams"]["s"]
    assert got["accuracy"] == m_eng.result()["accuracy"] and abs(got["nll"] - m_eng.result()["nll"]) < 1e-9, \
        "tuning must run the same adaptation as the duo's large_indep mode"

    # ---- storage, resume keys, ranking, grid-edge warnings ----------------------------------------------------------
    key = lambda t, **o: trial_key(t, model="m", tta="tent", norm="LN", stream_labels=["a", "b"], batch_size=64,  # noqa: E731
                                   seed=0, num_samples=None, **o)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "trials"
        assert load_result(d, key(big_trial)) is None
        store_result(d, key(big_trial), big)
        assert load_result(d, key(big_trial)) == json.loads(json.dumps(big)) and not list(d.glob(".*.tmp"))
    k0 = key(big_trial)
    for change in (dict(model="m2"), dict(tta="eata"), dict(norm="BN"), dict(stream_labels=["a"]), dict(batch_size=32),
                   dict(seed=1), dict(num_samples=100)):
        args = dict(model="m", tta="tent", norm="LN", stream_labels=["a", "b"], batch_size=64, seed=0, num_samples=None)
        assert trial_key(big_trial, **{**args, **change}) != k0, change
    assert trial_key(Trial("big", {**big_trial.optim, "LR": 0.06}), model="m", tta="tent", norm="LN",
                     stream_labels=["a", "b"], batch_size=64, seed=0, num_samples=None) != k0, "the optimizer is in the key"

    def fake(name, optim, acc):
        return {"trial": name, "optim": optim, "streams": {"a": {"accuracy": acc}, "b": {"accuracy": acc}},
                "macro_accuracy": acc, "macro_final_accuracy": acc, "macro_nll": 1.0}
    grid = build_trials(["Adam"], [1e-4, 1e-3, 1e-2], [0.9], [], [])
    res = [fake("frozen", None, 0.30), fake(grid[0].name, grid[0].optim, 0.41), fake(grid[1].name, grid[1].optim, 0.45),
           fake(grid[2].name, grid[2].optim, 0.20)]
    ranked = rank(res)
    assert [r["trial"] for r in ranked] == [grid[1].name, grid[0].name, grid[2].name], "best first, frozen not ranked"
    assert ranking_rows(res)[-1]["trial"] == "frozen" and ranking_rows(res)[0]["acc/a"] == 0.45
    assert edge_warnings(ranked[0], grid) == [], "an interior optimum raises no warning"
    assert "smallest" in edge_warnings(ranked[1], grid)[0]
    assert "largest" in edge_warnings(fake("e", grid[2].optim, 0.9), grid)[0]
    assert edge_warnings(fake("only", grid[0].optim, 0.9), grid[:1]) == [], "one learning rate has no edge"

    # ---- the tuned file reaches the experiments: load_duo overlays it on the duo YAML --------------------------------
    g = load_global()
    with tempfile.TemporaryDirectory() as tmp:
        g2 = GlobalConfig({**g.raw, "paths": {**g.raw["paths"], "tuned": tmp}}, g.source)
        rec = tuned_record(ranked[0], res[0], model="resnet50", tta="tent", norm="BN", batch_size=g.batch_size,
                           stream_labels=["a", "b"], seed=0, bn=True)
        assert rec["frozen_bn_batch_stats"] is True
        assert "frozen_bn_batch_stats" not in tuned_record(ranked[0], res[0], model="m", tta="tent", norm="LN",
                                                           batch_size=64, stream_labels=["a"], seed=0)
        out = write_tuned(Path(tmp) / "tent__resnet50.yaml", rec)
        assert yaml.safe_load(out.read_text()) == rec and out.read_text().startswith("# Written by")
        duo = load_duo("vitb16_rn50", g2)
        yaml_large = yaml.safe_load((DUO_DIR / "vitb16_rn50.yaml").read_text())["LARGE"]["OPTIM"]
        assert duo["SMALL"]["OPTIM"] == ranked[0]["optim"] and duo["LARGE"]["OPTIM"] == yaml_large
    print("tuning self-test passed")
