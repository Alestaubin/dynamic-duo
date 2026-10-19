"""
algorithms.py
=============
The base TTA algorithms of the Dynamic Duos experiments behind one interface, one member at a time (Phase 3, Stage D).

    runner = make_runner(spec, ctx)          # what ensure_logits (logit_runs.py) adapts the pair with
    z_l, z_s = runner.step_tensors(x_l, x_s, dataset)

  tent        the engine's own `MemberRunner` (src/experiments/members.py): its logits are bit for bit those of every cache made
              so far, and it stays on the existing path (author's decision 10)
  eta         ETA of the official EATA code (src/tta/ported/eata.py), no Fisher term
  eata        EATA: ETA plus the Fisher anti-forgetting penalty (weight 2000, Fisher from 2000 clean images of the training slice)
  sar         SAR of the official code (src/tta/ported/sar.py), SAM on SGD; `recovery` False (the duo base setting) or True

`eta` is NOT the repo's older `eata` method (src/tta/methods/eata.py): that one steps the optimizer on a batch whose samples were
all filtered out, so momentum keeps moving the weights, which upstream does not (tested in src/tta/ported/eata.py).

The ported algorithms follow the upstream benchmark's convention: images arrive as [0, 1] pixels and the model normalizes
INSIDE itself, behind an `ImageNormalizer` in front of it (`split_normalize`, `wrap`). The transform that builds the batches stops
before the normalization, so a batch is normalized exactly once (input_checks.py asserts it), and a later algorithm with pixel-space
augmentations (CoTTA, ROID, DeYO) can be ported without touching the pipeline.

`resolve_spec` writes every default hyperparameter into `RunSpec.tta_kwargs` before anything is keyed, so a member's key names the
hyperparameters it was run with and a changed default can never reuse an old cache by accident.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torchvision import transforms

from src.experiments.input_checks import assert_unit_range
from src.experiments.members import load_models
from src.experiments.specs import MODE_SIDES, RunSpec
from src.tta.methods import build_tta_method
from src.tta.ported.eata import EataMember, compute_fishers
from src.tta.ported.sar import SarMember

ALGO_DEFAULTS: dict[str, dict] = {
    "tent": {},
    "eta": {"e_margin_frac": 0.4, "d_margin": 0.05},
    "eata": {"e_margin_frac": 0.4, "d_margin": 0.05, "fisher_alpha": 2000.0, "fisher_samples": 2000, "fisher_slice_seed": 0},
    "sar": {"recovery": False, "rho": 0.05, "margin_frac": 0.4, "reset_constant_em": 0.2, "ema_decay": 0.9},
}
PORTED = ("eta", "eata", "sar")
ALGORITHMS = tuple(ALGO_DEFAULTS)


def resolve_spec(spec: RunSpec) -> RunSpec:
    """`spec` with its algorithm's default hyperparameters filled in (and unknown names refused)."""
    if spec.mode == "no_adapt":
        return spec                      # a frozen run applies no algorithm: its key forgets it
    if spec.tta not in ALGO_DEFAULTS:
        raise ValueError(f"unknown algorithm {spec.tta!r}: {ALGORITHMS}")
    extra = set(spec.tta_kwargs) - set(ALGO_DEFAULTS[spec.tta])
    if extra:
        raise ValueError(f"{spec.tta}: unknown hyperparameters {sorted(extra)}; known {sorted(ALGO_DEFAULTS[spec.tta])}")
    return dataclasses.replace(spec, tta_kwargs={**ALGO_DEFAULTS[spec.tta], **spec.tta_kwargs})


# ------------------------------------------------------------------------------------------ [0, 1] pixels + Normalize

class ImageNormalizer(nn.Module):
    """(x - mean) / std on a (B, 3, H, W) batch of [0, 1] pixels: the first layer of every member of the ported algorithms."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor([float(m) for m in mean]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([float(s) for s in std]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def split_normalize(preprocess):
    """A model's preprocess as (the same pipeline without its normalization, mean, std). A Compose loses its Normalize; a
    torchvision `Weights.transforms()` object, whose resize and crop cannot be rebuilt apart from it, becomes a copy with the
    identity mean 0 / std 1."""
    if isinstance(preprocess, transforms.Compose):
        norms = [t for t in preprocess.transforms if isinstance(t, transforms.Normalize)]
        if len(norms) != 1:
            raise ValueError(f"expected exactly one Normalize in {preprocess}, found {len(norms)}")
        rest = [t for t in preprocess.transforms if not isinstance(t, transforms.Normalize)]
        return transforms.Compose(rest), tuple(norms[0].mean), tuple(norms[0].std)
    if all(hasattr(preprocess, a) for a in ("mean", "std", "resize_size", "crop_size")):
        unit = copy.copy(preprocess)
        unit.mean, unit.std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        return unit, tuple(preprocess.mean), tuple(preprocess.std)
    raise ValueError(f"cannot split the normalization off {preprocess!r}")


def wrap(model: nn.Module, mean, std) -> nn.Sequential:
    return nn.Sequential(ImageNormalizer(mean, std), model)


# ------------------------------------------------------------------------------------------ members

class LossMember:
    """A TTAMethod of src/tta/methods (loss + update) on ONE model, as a member: the method is set up on it as the "large" side of
    a one-model duo, like tuning.SingleModel. `optim=None` freezes it (no updates): BatchNorm uses the test batch's statistics
    with `bn`, else the source ones. Scores the pre-update logits."""

    def __init__(self, model: nn.Module, norm: str, optim: dict | None, device: torch.device, tta: str = "tent",
                 tta_kwargs: dict | None = None, bn: bool = False):
        self.model, self.device, self.adapt = model.to(device), device, optim is not None
        side = {"NAME": "model", "NORM": norm, "OPTIM": optim or {}}
        self.tta = build_tta_method(tta, **(tta_kwargs or {}))
        self.tta.setup(self.model, nn.Identity(), {"LARGE": side, "SMALL": {**side, "NAME": "placeholder"}},
                       adapt_large=self.adapt, adapt_small=False, bn=bn)

    def reset(self) -> None:
        if self.adapt:
            self.tta.reset()

    def step(self, x: torch.Tensor, class_mask=None) -> torch.Tensor:
        with torch.set_grad_enabled(self.adapt):
            z = self.model(x)
            if self.adapt:
                self.tta.update(self.tta.loss(z[:, class_mask] if class_mask is not None else z, "large"), ["large"])
        return z.detach()

    def describe(self) -> dict:
        return self.tta.describe() if self.adapt else {"tta/method": "frozen"}


def build_member(kind: str, model: nn.Module, norm: str, optim: dict, kwargs: dict, *, adapting: bool, device: torch.device,
                 bn: bool = False, fisher_batches: Iterable[torch.Tensor] | None = None):
    """One member of algorithm `kind` (a key of ALGO_DEFAULTS; `kwargs` complete, see resolve_spec)."""
    if not adapting:
        return LossMember(model, norm, None, device, bn=bn)
    if kind == "tent":
        return LossMember(model, norm, optim, device, "tent")
    if kind in ("eta", "eata"):
        fishers = None
        if kind == "eata":
            if fisher_batches is None:
                raise ValueError("EATA needs clean source batches for its Fisher information (fisher_batches)")
            fishers = compute_fishers(model, fisher_batches)
        return EataMember(model, optim, fishers=fishers, fisher_alpha=kwargs.get("fisher_alpha", 0.0) if kind == "eata" else 0.0,
                          e_margin_frac=kwargs["e_margin_frac"], d_margin=kwargs["d_margin"])
    if kind == "sar":
        return SarMember(model, optim, recovery=kwargs["recovery"], rho=kwargs["rho"], margin_frac=kwargs["margin_frac"],
                         reset_constant_em=kwargs["reset_constant_em"], ema_decay=kwargs["ema_decay"])
    raise ValueError(f"unknown algorithm {kind!r}: {ALGORITHMS}")


class PairRunner:
    """Two independent members. Offers what `logit_runs.adapt_stream` uses of a MemberRunner: `device`, `large` / `small` (the
    models, normalizer included), `large_preprocess` / `small_preprocess` (the [0, 1] transforms), `reset()` and `step_tensors`."""

    def __init__(self, large, small, large_model: nn.Module, small_model: nn.Module, large_preprocess, small_preprocess,
                 device: torch.device, objective_logits: str = "masked"):
        self.members = {"large": large, "small": small}
        self.large, self.small = large_model, small_model
        self.large_preprocess, self.small_preprocess = large_preprocess, small_preprocess
        self.device, self.objective_logits = device, objective_logits
        self._checked = 0

    def reset(self) -> None:
        for m in self.members.values():
            m.reset()

    def step_tensors(self, x_l: torch.Tensor, x_s: torch.Tensor, dataset) -> tuple[torch.Tensor, torch.Tensor]:
        x_l, x_s = x_l.to(self.device, non_blocking=True), x_s.to(self.device, non_blocking=True)
        if self._checked < 3:                                   # the first batches: pixels, not normalized tensors
            assert_unit_range(x_l, "large batch")
            assert_unit_range(x_s, "small batch")
            self._checked += 1
        mask = dataset.class_mask if self.objective_logits == "masked" else None
        return self.members["large"].step(x_l, mask), self.members["small"].step(x_s, mask)

    def describe(self) -> dict:
        return {side: m.describe() for side, m in self.members.items()}


def slice_fisher_loader(ctx, unit_preprocess, n: int, seed: int, batch_size: int | None = None):
    """Batches of `n` clean images of the held-out training slice, as [0, 1] pixels (CPU tensors): EATA's Fisher source."""
    from src.experiments.tuning import SingleCollate
    from src.utils.datasets import ImageNetTrainSlice

    ds = ImageNetTrainSlice.from_config(ctx.g)
    return _TensorsOnly(ds.loader(batch_size=batch_size or ctx.batch_size, num_samples=n, seed=seed, workers=ctx.g.workers,
                                  collate_fn=SingleCollate(ds.input_transform(unit_preprocess))))


class _TensorsOnly:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for x, _ in self.loader:
            yield x


def make_runner(spec: RunSpec, ctx, fisher_source: Callable | None = None):
    """The runner that adapts the pair of `spec`: MemberRunner for tent, PairRunner of ported members for eta / eata / sar.
    `fisher_source(unit_preprocess, kwargs)` -> batches, EATA's Fisher data (default: the held-out clean training slice)."""
    from src.experiments.runner import _runner

    spec = resolve_spec(spec)
    if spec.mode == "no_adapt" or spec.tta == "tent":
        return _runner(spec, ctx)
    if spec.mode in ("both_duo", "large_duo", "small_duo"):
        raise ValueError(f"mode {spec.mode!r}: the ported algorithms run the members independently")
    (large, l_pre), (small, s_pre) = ctx.model_factory() if ctx.model_factory else load_models(ctx.duo, ctx.device)
    members, models, pres = {}, {}, {}
    for side, model, pre, adapting in (("large", large, l_pre, MODE_SIDES[spec.mode][0]),
                                       ("small", small, s_pre, MODE_SIDES[spec.mode][1])):
        unit_pre, mean, std = split_normalize(pre)
        wrapped = wrap(model, mean, std).to(ctx.device)
        cfg = ctx.duo[side.upper()]
        batches = None
        if spec.tta == "eata" and adapting:
            batches = (fisher_source or (lambda p, k: slice_fisher_loader(ctx, p, k["fisher_samples"], k["fisher_slice_seed"])))(
                unit_pre, spec.tta_kwargs)
        members[side] = build_member(spec.tta, wrapped, cfg["NORM"], cfg["OPTIM"], spec.tta_kwargs, adapting=adapting,
                                     device=ctx.device, bn=ctx.bn, fisher_batches=batches)
        models[side], pres[side] = wrapped, unit_pre
    return PairRunner(members["large"], members["small"], models["large"], models["small"], pres["large"], pres["small"],
                      ctx.device, ctx.objective_logits)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from torchvision import models as tvm

    from src.experiments._testing import DUO, FakeStream, TinyBN, TinyLN, make_context
    from src.experiments.input_checks import assert_normalized_once, count_normalizations
    from src.experiments.logit_store import LogitStore, PairView
    from src.experiments.members import MemberRunner, PairCollate
    from src.experiments.protocols import Segment
    from src.utils.model import native_224

    torch.set_num_threads(2)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    norm_pre = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def tiny_base(seed=0, sharp=25.0):
        """Tiny models whose final layer is scaled up so that some samples pass the reliability filter (a random 1000-way
        model has entropy near ln 1000, above every margin, and the algorithms would never update)."""
        torch.manual_seed(seed)
        models = []
        for cls in (TinyLN, TinyBN):
            m = cls().eval()
            with torch.no_grad():
                m.net[-1].weight.mul_(sharp)
            for p in m.parameters():
                p.requires_grad_(False)
            models.append((m, norm_pre))
        return tuple(models)

    # ---- 1. split_normalize: a Compose, and torchvision's Weights.transforms() ---------------------------------------------
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.RandomState(0).randint(0, 256, (224, 224, 3), dtype=np.uint8))
    for pre in (norm_pre, tvm.ResNet50_Weights.IMAGENET1K_V1.transforms(), tvm.ViT_B_16_Weights.DEFAULT.transforms()):
        unit, m_, s_ = split_normalize(pre)
        assert tuple(m_) == mean and tuple(s_) == std
        x_unit, x_full = native_224(unit)(img), native_224(pre)(img)
        assert 0.0 <= float(x_unit.min()) and float(x_unit.max()) <= 1.0
        torch.testing.assert_close(ImageNormalizer(m_, s_)(x_unit[None])[0], x_full, rtol=0, atol=1e-6)
        assert count_normalizations(native_224(unit), None) == 0 and count_normalizations(pre, None) == 1
        # the model's own clean-image pipeline (Resize, CenterCrop) still works on the unit copy, with no normalization
        clean = unit(Image.fromarray(np.random.RandomState(1).randint(0, 256, (300, 400, 3), dtype=np.uint8)))
        assert clean.shape == ((3, 224, 224) if hasattr(pre, "resize_size") else (3, 300, 400))
        assert 0.0 <= float(clean.min()) and float(clean.max()) <= 1.0
    assert tuple(tvm.ResNet50_Weights.IMAGENET1K_V1.transforms().mean) == mean, "the original object is left alone"
    try:
        split_normalize(transforms.ToTensor())
        raise AssertionError("no Normalize to split off must raise")
    except ValueError:
        pass
    wrapped = wrap(nn.Linear(3, 2), mean, std)
    assert_normalized_once(split_normalize(norm_pre)[0], wrapped, "unit pipeline + wrapper")

    # ---- 2. resolve_spec: defaults in the key, nothing unknown ---------------------------------------------------------------
    assert resolve_spec(RunSpec("d", tta="tent")).tta_kwargs == {}
    r = resolve_spec(RunSpec("d", tta="sar", tta_kwargs={"recovery": True}))
    assert r.tta_kwargs["recovery"] is True and r.tta_kwargs["rho"] == 0.05 and r.tta_kwargs["reset_constant_em"] == 0.2
    assert resolve_spec(RunSpec("d", tta="eata")).tta_kwargs["fisher_alpha"] == 2000.0
    assert resolve_spec(RunSpec("d", tta="nope", mode="no_adapt")).tta == "nope", "a frozen run applies no algorithm"
    for bad in (RunSpec("d", tta="nope"), RunSpec("d", tta="sar", tta_kwargs={"rhoo": 1})):
        try:
            resolve_spec(bad)
            raise AssertionError("must refuse")
        except ValueError:
            pass

    # ---- 3. the [0, 1] path equals the existing path for Tent (the parity check the author asked for) ----------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tiny_base()
        ctx = make_context(tmp, batch_size=32, base=base)
        sgd = {"METHOD": "SGD", "STEPS": 1, "LR": 0.01, "MOMENTUM": 0.9, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}
        ctx = dataclasses.replace(ctx, duo={**ctx.duo, "LARGE": {**ctx.duo["LARGE"], "OPTIM": sgd},
                                            "SMALL": {**ctx.duo["SMALL"], "OPTIM": sgd}})        # SAR and EATA run on SGD
        ds = FakeStream(150, seed=2)
        old = MemberRunner(ctx.duo, "tent", {}, "both_indep", ctx.device, ctx.objective_logits, models=ctx.model_factory(), bn=False)
        old_out = [(zl.clone(), zs.clone()) for zl, zs, _ in old.iter_batches(ds, batch_size=32, workers=0, seed=1, num_samples=None)]

        (large, l_pre), (small, s_pre) = ctx.model_factory()
        parts = []
        for model, pre, side in ((large, l_pre, "LARGE"), (small, s_pre, "SMALL")):
            unit, m_, s_ = split_normalize(pre)
            parts.append((LossMember(wrap(model, m_, s_), ctx.duo[side]["NORM"], ctx.duo[side]["OPTIM"], ctx.device), unit))
        new = PairRunner(parts[0][0], parts[1][0], parts[0][0].model, parts[1][0].model, parts[0][1], parts[1][1], ctx.device)
        loader = ds.loader(batch_size=32, seed=1, workers=0, collate_fn=PairCollate(ds.input_transform(parts[0][1]),
                                                                                       ds.input_transform(parts[1][1])))
        new_out = [new.step_tensors(xl, xs, ds) for xl, xs, _ in loader]
        assert len(old_out) == len(new_out) == 5
        for (a_l, a_s), (b_l, b_s) in zip(old_out, new_out):
            torch.testing.assert_close(a_l, b_l, rtol=0, atol=1e-4)
            torch.testing.assert_close(a_s, b_s, rtol=0, atol=1e-4)
        assert any(not torch.equal(a[0], new_out[0][0]) for a in old_out[1:]), "the models adapt between batches"
        assert_normalized_once(new.large_preprocess, new.large, "parity large")

        # ---- 4. make_runner: tent stays on MemberRunner; eta / eata / sar are PairRunners and end to end through the store ----
        from src.experiments.logit_runs import ensure_logits, run_member_specs
        assert isinstance(make_runner(RunSpec("tiny", tta="tent", seed=1), ctx), MemberRunner)
        assert isinstance(make_runner(RunSpec("tiny", tta="sar", seed=1), ctx), PairRunner)
        assert isinstance(make_runner(RunSpec("tiny", tta="tent", mode="no_adapt"), ctx), MemberRunner)

        def synthetic_fisher(unit_pre, kw):
            g = torch.Generator().manual_seed(9)
            return [torch.rand(8, 3, 8, 8, generator=g) for _ in range(3)]

        store = LogitStore.from_context(ctx, staging=None)
        segs = [Segment(f"s{i}/s5", FakeStream(150, seed=sd), True) for i, sd in enumerate((1, 2))]
        keys, outs = {}, {}
        for algo in ("tent", "eta", "eata", "sar"):
            spec = RunSpec("tiny", tta=algo, seed=1)
            if algo != "tent":
                assert_normalized_once(make_runner(spec, ctx, synthetic_fisher).large_preprocess,
                                       make_runner(spec, ctx, synthetic_fisher).large, algo)
            man = ensure_logits(spec, ctx, segs, store, check_normalization=False, fisher_source=synthetic_fisher)
            ml, ms = run_member_specs(spec, ctx)
            keys[algo] = (ml.key, ms.key)
            assert man["files_written"] == 4 and ml.algorithm == algo and man["tta_kwargs"] == ALGO_DEFAULTS[algo]
            outs[algo] = PairView(store, ml, ms).read("s0__s5", raw=True)
        assert len({k[0] for k in keys.values()}) == 4, "every algorithm has its own member keys"
        # against a frozen run of the same members: the first batch is scored before any update, so it is the frozen logits
        # (whatever the algorithm), and every algorithm has changed the model by the later batches
        frozen = RunSpec("tiny", tta="tent", seed=1, mode="no_adapt")
        ensure_logits(frozen, ctx, segs, store, check_normalization=False)
        z_frozen = store.read(run_member_specs(frozen, ctx)[0], "s0__s5", raw=True)["z"]
        n0 = outs["tent"]["batch_sizes"][0]
        for algo in ("tent", "eta", "eata", "sar"):
            z = outs[algo]["z_l"]
            torch.testing.assert_close(z[:n0], z_frozen[:n0], rtol=0, atol=1e-3)
            assert not torch.allclose(z[n0:], z_frozen[n0:], atol=1e-3), f"{algo} must have adapted the large member"
        # a rerun finds everything stored
        assert ensure_logits(RunSpec("tiny", tta="sar", seed=1), ctx, segs, store, check_normalization=False)["files_written"] == 0
        # the key names its hyperparameters: another SAR setting is another member
        k_rec = run_member_specs(RunSpec("tiny", tta="sar", seed=1, tta_kwargs={"recovery": True}), ctx)[0].key
        assert k_rec != keys["sar"][0]
        # EATA without a Fisher source is refused, a frozen member needs no algorithm
        try:
            make_runner(RunSpec("tiny", tta="eata"), dataclasses.replace(ctx, model_factory=ctx.model_factory),
                        fisher_source=lambda p, k: [])
            raise AssertionError("EATA with no source batches must raise")
        except ValueError:
            pass
        # a member that does not adapt (large_indep with SAR: the small one is frozen)
        mixed = make_runner(RunSpec("tiny", tta="sar", mode="large_indep"), ctx)
        assert isinstance(mixed.members["large"], SarMember) and isinstance(mixed.members["small"], LossMember) and not mixed.members["small"].adapt
        # joint modes are refused for the ported algorithms
        try:
            make_runner(RunSpec("tiny", tta="sar", mode="both_duo", loop_ts=(1.0, 1.0)), ctx)
            raise AssertionError("joint modes are not supported for ported algorithms")
        except ValueError:
            pass
    print("algorithms self-test passed")
