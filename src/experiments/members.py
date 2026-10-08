"""
members.py
==========
MemberRunner: the two members adapting independently, one batch at a time.

This is the independent-adaptation branch of src/tta/dynamic_duo.py::forward_and_adapt without the
calibrator, the diagnostics and wandb around it: both models see the same batch (each through its own
preprocessing), each computes its own TTA loss on its own logits and updates itself, and the logits
returned are the PRE-update logits of that step (the TENT convention, identical to the old engine).

Only the independent modes are supported: with `*_duo` modes the calibrator sits inside the adaptation
loop, so the gate could no longer be varied by replay. The paper reports the independent case and leaves
the joint mode to future work.

The 200-class natural shifts (ImageNet-A/R): `objective_logits` ("masked" | "full", cfgs/global.yaml
protocol.objective_logits_200class) decides whether the TTA loss sees the dataset's 200 logits or all 1000;
the returned logits are always the full 1000, and callers score accuracy on `dataset.apply_class_mask`.
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch

from src.tta.methods import build_tta_method, resolve_tta_spec
from src.utils.datasets.base import ShiftDataset
from src.utils.model import _preprocess_batch, get_model

logger = logging.getLogger(__name__)

# mode -> (adapt_large, adapt_small)
INDEP_MODES = {
    "both_indep": (True, True),
    "large_indep": (True, False),
    "small_indep": (False, True),
    "no_adapt": (False, False),
}


class PairCollate:
    """Batch collation that runs BOTH members' preprocessing (each model keeps its own transform) inside the
    DataLoader workers: (list of (PIL, label)) or an already-built (list of PIL, labels) pair
    -> (x_large, x_small, labels). Run in the main process, the two pipelines (resize + crop + normalize on 64
    images each) cost about as much as the GPU step itself."""

    def __init__(self, large_preprocess, small_preprocess):
        self.pre_l, self.pre_s = large_preprocess, small_preprocess

    def __call__(self, batch):
        if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[1], torch.Tensor):
            imgs, labels = batch
        else:
            imgs, labels = zip(*batch)
            imgs, labels = list(imgs), torch.tensor(labels)
        return (torch.stack([self.pre_l(i) for i in imgs]), torch.stack([self.pre_s(i) for i in imgs]), labels)


def load_models(duo_cfg: dict, device: torch.device):
    """((large, large_preprocess), (small, small_preprocess)), frozen and on `device` (the TTA method
    then unfreezes what it adapts)."""
    out = []
    for side in ("LARGE", "SMALL"):
        model, preprocess = get_model(duo_cfg[side]["NAME"], verbose=False)
        out.append((model.to(device), preprocess))
    return tuple(out)


class MemberRunner:
    def __init__(self, duo_cfg: dict, tta: str = "tent", tta_kwargs: dict | None = None,
                 mode: str = "both_indep", device: torch.device | str | None = None,
                 objective_logits: str = "masked", models=None):
        if mode not in INDEP_MODES:
            raise ValueError(
                f"mode {mode!r} is not supported here: only {sorted(INDEP_MODES)}. The *_duo modes put the "
                f"calibrator inside the adaptation loop, so the gate could not be replayed."
            )
        if objective_logits not in ("masked", "full"):
            raise ValueError(f"objective_logits must be 'masked' or 'full', got {objective_logits!r}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mode = mode
        self.objective_logits = objective_logits
        self.adapt_large, self.adapt_small = INDEP_MODES[mode]

        (self.large, self.large_preprocess), (self.small, self.small_preprocess) = (
            models if models is not None else load_models(duo_cfg, self.device)
        )
        name, kwargs = resolve_tta_spec(duo_cfg, tta, tta_kwargs)
        self.tta = build_tta_method(name, **kwargs)
        self.tta.setup(self.large, self.small, duo_cfg, self.adapt_large, self.adapt_small)

    def reset(self) -> None:
        """Both adapting members back to their source weights (and the method's own state)."""
        self.tta.reset()

    def step(self, imgs, dataset: ShiftDataset) -> tuple[torch.Tensor, torch.Tensor]:
        """`step_tensors` on a list of PIL images (preprocessed here, in the calling process)."""
        return self.step_tensors(_preprocess_batch(imgs, self.large_preprocess, self.device),
                                 _preprocess_batch(imgs, self.small_preprocess, self.device), dataset)

    def step_tensors(self, x_l: torch.Tensor, x_s: torch.Tensor, dataset: ShiftDataset
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """One batch of already-preprocessed inputs: forward both members, adapt each on its own logits.
        Returns the pre-update full (B, 1000) logits (z_large, z_small), detached, on the model device."""
        x_l, x_s = x_l.to(self.device, non_blocking=True), x_s.to(self.device, non_blocking=True)
        with torch.set_grad_enabled(self.adapt_large or self.adapt_small):
            z_l, z_s = self.large(x_l), self.small(x_s)
            for side, do, z in (("large", self.adapt_large, z_l), ("small", self.adapt_small, z_s)):
                if do:
                    self.tta.update(self.tta.loss(dataset.objective_logits(z, self.objective_logits), side), [side])
        return z_l.detach(), z_s.detach()

    def iter_batches(self, dataset: ShiftDataset, *, batch_size: int, workers: int, seed: int | None,
                     num_samples: int | None) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Adapt over one stream, yielding (z_l, z_s, labels) per batch (logits full 1000-way, on device;
        labels on CPU). The caller decides about resets: this method never resets."""
        loader = dataset.loader(batch_size=batch_size, num_samples=num_samples, seed=seed, workers=workers,
                                pin_memory=self.device.type == "cuda",
                                collate_fn=PairCollate(self.large_preprocess, self.small_preprocess))
        for x_l, x_s, labels in loader:
            z_l, z_s = self.step_tensors(x_l, x_s, dataset)
            yield z_l, z_s, labels
