"""
input_checks.py
===============
Assertions against the two loader pitfalls the Dynamic Duos experiments already hit (docs/design_section45.md, 3.3):

  * double (or missing) normalization: a batch is normalized exactly once, either by the transform that builds it or
    by the model that consumes it, never both. `count_normalizations` counts the normalization steps of a
    (transform, model) pair and `assert_normalized_once` insists on one. This repo normalizes in the transform and keeps
    the models bare; the upstream TTA benchmark does the opposite (ToTensor only, normalization inside the model), so a
    method ported from it must get the transform that stops at ToTensor AND a model wrapped with a Normalize layer.
    `assert_unit_range` checks the [0, 1] side of that convention on a batch.
  * single-class batches: a loader that does not shuffle delivers ImageNet-C class by class (50 images per class), so a
    batch of 64 holds one or two classes and every adaptation step sees a different, absurd label marginal.
    `check_batch_diversity` fails when a stream's batches hold, on average, fewer than half as many distinct labels as
    samples (a shuffled batch of 64 from 1000 classes holds about 62, from the 200 classes of ImageNet-R/A about 54).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torchvision import transforms

_NORMALIZER_NAMES = ("ImageNormalizer", "Normalize", "NormalizeLayer", "Normalization")


def _identity(mean, std) -> bool:
    """mean 0 and std 1 leave a batch as it is: not a normalization (the [0, 1] pipeline of the ported algorithms keeps such a
    step where a torchvision `Weights.transforms()` object cannot drop its Normalize)."""
    return all(float(m) == 0.0 for m in mean) and all(float(s) == 1.0 for s in std)


def count_normalizations(preprocess, model: nn.Module | None = None) -> int:
    """Normalization steps in `preprocess` (a transforms.Compose, or a torchvision `Weights.transforms()` object, which
    normalizes with its mean and std) plus the normalization modules inside `model`."""
    n = 0
    if isinstance(preprocess, transforms.Compose):
        n += sum(isinstance(t, transforms.Normalize) and not _identity(t.mean, t.std) for t in preprocess.transforms)
    elif all(hasattr(preprocess, a) for a in ("mean", "std")):
        n += 0 if _identity(preprocess.mean, preprocess.std) else 1
    if model is not None:
        n += sum(type(m).__name__ in _NORMALIZER_NAMES for m in model.modules())
    return n


def assert_normalized_once(preprocess, model: nn.Module | None = None, what: str = "") -> None:
    n = count_normalizations(preprocess, model)
    if n != 1:
        raise AssertionError(
            f"{what or 'input pipeline'}: the batch is normalized {n} time(s) between the image and the logits, expected "
            f"exactly 1 (double normalization if the transform normalizes and the model wraps a normalizer too; none if "
            f"neither does)")


def assert_unit_range(x: torch.Tensor, what: str = "batch") -> None:
    """A [0, 1] pixel batch (the convention of the ported upstream methods, whose augmentations clip to [0, 1])."""
    lo, hi = float(x.min()), float(x.max())
    if lo < 0.0 or hi > 1.0:
        raise AssertionError(f"{what} is not in [0, 1] (range [{lo:.3f}, {hi:.3f}]): it has been normalized already")


def check_batch_diversity(labels: torch.Tensor, batch_sizes: Sequence[int], min_fraction: float = 0.5) -> float:
    """Mean over the stream's full-size batches of (distinct labels / batch size); raises when it is below
    `min_fraction`. Returns the mean. A stream shorter than one batch is checked as a single batch."""
    labels = labels.cpu()
    full = max(batch_sizes)
    parts, start = [], 0
    for b in batch_sizes:
        if b == full or len(batch_sizes) == 1:
            parts.append(len(torch.unique(labels[start:start + b])) / b)
        start += b
    mean = sum(parts) / len(parts)
    if mean < min_fraction:
        raise AssertionError(
            f"batches hold {mean:.2f} distinct labels per sample on average (< {min_fraction}): the stream was not "
            f"shuffled, so every batch is one or two classes")
    return mean


if __name__ == "__main__":
    from torchvision import models

    class ImageNormalizer(nn.Module):               # the name robustbench / the upstream benchmark use
        def forward(self, x):
            return x

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    bare = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    wrapped = nn.Sequential(ImageNormalizer(), bare)

    # ours: the transform normalizes, the model is bare
    ours = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    assert count_normalizations(ours, bare) == 1
    assert_normalized_once(ours, bare)
    # torchvision's Weights.transforms() normalizes too, and so does its native_224 version
    from src.utils.model import native_224
    w = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    assert count_normalizations(w, bare) == 1 and count_normalizations(native_224(w), bare) == 1
    # upstream convention: ToTensor only, normalization in the model
    unit = transforms.Compose([transforms.ToTensor()])
    assert count_normalizations(unit, wrapped) == 1
    assert_normalized_once(unit, wrapped)
    ident = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
    assert count_normalizations(ident, wrapped) == 1 and count_normalizations(ident, bare) == 0, "identity is not a normalization"
    # the two ways to get it wrong
    for pre, model, kind in ((ours, wrapped, "double"), (unit, bare, "none")):
        try:
            assert_normalized_once(pre, model, "test")
            raise SystemExit(f"{kind} normalization must raise")
        except AssertionError as e:
            assert "expected exactly 1" in str(e) and "test" in str(e)

    assert_unit_range(torch.rand(2, 3, 4, 4))
    for bad in (torch.randn(2, 3, 4, 4) * 2, torch.rand(1, 3, 2, 2) + 0.5):
        try:
            assert_unit_range(bad)
            raise SystemExit("a normalized batch must raise")
        except AssertionError:
            pass

    g = torch.Generator().manual_seed(0)
    shuffled = torch.randint(0, 1000, (5000,), generator=g)
    sizes = [64] * 78 + [8]
    assert check_batch_diversity(shuffled, sizes) > 0.9
    assert check_batch_diversity(torch.randint(0, 200, (5000,), generator=g), sizes) > 0.7, "200 classes (ImageNet-R/A)"
    by_class = torch.arange(1000).repeat_interleave(5)            # unshuffled: class after class
    try:
        check_batch_diversity(by_class, sizes)
        raise SystemExit("class-sorted batches must raise")
    except AssertionError as e:
        assert "not shuffled" in str(e)
    assert check_batch_diversity(torch.randint(0, 1000, (16,), generator=g), [16]) > 0.5, "one short batch"
    print("input_checks self-test passed")
