"""Fixtures for the engine's self-tests: tiny CPU models and a synthetic stream. Not used by experiments."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.experiments.runner import Context
from src.utils.config import GlobalConfig, load_global
from src.utils.datasets.base import ShiftDataset

OPTIM = {"METHOD": "Adam", "STEPS": 1, "LR": 1e-2, "BETA": 0.9, "WD": 0.0}
DUO = {"NAME": "tiny", "LARGE": {"NAME": "tiny_ln", "NORM": "LN", "OPTIM": OPTIM},
       "SMALL": {"NAME": "tiny_bn", "NORM": "BN", "OPTIM": OPTIM}}


class TinyLN(nn.Module):       # stands in for the large ViT: LayerNorm is what gets adapted
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, 1000))

    def forward(self, x):
        return self.net(x)


class TinyBN(nn.Module):       # stands in for the ResNet: BatchNorm is what gets adapted
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 1000))

    def forward(self, x):
        return self.net(x)


def tiny_models(seed: int = 0):
    torch.manual_seed(seed)
    pre = transforms.ToTensor()
    large, small = TinyLN().eval(), TinyBN().eval()
    for m in (large, small):
        for p in m.parameters():
            p.requires_grad_(False)
    return (large, pre), (small, pre)


class _Images(Dataset):
    def __init__(self, n: int, seed: int, num_classes: int):
        g = torch.Generator().manual_seed(seed)
        self.pixels = torch.randint(0, 256, (n, 8, 8, 3), generator=g, dtype=torch.uint8).numpy()
        self.labels = torch.randint(0, num_classes, (n,), generator=g).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return Image.fromarray(self.pixels[i]), self.labels[i]


class FakeStream(ShiftDataset):
    """n random 8x8 images with random labels; optionally a 200-of-1000 class mask like ImageNet-A/R."""
    name, config_key = "fake", "fake"

    def __init__(self, n: int = 150, seed: int = 0, masked: bool = False):
        super().__init__(".")
        self.n, self.seed = n, seed
        if masked:
            self.num_classes, self.class_mask = 200, list(range(100, 300))

    def torch_dataset(self):
        return _Images(self.n, self.seed, self.num_classes)


def make_context(tmp: Path, batch_size: int = 32, base=None) -> Context:
    """A Context whose cache lives in `tmp` and whose models are tiny copies of one initialisation."""
    raw = copy.deepcopy(load_global().raw)
    raw["paths"]["cache"] = str(tmp / "cache")
    raw["protocol"]["adaptation_batch_size"] = batch_size
    g = GlobalConfig(raw, Path("<test>"))
    base = base or tiny_models(0)
    return Context(g, copy.deepcopy(DUO), torch.device("cpu"), model_factory=lambda: copy.deepcopy(base))


def tmp_dir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory()
