"""ImageNet-C: <root>/<corruption>/<severity>/<wnid>/*.JPEG, rendered from the ImageNet validation set.

One instance is one (corruption, severity) stream of 50,000 images. Episodic evaluation iterates
`ImageNetC.stream(...)` and resets between instances; the continual protocol iterates the same
instances in one of CoTTA's orders and does not reset. Because each instance gets its own loader,
a continual stream never mixes corruptions within a batch (a ConcatDataset + shuffle would).

The images are already 224x224 centre crops (`already_224`).
"""

from __future__ import annotations

from pathlib import Path

from src.utils.config import GlobalConfig
from src.utils.datasets.base import ShiftDataset, WnidFolder, read_wnids


class ImageNetC(ShiftDataset):
    name = "imagenet_c"
    config_key = "imagenet_c"
    already_224 = True

    def __init__(self, root, corruption: str, severity: int = 5):
        super().__init__(root)
        self.corruption = corruption
        self.severity = severity

    @classmethod
    def from_config(cls, g: GlobalConfig, corruption: str, severity: int | None = None):
        return cls(g.path(cls.config_key), corruption, g.severity if severity is None else severity)

    @classmethod
    def stream(cls, root, corruptions: list[str], severity: int = 5) -> list["ImageNetC"]:
        """The instances to visit, in order. Pass one of cfg.cotta_orders for the continual protocol."""
        return [cls(root, c, severity) for c in corruptions]

    @property
    def folder(self) -> Path:
        return self.root / self.corruption / str(self.severity)

    def torch_dataset(self):
        return WnidFolder(self.folder, read_wnids("1k"))

    def describe(self) -> str:
        return f"imagenet_c/{self.corruption}/s{self.severity}"
