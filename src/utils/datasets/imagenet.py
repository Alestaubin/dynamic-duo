"""ImageNet itself: the clean validation set and the held-out clean slice of the training set.

ImageNet-C and CCC are both rendered from the 50,000 *validation* images, so anything fitted on
clean data without touching the test streams (temperatures T_L/T_S, ATC thresholds, proxy source
statistics) must come from *train*. `ImageNetTrainSlice` is that slice: `images_per_class` images per
class, chosen deterministically from `seed`, with the chosen file list cached so the (slow, network
filesystem) directory listing of 1.28M files happens once.

Caveat the paper does not discuss: the pretrained members saw these training images, so they are
slightly more confident on them than on unseen data. Temperatures fitted here may therefore come out
a little low; scripts/fit_temperatures.py can report the val-fitted value for comparison.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.utils.config import GlobalConfig, REPO_ROOT
from src.utils.datasets.base import DatasetLayoutError, FileList, ShiftDataset, WnidFolder, read_wnids


class ImageNetVal(ShiftDataset):
    name = "imagenet_val"
    config_key = "imagenet_val"

    def torch_dataset(self):
        return WnidFolder(self.root, read_wnids("1k"))


class ImageNetTrainSlice(ShiftDataset):
    name = "imagenet_train_slice"
    config_key = "imagenet_train"

    def __init__(self, root, images_per_class: int = 10, seed: int = 0, cache_dir: str | Path | None = None):
        super().__init__(root)
        self.images_per_class = images_per_class
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else REPO_ROOT / "cache"

    @classmethod
    def from_config(cls, g: GlobalConfig, **kwargs):
        s = g.source_slice
        if s["dataset"] != "imagenet_train":
            raise NotImplementedError(f"source_slice.dataset={s['dataset']!r}; only imagenet_train is supported")
        kwargs.setdefault("images_per_class", s["images_per_class"])
        kwargs.setdefault("seed", s["seed"])
        kwargs.setdefault("cache_dir", g.path("cache"))
        return super().from_config(g, **kwargs)

    @property
    def _list_path(self) -> Path:
        return self.cache_dir / f"train_slice_k{self.images_per_class}_s{self.seed}.tsv"

    def _select(self) -> list[tuple[str, int]]:
        wnids = read_wnids("1k")
        if not self.root.is_dir():
            raise DatasetLayoutError(f"{self.root} is not a directory")
        samples = []
        for label, wnid in enumerate(wnids):
            class_dir = self.root / wnid
            if not class_dir.is_dir():
                raise DatasetLayoutError(f"{self.root} has no class folder {wnid}")
            files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".jpeg", ".jpg", ".png")))
            if len(files) < self.images_per_class:
                raise DatasetLayoutError(f"{class_dir} has only {len(files)} images")
            # One independent stream per class, so changing images_per_class never reshuffles other classes.
            pick = np.random.RandomState([self.seed, label]).choice(len(files), self.images_per_class, replace=False)
            samples += [(f"{wnid}/{files[i]}", label) for i in sorted(pick)]
        return samples

    def file_list(self) -> list[tuple[str, int]]:
        if self._list_path.exists():
            rows = [line.split("\t") for line in self._list_path.read_text().splitlines()]
            return [(p, int(y)) for p, y in rows]
        samples = self._select()
        self._list_path.parent.mkdir(parents=True, exist_ok=True)
        self._list_path.write_text("".join(f"{p}\t{y}\n" for p, y in samples))
        return samples

    def torch_dataset(self):
        return FileList([(str(self.root / p), y) for p, y in self.file_list()])
