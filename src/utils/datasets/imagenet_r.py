"""ImageNet-R (renditions): 30,000 images, 200 classes, <top>/<wnid>/*.jpg for the 200 wnids of
hendrycks/imagenet-r (third_party/imagenet-r/eval.py).

Scored with `logits[:, class_mask]` (their own eval.py does `net(data)[:, imagenet_r_mask]`); labels
are the position of the wnid among the 200 sorted wnids, which is also the order of `class_mask`.
A different 200-class subset from ImageNet-A, so it has its own mask. The class folders on disk must be
exactly those 200 wnids. May be an `imagenet-r.tar` (staged to node-local storage) or the extracted
directory (read in place).
"""

from __future__ import annotations

from src.utils.datasets.base import FolderDataset, read_wnids


class ImageNetR(FolderDataset):
    name = "imagenet_r"
    config_key = "imagenet_r"
    num_classes = 200
    expected_images = 30_000

    def __init__(self, root, verify_counts: bool = True):
        super().__init__(root, verify_counts)
        self._wnids = read_wnids("r")
        index = {w: i for i, w in enumerate(read_wnids("1k"))}
        self.class_mask = [index[w] for w in self._wnids]

    def class_dirs(self) -> list[str]:
        return self._wnids
