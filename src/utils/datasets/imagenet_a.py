"""ImageNet-A (natural adversarial examples): 7,500 images, 200 classes, <top>/<wnid>/*.jpg for the
wnids selected by `thousand_k_to_200` in hendrycks/natural-adv-examples (scored with
`net(data)[:, indices_in_1k]`). The class folders on disk must be exactly those 200 wnids. Usually the
archive `imagenet-a.tar`, staged to node-local storage.
"""

from __future__ import annotations

from src.utils.datasets.base import FolderDataset, read_wnids


class ImageNetA(FolderDataset):
    name = "imagenet_a"
    config_key = "imagenet_a"
    num_classes = 200
    expected_images = 7_500

    def __init__(self, root, verify_counts: bool = True):
        super().__init__(root, verify_counts)
        self._wnids = read_wnids("a")
        index = {w: i for i, w in enumerate(read_wnids("1k"))}
        self.class_mask = [index[w] for w in self._wnids]

    def class_dirs(self) -> list[str]:
        return self._wnids
