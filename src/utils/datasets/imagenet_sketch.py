"""ImageNet-Sketch: 50,889 images, 1000 classes, sketch/<wnid>/*.JPEG (ImageNet-Sketch.zip, staged to
node-local storage). Some files are not RGB; the loader converts every image to RGB. The folder names must
equal the canonical sorted 1000-wnid list, which is also the standard ImageNet-1k class index order.
"""

from __future__ import annotations

from src.utils.datasets.base import FolderDataset, read_wnids


class ImageNetSketch(FolderDataset):
    name = "imagenet_sketch"
    config_key = "imagenet_sketch"
    expected_images = 50_889

    def class_dirs(self) -> list[str]:
        return read_wnids("1k")
