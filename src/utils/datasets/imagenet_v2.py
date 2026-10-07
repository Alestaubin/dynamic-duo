"""ImageNet-V2 (matched-frequency): 10,000 images, 10 per class, <top>/<0..999>/* where <top> is
`imagenetv2-matched-frequency-format-val`.

The folders are named by ImageNet class INDEX, not wnid. torchvision's ImageFolder sorts them as
strings ('0', '1', '10', '100', ...) and so assigns nearly every class the wrong label; here the label is
simply int(folder name). Full 1000-way evaluation, no mask. The file in the shared folder is called
`.tar.gz` but is an uncompressed tar; `tar -xf` handles either.
"""

from __future__ import annotations

from src.utils.datasets.base import FolderDataset


class ImageNetV2(FolderDataset):
    name = "imagenet_v2"
    config_key = "imagenet_v2"
    expected_images = 10_000
    expected_per_class = 10
    class_hint = "ImageNet-V2 must contain exactly the folders 0..999 (ImageNet class indices)."

    def class_dirs(self) -> list[str]:
        return [str(i) for i in range(1000)]
