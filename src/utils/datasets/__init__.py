"""One class per dataset the paper uses. Each knows its layout, checks it, and returns loaders.

    from src.utils.datasets import get_dataset
    ds = get_dataset("imagenet_c", corruption="fog")        # paths come from cfgs/global.yaml
    for imgs, labels in ds.loader(batch_size=64, seed=0): ...
    scores = ds.apply_class_mask(model_logits)              # only does anything for ImageNet-R/A

Archive-backed datasets (ImageNet-A/R/V2/Sketch on the shared folder) are extracted to $SLURM_TMPDIR on first
use; see staging.py. `python -m src.utils.datasets` runs the self-test (fake on-disk datasets, no real data needed);
`python scripts/setup_check.py` checks the real data.
"""

from __future__ import annotations

from src.utils.config import GlobalConfig, load_global
from src.utils.datasets.base import DatasetLayoutError, ShiftDataset, make_loader, pil_collate, read_wnids
from src.utils.datasets.ccc import CCC
from src.utils.datasets.staging import StagingUnavailable, stage_archive, stage_dataset
from src.utils.datasets.imagenet import ImageNetTrainSlice, ImageNetVal
from src.utils.datasets.imagenet_a import ImageNetA
from src.utils.datasets.imagenet_c import ImageNetC
from src.utils.datasets.imagenet_r import ImageNetR
from src.utils.datasets.imagenet_sketch import ImageNetSketch
from src.utils.datasets.imagenet_v2 import ImageNetV2

DATASETS: dict[str, type[ShiftDataset]] = {
    cls.name: cls
    for cls in (ImageNetVal, ImageNetTrainSlice, ImageNetC, ImageNetR, ImageNetA, ImageNetSketch, ImageNetV2, CCC)
}

# Natural-shift evaluation sets of the generality table (episodic), in table-column order.
NATURAL_SHIFTS = ("imagenet_r", "imagenet_a", "imagenet_sketch", "imagenet_v2")


def get_dataset(name: str, g: GlobalConfig | None = None, **kwargs) -> ShiftDataset:
    """Build dataset `name` from the global config; kwargs are the dataset's own selectors
    (corruption/severity for imagenet_c, difficulty/speed/seed for ccc)."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name].from_config(g or load_global(), **kwargs)


__all__ = ["DATASETS", "NATURAL_SHIFTS", "get_dataset", "make_loader", "pil_collate", "read_wnids",
           "stage_dataset", "stage_archive", "StagingUnavailable",
           "ShiftDataset", "DatasetLayoutError", "ImageNetVal", "ImageNetTrainSlice", "ImageNetC",
           "ImageNetR", "ImageNetA", "ImageNetSketch", "ImageNetV2", "CCC"]
