import os
import yaml
import torch
from torchvision import datasets


def _pil_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

def _norm_logits(z: torch.Tensor) -> torch.Tensor:
    """zero-mean unit-std normalization across classes."""
    mu = z.mean(dim=-1, keepdim=True)
    sd = z.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (z - mu) / sd

def load_imagenetC(data_dir, severities, corruption_types, device, batch_size=256, num_workers=4, fraction=1.0, seed=None):
    """
    Load the ImageNet-C dataset for a given corruption type and severity level.
    Returns a DataLoader of (list[PIL.Image], LongTensor) batches — no preprocessing
    applied so each model can apply its own transform in forward.

    fraction: proportion of each subset to use (0 < fraction <= 1.0).
    """
    if isinstance(severities, int):
        severities = [severities]
    if isinstance(corruption_types, str):
        corruption_types = [corruption_types]

    subsets = []
    for corruption in corruption_types:
        for severity in severities:
            ds = datasets.ImageFolder(os.path.join(data_dir, corruption, str(severity)))
            if fraction < 1.0:
                n = max(1, int(len(ds) * fraction))
                generator = torch.Generator().manual_seed(seed) if seed is not None else None
                indices = torch.randperm(len(ds), generator=generator)[:n].tolist()
                ds = torch.utils.data.Subset(ds, indices)
            subsets.append(ds)

    combined = torch.utils.data.ConcatDataset(subsets)
    pin_memory = device.type == "cuda"
    return torch.utils.data.DataLoader(
        combined, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_pil_collate_fn,
    )

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
