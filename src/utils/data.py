import os
import yaml
import torch
from torchvision import datasets


def _pil_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)


def load_imagenetC(data_dir, severities, corruption_types, device, batch_size=256, num_workers=4):
    """
    Load the ImageNet-C dataset for a given corruption type and severity level.
    Returns a DataLoader of (list[PIL.Image], LongTensor) batches — no preprocessing
    applied so each model can apply its own transform in forward.
    """
    if isinstance(severities, int):
        severities = [severities]
    if isinstance(corruption_types, str):
        corruption_types = [corruption_types]

    subsets = [
        datasets.ImageFolder(os.path.join(data_dir, corruption, str(severity)))
        for corruption in corruption_types
        for severity in severities
    ]
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
