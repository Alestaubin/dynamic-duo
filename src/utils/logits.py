import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from src.utils.data import _pil_collate_fn, load_imagenetC
from src.utils.model import get_model, _preprocess_batch
from src.tta.tent import configure_model_frozen

def get_model_logits(
    model_name: str,
    val_dir: str,
    test_dir: str,
    cache_dir: str,
    batch_size: int,
    num_workers: int,
    corruption: str | None = None,
    severity: int | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    tent_mode: bool = False,
    norm_type: str = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (logits, labels) for the given model and data split.

    Pass corruption=None for ImageNet val; otherwise ImageNet-C with the
    given corruption type and severity level.  Results are saved under
    cache_dir/<model_name>/<split_key>.pt so subsequent calls are instant.
    """
    if tent_mode and norm_type is None:
        raise ValueError("norm_type must be specified when tent_mode is True")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_key = "val" if corruption is None else f"{corruption}_{severity}"
    if tent_mode:
        split_key += "_tent_mode"
    cache_path = os.path.join(cache_dir, model_name, f"{split_key}.pt")

    if os.path.exists(cache_path):
        if verbose:
            print(f"[logit cache] hit  {model_name}/{split_key}")
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        return saved["logits"], saved["labels"]

    if verbose:
        print(f"[logit cache] miss {model_name}/{split_key} — running inference...")
    model, preprocess = get_model(model_name, freeze=True)
    model = model.to(device).eval()

    if tent_mode:
        model = configure_model_frozen(model, norm_type=norm_type)

    if tent_mode and corruption is not None:
        loader = load_imagenetC(
            test_dir, severities=severity, corruption_types=[corruption],
            device=device, batch_size=batch_size, num_workers=num_workers,
            seed=seed,
        )
    else:
        ds = datasets.ImageFolder(val_dir if corruption is None
                                  else os.path.join(test_dir, corruption, str(severity)))
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            collate_fn=_pil_collate_fn,
        )

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"{model_name}/{split_key}"):
            x = _preprocess_batch(imgs, preprocess, device)
            all_logits.append(model(x).cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    os.makedirs(os.path.join(cache_dir, model_name), exist_ok=True)
    torch.save({"logits": logits, "labels": labels}, cache_path)
    if verbose:
        print(f"[logit cache] saved {cache_path}")

    return logits, labels
