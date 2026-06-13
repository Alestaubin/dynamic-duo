import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from src.utils.data import _pil_collate_fn
from src.utils.model import get_model, _preprocess_batch


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (logits, labels) for the given model and data split.

    Pass corruption=None for ImageNet val; otherwise ImageNet-C with the
    given corruption type and severity level.  Results are saved under
    cache_dir/<model_name>/<split_key>.pt so subsequent calls are instant.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_key = "val" if corruption is None else f"{corruption}_{severity}"
    cache_path = os.path.join(cache_dir, model_name, f"{split_key}.pt")

    if os.path.exists(cache_path):
        print(f"[logit cache] hit  {model_name}/{split_key}")
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        return saved["logits"], saved["labels"]

    print(f"[logit cache] miss {model_name}/{split_key} — running inference...")
    model, preprocess = get_model(model_name, freeze=True)
    model = model.to(device).eval()

    if corruption is None:
        ds = datasets.ImageFolder(val_dir)
    else:
        ds = datasets.ImageFolder(os.path.join(test_dir, corruption, str(severity)))

    # shuffle=False is required: logits are cached per-model and later aligned by
    # index across models, so every model must iterate the data in the same order.
    loader = DataLoader(
        ds,
        batch_size=batch_size, shuffle=False,
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
    print(f"[logit cache] saved {cache_path}")

    return logits, labels
