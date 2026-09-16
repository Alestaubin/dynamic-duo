import os
import yaml
import torch
from torchvision import datasets


def _pil_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

def load_imagenetC(data_dir, severities, corruption_types, device, batch_size=256, num_workers=4, num_samples=None, seed=None):
    """
    Load the ImageNet-C dataset for a given corruption type and severity level.
    Returns a DataLoader of (list[PIL.Image], LongTensor) batches — no preprocessing
    applied so each model can apply its own transform in forward.

    num_samples: number of samples to use from each subset (if None, use all).
    """

    shuffle = True

    if isinstance(severities, int):
        severities = [severities]
    if isinstance(corruption_types, str):
        corruption_types = [corruption_types]

    subsets = []
    for corruption in corruption_types:
        for severity in severities:
            ds = datasets.ImageFolder(os.path.join(data_dir, corruption, str(severity)))
            if num_samples is not None:
                n = min(num_samples, len(ds))
                print(f"Sampling {n} examples from {len(ds)} for corruption {corruption} severity {severity}")
                generator = torch.Generator().manual_seed(seed) if seed is not None else None
                indices = torch.randperm(len(ds), generator=generator)[:n].tolist() if shuffle else list(range(n))
                ds = torch.utils.data.Subset(ds, indices)
            subsets.append(ds)

    combined = torch.utils.data.ConcatDataset(subsets)
    pin_memory = device.type == "cuda"

    # Seed the DataLoader's shuffle so batch order is reproducible too.
    loader_generator = torch.Generator().manual_seed(seed) if seed is not None else None

    return torch.utils.data.DataLoader(
        combined, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_pil_collate_fn,
        generator=loader_generator,
    )

def load_imagenet_val(data_dir, device, batch_size=256, num_workers=4, num_samples=None, seed=None):
    ds = datasets.ImageFolder(os.path.join(data_dir, "val"))
    if num_samples is not None:
        n = min(num_samples, len(ds))
        print(f"Sampling {n} examples from {len(ds)} for clean val")
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        indices = torch.randperm(len(ds), generator=generator)[:n].tolist()
        ds = torch.utils.data.Subset(ds, indices)
    pin_memory = device.type == "cuda"
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_pil_collate_fn,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )

def _assert_no_duplicate_entries(cfg: dict, section: str) -> None:
    """EVAL/CALIBRATOR's CORRUPTIONS and SEVERITIES feed a nested for-loop
    (evaluate_dynamic_duo, src.reliability.setup.fit_beta) that has no notion
    of "already ran this pair" -- a duplicate entry in either list (e.g. from
    uncommenting two overlapping yaml blocks by mistake) silently re-runs that
    (corruption, severity) as a second, independent pass. Every per-corruption
    diagnostic keyed only by the "{corruption}/s{severity}" label
    (JointProxyWeighted's proxy log CSV, plot_run_diagnostics.py's
    batch_diagnostics.csv / per-corruption CSVs) then silently merges both
    passes' rows into one -- e.g. a requested num_samples=50000 stream
    showing up as 100000 samples in one per-corruption plot, as happened with
    a duplicated 'brightness' entry in dynamic_duo_config_vitb_resnet.yaml.
    Fail fast here instead of burning 2x the compute on a run whose
    per-corruption output is then quietly wrong.
    """
    sec = cfg.get(section)
    if not sec:
        return
    for key in ("CORRUPTIONS", "SEVERITIES"):
        values = sec.get(key)
        if not values:
            continue
        seen = set()
        dupes = sorted({v for v in values if v in seen or seen.add(v)}, key=str)
        if dupes:
            raise ValueError(
                f"{section}.{key} has duplicate entries {dupes} -- each duplicate silently "
                f"re-runs that corruption/severity as a second pass and merges into the SAME "
                f"per-corruption diagnostics (see src/utils/data.py's _assert_no_duplicate_entries "
                f"for why). Remove the duplicate(s) from the config."
            )


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    _assert_no_duplicate_entries(cfg, "EVAL")
    _assert_no_duplicate_entries(cfg, "CALIBRATOR")
    return cfg
