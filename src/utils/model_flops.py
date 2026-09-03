"""
model_flops.py
===============
Prints GFLOPs, GMACs, and parameter count for one or more models, computed
on a single real preprocessed image from ImageNet val — each model's own
`preprocess` (different input resolutions/normalization across families) is
applied, rather than a made-up fixed-size dummy tensor.

Usage:
    python -m src.utils.model_flops --config cfgs/dynamic_duo_config.yaml --models resnet50 vit_b_16 efficientnet_b4
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.flop_counter import FlopCounterMode
from torchvision import datasets

from src.utils.data import load_config
from src.utils.model import get_model


def count_flops(model, input_tensor):
    flop_counter = FlopCounterMode(display=False)
    with flop_counter, torch.no_grad():
        model(input_tensor)
    return flop_counter.get_total_flops()


def profile_model(model_name, image, device):
    model, preprocess = get_model(model_name, freeze=True, verbose=False)
    model = model.to(device).eval()
    x = preprocess(image).unsqueeze(0).to(device)

    flops = count_flops(model, x)
    params = sum(p.numel() for p in model.parameters())

    return {
        "model": model_name,
        "gflops": flops / 1e9,
        "gmacs": flops / 2e9,
        "params_m": params / 1e6,
        "input_shape": tuple(x.shape),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", required=True,
                         help="Model names as recognized by src.utils.model.get_model")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = datasets.ImageFolder(cfg["VAL_DIR"])
    image, _ = val_ds[0]

    rows = [profile_model(name, image, device) for name in args.models]

    header = f"{'model':<22}{'GFLOPs':>10}{'GMACs':>10}{'Params(M)':>12}{'Input shape':>20}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['model']:<22}{row['gflops']:>10.3f}{row['gmacs']:>10.3f}"
              f"{row['params_m']:>12.3f}{str(row['input_shape']):>20}")


if __name__ == "__main__":
    main()
