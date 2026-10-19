"""
grids.py
========
The learning-rate grids of the Dynamic Duos brief (Phase 3, "Settled decisions"): SGD with momentum 0.9 and batch size 64.

    ResNet-50   {5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3}     the published 2.5e-4 times {0.2, 0.4, 1, 2, 4}
    ViT-B/16    {1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3}   SAR's published 1e-3 times {0.1, 0.25, 0.5, 1, 2.5}

If a method's published learning rate falls outside its grid the grid is re-centred on it with the SAME MULTIPLIERS
(`recentred`): CoTTA and RMT publish 1e-2 for ResNet-50, giving {2e-3, 4e-3, 1e-2, 2e-2, 4e-2}. Other ViT grids stay as
they are, since the upstream benchmark publishes no ViT-B/16 configuration besides SAR's. (Both methods are deferred.)

Because the members adapt independently, run i of a sweep uses the i-th value of EACH model's grid, and the best large
and best small member are paired offline afterwards (src/experiments/logit_store.py: PairView), so a sweep costs as
many runs as grid points, not their product.
"""

from __future__ import annotations

MULTIPLIERS = {"resnet50": (0.2, 0.4, 1.0, 2.0, 4.0), "vit_b_16": (0.1, 0.25, 0.5, 1.0, 2.5)}
CENTRE = {"resnet50": 2.5e-4, "vit_b_16": 1e-3}                      # the value the multipliers are relative to
_FAMILY = {"resnet50": "resnet50", "resnet50_v1": "resnet50", "vit_b_16": "vit_b_16", "vit_b_16_augreg": "vit_b_16",
           "vit_b_16_augreg2": "vit_b_16", "vit_b_16_sar": "vit_b_16"}


def family(model_name: str) -> str:
    try:
        return _FAMILY[model_name]
    except KeyError:
        raise ValueError(f"no learning-rate grid for model {model_name!r}: known {sorted(_FAMILY)}") from None


def lr_grid(model_name: str) -> list[float]:
    fam = family(model_name)
    return [float(f"{CENTRE[fam] * m:.6g}") for m in MULTIPLIERS[fam]]


def recentred(model_name: str, published_lr: float) -> list[float]:
    """The model's grid with the same multipliers around `published_lr`."""
    fam = family(model_name)
    return [float(f"{published_lr * m:.6g}") for m in MULTIPLIERS[fam]]


def sgd_optim(lr: float, momentum: float = 0.9) -> dict:
    """An OPTIM block as src/tta/tent.py::setup_optimizer reads it."""
    return {"METHOD": "SGD", "STEPS": 1, "LR": lr, "MOMENTUM": momentum, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}


if __name__ == "__main__":
    assert lr_grid("resnet50_v1") == [5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3] == lr_grid("resnet50")
    assert lr_grid("vit_b_16") == [1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3] == lr_grid("vit_b_16_sar")
    assert recentred("resnet50_v1", 1e-2) == [2e-3, 4e-3, 1e-2, 2e-2, 4e-2]
    assert recentred("resnet50", 2.5e-4) == lr_grid("resnet50"), "re-centring on the published value changes nothing"
    assert sgd_optim(1e-3)["MOMENTUM"] == 0.9 and sgd_optim(1e-3, 0.0)["MOMENTUM"] == 0.0
    try:
        lr_grid("convnext_tiny")
        raise SystemExit("an unknown model has no grid")
    except ValueError:
        pass
    print("grids self-test passed")
