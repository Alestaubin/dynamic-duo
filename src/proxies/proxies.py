"""
proxies.py
==========
Per-batch reliability proxies for heterogeneous model pairs.

Provides three proxy functions (nuclear_norm, atc, prototype), the
ModelProxyConfig dataclass that holds per-model state (thresholds, prototypes,
calibration maps), FeatureExtractor for hook-based penultimate feature capture,
and build_proxy_configs for building both configs from a source dataloader.

Used by JointSoftAnchor (calibrator) and proxy_benchmark (eval harness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm


# ─── Raw proxy functions ─────────────────────────────────────────────────────

@torch.no_grad()
def nuclear_norm_score(logits: torch.Tensor) -> float:
    """Confidence + dispersity via the nuclear norm of the softmax matrix.

    Normalised by sqrt(N * min(N, C)) to keep values in [0, 1] and reduce
    batch-size dependence. Penalises TENT collapse (low rank) by construction.
    """
    p = torch.softmax(logits, dim=1)
    n, c = p.shape
    nuc = torch.linalg.matrix_norm(p, ord="nuc")
    return float(nuc / math.sqrt(n * min(n, c)))


@torch.no_grad()
def _atc_sample_scores(logits: torch.Tensor, kind: str = "neg_entropy") -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    if kind == "maxconf":
        return p.max(dim=1).values
    if kind == "neg_entropy":
        return (p * torch.log_softmax(logits, dim=1)).sum(dim=1)
    raise ValueError(kind)


@torch.no_grad()
def fit_atc_threshold(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    kind: str = "neg_entropy",
) -> float:
    """Fit ATC threshold on source data: t s.t. P(score < t) = source error rate."""
    scores = _atc_sample_scores(source_logits, kind)
    err_rate = 1.0 - (source_logits.argmax(1) == source_labels).float().mean().item()
    return torch.quantile(scores.float(), max(min(err_rate, 1.0), 0.0)).item()


@torch.no_grad()
def atc_score(logits: torch.Tensor, threshold: float, kind: str = "neg_entropy") -> float:
    """ATC predicted accuracy = fraction of samples with score >= threshold."""
    return float((_atc_sample_scores(logits, kind) >= threshold).float().mean())


@torch.no_grad()
def build_prototypes(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """L2-normalised mean penultimate feature per class, (num_classes, D).
    Empty classes get a zero vector (they are never nearest-prototype).
    """
    protos = torch.zeros(num_classes, features.shape[1], device=features.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            protos[c] = features[mask].mean(0)
    return F.normalize(protos, dim=1)


@torch.no_grad()
def prototype_score(features: torch.Tensor, prototypes: torch.Tensor) -> float:
    """Mean nearest-prototype cosine similarity.

    Cosine (not Euclidean) so 768-d ViT and 2048-d ResNet spaces are scale-comparable.
    features and prototypes must be on the same device.
    """
    f = F.normalize(features, dim=1)
    nearest = (f @ prototypes.t()).max(dim=1).values
    return float(nearest.mean())


# ─── Per-model proxy configuration ───────────────────────────────────────────

@dataclass
class ModelProxyConfig:
    """Everything proxy-related for ONE model.

    Built offline from source data; used by JointSoftAnchor and the benchmark.
    """
    name: str
    num_classes: int
    atc_threshold: float | None = None
    prototypes: torch.Tensor | None = None
    atc_kind: str = "neg_entropy"
    # isotonic maps: raw proxy value → predicted accuracy (fitted by benchmark)
    calib: dict[str, IsotonicRegression] = field(default_factory=dict)

    def raw_proxies(self, logits: torch.Tensor, features: torch.Tensor) -> dict[str, float]:
        out = {"nuclear_norm": nuclear_norm_score(logits)}
        if self.atc_threshold is not None:
            out["atc"] = atc_score(logits, self.atc_threshold, self.atc_kind)
        if self.prototypes is not None:
            out["prototype"] = prototype_score(features, self.prototypes)
        return out

    def predicted_acc(self, proxy_name: str, raw_value: float) -> float:
        if proxy_name in self.calib:
            return float(self.calib[proxy_name].predict([raw_value])[0])
        return raw_value


# ─── Feature extraction via forward hooks ────────────────────────────────────

def _hook_spec(model: nn.Module, model_name: str):
    """Return (layer_to_hook, output_transform) for penultimate feature capture.

    vit_b_16  : encoder output (N, seq, dim) → CLS token (N, dim=768)
    resnet50  : avgpool output (N, C, 1, 1)  → flatten (N, C=2048)
    """
    name = model_name.lower()
    if "vit" in name:
        return model.encoder, lambda out: out[:, 0, :]
    if "resnet" in name:
        return model.avgpool, lambda out: out.flatten(1)
    raise ValueError(
        f"No feature-hook spec for model '{model_name}'. Add it to _hook_spec()."
    )


class FeatureExtractor:
    """Wraps a model; captures penultimate features alongside logits via a forward hook."""

    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self._feats: torch.Tensor | None = None
        layer, self._transform = _hook_spec(model, model_name)
        self._handle = layer.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        self._feats = self._transform(out).detach()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        logits = self.model(x)
        return logits.detach(), self._feats

    def remove(self):
        self._handle.remove()


# ─── Source-data helpers ──────────────────────────────────────────────────────

@torch.no_grad()
def _source_pass(ext_l, preprocess_l, ext_s, preprocess_s, loader, device):
    """Run both models over the source loader; return logits, features, labels."""
    z_l, f_l, z_s, f_s, labs = [], [], [], [], []
    for imgs, labels in tqdm(loader, desc="source pass"):
        xl = torch.stack([preprocess_l(img) for img in imgs]).to(device)
        xs = torch.stack([preprocess_s(img) for img in imgs]).to(device)
        zl, fl = ext_l(xl)
        zs, fs = ext_s(xs)
        z_l.append(zl.cpu()); f_l.append(fl.cpu())
        z_s.append(zs.cpu()); f_s.append(fs.cpu())
        labs.append(labels.cpu())
    return (torch.cat(z_l), torch.cat(f_l),
            torch.cat(z_s), torch.cat(f_s),
            torch.cat(labs))


def build_proxy_configs(
    large_model: nn.Module,
    large_preprocess,
    large_name: str,
    small_model: nn.Module,
    small_preprocess,
    small_name: str,
    source_loader,
    device: torch.device,
    num_classes: int = 1000,
) -> tuple[ModelProxyConfig, ModelProxyConfig]:
    """Build ModelProxyConfigs for both models from clean source data.

    Registers and removes feature hooks internally; the models are left unchanged.
    Returns (cfg_large, cfg_small) with fitted ATC thresholds and prototypes.
    """
    ext_l = FeatureExtractor(large_model, large_name)
    ext_s = FeatureExtractor(small_model, small_name)
    try:
        zl, fl, zs, fs, labs = _source_pass(
            ext_l, large_preprocess, ext_s, small_preprocess, source_loader, device
        )
    finally:
        ext_l.remove()
        ext_s.remove()

    cfg_l = ModelProxyConfig(
        name=large_name, num_classes=num_classes,
        atc_threshold=fit_atc_threshold(zl, labs),
        prototypes=build_prototypes(fl, labs, num_classes),
    )
    cfg_s = ModelProxyConfig(
        name=small_name, num_classes=num_classes,
        atc_threshold=fit_atc_threshold(zs, labs),
        prototypes=build_prototypes(fs, labs, num_classes),
    )
    return cfg_l, cfg_s
