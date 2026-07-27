"""
prototype.py
============
Nearest-class-prototype reliability proxy, under one of two source-fitted
metrics selected by `metric`:

  "cosine"      — L2-normalised mean penultimate feature per class; score is
                  mean nearest-prototype cosine similarity.
  "mahalanobis" — raw per-class means + a tied (shared-across-classes)
                  precision matrix (Lee et al. 2018); score is the NEGATIVE
                  mean nearest-class squared Mahalanobis distance (negated so
                  higher is still "more reliable").
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from src.reliability.proxies.base import Proxy, register


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


@torch.no_grad()
def build_class_means(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (un-normalised) mean penultimate feature per PRESENT class.

    Returns (means, class_ids): means is (C_present, D), class_ids the matching
    class indices. Empty classes are dropped — unlike cosine (where a zero row
    is simply never nearest), a zero mean would be a spurious attractor under
    Mahalanobis.
    """
    means, present = [], []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            present.append(c)
            means.append(features[mask].mean(0))
    return torch.stack(means), torch.tensor(present, dtype=torch.long)


@torch.no_grad()
def build_tied_precision(
    features: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
    class_ids: torch.Tensor,
    shrinkage: float = 1e-2,
) -> torch.Tensor:
    """Inverse of the tied (shared-across-classes) within-class covariance.

    Pools per-class-centred deviations across all samples — the standard
    Mahalanobis-OOD estimator (Lee et al. 2018) — then diagonally shrinks for
    invertibility at D up to 2048:  Σ ← Σ + shrinkage · mean(diag Σ) · I.
    """
    D = features.shape[1]
    row_of = torch.full((int(labels.max()) + 1,), -1, dtype=torch.long)
    row_of[class_ids] = torch.arange(len(class_ids))
    rows = row_of[labels]
    valid = rows >= 0                                        # drop dropped-class samples
    centred = features[valid] - means[rows[valid]]          # (N, D)
    cov = (centred.t() @ centred) / centred.shape[0]        # (D, D)
    ridge = shrinkage * cov.diagonal().mean()
    cov = cov + ridge * torch.eye(D, dtype=cov.dtype, device=cov.device)
    return torch.linalg.inv(cov)


@torch.no_grad()
def mahalanobis_score(
    features: torch.Tensor, class_means: torch.Tensor, precision: torch.Tensor
) -> float:
    """NEGATIVE mean nearest-class squared Mahalanobis distance.

    Negated so that, like cosine, HIGHER = closer to the source class clusters =
    more reliable — keeping the proxy's "higher is better" convention and the
    isotonic calibration's increasing=True assumption.

    Uses dᴹ² = xᵀPx − 2 xᵀPμ + μᵀPμ to avoid materialising the (N, C, D)
    difference tensor. features / class_means / precision share a device.

    Unlike cosine, raw magnitude is NOT cross-model comparable (squared
    Mahalanobis is ~χ² with df = feature dim, so 768-d and 2048-d differ in
    scale); rely on the calibration map for cross-model selection.
    """
    fp = features @ precision                               # (N, D)
    term_x = (fp * features).sum(1, keepdim=True)           # (N, 1)
    mp = class_means @ precision                            # (C, D)
    term_m = (mp * class_means).sum(1)                      # (C,)
    cross = fp @ class_means.t()                            # (N, C)
    d2 = term_x - 2.0 * cross + term_m.unsqueeze(0)         # (N, C)
    nearest = d2.min(dim=1).values                         # (N,)
    return float((-nearest).mean())


@register
class PrototypeProxy(Proxy):
    name = "prototype"

    def __init__(
        self,
        metric: Literal["cosine", "mahalanobis"] = "cosine",
        cov_shrinkage: float = 1e-2,
    ):
        self.metric = metric
        self.cov_shrinkage = cov_shrinkage
        self.prototypes: torch.Tensor | None = None      # (C, D), cosine metric
        self.class_means: torch.Tensor | None = None     # (C', D), mahalanobis metric
        self.precision: torch.Tensor | None = None        # (D, D), mahalanobis metric

    @property
    def is_fitted(self) -> bool:
        if self.metric == "mahalanobis":
            return self.class_means is not None and self.precision is not None
        return self.prototypes is not None

    def fit_source(self, logits, features, labels, num_classes) -> None:
        if self.metric == "mahalanobis":
            means, ids = build_class_means(features, labels, num_classes)
            self.class_means = means
            self.precision = build_tied_precision(features, labels, means, ids, self.cov_shrinkage)
        else:
            self.prototypes = build_prototypes(features, labels, num_classes)

    def score(self, logits: torch.Tensor, features: torch.Tensor) -> float:
        """Lazily moves the stored source state onto `features`' device,
        caching it in place."""
        assert self.is_fitted, \
            "PrototypeProxy not fitted; call fit_source() or build_proxy_stats()"
        dev = features.device
        if self.metric == "mahalanobis":
            if self.class_means.device != dev:
                self.class_means = self.class_means.to(dev)
                self.precision = self.precision.to(dev)
            return mahalanobis_score(features, self.class_means, self.precision)
        if self.prototypes.device != dev:
            self.prototypes = self.prototypes.to(dev)
        return prototype_score(features, self.prototypes)

    def state_dict(self) -> dict:
        return {
            "metric": self.metric,
            "cov_shrinkage": self.cov_shrinkage,
            "prototypes": self.prototypes,
            "class_means": self.class_means,
            "precision": self.precision,
        }

    def load_state_dict(self, state: dict) -> None:
        self.metric = state.get("metric", "cosine")
        self.cov_shrinkage = state.get("cov_shrinkage", 1e-2)
        self.prototypes = state.get("prototypes")
        self.class_means = state.get("class_means")
        self.precision = state.get("precision")
