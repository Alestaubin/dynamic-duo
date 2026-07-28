"""
atc.py
======
Averaged Threshold Confidence: predicted accuracy = P(score >= t), with the
threshold t fitted on source data such that P(score < t) = source error rate.
"""

from __future__ import annotations

import torch

from src.reliability.proxies.base import Proxy, register


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


@register
class ATCProxy(Proxy):
    name = "atc"

    def __init__(self, kind: str = "neg_entropy"):
        self.kind = kind
        self.threshold: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.threshold is not None

    def fit_source(self, logits, features, labels, num_classes) -> None:
        self.threshold = fit_atc_threshold(logits, labels, self.kind)

    def score(self, logits: torch.Tensor, features: torch.Tensor | None, labels: torch.Tensor | None = None) -> float:
        assert self.threshold is not None, \
            "ATCProxy not fitted; call fit_source() or build_proxy_stats()"
        return atc_score(logits, self.threshold, self.kind)

    def state_dict(self) -> dict:
        return {"threshold": self.threshold, "kind": self.kind}

    def load_state_dict(self, state: dict) -> None:
        self.threshold = state["threshold"]
        self.kind = state.get("kind", "neg_entropy")
