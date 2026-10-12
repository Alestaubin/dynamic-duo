"""
oracle.py
=========
Cheating oracle proxy: reports the model's ACTUAL per-batch accuracy,
computed from ground-truth labels, as the reliability score.

This is not a real proxy — every other proxy in this package is label-free
by construction, since labels aren't available at test time — but it is a
useful reference point: gating JointProxyWeighted on this signal shows what
the duo could achieve with a perfect (label-aware) reliability estimate,
isolating how much of the gap to that ceiling is due to the label-free
proxies' imperfection versus the calibration/filter/gate stages downstream.

Requires labels every call; use only where they're actually available (e.g.
test-set evaluation, which does have labels — it just doesn't use them for
adaptation). Never use this as a real deployment signal.
"""

from __future__ import annotations

import torch

from src.reliability.proxies.base import Proxy, register


@torch.no_grad()
def oracle_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels.to(logits.device)).float().mean())


@register
class OracleProxy(Proxy):
    name = "oracle"
    requires_labels = True

    def score(
        self,
        logits: torch.Tensor,
        features: torch.Tensor | None,
        labels: torch.Tensor | None = None,
    ) -> float:
        assert labels is not None, \
            "OracleProxy.score() requires labels — it cheats by construction " \
            "and cannot be used where labels are unavailable."
        return oracle_score(logits, labels)


if __name__ == "__main__":
    torch.manual_seed(0)
    K, B = 10, 32

    labels = torch.randint(0, K, (B,))
    logits_all_correct = torch.zeros(B, K)
    logits_all_correct[torch.arange(B), labels] = 10.0
    assert oracle_score(logits_all_correct, labels) == 1.0

    # First half always predicts class 0 (correct only when labels happen to
    # be 0 there); second half is untouched (still all-correct).
    logits_half_correct = logits_all_correct.clone()
    logits_half_correct[: B // 2] = 0.0
    logits_half_correct[: B // 2, 0] = 10.0
    n_correct_first_half = int((labels[: B // 2] == 0).sum())
    expected = (n_correct_first_half + B // 2) / B
    got = oracle_score(logits_half_correct, labels)
    assert abs(got - expected) < 1e-6, (got, expected)

    proxy = OracleProxy()
    assert proxy.is_fitted  # stateless
    assert proxy.score(logits_all_correct, None, labels=labels) == 1.0
    try:
        proxy.score(logits_all_correct, None, labels=None)
        raise AssertionError("expected an AssertionError when labels is None")
    except AssertionError as e:
        assert "requires labels" in str(e)

    print("oracle self-test passed")
