"""
agreement.py
============
Cross-model agreement: for two independently-trained models, the batch
disagreement rate approximates the test error, and agreement-on-the-line
shows OOD agreement tracks ID agreement with the same slope as OOD accuracy
tracks ID accuracy. https://arxiv.org/abs/2106.13799, https://arxiv.org/abs/2206.13089

Unlike the other proxies in this package, agreement scores the *pair*, not
a single model — there is no r_l / r_s split, so it does not implement the
Proxy interface (Proxy.score(logits, features) is inherently per-model) and
is NOT a member of PROXY_KINDS / ProxyStats. JointProxyWeighted's Section-5
gate needs a per-model reliability gap (s_l - s_s) to weight the two models
against each other, which this signal cannot produce on its own.

It's provided as a standalone diagnostic/shift-detection signal (e.g. a
sudden drop in agreement_rate is itself evidence of a distribution shift,
independent of which model to trust more) rather than as a gate input.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def agreement_rate(z_l: torch.Tensor, z_s: torch.Tensor) -> float:
    """Fraction of the batch where the two models' argmax predictions match."""
    return float((z_l.argmax(dim=1) == z_s.argmax(dim=1)).float().mean())


if __name__ == "__main__":
    torch.manual_seed(0)

    # Identical logits agree everywhere.
    z = torch.randn(32, 10)
    assert agreement_rate(z, z) == 1.0

    # Disjoint argmax classes never agree (10 classes, one model always picks
    # class 0, the other always picks class 1).
    z_l = torch.zeros(16, 10); z_l[:, 0] = 10.0
    z_s = torch.zeros(16, 10); z_s[:, 1] = 10.0
    assert agreement_rate(z_l, z_s) == 0.0

    # Partial agreement lands strictly between 0 and 1.
    z_l = torch.randn(100, 5)
    z_s = z_l.clone()
    z_s[:40] = torch.randn(40, 5)  # perturb 40/100 rows so some argmax flip
    r = agreement_rate(z_l, z_s)
    assert 0.0 < r <= 1.0
    print("agreement_rate self-test passed")
