"""
cot.py
======
Confidence Optimal Transport (COT): scores a model by the negated Wasserstein
distance, under an L_inf ground metric, between the batch's softmax rows and
`b_t` one-hot vectors drawn from the source label marginal. https://arxiv.org/abs/2305.15640

Ground-cost simplification
---------------------------
For a softmax row c (on the simplex) and a one-hot vector at class k,
    ||c - onehot_k||_inf = max(1 - c_k, max_{i != k} c_i) = 1 - c_k,
since max_{i!=k} c_i <= sum_{i!=k} c_i = 1 - c_k always. So the bt x bt L_inf
cost matrix is simply `1 - P[:, sampled_classes]` — no need to materialise
full one-hot vectors.

Both point clouds are uniform over bt points, so the Wasserstein distance is
exactly the optimal (minimum total cost) assignment's average edge cost —
solved here by `scipy.optimize.linear_sum_assignment` (Hungarian algorithm,
O(b_t^3) worst case, matching the paper's stated complexity).

Needs the source label marginal (fit_source); stateful but lightweight (one
length-K probability vector).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.reliability.proxies.base import Proxy, register


def _min_cost_assignment(cost: np.ndarray) -> float:
    """Average edge cost of the min-total-cost perfect matching of a square cost matrix."""
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


@torch.no_grad()
def cot_score(logits: torch.Tensor, class_probs: torch.Tensor, generator: torch.Generator | None = None) -> float:
    """Negated W(batch softmax rows, source label marginal) under an L_inf ground metric."""
    p = torch.softmax(logits, dim=1)
    bt = p.shape[0]
    sampled = torch.multinomial(class_probs, bt, replacement=True, generator=generator)
    cost = (1.0 - p[:, sampled]).cpu().numpy().astype(np.float64)
    return -_min_cost_assignment(cost)


@register
class CotProxy(Proxy):
    name = "cot"

    def __init__(self):
        self.class_probs: torch.Tensor | None = None

    @property
    def is_fitted(self) -> bool:
        return self.class_probs is not None

    def fit_source(self, logits, features, labels, num_classes) -> None:
        counts = torch.bincount(labels, minlength=num_classes).float()
        self.class_probs = counts / counts.sum()

    def score(self, logits: torch.Tensor, features: torch.Tensor | None) -> float:
        assert self.class_probs is not None, \
            "CotProxy not fitted; call fit_source() or build_proxy_stats()"
        probs = self.class_probs.to(logits.device)
        return cot_score(logits, probs)

    def state_dict(self) -> dict:
        return {"class_probs": self.class_probs}

    def load_state_dict(self, state: dict) -> None:
        self.class_probs = state["class_probs"]


if __name__ == "__main__":
    from itertools import permutations

    # _min_cost_assignment matches brute-force on a tiny random cost matrix.
    rng = np.random.RandomState(0)
    for _ in range(20):
        n = 4
        cost = rng.rand(n, n)
        brute = min(
            sum(cost[i, j] for i, j in enumerate(perm)) / n
            for perm in permutations(range(n))
        )
        got = _min_cost_assignment(cost)
        assert abs(brute - got) < 1e-9, (brute, got)
    print("cot min-cost-assignment brute-force self-test passed")

    # A batch whose predictions are diverse and track the source marginal
    # scores higher (lower average transport cost) than one collapsed onto a
    # single class, on average over many independent draws of the source
    # comparison sample (a single draw is noisy — the paper's own motivation
    # for the temporal filter in Section 4 — so we average trials here).
    torch.manual_seed(0)
    K, bt = 10, 200
    class_probs = torch.full((K,), 1.0 / K)
    collapsed = torch.zeros(bt, K); collapsed[:, 0] = 10.0
    good_labels = torch.multinomial(class_probs, bt, replacement=True)
    good = torch.full((bt, K), -2.0)
    good[torch.arange(bt), good_labels] = 6.0

    n_trials = 20
    s_good = sum(
        cot_score(good, class_probs, generator=torch.Generator().manual_seed(t))
        for t in range(n_trials)
    ) / n_trials
    s_collapsed = sum(
        cot_score(collapsed, class_probs, generator=torch.Generator().manual_seed(t))
        for t in range(n_trials)
    ) / n_trials
    assert s_good > s_collapsed, (s_good, s_collapsed)
    print(f"cot proxy self-test passed (avg good={s_good:.3f} vs collapsed={s_collapsed:.3f})")

    # fit_source / score round-trip through the Proxy interface.
    proxy = CotProxy()
    labels = torch.randint(0, K, (200,))
    proxy.fit_source(None, None, labels, K)
    assert proxy.is_fitted
    val = proxy.score(good, None)
    assert isinstance(val, float)
    print("CotProxy self-test passed")
