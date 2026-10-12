"""
linear.py
=========
g(r) = alpha*r + gamma, least-squares fit — the default of the unsupervised
accuracy-estimation literature. Two parameters, so essentially immune to
overfitting, but assumes the score-accuracy relation is linear over the whole
operating range; output can leave [0, 1] (the caller clips).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class LinearMap:
    def __init__(self):
        self.alpha = 1.0
        self.gamma = 0.0

    def fit(self, xs: Iterable[float], ys: Iterable[float]) -> "LinearMap":
        x = np.asarray(list(xs), dtype=np.float64)
        y = np.asarray(list(ys), dtype=np.float64)
        A = np.stack([x, np.ones_like(x)], axis=1)
        (self.alpha, self.gamma), *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, x: float) -> float:
        return float(self.alpha * x + self.gamma)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    xs = np.linspace(0, 1, 50)
    ys = 0.6 * xs + 0.2 + rng.normal(0, 0.01, size=50)
    m = LinearMap().fit(xs, ys)
    assert abs(m.alpha - 0.6) < 0.05 and abs(m.gamma - 0.2) < 0.05
    assert abs(m.predict(0.5) - (m.alpha * 0.5 + m.gamma)) < 1e-9
    print("LinearMap self-test passed")
