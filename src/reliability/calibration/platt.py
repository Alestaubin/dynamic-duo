"""
platt.py
========
Platt scaling: g(r) = sigmoid(a*r + b), a two-parameter monotone sigmoid
fit by nonlinear least squares against the (raw_score, batch_accuracy)
calibration pairs. Relaxes linear.py's straight line to a fixed sigmoid
shape while keeping a low-dimensional (sample-efficient) fit.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit


def _sigmoid(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * x + b)))


class PlattMap:
    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, xs: Iterable[float], ys: Iterable[float]) -> "PlattMap":
        x = np.asarray(list(xs), dtype=np.float64)
        y = np.clip(np.asarray(list(ys), dtype=np.float64), 1e-6, 1 - 1e-6)
        (self.a, self.b), *_ = curve_fit(_sigmoid, x, y, p0=(1.0, 0.0), maxfev=5000)
        return self

    def predict(self, x: float) -> float:
        return float(_sigmoid(np.array(x), self.a, self.b))


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    xs = np.linspace(-5, 5, 60)
    ys = 1.0 / (1.0 + np.exp(-(2.0 * xs - 1.0))) + rng.normal(0, 0.01, size=60)
    ys = np.clip(ys, 0.0, 1.0)
    m = PlattMap().fit(xs, ys)
    assert abs(m.a - 2.0) < 0.3 and abs(m.b - (-1.0)) < 0.3, (m.a, m.b)
    # Monotone increasing in x for a > 0.
    assert m.predict(2.0) > m.predict(-2.0)
    print("PlattMap self-test passed")
