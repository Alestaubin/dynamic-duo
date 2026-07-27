"""
beta.py
=======
Beta calibration (Kull, Silva Filho & Flach 2017): a three-parameter family
designed for scores already in [0, 1], strictly generalizing Platt scaling.

    logit(g(r)) = a*ln(r) - b*ln(1-r) + c,   a, b >= 0

fit by nonlinear least squares on the (raw_score, batch_accuracy) pairs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit


def _beta_calib(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)
    z = a * np.log(x) - b * np.log(1 - x) + c
    return 1.0 / (1.0 + np.exp(-z))


class BetaMap:
    def __init__(self):
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0

    def fit(self, xs: Iterable[float], ys: Iterable[float]) -> "BetaMap":
        x = np.clip(np.asarray(list(xs), dtype=np.float64), 1e-6, 1 - 1e-6)
        y = np.clip(np.asarray(list(ys), dtype=np.float64), 1e-6, 1 - 1e-6)
        (self.a, self.b, self.c), *_ = curve_fit(
            _beta_calib, x, y, p0=(1.0, 1.0, 0.0),
            bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        return self

    def predict(self, x: float) -> float:
        return float(_beta_calib(np.array(x), self.a, self.b, self.c))


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    xs = np.clip(np.linspace(0.01, 0.99, 60), 1e-3, 1 - 1e-3)
    ys = np.clip(_beta_calib(xs, 2.0, 1.0, 0.5) + rng.normal(0, 0.01, size=60), 1e-3, 1 - 1e-3)
    m = BetaMap().fit(xs, ys)
    # Fit recovers a reasonable approximation and stays monotone increasing.
    assert m.predict(0.8) > m.predict(0.2)
    assert m.predict(0.9) > m.predict(0.8)
    residual = np.abs(np.array([m.predict(x) for x in xs]) - ys).mean()
    assert residual < 0.05, residual
    print("BetaMap self-test passed")
