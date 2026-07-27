"""
isotonic.py
===========
Isotonic regression (pool-adjacent-violators): the nonparametric map matched
exactly to the Section-3 monotonicity assumption. Corrects any monotone
distortion, at the cost of overfitting when the calibration set is small and
of boundary outputs in {0, 1} (handled by the y_min/y_max clip).

Thin wrapper around sklearn.isotonic.IsotonicRegression so it exposes the
same fit/predict(scalar) interface as the other calibration maps.
"""

from __future__ import annotations

from typing import Iterable

from sklearn.isotonic import IsotonicRegression


class IsotonicMap:
    def __init__(self, y_min: float = 0.0, y_max: float = 1.0, increasing: bool | str = True):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=y_min, y_max=y_max, increasing=increasing)

    def fit(self, xs: Iterable[float], ys: Iterable[float]) -> "IsotonicMap":
        self._iso.fit(list(xs), list(ys))
        return self

    def predict(self, x: float) -> float:
        return float(self._iso.predict([x])[0])


if __name__ == "__main__":
    xs = [0.1, 0.2, 0.3, 0.4, 0.5]
    ys = [0.2, 0.2, 0.5, 0.5, 0.9]
    m = IsotonicMap().fit(xs, ys)
    preds = [m.predict(x) for x in xs]
    assert all(preds[i] <= preds[i + 1] for i in range(len(preds) - 1)), preds
    assert m.predict(10.0) <= 1.0  # clipped
    print("IsotonicMap self-test passed")
