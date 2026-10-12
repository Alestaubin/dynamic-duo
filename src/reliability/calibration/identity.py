"""
identity.py
===========
g(r) = r — the baseline of the Section-3 calibration ladder. No fitting;
gates directly on the raw proxy gap. Measures what calibration is worth: if
it matches the other maps, the calibration stage (and its held-out-data
requirement) can be dropped.
"""

from __future__ import annotations

from typing import Iterable


class IdentityMap:
    def fit(self, xs: Iterable[float], ys: Iterable[float]) -> "IdentityMap":
        return self

    def predict(self, x: float) -> float:
        return float(x)


if __name__ == "__main__":
    m = IdentityMap().fit([0.1, 0.5, 0.9], [0.2, 0.6, 0.8])
    for x in (0.0, 0.3, 1.0):
        assert m.predict(x) == x
    print("IdentityMap self-test passed")
