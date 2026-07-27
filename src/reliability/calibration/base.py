"""
base.py
=======
Shared protocol for Section-3 calibration maps: fit a scalar map from a raw
proxy score to a predicted batch accuracy, then predict on new scores.

Every map in this package (identity, linear, platt, beta, isotonic) exposes
the same two methods so `maps.py`'s CalibrationMaps can hold, save, and load
any of them interchangeably. Per the paper (Section 3), the only requirement
on a map is monotonicity in the raw score; each candidate below trades off
where it sits on that ladder (fewer assumptions <-> more fitting data needed).
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class CalibrationMap(Protocol):
    def fit(self, xs: Iterable[float], ys: Iterable[float]) -> "CalibrationMap": ...
    def predict(self, x: float) -> float: ...
