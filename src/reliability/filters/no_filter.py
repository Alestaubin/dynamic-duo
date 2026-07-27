"""
no_filter.py
============
Baseline: gate directly on the raw calibrated score. Reacts instantly but
transmits the full per-batch proxy noise into the ensemble weights — the
reference point every other filter must beat.
"""

from __future__ import annotations

from src.reliability.filters.base import ScoreFilter


class NoFilter(ScoreFilter):
    def update(self, x: float) -> float:
        return x

    def reset(self) -> None:
        pass


if __name__ == "__main__":
    f = NoFilter()
    for x in (0.1, 5.0, -3.0):
        assert f.update(x) == x
    f.reset()  # no-op, doesn't raise
    print("NoFilter self-test passed")
