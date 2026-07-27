"""
running_mean.py
===============
Simple averaging (eq. 10): the running mean of every score since the last
reset — an EMA whose gain decays as 1/t. Noise vanishes asymptotically, but
the estimate becomes inert late in a corruption, when reliability may be
genuinely drifting (e.g. a model starting to collapse under adaptation).
Appropriate only if reliability is truly constant within a corruption.
"""

from __future__ import annotations

from src.reliability.filters.base import ScoreFilter


class RunningMean(ScoreFilter):
    def __init__(self):
        self._sum = 0.0
        self._n = 0

    def update(self, x: float) -> float:
        self._sum += x
        self._n += 1
        return self._sum / self._n

    def reset(self) -> None:
        self._sum = 0.0
        self._n = 0


if __name__ == "__main__":
    f = RunningMean()
    vals = [1.0, 2.0, 3.0, 4.0]
    running = []
    for v in vals:
        running.append(f.update(v))
    assert running == [1.0, 1.5, 2.0, 2.5], running

    f.reset()
    assert f.update(10.0) == 10.0  # back to a fresh mean after reset
    print("RunningMean self-test passed")
