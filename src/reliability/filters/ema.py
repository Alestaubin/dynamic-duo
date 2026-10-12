"""
ema.py
======
Exponential Moving Average (eq. 11) with smoothing factor alpha in (0, 1].
Resets to a fixed prior (typically logit(val_acc) for the model — see
src.reliability.calibration for the logit transform) at each corruption
boundary, matching the model's own TENT reset there.
"""

from __future__ import annotations

from src.reliability.filters.base import ScoreFilter


class EMA(ScoreFilter):
    def __init__(self, alpha: float, prior: float = 0.0):
        assert 0.0 < alpha <= 1.0, f"alpha must be in (0, 1], got {alpha}"
        self.alpha = alpha
        self.prior = prior
        self._x = prior

    def update(self, x: float) -> float:
        self._x = self.alpha * x + (1.0 - self.alpha) * self._x
        return self._x

    def reset(self) -> None:
        self._x = self.prior


if __name__ == "__main__":
    # alpha=1 reduces to NoFilter (no memory).
    f = EMA(alpha=1.0, prior=0.0)
    for x in (0.2, 5.0, -3.0):
        assert f.update(x) == x

    # A small alpha averages heavily; a large alpha tracks the input closely.
    slow = EMA(alpha=0.05, prior=0.0)
    fast = EMA(alpha=0.5, prior=0.0)
    for _ in range(20):
        s = slow.update(1.0)
        fa = fast.update(1.0)
    assert fa > s, (fa, s)  # fast filter tracks the constant-1.0 input faster

    # reset() restores the prior exactly.
    f = EMA(alpha=0.3, prior=1.23)
    for x in (5.0, -2.0, 3.0):
        f.update(x)
    assert f._x != 1.23
    f.reset()
    assert f._x == 1.23
    print("EMA self-test passed")
