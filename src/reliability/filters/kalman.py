"""
kalman.py
=========
Scalar Kalman filter (eq. 12-14): the EMA fixes its gain in advance; this
derives it from an explicit noise model. The latent reliability follows a
random walk with process noise `q`, observed through a noisy per-batch proxy
measurement with observation noise `r`:

    predict:  x_hat^- = x_hat_{t-1},          v^- = v_{t-1} + q
    gain:     kappa_t = v^- / (v^- + r)
    update:   x_hat_t = x_hat^- + kappa_t * (o_t - x_hat^-)
              v_t     = (1 - kappa_t) * v^-

Only the ratio q/r matters in steady state: as observations accumulate, the
gain converges to a fixed value K_inf, at which point the filter is exactly
an EMA with a constant smoothing factor (see `steady_state_gain`).

Resets to (prior_mean, prior_var) at each corruption boundary, matching the
model's own TENT reset there.
"""

from __future__ import annotations

import math

from src.reliability.filters.base import ScoreFilter


class Kalman(ScoreFilter):
    def __init__(self, q: float, r: float, prior_mean: float, prior_var: float):
        assert q >= 0.0, f"q must be >= 0, got {q}"
        assert r > 0.0, f"r must be > 0, got {r}"
        assert prior_var >= 0.0, f"prior_var must be >= 0, got {prior_var}"
        self.q = q
        self.r = r
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self._mean = prior_mean
        self._var = prior_var

    def update(self, x: float) -> float:
        pred_var = self._var + self.q
        gain = pred_var / (pred_var + self.r)
        self._mean = self._mean + gain * (x - self._mean)
        self._var = (1.0 - gain) * pred_var
        return self._mean

    def reset(self) -> None:
        self._mean = self.prior_mean
        self._var = self.prior_var

    @property
    def var(self) -> float:
        return self._var

    @property
    def steady_state_gain(self) -> float:
        """K_inf: the fixed-EMA smoothing factor the gain converges to as
        observations accumulate. 0 when q=0 (gain keeps shrinking toward 0)."""
        if self.q == 0.0:
            return 0.0
        disc = math.sqrt(self.q ** 2 + 4.0 * self.q * self.r)
        return (disc - self.q) / (2.0 * self.r + disc - self.q)


if __name__ == "__main__":
    import random
    import statistics

    rng = random.Random(0)

    # q=0, uninformative prior: converges to the running mean of a stationary
    # noisy stream (no assumed drift, prior's influence vanishes).
    obs = [2.0 + rng.gauss(0, 1.0) for _ in range(500)]
    f = Kalman(q=0.0, r=1.0, prior_mean=0.0, prior_var=1e6)
    m = None
    for o in obs:
        m = f.update(o)
    assert abs(m - statistics.fmean(obs)) < 1e-3
    print("Kalman q=0 running-mean self-test passed")

    # Step change: the filter tracks it with a lag governed by q/r.
    pre = [0.0 + rng.gauss(0, 0.1) for _ in range(50)]
    post = [5.0 + rng.gauss(0, 0.1) for _ in range(50)]
    f = Kalman(q=1e-2, r=1e-1, prior_mean=0.0, prior_var=1.0)
    for o in pre:
        mean_before = f.update(o)
    for o in post:
        mean_after = f.update(o)
    assert mean_before < 1.0 and mean_after > 3.0
    print(f"Kalman step-change self-test passed (before={mean_before:.3f}, after={mean_after:.3f})")

    # Higher q/r tracks a step faster (less lag).
    def _mean_after_n(q, r, n, val=5.0):
        flt = Kalman(q=q, r=r, prior_mean=0.0, prior_var=1.0)
        m = None
        for _ in range(n):
            m = flt.update(val)
        return m

    slow = _mean_after_n(q=1e-4, r=1.0, n=10)
    fast = _mean_after_n(q=1e-1, r=1.0, n=10)
    assert fast > slow
    print("Kalman q/r tracking-speed self-test passed")

    # reset() restores the prior exactly.
    f = Kalman(q=1e-2, r=1e-1, prior_mean=1.23, prior_var=0.5)
    for o in (3.0, -1.0, 2.0):
        f.update(o)
    f.reset()
    assert f._mean == 1.23 and f._var == 0.5
    print("Kalman reset self-test passed")

    # steady_state_gain: 0 at q=0; monotone increasing in q/r.
    assert Kalman(q=0.0, r=1.0, prior_mean=0.0, prior_var=1.0).steady_state_gain == 0.0
    g_small = Kalman(q=1e-3, r=1.0, prior_mean=0.0, prior_var=1.0).steady_state_gain
    g_large = Kalman(q=1.0, r=1.0, prior_mean=0.0, prior_var=1.0).steady_state_gain
    assert 0.0 < g_small < g_large < 1.0
    print("Kalman steady_state_gain self-test passed")
