"""
proxy_filter.py
================
Scalar Kalman filter for denoising any per-batch reliability proxy score, in
logit (log-odds) space.

Models the true (latent) reliability of a model as a slowly drifting random
walk, observed through a noisy per-batch proxy measurement:

    predict  x̂⁻ = x̂_{t-1}                    ;  P⁻ = P_{t-1} + q
    observe  o_t = x_t + v,  v ~ N(0, R)
    gain     K   = P⁻ / (P⁻ + R)
    update   x̂_t = x̂⁻ + K · (o_t − x̂⁻)       ;  P_t = (1 − K) · P⁻

`q` (process noise) sets how fast the latent reliability is allowed to drift
between batches — larger q tracks adaptation faster but is noisier. `r`
(observation noise) is how noisy a single-batch proxy reading is around the
true reliability. Only the ratio q/r matters for the filter's steady-state
behaviour: as observations accumulate the gain converges to a fixed value,
i.e. the filter becomes a plain EMA with a constant smoothing factor,

    K_inf = (sqrt(q^2 + 4*q*r) - q) / (2*r + sqrt(q^2 + 4*q*r) - q)

Filter in LOGIT space (unbounded), not raw [0,1] proxy/accuracy space, so the
additive-Gaussian observation model is a reasonable approximation — see
proxies.fit_val_reliability for how the prior (logit of val accuracy) is
computed.

Stateful: update() once per batch, reset() at each stream boundary (e.g. per
corruption) — the model itself resets to source at each corruption, so the
filter should return to its val-set prior there too.
"""

from __future__ import annotations

import math


class ProxyFilter:
    """Scalar Kalman filter tracking one model's latent (logit-space) reliability.

    Parameters
    ----------
    q : process noise (drift). How much the latent reliability is allowed to
        move between batches; 0 means the true reliability is assumed constant.
    r : observation noise. How noisy a single proxy reading is around the
        true reliability.
    prior_mean : prior belief about the latent reliability before any
        observations — typically logit(val_acc) for this model (see
        proxies.fit_val_reliability).
    prior_var : prior uncertainty (variance) around prior_mean.
    """

    def __init__(self, q: float, r: float, prior_mean: float, prior_var: float) -> None:
        assert q >= 0.0, f"q must be >= 0, got {q}"
        assert r > 0.0, f"r must be > 0, got {r}"
        assert prior_var >= 0.0, f"prior_var must be >= 0, got {prior_var}"
        self.q = q
        self.r = r
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self._mean = prior_mean
        self._var = prior_var

    def update(self, o: float) -> None:
        """Assimilate one new (logit-space) proxy observation."""
        pred_var = self._var + self.q
        gain = pred_var / (pred_var + self.r)
        self._mean = self._mean + gain * (o - self._mean)
        self._var = (1.0 - gain) * pred_var

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return self._var

    @property
    def steady_state_gain(self) -> float:
        """K_inf for this filter's (q, r) — the fixed-EMA smoothing factor the
        gain converges to as observations accumulate. 0 when q=0 (the gain
        keeps shrinking toward 0 as more observations are assimilated)."""
        if self.q == 0.0:
            return 0.0
        disc = math.sqrt(self.q ** 2 + 4.0 * self.q * self.r)
        return (disc - self.q) / (2.0 * self.r + disc - self.q)

    def reset(self) -> None:
        """Restore the filter to its prior (call at each corruption boundary,
        since the underlying model resets to source there too)."""
        self._mean = self.prior_mean
        self._var = self.prior_var


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    import statistics

    rng = random.Random(0)

    # q=0, uninformative prior: filter converges to the running mean of a
    # stationary noisy stream, since there's no assumed drift to track and
    # the prior's influence vanishes.
    true_mean = 2.0
    obs = [true_mean + rng.gauss(0, 1.0) for _ in range(500)]
    f = ProxyFilter(q=0.0, r=1.0, prior_mean=0.0, prior_var=1e6)
    for o in obs:
        f.update(o)
    running_mean = statistics.fmean(obs)
    assert abs(f.mean - running_mean) < 1e-3, (f.mean, running_mean)
    print("ProxyFilter q=0 running-mean self-test passed")

    # Step change: filter should track it with a lag governed by q/r.
    pre  = [0.0 + rng.gauss(0, 0.1) for _ in range(50)]
    post = [5.0 + rng.gauss(0, 0.1) for _ in range(50)]
    f = ProxyFilter(q=1e-2, r=1e-1, prior_mean=0.0, prior_var=1.0)
    for o in pre:
        f.update(o)
    mean_before = f.mean
    for o in post:
        f.update(o)
    mean_after = f.mean
    assert mean_before < 1.0, mean_before
    assert mean_after > 3.0, mean_after   # tracked most of the way to 5.0
    print(f"ProxyFilter step-change self-test passed "
          f"(before={mean_before:.3f}, after={mean_after:.3f})")

    # Higher q/r ratio tracks a step faster (less lag) than a lower ratio.
    def _mean_after_n_const_obs(q, r, n, obs_val=5.0):
        flt = ProxyFilter(q=q, r=r, prior_mean=0.0, prior_var=1.0)
        for _ in range(n):
            flt.update(obs_val)
        return flt.mean

    slow = _mean_after_n_const_obs(q=1e-4, r=1.0, n=10)
    fast = _mean_after_n_const_obs(q=1e-1, r=1.0, n=10)
    assert fast > slow, (fast, slow)
    print("ProxyFilter q/r tracking-speed self-test passed")

    # reset() restores the prior exactly.
    f = ProxyFilter(q=1e-2, r=1e-1, prior_mean=1.23, prior_var=0.5)
    for o in (3.0, -1.0, 2.0):
        f.update(o)
    assert f.mean != 1.23
    f.reset()
    assert f.mean == 1.23 and f.var == 0.5
    print("ProxyFilter reset self-test passed")

    # steady_state_gain: 0 when q=0; monotonically increases with q/r.
    assert ProxyFilter(q=0.0, r=1.0, prior_mean=0.0, prior_var=1.0).steady_state_gain == 0.0
    g_small = ProxyFilter(q=1e-3, r=1.0, prior_mean=0.0, prior_var=1.0).steady_state_gain
    g_large = ProxyFilter(q=1.0, r=1.0, prior_mean=0.0, prior_var=1.0).steady_state_gain
    assert 0.0 < g_small < g_large < 1.0
    print("ProxyFilter steady_state_gain self-test passed")
