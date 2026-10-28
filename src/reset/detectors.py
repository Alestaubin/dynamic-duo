"""
detectors.py
============
The drop detector of the reset rule `own_fast_vs_slow` (Phase 4). It watches one member's proxy score x_t, one value per proxy batch,
and alarms when a FAST moving average of x has fallen more than `delta` noise units below a SLOW one:

    d = FastVsSlow(alpha_fast=0.05, alpha_slow=0.005, delta=0.7, W0=20)
    alarm = d.update(x)           # True: x has dropped (one-sided: a rise never alarms)
    d.restart()                   # forget everything, start a new warm-up
    state = d.state_dict(); d.load_state_dict(state)      # plain floats / lists: JSON-serializable

  * **Warm-up**: no alarm for the first `W0` updates after construction or `restart()`. The warm-up window fixes the reference (its mean,
    where both averages start) and the initial running standard deviation.
  * **Alarm**: (slow_t - fast_t) / scale > delta, with fast_t = (1 - alpha_fast) fast_{t-1} + alpha_fast x_t and slow_t likewise with
    alpha_slow < alpha_fast. A sudden drop moves the fast average first; a slow decay moves the slow one later and the gap opens.
  * **Standardization** (`standardize=True`, the default): `scale` is a running standard deviation of x, an exponentially weighted variance
    with rate `alpha_var` (West's update, started from the warm-up variance), so `delta` is in noise units. The scale used to judge x_t is
    the one BEFORE x_t enters the variance, so a collapse does not inflate its own yardstick before it has been judged. With
    `standardize=False` the scale is 1 and `delta` is in the units of x. The scale is floored at `sigma_floor`.
  * **Statistic**: after every post-warm-up update `d.statistic` is the quantity compared with `delta` (NaN during the warm-up); the
    controller logs it.

The other procedures that were screened against this one (level, imm_vs_ema, slope, cusum) were dropped, with their code: it is in git at
the tag `pre-reset-cleanup`.
"""

from __future__ import annotations

import math

import numpy as np

SIGMA_FLOOR = 1e-6


class FastVsSlow:
    kind = "fast_vs_slow"

    def __init__(self, alpha_fast: float, alpha_slow: float, delta: float, *, W0: int = 20, standardize: bool = True, alpha_var: float = 0.01,
                 sigma_floor: float = SIGMA_FLOOR):
        if not 0 < alpha_slow < alpha_fast < 1:
            raise ValueError(f"need 0 < alpha_slow < alpha_fast < 1, got {alpha_slow}, {alpha_fast}")
        if W0 < 2:
            raise ValueError(f"W0 must be at least 2 (the warm-up fixes a mean and a variance), got {W0}")
        if not 0 < alpha_var < 1:
            raise ValueError(f"alpha_var must be in (0, 1), got {alpha_var}")
        self.alpha_fast, self.alpha_slow, self.delta = float(alpha_fast), float(alpha_slow), float(delta)
        self.W0, self.standardize, self.alpha_var, self.sigma_floor = int(W0), bool(standardize), float(alpha_var), float(sigma_floor)
        self.restart()

    def restart(self) -> None:
        self.n = 0                           # updates since the (re)start
        self._warm: list[float] = []
        self.mean0 = float("nan")            # the warm-up mean: where both averages start
        self._m = self._v = float("nan")     # running mean and variance for the standardizer
        self.statistic = float("nan")
        self.fast = self.slow = float("nan")

    @property
    def warming_up(self) -> bool:
        return self.n < self.W0

    @property
    def scale(self) -> float:
        return max(math.sqrt(self._v), self.sigma_floor) if self.standardize else 1.0

    def update(self, x: float) -> bool:
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"{self.kind}: non-finite input {x!r}")
        self.n += 1
        if self.n <= self.W0:
            self._warm.append(x)
            if self.n == self.W0:
                w = np.asarray(self._warm)
                self.mean0, self._m, self._v = float(w.mean()), float(w.mean()), float(w.var(ddof=1))
                self._warm = []
                self.fast = self.slow = self.mean0
            self.statistic = float("nan")
            return False
        scale = self.scale
        self.fast = (1.0 - self.alpha_fast) * self.fast + self.alpha_fast * x
        self.slow = (1.0 - self.alpha_slow) * self.slow + self.alpha_slow * x
        self.statistic = (self.slow - self.fast) / scale
        alarm = bool(self.statistic > self.delta)
        diff = x - self._m                                  # West's exponentially weighted variance, AFTER x was judged
        incr = self.alpha_var * diff
        self._m += incr
        self._v = (1.0 - self.alpha_var) * (self._v + diff * incr)
        return alarm

    # ---- state ------------------------------------------------------------------------------------------------------
    def params(self) -> dict:
        return {"alpha_fast": self.alpha_fast, "alpha_slow": self.alpha_slow, "delta": self.delta, "W0": self.W0, "standardize": self.standardize,
                "alpha_var": self.alpha_var}

    def state_dict(self) -> dict:
        return {"kind": self.kind, "n": self.n, "warm": list(self._warm), "mean0": self.mean0, "m": self._m, "v": self._v,
                "statistic": self.statistic, "fast": self.fast, "slow": self.slow}

    def load_state_dict(self, s: dict) -> None:
        if s["kind"] != self.kind:
            raise ValueError(f"state of a {s['kind']!r} detector loaded into a {self.kind!r} one")
        self.n, self._warm, self.mean0, self._m, self._v, self.statistic = s["n"], list(s["warm"]), s["mean0"], s["m"], s["v"], s["statistic"]
        self.fast, self.slow = s["fast"], s["slow"]

    def __repr__(self) -> str:
        return f"FastVsSlow({', '.join(f'{k}={v}' for k, v in self.params().items())})"


def make_detector(kind: str, **params) -> FastVsSlow:
    if kind != FastVsSlow.kind:
        raise ValueError(f"unknown detector {kind!r}: only {FastVsSlow.kind!r} is kept (the other procedures are in git at the tag pre-reset-cleanup)")
    return FastVsSlow(**params)


# ---------------------------------------------------------------------------------------------------- self-test

def _alarm_times(d: FastVsSlow, xs, restart_on_alarm: bool = False) -> list[int]:
    out = []
    for t, x in enumerate(xs):
        if d.update(x):
            out.append(t)
            if restart_on_alarm:
                d.restart()
    return out


if __name__ == "__main__":
    import json

    rng = np.random.default_rng(0)
    FS = dict(alpha_fast=0.2, alpha_slow=0.02, delta=2.0)

    # ---- warm-up, references, refusals ---------------------------------------------------------------------------------
    d = FastVsSlow(alpha_fast=0.1, alpha_slow=0.01, delta=2.0, W0=10)
    assert all(d.update(v) is False for v in rng.normal(3.0, 2.0, 10)) and not d.warming_up
    assert abs(d.mean0 - 3.0) < 2.0 and d.scale > 0.5, "the warm-up fixes the mean and the scale"
    assert math.isnan(d.statistic) and d.fast == d.slow == d.mean0, "both averages start at the warm-up mean"
    d.restart()
    assert d.warming_up and d.n == 0 and math.isnan(d.mean0) and math.isnan(d.fast)
    for bad in (dict(W0=1), dict(alpha_var=0.0), dict(alpha_var=1.0), dict(alpha_fast=0.01, alpha_slow=0.1), dict(alpha_fast=1.0), dict(alpha_slow=0.0)):
        try:
            FastVsSlow(**{**dict(alpha_fast=0.1, alpha_slow=0.01, delta=1.0), **bad})
            raise AssertionError(f"{bad} must be refused")
        except ValueError:
            pass
    try:
        FastVsSlow(**FS, W0=5).update(float("nan"))
        raise AssertionError("a non-finite input must be refused")
    except ValueError:
        pass
    try:
        make_detector("cusum", nu=0.5, h=8.0)
        raise AssertionError("a dropped procedure must be refused")
    except ValueError:
        pass

    # ---- standardization: the same drop in noise units is the same alarm at any scale of x; standardize=False reads delta in x's units ----
    for scale in (0.01, 1.0, 100.0):
        base = rng.normal(0.0, 1.0, 400)
        x = np.concatenate([base[:200], base[200:] - 6.0]) * scale + 7 * scale
        t = _alarm_times(FastVsSlow(**FS, W0=50), x)
        assert t and 200 <= t[0] <= 230, (scale, t[:3])
    raw = np.concatenate([np.full(30, 5.0) + rng.normal(0, 0.01, 30), np.full(30, 3.0)])
    assert _alarm_times(FastVsSlow(alpha_fast=0.5, alpha_slow=0.05, delta=0.5, standardize=False, W0=10), raw)[0] >= 30

    # ---- 1. pure noise: the false-alarm rate falls as delta rises (no restart: exactly monotone) ----------------------------
    noise = rng.normal(0.0, 1.0, 20000)
    values = [0.2, 0.3, 0.4, 0.5]
    counts = [len(_alarm_times(FastVsSlow(alpha_fast=0.1, alpha_slow=0.01, delta=v, W0=50), noise)) for v in values]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1], counts
    print(f"  noise: alarm steps per 20,000 at delta = {values}: {counts}")
    rates = [len(_alarm_times(FastVsSlow(alpha_fast=0.1, alpha_slow=0.01, delta=v, W0=50), noise, True)) for v in (0.2, 0.3, 0.4)]
    assert rates[0] > rates[1] >= rates[2], rates                      # with restart-on-alarm (how a controller uses it) the rate still falls

    # ---- 2. sudden step drop: it alarms soon after it; a rise never alarms --------------------------------------------------
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(-6, 1, 100)])
    t = _alarm_times(FastVsSlow(**FS, W0=50), x)
    assert t and 300 <= t[0] <= 330, t[:3]
    up = np.concatenate([rng.normal(0, 1, 300), rng.normal(+6, 1, 100)])
    assert not _alarm_times(FastVsSlow(**FS, W0=50), up)

    # ---- 3. slow ramp down: the slow average lags the fast one and the gap opens (the setting the reset runs use, W0 20) ---------------
    # (the running scale grows with a ramp, so a large delta in noise units can miss it: delta 2 with speeds (0.1, 0.01) never alarms here)
    t0 = 300
    for rate, within in ((0.01, 250), (0.03, 120), (0.1, 60)):         # noise units per batch
        for seed in range(3):
            ramp = np.random.default_rng(seed).normal(0, 1, 1000) - rate * np.maximum(0, np.arange(1000) - t0)
            t_fs = _alarm_times(FastVsSlow(alpha_fast=0.05, alpha_slow=0.005, delta=0.7, W0=20), ramp)
            assert t_fs and t0 <= t_fs[0] <= t0 + within and not [i for i in t_fs if i < t0], (rate, seed, t_fs[:3])
        print(f"  ramp of {rate} sigma/batch from batch {t0}: alarms at +{t_fs[0] - t0} (seed 2)")

    # ---- state: JSON round trip reproduces the alarms exactly ------------------------------------------------------------
    a, b = FastVsSlow(alpha_fast=0.2, alpha_slow=0.02, delta=0.5, W0=20), FastVsSlow(alpha_fast=0.2, alpha_slow=0.02, delta=0.5, W0=20)
    xs = rng.normal(0, 1, 300) - 0.01 * np.arange(300)
    for x in xs[:137]:
        a.update(x)
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    assert [a.update(x) for x in xs[137:]] == [b.update(x) for x in xs[137:]]
    assert json.dumps(a.state_dict()) == json.dumps(b.state_dict())
    try:
        b.load_state_dict({**a.state_dict(), "kind": "cusum"})
        raise AssertionError("state of another detector must be refused")
    except ValueError:
        pass
    print("detectors self-test passed")
