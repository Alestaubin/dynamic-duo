"""
detectors.py
============
Streaming drop detectors for the reset rules (Phase 4). One instance watches one quantity x_t of one member (its own
proxy score `own`, or its gap to the partner `gap`; see controller.py), one value per proxy batch:

    d = Cusum(nu=0.5, h=8.0, W0=20)
    alarm = d.update(x)           # True: x has dropped (a detector is one-sided and never fires on a rise)
    d.restart()                   # forget everything, start a new warm-up
    state = d.state_dict(); d.load_state_dict(state)      # plain floats / lists: JSON-serializable

Common behaviour
  * **Warm-up**: no alarm for the first `W0` updates after construction or `restart()`; the warm-up window also fixes the
    references (its mean, and the initial running standard deviation).
  * **Standardization** (`standardize=True`, the default): deviations are divided by a running standard deviation of x, an
    exponentially weighted variance with rate `alpha_var` (West's update, started from the warm-up variance), so thresholds are
    in noise units. The scale used to judge x_t is the one BEFORE x_t enters the variance, so a collapse does not inflate its
    own yardstick before it has been judged. With `standardize=False` the scale is 1 and the thresholds are in the units of x.
    The scale is floored at `sigma_floor` (a constant sequence has no noise to measure).
  * **Statistic**: after every post-warm-up update `d.statistic` is the quantity that was compared with the threshold (NaN
    during the warm-up), which the controller logs.

The five procedures (alarm when ...), with the standardized deviation z = (reference - x) / scale:

  level         (x_t - m0) / scale < tau          m0 = the warm-up mean; with standardize=False: x_t < tau
  imm_vs_ema    xbar_{t-1} - x_t > delta * scale  xbar_t = (1 - alpha) xbar_{t-1} + alpha x_t, started at m0. A slow ramp drags the
                                                  average along (gap = ramp rate / alpha), so this procedure does not see ramps.
  fast_vs_slow  xbar_slow_t - xbar_fast_t > delta * scale          both averages started at m0
  slope         the least-squares slope of the last W values / scale < -gamma for `k` consecutive batches (no alarm before W
                values exist: the warm-up is then effectively max(W0, W))
  cusum         S_t = max(0, S_{t-1} + (mu - x_t) / scale - nu) > h. `mu`: "warmup" (the warm-up mean, FROZEN until the next
                restart: an adaptive average would absorb slow collapses) for `own`, or a number (0 for `gap`).

After an alarm a detector keeps its state (CUSUM stays above h until restarted): the controller restarts it.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable

import numpy as np

SIGMA_FLOOR = 1e-6


class Detector:
    kind = ""

    def __init__(self, *, W0: int = 20, standardize: bool = True, alpha_var: float = 0.01, sigma_floor: float = SIGMA_FLOOR):
        if W0 < 2:
            raise ValueError(f"W0 must be at least 2 (the warm-up fixes a mean and a variance), got {W0}")
        if not 0 < alpha_var < 1:
            raise ValueError(f"alpha_var must be in (0, 1), got {alpha_var}")
        self.W0, self.standardize, self.alpha_var, self.sigma_floor = int(W0), bool(standardize), float(alpha_var), float(sigma_floor)
        self.restart()

    # ---- the common machinery -------------------------------------------------------------------------------------
    def restart(self) -> None:
        self.n = 0                       # updates since the (re)start
        self._warm: list[float] = []
        self.mean0 = float("nan")        # the warm-up mean: the reference of level / imm_vs_ema / fast_vs_slow / cusum("warmup")
        self._m = self._v = float("nan")     # running mean and variance for the standardizer
        self.statistic = float("nan")
        self._restart()

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
            self._observe_warmup(x)
            if self.n == self.W0:
                w = np.asarray(self._warm)
                self.mean0, self._m, self._v = float(w.mean()), float(w.mean()), float(w.var(ddof=1))
                self._warm = []
                self._init_after_warmup()
            self.statistic = float("nan")
            return False
        scale = self.scale
        alarm = bool(self._step(x, scale))
        diff = x - self._m                                  # West's exponentially weighted variance, AFTER x was judged
        incr = self.alpha_var * diff
        self._m += incr
        self._v = (1.0 - self.alpha_var) * (self._v + diff * incr)
        return alarm

    # ---- what a procedure supplies ---------------------------------------------------------------------------------
    def _restart(self) -> None: ...
    def _observe_warmup(self, x: float) -> None: ...
    def _init_after_warmup(self) -> None: ...

    def _step(self, x: float, scale: float) -> bool:
        raise NotImplementedError

    def _extra_state(self) -> dict:
        return {}

    def _load_extra(self, s: dict) -> None: ...

    # ---- state ------------------------------------------------------------------------------------------------------
    def params(self) -> dict:
        return {"W0": self.W0, "standardize": self.standardize, "alpha_var": self.alpha_var}

    def state_dict(self) -> dict:
        return {"kind": self.kind, "n": self.n, "warm": list(self._warm), "mean0": self.mean0, "m": self._m, "v": self._v,
                "statistic": self.statistic, **self._extra_state()}

    def load_state_dict(self, s: dict) -> None:
        if s["kind"] != self.kind:
            raise ValueError(f"state of a {s['kind']!r} detector loaded into a {self.kind!r} one")
        self.n, self._warm, self.mean0, self._m, self._v, self.statistic = s["n"], list(s["warm"]), s["mean0"], s["m"], s["v"], s["statistic"]
        self._load_extra(s)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v}' for k, v in self.params().items())})"


class Level(Detector):
    kind = "level"

    def __init__(self, tau: float, **kw):
        self.tau = float(tau)
        super().__init__(**kw)

    def _step(self, x, scale):
        self.statistic = (x - self.mean0) / scale if self.standardize else x
        return self.statistic < self.tau

    def params(self):
        return {"tau": self.tau, **super().params()}


class ImmVsEma(Detector):
    kind = "imm_vs_ema"

    def __init__(self, alpha: float, delta: float, **kw):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha, self.delta = float(alpha), float(delta)
        super().__init__(**kw)

    def _restart(self):
        self.ema = float("nan")

    def _init_after_warmup(self):
        self.ema = self.mean0

    def _step(self, x, scale):
        self.statistic = (self.ema - x) / scale               # the drop below the average BEFORE x joins it
        alarm = self.statistic > self.delta
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * x
        return alarm

    def _extra_state(self):
        return {"ema": self.ema}

    def _load_extra(self, s):
        self.ema = s["ema"]

    def params(self):
        return {"alpha": self.alpha, "delta": self.delta, **super().params()}


class FastVsSlow(Detector):
    kind = "fast_vs_slow"

    def __init__(self, alpha_fast: float, alpha_slow: float, delta: float, **kw):
        if not 0 < alpha_slow < alpha_fast < 1:
            raise ValueError(f"need 0 < alpha_slow < alpha_fast < 1, got {alpha_slow}, {alpha_fast}")
        self.alpha_fast, self.alpha_slow, self.delta = float(alpha_fast), float(alpha_slow), float(delta)
        super().__init__(**kw)

    def _restart(self):
        self.fast = self.slow = float("nan")

    def _init_after_warmup(self):
        self.fast = self.slow = self.mean0

    def _step(self, x, scale):
        self.fast = (1.0 - self.alpha_fast) * self.fast + self.alpha_fast * x
        self.slow = (1.0 - self.alpha_slow) * self.slow + self.alpha_slow * x
        self.statistic = (self.slow - self.fast) / scale
        return self.statistic > self.delta

    def _extra_state(self):
        return {"fast": self.fast, "slow": self.slow}

    def _load_extra(self, s):
        self.fast, self.slow = s["fast"], s["slow"]

    def params(self):
        return {"alpha_fast": self.alpha_fast, "alpha_slow": self.alpha_slow, "delta": self.delta, **super().params()}


class Slope(Detector):
    kind = "slope"

    def __init__(self, W: int, gamma: float, k: int = 1, **kw):
        if W < 3 or k < 1:
            raise ValueError(f"need W >= 3 and k >= 1, got W={W}, k={k}")
        self.W, self.gamma, self.k = int(W), float(gamma), int(k)
        c = np.arange(self.W, dtype=np.float64)
        self._c = c - c.mean()                               # centred time index: slope = sum(c x) / sum(c^2)
        self._c2 = float((self._c ** 2).sum())
        super().__init__(**kw)

    def _restart(self):
        self._win: deque[float] = deque(maxlen=self.W)
        self.run = 0                                         # consecutive batches below -gamma

    def _observe_warmup(self, x):
        self._win.append(x)

    def _step(self, x, scale):
        self._win.append(x)
        if len(self._win) < self.W:
            self.statistic, self.run = float("nan"), 0
            return False
        self.statistic = float(np.dot(self._c, np.fromiter(self._win, dtype=np.float64, count=self.W)) / self._c2) / scale
        self.run = self.run + 1 if self.statistic < -self.gamma else 0
        return self.run >= self.k

    def _extra_state(self):
        return {"win": list(self._win), "run": self.run}

    def _load_extra(self, s):
        self._win = deque(s["win"], maxlen=self.W)
        self.run = s["run"]

    def params(self):
        return {"W": self.W, "gamma": self.gamma, "k": self.k, **super().params()}


class Cusum(Detector):
    kind = "cusum"

    def __init__(self, nu: float, h: float, mu: float | str = "warmup", **kw):
        if not (mu == "warmup" or isinstance(mu, (int, float))):
            raise ValueError(f"mu must be 'warmup' or a number, got {mu!r}")
        self.nu, self.h, self.mu = float(nu), float(h), mu
        super().__init__(**kw)

    def _restart(self):
        self.S = 0.0

    @property
    def reference(self) -> float:
        return self.mean0 if self.mu == "warmup" else float(self.mu)

    def _step(self, x, scale):
        self.S = max(0.0, self.S + (self.reference - x) / scale - self.nu)
        self.statistic = self.S
        return self.S > self.h

    def _extra_state(self):
        return {"S": self.S}

    def _load_extra(self, s):
        self.S = s["S"]

    def params(self):
        return {"nu": self.nu, "h": self.h, "mu": self.mu, **super().params()}


DETECTORS: dict[str, Callable[..., Detector]] = {c.kind: c for c in (Level, ImmVsEma, FastVsSlow, Slope, Cusum)}
THRESHOLD = {"level": "tau", "imm_vs_ema": "delta", "fast_vs_slow": "delta", "slope": "gamma", "cusum": "h"}   # the swept parameter


def make_detector(kind: str, **params) -> Detector:
    if kind not in DETECTORS:
        raise ValueError(f"unknown detector {kind!r}: one of {sorted(DETECTORS)}")
    return DETECTORS[kind](**params)


# ---------------------------------------------------------------------------------------------------- self-test

def _alarm_times(d: Detector, xs, restart_on_alarm: bool = False) -> list[int]:
    out = []
    for t, x in enumerate(xs):
        if d.update(x):
            out.append(t)
            if restart_on_alarm:
                d.restart()
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- warm-up, references, standardization ------------------------------------------------------------------------
    d = Cusum(nu=0.5, h=5.0, W0=10)
    assert all(d.update(v) is False for v in rng.normal(3.0, 2.0, 10)) and not d.warming_up
    assert abs(d.mean0 - 3.0) < 2.0 and d.scale > 0.5, "the warm-up fixes the mean and the scale"
    assert math.isnan(d.statistic)
    frozen = d.mean0
    for v in rng.normal(3.0, 2.0, 200):
        d.update(v)
    assert d.mean0 == frozen, "the CUSUM reference is frozen until the next restart"
    d.restart()
    assert d.warming_up and d.n == 0 and d.S == 0.0 and math.isnan(d.mean0)
    for bad in (dict(W0=1), dict(alpha_var=0.0), dict(alpha_var=1.0)):
        try:
            Cusum(nu=0.5, h=5.0, **bad)
            raise AssertionError(f"{bad} must be refused")
        except ValueError:
            pass
    try:
        Level(tau=-3.0, W0=5).update(float("nan"))
        raise AssertionError("a non-finite input must be refused")
    except ValueError:
        pass
    # standardization: the same drop in noise units is the same alarm at any scale of x
    for scale in (0.01, 1.0, 100.0):
        base = rng.normal(0.0, 1.0, 400)
        x = np.concatenate([base[:200], base[200:] - 6.0]) * scale + 7 * scale
        t = _alarm_times(Cusum(nu=0.5, h=8.0, W0=50), x)
        assert t and 200 <= t[0] <= 210, (scale, t[:3])
    # ... and standardize=False reads thresholds in the units of x
    raw = np.concatenate([np.full(30, 5.0) + rng.normal(0, 0.01, 30), np.full(30, 3.0)])
    assert _alarm_times(Level(tau=4.0, standardize=False, W0=10), raw)[0] == 30

    # ---- 1. pure noise: the false-alarm rate falls as the threshold rises (no restart: exactly monotone) ----------------
    noise = rng.normal(0.0, 1.0, 20000)
    grids = {"level": ("tau", [-1.0, -2.0, -3.0, -4.0], {}), "imm_vs_ema": ("delta", [1.5, 2.5, 3.5, 4.5], {"alpha": 0.05}),
             "fast_vs_slow": ("delta", [0.5, 1.0, 1.5, 2.0], {"alpha_fast": 0.1, "alpha_slow": 0.01}),
             "slope": ("gamma", [0.005, 0.01, 0.02, 0.03], {"W": 32, "k": 2}), "cusum": ("h", [2.0, 4.0, 8.0, 12.0], {"nu": 0.5})}
    for kind, (thr, values, fixed) in grids.items():
        counts = [len(_alarm_times(make_detector(kind, **{thr: v}, **fixed, W0=50), noise)) for v in values]
        assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1], (kind, counts)
        print(f"  noise, {kind:12s} alarm steps per 20,000 at {thr} = {values}: {counts}")
    # with restart-on-alarm (how a controller uses them) the rate still falls
    rates = [len(_alarm_times(Cusum(nu=0.5, h=h, W0=50), noise, True)) for h in (2.0, 4.0, 8.0)]
    assert rates[0] > rates[1] >= rates[2], rates

    # ---- 2. sudden step drop: cusum detects within the expected delay -------------------------------------------------
    for depth in (3.0, 5.0, 8.0):
        x = np.concatenate([rng.normal(0, 1, 300), rng.normal(-depth, 1, 100)])
        t = _alarm_times(Cusum(nu=0.5, h=8.0, W0=50), x)
        expected = 8.0 / (depth - 0.5)                      # h / (drift per batch in noise units, minus the allowance nu)
        assert t and t[0] >= 300 and t[0] - 300 <= 2 * expected + 3, (depth, t[:3], expected)
        print(f"  step drop of {depth:.0f} sigma: cusum alarms {t[0] - 300} batches after it (h / (depth - nu) = {expected:.1f})")
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(-6, 1, 100)])
    for kind, params in (("level", dict(tau=-4.0)), ("imm_vs_ema", dict(alpha=0.05, delta=4.0)),
                         ("fast_vs_slow", dict(alpha_fast=0.2, alpha_slow=0.02, delta=2.0)), ("slope", dict(W=16, gamma=0.3, k=1))):
        t = _alarm_times(make_detector(kind, **params, W0=50), x)
        assert t and 300 <= t[0] <= 330 and t[0] >= 300, (kind, t[:3])
    # a rise is never an alarm
    up = np.concatenate([rng.normal(0, 1, 300), rng.normal(+6, 1, 100)])
    for kind, params in (("level", dict(tau=-4.0)), ("imm_vs_ema", dict(alpha=0.05, delta=4.0)), ("cusum", dict(nu=0.5, h=8.0)),
                         ("fast_vs_slow", dict(alpha_fast=0.2, alpha_slow=0.02, delta=2.0)), ("slope", dict(W=16, gamma=0.3, k=1))):
        assert not _alarm_times(make_detector(kind, **params, W0=50), up), kind

    # ---- 3. slow ramp down: slope and cusum catch it; imm_vs_ema does not ---------------------------------------------
    rate, t0 = 0.03, 300                                          # noise units per batch: 21 sigma over 700 batches
    ramp = rng.normal(0, 1, 1000) - rate * np.maximum(0, np.arange(1000) - t0)
    t_slope = _alarm_times(Slope(W=64, gamma=0.015, k=4, W0=50), ramp)
    t_cusum = _alarm_times(Cusum(nu=0.5, h=8.0, W0=50), ramp)
    t_imm = _alarm_times(ImmVsEma(alpha=0.05, delta=4.5, W0=50), ramp)
    t_fs = _alarm_times(FastVsSlow(alpha_fast=0.1, alpha_slow=0.01, delta=2.0, W0=50), ramp)
    assert t_slope and t0 <= t_slope[0] <= t0 + 150, t_slope[:3]
    assert t_cusum and t0 <= t_cusum[0] <= t0 + 150, t_cusum[:3]
    assert not t_imm, f"imm_vs_ema follows the ramp with its average (lag = rate / alpha = {rate / 0.05:.1f} sigma) and misses it: {t_imm[:3]}"
    # the pre-ramp part is quiet for every detector at these thresholds (so the alarms above are the ramp's)
    for name, t in (("slope", t_slope), ("cusum", t_cusum), ("fast_vs_slow", t_fs)):
        assert not [i for i in t if i < t0], (name, t[:3])
    print(f"  ramp of {rate} sigma/batch from batch {t0}: slope alarms at +{t_slope[0] - t0}, cusum at +{t_cusum[0] - t0}, "
          f"fast_vs_slow at {'+%d' % (t_fs[0] - t0) if t_fs else 'never'}, imm_vs_ema at {'never' if not t_imm else t_imm[0]}")

    # ---- the slope needs W values, k consecutive batches -----------------------------------------------------------------
    s = Slope(W=8, gamma=0.5, k=3, W0=4, standardize=False)
    line = -np.arange(40, dtype=float)
    ts = _alarm_times(s, line)
    assert ts[0] == 7 + 2, f"first full window at index 7, then k = 3 consecutive: {ts[:3]}"
    assert abs(s.statistic - (-1.0)) < 1e-9, "the least-squares slope of a line of slope -1 is -1"

    # ---- state: JSON round trip reproduces the alarms exactly --------------------------------------------------------------
    import json

    for kind, params in (("level", dict(tau=-1.0)), ("imm_vs_ema", dict(alpha=0.1, delta=1.0)), ("cusum", dict(nu=0.25, h=3.0)),
                         ("fast_vs_slow", dict(alpha_fast=0.2, alpha_slow=0.02, delta=0.5)), ("slope", dict(W=16, gamma=0.01, k=2))):
        a, b = make_detector(kind, **params, W0=20), make_detector(kind, **params, W0=20)
        xs = rng.normal(0, 1, 300) - 0.01 * np.arange(300)
        for x in xs[:137]:
            a.update(x)
        b.load_state_dict(json.loads(json.dumps(a.state_dict())))
        assert [a.update(x) for x in xs[137:]] == [b.update(x) for x in xs[137:]], kind
        assert a.state_dict() == b.state_dict(), kind
    try:
        Cusum(nu=0.5, h=1.0).load_state_dict(Level(tau=0.0).state_dict())
        raise AssertionError("state of another detector must be refused")
    except ValueError:
        pass
    print("detectors self-test passed")
