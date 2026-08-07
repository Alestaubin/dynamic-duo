"""
joint_optimal_w_oracle.py
==========================
Cheating oracle baseline: directly solves for the scalar w_l in [0, 1] that
minimizes THIS batch's own NLL, instead of routing through
JointProxyWeighted's sigmoid(beta * (x_l - x_s)) gate. An even tighter
ceiling than sweeping beta under proxy_kind='oracle' (see
scripts/calibrate_gate_oracle.py): that sweep still constrains w_l to the
sigmoid FORM, asking "what's the best beta" rather than "what's the best
possible w_l". This module removes that constraint entirely.

Exact, not a heuristic: for fixed T_l/T_s,

    z_duo(w_l) = w_l*(z_l/T_l) + (1-w_l)*(z_s/T_s)

is AFFINE in w_l, and cross-entropy is convex in its logit argument, so
NLL(w_l) is provably convex on [0, 1]. A bounded scalar optimizer (SciPy's
Brent-based `minimize_scalar`) is therefore guaranteed to find the GLOBAL
optimum, no local-minima risk, no beta grid needed.

Cheats by construction (solves using the batch's own test labels, injected
via set_labels() the same way JointProxyWeighted's proxy_kind='oracle'
does) -- an upper-bound diagnostic only, never a deployable method. Keeps
T_l/T_s FIXED from a frozen base_ts (never re-optimized here, same
convention as JointProxyWeighted -- unidentifiable jointly with the gate),
so it isolates the gating ceiling specifically, not a re-calibrated one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from src.calibrators.base import BaseJointCalibrator, _NoOpModule
from src.calibrators.joint_fixed_TS import JointFixedTS


def combine(z_l: torch.Tensor, z_s: torch.Tensor, w_l: float, T_l: float, T_s: float) -> torch.Tensor:
    """Section 5 combine, generalized to any scalar w_l -- identical formula
    to JointProxyWeighted._combine. A free function (not a method) so
    optimal_w_nll can call it many times per batch, once per candidate
    w_l, without an instantiated calibrator."""
    w_s = 1.0 - w_l
    return w_l * (z_l / T_l) + w_s * (z_s / T_s)


def optimal_w_nll(
    z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor,
    T_l: float, T_s: float,
) -> tuple[float, float]:
    """Solve for the scalar w_l in [0, 1] minimizing THIS batch's own NLL.
    Returns (w_l*, nll* at w_l*). See module docstring for why this is
    exact rather than a heuristic. z_l/z_s/labels should be detached
    (no_grad) -- SciPy's optimizer works on plain floats, not autograd
    tensors.
    """
    def _nll(w_l: float) -> float:
        return float(F.cross_entropy(combine(z_l, z_s, w_l, T_l, T_s), labels))

    result = minimize_scalar(_nll, bounds=(0.0, 1.0), method="bounded")
    return float(result.x), float(result.fun)


class JointOptimalWOracle(BaseJointCalibrator):
    """Cheating oracle calibrator: fits w_l fresh every batch by directly
    minimizing that batch's own NLL (see module docstring), rather than
    JointProxyWeighted's proxy -> calibrate -> filter -> sigmoid-gate
    pipeline. Mirrors JointCoca's self-adapting-per-batch shape, except it
    needs labels -- injected via set_labels(), the same convention
    JointProxyWeighted's proxy_kind='oracle' uses (see DynamicDuo.forward's
    calibration_mode check).

    Like JointProxyWeighted, DynamicDuo calls calibrate_with_grad (loss
    path) then calibrate (output path) on the SAME batch every adapting
    step; _pending caches the first result so the w_l solve and its NLL
    aren't computed twice for one batch.
    """

    def __init__(self, base_ts: JointFixedTS | None = None):
        super().__init__()
        self.base_ts = base_ts
        if base_ts is not None:
            for p in base_ts.parameters():
                p.requires_grad_(False)
        self._labels: torch.Tensor | None = None
        self._pending: torch.Tensor | None = None
        self.last_w_l: float = 0.5
        self.last_nll: float = float("nan")

    def set_labels(self, labels: torch.Tensor) -> None:
        self._labels = labels

    def _temps(self) -> tuple[float, float]:
        T_l = float(self.base_ts.Tl.item()) if self.base_ts is not None else 1.0
        T_s = float(self.base_ts.Ts.item()) if self.base_ts is not None else 1.0
        return T_l, T_s

    def _solve(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> float:
        assert self._labels is not None, (
            "JointOptimalWOracle requires labels via set_labels() -- it cheats "
            "by construction and cannot be used where labels are unavailable."
        )
        T_l, T_s = self._temps()
        labels = self._labels.to(logits_l.device)
        with torch.no_grad():
            w_l, nll = optimal_w_nll(logits_l.detach(), logits_s.detach(), labels, T_l, T_s)
        self.last_w_l, self.last_nll = w_l, nll
        self._labels = None  # consume
        return w_l

    def calibrate_with_grad(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        w_l = self._solve(logits_l, logits_s)
        T_l, T_s = self._temps()
        # w_l is a fixed (detached) scalar; grad still flows through
        # logits_l/logits_s for the TENT adaptation loss, same as
        # JointProxyWeighted.calibrate_with_grad.
        z_duo = combine(logits_l, logits_s, w_l, T_l, T_s)
        self._pending = z_duo.detach()
        return z_duo

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        if self._pending is not None:
            z_duo, self._pending = self._pending, None
            return z_duo
        w_l = self._solve(logits_l, logits_s)
        T_l, T_s = self._temps()
        with torch.no_grad():
            return combine(logits_l, logits_s, w_l, T_l, T_s)

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        pass  # nothing to fit ahead of time; solves fresh every batch

    @property
    def model(self):
        return _NoOpModule()


if __name__ == "__main__":
    torch.manual_seed(0)
    K, B = 10, 32
    z_l = torch.randn(B, K) * 3
    z_s = torch.randn(B, K) * 3
    labels = torch.randint(0, K, (B,))

    w_star, nll_star = optimal_w_nll(z_l, z_s, labels, 1.0, 1.0)
    assert 0.0 <= w_star <= 1.0

    # optimal_w_nll must match a fine brute-force grid search (proof the
    # bounded optimizer actually finds the convex minimum, not a
    # plausible-looking local point).
    grid = [i / 2000 for i in range(2001)]
    grid_nlls = [float(F.cross_entropy(combine(z_l, z_s, w, 1.0, 1.0), labels)) for w in grid]
    best_grid_nll = min(grid_nlls)
    assert abs(nll_star - best_grid_nll) < 1e-4, (nll_star, best_grid_nll)

    # The optimum must be at least as good as either single-model endpoint
    # (w_l=0 or w_l=1) -- it's a strict superset of "pick one".
    assert nll_star <= grid_nlls[0] + 1e-6
    assert nll_star <= grid_nlls[-1] + 1e-6

    # BaseJointCalibrator interface round-trip: calibrate() with set_labels()
    # matches the free function directly.
    calib = JointOptimalWOracle(base_ts=None)
    calib.set_labels(labels)
    z_duo = calib.calibrate(z_l, z_s)
    expected = combine(z_l, z_s, calib.last_w_l, 1.0, 1.0)
    assert torch.allclose(z_duo, expected)

    # calibrate() without set_labels() must refuse rather than silently
    # defaulting to some weight -- it cheats by construction.
    calib2 = JointOptimalWOracle()
    try:
        calib2.calibrate(z_l, z_s)
        raise AssertionError("expected an AssertionError when labels were never set")
    except AssertionError as e:
        assert "set_labels" in str(e)

    print("joint_optimal_w_oracle self-test passed")
