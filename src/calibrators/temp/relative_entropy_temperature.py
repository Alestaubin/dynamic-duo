"""
Per-batch single-parameter entropy minimization kernel.

Parameterization:
    T_l = exp( w / 2)
    T_s = exp(-w / 2)
    → T_l * T_s = 1  (product fixed; overall scale is not a free parameter)
    → T_l / T_s = exp(w) (the single degree of freedom)

w > 0: T_l > 1, T_s < 1 → small model logits amplified relative to large
w < 0: T_l < 1, T_s > 1 → large model logits amplified relative to small
w = 0: T_l = T_s = 1    → symmetric / equal weighting (init default)

Constraint: w ∈ (-w_max, w_max) enforced via tanh reparameterization:
    w = w_max * tanh(v),  v ∈ ℝ (unconstrained free parameter)
This makes the bound visible to the optimizer at every step, preventing
LBFGS from escaping to ±∞ and getting post-hoc clamped to the boundary.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from src.utils.logit_transforms import combine_logits
from src.tta.tent import softmax_entropy


class RelativeEntropyTemperature:
    """Per-batch entropy minimization over a single relative-weight scalar w.

    Parameters
    ----------
    init_w : float
        Initial value of w (0.0 = symmetric).
    t_max : float
        Maximum temperature; bounds w to [-2 log(t_max), 2 log(t_max)].
    reset_each_batch : bool
        If True (default), w is reset to init_w before each batch.
    device : optional
        Passed through to logit placement (unused; w is created fresh per call).
    """

    def __init__(
        self,
        init_w: float = 0.0,
        t_max: float = 10.0,
        reset_each_batch: bool = True,
        device=None,
    ) -> None:
        if t_max <= 1.0:
            raise ValueError("t_max must be > 1.0 so that the constraint is non-trivial.")
        self.init_w = init_w
        self._w_max = 2.0 * math.log(t_max)
        self.reset_each_batch = reset_each_batch
        self.device = torch.device(device) if device is not None else None

        self.w: float = init_w
        self.last_loss: float = float("nan")

    def _reset(self) -> None:
        self.w = self.init_w

    @property
    def temperature_l(self) -> float:
        return math.exp(self.w / 2.0)

    @property
    def temperature_s(self) -> float:
        return math.exp(-self.w / 2.0)

    def adapt(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> Tuple[float, float]:
        """Minimize ensemble entropy over w for one batch.

        Returns
        -------
        (tau_l, tau_s) : floats
            Fitted temperatures. Always satisfy tau_l * tau_s == 1.
        """
        if logits_l.shape != logits_s.shape or logits_l.dim() != 2:
            raise ValueError("Expected matching (B, C) logit tensors.")

        if self.reset_each_batch:
            self._reset()

        device = logits_l.device
        # Reparameterize as w = w_max * tanh(v) so v is unconstrained and
        # the bound is enforced inside every closure evaluation, not post-hoc.
        init_v = math.atanh(max(-0.9999, min(0.9999, self.w / self._w_max)))
        v = torch.tensor([init_v], device=device, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [v], lr=1.0, max_iter=100, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            w = self._w_max * torch.tanh(v)
            tau_l = (w / 2.0).exp()
            tau_s = (-w / 2.0).exp()
            z_ens = combine_logits(z_l=logits_l, z_s=logits_s, tau_l=tau_l, tau_s=tau_s)
            loss = softmax_entropy(z_ens).mean()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        self.last_loss = float(loss.detach()) if loss is not None else float("nan")

        self.w = float(self._w_max * math.tanh(float(v.detach())))
        return self.temperature_l, self.temperature_s
