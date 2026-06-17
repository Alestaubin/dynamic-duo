"""
Per-batch single-parameter entropy minimization kernel.

Parameterization:
    z = λ · z_l + (1 − λ) · z_s,   λ ∈ (lambda_min, lambda_max)

λ is reparameterized via the sigmoid so the optimizer sees an unconstrained
scalar:
    λ = sigmoid(ρ),   ρ ∈ ℝ

Boundary enforcement: ρ is clamped to [logit(lambda_min), logit(lambda_max)]
after the L-BFGS step.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from src.tta.tent import softmax_entropy


class LambdaEntropyTemperature:
    """Per-batch entropy minimization over a single mixture weight λ.

    Parameters
    ----------
    init_lambda : float
        Initial λ ∈ (0, 1). Defaults to 0.5 (equal mixture).
    lambda_min, lambda_max : float
        Inclusive bounds for λ; must satisfy 0 ≤ lambda_min < lambda_max ≤ 1.
    max_iter : int
        Maximum L-BFGS iterations per batch.
    reset_each_batch : bool
        If True (default), ρ is reset to logit(init_lambda) before each batch.
        Set to False for a warm-started ρ across batches.
    """

    def __init__(
        self,
        init_lambda: float = 0.5,
        lambda_min: float = 0.0,
        lambda_max: float = 1.0,
        max_iter: int = 50,
        reset_each_batch: bool = True,
    ) -> None:
        if not (0.0 < init_lambda < 1.0):
            raise ValueError("init_lambda must be strictly in (0, 1)")
        if not (0.0 <= lambda_min < lambda_max <= 1.0):
            raise ValueError("Need 0 <= lambda_min < lambda_max <= 1")

        self.init_lambda = init_lambda
        self.max_iter = max_iter
        self.reset_each_batch = reset_each_batch

        self._logit_min = math.log(lambda_min / (1.0 - lambda_min)) if lambda_min > 0.0 else -math.inf
        self._logit_max = math.log(lambda_max / (1.0 - lambda_max)) if lambda_max < 1.0 else math.inf
        self._init_rho = math.log(init_lambda / (1.0 - init_lambda))
        self._rho: float = self._init_rho

        self.last_lambda: float = init_lambda
        self.last_loss: float = float("nan")

    def _reset(self) -> None:
        self._rho = self._init_rho

    def adapt(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> float:
        """Minimize ensemble entropy over λ for one batch.

        Parameters
        ----------
        logits_l : (B, C) tensor — large model logits.
        logits_s : (B, C) tensor — small model logits.

        Returns
        -------
        lambda_ : float
            Fitted mixture weight (λ closer to 1 → more weight on large model).
        """
        if logits_l.shape != logits_s.shape or logits_l.dim() != 2:
            raise ValueError("Expected matching (B, C) logit tensors.")

        if self.reset_each_batch:
            self._reset()

        rho = torch.tensor([self._rho], device=logits_l.device,
                           dtype=logits_l.dtype, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [rho], lr=1.0, max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            lam = torch.sigmoid(rho)
            z = lam * logits_l + (1.0 - lam) * logits_s
            loss = softmax_entropy(z).mean()
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        with torch.no_grad():
            lo = self._logit_min if math.isfinite(self._logit_min) else -1e9
            hi = self._logit_max if math.isfinite(self._logit_max) else  1e9
            rho.clamp_(lo, hi)

        self._rho = float(rho.detach())
        self.last_lambda = float(torch.sigmoid(rho).detach())
        self.last_loss = float(loss.detach()) if loss is not None else float("nan")
        return self.last_lambda
