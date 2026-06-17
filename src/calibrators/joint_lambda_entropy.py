import math
import logging

import torch

from src.calibrators.base import BaseJointCalibrator, _NoOpModule
from src.tta.tent import softmax_entropy

logger = logging.getLogger(__name__)


class JointLambdaEntropy(BaseJointCalibrator):
    """Per-batch entropy-minimizing linear mixture calibrator.

    Combines large and small model logits as:
        z = λ · z_l + (1 - λ) · z_s,   λ ∈ [lambda_min, lambda_max]

    λ is a single scalar fitted per batch by minimizing mean Shannon entropy
    of softmax(z). Reparameterized via logit/sigmoid so unconstrained
    optimization stays in [lambda_min, lambda_max]; optimized with L-BFGS.
    """

    def __init__(
        self,
        init_lambda: float = 0.5,
        lambda_min: float = 0.0,
        lambda_max: float = 1.0,
        max_iter: int = 50,
        reset_each_batch: bool = True,
    ):
        super().__init__()
        if not (0.0 < init_lambda < 1.0):
            raise ValueError("init_lambda must be strictly in (0, 1)")
        if not (0.0 <= lambda_min < lambda_max <= 1.0):
            raise ValueError("Need 0 <= lambda_min < lambda_max <= 1")

        self.init_lambda = init_lambda
        self.max_iter = max_iter
        self.reset_each_batch = reset_each_batch

        # logit bounds for clamping ρ = logit(λ)
        self._logit_min = math.log(lambda_min / (1.0 - lambda_min)) if lambda_min > 0.0 else -math.inf
        self._logit_max = math.log(lambda_max / (1.0 - lambda_max)) if lambda_max < 1.0 else math.inf
        self._init_rho = math.log(init_lambda / (1.0 - init_lambda))
        self._rho = self._init_rho  # warm-start state across batches

        self.last_lambda: float = init_lambda
        self.last_loss: float = float("nan")

    # ------------------------------------------------------------------ #
    def _fit(self, z_l: torch.Tensor, z_s: torch.Tensor) -> float:
        if self.reset_each_batch:
            self._rho = self._init_rho

        rho = torch.tensor([self._rho], device=z_l.device, dtype=z_l.dtype, requires_grad=True)
        optimizer = torch.optim.LBFGS([rho], lr=1.0, max_iter=self.max_iter,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            lam = torch.sigmoid(rho)
            z = lam * z_l + (1.0 - lam) * z_s
            loss = softmax_entropy(z).mean()
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        with torch.no_grad():
            if math.isfinite(self._logit_min) or math.isfinite(self._logit_max):
                rho.clamp_(
                    self._logit_min if math.isfinite(self._logit_min) else -1e9,
                    self._logit_max if math.isfinite(self._logit_max) else 1e9,
                )

        self._rho = float(rho.detach())
        self.last_lambda = float(torch.sigmoid(rho).detach())
        self.last_loss = float(loss.detach()) if loss is not None else float("nan")
        logger.debug("lambda=%.4f  H=%.4f", self.last_lambda, self.last_loss)
        return self.last_lambda

    @staticmethod
    def _combine(z_l: torch.Tensor, z_s: torch.Tensor, lam: float) -> torch.Tensor:
        return lam * z_l + (1.0 - lam) * z_s

    # --- BaseJointCalibrator interface -------------------------------- #
    def calibrate_with_grad(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        lam = self._fit(logits_l.detach(), logits_s.detach())
        return self._combine(logits_l, logits_s, lam)

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        lam = self._fit(logits_l.detach(), logits_s.detach())
        with torch.no_grad():
            return self._combine(logits_l, logits_s, lam)

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        logger.info("JointLambdaEntropy is self-adapting; tune() is a no-op.")

    @property
    def model(self):
        return _NoOpModule()
