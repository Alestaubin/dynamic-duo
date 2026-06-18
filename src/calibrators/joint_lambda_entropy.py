import logging

import torch

from src.calibrators.temp.lambda_entropy_temperature import LambdaEntropyTemperature
from src.calibrators.base import BaseJointCalibrator, _NoOpModule

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class JointLambdaEntropy(BaseJointCalibrator):
    """Per-batch entropy-minimizing linear mixture calibrator.

    Combines large and small model logits as:
        z = λ · z_l + (1 - λ) · z_s,   λ ∈ [lambda_min, lambda_max]

    λ is a single scalar fitted per batch by minimizing mean Shannon entropy
    of softmax(z). Delegates optimization to LambdaEntropyTemperature.

    Convention: the FIRST logits argument is the LARGE model,
    matching DynamicDuo's calibrate(z_large, z_small).
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
        self.kernel = LambdaEntropyTemperature(
            init_lambda=init_lambda,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            max_iter=max_iter,
            reset_each_batch=reset_each_batch,
        )
        self.last_lambda: float = init_lambda
        self.last_loss: float = float("nan")

    # ------------------------------------------------------------------ #
    def _fit(self, z_l: torch.Tensor, z_s: torch.Tensor) -> float:
        self.last_lambda = self.kernel.adapt(z_l, z_s)
        self.last_loss = self.kernel.last_loss
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
