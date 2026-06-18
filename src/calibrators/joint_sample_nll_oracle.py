import logging

import torch

from src.calibrators.temp.sample_nll_temperature import SampleNLLTemperature
from src.calibrators.base import _NoOpModule, BaseJointCalibrator
from src.utils.logit_transforms import combine_logits

logger = logging.getLogger(__name__)


class JointSampleNLLOracle(BaseJointCalibrator):
    """Per-sample oracle calibrator: one (T_l, T_s) pair per sample per batch.

    Fits temperatures that minimise NLL on each individual sample — the
    tightest possible per-instance fixed-temperature baseline.  Uses batch
    labels: this is a cheating oracle, not a valid test-time method.
    """

    def __init__(
        self,
        num_steps: int = 20,
        lr: float = 5e-2,
        init_temp_l: float = 1.0,
        init_temp_s: float = 1.0,
        t_min: float = 0.05,
        t_max: float = 20.0,
    ):
        super().__init__()
        self.kernel = SampleNLLTemperature(
            num_steps=num_steps,
            lr=lr,
            init_temp_l=init_temp_l,
            init_temp_s=init_temp_s,
            t_min=t_min,
            t_max=t_max,
        )
        self._labels: torch.Tensor | None = None
        self.last_tau_l: torch.Tensor | None = None  # (B, 1) after each fit
        self.last_tau_s: torch.Tensor | None = None
        self.last_nll: float = float("nan")

    # ------------------------------------------------------------------ #
    def set_labels(self, labels: torch.Tensor) -> None:
        """Provide the batch labels before the next calibrate call."""
        self._labels = labels

    def _fit(
        self, z_l: torch.Tensor, z_s: torch.Tensor
    ):
        if self._labels is None:
            raise RuntimeError(
                "JointSampleNLLOracle requires labels before each calibrate call. "
                "Call set_labels(labels) first, or use DynamicDuo with "
                "calibration_mode='sample_oracle_ts'."
            )
        tau_l, tau_s = self.kernel.adapt(z_l, z_s, self._labels)
        self.last_tau_l = tau_l
        self.last_tau_s = tau_s
        self.last_nll = self.kernel.last_loss
        return tau_l, tau_s

    # --- BaseJointCalibrator interface -------------------------------- #
    def calibrate_with_grad(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        tau_l, tau_s = self._fit(logits_l, logits_s)
        return combine_logits(z_l=logits_l, z_s=logits_s, tau_l=tau_l, tau_s=tau_s)

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        tau_l, tau_s = self._fit(logits_l, logits_s)
        with torch.no_grad():
            return combine_logits(z_l=logits_l, z_s=logits_s, tau_l=tau_l, tau_s=tau_s)

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        logger.info("JointSampleNLLOracle is a per-batch oracle; tune() is a no-op.")

    @property
    def model(self):
        return _NoOpModule()
