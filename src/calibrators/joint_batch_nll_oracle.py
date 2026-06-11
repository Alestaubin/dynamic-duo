import logging

import torch

from src.calibrators.batch_nll_temperature import BatchNLLTemperature
from src.calibrators.base import _NoOpModule, BaseJointCalibrator

logger = logging.getLogger(__name__)


class JointBatchNLLOracle(BaseJointCalibrator):
    """Per-batch oracle calibrator: fits T_l and T_s to minimize NLL on each batch.

    Uses batch labels — this is a cheating oracle, not a valid test-time method.
    Its purpose is to bound how much temperature scaling can help per batch.

    Usage with DynamicDuo
    ---------------------
    Because BaseJointCalibrator.calibrate() does not receive labels, the caller
    must supply them via set_labels(labels) before each forward pass.
    DynamicDuo.forward() does this automatically when calibration_mode is
    "batch_oracle_ts".

    Convention: the FIRST logits argument is the LARGE (anchor) model,
    matching DynamicDuo's calibrate(z_large, z_small).
    """

    def __init__(
        self,
        num_steps: int = 20,
        lr: float = 5e-2,
        init_temp_l: float = 1.0,
        init_temp_s: float = 1.0,
        reset_each_batch: bool = True,
        t_min: float = 0.05,
        t_max: float = 20.0,
        device=None,
    ):
        super().__init__()
        self.kernel = BatchNLLTemperature(
            num_steps=num_steps,
            lr=lr,
            init_temp_l=init_temp_l,
            init_temp_s=init_temp_s,
            reset_each_batch=reset_each_batch,
            t_min=t_min,
            t_max=t_max,
            device=device,
        )
        self._labels: torch.Tensor | None = None
        self.last_tau_l: float = float("nan")
        self.last_tau_s: float = float("nan")
        self.last_nll: float = float("nan")

    # ------------------------------------------------------------------ #
    def set_labels(self, labels: torch.Tensor) -> None:
        """Provide the batch labels before the next calibrate call."""
        self._labels = labels

    def _align_device(self, logits: torch.Tensor) -> None:
        if self.kernel.rho_l.device != logits.device:
            self.kernel.rho_l = torch.nn.Parameter(
                self.kernel.rho_l.detach().to(logits.device)
            )
            self.kernel.rho_s = torch.nn.Parameter(
                self.kernel.rho_s.detach().to(logits.device)
            )
            self.kernel.device = logits.device
            self.kernel._build_optimizer()

    def _fit(self, z_l: torch.Tensor, z_s: torch.Tensor):
        if self._labels is None:
            raise RuntimeError(
                "JointBatchNLLOracle requires labels before each calibrate call. "
                "Call set_labels(labels) first, or use DynamicDuo with "
                "calibration_mode='batch_oracle_ts'."
            )
        self._align_device(z_l)
        self.last_tau_l, self.last_tau_s = self.kernel.adapt(z_l, z_s, self._labels)
        self.last_nll = self.kernel.last_loss
        return self.last_tau_l, self.last_tau_s

    def _aggregate(
        self, z_l: torch.Tensor, z_s: torch.Tensor, tau_l: float, tau_s: float
    ) -> torch.Tensor:
        return (z_l / tau_l + z_s / tau_s) / 2.0

    # --- BaseJointCalibrator interface -------------------------------- #
    def calibrate_with_grad(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        tau_l, tau_s = self._fit(logits_l, logits_s)
        return self._aggregate(logits_l, logits_s, tau_l, tau_s)

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        tau_l, tau_s = self._fit(logits_l, logits_s)
        with torch.no_grad():
            return self._aggregate(logits_l, logits_s, tau_l, tau_s)

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        logger.info("JointBatchNLLOracle is a per-batch oracle; tune() is a no-op.")

    @property
    def model(self):
        return _NoOpModule()
