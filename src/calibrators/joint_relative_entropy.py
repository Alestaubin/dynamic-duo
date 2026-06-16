import logging

import torch

from src.calibrators.base import _NoOpModule, BaseJointCalibrator
from src.calibrators.temp.relative_entropy_temperature import RelativeEntropyTemperature
from src.utils.logit_transforms import combine_logits

logger = logging.getLogger(__name__)


class JointRelativeEntropy(BaseJointCalibrator):
    """Per-batch entropy-minimizing calibrator with one degree of freedom.

    Parameterization: T_l = exp(w/2), T_s = exp(-w/2), so T_l * T_s = 1.
    Only the ratio T_l/T_s = exp(w) is optimized; the overall scale is fixed.
    """

    def __init__(
        self,
        init_w: float = 0.0,
        t_max: float = 10.0,
        reset_each_batch: bool = True,
        device=None,
    ):
        super().__init__()
        self.ent = RelativeEntropyTemperature(
            init_w=init_w,
            t_max=t_max,
            reset_each_batch=reset_each_batch,
            device=device,
        )
        self.last_tau_l: float = float("nan")
        self.last_tau_s: float = float("nan")
        self.last_entropy: float = float("nan")

    def _fit(self, z_l: torch.Tensor, z_s: torch.Tensor):
        with torch.no_grad():
            agree = (z_l.argmax(dim=1) == z_s.argmax(dim=1)).float().mean().item()
        self.last_tau_l, self.last_tau_s = self.ent.adapt(z_l, z_s)
        self.last_entropy = self.ent.last_loss
        print(
            f"batch agree={agree:.3f}  w={self.ent.w:.4f}"
            f"  tau_l={float(self.last_tau_l):.4f}  tau_s={float(self.last_tau_s):.4f}"
            f"  H={float(self.last_entropy):.4f}"
        )
        return self.last_tau_l, self.last_tau_s

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
        logger.info("JointRelativeEntropy is self-adapting; tune() is a no-op.")

    @property
    def model(self):
        return _NoOpModule()
