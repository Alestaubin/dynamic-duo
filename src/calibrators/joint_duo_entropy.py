import logging

from src.utils.logit_transforms import combine_logits
import torch

from src.calibrators.temp.duo_entropy_temperature import DuoEntropyTemperature, LBFGSDuoEntropyTemperature
from src.calibrators.base import _NoOpModule, BaseJointCalibrator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG) 

class JointDuoEntropy(BaseJointCalibrator):
    """Per-batch entropy-minimizing calibrator with two temperatures.

    Each batch it fits T_l (large model) and T_s (small model) to minimize
    the mean Shannon entropy of duo logits. Calls DuoEntropyTemperature, which does the actual optimization.

    """

    def __init__(
        self,
        num_steps: int = 10,
        lr: float = 5e-2,
        init_temp_l: float = 1.0,
        init_temp_s: float = 1.0,
        reset_each_batch: bool = True,
        t_min: float = 0.05,
        t_max: float = 10.0,
        eps: float = 1e-8,
        device=None,
    ):
        super().__init__()
        # self.ent = DuoEntropyTemperature(
        #     num_steps=num_steps,
        #     lr=lr,
        #     init_temp_l=init_temp_l,
        #     init_temp_s=init_temp_s,
        #     reset_each_batch=reset_each_batch,
        #     t_min=t_min,
        #     t_max=t_max,
        #     eps=eps,
        #     device=device,
        # )
        
        self.ent = LBFGSDuoEntropyTemperature(
            init_temp_l=init_temp_l,
            init_temp_s=init_temp_s,
            device=device,
        )

        self.last_tau_l: float = float("nan")
        self.last_tau_s: float = float("nan")
        self.last_entropy: float = float("nan")

    # ------------------------------------------------------------------ #
    def _align_device(self, logits: torch.Tensor) -> None:
        """Keep DuoEntropyTemperature's parameters on the logits' device."""
        rho = self.ent.rho_l
        if not isinstance(rho, torch.Tensor):
            return
        if rho.device != logits.device:
            self.ent.rho_l = torch.nn.Parameter(self.ent.rho_l.detach().to(logits.device))
            self.ent.rho_s = torch.nn.Parameter(self.ent.rho_s.detach().to(logits.device))
            self.ent.device = logits.device
            self.ent._build_optimizer()

    def _fit(self, z_l: torch.Tensor, z_s: torch.Tensor):
        self._align_device(z_l)
        with torch.no_grad():
            agree = (z_l.argmax(dim=1) == z_s.argmax(dim=1)).float().mean().item()
        self.last_tau_l, self.last_tau_s = self.ent.adapt(z_l, z_s)
        self.last_entropy = self.ent.last_loss
        # logger.debug(
        #         "batch agree=%.3f  tau_l=%.4f  tau_s=%.4f  H=%.4f",
        #         agree, self.last_tau_l, self.last_tau_s, self.last_entropy,
        # )
        print(f"batch agree={agree:.3f}  tau_l={float(self.last_tau_l):.4f}  tau_s={float(self.last_tau_s):.4f}  H={float(self.last_entropy):.4f}")
        return self.last_tau_l, self.last_tau_s

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
        """JointDuoEntropy is self-adapting; there is nothing to fit before test time."""
        logger.info("JointDuoEntropy is self-adapting; tune() is a no-op.")

    @property
    def model(self):
        return _NoOpModule()
