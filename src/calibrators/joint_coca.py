import logging

import torch

from src.calibrators.coca_temperature import CocaTemperature
from src.calibrators.base import _NoOpModule, BaseJointCalibrator
logger = logging.getLogger(__name__)



class JointCoca(BaseJointCalibrator):
    """Self-adapting COCA calibrator for an asymmetric duo.

    Each batch it fits a single temperature tau that aligns the smaller
    (source) model to the larger (anchor) model in exponential space
    (COCA Eqns 1-2, via CocaTemperature), then returns aggregated ensemble
    logits.

    Aggregation (logit space):
        p_e = (z_l + z_s / tau) / 2          # anchor + tau-aligned source mean

    Sharpness-preserving rescale (per sample, detached):
        T   = max_c p_e / max_c z_l
        out = p_e / T                        # => max_c out == max_c z_l

    so aggregation never changes the magnitude of the winning logit relative to
    the anchor. T is treated as a constant (no grad through the max), so on the
    duo loss path gradients flow into the model logits exactly as with any other
    calibrator.

    Regime: self-adapting. It owns its optimizer (inside CocaTemperature) and
    fits tau itself per batch -- DynamicDuo neither freezes nor steps it.

    Note: DynamicDuo calls the calibrator twice per batch (calibrate_with_grad
    on the loss path, then calibrate on the output path), so tau is fit twice.
    With reset_each_batch=True the two fits run on the same logits from tau=1 and
    are identical, and the cost is K gradient steps on one scalar -- negligible
    next to a model forward. If you switch to reset_each_batch=False (warm start)
    note that this gives tau 2K steps per batch.

    Convention: the FIRST logits argument is the ANCHOR (the larger model),
    matching DynamicDuo's calibrate(z_large, z_small).
    """

    def __init__(
        self,
        num_steps: int = 10,
        lr: float = 5e-2,
        init_temp: float = 1.0,
        reset_each_batch: bool = True,
        t_min: float = 0.1,
        t_max: float = 10.0,
        eps: float = 1e-4,
        device=None,
    ):
        super().__init__()
        self.coca = CocaTemperature(
            num_steps=num_steps,
            lr=lr,
            init_temp=init_temp,
            reset_each_batch=reset_each_batch,
            device=device,
        )
        self.t_min = t_min
        self.t_max = t_max
        self.eps = eps

        self.last_tau: float = float("nan")
        self.last_disc_loss: float = float("nan")

    # ------------------------------------------------------------------ #
    def _align_device(self, logits: torch.Tensor) -> None:
        """Keep CocaTemperature's parameter on the logits' device."""
        if self.coca.rho.device != logits.device:
            self.coca.rho = torch.nn.Parameter(self.coca.rho.detach().to(logits.device))
            self.coca.device = logits.device
            self.coca._build_optimizer()

    def _fit(self, z_anchor: torch.Tensor, z_source: torch.Tensor) -> float:
        self._align_device(z_anchor)
        # adapt() detaches internally and runs its own K-step optimization.
        self.last_tau = self.coca.adapt(z_anchor, z_source)
        self.last_disc_loss = self.coca.last_loss
        return self.last_tau

    def _aggregate(self, z_anchor: torch.Tensor, z_source: torch.Tensor, tau: float) -> torch.Tensor:
        # Anchor stays at T=1 (it is the fixed target); only the source is scaled.
        p_e = (z_anchor + z_source / tau) / 2.0

        with torch.no_grad():
            anchor_max = z_anchor.max(dim=1).values            # (B,)
            ens_max = p_e.max(dim=1).values                    # (B,)
            T = ens_max / anchor_max.clamp_min(self.eps)
            # Fall back to no rescale where the anchor's top logit is ~0 or
            # negative (degenerate / very flat predictions, e.g. heavy blur),
            # which would otherwise produce a meaningless or sign-flipped T.
            T = torch.where(
                torch.isfinite(T) & (anchor_max > self.eps),
                T, torch.ones_like(T),
            )
            T = T.clamp(self.t_min, self.t_max).unsqueeze(1)   # (B, 1), detached

        return p_e / T

    # --- BaseJointCalibrator interface -------------------------------- #
    def calibrate_with_grad(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        tau = self._fit(logits_l, logits_s)
        return self._aggregate(logits_l, logits_s, tau)

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        tau = self._fit(logits_l, logits_s)
        with torch.no_grad():
            return self._aggregate(logits_l, logits_s, tau)

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        """COCA is self-adapting; there is nothing to fit before test time."""
        logger.info("JointCoca is self-adapting; tune() is a no-op.")

    @property
    def model(self):
        return _NoOpModule()