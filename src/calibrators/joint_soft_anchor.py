"""
joint_soft_anchor.py
====================
Reliability-proxy-based soft-anchor calibrator for an asymmetric duo.

Per batch:
  1. Computes a scalar reliability score r per model
     (nuclear_norm / atc / prototype).
  2. Builds a detached soft anchor:
       a  = sigmoid((r_l - r_s) / tau_gate)        # routing weight ∈ (0,1)
       q  = a·p_l + (1-a)·p_s                       # teacher distribution
  3. Fits T_l, T_s for num_steps steps of
       KL(q || p(z_l/T_l)) + KL(q || p(z_s/T_s))
  4. Returns log(p_duo) = log(a·p(z_l/T_l) + (1-a)·p(z_s/T_s))
     so that the outer softmax_entropy loss adapts the BN/LN parameters.

Note: since softmax(log(p_duo)) = p_duo (p_duo sums to 1),
softmax_entropy(log(p_duo)) = entropy(p_duo) exactly — the TENT signal
on the mixture is well-defined.
"""

from __future__ import annotations

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.calibrators.base import BaseJointCalibrator, _NoOpModule
from src.proxies.proxies import (
    ModelProxyConfig,
    FeatureExtractor,
    nuclear_norm_score,
    atc_score,
    prototype_score,
)

logger = logging.getLogger(__name__)

_PROXY_KINDS = {"nuclear_norm", "atc", "prototype"}
_SOFTPLUS_INV_1 = math.log(math.exp(1.0) - 1.0)  # softplus^{-1}(1) ≈ 0.5413


class JointSoftAnchor(BaseJointCalibrator):
    """Soft-anchor calibrator for an asymmetric duo.

    Proxy kinds and their source-data requirements:
      "nuclear_norm"  — no source data needed; purely logit-based.
      "atc"           — needs cfg.atc_threshold fitted on source logits.
      "prototype"     — needs cfg.prototypes AND register_hooks() called.

    Parameters
    ----------
    proxy_kind : "nuclear_norm" | "atc" | "prototype"
    cfg_l, cfg_s : ModelProxyConfig for large and small model respectively.
    tau_gate : gate sharpness for the sigmoid anchor weight. Larger = harder gate.
    num_steps : gradient steps per batch for T_l, T_s fitting.
    lr : learning rate for the per-batch temperature optimizer.
    reset_each_batch : if True, reset T_l=T_s=1 before each batch's fitting.
    """

    def __init__(
        self,
        proxy_kind: str,
        cfg_l: ModelProxyConfig,
        cfg_s: ModelProxyConfig,
        tau_gate: float = 1.0,
        num_steps: int = 5,
        lr: float = 5e-2,
        reset_each_batch: bool = True,
    ):
        super().__init__()
        assert proxy_kind in _PROXY_KINDS, \
            f"proxy_kind must be one of {_PROXY_KINDS}, got '{proxy_kind}'"
        self.proxy_kind = proxy_kind
        self.cfg_l = cfg_l
        self.cfg_s = cfg_s
        self.tau_gate = tau_gate
        self.num_steps = num_steps
        self.lr = lr
        self.reset_each_batch = reset_each_batch

        self.raw_T_l = nn.Parameter(torch.tensor(_SOFTPLUS_INV_1))
        self.raw_T_s = nn.Parameter(torch.tensor(_SOFTPLUS_INV_1))
        self._optimizer: torch.optim.Optimizer | None = None

        # Feature hooks (registered by setup_duo when proxy_kind == "prototype")
        self._ext_l: FeatureExtractor | None = None
        self._ext_s: FeatureExtractor | None = None

        logger.info(
            "JointSoftAnchor | proxy=%s tau_gate=%.2f num_steps=%d lr=%.4f "
            "reset_each_batch=%s",
            proxy_kind, tau_gate, num_steps, lr, reset_each_batch,
        )

    # ── Hook management ───────────────────────────────────────────────────── #

    def register_hooks(self, model_l: nn.Module, model_s: nn.Module) -> None:
        """Register penultimate-feature hooks on both models (prototype proxy)."""
        self._ext_l = FeatureExtractor(model_l, self.cfg_l.name)
        self._ext_s = FeatureExtractor(model_s, self.cfg_s.name)
        logger.info(
            "JointSoftAnchor: registered feature hooks on %s and %s",
            self.cfg_l.name, self.cfg_s.name,
        )

    def remove_hooks(self) -> None:
        if self._ext_l is not None:
            self._ext_l.remove(); self._ext_l = None
        if self._ext_s is not None:
            self._ext_s.remove(); self._ext_s = None

    # ── Internal helpers ──────────────────────────────────────────────────── #

    def _get_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                [self.raw_T_l, self.raw_T_s], lr=self.lr
            )
        return self._optimizer

    def _align_device(self, device: torch.device) -> None:
        if self.raw_T_l.device != device:
            self.raw_T_l = nn.Parameter(self.raw_T_l.detach().to(device))
            self.raw_T_s = nn.Parameter(self.raw_T_s.detach().to(device))
            self._optimizer = None

    @torch.no_grad()
    def _proxy_scores(
        self, z_l: torch.Tensor, z_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (r_l, r_s) as scalar tensors on z_l.device."""
        device = z_l.device
        if self.proxy_kind == "nuclear_norm":
            r_l = nuclear_norm_score(z_l)
            r_s = nuclear_norm_score(z_s)
        elif self.proxy_kind == "atc":
            assert self.cfg_l.atc_threshold is not None, \
                "atc proxy requires cfg_l.atc_threshold (build with build_proxy_configs)"
            r_l = atc_score(z_l, self.cfg_l.atc_threshold, self.cfg_l.atc_kind)
            r_s = atc_score(z_s, self.cfg_s.atc_threshold, self.cfg_s.atc_kind)
        else:  # prototype
            assert self._ext_l is not None, \
                "prototype proxy requires register_hooks() called before inference"
            assert self.cfg_l.prototypes is not None, \
                "prototype proxy requires cfg_l.prototypes (build with build_proxy_configs)"
            r_l = prototype_score(self._ext_l._feats, self.cfg_l.prototypes.to(device))
            r_s = prototype_score(self._ext_s._feats, self.cfg_s.prototypes.to(device))
        return (torch.tensor(r_l, device=device, dtype=torch.float32),
                torch.tensor(r_s, device=device, dtype=torch.float32))

    def _fit_temps(
        self,
        z_l: torch.Tensor,
        z_s: torch.Tensor,
        r_l: torch.Tensor,
        r_s: torch.Tensor,
    ) -> None:
        """Fit T_l, T_s for num_steps steps of KL(q || p(z/T))."""
        if self.reset_each_batch:
            with torch.no_grad():
                self.raw_T_l.fill_(_SOFTPLUS_INV_1)
                self.raw_T_s.fill_(_SOFTPLUS_INV_1)
            self._optimizer = None  # discard stale Adam moment estimates

        opt = self._get_optimizer()
        z_l_d, z_s_d = z_l.detach(), z_s.detach()

        a = torch.sigmoid((r_l - r_s) / self.tau_gate).detach()
        q = (a * F.softmax(z_l_d, dim=1)
             + (1.0 - a) * F.softmax(z_s_d, dim=1)).detach()

        for _ in range(self.num_steps):
            opt.zero_grad(set_to_none=True)
            T_l = F.softplus(self.raw_T_l) + 1e-4
            T_s = F.softplus(self.raw_T_s) + 1e-4
            loss = (F.kl_div(F.log_softmax(z_l_d / T_l, dim=1), q, reduction="batchmean")
                  + F.kl_div(F.log_softmax(z_s_d / T_s, dim=1), q, reduction="batchmean"))
            loss.backward()
            opt.step()

    def _mix(
        self,
        z_l: torch.Tensor,
        z_s: torch.Tensor,
        r_l: torch.Tensor,
        r_s: torch.Tensor,
    ) -> torch.Tensor:
        """Return log(p_duo) using current T_l, T_s.

        softmax_entropy(log(p_duo)) = entropy(p_duo) since p_duo sums to 1.
        """
        T_l = F.softplus(self.raw_T_l) + 1e-4
        T_s = F.softplus(self.raw_T_s) + 1e-4
        a = torch.sigmoid((r_l - r_s) / self.tau_gate).detach()
        p_duo = (a * F.log_softmax(z_l / T_l, dim=1).exp()
                 + (1.0 - a) * F.log_softmax(z_s / T_s, dim=1).exp())
        return torch.log(p_duo.clamp(min=1e-8))

    # ── BaseJointCalibrator interface ─────────────────────────────────────── #

    def calibrate_with_grad(
        self, logits_l: torch.Tensor, logits_s: torch.Tensor
    ) -> torch.Tensor:
        self._align_device(logits_l.device)
        r_l, r_s = self._proxy_scores(logits_l, logits_s)
        self._fit_temps(logits_l, logits_s, r_l, r_s)  # fits T_l, T_s internally
        return self._mix(logits_l, logits_s, r_l, r_s)  # log(p_duo) for outer entropy

    def calibrate(
        self, logits_l: torch.Tensor, logits_s: torch.Tensor
    ) -> torch.Tensor:
        # T_l, T_s already fitted this batch by calibrate_with_grad
        self._align_device(logits_l.device)
        with torch.no_grad():
            r_l, r_s = self._proxy_scores(logits_l, logits_s)
            return self._mix(logits_l, logits_s, r_l, r_s)

    def forward(
        self, logits_l: torch.Tensor, logits_s: torch.Tensor
    ) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        logger.info("JointSoftAnchor is self-adapting; tune() is a no-op.")

    @property
    def model(self):
        return _NoOpModule()
