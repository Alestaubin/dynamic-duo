"""
Per-sample two-temperature NLL minimization kernel.

For each batch, solves B independent temperature-scaling problems in parallel:

    T_l^i, T_s^i = argmin_{T_l, T_s} CE( (z_l_i/T_l + z_s_i/T_s) / 2,  y_i )

implemented as B-dimensional parameter vectors [rho_l_1 ... rho_l_B] and
[rho_s_1 ... rho_s_B] optimised with a single Adam pass over K steps.
Independence is structural: the loss for sample i only involves rho_l_i and
rho_s_i, so their gradients are automatically disjoint and can be vectorised.

This is a cheating oracle: it requires the per-sample true labels y at
calibration time.  Its purpose is to measure the ceiling of per-sample
temperature scaling — the tightest possible per-instance fixed-T baseline.

Unlike BatchNLLTemperature (two shared scalars), temperature parameters here
are allocated fresh each adapt() call because batch size B can vary.

Conventions
-----------
* z_l, z_s : (B, C) logits — detached before optimisation.
* labels   : (B,) long tensor of true class indices.
* Returns  : tau_l, tau_s of shape (B, 1) for direct broadcasting in the
             ensemble aggregation (z_l / tau_l + z_s / tau_s) / 2.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Type

import torch
import torch.nn.functional as F


class SampleNLLTemperature:
    """Per-sample NLL minimization kernel. B independent problems solved in parallel.

    Parameters
    ----------
    num_steps : int
        K Adam steps per batch.
    lr : float
        Learning rate.
    init_temp_l, init_temp_s : float
        Initial temperature for every sample's T_l / T_s at the start of each
        adapt() call.  (There is no warm-starting across batches: parameters
        are re-allocated each call because B can vary.)
    t_min, t_max : float
        Hard bounds via projected gradient on the log-temperature vectors.
    """

    def __init__(
        self,
        num_steps: int = 20,
        lr: float = 5e-2,
        init_temp_l: float = 1.0,
        init_temp_s: float = 1.0,
        t_min: float = 0.05,
        t_max: float = 20.0,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if init_temp_l <= 0 or init_temp_s <= 0:
            raise ValueError("init_temp_l and init_temp_s must be positive.")
        if t_min <= 0 or t_min >= t_max:
            raise ValueError("Need 0 < t_min < t_max.")

        self.num_steps = num_steps
        self.lr = lr
        self.init_log_temp_l = math.log(init_temp_l)
        self.init_log_temp_s = math.log(init_temp_s)
        self._log_t_min = math.log(t_min)
        self._log_t_max = math.log(t_max)
        self.optimizer_cls = optimizer_cls
        self.dtype = dtype

        self.last_loss: float = float("nan")
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def adapt(
        self,
        logits_l: torch.Tensor,
        logits_s: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit one (T_l, T_s) pair per sample; return (tau_l, tau_s) of shape (B, 1).

        Parameters
        ----------
        logits_l, logits_s : (B, C) tensors
        labels : (B,) long tensor of true class indices

        Returns
        -------
        tau_l, tau_s : (B, 1) detached tensors
        """
        if logits_l.shape != logits_s.shape:
            raise ValueError("logits_l and logits_s must have the same shape.")
        if logits_l.dim() != 2:
            raise ValueError("Expected logits of shape (B, C).")

        B = logits_l.shape[0]
        z_l = logits_l.detach().to(self.dtype)
        z_s = logits_s.detach().to(self.dtype)
        y = labels.to(device=z_l.device, dtype=torch.long)

        # Fresh B-dim parameter vectors allocated on the logits' device.
        rho_l = torch.nn.Parameter(
            torch.full((B,), self.init_log_temp_l, dtype=self.dtype, device=z_l.device)
        )
        rho_s = torch.nn.Parameter(
            torch.full((B,), self.init_log_temp_s, dtype=self.dtype, device=z_l.device)
        )
        optimizer = self.optimizer_cls([rho_l, rho_s], lr=self.lr)

        self.loss_history = []
        for _ in range(self.num_steps):
            optimizer.zero_grad(set_to_none=True)
            tau_l = rho_l.exp().unsqueeze(1)          # (B, 1)
            tau_s = rho_s.exp().unsqueeze(1)          # (B, 1)
            z_ens = (z_l / tau_l + z_s / tau_s) / 2.0
            loss = F.cross_entropy(z_ens, y)          # mean over B; grad flows to rho_l_i, rho_s_i independently
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                rho_l.clamp_(self._log_t_min, self._log_t_max)
                rho_s.clamp_(self._log_t_min, self._log_t_max)
            self.last_loss = float(loss.detach())
            self.loss_history.append(self.last_loss)

        return rho_l.exp().detach().unsqueeze(1), rho_s.exp().detach().unsqueeze(1)


# ====================================================================== #
# Self-contained sanity check.
# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C = 32, 1000

    labels = torch.randint(0, C, (B,))
    z_base = torch.randn(B, C)
    # Make each sample correctly predicted (boost correct-class logit).
    z_base.scatter_(1, labels.unsqueeze(1), 5.0)
    z_l = 2.0 * z_base
    z_s = 2.0 * z_base

    kernel = SampleNLLTemperature(num_steps=100, lr=5e-2, t_min=0.1, t_max=20.0)
    tau_l, tau_s = kernel.adapt(z_l, z_s, labels)

    print(f"tau_l shape     : {tau_l.shape}")          # (B, 1)
    print(f"tau_l range     : [{tau_l.min():.4f}, {tau_l.max():.4f}]")
    print(f"NLL[0] -> [-1]  : {kernel.loss_history[0]:.4e} -> {kernel.loss_history[-1]:.4e}")
    assert tau_l.shape == (B, 1) and tau_s.shape == (B, 1)
    assert kernel.loss_history[-1] < kernel.loss_history[0], "NLL did not decrease"
    assert tau_l.min() >= 0.09, "temperature hit below t_min"
    print("OK")
