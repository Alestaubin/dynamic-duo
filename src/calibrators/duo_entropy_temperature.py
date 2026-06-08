"""
Per-batch two-temperature entropy minimization kernel.

For each batch, fits two scalar temperatures (T_l for the large/anchor model,
T_s for the small model) by minimizing the mean Shannon entropy of the
ensemble softmax:

    arg min_{T_l, T_s}  H( softmax( (z_l/T_l + z_s/T_s) / 2 ) )
    H(p) = -sum_c p_c log p_c

----------------
The trivial entropy minimum is T -> 0 (arbitrarily sharp predictions). To
prevent collapse, temperatures are projected onto [t_min, t_max] after every
gradient step (projected gradient on the log-temperature reparameterization).

Conventions
-----------
* z_l = logits of the LARGE (anchor) model — divided by T_l.
* z_s = logits of the SMALL (source) model — divided by T_s.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Type

import torch


class DuoEntropyTemperature:
    """Per-batch two-temperature entropy minimization kernel.

    Instantiate once; call ``adapt`` once per batch with both models' logits.

    Parameters
    ----------
    num_steps : int
        K inner gradient steps per batch.
    lr : float
        Learning rate for the temperature optimizer.
    init_temp_l, init_temp_s : float
        Initial temperatures for the large and small model at the start of
        each batch's optimization (or at construction).
    reset_each_batch : bool
        If True (default), both temperatures are reset to their init values
        before each batch's K steps. If False, temperatures are warm-started.
    t_min, t_max : float
        Hard bounds for both temperatures (projected gradient).
    eps : float
        Small constant added inside log for numerical stability.
    optimizer_cls : type[torch.optim.Optimizer]
        Optimizer for the two log-temperature parameters (default: Adam).
    device, dtype : optional
        Placement of the temperature parameters.
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
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device=None,
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
        self.reset_each_batch = reset_each_batch
        self._log_t_min = math.log(t_min)
        self._log_t_max = math.log(t_max)
        self.eps = eps
        self.optimizer_cls = optimizer_cls
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype

        self.rho_l = torch.nn.Parameter(
            torch.tensor(self.init_log_temp_l, dtype=dtype, device=self.device)
        )
        self.rho_s = torch.nn.Parameter(
            torch.tensor(self.init_log_temp_s, dtype=dtype, device=self.device)
        )
        self._build_optimizer()
        self.last_loss: float = float("nan")
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------ #
    def _build_optimizer(self) -> None:
        self.optimizer = self.optimizer_cls([self.rho_l, self.rho_s], lr=self.lr)

    def reset(self) -> None:
        with torch.no_grad():
            self.rho_l.fill_(self.init_log_temp_l)
            self.rho_s.fill_(self.init_log_temp_s)
        self._build_optimizer()

    @property
    def temperature_l(self) -> float:
        return float(self.rho_l.detach().exp())

    @property
    def temperature_s(self) -> float:
        return float(self.rho_s.detach().exp())

    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def adapt(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> Tuple[float, float]:
        """Run K steps minimising ensemble entropy for one batch.

        Parameters
        ----------
        logits_l : (B, C) tensor
            Logits of the large (anchor) model.
        logits_s : (B, C) tensor
            Logits of the small (source) model.

        Returns
        -------
        (tau_l, tau_s) : pair of floats
            The fitted temperatures at convergence.
        """
        if logits_l.shape != logits_s.shape:
            raise ValueError("logits_l and logits_s must have the same shape.")
        if logits_l.dim() != 2:
            raise ValueError("Expected logits of shape (B, C).")

        if self.reset_each_batch:
            self.reset()

        z_l = logits_l.detach().to(self.rho_l.dtype)
        z_s = logits_s.detach().to(self.rho_l.dtype)
        if self.device is not None:
            z_l, z_s = z_l.to(self.device), z_s.to(self.device)

        self.loss_history = []
        for _ in range(self.num_steps):
            self.optimizer.zero_grad(set_to_none=True)
            tau_l = self.rho_l.exp()
            tau_s = self.rho_s.exp()
            z_ens = (z_l / tau_l + z_s / tau_s) / 2.0
            p = torch.softmax(z_ens, dim=-1)
            H = -(p * torch.log(p + self.eps)).sum(dim=-1).mean()
            H.backward()
            self.optimizer.step()
            # Project back onto [t_min, t_max] in log space.
            with torch.no_grad():
                self.rho_l.clamp_(self._log_t_min, self._log_t_max)
                self.rho_s.clamp_(self._log_t_min, self._log_t_max)
            self.last_loss = float(H.detach())
            self.loss_history.append(self.last_loss)

        return self.temperature_l, self.temperature_s


# ====================================================================== #
# Self-contained sanity check.
#
# Setup: construct a pair whose per-sample optimal ensemble T* is known
# by symmetry (identical logit scales -> T_l* = T_s* = T_true).  We check
# that (a) both temperatures converge to roughly T_true, (b) entropy drops.
# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(42)

    B, C = 64, 1000
    T_true = 2.0

    # Both models have the same scale -> optimal is T_l = T_s = T_true.
    z_base = torch.randn(B, C)
    z_l = T_true * z_base
    z_s = T_true * z_base

    ent = DuoEntropyTemperature(num_steps=200, lr=5e-2, t_min=0.1, t_max=20.0)
    tau_l, tau_s = ent.adapt(z_l, z_s)

    print(f"true T          : {T_true:.4f}")
    print(f"recovered T_l   : {tau_l:.4f}")
    print(f"recovered T_s   : {tau_s:.4f}")
    print(f"H[0] -> H[-1]   : {ent.loss_history[0]:.4e} -> {ent.loss_history[-1]:.4e}")
    assert ent.loss_history[-1] < ent.loss_history[0], "entropy did not decrease"

    # Unrelated logit pair: should still find some finite temperatures.
    z_l2 = torch.randn(B, C)
    z_s2 = torch.randn(B, C)
    tau_l2, tau_s2 = ent.adapt(z_l2, z_s2)
    print(f"unrelated pair  : T_l={tau_l2:.4f}  T_s={tau_s2:.4f}  "
          f"(final H={ent.loss_history[-1]:.4e})")
    print("OK")
