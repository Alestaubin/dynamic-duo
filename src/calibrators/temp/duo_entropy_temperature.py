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
from src.utils.logit_transforms import combine_logits

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
        """Run K steps minimizing ensemble entropy for one batch.

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
            z_ens = combine_logits(z_l, z_s, tau_l, tau_s)
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


class LBFGSDuoEntropyTemperature():
    """Same as DuoEntropyTemperature but with L-BFGS optimizer and line search.
    """
    def __init__(
        self,
        max_iter: int = 50,
        lr: float = 0.01,
        init_temp_l: float = 1.0,
        init_temp_s: float = 1.0,
        reset_each_batch: bool = True,
        t_min: float = 0.05,
        t_max: float = 10.0,
        eps: float = 1e-8,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.init_temp_l = init_temp_l
        self.init_temp_s = init_temp_s

        self.prev_temp_l = init_temp_l
        self.prev_temp_s = init_temp_s

        self.max_iter = max_iter
        self.lr = lr

    def adapt(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> Tuple[float, float]:
        """Run L-BFGS with line search to minimize ensemble entropy for one batch."""
        Tl = torch.tensor(self.prev_temp_l, requires_grad=True, device=logits_l.device)
        Ts = torch.tensor(self.prev_temp_s, requires_grad=True, device=logits_s.device)
        optimizer = torch.optim.LBFGS([Tl, Ts], lr=self.lr, max_iter=self.max_iter)

        def closure():
            optimizer.zero_grad()
            z_ens = combine_logits(logits_l, logits_s, Tl, Ts)
            p = torch.softmax(z_ens, dim=-1)
            loss.backward()
            return loss

