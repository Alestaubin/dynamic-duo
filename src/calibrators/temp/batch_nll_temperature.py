"""
Per-batch two-temperature NLL minimization kernel.

For each batch, fits T_l and T_s to minimize the cross-entropy loss of the
ensemble against the batch's true labels:

    arg min_{T_l, T_s}  CE( (z_l/T_l + z_s/T_s) / 2,  y )

This is a cheating oracle: it uses the batch labels that are not available
at real test time. The resulting temperatures are the tightest per-batch
fixed-temperature baseline possible.

Unlike entropy minimization (which collapses to t_min for any correctly
predicted batch), NLL minimization finds a non-trivial T* that balances
correct-class confidence against the mix of correct/incorrect predictions in
the batch. On corrupted data the optimizer will tend to increase T (soften
predictions) when the ensemble is confused, and decrease T (sharpen) when it
is correct.

Conventions
-----------
* z_l = logits of the LARGE model — divided by T_l.
* z_s = logits of the SMALL model — divided by T_s.
* Both are detached: only T_l and T_s are optimised.
* labels = (B,) long tensor of true class indices.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Type

import torch
import torch.nn.functional as F


class BatchNLLTemperature:
    """Per-batch two-temperature NLL minimization kernel.

    Instantiate once; call ``adapt`` once per batch.

    Parameters
    ----------
    num_steps : int
        K inner gradient steps per batch.
    lr : float
        Learning rate for the Adam optimiser.
    init_temp_l, init_temp_s : float
        Initial temperatures for the large and small model.
    reset_each_batch : bool
        If True (default), temperatures are reset to init before each batch.
    t_min, t_max : float
        Hard bounds applied via projected gradient.
    optimizer_cls : type[torch.optim.Optimizer]
        Defaults to Adam.
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
    def adapt(
        self,
        logits_l: torch.Tensor,
        logits_s: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """Run K steps minimizing CE((z_l/T_l + z_s/T_s)/2, labels) for one batch.

        Parameters
        ----------
        logits_l, logits_s : (B, C) tensors
        labels : (B,) long tensor of true class indices

        Returns
        -------
        (tau_l, tau_s) : fitted temperatures
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
        y = labels.to(device=z_l.device, dtype=torch.long)

        self.loss_history = []
        for _ in range(self.num_steps):
            self.optimizer.zero_grad(set_to_none=True)
            tau_l = self.rho_l.exp()
            tau_s = self.rho_s.exp()
            z_ens = (z_l / tau_l + z_s / tau_s) / 2.0
            loss = F.cross_entropy(z_ens, y)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.rho_l.clamp_(self._log_t_min, self._log_t_max)
                self.rho_s.clamp_(self._log_t_min, self._log_t_max)
            self.last_loss = float(loss.detach())
            self.loss_history.append(self.last_loss)

        return self.temperature_l, self.temperature_s


# ====================================================================== #
# Self-contained sanity check.
# Mix of correctly and incorrectly predicted samples to get non-trivial T*.
# ====================================================================== #
if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    B, C = 128, 1000

    # Half of the batch: correct class has the highest logit.
    # Other half: logits are random (likely wrong).
    labels = torch.randint(0, C, (B,))
    z_base = torch.randn(B, C)
    # Boost correct-class logit for the first half so ensemble is ~50% accurate.
    z_base[: B // 2].scatter_(1, labels[: B // 2].unsqueeze(1), 5.0)

    z_l = 1.5 * z_base
    z_s = 0.8 * z_base

    kernel = BatchNLLTemperature(num_steps=50, lr=5e-2, t_min=0.1, t_max=20.0)
    tau_l, tau_s = kernel.adapt(z_l, z_s, labels)

    print(f"fitted T_l      : {tau_l:.4f}")
    print(f"fitted T_s      : {tau_s:.4f}")
    print(f"NLL[0] -> NLL[-1]: {kernel.loss_history[0]:.4e} -> {kernel.loss_history[-1]:.4e}")
    assert kernel.loss_history[-1] < kernel.loss_history[0], "NLL did not decrease"
    assert tau_l >= 0.09 and tau_s >= 0.09, "temperature hit below t_min"
    print("OK")
