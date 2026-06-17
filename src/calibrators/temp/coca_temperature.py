"""
coca_temperature.py

Exact single-temperature, two-model temperature-finding kernel from COCA
(Yi et al., "When Small Guides Large: Cross-Model Co-Learning for Test-Time
Adaptation", 2025), corresponding to the anchor-guided alignment loss:

    arg min_tau  L_s( p_a(x), p_s(x) / tau )                          (Eqn. 1)
    L_s(p_a, p_s) = || exp(p_a) - exp(p_s) ||_1                        (Eqn. 2)

and the per-batch update loop:

    for each batch of test samples:
        compute logits from f_theta1, f_theta2
        for i = 1 to K:
            update tau via Eqn. 1

"""

from __future__ import annotations

from typing import List, Literal, Type

import torch
from src.tta.tent import softmax_entropy


class CocaTemperature:
    """COCA single-temperature anchor-guided alignment (Eqns 1-2), two models.

    Instantiate once; call ``adapt`` once per batch with the two models' logits.

    Parameters
    ----------
    num_steps : int
        K inner gradient steps on tau per batch (the inner loop of the algorithm).
    lr : float
        Learning rate for the temperature optimiser.
    init_temp : float
        Temperature at the start of each optimisation (tau = init_temp).
    reset_each_batch : bool
        If True (default), tau is reset to ``init_temp`` before each batch's K
        steps -- the literal reading of "for each batch ... update tau". If
        False, tau persists and is warm-started across batches.
    optimizer_cls : type[torch.optim.Optimizer]
        Optimiser for the single log-temperature parameter (default: Adam).
    device, dtype : optional
        Placement of the temperature parameter.
    """

    def __init__(
        self,
        num_steps: int = 10,
        lr: float = 1e-2,
        init_temp: float = 1.0,
        reset_each_batch: bool = True,
        loss: Literal["l1", "entropy"] = "l1",
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if init_temp <= 0:
            raise ValueError("init_temp must be positive.")
        if loss not in ("l1", "entropy"):
            raise ValueError(f"loss must be 'l1' or 'entropy', got {loss!r}.")
        self.num_steps = num_steps
        self.lr = lr
        self.loss = loss
        self.init_log_temp = float(torch.log(torch.tensor(init_temp)))
        self.reset_each_batch = reset_each_batch
        self.optimizer_cls = optimizer_cls
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype

        self.rho = torch.nn.Parameter(
            torch.tensor(self.init_log_temp, dtype=dtype, device=self.device)
        )
        self._build_optimizer()
        self.last_loss: float = float("nan")
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------ #
    def _build_optimizer(self) -> None:
        self.optimizer = self.optimizer_cls([self.rho], lr=self.lr)

    def reset(self) -> None:
        with torch.no_grad():
            self.rho.fill_(self.init_log_temp)
        self._build_optimizer()

    @property
    def temperature(self) -> float:
        """Current scalar temperature tau (detached)."""
        return float(self.rho.detach().exp())

    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def adapt(self, anchor_logits: torch.Tensor, scaled_logits: torch.Tensor) -> float:
        """Run K steps minimising Eqn. 1 for one batch; return the learned tau.

        Parameters
        ----------
        anchor_logits : (B, C) tensor
            Logits of the anchor (larger) model -- the fixed target p_a.
        scaled_logits : (B, C) tensor
            Logits of the scaled (smaller) model p_s, to be divided by tau.

        Both are detached internally: only tau is optimised, never the models.
        """
        if anchor_logits.shape != scaled_logits.shape:
            raise ValueError("anchor_logits and scaled_logits must have the same shape.")
        if anchor_logits.dim() != 2:
            raise ValueError("Expected logits of shape (B, C).")

        if self.reset_each_batch:
            self.reset()

        p_a = anchor_logits.detach().to(self.rho.dtype)
        p_s = scaled_logits.detach().to(self.rho.dtype)
        if self.device is not None:
            p_a, p_s = p_a.to(self.device), p_s.to(self.device)

        # if self.loss == "l1":
        #     # Subtract per-sample max of p_a before exp to prevent overflow on
        #     # large ImageNet logits (can be 10-50+). The shift c is a positive
        #     # constant wrt tau, so it doesn't change the argmin:
        #     #   ||exp(p_a) - exp(p_s/tau)||_1 ∝ ||exp(p_a-c) - exp(p_s/tau-c)||_1
        #     c = p_a.amax(dim=-1, keepdim=True)  # (B, 1), no grad
        #     exp_pa = (p_a - c).exp()             # fixed target; values in (0, 1]
        
        exp_pa = p_a.exp()
        self.loss_history = []
        for _ in range(self.num_steps):
            self.optimizer.zero_grad(set_to_none=True)
            tau = self.rho.exp()
            if self.loss == "l1":
                exp_ps = (p_s / tau).exp()
                loss = (exp_pa - exp_ps).abs().sum(dim=-1).mean()
            else:  # entropy
                z_ens = (p_a + p_s / tau) / 2.0
                loss = softmax_entropy(z_ens).mean()
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach())
            self.loss_history.append(self.last_loss)

        return self.temperature

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def apply(self, scaled_logits: torch.Tensor) -> torch.Tensor:
        """Return the temperature-scaled logits p_s / tau at the current tau."""
        return scaled_logits / self.temperature

    @torch.no_grad()
    def calibrated_softmax(self, scaled_logits: torch.Tensor) -> torch.Tensor:
        """Softmax of the temperature-scaled scaled-model logits, (B, C)."""
        return torch.softmax(self.apply(scaled_logits), dim=-1)


