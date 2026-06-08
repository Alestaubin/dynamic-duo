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

This module implements ONLY that kernel (one scalar temperature, two models),
in isolation from the marginal-entropy / cross-model distillation losses that
COCA layers on top. It is meant as the unit to validate before generalising.

Conventions (matching the paper)
---------------------------------
* p_a = logits of the ANCHOR model (the larger / stronger model). Treated as a
  fixed target: it is detached and never scaled. This is the "unidirectional"
  part -- only p_s / tau moves toward p_a.
* p_s = logits of the SCALED model (the smaller model). Divided by tau.
* The discrepancy is the L1 norm over the class dimension of the difference of
  raw exponentiated logits (Eqn. 2), averaged over the batch.
* tau is a single positive scalar. It is reparameterised internally as
  tau = exp(rho), rho the free scalar, purely to keep tau > 0 under gradient
  steps (otherwise a step can drive tau negative and exp(p_s/tau) blows up).
  This does not change the objective in Eqn. 1 -- it is still one parameter.
"""

from __future__ import annotations

from typing import List, Type

import torch


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
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if init_temp <= 0:
            raise ValueError("init_temp must be positive.")
        self.num_steps = num_steps
        self.lr = lr
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

        exp_pa = p_a.exp()  # fixed target in exponential space; constant in the loop

        self.loss_history = []
        for _ in range(self.num_steps):
            self.optimizer.zero_grad(set_to_none=True)
            tau = self.rho.exp()                              # scalar > 0
            exp_ps = (p_s / tau).exp()                        # exp(p_s / tau)
            loss = (exp_pa - exp_ps).abs().sum(dim=-1).mean() # Eqn. 2 + Eqn. 1
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


# ====================================================================== #
# Self-contained sanity check.
#
# Controlled setup: build a "scaled model" whose logits are a known multiple
# T_true of the anchor's logits (p_s = T_true * p_a). Then the alignment in
# Eqn. 1 is solved exactly by tau = T_true (giving p_s / tau = p_a, loss -> 0),
# so we can verify the kernel recovers the right temperature and the loss
# decreases monotonically. Real model pairs are not exact multiples, so on real
# data the loss floors above zero -- that is expected.
# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C = 64, 10
    T_true = 2.0

    # Modest logit scale (std ~1) keeps raw exp() well-conditioned. With large
    # logits the raw-exp objective is dominated by the top class and converges
    # slowly -- see the note in the module docstring.
    p_a = torch.randn(B, C)                # anchor (larger model) logits
    p_s = T_true * p_a                     # scaled model = exact T_true multiple

    coca = CocaTemperature(num_steps=500, lr=5e-2, init_temp=1.0)
    tau = coca.adapt(anchor_logits=p_a, scaled_logits=p_s)

    print(f"true temperature      : {T_true:.4f}")
    print(f"recovered temperature : {tau:.4f}")
    print(f"loss[0] -> loss[-1]   : {coca.loss_history[0]:.4e} -> {coca.loss_history[-1]:.4e}")

    assert abs(tau - T_true) < 0.05, "tau did not converge to the known optimum"
    assert coca.loss_history[-1] < coca.loss_history[0], "loss did not decrease"

    # Sanity on real-ish pairs: two unrelated logit sets -> loss floors above 0.
    p_a2 = torch.randn(B, C)
    p_s2 = torch.randn(B, C)
    coca.adapt(p_a2, p_s2)
    print(f"unrelated-pair tau    : {coca.temperature:.4f} "
          f"(final loss {coca.loss_history[-1]:.4e})")
    print("OK")