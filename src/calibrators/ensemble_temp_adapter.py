"""
Test-time temperature adaptation for an ensemble of n models, generalising the
unidirectional discrepancy-based temperature optimisation of

    arg min_tau  L_s( p_a(x), p_s(x) / tau )                          (Eqn. 1)
    L_s(p_a, p_s) = || exp(p_a) - exp(p_s) ||_1                        (Eqn. 2)

from a single model pair (one fixed anchor + one scaled model, one temperature)
to an arbitrary ensemble of n models, learning one temperature per model.

Why this is not a mechanical generalisation
--------------------------------------------
The paper's objective is *unidirectional*: p_a is a fixed anchor and only
p_s / tau is moved. A naive symmetric pairwise loss with n free temperatures,

    sum_{i<j} || exp(z_i / tau_i) - exp(z_j / tau_j) ||_1,

is degenerate: its global minimum drives every tau_i -> inf, so z_i / tau_i -> 0,
every exp(0) = 1, and all members trivially agree at a *uniform* prediction.

To keep the unidirectional property with n temperatures, the default target_mode
aligns each member to a **stop-gradient** consensus of the others. The detached
target is the direct analogue of the fixed anchor: the gradient w.r.t. tau_i
flows only through member i, never through its target, which removes the
"inflate every tau together" gradient path. For n = 2 this reduces to the
paper's structure (each side calibrated toward the detached other).

Usage (per-batch, from the main TTA loop)
-----------------------------------------
    adapter = EnsembleTemperatureAdapter(n_models=len(models), num_steps=K)

    for batch in test_loader:
        # logits: list of n tensors (B, C), or a stacked tensor (n, B, C)
        logits = [m(batch) for m in models]
        adapter.adapt(logits)                  # K inner steps on tau, in place
        probs = adapter.predict_proba(logits)  # calibrated ensemble (B, C)
        # adapter.last_loss / adapter.temperatures are handy for W&B logging
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
import torch.nn.functional as F

LogitsInput = Union[torch.Tensor, Sequence[torch.Tensor]]


class EnsembleTemperatureAdapter:
    """Stateful per-batch temperature optimiser for an ensemble of n models.

    One scalar temperature is learned per model. Instantiate once; call
    ``adapt`` once per batch. Temperatures are stored as state so they can be
    warm-started across batches (online TTA) or reset per batch (episodic TTA)
    via ``reset_each_batch``.

    Parameters
    ----------
    n_models : int
        Number of models in the ensemble (n).
    num_steps : int
        K inner optimisation steps performed on tau for each batch.
    lr : float
        Learning rate for the temperature optimiser.
    target_mode : {"consensus", "anchor", "pairwise"}
        How the discrepancy is formed across the n members:
          - "consensus" (default): each member is pulled toward a stop-gradient
            mean of the other members' projected logits. Unidirectional and
            symmetric; the recommended generalisation.
          - "anchor": member ``anchor_idx`` provides a stop-gradient target and
            all others are aligned to it. Recovers Eqn. 1 exactly for n = 2.
            The anchor's own temperature receives no gradient (stays at init).
          - "pairwise": sum of all unordered pair discrepancies. Faithful in
            form but degenerate without regularisation (see module docstring);
            use ``reg_weight`` / ``max_temp`` if you select this.
    anchor_idx : int
        Index of the anchor member when ``target_mode == "anchor"``.
    leave_one_out : bool
        For "consensus", whether the target for member i excludes member i
        (True, default) or is the full ensemble mean (False).
    space : {"exp", "softmax"}
        Projection applied before the L1 discrepancy. "exp" is faithful to
        Eqn. 2 (raw exponentials of logits); "softmax" is the numerically
        stable normalised variant and is usually the safer choice.
    init_temp : float
        Initial temperature for every member.
    min_temp, max_temp : float
        Temperatures are clamped to this range when applied. ``max_temp`` also
        acts as a hard cap against runaway softening / collapse.
    reg_weight : float
        Optional L2 penalty on log-temperature, sum_i (log tau_i)^2, added to
        the discrepancy. Discourages drift away from tau = 1. Off by default.
    reset_each_batch : bool
        If True, temperatures (and optimiser state) are reset to init before
        each batch's K steps (episodic). If False (default), they persist and
        are warm-started across batches (online).
    optimizer_cls : type[torch.optim.Optimizer]
        Optimiser class used for the temperatures (default: Adam).
    device, dtype : optional
        Where the temperature parameters live.
    """

    def __init__(
        self,
        n_models: int,
        num_steps: int = 10,
        lr: float = 1e-2,
        target_mode: str = "consensus",
        anchor_idx: int = 0,
        leave_one_out: bool = True,
        space: str = "exp",
        init_temp: float = 1.0,
        min_temp: float = 1e-2,
        max_temp: float = 1e2,
        reg_weight: float = 0.0,
        reset_each_batch: bool = False,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: Union[str, torch.device, None] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if n_models < 2:
            raise ValueError("n_models must be >= 2.")
        if target_mode not in {"consensus", "anchor", "pairwise"}:
            raise ValueError(f"Unknown target_mode: {target_mode!r}.")
        if space not in {"exp", "softmax"}:
            raise ValueError(f"Unknown space: {space!r}.")
        if not (0 <= anchor_idx < n_models):
            raise ValueError("anchor_idx out of range.")

        self.n_models = n_models
        self.num_steps = num_steps
        self.lr = lr
        self.target_mode = target_mode
        self.anchor_idx = anchor_idx
        self.leave_one_out = leave_one_out
        self.space = space
        self.init_log_temp = float(torch.log(torch.tensor(init_temp)))
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.reg_weight = reg_weight
        self.reset_each_batch = reset_each_batch
        self.optimizer_cls = optimizer_cls
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype

        # Free parameters are log-temperatures (guarantees tau > 0).
        self.log_tau = torch.nn.Parameter(
            torch.full((n_models,), self.init_log_temp, dtype=dtype, device=self.device)
        )
        self._build_optimizer()
        self.last_loss: float = float("nan")

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #
    def _build_optimizer(self) -> None:
        self.optimizer = self.optimizer_cls([self.log_tau], lr=self.lr)

    def reset(self) -> None:
        """Reset temperatures to their initial value and clear optimiser state."""
        with torch.no_grad():
            self.log_tau.fill_(self.init_log_temp)
        self._build_optimizer()

    @property
    def temperatures(self) -> torch.Tensor:
        """Current per-model temperatures, shape (n,), detached and clamped."""
        return self.log_tau.detach().exp().clamp(self.min_temp, self.max_temp)

    # ------------------------------------------------------------------ #
    # Core computation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stack(logits: LogitsInput) -> torch.Tensor:
        """Accept a list of (B, C) tensors or a stacked (n, B, C) tensor."""
        if isinstance(logits, torch.Tensor):
            stacked = logits
        else:
            stacked = torch.stack(list(logits), dim=0)
        if stacked.dim() != 3:
            raise ValueError(
                "Expected logits as (n, B, C) tensor or a length-n list of (B, C)."
            )
        return stacked

    def _scaled(self, logits_nbc: torch.Tensor) -> torch.Tensor:
        """Apply per-model temperatures: z_i / tau_i. Returns (n, B, C)."""
        tau = self.log_tau.exp().clamp(self.min_temp, self.max_temp)  # (n,)
        return logits_nbc / tau.view(self.n_models, 1, 1)

    def _project(self, scaled_nbc: torch.Tensor) -> torch.Tensor:
        """Project scaled logits into the discrepancy space. Returns (n, B, C)."""
        if self.space == "exp":
            return torch.exp(scaled_nbc)
        return F.softmax(scaled_nbc, dim=-1)

    @staticmethod
    def _l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """L1 over the class dim (Eqn. 2), averaged over the batch -> scalar."""
        return (a - b).abs().sum(dim=-1).mean()

    def _discrepancy(self, proj: torch.Tensor) -> torch.Tensor:
        """Total discrepancy loss over the n projected members. proj: (n, B, C)."""
        n = self.n_models

        if self.target_mode == "anchor":
            target = proj[self.anchor_idx].detach()  # fixed reference (B, C)
            loss = proj.new_zeros(())
            for i in range(n):
                if i == self.anchor_idx:
                    continue
                loss = loss + self._l1(proj[i], target)
            loss = loss / max(n - 1, 1)

        elif self.target_mode == "consensus":
            loss = proj.new_zeros(())
            sum_all = proj.sum(dim=0)  # (B, C)
            for i in range(n):
                if self.leave_one_out:
                    target = (sum_all - proj[i]) / (n - 1)
                else:
                    target = sum_all / n
                # Stop-gradient on the target -> preserves unidirectionality.
                loss = loss + self._l1(proj[i], target.detach())
            loss = loss / n

        else:  # "pairwise"
            loss = proj.new_zeros(())
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    loss = loss + self._l1(proj[i], proj[j])
                    count += 1
            loss = loss / max(count, 1)

        if self.reg_weight > 0.0:
            loss = loss + self.reg_weight * (self.log_tau ** 2).sum()
        return loss

    # ------------------------------------------------------------------ #
    # Public API — called once per batch from the main TTA module
    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def adapt(self, logits: LogitsInput) -> torch.Tensor:
        """Run K optimisation steps on the temperatures for one batch.

        The incoming logits are treated as constants (detached) so only the
        temperatures are updated — no gradient reaches the underlying models.

        Returns the updated per-model temperatures, shape (n,).
        """
        if self.reset_each_batch:
            self.reset()

        z = self._stack(logits).detach().to(self.log_tau.dtype)
        if self.device is not None:
            z = z.to(self.device)

        for _ in range(self.num_steps):
            self.optimizer.zero_grad(set_to_none=True)
            proj = self._project(self._scaled(z))
            loss = self._discrepancy(proj)
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach())

        return self.temperatures

    @torch.no_grad()
    def apply(self, logits: LogitsInput) -> torch.Tensor:
        """Return temperature-scaled logits z_i / tau_i, shape (n, B, C)."""
        z = self._stack(logits).to(self.log_tau.dtype)
        if self.device is not None:
            z = z.to(self.device)
        return self._scaled(z)

    @torch.no_grad()
    def predict_proba(
        self, logits: LogitsInput, weights: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """Calibrated ensemble probabilities, shape (B, C).

        Averages each member's temperature-scaled softmax. Pass ``weights``
        (shape (n,)) for a weighted average; defaults to a uniform mean.
        """
        probs = F.softmax(self.apply(logits), dim=-1)  # (n, B, C)
        if weights is None:
            return probs.mean(dim=0)
        w = weights.to(probs.dtype).to(probs.device).view(self.n_models, 1, 1)
        return (probs * w).sum(dim=0) / w.sum()