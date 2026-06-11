import torch

def logit_pnorm(z: torch.Tensor, p: float, tau: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Logit p-normalization (centralized) https://arxiv.org/pdf/2305.15508

    Args:
        z:   logits, shape (..., C)
        p:   p-norm order
        tau: temperature
        eps: floor for the norm to avoid division by zero

    Returns:
        Normalized logits, same shape as z.
    """
    centered = z - z.mean(dim=-1, keepdim=True)
    norm = centered.abs().pow(p).sum(dim=-1, keepdim=True).pow(1.0 / p)
    return centered / (tau * norm.clamp(min=eps))

def combine_logits(z_l: torch.Tensor, z_s: torch.Tensor, tau_l: float, tau_s: float) -> torch.Tensor:
    """Combine two sets of logits with per-model temperatures."""
    return (z_l / tau_l + z_s / tau_s) / 2.0