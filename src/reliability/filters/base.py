"""
base.py
=======
Shared interface for Section-4 temporal filters: denoise a stream of
per-batch calibrated (logit-space) reliability scores into a slowly-drifting
latent-reliability estimate.

Filters are proxy-agnostic: they consume whatever scalar stream Section 3's
calibration + logit-transform (eq. 8) produces and know nothing about how it
was computed. `reset()` is called at each corruption boundary, since the
model itself resets to source there too — the filtered estimate should
return to its prior at the same points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ScoreFilter(ABC):
    @abstractmethod
    def update(self, x: float) -> float:
        """Assimilate one new observation; return the filtered value."""

    @abstractmethod
    def reset(self) -> None:
        """Restore to the prior/initial state (call at each corruption boundary)."""
