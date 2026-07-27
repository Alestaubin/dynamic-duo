"""
base.py
=======
Shared parent class for reliability proxies, plus a name -> class registry.

A proxy scores how reliable a model's prediction is on one batch (higher =
more reliable). Some proxies need no source-data fitting (nuclear_norm);
others fit scalar or tensor state on clean source data first (atc,
prototype) via fit_source().

To add a new proxy: create a Proxy subclass in its own file under
src/proxies/, decorate it with @register, and import that module from
proxies.py. It then appears automatically in PROXY_KINDS and in every
ProxyStats via build_all().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import torch

_REGISTRY: dict[str, type["Proxy"]] = {}

_ProxyT = TypeVar("_ProxyT", bound="Proxy")


def register(cls: type[_ProxyT]) -> type[_ProxyT]:
    _REGISTRY[cls.name] = cls
    return cls


def build_all() -> dict[str, "Proxy"]:
    """One freshly-constructed instance of every registered proxy."""
    return {name: cls() for name, cls in _REGISTRY.items()}


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


class Proxy(ABC):
    """One reliability proxy for a single model.

    Subclasses set a class-level `name` and are decorated with @register.
    """

    name: str = ""

    def fit_source(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
    ) -> None:
        """Fit any source-data-derived state. No-op for stateless proxies."""

    @property
    def is_fitted(self) -> bool:
        """Whether this proxy has enough state to score. True if stateless."""
        return True

    @abstractmethod
    def score(self, logits: torch.Tensor, features: torch.Tensor | None) -> float:
        """Reliability score for one batch; higher = more reliable."""

    def state_dict(self) -> dict:
        """Source-fitted state to persist. Empty for stateless proxies."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore state saved by state_dict()."""
