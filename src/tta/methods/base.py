"""
base.py
=======
Shared parent class for test-time-adaptation methods, plus a name -> class
registry.

A TTAMethod owns EVERYTHING that adapts the duo's models: which parameters
are trainable, the optimizer(s), the per-batch adaptation loss, the update
step, and restoring all of that on reset(). DynamicDuo only decides WHICH
models adapt and on WHICH logits (its `mode`: "duo" = the joint calibrated
output, "indep" = each model's own logits) -- never HOW.

One instance serves both models (the duo loss is joint, so it can't be split
per model), and per-model state is keyed by side ("large"/"small").

To add a new method: create a TTAMethod subclass in its own file under
src/tta/methods/, set a class-level `name`, decorate it with @register, and
import that module from src/tta/methods/__init__.py. It then appears
automatically in TTA_METHODS and is selectable via --tta_method, the duo
YAML's `TTA:` block, or setup_duo(tta_method=...).

Non-adapting sides are NOT the method's business: setup() always puts them in
the shared frozen configuration (configure_model_frozen -- live batch stats,
no grad), so `no_adapt` logits are identical whichever method is selected and
every frozen-logit cache stays valid across methods.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Iterable, TypeVar

import torch
import torch.nn as nn

from src.tta.tent import configure_model_frozen

logger = logging.getLogger(__name__)

DEFAULT_TTA_METHOD = "tent"

SIDES = ("large", "small")

_REGISTRY: dict[str, type["TTAMethod"]] = {}

_TTAT = TypeVar("_TTAT", bound="TTAMethod")


def register(cls: type[_TTAT]) -> type[_TTAT]:
    _REGISTRY[cls.name] = cls
    return cls


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


def build_tta_method(name: str, **kwargs) -> "TTAMethod":
    """A freshly-constructed (not yet set up) instance of the named method."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown TTA method {name!r}. Must be one of {registered_names()}.")
    return _REGISTRY[name](**kwargs)


def resolve_tta_spec(cfg: dict, tta_method: str | None = None,
                     tta_kwargs: dict | None = None) -> tuple[str, dict]:
    """Which (method name, kwargs) a run uses.

    Precedence for the NAME: explicit argument (CLI / run_cfg) > the duo
    YAML's `TTA: {METHOD: ...}` block > DEFAULT_TTA_METHOD. The YAML block's
    KWARGS only apply when the resolved name IS the YAML's own METHOD (they
    belong to that method; handing them to a different one would be a
    guaranteed TypeError or, worse, a silent mismatch). Explicit kwargs are
    merged on top.
    """
    block = cfg.get("TTA") or {}
    yaml_name = block.get("METHOD")
    name = tta_method or yaml_name or DEFAULT_TTA_METHOD
    kwargs = dict(block.get("KWARGS") or {}) if name == yaml_name else {}
    kwargs.update(tta_kwargs or {})
    return name, kwargs


class TTAMethod(ABC):
    """One test-time-adaptation method for an asymmetric duo.

    Lifecycle: build_tta_method(name, **hparams) -> setup(...) once -> per
    batch loss() then update() on each adapting step -> reset() at every
    corruption boundary.
    """

    name: ClassVar[str] = ""

    def __init__(self):
        self.cfg: dict | None = None
        self.models: dict[str, nn.Module] = {}
        self.adapting: set[str] = set()

    # --- setup ------------------------------------------------------------ #
    def setup(self, large: nn.Module, small: nn.Module, cfg: dict,
              adapt_large: bool, adapt_small: bool) -> None:
        """Configure both models. Adapting sides go through the method's own
        _setup_side(); the rest get the shared frozen configuration."""
        self.cfg = cfg
        self.models = {"large": large, "small": small}
        self.adapting = {s for s, do in (("large", adapt_large), ("small", adapt_small)) if do}
        for side, model in self.models.items():
            side_cfg = cfg[side.upper()]
            if side in self.adapting:
                logger.info(f"Configuring {side} model for TTA method '{self.name}' "
                            f"with norm={side_cfg['NORM']}")
                self._setup_side(side, model, side_cfg)
            else:
                logger.info(f"Configuring {side} model (frozen, batch stats) with norm={side_cfg['NORM']}")
                configure_model_frozen(model, side_cfg["NORM"])

    @abstractmethod
    def _setup_side(self, side: str, model: nn.Module, side_cfg: dict) -> None:
        """Make `model` adaptable: set train/grad state, pick trainable
        params, build whatever optimizer/state the method needs, and snapshot
        everything reset() must restore. side_cfg is cfg['LARGE'] or
        cfg['SMALL']."""

    # --- per batch -------------------------------------------------------- #
    @abstractmethod
    def loss(self, logits: torch.Tensor, signal: str) -> torch.Tensor:
        """Differentiable scalar adaptation loss on `logits`. `signal` is
        "duo" (logits = the joint calibrated output) or "large"/"small"
        (logits = that model's own output) -- also the key a stateful method
        must use to keep its per-signal state apart."""

    @abstractmethod
    def update(self, loss: torch.Tensor, sides: Iterable[str]) -> None:
        """Backpropagate `loss` and apply one update to each of `sides`
        (always a subset of the adapting sides)."""

    # --- lifecycle -------------------------------------------------------- #
    def reset(self) -> None:
        """Restore every adapting side to its pre-adaptation state (and any
        method-level state -- override and call super() if there is some)."""
        for side in sorted(self.adapting):
            logger.info(f"Resetting {side} model to pre-adaptation state")
            self._reset_side(side)

    @abstractmethod
    def _reset_side(self, side: str) -> None:
        """Restore one adapting side's model/optimizer/method state."""

    def describe(self) -> dict:
        """Flat, wandb-config-shaped description of this method's setup
        (called after setup())."""
        return {"tta/method": self.name}


if __name__ == "__main__":
    # Self-test: spec resolution precedence + registry errors. `python -m` runs
    # this file as `__main__`, a SECOND copy of the module whose registry is empty
    # (the method modules registered themselves into `src.tta.methods.base`), so
    # test the package's copy.
    from src.tta.methods.base import build_tta_method, registered_names, resolve_tta_spec

    assert "tent" in registered_names()

    assert resolve_tta_spec({}) == ("tent", {})
    yaml_cfg = {"TTA": {"METHOD": "tent", "KWARGS": {"a": 1}}}
    assert resolve_tta_spec(yaml_cfg) == ("tent", {"a": 1}), "YAML block used when nothing explicit"
    assert resolve_tta_spec(yaml_cfg, None, {"b": 2}) == ("tent", {"a": 1, "b": 2}), "explicit kwargs merge on top"
    assert resolve_tta_spec(yaml_cfg, None, {"a": 9}) == ("tent", {"a": 9}), "explicit kwargs win"
    # YAML kwargs belong to the YAML's method: overriding the method drops them.
    assert resolve_tta_spec(yaml_cfg, "other", {"b": 2}) == ("other", {"b": 2})
    assert resolve_tta_spec({"TTA": None}) == ("tent", {})

    try:
        build_tta_method("definitely_not_a_method")
        raise AssertionError("expected ValueError for an unregistered method")
    except ValueError as e:
        assert "tent" in str(e), "error must list the valid names"
    assert build_tta_method("tent").name == "tent"
    try:
        build_tta_method("tent", bogus=1)
        raise AssertionError("expected TypeError for an unknown hyperparameter")
    except TypeError:
        pass

    print("tta methods base self-test passed")
