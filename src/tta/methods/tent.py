"""
tent.py
=======
TENT (Wang et al., 2021) as a TTAMethod: entropy minimization over the
normalization layers' affine parameters. A thin adapter over the helpers in
src/tta/tent.py (which also back the single-model run_tent.py), so the duo's
TENT behavior and the single-model one can never drift apart.

The loss is the mean softmax entropy of whatever logits the duo mode feeds in
(the joint calibrated output for *_duo modes, each model's own logits for
*_indep modes). Optimizer/LR/NORM come from the per-side LARGE/SMALL config
blocks; TENT itself has no extra hyperparameters.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn

from src.tta.methods.base import TTAMethod, register
from src.tta.tent import (
    collect_params, configure_model, copy_model_and_optimizer,
    load_model_and_optimizer, setup_optimizer, softmax_entropy,
)

logger = logging.getLogger(__name__)


@register
class TentMethod(TTAMethod):
    name = "tent"

    def __init__(self):
        super().__init__()
        self._optimizers: dict[str, torch.optim.Optimizer] = {}
        self._snapshots: dict[str, tuple[dict, dict]] = {}

    def _setup_side(self, side: str, model: nn.Module, side_cfg: dict) -> None:
        model = configure_model(model, side_cfg["NORM"])
        params, param_names = collect_params(model, side_cfg["NORM"])
        if not params:
            raise ValueError("No parameters found for adaptation. Check if model has Norm layers.")

        optimizer = setup_optimizer(params, side_cfg["OPTIM"])
        logger.info("model for adaptation: %s", model)
        logger.info("params for adaptation: %s", param_names)
        logger.info("optimizer for adaptation: %s", optimizer)

        self._optimizers[side] = optimizer
        self._snapshots[side] = copy_model_and_optimizer(model, optimizer)

    def loss(self, logits: torch.Tensor, signal: str) -> torch.Tensor:
        return softmax_entropy(logits).mean(0)

    def update(self, loss: torch.Tensor, sides: Iterable[str]) -> None:
        sides = list(sides)
        loss.backward()
        for side in sides:
            self._optimizers[side].step()
        for side in sides:
            self._optimizers[side].zero_grad(set_to_none=True)

    def _reset_side(self, side: str) -> None:
        model_state, optimizer_state = self._snapshots[side]
        load_model_and_optimizer(self.models[side], self._optimizers[side],
                                 model_state, optimizer_state)

    def describe(self) -> dict:
        d = super().describe()
        for side in ("large", "small"):
            side_cfg = self.cfg[side.upper()]
            d[f"{side}/norm"] = side_cfg["NORM"]
            d[f"{side}/lr"] = side_cfg["OPTIM"]["LR"]
            d[f"{side}/optim"] = side_cfg["OPTIM"]["METHOD"]
        return d


if __name__ == "__main__":
    # Self-test on tiny CPU models: setup/loss/update/reset semantics.
    torch.manual_seed(0)

    def _tiny():
        return nn.Sequential(nn.Linear(4, 6), nn.LayerNorm(6), nn.Linear(6, 3))

    def _side_cfg():
        return {"NAME": "tiny", "NORM": "LN",
                "OPTIM": {"METHOD": "Adam", "LR": 1e-2, "BETA": 0.9, "WD": 0.0}}

    cfg = {"LARGE": _side_cfg(), "SMALL": _side_cfg()}
    large, small = _tiny(), _tiny()
    large_w0 = large[1].weight.detach().clone()
    small_w0 = small[1].weight.detach().clone()

    m = TentMethod()
    m.setup(large, small, cfg, adapt_large=True, adapt_small=False)
    assert m.adapting == {"large"}
    assert large[1].weight.requires_grad and not small[1].weight.requires_grad, \
        "only the adapting side may have trainable norm params"

    x = torch.randn(8, 4)
    z = large(x)
    loss = m.loss(z, "large")
    assert torch.allclose(loss, softmax_entropy(z).mean(0))
    m.update(loss, ["large"])
    assert not torch.equal(large[1].weight.detach(), large_w0), "update() must move the params"
    assert large[1].weight.grad is None, "update() must clear grads"
    assert torch.equal(small[1].weight.detach(), small_w0), "frozen side must not move"

    m.reset()
    assert torch.equal(large[1].weight.detach(), large_w0), "reset() must restore the snapshot"
    assert m.describe()["tta/method"] == "tent" and m.describe()["large/lr"] == 1e-2

    # An adapting side with no norm layers of the requested type must fail loudly.
    bad = TentMethod()
    try:
        bad.setup(nn.Linear(4, 3), _tiny(), {"LARGE": {**_side_cfg(), "NORM": "BN"}, "SMALL": _side_cfg()},
                  adapt_large=True, adapt_small=False)
        raise AssertionError("expected ValueError for a model with no params to adapt")
    except ValueError:
        pass

    print("TentMethod self-test passed")
