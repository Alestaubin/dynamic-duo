"""
eata.py
=======
EATA-style filtered, weighted entropy minimization (Niu et al., 2022) as a
TTAMethod -- the first non-TENT method, here to prove the abstraction: it is a
single forward/backward pass per step, so it needs nothing from DynamicDuo
beyond what TENT already uses.

Same trainable parameters and optimizer as TENT (norm-layer affine params,
LARGE/SMALL OPTIM blocks), so it subclasses TentMethod for setup/update/reset
and only replaces the loss. Per sample, with entropy E(x) and C classes:

  * reliability filter + weight: keep E(x) < e_margin (= e_margin_frac * ln C),
    weighted by 1 / exp(E(x) - e_margin) -- confident samples count more, the
    near-uniform ones that make entropy minimization collapse are dropped.
  * diversity filter: keep only samples whose softmax has |cos| < d_margin
    against a running average of the previously kept predictions, so
    near-duplicate (already-seen) predictions don't dominate the update.
  * loss = mean over kept samples of (weight * E(x)); zero (with a live graph,
    so update() still works) if nothing survives the filters.

NOT included: EATA's Fisher anti-forgetting regularizer, which needs a labelled
source subset to estimate. Per-signal state (the running average of kept
softmaxes) is keyed by `signal` ("duo"/"large"/"small") because in *_indep
modes each model adapts on its own logits and must not see the other's history;
it is cleared by reset() at every corruption boundary.

loss() is stateful: call it exactly once per update (DynamicDuo does).

Hyperparameters (--tta_kwargs / the duo YAML's TTA.KWARGS):
  e_margin_frac  entropy threshold as a fraction of ln(C)    (default 0.4, as in EATA)
  d_margin       cosine-similarity threshold for diversity   (default 0.05, ImageNet value in EATA;
                 pass a value > 1 to disable the diversity filter)
  ma_decay       decay of the running average of kept probs  (default 0.9, i.e. EATA's 0.1 update rate)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tta.methods.base import register
from src.tta.methods.tent import TentMethod
from src.tta.tent import softmax_entropy


@register
class EataMethod(TentMethod):
    name = "eata"

    def __init__(self, e_margin_frac: float = 0.4, d_margin: float = 0.05, ma_decay: float = 0.9):
        super().__init__()
        assert e_margin_frac > 0, "e_margin_frac must be positive"
        assert 0.0 <= ma_decay < 1.0, "ma_decay must be in [0, 1)"
        self.e_margin_frac = float(e_margin_frac)
        self.d_margin = float(d_margin)
        self.ma_decay = float(ma_decay)
        self._ma_probs: dict[str, torch.Tensor] = {}
        self.last_stats: dict[str, dict[str, float]] = {}

    def loss(self, logits: torch.Tensor, signal: str) -> torch.Tensor:
        ent = softmax_entropy(logits)                                    # (B,)
        e_margin = self.e_margin_frac * math.log(logits.shape[1])

        keep = ent.detach() < e_margin
        probs = logits.detach().softmax(1)

        prev = self._ma_probs.get(signal)
        if prev is not None:
            keep &= F.cosine_similarity(prev.unsqueeze(0), probs, dim=1).abs() < self.d_margin

        n_keep = int(keep.sum())
        self.last_stats[signal] = {"kept_frac": n_keep / ent.shape[0]}
        if n_keep == 0:
            return ent.sum() * 0.0                                       # live graph, zero grad

        batch_mean = probs[keep].mean(0)
        self._ma_probs[signal] = (batch_mean if prev is None
                                  else self.ma_decay * prev + (1.0 - self.ma_decay) * batch_mean)

        weight = torch.exp(-(ent.detach() - e_margin))
        return (ent * weight)[keep].mean()

    def reset(self) -> None:
        super().reset()
        self._ma_probs.clear()
        self.last_stats.clear()

    def describe(self) -> dict:
        d = super().describe()
        d.update({"tta/e_margin_frac": self.e_margin_frac, "tta/d_margin": self.d_margin,
                  "tta/ma_decay": self.ma_decay})
        return d


if __name__ == "__main__":
    # Self-test on tiny CPU models: filter/weight maths, per-signal state, reset, end-to-end update.
    torch.manual_seed(0)
    C = 3
    e0 = 0.4 * math.log(C)

    def logits_of(rows):
        return torch.tensor(rows, dtype=torch.float32, requires_grad=True)

    sure0 = [10.0, 0.0, 0.0]       # confident in class 0: entropy ~ 1e-3 << e0
    sure2 = [0.0, 0.0, 10.0]       # confident in class 2: orthogonal to sure0
    flat = [0.0, 0.0, 0.0]         # uniform: entropy ln C > e0 -> reliability-filtered

    m = EataMethod()

    # 1. reliability filter + weight: only the confident row survives, weighted by exp(e0 - E).
    z = logits_of([sure0, flat])
    ent = softmax_entropy(z)
    assert ent[0] < e0 < ent[1]
    loss = m.loss(z, "duo")
    expected = ent[0] * torch.exp(e0 - ent[0])
    assert torch.allclose(loss, expected.detach()), (loss, expected)
    assert m.last_stats["duo"]["kept_frac"] == 0.5
    loss.backward()
    assert z.grad is not None and z.grad[1].abs().sum() == 0, "filtered sample must get no gradient"

    # 2. diversity filter: the running average is now ~sure0, so another sure0 is dropped,
    #    but sure2 (cos ~ 0) is kept.
    z = logits_of([sure0])
    loss = m.loss(z, "duo")
    assert loss.item() == 0.0 and m.last_stats["duo"]["kept_frac"] == 0.0
    loss.backward()                                   # all-filtered loss must still be backprop-able
    assert z.grad is not None and z.grad.abs().sum() == 0
    z = logits_of([sure2])
    assert m.loss(z, "duo").item() > 0.0 and m.last_stats["duo"]["kept_frac"] == 1.0

    # 3. state is per signal: "small" has seen nothing, so sure0 is kept there.
    z = logits_of([sure0])
    assert m.loss(z, "small").item() > 0.0

    # 4. reset() clears all running state.
    m.reset()                                         # no sides set up yet: only the method-level state
    assert not m._ma_probs

    # 5. disabling the diversity filter (d_margin > 1) keeps repeats.
    m_nodiv = EataMethod(d_margin=2.0)
    m_nodiv.loss(logits_of([sure0]), "duo")
    assert m_nodiv.loss(logits_of([sure0]), "duo").item() > 0.0

    # 6. end to end through TentMethod's setup/update/reset on tiny models.
    def _tiny():
        return nn.Sequential(nn.Linear(4, 6), nn.LayerNorm(6), nn.Linear(6, C))

    def _side_cfg():
        return {"NAME": "tiny", "NORM": "LN",
                "OPTIM": {"METHOD": "Adam", "LR": 1e-2, "BETA": 0.9, "WD": 0.0}}

    cfg = {"LARGE": _side_cfg(), "SMALL": _side_cfg()}
    large, small = _tiny(), _tiny()
    with torch.no_grad():                              # make the tiny model confident so samples pass the filter
        large[2].weight.mul_(50.0)
    w0 = large[1].weight.detach().clone()

    m = EataMethod()
    m.setup(large, small, cfg, adapt_large=True, adapt_small=False)
    x = torch.randn(16, 4)
    z = large(x)
    loss = m.loss(z, "large")
    assert m.last_stats["large"]["kept_frac"] > 0.0, "tiny confident model should pass the reliability filter"
    m.update(loss, ["large"])
    assert not torch.equal(large[1].weight.detach(), w0), "update() must move the params"
    m.update(m.loss(large(x), "large"), ["large"])    # second step: diversity filter may drop everything; must not crash
    m.reset()
    assert torch.equal(large[1].weight.detach(), w0) and not m._ma_probs
    d = m.describe()
    assert d["tta/method"] == "eata" and d["tta/d_margin"] == 0.05 and d["large/lr"] == 1e-2

    # 7. the method is registered under its name and takes its hyperparameters.
    from src.tta.methods import build_tta_method, registered_names
    assert "eata" in registered_names()
    assert build_tta_method("eata", d_margin=0.1).d_margin == 0.1

    print("EataMethod self-test passed")
