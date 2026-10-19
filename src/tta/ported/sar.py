"""
SAR (Sharpness-Aware and Reliable entropy minimization, Niu et al., ICLR 2023) for one member.

Ported from third_party/SAR/sar.py (SAR repository, commit 20f6e24b17525f34503510afccedc0629b67b7c4).

    Copyright (c) niushuaicheng 2023, All rights reserved.  BSD 3-Clause License.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met: redistributions of source code must retain the above copyright notice, this list of
    conditions and the following disclaimer; redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation and/or other materials provided with the
    distribution; neither the name of the copyright holder nor the names of its contributors may be used to endorse or
    promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

What differs from the official code (everything else is its code, step for step, quirks included):
  * one member as a class (`SarMember`) with `reset()` / `step(x)` instead of the `SAR(nn.Module)` wrapper; `recovery=False`
    skips the model-recovery reset (the setting for SAR as a duo base algorithm; True is the official behavior, including that
    the moving average is restored AFTER the reset, so a recovered model keeps its low average);
  * the margin is `margin_frac * ln(K)` with K the number of classes the loss sees (the official 0.4 ln 1000 on ImageNet);
  * `collect_params` skips the top layers by the official name patterns (`layer4`, `blocks.9-11`, the final `norm`), which
    are timm's names; torchvision's ViT-B/16 names its blocks `encoder.layers.encoder_layer_N` and its final norm
    `encoder.ln`, so those are skipped too, i.e. the same top three blocks and final norm, not nothing;
  * `steps` is 1 and the progress print on recovery is dropped.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from src.tta.ported.sam import SAM

_TORCHVISION_VIT_TOP = re.compile(r"encoder_layer_(9|10|11)(\.|$)")


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def update_ema(ema, new_data, decay: float = 0.9):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return decay * ema + (1 - decay) * new_data


def skip_top_layers(nm: str) -> bool:
    """The official skip rules, plus torchvision's names for the same layers of a ViT-B/16."""
    return ('layer4' in nm or 'blocks.9' in nm or 'blocks.10' in nm or 'blocks.11' in nm or 'norm.' in nm
            or nm in ['norm'] or bool(_TORCHVISION_VIT_TOP.search(nm)) or nm.endswith("encoder.ln"))


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers, except the top layers' (official SAR)."""
    params, names = [], []
    for nm, m in model.named_modules():
        if skip_top_layers(nm):
            continue
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


def configure_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


class SarMember:
    name = "sar"

    def __init__(self, model: nn.Module, optim: dict, *, recovery: bool = False, rho: float = 0.05,
                 margin_frac: float = 0.4, reset_constant_em: float = 0.2, ema_decay: float = 0.9):
        if optim["METHOD"] != "SGD":
            raise ValueError(f"SAR runs SAM on SGD, got OPTIM METHOD {optim['METHOD']!r}")
        self.model = configure_model(model)
        params, self.param_names = collect_params(self.model)
        if not params:
            raise ValueError("SAR found no parameters to adapt: check the model's normalization layers")
        self.optimizer = SAM(params, torch.optim.SGD, rho=rho, lr=optim["LR"], momentum=optim.get("MOMENTUM", 0.0))
        self.recovery, self.margin_frac = recovery, margin_frac
        self.reset_constant_em, self.ema_decay = reset_constant_em, ema_decay
        self.ema = None                       # moving average of the second-pass entropy: the recovery criterion
        self.n_resets = 0
        self.model_state, self.optimizer_state = deepcopy(self.model.state_dict()), deepcopy(self.optimizer.state_dict())

    def reset(self) -> None:
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.ema = None

    @torch.enable_grad()
    def step(self, x: torch.Tensor, class_mask=None) -> torch.Tensor:
        sel = (lambda z: z[:, class_mask]) if class_mask is not None else (lambda z: z)
        self.optimizer.zero_grad()
        outputs = self.model(x)
        k = sel(outputs).shape[1]
        margin = self.margin_frac * math.log(k)
        entropys = softmax_entropy(sel(outputs))
        filter_ids_1 = torch.where(entropys < margin)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()

        self.optimizer.first_step(zero_grad=True)       # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
        entropys2 = softmax_entropy(sel(self.model(x)))
        entropys2 = entropys2[filter_ids_1]             # second time forward
        filter_ids_2 = torch.where(entropys2 < margin)  # filter again: the weights have changed to \Theta + \hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            self.ema = update_ema(self.ema, loss_second.item(), self.ema_decay)   # moving average for model recovery

        loss_second.backward()                          # gradients at \Theta + \hat{\epsilon(\Theta)} update the weights
        self.optimizer.second_step(zero_grad=True)

        if self.recovery and self.ema is not None and self.ema < self.reset_constant_em:
            ema = self.ema
            self.reset()                                # the official forward(): reset, THEN `self.ema = ema`
            self.ema = ema
            self.n_resets += 1
        return outputs.detach()

    def describe(self) -> dict:
        return {"tta/method": "sar", "tta/recovery": self.recovery, "tta/rho": self.optimizer.defaults["rho"],
                "tta/margin_frac": self.margin_frac, "tta/n_params": len(self.param_names)}


if __name__ == "__main__":
    import importlib.util
    from pathlib import Path

    import torchvision

    from src.utils.config import REPO_ROOT

    def load_official(rel: str):
        p = REPO_ROOT / "third_party" / rel
        spec = importlib.util.spec_from_file_location(f"official_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    off_sar, off_sam = load_official("SAR/sar.py"), load_official("SAR/sam.py")
    OPT = {"METHOD": "SGD", "STEPS": 1, "LR": 0.05, "MOMENTUM": 0.9}

    # ---- 1. which parameters: the official rules on timm's names, the same layers on torchvision's -------------------
    import timm
    tv_vit = nn.Sequential(nn.Identity(), torchvision.models.vit_b_16(weights=None))      # wrapped like the real members
    timm_vit = timm.create_model("vit_base_patch16_224", pretrained=False)
    p_tv, n_tv = collect_params(tv_vit)
    p_timm, n_timm = off_sar.collect_params(timm_vit)
    assert len(p_tv) == len(p_timm) == 36, (len(p_tv), len(p_timm))                     # 9 blocks x 2 LayerNorms x (weight, bias)
    assert not any(re.search(r"encoder_layer_(9|10|11)\b|encoder\.ln", n) for n in n_tv)
    assert any("encoder_layer_8.ln_1" in n for n in n_tv) and any("encoder_layer_0.ln_1" in n for n in n_tv)
    assert n_timm[-1].startswith("blocks.8") and not any(n.startswith(("blocks.9", "blocks.10", "blocks.11", "norm.")) for n in n_timm)
    tv_rn = nn.Sequential(nn.Identity(), torchvision.models.resnet50(weights=None))
    p_rn, n_rn = collect_params(tv_rn)
    p_rn_off, _ = off_sar.collect_params(torchvision.models.resnet50(weights=None))
    assert len(p_rn) == len(p_rn_off) > 0 and not any("layer4" in n for n in n_rn)

    # ---- 2. the step, against the official class, step for step ---------------------------------------------------------
    K = 10

    def tiny(scale: float):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(8, 16), nn.LayerNorm(16), nn.ReLU(), nn.Linear(16, K))
        with torch.no_grad():
            m[3].weight.mul_(scale)
        return m

    def run(scale: float, recovery: bool, steps: int = 14):
        m_off = off_sar.configure_model(tiny(scale))
        params, _ = off_sar.collect_params(m_off)
        official = off_sar.SAR(m_off, off_sam.SAM(params, torch.optim.SGD, lr=OPT["LR"], momentum=0.9),
                               margin_e0=0.4 * math.log(K), reset_constant_em=0.2)
        mine = SarMember(tiny(scale), OPT, recovery=recovery)
        g = torch.Generator().manual_seed(1)
        diffs = []
        for _ in range(steps):
            x = torch.randn(32, 8, generator=g)
            diffs.append((official(x), mine.step(x)))
        return official, mine, diffs

    for scale in (6.0, 40.0):           # moderately sure (updates, no recovery) and very sure (the average falls under 0.2: recovery)
        official, mine, diffs = run(scale, recovery=True)
        for a, b in diffs:
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
        for pa, pb in zip(official.model.parameters(), mine.model.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=1e-6)
        moved = [not torch.equal(a, b) for a, b in zip(tiny(scale).parameters(), mine.model.parameters())]
        if scale > 10:
            assert mine.n_resets > 0, "the recovery path must have been exercised"
        else:
            assert mine.n_resets == 0 and any(moved), "no recovery here, and the model must really be updating"
    # recovery off never resets, so on the confident model it ends where the official (which keeps resetting) does not
    official, mine, _ = run(40.0, recovery=False)
    assert mine.n_resets == 0 and not all(torch.equal(pa, pb) for pa, pb in zip(official.model.parameters(), mine.model.parameters()))

    # ---- 3. reset, the scored logits, the class mask ---------------------------------------------------------------------
    m = SarMember(tiny(6.0), OPT)
    w0 = [p.detach().clone() for p in m.model.parameters()]
    x = torch.randn(32, 8)
    z = m.step(x)
    assert z.shape == (32, K) and not z.requires_grad
    assert torch.equal(z, tiny(6.0)(x)), "the scored logits are the first pass, before the update"
    for _ in range(3):
        m.step(torch.randn(32, 8))
    assert any(not torch.equal(a, b) for a, b in zip(w0, m.model.parameters())) and m.ema is not None
    m.reset()
    assert all(torch.equal(a, b) for a, b in zip(w0, m.model.parameters())) and m.ema is None
    mm = SarMember(tiny(6.0), OPT)
    assert mm.step(x, class_mask=[0, 1, 2, 3]).shape == (32, K), "scored logits stay 1000-way (here 10); only the loss is masked"
    try:
        SarMember(tiny(6.0), {"METHOD": "Adam", "LR": 1e-3})
        raise AssertionError("SAR needs SGD")
    except ValueError:
        pass
    print("SAR (ported) self-test passed")
