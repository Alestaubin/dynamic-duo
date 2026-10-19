"""
ETA / EATA (Efficient Anti-forgetting Test-time Adaptation, Niu et al., ICML 2022) for one member.

Ported from third_party/EATA/eata.py and the Fisher computation of third_party/EATA/main.py (EATA repository, commit
f739b3668cc7617e9b9f1979c1a358497a3472c3). MIT License, Copyright (c) 2023 Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang,
Yaofo Chen, Shijian Zheng, Peilin Zhao, Mingkui Tan. Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, subject to the inclusion of this copyright notice and this permission notice in all copies or substantial
portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

ETA is EATA without the Fisher regularizer (`fishers=None`, or `fisher_alpha=0`); EATA adds
`fisher_alpha * sum(F * (theta - theta_source)^2)` to the loss, with F the diagonal Fisher information of the SOURCE model
estimated on clean source images (`compute_fishers`).

What differs from the official code (everything else is its code, step for step, quirks included):
  * one member as a class with `reset()` / `step(x)`; the parameters are the affine parameters of ALL norm layers (BatchNorm,
    LayerNorm, GroupNorm), as for Tent here, where the official code collects BatchNorm only;
  * the moving average of the kept predictions is cleared by `reset()`: the official class resets only the weights and the
    optimizer, so its average would carry over from one corruption to the next in its each-shift-reset experiments, while
    here every stream starts from a clean state (as the repo's own ETA, src/tta/methods/eata.py, does);
  * the margin is `e_margin_frac * ln(K)` with K the number of classes the loss sees (the official 0.4 ln 1000);
  * `steps` is 1; the counters of reliable samples are dropped.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def collect_params(model):
    """The affine scale + shift parameters of every norm layer, and their names."""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


def configure_model(model):
    """Configure model for use with EATA: train mode, gradients for the norm layers only, batch statistics."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False       # force use of batch stats in train and eval modes
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def compute_fishers(model: nn.Module, batches: Iterable[torch.Tensor]) -> dict:
    """The diagonal Fisher of the source model, official EATA main.py: for every batch of clean source images the cross
    entropy against the model's OWN predictions as labels, the squared gradient of every adapted parameter, averaged over the
    batches. Returns {parameter name: [fisher, copy of the source parameter]}. The model is configured (and left) for EATA."""
    model = configure_model(model)
    device = next(model.parameters()).device
    fishers: dict = {}
    n = 0
    for images in batches:
        n += 1
        outputs = model(images.to(device))
        _, targets = outputs.max(1)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                sq = param.grad.data.clone().detach() ** 2
                fishers[name] = [sq + fishers[name][0] if name in fishers else sq, param.data.clone().detach()]
        model.zero_grad()
    if n == 0:
        raise ValueError("no source batches: the Fisher needs clean source images")
    for name in fishers:
        fishers[name][0] = fishers[name][0] / n
    return fishers


class EataMember:
    name = "eata"

    def __init__(self, model: nn.Module, optim: dict, *, fishers: dict | None = None, fisher_alpha: float = 2000.0,
                 e_margin_frac: float = 0.4, d_margin: float = 0.05):
        if optim["METHOD"] != "SGD":
            raise ValueError(f"EATA runs on SGD, got OPTIM METHOD {optim['METHOD']!r}")
        self.model = configure_model(model)
        params, self.param_names = collect_params(self.model)
        if not params:
            raise ValueError("EATA found no parameters to adapt: check the model's normalization layers")
        self.optimizer = torch.optim.SGD(params, optim["LR"], momentum=optim.get("MOMENTUM", 0.0))
        self.fishers, self.fisher_alpha = fishers, fisher_alpha
        self.e_margin_frac, self.d_margin = e_margin_frac, d_margin
        self.current_model_probs = None                  # the moving average of the kept probability vectors (Eqn. 4)
        self.model_state, self.optimizer_state = deepcopy(self.model.state_dict()), deepcopy(self.optimizer.state_dict())

    def reset(self) -> None:
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.current_model_probs = None

    @torch.enable_grad()
    def step(self, x: torch.Tensor, class_mask=None) -> torch.Tensor:
        outputs = self.model(x)
        obj = outputs[:, class_mask] if class_mask is not None else outputs
        e_margin = self.e_margin_frac * math.log(obj.shape[1])
        entropys = softmax_entropy(obj)
        filter_ids_1 = torch.where(entropys < e_margin)       # filter unreliable samples
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]
        if self.current_model_probs is not None:              # filter redundant samples
            cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0),
                                                      obj[filter_ids_1].softmax(1), dim=1)
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = update_model_probs(self.current_model_probs, obj[filter_ids_1][filter_ids_2].softmax(1))
        else:
            updated_probs = update_model_probs(self.current_model_probs, obj[filter_ids_1].softmax(1))
        coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
        entropys = entropys.mul(coeff)                        # reweight entropy losses for different samples
        loss = entropys.mean(0)
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
            loss += ewc_loss
        if x[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.current_model_probs = updated_probs
        return outputs.detach()

    def describe(self) -> dict:
        return {"tta/method": "eata" if self.fishers is not None else "eta", "tta/fisher_alpha": self.fisher_alpha,
                "tta/e_margin_frac": self.e_margin_frac, "tta/d_margin": self.d_margin, "tta/n_params": len(self.param_names)}


if __name__ == "__main__":
    import importlib.util

    from src.experiments.tuning import SingleModel
    from src.utils.config import REPO_ROOT

    def load_official(rel: str):
        p = REPO_ROOT / "third_party" / rel
        spec = importlib.util.spec_from_file_location(f"official_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    off = load_official("EATA/eata.py")
    K = 10
    OPT = {"METHOD": "SGD", "STEPS": 1, "LR": 0.05, "MOMENTUM": 0.9, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}

    def tiny(scale: float | None = None, norm: str = "bn"):
        scale = scale if scale is not None else (30.0 if norm == "bn" else 8.0)     # where the entropy filter keeps some samples
        torch.manual_seed(0)
        if norm == "bn":
            m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(), nn.Linear(8, K))
            last = m[5]
        else:
            m = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 16), nn.LayerNorm(16), nn.ReLU(), nn.Linear(16, K))
            last = m[4]
        with torch.no_grad():
            last.weight.mul_(scale)
        return m

    def batches(n=8, bs=16, seed=1, size=8):
        g = torch.Generator().manual_seed(seed)
        return [torch.rand(bs, 3, size, size, generator=g) for _ in range(n)]

    # ---- 1. the Fisher: against an independent computation ------------------------------------------------------------
    src = batches(n=5, bs=16, seed=3)
    m = tiny()
    fishers = compute_fishers(m, src)
    ref = tiny()
    ref = configure_model(ref)
    params, names = collect_params(ref)
    acc = [torch.zeros_like(p) for p in params]
    for x in src:
        out = ref(x)
        loss = F.cross_entropy(out, out.argmax(1))
        for a, g in zip(acc, torch.autograd.grad(loss, params)):
            a += g ** 2
    assert set(fishers) == set(names)
    for nm, a, p in zip(names, acc, params):
        torch.testing.assert_close(fishers[nm][0], a / len(src), rtol=1e-5, atol=1e-10)
        assert torch.equal(fishers[nm][1], p.detach()), "the second entry is the source parameter"
    assert all((f[0] >= 0).all() for f in fishers.values()) and any(f[0].sum() > 0 for f in fishers.values())
    try:
        compute_fishers(tiny(), [])
        raise AssertionError("no source batches must raise")
    except ValueError:
        pass

    # ---- 2. EATA (with the Fisher penalty) against the official class, step for step ---------------------------------
    def official_run(fishers_, alpha, d_margin, data):
        m_off = off.configure_model(tiny())
        p_off, _ = off.collect_params(m_off)
        opt = torch.optim.SGD(p_off, OPT["LR"], momentum=0.9)
        model = off.EATA(m_off, opt, fishers_, alpha, e_margin=0.4 * math.log(K), d_margin=d_margin)
        return model, [model(x) for x in data]

    data = batches(n=10, bs=16, seed=5)
    # the real Fisher of this confident toy is ~0, so a synthetic one (stable: lr * 2 * alpha * F = 0.1) makes the penalty bite
    synthetic = {k: [torch.full_like(v[0], 0.05), v[1]] for k, v in fishers.items()}
    # d_margin 0.05 is the official value; with it the diversity filter drops every batch after the first as redundant, so
    # 2.0 (which disables the filter) is run too, to compare the model at every one of ten updates
    for alpha, fish, d_margin in ((2000.0, fishers, 0.05), (2000.0, fishers, 2.0), (20.0, synthetic, 2.0),
                                  (20.0, synthetic, 0.05), (0.0, None, 2.0), (0.0, None, 0.05)):
        official, outs_off = official_run(fish, alpha, d_margin, data)
        mine = EataMember(tiny(), OPT, fishers=fish, fisher_alpha=alpha, d_margin=d_margin)
        outs = [mine.step(x) for x in data]
        for a, b in zip(outs_off, outs):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
        for pa, pb in zip(official.model.parameters(), mine.model.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=1e-6)
        moved = any(not torch.equal(a, b) for a, b in zip(tiny().parameters(), mine.model.parameters()))
        assert moved, "the model must really be updating, or this proves nothing"
        assert mine.current_model_probs is not None
    # the penalty matters: with it the parameters stay closer to the source than without
    with_f = EataMember(tiny(), OPT, fishers=synthetic, fisher_alpha=20.0, d_margin=2.0)
    without = EataMember(tiny(), OPT, d_margin=2.0)
    for x in data:
        with_f.step(x), without.step(x)
    src_params = [p.detach().clone() for p in tiny().parameters()]
    drift = lambda mem: sum(float(((p - s) ** 2).sum()) for p, s in zip(mem.model.parameters(), src_params))   # noqa: E731
    assert 0 < drift(with_f) < drift(without), (drift(with_f), drift(without))

    # ---- 3. ETA against the repo's own EataMethod (src/tta/methods/eata.py) ---------------------------------------------
    # They agree wherever every batch keeps some samples (d_margin 2.0 disables the diversity filter). They do NOT agree when a
    # batch is filtered out entirely: upstream skips the optimizer step then (`if x[ids1][ids2].size(0) != 0`), while the
    # repo's TentMethod.update still steps on its zero loss, so SGD's momentum keeps moving the weights. With the official
    # d_margin 0.05 every batch after the first is redundant, so the two diverge. This is why ETA runs on THIS port.
    for norm, nm in (("bn", "BN"), ("ln", "LN")):
        data_n = batches(n=8, bs=16, seed=7, size=4 if norm == "ln" else 8)

        def pair(d_margin):
            mine = EataMember(tiny(norm=norm), OPT, d_margin=d_margin)
            existing = SingleModel(tiny(norm=norm), lambda x: x, nm, OPT, torch.device("cpu"), tta="eata")
            existing.tta.d_margin = d_margin
            return mine, existing, [(mine.step(x), existing.step(x)) for x in data_n]

        mine, existing, outs = pair(2.0)
        for a_, b_ in outs:
            torch.testing.assert_close(a_, b_, rtol=0, atol=1e-5)
        for pa, pb in zip(mine.model.parameters(), existing.model.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=1e-5)
        assert any(not torch.equal(a_, b_) for a_, b_ in zip(tiny(norm=norm).parameters(), mine.model.parameters()))
        mine, existing, outs = pair(0.05)
        assert max(float((a_ - b_).abs().max()) for a_, b_ in outs) > 1e-3, "the documented difference should show"
    mine.reset()
    assert mine.current_model_probs is None
    assert all(torch.equal(a_, b_) for a_, b_ in zip(tiny(norm="ln").parameters(), mine.model.parameters()))
    # a reset stream starts from the same state as the first one
    a = [EataMember(tiny(), OPT)]
    first = [a[0].step(x) for x in data]
    a[0].reset()
    again = [a[0].step(x) for x in data]
    assert all(torch.equal(u, v) for u, v in zip(first, again)), "after reset() the member replays identically"

    # ---- 4. what it reports --------------------------------------------------------------------------------------------
    z = EataMember(tiny(), OPT).step(data[0])
    assert z.shape == (16, K) and not z.requires_grad and torch.equal(z, tiny()(data[0]))
    assert EataMember(tiny(), OPT).step(data[0], class_mask=[0, 1, 2, 3]).shape == (16, K)
    assert EataMember(tiny(), OPT, fishers=fishers).describe()["tta/method"] == "eata" and EataMember(tiny(), OPT).describe()["tta/method"] == "eta"
    try:
        EataMember(tiny(), {"METHOD": "Adam", "LR": 1e-3})
        raise AssertionError("EATA needs SGD")
    except ValueError:
        pass
    print("EATA (ported) self-test passed")
