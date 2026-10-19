"""
coca_run.py
===========
COCA ("When Small Guides Large", the coupled small/large duo method) run IN PLACE from its checkout (Phase 3, Stage D).

The repository third_party/COCA has no LICENSE file, so none of its code is copied into src/ (the author's decision B): `COCAV4.py`
and `tent.py` are loaded from the checkout by path at run time and used as they are. This module only builds what COCA's own
`main.py` builds for `--method coca --plusmethod tent` and drives it over our streams:

    models      the duo's two models, configured with COCA's `tent.configure_model` (norm layers train, batch statistics)
    optimizer   SGD, momentum 0.9, one parameter group per model: the ResNet (the small member) and the ViT (the large one),
                each with its own learning rate (the brief's published 2.5e-4 and 1e-3 are the centres of the grids)
    step        `Coca(resnet, vit, optimizer, args)(x_resnet, x_vit)` -> (resnet logits, vit logits, the combination), all
                computed BEFORE the update of that step; the combination `(vit + resnet / T) / 2` (rescaled) is COCA's prediction
                and T is its temperature, fitted by 10 AdamW steps per batch (the official code; the brief says 3 to 5, the author
                chose the official 10) and carried from batch to batch

Two facts about the released code. `Coca.forward` and its loss read `self.tryT` and `self.halfcombine`, which nothing in the
repository ever assigns (tested below: without them the first call raises AttributeError); both are set to False here, the only
values under which the method adapts and uses its learned temperature. And `Coca` has no reset, so `reset()` restores the two
models, the optimizer and T (back to 1, with a fresh AdamW) by hand: every stream starts from the source weights (episodic).

COCA is a duo method with its own run: its prediction depends on T's history, which cannot be replayed from the members' logits,
so it is scored online (accuracy, ECE, NLL of the combination and of each member) and not stored in the logit store.
Streams with a class mask (ImageNet-A/R) are not supported here.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch

from src.experiments.evaluate import StreamMetrics
from src.experiments.members import PairCollate, load_models
from src.experiments.protocols import Segment
from src.utils.config import REPO_ROOT

COCA_DIR = REPO_ROOT / "third_party" / "COCA" / "COCA-maincode"
_loaded: dict = {}


def load_coca():
    """(COCAV4 module, COCA's tent module), loaded from the checkout by path."""
    if not _loaded:
        for name in ("COCAV4", "tent"):
            path = COCA_DIR / f"{name}.py"
            if not path.is_file():
                raise FileNotFoundError(f"{path} is missing: python third_party/fetch.py COCA")
            spec = importlib.util.spec_from_file_location(f"coca_official_{name}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            _loaded[name] = mod
    return _loaded["COCAV4"], _loaded["tent"]


class CocaRunner:
    def __init__(self, ctx, lr_large: float, lr_small: float, momentum: float = 0.9, models=None):
        coca_mod, tent_mod = load_coca()
        (self.large, self.large_preprocess), (self.small, self.small_preprocess) = (
            models if models is not None else (ctx.model_factory() if ctx.model_factory else load_models(ctx.duo, ctx.device)))
        self.device = ctx.device
        for m in (self.large, self.small):
            tent_mod.configure_model(m)
        p_l, _ = tent_mod.collect_params(self.large)
        p_s, _ = tent_mod.collect_params(self.small)
        self.optimizer = torch.optim.SGD([{"params": p_s, "lr": lr_small, "momentum": momentum},
                                          {"params": p_l, "lr": lr_large, "momentum": momentum}])
        args = SimpleNamespace(gpu=self.device, method="coca", plusmethod="tent", rho=0.05)
        self.coca = coca_mod.Coca(self.small, self.large, self.optimizer, args)      # (resnet, vit, optimizer, args)
        self.coca.tryT, self.coca.halfcombine = False, False                          # never assigned in the released code
        self._tent_mod = tent_mod
        self._state = (copy.deepcopy(self.large.state_dict()), copy.deepcopy(self.small.state_dict()),
                       copy.deepcopy(self.optimizer.state_dict()))
        self.lr_large, self.lr_small = lr_large, lr_small

    def reset(self) -> None:
        self.large.load_state_dict(self._state[0], strict=True)
        self.small.load_state_dict(self._state[1], strict=True)
        self.optimizer.load_state_dict(self._state[2])
        with torch.no_grad():
            self.coca.T.fill_(1.0)
        self.coca.T_optimizer = torch.optim.AdamW([self.coca.T], 0.01, weight_decay=0)

    def step_tensors(self, x_l: torch.Tensor, x_s: torch.Tensor):
        """-> (vit logits, resnet logits, COCA's combination, T), the logits before this step's update."""
        z_s, z_l, z_c = self.coca(x_s.to(self.device, non_blocking=True), x_l.to(self.device, non_blocking=True))
        return z_l.detach(), z_s.detach(), z_c.detach(), float(self.coca.T.detach())


def run_coca(ctx, segments: Sequence[Segment], *, lr_large: float, lr_small: float, seed: int,
             num_samples: int | None = None, runner: CocaRunner | None = None) -> dict:
    """COCA over episodic `segments`. Returns {"rows": [one per (stream, series): coca / large / small, with accuracy, ece, nll,
    entropy], "tau": {stream label: [T after every batch]}}."""
    for seg in segments:
        if seg.dataset.class_mask is not None:
            raise ValueError(f"{seg.label}: COCA runs on 1000-class streams only here")
    runner = runner or CocaRunner(ctx, lr_large, lr_small)
    rows, taus = [], {}
    for seg in segments:
        runner.reset()
        ds = seg.dataset
        loader = ds.loader(batch_size=ctx.batch_size, num_samples=num_samples, seed=seed, workers=ctx.g.workers,
                           pin_memory=runner.device.type == "cuda",
                           collate_fn=PairCollate(ds.input_transform(runner.large_preprocess),
                                                  ds.input_transform(runner.small_preprocess)))
        metrics = {s: StreamMetrics(ctx.g.ece_bins) for s in ("coca", "large", "small")}
        taus[seg.label] = []
        for x_l, x_s, y in loader:
            z_l, z_s, z_c, tau = runner.step_tensors(x_l, x_s)
            metrics["coca"].update(z_c, y)
            metrics["large"].update(z_l, y)
            metrics["small"].update(z_s, y)
            taus[seg.label].append(tau)
        rows += [{"segment": seg.label, "series": s, "seed": seed, "lr_large": runner.lr_large, "lr_small": runner.lr_small,
                  **m.result()} for s, m in metrics.items()]
    return {"rows": rows, "tau": taus}


if __name__ == "__main__":
    import tempfile

    from src.experiments._testing import FakeStream, TinyBN, TinyLN, make_context
    from torchvision import transforms

    torch.set_num_threads(2)

    def tiny_pair(sharp=3.0):
        torch.manual_seed(0)
        out = []
        for cls in (TinyLN, TinyBN):
            m = cls().eval()
            with torch.no_grad():
                m.net[-1].weight.mul_(sharp)
            for p in m.parameters():
                p.requires_grad_(False)
            out.append((m, transforms.ToTensor()))
        return tuple(out)

    coca_mod, tent_mod = load_coca()
    assert COCA_DIR.is_dir() and coca_mod.__file__.startswith(str(COCA_DIR)) and tent_mod.__file__.startswith(str(COCA_DIR))

    # ---- the released code needs tryT and halfcombine, which it never assigns ----------------------------------------------
    m1, m2 = tiny_pair()[1][0], tiny_pair()[0][0]
    bare = coca_mod.Coca(m1, m2, torch.optim.SGD([torch.zeros(1, requires_grad=True)], 0.1),
                         SimpleNamespace(gpu=torch.device("cpu"), method="coca", plusmethod="tent", rho=0.05))
    try:
        bare(torch.rand(4, 3, 8, 8), torch.rand(4, 3, 8, 8))
        raise AssertionError("the released Coca must fail without tryT")
    except AttributeError as e:
        assert "tryT" in str(e)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_context(Path(tmp), batch_size=32, base=tiny_pair())
        segs = [Segment(f"s{i}/s5", FakeStream(150, seed=sd), True) for i, sd in enumerate((1, 2))]
        res = run_coca(ctx, segs, lr_large=0.05, lr_small=0.05, seed=3)
        assert {r["series"] for r in res["rows"]} == {"coca", "large", "small"} and len(res["rows"]) == 6
        assert all(r["n"] == 150 and 0 <= r["accuracy"] <= 1 and r["lr_large"] == 0.05 for r in res["rows"])
        assert all(len(v) == 5 for v in res["tau"].values()), "one temperature per batch (150 samples in batches of 32)"
        assert all(abs(v[-1] - 1.0) > 1e-4 for v in res["tau"].values()), "the temperature is fitted, so it leaves 1"

        # the members are scored BEFORE the update, and adapt afterwards; reset restores everything including T
        runner = CocaRunner(ctx, 0.05, 0.05)
        ds = segs[0].dataset
        frozen_l, frozen_s = tiny_pair()[0][0], tiny_pair()[1][0]
        loader = ds.loader(batch_size=32, seed=3, workers=0, collate_fn=PairCollate(
            ds.input_transform(runner.large_preprocess), ds.input_transform(runner.small_preprocess)))
        w0 = [p.detach().clone() for p in runner.large.parameters()]
        outs = [runner.step_tensors(xl, xs) for xl, xs, _ in loader]
        first = next(iter(ds.loader(batch_size=32, seed=3, workers=0, collate_fn=PairCollate(
            ds.input_transform(runner.large_preprocess), ds.input_transform(runner.small_preprocess)))))
        with torch.no_grad():
            torch.testing.assert_close(outs[0][0], frozen_l(first[0]), rtol=0, atol=1e-6)
        assert any(not torch.equal(a, b) for a, b in zip(w0, runner.large.parameters())), "the ViT stand-in must adapt"
        assert abs(outs[-1][3] - 1.0) > 1e-4
        runner.reset()
        assert all(torch.equal(a, b) for a, b in zip(w0, runner.large.parameters())) and float(runner.coca.T) == 1.0
        # a reset run replays identically (CPU, same seed)
        again = [runner.step_tensors(xl, xs) for xl, xs, _ in ds.loader(batch_size=32, seed=3, workers=0, collate_fn=PairCollate(
            ds.input_transform(runner.large_preprocess), ds.input_transform(runner.small_preprocess)))]
        assert all(torch.equal(a[0], b[0]) and torch.equal(a[2], b[2]) and a[3] == b[3] for a, b in zip(outs, again))
        try:
            run_coca(ctx, [Segment("m/s5", FakeStream(150, seed=1, masked=True), True)], lr_large=0.05, lr_small=0.05, seed=3)
            raise AssertionError("masked streams are refused")
        except ValueError:
            pass
    print("coca_run self-test passed")
