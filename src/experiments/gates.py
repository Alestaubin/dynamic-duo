"""
gates.py
========
GateSpec -> a calibrator with the BaseJointCalibrator interface (`calibrate(z_l, z_s)` -> combined logits),
built WITHOUT models: every gate evaluated here is a function of the two members' logits only, which is
what lets one member-logit cache serve every gate variant.

  fixed_ts        JointFixedTS: 0.5 * z_l / T_L + 0.5 * z_s / T_S  (the beta = 0 case of the gate)
  proxy_weighted  JointProxyWeighted: w_l = sigmoid(beta * (x_l - x_s)) from a label-free reliability proxy,
                  log or linear pooling, optional temporal filter

`MemberOnly` wraps one member so the members go through the same metric path as every calibrator.

Replay caveat: JointProxyWeighted combines the slice that COMPLETES a proxy batch with the weight computed
from that batch (itself included) and earlier slices with the previous weight, so its output depends on how
the stream is cut into calibrate() calls. Replays therefore feed the original adaptation batch size
(recorded in the cache manifest), which reproduces a live run exactly.

Not supported yet (raise NotImplementedError, each is a Phase 3/5 item): proxies that need a source-data
fit (atc, prototype, cot) and non-identity calibration maps. The paper's proxies other than ATC are
stateless.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.calibrators.base import BaseJointCalibrator, _NoOpModule
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_proxy_weighted import JointProxyWeighted
from src.experiments.specs import GateSpec
from src.reliability.calibration.logit import to_logit
from src.reliability.proxies.stats import ProxyStats
from src.utils.config import REPO_ROOT

_SOURCE_FIT_PROXIES = {"atc", "prototype", "cot"}


class MemberOnly(BaseJointCalibrator):
    """Predict with one member and ignore the other."""

    def __init__(self, side: str):
        super().__init__()
        assert side in ("large", "small")
        self.side = side

    def calibrate(self, logits_l, logits_s):
        return logits_l if self.side == "large" else logits_s

    def calibrate_with_grad(self, logits_l, logits_s):
        return self.calibrate(logits_l, logits_s)

    def forward(self, logits_l, logits_s):
        return self.calibrate(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        pass

    @property
    def model(self):
        return _NoOpModule()


def load_fixed_ts(path: str | None) -> JointFixedTS | None:
    """T_L, T_S from a `config.json` written by JointFixedTS.save (read directly: JointFixedTS.load prints)."""
    if path is None:
        return None
    folder = Path(path) if str(path).startswith("/") else REPO_ROOT / path
    cfg = json.loads((folder / "config.json").read_text())
    if cfg.get("class_name") != "JointFixedTS":
        raise ValueError(f"{folder}/config.json is not a JointFixedTS checkpoint")
    ts = JointFixedTS(Tl=cfg["T_l"], Ts=cfg["T_s"], verbose=False)
    for prm in ts.parameters():
        prm.requires_grad_(False)
    return ts


def build_calibrator(spec: GateSpec, *, num_classes: int = 1000, member_names: tuple[str, str] = ("large", "small")):
    """The calibrator for `spec`. `num_classes` is the dataset's class count (200 on ImageNet-A/R, whose
    logits are masked before they reach the gate)."""
    if spec.kind == "fixed_ts":
        ts = load_fixed_ts(spec.fixed_ts) or JointFixedTS(verbose=False)
        ts.verbose = False
        return ts

    if spec.kind == "proxy_weighted":
        if spec.proxy in _SOURCE_FIT_PROXIES:
            raise NotImplementedError(
                f"proxy {spec.proxy!r} needs a source-data fit (clean train slice); not wired into the "
                f"replay engine yet (plan.md Phase 3)."
            )
        if spec.calib_method != "identity":
            raise NotImplementedError(
                f"calib_method {spec.calib_method!r}: only 'identity' (the paper's setting) is supported."
            )
        cal = JointProxyWeighted(
            proxy_kind=spec.proxy,
            cfg_l=ProxyStats(name=member_names[0], num_classes=num_classes),
            cfg_s=ProxyStats(name=member_names[1], num_classes=num_classes),
            beta=spec.beta,
            filter_kind=spec.filter,
            filter_kwargs=dict(spec.filter_kwargs),
            prior_l=to_logit(0.5), prior_s=to_logit(0.5),
            base_ts=load_fixed_ts(spec.fixed_ts),
            proxy_batch_size=spec.proxy_batch_size,
            pool=spec.pool,
            verbose=False,
        )
        return cal
    raise ValueError(f"unknown gate kind {spec.kind!r}")


if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)
    z_l, z_s = torch.randn(256, 1000) * 3, torch.randn(256, 1000) * 3
    with tempfile.TemporaryDirectory() as tmp:
        JointFixedTS(Tl=1.7, Ts=0.9, verbose=False).save(tmp)
        ts_spec = GateSpec("fixed", kind="fixed_ts", fixed_ts=tmp)
        fixed = build_calibrator(ts_spec)
        want = 0.5 * z_l / 1.7 + 0.5 * z_s / 0.9
        assert torch.allclose(fixed.calibrate(z_l, z_s).cpu(), want, atol=1e-5)

        # the paper's default gate; beta = 0 must reduce to fixed TS exactly (w_l = 0.5)
        ours = build_calibrator(GateSpec("ours", fixed_ts=tmp, beta=0.5, proxy_batch_size=64))
        zero = build_calibrator(GateSpec("b0", fixed_ts=tmp, beta=0.0, proxy_batch_size=64))
        ours.records, zero.records = [], []
        for cal in (ours, zero):
            cal.set_corruption("seg", total_samples=256)
        out_zero = torch.cat([zero.calibrate(z_l[i:i + 64], z_s[i:i + 64]).cpu() for i in range(0, 256, 64)])
        assert torch.allclose(out_zero, want, atol=1e-5), "beta = 0 is the fixed combination"
        out = torch.cat([ours.calibrate(z_l[i:i + 64], z_s[i:i + 64]).cpu() for i in range(0, 256, 64)])
        assert len(ours.records) == 4 and all(r["corruption"] == "seg" and 0 <= r["w_l"] <= 1 for r in ours.records)
        assert not torch.allclose(out, want, atol=1e-3), "beta > 0 must move the weights when the members differ"
        # NOT chunk-invariant, by design of JointProxyWeighted: the slice that completes a proxy batch is
        # combined with the weight computed FROM that batch, earlier slices with the previous weight. So a
        # replay must feed the original adaptation batch size. Same chunking => identical output:
        ours2 = build_calibrator(GateSpec("ours", fixed_ts=tmp, beta=0.5, proxy_batch_size=64))
        ours2.set_corruption("seg", total_samples=256)
        out2 = torch.cat([ours2.calibrate(z_l[i:i + 64], z_s[i:i + 64]).cpu() for i in range(0, 256, 64)])
        assert torch.equal(out, out2), "replaying with the same chunking must be deterministic"

        # members
        assert MemberOnly("large").calibrate(z_l, z_s) is z_l and MemberOnly("small").calibrate(z_l, z_s) is z_s
        # unsupported settings fail loudly instead of silently doing something else
        for bad in (GateSpec("a", proxy="atc"), GateSpec("b", calib_method="isotonic")):
            try:
                build_calibrator(bad)
                raise AssertionError("must raise")
            except NotImplementedError:
                pass
    print("gates self-test passed")
