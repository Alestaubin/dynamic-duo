"""
joint_dd.py
===========
Live gate and Joint DD runner for the Dynamic Duos proxies of src/reliability/proxies/pure.py (Phase 3, Stage C).

`PureProxyGate` is `JointProxyWeighted` configured the way the paper's evaluator does (filter none, log pool, prior weight 0.5,
source statistics handed in, any `pure.PROXIES` kind): the bucket logic, the gate formula and the clip eps are the live gate's
own, and so are the proxies (JointProxyWeighted scores through pure.py), so it is the reference the offline evaluator
(dd_eval.py) must reproduce exactly. In Vanilla DD it
sits outside the adaptation loop (the members never see it); in **Joint DD** it sits INSIDE it: `run_joint_dd` adapts both
members on the entropy of the gated prediction, with the weight a constant of each step, as the brief defines it.

    gate = PureProxyGate("nnn", beta=0.5, b_t=128, T_l=0.77, T_s=1.22)
    res  = run_joint_dd(ctx, segments, gate_kwargs=dict(proxy="nnn", beta=0.5, b_t=128, T_l=0.77, T_s=1.22), seed=0)

With beta = 0 the gate weight is exactly 0.5 and Joint DD is the engine's joint duo mode (`both_duo` with a static
JointFixedTS): same logits, same updates (equivalence test 3, below and in the Stage C smoke job).

Joint DD gates the 1000-way logits, so it does not support the 200-class natural shifts (their gate needs the masked logits).
"""

from __future__ import annotations

import torch

from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_proxy_weighted import JointProxyWeighted
from src.experiments.evaluate import StreamMetrics
from src.experiments.members import MemberRunner
from src.reliability.proxies import pure
from src.reliability.proxies.stats import ProxyStats


class PureProxyGate(JointProxyWeighted):
    """JointProxyWeighted whose proxy is one of `pure.PROXIES`. `stats_l` / `stats_s`: `fit_source_stats` of each member
    (needed by the stateful proxies). eps defaults to the live gate's 1e-3."""

    def __init__(self, proxy: str, *, beta: float, b_t: int, T_l: float, T_s: float, stats_l: dict | None = None,
                 stats_s: dict | None = None, num_classes: int = 1000, eps: float = 1e-3):
        if proxy not in pure.PROXIES:
            raise ValueError(f"unknown proxy {proxy!r}: {sorted(pure.PROXIES)}")
        if pure.PROXIES[proxy].stateful and (stats_l is None or stats_s is None):
            raise ValueError(f"proxy {proxy!r} is stateful: pass stats_l and stats_s (fit_source_stats)")
        ts = JointFixedTS(Tl=T_l, Ts=T_s, verbose=False)
        for prm in ts.parameters():
            prm.requires_grad_(False)
        cfg_l, cfg_s = ProxyStats("large", num_classes), ProxyStats("small", num_classes)
        cfg_l.source_stats, cfg_s.source_stats = stats_l, stats_s
        super().__init__(proxy_kind=proxy, cfg_l=cfg_l, cfg_s=cfg_s, beta=beta, filter_kind="none", prior_l=0.0,
                         prior_s=0.0, base_ts=ts, eps=eps, proxy_batch_size=b_t, pool="log", verbose=False)


class RecordingCalibrator:
    """Wraps a calibrator used as a MemberRunner `loop_calibrator` and keeps the combined logits of the last step (the
    pre-update prediction the run scores), which the runner itself does not return."""

    def __init__(self, inner):
        self.inner, self.last = inner, None

    def calibrate_with_grad(self, logits_l, logits_s):
        out = self.inner.calibrate_with_grad(logits_l=logits_l, logits_s=logits_s)
        self.last = out.detach()
        return out

    def calibrate(self, logits_l, logits_s):
        return self.inner.calibrate(logits_l, logits_s)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def run_joint_dd(ctx, segments, *, gate_kwargs: dict, seed: int, num_samples: int | None = None, tta: str = "tent",
                 tta_kwargs: dict | None = None, gate: PureProxyGate | None = None) -> dict:
    """Joint DD, live: both members adapt on the entropy of the gated prediction (the weight is a constant of the step),
    source weights and a fresh gate before every stream (episodic). Returns {"rows": [...], "gate_records": {...}}: one
    row per (stream, series) with series duo / large / small and the accuracy, ECE, NLL, entropy of the PRE-update logits,
    and the gate's per-bucket records (proxy values, weights) per stream."""
    for seg in segments:
        if seg.dataset.class_mask is not None:
            raise ValueError(f"{seg.label}: Joint DD gates the 1000-way logits; 200-class datasets are not supported")
    gate = gate or PureProxyGate(**gate_kwargs)
    gate.records = []
    rec = RecordingCalibrator(gate)
    runner = MemberRunner(ctx.duo, tta, tta_kwargs or {}, "both_duo", ctx.device, ctx.objective_logits,
                          models=ctx.model_factory() if ctx.model_factory else None, loop_calibrator=rec, bn=ctx.bn)
    rows, records = [], {}
    for seg in segments:
        runner.reset()
        n = min(num_samples, len(seg.dataset)) if num_samples else len(seg.dataset)
        gate.set_corruption(seg.label, total_samples=n)
        metrics = {s: StreamMetrics(ctx.g.ece_bins) for s in ("duo", "large", "small")}
        for z_l, z_s, y in runner.iter_batches(seg.dataset, batch_size=ctx.batch_size, workers=ctx.g.workers, seed=seed,
                                               num_samples=num_samples):
            metrics["duo"].update(rec.last, y)
            metrics["large"].update(z_l, y)
            metrics["small"].update(z_s, y)
        rows += [{"segment": seg.label, "series": s, **m.result()} for s, m in metrics.items()]
        records[seg.label], gate.records = gate.records, []
    return {"rows": rows, "gate_records": records}


if __name__ == "__main__":
    import tempfile
    from pathlib import Path


    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.experiments.protocols import Segment

    torch.set_num_threads(2)
    torch.manual_seed(0)
    T_L, T_S = 1.4, 0.9

    # ---- the gate: every pure proxy runs inside the live JointProxyWeighted machinery -----------------------------------
    z_l, z_s = torch.randn(300, 50) * 3, torch.randn(300, 50) * 2
    y = torch.randint(0, 50, (300,))
    src_l, src_s = pure.fit_source_stats(z_l, y), pure.fit_source_stats(z_s, y)
    for name in pure.PROXIES:
        g = PureProxyGate(name, beta=1.0, b_t=64, T_l=T_L, T_s=T_S, stats_l=src_l, stats_s=src_s, num_classes=50)
        g.records = []
        g.set_corruption("s", total_samples=300)
        out = torch.cat([g.calibrate(z_l[i:i + 32], z_s[i:i + 32]) for i in range(0, 300, 32)])
        assert out.shape == (300, 50) and len(g.records) == 5 and all(0 <= r["w_l"] <= 1 for r in g.records), name
        assert g.records[0]["r_l"] == pure.score(name, z_l[:64], None, src_l, **(
            {"duo_logits": 0.5 * (z_l[:64] / T_L) + 0.5 * (z_s[:64] / T_S)} if pure.PROXIES[name].needs_duo else {})), name
    try:
        PureProxyGate("energy", beta=1.0, b_t=64, T_l=1.0, T_s=1.0)
        raise AssertionError("a stateful proxy needs its statistics")
    except ValueError:
        pass
    g0 = PureProxyGate("nnn", beta=0.0, b_t=64, T_l=T_L, T_s=T_S, num_classes=50)
    g0.set_corruption("s", total_samples=300)
    assert torch.equal(torch.cat([g0.calibrate(z_l[i:i + 32], z_s[i:i + 32]) for i in range(0, 300, 32)]),
                       0.5 * (z_l / T_L) + 0.5 * (z_s / T_S)), "beta = 0 is the fixed combination, exactly"

    # ---- equivalence test 3: Joint DD with beta = 0 equals the engine's joint duo mode -----------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_context(Path(tmp), batch_size=32, base=tiny_models(0))
        segs = [Segment(f"s{i}/s5", FakeStream(150, seed=sd), True) for i, sd in enumerate((1, 2))]

        static = JointFixedTS(Tl=T_L, Ts=T_S, verbose=False)
        static.device = torch.device("cpu")
        for prm in static.parameters():
            prm.requires_grad_(False)
        ref = MemberRunner(ctx.duo, "tent", {}, "both_duo", ctx.device, ctx.objective_logits, models=ctx.model_factory(),
                           loop_calibrator=RecordingCalibrator(static), bn=False)
        ref_out = []
        for seg in segs:
            ref.reset()
            for zl, zs, yy in ref.iter_batches(seg.dataset, batch_size=32, workers=0, seed=3, num_samples=None):
                ref_out.append((zl.clone(), zs.clone(), ref.loop_calibrator.last.clone()))

        res0 = run_joint_dd(ctx, segs, gate_kwargs=dict(proxy="nnn", beta=0.0, b_t=128, T_l=T_L, T_s=T_S), seed=3)
        # the same run again, capturing logits, to compare them bit for bit
        gate = PureProxyGate("nnn", beta=0.0, b_t=128, T_l=T_L, T_s=T_S)
        rec = RecordingCalibrator(gate)
        jd = MemberRunner(ctx.duo, "tent", {}, "both_duo", ctx.device, ctx.objective_logits, models=ctx.model_factory(),
                          loop_calibrator=rec, bn=False)
        k = 0
        for seg in segs:
            jd.reset()
            gate.set_corruption(seg.label, total_samples=150)
            for zl, zs, yy in jd.iter_batches(seg.dataset, batch_size=32, workers=0, seed=3, num_samples=None):
                assert zl.equal(ref_out[k][0]) and zs.equal(ref_out[k][1]), f"batch {k}: members' logits differ"
                assert rec.last.equal(ref_out[k][2]), f"batch {k}: the duo's logits differ"
                k += 1
        assert k == len(ref_out) == 10
        assert {r["series"] for r in res0["rows"]} == {"duo", "large", "small"} and len(res0["rows"]) == 6
        assert sorted(res0["gate_records"]) == ["s0/s5", "s1/s5"] and all(len(v) == 2 for v in res0["gate_records"].values())

        # beta > 0 really changes what the members learn
        res1 = run_joint_dd(ctx, segs, gate_kwargs=dict(proxy="msp", beta=8.0, b_t=64, T_l=T_L, T_s=T_S), seed=3)
        w = [r["w_l"] for v in res1["gate_records"].values() for r in v]
        assert any(abs(x - 0.5) > 1e-3 for x in w), "the gate must move off 0.5"
        d0 = {(r["segment"], r["series"]): r for r in res0["rows"]}
        d1 = {(r["segment"], r["series"]): r for r in res1["rows"]}
        assert any(d0[k_]["nll"] != d1[k_]["nll"] for k_ in d0 if k_[1] == "duo")
        # the gate starts fresh on every stream, and a 200-class stream is refused
        assert [len(res1["gate_records"][s]) for s in ("s0/s5", "s1/s5")] == [3, 3], \
            "150 samples in buckets of 64: 64 + 64 + 22 on each stream, none straddling the boundary"
        try:
            run_joint_dd(ctx, [Segment("m/s5", FakeStream(150, seed=1, masked=True), True)],
                         gate_kwargs=dict(proxy="nnn", beta=0.0, b_t=64, T_l=1.0, T_s=1.0), seed=3)
            raise AssertionError("a masked stream must be refused")
        except ValueError:
            pass

    print("joint_dd self-test passed")
