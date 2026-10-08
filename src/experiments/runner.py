"""
runner.py
=========
Compute-or-load the members' logits, then replay gates on them.

    ctx   = Context.create("vitb16_rn50")
    spec  = RunSpec("vitb16_rn50", tta="tent", seed=0)
    gates = [GateSpec("fixed TS", kind="fixed_ts"), GateSpec.from_global(ctx.g, "ours")]
    res   = evaluate(spec, ctx, gates)          # res.rows: one dict per (segment, series)

`evaluate` = `ensure_members` (adapt the two models over the stream and cache their logits, only for the
segments not cached yet) + `replay` (feed the cached logits, in the original adaptation-batch chunks, to every
gate). `report_only=True` never runs a model: it fails, listing what is missing, if the cache is incomplete.
CCC streams (7.5M images) are not cached; `evaluate_live` streams the members straight into the gates.

Episodic segments are independent (each starts from source weights), so a killed job resumes at the first
missing segment. Continual/CCC segments depend on all earlier ones, so those are recomputed from the start.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from tqdm import tqdm

from src.experiments.cache import MemberCache
from src.experiments.evaluate import OnlineEval
from src.experiments.gates import MemberOnly, build_calibrator  # noqa: F401 (MemberOnly re-exported for scripts)
from src.experiments.members import MemberRunner
from src.experiments.protocols import Segment, build_segments, resets_between_segments
from src.experiments.results import annotate
from src.experiments.specs import GateSpec, RunSpec
from src.utils.config import GlobalConfig, load_duo, load_global


@dataclass
class Context:
    g: GlobalConfig
    duo: dict
    device: torch.device
    # Returns fresh ((large, preprocess), (small, preprocess)); None = load the duo's pretrained models.
    # Lets tests run the whole engine on tiny CPU models.
    model_factory: Callable | None = None

    @classmethod
    def create(cls, duo_name: str, g: GlobalConfig | None = None, device: str | None = None) -> "Context":
        g = g or load_global()
        return cls(g, load_duo(duo_name, g), torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))

    @property
    def batch_size(self) -> int:
        return self.g.batch_size

    @property
    def objective_logits(self) -> str:
        return self.g.objective_logits_200class


@dataclass
class EvalResult:
    rows: list[dict]                       # one per (segment, series): n, accuracy, ece, nll, entropy
    records: dict[str, list[dict]]         # per-proxy-batch gate records, by gate name
    curves: dict[str, dict]                # per-batch correct counts, by series (for accuracy-vs-time plots)
    segment_batches: list[tuple[str, int]]


def run_dir(spec: RunSpec, ctx: Context) -> Path:
    if spec.duo != ctx.duo["NAME"]:
        raise ValueError(f"spec.duo={spec.duo!r} but the context holds duo {ctx.duo['NAME']!r}")
    fp = spec.fingerprint(ctx.duo, ctx.batch_size, ctx.objective_logits)
    return ctx.g.path("cache") / "members" / spec.duo / spec.tta / f"{spec.protocol_tag}_seed{spec.seed}_{fp}"


def _member_names(ctx: Context) -> tuple[str, str]:
    return ctx.duo["LARGE"]["NAME"], ctx.duo["SMALL"]["NAME"]


def _store_segment(runner: MemberRunner, seg: Segment, spec: RunSpec, ctx: Context, cache: MemberCache) -> int:
    zs_l, zs_s, ys = [], [], []
    loader_n = spec.num_samples
    for z_l, z_s, labels in tqdm(
        runner.iter_batches(seg.dataset, batch_size=ctx.batch_size, workers=ctx.g.workers,
                            seed=spec.seed, num_samples=loader_n),
        desc=f"{spec.tta} {seg.label}", leave=False,
    ):
        zs_l.append(z_l.cpu()); zs_s.append(z_s.cpu()); ys.append(labels)
    cache.write(seg.file_stem, torch.cat(zs_l), torch.cat(zs_s), torch.cat(ys))
    return sum(len(y) for y in ys)


def _runner(spec: RunSpec, ctx: Context) -> MemberRunner:
    return MemberRunner(ctx.duo, spec.tta, spec.tta_kwargs, spec.mode, ctx.device, ctx.objective_logits,
                        models=ctx.model_factory() if ctx.model_factory else None)


def ensure_members(spec: RunSpec, ctx: Context, segments: list[Segment] | None = None) -> MemberCache:
    """Adapt the members over the stream and cache their logits, for the segments not cached yet."""
    segments = segments or build_segments(spec, ctx.g)
    cache = MemberCache(run_dir(spec, ctx))
    missing = [s for s in segments if not cache.has(s.file_stem)]
    if not missing:
        return cache
    independent = resets_between_segments(spec)
    todo = missing if independent else segments          # continual: segment k needs all before it

    runner = _runner(spec, ctx)
    t0, n_images, sizes = time.time(), 0, {}
    for seg in todo:
        if seg.reset_before:
            runner.reset()
        n = _store_segment(runner, seg, spec, ctx, cache)
        sizes[seg.label] = n
        n_images += n
        cache.write_manifest({
            "spec": spec.describe(), "batch_size": ctx.batch_size, "objective_logits": ctx.objective_logits,
            "duo": {s: ctx.duo[s] for s in ("LARGE", "SMALL")}, "tta": runner.tta.describe(),
            "segments": {**(cache.read_manifest().get("segments", {}) if cache.manifest_path.exists() else {}), **sizes},
            "seconds": time.time() - t0, "images_adapted": n_images,
        })
    return cache


def _check_complete(spec: RunSpec, cache: MemberCache, segments: list[Segment]) -> None:
    missing = [s.label for s in segments if not cache.has(s.file_stem)]
    if missing:
        raise FileNotFoundError(
            f"member logits missing for {len(missing)}/{len(segments)} segments ({missing[:4]}...) in {cache.dir}. "
            f"Run the script without --report_only to compute them (it runs on a GPU)."
        )


def _new_eval(gates: Sequence[GateSpec], ctx: Context, segments: list[Segment]) -> OnlineEval:
    k = {s.dataset.num_classes for s in segments}
    if len(k) != 1:
        raise ValueError(f"segments disagree on the class count: {k}")
    n_classes = k.pop()
    calibrators = {g.name: build_calibrator(g, num_classes=n_classes, member_names=_member_names(ctx)) for g in gates}
    return OnlineEval(calibrators, num_bins=ctx.g.ece_bins, device=ctx.device)


def _finish(ev: OnlineEval, spec: RunSpec) -> EvalResult:
    return EvalResult(rows=annotate(ev.rows, spec), records=ev.records, curves=ev.curves,
                      segment_batches=ev.segment_batches)


def replay(spec: RunSpec, ctx: Context, gates: Sequence[GateSpec],
           segments: list[Segment] | None = None) -> EvalResult:
    """Evaluate `gates` (and the two members) on the cached member logits. Touches no model and no dataset."""
    segments = segments or build_segments(spec, ctx.g)
    cache = MemberCache(run_dir(spec, ctx))
    _check_complete(spec, cache, segments)
    bs = int(cache.read_manifest()["batch_size"])        # must match the live run: see gates.py's caveat
    sizes = {s.label: cache.n_samples(s.file_stem) for s in segments}
    ev = _new_eval(gates, ctx, segments)
    for seg in segments:
        d = cache.read(seg.file_stem)
        ev.start_segment(seg.label, seg.dataset, reset=seg.reset_before, n_samples=d["n"], stream_total=sum(sizes.values()))
        for i in range(0, d["n"], bs):
            ev.update(d["z_l"][i:i + bs], d["z_s"][i:i + bs], d["labels"][i:i + bs])
        ev.end_segment()
    return _finish(ev, spec)


def evaluate_live(spec: RunSpec, ctx: Context, gates: Sequence[GateSpec],
                  segments: list[Segment] | None = None) -> EvalResult:
    """Adapt the members and feed the gates in the same pass, caching nothing (CCC-sized streams)."""
    segments = segments or build_segments(spec, ctx.g)
    ev = _new_eval(gates, ctx, segments)
    runner = _runner(spec, ctx)
    total = sum(spec.num_samples or len(s.dataset) for s in segments)
    for seg in segments:
        if seg.reset_before:
            runner.reset()
        n = spec.num_samples or len(seg.dataset)
        ev.start_segment(seg.label, seg.dataset, reset=seg.reset_before, n_samples=n, stream_total=total)
        for z_l, z_s, labels in tqdm(
            runner.iter_batches(seg.dataset, batch_size=ctx.batch_size, workers=ctx.g.workers, seed=spec.seed,
                                num_samples=spec.num_samples),
            desc=f"{spec.tta} {seg.label}", total=-(-n // ctx.batch_size), leave=False,
        ):
            ev.update(z_l, z_s, labels)
        ev.end_segment()
    return _finish(ev, spec)


def evaluate(spec: RunSpec, ctx: Context, gates: Sequence[GateSpec], *, report_only: bool = False,
             segments: list[Segment] | None = None) -> EvalResult:
    if spec.protocol == "ccc":
        if report_only:
            raise ValueError("CCC streams are not cached, so there is nothing to report from; run without --report_only")
        return evaluate_live(spec, ctx, gates, segments)
    if not report_only:
        ensure_members(spec, ctx, segments)
    return replay(spec, ctx, gates, segments)


if __name__ == "__main__":
    import copy
    import os
    import tempfile

    from src.calibrators.joint_fixed_TS import JointFixedTS
    from src.experiments._testing import DUO, FakeStream, make_context, tiny_models
    from src.experiments.cache import center_fp16
    from src.tta.dynamic_duo import forward_and_adapt
    from src.tta.methods import build_tta_method

    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tiny_models(0)
        ctx = make_context(tmp, batch_size=32, base=base)
        ts_dir = tmp / "ts"
        JointFixedTS(Tl=1.4, Ts=0.9, verbose=False).save(str(ts_dir))
        gates = [GateSpec("fixed", kind="fixed_ts", fixed_ts=str(ts_dir)),
                 GateSpec("ours", fixed_ts=str(ts_dir), proxy_batch_size=64, beta=2.0)]

        def segs(*seeds, resets=True):
            return [Segment(f"s{i}/s5", FakeStream(150, seed=sd), True if resets else i == 0)
                    for i, sd in enumerate(seeds)]

        # ---- 1. the new runner reproduces the old engine's member logits bit for bit -----------------------
        spec = RunSpec("tiny", seed=3)
        two = segs(1, 2)
        cache = ensure_members(spec, ctx, two)
        (large, lp), (small, sp) = copy.deepcopy(base)
        tta = build_tta_method("tent")
        tta.setup(large, small, DUO, True, True)
        for seg in two:
            tta.reset()
            zl, zs = [], []
            for imgs, _ in seg.dataset.loader(32, None, 3, 0):
                _, a, b = forward_and_adapt(imgs, large, lp, small, sp, JointFixedTS(verbose=False), tta, "both_indep")
                zl.append(a.detach()); zs.append(b.detach())
            d = cache.read(seg.file_stem)
            assert d["z_l"].equal(center_fp16(torch.cat(zl)).float()), "large logits differ from the old engine"
            assert d["z_s"].equal(center_fp16(torch.cat(zs)).float()), "small logits differ from the old engine"
            assert d["n"] == 150 and d["labels"].shape == (150,)
        m = cache.read_manifest()
        assert m["batch_size"] == 32 and set(m["segments"]) == {"s0/s5", "s1/s5"} and m["spec"]["tta"] == "tent"

        # ---- 2. episodic resets between segments, continual does not ---------------------------------------
        same = [Segment("x/s5", FakeStream(150, seed=7), True), Segment("y/s5", FakeStream(150, seed=7), True)]
        c_ep = ensure_members(RunSpec("tiny", seed=0, protocol_args={"corruptions": ["x", "y"]}), ctx, same)
        assert c_ep.read("x__s5")["z_l"].equal(c_ep.read("y__s5")["z_l"]), "reset must restore source weights"
        cont = RunSpec("tiny", seed=0, protocol="continual", protocol_args={"order": 0})
        same_c = [Segment("x/s5", FakeStream(150, seed=7), True), Segment("y/s5", FakeStream(150, seed=7), False)]
        c_co = ensure_members(cont, ctx, same_c)
        assert c_co.read("x__s5")["z_l"].equal(c_ep.read("x__s5")["z_l"]), "first continual segment starts from source"
        assert not c_co.read("y__s5")["z_l"].equal(c_co.read("x__s5")["z_l"]), "continual must carry adaptation over"

        # ---- 3. resume: episodic recomputes only what is missing, continual everything ---------------------
        a_file, b_file = cache.file("s0__s5"), cache.file("s1__s5")
        a_mtime, b_before = a_file.stat().st_mtime_ns, cache.read("s1__s5")["z_l"]
        b_file.unlink()
        ensure_members(spec, ctx, two)
        assert a_file.stat().st_mtime_ns == a_mtime, "an intact episodic segment must not be recomputed"
        assert cache.read("s1__s5")["z_l"].equal(b_before), "a recomputed segment must be identical"
        mt = {p: p.stat().st_mtime_ns for p in (c_co.file("x__s5"), c_co.file("y__s5"))}
        c_co.file("y__s5").unlink()
        ensure_members(cont, ctx, same_c)
        assert c_co.file("x__s5").stat().st_mtime_ns != mt[c_co.file("x__s5")], "continual recomputes the whole stream"

        # ---- 4. replay agrees with a live pass (cache = centered fp16, live = fp32) -------------------------
        rep, live = replay(spec, ctx, gates, two), evaluate_live(spec, ctx, gates, two)
        assert [(r["segment"], r["series"]) for r in rep.rows] == [(r["segment"], r["series"]) for r in live.rows]
        for a, b in zip(rep.rows, live.rows):
            assert abs(a["accuracy"] - b["accuracy"]) <= 2 / 150 and abs(a["entropy"] - b["entropy"]) < 5e-3, (a, b)
            assert a["duo"] == "tiny" and a["tta"] == "tent" and a["seed"] == 3
        wr = [r["w_l"] for r in rep.records["ours"]]
        wl = [r["w_l"] for r in live.records["ours"]]
        assert len(wr) == len(wl) == 2 * (-(-150 // 64)) and max(abs(x - y) for x, y in zip(wr, wl)) < 5e-3
        assert any(abs(w - 0.5) > 1e-3 for w in wr), "the gate must actually move off 0.5 on this stream"
        # the fixed gate is the beta = 0 case; the gate's own records label the segment they fall in
        assert {r["corruption"] for r in rep.records["ours"]} == {"s0/s5", "s1/s5"}

        # ---- 5. report_only touches neither models nor data; an incomplete cache is an error ----------------
        class Poisoned(FakeStream):
            def torch_dataset(self):
                raise AssertionError("replay must not touch the dataset")
        poisoned = [Segment("s0/s5", Poisoned(150, 1), True), Segment("s1/s5", Poisoned(150, 2), True)]
        assert replay(spec, ctx, gates, poisoned).rows == rep.rows
        no_models = Context(ctx.g, ctx.duo, ctx.device, model_factory=lambda: (_ for _ in ()).throw(AssertionError("no models")))
        assert evaluate(spec, no_models, gates, report_only=True, segments=two).rows == rep.rows
        try:
            replay(RunSpec("tiny", seed=99), ctx, gates, two)
            raise AssertionError("an uncached run must fail in report-only mode")
        except FileNotFoundError as e:
            assert "missing" in str(e)

        # ---- 6. objective logits: the loss sees the masked 200 or the full 1000 logits ----------------------
        seen = {}
        for mode in ("masked", "full"):
            r = MemberRunner(DUO, mode="both_indep", device="cpu", objective_logits=mode, models=copy.deepcopy(base))
            orig = r.tta.loss
            r.tta.loss = lambda z, side, orig=orig, mode=mode: (seen.setdefault(mode, z.shape[1]), orig(z, side))[1]
            ds = FakeStream(64, seed=5, masked=True)
            imgs, _ = next(iter(ds.loader(32, seed=0, workers=0)))
            z_l, z_s = r.step(imgs, ds)
            assert z_l.shape == (32, 1000), "returned logits are always the full 1000"
        assert seen == {"masked": 200, "full": 1000}, seen
        try:
            MemberRunner(DUO, mode="both_duo", device="cpu", models=copy.deepcopy(base))
            raise AssertionError("duo modes must be rejected")
        except ValueError as e:
            assert "calibrator" in str(e)

        # ---- 6b. preprocessing in DataLoader workers changes nothing but speed ---------------------------------
        outs = {}
        for w in (0, 2):
            r = MemberRunner(DUO, mode="both_indep", device="cpu", models=copy.deepcopy(base))
            outs[w] = [(a, b, y) for a, b, y in r.iter_batches(FakeStream(100, seed=9), batch_size=32, workers=w,
                                                               seed=1, num_samples=None)]
        assert len(outs[0]) == len(outs[2]) == 4
        assert all(a.equal(c) and b.equal(d) and y.equal(e) for (a, b, y), (c, d, e) in zip(outs[0], outs[2])), \
            "workers must not change the logits"

        # ---- 7. a 200-class dataset is gated on its 200 classes ---------------------------------------------
        nat = [Segment("fake200", FakeStream(150, seed=4, masked=True), True)]
        nspec = RunSpec("tiny", protocol="natural", protocol_args={"dataset": "fake200"})
        res = evaluate(nspec, ctx, gates, segments=nat)
        assert all(r["n"] == 150 and 0 <= r["accuracy"] <= 1 for r in res.rows)
    print("runner self-test passed")
