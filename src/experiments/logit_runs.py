"""
logit_runs.py
=============
Adapt a duo's members and put their logits in the per-member `LogitStore` (Phase 3, Stage B).

    spec  = RunSpec("vitb16_rn50v1", tta="tent", seed=0)                 # mode both_indep: the members never see each other
    ctx   = with_optim(Context.create("vitb16_rn50v1"), large=sgd_optim(5e-4), small=sgd_optim(2.5e-4))
    segs  = [Segment("speckle_noise/s5", ImageNetC.from_config(ctx.g, "speckle_noise"), True), ...]
    run   = ensure_logits(spec, ctx, segs)          # adapts, writes what is missing, returns the run manifest

It is `runner.ensure_members` with the other storage: the SAME `MemberRunner` does the adaptation (so Tent's logits are
bit for bit the ones of the old cache, checked in the self-test), but each member's streams go to their own files under a
member key (logit_store.py), staged and checksummed. A stream that a member already has is not written again, so a run
that shares a member with an earlier run (same model, algorithm, hyperparameters, seed) only adds the other member, and
any large member can be paired with any small one afterwards through `PairView`.

Both members are always run together, even when only one is missing: they share the DataLoader (one JPEG decode per image
for both), and the members are independent, so this costs the other member's forward pass and nothing else.

Checks on the way (input_checks.py): the models are normalized exactly once (`check_normalization`), the loader's order
is the emulated seeded order (so it shuffled, and `PairView` can compare members), and its batches are not single-class.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import socket
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from src.experiments.input_checks import assert_normalized_once, check_batch_diversity
from src.experiments.logit_store import LogitStore, MemberSpec, StoreError, dataset_labels, stream_order
from src.experiments.members import PairCollate
from src.experiments.protocols import Segment, build_segments
from src.experiments.runner import Context, _runner
from src.experiments.specs import RunSpec
from src.utils.config import GlobalConfig
from src.utils.datasets import ImageNetC, ImageNetTrainSlice, ImageNetVal, get_dataset


NATURAL = ("imagenet_r", "imagenet_a", "imagenet_sketch", "imagenet_v2")
STREAM_NAMES = ("heldout", "test", "clean", "clean_train_slice", *NATURAL)


def stream_segments(names: Sequence[str], g: GlobalConfig) -> list[Segment]:
    """`heldout` (the 4 held-out corruptions) and `test` (the 15 test ones) expand to ImageNet-C streams at the protocol's
    severity, `clean` is the 50,000 validation images, `clean_train_slice` the held-out clean slice of the TRAINING images
    (temperatures, source statistics; never the validation images), and a natural-shift name is one stream. Every
    stream is episodic: the members are reset before it."""
    out: list[Segment] = []
    for name in names:
        if name in ("heldout", "test"):
            cs = g.tuning_corruptions if name == "heldout" else g.test_corruptions
            out += [Segment(f"{c}/s{g.severity}", ImageNetC.from_config(g, c, g.severity), True) for c in cs]
        elif name == "clean":
            out.append(Segment("clean", ImageNetVal.from_config(g), True))
        elif name == "clean_train_slice":
            out.append(Segment("clean_train_slice", ImageNetTrainSlice.from_config(g), True))
        elif name in NATURAL:
            out.append(Segment(name, get_dataset(name, g), True))
        else:
            raise ValueError(f"unknown stream {name!r}: one of {STREAM_NAMES}")
    labels = [s.label for s in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"a stream is listed twice: {labels}")
    return out


def with_optim(ctx: Context, large: dict | None = None, small: dict | None = None) -> Context:
    """A copy of `ctx` whose members' OPTIM blocks are replaced (a grid point of a sweep). The tuned settings that
    `Context.create` overlaid stay for the side that is not replaced."""
    duo = copy.deepcopy(ctx.duo)
    for side, optim in (("LARGE", large), ("SMALL", small)):
        if optim is not None:
            duo[side]["OPTIM"] = dict(optim)
    return dataclasses.replace(ctx, duo=duo)


def run_member_specs(spec: RunSpec, ctx: Context) -> tuple[MemberSpec, MemberSpec]:
    kw = dict(batch_size=ctx.batch_size, objective_logits=ctx.objective_logits, bn=ctx.bn)
    return (MemberSpec.from_run(spec, ctx.duo, "large", **kw), MemberSpec.from_run(spec, ctx.duo, "small", **kw))


def run_id(spec: RunSpec, ctx: Context) -> str:
    ml, ms = run_member_specs(spec, ctx)
    return f"{spec.duo}__{spec.mode}__seed{spec.seed}__{ml.key}_{ms.key}"


def missing_streams(spec: RunSpec, ctx: Context, segments: Sequence[Segment], store: LogitStore
                    ) -> dict[str, list[str]]:
    """{"large": [stems not stored], "small": [...]} for the streams of `segments`."""
    ml, ms = run_member_specs(spec, ctx)
    return {side: [s.file_stem for s in segments if not store.has(m, s.file_stem)]
            for side, m in (("large", ml), ("small", ms))}


def adapt_stream(runner, seg: Segment, spec: RunSpec, ctx: Context) -> dict:
    """Source weights are the caller's business; this adapts the members over one stream and returns
    {"z_l", "z_s", "labels", "order", "batch_sizes"} (raw logits on the CPU), after checking that the loader delivered
    the seeded order (it shuffled) and that its batches are not single-class."""
    ds = seg.dataset
    loader = ds.loader(batch_size=ctx.batch_size, num_samples=spec.num_samples, seed=spec.seed, workers=ctx.g.workers,
                       pin_memory=runner.device.type == "cuda",
                       collate_fn=PairCollate(ds.input_transform(runner.large_preprocess),
                                              ds.input_transform(runner.small_preprocess)))
    zl, zs, ys, sizes = [], [], [], []
    for x_l, x_s, y in tqdm(loader, desc=f"{spec.tta} {seg.label}", leave=False):
        z_l, z_s = runner.step_tensors(x_l, x_s, ds)
        zl.append(z_l.cpu()); zs.append(z_s.cpu()); ys.append(y); sizes.append(len(y))
    labels = torch.cat(ys)

    base = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
    order = stream_order(len(base), spec.num_samples, spec.seed)
    want = dataset_labels(base)
    if want is not None and not torch.equal(want[torch.from_numpy(order).long()], labels.long()):
        raise StoreError(
            f"{seg.label}: the loader's sample order is not the seeded order that stream_order() replays (a loader that "
            f"does not shuffle, or a torch release that draws differently); members could not be checked against "
            f"each other")
    check_batch_diversity(labels, sizes)
    return {"z_l": torch.cat(zl), "z_s": torch.cat(zs), "labels": labels, "order": order, "batch_sizes": sizes}


def ensure_logits(spec: RunSpec, ctx: Context, segments: Sequence[Segment] | None = None,
                  store: LogitStore | None = None, *, compute: bool = True, check_normalization: bool = True) -> dict:
    """Adapt the members over `segments` (default: the spec's protocol) and store what is missing, per member and
    stream. Returns the run manifest (also written under `_runs/`). `compute=False` never touches a model: it raises
    FileNotFoundError, listing what is missing, if the store is incomplete (the --report_only behaviour)."""
    segments = list(segments or build_segments(spec, ctx.g))
    store = store or LogitStore.from_context(ctx)
    ml, ms = run_member_specs(spec, ctx)
    members = {"large": ml, "small": ms}
    missing = missing_streams(spec, ctx, segments, store)
    todo = [s for s in segments if s.file_stem in missing["large"] or s.file_stem in missing["small"]]
    if todo and not compute:
        raise FileNotFoundError(f"logits missing in {store.root}: large {missing['large'][:4]}, small "
                                f"{missing['small'][:4]} (run without --report_only to compute them on a GPU)")

    t0, written, n_images = time.time(), 0, 0
    if todo:
        runner = _runner(spec, ctx)
        if check_normalization:
            for side, model, pre in (("large", runner.large, runner.large_preprocess),
                                     ("small", runner.small, runner.small_preprocess)):
                for seg in todo[:1]:
                    assert_normalized_once(seg.dataset.input_transform(pre), model, f"{side} member, {seg.label}")
        for seg in todo:
            runner.reset()
            out = adapt_stream(runner, seg, spec, ctx)
            n_images += len(out["labels"])
            for side, z in (("large", out["z_l"]), ("small", out["z_s"])):
                m = members[side]
                if not store.has(m, seg.file_stem):
                    store.write(m, seg.file_stem, z, out["labels"], out["order"], out["batch_sizes"], stream=seg.label)
                    written += 1

    streams = {}
    for seg in segments:
        recs = {side: store.record(m, seg.file_stem) for side, m in members.items()}
        streams[seg.file_stem] = {"label": seg.label, **recs}
    manifest = {
        "duo": spec.duo, "tta": spec.tta, "mode": spec.mode, "seed": spec.seed, "num_samples": spec.num_samples,
        "batch_size": ctx.batch_size, "objective_logits": ctx.objective_logits, "bn": ctx.bn,
        "members": {side: {"key": m.key, "dir": str(m.relpath), **m.describe()} for side, m in members.items()},
        "streams": streams, "seconds": round(time.time() - t0, 1), "streams_computed": len(todo),
        "files_written": written, "images_adapted": n_images, "host": socket.gethostname(),
        "staging": str(store.staging) if store.staging else None,
    }
    store.write_run_manifest(run_id(spec, ctx), manifest)
    return manifest


if __name__ == "__main__":
    import tempfile

    from src.experiments._testing import FakeStream, make_context, tiny_models
    from src.experiments.grids import sgd_optim
    from src.experiments.logit_store import OrderMismatch, PairView, sha256_file
    from src.experiments.runner import ensure_members
    from src.utils.datasets.base import make_loader

    torch.set_num_threads(2)
    adam = lambda lr: {"METHOD": "Adam", "STEPS": 1, "LR": lr, "BETA": 0.9, "WD": 0.0}     # noqa: E731

    def segs(*seeds, masked=False):
        return [Segment(f"s{i}/s5", FakeStream(150, seed=sd, masked=masked), True) for i, sd in enumerate(seeds)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tiny_models(0)
        ctx = make_context(tmp, batch_size=32, base=base, bn=True)
        store = LogitStore.from_context(ctx, staging=tmp / "node")
        two = segs(1, 2)
        spec = RunSpec("tiny", seed=3)

        # ---- 1. the same logits as the old cache, bit for bit (same MemberRunner), stored per member -----------------
        old = ensure_members(spec, ctx, two)
        man = ensure_logits(spec, ctx, two, store, check_normalization=False)
        ml, ms = run_member_specs(spec, ctx)
        for seg in two:
            o, l, s = old.read(seg.file_stem), store.read(ml, seg.file_stem), store.read(ms, seg.file_stem)
            assert l["z"].equal(o["z_l"]) and s["z"].equal(o["z_s"]) and l["labels"].equal(o["labels"]), seg.label
            assert l["batch_sizes"] == [32, 32, 32, 32, 22] and l["n"] == 150 and l["stream"] == seg.label
            assert np.array_equal(l["order"], s["order"]) and sorted(l["order"].tolist()) == list(range(150))
            for m in (ml, ms):
                store.verify(m, seg.file_stem)
        assert man["files_written"] == 4 and man["streams_computed"] == 2 and set(man["streams"]) == {"s0__s5", "s1__s5"}
        assert man["members"]["large"]["key"] == ml.key and man["members"]["small"]["algorithm"] == "tent"
        assert man["streams"]["s0__s5"]["large"]["npy_sha256"] == sha256_file(store.npy(ml, "s0__s5"))
        assert store.read_run_manifest(run_id(spec, ctx))["git_commit"] and not any((tmp / "node").rglob("*.tmp*"))

        # ---- 2. resume and sharing: nothing is rewritten; a run with another small lr only adds the small member ----
        writes = []
        real_write = store.write
        store.write = lambda m, stem, *a, **k: (writes.append((m.key, stem)), real_write(m, stem, *a, **k))[1]
        again = ensure_logits(spec, ctx, two, store, check_normalization=False)
        assert writes == [] and again["streams_computed"] == 0 and again["files_written"] == 0
        mt = {p: p.stat().st_mtime_ns for p in store.dir(ml).glob("*")}
        ctx_b = with_optim(ctx, small=adam(5e-3))
        spec_b = RunSpec("tiny", seed=3)
        mlb, msb = run_member_specs(spec_b, ctx_b)
        assert mlb.key == ml.key and msb.key != ms.key
        man_b = ensure_logits(spec_b, ctx_b, two, store, check_normalization=False)
        assert sorted(writes) == sorted([(msb.key, "s0__s5"), (msb.key, "s1__s5")]), "only the new small member is written"
        assert all(p.stat().st_mtime_ns == t for p, t in mt.items()), "the shared large member is untouched"
        assert man_b["files_written"] == 2
        assert not store.read(msb, "s0__s5")["z"].equal(store.read(ms, "s0__s5")["z"]), "another lr, another logits"
        # report-only: complete stores pass, an incomplete one raises without touching a model
        ensure_logits(spec, ctx, two, store, compute=False, check_normalization=False)
        try:
            ensure_logits(spec, ctx, two + [Segment("other/s5", FakeStream(150, seed=9), True)], store, compute=False,
                          check_normalization=False)
            raise AssertionError("an incomplete store must raise with compute=False")
        except FileNotFoundError as e:
            assert "missing" in str(e)
        store.write = real_write

        # ---- 3. composition (equivalence test 4): large of run A, small of run B == the run made with that pair ------
        ctx_a, ctx_c = with_optim(ctx, large=adam(1e-3), small=adam(2e-3)), with_optim(ctx, large=adam(4e-3), small=adam(8e-3))
        ctx_d = with_optim(ctx, large=adam(1e-3), small=adam(8e-3))               # the pair (A's large, C's small)
        for c in (ctx_a, ctx_c):
            ensure_logits(spec, c, two, store, check_normalization=False)
        al, _ = run_member_specs(spec, ctx_a)
        _, cs = run_member_specs(spec, ctx_c)
        composed = PairView(store, al, cs)
        direct_store = LogitStore(tmp / "direct", staging=None)
        ensure_logits(spec, ctx_d, two, direct_store, check_normalization=False)
        dl, ds_ = run_member_specs(spec, ctx_d)
        assert (dl.key, ds_.key) == (al.key, cs.key), "the direct run has exactly the composed members' keys"
        direct = PairView(direct_store, dl, ds_)
        assert composed.stems() == direct.stems() == ["s0__s5", "s1__s5"]
        for stem in composed.stems():
            a, b = composed.read(stem), direct.read(stem)
            assert a["z_l"].equal(b["z_l"]) and a["z_s"].equal(b["z_s"]) and a["labels"].equal(b["labels"]), stem
            assert a["batch_sizes"] == b["batch_sizes"] and np.array_equal(a["order"], b["order"])
        # raw logits come back as well
        raw = composed.read("s0__s5", raw=True)
        assert raw["z_l"].shape == (150, 1000) and raw["z_l"].max(1).values.equal(store.read(al, "s0__s5")["rowmax"])

        # ---- 4. other seeds / other batches do not pair --------------------------------------------------------------
        other_seed = RunSpec("tiny", seed=4)
        ensure_logits(other_seed, ctx_c, two, store, check_normalization=False)
        try:
            PairView(store, al, run_member_specs(other_seed, ctx_c)[1])
            raise AssertionError("members with different seeds must not pair")
        except OrderMismatch:
            pass

        # ---- 5. frozen runs: the algorithm drops out of the key, --bn keeps its own ----------------------------------
        frozen = RunSpec("tiny", seed=3, mode="no_adapt")
        ensure_logits(frozen, ctx, two, store, check_normalization=False)
        fl, fs = run_member_specs(frozen, ctx)
        assert fl.algorithm == "frozen" and store.has(fl, "s0__s5") and store.has(fs, "s1__s5")
        assert not store.read(fl, "s0__s5")["z"].equal(store.read(ml, "s0__s5")["z"]), "adaptation changes the large member"
        assert run_member_specs(RunSpec("tiny", seed=3, tta="eata", mode="no_adapt"), ctx)[0].key == fl.key

        # ---- 6. a loader that does not shuffle is caught ------------------------------------------------------------
        class Unshuffled(FakeStream):
            def loader(self, batch_size, num_samples=None, seed=None, workers=4, pin_memory=False, shuffle=True,
                       collate_fn=None):
                return make_loader(self.torch_dataset(), batch_size=batch_size, shuffle=False, workers=workers,
                                   num_samples=num_samples, seed=seed, collate_fn=collate_fn)

        try:
            ensure_logits(RunSpec("tiny", seed=5), ctx, [Segment("u/s5", Unshuffled(150, seed=1), True)], store,
                          check_normalization=False)
            raise AssertionError("an unshuffled loader must be caught")
        except StoreError as e:
            assert "shuffle" in str(e)
        assert not store.has(run_member_specs(RunSpec("tiny", seed=5), ctx)[0], "u__s5"), "nothing was stored"

        # ---- 7. the normalization check runs by default and fails on the tiny models (ToTensor only) ----------------
        try:
            ensure_logits(RunSpec("tiny", seed=6), ctx, two[:1], store)
            raise AssertionError("a pipeline that normalizes zero times must raise")
        except AssertionError as e:
            assert "expected exactly 1" in str(e)

        # ---- 8. a 200-class stream stores the full 1000 logits ----------------------------------------------------
        m200 = ensure_logits(RunSpec("tiny", seed=7), ctx, segs(1, masked=True), store, check_normalization=False)
        assert store.read(run_member_specs(RunSpec("tiny", seed=7), ctx)[0], "s0__s5")["z"].shape == (150, 1000)
        assert m200["streams"]["s0__s5"]["large"]["n"] == 150

    print("logit_runs self-test passed")
