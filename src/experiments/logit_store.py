"""
logit_store.py
==============
Per-MEMBER store of adapted logits (Phase 3, Stage B of docs/design_section45.md).

In the independent modes (`*_indep`, `no_adapt`) a member's logits depend only on that member (its model, its
algorithm and hyperparameters, whether it adapts), the stream, the seed and the batch size, never on the other member.
So they are stored per member, under a key made of exactly those inputs, and a "run" is a PAIR of member keys: the
best large member of one run can be paired offline with the best small member of another (`PairView`), and two runs
that share a member store it once. (The older `MemberCache` in cache.py stores both members per run and stays readable.)

    <root>/<model>/<algorithm>/<member_key>/member.json            what the key stands for (written once)
    <root>/<model>/<algorithm>/<member_key>/<stream>.npy           (N, 1000) float16, logits minus their row maximum
    <root>/<model>/<algorithm>/<member_key>/<stream>.meta.npz      labels, sample order, batch sizes, row maximum, ...
    <root>/_runs/<run_id>.json                                     one manifest per run: config, git hash, seed, sha256s

`<stream>.meta.npz` is renamed into place LAST and holds the sha256 of the `.npy`, so a stream exists exactly when its
meta file does; there is no shared file that two jobs could race on (an array may fill different streams of one member).
The `.npy` is centered like cache.py's (the largest logits stay near 0 where fp16 is finest) and the float32 row maximum
is kept in the meta file, so the raw logits come back as `centered + rowmax` (energy scores need them: logsumexp is not
invariant to a per-row constant). Raw logits are exact to the fp16 rounding of the centered values.

Writes are staged: with $SLURM_TMPDIR set, a stream is first written on the node, then copied to the store, the copy's
sha256 is compared with the staged file's, and only then is it renamed and the meta file written. A job that dies
between streams therefore loses at most the stream in flight, and a copy that went wrong is never recorded.

The sample order (dataset indices, in the order the loader delivered them) is stored so that members from different runs
can be checked to have seen the same batches before they are paired. It is computed by `stream_order`, which replays
make_loader's seeded subset and shuffle, and the writer checks it against the labels the loader really delivered.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.experiments.cache import git_commit
from src.experiments.specs import DUO_MODES, INDEPENDENT_PROTOCOLS, INPUTS_TAG, MODE_SIDES, RunSpec

FORMAT_VERSION = 1


class StoreError(RuntimeError):
    """A file in the store is missing, truncated or corrupt, or a write could not be verified."""


class OrderMismatch(RuntimeError):
    """Two members were not run on the same batches, so their logits cannot be paired."""


# --------------------------------------------------------------------------------------------- member identity

@dataclass(frozen=True)
class MemberSpec:
    """Everything that determines ONE member's logits on a stream (the stream itself is the file name).

    A member that does not adapt ("frozen") has no algorithm, kwargs or OPTIM in its key: they cannot change its logits,
    so two frozen runs that differ only in them share their files. `frozen_bn` ("batch" | "source") is set only for a
    frozen BatchNorm member, the one case where the --bn flag changes the logits.
    """

    model: dict                       # the duo's side entry: NAME, NORM, OPTIM
    algorithm: str                    # TTA method name, or "frozen"
    kwargs: dict
    adapting: bool
    seed: int
    num_samples: int | None
    batch_size: int
    objective_logits: str = "masked"
    inputs: str = INPUTS_TAG
    frozen_bn: str | None = None

    @classmethod
    def from_run(cls, spec: RunSpec, duo_cfg: dict, side: str, *, batch_size: int, objective_logits: str = "masked",
                 bn: bool = False) -> "MemberSpec":
        """The member of `side` ("large" | "small") in a run. Only runs whose members are independent qualify."""
        if spec.mode in DUO_MODES:
            raise ValueError(f"mode {spec.mode!r} adapts on the duo's entropy, so a member's logits depend on the other "
                             f"one and cannot be stored per member")
        if spec.protocol not in INDEPENDENT_PROTOCOLS:
            raise ValueError(f"protocol {spec.protocol!r} carries adaptation across streams: a member's stream cannot "
                             f"be stored on its own (episodic, tuning and natural only)")
        if side not in ("large", "small"):
            raise ValueError(f"side must be 'large' or 'small', got {side!r}")
        adapting = MODE_SIDES[spec.mode][0 if side == "large" else 1]
        cfg = duo_cfg[side.upper()]
        frozen_bn = (("batch" if bn else "source") if (not adapting and cfg.get("NORM") == "BN") else None)
        return cls(model={k: cfg[k] for k in ("NAME", "NORM", "OPTIM")} if adapting else
                   {k: cfg[k] for k in ("NAME", "NORM")},
                   algorithm=spec.tta if adapting else "frozen", kwargs=dict(spec.tta_kwargs) if adapting else {},
                   adapting=adapting, seed=spec.seed, num_samples=spec.num_samples, batch_size=batch_size,
                   objective_logits=objective_logits, frozen_bn=frozen_bn)

    def payload(self) -> dict:
        d = {"model": self.model, "algorithm": self.algorithm, "kwargs": self.kwargs, "adapting": self.adapting,
             "seed": self.seed, "num_samples": self.num_samples, "batch_size": self.batch_size,
             "objective_logits": self.objective_logits, "inputs": self.inputs}
        if self.frozen_bn is not None:
            d["frozen_bn"] = self.frozen_bn
        return d

    @property
    def key(self) -> str:
        return hashlib.sha1(json.dumps(self.payload(), sort_keys=True, default=str).encode()).hexdigest()[:12]

    @property
    def relpath(self) -> Path:
        return Path(self.model["NAME"]) / self.algorithm / self.key

    def describe(self) -> dict:
        return {**self.payload(), "key": self.key, "format_version": FORMAT_VERSION}


# --------------------------------------------------------------------------------------------- order and labels

def stream_order(dataset_len: int, num_samples: int | None, seed: int) -> np.ndarray:
    """Dataset indices in the order `make_loader(..., shuffle=True, seed=seed)` delivers them (int32).

    Replays its two seeded draws: the random subset (only with `num_samples`), then the loader's shuffle, which comes
    AFTER the DataLoader has drawn its worker base seed from the same generator. The writer checks this against the
    labels the loader delivered, so a torch release that orders differently is an error, not a silent mismatch."""
    idx = torch.arange(dataset_len)
    if num_samples is not None:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(dataset_len, generator=gen)[:min(num_samples, dataset_len)]
    gen = torch.Generator().manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=gen)
    return idx[torch.randperm(len(idx), generator=gen)].numpy().astype(np.int32)


def dataset_labels(ds) -> torch.Tensor | None:
    """Labels of a map-style dataset in index order, without decoding images (None if it does not expose them)."""
    if hasattr(ds, "samples"):
        return torch.tensor([label for _, label in ds.samples])
    if hasattr(ds, "labels"):
        return torch.as_tensor(list(ds.labels))
    return None


# --------------------------------------------------------------------------------------------- files

def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


class LogitStore:
    def __init__(self, root: str | Path, staging: str | Path | None = "auto"):
        """`staging`: a node-local directory to write streams in first ("auto": $SLURM_TMPDIR/logit_stage if that is
        set, else none, and the stream is written straight into the store)."""
        self.root = Path(root)
        if staging == "auto":
            tmp = os.environ.get("SLURM_TMPDIR")
            staging = Path(tmp) / "logit_stage" if tmp else None
        self.staging = Path(staging) if staging else None

    @classmethod
    def from_context(cls, ctx, staging: str | Path | None = "auto") -> "LogitStore":
        return cls(ctx.g.path("cache") / "logits", staging)

    # ----------------------------------------------------------------------------- paths
    def dir(self, m: MemberSpec) -> Path:
        return self.root / m.relpath

    def npy(self, m: MemberSpec, stem: str) -> Path:
        return self.dir(m) / f"{stem}.npy"

    def meta(self, m: MemberSpec, stem: str) -> Path:
        return self.dir(m) / f"{stem}.meta.npz"

    # ----------------------------------------------------------------------------- queries
    def has(self, m: MemberSpec, stem: str) -> bool:
        """The stream is complete: its meta file exists and the `.npy` it describes has the recorded size."""
        meta, npy = self.meta(m, stem), self.npy(m, stem)
        if not (meta.is_file() and npy.is_file()):
            return False
        try:
            with np.load(meta, allow_pickle=False) as z:
                return int(z["npy_bytes"]) == npy.stat().st_size
        except (OSError, ValueError, KeyError):
            return False

    def stems(self, m: MemberSpec) -> list[str]:
        d = self.dir(m)
        if not d.is_dir():
            return []
        return sorted(p.name[:-len(".meta.npz")] for p in d.glob("*.meta.npz") if self.has(m, p.name[:-len(".meta.npz")]))

    # ----------------------------------------------------------------------------- write
    def write(self, m: MemberSpec, stem: str, z: torch.Tensor, labels: torch.Tensor, order: np.ndarray,
              batch_sizes: Sequence[int], stream: str | None = None) -> dict:
        """Store one stream of one member. `z`: (N, K) raw logits; `order`: dataset indices as delivered;
        `batch_sizes`: the loader's batch sizes (they sum to N). Returns the file record (sha256s, bytes)."""
        t0 = time.time()
        n, k = z.shape
        if not (len(labels) == len(order) == n and sum(batch_sizes) == n):
            raise ValueError(f"length mismatch: {n} logits, {len(labels)} labels, {len(order)} order entries, "
                             f"batches sum to {sum(batch_sizes)}")
        z = z.detach().float().cpu()
        if not torch.isfinite(z).all():
            raise ValueError("non-finite logits cannot be stored")
        rowmax = z.max(dim=1).values
        centered = (z - rowmax[:, None]).half().numpy()

        d = self.dir(m)
        d.mkdir(parents=True, exist_ok=True)
        member_json = d / "member.json"
        if not member_json.exists():
            _atomic_json(member_json, m.describe())
        tag = f".tmp{os.getpid()}"
        final_npy, final_meta = self.npy(m, stem), self.meta(m, stem)
        final_meta.unlink(missing_ok=True)            # while the stream is being replaced it does not exist

        work = (self.staging / m.key) if self.staging else d
        work.mkdir(parents=True, exist_ok=True)
        staged = work / f"{stem}.npy{tag}"
        with open(staged, "wb") as f:
            np.save(f, centered)
        digest = sha256_file(staged)
        if self.staging:
            landing = d / f"{stem}.npy{tag}"
            try:
                shutil.copyfile(staged, landing)
                if sha256_file(landing) != digest:
                    raise StoreError(f"copy-out of {stem} to {d} failed its checksum")
            except BaseException:
                landing.unlink(missing_ok=True)
                raise
            finally:
                staged.unlink(missing_ok=True)
        else:
            landing = staged
        os.replace(landing, final_npy)

        meta_tmp = d / f"{stem}.meta.tmp{os.getpid()}.npz"
        with open(meta_tmp, "wb") as f:
            np.savez(f, labels=labels.cpu().to(torch.int16).numpy(), order=np.asarray(order, dtype=np.int32),
                     batch_sizes=np.asarray(batch_sizes, dtype=np.int32), rowmax=rowmax.numpy().astype(np.float32),
                     npy_sha256=np.array(digest), npy_bytes=np.array(final_npy.stat().st_size), n=np.array(n),
                     k=np.array(k), stream=np.array(stream or stem), member_key=np.array(m.key),
                     format_version=np.array(FORMAT_VERSION))
        os.replace(meta_tmp, final_meta)                # last: the stream now exists
        return {"npy_sha256": digest, "npy_bytes": final_npy.stat().st_size, "meta_sha256": sha256_file(final_meta),
                "n": n, "seconds": round(time.time() - t0, 2)}

    # ----------------------------------------------------------------------------- read
    def read(self, m: MemberSpec, stem: str, raw: bool = False) -> dict:
        """{"z": (N, K) float32 (centered, or raw with raw=True), "labels": int64, "order": int32, "batch_sizes": list,
        "rowmax": float32, "n": int, "stream": str}"""
        if not self.has(m, stem):
            raise StoreError(f"stream {stem!r} of member {m.key} is not in {self.dir(m)}")
        with np.load(self.meta(m, stem), allow_pickle=False) as meta:
            meta = {k: meta[k] for k in meta.files}
        z = torch.from_numpy(np.load(self.npy(m, stem), mmap_mode="r").astype(np.float32))
        rowmax = torch.from_numpy(meta["rowmax"])
        if raw:
            z = z + rowmax[:, None]
        return {"z": z, "labels": torch.from_numpy(meta["labels"].astype(np.int64)), "order": meta["order"],
                "batch_sizes": meta["batch_sizes"].tolist(), "rowmax": rowmax, "n": int(meta["n"]),
                "stream": str(meta["stream"])}

    def verify(self, m: MemberSpec, stem: str) -> dict:
        """Recompute the `.npy`'s sha256 and compare it with the one recorded when it was written."""
        with np.load(self.meta(m, stem), allow_pickle=False) as meta:
            want = str(meta["npy_sha256"])
        got = sha256_file(self.npy(m, stem))
        if got != want:
            raise StoreError(f"{self.npy(m, stem)} does not match its recorded sha256")
        return {"npy_sha256": got, "meta_sha256": sha256_file(self.meta(m, stem)),
                "npy_bytes": self.npy(m, stem).stat().st_size}

    def record(self, m: MemberSpec, stem: str) -> dict:
        """The file record of a stored stream, for a run manifest (reads the meta file, hashes it; the `.npy`'s hash is the
        one recorded at write time, `verify` recomputes it)."""
        with np.load(self.meta(m, stem), allow_pickle=False) as meta:
            rec = {"npy_sha256": str(meta["npy_sha256"]), "npy_bytes": int(meta["npy_bytes"]), "n": int(meta["n"])}
        rec["meta_sha256"] = sha256_file(self.meta(m, stem))
        return rec

    # ----------------------------------------------------------------------------- run manifests
    def write_run_manifest(self, run_id: str, payload: dict) -> Path:
        d = self.root / "_runs"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{run_id}.json"
        _atomic_json(path, {**payload, "run_id": run_id, "git_commit": git_commit(),
                            "written": time.strftime("%Y-%m-%d %H:%M:%S")})
        return path

    def read_run_manifest(self, run_id: str) -> dict:
        return json.loads((self.root / "_runs" / f"{run_id}.json").read_text())


# --------------------------------------------------------------------------------------------- pairing

def check_aligned(a: dict, b: dict, stem: str = "") -> None:
    """Raise OrderMismatch unless two members' streams were delivered in the same order, in the same batches."""
    where = f" (stream {stem})" if stem else ""
    if a["n"] != b["n"] or not np.array_equal(a["order"], b["order"]):
        raise OrderMismatch(f"the members saw different sample orders{where}: n={a['n']} vs {b['n']}")
    if not torch.equal(a["labels"], b["labels"]):
        raise OrderMismatch(f"the members' labels differ{where} although the orders agree: a corrupt or foreign file")
    if list(a["batch_sizes"]) != list(b["batch_sizes"]):
        raise OrderMismatch(f"the members were run with different batch boundaries{where}")


@dataclass
class PairView:
    """A large and a small member read from a store as one pair, whichever runs they came from.

    `read(stem)` returns what the replay engine needs: {"z_l", "z_s", "labels", "n", "batch_sizes", "order"}, after
    checking that the two members saw exactly the same samples in the same batches."""

    store: LogitStore
    large: MemberSpec
    small: MemberSpec
    streams: list[str] = field(default_factory=list)

    def __post_init__(self):
        for attr in ("seed", "num_samples", "batch_size"):
            if getattr(self.large, attr) != getattr(self.small, attr):
                raise OrderMismatch(f"members differ in {attr}: {getattr(self.large, attr)!r} vs "
                                    f"{getattr(self.small, attr)!r}; their batches cannot match")

    def stems(self) -> list[str]:
        s = set(self.store.stems(self.large)) & set(self.store.stems(self.small))
        return sorted(s & set(self.streams)) if self.streams else sorted(s)

    def read(self, stem: str, raw: bool = False) -> dict:
        a, b = self.store.read(self.large, stem, raw), self.store.read(self.small, stem, raw)
        check_aligned(a, b, stem)
        return {"z_l": a["z"], "z_s": b["z"], "labels": a["labels"], "n": a["n"], "batch_sizes": a["batch_sizes"],
                "order": a["order"], "stream": a["stream"]}


if __name__ == "__main__":
    import tempfile

    from torch.utils.data import Dataset

    from src.experiments.cache import center_fp16
    from src.utils.datasets.base import make_loader

    torch.manual_seed(0)
    OPT = {"METHOD": "SGD", "STEPS": 1, "LR": 1e-3, "MOMENTUM": 0.9, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}
    duo = {"LARGE": {"NAME": "vit", "NORM": "LN", "OPTIM": OPT}, "SMALL": {"NAME": "rn", "NORM": "BN", "OPTIM": OPT}}

    def member(spec, side, **kw):
        return MemberSpec.from_run(spec, duo, side, batch_size=64, **kw)

    # ---- member keys: what is in the key, and what is not ------------------------------------------------------
    run = RunSpec("d", seed=0)
    kl, ks = member(run, "large"), member(run, "small")
    assert kl.key != ks.key and kl.relpath == Path("vit/tent") / kl.key
    other_s = {**duo, "SMALL": {**duo["SMALL"], "OPTIM": {**OPT, "LR": 2e-3}}}
    assert MemberSpec.from_run(run, other_s, "large", batch_size=64).key == kl.key, "the other member's lr is not in the key"
    assert MemberSpec.from_run(run, other_s, "small", batch_size=64).key != ks.key
    assert member(RunSpec("d", seed=1), "large").key != kl.key and member(RunSpec("d", seed=0, num_samples=100), "large").key != kl.key
    assert member(RunSpec("d", tta="eata"), "large").key != kl.key
    assert member(RunSpec("d", mode="large_indep"), "large").key == kl.key, "the large member adapts in both modes"
    assert member(RunSpec("d", mode="large_indep"), "small").key != ks.key and member(RunSpec("d", mode="large_indep"), "small").algorithm == "frozen"
    # a frozen member forgets its algorithm and optimizer; --bn only matters for a frozen BatchNorm member
    f1, f2 = member(RunSpec("d", mode="no_adapt"), "small"), member(RunSpec("d", tta="eata", mode="no_adapt"), "small")
    assert f1.key == f2.key and f1.model == {"NAME": "rn", "NORM": "BN"} and f1.frozen_bn == "source"
    assert member(RunSpec("d", mode="no_adapt"), "small", bn=True).key != f1.key
    assert member(RunSpec("d", mode="no_adapt"), "large", bn=True).key == member(RunSpec("d", mode="no_adapt"), "large").key
    assert member(RunSpec("d", mode="no_adapt"), "large").frozen_bn is None
    for bad in (RunSpec("d", mode="both_duo", loop_ts=(1.0, 1.0)), RunSpec("d", protocol="continual", protocol_args={"order": 0})):
        try:
            member(bad, "large")
            raise AssertionError("must refuse")
        except ValueError:
            pass

    # ---- write / read / checksums / atomicity -----------------------------------------------------------------
    n = 300
    z = torch.randn(n, 1000) * 2.0
    z[torch.arange(n), torch.randint(0, 1000, (n,))] += 16.0
    z[7] += 40.0                                           # a row with a big offset
    labels = torch.randint(0, 1000, (n,))
    order = np.random.RandomState(0).permutation(n).astype(np.int32)
    sizes = [64] * 4 + [44]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        store = LogitStore(tmp / "logits", staging=None)
        assert not store.has(kl, "fog__s5") and store.stems(kl) == []
        rec = store.write(kl, "fog__s5", z, labels, order, sizes, stream="fog/s5")
        assert store.has(kl, "fog__s5") and store.stems(kl) == ["fog__s5"]
        assert rec["npy_sha256"] == sha256_file(store.npy(kl, "fog__s5")) == store.verify(kl, "fog__s5")["npy_sha256"]
        assert not list((tmp / "logits").rglob("*.tmp*")) and (store.dir(kl) / "member.json").is_file()
        assert json.loads((store.dir(kl) / "member.json").read_text())["key"] == kl.key
        d = store.read(kl, "fog__s5")
        assert d["z"].equal(center_fp16(z).float()), "the centered values are cache.py's"
        assert d["z"].max(1).values.eq(0).all() and d["labels"].equal(labels) and np.array_equal(d["order"], order)
        assert d["batch_sizes"] == sizes and d["stream"] == "fog/s5" and d["n"] == n
        raw = store.read(kl, "fog__s5", raw=True)["z"]
        assert (raw - z).abs().max() < 0.03, (raw - z).abs().max()        # fp16 rounding of centered values up to ~|40|
        assert abs(raw.logsumexp(1)[7] - z.logsumexp(1)[7]) < 0.03, "raw logits keep their absolute scale (energy)"
        assert raw.argmax(1).equal(z.argmax(1)) and (raw.max(1).values - z.max(1).values).abs().max() < 1e-5, "row maximum is exact"
        # corruption is caught: by the checksum, and by the size check that `has` does on every look
        p = store.npy(kl, "fog__s5")
        before = p.read_bytes()
        p.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
        try:
            store.verify(kl, "fog__s5")
            raise AssertionError("a flipped byte must fail verify")
        except StoreError:
            pass
        assert store.has(kl, "fog__s5"), "same size: only verify can see it"
        p.write_bytes(before[:-100])
        assert not store.has(kl, "fog__s5"), "a truncated file is not a stream"
        p.write_bytes(before)
        assert store.has(kl, "fog__s5") and store.verify(kl, "fog__s5")
        # rewriting a stream replaces it
        store.write(kl, "fog__s5", z + 1.0, labels, order, sizes)
        assert (store.read(kl, "fog__s5", raw=True)["z"] - (z + 1.0)).abs().max() < 0.03
        for bad in (dict(labels=labels[:10]), dict(batch_sizes=[64])):
            try:
                store.write(kl, "bad", z, **{"labels": labels, "order": order, "batch_sizes": sizes, **bad})
                raise AssertionError("length mismatch must raise")
            except ValueError:
                pass
        assert not store.has(kl, "bad")
        try:
            store.write(kl, "nan", torch.full((2, 3), float("nan")), labels[:2], order[:2], [2])
            raise AssertionError("non-finite must raise")
        except ValueError:
            pass

        # ---- staging: copy out, verify, then record -------------------------------------------------------------
        stage = tmp / "node_local"
        staged_store = LogitStore(tmp / "logits2", staging=stage)
        staged_store.write(ks, "fog__s5", z, labels, order, sizes)
        assert staged_store.has(ks, "fog__s5") and not any(stage.rglob("*.tmp*")), "the staged copy is removed"
        assert staged_store.verify(ks, "fog__s5")["npy_bytes"] == staged_store.npy(ks, "fog__s5").stat().st_size
        real_copy = shutil.copyfile

        def bad_copy(src, dst, *a, **k):
            out = real_copy(src, dst, *a, **k)
            with open(dst, "r+b") as f:
                f.seek(200)
                f.write(b"\xff")
            return out

        shutil.copyfile = bad_copy
        try:
            staged_store.write(ks, "snow__s5", z, labels, order, sizes)
            raise AssertionError("a bad copy must raise")
        except StoreError:
            pass
        finally:
            shutil.copyfile = real_copy
        assert not staged_store.has(ks, "snow__s5") and not list((tmp / "logits2").rglob("snow*")), \
            "a copy that failed its checksum leaves nothing that looks like a stream"
        assert not any(stage.rglob("snow*")), "and nothing on the node either"

        # ---- pairing: alignment is checked, from data ------------------------------------------------------------
        store.write(ks, "fog__s5", z * 0.5, labels, order, sizes)
        pv = PairView(store, kl, ks)
        pair = pv.read("fog__s5")
        assert pv.stems() == ["fog__s5"] and pair["z_l"].shape == (n, 1000) and pair["n"] == n
        store.write(ks, "snow__s5", z, labels, order, sizes)
        store.write(kl, "snow__s5", z, labels, order[::-1].copy(), sizes)       # same labels, another order
        try:
            pv.read("snow__s5")
            raise AssertionError("different sample orders must not pair")
        except OrderMismatch:
            pass
        store.write(kl, "contrast__s5", z, labels, order, sizes)
        store.write(ks, "contrast__s5", z, labels, order, [100, 200])
        try:
            pv.read("contrast__s5")
            raise AssertionError("different batch boundaries must not pair")
        except OrderMismatch:
            pass
        for attr, other in (("seed", RunSpec("d", seed=1)), ("num_samples", RunSpec("d", num_samples=50))):
            try:
                PairView(store, kl, member(other, "small"))
                raise AssertionError(f"members with another {attr} must not pair")
            except OrderMismatch:
                pass

        # ---- run manifest ----------------------------------------------------------------------------------------
        mp = store.write_run_manifest("r1", {"members": {"large": kl.key}})
        assert store.read_run_manifest("r1")["run_id"] == "r1" and "git_commit" in store.read_run_manifest("r1") and mp.is_file()

    # ---- stream_order against the real DataLoader --------------------------------------------------------------
    class Index(Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return i, i

    def delivered(n, num_samples, seed, workers):
        dl = make_loader(Index(n), batch_size=64, workers=workers, num_samples=num_samples, seed=seed,
                         collate_fn=lambda b: torch.tensor([i for i, _ in b]))
        return torch.cat(list(dl)).numpy()

    for n_, cap, seed_, w in ((1000, None, 0, 0), (1000, None, 7, 2), (1000, 300, 0, 0), (1000, 300, 3, 2), (200, 500, 1, 0)):
        got = delivered(n_, cap, seed_, w)
        assert np.array_equal(got, stream_order(n_, cap, seed_)), (n_, cap, seed_, w)
    assert not np.array_equal(stream_order(1000, None, 0), stream_order(1000, None, 1))
    assert not np.array_equal(stream_order(1000, None, 0), np.arange(1000)), "the loader shuffles"
    print("logit_store self-test passed")
