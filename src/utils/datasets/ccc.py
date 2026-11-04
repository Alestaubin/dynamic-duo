"""
CCC (Continuously Changing Corruptions, Press et al. 2023): one 7.5M-image stream per
(difficulty, speed, seed), generated locally by third_party/CCC/generate.py from ImageNet val.

On-disk layout (what generate.py writes):
    <root>/baseline_<0|20|40>_transition+speed_<speed>_seed_<seed>/serial_00000.tar ...
Each shard is a webdataset tar of JPEG-85 224x224 images; a sample is the three members
`<key>.input.jpg`, `<key>.output.cls` (ASCII class index) and `<key>.info`. Shards hold 25,000 images
(rounded up to a multiple of the speed, so 26,000 at speed 2000).

Reading is by byte offset from a per-shard index (cached under <cache>/ccc_index/), so the stream can be
read by several DataLoader workers while the batch ORDER stays exactly the stream order: there is no
shuffling anywhere, since the temporal order of the corruptions is the point of the benchmark. It needs
neither the webdataset package nor any decompression of whole shards.

Differences from the reference third_party/CCC/eval.py: the stream is cut at exactly `images_per_stream`
(eval.py stops after the first batch that reaches it, overshooting by < one batch), and the images are
returned as PIL like every other dataset here; they are already 224x224 centre crops (`already_224`).
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils.config import GlobalConfig, REPO_ROOT
from src.utils.datasets.base import DatasetLayoutError, ShiftDataset, worker_init

BASELINE = {"easy": 40, "medium": 20, "hard": 0}   # --baseline passed to generate.py


def _identity(x):
    return x


def index_shard(path: Path) -> np.ndarray:
    """(N, 4) int64 rows (img_offset, img_size, cls_offset, cls_size), one per sample, in archive order."""
    entries: dict[str, dict] = {}
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, _, ext = m.name.partition(".")
            if ext == "input.jpg":
                entries.setdefault(key, {})["img"] = (m.offset_data, m.size)
            elif ext == "output.cls":
                entries.setdefault(key, {})["cls"] = (m.offset_data, m.size)
    rows = [(*e["img"], *e["cls"]) for e in entries.values() if "img" in e and "cls" in e]
    if not rows:
        raise DatasetLayoutError(f"{path} holds no <key>.input.jpg / <key>.output.cls samples")
    return np.asarray(rows, dtype=np.int64)


class _Stream(Dataset):
    """The whole stream as one map-style dataset of (PIL, label), in stream order."""

    def __init__(self, shards: list[Path], indexes: list[np.ndarray]):
        self.shards = shards
        self.indexes = indexes
        self.ends = np.cumsum([len(ix) for ix in indexes])   # exclusive end of each shard, in stream coordinates
        self._fds: dict[int, int] = {}                         # opened lazily, per worker process

    def __len__(self) -> int:
        return int(self.ends[-1])

    def _read(self, shard: int, offset: int, size: int) -> bytes:
        if shard not in self._fds:
            self._fds[shard] = os.open(self.shards[shard], os.O_RDONLY)
        return os.pread(self._fds[shard], size, offset)

    def __getitem__(self, i: int):
        shard = int(np.searchsorted(self.ends, i, side="right"))
        local = i - (int(self.ends[shard - 1]) if shard else 0)
        io_, is_, co, cs = self.indexes[shard][local]
        img = Image.open(io.BytesIO(self._read(shard, io_, is_))).convert("RGB")
        return img, int(self._read(shard, co, cs).decode())


class _Batches(Dataset):
    """Item b is stream samples [start + b*bs, start + (b+1)*bs): lets DataLoader workers prefetch in order. `start` (a resumed run) skips the first samples
    without reading them."""

    def __init__(self, stream: _Stream, batch_size: int, limit: int, start: int = 0):
        self.stream, self.bs, self.limit, self.start = stream, batch_size, limit, start

    def __len__(self) -> int:
        return max(0, -(-(self.limit - self.start) // self.bs))

    def __getitem__(self, b: int):
        lo = self.start + b * self.bs
        items = [self.stream[i] for i in range(lo, min(lo + self.bs, self.limit))]
        images, labels = zip(*items)
        return list(images), torch.tensor(labels)


class CCC(ShiftDataset):
    name = "ccc"
    config_key = "ccc"
    already_224 = True
    supports_start = True          # loader(start=...): a resumed live run skips what it has seen without reading it (src/reset/live_runs.py)

    def __init__(self, root, difficulty: str, speed: int, seed: int,
                 images_per_stream: int = 7_500_000, cache_dir: str | Path | None = None):
        super().__init__(root)
        if difficulty not in BASELINE:
            raise ValueError(f"difficulty must be one of {sorted(BASELINE)}, got {difficulty!r}")
        self.difficulty, self.speed, self.seed = difficulty, speed, seed
        self.images_per_stream = images_per_stream
        self.cache_dir = Path(cache_dir) if cache_dir else REPO_ROOT / "cache"

    @classmethod
    def from_config(cls, g: GlobalConfig, difficulty: str, speed: int, seed: int):
        c = g.ccc
        if speed not in c["speeds"] or seed not in c["seeds"]:
            raise ValueError(f"CCC has speeds {c['speeds']} and seeds {c['seeds']}; got speed={speed} seed={seed}")
        return cls(g.path(cls.config_key), difficulty, speed, seed,
                   images_per_stream=c["images_per_stream"], cache_dir=g.path("cache"))

    @property
    def dir_name(self) -> str:
        return f"baseline_{BASELINE[self.difficulty]}_transition+speed_{self.speed}_seed_{self.seed}"

    @property
    def folder(self) -> Path:
        return self.root / self.dir_name

    def shards(self) -> list[Path]:
        shards = sorted(self.folder.glob("serial_*.tar"))
        if not shards:
            raise DatasetLayoutError(f"no serial_*.tar shards in {self.folder} (generate the stream first)")
        numbers = [int(p.stem.split("_")[1]) for p in shards]
        if numbers != list(range(len(numbers))):
            gaps = sorted(set(range(numbers[-1] + 1)) - set(numbers))
            raise DatasetLayoutError(f"{self.folder} is missing shards (first missing: serial_{gaps[0]:05d}.tar)")
        return shards

    def _index(self, shard: Path) -> np.ndarray:
        cache = self.cache_dir / "ccc_index" / self.dir_name / f"{shard.stem}.{shard.stat().st_size}.npy"
        if cache.exists():
            return np.load(cache)
        ix = index_shard(shard)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, ix)
        return ix

    def __len__(self) -> int:
        """The stream's nominal length (cfgs/protocols.yaml), without reading any shard: callers size their runs with min(images, len); the loader checks that
        the prefix it is asked for exists (indexing every shard here would read shards that are missing or still being written)."""
        return self.images_per_stream

    def torch_dataset(self, limit: int | None = None) -> _Stream:
        """The whole stream, or (`limit`) only the shards that hold its first `limit` images, indexed in order: a prefix can be read while later shards are
        missing or still being written (a shard being written is not a valid tar yet)."""
        if limit is None:
            shards = self.shards()
            return _Stream(shards, [self._index(s) for s in shards])
        used, indexes, n, k = [], [], 0, 0
        while n < limit:
            shard = self.folder / f"serial_{k:05d}.tar"
            if not shard.is_file():
                break
            used.append(shard)
            indexes.append(self._index(shard))
            n += len(indexes[-1])
            k += 1
        if not used:
            raise DatasetLayoutError(f"no serial_00000.tar in {self.folder} (generate the stream first)")
        return _Stream(used, indexes)

    def _limited(self, num_samples: int | None) -> tuple[_Stream, int]:
        limit = self.images_per_stream if num_samples is None else num_samples
        stream = self.torch_dataset(limit)
        if len(stream) < limit:
            raise DatasetLayoutError(
                f"{self.folder} holds {len(stream):,} images in its first {len(stream.shards)} contiguous shards but {limit:,} were requested; "
                f"the stream is only partly generated (next missing: serial_{len(stream.shards):05d}.tar)."
            )
        return stream, limit

    def check(self) -> dict:
        shards = self.shards()
        first = _Stream([shards[0]], [self._index(shards[0])])
        img, label = first[0]
        if img.size != (224, 224) or not 0 <= label < 1000:
            raise DatasetLayoutError(f"{shards[0]}: first sample is {img.size} / label {label}, expected 224x224 / 0..999")
        done = f"{len(shards)} shards (first holds {len(first):,} images)"
        return {"images": None, "classes": 1000, "shards": len(shards), "note": done}

    def loader(self, batch_size: int, num_samples: int | None = None, seed: int | None = None,
               workers: int = 4, pin_memory: bool = False, shuffle: bool = False, collate_fn=None, start: int = 0) -> DataLoader:
        if shuffle:
            raise ValueError("CCC streams are ordered; shuffle=True would destroy the benchmark")
        stream, limit = self._limited(num_samples)
        # Items are already whole batches (list of PIL, labels); a custom collate_fn receives that pair.
        if not 0 <= start <= limit:
            raise ValueError(f"start {start} is outside the stream [0, {limit}]")
        return DataLoader(_Batches(stream, batch_size, limit, start), batch_size=None, shuffle=False,
                          num_workers=workers, pin_memory=pin_memory, collate_fn=collate_fn or _identity,
                          worker_init_fn=worker_init if workers > 0 else None)

    def describe(self) -> str:
        return f"ccc/{self.difficulty}/speed{self.speed}/seed{self.seed}"


if __name__ == "__main__":
    import tempfile

    # Fake stream: 3 shards of 10 / 7 / 9 samples, labels 0..25 in stream order, written the way
    # webdataset.TarWriter does. Batches of 4 straddle shard boundaries.
    sizes, n_total = [10, 7, 9], 26
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ds = CCC(tmp / "ccc", "medium", 1000, 44, images_per_stream=n_total, cache_dir=tmp / "cache")
        ds.folder.mkdir(parents=True)
        label = 0
        for s, n in enumerate(sizes):
            with tarfile.open(ds.folder / f"serial_{s:05d}.tar", "w") as tf:
                for j in range(n):
                    buf = io.BytesIO()
                    Image.new("RGB", (224, 224), (label * 9 % 256, 0, 0)).save(buf, format="JPEG", quality=95)
                    for ext, data in (("info", b"x"), ("input.jpg", buf.getvalue()), ("output.cls", str(label).encode())):
                        info = tarfile.TarInfo(f"sample_{label}.{ext}")
                        info.size = len(data)
                        tf.addfile(info, io.BytesIO(data))
                    label += 1
        assert ds.check()["shards"] == 3
        for workers in (0, 2):
            labels = [int(y) for _, ys in ds.loader(4, workers=workers) for y in ys]
            assert labels == list(range(n_total)), (workers, labels)
        batches = list(ds.loader(4, num_samples=21, workers=0))
        assert [len(b[0]) for b in batches] == [4] * 5 + [1] and batches[-1][1].tolist() == [20]
        # a resumed run: start=8 is the stream from sample 8 on, the same batches as the uninterrupted loader's from its third on (and nothing at the end)
        for workers in (0, 2):
            rest = [ys.tolist() for _, ys in ds.loader(4, num_samples=21, workers=workers, start=8)]
            assert rest == [b[1].tolist() for b in batches[2:]], (workers, rest)
        assert list(ds.loader(4, num_samples=21, workers=0, start=21)) == [] and CCC.supports_start
        # a prefix reads only the shards it needs: with the last shard missing (or still being written) the first 17 images still load, 18 do not
        last = ds.folder / "serial_00002.tar"
        last.rename(last.with_suffix(".tmp"))
        assert [int(y) for _, ys in ds.loader(4, num_samples=17, workers=0) for y in ys] == list(range(17))
        try:
            ds.loader(4, num_samples=18, workers=0)
            raise AssertionError("a prefix longer than the contiguous shards must be refused")
        except DatasetLayoutError:
            pass
        last.with_suffix(".tmp").rename(last)
        assert len(ds) == n_total, "len is the nominal length, read from no shard"
        img, y = ds.torch_dataset()[12]
        assert y == 12 and img.size == (224, 224) and abs(img.getpixel((5, 5))[0] - 12 * 9 % 256) < 8
        for bad, msg in ((lambda: ds.loader(4, num_samples=27, workers=0), "partly generated"),
                         (lambda: ds.loader(4, shuffle=True), "ordered")):
            try:
                bad()
                raise AssertionError(msg)
            except (DatasetLayoutError, ValueError) as e:
                assert msg in str(e), e
        (ds.folder / "serial_00001.tar").rename(ds.folder / "serial_00002.tar.bak")
        (ds.folder / "serial_00002.tar").rename(ds.folder / "serial_00005.tar")
        try:
            ds.shards()
            raise AssertionError("missing shard must be detected")
        except DatasetLayoutError as e:
            assert "serial_00001.tar" in str(e)
    print("ccc self-test passed")
