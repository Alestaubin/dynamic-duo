"""
ccc_generate.py
===============
Generates a CCC stream (Press et al. 2023) locally with the upstream generator `third_party/CCC/generate.py`, as a Slurm CPU array (one worker per task), and
times the corruptions. The upstream file is not edited: its source is executed with three changes made here, each checked to apply exactly once:

  1. the length: upstream stops at 7.5M images (`cutoff = 7500000`); `--images` cuts the stream at a prefix (the DD sprint uses the first 2M).
  2. the RNG of the images: upstream seeds NumPy and torch ONCE per worker and advances them only on the shards that worker writes, so with N workers shards
     0, 1, ..., N-1 would hold the same images (same shuffles, same crops) under different corruptions, and the stream's content would depend on N. Here both
     are reseeded from (seed, shard, step) before every step's image draw (`ApplyTransforms`) and before a shard is written, so a shard depends only on its index.
     The corruption WALK (which corruptions at which severities, in which order) uses Python's `random`, which upstream advances on every shard of every worker:
     it is upstream's, unchanged and independent of N.
  3. the frost textures come from third_party/robustness (no download); the accuracy matrix must already be in the destination (downloaded once on a login node).

    # time each corruption (ms per image at severity 3 and 5), one CPU job:
    sbatch -t 01:00:00 slurm/job_ccc.sh scripts/ccc_generate.py --timing --n 40
    # generate the first 2M images of CCC-Medium, speed 1000, seed 43, with 80 workers (one shard of 25,000 images each):
    sbatch --array=0-79 -t 12:00:00 slurm/job_ccc.sh scripts/ccc_generate.py --difficulty medium --speed 1000 --seed 43 --images 2000000 --workers 80

Output: <dest>/baseline_<b>_transition+speed_<speed>_seed_<seed>/serial_<k>.tar (the layout src/utils/datasets/ccc.py reads; set paths.ccc to <dest>).
Worker w writes shards w, w + N, w + 2N, ... below the cut. ImageNet val is copied to the node's local disk first (the generator lists every val file at each step).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CCC = REPO / "third_party" / "CCC"
FROST = REPO / "third_party" / "robustness" / "ImageNet-C" / "create_c"
FROST_FILES = ("frost1.png", "frost2.png", "frost3.png", "frost4.jpg", "frost5.jpg", "frost6.jpg")
DEFAULT_DEST = Path("/project/6104790/alxstaub/datasets/ccc")
BASELINE = {"easy": 40, "medium": 20, "hard": 0}
SPEEDS, SEEDS = (1000, 2000, 5000), (43, 44, 45)
IMGS_PER = 25000

RESEED = '''
def _dd_reseed(seed, serial, step):
    """DD patch: the images of a shard depend only on (seed, shard, step), not on the number of workers."""
    s = (int(seed) * 1_000_003 + int(serial) * 1_009 + int(step) + 7) % (2 ** 32)
    np.random.seed(s)
    torch.manual_seed(s)
'''
PATCHES = [
    ("    cutoff = 7500000  # total amount of images to be generated", "    cutoff = {images}  # DD patch: --images"),
    ("                cur_data = ApplyTransforms(data_root, n1, n2, s1, s2, speed)",
     "                _dd_reseed(seed, serial, len(current_datasets))\n                cur_data = ApplyTransforms(data_root, n1, n2, s1, s2, speed)"),
    ("                    write_dataset(concatenated_datasets, current_destination_path)",
     "                    _dd_reseed(seed, serial, -1)\n                    write_dataset(concatenated_datasets, current_destination_path)"),
    ("def GenerateDataset(", RESEED + "\n\ndef GenerateDataset("),
]


def patched_source(images: int) -> str:
    src = (CCC / "generate.py").read_text()
    for old, new in PATCHES:
        if src.count(old) != 1:
            raise SystemExit(f"generate.py changed upstream: {old.strip()[:60]!r} is not found exactly once")
        src = src.replace(old, new.format(images=images))
    return src.replace('if __name__ == "__main__":', 'if False:  # DD: driven by scripts/ccc_generate.py')


def workdir() -> Path:
    """A node-local folder with the frost textures (the corruption reads ./frost/) and, if possible, a local copy of ImageNet val."""
    base = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp") / f"ccc_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    (base / "frost").mkdir(parents=True, exist_ok=True)
    for f in FROST_FILES:
        shutil.copy2(FROST / f, base / "frost" / f)
    return base


def stage_val(val: Path, base: Path, stage: bool = False) -> Path:
    """ImageNet val, read in place (default: 80 tasks copying 6.3 GB each would read 500 GB from the shared file system; the generator lists the folder
    about 25 times per shard), or copied to the node's local disk with `--stage`."""
    if not stage or not os.environ.get("SLURM_TMPDIR"):
        return val
    local = base / "val"
    if not (local / "n15075141").is_dir():
        t0 = time.time()
        shutil.copytree(val, local, dirs_exist_ok=True)
        print(f"staged ImageNet val to {local} in {time.time() - t0:.0f} s", flush=True)
    n = sum(len(f) for _, _, f in os.walk(local))
    if n != 50000:
        raise SystemExit(f"{local} holds {n} files, expected 50,000")
    return local


def _skimage_compat() -> None:
    """Upstream calls skimage.filters.gaussian(..., multichannel=True), an argument scikit-image removed (0.19 deprecated it); `channel_axis=-1` is its exact
    replacement. Patched before make_imagenet_c imports `gaussian`."""
    import skimage.filters as skf
    if getattr(skf.gaussian, "_dd_compat", False):
        return
    orig = skf.gaussian

    def gaussian(image, *args, multichannel=None, **kwargs):
        if multichannel is not None:
            kwargs.setdefault("channel_axis", -1 if multichannel else None)
        return orig(image, *args, **kwargs)

    gaussian._dd_compat = True
    skf.gaussian = gaussian
    # NumPy 2 removed the binary mode of np.fromstring (upstream's motion blur and snow decode Wand blobs with it); np.frombuffer is the same bytes
    import numpy as np
    orig_fs = np.fromstring

    def fromstring(string, dtype=float, count=-1, *, sep="", **kw):
        if sep == "":
            return np.frombuffer(string, dtype=dtype, count=count).copy()
        return orig_fs(string, dtype=dtype, count=count, sep=sep, **kw)

    np.fromstring = fromstring
    # and the aliases NumPy 2 removed (upstream's fog uses np.float_), restored to what they always meant
    for alias, real in (("float_", "float64"), ("complex_", "complex128"), ("bool8", "bool_"), ("unicode_", "str_"), ("string_", "bytes_")):
        if alias not in np.__dict__:
            setattr(np, alias, getattr(np, real))


def load_module(images: int):
    _skimage_compat()
    sys.path.insert(0, str(CCC))
    ns: dict = {"__name__": "ccc_generate_patched", "__file__": str(CCC / "generate.py")}
    exec(compile(patched_source(images), str(CCC / "generate.py"), "exec"), ns)
    return ns


def generate(a) -> int:
    from src.utils.config import load_global

    g = load_global()
    val = Path(a.imagenet_val or g.path("imagenet_val")).resolve()
    dest = Path(a.dest)
    if not (dest / "ccc_accuracy_matrix.pickle").is_file():
        raise SystemExit(f"{dest}/ccc_accuracy_matrix.pickle is missing: download it once on a login node (compute nodes may have no internet)")
    if a.images % IMGS_PER:
        raise SystemExit(f"--images must be a multiple of {IMGS_PER} (whole shards)")
    w = a.worker if a.worker is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if not 0 <= w < a.workers:
        raise SystemExit(f"worker index {w} is not in [0, {a.workers}): pass --worker or run as an array task")
    idx = SPEEDS.index(a.speed) + 3 * SEEDS.index(a.seed)              # ccc_config.stream_config: index % 9 -> (speed, seed)
    base = workdir()
    data_root = stage_val(val, base, a.stage)
    os.chdir(base)
    ns = load_module(a.images)
    assert ns["stream_config"](9 * w + idx) == (a.speed, a.seed)
    shards = [k for k in range(a.images // IMGS_PER) if k % a.workers == w]
    print(f"CCC-{a.difficulty} (baseline {BASELINE[a.difficulty]}), speed {a.speed}, seed {a.seed}: worker {w} of {a.workers} writes shards {shards} "
          f"of the first {a.images:,} images into {dest}", flush=True)
    t0 = time.time()
    ns["GenerateDataset"](data_root=str(data_root), destination_folder=str(dest), speed=a.speed, seed=a.seed, baseline=BASELINE[a.difficulty],
                          serial_ind=w, totalprocesses=a.workers)
    out = dest / ns["dataset_name"](BASELINE[a.difficulty], a.speed, a.seed)
    done = [k for k in shards if (out / f"serial_{k:05d}.tar").is_file()]
    print(f"done in {(time.time() - t0) / 60:.1f} min: {len(done)} of {len(shards)} shards present ({out})", flush=True)
    return 0 if len(done) == len(shards) else 1


def timing(a) -> int:
    """ms per image of every corruption of CCC (severity 3 and 5) on `--n` val images: what a shard costs and which corruptions dominate."""
    from src.utils.config import load_global
    import numpy as np
    from PIL import Image
    import torchvision.transforms as trn

    base = workdir()
    os.chdir(base)
    ns = load_module(7500000)
    d = ns["noise_dict"]()
    val = Path(a.imagenet_val or load_global().path("imagenet_val"))
    paths = [sorted((val / c).iterdir())[0] for c in sorted(os.listdir(val))[:a.n]]          # one image of each of the first n classes
    crop = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomResizedCrop(224)])
    imgs = [crop(Image.open(p).convert("RGB")) for p in paths]
    rows = []
    for name, f in d.items():
        for sev in (3.0, 5.0):
            t0 = time.time()
            for im in imgs:
                x = f(im, sev)
                Image.fromarray(np.uint8(x)).save(os.devnull, format="JPEG", quality=85)
            rows.append((name, sev, 1000 * (time.time() - t0) / len(imgs)))
            print(f"  {name:20s} severity {sev:g}: {rows[-1][2]:8.1f} ms/image", flush=True)
    per = {}
    for name, _, ms in rows:
        per.setdefault(name, []).append(ms)
    mean = sum(sum(v) / len(v) for v in per.values()) / len(per)
    print(f"mean over corruptions and severities: {mean:.1f} ms/image -> one shard of {IMGS_PER:,} images with TWO corruptions applied ~ "
          f"{2 * mean * IMGS_PER / 3.6e6:.1f} h (upper bound: most steps apply both)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing", action="store_true", help="time every corruption instead of generating")
    ap.add_argument("--n", type=int, default=40, help="--timing: images per corruption and severity")
    ap.add_argument("--difficulty", choices=tuple(BASELINE), default="medium")
    ap.add_argument("--speed", type=int, choices=SPEEDS, default=1000)
    ap.add_argument("--seed", type=int, choices=SEEDS, default=43)
    ap.add_argument("--images", type=int, default=2_000_000, help="the prefix of the stream to generate (a multiple of 25,000; upstream: 7,500,000)")
    ap.add_argument("--workers", type=int, default=80, help="number of array tasks (worker w writes shards w, w + N, ...)")
    ap.add_argument("--worker", type=int, default=None, help="this worker's index (default: $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--imagenet_val", default=None, help="default: paths.imagenet_val of cfgs/global.yaml")
    ap.add_argument("--stage", action="store_true", help="copy ImageNet val to the node's local disk first (default: read it in place)")
    a = ap.parse_args()
    return timing(a) if a.timing else generate(a)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    sys.exit(main())
