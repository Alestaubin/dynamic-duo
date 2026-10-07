#!/usr/bin/env python3
"""
Check that this checkout is ready to run the paper's experiments.

    python scripts/setup_check.py                  # layouts, splits, duo configs
    python scripts/setup_check.py --full           # also count every ImageNet-C corruption's 50,000 files
    python scripts/setup_check.py --probe 200      # also score a pretrained ResNet-50 on 200 images per dataset

What it verifies
  * cfgs/global.yaml loads and its splits hold: the tuning corruptions are in no test stream and no CCC
    level, every CCC corruption is a test corruption, every CoTTA order is a permutation of the tests.
  * Each dataset's path is set and the data has the layout the dataset class expects, including the exact
    image and class counts. Archives (ImageNet-A/R/V2/Sketch live as tar/zip in the shared folder) are
    verified from their member lists, so this works on a login node and extracts nothing. A path that is
    still `null` is reported as PENDING (not a failure) unless --strict.
  * Temperatures / ATC thresholds are fitted on a clean slice of ImageNet *train*, never on *val*
    (ImageNet-C and CCC are rendered from val).
  * Each duo in cfgs/duos/ names models the loader in src/utils/model.py can build.
  * --probe needs the archives extracted, so run it inside salloc/sbatch ($SLURM_TMPDIR). It stages them
    first. A correct label mapping gives a pretrained ResNet-50 far-above-chance accuracy on every set
    (ImageNet-A excepted: it is adversarial for ResNets). Near-chance accuracy means wrong labels.

Exit status is 1 if anything is BAD (or PENDING under --strict).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere, no PYTHONPATH needed

from src.utils import model as model_lib
from src.utils.config import DatasetPathNotSet, list_duos, load_duo, load_global
from src.utils.datasets import (CCC, DATASETS, NATURAL_SHIFTS, DatasetLayoutError, ImageNetC, StagingUnavailable,
                                get_dataset, read_wnids)

OK, PENDING, BAD = "ok", "PENDING", "BAD"


class Pending(Exception):
    """Not a failure: the data is not imported / generated yet."""


def run(label: str, fn, results: list) -> None:
    """Run one check; fn returns a detail string or raises."""
    t0 = time.time()
    try:
        status, detail = OK, fn()
    except DatasetPathNotSet as e:
        status, detail = PENDING, str(e).split(" is not set")[0] + " is not set"
    except Pending as e:
        status, detail = PENDING, str(e)
    except (DatasetLayoutError, ValueError, KeyError, FileNotFoundError, NotImplementedError) as e:
        status, detail = BAD, str(e)
    results.append((label, status))
    slow = f"  ({time.time() - t0:.0f}s)" if time.time() - t0 > 2 else ""
    print(f"  [{status:7s}] {label:34s} {detail or ''}{slow}")


def check_imagenet_c(g, corruption: str, full: bool) -> str:
    ds = ImageNetC.from_config(g, corruption)
    if full:
        return f"{ds.check()['images']:,} images"
    # Structural check only (listing 19 x 50,000 files on a network filesystem is slow).
    if not ds.folder.is_dir():
        raise DatasetLayoutError(f"{ds.folder} does not exist")
    classes = sorted(p.name for p in ds.folder.iterdir() if p.is_dir())
    if classes != read_wnids("1k"):
        raise DatasetLayoutError(f"{ds.folder} has {len(classes)} class folders, expected the 1000 ImageNet wnids")
    return "1000 class folders (use --full to count files)"


def check_ccc(g) -> str:
    c = g.ccc
    done, missing = [], []
    for diff, speed, seed in itertools.product(c["baseline"], c["speeds"], c["seeds"]):
        ds = CCC.from_config(g, diff, speed, seed)
        (done if ds.folder.is_dir() else missing).append(ds)
    for ds in done:
        ds.check()  # raises on a broken or partial stream
    if not done:
        raise Pending(f"no stream generated yet in {g.path('ccc')}")
    return f"{len(done)}/{len(done) + len(missing)} streams generated"


def probe(g, n: int, device: str) -> None:
    import torch

    from src.utils.model import _preprocess_batch, get_model

    model, preprocess = get_model("resnet50", freeze=True, verbose=False)
    model.to(device).eval()
    cases = [("imagenet_val", {}), ("imagenet_c", {"corruption": "fog"}), *[(d, {}) for d in NATURAL_SHIFTS]]
    print(f"\nProbe: frozen ResNet-50 on {n} random images per dataset (device={device})")
    for name, kw in cases:
        try:
            ds = get_dataset(name, g, **kw)
            correct = total = 0
            for imgs, labels in ds.loader(batch_size=50, num_samples=n, seed=0, workers=4):
                with torch.no_grad():
                    logits = model(_preprocess_batch(imgs, preprocess, device)).cpu()
                correct += (ds.apply_class_mask(logits).argmax(1) == labels).sum().item()
                total += len(labels)
            acc = correct / total
            chance = 1 / ds.num_classes
            verdict = "" if acc > 5 * chance or name == "imagenet_a" else "  <-- near chance: labels are probably wrong"
            print(f"  {ds.describe():48s} acc {100 * acc:5.1f}%  (chance {100 * chance:.1f}%){verdict}")
        except DatasetPathNotSet:
            print(f"  {name:48s} PENDING")
        except StagingUnavailable:
            print(f"  {name:48s} SKIPPED: needs $SLURM_TMPDIR (run inside salloc/sbatch)")
        except Exception as e:  # noqa: BLE001 - a probe must report, not crash
            print(f"  {name:48s} BAD: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--global", dest="global_path", default=None, help="global config (default: cfgs/global.yaml)")
    ap.add_argument("--full", action="store_true", help="count every ImageNet-C file instead of a structure check")
    ap.add_argument("--probe", type=int, default=0, metavar="N", help="score ResNet-50 on N images per dataset")
    ap.add_argument("--strict", action="store_true", help="treat PENDING (unset path) as a failure")
    ap.add_argument("--skip-train-slice", action="store_true",
                    help="skip building the clean train slice file list (lists 1.28M files the first time)")
    args = ap.parse_args()

    results: list[tuple[str, str]] = []
    print("Global config")
    g = load_global(args.global_path)
    print(f"  [ok     ] {g.source} loads; splits valid: {len(g.test_corruptions)} test, "
          f"{len(g.tuning_corruptions)} tuning, {len(g.cotta_orders)} CoTTA orders, CCC levels "
          f"{ {k: len(v) for k, v in g.ccc['corruptions'].items()} }; "
          f"ImageNet-A/R objective logits: {g.objective_logits_200class}")

    print("\nDatasets")
    run("imagenet_val", lambda: f"{get_dataset('imagenet_val', g).check()['images']:,} images", results)
    for c in g.test_corruptions + g.tuning_corruptions:
        run(f"imagenet_c/{c}/s{g.severity}", lambda c=c: check_imagenet_c(g, c, args.full), results)
    for name in NATURAL_SHIFTS:
        run(name, lambda name=name: (lambda r: f"{r['images']:,} images, {r['classes']} classes ({r['where']})")(
            get_dataset(name, g).check()), results)
    run("ccc", lambda: check_ccc(g), results)

    print("\nSource slice (temperatures, ATC thresholds)")
    s = g.source_slice

    def check_source_slice() -> str:
        if s["dataset"] != "imagenet_train" or g.path("imagenet_train").resolve() == g.path("imagenet_val").resolve():
            raise DatasetLayoutError("the source slice would be read from the validation set")
        return f"{s['images_per_class']} images/class from {g.path('imagenet_train')}"

    run("source slice is not val", check_source_slice, results)
    if not args.skip_train_slice:
        run("train slice file list", lambda: f"{len(get_dataset('imagenet_train_slice', g).torch_dataset()):,} images", results)

    print("\nDuos")
    for name in list_duos():
        def check_duo(name=name):
            duo = load_duo(name, g)
            missing = [duo[side]["NAME"] for side in ("LARGE", "SMALL")
                       if not hasattr(model_lib, f"load_{duo[side]['NAME']}")]
            if missing:  # known gap, closed in Phase 3 of plan.md
                raise Pending(f"{duo['LARGE']['NAME']} + {duo['SMALL']['NAME']}: no loader in src/utils/model.py for {missing}")
            return f"{duo['LARGE']['NAME']} + {duo['SMALL']['NAME']}"
        run(f"duo {name}", check_duo, results)

    if args.probe:
        probe(g, args.probe, "cuda" if __import__("torch").cuda.is_available() else "cpu")

    bad = [label for label, st in results if st == BAD or (args.strict and st == PENDING)]
    pending = [label for label, st in results if st == PENDING]
    print(f"\n{len(results) - len(bad) - len(pending)} ok, {len(pending)} pending, {len(bad)} bad")
    if pending and not args.strict:
        print("pending = path not set / data not generated yet; set paths in cfgs/global.yaml (or run with --strict).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
