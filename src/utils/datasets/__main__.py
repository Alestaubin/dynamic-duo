"""Self-test: builds fake on-disk copies of every dataset layout and checks labels, masks and ordering."""

import copy
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import torch
from PIL import Image

from src.utils.config import GlobalConfig, load_global
from src.utils.data import load_imagenetC               # the loader these classes replace
from src.utils.datasets import (DATASETS, DatasetLayoutError, ImageNetA, ImageNetC, ImageNetR, ImageNetSketch,
                                ImageNetTrainSlice, ImageNetV2, ImageNetVal, StagingUnavailable, get_dataset,
                                read_wnids, stage_dataset)
from src.utils.datasets import staging


def make_tree(root: Path, class_dirs, per_class: int, size=(8, 8)) -> None:
    for c in class_dirs:
        (root / c).mkdir(parents=True, exist_ok=True)
        for j in range(per_class):
            Image.new("RGB", size, (j * 20 % 256, 5, 5)).save(root / c / f"img_{j:03d}.JPEG")


def labels_of(loader) -> list[int]:
    return [int(y) for _, ys in loader for y in ys]


w1k, wr, wa = read_wnids("1k"), read_wnids("r"), read_wnids("a")
assert len(w1k) == 1000 and w1k == sorted(w1k) and len(wr) == 200 and len(wa) == 200
assert set(wr) <= set(w1k) and set(wa) <= set(w1k) and set(wr) != set(wa)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # --- ImageNet-C: same batches as the old load_imagenetC, for several seeds / subset sizes ---------
    make_tree(tmp / "c" / "fog" / "5", w1k, per_class=3)
    for seed in (0, 1):
        for n in (None, 37):
            old = labels_of(load_imagenetC(str(tmp / "c"), 5, "fog", torch.device("cpu"), batch_size=16,
                                           num_workers=0, num_samples=n, seed=seed))
            new = labels_of(ImageNetC(tmp / "c", "fog", 5).loader(16, num_samples=n, seed=seed, workers=0))
            assert old == new and len(new) == (n or 3000), (seed, n)
    assert labels_of(ImageNetC(tmp / "c", "fog", 5).loader(16, seed=0, workers=0)) != \
           labels_of(ImageNetC(tmp / "c", "fog", 5).loader(16, seed=1, workers=0)), "seed must change the order"
    try:
        ImageNetC(tmp / "c", "snow", 5).torch_dataset()
        raise AssertionError("missing corruption must raise")
    except DatasetLayoutError:
        pass
    assert [d.corruption for d in ImageNetC.stream(tmp / "c", ["fog", "snow"])] == ["fog", "snow"]

    # --- a missing class folder is an error, not a silent label shift --------------------------------
    make_tree(tmp / "val", w1k[:999], per_class=1)
    try:
        ImageNetVal(tmp / "val").check()
        raise AssertionError("999 classes must raise")
    except DatasetLayoutError as e:
        assert "missing 1/1000" in str(e), e
    make_tree(tmp / "val", w1k[999:], per_class=1)
    assert ImageNetVal(tmp / "val").check() == {"images": 1000, "classes": 1000}

    # --- ImageNet-R / A: 200 classes, labels follow the sorted wnid subset, mask indexes the 1k logits --
    for cls, wnids, sub in ((ImageNetR, wr, "r"), (ImageNetA, wa, "a")):
        make_tree(tmp / sub, wnids, per_class=2)
        ds = cls(tmp / sub, verify_counts=False)
        assert ds.check() == {"images": 400, "classes": 200, "where": str(tmp / sub)} and ds.num_classes == 200
        assert [w1k[i] for i in ds.class_mask] == wnids and ds.class_mask == sorted(ds.class_mask)
        logits = torch.zeros(1, 1000)
        logits[0, ds.class_mask[37]] = 1.0
        assert ds.apply_class_mask(logits).argmax().item() == 37
        first = ds.torch_dataset().samples[0]
        assert first[0].split("/")[-2] == wnids[0] and first[1] == 0
    assert ImageNetR(tmp / "r").class_mask != ImageNetA(tmp / "a").class_mask
    ds = ImageNetA(tmp / "a", verify_counts=False)
    z = torch.randn(3, 1000)
    assert ds.objective_logits(z, "masked").shape == (3, 200) and ds.objective_logits(z, "full") is z
    assert ds.objective_logits(z, "masked").equal(ds.apply_class_mask(z))
    try:
        ds.objective_logits(z, "half")
        raise AssertionError("bad mode must raise")
    except ValueError:
        pass
    # the real counts are enforced: the 400-image fake is not ImageNet-A (7,500) and is rejected by default
    for strict_ds, msg in ((ImageNetA(tmp / "a"), "expected 7,500"), (ImageNetR(tmp / "r"), "expected 30,000")):
        try:
            strict_ds.check()
            raise AssertionError("count mismatch must raise")
        except DatasetLayoutError as e:
            assert msg in str(e), e
    # a class folder outside the expected 200 means a different subset than the label mapping assumes
    (tmp / "r" / "n99999999").mkdir()
    try:
        ImageNetR(tmp / "r", verify_counts=False).torch_dataset()
        raise AssertionError("extra class folder must raise")
    except DatasetLayoutError as e:
        assert "outside the expected set" in str(e), e
    (tmp / "r" / "n99999999").rmdir()
    (tmp / "r" / "README.txt").write_text("non-directory entries are ignored")
    assert ImageNetR(tmp / "r", verify_counts=False).check()["images"] == 400

    # --- ImageNet-V2: label = int(folder name); ImageFolder would give '10' -> 2 ---------------------
    make_tree(tmp / "v2", [str(i) for i in range(1000)], per_class=1)
    ds = ImageNetV2(tmp / "v2", verify_counts=False)
    by_folder = {Path(p).parent.name: y for p, y in ds.torch_dataset().samples}
    assert by_folder["10"] == 10 and by_folder["999"] == 999 and by_folder["100"] == 100
    import torchvision
    assert torchvision.datasets.ImageFolder(str(tmp / "v2")).class_to_idx["10"] != 10, "the pitfall this class avoids"
    try:
        ImageNetV2(tmp / "v2").check()
        raise AssertionError("10 images per class is required")
    except DatasetLayoutError as e:
        assert "expected 10,000" in str(e)
    (tmp / "v2" / "7").rename(tmp / "v2" / "seven")
    try:
        ImageNetV2(tmp / "v2", verify_counts=False).check()
        raise AssertionError("bad folder names must raise")
    except DatasetLayoutError as e:
        assert "0..999" in str(e)

    # --- ImageNet-Sketch --------------------------------------------------------------------------
    make_tree(tmp / "sk", w1k, per_class=1)
    assert ImageNetSketch(tmp / "sk", verify_counts=False).check()["images"] == 1000
    try:
        ImageNetSketch(tmp / "sk").check()
        raise AssertionError("Sketch must have 50,889 images")
    except DatasetLayoutError as e:
        assert "expected 50,889" in str(e)

    # --- clean train slice: k per class, deterministic, cached --------------------------------------
    make_tree(tmp / "train", w1k, per_class=5)
    a = ImageNetTrainSlice(tmp / "train", images_per_class=2, seed=0, cache_dir=tmp / "cache")
    fl = a.file_list()
    assert len(fl) == 2000 and all(sum(y == c for _, y in fl) == 2 for c in (0, 500, 999))
    assert (tmp / "cache" / "train_slice_k2_s0.tsv").exists()
    assert ImageNetTrainSlice(tmp / "train", 2, 0, tmp / "cache2").file_list() == fl, "same seed, same slice"
    assert ImageNetTrainSlice(tmp / "train", 2, 1, tmp / "cache3").file_list() != fl, "different seed, different slice"
    assert ImageNetTrainSlice(tmp / "train", 2, 0, tmp / "cache").file_list() == fl, "cache round-trips"

    # --- staging: archives -> node-local dir -------------------------------------------------------
    for k in ("DUO_STAGING_DIR", "SLURM_TMPDIR"):
        os.environ.pop(k, None)
    shared, node = tmp / "shared", tmp / "node_local"
    shared.mkdir()
    src = tmp / "src_a" / "imagenet-a"
    make_tree(src, wa, per_class=2)
    (src / "README.txt").write_text("top-level file")

    def pack(kind: str, name: str) -> Path:
        out = shared / name
        if kind == "tar":
            subprocess.run(["tar", "-cf", out, "-C", src.parent, "imagenet-a"], check=True)
        elif kind == "tgz":
            subprocess.run(["tar", "-czf", out, "-C", src.parent, "imagenet-a"], check=True)
        else:
            with zipfile.ZipFile(out, "w") as zf:
                for f in sorted(src.rglob("*")):
                    zf.write(f, f.relative_to(src.parent))
        return out

    a_tar, a_tgz, a_zip = pack("tar", "imagenet-a.tar"), pack("tgz", "a.tar.gz"), pack("zip", "a.zip")
    shared_before = sorted(os.listdir(shared))

    # listing-based check works with NO staging directory (login node) and asserts the same things
    ds = ImageNetA(a_tar, verify_counts=False)
    assert ds.archive_backed and ds.check() == {"images": 400, "classes": 200, "where": "archive listing (not extracted)"}
    ds.verify_counts, ds.expected_images = True, 400
    assert ds.check()["images"] == 400
    ds.expected_images = 401
    try:
        ds.check()
        raise AssertionError("count mismatch must be caught from the listing")
    except DatasetLayoutError as e:
        assert "expected 401" in str(e)
    ds.expected_images = 400
    try:
        ds.root  # noqa: B018 - touching .root must stage, which is impossible outside a job
        raise AssertionError("staging without $SLURM_TMPDIR must raise")
    except StagingUnavailable as e:
        assert "SLURM_TMPDIR" in str(e)

    # inside a "job": $SLURM_TMPDIR/data is the staging dir
    os.environ["SLURM_TMPDIR"] = str(tmp / "slurm")
    assert staging.staging_root() == tmp / "slurm" / "data"
    os.environ["DUO_STAGING_DIR"] = str(node)     # explicit override wins (used for local tests)
    assert staging.staging_root() == node
    assert staging.peek_staged(a_tar) is None
    root = ds.root
    assert root == node / "imagenet-a" and (root / wa[0]).is_dir() and (root / "README.txt").is_file()
    assert staging.peek_staged(a_tar) == root and ds.check()["where"] == str(root)
    assert len(ds.torch_dataset()) == 400
    assert sorted(os.listdir(shared)) == shared_before, "nothing may be written to the shared folder"
    assert not list(node.glob(".partial_*")), "no partial extraction left behind"

    # idempotent: a second dataset object / stage_dataset call does not extract again
    calls = []
    real_extract = staging._extract
    staging._extract = lambda *a, **k: calls.append(a) or real_extract(*a, **k)
    assert ImageNetA(a_tar, verify_counts=False).root == root and not calls
    # a changed archive is re-extracted (and the stale tree replaced)
    (src / "extra.txt").write_text("changed")
    (src / wa[0] / "new_000.JPEG").write_bytes((src / wa[0] / "img_000.JPEG").read_bytes())
    os.remove(a_tar)
    pack("tar", "imagenet-a.tar")
    assert ImageNetA(a_tar, verify_counts=False).root == root and len(calls) == 1
    assert (root / wa[0] / "new_000.JPEG").is_file()
    # a crashed extraction (leftover .partial_ dir, no marker) is cleaned up and redone
    for m in node.glob(".staged_*imagenet-a.tar.json"):
        m.unlink()
    (node / ".partial_imagenet-a.tar").mkdir(exist_ok=True)
    (node / ".partial_imagenet-a.tar" / "junk").write_text("x")
    assert ImageNetA(a_tar, verify_counts=False).root == root and len(calls) == 2
    assert not list(node.glob(".partial_*"))
    staging._extract = real_extract

    # .tar.gz and .zip stage too (both unpack to the same single top-level directory name)
    for archive in (a_tgz, a_zip):
        shutil_node = tmp / f"node_{archive.suffix.strip('.')}"
        os.environ["DUO_STAGING_DIR"] = str(shutil_node)
        r = ImageNetA(archive, verify_counts=False).root
        assert r == shutil_node / "imagenet-a" and len(list(r.glob("n*/*.JPEG"))) >= 400, archive
    os.environ["DUO_STAGING_DIR"] = str(node)

    # a directory is read in place (ImageNet-R's fallback): nothing is staged
    before = sorted(os.listdir(node))
    assert ImageNetA(src, verify_counts=False).root == src and sorted(os.listdir(node)) == before
    assert sorted(os.listdir(shared)) == shared_before

    # stage_dataset(name) goes through the config: candidate list, first existing wins
    raw = copy.deepcopy(load_global().raw)
    raw["paths"]["imagenet_a"] = [str(shared / "missing.tar"), str(a_tar)]
    gg = GlobalConfig(raw, Path("<test>"))
    assert stage_dataset("imagenet_a", gg) == root
    raw["paths"]["imagenet_a"] = str(src)
    assert stage_dataset("imagenet_a", GlobalConfig(raw, Path("<test>"))) == src
    os.environ.pop("DUO_STAGING_DIR"), os.environ.pop("SLURM_TMPDIR")

    # --- registry + from_config with a throwaway global config ---------------------------------------
    raw = copy.deepcopy(load_global().raw)
    raw["paths"].update(imagenet_c=str(tmp / "c"), imagenet_val=str(tmp / "val"), imagenet_r=str(tmp / "r"),
                        imagenet_train=str(tmp / "train"), cache=str(tmp / "cache4"))
    raw["source_slice"]["images_per_class"] = 3   # the fake train tree has 5 images per class
    g = GlobalConfig(raw, Path("<test>"))
    assert get_dataset("imagenet_c", g, corruption="fog").describe().endswith("imagenet_c/fog/s5")
    assert get_dataset("imagenet_r", g, verify_counts=False).num_classes == 200
    assert len(get_dataset("imagenet_train_slice", g).torch_dataset()) == 3 * 1000
    g = GlobalConfig({**raw, "paths": {**raw["paths"], "imagenet_v2": None}}, Path("<test>"))
    try:
        get_dataset("imagenet_v2", g)
        raise AssertionError("unset path must raise")
    except RuntimeError as e:
        assert "paths.imagenet_v2" in str(e)
    assert set(DATASETS) == {"imagenet_val", "imagenet_train_slice", "imagenet_c", "imagenet_r", "imagenet_a",
                             "imagenet_sketch", "imagenet_v2", "ccc"}

print("datasets self-test passed (run `python -m src.utils.datasets.ccc` for the CCC reader)")
