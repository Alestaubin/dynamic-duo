"""
staging.py
==========
Extract dataset archives to node-local storage before reading them.

The natural-shift datasets live as archives in the lab's shared folder, which has a file-count quota, so
they are NEVER extracted there. Each SLURM job (and each job-array task) gets its own $SLURM_TMPDIR on the
node's local SSD, wiped when the job ends; the first time a dataset is used in a job it is extracted to
$SLURM_TMPDIR/data and read from there.

    stage_dataset("imagenet_a")      # key of `paths:` in cfgs/global.yaml -> directory to read from
    stage_archive(Path(...tar))      # same, for an explicit archive path

  * A path that is already a directory is returned as is (e.g. ImageNet-R's extracted directory when no
    imagenet-r.tar exists yet): nothing to stage.
  * Extraction is idempotent: a marker records which archive (path, size, mtime) was extracted, so a
    second call in the same job is free, and a changed archive is re-extracted.
  * Extraction goes to a `.partial_*` directory and is renamed into place, so a job that dies mid-extract
    never leaves a half-populated dataset that looks complete. A file lock serializes processes of one job.
  * `tar -xf` (no `z`: GNU tar detects compression, and imagenetv2-matched-frequency.tar.gz is in fact an
    uncompressed tar) and `unzip -q`, as in the cluster notes.

The staging directory is $DUO_STAGING_DIR if set (local tests), else $SLURM_TMPDIR/data. Outside a job
there is deliberately no default (a login node's /tmp is small and the repo is on a quota'd filesystem):
StagingUnavailable says what to do. Listing an archive's contents needs no staging, which is how
`scripts/setup_check.py` verifies the data from a login node.

    python -c 'from src.utils.datasets.staging import main; main()' imagenet_a imagenet_r   # what slurm/job.sh runs first
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".zip")


class StagingUnavailable(RuntimeError):
    """No node-local staging directory: not running inside a SLURM job/allocation."""


def is_archive(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and p.name.lower().endswith(ARCHIVE_SUFFIXES)


def staging_root() -> Path:
    override = os.environ.get("DUO_STAGING_DIR")
    if override:
        return Path(override)
    tmp = os.environ.get("SLURM_TMPDIR")
    if tmp:
        return Path(tmp) / "data"
    raise StagingUnavailable(
        "no node-local staging directory: $SLURM_TMPDIR is not set. Run inside a SLURM job or an salloc "
        "allocation (each gets its own $SLURM_TMPDIR), or set DUO_STAGING_DIR for a local test. "
        "Never extract into the shared dataset folder (file-count quota)."
    )


def list_archive_files(archive: str | Path) -> list[str]:
    """Names of all non-directory members, without extracting anything."""
    archive = Path(archive)
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            return [i.filename for i in zf.infolist() if not i.is_dir()]
    with tarfile.open(archive, "r:*") as tf:
        return [m.name for m in tf if m.isfile()]


def _slug(archive: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", archive.name)


def _stamp(archive: Path) -> dict:
    st = archive.stat()
    return {"archive": str(archive), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _read_marker(marker: Path) -> dict | None:
    try:
        return json.loads(marker.read_text())
    except (OSError, ValueError):
        return None


def peek_staged(archive: str | Path) -> Path | None:
    """The staged directory of `archive` if it was already extracted in this job, else None. Never extracts."""
    archive = Path(archive).resolve()
    try:
        dest = staging_root()
    except StagingUnavailable:
        return None
    meta = _read_marker(dest / f".staged_{_slug(archive)}.json")
    stamp = _stamp(archive)
    if meta and all(meta.get(k) == v for k, v in stamp.items()) and (dest / meta["top"]).is_dir():
        return dest / meta["top"]
    return None


@contextlib.contextmanager
def _locked(path: Path):
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _extract(archive: Path, into: Path) -> None:
    if archive.name.lower().endswith(".zip"):
        if shutil.which("unzip"):
            # exit status 1 = "completed with warnings", which is fine
            r = subprocess.run(["unzip", "-q", str(archive), "-d", str(into)])
            if r.returncode not in (0, 1):
                raise subprocess.CalledProcessError(r.returncode, r.args)
        else:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(into)
    else:
        subprocess.run(["tar", "-xf", str(archive), "-C", str(into)], check=True)


def stage_archive(archive: str | Path) -> Path:
    """Extract `archive` into the staging directory (once) and return its single top-level directory."""
    archive = Path(archive).resolve()
    if not is_archive(archive):
        raise FileNotFoundError(f"{archive} is not an existing archive ({', '.join(ARCHIVE_SUFFIXES)})")
    dest = staging_root()
    dest.mkdir(parents=True, exist_ok=True)
    slug = _slug(archive)
    with _locked(dest / f".lock_{slug}"):
        staged = peek_staged(archive)
        if staged is not None:
            return staged
        marker = dest / f".staged_{slug}.json"
        marker.unlink(missing_ok=True)
        partial = dest / f".partial_{slug}"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir()
        t0 = time.time()
        _extract(archive, partial)
        tops = [p for p in partial.iterdir() if p.is_dir()]
        if len(tops) != 1:
            raise RuntimeError(f"{archive.name} should hold exactly one top-level directory, found {[p.name for p in tops]}")
        top = tops[0].name
        if (dest / top).exists():
            shutil.rmtree(dest / top)
        os.replace(tops[0], dest / top)
        shutil.rmtree(partial)
        marker.write_text(json.dumps({**_stamp(archive), "top": top}))
        print(f"[stage] {archive.name} -> {dest / top} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return dest / top


def stage_dataset(name: str, g=None) -> Path:
    """Directory to read dataset `name` (a key of `paths:` in cfgs/global.yaml) from, staging it if it is an archive."""
    from src.utils.config import load_global

    src = (g or load_global()).path(name)
    return stage_archive(src) if is_archive(src) else src


def main(names: list[str] | None = None) -> None:
    """CLI used by slurm/job.sh: stage the given dataset keys (default: the four natural shifts)."""
    from src.utils.config import load_global

    g = load_global()
    for n in names or sys.argv[1:] or ["imagenet_a", "imagenet_r", "imagenet_v2", "imagenet_sketch"]:
        print(f"{n}: {stage_dataset(n, g)}")


if __name__ == "__main__":
    main()
