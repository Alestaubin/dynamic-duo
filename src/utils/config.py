"""
config.py
=========
Loads the repo-wide configuration (cfgs/global.yaml) and the per-duo model configs (cfgs/duos/*.yaml).

    g = load_global()                     # honours $DUO_GLOBAL_CONFIG
    g.path("imagenet_c")                  # Path, resolved against the repo root
    g.test_corruptions                    # the 15 ImageNet-C test corruptions, in table-column order
    duo = load_duo("vitb16_rn50")         # LARGE/SMALL + the legacy keys src/tta still reads

A dataset path set to `null` in global.yaml means "not imported yet": `g.path(key)` then raises
DatasetPathNotSet naming the key to fill in, instead of failing later with a confusing IO error.
A path may also be a LIST of candidates (first one that exists wins), e.g. a tarball with the
extracted directory as fallback, or a `.tar` vs `.tar.gz` file name that differs between copies.
A path may point at a directory or at an archive (.tar/.tar.gz/.zip); archives are extracted to
node-local storage on first use (see src/utils/datasets/staging.py).

Validation (run on every load) enforces the splits the paper relies on: the tuning corruptions
appear in no test stream and in no CCC level, every CCC corruption is one of the test corruptions,
and each CoTTA order is a permutation of the test corruptions.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_VAR = "DUO_GLOBAL_CONFIG"
DEFAULT_GLOBAL = REPO_ROOT / "cfgs" / "global.yaml"
DUO_DIR = REPO_ROOT / "cfgs" / "duos"


class DatasetPathNotSet(RuntimeError):
    """A path in cfgs/global.yaml that the caller needs is still `null`."""


def _resolve(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def _dupes(xs: list) -> list:
    seen: set = set()
    return sorted({x for x in xs if x in seen or seen.add(x)}, key=str)


class GlobalConfig:
    def __init__(self, raw: dict, source: Path):
        self.raw = raw
        self.source = source
        self._validate()

    # --- paths ------------------------------------------------------------ #
    def path_or_none(self, key: str) -> Path | None:
        if key not in self.raw["paths"]:
            raise KeyError(f"unknown path key {key!r}; known: {sorted(self.raw['paths'])}")
        value = self.raw["paths"][key]
        if value is None:
            return None
        candidates = [_resolve(c) for c in ([value] if isinstance(value, (str, Path)) else value)]
        for c in candidates:
            if c.exists():
                return c
        if len(candidates) == 1:
            return candidates[0]  # the dataset class reports the missing path with its own message
        raise FileNotFoundError(f"none of the candidates for paths.{key} exists: {[str(c) for c in candidates]}")

    def path(self, key: str) -> Path:
        p = self.path_or_none(key)
        if p is None:
            raise DatasetPathNotSet(
                f"paths.{key} is not set (null) in {self.source}. Import the dataset and set that key; "
                f"the expected layout is documented next to it."
            )
        return p

    # --- protocol constants ----------------------------------------------- #
    @property
    def severity(self) -> int:
        return self.raw["protocol"]["severity"]

    @property
    def batch_size(self) -> int:
        return self.raw["protocol"]["adaptation_batch_size"]

    @property
    def seeds(self) -> list[int]:
        return list(self.raw["protocol"]["seeds"])

    @property
    def samples_per_stream(self) -> int | None:
        return self.raw["protocol"]["samples_per_stream"]

    @property
    def ece_bins(self) -> int:
        return self.raw["protocol"]["ece_bins"]

    @property
    def objective_logits_200class(self) -> str:
        """"masked": the TTA objective (e.g. entropy) sees the 200-way logits on ImageNet-A/R, as in the
        reference benchmark. "full": it sees all 1000 logits. Accuracy/ECE always use the masked logits,
        and the same setting applies to every compared method."""
        return self.raw["protocol"]["objective_logits_200class"]

    @property
    def workers(self) -> int:
        return self.raw["runtime"]["workers"]

    @property
    def gate(self) -> dict:
        return dict(self.raw["gate"])

    @property
    def source_slice(self) -> dict:
        return dict(self.raw["source_slice"])

    # --- corruptions ------------------------------------------------------ #
    @property
    def corruption_families(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self.raw["corruptions"]["test"].items()}

    @property
    def test_corruptions(self) -> list[str]:
        return [c for fam in self.raw["corruptions"]["test"].values() for c in fam]

    @property
    def tuning_corruptions(self) -> list[str]:
        return list(self.raw["corruptions"]["tuning"])

    @property
    def collapse_corruptions(self) -> list[str]:
        return list(self.raw["corruptions"]["collapse"])

    # --- generated protocol facts (cfgs/protocols.yaml) -------------------- #
    @functools.cached_property
    def protocols(self) -> dict:
        return yaml.safe_load(_resolve(self.raw["protocols_file"]).read_text())

    @property
    def cotta_orders(self) -> list[list[str]]:
        return [list(o) for o in self.protocols["cotta_orders"]]

    @property
    def ccc(self) -> dict:
        return self.protocols["ccc"]

    # --- validation ------------------------------------------------------- #
    def _validate(self) -> None:
        test, tuning = self.test_corruptions, self.tuning_corruptions
        errors = []
        for name, xs in (("corruptions.test", test), ("corruptions.tuning", tuning)):
            if _dupes(xs):
                errors.append(f"{name} has duplicate entries {_dupes(xs)} (a duplicate silently re-runs that "
                              f"corruption as a second pass)")
        if self.objective_logits_200class not in ("masked", "full"):
            errors.append(f"protocol.objective_logits_200class must be 'masked' or 'full', "
                          f"got {self.objective_logits_200class!r}")
        if set(test) & set(tuning):
            errors.append(f"tuning corruptions {sorted(set(test) & set(tuning))} are also test corruptions")
        if not set(self.collapse_corruptions) <= set(test):
            errors.append(f"collapse corruptions {sorted(set(self.collapse_corruptions) - set(test))} are not test corruptions")
        for level, names in self.ccc["corruptions"].items():
            if set(names) & set(tuning):
                errors.append(f"tuning corruptions {sorted(set(names) & set(tuning))} appear in CCC-{level}")
            if not set(names) <= set(test):
                errors.append(f"CCC-{level} uses {sorted(set(names) - set(test))}, which are not test corruptions")
        for i, order in enumerate(self.cotta_orders):
            if sorted(order) != sorted(test):
                errors.append(f"CoTTA order {i} is not a permutation of corruptions.test")
        if errors:
            raise ValueError(f"invalid {self.source}:\n  - " + "\n  - ".join(errors))


@functools.lru_cache(maxsize=None)
def _load(path: str) -> GlobalConfig:
    p = Path(path)
    return GlobalConfig(yaml.safe_load(p.read_text()), p)


def load_global(path: str | Path | None = None) -> GlobalConfig:
    """The repo-wide config. Precedence: explicit `path` > $DUO_GLOBAL_CONFIG > cfgs/global.yaml."""
    chosen = path or os.environ.get(ENV_VAR) or DEFAULT_GLOBAL
    return _load(str(_resolve(chosen)))


def load_duo(name_or_path: str | Path, global_cfg: GlobalConfig | None = None) -> dict:
    """A duo config (LARGE/SMALL) merged with the global values under the key names the existing
    src/tta engine reads (TEST_DIR, VAL_DIR, BS, WORKERS, EVAL, CALIBRATOR). `name_or_path` is a file
    in cfgs/duos/ (with or without .yaml) or a path to a duo YAML."""
    g = global_cfg or load_global()
    p = Path(name_or_path)
    if not p.suffix:
        p = DUO_DIR / f"{p.name}.yaml"
    elif not p.is_absolute() and not p.exists():
        p = DUO_DIR / p.name
    duo = yaml.safe_load(_resolve(p).read_text())
    for side in ("LARGE", "SMALL"):
        if side not in duo or "NAME" not in duo[side] or "NORM" not in duo[side]:
            raise ValueError(f"{p}: {side} must define NAME and NORM")
    sev = [g.severity]
    duo.update(
        TEST_DIR=str(g.path_or_none("imagenet_c") or ""),
        VAL_DIR=str(g.path_or_none("imagenet_val") or ""),
        BS=g.batch_size,
        WORKERS=g.workers,
        EVAL={"CORRUPTIONS": g.test_corruptions, "SEVERITIES": sev},
        CALIBRATOR={"CORRUPTIONS": g.tuning_corruptions, "SEVERITIES": sev},
    )
    return duo


def list_duos() -> list[str]:
    return sorted(p.stem for p in DUO_DIR.glob("*.yaml"))


if __name__ == "__main__":
    import copy
    import tempfile

    g = load_global()
    assert g.source == DEFAULT_GLOBAL or os.environ.get(ENV_VAR)
    assert len(g.test_corruptions) == 15 and len(g.tuning_corruptions) == 4
    assert len(g.cotta_orders) == 10 and set(g.ccc["corruptions"]) == {"easy", "medium", "hard"}
    assert g.path("imagenet_c").name == "ImageNet-C"
    assert g.objective_logits_200class in ("masked", "full")
    try:
        GlobalConfig({**g.raw, "paths": {**g.raw["paths"], "imagenet_r": None}}, g.source).path("imagenet_r")
        raise AssertionError("an unset dataset path must raise")
    except DatasetPathNotSet as e:
        assert "paths.imagenet_r" in str(e)
    # candidate lists: first existing wins; a single missing path is returned as is; all-missing raises
    here = Path(__file__)
    cand = GlobalConfig({**g.raw, "paths": {**g.raw["paths"], "x": ["/nonexistent/a.tar", str(here)]}}, g.source)
    assert cand.path("x") == here
    single = GlobalConfig({**g.raw, "paths": {**g.raw["paths"], "x": "/nonexistent/a.tar"}}, g.source)
    assert single.path("x") == Path("/nonexistent/a.tar")
    none = GlobalConfig({**g.raw, "paths": {**g.raw["paths"], "x": ["/nonexistent/a", "/nonexistent/b"]}}, g.source)
    try:
        none.path("x")
        raise AssertionError("all-missing candidates must raise")
    except FileNotFoundError as e:
        assert "/nonexistent/b" in str(e)

    for name in list_duos():
        duo = load_duo(name)
        assert duo["EVAL"]["CORRUPTIONS"] == g.test_corruptions and duo["BS"] == 64, name
        assert set(duo["EVAL"]["CORRUPTIONS"]).isdisjoint(duo["CALIBRATOR"]["CORRUPTIONS"]), name

    # The validator must reject each way the splits can silently break.
    def broken(mutate) -> str:
        raw = copy.deepcopy(g.raw)
        mutate(raw)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(raw, f)
        try:
            GlobalConfig(raw, Path(f.name))
        except ValueError as e:
            return str(e)
        raise AssertionError("expected ValueError")

    assert "also test corruptions" in broken(lambda r: r["corruptions"]["tuning"].append("fog"))
    assert "duplicate" in broken(lambda r: r["corruptions"]["test"]["noise"].append("shot_noise"))
    assert "not test corruptions" in broken(lambda r: r["corruptions"]["collapse"].append("spatter"))
    assert "masked" in broken(lambda r: r["protocol"].update(objective_logits_200class="half"))
    print("config self-test passed")
