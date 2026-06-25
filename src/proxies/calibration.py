"""
calibration.py
==============
Builds, saves, and loads the per-model maps that turn a raw reliability proxy
value into a predicted accuracy.

The map is a first-class artifact (CalibrationMaps), decoupled from
ProxyStats and from the proxy-stats .pt file. Each model gets one
isotonic regressor per proxy: raw_proxy_value -> predicted_accuracy. Fitting
pools batches across corruptions so the map generalises; an optional
holdout set is excluded from the fit so you can measure held-out ranking
quality.

Persistence: maps live in their own directory (DEFAULT_CALIB_DIR) and nothing
else does. One file per fitted map-set, suffixed CALIB_MAP_SUFFIX, so the
folder only ever contains calibration maps. attach() wires a loaded map-set
onto two ProxyStatss' .calib dicts.

Pipeline:
    records = collect_records(cfg_l, cfg_s, ext_l, pp_l, ext_s, pp_s, streams, device)
    maps    = fit_calibration_maps(records, ["nuclear_norm", "atc", "prototype"],
                                   cfg_l.name, cfg_s.name, holdout_corruptions={"snow"})
    save_calibration_maps(maps, "resnet50_vitb16_dev")
    ...
    maps = load_calibration_maps("resnet50_vitb16_dev")
    maps.attach(cfg_l, cfg_s)   # now cfg.predicted_acc(proxy, raw) is calibrated
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import sklearn
import torch
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

from src.proxies.proxies import ProxyStats, FeatureExtractor

__all__ = [
    "BatchRecord",
    "make_record",
    "collect_records",
    "CalibrationMaps",
    "fit_calibration_maps",
    "fit_calibration",
    "save_calibration_maps",
    "load_calibration_maps",
    "list_calibration_maps",
    "DEFAULT_CALIB_DIR",
    "CALIB_MAP_SUFFIX",
]

# Dedicated, maps-only directory + distinctive suffix so the folder is
# unambiguous: every file in it is a calibration map.
DEFAULT_CALIB_DIR = Path("data/calibration_maps")
CALIB_MAP_SUFFIX = ".calibmap.pt"


# ─── Per-batch dev record ────────────────────────────────────────────────────

@dataclass
class BatchRecord:
    corruption: str
    severity: int
    raw_l: dict[str, float]
    raw_s: dict[str, float]
    acc_l: float
    acc_s: float


@torch.no_grad()
def make_record(
    cfg_l: ProxyStats,
    cfg_s: ProxyStats,
    logits_l: torch.Tensor,
    logits_s: torch.Tensor,
    feats_l: torch.Tensor,
    feats_s: torch.Tensor,
    labels: torch.Tensor,
    corruption: str,
    severity: int,
) -> BatchRecord:
    return BatchRecord(
        corruption=corruption,
        severity=severity,
        raw_l=cfg_l.raw_proxies(logits_l, feats_l),
        raw_s=cfg_s.raw_proxies(logits_s, feats_s),
        acc_l=float((logits_l.argmax(1) == labels).float().mean()),
        acc_s=float((logits_s.argmax(1) == labels).float().mean()),
    )


# ─── Record collection ───────────────────────────────────────────────────────

@torch.no_grad()
def _collect_records(
    cfg_l, cfg_s,
    ext_l: FeatureExtractor, preprocess_l,
    ext_s: FeatureExtractor, preprocess_s,
    loader, device, corruption, severity,
) -> list[BatchRecord]:
    """One BatchRecord per batch of a single (corruption, severity) loader."""
    records = []
    for imgs, labels in tqdm(loader, desc=f"{corruption}/s{severity}"):
        xl = torch.stack([preprocess_l(img) for img in imgs]).to(device)
        xs = torch.stack([preprocess_s(img) for img in imgs]).to(device)
        zl, fl = ext_l(xl)
        zs, fs = ext_s(xs)
        records.append(make_record(
            cfg_l, cfg_s,
            zl.cpu(), zs.cpu(), fl.cpu(), fs.cpu(),
            labels,  # CPU from _pil_collate_fn
            corruption, severity,
        ))
    return records


@torch.no_grad()
def collect_records(
    cfg_l: ProxyStats,
    cfg_s: ProxyStats,
    ext_l: FeatureExtractor, preprocess_l,
    ext_s: FeatureExtractor, preprocess_s,
    streams: Iterable[tuple[str, int, object]],
    device: torch.device,
) -> list[BatchRecord]:
    """Collect records across many (corruption, severity, loader) streams.

    `streams` yields (corruption, severity, loader). Returns the flat list of
    per-batch records, ready for fit_calibration_maps.
    """
    records: list[BatchRecord] = []
    for corruption, severity, loader in streams:
        records += _collect_records(
            cfg_l, cfg_s, ext_l, preprocess_l, ext_s, preprocess_s,
            loader, device, corruption, severity,
        )
    return records


# ─── The map artifact ────────────────────────────────────────────────────────

@dataclass
class CalibrationMaps:
    """Per-model isotonic maps raw_proxy -> predicted accuracy, plus provenance."""
    large_name: str
    small_name: str
    calib_l: dict[str, IsotonicRegression] = field(default_factory=dict)
    calib_s: dict[str, IsotonicRegression] = field(default_factory=dict)
    proxy_name: str = ""
    n_fit_records: int = 0
    fit_corruptions: list[str] = field(default_factory=list)
    sklearn_version: str = sklearn.__version__

    @property
    def proxy_names(self) -> list[str]:
        return [self.proxy_name] if self.proxy_name else list(self.calib_l.keys())

    def predict_l(self, proxy: str, raw: float) -> float:
        iso = self.calib_l.get(proxy)
        return float(iso.predict([raw])[0]) if iso is not None else raw

    def predict_s(self, proxy: str, raw: float) -> float:
        iso = self.calib_s.get(proxy)
        return float(iso.predict([raw])[0]) if iso is not None else raw

    def attach(self, cfg_l: ProxyStats, cfg_s: ProxyStats) -> None:
        """Wire these maps onto two ProxyStats' .calib dicts (in place)."""
        if cfg_l.name != self.large_name or cfg_s.name != self.small_name:
            warnings.warn(
                f"calib map names ({self.large_name}, {self.small_name}) "
                f"!= stats names ({cfg_l.name}, {cfg_s.name})"
            )
        cfg_l.calib.update(self.calib_l)
        cfg_s.calib.update(self.calib_s)


# ─── Fitting ─────────────────────────────────────────────────────────────────

def fit_calibration_maps(
    records: list[BatchRecord],
    proxy_name: str,
    large_name: str,
    small_name: str,
    y_min: float = 0.0,
    y_max: float = 1.0,
    increasing: bool | str = True,
) -> CalibrationMaps:
    """Fit per-model isotonic maps raw_proxy -> accuracy for a single proxy.

    The caller is responsible for passing only the records that should be used
    for fitting (i.e. holdout filtering happens upstream, not here).

    A model with fewer than 2 distinct raw values skips fitting; predict_* /
    predicted_acc then falls back to the identity for that model.
    """
    calib_l: dict[str, IsotonicRegression] = {}
    calib_s: dict[str, IsotonicRegression] = {}
    for calib, side in ((calib_l, "l"), (calib_s, "s")):
        pairs = [
            (getattr(r, f"raw_{side}")[proxy_name], getattr(r, f"acc_{side}"))
            for r in records if proxy_name in getattr(r, f"raw_{side}")
        ]
        if len({p[0] for p in pairs}) < 2:
            continue
        xs, ys = zip(*pairs)
        iso = IsotonicRegression(
            out_of_bounds="clip", y_min=y_min, y_max=y_max, increasing=increasing
        )
        iso.fit(list(xs), list(ys))
        calib[proxy_name] = iso

    return CalibrationMaps(
        large_name=large_name,
        small_name=small_name,
        calib_l=calib_l,
        calib_s=calib_s,
        proxy_name=proxy_name,
        n_fit_records=len(records),
        fit_corruptions=sorted({r.corruption for r in records}),
    )


def fit_calibration(
    cfg_l: ProxyStats,
    cfg_s: ProxyStats,
    dev_records: list[BatchRecord],
    proxy_name: str,
) -> CalibrationMaps:
    """Convenience wrapper: fit maps and attach onto the stats in place."""
    maps = fit_calibration_maps(dev_records, proxy_name, cfg_l.name, cfg_s.name)
    maps.attach(cfg_l, cfg_s)
    return maps


# ─── Persistence (dedicated maps-only folder) ────────────────────────────────

def _resolve_path(name_or_path: str | Path, directory: str | Path) -> Path:
    """A bare name -> directory/<name><suffix>; a path with the suffix -> itself."""
    p = Path(name_or_path)
    if p.name.endswith(CALIB_MAP_SUFFIX):
        return p
    return Path(directory) / f"{p.name}{CALIB_MAP_SUFFIX}"


def save_calibration_maps(
    maps: CalibrationMaps,
    name: str,
    directory: str | Path = DEFAULT_CALIB_DIR,
) -> Path:
    """Save a map-set into the dedicated calibration-maps folder as <name>."""
    path = _resolve_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "large_name": maps.large_name,
            "small_name": maps.small_name,
            "calib_l": maps.calib_l,
            "calib_s": maps.calib_s,
            "proxy_name": maps.proxy_name,
            "n_fit_records": maps.n_fit_records,
            "fit_corruptions": maps.fit_corruptions,
            "sklearn_version": maps.sklearn_version,
        },
        path,
    )
    print(f"[calib map] saved → {path}  "
          f"(proxies={maps.proxy_names}, n={maps.n_fit_records})")
    return path


def load_calibration_maps(
    name: str | Path,
    directory: str | Path = DEFAULT_CALIB_DIR,
) -> CalibrationMaps:
    """Load a map-set by bare name (from the dedicated folder) or by full path."""
    path = _resolve_path(name, directory)
    data = torch.load(path, map_location="cpu", weights_only=False)

    saved_ver = data.get("sklearn_version")
    if saved_ver and saved_ver != sklearn.__version__:
        warnings.warn(
            f"calib map fitted with sklearn {saved_ver}, "
            f"loading under {sklearn.__version__}; unpickled regressors may differ"
        )

    maps = CalibrationMaps(
        large_name=data["large_name"],
        small_name=data["small_name"],
        calib_l=data.get("calib_l", {}),
        calib_s=data.get("calib_s", {}),
        # backward compat: old files saved proxy_names list
        proxy_name=data.get("proxy_name") or (data.get("proxy_names") or [""])[0],
        n_fit_records=data.get("n_fit_records", 0),
        fit_corruptions=data.get("fit_corruptions", []),
        sklearn_version=saved_ver or sklearn.__version__,
    )
    print(f"[calib map] loaded ← {path}  (proxies={maps.proxy_names})")
    return maps


def list_calibration_maps(directory: str | Path = DEFAULT_CALIB_DIR) -> list[str]:
    """Bare names of every map-set in the folder (the folder holds only maps)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        p.name[: -len(CALIB_MAP_SUFFIX)]
        for p in directory.glob(f"*{CALIB_MAP_SUFFIX}")
    )


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    import tempfile

    # Synthetic dev records: proxy correlated with accuracy + noise.
    rng = random.Random(0)
    recs: list[BatchRecord] = []
    for c in ("gauss", "snow", "fog"):
        for _ in range(40):
            a_l = min(max(rng.random(), 0.0), 1.0)
            a_s = min(max(rng.random(), 0.0), 1.0)
            recs.append(BatchRecord(
                corruption=c, severity=3,
                raw_l={"nuclear_norm": a_l + rng.gauss(0, 0.05), "atc": a_l},
                raw_s={"nuclear_norm": a_s + rng.gauss(0, 0.05), "atc": a_s},
                acc_l=a_l, acc_s=a_s,
            ))

    maps = fit_calibration_maps(recs, "nuclear_norm", "resnet50", "vit_b_16")
    assert maps.proxy_name == "nuclear_norm"
    assert maps.proxy_names == ["nuclear_norm"]

    with tempfile.TemporaryDirectory() as d:
        save_calibration_maps(maps, "selftest", directory=d)
        assert list_calibration_maps(d) == ["selftest"]
        reloaded = load_calibration_maps("selftest", directory=d)

    # Round-trip predictions match.
    for raw in (0.2, 0.5, 0.8):
        assert abs(maps.predict_l("nuclear_norm", raw) - reloaded.predict_l("nuclear_norm", raw)) < 1e-9
    print("calibration self-test passed")