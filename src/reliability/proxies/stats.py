"""
stats.py
========
Per-batch reliability proxies for heterogeneous model pairs.

Two layers feed one scoring entry point, ProxyStats.score:
  * pure.py — every proxy that is a function of one member's logits (nnn, msp,
    energy, atc, doc, cot, ...; `PROXIES`). This is the one library: the paper's
    evaluator (dd_eval.py) and the live gate (JointProxyWeighted) both use it.
    The old kind names (`nuclear_norm`, `ac_mc`) resolve to it through
    pure.LEGACY_KINDS, so configs written against them still work. (The old
    registry `atc` was the negative-entropy ATC: ask for `atc_negent`; `atc`
    is pure.py's max-confidence ATC.)
  * the Proxy registry (base.py; prototype.py, oracle.py) — only the two proxies
    a logits-only function cannot express: `prototype` reads penultimate
    features, `oracle` reads ground-truth labels.
This module provides ProxyStats, the dataclass that holds per-model SOURCE-FITTED
state (the registry proxies' state + `source_stats`, the dict the stateful pure
proxies read), FeatureExtractor for hook-based penultimate feature capture, and
build_proxy_stats for building both ProxyStats from a source dataloader.

Adding a proxy of logits alone: a function in pure.py plus its `PROXIES` entry.
Adding one that needs features or labels: a Proxy subclass in its own file,
decorated with @register, imported below — it then appears automatically in
PROXY_KINDS and in every ProxyStats.

Persistence: stats live in their own directory (DEFAULT_PROXY_DIR) and
nothing else does — one file per (large, small) pair. The calib maps that turn
a raw proxy into predicted accuracy are NOT stored here; they are a separate
artifact owned by src.reliability.calibration.maps and attached at runtime
onto .calib.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from src.reliability.proxies import pure
from src.reliability.proxies.base import Proxy, build_all, registered_names
from src.reliability.proxies.prototype import (
    PrototypeProxy,
    prototype_score,
    mahalanobis_score,
    build_prototypes,
    build_class_means,
    build_tied_precision,
)
from src.reliability.proxies.oracle import OracleProxy, oracle_score
from src.reliability.calibration.base import CalibrationMap

__all__ = [
    "PROXY_KINDS",
    "ProxyStats",
    "FeatureExtractor",
    "build_proxy_stats",
    "save_proxy_stats",
    "load_proxy_stats",
    "list_proxy_stats",
    "DEFAULT_PROXY_DIR",
    "PROXY_STATS_SUFFIX",
    "needs_source_fit",
    "PrototypeProxy", "prototype_score", "mahalanobis_score",
    "build_prototypes", "build_class_means", "build_tied_precision",
    "OracleProxy", "oracle_score",
]

# Dedicated, stats-only directory + distinctive suffix so the folder is
# unambiguous: every file in it is a proxy-stats pair.
DEFAULT_PROXY_DIR = Path("data/proxy_stats")
PROXY_STATS_SUFFIX = ".proxystats.pt"

# Every kind ProxyStats.score accepts: the registry's (prototype, oracle), pure.py's (nnn, msp, energy, cot, ...) and the
# legacy names that resolve to pure.py's (nuclear_norm, ac_mc).
PROXY_KINDS = frozenset(registered_names()) | frozenset(pure.PROXIES) | frozenset(pure.LEGACY_KINDS)

# raw_proxies() reports the kinds the registry used to, so calibration-map fitting and the sweeps see the same columns: the
# two stateless ones always, the two that used to be fitted on source data once `source_stats` exists.
_RAW_ALWAYS = ("nuclear_norm", "ac_mc")
_RAW_AFTER_FIT = ("atc_negent", "cot")


def needs_source_fit(kind: str) -> bool:
    """Whether scoring `kind` needs ProxyStats.fit_source first: prototype (class means) or a stateful pure proxy
    (energy, atc, atc_negent, doc: their source statistics)."""
    name = pure.canonical(kind)
    return kind == "prototype" or (name in pure.PROXIES and pure.PROXIES[name].stateful)


# ─── Per-model proxy stats ───────────────────────────────────────────

@dataclass
class ProxyStats:
    """Source-fitted state for computing reliability proxies on ONE model.

    Holds one Proxy instance per REGISTRY proxy kind (prototype, oracle; see
    base.py) and `source_stats`, the dict of source-slice statistics pure.py's
    stateful proxies read. Built offline from clean source data via
    fit_source(); raw_proxies() then scores a batch under every fitted proxy.
    """
    name: str
    num_classes: int
    proxies: dict[str, Proxy] = field(default_factory=build_all)
    # pure.fit_source_stats of this model's logits on the clean source slice; None until fit_source().
    source_stats: dict | None = None
    # raw proxy → predicted accuracy. Populated at runtime by
    # calibration.maps.CalibrationMaps.attach(); NOT persisted with these stats.
    calib: dict[str, CalibrationMap] = field(default_factory=dict)

    def fit_source(
        self, logits: torch.Tensor, features: torch.Tensor,
        labels: torch.Tensor, num_classes: int,
    ) -> None:
        for proxy in self.proxies.values():
            proxy.fit_source(logits, features, labels, num_classes)
        self.source_stats = pure.fit_source_stats(logits, labels)

    def raw_proxies(
        self, logits: torch.Tensor, features: torch.Tensor, labels: torch.Tensor | None = None,
    ) -> dict[str, float]:
        out = {
            name: proxy.score(logits, features, labels=labels)
            for name, proxy in self.proxies.items()
            if proxy.is_fitted and not (proxy.requires_labels and labels is None)
        }
        for name in _RAW_ALWAYS + (_RAW_AFTER_FIT if self.source_stats is not None else ()):
            out[name] = self.score(name, logits, features)
        return out

    def score(
        self, proxy_name: str, logits: torch.Tensor, features: torch.Tensor | None,
        labels: torch.Tensor | None = None, **kw,
    ) -> float:
        """One kind on one batch: a registry proxy (prototype, oracle) if it is one, else pure.py's. `kw` reaches
        the pure function (`duo_logits` for agree_duo, `seed` for cot)."""
        if proxy_name in self.proxies:
            return self.proxies[proxy_name].score(logits, features, labels=labels)
        return pure.score(proxy_name, logits, None, self.source_stats, **kw)

    def predicted_acc(self, proxy_name: str, raw_value: float) -> float:
        if proxy_name in self.calib:
            return self.calib[proxy_name].predict(raw_value)
        return raw_value


# ─── Feature extraction via forward hooks ────────────────────────────────────

# Attribute names torchvision classification models use for their final
# Linear layer: resnet/resnext/wide_resnet -> fc; efficientnet/convnext/
# densenet/mobilenet/vgg -> classifier (a Sequential ending in Linear);
# vit -> heads; swin -> head.
_CLASSIFIER_ATTRS = ("fc", "classifier", "heads", "head")


def _find_final_linear(model: nn.Module, model_name: str) -> nn.Linear:
    """Locate a model's final classification Linear layer by checking common
    torchvision attribute names, descending into a Sequential to its last
    Linear submodule if the attribute isn't a bare Linear itself."""
    for attr in _CLASSIFIER_ATTRS:
        module = getattr(model, attr, None)
        if isinstance(module, nn.Linear):
            return module
        if isinstance(module, nn.Sequential):
            linears = [m for m in module if isinstance(m, nn.Linear)]
            if linears:
                return linears[-1]
    raise ValueError(
        f"Could not locate a final Linear classifier on '{model_name}' "
        f"(checked attributes: {_CLASSIFIER_ATTRS}). Add explicit support "
        f"in _find_final_linear()."
    )


class FeatureExtractor:
    """Wraps a model; captures the penultimate feature alongside logits via a
    forward PRE-hook on the model's final Linear classifier.

    Architecture-agnostic by construction: whatever tensor a model's own
    final Linear layer consumes IS the penultimate feature — a pooled/
    flattened CNN feature map (resnet, efficientnet, convnext, densenet,
    mobilenet, ...) or a transformer's CLS token (vit, swin, ...) alike — so
    there is no need to special-case each architecture's internal pooling.
    """

    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self._feats: torch.Tensor | None = None
        linear = _find_final_linear(model, model_name)
        self._handle = linear.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        self._feats = inputs[0].detach()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        logits = self.model(x)
        return logits.detach(), self._feats

    def remove(self):
        self._handle.remove()


# ─── Source-data helpers ──────────────────────────────────────────────────────

@torch.no_grad()
def _source_pass(ext_l, preprocess_l, ext_s, preprocess_s, loader, device):
    """Run both models over the source loader; return logits, features, labels."""
    z_l, f_l, z_s, f_s, labs = [], [], [], [], []
    for imgs, labels in tqdm(loader, desc="source pass"):
        xl = torch.stack([preprocess_l(img) for img in imgs]).to(device)
        xs = torch.stack([preprocess_s(img) for img in imgs]).to(device)
        zl, fl = ext_l(xl)
        zs, fs = ext_s(xs)
        z_l.append(zl.cpu()); f_l.append(fl.cpu())
        z_s.append(zs.cpu()); f_s.append(fs.cpu())
        labs.append(labels.cpu())
    return (torch.cat(z_l), torch.cat(f_l),
            torch.cat(z_s), torch.cat(f_s),
            torch.cat(labs))


# ─── Persistence (dedicated stats-only folder) ─────────────────────────────

def _resolve_path(name_or_path: str | Path, directory: str | Path) -> Path:
    """A bare name -> directory/<name><suffix>; a path with the suffix -> itself."""
    p = Path(name_or_path)
    if p.name.endswith(PROXY_STATS_SUFFIX):
        return p
    return Path(directory) / f"{p.name}{PROXY_STATS_SUFFIX}"


def save_proxy_stats(
    cfg_l: ProxyStats,
    cfg_s: ProxyStats,
    name: str | Path,
    directory: str | Path = DEFAULT_PROXY_DIR,
) -> Path:
    """Save a (cfg_l, cfg_s) pair into the dedicated proxy-stats folder.

    Stores only source-fitted state (one state_dict() per proxy); calib maps
    are a separate artifact (see calibration.py).
    """
    path = _resolve_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "large_name":  cfg_l.name,
        "small_name":  cfg_s.name,
        "num_classes": cfg_l.num_classes,
        "proxies_l":   {n: p.state_dict() for n, p in cfg_l.proxies.items()},
        "proxies_s":   {n: p.state_dict() for n, p in cfg_s.proxies.items()},
        "source_stats_l": cfg_l.source_stats,
        "source_stats_s": cfg_s.source_stats,
    }, path)
    print(f"[proxy stats] saved → {path} (proxies={sorted(cfg_l.proxies)})")
    return path


def load_proxy_stats(
    name: str | Path,
    directory: str | Path = DEFAULT_PROXY_DIR,
) -> tuple[ProxyStats, ProxyStats]:
    """Load a (cfg_l, cfg_s) pair by bare name (from the dedicated folder) or
    by full path. .calib starts empty; attach a CalibrationMaps to populate it."""
    path = _resolve_path(name, directory)
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg_l = ProxyStats(name=data["large_name"], num_classes=data["num_classes"])
    cfg_s = ProxyStats(name=data["small_name"], num_classes=data["num_classes"])
    for cfg, key in ((cfg_l, "proxies_l"), (cfg_s, "proxies_s")):
        for proxy_name, state in data[key].items():
            if proxy_name in cfg.proxies:
                cfg.proxies[proxy_name].load_state_dict(state)
    # absent in files written before the registry's atc/cot/nuclear_norm/ac_mc were retired (build_proxy_stats warns)
    cfg_l.source_stats, cfg_s.source_stats = data.get("source_stats_l"), data.get("source_stats_s")
    print(f"[proxy stats] loaded ← {path} (proxies={sorted(data['proxies_l'])})")
    return cfg_l, cfg_s


def list_proxy_stats(directory: str | Path = DEFAULT_PROXY_DIR) -> list[str]:
    """Bare names of every stats pair in the folder (the folder holds only stats)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        p.name[: -len(PROXY_STATS_SUFFIX)]
        for p in directory.glob(f"*{PROXY_STATS_SUFFIX}")
    )


def build_proxy_stats(
    large_model: nn.Module,
    large_preprocess,
    large_name: str,
    small_model: nn.Module,
    small_preprocess,
    small_name: str,
    source_loader,
    device: torch.device,
    num_classes: int = 1000,
    proto_metric: Literal["cosine", "mahalanobis"] = "cosine",
    cov_shrinkage: float = 1e-2,
    cache_path: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_PROXY_DIR,
) -> tuple[ProxyStats, ProxyStats]:
    """Build ProxyStats for both models from clean source data.

    proto_metric selects the prototype-proxy distance built from the source
    features: "cosine" (L2-normalised class prototypes) or "mahalanobis" (raw
    class means + a tied, shrinkage-regularised precision matrix). cov_shrinkage
    is the diagonal ridge for the Mahalanobis covariance.

    `cache_path` may be a bare name (resolved into `cache_dir`) or a full path.
    If it resolves to an existing file, loads from cache and skips the source
    pass; otherwise runs the pass and, if `cache_path` is given, saves the result.

    Registers and removes feature hooks internally; the models are left unchanged.
    """
    if cache_path is not None and _resolve_path(cache_path, cache_dir).exists():
        cfg_l, cfg_s = load_proxy_stats(cache_path, cache_dir)
        if cfg_l.source_stats is None:
            warnings.warn(
                f"cached proxy stats {cache_path!r} predate pure.py's source statistics: the source-fitted kinds "
                f"(atc, cot, energy, doc, ...) cannot be scored from it. Use a different --proxy_cache name to rebuild."
            )
        cached_proto = cfg_l.proxies["prototype"]
        cached_metric = cached_proto.metric if isinstance(cached_proto, PrototypeProxy) else "cosine"
        if cached_metric != proto_metric:
            warnings.warn(
                f"cached proxy stats use proto_metric='{cached_metric}' but "
                f"proto_metric='{proto_metric}' was requested; using the cache. "
                f"Use a different --proxy_cache name to rebuild with the new metric."
            )
        return cfg_l, cfg_s

    ext_l = FeatureExtractor(large_model, large_name)
    ext_s = FeatureExtractor(small_model, small_name)
    try:
        zl, fl, zs, fs, labels = _source_pass(
            ext_l, large_preprocess, ext_s, small_preprocess, source_loader, device
        )
    finally:
        ext_l.remove()
        ext_s.remove()

    cfg_l = ProxyStats(name=large_name, num_classes=num_classes)
    cfg_s = ProxyStats(name=small_name, num_classes=num_classes)
    for cfg in (cfg_l, cfg_s):
        cfg.proxies["prototype"] = PrototypeProxy(metric=proto_metric, cov_shrinkage=cov_shrinkage)

    cfg_l.fit_source(zl, fl, labels, num_classes)
    cfg_s.fit_source(zs, fs, labels, num_classes)

    if cache_path is not None:
        save_proxy_stats(cfg_l, cfg_s, cache_path, cache_dir)

    return cfg_l, cfg_s
