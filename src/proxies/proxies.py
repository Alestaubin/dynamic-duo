"""
proxies.py
==========
Per-batch reliability proxies for heterogeneous model pairs.

Provides three proxy functions (nuclear_norm, atc, prototype), the
ProxyStats dataclass that holds per-model SOURCE-FITTED state (ATC
threshold, class prototypes), FeatureExtractor for hook-based penultimate
feature capture, and build_proxy_stats for building both ProxyStats from a
source dataloader.

Persistence: stats live in their own directory (DEFAULT_PROXY_DIR) and
nothing else does — one file per (large, small) pair. The calib maps that turn
a raw proxy into predicted accuracy are NOT stored here; they are a separate
artifact owned by calibration.py and attached at runtime onto .calib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

# Dedicated, stats-only directory + distinctive suffix so the folder is
# unambiguous: every file in it is a proxy-stats pair.
DEFAULT_PROXY_DIR = Path("data/proxy_stats")
PROXY_STATS_SUFFIX = ".proxystats.pt"


# ─── Raw proxy functions ─────────────────────────────────────────────────────

@torch.no_grad()
def nuclear_norm_score(logits: torch.Tensor) -> float:
    """Confidence + dispersity via the nuclear norm of the softmax matrix.
    https://github.com/cuishuhao/BNM/blob/2d23c61f864af489d84fe5f8b66bc0a5ca51cda9/UODR/train_loader.py#L197
    """
    p = torch.softmax(logits, dim=1)
    n, c = p.shape
    nuc = torch.linalg.matrix_norm(p, ord="nuc")
    # return float(nuc / (n * min(n, c)) ** 0.5)
    return float(nuc / n)


@torch.no_grad()
def _atc_sample_scores(logits: torch.Tensor, kind: str = "neg_entropy") -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    if kind == "maxconf":
        return p.max(dim=1).values
    if kind == "neg_entropy":
        return (p * torch.log_softmax(logits, dim=1)).sum(dim=1)
    raise ValueError(kind)


@torch.no_grad()
def fit_atc_threshold(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    kind: str = "neg_entropy",
) -> float:
    """Fit ATC threshold on source data: t s.t. P(score < t) = source error rate."""
    scores = _atc_sample_scores(source_logits, kind)
    err_rate = 1.0 - (source_logits.argmax(1) == source_labels).float().mean().item()
    return torch.quantile(scores.float(), max(min(err_rate, 1.0), 0.0)).item()


@torch.no_grad()
def atc_score(logits: torch.Tensor, threshold: float, kind: str = "neg_entropy") -> float:
    """ATC predicted accuracy = fraction of samples with score >= threshold."""
    return float((_atc_sample_scores(logits, kind) >= threshold).float().mean())


@torch.no_grad()
def build_prototypes(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """L2-normalised mean penultimate feature per class, (num_classes, D).
    Empty classes get a zero vector (they are never nearest-prototype).
    """
    protos = torch.zeros(num_classes, features.shape[1], device=features.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            protos[c] = features[mask].mean(0)
    return F.normalize(protos, dim=1)


@torch.no_grad()
def prototype_score(features: torch.Tensor, prototypes: torch.Tensor) -> float:
    """Mean nearest-prototype cosine similarity.

    Cosine (not Euclidean) so 768-d ViT and 2048-d ResNet spaces are scale-comparable.
    features and prototypes must be on the same device.
    """
    f = F.normalize(features, dim=1)
    nearest = (f @ prototypes.t()).max(dim=1).values
    return float(nearest.mean())


# ─── Per-model proxy stats ───────────────────────────────────────────

@dataclass
class ProxyStats:
    """Source-fitted state for computing reliability proxies on ONE model.

    Built offline from clean source data (ATC threshold, class prototypes);
    """
    name: str
    num_classes: int
    atc_threshold: float | None = None
    prototypes: torch.Tensor | None = None
    atc_kind: str = "neg_entropy"
    # raw proxy → predicted accuracy. Populated at runtime by
    # calibration.CalibrationMaps.attach(); NOT persisted with these stats.
    calib: dict[str, IsotonicRegression] = field(default_factory=dict)

    def raw_proxies(self, logits: torch.Tensor, features: torch.Tensor) -> dict[str, float]:
        out = {"nuclear_norm": nuclear_norm_score(logits)}
        if self.atc_threshold is not None:
            out["atc"] = atc_score(logits, self.atc_threshold, self.atc_kind)
        if self.prototypes is not None:
            out["prototype"] = prototype_score(features, self.prototypes)
        return out

    def predicted_acc(self, proxy_name: str, raw_value: float) -> float:
        if proxy_name in self.calib:
            return float(self.calib[proxy_name].predict([raw_value])[0])
        return raw_value


# ─── Feature extraction via forward hooks ────────────────────────────────────

def _hook_spec(model: nn.Module, model_name: str):
    """Return (layer_to_hook, output_transform) for penultimate feature capture.

    vit_b_16  : encoder output (N, seq, dim) → CLS token (N, dim=768)
    resnet50  : avgpool output (N, C, 1, 1)  → flatten (N, C=2048)
    """
    name = model_name.lower()
    if "vit" in name:
        return model.encoder, lambda out: out[:, 0, :]
    if "resnet" in name:
        return model.avgpool, lambda out: out.flatten(1)
    raise ValueError(
        f"No feature-hook spec for model '{model_name}'. Add it to _hook_spec()."
    )


class FeatureExtractor:
    """Wraps a model; captures penultimate features alongside logits via a forward hook."""

    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self._feats: torch.Tensor | None = None
        layer, self._transform = _hook_spec(model, model_name)
        self._handle = layer.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        self._feats = self._transform(out).detach()

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

    Stores only source-fitted state (thresholds, prototypes); calib maps are a
    separate artifact (see calibration.py).
    """
    path = _resolve_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "large_name":      cfg_l.name,
        "small_name":      cfg_s.name,
        "num_classes":     cfg_l.num_classes,
        "atc_threshold_l": cfg_l.atc_threshold,
        "atc_threshold_s": cfg_s.atc_threshold,
        "atc_kind_l":      cfg_l.atc_kind,
        "atc_kind_s":      cfg_s.atc_kind,
        "prototypes_l":    cfg_l.prototypes,
        "prototypes_s":    cfg_s.prototypes,
    }, path)
    print(f"[proxy stats] saved → {path}")
    return path


def load_proxy_stats(
    name: str | Path,
    directory: str | Path = DEFAULT_PROXY_DIR,
) -> tuple[ProxyStats, ProxyStats]:
    """Load a (cfg_l, cfg_s) pair by bare name (from the dedicated folder) or
    by full path. .calib starts empty; attach a CalibrationMaps to populate it."""
    path = _resolve_path(name, directory)
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg_l = ProxyStats(
        name=data["large_name"],
        num_classes=data["num_classes"],
        atc_threshold=data["atc_threshold_l"],
        prototypes=data["prototypes_l"],
        atc_kind=data.get("atc_kind_l", "neg_entropy"),
    )
    cfg_s = ProxyStats(
        name=data["small_name"],
        num_classes=data["num_classes"],
        atc_threshold=data["atc_threshold_s"],
        prototypes=data["prototypes_s"],
        atc_kind=data.get("atc_kind_s", "neg_entropy"),
    )
    print(f"[proxy stats] loaded ← {path}")
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
    cache_path: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_PROXY_DIR,
) -> tuple[ProxyStats, ProxyStats]:
    """Build ProxyStats for both models from clean source data.

    `cache_path` may be a bare name (resolved into `cache_dir`) or a full path.
    If it resolves to an existing file, loads from cache and skips the source
    pass; otherwise runs the pass and, if `cache_path` is given, saves the result.

    Registers and removes feature hooks internally; the models are left unchanged.
    Returns (cfg_large, cfg_small) with fitted ATC thresholds and prototypes.
    """
    if cache_path is not None and _resolve_path(cache_path, cache_dir).exists():
        return load_proxy_stats(cache_path, cache_dir)

    ext_l = FeatureExtractor(large_model, large_name)
    ext_s = FeatureExtractor(small_model, small_name)
    try:
        zl, fl, zs, fs, labels = _source_pass(
            ext_l, large_preprocess, ext_s, small_preprocess, source_loader, device
        )
    finally:
        ext_l.remove()
        ext_s.remove()

    cfg_l = ProxyStats(
        name=large_name, num_classes=num_classes,
        atc_threshold=fit_atc_threshold(zl, labels),
        prototypes=build_prototypes(fl, labels, num_classes),
    )
    cfg_s = ProxyStats(
        name=small_name, num_classes=num_classes,
        atc_threshold=fit_atc_threshold(zs, labels),
        prototypes=build_prototypes(fs, labels, num_classes),
    )

    if cache_path is not None:
        save_proxy_stats(cfg_l, cfg_s, cache_path, cache_dir)

    return cfg_l, cfg_s


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    cfg_l = ProxyStats(
        name="resnet50", num_classes=10,
        atc_threshold=-0.5, prototypes=F.normalize(torch.randn(10, 2048), dim=1),
    )
    cfg_s = ProxyStats(
        name="vit_b_16", num_classes=10,
        atc_threshold=-0.7, prototypes=F.normalize(torch.randn(10, 768), dim=1),
    )

    with tempfile.TemporaryDirectory() as d:
        save_proxy_stats(cfg_l, cfg_s, "selftest", directory=d)
        assert list_proxy_stats(d) == ["selftest"]
        rl, rs = load_proxy_stats("selftest", directory=d)

    assert rl.name == "resnet50" and rs.name == "vit_b_16"
    assert rl.atc_threshold == -0.5 and rs.atc_threshold == -0.7
    assert torch.allclose(rl.prototypes, cfg_l.prototypes)
    assert torch.allclose(rs.prototypes, cfg_s.prototypes)
    assert rl.calib == {} and rs.calib == {}  # calib not persisted here
    print("proxies self-test passed")