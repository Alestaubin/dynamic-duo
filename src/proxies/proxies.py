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

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

# Dedicated, stats-only directory + distinctive suffix so the folder is
# unambiguous: every file in it is a proxy-stats pair.
DEFAULT_PROXY_DIR = Path("data/proxy_stats")
PROXY_STATS_SUFFIX = ".proxystats.pt"


def _logit(p: float, eps: float = 1e-6) -> float:
    """Scalar log-odds, clamped so p=0/1 don't blow up."""
    import math
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


# ─── Raw proxy functions ─────────────────────────────────────────────────────

@torch.no_grad()
def nuclear_norm_score(logits: torch.Tensor) -> float:
    """Confidence + dispersity via the nuclear norm of the softmax matrix.
    https://github.com/cuishuhao/BNM/blob/2d23c61f864af489d84fe5f8b66bc0a5ca51cda9/UODR/train_loader.py#L197
    https://arxiv.org/pdf/2302.01094
    """
    p = torch.softmax(logits, dim=1)
    n, c = p.shape
    nuc = torch.linalg.matrix_norm(p, ord="nuc")
    return float(nuc / (n * min(n, c)) ** 0.5)
    # return float(nuc / n)


class RunningNuclearNorm:
    """Recency-weighted cumulative nuclear norm of the softmax prediction matrix
    — a less noisy reliability proxy than the single-batch score, that can also
    track a model whose reliability is drifting as it adapts.

    The full nuclear norm needs only the running k×k Gram matrix G = Σ_b P_bᵀ P_b,
    not the stored (n_total × k) prediction matrix, because
        ‖P‖_* = Σ_i σ_i(P) = Σ_i √λ_i(PᵀP) = Σ_i √λ_i(G).
    score() applies the same bounded normalisation as nuclear_norm_score, now
    over all pooled data, so single-batch noise averages out.

    `decay` (λ ∈ [0, 1]) exponentially down-weights older batches in both the
    Gram matrix and the effective sample count:
        G_t = (1 - λ) · G_{t-1} + P_tᵀ P_t
        n_t = (1 - λ) · n_{t-1} + b_t
    λ=0 reproduces the original cumulative (unweighted) average exactly; λ=1
    keeps only the latest batch (equivalent to nuclear_norm_score of it).

    Stateful: update() once per batch, reset() at each stream boundary
    (e.g. per corruption).
    """

    def __init__(self, decay: float = 0.0) -> None:
        assert 0.0 <= decay <= 1.0, f"decay must be in [0, 1], got {decay}"
        self.decay = decay
        self._gram: torch.Tensor | None = None
        self._n: float = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor) -> None:
        p = torch.softmax(logits, dim=1)
        batch_gram = p.t() @ p
        b = p.shape[0]
        if self._gram is None:
            self._gram = batch_gram
            self._n = float(b)
        else:
            lam = self.decay
            self._gram = (1.0 - lam) * self._gram + batch_gram
            self._n = (1.0 - lam) * self._n + b

    @torch.no_grad()
    def score(self, logits: torch.Tensor | None = None) -> float:
        """Recency-weighted bounded nuclear norm over all updated batches. Before
        the first update, falls back to the single-batch score of `logits`."""
        if self._gram is None:
            return nuclear_norm_score(logits) if logits is not None else float("nan")
        c = self._gram.shape[0]
        nuc = torch.linalg.eigvalsh(self._gram).clamp_min(0).sqrt().sum()
        return float(nuc / (self._n * min(self._n, c)) ** 0.5)

    def reset(self) -> None:
        self._gram = None
        self._n = 0.0


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


@torch.no_grad()
def build_class_means(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (un-normalised) mean penultimate feature per PRESENT class.

    Returns (means, class_ids): means is (C_present, D), class_ids the matching
    class indices. Empty classes are dropped — unlike cosine (where a zero row
    is simply never nearest), a zero mean would be a spurious attractor under
    Mahalanobis.
    """
    means, present = [], []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            present.append(c)
            means.append(features[mask].mean(0))
    return torch.stack(means), torch.tensor(present, dtype=torch.long)


@torch.no_grad()
def build_tied_precision(
    features: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
    class_ids: torch.Tensor,
    shrinkage: float = 1e-2,
) -> torch.Tensor:
    """Inverse of the tied (shared-across-classes) within-class covariance.

    Pools per-class-centred deviations across all samples — the standard
    Mahalanobis-OOD estimator (Lee et al. 2018) — then diagonally shrinks for
    invertibility at D up to 2048:  Σ ← Σ + shrinkage · mean(diag Σ) · I.
    """
    D = features.shape[1]
    row_of = torch.full((int(labels.max()) + 1,), -1, dtype=torch.long)
    row_of[class_ids] = torch.arange(len(class_ids))
    rows = row_of[labels]
    valid = rows >= 0                                        # drop dropped-class samples
    centred = features[valid] - means[rows[valid]]          # (N, D)
    cov = (centred.t() @ centred) / centred.shape[0]        # (D, D)
    ridge = shrinkage * cov.diagonal().mean()
    cov = cov + ridge * torch.eye(D, dtype=cov.dtype, device=cov.device)
    return torch.linalg.inv(cov)


@torch.no_grad()
def mahalanobis_score(
    features: torch.Tensor, class_means: torch.Tensor, precision: torch.Tensor
) -> float:
    """NEGATIVE mean nearest-class squared Mahalanobis distance.

    Negated so that, like cosine, HIGHER = closer to the source class clusters =
    more reliable — keeping the proxy's "higher is better" convention and the
    isotonic calibration's increasing=True assumption.

    Uses dᴹ² = xᵀPx − 2 xᵀPμ + μᵀPμ to avoid materialising the (N, C, D)
    difference tensor. features / class_means / precision share a device.

    Unlike cosine, raw magnitude is NOT cross-model comparable (squared
    Mahalanobis is ~χ² with df = feature dim, so 768-d and 2048-d differ in
    scale); rely on the calibration map for cross-model selection.
    """
    fp = features @ precision                               # (N, D)
    term_x = (fp * features).sum(1, keepdim=True)           # (N, 1)
    mp = class_means @ precision                            # (C, D)
    term_m = (mp * class_means).sum(1)                      # (C,)
    cross = fp @ class_means.t()                            # (N, C)
    d2 = term_x - 2.0 * cross + term_m.unsqueeze(0)         # (N, C)
    nearest = d2.min(dim=1).values                         # (N,)
    return float((-nearest).mean())


# ─── Per-model proxy stats ───────────────────────────────────────────

@dataclass
class ProxyStats:
    """Source-fitted state for computing reliability proxies on ONE model.

    Built offline from clean source data: ATC threshold, and — for the prototype
    proxy — either L2-normalised class prototypes (cosine metric) or raw class
    means + a tied precision matrix (Mahalanobis metric), selected by proto_metric.
    """
    name: str
    num_classes: int
    atc_threshold: float | None = None
    prototypes: torch.Tensor | None = None          # (C, D) L2-normalised, cosine metric
    class_means: torch.Tensor | None = None         # (C', D) raw means, Mahalanobis metric
    precision: torch.Tensor | None = None           # (D, D) tied Σ⁻¹, Mahalanobis metric
    proto_metric: Literal["cosine", "mahalanobis"] = "cosine"
    atc_kind: str = "neg_entropy"
    # raw proxy → predicted accuracy. Populated at runtime by
    # calibration.CalibrationMaps.attach(); NOT persisted with these stats.
    calib: dict[str, IsotonicRegression] = field(default_factory=dict)
    # Validation-set reliability prior + per-proxy scale, used by
    # JointProxyWeighted (soft weighting). Populated by fit_val_reliability();
    # NOT persisted with these stats (same treatment as .calib).
    val_acc: float | None = None
    proxy_mean: dict[str, float] = field(default_factory=dict)
    proxy_std: dict[str, float] = field(default_factory=dict)

    def _has_prototype_state(self) -> bool:
        if self.proto_metric == "mahalanobis":
            return self.class_means is not None and self.precision is not None
        return self.prototypes is not None

    def prototype_proxy(self, features: torch.Tensor) -> float:
        """Nearest-class-prototype reliability score under the configured metric
        (higher = closer to the source clusters). Lazily moves the stored source
        state onto the features' device, caching it in place."""
        dev = features.device
        if self.proto_metric == "mahalanobis":
            assert self.class_means is not None and self.precision is not None, \
                "mahalanobis proto_metric needs class_means + precision " \
                "(build_proxy_stats(..., proto_metric='mahalanobis'))"
            if self.class_means.device != dev:
                self.class_means = self.class_means.to(dev)
                self.precision = self.precision.to(dev)
            return mahalanobis_score(features, self.class_means, self.precision)
        assert self.prototypes is not None, \
            "cosine proto_metric needs prototypes (build_proxy_stats)"
        if self.prototypes.device != dev:
            self.prototypes = self.prototypes.to(dev)
        return prototype_score(features, self.prototypes)

    def raw_proxies(self, logits: torch.Tensor, features: torch.Tensor) -> dict[str, float]:
        out = {"nuclear_norm": nuclear_norm_score(logits)}
        if self.atc_threshold is not None:
            out["atc"] = atc_score(logits, self.atc_threshold, self.atc_kind)
        if self._has_prototype_state():
            out["prototype"] = self.prototype_proxy(features)
        return out

    def predicted_acc(self, proxy_name: str, raw_value: float) -> float:
        if proxy_name in self.calib:
            return float(self.calib[proxy_name].predict([raw_value])[0])
        return raw_value

    def reliability_score(
        self,
        proxy_name: str,
        raw_value: float,
        mode: Literal["zscore", "isotonic"] = "zscore",
    ) -> float:
        """Put a raw proxy value on a common, val-based reliability scale so two
        heterogeneous models' proxies become comparable (JointProxyWeighted).

        "zscore" (default, lightweight): standardise by the val-set proxy
        mean/std (fit_val_reliability). No fitted curve, so it can't overfit —
        just recentres/rescales.
        "isotonic": map through the fitted calib curve (calibration.py) to a
        predicted accuracy, then take its logit, so it lands in the same
        unbounded logit space the zscore mode already produces.
        """
        if mode == "zscore":
            mean = self.proxy_mean.get(proxy_name)
            std = self.proxy_std.get(proxy_name)
            assert mean is not None and std is not None, (
                f"no val proxy_mean/proxy_std for proxy '{proxy_name}' on model "
                f"'{self.name}'; call fit_val_reliability first"
            )
            return (raw_value - mean) / max(std, 1e-6)
        if mode == "isotonic":
            return _logit(self.predicted_acc(proxy_name, raw_value))
        raise ValueError(f"mode must be 'zscore' or 'isotonic', got '{mode}'")


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


@torch.no_grad()
def fit_val_reliability(
    cfg: ProxyStats,
    logits: torch.Tensor,
    feats: torch.Tensor | None,
    labels: torch.Tensor,
    proxy_name: str,
    batch_size: int = 64,
) -> None:
    """Fill cfg.val_acc and cfg.proxy_mean/proxy_std[proxy_name] from a pool of
    clean validation logits/features/labels (e.g. from _source_pass).

    val_acc becomes the Kalman prior mean's source (via logit(val_acc)) for
    ProxyFilter. proxy_mean/std are fit by chunking the pool into batch_size
    groups and scoring each the same way a live test batch would — so they are
    on the same scale as the online per-batch proxy stream, not the (usually
    much less noisy) full-pool score. proxy_name="nuclear_norm_cum" reuses the
    "nuclear_norm" per-batch score: the running accumulator differs only in how
    batches are pooled over time at test time, not in the per-batch quantity.
    """
    cfg.val_acc = float((logits.argmax(1) == labels).float().mean())

    lookup_name = "nuclear_norm" if proxy_name == "nuclear_norm_cum" else proxy_name
    n = logits.shape[0]
    scores: list[float] = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        z_b = logits[start:end]
        f_b = feats[start:end] if feats is not None else None
        raw = cfg.raw_proxies(z_b, f_b)
        assert lookup_name in raw, (
            f"proxy '{proxy_name}' not computable from raw_proxies() for model "
            f"'{cfg.name}' (missing atc_threshold / prototype state?)"
        )
        scores.append(raw[lookup_name])

    scores_t = torch.tensor(scores)
    cfg.proxy_mean[proxy_name] = float(scores_t.mean())
    cfg.proxy_std[proxy_name] = float(scores_t.std(unbiased=False).clamp_min(1e-6)) \
        if len(scores) > 1 else 1.0


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
        "class_means_l":   cfg_l.class_means,
        "class_means_s":   cfg_s.class_means,
        "precision_l":     cfg_l.precision,
        "precision_s":     cfg_s.precision,
        "proto_metric":    cfg_l.proto_metric,
    }, path)
    print(f"[proxy stats] saved → {path} (proto_metric={cfg_l.proto_metric})")
    return path


def load_proxy_stats(
    name: str | Path,
    directory: str | Path = DEFAULT_PROXY_DIR,
) -> tuple[ProxyStats, ProxyStats]:
    """Load a (cfg_l, cfg_s) pair by bare name (from the dedicated folder) or
    by full path. .calib starts empty; attach a CalibrationMaps to populate it."""
    path = _resolve_path(name, directory)
    data = torch.load(path, map_location="cpu", weights_only=False)
    proto_metric = data.get("proto_metric", "cosine")
    cfg_l = ProxyStats(
        name=data["large_name"],
        num_classes=data["num_classes"],
        atc_threshold=data["atc_threshold_l"],
        prototypes=data["prototypes_l"],
        class_means=data.get("class_means_l"),
        precision=data.get("precision_l"),
        proto_metric=proto_metric,
        atc_kind=data.get("atc_kind_l", "neg_entropy"),
    )
    cfg_s = ProxyStats(
        name=data["small_name"],
        num_classes=data["num_classes"],
        atc_threshold=data["atc_threshold_s"],
        prototypes=data["prototypes_s"],
        class_means=data.get("class_means_s"),
        precision=data.get("precision_s"),
        proto_metric=proto_metric,
        atc_kind=data.get("atc_kind_s", "neg_entropy"),
    )
    print(f"[proxy stats] loaded ← {path} (proto_metric={proto_metric})")
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
        if cfg_l.proto_metric != proto_metric:
            warnings.warn(
                f"cached proxy stats use proto_metric='{cfg_l.proto_metric}' but "
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

    def _proto_state(feats: torch.Tensor) -> dict:
        if proto_metric == "mahalanobis":
            means, ids = build_class_means(feats, labels, num_classes)
            prec = build_tied_precision(feats, labels, means, ids, cov_shrinkage)
            return {"class_means": means, "precision": prec, "proto_metric": "mahalanobis"}
        return {"prototypes": build_prototypes(feats, labels, num_classes),
                "proto_metric": "cosine"}

    cfg_l = ProxyStats(
        name=large_name, num_classes=num_classes,
        atc_threshold=fit_atc_threshold(zl, labels), **_proto_state(fl),
    )
    cfg_s = ProxyStats(
        name=small_name, num_classes=num_classes,
        atc_threshold=fit_atc_threshold(zs, labels), **_proto_state(fs),
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
    assert rl.proto_metric == "cosine"
    print("proxies cosine self-test passed")

    # Mahalanobis build / score / round-trip
    torch.manual_seed(0)
    feats = torch.randn(400, 16)
    labs = torch.randint(0, 4, (400,))
    means, ids = build_class_means(feats, labs, num_classes=4)
    prec = build_tied_precision(feats, labs, means, ids, shrinkage=1e-2)
    assert means.shape == (4, 16) and prec.shape == (16, 16)
    cfg_m = ProxyStats(name="m", num_classes=4, class_means=means,
                       precision=prec, proto_metric="mahalanobis")
    score = cfg_m.prototype_proxy(feats)            # negated distance -> < 0
    assert score < 0.0
    assert "prototype" in cfg_m.raw_proxies(torch.randn(400, 4), feats)
    with tempfile.TemporaryDirectory() as d:
        save_proxy_stats(cfg_m, cfg_m, "maha", directory=d)
        a, b = load_proxy_stats("maha", directory=d)
    assert a.proto_metric == "mahalanobis"
    assert torch.allclose(a.class_means, means) and torch.allclose(a.precision, prec)
    assert abs(a.prototype_proxy(feats) - score) < 1e-5
    print("proxies mahalanobis self-test passed")

    # ─── RunningNuclearNorm decay self-test (Task 1.1) ───────────────────────
    import statistics

    torch.manual_seed(1)
    batches = [torch.randn(8, 5) for _ in range(6)]

    def _old_cumulative_score(gram: torch.Tensor, n: float) -> float:
        c = gram.shape[0]
        nuc = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt().sum()
        return float(nuc / (n * min(n, c)) ** 0.5)

    # decay=0 reproduces the original unweighted cumulative average exactly.
    rnn0 = RunningNuclearNorm(decay=0.0)
    gram_ref, n_ref = None, 0
    for b in batches:
        p = torch.softmax(b, dim=1)
        gram_ref = p.t() @ p if gram_ref is None else gram_ref + p.t() @ p
        n_ref += p.shape[0]
        rnn0.update(b)
        assert abs(rnn0.score() - _old_cumulative_score(gram_ref, n_ref)) < 1e-5
    print("RunningNuclearNorm decay=0 regression self-test passed")

    # decay=1 keeps only the latest batch (== single-batch score).
    rnn1 = RunningNuclearNorm(decay=1.0)
    for b in batches:
        rnn1.update(b)
        assert abs(rnn1.score() - nuclear_norm_score(b)) < 1e-5
    print("RunningNuclearNorm decay=1 self-test passed")

    # 0<decay<1 on a stationary stream: lower score variance than per-batch.
    torch.manual_seed(2)
    stream = [torch.randn(16, 10) for _ in range(60)]
    rnn_ema = RunningNuclearNorm(decay=0.1)
    per_batch_scores, ema_scores = [], []
    for b in stream:
        per_batch_scores.append(nuclear_norm_score(b))
        rnn_ema.update(b)
        ema_scores.append(rnn_ema.score())
    var_per_batch = statistics.pvariance(per_batch_scores[20:])  # skip burn-in
    var_ema = statistics.pvariance(ema_scores[20:])
    assert var_ema < var_per_batch, \
        f"EMA-Gram variance {var_ema} not lower than per-batch {var_per_batch}"
    print(f"RunningNuclearNorm decay=0.1 variance-reduction self-test passed "
          f"(per_batch={var_per_batch:.2e}, ema={var_ema:.2e})")

    # Step-change stream: EMA-Gram tracks the shift (not stuck at the old level).
    torch.manual_seed(3)
    pre_step  = [torch.randn(16, 10) - 3.0 for _ in range(30)]   # low-confidence regime
    post_step = [torch.randn(16, 10) + 3.0 for _ in range(30)]   # high-confidence regime
    rnn_step = RunningNuclearNorm(decay=0.1)
    for b in pre_step:
        rnn_step.update(b)
    score_before = rnn_step.score()
    for b in post_step:
        rnn_step.update(b)
    score_after = rnn_step.score()
    assert score_after > score_before, \
        "EMA-Gram did not track a step change in the underlying stream"
    print("RunningNuclearNorm decay=0.1 step-change self-test passed")

    # ─── Val reliability prior + calibration self-test (Task 2.1) ────────────
    torch.manual_seed(4)
    val_logits = torch.randn(256, 10)
    val_labels = torch.randint(0, 10, (256,))
    cfg_val = ProxyStats(name="dummy", num_classes=10)
    fit_val_reliability(cfg_val, val_logits, None, val_labels, "nuclear_norm", batch_size=16)
    assert cfg_val.val_acc is not None and 0.0 <= cfg_val.val_acc <= 1.0
    assert "nuclear_norm" in cfg_val.proxy_mean and "nuclear_norm" in cfg_val.proxy_std
    assert cfg_val.proxy_std["nuclear_norm"] > 0.0
    print("fit_val_reliability self-test passed")

    # nuclear_norm_cum reuses the nuclear_norm per-batch scoring.
    cfg_cum = ProxyStats(name="dummy2", num_classes=10)
    fit_val_reliability(cfg_cum, val_logits, None, val_labels, "nuclear_norm_cum", batch_size=16)
    assert abs(cfg_cum.proxy_mean["nuclear_norm_cum"] - cfg_val.proxy_mean["nuclear_norm"]) < 1e-6
    print("fit_val_reliability nuclear_norm_cum aliasing self-test passed")

    # zscore reliability_score: a raw value at the mean scores 0; symmetric std shift scores +/-1.
    mean = cfg_val.proxy_mean["nuclear_norm"]
    std = cfg_val.proxy_std["nuclear_norm"]
    assert abs(cfg_val.reliability_score("nuclear_norm", mean, mode="zscore")) < 1e-6
    assert abs(cfg_val.reliability_score("nuclear_norm", mean + std, mode="zscore") - 1.0) < 1e-4
    print("reliability_score zscore self-test passed")

    # isotonic reliability_score: needs a calib map; predicted_acc=0.5 -> logit=0.
    from sklearn.isotonic import IsotonicRegression as _IR
    iso = _IR(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit([0.0, 0.5, 1.0], [0.1, 0.5, 0.9])
    cfg_val.calib["nuclear_norm"] = iso
    assert abs(cfg_val.reliability_score("nuclear_norm", 0.5, mode="isotonic")) < 1e-4
    print("reliability_score isotonic self-test passed")