"""
proxy_benchmark.py
==================
Benchmarking harness for per-model, per-batch reliability proxies in a
heterogeneous duo (e.g. ViT-B/16 + ResNet-50) under ImageNet-C shift.

Goal: decide which unsupervised proxy best answers "which of the two models is
more accurate on THIS batch", BEFORE building any anchor mechanism on top of it.

Three proxies are implemented, each producing a per-batch scalar PER MODEL:
  1. nuclear_norm     -- confidence + dispersity   (Deng et al., "Confidence and Dispersity as Signals: Unsupervised
                        Model Evaluation and Ranking").
  2. atc              -- Average Thresholded Confidence (Garg et al., 2022).
                         Returns a predicted-accuracy value directly.
  3. prototype        -- mean nearest-source-prototype cosine similarity
                         (Trust-Score / T3A / FOA family). Needs source prototypes.

Comparability across the two heterogeneous models is handled by fitting a
PER-MODEL isotonic calibration map  raw_proxy -> predicted accuracy  on a labeled
dev split, fit ACROSS corruptions (optionally leave-corruptions-out). Ranking is
then done on calibrated (accuracy-unit) scores.

Evaluation reports, per corruption and pooled:
  - per-model signal quality:  Spearman(raw proxy, true batch accuracy)
  - model-selection accuracy:   did sign(pred_gap) match sign(true_gap)
  - gap correlation:            Spearman(pred_gap, true_gap)
  - a selection risk-coverage curve + its AURC  (commit only when |pred_gap| large)

The risk-coverage view is the one that matters for the soft anchor: it tells you
how reliable the trust direction is as a function of the margin you gate on.

Author: scaffold for the dynamic-duo project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


# ----------------------------------------------------------------------------
# Raw proxy functions.  Each takes a single model's batch outputs and returns a
# python float (one scalar per batch).  Higher = "more reliable" for all three
# (for ATC, higher = higher predicted accuracy).
# ----------------------------------------------------------------------------

@torch.no_grad()
def nuclear_norm_score(logits: torch.Tensor) -> float:
    """Confidence + dispersity via the nuclear norm of the prediction matrix.

    P is (N, C) row-stochastic. ||P||_* is large when rows are confident (peaky)
    AND the batch uses many distinct classes (high rank). A model that has
    collapsed to one confident class has low rank -> low nuclear norm, so this
    penalises exactly the TENT collapse failure mode.

    Normalised by sqrt(N * min(N, C)) (an upper bound on ||P||_*) to keep the
    value in a roughly [0, 1] range and reduce batch-size dependence.
    """
    p = torch.softmax(logits, dim=1)
    n, c = p.shape
    nuc = torch.linalg.matrix_norm(p, ord="nuc")
    denom = math.sqrt(n * min(n, c))
    return float(nuc / denom)


@torch.no_grad()
def _atc_sample_scores(logits: torch.Tensor, kind: str = "neg_entropy") -> torch.Tensor:
    """Per-sample ATC score function. neg_entropy is usually stronger than maxconf."""
    p = torch.softmax(logits, dim=1)
    if kind == "maxconf":
        return p.max(dim=1).values
    elif kind == "neg_entropy":
        logp = torch.log_softmax(logits, dim=1)
        return (p * logp).sum(dim=1)  # = -entropy (higher = more confident)
    raise ValueError(kind)


@torch.no_grad()
def fit_atc_threshold(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    kind: str = "neg_entropy",
) -> float:
    """Learn the ATC threshold t on labeled source/ID data: choose t so that the
    fraction of source points scoring BELOW t equals the source ERROR rate.
    Then predicted_acc(target) = fraction of target points scoring >= t.
    """
    scores = _atc_sample_scores(source_logits, kind)
    correct = source_logits.argmax(1) == source_labels
    err_rate = 1.0 - correct.float().mean().item()
    # threshold = err_rate quantile of the score distribution
    t = torch.quantile(scores.float(), max(min(err_rate, 1.0), 0.0)).item()
    return t


@torch.no_grad()
def atc_score(logits: torch.Tensor, threshold: float, kind: str = "neg_entropy") -> float:
    """ATC predicted accuracy = fraction of samples with score >= threshold."""
    scores = _atc_sample_scores(logits, kind)
    return float((scores >= threshold).float().mean())


@torch.no_grad()
def build_prototypes(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Class prototypes = L2-normalised mean penultimate feature per class.
    Computed once, offline, from a FROZEN (pre-adaptation) snapshot of the model.
    Returns (num_classes, D); empty classes get a zero vector (never nearest).
    """
    d = features.shape[1]
    protos = torch.zeros(num_classes, d, device=features.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            protos[c] = features[mask].mean(0)
    return torch.nn.functional.normalize(protos, dim=1)


@torch.no_grad()
def prototype_score(features: torch.Tensor, prototypes: torch.Tensor) -> float:
    """Mean nearest-prototype cosine similarity. Cosine (not Euclidean) so the
    768-d ViT and 2048-d ResNet spaces are scale-comparable. Higher = closer to
    the model's own source structure.
    """
    f = torch.nn.functional.normalize(features, dim=1)
    sims = f @ prototypes.t()           # (N, C)
    nearest = sims.max(dim=1).values    # (N,)
    return float(nearest.mean())

# ----------------------------------------------------------------------------
# Per-model proxy configuration: prototypes + ATC threshold + calibration maps.
# ----------------------------------------------------------------------------

@dataclass
class ModelProxyConfig:
    """Holds everything proxy-related for ONE model. Built offline from source +
    a labeled dev split."""
    name: str
    num_classes: int
    atc_threshold: float | None = None
    prototypes: torch.Tensor | None = None
    atc_kind: str = "neg_entropy"
    # calibration maps: raw proxy value -> predicted accuracy, one per proxy
    calib: dict[str, IsotonicRegression] = field(default_factory=dict)

    def raw_proxies(self, logits: torch.Tensor, features: torch.Tensor) -> dict[str, float]:
        out = {"nuclear_norm": nuclear_norm_score(logits)}
        if self.atc_threshold is not None:
            out["atc"] = atc_score(logits, self.atc_threshold, self.atc_kind)
        if self.prototypes is not None:
            out["prototype"] = prototype_score(features, self.prototypes)
        return out

    def predicted_acc(self, proxy_name: str, raw_value: float) -> float:
        """Apply the fitted calibration map. Falls back to identity if unfit
        (e.g. ATC is already in accuracy units and may be used uncalibrated)."""
        if proxy_name in self.calib:
            return float(self.calib[proxy_name].predict([raw_value])[0])
        return raw_value


# ----------------------------------------------------------------------------
# A single dev/eval record.
# ----------------------------------------------------------------------------

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
    cfg_l: ModelProxyConfig,
    cfg_s: ModelProxyConfig,
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


# ----------------------------------------------------------------------------
# Calibration: fit per-model isotonic maps raw_proxy -> accuracy on dev records.
# ----------------------------------------------------------------------------

def fit_calibration(
    cfg_l: ModelProxyConfig,
    cfg_s: ModelProxyConfig,
    dev_records: list[BatchRecord],
    proxy_names: list[str],
    holdout_corruptions: set[str] | None = None,
) -> None:
    """Fit, in place, cfg.calib[proxy] for each model and proxy. Pools batches
    across corruptions so the map generalises; optionally excludes holdout
    corruptions from the FIT so you can measure held-out ranking quality."""
    holdout = holdout_corruptions or set()
    fit_recs = [r for r in dev_records if r.corruption not in holdout]
    for proxy in proxy_names:
        for cfg, side in ((cfg_l, "l"), (cfg_s, "s")):
            xs, ys = [], []
            for r in fit_recs:
                raw = r.raw_l if side == "l" else r.raw_s
                acc = r.acc_l if side == "l" else r.acc_s
                if proxy in raw:
                    xs.append(raw[proxy])
                    ys.append(acc)
            if len(set(xs)) < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(xs, ys)
            cfg.calib[proxy] = iso


# ----------------------------------------------------------------------------
# Evaluation.
# ----------------------------------------------------------------------------

def _selection_risk_coverage(pred_gap: np.ndarray, true_gap: np.ndarray):
    """Risk-coverage curve for the model-SELECTION decision.

    Coverage c = fraction of batches we 'commit' on (those with the largest
    |pred_gap|). Risk = selection error on the committed set (we picked the model
    that was NOT actually better, ignoring near-ties). Returns (coverages, risks,
    aurc). Lower AURC = the proxy's margin is a trustworthy gate for the anchor.
    """
    order = np.argsort(-np.abs(pred_gap))          # most confident first
    pg, tg = pred_gap[order], true_gap[order]
    # a selection is 'wrong' when predicted and true gap disagree in sign
    # (true ties contribute 0.5 error -- no better-model to pick)
    wrong = np.where(tg == 0, 0.5, (np.sign(pg) != np.sign(tg)).astype(float))
    n = len(pg)
    coverages, risks = [], []
    cum = 0.0
    for i in range(n):
        cum += wrong[i]
        coverages.append((i + 1) / n)
        risks.append(cum / (i + 1))
    coverages, risks = np.array(coverages), np.array(risks)
    aurc = float(np.trapz(risks, coverages))
    return coverages, risks, aurc


def evaluate_proxy(eval_records: list[BatchRecord], proxy: str, cfg_l, cfg_s) -> dict:
    """Full ranking diagnostics for one proxy over the eval records."""
    raw_l, raw_s, acc_l, acc_s, corrs = [], [], [], [], []
    pred_l, pred_s = [], []
    for r in eval_records:
        if proxy not in r.raw_l or proxy not in r.raw_s:
            continue
        raw_l.append(r.raw_l[proxy]); raw_s.append(r.raw_s[proxy])
        acc_l.append(r.acc_l); acc_s.append(r.acc_s)
        pred_l.append(cfg_l.predicted_acc(proxy, r.raw_l[proxy]))
        pred_s.append(cfg_s.predicted_acc(proxy, r.raw_s[proxy]))
        corrs.append(r.corruption)

    raw_l = np.array(raw_l); raw_s = np.array(raw_s)
    acc_l = np.array(acc_l); acc_s = np.array(acc_s)
    pred_l = np.array(pred_l); pred_s = np.array(pred_s)
    pred_gap = pred_l - pred_s
    true_gap = acc_l - acc_s

    # per-model signal quality (raw proxy vs true accuracy)
    sig_l = _safe_spearman(raw_l, acc_l)
    sig_s = _safe_spearman(raw_s, acc_s)

    # model selection: did we pick the truly-better model (ignore true ties)
    nontie = true_gap != 0
    if nontie.any():
        sel_acc = float((np.sign(pred_gap[nontie]) == np.sign(true_gap[nontie])).mean())
    else:
        sel_acc = float("nan")
    gap_corr = _safe_spearman(pred_gap, true_gap)
    cov, risk, aurc = _selection_risk_coverage(pred_gap, true_gap)

    # per-corruption selection accuracy
    per_corr = {}
    corrs = np.array(corrs)
    for c in sorted(set(corrs.tolist())):
        m = (corrs == c) & nontie
        if m.any():
            per_corr[c] = float((np.sign(pred_gap[m]) == np.sign(true_gap[m])).mean())

    return {
        "proxy": proxy,
        "n_batches": int(len(raw_l)),
        "signal_spearman_l": sig_l,
        "signal_spearman_s": sig_s,
        "selection_accuracy": sel_acc,
        "gap_spearman": gap_corr,
        "selection_aurc": aurc,
        "per_corruption_selection_acc": per_corr,
        "_rc_curve": (cov, risk),  # for plotting / W&B
    }


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


# ----------------------------------------------------------------------------
# W&B logging (optional; no-op if wandb absent or run is None).
# ----------------------------------------------------------------------------

def log_to_wandb(results: list[dict], run=None) -> None:
    try:
        import wandb
    except ImportError:
        return
    if run is None and wandb.run is None:
        return
    logger = run or wandb
    table = wandb.Table(columns=[
        "proxy", "n_batches", "selection_accuracy", "gap_spearman",
        "selection_aurc", "signal_spearman_l", "signal_spearman_s",
    ])
    for res in results:
        table.add_data(
            res["proxy"], res["n_batches"], res["selection_accuracy"],
            res["gap_spearman"], res["selection_aurc"],
            res["signal_spearman_l"], res["signal_spearman_s"],
        )
        # per-corruption selection accuracy as a separate logged dict
        logger.log({f"sel_acc/{res['proxy']}/{c}": v
                    for c, v in res["per_corruption_selection_acc"].items()})
    logger.log({"proxy_benchmark/summary": table})


def print_report(results: list[dict]) -> None:
    print(f"\n{'proxy':<14}{'n':>5}{'sel_acc':>9}{'gap_rho':>9}"
          f"{'sel_AURC':>10}{'sig_L':>8}{'sig_S':>8}")
    print("-" * 63)
    for r in sorted(results, key=lambda x: -_nan_to_neg(x["selection_accuracy"])):
        print(f"{r['proxy']:<14}{r['n_batches']:>5}"
              f"{r['selection_accuracy']:>9.3f}{r['gap_spearman']:>9.3f}"
              f"{r['selection_aurc']:>10.3f}{r['signal_spearman_l']:>8.3f}"
              f"{r['signal_spearman_s']:>8.3f}")
    print("\nLower sel_AURC = margin is a more trustworthy anchor gate.")
    print("sel_acc = P(picked the truly-better model on non-tie batches).\n")


def _nan_to_neg(x):
    return -1.0 if (x != x) else x

# ============================================================================
# REAL-MODEL BENCHMARK RUNNER
# ----------------------------------------------------------------------------
# Connects the proxy library to the dynamic-duo project.
# Large model: vit_b_16  (hook: encoder → CLS token, dim=768)
# Small model: resnet50  (hook: avgpool → flatten, dim=2048)
#
# Usage:
#   python src/proxies/proxy_benchmark.py \
#       --config cfgs/dynamic_duo_config.yaml [--num_samples 1000] [--seed 0]
#   python src/proxies/proxy_benchmark.py --test   # synthetic self-test
# ============================================================================

class FeatureExtractor:
    """Wraps a model; captures penultimate features alongside logits via a forward hook."""

    def __init__(self, model: torch.nn.Module, model_name: str):
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


def _hook_spec(model: torch.nn.Module, model_name: str):
    """Return (layer_to_hook, output_transform) for penultimate feature capture."""
    name = model_name.lower()
    if "vit" in name:
        # torchvision ViT encoder → (N, seq_len, hidden_dim); CLS token at index 0
        return model.encoder, lambda out: out[:, 0, :]
    if "resnet" in name:
        # ResNet avgpool → (N, C, 1, 1); flatten to (N, C)
        return model.avgpool, lambda out: out.flatten(1)
    raise ValueError(f"No feature-hook spec for model '{model_name}'. Add it to _hook_spec().")


def _fwd(extractor: FeatureExtractor, preprocess, imgs: list, device: torch.device):
    """Preprocess a list of PIL images and forward through extractor."""
    x = torch.stack([preprocess(img) for img in imgs]).to(device)
    return extractor(x)


@torch.no_grad()
def _collect_source(extractor_l, preprocess_l, extractor_s, preprocess_s, loader, device):
    """Full source pass for ATC-threshold fitting and prototype building."""
    from tqdm import tqdm
    z_l, f_l, z_s, f_s, labs = [], [], [], [], []
    for imgs, labels in tqdm(loader, desc="source"):
        zl, fl = _fwd(extractor_l, preprocess_l, imgs, device)
        zs, fs = _fwd(extractor_s, preprocess_s, imgs, device)
        z_l.append(zl.cpu()); f_l.append(fl.cpu())
        z_s.append(zs.cpu()); f_s.append(fs.cpu())
        labs.append(labels.cpu())
    return (torch.cat(z_l), torch.cat(f_l),
            torch.cat(z_s), torch.cat(f_s),
            torch.cat(labs))


@torch.no_grad()
def _collect_records(cfg_l, cfg_s, extractor_l, preprocess_l, extractor_s, preprocess_s,
                     loader, device, corruption, severity):
    """Create one BatchRecord per batch in the loader."""
    from tqdm import tqdm
    records = []
    for imgs, labels in tqdm(loader, desc=f"{corruption}/s{severity}"):
        zl, fl = _fwd(extractor_l, preprocess_l, imgs, device)
        zs, fs = _fwd(extractor_s, preprocess_s, imgs, device)
        # Move to CPU: prototypes were built from CPU tensors
        records.append(make_record(
            cfg_l, cfg_s,
            zl.cpu(), zs.cpu(), fl.cpu(), fs.cpu(),
            labels,  # already CPU from _pil_collate_fn
            corruption, severity,
        ))
    return records


def run_proxy_benchmark(cfg, device, num_samples=None, seed=None, wandb_project=None):
    """Full proxy benchmark pipeline for the duo (large=vit_b_16, small=resnet50)."""
    import torch.utils.data
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from src.utils.model import get_model
    from src.utils.data import load_imagenetC, _pil_collate_fn

    NUM_CLASSES = 1000

    large_model, large_pre = get_model(cfg["LARGE"]["NAME"])
    small_model, small_pre = get_model(cfg["SMALL"]["NAME"])
    large_model = large_model.to(device).eval()
    small_model = small_model.to(device).eval()

    ext_l = FeatureExtractor(large_model, cfg["LARGE"]["NAME"])
    ext_s = FeatureExtractor(small_model, cfg["SMALL"]["NAME"])

    try:
        # Source pass: clean ImageNet val → ATC thresholds + prototypes
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        src_ds = datasets.ImageFolder(cfg["VAL_DIR"])
        if num_samples is not None:
            n = min(num_samples, len(src_ds))
            src_ds = torch.utils.data.Subset(src_ds, torch.randperm(len(src_ds), generator=gen)[:n].tolist())
        src_loader = DataLoader(
            src_ds, batch_size=cfg["BS"], shuffle=False,
            num_workers=cfg["WORKERS"], pin_memory=(device.type == "cuda"),
            collate_fn=_pil_collate_fn,
        )
        zl_src, fl_src, zs_src, fs_src, labs_src = _collect_source(
            ext_l, large_pre, ext_s, small_pre, src_loader, device
        )

        cfg_l = ModelProxyConfig(
            name=cfg["LARGE"]["NAME"], num_classes=NUM_CLASSES,
            atc_threshold=fit_atc_threshold(zl_src, labs_src),
            prototypes=build_prototypes(fl_src, labs_src, NUM_CLASSES),
        )
        cfg_s = ModelProxyConfig(
            name=cfg["SMALL"]["NAME"], num_classes=NUM_CLASSES,
            atc_threshold=fit_atc_threshold(zs_src, labs_src),
            prototypes=build_prototypes(fs_src, labs_src, NUM_CLASSES),
        )

        # Dev pass: calibrator corruptions → fit isotonic calibration maps
        dev_records = []
        for severity in cfg["CALIBRATOR"]["SEVERITIES"]:
            for corruption in cfg["CALIBRATOR"]["CORRUPTIONS"]:
                loader = load_imagenetC(
                    cfg["TEST_DIR"], severities=severity,
                    corruption_types=[corruption], device=device,
                    batch_size=cfg["BS"], num_workers=cfg["WORKERS"],
                    num_samples=num_samples, seed=seed,
                )
                dev_records.extend(_collect_records(
                    cfg_l, cfg_s, ext_l, large_pre, ext_s, small_pre,
                    loader, device, corruption, severity,
                ))

        proxies = ["nuclear_norm", "atc", "prototype"]
        fit_calibration(cfg_l, cfg_s, dev_records, proxies)
        print(f"\nCalibration fitted on {len(dev_records)} dev batches "
              f"({len(cfg['CALIBRATOR']['CORRUPTIONS'])} corruptions × "
              f"{len(cfg['CALIBRATOR']['SEVERITIES'])} severities).")

        # Eval pass: eval corruptions
        eval_records = []
        for severity in cfg["EVAL"]["SEVERITIES"]:
            for corruption in cfg["EVAL"]["CORRUPTIONS"]:
                loader = load_imagenetC(
                    cfg["TEST_DIR"], severities=severity,
                    corruption_types=[corruption], device=device,
                    batch_size=cfg["BS"], num_workers=cfg["WORKERS"],
                    num_samples=num_samples, seed=seed,
                )
                eval_records.extend(_collect_records(
                    cfg_l, cfg_s, ext_l, large_pre, ext_s, small_pre,
                    loader, device, corruption, severity,
                ))

        results = [evaluate_proxy(eval_records, p, cfg_l, cfg_s) for p in proxies]
        print_report(results)

        if wandb_project is not None:
            import wandb
            wrun = wandb.init(
                project=wandb_project,
                name=f"{cfg['LARGE']['NAME']}+{cfg['SMALL']['NAME']}_proxy_benchmark",
            )
            log_to_wandb(results, wrun)
            wrun.finish()

        return results

    finally:
        ext_l.remove()
        ext_s.remove()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Proxy benchmark for the dynamic-duo (vit_b_16 + resnet50)."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to dynamic_duo_config.yaml (omit to run synthetic test).")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max samples per corruption/severity subset.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Run synthetic self-test instead of real models.")
    args = parser.parse_args()

    if args.test or args.config is None:
        _run_self_test()
    else:
        from src.utils.data import load_config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        cfg = load_config(args.config)
        run_proxy_benchmark(cfg, device, num_samples=args.num_samples,
                            seed=args.seed, wandb_project=args.wandb_project)