"""
proxy_benchmark.py
==================
Benchmarking harness for per-model, per-batch reliability proxies in a
heterogeneous duo (e.g. ViT-B/16 + ResNet-50) under ImageNet-C shift.

Goal: decide which unsupervised proxy best answers "which of the two models is
more accurate on THIS batch", BEFORE building any anchor mechanism on top of it.

Three proxies are implemented, each producing a per-batch scalar PER MODEL:
  1. nuclear_norm  -- confidence + dispersity (Deng et al., "Confidence and
                      Dispersity as Signals"). Collapse-aware by construction.
  2. atc           -- Average Thresholded Confidence (Garg et al., 2022).
                      Returns a predicted-accuracy value directly.
  3. prototype     -- mean nearest-source-prototype cosine similarity
                      (Trust-Score / T3A / FOA family). Needs source prototypes.

Proxy functions and ModelProxyConfig live in src.proxies.proxies.
This module provides the benchmark scaffolding: batch records, calibration
fitting, evaluation metrics, and the real-data runner.

Evaluation reports, per corruption and pooled:
  - per-model signal quality:  Spearman(raw proxy, true batch accuracy)
  - model-selection accuracy:  did sign(pred_gap) match sign(true_gap)?
  - gap correlation:           Spearman(pred_gap, true_gap)
  - selection risk-coverage curve + AURC (commit only when |pred_gap| large)

Usage:
  python src/proxies/proxy_benchmark.py \
      --config cfgs/dynamic_duo_config.yaml [--num_samples 1000] [--seed 0]
  python src/proxies/proxy_benchmark.py --test   # synthetic self-test
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

_trapz = getattr(np, "trapezoid", None) or np.trapz

from src.proxies.proxies import (
    ModelProxyConfig,
    FeatureExtractor,
    fit_atc_threshold,
    build_prototypes,
    build_proxy_configs,
)


# ─── Per-batch record ─────────────────────────────────────────────────────────

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


# ─── Calibration ─────────────────────────────────────────────────────────────

def fit_calibration(
    cfg_l: ModelProxyConfig,
    cfg_s: ModelProxyConfig,
    dev_records: list[BatchRecord],
    proxy_names: list[str],
    holdout_corruptions: set[str] | None = None,
) -> None:
    """Fit per-model isotonic maps raw_proxy→accuracy on dev records, in place."""
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


# ─── Evaluation ──────────────────────────────────────────────────────────────

def _selection_risk_coverage(pred_gap: np.ndarray, true_gap: np.ndarray):
    """Risk-coverage curve for the model-selection decision.

    Commits on batches with the largest |pred_gap|. Risk = selection error.
    Returns (coverages, risks, aurc). Lower AURC = more trustworthy gate.
    """
    order = np.argsort(-np.abs(pred_gap))
    pg, tg = pred_gap[order], true_gap[order]
    wrong = np.where(tg == 0, 0.5, (np.sign(pg) != np.sign(tg)).astype(float))
    n = len(pg)
    coverages, risks, cum = [], [], 0.0
    for i in range(n):
        cum += wrong[i]
        coverages.append((i + 1) / n)
        risks.append(cum / (i + 1))
    coverages, risks = np.array(coverages), np.array(risks)
    return coverages, risks, float(_trapz(risks, coverages))


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

    raw_l  = np.array(raw_l);  raw_s  = np.array(raw_s)
    acc_l  = np.array(acc_l);  acc_s  = np.array(acc_s)
    pred_l = np.array(pred_l); pred_s = np.array(pred_s)
    pred_gap = pred_l - pred_s
    true_gap = acc_l - acc_s

    sig_l = _safe_spearman(raw_l, acc_l)
    sig_s = _safe_spearman(raw_s, acc_s)

    nontie = true_gap != 0
    sel_acc = (float((np.sign(pred_gap[nontie]) == np.sign(true_gap[nontie])).mean())
               if nontie.any() else float("nan"))
    gap_corr = _safe_spearman(pred_gap, true_gap)
    cov, risk, aurc = _selection_risk_coverage(pred_gap, true_gap)

    corrs_arr = np.array(corrs)
    per_corr = {
        c: float((np.sign(pred_gap[m]) == np.sign(true_gap[m])).mean())
        for c in sorted(set(corrs))
        if (m := (corrs_arr == c) & nontie).any()
    }

    return {
        "proxy": proxy,
        "n_batches": int(len(raw_l)),
        "signal_spearman_l": sig_l,
        "signal_spearman_s": sig_s,
        "selection_accuracy": sel_acc,
        "gap_spearman": gap_corr,
        "selection_aurc": aurc,
        "per_corruption_selection_acc": per_corr,
        "_rc_curve": (cov, risk),
    }


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


# ─── Reporting ────────────────────────────────────────────────────────────────

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
# ============================================================================

@torch.no_grad()
def _collect_records(
    cfg_l, cfg_s,
    ext_l: FeatureExtractor, preprocess_l,
    ext_s: FeatureExtractor, preprocess_s,
    loader, device, corruption, severity,
) -> list[BatchRecord]:
    """One BatchRecord per batch."""
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

    # Source pass: clean ImageNet val → ATC thresholds + prototypes
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    src_ds = datasets.ImageFolder(cfg["VAL_DIR"])
    if num_samples is not None:
        n = min(num_samples, len(src_ds))
        src_ds = torch.utils.data.Subset(
            src_ds, torch.randperm(len(src_ds), generator=gen)[:n].tolist()
        )
    src_loader = DataLoader(
        src_ds, batch_size=cfg["BS"], shuffle=False,
        num_workers=cfg["WORKERS"], pin_memory=(device.type == "cuda"),
        collate_fn=_pil_collate_fn,
    )
    cfg_l, cfg_s = build_proxy_configs(
        large_model, large_pre, cfg["LARGE"]["NAME"],
        small_model, small_pre, cfg["SMALL"]["NAME"],
        src_loader, device, NUM_CLASSES,
    )

    # Dev + eval passes need persistent feature extractors
    ext_l = FeatureExtractor(large_model, cfg["LARGE"]["NAME"])
    ext_s = FeatureExtractor(small_model, cfg["SMALL"]["NAME"])
    try:
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

        # Eval pass
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
                        help="Path to dynamic_duo_config.yaml")
    parser.add_argument("--num_samples", type=int, default=None)
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
