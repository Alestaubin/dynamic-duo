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
      --config cfgs/dynamic_duo_config.yaml --num_samples 1000 --seed 0
  python src/proxies/proxy_benchmark.py --test   # synthetic self-test
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

_trapz = getattr(np, "trapezoid", None) or np.trapz

from src.proxies.proxies import (
    ModelProxyConfig,
    FeatureExtractor,
    fit_atc_threshold,
    build_prototypes,
    build_proxy_configs,
    nuclear_norm_score,
    atc_score,
    prototype_score,
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


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _nan_to_neg(x):
    return -1.0 if (x != x) else x


def _metrics_from_arrays(
    raw_l: np.ndarray, raw_s: np.ndarray,
    acc_l: np.ndarray, acc_s: np.ndarray,
    pred_l: np.ndarray, pred_s: np.ndarray,
) -> dict:
    """Compute all selection metrics from pre-built arrays."""
    pred_gap = pred_l - pred_s
    true_gap = acc_l - acc_s
    nontie = true_gap != 0
    sel_acc = (float((np.sign(pred_gap[nontie]) == np.sign(true_gap[nontie])).mean())
               if nontie.any() else float("nan"))
    _, _, aurc = _selection_risk_coverage(pred_gap, true_gap)
    return {
        "n_batches":  int(len(raw_l)),
        "sel_acc":    sel_acc,
        "gap_rho":    _safe_spearman(pred_gap, true_gap),
        "sel_aurc":   aurc,
        "sig_l":      _safe_spearman(raw_l, acc_l),
        "sig_s":      _safe_spearman(raw_s, acc_s),
    }


def evaluate_proxy(eval_records: list[BatchRecord], proxy: str, cfg_l, cfg_s) -> dict:
    """Full ranking diagnostics for one proxy over the eval records.

    Returns a dict with overall metrics and a per_corruption breakdown.
    """
    raw_l, raw_s, acc_l, acc_s, corrs = [], [], [], [], []
    pred_l, pred_s = [], []
    for r in eval_records:
        if proxy not in r.raw_l or proxy not in r.raw_s:
            continue
        raw_l.append(r.raw_l[proxy]); raw_s.append(r.raw_s[proxy])
        acc_l.append(r.acc_l);        acc_s.append(r.acc_s)
        pred_l.append(cfg_l.predicted_acc(proxy, r.raw_l[proxy]))
        pred_s.append(cfg_s.predicted_acc(proxy, r.raw_s[proxy]))
        corrs.append(r.corruption)

    raw_l  = np.array(raw_l);  raw_s  = np.array(raw_s)
    acc_l  = np.array(acc_l);  acc_s  = np.array(acc_s)
    pred_l = np.array(pred_l); pred_s = np.array(pred_s)
    corrs_arr = np.array(corrs)

    overall = _metrics_from_arrays(raw_l, raw_s, acc_l, acc_s, pred_l, pred_s)
    cov, risk, _ = _selection_risk_coverage(pred_l - pred_s, acc_l - acc_s)

    per_corruption: dict[str, dict] = {}
    for c in sorted(set(corrs)):
        mask = corrs_arr == c
        if mask.sum() >= 2:
            per_corruption[c] = _metrics_from_arrays(
                raw_l[mask], raw_s[mask],
                acc_l[mask], acc_s[mask],
                pred_l[mask], pred_s[mask],
            )

    return {
        "proxy":              proxy,
        "n_batches":          overall["n_batches"],
        "selection_accuracy": overall["sel_acc"],
        "gap_spearman":       overall["gap_rho"],
        "selection_aurc":     overall["sel_aurc"],
        "signal_spearman_l":  overall["sig_l"],
        "signal_spearman_s":  overall["sig_s"],
        "per_corruption":     per_corruption,
        "_rc_curve":          (cov, risk),
    }


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
        logger.log({f"sel_acc/{res['proxy']}/{c}": m["sel_acc"]
                    for c, m in res["per_corruption"].items()})
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


# Metrics included in the CSV (short name → result-dict key within per_corruption)
_CSV_METRICS = [
    ("sel_acc",  "sel_acc"),
    ("gap_rho",  "gap_rho"),
    ("sel_aurc", "sel_aurc"),
    ("sig_l",    "sig_l"),
    ("sig_s",    "sig_s"),
]


def save_results_csv(results: list[dict], path: str | Path) -> None:
    """Save per-corruption metrics for all proxies to a CSV file.

    Rows: one per corruption + a final "ALL" overall row.
    Columns: corruption, n_batches, {proxy}_{metric} for all proxy×metric combos.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    proxies = [r["proxy"] for r in results]
    # collect all corruptions seen across any proxy
    all_corruptions: list[str] = sorted(
        {c for r in results for c in r["per_corruption"]}
    )

    fieldnames = ["corruption", "n_batches"] + [
        f"{p}_{m}" for p in proxies for m, _ in _CSV_METRICS
    ]

    def _row(label: str, n: int, metrics_by_proxy: dict[str, dict]) -> dict:
        row: dict = {"corruption": label, "n_batches": n}
        for p in proxies:
            m = metrics_by_proxy.get(p, {})
            for col, key in _CSV_METRICS:
                v = m.get(key, float("nan"))
                row[f"{p}_{col}"] = f"{v:.4f}" if not math.isnan(v) else "nan"
        return row

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in all_corruptions:
            n = next(
                (r["per_corruption"][c]["n_batches"]
                 for r in results if c in r["per_corruption"]),
                0,
            )
            writer.writerow(_row(
                c, n,
                {r["proxy"]: r["per_corruption"].get(c, {}) for r in results},
            ))
        # overall row
        writer.writerow(_row(
            "ALL", results[0]["n_batches"] if results else 0,
            {r["proxy"]: {
                "sel_acc":  r["selection_accuracy"],
                "gap_rho":  r["gap_spearman"],
                "sel_aurc": r["selection_aurc"],
                "sig_l":    r["signal_spearman_l"],
                "sig_s":    r["signal_spearman_s"],
            } for r in results},
        ))

    print(f"[proxy benchmark] results saved → {path}")


def print_latex_table(results: list[dict]) -> None:
    """Print a LaTeX table: rows=corruptions, columns=proxy×{sel_acc, sel_aurc}.

    Best sel_acc per row is bolded.
    """
    proxies = [r["proxy"] for r in results]
    all_corruptions: list[str] = sorted(
        {c for r in results for c in r["per_corruption"]}
    )
    # map proxy → metrics dict per corruption
    by_proxy: dict[str, dict] = {r["proxy"]: r for r in results}

    n_metrics = 2  # sel_acc, sel_aurc
    n_cols = 1 + len(proxies) * n_metrics  # corruption + proxy cols

    col_spec = "l" + "".join("rr" for _ in proxies)
    proxy_headers = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{p}}}" for p in proxies
    )
    sub_headers = " & ".join(
        "sel\\_acc & sel\\_AURC" for _ in proxies
    )

    lines = [
        "\\begin{table}[h]",
        "  \\centering",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    Corruption & {proxy_headers} \\\\",
        "    \\cmidrule(lr){" + "2-" + str(n_cols) + "}",
        f"    & {sub_headers} \\\\",
        "    \\midrule",
    ]

    def _fmt(v: float, bold: bool) -> str:
        if math.isnan(v):
            return "--"
        s = f"{v:.3f}"
        return f"\\textbf{{{s}}}" if bold else s

    for c in all_corruptions:
        sel_accs = [
            by_proxy[p]["per_corruption"].get(c, {}).get("sel_acc", float("nan"))
            for p in proxies
        ]
        best_sel = max((v for v in sel_accs if not math.isnan(v)), default=float("nan"))
        cells = []
        for p in proxies:
            m = by_proxy[p]["per_corruption"].get(c, {})
            sa   = m.get("sel_acc",  float("nan"))
            aurc = m.get("sel_aurc", float("nan"))
            cells.append(_fmt(sa, not math.isnan(sa) and sa == best_sel))
            cells.append(_fmt(aurc, False))
        lines.append(f"    {c} & {' & '.join(cells)} \\\\")

    # overall row
    lines.append("    \\midrule")
    sel_accs_all = [
        by_proxy[p]["selection_accuracy"] for p in proxies
    ]
    best_all = max((v for v in sel_accs_all if not math.isnan(v)), default=float("nan"))
    cells = []
    for p in proxies:
        sa   = by_proxy[p]["selection_accuracy"]
        aurc = by_proxy[p]["selection_aurc"]
        cells.append(_fmt(sa, not math.isnan(sa) and sa == best_all))
        cells.append(_fmt(aurc, False))
    lines.append(f"    ALL & {' & '.join(cells)} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{Proxy selection accuracy and AURC per corruption.}",
        "  \\label{tab:proxy_benchmark}",
        "\\end{table}",
    ]
    print("\n" + "\n".join(lines) + "\n")


def print_latex_signal_table(results: list[dict]) -> None:
    """LaTeX table of Spearman ρ(raw proxy, true batch accuracy) per corruption.

    This measures the proxy's raw signal quality before any isotonic calibration —
    i.e. whether higher proxy score actually tracks higher model accuracy on each batch.
    Four columns per proxy: ρ_large, ρ_small (best ρ per row is bolded).
    """
    proxies = [r["proxy"] for r in results]
    all_corruptions: list[str] = sorted(
        {c for r in results for c in r["per_corruption"]}
    )
    by_proxy: dict[str, dict] = {r["proxy"]: r for r in results}

    n_metrics = 2  # sig_l, sig_s
    n_cols = 1 + len(proxies) * n_metrics

    col_spec = "l" + "".join("rr" for _ in proxies)
    proxy_headers = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{p}}}" for p in proxies
    )
    sub_headers = " & ".join(
        "$\\rho_L$ & $\\rho_S$" for _ in proxies
    )

    lines = [
        "\\begin{table}[h]",
        "  \\centering",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    Corruption & {proxy_headers} \\\\",
        "    \\cmidrule(lr){" + "2-" + str(n_cols) + "}",
        f"    & {sub_headers} \\\\",
        "    \\midrule",
    ]

    def _fmt(v: float, bold: bool) -> str:
        if math.isnan(v):
            return "--"
        s = f"{v:.3f}"
        return f"\\textbf{{{s}}}" if bold else s

    all_sig_values = lambda c: [
        v
        for p in proxies
        for v in (
            by_proxy[p]["per_corruption"].get(c, {}).get("sig_l", float("nan")),
            by_proxy[p]["per_corruption"].get(c, {}).get("sig_s", float("nan")),
        )
        if not math.isnan(v)
    ]

    for c in all_corruptions:
        best = max(all_sig_values(c), default=float("nan"))
        cells = []
        for p in proxies:
            m = by_proxy[p]["per_corruption"].get(c, {})
            sl = m.get("sig_l", float("nan"))
            ss = m.get("sig_s", float("nan"))
            cells.append(_fmt(sl, not math.isnan(sl) and sl == best))
            cells.append(_fmt(ss, not math.isnan(ss) and ss == best))
        lines.append(f"    {c} & {' & '.join(cells)} \\\\")

    lines.append("    \\midrule")
    all_sig_overall = [
        v
        for p in proxies
        for v in (by_proxy[p]["signal_spearman_l"], by_proxy[p]["signal_spearman_s"])
        if not math.isnan(v)
    ]
    best_all = max(all_sig_overall, default=float("nan"))
    cells = []
    for p in proxies:
        sl = by_proxy[p]["signal_spearman_l"]
        ss = by_proxy[p]["signal_spearman_s"]
        cells.append(_fmt(sl, not math.isnan(sl) and sl == best_all))
        cells.append(_fmt(ss, not math.isnan(ss) and ss == best_all))
    lines.append(f"    ALL & {' & '.join(cells)} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{Spearman $\\rho$ between raw proxy score and true batch accuracy"
        " ($\\rho_L$: large model, $\\rho_S$: small model). No isotonic calibration applied.}",
        "  \\label{tab:proxy_signal}",
        "\\end{table}",
    ]
    print("\n" + "\n".join(lines) + "\n")


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


def run_proxy_benchmark(
    cfg,
    device,
    num_samples=None,
    seed=None,
    wandb_project=None,
    cache_path=None,
    csv_path=None,
):
    """Full proxy benchmark pipeline for the duo (large=vit_b_16, small=resnet50).

    Parameters
    ----------
    cache_path : path to a .pt file for caching ATC thresholds + prototypes from
                 the source pass. Created on first run, loaded on subsequent runs.
    csv_path   : if given, per-corruption metrics for all proxies are written here.
    """
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
    # (skipped if cache_path points to an existing file)
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
        cache_path=cache_path,
    )

    proxies = ["nuclear_norm", "atc", "prototype"]

    # Dev + eval passes need persistent feature extractors
    ext_l = FeatureExtractor(large_model, cfg["LARGE"]["NAME"])
    ext_s = FeatureExtractor(small_model, cfg["SMALL"]["NAME"])
    try:
        # Dev pass: calibrator corruptions → fit isotonic calibration maps.
        # Skipped when all calib maps are already present in the loaded cache.
        if cfg_l.calib:
            print(f"\n[proxy cache] calibration maps already loaded "
                  f"({sorted(cfg_l.calib)}), skipping dev pass.")
        else:
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

            fit_calibration(cfg_l, cfg_s, dev_records, proxies)
            print(f"\nCalibration fitted on {len(dev_records)} dev batches "
                  f"({len(cfg['CALIBRATOR']['CORRUPTIONS'])} corruptions × "
                  f"{len(cfg['CALIBRATOR']['SEVERITIES'])} severities).")

            if cache_path is not None:
                from src.proxies.proxies import save_proxy_configs
                save_proxy_configs(cfg_l, cfg_s, cache_path)

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
        print_latex_table(results)
        print_latex_signal_table(results)

        if csv_path is not None:
            save_results_csv(results, csv_path)

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


# ============================================================================
# SPEARMAN TEST — per-corruption proxy vs. accuracy correlation
# ============================================================================

@torch.no_grad()
def _collect_corruption_aggregate(
    cfg_l: ModelProxyConfig,
    cfg_s: ModelProxyConfig,
    ext_l: FeatureExtractor, preprocess_l,
    ext_s: FeatureExtractor, preprocess_s,
    loader, device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one corruption through both models; return concatenated tensors."""
    z_l, f_l, z_s, f_s, labs = [], [], [], [], []
    for imgs, labels in tqdm(loader, leave=False):
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


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson r and two-sided p-value; returns (nan, nan) on degenerate input."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    r, p = pearsonr(a, b)
    return float(r), float(p)


def _safe_spearman_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman ρ and p-value; returns (nan, nan) on degenerate input."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    res = spearmanr(a, b)
    return float(res.statistic), float(res.pvalue)


def run_spearman_test(
    cfg,
    device,
    num_samples=None,
    seed=None,
    cache_path=None,
    csv_path=None,
):
    """Per-corruption proxy signal quality: Spearman ρ and Pearson r.

    For each corruption/severity, runs all samples through both models and
    computes ONE proxy score for the full set (not per batch). Each
    corruption/severity is then one data point. Reports Spearman ρ and
    Pearson r between proxy scores and true accuracies across all
    corruption/severity pairs.

    Parameters
    ----------
    cache_path : .pt file for ATC thresholds + prototypes (created if absent).
    csv_path   : optional path to write the per-corruption table as CSV.
    """
    import torch.utils.data
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from src.utils.model import get_model
    from src.utils.data import load_imagenetC, _pil_collate_fn

    NUM_CLASSES = 1000
    PROXIES = ["nuclear_norm", "atc", "prototype"]

    large_model, large_pre = get_model(cfg["LARGE"]["NAME"])
    small_model, small_pre  = get_model(cfg["SMALL"]["NAME"])
    large_model = large_model.to(device).eval()
    small_model = small_model.to(device).eval()

    # Source pass for ATC thresholds and prototypes (skipped if cache present)
    src_ds = datasets.ImageFolder(cfg["VAL_DIR"])
    src_loader = DataLoader(
        src_ds, batch_size=cfg["BS"], shuffle=False,
        num_workers=cfg["WORKERS"], pin_memory=(device.type == "cuda"),
        collate_fn=_pil_collate_fn,
    )
    cfg_l, cfg_s = build_proxy_configs(
        large_model, large_pre, cfg["LARGE"]["NAME"],
        small_model, small_pre, cfg["SMALL"]["NAME"],
        src_loader, device, NUM_CLASSES,
        cache_path=cache_path,
    )

    # ── Collect one aggregate data point per corruption/severity ─────────── #
    # Each entry: corruption, severity, proxy_l, proxy_s, acc_l, acc_s
    rows: list[dict] = []

    ext_l = FeatureExtractor(large_model, cfg["LARGE"]["NAME"])
    ext_s = FeatureExtractor(small_model, cfg["SMALL"]["NAME"])
    try:
        for severity in cfg["EVAL"]["SEVERITIES"]:
            for corruption in cfg["EVAL"]["CORRUPTIONS"]:
                loader = load_imagenetC(
                    cfg["TEST_DIR"], severities=severity,
                    corruption_types=[corruption], device=device,
                    batch_size=cfg["BS"], num_workers=cfg["WORKERS"],
                    num_samples=num_samples, seed=seed,
                )
                zl, fl, zs, fs, labs = _collect_corruption_aggregate(
                    cfg_l, cfg_s,
                    ext_l, large_pre, ext_s, small_pre,
                    loader, device,
                )
                acc_l = float((zl.argmax(1) == labs).float().mean())
                acc_s = float((zs.argmax(1) == labs).float().mean())
                protos_l = cfg_l.prototypes.to(zl.device) if cfg_l.prototypes is not None else None
                protos_s = cfg_s.prototypes.to(zs.device) if cfg_s.prototypes is not None else None
                row = {
                    "corruption": corruption,
                    "severity":   severity,
                    "n_samples":  int(labs.shape[0]),
                    "acc_l":      acc_l,
                    "acc_s":      acc_s,
                    "nuclear_norm_l": float(nuclear_norm_score(zl)),
                    "nuclear_norm_s": float(nuclear_norm_score(zs)),
                    "atc_l": (float(atc_score(zl, cfg_l.atc_threshold, cfg_l.atc_kind))
                              if cfg_l.atc_threshold is not None else float("nan")),
                    "atc_s": (float(atc_score(zs, cfg_s.atc_threshold, cfg_s.atc_kind))
                              if cfg_s.atc_threshold is not None else float("nan")),
                    "prototype_l": (float(prototype_score(fl, protos_l))
                                    if protos_l is not None else float("nan")),
                    "prototype_s": (float(prototype_score(fs, protos_s))
                                    if protos_s is not None else float("nan")),
                }
                print(f"  {corruption}/s{severity}: n={row['n_samples']}  "
                      f"acc_l={acc_l:.3f}  acc_s={acc_s:.3f}  "
                      f"nn_l={row['nuclear_norm_l']:.3f}  proto_l={row['prototype_l']:.3f}")
                rows.append(row)
    finally:
        ext_l.remove()
        ext_s.remove()

    n_pts = len(rows)
    print(f"\n{n_pts} data points "
          f"({len(cfg['EVAL']['CORRUPTIONS'])} corruptions × "
          f"{len(cfg['EVAL']['SEVERITIES'])} severities)\n")

    # ── Compute correlations ─────────────────────────────────────────────── #
    # stats[proxy][model] = {"rho": float, "rho_p": float, "r": float, "r_p": float}
    stats: dict[str, dict[str, dict]] = {}
    for proxy in PROXIES:
        stats[proxy] = {}
        for model, side in (("large", "l"), ("small", "s")):
            proxy_vals = np.array([r[f"{proxy}_{side}"] for r in rows])
            acc_vals   = np.array([r[f"acc_{side}"]     for r in rows])
            valid = ~np.isnan(proxy_vals)
            rho, rho_p = _safe_spearman_p(proxy_vals[valid], acc_vals[valid])
            r,   r_p   = _safe_pearson(proxy_vals[valid], acc_vals[valid])
            stats[proxy][model] = {"rho": rho, "rho_p": rho_p, "r": r, "r_p": r_p,
                                   "n": int(valid.sum())}

    # ── Print summary table ──────────────────────────────────────────────── #
    print(f"{'proxy':<14}  {'model':<6}  "
          f"{'Spearman ρ':>11}  {'p':>7}  {'Pearson r':>10}  {'p':>7}  {'n':>4}")
    print("-" * 65)
    for proxy in PROXIES:
        for model in ("large", "small"):
            s = stats[proxy][model]
            print(f"{proxy:<14}  {model:<6}  "
                  f"{s['rho']:>11.3f}  {s['rho_p']:>7.4f}  "
                  f"{s['r']:>10.3f}  {s['r_p']:>7.4f}  {s['n']:>4}")
        print()

    # ── Save CSV ─────────────────────────────────────────────────────────── #
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (["corruption", "severity", "n_samples", "acc_l", "acc_s"]
                      + [f"{p}_{s}" for p in PROXIES for s in ("l", "s")])
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                                 for k, v in row.items()})
        print(f"[spearman test] per-corruption data saved → {path}")

    return stats, rows


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
    parser.add_argument("--proxy_cache", type=str, default=None,
                        help="Path to .pt cache for ATC thresholds + prototypes.")
    parser.add_argument("--csv", type=str, default=None, dest="csv_path",
                        help="Save per-corruption results table to this CSV path.")
    parser.add_argument("--spearman", action="store_true",
                        help="Run the lightweight Spearman ρ test (no dev/calibration pass).")
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
        if args.spearman:
            run_spearman_test(cfg, device, num_samples=args.num_samples,
                              seed=args.seed, cache_path=args.proxy_cache,
                              csv_path=args.csv_path)
        else:
            run_proxy_benchmark(cfg, device, num_samples=args.num_samples,
                                seed=args.seed, wandb_project=args.wandb_project,
                                cache_path=args.proxy_cache, csv_path=args.csv_path)
