"""
joint_proxy_anchor_coca.py
==========================
COCA TS calibrator with per-batch proxy-driven anchor selection.

Each batch:
  1. Compute proxy scores r_l, r_s (nuclear_norm / atc / prototype).
  2. Select anchor = the model with the higher selection score: the raw proxy,
     or the calibrated predicted accuracy when calibrated_selection=True.
  3. Fit COCA temperature tau aligning the source to the anchor.
  4. Return aggregated ensemble logits (z_anchor + z_source/tau) / 2.

Records per-batch proxy scores and GT accuracies (when labels are injected
via set_labels). After each corruption, report_and_reset_corruption_stats()
computes R², Pearson r, and Spearman ρ between proxy scores and per-batch
accuracies across all batches of that corruption.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from src.calibrators.base import BaseJointCalibrator, _NoOpModule
from src.calibrators.temp.coca_temperature import CocaTemperature
from src.proxies.proxies import (
    ProxyStats,
    FeatureExtractor,
    nuclear_norm_score,
    atc_score,
)

_PROXY_KINDS = {"nuclear_norm", "atc", "prototype"}


def _corr_stats(xs: list[float], ys: list[float]) -> dict:
    """R², Pearson r, Spearman ρ between xs and ys. Returns nans when n < 3."""
    n = len(xs)
    nan_result = {"r2": float("nan"), "pearson_r": float("nan"), "spearman_rho": float("nan"), "n": n}
    if n < 3:
        return nan_result
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return nan_result
    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    return {"r2": float(pr ** 2), "pearson_r": float(pr), "spearman_rho": float(sr), "n": n}


class JointProxyAnchorCoca(BaseJointCalibrator):
    """COCA TS with per-batch proxy-driven anchor selection.

    The model with the higher proxy score becomes the COCA anchor for that
    batch; the other model is the source whose temperature is scaled.

    Parameters
    ----------
    proxy_kind : "nuclear_norm" | "atc" | "prototype"
    cfg_l, cfg_s : ProxyStats for large and small model.
    calibrated_selection : if True, pick the anchor by comparing each model's
        calibrated predicted accuracy (cfg.predicted_acc(proxy_kind, raw))
        instead of the raw proxy scores. Requires a CalibrationMaps attached to
        both cfg_l and cfg_s for this proxy_kind (asserted at first batch). The
        raw r_l/r_s are still logged and correlated unchanged, so the per-
        corruption proxy diagnostics stay comparable across this flag.
    num_steps : COCA gradient steps per batch.
    lr : COCA learning rate.
    loss : COCA alignment loss ("l1" or "entropy").
    t_min, t_max, eps : aggregation / stability params (same as JointCoca).
    log_every : print a running-average summary every N batches (0 = off).
    csv_path : if given, append per-batch diagnostics rows to this CSV file.
               The file is created (with a header) on first write if it doesn't
               exist, or appended to if it already does.
    """

    _CSV_FIELDS = [
        "corruption", "batch_in_corruption",
        "r_l", "r_s", "pred_l", "pred_s", "anchor", "tau",
        "acc_l", "acc_s", "duo_acc",
        "true_better", "anchor_correct",
    ]

    def __init__(
        self,
        proxy_kind: str,
        cfg_l: ProxyStats,
        cfg_s: ProxyStats,
        num_steps: int = 10,
        lr: float = 5e-2,
        loss: Literal["l1", "entropy"] = "l1",
        t_min: float = 0.1,
        t_max: float = 10.0,
        eps: float = 1e-4,
        log_every: int = 10,
        csv_path: str | None = None,
        calibrated_selection: bool = False,
    ):
        super().__init__()
        assert proxy_kind in _PROXY_KINDS, \
            f"proxy_kind must be one of {_PROXY_KINDS}, got '{proxy_kind}'"
        self.proxy_kind = proxy_kind
        self.cfg_l = cfg_l
        self.cfg_s = cfg_s
        self.calibrated_selection = calibrated_selection
        self._calib_checked = False
        self.t_min = t_min
        self.t_max = t_max
        self.eps = eps
        self.log_every = log_every
        if csv_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = Path(csv_path)
            self._csv_path: Path | None = p.parent / f"{p.name}_{ts}.csv"
        else:
            self._csv_path = None

        self._coca = CocaTemperature(num_steps=num_steps, lr=lr, loss=loss)

        # Feature hooks for prototype proxy (registered by setup_duo)
        self._ext_l: FeatureExtractor | None = None
        self._ext_s: FeatureExtractor | None = None

        # Per-batch state
        self._labels: torch.Tensor | None = None
        self._diag_done: bool = False
        self._sel_scores: tuple[float, float] | None = None

        # Per-corruption accumulators (cleared by report_and_reset_corruption_stats)
        self._corr_r_l:     list[float] = []
        self._corr_r_s:     list[float] = []
        self._corr_pred_l:  list[float] = []
        self._corr_pred_s:  list[float] = []
        self._corr_acc_l:   list[float] = []
        self._corr_acc_s:   list[float] = []
        self._corr_duo_acc: list[float] = []

        # Selection accuracy counters (non-tie batches only)
        self._n_sel_correct: int = 0
        self._n_sel_total:   int = 0

        # Current corruption label (set by set_corruption before each corruption)
        self._current_corruption: str = ""

        # Running counters for the per-N summary line
        self._n_batches: int = 0
        self._n_large_anchor: int = 0

    # ── Hook management ───────────────────────────────────────────────────── #

    def register_hooks(self, model_l: nn.Module, model_s: nn.Module) -> None:
        self._ext_l = FeatureExtractor(model_l, self.cfg_l.name)
        self._ext_s = FeatureExtractor(model_s, self.cfg_s.name)

    def remove_hooks(self) -> None:
        if self._ext_l is not None:
            self._ext_l.remove(); self._ext_l = None
        if self._ext_s is not None:
            self._ext_s.remove(); self._ext_s = None

    # ── Corruption / label injection ──────────────────────────────────────── #

    def set_corruption(self, label: str) -> None:
        self._current_corruption = label

    def set_labels(self, labels: torch.Tensor) -> None:
        self._labels = labels
        self._diag_done = False

    # ── Internals ─────────────────────────────────────────────────────────── #

    def _align_device(self, device: torch.device) -> None:
        if self._coca.rho.device != device:
            self._coca.rho = nn.Parameter(self._coca.rho.detach().to(device))
            self._coca.device = device
            self._coca._build_optimizer()

    def _check_calib(self) -> None:
        """Once: assert both models have a calib map for this proxy. Without it
        predicted_acc silently returns the raw value, so calibrated_selection
        would be a no-op — fail loudly instead."""
        if self._calib_checked:
            return
        missing = [
            tag for cfg, tag in ((self.cfg_l, "cfg_l"), (self.cfg_s, "cfg_s"))
            if self.proxy_kind not in cfg.calib
        ]
        assert not missing, (
            f"calibrated_selection=True but {missing} has no calib map for proxy "
            f"'{self.proxy_kind}'; attach a CalibrationMaps before inference."
        )
        self._calib_checked = True

    @torch.no_grad()
    def _proxy_scores(self, z_l: torch.Tensor, z_s: torch.Tensor) -> tuple[float, float]:
        if self.proxy_kind == "nuclear_norm":
            return float(nuclear_norm_score(z_l)), float(nuclear_norm_score(z_s))
        elif self.proxy_kind == "atc":
            assert self.cfg_l.atc_threshold is not None, \
                "atc proxy requires cfg_l.atc_threshold (build with build_proxy_stats)"
            return (float(atc_score(z_l, self.cfg_l.atc_threshold, self.cfg_l.atc_kind)),
                    float(atc_score(z_s, self.cfg_s.atc_threshold, self.cfg_s.atc_kind)))
        else:  # prototype (cosine or mahalanobis, per cfg.proto_metric)
            assert self._ext_l is not None and self._ext_s is not None, \
                "prototype proxy requires register_hooks() and build_proxy_stats()"
            return (self.cfg_l.prototype_proxy(self._ext_l._feats),
                    self.cfg_s.prototype_proxy(self._ext_s._feats))

    def _aggregate(self, z_anchor: torch.Tensor, z_source: torch.Tensor, tau: float) -> torch.Tensor:
        """Anchor-guided aggregation identical to JointCoca._aggregate."""
        p_e = (z_anchor + z_source / tau) / 2.0
        with torch.no_grad():
            anchor_max = z_anchor.max(dim=1).values
            ens_max    = p_e.max(dim=1).values
            T = ens_max / anchor_max.clamp_min(self.eps)
            T = torch.where(
                torch.isfinite(T) & (anchor_max > self.eps),
                T, torch.ones_like(T),
            )
            T = T.clamp(self.t_min, self.t_max).unsqueeze(1)
        return p_e / T

    def _forward(
        self, z_l: torch.Tensor, z_s: torch.Tensor
    ) -> tuple[torch.Tensor, float, float, str, float]:
        """Proxy → anchor selection → COCA fit → aggregate.

        Returns (z_duo, r_l, r_s, anchor_name, tau).
        """
        self._align_device(z_l.device)
        r_l, r_s = self._proxy_scores(z_l, z_s)

        # Selection score: calibrated predicted-accuracy or the raw proxy.
        if self.calibrated_selection:
            self._check_calib()
            s_l = self.cfg_l.predicted_acc(self.proxy_kind, r_l)
            s_s = self.cfg_s.predicted_acc(self.proxy_kind, r_s)
        else:
            s_l, s_s = r_l, r_s
        self._sel_scores = (s_l, s_s)

        if s_l >= s_s:
            z_anchor, z_source, anchor_name = z_l, z_s, "large"
        else:
            z_anchor, z_source, anchor_name = z_s, z_l, "small"

        tau = self._coca.adapt(z_anchor, z_source)  # @torch.enable_grad() inside
        z_duo = self._aggregate(z_anchor, z_source, tau)
        return z_duo, r_l, r_s, anchor_name, tau

    def _log_batch(
        self,
        z_duo: torch.Tensor,
        r_l: float, r_s: float,
        anchor: str, tau: float,
        z_l: torch.Tensor, z_s: torch.Tensor,
    ) -> None:
        """Print per-batch line and accumulate for per-corruption stats."""
        self._n_batches += 1
        n = self._n_batches
        if anchor == "large":
            self._n_large_anchor += 1

        parts = [
            f"[ProxyAnchorCoca {self.proxy_kind} batch {n:4d}]",
            f"r_l={r_l:.3f} r_s={r_s:.3f}",
            f"anchor={anchor}",
            f"tau={tau:.3f}",
        ]
        if self.calibrated_selection and self._sel_scores is not None:
            s_l, s_s = self._sel_scores
            parts.insert(2, f"acc_hat_l={s_l:.3f} acc_hat_s={s_s:.3f}")

        if self._labels is not None:
            labels = self._labels.to(z_l.device)
            acc_l   = float((z_l.detach().argmax(1) == labels).float().mean())
            acc_s   = float((z_s.detach().argmax(1) == labels).float().mean())
            duo_acc = float((z_duo.detach().argmax(1) == labels).float().mean())

            self._corr_r_l.append(r_l);    self._corr_acc_l.append(acc_l)
            self._corr_r_s.append(r_s);    self._corr_acc_s.append(acc_s)
            self._corr_duo_acc.append(duo_acc)

            pred_l = self.cfg_l.predicted_acc(self.proxy_kind, r_l)
            pred_s = self.cfg_s.predicted_acc(self.proxy_kind, r_s)
            self._corr_pred_l.append(pred_l)
            self._corr_pred_s.append(pred_s)

            # Selection accuracy: does proxy pick the truly-better model?
            if abs(acc_l - acc_s) > 1e-6:  # skip ties
                true_better = "large" if acc_l > acc_s else "small"
                self._n_sel_total += 1
                if anchor == true_better:
                    self._n_sel_correct += 1
            else:
                true_better = "tie"
            anchor_correct = (anchor == true_better)

            if self._csv_path is not None:
                need_header = not self._csv_path.exists()
                with self._csv_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
                    if need_header:
                        writer.writeheader()
                    writer.writerow({
                        "corruption": self._current_corruption,
                        "batch_in_corruption": n,
                        "r_l": r_l, "r_s": r_s,
                        "pred_l": pred_l, "pred_s": pred_s,
                        "anchor": anchor, "tau": tau,
                        "acc_l": acc_l, "acc_s": acc_s, "duo_acc": duo_acc,
                        "true_better": true_better,
                        "anchor_correct": anchor_correct,
                    })

            parts += [
                f"acc_l={acc_l:.3f} acc_s={acc_s:.3f} duo={duo_acc:.3f}",
                f"anchor={'✓' if anchor_correct else '✗'}",
            ]
            self._labels = None  # consume

        print(" ".join(parts))

        if self.log_every > 0 and n % self.log_every == 0:
            large_pct = 100.0 * self._n_large_anchor / n
            sel_str = ""
            if self._n_sel_total > 0:
                sel_acc = self._n_sel_correct / self._n_sel_total
                sel_str = (f"  sel_acc={sel_acc:.3f} "
                           f"({self._n_sel_correct}/{self._n_sel_total})")
            print(f"[ProxyAnchorCoca {self.proxy_kind} n={n}] "
                  f"large_anchor_rate={large_pct:.1f}%{sel_str}")

        self._diag_done = True

    def report_and_reset_corruption_stats(self, label: str) -> dict:
        """Compute stats for this corruption, print, and clear accumulators."""
        stats_l      = _corr_stats(self._corr_r_l,    self._corr_acc_l)
        stats_s      = _corr_stats(self._corr_r_s,    self._corr_acc_s)
        stats_pred_l = _corr_stats(self._corr_pred_l, self._corr_acc_l)
        stats_pred_s = _corr_stats(self._corr_pred_s, self._corr_acc_s)
        n = stats_l["n"]

        sel_acc = (self._n_sel_correct / self._n_sel_total
                   if self._n_sel_total > 0 else float("nan"))

        def _fmt(v: float) -> str:
            return f"{v:.3f}" if not (v != v) else "nan"

        if n > 0:
            sel_str = (f"  sel_acc={_fmt(sel_acc)} "
                       f"({self._n_sel_correct}/{self._n_sel_total} non-tie batches)")
            print(
                f"[ProxyAnchorCoca {self.proxy_kind} {label}] "
                f"n={n} batches with labels{sel_str}\n"
                f"  large  (proxy): R²={_fmt(stats_l['r2'])}  "
                f"r={_fmt(stats_l['pearson_r'])}  "
                f"ρ={_fmt(stats_l['spearman_rho'])}\n"
                f"  large  (pred):  R²={_fmt(stats_pred_l['r2'])}  "
                f"r={_fmt(stats_pred_l['pearson_r'])}  "
                f"ρ={_fmt(stats_pred_l['spearman_rho'])}\n"
                f"  small  (proxy): R²={_fmt(stats_s['r2'])}  "
                f"r={_fmt(stats_s['pearson_r'])}  "
                f"ρ={_fmt(stats_s['spearman_rho'])}\n"
                f"  small  (pred):  R²={_fmt(stats_pred_s['r2'])}  "
                f"r={_fmt(stats_pred_s['pearson_r'])}  "
                f"ρ={_fmt(stats_pred_s['spearman_rho'])}"
            )

        sel_correct = self._n_sel_correct
        sel_total   = self._n_sel_total

        self._corr_r_l.clear();    self._corr_acc_l.clear()
        self._corr_r_s.clear();    self._corr_acc_s.clear()
        self._corr_pred_l.clear(); self._corr_pred_s.clear()
        self._corr_duo_acc.clear()
        self._n_sel_correct = 0
        self._n_sel_total   = 0

        return {
            "sel_acc": sel_acc,
            "sel_correct": sel_correct,
            "sel_total": sel_total,
            "l_r2": stats_l["r2"],
            "l_pearson_r": stats_l["pearson_r"],
            "l_spearman_rho": stats_l["spearman_rho"],
            "l_pred_r2": stats_pred_l["r2"],
            "l_pred_pearson_r": stats_pred_l["pearson_r"],
            "l_pred_spearman_rho": stats_pred_l["spearman_rho"],
            "s_r2": stats_s["r2"],
            "s_pearson_r": stats_s["pearson_r"],
            "s_spearman_rho": stats_s["spearman_rho"],
            "s_pred_r2": stats_pred_s["r2"],
            "s_pred_pearson_r": stats_pred_s["pearson_r"],
            "s_pred_spearman_rho": stats_pred_s["spearman_rho"],
            "n": n,
        }

    # ── BaseJointCalibrator interface ─────────────────────────────────────── #

    def calibrate_with_grad(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        z_duo, r_l, r_s, anchor, tau = self._forward(logits_l, logits_s)
        self._log_batch(z_duo, r_l, r_s, anchor, tau, logits_l, logits_s)
        return z_duo

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_duo, r_l, r_s, anchor, tau = self._forward(logits_l, logits_s)
        if not self._diag_done:
            # no_adapt mode: calibrate_with_grad was never called this batch
            self._log_batch(z_duo, r_l, r_s, anchor, tau, logits_l, logits_s)
        return z_duo

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        pass

    @property
    def model(self):
        return _NoOpModule()