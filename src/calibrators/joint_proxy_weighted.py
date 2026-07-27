"""
joint_proxy_weighted.py
========================
Filtered-proxy soft weighting (paper Sections 2-5): continuous, reliability-
weighted combination of the two models.

Each batch:
  1. Raw proxy scores r_l, r_s (Section 2 — any src.reliability.proxies kind
     except "agreement", which scores the pair rather than a single model
     and so has no r_l/r_s split; see proxies/agreement.py).
  2. Calibrate each raw score onto a common predicted-accuracy scale via the
     attached CalibrationMaps (Section 3), then to log-odds (eq. 8,
     src.reliability.calibration.logit.to_logit).
  3. Denoise the log-odds score with a per-model temporal filter (Section 4:
     none / running_mean / ema / kalman — src.reliability.filters).
  4. Gate: w_l = sigmoid(beta * (x_l - x_s)), w_s = 1 - w_l (eq. 15).
  5. Combine against a frozen JointFixedTS's (T_l, T_s) prior — the proxy
     only sets the weight ratio between the two models, not their base
     scale (unidentifiable together, so T_l/T_s stay FIXED here; do not
     also fit temperatures inside this calibrator). Two pooling modes:
       "log"    z_duo = w_l*(z_l/T_l) + w_s*(z_s/T_s)
                Matches the product-of-experts aggregation used elsewhere in
                this codebase, but a single confidently-wrong (collapsed)
                model poisons the product.
       "linear" p_duo = w_l*softmax(z_l/T_l) + w_s*softmax(z_s/T_s)
                z_duo = log(p_duo)
                Degrades gracefully under model collapse: a bad model
                contributes at most its weight, not a multiplicative penalty.

Records per-batch diagnostics and GT accuracies (when labels are injected via
set_labels). After each corruption, report_and_reset_corruption_stats()
reports proxy<->accuracy correlation and the mean gate weight.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.calibrators.base import BaseJointCalibrator, _NoOpModule
from src.calibrators.joint_fixed_TS import JointFixedTS
from src.reliability.proxies.stats import ProxyStats, FeatureExtractor, PROXY_KINDS
from src.reliability.calibration.logit import to_logit
from src.reliability.filters.base import ScoreFilter
from src.reliability.filters.no_filter import NoFilter
from src.reliability.filters.running_mean import RunningMean
from src.reliability.filters.ema import EMA
from src.reliability.filters.kalman import Kalman

_PROXY_KINDS = PROXY_KINDS  # "agreement" is intentionally excluded (pair-level, no r_l/r_s split)
_FILTER_KINDS = {"none", "running_mean", "ema", "kalman"}
_POOL_KINDS = {"log", "linear"}


def _corr_stats(xs: list[float], ys: list[float]) -> dict:
    """R², Pearson r, Spearman ρ between xs and ys. Returns nans when n < 3."""
    from scipy.stats import pearsonr, spearmanr

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


def _build_filter(kind: str, prior: float, filter_kwargs: dict) -> ScoreFilter:
    if kind == "none":
        return NoFilter()
    if kind == "running_mean":
        return RunningMean()
    if kind == "ema":
        return EMA(alpha=filter_kwargs.get("alpha", 0.1), prior=prior)
    if kind == "kalman":
        return Kalman(
            q=filter_kwargs.get("q", 1e-3),
            r=filter_kwargs.get("r", 1e-1),
            prior_mean=prior,
            prior_var=filter_kwargs.get("prior_var", 1.0),
        )
    raise ValueError(f"filter kind must be one of {_FILTER_KINDS}, got '{kind}'")


class JointProxyWeighted(BaseJointCalibrator):
    """Continuous reliability-weighted combination of two models.

    Parameters
    ----------
    proxy_kind : any src.reliability.proxies.stats.PROXY_KINDS member
        ("nuclear_norm" | "atc" | "prototype" | "ac_mc" | "cot").
    cfg_l, cfg_s : ProxyStats for large and small model, with a
        CalibrationMaps for this proxy_kind attached to .calib (see
        src.reliability.calibration.maps).
    beta : gate sharpness (eq. 15). beta -> inf recovers hard anchor
        selection; beta = 0 is the equal-weight ensemble.
    pool : "log" or "linear" (see module docstring).
    filter_kind : "none" | "running_mean" | "ema" | "kalman" (Section 4).
    filter_kwargs : filter-specific knobs — ema: {"alpha"}; kalman:
        {"q", "r", "prior_var"}.
    prior_l, prior_s : each model's logit-space prior (typically
        to_logit(val_acc) — see src.reliability.calibration.logit), used as
        the ema/kalman reset point at each corruption boundary.
    base_ts : frozen JointFixedTS supplying T_l, T_s. If None, T_l = T_s = 1.0.
    eps : clip predicted accuracies into (eps, 1-eps) before the logit
        transform (eq. 6).
    csv_path : if given, append per-batch diagnostics rows to this CSV file.
    """

    _CSV_FIELDS = [
        "corruption", "batch_in_corruption",
        "r_l", "r_s", "a_l", "a_s", "x_l", "x_s", "w_l", "w_s",
        "acc_l", "acc_s", "duo_acc",
    ]

    def __init__(
        self,
        proxy_kind: str,
        cfg_l: ProxyStats,
        cfg_s: ProxyStats,
        beta: float = 4.0,
        pool: Literal["log", "linear"] = "linear",
        filter_kind: Literal["none", "running_mean", "ema", "kalman"] = "kalman",
        filter_kwargs: dict | None = None,
        prior_l: float = 0.0,
        prior_s: float = 0.0,
        base_ts: JointFixedTS | None = None,
        eps: float = 1e-3,
        csv_path: str | None = None,
        log_every: int = 10,
    ):
        super().__init__()
        assert proxy_kind in _PROXY_KINDS, \
            f"proxy_kind must be one of {sorted(_PROXY_KINDS)}, got '{proxy_kind}'"
        assert pool in _POOL_KINDS, f"pool must be one of {_POOL_KINDS}, got '{pool}'"
        assert filter_kind in _FILTER_KINDS, \
            f"filter_kind must be one of {_FILTER_KINDS}, got '{filter_kind}'"

        self.proxy_kind = proxy_kind
        self.cfg_l = cfg_l
        self.cfg_s = cfg_s
        self.beta = beta
        self.pool = pool
        self.filter_kind = filter_kind
        self.filter_kwargs = filter_kwargs or {}
        self.prior_l = prior_l
        self.prior_s = prior_s
        self.eps = eps
        self.log_every = log_every

        # base_ts is registered as a submodule (if given) so DynamicDuo's
        # `joint_calibrator.to(device)` moves its T_l/T_s along with everything
        # else; it is never fit here (unidentifiable jointly with the gate).
        self.base_ts = base_ts
        if base_ts is not None:
            for p in base_ts.parameters():
                p.requires_grad_(False)

        self._filter_l = _build_filter(filter_kind, prior_l, self.filter_kwargs)
        self._filter_s = _build_filter(filter_kind, prior_s, self.filter_kwargs)

        if csv_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = Path(csv_path)
            self._csv_path: Path | None = p.parent / f"{p.name}_{ts}.csv"
        else:
            self._csv_path = None

        # Feature hooks for the prototype proxy (registered by setup_duo).
        self._ext_l: FeatureExtractor | None = None
        self._ext_s: FeatureExtractor | None = None

        # Per-batch state. `_pending` caches calibrate_with_grad()'s result so
        # that the same-batch calibrate() call (dynamic_duo.py's no-grad
        # inference pass runs both, on the SAME logits, every batch) reuses it
        # instead of recomputing — the temporal filters are stateful, so
        # calling _forward() twice per batch would double-assimilate the same
        # observation and corrupt their tracked reliability.
        self._pending: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._current_corruption: str = ""
        self._n_batches: int = 0

        # Per-corruption accumulators (cleared by report_and_reset_corruption_stats)
        self._corr_r_l:   list[float] = []
        self._corr_r_s:   list[float] = []
        self._corr_acc_l: list[float] = []
        self._corr_acc_s: list[float] = []
        self._corr_w_l:   list[float] = []

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
        """New corruption stream: the model resets to source here, so the
        temporal filters reset to their priors too (Section 4)."""
        self._current_corruption = label
        self._filter_l.reset()
        self._filter_s.reset()

    def set_labels(self, labels: torch.Tensor) -> None:
        self._labels = labels

    # ── Internals ─────────────────────────────────────────────────────────── #

    @torch.no_grad()
    def _proxy_scores(self, z_l: torch.Tensor, z_s: torch.Tensor) -> tuple[float, float]:
        f_l = self._ext_l._feats if self._ext_l is not None else None
        f_s = self._ext_s._feats if self._ext_s is not None else None
        return (self.cfg_l.score(self.proxy_kind, z_l, f_l),
                self.cfg_s.score(self.proxy_kind, z_s, f_s))

    def _gate_and_combine(
        self, z_l: torch.Tensor, z_s: torch.Tensor, r_l: float, r_s: float,
    ) -> tuple[torch.Tensor, float, float, float, float, float]:
        """Sections 3-5: calibrate -> logit -> filter -> gate -> combine.

        Returns (z_duo, a_l, a_s, x_l, x_s, w_l).
        """
        a_l = min(max(self.cfg_l.predicted_acc(self.proxy_kind, r_l), self.eps), 1.0 - self.eps)
        a_s = min(max(self.cfg_s.predicted_acc(self.proxy_kind, r_s), self.eps), 1.0 - self.eps)

        x_l = self._filter_l.update(to_logit(a_l, self.eps))
        x_s = self._filter_s.update(to_logit(a_s, self.eps))

        w_l = float(torch.sigmoid(torch.tensor(self.beta * (x_l - x_s))))
        w_s = 1.0 - w_l

        T_l = float(self.base_ts.Tl.item()) if self.base_ts is not None else 1.0
        T_s = float(self.base_ts.Ts.item()) if self.base_ts is not None else 1.0

        if self.pool == "log":
            z_duo = w_l * (z_l / T_l) + w_s * (z_s / T_s)
        else:  # linear
            p_duo = w_l * F.softmax(z_l / T_l, dim=1) + w_s * F.softmax(z_s / T_s, dim=1)
            z_duo = torch.log(p_duo.clamp(min=1e-8))

        return z_duo, a_l, a_s, x_l, x_s, w_l

    def _forward(
        self, z_l: torch.Tensor, z_s: torch.Tensor
    ) -> tuple[torch.Tensor, float, float, float, float, float, float]:
        """Returns (z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l)."""
        r_l, r_s = self._proxy_scores(z_l, z_s)
        z_duo, a_l, a_s, x_l, x_s, w_l = self._gate_and_combine(z_l, z_s, r_l, r_s)
        return z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l

    def _log_batch(
        self,
        z_duo: torch.Tensor,
        r_l: float, r_s: float, a_l: float, a_s: float,
        x_l: float, x_s: float, w_l: float,
        z_l: torch.Tensor, z_s: torch.Tensor,
    ) -> None:
        self._n_batches += 1
        n = self._n_batches
        w_s = 1.0 - w_l

        parts = [
            f"[ProxyWeighted {self.proxy_kind} batch {n:4d}]",
            f"r_l={r_l:.3f} r_s={r_s:.3f}",
            f"a_l={a_l:.3f} a_s={a_s:.3f}",
            f"w_l={w_l:.3f} w_s={w_s:.3f}",
        ]

        acc_l = acc_s = duo_acc = float("nan")
        if self._labels is not None:
            labels = self._labels.to(z_l.device)
            acc_l   = float((z_l.detach().argmax(1) == labels).float().mean())
            acc_s   = float((z_s.detach().argmax(1) == labels).float().mean())
            duo_acc = float((z_duo.detach().argmax(1) == labels).float().mean())

            self._corr_r_l.append(r_l);   self._corr_acc_l.append(acc_l)
            self._corr_r_s.append(r_s);   self._corr_acc_s.append(acc_s)
            self._corr_w_l.append(w_l)

            if self._csv_path is not None:
                need_header = not self._csv_path.exists()
                with self._csv_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
                    if need_header:
                        writer.writeheader()
                    writer.writerow({
                        "corruption": self._current_corruption,
                        "batch_in_corruption": n,
                        "r_l": r_l, "r_s": r_s, "a_l": a_l, "a_s": a_s,
                        "x_l": x_l, "x_s": x_s, "w_l": w_l, "w_s": w_s,
                        "acc_l": acc_l, "acc_s": acc_s, "duo_acc": duo_acc,
                    })

            parts.append(f"acc_l={acc_l:.3f} acc_s={acc_s:.3f} duo={duo_acc:.3f}")
            self._labels = None  # consume

        print(" ".join(parts))

        if self.log_every > 0 and n % self.log_every == 0 and self._corr_w_l:
            avg_w_l = sum(self._corr_w_l) / len(self._corr_w_l)
            print(f"[ProxyWeighted {self.proxy_kind} n={n}] avg w_l={avg_w_l:.3f}")

    def report_and_reset_corruption_stats(self, label: str) -> dict:
        """R²/Pearson/Spearman between proxy and true per-batch accuracy for
        this corruption, plus the mean gate weight. Prints and clears."""
        stats_l = _corr_stats(self._corr_r_l, self._corr_acc_l)
        stats_s = _corr_stats(self._corr_r_s, self._corr_acc_s)
        n = stats_l["n"]
        mean_w_l = sum(self._corr_w_l) / len(self._corr_w_l) if self._corr_w_l else float("nan")

        def _fmt(v: float) -> str:
            return f"{v:.3f}" if not (v != v) else "nan"

        if n > 0:
            print(
                f"[ProxyWeighted {self.proxy_kind} {label}] n={n} batches with labels  "
                f"mean w_l={_fmt(mean_w_l)}\n"
                f"  large: R²={_fmt(stats_l['r2'])}  r={_fmt(stats_l['pearson_r'])}  "
                f"ρ={_fmt(stats_l['spearman_rho'])}\n"
                f"  small: R²={_fmt(stats_s['r2'])}  r={_fmt(stats_s['pearson_r'])}  "
                f"ρ={_fmt(stats_s['spearman_rho'])}"
            )

        result = {
            "mean_w_l": mean_w_l,
            "l_r2": stats_l["r2"], "l_pearson_r": stats_l["pearson_r"], "l_spearman_rho": stats_l["spearman_rho"],
            "s_r2": stats_s["r2"], "s_pearson_r": stats_s["pearson_r"], "s_spearman_rho": stats_s["spearman_rho"],
            "n": n,
        }

        self._corr_r_l.clear();   self._corr_acc_l.clear()
        self._corr_r_s.clear();   self._corr_acc_s.clear()
        self._corr_w_l.clear()
        return result

    # ── BaseJointCalibrator interface ─────────────────────────────────────── #

    def calibrate_with_grad(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l = self._forward(logits_l, logits_s)
        self._log_batch(z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l, logits_l, logits_s)
        self._pending = z_duo.detach()
        return z_duo

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        if self._pending is not None:
            # duo-adapting modes: calibrate_with_grad already ran (and logged)
            # on these same logits this batch — reuse it rather than running
            # _forward() again, which would double-assimilate the stateful
            # temporal filters.
            z_duo, self._pending = self._pending, None
            return z_duo
        with torch.no_grad():
            z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l = self._forward(logits_l, logits_s)
        self._log_batch(z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l, logits_l, logits_s)
        return z_duo

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        pass  # gate/filter have no offline-tunable params here; see setup.fit_beta

    @property
    def model(self):
        return _NoOpModule()


if __name__ == "__main__":
    torch.manual_seed(0)
    K, B = 10, 16

    def _make_calibrator(filter_kind="none", pool="linear", beta=4.0):
        cfg_l = ProxyStats(name="large", num_classes=K)
        cfg_s = ProxyStats(name="small", num_classes=K)
        # No CalibrationMaps attached: predicted_acc() falls back to the raw
        # (identity-calibrated) score, which is fine for a nuclear_norm proxy
        # since it's already roughly in [0, 1].
        return JointProxyWeighted(
            proxy_kind="nuclear_norm", cfg_l=cfg_l, cfg_s=cfg_s,
            beta=beta, pool=pool, filter_kind=filter_kind, log_every=0,
        )

    # A confidently-correct large model + a near-uniform (unreliable) small
    # model should gate mostly toward the large model.
    calib = _make_calibrator()
    labels = torch.randint(0, K, (B,))
    z_l_confident = torch.full((B, K), -2.0)
    z_l_confident[torch.arange(B), labels] = 8.0
    z_s_uniform = torch.zeros(B, K)
    calib.set_corruption("test")
    calib.set_labels(labels)
    z_duo = calib.calibrate_with_grad(z_l_confident, z_s_uniform)
    assert z_duo.shape == (B, K)
    assert (z_duo.argmax(1) == labels).float().mean() > 0.8

    # Calling calibrate() on the SAME batch must reuse the cached result, not
    # recompute (which would double-assimilate the stateful filters).
    calib2 = _make_calibrator(filter_kind="running_mean")
    calib2.set_corruption("test")
    calib2.set_labels(labels)
    z1 = calib2.calibrate_with_grad(z_l_confident, z_s_uniform)
    z2 = calib2.calibrate(z_l_confident, z_s_uniform)
    assert torch.allclose(z1, z2)
    assert calib2._filter_l._n == 1 and calib2._filter_s._n == 1, \
        "calibrate() must not re-assimilate an already-processed batch into the filter"

    # A fresh batch (no calibrate_with_grad call this time -> no_adapt-style
    # path) does update the filter exactly once via calibrate() alone.
    z3 = calib2.calibrate(z_l_confident, z_s_uniform)
    assert calib2._filter_l._n == 2 and calib2._filter_s._n == 2

    # set_corruption resets the temporal filters to their priors.
    kalman_calib = _make_calibrator(filter_kind="kalman")
    kalman_calib.set_corruption("c1")
    kalman_calib.calibrate(z_l_confident, z_s_uniform)
    assert kalman_calib._filter_l._mean != kalman_calib._filter_l.prior_mean
    kalman_calib.set_corruption("c2")
    assert kalman_calib._filter_l._mean == kalman_calib._filter_l.prior_mean

    # report_and_reset_corruption_stats returns a well-formed summary and
    # clears its accumulators.
    calib3 = _make_calibrator()
    calib3.set_corruption("c")
    for _ in range(5):
        labels = torch.randint(0, K, (B,))
        calib3.set_labels(labels)
        calib3.calibrate(z_l_confident, z_s_uniform)
    stats = calib3.report_and_reset_corruption_stats("c")
    assert stats["n"] == 5
    assert calib3._corr_r_l == []

    # "log" pool and "linear" pool both produce valid probability-normalisable
    # logits and agree at beta=0 (equal weighting) up to the pooling formula
    # difference (log-pool == a 50/50 average of logits; linear == of probs).
    for pool in ("log", "linear"):
        c = _make_calibrator(pool=pool, beta=0.0)
        c.set_corruption("t"); c.set_labels(torch.randint(0, K, (B,)))
        out = c.calibrate(z_l_confident, z_s_uniform)
        assert torch.isfinite(out).all()

    print("JointProxyWeighted self-test passed")
