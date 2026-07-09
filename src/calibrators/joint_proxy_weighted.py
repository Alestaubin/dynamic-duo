"""
joint_proxy_weighted.py
========================
Continuous, reliability-weighted combination of the two models — the soft
counterpart to JointProxyAnchorCoca's hard argmax anchor selection.

Each batch:
  1. Raw proxy scores r_l, r_s (nuclear_norm / nuclear_norm_cum / atc /
     prototype — see proxies.py). If `filter="ema_gram"`, the nuclear-norm
     proxies are computed from a recency-weighted running Gram matrix
     (RunningNuclearNorm(decay=...)) instead of the plain per-batch or
     unweighted-cumulative score — this filter needs the raw softmax matrix,
     so it necessarily runs BEFORE calibration.
  2. Calibrate each raw score onto a common, val-based reliability scale
     s_l, s_s (ProxyStats.reliability_score, `calib_mode` = "zscore" or
     "isotonic") so the two heterogeneous models' proxies are comparable.
  3. If `filter="kalman"`, denoise s_l, s_s with a scalar Kalman filter
     (ProxyFilter) in this already-unbounded logit-like space — this is the
     general-purpose filter, valid for any proxy_kind, and is intentionally
     applied post-calibration (see proxy_filter.py's docstring: filter in
     logit space, not raw [0,1] proxy space).
  4. Gate: w_l = sigmoid(beta * (s_l - s_s)), w_s = 1 - w_l. beta -> inf
     recovers hard anchor selection (JointProxyAnchorCoca); beta = 0 is the
     equal-weight ensemble.
  5. Combine using the validation-tuned base temperatures T_l, T_s (a frozen
     JointFixedTS) as the per-model prior scale — the proxy sets only the
     weight ratio between them. Two pooling modes:
       "log"    z_duo = w_l * (z_l / T_l) + w_s * (z_s / T_s)
                Matches today's product-of-experts aggregation, but a single
                confidently-wrong (collapsed) model poisons the product.
       "linear" p_duo = w_l * softmax(z_l / T_l) + w_s * softmax(z_s / T_s)
                z_duo = log(p_duo)
                Degrades gracefully under model collapse: a bad model
                contributes at most its weight, not a multiplicative penalty.
     Weight and temperature are unidentifiable together in the log pool, so
     T_l, T_s are held FIXED (from val NLL) here — do not also fit
     temperatures inside this calibrator.

Note: marginal accuracy is the optimal weight only under independent model
errors; ViT/ResNet errors on ImageNet-C are partially correlated, so
logit(predicted accuracy) alone is a slightly too-aggressive weight — beta is
fit on held-out dev-shift data (see calibrator_setup.fit_beta) rather than
trusted at face value.

Records per-batch diagnostics and GT accuracies (when labels are injected via
set_labels), mirroring JointProxyAnchorCoca. After each corruption,
report_and_reset_corruption_stats() reports proxy<->accuracy correlation, mean
w_l, and the duo-vs-best-single-model accuracy gap.
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
from src.calibrators.joint_proxy_anchor_coca import _PROXY_KINDS, _corr_stats
from src.proxies.proxies import (
    ProxyStats,
    FeatureExtractor,
    RunningNuclearNorm,
    nuclear_norm_score,
    atc_score,
    _logit,
)
from src.proxies.proxy_filter import ProxyFilter

_NUCLEAR_KINDS = {"nuclear_norm", "nuclear_norm_cum"}


class JointProxyWeighted(BaseJointCalibrator):
    """Continuous reliability-weighted combination of two models.

    Parameters
    ----------
    proxy_kind : "nuclear_norm" | "nuclear_norm_cum" | "atc" | "prototype"
    cfg_l, cfg_s : ProxyStats for large and small model. Must have
        val_acc / proxy_mean / proxy_std populated for this proxy_kind
        (calib_mode="zscore") and/or a .calib map attached (calib_mode=
        "isotonic") — see proxies.fit_val_reliability / calibration.py.
    beta : gate sharpness. beta -> inf = hard anchor, beta = 0 = equal weight.
    pool : "log" (sharpening, matches today's aggregation) or "linear"
        (mixture, collapse-robust). See module docstring.
    calib_mode : "zscore" (default, lightweight) or "isotonic".
    filter : "none" | "ema_gram" | "kalman". "ema_gram" only applies to
        nuclear-norm proxies (needs the raw softmax matrix); "kalman" applies
        to the calibrated score and needs cfg_l.val_acc / cfg_s.val_acc set.
    filter_kwargs : dict of filter-specific knobs:
        ema_gram -> {"decay": float}                 (default 0.1)
        kalman   -> {"q": float, "r": float, "prior_var": float}
                    (defaults 1e-3, 1e-1, 1.0)
    base_ts : frozen JointFixedTS supplying T_l, T_s (val-tuned priors). If
        None, T_l = T_s = 1.0.
    precision_weight : if True and filter="kalman", temper beta by the
        filters' posterior variance (beta_eff = beta / (1 + P_l + P_s)) so an
        uncertain reliability estimate pulls weights toward equal. Off by
        default.
    csv_path : if given, append per-batch diagnostics rows to this CSV file
        (timestamped filename, like JointProxyAnchorCoca).
    """

    _CSV_FIELDS = [
        "corruption", "batch_in_corruption",
        "r_l", "r_s", "s_l", "s_s", "w_l", "w_s",
        "acc_l", "acc_s", "duo_acc", "best_single_acc", "duo_vs_best_gap",
    ]

    def __init__(
        self,
        proxy_kind: str,
        cfg_l: ProxyStats,
        cfg_s: ProxyStats,
        *,
        beta: float = 4.0,
        pool: Literal["log", "linear"] = "linear",
        calib_mode: Literal["zscore", "isotonic"] = "zscore",
        filter: Literal["none", "ema_gram", "kalman"] = "kalman",
        filter_kwargs: dict | None = None,
        base_ts: JointFixedTS | None = None,
        precision_weight: bool = False,
        csv_path: str | None = None,
        log_every: int = 10,
    ):
        super().__init__()
        assert proxy_kind in _PROXY_KINDS, \
            f"proxy_kind must be one of {_PROXY_KINDS}, got '{proxy_kind}'"
        assert pool in ("log", "linear"), f"pool must be 'log' or 'linear', got '{pool}'"
        assert calib_mode in ("zscore", "isotonic"), \
            f"calib_mode must be 'zscore' or 'isotonic', got '{calib_mode}'"
        assert filter in ("none", "ema_gram", "kalman"), \
            f"filter must be 'none', 'ema_gram', or 'kalman', got '{filter}'"
        if filter == "ema_gram":
            assert proxy_kind in _NUCLEAR_KINDS, (
                "filter='ema_gram' needs the raw softmax matrix (Gram trick) "
                f"and only applies to {_NUCLEAR_KINDS}, got proxy_kind='{proxy_kind}'"
            )
        if filter == "kalman":
            assert cfg_l.val_acc is not None and cfg_s.val_acc is not None, (
                "filter='kalman' needs cfg_l.val_acc / cfg_s.val_acc as the filter "
                "prior; call proxies.fit_val_reliability on both first."
            )

        self.proxy_kind = proxy_kind
        self.cfg_l = cfg_l
        self.cfg_s = cfg_s
        self.beta = beta
        self.pool = pool
        self.calib_mode = calib_mode
        self.filter = filter
        self.filter_kwargs = filter_kwargs or {}
        self.base_ts = base_ts
        self.precision_weight = precision_weight
        self.log_every = log_every

        if csv_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = Path(csv_path)
            self._csv_path: Path | None = p.parent / f"{p.name}_{ts}.csv"
        else:
            self._csv_path = None

        # Feature hooks for prototype proxy (registered by setup_duo)
        self._ext_l: FeatureExtractor | None = None
        self._ext_s: FeatureExtractor | None = None

        # Nuclear-norm accumulator: active whenever proxy_kind is a nuclear
        # variant. decay=0 reproduces the plain cumulative average
        # (proxy_kind="nuclear_norm_cum", filter != "ema_gram"); a nonzero
        # decay from filter_kwargs is used when filter="ema_gram" (which
        # applies regardless of whether proxy_kind said "nuclear_norm" or
        # "nuclear_norm_cum" — both mean "use the Gram accumulator" once
        # ema_gram is selected).
        self._use_nuc_accumulator = proxy_kind in _NUCLEAR_KINDS and (
            proxy_kind == "nuclear_norm_cum" or filter == "ema_gram"
        )
        if self._use_nuc_accumulator:
            decay = self.filter_kwargs.get("decay", 0.1) if filter == "ema_gram" else 0.0
            self._acc_l = RunningNuclearNorm(decay=decay)
            self._acc_s = RunningNuclearNorm(decay=decay)

        # Scalar Kalman filters on the calibrated (logit-space) score.
        if filter == "kalman":
            q = self.filter_kwargs.get("q", 1e-3)
            r = self.filter_kwargs.get("r", 1e-1)
            prior_var = self.filter_kwargs.get("prior_var", 1.0)
            self._kf_l = ProxyFilter(q=q, r=r, prior_mean=_logit(cfg_l.val_acc), prior_var=prior_var)
            self._kf_s = ProxyFilter(q=q, r=r, prior_mean=_logit(cfg_s.val_acc), prior_var=prior_var)

        # Per-batch state
        self._labels: torch.Tensor | None = None
        self._diag_done: bool = False

        # Per-corruption accumulators (cleared by report_and_reset_corruption_stats)
        self._corr_r_l:   list[float] = []
        self._corr_r_s:   list[float] = []
        self._corr_s_l:   list[float] = []
        self._corr_s_s:   list[float] = []
        self._corr_acc_l: list[float] = []
        self._corr_acc_s: list[float] = []
        self._corr_w_l:   list[float] = []
        self._corr_gap:   list[float] = []

        self._current_corruption: str = ""
        self._n_batches: int = 0

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

    @torch.no_grad()
    def _raw_scores(self, z_l: torch.Tensor, z_s: torch.Tensor) -> tuple[float, float]:
        if self.proxy_kind in _NUCLEAR_KINDS:
            if self._use_nuc_accumulator:
                # Fold each batch into the running Gram exactly once per batch
                # (calibrate_with_grad + calibrate can both run per batch).
                if not self._diag_done:
                    self._acc_l.update(z_l)
                    self._acc_s.update(z_s)
                return self._acc_l.score(z_l), self._acc_s.score(z_s)
            return float(nuclear_norm_score(z_l)), float(nuclear_norm_score(z_s))
        elif self.proxy_kind == "atc":
            assert self.cfg_l.atc_threshold is not None, \
                "atc proxy requires cfg_l.atc_threshold (build with build_proxy_stats)"
            return (float(atc_score(z_l, self.cfg_l.atc_threshold, self.cfg_l.atc_kind)),
                    float(atc_score(z_s, self.cfg_s.atc_threshold, self.cfg_s.atc_kind)))
        else:  # prototype
            assert self._ext_l is not None and self._ext_s is not None, \
                "prototype proxy requires register_hooks() and build_proxy_stats()"
            return (self.cfg_l.prototype_proxy(self._ext_l._feats),
                    self.cfg_s.prototype_proxy(self._ext_s._feats))

    @torch.no_grad()
    def _calibrated_scores(self, r_l: float, r_s: float) -> tuple[float, float]:
        """raw -> calibrated (s_l, s_s) -> optional Kalman denoise.

        Split out from _gate so callers that need to grid-search beta (see
        calibrator_setup.fit_beta) can cache (s_l, s_s) per batch and re-weight
        cheaply per candidate, without re-running the stateful filter/calib
        pipeline once per beta.
        """
        s_l = self.cfg_l.reliability_score(self.proxy_kind, r_l, mode=self.calib_mode)
        s_s = self.cfg_s.reliability_score(self.proxy_kind, r_s, mode=self.calib_mode)

        if self.filter == "kalman":
            if not self._diag_done:
                self._kf_l.update(s_l)
                self._kf_s.update(s_s)
            s_l, s_s = self._kf_l.mean, self._kf_s.mean

        return s_l, s_s

    @torch.no_grad()
    def _weights(self, s_l: float, s_s: float, beta: float | None = None) -> tuple[float, float]:
        """calibrated scores -> gate weights. `beta` overrides self.beta (used
        by the beta grid-search); defaults to self.beta."""
        beta_eff = self.beta if beta is None else beta
        if self.precision_weight and self.filter == "kalman":
            beta_eff = beta_eff / (1.0 + self._kf_l.var + self._kf_s.var)

        gap = torch.tensor(beta_eff * (s_l - s_s))
        w_l = float(torch.sigmoid(gap))
        w_s = 1.0 - w_l
        return w_l, w_s

    def _gate(self, r_l: float, r_s: float) -> tuple[float, float, float, float]:
        s_l, s_s = self._calibrated_scores(r_l, r_s)
        w_l, w_s = self._weights(s_l, s_s)
        return s_l, s_s, w_l, w_s

    def _combine(self, z_l: torch.Tensor, z_s: torch.Tensor, w_l: float, w_s: float) -> torch.Tensor:
        T_l = self.base_ts.Tl.item() if self.base_ts is not None else 1.0
        T_s = self.base_ts.Ts.item() if self.base_ts is not None else 1.0
        if self.pool == "log":
            return w_l * (z_l / T_l) + w_s * (z_s / T_s)
        p_duo = w_l * F.softmax(z_l / T_l, dim=1) + w_s * F.softmax(z_s / T_s, dim=1)
        return torch.log(p_duo.clamp_min(1e-8))

    def _forward(
        self, z_l: torch.Tensor, z_s: torch.Tensor
    ) -> tuple[torch.Tensor, float, float, float, float, float, float]:
        """proxy -> calibrate/filter -> gate -> combine.

        Returns (z_duo, r_l, r_s, s_l, s_s, w_l, w_s).
        """
        r_l, r_s = self._raw_scores(z_l, z_s)
        s_l, s_s, w_l, w_s = self._gate(r_l, r_s)
        z_duo = self._combine(z_l, z_s, w_l, w_s)
        return z_duo, r_l, r_s, s_l, s_s, w_l, w_s

    def _log_batch(
        self,
        z_duo: torch.Tensor,
        r_l: float, r_s: float, s_l: float, s_s: float, w_l: float, w_s: float,
        z_l: torch.Tensor, z_s: torch.Tensor,
    ) -> None:
        self._n_batches += 1
        n = self._n_batches

        parts = [
            f"[ProxyWeighted {self.proxy_kind} batch {n:4d}]",
            f"r_l={r_l:.3f} r_s={r_s:.3f}",
            f"s_l={s_l:.3f} s_s={s_s:.3f}",
            f"w_l={w_l:.3f}",
        ]

        if self._labels is not None:
            labels = self._labels.to(z_l.device)
            acc_l   = float((z_l.detach().argmax(1) == labels).float().mean())
            acc_s   = float((z_s.detach().argmax(1) == labels).float().mean())
            duo_acc = float((z_duo.detach().argmax(1) == labels).float().mean())
            best_single = max(acc_l, acc_s)
            gap = duo_acc - best_single

            self._corr_r_l.append(r_l);   self._corr_s_l.append(s_l); self._corr_acc_l.append(acc_l)
            self._corr_r_s.append(r_s);   self._corr_s_s.append(s_s); self._corr_acc_s.append(acc_s)
            self._corr_w_l.append(w_l)
            self._corr_gap.append(gap)

            if self._csv_path is not None:
                need_header = not self._csv_path.exists()
                with self._csv_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
                    if need_header:
                        writer.writeheader()
                    writer.writerow({
                        "corruption": self._current_corruption,
                        "batch_in_corruption": n,
                        "r_l": r_l, "r_s": r_s, "s_l": s_l, "s_s": s_s,
                        "w_l": w_l, "w_s": w_s,
                        "acc_l": acc_l, "acc_s": acc_s, "duo_acc": duo_acc,
                        "best_single_acc": best_single, "duo_vs_best_gap": gap,
                    })

            parts += [f"acc_l={acc_l:.3f} acc_s={acc_s:.3f} duo={duo_acc:.3f} gap={gap:+.3f}"]
            self._labels = None  # consume

        print(" ".join(parts))

        if self.log_every > 0 and n % self.log_every == 0 and self._corr_w_l:
            mean_w_l = sum(self._corr_w_l) / len(self._corr_w_l)
            mean_gap = sum(self._corr_gap) / len(self._corr_gap)
            print(f"[ProxyWeighted {self.proxy_kind} n={n}] "
                  f"mean_w_l={mean_w_l:.3f}  mean_duo_vs_best_gap={mean_gap:+.4f}")

        self._diag_done = True

    def report_and_reset_corruption_stats(self, label: str) -> dict:
        stats_l = _corr_stats(self._corr_r_l, self._corr_acc_l)
        stats_s = _corr_stats(self._corr_r_s, self._corr_acc_s)
        n = stats_l["n"]

        mean_w_l = sum(self._corr_w_l) / len(self._corr_w_l) if self._corr_w_l else float("nan")
        mean_gap = sum(self._corr_gap) / len(self._corr_gap) if self._corr_gap else float("nan")

        def _fmt(v: float) -> str:
            return f"{v:.3f}" if not (v != v) else "nan"

        if n > 0:
            print(
                f"[ProxyWeighted {self.proxy_kind} {label}] n={n} batches with labels  "
                f"mean_w_l={_fmt(mean_w_l)}  mean_duo_vs_best_gap={_fmt(mean_gap)}\n"
                f"  large (proxy): R²={_fmt(stats_l['r2'])}  r={_fmt(stats_l['pearson_r'])}  "
                f"ρ={_fmt(stats_l['spearman_rho'])}\n"
                f"  small (proxy): R²={_fmt(stats_s['r2'])}  r={_fmt(stats_s['pearson_r'])}  "
                f"ρ={_fmt(stats_s['spearman_rho'])}"
            )

        result = {
            "mean_w_l": mean_w_l,
            "mean_duo_vs_best_gap": mean_gap,
            "l_r2": stats_l["r2"], "l_pearson_r": stats_l["pearson_r"], "l_spearman_rho": stats_l["spearman_rho"],
            "s_r2": stats_s["r2"], "s_pearson_r": stats_s["pearson_r"], "s_spearman_rho": stats_s["spearman_rho"],
            "n": n,
        }

        self._corr_r_l.clear();  self._corr_acc_l.clear()
        self._corr_r_s.clear();  self._corr_acc_s.clear()
        self._corr_s_l.clear();  self._corr_s_s.clear()
        self._corr_w_l.clear();  self._corr_gap.clear()
        if self._use_nuc_accumulator:
            self._acc_l.reset(); self._acc_s.reset()
        if self.filter == "kalman":
            self._kf_l.reset(); self._kf_s.reset()

        return result

    # ── BaseJointCalibrator interface ─────────────────────────────────────── #

    def calibrate_with_grad(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        z_duo, r_l, r_s, s_l, s_s, w_l, w_s = self._forward(logits_l, logits_s)
        self._log_batch(z_duo, r_l, r_s, s_l, s_s, w_l, w_s, logits_l, logits_s)
        return z_duo

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_duo, r_l, r_s, s_l, s_s, w_l, w_s = self._forward(logits_l, logits_s)
        if not self._diag_done:
            self._log_batch(z_duo, r_l, r_s, s_l, s_s, w_l, w_s, logits_l, logits_s)
        return z_duo

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        pass

    @property
    def model(self):
        return _NoOpModule()


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    C = 10

    def _mk_cfg(name, val_acc, mean, std):
        cfg = ProxyStats(name=name, num_classes=C)
        cfg.val_acc = val_acc
        cfg.proxy_mean["nuclear_norm"] = mean
        cfg.proxy_std["nuclear_norm"] = std
        return cfg

    cfg_l = _mk_cfg("large", val_acc=0.8, mean=0.5, std=0.1)
    cfg_s = _mk_cfg("small", val_acc=0.6, mean=0.5, std=0.1)

    def _logits_with_score(target_score, n=16, c=C):
        # Higher target_score -> peaked softmax rows (higher nuclear norm).
        base = torch.randn(n, c)
        return base * (1.0 + 10.0 * target_score)

    # ── beta -> inf recovers hard argmax selection ────────────────────────
    jpw_hard = JointProxyWeighted(
        "nuclear_norm", cfg_l, cfg_s, beta=1e6, pool="linear",
        calib_mode="zscore", filter="none", log_every=0,
    )
    z_l = _logits_with_score(0.9)   # r_l high -> s_l high (large more reliable)
    z_s = _logits_with_score(0.1)   # r_s low
    _, r_l, r_s, s_l, s_s, w_l, w_s = jpw_hard._forward(z_l, z_s)
    assert s_l > s_s
    assert w_l > 0.999 and w_s < 0.001, (w_l, w_s)
    print("JointProxyWeighted beta->inf hard-selection self-test passed")

    # Flip which model looks better; weight should flip too.
    _, *_, w_l2, w_s2 = jpw_hard._forward(z_s, z_l)
    assert w_l2 < 0.001 and w_s2 > 0.999, (w_l2, w_s2)
    print("JointProxyWeighted beta->inf flip self-test passed")

    # ── beta = 0 is the equal-weight ensemble ──────────────────────────────
    jpw_equal = JointProxyWeighted(
        "nuclear_norm", cfg_l, cfg_s, beta=0.0, pool="linear",
        calib_mode="zscore", filter="none", log_every=0,
    )
    _, *_, w_l3, w_s3 = jpw_equal._forward(z_l, z_s)
    assert abs(w_l3 - 0.5) < 1e-9 and abs(w_s3 - 0.5) < 1e-9
    print("JointProxyWeighted beta=0 equal-weight self-test passed")

    # ── log vs linear pool both run end-to-end; linear is collapse-robust ──
    torch.manual_seed(1)
    z_l_ok = torch.randn(8, C)
    z_s_collapsed = torch.zeros(8, C); z_s_collapsed[:, 0] = 50.0  # degenerate, over-confident & wrong half the time
    labels = torch.randint(1, C, (8,))  # never class 0 -> small model always wrong

    for pool in ("log", "linear"):
        jpw = JointProxyWeighted(
            "nuclear_norm", cfg_l, cfg_s, beta=2.0, pool=pool,
            calib_mode="zscore", filter="none", log_every=0,
        )
        z_duo, *_ , w_l4, w_s4 = jpw._forward(z_l_ok, z_s_collapsed)
        nll = F.cross_entropy(z_duo, labels).item()
        if pool == "linear":
            linear_nll = nll
        else:
            log_nll = nll
    assert linear_nll < log_nll, (
        f"linear pool should be more robust to a collapsed model than log pool "
        f"(linear={linear_nll:.3f}, log={log_nll:.3f})"
    )
    print(f"JointProxyWeighted pool collapse-robustness self-test passed "
          f"(linear_nll={linear_nll:.3f} < log_nll={log_nll:.3f})")

    # ── reset clears filters/accumulators ──────────────────────────────────
    jpw_kf = JointProxyWeighted(
        "nuclear_norm", cfg_l, cfg_s, beta=2.0, pool="linear",
        calib_mode="zscore", filter="kalman",
        filter_kwargs={"q": 1e-2, "r": 1e-1}, log_every=0,
    )
    jpw_kf.set_labels(torch.randint(0, C, (8,)))
    jpw_kf.calibrate_with_grad(z_l_ok, z_l_ok * 2)
    assert jpw_kf._kf_l.mean != jpw_kf._kf_l.prior_mean
    jpw_kf.report_and_reset_corruption_stats("dummy")
    assert jpw_kf._kf_l.mean == jpw_kf._kf_l.prior_mean
    print("JointProxyWeighted reset self-test passed")

    # ── ema_gram filter requires a nuclear proxy_kind ──────────────────────
    try:
        JointProxyWeighted("atc", cfg_l, cfg_s, filter="ema_gram")
        raise AssertionError("expected assertion error for ema_gram + non-nuclear proxy_kind")
    except AssertionError as e:
        assert "ema_gram" in str(e)
    print("JointProxyWeighted ema_gram validation self-test passed")

    print("joint_proxy_weighted self-test passed")
