"""
joint_proxy_weighted.py
========================
Filtered-proxy soft weighting (paper Sections 2-5): continuous, reliability-
weighted combination of the two models.

proxy_batch_size vs. adaptation_batch_size (paper Section 1): these are two
ENTIRELY INDEPENDENT knobs. adaptation_batch_size is simply however many
samples arrive in one calibrate()/calibrate_with_grad() call (whatever the
caller's DataLoader batch size happens to be) -- this class never needs to
be told it explicitly. Samples accumulate in a proxy bucket until
proxy_batch_size is reached, at which point the proxy score / calibration /
filter / gate pipeline below runs fresh on everything in the bucket. Every
adaptation batch still gets its own combined (gated) output every call (the
TENT loss needs one every step) -- but if adaptation_batch_size >
proxy_batch_size, that bucket can fill (and the gate weight can change)
SEVERAL times within a single call: different rows of the same returned
combination then use different w_l, which is the correct behavior for a
continuous, call-boundary-agnostic stream, not an approximation of it.
proxy_batch_size=1 (the default) means the bucket is full after every
SINGLE sample -- there is no implicit floor at "one adaptation batch".

Each proxy batch:
  1. Raw proxy scores r_l, r_s (Section 2 — any src.reliability.proxies kind
     except "agreement", which scores the pair rather than a single model
     and so has no r_l/r_s split; see proxies/agreement.py).
  2. Calibrate each raw score onto a common predicted-accuracy scale via the
     attached CalibrationMaps (Section 3), then to log-odds (eq. 8,
     src.reliability.calibration.logit.to_logit).
  3. Denoise the log-odds score with a per-model temporal filter (Section 4:
     none / running_mean / ema / kalman — src.reliability.filters). The
     filter's own stream is thus indexed by proxy batches, matching the
     paper's t = 1, 2, ... (Section 1), not by adaptation batches.
  4. Gate: w_l = sigmoid(beta * (x_l - x_s)), w_s = 1 - w_l (eq. 15).

Every adaptation batch (Section 5 combination) then:
  5. Combines against a frozen JointFixedTS's (T_l, T_s) prior — the proxy
     only sets the weight ratio between the two models, not their base
     scale (unidentifiable together, so T_l/T_s stay FIXED here; do not
     also fit temperatures inside this calibrator):
       z_duo = w_l*(z_l/T_l) + w_s*(z_s/T_s)
     Product-of-experts logit pooling, matching the aggregation used
     elsewhere in this codebase (JointFixedTS.combine_logits).

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
        ("nuclear_norm" | "atc" | "prototype" | "ac_mc" | "cot" | "oracle").
        "oracle" is a cheat (uses ground-truth labels as the score) meant only
        as an upper-bound reference — never a real deployment signal.
    cfg_l, cfg_s : ProxyStats for large and small model, with a
        CalibrationMaps for this proxy_kind attached to .calib (see
        src.reliability.calibration.maps).
    beta : gate sharpness (eq. 15). beta -> inf recovers hard anchor
        selection; beta = 0 is the equal-weight ensemble.
    filter_kind : "none" | "running_mean" | "ema" | "kalman" (Section 4).
    filter_kwargs : filter-specific knobs — ema: {"alpha"}; kalman:
        {"q", "r", "prior_var"}.
    prior_l, prior_s : each model's logit-space prior (typically
        to_logit(val_acc) — see src.reliability.calibration.logit), used as
        the ema/kalman reset point at each corruption boundary, and as the
        initial gate weight (via the sigmoid) before the first proxy batch
        completes.
    base_ts : frozen JointFixedTS supplying T_l, T_s. If None, T_l = T_s = 1.0.
    eps : clip predicted accuracies into (eps, 1-eps) before the logit
        transform (eq. 6).
    proxy_batch_size : number of samples to aggregate into one proxy
        computation (Section 1's b_t), independent of the adaptation batch
        size. 1 (default) recomputes the gate every adaptation batch.
    csv_path : if given, append per-batch diagnostics rows to this CSV file.
    """

    _CSV_FIELDS = [
        "corruption", "n_refreshes",
        "r_l", "r_s", "a_l", "a_s", "x_l", "x_s", "w_l", "w_s",
        "acc_l", "acc_s", "duo_acc", "n",
    ]

    def __init__(
        self,
        proxy_kind: str,
        cfg_l: ProxyStats,
        cfg_s: ProxyStats,
        beta: float = 4.0,
        filter_kind: Literal["none", "running_mean", "ema", "kalman"] = "kalman",
        filter_kwargs: dict | None = None,
        prior_l: float = 0.0,
        prior_s: float = 0.0,
        base_ts: JointFixedTS | None = None,
        eps: float = 1e-3,
        proxy_batch_size: int = 1,
        csv_path: str | None = None,
        log_every: int = 10,
        verbose: bool = True,
    ):
        super().__init__()
        assert proxy_kind in _PROXY_KINDS, \
            f"proxy_kind must be one of {sorted(_PROXY_KINDS)}, got '{proxy_kind}'"
        assert filter_kind in _FILTER_KINDS, \
            f"filter_kind must be one of {_FILTER_KINDS}, got '{filter_kind}'"
        assert proxy_batch_size >= 1, f"proxy_batch_size must be >= 1, got {proxy_batch_size}"

        self.proxy_kind = proxy_kind
        self.cfg_l = cfg_l
        self.cfg_s = cfg_s
        self.beta = beta
        self.filter_kind = filter_kind
        self.filter_kwargs = filter_kwargs or {}
        self.prior_l = prior_l
        self.prior_s = prior_s
        self.eps = eps
        self.proxy_batch_size = proxy_batch_size
        self.log_every = log_every
        self.verbose = verbose

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

        # `_pending` caches calibrate_with_grad()'s result so that the
        # same-batch calibrate() call (dynamic_duo.py's no-grad inference
        # pass runs both, on the SAME logits, every batch) reuses it
        # instead of recomputing — the temporal filters are stateful, so
        # calling _forward() twice per batch would double-assimilate the
        # same observation and corrupt their tracked reliability.
        self._pending: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._current_corruption: str = ""
        self.n_refreshes: int = 0  # how many times the proxy bucket has been solved

        # Proxy-batch accumulation buffer (Section 1's b_t): filled sample
        # by sample as calls come in, flushed (see _flush_bucket) every
        # time it reaches proxy_batch_size — possibly several times within
        # one call. Cleared on set_corruption() too, so a partial buffer
        # never leaks across a corruption boundary.
        self._buf_z_l: list[torch.Tensor] = []
        self._buf_z_s: list[torch.Tensor] = []
        self._buf_f_l: list[torch.Tensor] = []
        self._buf_f_s: list[torch.Tensor] = []
        self._buf_labels: list[torch.Tensor] = []
        self._buf_n: int = 0

        # Stream-length bookkeeping (set by set_corruption's total_samples):
        # when a stream's length isn't a multiple of proxy_batch_size, the
        # trailing remainder would otherwise never reach the threshold below
        # and silently combine with a STALE gate from the previous proxy
        # batch — see _flush_bucket's stream_exhausted check.
        self._stream_total_samples: int | None = None
        self._stream_samples_seen: int = 0

        # Cached gate outputs, held constant between proxy-batch flushes and
        # used to combine every adaptation batch (or sub-slice of one) in
        # the meantime. Initialised from the priors so the very first
        # (possibly incomplete) proxy batch still gates sensibly.
        self._cached_r_l: float = 0.0
        self._cached_r_s: float = 0.0
        self._cached_a_l: float = 0.5
        self._cached_a_s: float = 0.5
        self._cached_x_l: float = prior_l
        self._cached_x_s: float = prior_s
        self._cached_w_l: float = float(torch.sigmoid(torch.tensor(beta * (prior_l - prior_s))))

        # Per-corruption accumulators (cleared by report_and_reset_corruption_stats)
        self._corr_r_l:   list[float] = []
        self._corr_r_s:   list[float] = []
        self._corr_acc_l: list[float] = []
        self._corr_acc_s: list[float] = []
        self._corr_w_l:   list[float] = []

        if verbose:
            T_l = float(base_ts.Tl.item()) if base_ts is not None else 1.0
            T_s = float(base_ts.Ts.item()) if base_ts is not None else 1.0
            print(
                f"\n{'#' * 78}\n"
                f"# JointProxyWeighted CONFIG\n"
                f"#   proxy_kind={proxy_kind}  PROXY_BATCH_SIZE={proxy_batch_size}  beta={beta}\n"
                f"#   filter_kind={filter_kind}  filter_kwargs={self.filter_kwargs}\n"
                f"#   prior_l={prior_l}  prior_s={prior_s}  eps={eps}  verbose={verbose}\n"
                f"#   base_ts: T_l={T_l:.4f}  T_s={T_s:.4f}"
                f"{'  (base_ts=None, defaulted to 1.0/1.0)' if base_ts is None else ''}\n"
                f"{'#' * 78}\n"
            )

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

    def set_corruption(self, label: str, total_samples: int | None = None) -> None:
        """New corruption stream: the model resets to source here, so the
        temporal filters reset to their priors too (Section 4), and any
        partially-filled proxy batch from the previous corruption is
        discarded rather than mixed into the new stream.

        total_samples, if given (the stream's exact sample count — e.g.
        len(loader.dataset)), lets _flush_bucket detect when the trailing
        remainder of a stream can never reach proxy_batch_size on its own
        and force a flush anyway, instead of that tail silently combining
        with a stale gate from the previous proxy batch.
        """
        self._current_corruption = label
        self._filter_l.reset()
        self._filter_s.reset()
        self._buf_z_l.clear(); self._buf_z_s.clear()
        self._buf_f_l.clear(); self._buf_f_s.clear()
        self._buf_labels.clear()
        self._buf_n = 0
        self._stream_total_samples = total_samples
        self._stream_samples_seen = 0
        self._cached_x_l, self._cached_x_s = self.prior_l, self.prior_s
        self._cached_w_l = float(torch.sigmoid(torch.tensor(self.beta * (self.prior_l - self.prior_s))))

    def set_labels(self, labels: torch.Tensor) -> None:
        self._labels = labels

    # ── Internals ─────────────────────────────────────────────────────────── #

    @torch.no_grad()
    def _proxy_scores(
        self,
        z_l: torch.Tensor, z_s: torch.Tensor,
        f_l: torch.Tensor | None, f_s: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> tuple[float, float]:
        # Only the cheating OracleProxy (proxy_kind="oracle") actually uses
        # labels; every other proxy ignores the kwarg.
        return (self.cfg_l.score(self.proxy_kind, z_l, f_l, labels=labels),
                self.cfg_s.score(self.proxy_kind, z_s, f_s, labels=labels))

    def _gate(self, r_l: float, r_s: float) -> tuple[float, float, float, float, float]:
        """Sections 3-4: calibrate -> logit -> filter -> gate.

        Returns (a_l, a_s, x_l, x_s, w_l). Runs once per proxy batch (see
        _flush_bucket).
        """
        a_l = min(max(self.cfg_l.predicted_acc(self.proxy_kind, r_l), self.eps), 1.0 - self.eps)
        a_s = min(max(self.cfg_s.predicted_acc(self.proxy_kind, r_s), self.eps), 1.0 - self.eps)

        x_l = self._filter_l.update(to_logit(a_l, self.eps))
        x_s = self._filter_s.update(to_logit(a_s, self.eps))

        w_l = float(torch.sigmoid(torch.tensor(self.beta * (x_l - x_s))))
        return a_l, a_s, x_l, x_s, w_l

    def _combine(self, z_l: torch.Tensor, z_s: torch.Tensor, w_l: float) -> torch.Tensor:
        """Section 5: combine a slice of logits at a given gate weight.
        Product-of-experts logit pooling (see module docstring)."""
        w_s = 1.0 - w_l
        T_l = float(self.base_ts.Tl.item()) if self.base_ts is not None else 1.0
        T_s = float(self.base_ts.Ts.item()) if self.base_ts is not None else 1.0
        return w_l * (z_l / T_l) + w_s * (z_s / T_s)

    def _flush_bucket(self) -> None:
        """Run the proxy pipeline (Sections 2-4) on everything currently
        buffered, cache the result, clear the buffer, and (if verbose)
        announce the refresh — called the moment the proxy bucket reaches
        proxy_batch_size (or the stream is exhausted, see set_corruption).
        Also updates the per-corruption correlation-diagnostic accumulators
        and the CSV log, keyed to THIS bucket's own accuracy — more correct
        than keying them to whatever adaptation batch happened to trigger
        the flush, since r_l/w_l are only ever fresh right here.
        """
        agg_z_l = torch.cat(self._buf_z_l, dim=0)
        agg_z_s = torch.cat(self._buf_z_s, dim=0)
        agg_f_l = torch.cat(self._buf_f_l, dim=0) if self._buf_f_l else None
        agg_f_s = torch.cat(self._buf_f_s, dim=0) if self._buf_f_s else None
        agg_labels = torch.cat(self._buf_labels, dim=0) if self._buf_labels else None
        n = agg_z_l.shape[0]

        r_l, r_s = self._proxy_scores(agg_z_l, agg_z_s, agg_f_l, agg_f_s, agg_labels)
        a_l, a_s, x_l, x_s, w_l = self._gate(r_l, r_s)
        self._cached_r_l, self._cached_r_s = r_l, r_s
        self._cached_a_l, self._cached_a_s = a_l, a_s
        self._cached_x_l, self._cached_x_s = x_l, x_s
        self._cached_w_l = w_l
        self.n_refreshes += 1

        self._buf_z_l.clear(); self._buf_z_s.clear()
        self._buf_f_l.clear(); self._buf_f_s.clear()
        self._buf_labels.clear()
        self._buf_n = 0

        acc_l = acc_s = duo_acc = float("nan")
        if agg_labels is not None:
            with torch.no_grad():
                z_duo = self._combine(agg_z_l, agg_z_s, w_l)
            acc_l = float((agg_z_l.argmax(1) == agg_labels).float().mean())
            acc_s = float((agg_z_s.argmax(1) == agg_labels).float().mean())
            duo_acc = float((z_duo.argmax(1) == agg_labels).float().mean())
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
                        "n_refreshes": self.n_refreshes,
                        "r_l": r_l, "r_s": r_s, "a_l": a_l, "a_s": a_s,
                        "x_l": x_l, "x_s": x_s, "w_l": w_l, "w_s": 1.0 - w_l,
                        "acc_l": acc_l, "acc_s": acc_s, "duo_acc": duo_acc, "n": n,
                    })

        if self.verbose:
            parts = [
                f"[ProxyWeighted {self.proxy_kind} PROXY BATCH #{self.n_refreshes} full, n={n}]",
                f"r_l={r_l:.3f} r_s={r_s:.3f}",
                f"a_l={a_l:.3f} a_s={a_s:.3f}",
                f"w_l={w_l:.3f} w_s={1.0 - w_l:.3f}",
            ]
            if agg_labels is not None:
                parts.append(f"acc_l={acc_l:.3f} acc_s={acc_s:.3f} duo={duo_acc:.3f}")
            print(" ".join(parts))
            if self.log_every > 0 and self.n_refreshes % self.log_every == 0 and self._corr_w_l:
                avg_w_l = sum(self._corr_w_l) / len(self._corr_w_l)
                print(f"[ProxyWeighted {self.proxy_kind} n_refreshes={self.n_refreshes}] avg w_l={avg_w_l:.3f}")

    def _forward(
        self, z_l: torch.Tensor, z_s: torch.Tensor
    ) -> tuple[torch.Tensor, float, float, float, float, float, float, float]:
        """Combine one adaptation batch (adaptation_batch_size = z_l.shape[0]
        — see module docstring), feeding the proxy bucket sample-range by
        sample-range and flushing it (see _flush_bucket) every time it
        fills — possibly several times within one call if proxy_batch_size
        < adaptation_batch_size. Each slice is combined with whatever w_l
        was current at that point in the stream, then all slices are
        concatenated back into one adaptation_batch_size-sized output, in
        order. Returns (z_duo, r_l, r_s, a_l, a_s, x_l, x_s, w_l) — the
        LATEST cached values as of the end of this call.

        Deliberately quiet by itself (only _flush_bucket prints, gated by
        self.verbose) regardless of caller — fit_beta and
        calibrate_gate_oracle.py's sweep call this directly across many
        batches and rely on staying quiet; set verbose=False at
        construction (or toggle self.verbose) for those.
        """
        f_l_full = self._ext_l._feats.detach() if self._ext_l is not None else None
        f_s_full = self._ext_s._feats.detach() if self._ext_s is not None else None
        labels_full = self._labels.to(z_l.device) if self._labels is not None else None
        self._labels = None  # consume

        adaptation_batch_size = z_l.shape[0]
        out_chunks = []
        start = 0
        while start < adaptation_batch_size:
            take = min(self.proxy_batch_size - self._buf_n, adaptation_batch_size - start)
            sl = slice(start, start + take)

            self._buf_z_l.append(z_l[sl].detach())
            self._buf_z_s.append(z_s[sl].detach())
            if f_l_full is not None:
                self._buf_f_l.append(f_l_full[sl])
                self._buf_f_s.append(f_s_full[sl])
            if labels_full is not None:
                self._buf_labels.append(labels_full[sl].detach())
            self._buf_n += take
            self._stream_samples_seen += take

            stream_exhausted = (
                self._stream_total_samples is not None
                and self._stream_samples_seen >= self._stream_total_samples
            )
            if self._buf_n >= self.proxy_batch_size or stream_exhausted:
                self._flush_bucket()

            out_chunks.append(self._combine(z_l[sl], z_s[sl], self._cached_w_l))
            start += take

        z_duo = torch.cat(out_chunks, dim=0)
        return (z_duo, self._cached_r_l, self._cached_r_s, self._cached_a_l,
                self._cached_a_s, self._cached_x_l, self._cached_x_s, self._cached_w_l)

    def report_and_reset_corruption_stats(self, label: str) -> dict:
        """R²/Pearson/Spearman between proxy and true per-PROXY-BATCH
        accuracy for this corruption, plus the mean gate weight. Each data
        point is one proxy-bucket flush (see _flush_bucket), not one
        adaptation batch. Prints and clears."""
        stats_l = _corr_stats(self._corr_r_l, self._corr_acc_l)
        stats_s = _corr_stats(self._corr_r_s, self._corr_acc_s)
        n = stats_l["n"]
        mean_w_l = sum(self._corr_w_l) / len(self._corr_w_l) if self._corr_w_l else float("nan")

        def _fmt(v: float) -> str:
            return f"{v:.3f}" if not (v != v) else "nan"

        if n > 0:
            print(
                f"[ProxyWeighted {self.proxy_kind} {label}] n={n} proxy batches with labels  "
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
        z_duo, *_ = self._forward(logits_l, logits_s)
        self._pending = z_duo.detach()
        return z_duo

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        if self._pending is not None:
            # duo-adapting modes: calibrate_with_grad already ran (and
            # possibly flushed the bucket) on these same logits this batch
            # — reuse it rather than running _forward() again, which would
            # double-assimilate the stateful temporal filters.
            z_duo, self._pending = self._pending, None
            return z_duo
        with torch.no_grad():
            z_duo, *_ = self._forward(logits_l, logits_s)
        return z_duo

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        pass  # gate/filter have no offline-tunable params here; see setup.fit_beta

    @property
    def last_w_l(self) -> float:
        """The gate weight actually used to combine the most recently
        processed slice — the cached value, whether freshly refreshed this
        call or reused from the last completed proxy batch (see
        _flush_bucket). Uniform introspection point with
        JointOptimalWOracle.last_w_l and JointFixedTS.last_w_l, e.g. for
        scripts/plot_optimal_w_sanity.py comparing arbitrary w_l-choosing
        strategies against the optimum."""
        return self._cached_w_l

    @property
    def model(self):
        return _NoOpModule()


