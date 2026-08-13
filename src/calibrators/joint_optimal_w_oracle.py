"""
joint_optimal_w_oracle.py
==========================
Cheating oracle baseline: directly solves for the scalar w_l in [0, 1] that
minimizes THIS batch's own NLL, instead of routing through
JointProxyWeighted's sigmoid(beta * (x_l - x_s)) gate. An even tighter
ceiling than sweeping beta under proxy_kind='oracle' (see
scripts/calibrate_gate_oracle.py): that sweep still constrains w_l to the
sigmoid FORM, asking "what's the best beta" rather than "what's the best
possible w_l". This module removes that constraint entirely.

Exact, not a heuristic: for fixed T_l/T_s,

    z_duo(w_l) = w_l*(z_l/T_l) + (1-w_l)*(z_s/T_s)

is AFFINE in w_l, and cross-entropy is convex in its logit argument, so
NLL(w_l) is provably convex on [0, 1]. A bounded scalar optimizer (SciPy's
Brent-based `minimize_scalar`) is therefore guaranteed to find the GLOBAL
optimum, no local-minima risk, no beta grid needed.

Cheats by construction (solves using the batch's own test labels, injected
via set_labels() the same way JointProxyWeighted's proxy_kind='oracle'
does) -- an upper-bound diagnostic only, never a deployable method. Keeps
T_l/T_s FIXED from a frozen base_ts (never re-optimized here, same
convention as JointProxyWeighted -- unidentifiable jointly with the gate),
so it isolates the gating ceiling specifically, not a re-calibrated one.

proxy_batch_size vs. adaptation_batch_size
-------------------------------------------
These are two ENTIRELY INDEPENDENT knobs. adaptation_batch_size is simply
however many samples arrive in one calibrate()/calibrate_with_grad() call
(whatever the caller's DataLoader batch size happens to be) -- this class
never needs to be told it explicitly. Samples accumulate in a proxy bucket
until proxy_batch_size is reached, at which point w_l is resolved fresh
from everything currently in the bucket. If adaptation_batch_size >
proxy_batch_size, that bucket can fill (and w_l can change) SEVERAL times
within a single call -- different rows of the same returned combination
then use different w_l, which is the correct behavior for a continuous,
call-boundary-agnostic stream of samples, not an approximation of it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from src.calibrators.base import BaseJointCalibrator, _NoOpModule
from src.calibrators.joint_fixed_TS import JointFixedTS


def combine(z_l: torch.Tensor, z_s: torch.Tensor, w_l: float, T_l: float, T_s: float) -> torch.Tensor:
    """Section 5 combine, generalized to any scalar w_l -- identical formula
    to JointProxyWeighted._combine. A free function (not a method) so
    optimal_w_nll can call it many times per batch, once per candidate
    w_l, without an instantiated calibrator."""
    w_s = 1.0 - w_l
    return w_l * (z_l / T_l) + w_s * (z_s / T_s)


def optimal_w_nll(
    z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor,
    T_l: float, T_s: float,
) -> tuple[float, float]:
    """Solve for the scalar w_l in [0, 1] minimizing THIS batch's own NLL.
    Returns (w_l*, nll* at w_l*). See module docstring for why this is
    exact rather than a heuristic. z_l/z_s/labels should be detached
    (no_grad) -- SciPy's optimizer works on plain floats, not autograd
    tensors.
    """
    def _nll(w_l: float) -> float:
        return float(F.cross_entropy(combine(z_l, z_s, w_l, T_l, T_s), labels))

    result = minimize_scalar(_nll, bounds=(0.0, 1.0), method="bounded")
    return float(result.x), float(result.fun)


class JointOptimalWOracle(BaseJointCalibrator):
    """Cheating oracle calibrator: solves for w_l by directly minimizing NLL
    (see module docstring), rather than JointProxyWeighted's proxy ->
    calibrate -> filter -> sigmoid-gate pipeline. Needs labels -- injected
    via set_labels(), the same convention JointProxyWeighted's
    proxy_kind='oracle' uses (see DynamicDuo.forward's calibration_mode
    check).

    proxy_batch_size and adaptation_batch_size are independent -- see
    module docstring. verbose (default True, matching JointProxyWeighted's
    always-prints-unless-bypassed convention) prints one line every time
    the proxy bucket actually fills and w_l is resolved -- NOT once per
    adaptation batch, since a single call can trigger several of these.

    Like JointProxyWeighted, DynamicDuo calls calibrate_with_grad (loss
    path) then calibrate (output path) on the SAME batch every adapting
    step; _pending caches the first result so the buffer/solve isn't
    double-assimilated for one batch.
    """

    def __init__(self, base_ts: JointFixedTS | None = None, proxy_batch_size: int = 1, verbose: bool = True):
        super().__init__()
        assert proxy_batch_size >= 1, f"proxy_batch_size must be >= 1, got {proxy_batch_size}"
        self.base_ts = base_ts
        if base_ts is not None:
            for p in base_ts.parameters():
                p.requires_grad_(False)
        self.proxy_batch_size = proxy_batch_size
        self.verbose = verbose
        self._labels: torch.Tensor | None = None
        self._pending: torch.Tensor | None = None
        self.last_w_l: float = 0.5
        self.last_nll: float = float("nan")
        self.n_refreshes: int = 0  # how many times the proxy bucket has been solved

        # Proxy-batch accumulation buffer (Section 1's b_t): filled sample by
        # sample as calls come in, flushed (solved) every time it reaches
        # proxy_batch_size -- possibly several times within one call.
        self._buf_z_l: list[torch.Tensor] = []
        self._buf_z_s: list[torch.Tensor] = []
        self._buf_labels: list[torch.Tensor] = []
        self._buf_n: int = 0

        # Stream-length bookkeeping (set by set_corruption's total_samples):
        # forces a flush of a trailing proxy-batch remainder instead of
        # silently leaving it stale/discarded -- see
        # JointProxyWeighted._flush_bucket for the identical fix.
        self._stream_total_samples: int | None = None
        self._stream_samples_seen: int = 0

        if verbose:
            T_l, T_s = self._temps()
            print(
                f"\n{'#' * 78}\n"
                f"# JointOptimalWOracle CONFIG\n"
                f"#   PROXY_BATCH_SIZE={proxy_batch_size}  verbose={verbose}\n"
                f"#   base_ts: T_l={T_l:.4f}  T_s={T_s:.4f}"
                f"{'  (base_ts=None, defaulted to 1.0/1.0)' if base_ts is None else ''}\n"
                f"{'#' * 78}\n"
            )

    def set_labels(self, labels: torch.Tensor) -> None:
        self._labels = labels

    def set_corruption(self, label: str, total_samples: int | None = None) -> None:
        """New stream: clear any partial buffer from the previous one and
        reset stream-length tracking (see class docstring)."""
        self._buf_z_l.clear(); self._buf_z_s.clear(); self._buf_labels.clear()
        self._buf_n = 0
        self._stream_total_samples = total_samples
        self._stream_samples_seen = 0

    def _temps(self) -> tuple[float, float]:
        T_l = float(self.base_ts.Tl.item()) if self.base_ts is not None else 1.0
        T_s = float(self.base_ts.Ts.item()) if self.base_ts is not None else 1.0
        return T_l, T_s

    def _flush_bucket(self, T_l: float, T_s: float) -> None:
        """Solve optimal_w_nll on everything currently buffered, cache the
        result, clear the buffer, and (if verbose) announce it -- called
        the moment the proxy bucket reaches proxy_batch_size."""
        agg_z_l = torch.cat(self._buf_z_l, dim=0)
        agg_z_s = torch.cat(self._buf_z_s, dim=0)
        agg_labels = torch.cat(self._buf_labels, dim=0)
        with torch.no_grad():
            w_l, nll = optimal_w_nll(agg_z_l, agg_z_s, agg_labels, T_l, T_s)
        self.last_w_l, self.last_nll = w_l, nll
        self.n_refreshes += 1
        n = agg_z_l.shape[0]
        self._buf_z_l.clear(); self._buf_z_s.clear(); self._buf_labels.clear()
        self._buf_n = 0
        if self.verbose:
            print(
                f"[OptimalWOracle PROXY BATCH #{self.n_refreshes} full, n={n}] "
                f"solved w_l={w_l:.4f} w_s={1.0 - w_l:.4f}  nll={nll:.4f}"
            )

    def _combine_batch(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        """Combine one adaptation batch (adaptation_batch_size = however
        many rows logits_l/logits_s have -- see module docstring), feeding
        the proxy bucket sample-range by sample-range and flushing it
        (see _flush_bucket) every time it fills, so w_l can change
        mid-batch. Each slice is combined with whatever w_l was current at
        that point in the stream, then all slices are concatenated back
        into one adaptation_batch_size-sized output, in order.
        """
        assert self._labels is not None, (
            "JointOptimalWOracle requires labels via set_labels() -- it cheats "
            "by construction and cannot be used where labels are unavailable."
        )
        labels = self._labels.to(logits_l.device)
        self._labels = None  # consume
        adaptation_batch_size = logits_l.shape[0]
        T_l, T_s = self._temps()

        out_chunks = []
        start = 0
        while start < adaptation_batch_size:
            take = min(self.proxy_batch_size - self._buf_n, adaptation_batch_size - start)
            sl = slice(start, start + take)

            self._buf_z_l.append(logits_l[sl].detach())
            self._buf_z_s.append(logits_s[sl].detach())
            self._buf_labels.append(labels[sl].detach())
            self._buf_n += take
            self._stream_samples_seen += take

            stream_exhausted = (
                self._stream_total_samples is not None
                and self._stream_samples_seen >= self._stream_total_samples
            )
            if self._buf_n >= self.proxy_batch_size or stream_exhausted:
                self._flush_bucket(T_l, T_s)

            out_chunks.append(combine(logits_l[sl], logits_s[sl], self.last_w_l, T_l, T_s))
            start += take

        return torch.cat(out_chunks, dim=0)

    def calibrate_with_grad(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        # Grad still flows through logits_l/logits_s for the TENT
        # adaptation loss -- only each slice's w_l is a fixed (detached)
        # scalar, same as before.
        z_duo = self._combine_batch(logits_l, logits_s)
        self._pending = z_duo.detach()
        return z_duo

    def calibrate(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        if self._pending is not None:
            z_duo, self._pending = self._pending, None
            return z_duo
        with torch.no_grad():
            return self._combine_batch(logits_l, logits_s)

    def forward(self, logits_l: torch.Tensor, logits_s: torch.Tensor) -> torch.Tensor:
        return self.calibrate_with_grad(logits_l, logits_s)

    def tune(self, *args, **kwargs):
        pass  # nothing to fit ahead of time; solves fresh per proxy batch

    @property
    def model(self):
        return _NoOpModule()


if __name__ == "__main__":
    torch.manual_seed(0)
    K, B = 10, 32
    z_l = torch.randn(B, K) * 3
    z_s = torch.randn(B, K) * 3
    labels = torch.randint(0, K, (B,))

    w_star, nll_star = optimal_w_nll(z_l, z_s, labels, 1.0, 1.0)
    assert 0.0 <= w_star <= 1.0

    # optimal_w_nll must match a fine brute-force grid search (proof the
    # bounded optimizer actually finds the convex minimum, not a
    # plausible-looking local point).
    grid = [i / 2000 for i in range(2001)]
    grid_nlls = [float(F.cross_entropy(combine(z_l, z_s, w, 1.0, 1.0), labels)) for w in grid]
    best_grid_nll = min(grid_nlls)
    assert abs(nll_star - best_grid_nll) < 1e-4, (nll_star, best_grid_nll)

    # The optimum must be at least as good as either single-model endpoint
    # (w_l=0 or w_l=1) -- it's a strict superset of "pick one".
    assert nll_star <= grid_nlls[0] + 1e-6
    assert nll_star <= grid_nlls[-1] + 1e-6

    # BaseJointCalibrator interface round-trip: calibrate() with set_labels()
    # matches the free function directly, at a proxy_batch_size >=
    # adaptation_batch_size so exactly one flush happens on the whole batch
    # (see the pbs=1 test below for the per-SAMPLE-granularity case).
    calib = JointOptimalWOracle(base_ts=None, proxy_batch_size=B, verbose=False)
    calib.set_labels(labels)
    z_duo = calib.calibrate(z_l, z_s)
    expected = combine(z_l, z_s, calib.last_w_l, 1.0, 1.0)
    assert torch.allclose(z_duo, expected)
    assert calib.n_refreshes == 1

    # proxy_batch_size=1 (the default) means the bucket is FULL after every
    # single sample -- true independence from adaptation_batch_size means
    # there is no implicit floor at "one adaptation batch"; a 32-sample
    # call resolves w_l 32 times, once per sample. This is a real behavior
    # change from treating "adaptation batch" as the minimum granularity;
    # anyone relying on the old assumption needs to pass an explicit
    # proxy_batch_size instead of the default.
    calib1s = JointOptimalWOracle(base_ts=None, proxy_batch_size=1, verbose=False)
    calib1s.set_corruption("synthetic")
    calib1s.set_labels(labels)
    calib1s.calibrate(z_l, z_s)
    assert calib1s.n_refreshes == B, f"pbs=1 should resolve once per sample ({B}), got {calib1s.n_refreshes}"
    assert calib1s._buf_n == 0

    # calibrate() without set_labels() must refuse rather than silently
    # defaulting to some weight -- it cheats by construction.
    calib2 = JointOptimalWOracle(verbose=False)
    try:
        calib2.calibrate(z_l, z_s)
        raise AssertionError("expected an AssertionError when labels were never set")
    except AssertionError as e:
        assert "set_labels" in str(e)

    # proxy_batch_size buffering across MULTIPLE calls: below the
    # threshold, last_w_l must NOT change (reused, not resolved); once the
    # buffer reaches proxy_batch_size, the solve must match optimal_w_nll
    # on the exact AGGREGATE of everything buffered so far.
    chunk1, chunk2, chunk3 = z_l[:10], z_l[10:20], z_l[20:28]
    schunk1, schunk2, schunk3 = z_s[:10], z_s[10:20], z_s[20:28]
    lchunk1, lchunk2, lchunk3 = labels[:10], labels[10:20], labels[20:28]

    calib3 = JointOptimalWOracle(base_ts=None, proxy_batch_size=24, verbose=False)
    calib3.set_corruption("synthetic")
    calib3.set_labels(lchunk1)
    calib3.calibrate(chunk1, schunk1)
    assert calib3._buf_n == 10 and calib3.last_w_l == 0.5, "should still be buffering, not solved yet"
    calib3.set_labels(lchunk2)
    calib3.calibrate(chunk2, schunk2)
    # threshold is 24 > 20 samples seen so far -- must still be buffering.
    assert calib3._buf_n == 20 and calib3.last_w_l == 0.5
    calib3.set_labels(lchunk3)
    calib3.calibrate(chunk3, schunk3)  # 20 + 8 = 28 samples seen, threshold 24
    # The bucket only takes EXACTLY enough of chunk3 to reach 24 (4 of its 8
    # samples); the remaining 4 (indices 24:28) start a fresh bucket rather
    # than all 8 being swept into one flush -- true per-sample granularity,
    # not "flush whatever chunk was in flight when the threshold was hit".
    assert calib3._buf_n == 4, "the 4 samples past the threshold should start a new bucket"
    expected_w_l, expected_nll = optimal_w_nll(z_l[:24], z_s[:24], labels[:24], 1.0, 1.0)
    assert abs(calib3.last_w_l - expected_w_l) < 1e-9
    assert abs(calib3.last_nll - expected_nll) < 1e-9
    assert calib3.n_refreshes == 1

    # proxy_batch_size < adaptation_batch_size, WITHIN A SINGLE CALL: the
    # bucket must fill and refresh multiple times in one calibrate() call,
    # and different ROWS of the same returned z_duo must use different w_l
    # -- not one uniform w_l for the whole call.
    calib5 = JointOptimalWOracle(base_ts=None, proxy_batch_size=10, verbose=False)
    calib5.set_corruption("synthetic")
    calib5.set_labels(labels)  # B=32 samples, threshold=10 -> 3 flushes (10,10,10) + buffers 2
    z_duo5 = calib5.calibrate(z_l, z_s)
    assert calib5.n_refreshes == 3, f"expected 3 flushes at pbs=10 over 32 samples, got {calib5.n_refreshes}"
    assert calib5._buf_n == 2, "the trailing 2 samples (30..32) should still be buffered, not flushed"
    w_l_first, _ = optimal_w_nll(z_l[:10], z_s[:10], labels[:10], 1.0, 1.0)
    w_l_second, _ = optimal_w_nll(z_l[10:20], z_s[10:20], labels[10:20], 1.0, 1.0)
    assert w_l_first != w_l_second, "test setup should produce distinguishable w_l between slices"
    expected_row0 = combine(z_l[0:1], z_s[0:1], w_l_first, 1.0, 1.0)
    expected_row10 = combine(z_l[10:11], z_s[10:11], w_l_second, 1.0, 1.0)
    assert torch.allclose(z_duo5[0:1], expected_row0), "rows 0-9 should use the FIRST flush's w_l"
    assert torch.allclose(z_duo5[10:11], expected_row10), "rows 10-19 should use the SECOND flush's w_l"

    # set_corruption(total_samples=...) must force-flush a trailing
    # remainder that would otherwise never reach proxy_batch_size, instead
    # of leaving it stale -- same fix as JointProxyWeighted.
    calib4 = JointOptimalWOracle(base_ts=None, proxy_batch_size=100, verbose=False)
    calib4.set_corruption("stream", total_samples=B)  # B=32 < 100
    calib4.set_labels(labels)
    calib4.calibrate(z_l, z_s)
    assert calib4._buf_n == 0, "trailing remainder should have been force-flushed"
    assert abs(calib4.last_w_l - w_star) < 1e-6, "should match the full-batch optimum"

    print("joint_optimal_w_oracle self-test passed")
