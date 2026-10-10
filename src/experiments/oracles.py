"""
oracles.py
==========
Fitting the duo's temperatures WITH LABELS, at three granularities (per stream, per batch, per sample).
Everything here cheats on purpose: it is the upper-bound side of the temperature-scaling analysis
(scripts/01_tab_temperatures.py), never a deployable method.

The duo's combined logits are  (z_l / T_l + z_s / T_s) / 2  (JointFixedTS). Cross-entropy is convex in the
logits and the logits are linear in (a, b) = (1/(2 T_l), 1/(2 T_s)), so

  * the NLL is CONVEX in (a, b): `fit_temperatures` finds the global optimum with a damped, box-constrained
    Newton method (exact 2x2 Hessian, line search), vectorised over a leading batch dimension so thousands of
    per-batch fits run in one pass. Temperatures are boxed to [0.05, 50], the box JointFixedTS.tune uses: when
    a model is unhelpful the NLL keeps improving as its temperature grows without bound, and the box turns
    "infinity" into a reproducible boundary value.
  * per sample there is a single true label: `per_sample_temperatures` runs gradient descent (Adam in
    log-temperature space, same box) independently for every sample. A sample the duo can get right has no
    finite optimum (more confidence always lowers its NLL), so those sit on the lower boundary; this is the
    "absolute ceiling" of the definition, bounded by the box.

All functions take float32 logits (centered fp16 from the cache is fine: every quantity here is invariant to a
per-row constant) shaped (N, K) or, for the batched fits, (B, n, K).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

T_MIN, T_MAX = 0.05, 50.0
A_MIN, A_MAX = 0.5 / T_MAX, 0.5 / T_MIN          # a = 1 / (2 T)
_ELEMS = 48_000_000                              # cap on B*n*K elements per chunk (a few hundred MB per temporary)


def combine(z_l: torch.Tensor, z_s: torch.Tensor, T_l, T_s) -> torch.Tensor:
    """(z_l / T_l + z_s / T_s) / 2; T_l, T_s broadcast against the (..., K) logits."""
    return 0.5 * (z_l / T_l + z_s / T_s)


# --------------------------------------------------------------------------------------------------------- metrics

@torch.no_grad()
def acc_nll(z_l: torch.Tensor, z_s: torch.Tensor, y: torch.Tensor, T_l, T_s, chunk: int = 16384
            ) -> tuple[float, float]:
    """(accuracy, mean NLL) of the combination over all N samples. T_l, T_s: floats or per-sample (N,) tensors."""
    n_correct, nll_sum = 0, 0.0
    for i in range(0, len(y), chunk):
        sl = slice(i, i + chunk)
        args = [(v[sl].unsqueeze(1) if torch.is_tensor(v) and v.dim() == 1 else v) for v in (T_l, T_s)]
        z = combine(z_l[sl], z_s[sl], *args)
        n_correct += int((z.argmax(1) == y[sl]).sum())
        nll_sum += float(F.cross_entropy(z, y[sl], reduction="sum"))
    return n_correct / len(y), nll_sum / len(y)


# ------------------------------------------------------------------------------- global / per-batch temperatures

def _stats(zl: torch.Tensor, zs: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, second: bool):
    """Mean NLL f (B,), and if `second` its gradient (B,2) and Hessian (B,2,2) in (a, b), in float64."""
    yi = y.unsqueeze(2)
    zly, zsy = zl.gather(2, yi).squeeze(2), zs.gather(2, yi).squeeze(2)
    s = a.view(-1, 1, 1) * zl + b.view(-1, 1, 1) * zs
    f = (torch.logsumexp(s, 2) - a.view(-1, 1) * zly - b.view(-1, 1) * zsy).double().mean(1)
    if not second:
        return f
    p = torch.softmax(s, 2)
    El, Es = (p * zl).sum(2), (p * zs).sum(2)
    Hll = ((p * zl * zl).sum(2) - El * El).double().mean(1)
    Hss = ((p * zs * zs).sum(2) - Es * Es).double().mean(1)
    Hls = ((p * zl * zs).sum(2) - El * Es).double().mean(1)
    g = torch.stack([(El - zly).double().mean(1), (Es - zsy).double().mean(1)], 1)
    H = torch.stack([torch.stack([Hll, Hls], 1), torch.stack([Hls, Hss], 1)], 1)
    return f, g, H


def _fit_chunk(zl, zs, y, init, iters: int, tol: float):
    B, dev = zl.shape[0], zl.device
    x = torch.tensor([0.5 / init[0], 0.5 / init[1]], dtype=torch.float64, device=dev).expand(B, 2).clone()
    x = x.clamp(A_MIN, A_MAX)
    eye = torch.eye(2, dtype=torch.float64, device=dev)
    for _ in range(iters):
        f, g, H = _stats(zl, zs, y, x[:, 0].float(), x[:, 1].float(), True)
        # projected Newton: a variable sitting on a bound with the gradient pushing outward stays put
        fixed = ((x <= A_MIN * (1 + 1e-9)) & (g > 0)) | ((x >= A_MAX * (1 - 1e-9)) & (g < 0))
        g = torch.where(fixed, torch.zeros_like(g), g)
        if g.abs().max() < tol:
            break
        free = (~fixed).double()
        H = H * free.unsqueeze(2) * free.unsqueeze(1) + torch.diag_embed(fixed.double())
        lam = 1e-9 * (H.diagonal(dim1=1, dim2=2).sum(1) + 1e-12)
        d = -torch.linalg.solve(H + lam.view(-1, 1, 1) * eye, g.unsqueeze(2)).squeeze(2) * free
        t, accepted, x_new = torch.ones(B, dtype=torch.float64, device=dev), torch.zeros(B, dtype=torch.bool, device=dev), x.clone()
        for _ in range(30):                                       # backtracking: accept any non-increase
            cand = (x + t.unsqueeze(1) * d).clamp(A_MIN, A_MAX)
            ok = (_stats(zl, zs, y, cand[:, 0].float(), cand[:, 1].float(), False) <= f) & ~accepted
            x_new[ok], accepted = cand[ok], accepted | ok
            if accepted.all():
                break
            t = torch.where(accepted, t, t * 0.5)
        if (x_new - x).abs().max() < 1e-12:
            break
        x = x_new
    f = _stats(zl, zs, y, x[:, 0].float(), x[:, 1].float(), False)
    return 0.5 / x[:, 0], 0.5 / x[:, 1], f


@torch.no_grad()
def fit_temperatures(z_l: torch.Tensor, z_s: torch.Tensor, y: torch.Tensor, *, init=(1.0, 1.0),
                     iters: int = 60, tol: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global-optimum (T_l, T_s) minimizing the NLL of (z_l/T_l + z_s/T_s)/2, within [0.05, 50].

    z_l, z_s: (n, K) or (B, n, K); y: (n,) or (B, n). Returns (T_l, T_s, nll), each a scalar tensor for 2-D input
    and a (B,) tensor for 3-D input (one independent fit per leading index)."""
    squeeze = z_l.dim() == 2
    if squeeze:
        z_l, z_s, y = z_l[None], z_s[None], y[None]
    B, n, K = z_l.shape
    step = max(1, _ELEMS // (n * K))
    out = [_fit_chunk(z_l[i:i + step], z_s[i:i + step], y[i:i + step], init, iters, tol) for i in range(0, B, step)]
    T_l, T_s, f = (torch.cat([o[j] for o in out]).float() for j in range(3))
    return (T_l[0], T_s[0], f[0]) if squeeze else (T_l, T_s, f)


# --------------------------------------------------------------------------------------------- per-sample

@torch.enable_grad()      # callers may be under no_grad; this one optimizes by gradient descent
def per_sample_temperatures(z_l: torch.Tensor, z_s: torch.Tensor, y: torch.Tensor, *, init=(1.0, 1.0),
                            steps: int = 200, lr: float = 0.1, chunk: int = 8192
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """(T_l(i), T_s(i)) per sample by gradient descent on that sample's own NLL, within [0.05, 50].

    Adam in log-temperature space, started at `init` (the Fixed-TS temperatures), clamped to the box after every
    step. The loss is a SUM over samples, and Adam normalizes per element, so each sample's pair evolves exactly
    as if it were optimized alone. The better of the final and the initial pair is kept (never worse than init)."""
    lo, hi = torch.log(torch.tensor(T_MIN)), torch.log(torch.tensor(T_MAX))
    Ts_l, Ts_s = [], []
    for i in range(0, len(y), chunk):
        zl, zs, yy = z_l[i:i + chunk], z_s[i:i + chunk], y[i:i + chunk]
        u = torch.full((len(yy),), float(torch.log(torch.tensor(float(init[0])))), device=zl.device, requires_grad=True)
        v = torch.full((len(yy),), float(torch.log(torch.tensor(float(init[1])))), device=zl.device, requires_grad=True)
        u0, v0 = u.detach().clone(), v.detach().clone()
        opt = torch.optim.Adam([u, v], lr=lr)

        def nll_each(uu, vv):
            return F.cross_entropy(combine(zl, zs, uu.exp().unsqueeze(1), vv.exp().unsqueeze(1)), yy, reduction="none")

        for _ in range(steps):
            opt.zero_grad()
            nll_each(u, v).sum().backward()
            opt.step()
            with torch.no_grad():
                u.clamp_(float(lo), float(hi)); v.clamp_(float(lo), float(hi))
        with torch.no_grad():
            better = nll_each(u, v) <= nll_each(u0, v0)
            Ts_l.append(torch.where(better, u.detach(), u0).exp())
            Ts_s.append(torch.where(better, v.detach(), v0).exp())
    return torch.cat(Ts_l), torch.cat(Ts_s)


# ----------------------------------------------------------------------------------- consecutive-batch helpers

def _batched(fn, tensors: list[torch.Tensor], batch_size: int):
    """Apply `fn(*batched_tensors)` (leading dim = batch index) to the full batches of consecutive samples, and
    separately to the shorter last batch. Returns the list of (start, end, result) pieces."""
    n = len(tensors[0])
    n_full = n // batch_size
    pieces = []
    if n_full:
        m = n_full * batch_size
        res = fn(*[t[:m].reshape(n_full, batch_size, *t.shape[1:]) for t in tensors])
        pieces.append((0, m, res, batch_size))
    if n % batch_size:
        res = fn(*[t[n_full * batch_size:][None] for t in tensors])
        pieces.append((n_full * batch_size, n, res, n - n_full * batch_size))
    return pieces


@torch.no_grad()
def batchwise_temperatures(z_l, z_s, y, batch_size: int, init=(1.0, 1.0)) -> tuple[torch.Tensor, torch.Tensor]:
    """Refit (T_l, T_s) on every consecutive batch of `batch_size` samples; per-sample vectors (N,) holding each
    sample's batch's temperatures."""
    Tl, Ts = [], []
    for _, _, (tl, ts, _), size in _batched(lambda a, b, c: fit_temperatures(a, b, c, init=init), [z_l, z_s, y], batch_size):
        Tl.append(tl.repeat_interleave(size)); Ts.append(ts.repeat_interleave(size))
    return torch.cat(Tl), torch.cat(Ts)


if __name__ == "__main__":
    import math

    from src.calibrators.joint_fixed_TS import JointFixedTS

    torch.manual_seed(0)

    def make(n, K, sharp_l, sharp_s, acc_l=0.7, acc_s=0.6):
        """Two overconfident-ish models: each puts `sharp` on its predicted class, right with prob acc."""
        y = torch.randint(0, K, (n,))
        out = []
        for sharp, acc in ((sharp_l, acc_l), (sharp_s, acc_s)):
            z = torch.randn(n, K) * 1.2
            pred = torch.where(torch.rand(n) < acc, y, torch.randint(0, K, (n,)))
            z[torch.arange(n), pred] += sharp
            out.append(z)
        return out[0], out[1], y

    z_l, z_s, y = make(3000, 40, 6.0, 3.5)

    def grid_best(zl, zs, yy):
        best = (1e9, None)
        for tl in torch.logspace(math.log10(T_MIN), math.log10(T_MAX), 70):
            for ts in torch.logspace(math.log10(T_MIN), math.log10(T_MAX), 70):
                v = float(F.cross_entropy(combine(zl, zs, tl, ts), yy))
                if v < best[0]:
                    best = (v, (float(tl), float(ts)))
        return best

    # 1. the Newton fit reaches the global optimum: never worse than a fine grid, matches JointFixedTS.tune
    Tl, Ts, nll = fit_temperatures(z_l, z_s, y)
    gb, _ = grid_best(z_l, z_s, y)
    assert float(nll) <= gb + 1e-6, (float(nll), gb)
    ref = JointFixedTS(verbose=False)
    ref.tune(z_l, z_s, y)
    ref_nll = float(F.cross_entropy(combine(z_l, z_s, ref.Tl.item(), ref.Ts.item()), y))
    assert float(nll) <= ref_nll + 1e-5, (float(nll), ref_nll)
    assert abs(float(F.cross_entropy(combine(z_l, z_s, float(Tl), float(Ts)), y)) - float(nll)) < 1e-5

    # 2. an unhelpful second model: its temperature runs to the box boundary instead of "infinity"
    junk = torch.randn(3000, 40) * 0.01
    Tl2, Ts2, _ = fit_temperatures(z_l, junk, y)
    assert float(Ts2) > 0.99 * T_MAX or float(Tl2) < 0.99 * float(Ts2), (float(Tl2), float(Ts2))
    Tl3, Ts3, _ = fit_temperatures(z_l * 0.001, z_s, y)       # large model near-uniform
    assert 0.05 <= float(Tl3) <= 50 and 0.05 <= float(Ts3) <= 50

    # 3. per-batch fits == the same fits done one by one; a short last batch is handled
    B, n = 6, 128
    zb_l, zb_s, yb = z_l[:B * n].reshape(B, n, -1), z_s[:B * n].reshape(B, n, -1), y[:B * n].reshape(B, n)
    tl, ts, f = fit_temperatures(zb_l, zb_s, yb)
    for i in range(B):
        a, b, fi = fit_temperatures(zb_l[i], zb_s[i], yb[i])
        assert abs(float(a) - float(tl[i])) < 1e-3 * float(a) and abs(float(fi) - float(f[i])) < 1e-5
    ptl, pts = batchwise_temperatures(z_l[:300], z_s[:300], y[:300], 128)
    assert ptl.shape == (300,) and ptl[0] == ptl[127] != ptl[128] == ptl[255] != ptl[256] == ptl[299]

    # 4. per-sample GD: never worse than its start, and as good as a per-sample grid search
    zl_s, zs_s, ys = z_l[:24], z_s[:24], y[:24]
    Tl_i, Ts_i = per_sample_temperatures(zl_s, zs_s, ys, init=(float(Tl), float(Ts)), steps=300)
    assert Tl_i.shape == (24,) and (Tl_i >= T_MIN * 0.999).all() and (Tl_i <= T_MAX * 1.001).all()
    for i in range(24):
        start = float(F.cross_entropy(combine(zl_s[i:i + 1], zs_s[i:i + 1], float(Tl), float(Ts)), ys[i:i + 1]))
        now = float(F.cross_entropy(combine(zl_s[i:i + 1], zs_s[i:i + 1], float(Tl_i[i]), float(Ts_i[i])), ys[i:i + 1]))
        best, _ = grid_best(zl_s[i:i + 1], zs_s[i:i + 1], ys[i:i + 1])
        assert now <= start + 1e-6 and now <= best + 5e-3, (i, start, now, best)

    # 5. increasing granularity can only lower the NLL (each level contains the one before it)
    batch = 128
    a_fixed = acc_nll(z_l, z_s, y, float(Tl) * 1.4, float(Ts) * 0.7)                    # some fixed (off-optimal) temps
    a_corr = acc_nll(z_l, z_s, y, float(Tl), float(Ts))                                  # corruption-wise optimum
    btl, bts = batchwise_temperatures(z_l, z_s, y, batch)
    a_batch = acc_nll(z_l, z_s, y, btl, bts)
    ptl_, pts_ = per_sample_temperatures(z_l[:1024], z_s[:1024], y[:1024], init=(float(Tl), float(Ts)), steps=150)
    a_samp = acc_nll(z_l[:1024], z_s[:1024], y[:1024], ptl_, pts_)
    a_corr_1k = acc_nll(z_l[:1024], z_s[:1024], y[:1024], float(Tl), float(Ts))
    assert a_corr[1] <= a_fixed[1] + 1e-9 and a_batch[1] <= a_corr[1] + 1e-9
    btl16, bts16 = batchwise_temperatures(z_l, z_s, y, 16)           # 16 divides 128: its batches nest inside the 128s
    assert acc_nll(z_l, z_s, y, btl16, bts16)[1] <= a_batch[1] + 1e-6
    assert a_samp[1] <= a_corr_1k[1] + 1e-9
    # chunked metrics == direct computation
    assert abs(acc_nll(z_l, z_s, y, 1.0, 1.0, chunk=700)[1] - float(F.cross_entropy(combine(z_l, z_s, 1.0, 1.0), y))) < 1e-5
    print("oracles self-test passed")
