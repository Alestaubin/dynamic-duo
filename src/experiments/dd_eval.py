"""
dd_eval.py
==========
The offline evaluator of Vanilla Dynamic Duos (Phase 3, Stage C of docs/design_section45.md): everything that can be computed
from the two members' cached logits, without a model, a dataset or the live gate.

Per stream (and averaged over streams, `average_over_streams`), from the members' RAW logits:

  members alone            each adapted member, raw softmax
  fixed combinations       0.5 (z_L / T_L + z_S / T_S) with the clean TS and with the held-out TS
  Vanilla DD               every proxy x beta x b_t:  w = sigmoid(beta (s_L - s_S)),  z = w z_L / T_L + (1 - w) z_S / T_S
  oracle selection         per proxy batch, the member with the higher batch accuracy, on its temperature-scaled logits
  batch-wise oracle weight per proxy batch, the w on a grid of 101 values that minimizes the batch's NLL (in-sample)
  proxy diagnostics        per (proxy, b_t): ordering accuracy and Spearman correlation of s_L - s_S with the true log-odds gap

Metrics: top-1 accuracy, ECE (15 equal-width bins, top label) and NLL, from per-sample confidence, correctness and log-loss.

**The live gate's semantics, reproduced exactly** (docs/design_section45.md, section 1; the author kept them as the definition
of Vanilla DD). The stream is cut into proxy batches ("buckets") of b_t consecutive samples (the last one is smaller). A
bucket's weight w_j is computed from the bucket's own logits, but it is applied only to the adaptation-batch slice that
completes the bucket: the bucket's earlier slices keep the previous bucket's weight (0.5 before the first bucket). `plan_buckets`
turns that rule into one weight index per sample; `JointProxyWeighted` fed the same batches gives the same predictions (tested
below for every proxy, against `PureProxyGate`, the live gate with the library's proxies). With the filter `none` and the
`identity` map a bucket's weight is a pure function of its logits, so proxy scores are computed once per (proxy, b_t) and every
beta is one line of arithmetic. The clip of the predicted accuracy before the logit is `eps` = 1e-3, the live gate's.

Temperature sets are fitted per base algorithm and pair (decision 5): the held-out TS on the held-out streams pooled, the clean
TS and the source statistics of the stateful proxies on the clean held-out slice of the training images.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.experiments import oracles as O
from src.experiments.evaluate import StreamMetrics
from src.reliability.proxies import pure
from src.utils.metrics import top_label_ece

GRID_BETAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
GRID_BT = (64, 128, 256, 512, 1024, 2048)


@dataclass(frozen=True)
class GridConfig:
    proxies: tuple = tuple(pure.PROXIES)
    betas: tuple = GRID_BETAS
    b_ts: tuple = GRID_BT                 # add 16 and 32: the live gate supports buckets smaller than an adaptation batch
    eps: float = 1e-3                     # the live gate's clip of the predicted accuracy (the author's decision 2)
    bins: int = 15
    oracle_points: int = 101
    ts: str = "heldout"                   # the temperature set that drives Vanilla DD and the oracles
    chunk_elems: int = 64_000_000         # cap on (weights x samples x classes) elements per temporary


# ------------------------------------------------------------------------------------------------ the bucket rule

@dataclass
class BucketPlan:
    b_t: int
    n: int
    n_buckets: int
    bucket: np.ndarray        # (n,) the bucket each sample belongs to
    widx: np.ndarray          # (n,) the bucket whose weight scores the sample (-1: the prior weight, 0.5)


def plan_buckets(batch_sizes: Sequence[int], n: int, b_t: int) -> BucketPlan:
    """Which bucket's weight scores each sample under JointProxyWeighted._forward (see the module docstring).

    Bucket j is samples [j b_t, (j + 1) b_t). It is flushed when its last sample is appended, which happens inside the
    adaptation batch containing that sample, so its weight scores the samples of the bucket from the start of THAT batch
    (or of the bucket, if later); the rest of the bucket was scored with the weight before."""
    if sum(batch_sizes) != n:
        raise ValueError(f"batch sizes sum to {sum(batch_sizes)}, not {n}")
    idx = np.arange(n)
    nb = -(-n // b_t)
    bucket = idx // b_t
    last = np.minimum((np.arange(nb) + 1) * b_t, n) - 1                       # the sample that fills each bucket
    batch_starts = np.concatenate([[0], np.cumsum(batch_sizes)[:-1]])
    batch_start_of_last = batch_starts[np.searchsorted(batch_starts, last, side="right") - 1]
    own_from = np.maximum(np.arange(nb) * b_t, batch_start_of_last)           # first sample scored with the bucket's own weight
    widx = np.where(idx >= own_from[bucket], bucket, bucket - 1)
    return BucketPlan(b_t, n, nb, bucket.astype(np.int64), widx.astype(np.int64))


def gate_weights(r_l: np.ndarray, r_s: np.ndarray, beta: float, eps: float = 1e-3) -> np.ndarray:
    """w_L of every bucket: sigmoid(beta (x_L - x_S)), x = logit(clip(r, eps, 1 - eps))."""
    a_l, a_s = np.clip(r_l, eps, 1 - eps), np.clip(r_s, eps, 1 - eps)
    x = np.log(a_l / (1 - a_l)) - np.log(a_s / (1 - a_s))
    return 1.0 / (1.0 + np.exp(-beta * x))


def sample_weights(w_bucket: np.ndarray, plan: BucketPlan) -> np.ndarray:
    """(n,) float32 weight of each sample: its widx-th bucket weight, 0.5 where the prior still holds."""
    return np.concatenate([[0.5], w_bucket]).astype(np.float32)[plan.widx + 1]


@torch.no_grad()
def bucket_scores(proxy: str, z_l: torch.Tensor, z_s: torch.Tensor, plan: BucketPlan, stats: tuple | None,
                  duo: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(r_L, r_S): the proxy of each member on each bucket's logits. `stats` = (stats_l, stats_s), `duo` the fixed
    combination's logits (agree_duo)."""
    d = pure.PROXIES[proxy]
    st_l, st_s = stats if stats is not None else (None, None)
    r_l, r_s = np.empty(plan.n_buckets), np.empty(plan.n_buckets)
    for j in range(plan.n_buckets):
        lo, hi = j * plan.b_t, min((j + 1) * plan.b_t, plan.n)
        kw = {"duo_logits": duo[lo:hi]} if d.needs_duo else {}
        r_l[j] = d.fn(z_l[lo:hi], None, st_l, **kw)
        r_s[j] = d.fn(z_s[lo:hi], None, st_s, **kw)
    return r_l, r_s


def bucket_diagnostics(r_l: np.ndarray, r_s: np.ndarray, acc_l: np.ndarray, acc_s: np.ndarray, eps: float = 1e-3) -> dict:
    """Independent of beta. Ordering accuracy: over the buckets where the members' accuracies differ, the share where
    sign(s_L - s_S) is the sign of the true accuracy gap (a zero score gap is a miss). Spearman: s_L - s_S against the true
    log-odds gap logit(acc_L) - logit(acc_S), accuracies clipped to [eps, 1 - eps]."""
    lg = lambda a: np.log(np.clip(a, eps, 1 - eps) / (1 - np.clip(a, eps, 1 - eps)))    # noqa: E731
    gap = lg(r_l) - lg(r_s)
    true = acc_l - acc_s
    nz = true != 0
    order = float(np.mean(np.sign(gap[nz]) == np.sign(true[nz]))) if nz.any() else float("nan")
    tg = lg(acc_l) - lg(acc_s)
    rho = float("nan")
    if len(gap) > 2 and gap.std() > 0 and tg.std() > 0:
        rho = float(spearmanr(gap, tg).statistic)
    return {"ordering_acc": order, "spearman": rho, "n_buckets": len(gap), "n_ties": int((~nz).sum())}


# ------------------------------------------------------------------------------------------------ scoring combinations

class Scorer:
    """Metrics of w z_L / T_L + (1 - w) z_S / T_S for many weight vectors w (one per row of W, one weight per sample)."""

    def __init__(self, z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor, T_l: float, T_s: float,
                 bins: int = 15, chunk_elems: int = 64_000_000):
        self.A, self.B = z_l / T_l, z_s / T_s
        self.y, self.bins, self.chunk_elems = labels, bins, chunk_elems

    def _chunks(self, m: int):
        step = max(1, self.chunk_elems // (m * self.A.shape[1]))
        for i in range(0, len(self.y), step):
            yield slice(i, i + step)

    def _lp(self, W: torch.Tensor, sl: slice) -> torch.Tensor:
        w = W[:, sl].unsqueeze(-1)
        return torch.log_softmax(w * self.A[sl] + (1 - w) * self.B[sl], dim=2)          # (m, chunk, K)

    def _weights(self, W) -> torch.Tensor:
        W = torch.as_tensor(W, dtype=torch.float32, device=self.A.device)
        return W[None] if W.dim() == 1 else W

    @torch.no_grad()
    def metrics(self, W) -> list[dict]:
        W = self._weights(W)
        m, n = W.shape
        conf = torch.empty(m, n, device=W.device)
        correct = torch.empty(m, n, dtype=torch.bool, device=W.device)
        nll = torch.zeros(m, dtype=torch.float64, device=W.device)
        for sl in self._chunks(m):
            lp = self._lp(W, sl)
            c, pred = lp.max(dim=2)
            y = self.y[sl].expand(m, -1)
            conf[:, sl], correct[:, sl] = c.exp(), pred == y
            nll += -lp.gather(2, y.unsqueeze(-1)).squeeze(-1).double().sum(1)
        conf, correct = conf.cpu(), correct.cpu()
        return [{"accuracy": float(correct[i].double().mean()), "ece": float(top_label_ece(conf[i], correct[i], self.bins)),
                 "nll": float(nll[i]) / n, "n": n} for i in range(m)]

    @torch.no_grad()
    def nll_per_sample(self, W) -> torch.Tensor:
        """(m, n) log-loss of every sample under every weight vector."""
        W = self._weights(W)
        out = torch.empty(W.shape, dtype=torch.float32, device=W.device)
        for sl in self._chunks(W.shape[0]):
            out[:, sl] = -self._lp(W, sl).gather(2, self.y[sl].expand(W.shape[0], -1).unsqueeze(-1)).squeeze(-1)
        return out


@torch.no_grad()
def member_metrics(z: torch.Tensor, labels: torch.Tensor, bins: int = 15, chunk: int = 8192) -> dict:
    sm = StreamMetrics(bins)
    for i in range(0, len(labels), chunk):
        sm.update(z[i:i + chunk], labels[i:i + chunk])
    r = sm.result()
    return {"accuracy": r["accuracy"], "ece": r["ece"], "nll": r["nll"], "n": r["n"]}


# ------------------------------------------------------------------------------------------------ one stream

@torch.no_grad()
def evaluate_stream(sd: dict, cfg: GridConfig, temps: dict, stats: tuple | None, device: torch.device | str = "cpu"
                    ) -> dict:
    """sd: {"stem", "z_l", "z_s" (RAW logits), "labels", "batch_sizes"}; temps: {"heldout": (T_L, T_S)[, "clean": ...]};
    stats: (stats_l, stats_s) of the stateful proxies, or None (they are skipped). Returns
    {"rows": [...], "diag": [...], "records": {(proxy, b_t): {...}}}."""
    device = torch.device(device)
    z_l, z_s, y = sd["z_l"].to(device).float(), sd["z_s"].to(device).float(), sd["labels"].to(device).long()
    n, stem = len(y), sd["stem"]
    rows, diag, records = [], [], {}

    def add(series, res, **kw):
        rows.append({"stream": stem, "series": series, "proxy": None, "beta": None, "b_t": None, "ts": None, **kw, **res})

    add("large", member_metrics(z_l, y, cfg.bins))
    add("small", member_metrics(z_s, y, cfg.bins))
    half = torch.full((1, n), 0.5, device=device)
    fixed = {}
    for name, (T_l, T_s) in temps.items():
        fixed[name] = Scorer(z_l, z_s, y, T_l, T_s, cfg.bins, cfg.chunk_elems).metrics(half)[0]
        add("fixed", fixed[name], ts=name)

    sc = Scorer(z_l, z_s, y, *temps[cfg.ts], cfg.bins, cfg.chunk_elems)
    duo = 0.5 * sc.A + 0.5 * sc.B
    w_grid = torch.linspace(0.0, 1.0, cfg.oracle_points, device=device)
    nll_grid = sc.nll_per_sample(w_grid[:, None].expand(-1, n))                 # (points, n), independent of b_t
    ok_l, ok_s = (z_l.argmax(1) == y).float(), (z_s.argmax(1) == y).float()

    for b_t in cfg.b_ts:
        plan = plan_buckets(sd["batch_sizes"], n, b_t)
        bucket = torch.from_numpy(plan.bucket).to(device)
        size = torch.bincount(bucket, minlength=plan.n_buckets).float()
        acc_l = torch.bincount(bucket, weights=ok_l, minlength=plan.n_buckets) / size
        acc_s = torch.bincount(bucket, weights=ok_s, minlength=plan.n_buckets) / size
        # oracles, in-sample on the bucket: the better member (ties: the large one), the best of 101 weights
        w_select = (acc_l >= acc_s).float()[bucket]
        nll_bucket = torch.zeros(cfg.oracle_points, plan.n_buckets, device=device).index_add_(1, bucket, nll_grid)
        w_best = w_grid[nll_bucket.argmin(0)][bucket]
        res = sc.metrics(torch.stack([w_select, w_best]))
        add("oracle_select", res[0], b_t=b_t, ts=cfg.ts)
        add("oracle_w", res[1], b_t=b_t, ts=cfg.ts)

        for proxy in cfg.proxies:
            if pure.PROXIES[proxy].stateful and stats is None:
                continue
            r_l, r_s = bucket_scores(proxy, z_l, z_s, plan, stats, duo)
            diag.append({"stream": stem, "proxy": proxy, "b_t": b_t,
                         **bucket_diagnostics(r_l, r_s, acc_l.cpu().numpy(), acc_s.cpu().numpy(), cfg.eps)})
            w = np.stack([gate_weights(r_l, r_s, beta, cfg.eps) for beta in cfg.betas])           # (betas, buckets)
            records[(proxy, b_t)] = {"r_l": r_l, "r_s": r_s, "acc_l": acc_l.cpu().numpy(), "acc_s": acc_s.cpu().numpy(),
                                     "w_l": w, "betas": np.asarray(cfg.betas)}
            moving = [i for i, b in enumerate(cfg.betas) if b != 0]
            res = iter(sc.metrics(np.stack([sample_weights(w[i], plan) for i in moving])) if moving else [])
            for beta in cfg.betas:
                # beta = 0 is w = 0.5 everywhere: the fixed combination, whatever the proxy
                add("dd", fixed[cfg.ts] if beta == 0 else next(res), proxy=proxy, beta=beta, b_t=b_t, ts=cfg.ts)
    return {"rows": rows, "diag": diag, "records": records}


# ------------------------------------------------------------------------------------------------ across streams

def average_over_streams(rows: list[dict], streams: Sequence[str] | None = None) -> pd.DataFrame:
    """Macro average over `streams` (default all) of accuracy, ECE and NLL, per (series, proxy, beta, b_t, ts)."""
    df = pd.DataFrame(rows)
    if streams is not None:
        df = df[df["stream"].isin(streams)]
    keys = ["series", "proxy", "beta", "b_t", "ts"]
    out = df.groupby(keys, dropna=False)[["accuracy", "ece", "nll"]].mean().reset_index()
    out["n_streams"] = df.groupby(keys, dropna=False)["stream"].nunique().values
    return out


def pooled_diagnostics(records: dict, streams: Sequence[str], eps: float = 1e-3) -> pd.DataFrame:
    """Ordering accuracy and Spearman over ALL the buckets of `streams` pooled, per (proxy, b_t). `records`: {(stem, proxy,
    b_t): {...}} as `run_pair` returns it."""
    out = []
    for (proxy, b_t) in sorted({(p, b) for (_, p, b) in records}):
        cat = {k: np.concatenate([records[(s, proxy, b_t)][k] for s in streams if (s, proxy, b_t) in records])
               for k in ("r_l", "r_s", "acc_l", "acc_s")}
        out.append({"proxy": proxy, "b_t": b_t, **bucket_diagnostics(cat["r_l"], cat["r_s"], cat["acc_l"], cat["acc_s"], eps)})
    return pd.DataFrame(out)


def load_stream(pair, stem: str) -> dict:
    d = pair.read(stem, raw=True)
    return {"stem": stem, "z_l": d["z_l"], "z_s": d["z_s"], "labels": d["labels"], "batch_sizes": d["batch_sizes"]}


@torch.no_grad()
def run_pair(pair, heldout: Sequence[str], test: Sequence[str], cfg: GridConfig = GridConfig(), *,
             clean_stem: str | None = None, source_stem: str | None = None, extra: Sequence[str] = (),
             device: torch.device | str | None = None) -> dict:
    """Evaluate a stored pair (logit_store.PairView) on the held-out streams, the test streams and `extra` ones (e.g. the clean
    validation stream, reported separately).

    heldout        stems whose pooled logits fit the held-out TS (and are evaluated too)
    clean_stem     the clean held-out slice of the training images: fits the clean TS
    source_stem    the same slice: fits the source statistics of the stateful proxies (default: clean_stem). Without one,
                   the stateful proxies are skipped
    Returns {"rows", "diag", "records": {(stem, proxy, b_t): ...}, "temps", "stats"}."""
    from src.utils.device import get_device
    device = get_device(device)
    held = [load_stream(pair, s) for s in heldout]
    pooled = [torch.cat([h[k] for h in held]).to(device) for k in ("z_l", "z_s", "labels")]
    tl, ts, _ = O.fit_temperatures(pooled[0], pooled[1], pooled[2].long())
    temps = {"heldout": (float(tl), float(ts))}
    del pooled
    source_stem = source_stem or clean_stem
    stats = None
    if clean_stem or source_stem:
        src = load_stream(pair, source_stem or clean_stem)
        if clean_stem:
            c = src if clean_stem == (source_stem or clean_stem) else load_stream(pair, clean_stem)
            tl, ts, _ = O.fit_temperatures(c["z_l"].to(device), c["z_s"].to(device), c["labels"].to(device).long())
            temps["clean"] = (float(tl), float(ts))
        stats = (pure.fit_source_stats(src["z_l"], src["labels"]), pure.fit_source_stats(src["z_s"], src["labels"]))
    if cfg.ts not in temps:
        raise ValueError(f"cfg.ts={cfg.ts!r} but only {sorted(temps)} were fitted")

    out = {"rows": [], "diag": [], "records": {}, "temps": temps, "stats": stats}
    for stem in [*heldout, *test, *extra]:
        sd = next((h for h in held if h["stem"] == stem), None) or load_stream(pair, stem)
        r = evaluate_stream(sd, cfg, temps, stats, device)
        out["rows"] += r["rows"]
        out["diag"] += r["diag"]
        out["records"].update({(stem, p, b): v for (p, b), v in r["records"].items()})
    return out


def save_results(out_dir: str | Path, name: str, res: dict) -> dict:
    """<name>.rows.csv, <name>.diag.csv, <name>.records.npz (the per-bucket proxy values, accuracies and gate weights of
    every stream, for figures), <name>.temps.json. Four files per evaluation."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["rows"]).to_csv(d / f"{name}.rows.csv", index=False)
    pd.DataFrame(res["diag"]).to_csv(d / f"{name}.diag.csv", index=False)
    np.savez_compressed(d / f"{name}.records.npz", **{f"{s}|{p}|{b}|{k}": v for (s, p, b), rec in res["records"].items()
                                                      for k, v in rec.items()})
    (d / f"{name}.temps.json").write_text(json.dumps({"temps": res["temps"], "stats": res["stats"]}, indent=2))
    return {k: str(d / f"{name}.{k}") for k in ("rows.csv", "diag.csv", "records.npz", "temps.json")}


if __name__ == "__main__":
    import tempfile

    from src.calibrators.joint_fixed_TS import JointFixedTS
    from src.experiments.joint_dd import PureProxyGate
    from src.experiments.logit_store import LogitStore, MemberSpec, PairView
    from src.experiments.specs import RunSpec

    torch.manual_seed(0)
    K, N = 50, 300
    T_L, T_S = 1.3, 0.8
    BS = [32] * 9 + [12]                                                   # a stream of 300 in batches of 32, the last short

    def stream(n=N, seed=0, acc_l=0.6, acc_s=0.4):
        g = torch.Generator().manual_seed(seed)
        y = torch.randint(0, K, (n,), generator=g)

        def member(acc, scale):
            z = torch.randn(n, K, generator=g) * 1.5
            right = torch.rand(n, generator=g) < acc
            z[torch.arange(n), torch.where(right, y, torch.randint(0, K, (n,), generator=g))] += scale
            return z
        return member(acc_l, 5.0), member(acc_s, 3.0), y

    z_l, z_s, y = stream()
    src_l, src_s = stream(seed=11)[:2]
    ysrc = stream(seed=11)[2]
    stats = (pure.fit_source_stats(src_l, ysrc), pure.fit_source_stats(src_s, ysrc))

    # ---- 1. the bucket rule: every proxy, offline == the live gate, for several (batch, bucket) cuts -------------------------
    def live(proxy, beta, b_t, batches):
        g = PureProxyGate(proxy, beta=beta, b_t=b_t, T_l=T_L, T_s=T_S, stats_l=stats[0], stats_s=stats[1], num_classes=K)
        g.set_corruption("s", total_samples=N)
        edges = np.concatenate([[0], np.cumsum(batches)])
        return torch.cat([g.calibrate(z_l[a:b], z_s[a:b]) for a, b in zip(edges[:-1], edges[1:])])

    A, B = z_l / T_L, z_s / T_S
    duo = 0.5 * A + 0.5 * B
    for batches, b_t in ((BS, 64), (BS, 128), (BS, 16), (BS, 96), (BS, 32), ([64] * 4 + [44], 128), ([300], 64), (BS, 512)):
        plan = plan_buckets(batches, N, b_t)
        for proxy in pure.PROXIES:
            r_l, r_s = bucket_scores(proxy, z_l, z_s, plan, stats, duo)
            for beta in (0.0, 2.0):
                w = sample_weights(gate_weights(r_l, r_s, beta), plan)
                off = torch.from_numpy(w)[:, None] * A + (1 - torch.from_numpy(w)[:, None]) * B
                ref = live(proxy, beta, b_t, batches)
                assert torch.allclose(off, ref, atol=1e-5), (proxy, beta, b_t, len(batches), float((off - ref).abs().max()))
    # the rule itself, spelled out for the brief's case: b_t = 128 over batches of 64 scores the second half with its own weight
    p = plan_buckets([64] * 5 + [16], 336, 128)
    assert p.widx[:64].tolist() == [-1] * 64 and p.widx[64:128].tolist() == [0] * 64
    assert p.widx[128:192].tolist() == [0] * 64 and p.widx[192:256].tolist() == [1] * 64
    assert p.widx[256:320].tolist() == [1] * 64 and p.widx[320:].tolist() == [2] * 16, "the last (partial) bucket is its own"
    assert (plan_buckets([64] * 5, 320, 64).widx == np.arange(320) // 64).all(), "b_t = batch: every batch its own weight"
    share = lambda b_t: float((plan_buckets([64] * 781 + [16], 50000, b_t).widx == plan_buckets([64] * 781 + [16], 50000, b_t).bucket).mean())   # noqa: E731
    assert [round(share(b), 3) for b in (16, 64, 128, 256)] == [1.0, 1.0, 0.5, 0.25] and 0.03 < share(2048) < 0.04

    # ---- 2. beta = 0 is the fixed combination: exactly, and against the live JointFixedTS -----------------------------------
    cfg = GridConfig(proxies=("nnn", "msp", "energy", "agree_duo"), b_ts=(64, 128), betas=(0.0, 0.5, 2.0))
    temps = {"heldout": (T_L, T_S), "clean": (1.1, 1.0)}
    res = evaluate_stream({"stem": "s", "z_l": z_l, "z_s": z_s, "labels": y, "batch_sizes": BS}, cfg, temps, stats)
    df = pd.DataFrame(res["rows"])
    fx = df[(df.series == "fixed") & (df.ts == "heldout")].iloc[0]
    for _, r in df[(df.series == "dd") & (df.beta == 0.0)].iterrows():
        assert (r.accuracy, r.ece, r.nll) == (fx.accuracy, fx.ece, fx.nll), "beta = 0 must be the fixed combination exactly"
    ts_live = JointFixedTS(Tl=T_L, Ts=T_S, verbose=False)
    z_fixed = ts_live.calibrate(z_l, z_s).cpu()
    assert torch.allclose(z_fixed, duo, atol=1e-6) and (z_fixed.argmax(1) == duo.argmax(1)).all()
    assert fx.accuracy == float((z_fixed.argmax(1) == y).double().mean())
    assert abs(fx.nll - float(torch.nn.functional.cross_entropy(z_fixed, y))) < 1e-5
    assert fx.n == N

    # ---- 3. a DD cell equals the live gate's metrics (accuracy, ECE, NLL) ----------------------------------------------------
    for proxy, beta, b_t in (("nnn", 2.0, 128), ("msp", 0.5, 64), ("energy", 2.0, 128), ("agree_duo", 2.0, 64)):
        cell = df[(df.series == "dd") & (df.proxy == proxy) & (df.beta == beta) & (df.b_t == b_t)].iloc[0]
        sm = StreamMetrics(15)
        sm.update(live(proxy, beta, b_t, BS), y)
        want = sm.result()
        assert abs(cell.accuracy - want["accuracy"]) < 1e-12 and abs(cell.ece - want["ece"]) < 1e-6 and abs(cell.nll - want["nll"]) < 1e-4, \
            (proxy, beta, b_t, cell.to_dict(), want)
    assert {"large", "small", "fixed", "dd", "oracle_select", "oracle_w"} == set(df.series)
    assert len(df[df.series == "dd"]) == 4 * 3 * 2 and len(df[df.series == "fixed"]) == 2
    assert df[df.series == "large"].iloc[0].accuracy == float((z_l.argmax(1) == y).double().mean())

    # ---- 4. oracles: never worse than what they choose among ---------------------------------------------------------------
    sel = df[df.series == "oracle_select"].set_index("b_t")
    orw = df[df.series == "oracle_w"].set_index("b_t")
    acc_l, acc_s = float((z_l.argmax(1) == y).double().mean()), float((z_s.argmax(1) == y).double().mean())
    for b_t in (64, 128):
        assert sel.loc[b_t].accuracy >= max(acc_l, acc_s) - 1e-9, "per-bucket best member beats either member overall"
        assert orw.loc[b_t].nll <= fx.nll + 1e-6, "w = 0.5 is on the grid, so the oracle weight cannot lose to it"
    assert orw.loc[64].nll <= orw.loc[128].nll + 1e-6, "smaller buckets nest in larger ones: finer oracle, lower NLL"

    # ---- 5. diagnostics: a proxy that IS the accuracy orders perfectly; its mirror image orders never ----------------------
    rng = np.random.RandomState(0)
    a_l, a_s = rng.uniform(0.05, 0.95, 40), rng.uniform(0.05, 0.95, 40)
    d = bucket_diagnostics(a_l, a_s, a_l, a_s)
    assert d["ordering_acc"] == 1.0 and abs(d["spearman"] - 1.0) < 1e-9
    d = bucket_diagnostics(a_s, a_l, a_l, a_s)
    assert d["ordering_acc"] == 0.0 and abs(d["spearman"] + 1.0) < 1e-9
    d = bucket_diagnostics(np.full(10, 0.5), np.full(10, 0.5), a_l[:10], a_s[:10])
    assert d["ordering_acc"] == 0.0 and math.isnan(d["spearman"]), "a constant score never orders and has no rank correlation"
    t = bucket_diagnostics(a_l, a_s, np.round(a_l, 1), np.round(a_l, 1))
    assert t["n_ties"] == 40 and math.isnan(t["ordering_acc"]), "ties in the true gap are not counted"
    assert all({"ordering_acc", "spearman", "n_buckets"} <= set(x) for x in res["diag"]) and len(res["diag"]) == 4 * 2

    # ---- 6. records and Scorer -----------------------------------------------------------------------------------------------
    rec = res["records"][("nnn", 128)]
    assert rec["w_l"].shape == (3, 3) and (rec["w_l"][0] == 0.5).all() and rec["r_l"].shape == (3,)   # 300 / 128 -> 3 buckets
    sc = Scorer(z_l, z_s, y, T_L, T_S)
    W = torch.rand(3, N)
    for i, r in enumerate(sc.metrics(W)):
        logits = W[i, :, None] * A + (1 - W[i, :, None]) * B
        assert r["accuracy"] == float((logits.argmax(1) == y).double().mean())
        assert abs(r["nll"] - float(torch.nn.functional.cross_entropy(logits, y))) < 1e-5
    assert torch.allclose(sc.nll_per_sample(W)[1], torch.nn.functional.cross_entropy(W[1, :, None] * A + (1 - W[1, :, None]) * B, y, reduction="none"), atol=1e-5)
    small = Scorer(z_l, z_s, y, T_L, T_S, chunk_elems=3 * K * 7)                    # tiny chunks give the same answer
    for a, b in zip(small.metrics(W), sc.metrics(W)):
        assert a["accuracy"] == b["accuracy"] and a["ece"] == b["ece"] and abs(a["nll"] - b["nll"]) < 1e-9

    # ---- 7. averaging, and a stored pair end to end -------------------------------------------------------------------
    other = [{**r, "stream": "t", "accuracy": r["accuracy"] + 0.1} for r in res["rows"]]
    avg = average_over_streams(res["rows"] + other)
    big = avg[avg.series == "large"].iloc[0]
    assert big.n_streams == 2 and abs(big.accuracy - (df[df.series == "large"].iloc[0].accuracy + 0.05)) < 1e-12
    assert len(avg) == len(res["rows"]) and set(average_over_streams(res["rows"] + other, ["t"]).n_streams) == {1}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        store = LogitStore(tmp / "logits", staging=None)
        opt = {"METHOD": "SGD", "STEPS": 1, "LR": 1e-3, "MOMENTUM": 0.9, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}
        duo_cfg = {"LARGE": {"NAME": "vit", "NORM": "LN", "OPTIM": opt}, "SMALL": {"NAME": "rn", "NORM": "BN", "OPTIM": opt}}
        ml, ms = (MemberSpec.from_run(RunSpec("d", seed=0), duo_cfg, s, batch_size=32) for s in ("large", "small"))
        names = ["a__s5", "b__s5", "c__s5", "d__s5", "t1__s5", "t2__s5", "clean_train_slice"]
        for i, nm in enumerate(names):
            a, b, yy = stream(seed=20 + i) if nm != "clean_train_slice" else (src_l, src_s, ysrc)
            for m, z in ((ml, a), (ms, b)):
                store.write(m, nm, z, yy, np.arange(len(yy)), [32] * (len(yy) // 32) + ([len(yy) % 32] if len(yy) % 32 else []))
        pair = PairView(store, ml, ms)
        small_cfg = GridConfig(proxies=("nnn", "msp", "doc", "cot"), b_ts=(64,), betas=(0.0, 1.0))
        out = run_pair(pair, ["a__s5", "b__s5", "c__s5", "d__s5"], ["t1__s5", "t2__s5"], small_cfg, clean_stem="clean_train_slice",
                       device="cpu")
        assert set(out["temps"]) == {"heldout", "clean"} and out["stats"] is not None
        streams = {r["stream"] for r in out["rows"]}
        assert streams == {"a__s5", "b__s5", "c__s5", "d__s5", "t1__s5", "t2__s5"}
        t_l, t_s, _ = O.fit_temperatures(torch.cat([stream(seed=20 + i)[0] for i in range(4)]),
                                         torch.cat([stream(seed=20 + i)[1] for i in range(4)]),
                                         torch.cat([stream(seed=20 + i)[2] for i in range(4)]))
        assert abs(out["temps"]["heldout"][0] - float(t_l)) < 1e-3 and abs(out["temps"]["heldout"][1] - float(t_s)) < 1e-3
        no_src = run_pair(pair, ["a__s5", "b__s5"], ["t1__s5"], small_cfg, device="cpu")
        assert set(no_src["temps"]) == {"heldout"} and no_src["stats"] is None
        assert {r["proxy"] for r in no_src["rows"] if r["series"] == "dd"} == {"nnn", "msp", "cot"}, "stateful proxies skipped"
        pooled = pooled_diagnostics(out["records"], ["t1__s5", "t2__s5"])
        assert set(pooled.proxy) == {"nnn", "msp", "doc", "cot"} and (pooled.n_buckets == 2 * 5).all()
        files = save_results(tmp / "res", "run", out)
        back = pd.read_csv(files["rows.csv"])
        assert len(back) == len(out["rows"]) and set(np.load(files["records.npz"]).files)
        try:
            run_pair(pair, ["a__s5"], [], GridConfig(ts="clean"), device="cpu")
            raise AssertionError("cfg.ts = clean needs a clean stream")
        except ValueError:
            pass
    print("dd_eval self-test passed")
