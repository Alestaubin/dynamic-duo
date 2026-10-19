"""
pure.py
=======
The Dynamic Duos proxy library (Phase 3, Stage C of docs/design_section45.md) as PURE FUNCTIONS:

    proxy(logits_batch, T=None, source_stats=None) -> float in [0, 1]       higher = more reliable

`logits_batch` is one member's (b, K) logits on a proxy batch (K = the dataset's class count, so 200 on ImageNet-A/R once the
logits are masked). p = softmax(z / T); the author decided that every proxy runs on the RAW softmax (T=None), like the
nuclear norm always did, so the evaluator never passes a temperature; the argument stays for a tempered variant. The
stateful proxies read `source_stats`, a dict made by `fit_source_stats` from the member's logits on the held-out clean slice
of the training images, never from validation images. Nothing here touches a model or the gate; `PROXIES` is the table the
evaluator iterates over, and the registry classes of src/reliability/proxies/ (used by the live gate) are untouched.

| name               | definition                                                                              | stateful |
| nnn                | ||P||_* / sqrt(b min(b, K)), the repo's nuclear_norm_score on the raw softmax            | no  |
| msp                | mean max p                                                                               | no  |
| neg_entropy        | 1 - mean H(p) / log K                                                                    | no  |
| energy             | sigmoid of the z-score of mean logsumexp(z) against the source's per-sample mean and sd  | yes |
| atc, atc_negent    | share of samples whose max p (or -H(p)) reaches the threshold t fitted on the source so  | yes |
|                    | that the share equals the source accuracy (Garg et al., ICLR 2022)                       |     |
| doc                | clip(acc_src - (msp_src - mean max p), 0, 1) (Guillory et al., ICCV 2021)                | yes |
| confidence_fro     | ||P||_F / sqrt(b) = sqrt(mean_i ||p_i||^2)                                               | no  |
| dispersity         | (||P||_* / ||P||_F - 1) / (sqrt(min(b, K)) - 1)                                          | no  |
| asr_concentration  | H(softmax(mean over the batch of z)) / log K                                             | no  |
| marginal_entropy   | H(mean over the batch of p) / log K                                                      | no  |
| cot                | 1 - the optimal-transport estimate of the error, see `cot`                               | no  |
| mutual_info        | (H(mean p) - mean H(p)) / log min(b, K)                                                  | no  |
| agree_duo          | share of the member's argmax equal to the fixed-combination duo's argmax (beta = 0)      | no  |

`agree_duo` needs more than one member's batch (the duo's logits, from both members and the temperatures), so it takes them
as `duo_logits`; the evaluator builds them from the held-out temperatures.

COT (Lu et al., NeurIPS 2023, github.com/luyuzhe111/COT, commit 392c0ff). The official code estimates a model's error on a
target batch as  emd2(w, w, M) / 2  with M = cdist(onehot(labels), softmax rows, p=1), both marginals uniform over n points,
and `labels` drawn by `sample_label_dist`: floor(prior_k * n) copies of every class k, the remainder sampled from the prior.
For a one-hot target e_k and a softmax row c the L1 distance is 2 (1 - c_k), so the estimate is the mean of (1 - c_{i, k_i})
over the optimal matching of rows to labels; with equal weights the optimum is an assignment, solved exactly by the Hungarian
algorithm (scipy's `linear_sum_assignment`), which is what POT's emd2 returns for these marginals. Here the prior is uniform
over the K classes, the remainder is drawn by a generator seeded with `seed` (so a bucket's score is reproducible; the same
seed for every bucket), and the score is 1 - estimate, the estimated ACCURACY, in [0, 1]. The repo's older cot.py scores the
negated cost with labels sampled from a fitted marginal; it is not used here. COT needs b of the order of K or more: with
b << K most random labels name classes no row is confident about, and the score collapses toward 0 whatever the model does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from scipy.optimize import linear_sum_assignment

from src.reliability.proxies.nuclear_norm import nuclear_norm_score

_TINY = 1e-12


def _unit(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _logp(z: torch.Tensor, T: float | None) -> torch.Tensor:
    z = z.detach().float()
    return torch.log_softmax(z / T if T else z, dim=1)


def _entropy(logp: torch.Tensor) -> torch.Tensor:
    """H(p_i) of every row, in nats."""
    return -(logp.exp() * logp).sum(1)


def _need(stats: dict | None, *keys: str, who: str) -> dict:
    if not stats or any(k not in stats for k in keys):
        raise ValueError(f"proxy {who!r} is stateful: it needs source_stats with {list(keys)} "
                         f"(fit_source_stats on the held-out clean slice)")
    return stats


# ------------------------------------------------------------------------------------------ stateless proxies

@torch.no_grad()
def nnn(z, T=None, source_stats=None) -> float:
    return _unit(nuclear_norm_score(z.detach().float() / T if T else z.detach().float()))


@torch.no_grad()
def msp(z, T=None, source_stats=None) -> float:
    return _unit(float(_logp(z, T).exp().max(1).values.mean()))


@torch.no_grad()
def neg_entropy(z, T=None, source_stats=None) -> float:
    return _unit(1.0 - float(_entropy(_logp(z, T)).mean()) / math.log(z.shape[1]))


@torch.no_grad()
def confidence_fro(z, T=None, source_stats=None) -> float:
    p = _logp(z, T).exp()
    return _unit(float(torch.linalg.matrix_norm(p, ord="fro")) / math.sqrt(p.shape[0]))


@torch.no_grad()
def dispersity(z, T=None, source_stats=None) -> float:
    p = _logp(z, T).exp()
    m = min(p.shape)
    if m <= 1:
        return 0.0
    s = torch.linalg.svdvals(p)
    return _unit((float(s.sum()) / max(float(s.pow(2).sum().sqrt()), _TINY) - 1.0) / (math.sqrt(m) - 1.0))


@torch.no_grad()
def asr_concentration(z, T=None, source_stats=None) -> float:
    zz = z.detach().float()
    zz = zz / T if T else zz
    logp = torch.log_softmax(zz.mean(0, keepdim=True), dim=1)
    return _unit(float(_entropy(logp)[0]) / math.log(z.shape[1]))


@torch.no_grad()
def marginal_entropy(z, T=None, source_stats=None) -> float:
    pbar = _logp(z, T).exp().mean(0, keepdim=True)
    return _unit(float(_entropy(pbar.clamp_min(_TINY).log())[0]) / math.log(z.shape[1]))


@torch.no_grad()
def mutual_info(z, T=None, source_stats=None) -> float:
    logp = _logp(z, T)
    pbar = logp.exp().mean(0, keepdim=True)
    h_marginal = float(_entropy(pbar.clamp_min(_TINY).log())[0])
    m = min(z.shape)
    if m <= 1:
        return 0.0
    return _unit((h_marginal - float(_entropy(logp).mean())) / math.log(m))


@torch.no_grad()
def cot(z, T=None, source_stats=None, seed: int = 0) -> float:
    """1 - the COT error estimate with a uniform label prior; see the module docstring."""
    p = _logp(z, T).exp()
    b, k = p.shape
    gen = torch.Generator().manual_seed(seed)
    labels = torch.cat([torch.arange(k).repeat_interleave(b // k), torch.randint(0, k, (b - (b // k) * k,), generator=gen)])
    cost = (1.0 - p[:, labels]).double().cpu().numpy()                 # = cdist(onehot, softmax, p=1) / 2
    rows, cols = linear_sum_assignment(cost)
    return _unit(1.0 - float(cost[rows, cols].mean()))


@torch.no_grad()
def agree_duo(z, T=None, source_stats=None, duo_logits: torch.Tensor | None = None) -> float:
    if duo_logits is None:
        raise ValueError("agree_duo needs duo_logits, the fixed-combination duo's logits on the same batch")
    return _unit(float((z.detach().argmax(1) == duo_logits.detach().argmax(1)).float().mean()))


# ------------------------------------------------------------------------------------------ stateful proxies

@torch.no_grad()
def fit_source_stats(z_src: torch.Tensor, labels: torch.Tensor, T: float | None = None) -> dict:
    """Everything the stateful proxies read, from one member's RAW logits on the held-out clean slice of the training
    images (z_src: (n, K), labels: (n,)). Raw logits, not centered: the energy depends on their absolute level."""
    z = z_src.detach().float()
    zt = z / T if T else z
    logp = torch.log_softmax(zt, dim=1)
    conf = logp.exp().max(1).values
    neg_h = -_entropy(logp)
    acc = float((z.argmax(1) == labels.to(z.device)).float().mean())
    err = 1.0 - acc
    lse = torch.logsumexp(zt, dim=1)
    return {"n": len(z), "acc": acc, "msp": float(conf.mean()),
            "energy_mu": float(lse.mean()), "energy_sd": max(float(lse.std()), 1e-6),
            "atc_thr_maxconf": float(torch.quantile(conf.float(), min(max(err, 0.0), 1.0))),
            "atc_thr_negent": float(torch.quantile(neg_h.float(), min(max(err, 0.0), 1.0)))}


@torch.no_grad()
def energy(z, T=None, source_stats=None) -> float:
    st = _need(source_stats, "energy_mu", "energy_sd", who="energy")
    zz = z.detach().float()
    e = float(torch.logsumexp(zz / T if T else zz, dim=1).mean())
    return _unit(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, (e - st["energy_mu"]) / st["energy_sd"])))))


@torch.no_grad()
def atc(z, T=None, source_stats=None) -> float:
    st = _need(source_stats, "atc_thr_maxconf", who="atc")
    return _unit(float((_logp(z, T).exp().max(1).values >= st["atc_thr_maxconf"]).float().mean()))


@torch.no_grad()
def atc_negent(z, T=None, source_stats=None) -> float:
    st = _need(source_stats, "atc_thr_negent", who="atc_negent")
    return _unit(float((-_entropy(_logp(z, T)) >= st["atc_thr_negent"]).float().mean()))


@torch.no_grad()
def doc(z, T=None, source_stats=None) -> float:
    st = _need(source_stats, "acc", "msp", who="doc")
    return _unit(st["acc"] - (st["msp"] - float(_logp(z, T).exp().max(1).values.mean())))


# ------------------------------------------------------------------------------------------ the table

@dataclass(frozen=True)
class ProxyDef:
    name: str
    fn: Callable
    stateful: bool
    family: str                 # the grouping of the paper's proxy table
    needs_duo: bool = False


PROXIES: dict[str, ProxyDef] = {d.name: d for d in (
    ProxyDef("nnn", nnn, False, "confidence + diversity"),
    ProxyDef("msp", msp, False, "confidence"),
    ProxyDef("neg_entropy", neg_entropy, False, "confidence"),
    ProxyDef("energy", energy, True, "source-calibrated"),
    ProxyDef("atc", atc, True, "source-calibrated"),
    ProxyDef("atc_negent", atc_negent, True, "source-calibrated"),
    ProxyDef("doc", doc, True, "source-calibrated"),
    ProxyDef("confidence_fro", confidence_fro, False, "confidence"),
    ProxyDef("dispersity", dispersity, False, "confidence + diversity"),
    ProxyDef("asr_concentration", asr_concentration, False, "diversity"),
    ProxyDef("marginal_entropy", marginal_entropy, False, "diversity"),
    ProxyDef("cot", cot, False, "optimal transport"),
    ProxyDef("mutual_info", mutual_info, False, "confidence + diversity"),
    ProxyDef("agree_duo", agree_duo, False, "agreement", needs_duo=True),
)}


def score(name: str, z: torch.Tensor, T: float | None = None, source_stats: dict | None = None, **kw) -> float:
    """One proxy on one batch; `kw` carries `duo_logits` (agree_duo) or `seed` (cot)."""
    if name not in PROXIES:
        raise ValueError(f"unknown proxy {name!r}: {sorted(PROXIES)}")
    return PROXIES[name].fn(z, T, source_stats, **kw)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, K = 128, 1000

    def onehot_logits(classes, k=K, sharp=20.0):
        z = torch.zeros(len(classes), k)
        z[torch.arange(len(classes)), torch.as_tensor(classes)] = sharp
        return z

    diverse = onehot_logits(torch.arange(B))                       # confident, every row another class
    collapsed = onehot_logits(torch.full((B,), 7))                 # identical one-hot predictions
    uniform = torch.zeros(B, K)
    moderate = torch.randn(B, K) * 2.0
    scenarios = {"diverse": diverse, "collapsed": collapsed, "uniform": uniform, "moderate": moderate}

    # source statistics from a synthetic clean slice: a member that is right 80% of the time, confident
    n_src = 2000
    y = torch.randint(0, K, (n_src,))
    pred = torch.where(torch.rand(n_src) < 0.8, y, (y + 1 + torch.randint(0, K - 1, (n_src,))) % K)
    z_src = torch.randn(n_src, K) * 0.5
    z_src[torch.arange(n_src), pred] += 10.0
    stats = fit_source_stats(z_src, y)
    assert abs(stats["acc"] - 0.8) < 0.04 and 0.5 < stats["msp"] <= 1.0
    # the ATC thresholds put the source's own share above them at the source accuracy (Garg et al.)
    for key, fn_ in (("atc_thr_maxconf", atc), ("atc_thr_negent", atc_negent)):
        assert abs(fn_(z_src, source_stats=stats) - stats["acc"]) < 0.02, key

    # 1. every proxy stays in [0, 1] on every scenario, and on a batch smaller than K and one larger than K
    extra = {"duo_logits": moderate.flip(0)}
    for name, d in PROXIES.items():
        for sname, z in scenarios.items():
            v = score(name, z, source_stats=stats, **(extra if d.needs_duo else {}))
            assert 0.0 <= v <= 1.0 and math.isfinite(v), (name, sname, v)
        assert 0.0 <= score(name, torch.randn(20, 5), source_stats=fit_source_stats(torch.randn(50, 5), torch.randint(0, 5, (50,))),
                            **({"duo_logits": torch.randn(20, 5)} if d.needs_duo else {})) <= 1.0, name
        assert 0.0 <= score(name, torch.randn(1500, 12), source_stats=fit_source_stats(torch.randn(50, 12), torch.randint(0, 12, (50,))),
                            **({"duo_logits": torch.randn(1500, 12)} if d.needs_duo else {})) <= 1.0, name

    # 2. a collapsed batch (identical one-hot predictions) scores low on the dispersity proxies
    assert dispersity(collapsed) < 1e-3, dispersity(collapsed)
    assert nnn(collapsed) < 0.15 and mutual_info(collapsed) < 0.02 and marginal_entropy(collapsed) < 0.02
    assert asr_concentration(collapsed) < 0.02
    #    ... and a diverse one high where confidence and diversity are combined
    assert nnn(diverse) > 0.95 and dispersity(diverse) > 0.95 and mutual_info(diverse) > 0.95
    assert nnn(diverse) > 10 * nnn(collapsed) and dispersity(diverse) > dispersity(moderate)

    # 3. uniform predictions score low on the confidence proxies
    assert abs(msp(uniform) - 1 / K) < 1e-6 and neg_entropy(uniform) < 1e-6
    assert abs(confidence_fro(uniform) - 1 / math.sqrt(K)) < 1e-4
    assert atc(uniform, source_stats=stats) == 0.0 and atc_negent(uniform, source_stats=stats) == 0.0
    assert doc(uniform, source_stats=stats) < 0.05 and energy(uniform, source_stats=stats) < 0.01
    assert nnn(uniform) < 0.05
    #    confident rows score high on them (whatever their diversity)
    for c in (collapsed, diverse):
        assert msp(c) > 0.99 and neg_entropy(c) > 0.95 and confidence_fro(c) > 0.99

    # 4. the source-calibrated proxies order a confident batch above an unsure one, and need their statistics
    assert energy(diverse * 0.5 + 5.0, source_stats=stats) > energy(uniform, source_stats=stats)
    assert doc(z_src, source_stats=stats) > doc(moderate, source_stats=stats)
    assert abs(doc(z_src, source_stats=stats) - stats["acc"]) < 0.02, "on the source batch itself DoC returns the source accuracy"
    for name in ("energy", "atc", "atc_negent", "doc"):
        try:
            score(name, diverse)
            raise AssertionError(f"{name} without statistics must raise")
        except ValueError as e:
            assert "stateful" in str(e)
    try:
        agree_duo(diverse)
        raise AssertionError("agree_duo without duo logits must raise")
    except ValueError:
        pass

    # 5. agreement with the duo
    assert agree_duo(diverse, duo_logits=diverse) == 1.0 and agree_duo(diverse, duo_logits=diverse.roll(1, 1)) == 0.0

    # 6. COT: with K <= b the label prior is matched exactly (floor(b/K) copies of each class), so balanced confident
    #    predictions score ~1, collapsed ones ~1/K, uniform ones 1/K
    kk, bb = 10, 100
    balanced = onehot_logits(torch.arange(bb) % kk, kk)
    assert cot(balanced) > 0.999
    assert abs(cot(onehot_logits(torch.full((bb,), 3), kk)) - 1 / kk) < 1e-3
    assert abs(cot(torch.zeros(bb, kk)) - 1 / kk) < 1e-3
    assert cot(balanced) == cot(balanced), "seeded: the same batch gives the same score"
    # b < K: the labels are random classes, so even a perfect diverse batch scores low -- the documented limit
    assert cot(diverse) < 0.5
    # it is the assignment optimum: equal to a brute-force search over permutations on a tiny case
    import itertools
    tiny = torch.randn(5, 5) * 2
    p5, lab = tiny.softmax(1), torch.cat([torch.arange(5).repeat_interleave(1)])
    best = min(sum(1.0 - float(p5[i, lab[j]]) for i, j in enumerate(perm)) for perm in itertools.permutations(range(5))) / 5
    assert abs(cot(tiny) - (1.0 - best)) < 1e-6

    # 7. raw is the default; a temperature changes the softmax proxies and 1 is the identity
    z = torch.randn(64, 50)
    assert msp(z) == msp(z, T=1.0) and nnn(z) == nnn(z, T=1.0)
    assert msp(z, T=3.0) < msp(z) < msp(z, T=0.5)
    assert score("nnn", z) == nnn(z)
    try:
        score("nope", z)
        raise AssertionError("an unknown proxy must raise")
    except ValueError:
        pass
    assert len(PROXIES) == 14 and [n for n, d in PROXIES.items() if d.stateful] == ["energy", "atc", "atc_negent", "doc"]
    print("proxies (pure) self-test passed")
