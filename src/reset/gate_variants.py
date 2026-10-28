"""
gate_variants.py
================
EXTRA GATES for a live reset run (`extra_gates` of src/reset/live_runs.run_live): they score the same members' logits as the run's main gate, outside the loop (the
members never see them), so one run yields the correctness of many gates on one trajectory and one set of resets. The members' logits are not stored, which is why
this has to happen in the run (the final test stored only the gate at beta 0.5).

A gate is named by a string (`parse`), on the run's proxy and its main gate (the "leader"):

    beta<b>            the leader's gate at another beta:             w_l = sigmoid(b * (x_l - x_s));   beta0 is the fixed-TS duo exactly
    route<rho>w<W>     reset-aware routing on top of the leader's own beta: for `W` samples after a member's reset that member gets the fixed weight `rho` (rho = 0
                       routes everything to the other member, 0.5 is the fixed-TS duo), then the gate takes over again. If both members are reset together there is
                       no "other" member, and the gate stays as it is
    <proxy>@<b>        another label-free proxy of src/reliability/proxies/pure.py (not the stateful ones) at beta b

    gates = {n: make_gate(n, leader, T_l=T_l, T_s=T_s, b_t=b_t) for n in ("beta0", "beta2", "route0.2w5120", "agree_duo@1")}

The beta variants and the routed gate FOLLOW the leader: the proxy values of a bucket (the expensive part, an SVD per member) are the leader's, read from a log of its
flushes (the k-th flush of a follower takes the k-th of the leader), so they cost nothing and are the leader's to the bit; only the weight differs. The leader must be
called first on every batch, and followers made before its first flush.
"""

from __future__ import annotations

import re
from typing import Callable

import torch

from src.experiments.joint_dd import PureProxyGate
from src.reliability.proxies import pure

_BETA = re.compile(r"beta(?P<beta>[0-9]+(?:\.[0-9]+)?)")
_ROUTE = re.compile(r"route(?P<rho>[0-9]+(?:\.[0-9]+)?)w(?P<window>[0-9]+)")
_PROXY = re.compile(r"(?P<proxy>[a-z_]+)@(?P<beta>[0-9]+(?:\.[0-9]+)?)")


# the beta ladder around the final test's 0.5 (beta0 = the fixed-TS duo, a check; the main gate of the run is beta 0.5); scripts/46_gate_routing_runs.py's default
LADDER = ("beta0", "beta0.25", "beta1", "beta2", "beta4", "beta8")

# the default set of scripts/38_run_reset_live.py --extra_gates default: the beta ladder around the final test's 0.5 (beta0 = the fixed-TS duo, a check) and routing
# after a reset: the reset member's weight 0.35 / 0.2 / 0 for 5,120 samples (20 proxy batches, the window of scripts/45_gate_ablation.py) and 0.2 for 1,280, plus one
# proxy that does not measure confidence
DEFAULT_SET = ("beta0", "beta0.25", "beta1", "beta2", "beta4", "beta8", "route0.35w5120", "route0.2w5120", "route0w5120", "route0.2w1280", "agree_duo@1")


def _tap(leader: PureProxyGate) -> list:
    """The leader's log of proxy values, one (r_l, r_s) per flush (installed on first use)."""
    if getattr(leader, "_flush_log", None) is None:
        leader._flush_log = []
        scores = leader._proxy_scores

        def logged(*args, **kwargs):
            out = scores(*args, **kwargs)
            leader._flush_log.append(out)
            return out
        leader._proxy_scores = logged
    return leader._flush_log


class FollowerGate(PureProxyGate):
    """A PureProxyGate that does not score the proxy itself: its k-th flush takes the k-th proxy values of the `leader` (the same bucket: both see the same samples, the
    leader called first on every batch). Everything else, the bucket logic included, is the gate's own, so a follower is exactly an independent gate of the same proxy
    and its own beta."""

    def __init__(self, leader: PureProxyGate, *, beta: float, T_l: float, T_s: float, b_t: int, num_classes: int = 1000):
        if leader.n_refreshes:
            raise ValueError(f"a follower is made before its leader's first flush (the leader has made {leader.n_refreshes})")
        super().__init__(leader.proxy_kind, beta=beta, b_t=b_t, T_l=T_l, T_s=T_s, num_classes=num_classes)
        self.leader, self._log = leader, _tap(leader)

    def _proxy_scores(self, z_l, z_s, f_l, f_s, labels):
        if self.n_refreshes >= len(self._log):
            raise RuntimeError(f"a follower gate flushes after its leader (the leader has made {len(self._log)} flushes, this is the follower's "
                               f"{self.n_refreshes + 1}th): call the leader first on every batch")
        return self._log[self.n_refreshes]


class RoutedGate:
    """Reset-aware routing around a gate `inner` (see the module text). `notify_reset(member, position)` is called by the run loop when a member goes back to source,
    with the number of samples seen so far (= the first sample that member's reset weights act on); `calibrate` counts the samples it is given, so the two agree."""

    def __init__(self, inner: PureProxyGate, rho: float, window: int):
        if not 0.0 <= rho <= 1.0 or window < 1:
            raise ValueError(f"route needs a weight in [0, 1] and a window of at least 1 sample, got rho={rho}, window={window}")
        self.inner, self.rho, self.window = inner, float(rho), int(window)
        self.seen = 0
        self._override: tuple[str, int, int] | None = None            # (member, first sample, one past the last sample)

    def notify_reset(self, member: str, position: int) -> None:
        if self._override is not None and self._override[1] == position and self._override[0] != member:
            self._override = None                                      # both members at once: nobody to route to
        else:
            self._override = (member, int(position), int(position) + self.window)

    def calibrate(self, z_l: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        out = self.inner.calibrate(z_l, z_s)
        first, self.seen = self.seen, self.seen + z_l.shape[0]
        if self._override is None:
            return out
        member, start, end = self._override
        lo, hi = max(start, first) - first, min(end, self.seen) - first
        if first >= end:
            self._override = None
        if hi <= lo:
            return out
        out = out.clone()
        out[lo:hi] = self.inner._combine(z_l[lo:hi], z_s[lo:hi], self.rho if member == "large" else 1.0 - self.rho)
        return out

    def __getattr__(self, name):                                       # set_corruption, mark_segment, records, ...: the inner gate's
        return getattr(self.inner, name)


def parse(name: str) -> Callable[[PureProxyGate, float, float, int, int], object]:
    """The constructor `(leader, T_l, T_s, b_t, num_classes) -> gate` of a gate name; ValueError for a name that is none (checked before a run starts)."""
    if (m := _BETA.fullmatch(name)):
        beta = float(m["beta"])
        return lambda leader, T_l, T_s, b_t, K: FollowerGate(leader, beta=beta, T_l=T_l, T_s=T_s, b_t=b_t, num_classes=K)
    if (m := _ROUTE.fullmatch(name)):
        rho, window = float(m["rho"]), int(m["window"])
        if not 0.0 <= rho <= 1.0 or window < 1:
            raise ValueError(f"{name!r}: the weight of a reset member is in [0, 1] and the window at least 1 sample")
        return lambda leader, T_l, T_s, b_t, K: RoutedGate(FollowerGate(leader, beta=leader.beta, T_l=T_l, T_s=T_s, b_t=b_t, num_classes=K), rho, window)
    if (m := _PROXY.fullmatch(name)):
        proxy, beta = m["proxy"], float(m["beta"])
        if proxy not in pure.PROXIES or pure.PROXIES[proxy].stateful:
            raise ValueError(f"{name!r}: {proxy!r} is not a stateless proxy of pure.PROXIES ({sorted(k for k, d in pure.PROXIES.items() if not d.stateful)})")
        return lambda leader, T_l, T_s, b_t, K: PureProxyGate(proxy, beta=beta, b_t=b_t, T_l=T_l, T_s=T_s, num_classes=K)
    raise ValueError(f"unknown gate {name!r}: beta<b>, route<rho>w<samples> or <proxy>@<b>")


def make_gate(name: str, leader: PureProxyGate, *, T_l: float, T_s: float, b_t: int, num_classes: int = 1000):
    return parse(name)(leader, T_l, T_s, b_t, num_classes)


if __name__ == "__main__":
    torch.manual_seed(0)
    T_L, T_S, B_T, K = 1.4, 0.9, 80, 50
    z_l, z_s = torch.randn(600, K) * 3, torch.randn(600, K) * 2
    BATCHES = [(i, min(i + 32, 600)) for i in range(0, 600, 32)]        # b_t 80 is not a multiple of 32: buckets flush inside a batch

    def mk(beta=0.5):
        return PureProxyGate("nnn", beta=beta, b_t=B_T, T_l=T_L, T_s=T_S, num_classes=K)

    def run(gates: dict, hook=None) -> dict:
        for g in gates.values():
            g.set_corruption("s", total_samples=600)
        outs = {n: [] for n in gates}
        for a, b in BATCHES:
            for n, g in gates.items():                                  # the leader first
                outs[n].append(g.calibrate(z_l[a:b], z_s[a:b]))
            if hook:
                hook(a, b, gates)
        return {n: torch.cat(v) for n, v in outs.items()}

    # ---- a follower is an independent gate of its own beta, to the bit; beta 0 is the fixed-TS duo --------------------------------------------
    lead = mk(0.5)
    got = run({"lead": lead, **{n: make_gate(n, lead, T_l=T_L, T_s=T_S, b_t=B_T, num_classes=K) for n in ("beta0", "beta0.25", "beta2", "beta8")}})
    for beta, n in ((0.5, "lead"), (0.0, "beta0"), (0.25, "beta0.25"), (2.0, "beta2"), (8.0, "beta8")):
        assert torch.equal(got[n], run({"g": mk(beta)})["g"]), n
    assert torch.equal(got["beta0"], 0.5 * (z_l / T_L) + 0.5 * (z_s / T_S))
    assert not torch.equal(got["beta2"], got["lead"]) and lead.n_refreshes == 8      # 7 full buckets of 80 and the 40-sample tail (the stream is exhausted)
    print("  followers: bit-equal to independent gates of their beta (a bucket flushing inside a batch included); beta0 is the fixed-TS duo")

    # ---- a follower must not run ahead of its leader; two flushes in one batch (a bucket filled, then the stream ends) are followed ----------
    lone = FollowerGate(mk(), beta=1.0, T_l=T_L, T_s=T_S, b_t=B_T, num_classes=K)
    try:
        run({"f": lone})
        raise AssertionError("a follower without a flush of its leader must raise")
    except RuntimeError:
        pass
    try:
        FollowerGate(lead, beta=1.0, T_l=T_L, T_s=T_S, b_t=B_T, num_classes=K)
        raise AssertionError("a follower made after its leader flushed must raise")
    except ValueError:
        pass
    for total, bt, bs in ((120, 100, 32), (96, 16, 32), (50, 7, 50)):    # the last batch fills a bucket and leaves a tail; several buckets per batch; one batch
        zl, zs = z_l[:total], z_s[:total]
        lead2 = PureProxyGate("nnn", beta=0.5, b_t=bt, T_l=T_L, T_s=T_S, num_classes=K)
        fol = make_gate("beta3", lead2, T_l=T_L, T_s=T_S, b_t=bt, num_classes=K)
        indep = PureProxyGate("nnn", beta=3.0, b_t=bt, T_l=T_L, T_s=T_S, num_classes=K)
        for g in (lead2, fol, indep):
            g.set_corruption("s", total_samples=total)
        outs = [torch.cat([g.calibrate(zl[a:a + bs], zs[a:a + bs]) for a in range(0, total, bs)]) for g in (lead2, fol, indep)]
        assert torch.equal(outs[1], outs[2]) and fol.n_refreshes == lead2.n_refreshes == indep.n_refreshes and len(lead2._flush_log) == lead2.n_refreshes, (total, bt, bs)
    print("  followers: refuse to run ahead of the leader; follow it through batches that flush twice (last batch of a stream) or many times")

    # ---- routing: untouched without a reset; the fixed weight on the reset member for exactly `window` samples; both at once = no routing ---
    def routed(rho, window, events):
        lead = mk()
        r = make_gate(f"route{rho:g}w{window}", lead, T_l=T_L, T_s=T_S, b_t=B_T, num_classes=K)
        fired = set()

        def hook(a, b, gates):                                         # after batch [a, b): the run loop's apply()
            for pos, member in events:
                if pos == b and (pos, member) not in fired:
                    fired.add((pos, member))
                    r.notify_reset(member, pos)
        out = run({"lead": lead, "r": r}, hook)
        return out["lead"], out["r"]

    base, none = routed(0.2, 100, [])
    assert torch.equal(base, none)
    base, big = routed(0.2, 100, [(160, "large")])                    # samples 160..259 (inside batches 160..192 .. 256..288)
    expect = base.clone()
    expect[160:260] = 0.2 * (z_l[160:260] / T_L) + 0.8 * (z_s[160:260] / T_S)
    assert torch.equal(big, expect) and not torch.equal(big, base)
    _, small_side = routed(0.2, 100, [(160, "small")])
    expect = base.clone()
    expect[160:260] = 0.8 * (z_l[160:260] / T_L) + 0.2 * (z_s[160:260] / T_S)
    assert torch.equal(small_side, expect)
    _, hard = routed(0.0, 64, [(96, "large")])
    assert torch.equal(hard[96:160], z_s[96:160] / T_S) and torch.equal(hard[:96], base[:96])
    _, both = routed(0.2, 100, [(160, "large"), (160, "small")])
    assert torch.equal(both, base), "both members reset together: nobody to route to"
    _, later = routed(0.2, 100, [(160, "large"), (320, "small")])     # the later reset replaces the earlier (expired) one
    expect = base.clone()
    expect[160:260] = 0.2 * (z_l[160:260] / T_L) + 0.8 * (z_s[160:260] / T_S)
    expect[320:420] = 0.8 * (z_l[320:420] / T_L) + 0.2 * (z_s[320:420] / T_S)
    assert torch.equal(later, expect)
    print("  routing: the reset member gets rho for exactly `window` samples (mid-batch edges included), the other side mirrors, both at once and expiry handled")

    # ---- names ---------------------------------------------------------------------------------------------------------------------------------
    for bad in ("beta", "route0.2", "route2w10", "route0.2w0", "energy@1", "nope@1", "beta-1", "fixed1000"):
        try:
            parse(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    ag = make_gate("agree_duo@1", mk(), T_l=T_L, T_s=T_S, b_t=B_T, num_classes=K)
    assert isinstance(ag, PureProxyGate) and not isinstance(ag, FollowerGate) and ag.proxy_kind == "agree_duo" and ag.beta == 1.0
    assert run({"g": ag})["g"].shape == (600, K)
    print("gate_variants self-test passed")
