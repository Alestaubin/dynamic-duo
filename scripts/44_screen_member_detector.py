#!/usr/bin/env python3
"""
Per-member screen of the `fast_vs_slow` detector (plan.md, "Per-member detectors for own_fast_vs_slow"): the SAME offline replay as the screen of
scripts/37b_screen_resets_stress.py (src/reset/screening.py), but scored on ONE member only (default the ResNet) and over a wider grid of the detector's
parameters, to find out whether the ResNet's gradual decay wants other settings than the shared one (the ViT's abrupt collapse is caught within 11-32 batches).

    calibration streams ONLY (the stress streams S and the healthy streams H of 04c; the test streams T are NOT read: nothing is chosen on test)
    member            the degraded stretches (1 point below the frozen copy, M = 8, L = 16) and the false alarms of that member only; false alarms per
                      1,000 proxy batches of that member
    grid              (alpha_fast, alpha_slow) in 10 pairs from (0.1, 0.01) to (0.005, 0.00125) x warm-up W0 in {20, 50, 100} (cooldown K = W0) x delta on the
                      sqrt-2 grid 0.35 .. 8; the shared setting of the two members is (0.05, 0.005), W0 20
    selection         per false-alarm budget (0.5, 1, 2) the lowest-delay setting (then fewest misses, then fewest false alarms), as `screening.select`;
                      the live candidates are the two best at budget 1 with distinct speed pairs other than the shared pair
    noise floor       the screen reports whether the best setting's delay at budget 1 is at least 20% below the shared setting's (the screen's noise floor); the live
                      tuning (scripts/40_tune_reset.py member) decides on the live runs either way: per-member settings are the default, the shared setting is a candidate

    diagnostics       (always, a few seconds) per degraded stretch of the member the batch at which its proxy first falls 2 standard deviations below its
                      level in the 50 batches before the onset (the proxy's lag), with the mean standardized proxy inside the stretch (positive: the proxy
                      ROSE while the accuracy fell)

    python scripts/44_screen_member_detector.py --duo vitb16_rn50v1 --bn --member small

Writes <paths.out>/results/screen_resets_stress/member_detector__<member>.{json,csv}.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.experiments.cli import add_common_args, setup
from src.reset import screening as SC
from src.reset.controller import MEMBERS, RuleSpec

SPEEDS = [(0.1, 0.01), (0.1, 0.025), (0.05, 0.005), (0.05, 0.0125), (0.02, 0.002), (0.02, 0.005), (0.01, 0.001), (0.01, 0.0025), (0.005, 0.0005), (0.005, 0.00125)]
W0S = (20, 50, 100)
DELTAS = (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
SHARED = {"alpha_fast": 0.05, "alpha_slow": 0.005, "W0": 20}
GO_MARGIN = 0.20


def spec_of(alpha_fast: float, alpha_slow: float, w0: int, delta: float) -> RuleSpec:
    return RuleSpec.of(own=dict(kind="fast_vs_slow", delta=delta, alpha_fast=alpha_fast, alpha_slow=alpha_slow), W0=w0)


def score_one(spec: RuleSpec, streams: list[SC.ScreenStream], member: str) -> dict:
    """Pooled delay / misses / false alarms of `spec` on `member` over `streams` (false alarms per 1,000 proxy batches of that member)."""
    events = misses = false = batches = 0
    delays, detected = [], []
    for st in streams:
        s = SC.score_member(SC.replay(spec, st)[member], st.stretches(member), st.n)
        events, misses, false, batches = events + s["events"], misses + s["misses"], false + s["false"], batches + st.n
        delays += s["delays"]
        detected += s["detected"]
    nan = float("nan")
    return {"events": events, "misses": misses, "false": false, "batches": batches, "fa": 1000.0 * false / max(batches, 1),
            "miss_rate": misses / events if events else nan, "delay": float(np.mean(delays)) if delays else nan,
            "delay_detected": float(np.mean(detected)) if detected else nan}


def proxy_lag(streams: list[SC.ScreenStream], member: str, pre: tuple[int, int] = (60, 10), z: float = 2.0) -> list[dict]:
    """Per degraded stretch of `member`: onset, length, the first batch (counted from the onset) at which its proxy logit is `z` standard deviations below its mean in
    the batches `pre` before the onset (None: never inside the stretch), and the mean / final standardized proxy inside the stretch. Stretches with a short pre-window are skipped."""
    from src.reset.controller import logit_score
    out = []
    for st in streams:
        r = st.r_s if member == "small" else st.r_l
        x = np.array([logit_score(float(v), 1e-6) for v in r])
        for a, b in st.stretches(member):
            base = x[max(a - pre[0], 0):max(a - pre[1], 1)]
            if len(base) < 10:
                continue
            zz = (x - base.mean()) / (base.std() + 1e-9)
            hit = np.flatnonzero(zz[a:b] < -z)
            out.append({"stream": st.name, "onset": int(a), "length": int(b - a), "first_below": int(hit[0]) if len(hit) else None,
                        "mean_z": float(zz[a:b].mean()), "final_z": float(zz[b - 1])})
    return out


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), tta=False, seeds=False)
    ap.add_argument("--member", choices=MEMBERS, default="small", help="large = ViT, small = ResNet")
    ap.add_argument("--stress", default="S")
    ap.add_argument("--healthy", default="H")
    ap.add_argument("--beta", type=float, default=2.0, help="the gate beta of the 04c runs (in their file names)")
    ap.add_argument("--b_t", type=int, default=256)
    ap.add_argument("--budgets", type=float, nargs="+", default=list(SC.BUDGETS))
    args = ap.parse_args()
    ctx, _, out = setup(args, "screen_resets_stress", default_seeds=[0])
    folder = out.parent / "gate_stress" / f"{args.duo}__tent{'_bn' if args.bn else ''}"
    gate_tag = "" if (args.b_t, args.beta) == (256, 0.5) else f"__nnn_bt{args.b_t}_beta{args.beta:g}"

    def load(name: str) -> list[SC.ScreenStream]:
        found = []
        for f in sorted(glob.glob(str(folder / f"exp-{name}__heldout*{gate_tag}__seed*.npz"))):
            if f.endswith("__frozen.npz"):
                continue
            run, frozen = Path(f), Path(f[:-4] + "__frozen.npz")
            if frozen.is_file() and run.with_suffix(".gate.json").is_file():
                found.append(SC.ScreenStream.load(run, frozen, args.b_t, f"{name}/{run.stem}"))
        return found

    calib = load(args.stress) + (load(args.healthy) if args.healthy else [])
    if not calib:
        print(f"no calibration runs in {folder}")
        return 1
    m = args.member
    n_events = sum(len(s.stretches(m)) for s in calib)
    print(f"{len(calib)} calibration streams, {sum(s.n for s in calib)} proxy batches, {n_events} degraded stretches of the {'ResNet' if m == 'small' else 'ViT'}")

    rows = []
    for af, as_ in SPEEDS:
        for w0 in W0S:
            for d in DELTAS:
                rows.append({"alpha_fast": af, "alpha_slow": as_, "W0": w0, "delta": d, **score_one(spec_of(af, as_, w0, d), calib, m)})
    df = pd.DataFrame(rows)
    ref = {f"delta {d:g}": score_one(spec_of(SHARED["alpha_fast"], SHARED["alpha_slow"], SHARED["W0"], d), calib, m) for d in (0.7, 1.0)}
    key = lambda r: (r["delay"] if r["delay"] == r["delay"] else float("inf"), r["miss_rate"] if r["miss_rate"] == r["miss_rate"] else float("inf"), r["fa"])
    selected, candidates = {}, []
    for b in args.budgets:
        ok = [r for r in rows if r["fa"] <= b and r["events"] > 0]
        selected[b] = min(ok, key=key) if ok else None
    b1 = min(args.budgets, key=lambda b: abs(b - 1.0))
    ok1 = sorted((r for r in rows if r["fa"] <= b1 and r["events"] > 0), key=key)
    taken = {(SHARED["alpha_fast"], SHARED["alpha_slow"])}           # the live candidates: the two best settings at budget 1 with distinct speed pairs, other than the shared pair
    for r in ok1:
        pair = (r["alpha_fast"], r["alpha_slow"])
        if pair not in taken:
            candidates.append(r)
            taken.add(pair)
        if len(candidates) == 2:
            break

    show = lambda r: (f"speeds ({r['alpha_fast']:g}, {r['alpha_slow']:g}) W0 {r['W0']} delta {r['delta']:g}: delay {r['delay']:.1f} (detected {r['delay_detected']:.1f}), "
                      f"miss {r['miss_rate']:.2f} ({r['misses']}/{r['events']}), FA {r['fa']:.2f} per 1,000")
    print("\nshared setting on the same member:")
    for k, r in ref.items():
        print(f"  speeds (0.05, 0.005) W0 20 {k}: delay {r['delay']:.1f} (detected {r['delay_detected']:.1f}), miss {r['miss_rate']:.2f} ({r['misses']}/{r['events']}), FA {r['fa']:.2f} per 1,000")
    for b, r in selected.items():
        print(f"\nbest at budget {b:g}: " + (show(r) if r else "none"))
    print(f"\nbest delay per speed pair at budget {b1:g}:")
    for af, as_ in SPEEDS:
        sub = [r for r in ok1 if (r["alpha_fast"], r["alpha_slow"]) == (af, as_)]
        print(f"  ({af:g}, {as_:g}): " + (show(sub[0]) if sub else "no point within the budget"))
    shared_ok = [r for r in rows if (r["alpha_fast"], r["alpha_slow"], r["W0"]) == (SHARED["alpha_fast"], SHARED["alpha_slow"], SHARED["W0"]) and r["fa"] <= b1 and r["events"] > 0]
    ref_best = min(shared_ok, key=key) if shared_ok else None
    verdict = None
    if ok1 and ref_best:
        gain = 1.0 - ok1[0]["delay"] / ref_best["delay"]
        verdict = {"shared_best": ref_best, "best": ok1[0], "relative_delay_reduction": gain, "above_noise_floor": bool(gain >= GO_MARGIN)}
        print(f"\nshared setting at its own best delta within the budget: {show(ref_best)}")
        print(f"best setting overall: {show(ok1[0])}\nrelative delay reduction {100 * gain:.0f}% (the screen's noise floor is about {100 * GO_MARGIN:.0f}%: {'above' if gain >= GO_MARGIN else 'below'} it)")
        print("live candidates (distinct speed pairs, other than the shared one): " + "; ".join(show(c) for c in candidates))
    lag = proxy_lag(calib, m)
    print(f"\nthe {'ResNet' if m == 'small' else 'ViT'}'s proxy around its {len(lag)} degraded stretches (first batch after the onset at which the proxy is 2 sd below its pre-onset level; mean standardized proxy inside):")
    for r in sorted(lag, key=lambda r: r["length"]):
        print(f"  length {r['length']:5d}: first {r['first_below']}, mean z {r['mean_z']:+.1f}, final z {r['final_z']:+.1f}")
    short = [r for r in lag if r["length"] < 100]
    print(f"{len(short)} of {len(lag)} stretches are shorter than 100 batches; proxy never falls 2 sd in {sum(r['first_below'] is None for r in lag)}; mean z positive in {sum(r['mean_z'] > 0 for r in lag)}")
    tag = f"member_detector__{m}"
    (out / f"{tag}.json").write_text(json.dumps({"member": m, "streams": [s.name for s in calib], "events": n_events, "grid": {"speeds": SPEEDS, "W0": W0S, "delta": DELTAS},
                                                 "reference": ref, "selected": {str(b): r for b, r in selected.items()}, "candidates": candidates, "verdict": verdict, "proxy_lag": lag, "rows": rows}, default=float, indent=1))
    df.to_csv(out / f"{tag}.csv", index=False)
    print(f"\nwrote {out / f'{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
