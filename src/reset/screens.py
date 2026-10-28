"""
screens.py
==========
The two offline screens of the `own_fast_vs_slow` detector, on the stored calibration streams of ANY duo (CPU, about a minute each). Both replay the
detector over the proxy sequences of the held-out calibration runs (src/reset/screening.py) and score it against the degradation labels (a member 1 point
below its frozen copy, M = 8, L = 16): CALIBRATION streams only, nothing is chosen on a test stream.

    S   stress pair on the held-out stream (rotations 0..3, seeds 0 1): both members collapse; the collapses are the events, the healthy stretches before
        them count for false alarms
    H   healthy pair (the conservative rates) on the held-out stream, rotations 0 and 2: more healthy time for the false-alarm budget
    T   (optional, evaluation only) the stress pair on the test stream: if its files exist the screen also scores the chosen settings on them

    shared_screen(P, ...)     the setting both members share: per false-alarm budget (0.5, 1, 2 per 1,000 proxy batches) the (configuration, threshold)
                              with the lowest delay. The budget-1 choice is the SHARED candidate of the live tuning. -> P.shared_screen
    member_screen(P, side, shared, ...)   one member alone over a wider grid (10 speed pairs x W0 x threshold): the two best settings at budget 1 with
                              speed pairs other than the shared one are the member's CANDIDATES. -> P.member_screen(side)

The replay does not change the trajectory (the reset model cannot be simulated offline), so only the FIRST collapse of a member is measured: the live
runs decide. Fifteen families of rules were screened this way to choose the rule (the others were dropped: git tag `pre-reset-cleanup`).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.reset import screening as SC
from src.reset.controller import MEMBERS, RuleSpec, logit_score
from src.reset.layout import Paths
from src.reset.member_rules import Setting

SPEEDS = [(0.1, 0.01), (0.1, 0.025), (0.05, 0.005), (0.05, 0.0125), (0.02, 0.002), (0.02, 0.005), (0.01, 0.001), (0.01, 0.0025), (0.005, 0.0005), (0.005, 0.00125)]
W0S = (20, 50, 100)
DELTAS = (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
GO_MARGIN = 0.20                      # the per-member screen's noise floor: a gain in delay below 20% is not told from noise (reported, not a stop)
FAMILY = SC.FAMILY


class ScreenError(RuntimeError):
    """The calibration streams cannot support the screen (missing runs, or no collapse to detect): the message says what to do."""


# ------------------------------------------------------------------------------------------------------------------ the streams

def load_streams(P: Paths, name: str, *, b_t: int, beta: float, heldout: bool = True, proxy: str = "nnn", short_label: bool = False) -> list[SC.ScreenStream]:
    """The stored runs of calibration experiment `name` that have a frozen run and gate records beside them. `heldout`: the held-out stream (S, H) or the test one (T)."""
    gate_tag = "" if (proxy, b_t, beta) == ("nnn", 256, 0.5) else f"__{proxy}_bt{b_t}_beta{beta:g}"
    out = []
    for f in sorted(glob.glob(str(P.gate_stress_dir / f"exp-{name}__{'heldout' if heldout else 'L'}*{gate_tag}__seed*.npz"))):
        if f.endswith("__frozen.npz"):
            continue
        run, frozen = Path(f), Path(f[:-4] + "__frozen.npz")
        if not (frozen.is_file() and run.with_suffix(".gate.json").is_file()):
            print(f"skipped (no frozen run or gate records): {run.name}")
            continue
        label = run.stem.replace(f"exp-{name}__", "").replace(gate_tag, "") if short_label else run.stem
        out.append(SC.ScreenStream.load(run, frozen, b_t, f"{name}/{label}"))
    return out


def score_member(spec: RuleSpec, streams: list[SC.ScreenStream], member: str) -> dict:
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


def member_scores(spec: RuleSpec, streams: list[SC.ScreenStream]) -> dict[str, dict]:
    """`SC.evaluate`'s pooled numbers, per member."""
    out = {}
    for m in MEMBERS:
        r = score_member(spec, streams, m)
        out[m] = {k: r[k] for k in ("events", "misses", "fa", "delay", "miss_rate")}
    return out


# ------------------------------------------------------------------------------------------------------------------ the shared setting

def shared_screen(P: Paths, *, beta: float, b_t: int = 256, budgets=SC.BUDGETS, stress: str = "S", healthy: str = "H", test: str = "T", log=print) -> dict:
    """Screen the detector's grid (src/reset/screening.py) on the calibration streams, freeze the choice per budget, score it on the test streams if there are
    any. Writes P.shared_screen (and a .csv of the grid and of the budget-1 table beside it). Returns the JSON."""
    tuning = load_streams(P, stress, b_t=b_t, beta=beta, short_label=True)
    hl = load_streams(P, healthy, b_t=b_t, beta=beta, short_label=True) if healthy else []
    tst = load_streams(P, test, b_t=b_t, beta=beta, heldout=False, short_label=True) if test else []
    if not tuning:
        raise ScreenError(f"no calibration runs of experiment {stress!r} in {P.gate_stress_dir}")
    calib = tuning + hl
    for st in calib + tst:
        log(f"{st.name:60s} {st.n:5d} proxy batches; degraded stretches (start, length in batches): " + "; ".join(f"{m} {[(a, b - a) for a, b in st.stretches(m)]}" for m in MEMBERS))
    log(f"\ncalibration: {len(tuning)} stress + {len(hl)} healthy streams, {sum(s.n for s in calib)} proxy batches, "
        f"{sum(len(s.stretches(m)) for s in calib for m in MEMBERS)} degraded stretches; test: {len(tst)} streams")
    rows = SC.sweep(calib)
    selected = {b: SC.select(rows, b) for b in budgets}
    test_scores = {(b, fam): SC.evaluate(RuleSpec.from_json(row["spec"]), tst) for b, sel in selected.items() for fam, row in sel.items() if row is not None and tst}
    members = {(b, fam): {"tune": member_scores(RuleSpec.from_json(row["spec"]), calib), "test": member_scores(RuleSpec.from_json(row["spec"]), tst) if tst else None}
               for b, sel in selected.items() for fam, row in sel.items() if row is not None}
    b1 = min(budgets, key=lambda b: abs(b - 1.0))
    if all(row is None for row in selected[b1].values()):
        raise ScreenError(f"no setting has a point within {b1:g} false alarms per 1,000 batches, or the calibration streams have no degraded stretch to detect "
                          f"(the stress rates did not collapse the members: raise them, see `stress_up` in {P.config_path.name})")
    table = []
    for fam, row in selected[b1].items():
        if row is None:
            table.append({"rule": fam, "config": "no point within the budget"})
            continue
        t, mm = test_scores.get((b1, fam)), members[(b1, fam)]
        table.append({"rule": fam, "config": json.dumps({**row["config"], "thr": row["threshold"]}), "tune_delay": row["delay"], "tune_miss": row["miss_rate"],
                      "tune_fa": row["fa"], "vit_delay": mm["tune"]["large"]["delay"], "resnet_delay": mm["tune"]["small"]["delay"],
                      "test_delay": t and t["delay"], "test_miss": t and t["miss_rate"], "test_fa": t and t["fa"],
                      "test_vit_delay": mm["test"] and mm["test"]["large"]["delay"], "test_resnet_delay": mm["test"] and mm["test"]["small"]["delay"]})
    table = pd.DataFrame(table).sort_values("tune_delay", na_position="last")
    log(f"\nbudget {b1:g} false alarms per 1,000 proxy batches (delays in proxy batches of {b_t} images; a miss counts its whole stretch): the setting is chosen "
        f"on the calibration streams" + (", then scored on the test streams" if tst else "") + "\n" + table.drop(columns="config").round(2).to_string(index=False))
    sensitivity = {r: SC.label_sensitivity(RuleSpec.from_json(selected[b1][r]["spec"]), calib) for r in table["rule"] if selected[b1][r] is not None}
    for r, sens in sensitivity.items():
        log(f"label sensitivity of {r}: " + "; ".join(f"M{s['M']}/m{s['margin']:g}/L{s['L']}: delay {s['delay']:.1f} miss {s['miss_rate']:.2f}" for s in sens))
    out = {"duo": P.duo, "beta": beta, "b_t": b_t, "label": SC.LABEL, "tolerance": SC.TOLERANCE, "calibration_streams": [s.name for s in calib], "test_streams": [s.name for s in tst],
           "selected": {str(b): sel for b, sel in selected.items()}, "test_scores": {f"{b}|{fam}": s for (b, fam), s in test_scores.items()},
           "per_member": {f"{b}|{fam}": v for (b, fam), v in members.items()}, "sensitivity": sensitivity, "rows": rows}
    P.screens_dir.mkdir(parents=True, exist_ok=True)
    P.shared_screen.write_text(json.dumps(out, default=float))
    pd.DataFrame(rows).drop(columns="spec").to_csv(P.shared_screen.with_suffix(".csv"), index=False)
    table.to_csv(P.shared_screen.with_name(P.shared_screen.stem + f"__budget{b1:g}.csv"), index=False)
    log(f"wrote {P.shared_screen}")
    return out


def shared_setting(screen: dict, budget: float = 1.0) -> Setting:
    """The shared candidate: the budget's selected (speeds, W0, threshold) of the shared screen."""
    row = screen["selected"][str(float(budget))][FAMILY]
    if row is None:
        raise ScreenError(f"the shared screen has no setting within budget {budget:g}")
    (af, as_), w0 = row["config"]["speed"], row["config"]["W0"]
    return Setting(float(af), float(as_), int(w0), float(row["threshold"]))


# ------------------------------------------------------------------------------------------------------------------ one member

def spec_of(s: Setting) -> RuleSpec:
    return RuleSpec.of(own=dict(kind="fast_vs_slow", delta=s.delta, alpha_fast=s.alpha_fast, alpha_slow=s.alpha_slow), W0=s.W0)


def proxy_lag(streams: list[SC.ScreenStream], member: str, pre: tuple[int, int] = (60, 10), z: float = 2.0) -> list[dict]:
    """Per degraded stretch of `member`: onset, length, the first batch (counted from the onset) at which its proxy logit is `z` standard deviations below its mean in
    the batches `pre` before the onset (None: never inside the stretch), and the mean / final standardized proxy inside the stretch. Stretches with a short pre-window are skipped."""
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


def member_screen(P: Paths, side: str, shared: Setting, *, beta: float, b_t: int = 256, budgets=SC.BUDGETS, stress: str = "S", healthy: str = "H", log=print) -> dict:
    """Screen one member alone over the wider grid (SPEEDS x W0S x DELTAS) on the calibration streams. Candidates: the two best settings at budget 1 whose speed pairs
    are neither the shared one nor each other's. Writes P.member_screen(side) (+ .csv). Returns the JSON."""
    calib = load_streams(P, stress, b_t=b_t, beta=beta) + (load_streams(P, healthy, b_t=b_t, beta=beta) if healthy else [])
    if not calib:
        raise ScreenError(f"no calibration runs in {P.gate_stress_dir}")
    n_events = sum(len(s.stretches(side)) for s in calib)
    name = "ResNet" if side == "small" else "ViT"
    log(f"{len(calib)} calibration streams, {sum(s.n for s in calib)} proxy batches, {n_events} degraded stretches of the {side} member")
    if n_events == 0:
        raise ScreenError(f"the {side} member never collapses in the calibration streams, so there is no delay to screen: raise its stress rate (`stress_up` in {P.config_path.name})")
    rows = [{"alpha_fast": af, "alpha_slow": as_, "W0": w0, "delta": d, **score_member(spec_of(Setting(af, as_, w0, d)), calib, side)}
            for af, as_ in SPEEDS for w0 in W0S for d in DELTAS]
    df = pd.DataFrame(rows)
    ref = {f"delta {d:g}": score_member(spec_of(Setting(shared.alpha_fast, shared.alpha_slow, shared.W0, d)), calib, side) for d in sorted({shared.delta, 1.0})}
    key = lambda r: (r["delay"] if r["delay"] == r["delay"] else float("inf"), r["miss_rate"] if r["miss_rate"] == r["miss_rate"] else float("inf"), r["fa"])
    selected, candidates = {}, []
    for b in budgets:
        ok = [r for r in rows if r["fa"] <= b and r["events"] > 0]
        selected[b] = min(ok, key=key) if ok else None
    b1 = min(budgets, key=lambda b: abs(b - 1.0))
    ok1 = sorted((r for r in rows if r["fa"] <= b1 and r["events"] > 0), key=key)
    taken = {(shared.alpha_fast, shared.alpha_slow)}
    for r in ok1:
        pair = (r["alpha_fast"], r["alpha_slow"])
        if pair not in taken:
            candidates.append(r)
            taken.add(pair)
        if len(candidates) == 2:
            break
    show = lambda r: (f"speeds ({r['alpha_fast']:g}, {r['alpha_slow']:g}) W0 {r['W0']} delta {r['delta']:g}: delay {r['delay']:.1f} (detected {r['delay_detected']:.1f}), "
                      f"miss {r['miss_rate']:.2f} ({r['misses']}/{r['events']}), FA {r['fa']:.2f} per 1,000")
    log(f"\nshared setting ({shared.alpha_fast:g}, {shared.alpha_slow:g}) W0 {shared.W0} on the same member:")
    for k, r in ref.items():
        log(f"  {k}: delay {r['delay']:.1f} (detected {r['delay_detected']:.1f}), miss {r['miss_rate']:.2f} ({r['misses']}/{r['events']}), FA {r['fa']:.2f} per 1,000")
    for b, r in selected.items():
        log(f"\nbest at budget {b:g}: " + (show(r) if r else "none"))
    shared_ok = [r for r in rows if (r["alpha_fast"], r["alpha_slow"], r["W0"]) == (shared.alpha_fast, shared.alpha_slow, shared.W0) and r["fa"] <= b1 and r["events"] > 0]
    ref_best = min(shared_ok, key=key) if shared_ok else None
    verdict = None
    if ok1 and ref_best:
        gain = 1.0 - ok1[0]["delay"] / ref_best["delay"]
        verdict = {"shared_best": ref_best, "best": ok1[0], "relative_delay_reduction": gain, "above_noise_floor": bool(gain >= GO_MARGIN)}
        log(f"\nshared setting at its own best delta within the budget: {show(ref_best)}\nbest setting overall: {show(ok1[0])}\n"
            f"relative delay reduction {100 * gain:.0f}% (the screen's noise floor is about {100 * GO_MARGIN:.0f}%: {'above' if gain >= GO_MARGIN else 'below'} it)")
    log("live candidates (distinct speed pairs, other than the shared one): " + "; ".join(show(c) for c in candidates))
    lag = proxy_lag(calib, side)
    short = [r for r in lag if r["length"] < 100]
    log(f"\nthe {name}'s proxy around its {len(lag)} degraded stretches: {len(short)} are shorter than 100 batches; the proxy never falls 2 sd in "
        f"{sum(r['first_below'] is None for r in lag)}; its mean standardized value is positive in {sum(r['mean_z'] > 0 for r in lag)}")
    out = {"member": side, "streams": [s.name for s in calib], "events": n_events, "grid": {"speeds": SPEEDS, "W0": W0S, "delta": DELTAS}, "reference": ref,
           "selected": {str(b): r for b, r in selected.items()}, "candidates": candidates, "verdict": verdict, "proxy_lag": lag, "rows": rows}
    P.screens_dir.mkdir(parents=True, exist_ok=True)
    P.member_screen(side).write_text(json.dumps(out, default=float, indent=1))
    df.to_csv(P.member_screen(side).with_suffix(".csv"), index=False)
    log(f"wrote {P.member_screen(side)}")
    return out


def candidates_of(P: Paths, side: str) -> list[Setting]:
    """The member's candidate settings from its screen (not the shared one)."""
    d = json.loads(P.member_screen(side).read_text())
    return [Setting(c["alpha_fast"], c["alpha_slow"], int(c["W0"]), c["delta"]) for c in d["candidates"]]


def synthetic_stream(path: Path, name: str, collapse_at: int | None, seed: int, *, lrs=(0.0001, 0.0005), rotation: int = 0, n: int = 400, b_t: int = 256,
                     beta: float = 0.5, collapse=("small",)) -> Path:
    """A synthetic calibration run on disk, for the self-tests (the files a calibration job writes): the members in `collapse` collapse at batch `collapse_at`
    (accuracy to 0.05, proxy to 0.2), the others never do. Returns the adaptive run's path."""
    g = np.random.default_rng(seed)
    r_l = np.clip(0.7 + 0.03 * g.standard_normal(n), 0.01, 0.99)
    r_s = np.clip(0.6 + 0.03 * g.standard_normal(n), 0.01, 0.99)
    acc_l = np.clip(0.5 + 0.02 * g.standard_normal(n), 0, 1)
    acc_s = np.clip(0.4 + 0.02 * g.standard_normal(n), 0, 1)
    if collapse_at is not None:
        ramp = np.clip((np.arange(n) - collapse_at) / 10, 0, 1)
        if "large" in collapse:
            r_l, acc_l = r_l * (1 - ramp) + 0.2 * ramp, acc_l * (1 - ramp) + 0.05 * ramp
        if "small" in collapse:
            r_s, acc_s = r_s * (1 - ramp) + 0.2 * ramp, acc_s * (1 - ramp) + 0.05 * ramp
    u = g.random(n * b_t)                                   # one draw per image, shared by every series: the same images, as in a real run
    corr = lambda acc: (u < np.repeat(acc, b_t)).astype(np.uint8)
    gate = "" if (beta, b_t) == (0.5, 256) else f"__nnn_bt{b_t}_beta{beta:g}"
    stem = path / f"exp-{name}__heldout{f'__rot{rotation}' if rotation else ''}__L{lrs[0]:g}_S{lrs[1]:g}{gate}__seed{seed}"
    np.savez(f"{stem}.npz", large=corr(acc_l), small=corr(acc_s), ts=corr(np.full(n, 0.5)), gate=corr(np.full(n, 0.5)))
    np.savez(f"{stem}__frozen.npz", large=corr(np.full(n, 0.5)), small=corr(np.full(n, 0.4)))
    Path(f"{stem}.gate.json").write_text(json.dumps([{"r_l": float(a), "r_s": float(b)} for a, b in zip(r_l, r_s)]))
    return Path(f"{stem}.npz")


if __name__ == "__main__":
    import tempfile

    from src.utils.config import load_global

    with tempfile.TemporaryDirectory() as tmp:
        g = load_global()
        P = Paths(type("G", (), {"path": lambda self, k, _t=tmp: Path(_t)})(), "toyduo")
        P.gate_stress_dir.mkdir(parents=True)
        for seed, c in ((1, 200), (2, 150)):
            synthetic_stream(P.gate_stress_dir, "S", c, seed)
        synthetic_stream(P.gate_stress_dir, "H", None, 3)
        quiet = lambda *a, **k: None
        out = shared_screen(P, beta=0.5, test="", log=quiet)
        assert P.shared_screen.is_file() and out["selected"]["1.0"][FAMILY] is not None and len(out["calibration_streams"]) == 3
        sh = shared_setting(json.loads(P.shared_screen.read_text()))
        assert isinstance(sh, Setting) and sh.alpha_fast > sh.alpha_slow
        m = member_screen(P, "small", sh, beta=0.5, log=quiet)
        assert P.member_screen("small").is_file() and m["events"] == 2 and len(m["rows"]) == len(SPEEDS) * len(W0S) * len(DELTAS), (m["events"], len(m["rows"]))
        assert len(candidates_of(P, "small")) == len(m["candidates"]) and all((c.alpha_fast, c.alpha_slow) != (sh.alpha_fast, sh.alpha_slow) for c in candidates_of(P, "small"))
        try:                                                  # the healthy large member never collapses: nothing to screen for it
            member_screen(P, "large", sh, beta=0.5, log=quiet)
            raise AssertionError("a member without a collapse must be refused")
        except ScreenError as e:
            assert "never collapses" in str(e)
        try:
            shared_screen(Paths(P.g, "otherduo"), beta=0.5, log=quiet)
            raise AssertionError("missing calibration runs must be refused")
        except ScreenError:
            pass
    print("screens self-test passed")
