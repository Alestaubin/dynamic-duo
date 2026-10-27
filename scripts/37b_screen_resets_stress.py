#!/usr/bin/env python3
"""
Offline screening of the reset rules at FIXED STRESS RATES (Phase 4, R-D; src/reset/screening.py): both members at a rate that collapses it
(ViT 1.4e-4, ResNet 5e-4: both die on the held-out stream and on the test stream), Vanilla DD, no reset. 37_screen_resets.py does the same for the
episodic-tuned rates; this one reads the runs of scripts/04c_gate_stress.py (the gate attached, the frozen run over the same images):

    S   stress pair on the held-out stream (the 4 held-out shifts x 4 rounds, rotations 0..3, seeds 0 1): stops once both members are dead.
        The CALIBRATION streams: their collapses are the events, their healthy stretches before them count for false alarms.
    H   healthy pair (the 04b conservative rates 1.26e-5 / 2.2e-5) on the held-out stream, full length: more healthy stretches for the
        false-alarm budget (both members never collapse in them, so every alarm there is a false one).
    T   stress pair on the TEST stream, seeds 0 1 2: EVALUATION only; no parameter is chosen on it.

    python scripts/04c_gate_stress.py --duo vitb16_rn50v1 --bn --stream heldout --beta 2 --experiments --extra S:1.4e-4:5e-4 --seeds 0 1 \\
        --rotations 0 1 2 3 --stop_when all --after 20000 --shard auto          # array 0-7
    python scripts/04c_gate_stress.py ... --stream heldout --extra H:1.26e-5:2.2e-5 --seeds 0 --rotations 0 2 --shard auto     # array 0-1
    python scripts/04c_gate_stress.py ... --stream test --extra T:1.4e-4:5e-4 --seeds 0 1 2 --stop_when all --after 20000 --shard auto   # array 0-2
    sbatch -t 00:40:00 slurm/job_cpu.sh scripts/37b_screen_resets_stress.py --duo vitb16_rn50v1 --bn

Per rule family (own / gap / both x level, imm_vs_ema, fast_vs_slow, slope, cusum) and false-alarm budget (0.5, 1, 2 per 1,000 proxy batches) it
picks on the calibration streams the configuration and threshold with the lowest delay, freezes it and scores it on the test streams, pooled and per
member (ViT = large, ResNet = small). The replay does not change the trajectory (the reset model cannot be simulated offline), so only the FIRST
collapse of a member is measured: the live runs rank the finalists. Writes <paths.out>/results/screen_resets_stress/<tag>.{json,csv}, the budget table
and the delay-against-false-alarm curves.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.experiments.cli import add_common_args, setup
from src.reset import screening as SC
from src.reset.controller import MEMBERS, RuleSpec
from src.utils import provenance

_spec = importlib.util.spec_from_file_location("screen37", Path(__file__).with_name("37_screen_resets.py"))
_screen37 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_screen37)                 # its plot_curves (a script name that starts with digits cannot be imported)


def member_scores(spec: RuleSpec, streams: list[SC.ScreenStream]) -> dict[str, dict]:
    """`SC.evaluate`'s pooled numbers, per member."""
    out = {}
    for m in MEMBERS:
        events = misses = false = batches = 0
        delays = []
        for st in streams:
            s = SC.score_member(SC.replay(spec, st)[m], st.stretches(m), st.n)
            events, misses, false, batches, delays = events + s["events"], misses + s["misses"], false + s["false"], batches + st.n, delays + s["delays"]
        out[m] = {"events": events, "misses": misses, "fa": 1000.0 * false / max(batches, 1),
                  "delay": float(np.mean(delays)) if delays else float("nan"), "miss_rate": misses / events if events else float("nan")}
    return out


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), tta=False, seeds=False)
    ap.add_argument("--stress", default="S", help="name of the stress experiment on the held-out stream (04c --extra)")
    ap.add_argument("--healthy", default="H", help="name of the healthy experiment on the held-out stream ('' for none)")
    ap.add_argument("--test", default="T", help="name of the stress experiment on the test stream ('' for none)")
    ap.add_argument("--beta", type=float, default=2.0, help="the gate beta of those runs (in their file names; the proxy sequences do not depend on it)")
    ap.add_argument("--b_t", type=int, default=256)
    ap.add_argument("--budgets", type=float, nargs="+", default=list(SC.BUDGETS), help="false alarms per 1,000 proxy batches")
    ap.add_argument("--families", nargs="+", default=None, help="default: all 15 (own / gap / both x the five procedures)")
    args = ap.parse_args()
    tta = "tent"

    ctx, _, out = setup(args, "screen_resets_stress", default_seeds=[0])
    folder = out.parent / "gate_stress" / f"{args.duo}__{tta}{'_bn' if args.bn else ''}"
    gate_tag = "" if (args.b_t, args.beta) == (256, 0.5) else f"__nnn_bt{args.b_t}_beta{args.beta:g}"

    def find(name: str, heldout: bool) -> list[tuple[str, Path, Path]]:
        """(label, run, frozen) of every stored run of experiment `name`."""
        found = []
        for f in sorted(glob.glob(str(folder / f"exp-{name}__{'heldout' if heldout else 'L'}*{gate_tag}__seed*.npz"))):
            if f.endswith("__frozen.npz"):
                continue
            run, frozen = Path(f), Path(f[:-4] + "__frozen.npz")
            if not (frozen.is_file() and run.with_suffix(".gate.json").is_file()):
                print(f"skipped (no frozen run or gate records): {run.name}")
                continue
            found.append((run.stem.replace(f"exp-{name}__", "").replace(gate_tag, ""), run, frozen))
        return found

    tuning = [SC.ScreenStream.load(r, f, args.b_t, f"{args.stress}/{label}") for label, r, f in find(args.stress, True)]
    healthy = [SC.ScreenStream.load(r, f, args.b_t, f"{args.healthy}/{label}") for label, r, f in find(args.healthy, True)] if args.healthy else []
    test = [SC.ScreenStream.load(r, f, args.b_t, f"{args.test}/{label}") for label, r, f in find(args.test, False)] if args.test else []
    if not tuning:
        print(f"no calibration runs of experiment {args.stress!r} in {folder}: run scripts/04c_gate_stress.py first (see --help)")
        return 1
    calib = tuning + healthy
    for st in calib + test:
        print(f"{st.name:60s} {st.n:5d} proxy batches; degraded stretches (start, length in batches): " +
              "; ".join(f"{m} {[(a, b - a) for a, b in st.stretches(m)]}" for m in MEMBERS))
    print(f"\ncalibration: {len(tuning)} stress + {len(healthy)} healthy streams, {sum(s.n for s in calib)} proxy batches, "
          f"{sum(len(s.stretches(m)) for s in calib for m in MEMBERS)} degraded stretches; test: {len(test)} streams")

    fams = args.families or SC.families()
    rows = SC.sweep(calib, fams)
    selected = {b: SC.select(rows, b) for b in args.budgets}
    test_scores = {(b, fam): SC.evaluate(RuleSpec.from_json(row["spec"]), test) for b, sel in selected.items() for fam, row in sel.items() if row is not None and test}
    members = {(b, fam): {"tune": member_scores(RuleSpec.from_json(row["spec"]), calib),
                          "test": member_scores(RuleSpec.from_json(row["spec"]), test) if test else None}
               for b, sel in selected.items() for fam, row in sel.items() if row is not None}

    b1 = min(args.budgets, key=lambda b: abs(b - 1.0))
    if all(row is None for row in selected[b1].values()):
        print(f"no rule has a point within {b1:g} false alarms per 1,000 batches (or the calibration streams have no degraded stretch to detect): nothing to select")
        return 1
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
    print(f"\nbudget {b1:g} false alarms per 1,000 proxy batches (delays in proxy batches of {args.b_t} images; a miss counts its whole stretch): "
          f"the rule is chosen on the calibration streams, then scored on the test streams\n" + table.drop(columns="config").round(2).to_string(index=False))
    top = [r for r in table["rule"] if selected[b1][r] is not None][:3]
    sensitivity = {r: SC.label_sensitivity(RuleSpec.from_json(selected[b1][r]["spec"]), calib) for r in top}
    print(f"\ntop three on the calibration streams: {top}")
    for r in top:
        print(f"label sensitivity of {r}: " + "; ".join(f"M{s['M']}/m{s['margin']:g}/L{s['L']}: delay {s['delay']:.1f} miss {s['miss_rate']:.2f}" for s in sensitivity[r]))

    tag = f"{tta}__{args.duo}{'_bn' if args.bn else ''}__{args.stress}{args.healthy}{args.test}__nnn_bt{args.b_t}__n{len(calib)}"
    note = provenance.footer(ctx.duo, tta, args.bn, f"reset-rule screening at stress rates, calibrated on {len(tuning)} stress + {len(healthy)} healthy held-out streams, "
                             f"evaluated on {len(test)} test streams; Vanilla DD, no reset, replay without changing the trajectory")
    (out / f"{tag}.json").write_text(json.dumps({
        "args": vars(args), "label": SC.LABEL, "tolerance": SC.TOLERANCE, "calibration_streams": [s.name for s in calib], "test_streams": [s.name for s in test],
        "selected": {str(b): sel for b, sel in selected.items()}, "test_scores": {f"{b}|{fam}": s for (b, fam), s in test_scores.items()},
        "per_member": {f"{b}|{fam}": v for (b, fam), v in members.items()}, "sensitivity": sensitivity, "rows": rows}, default=float))
    pd.DataFrame(rows).drop(columns="spec").to_csv(out / f"{tag}.csv", index=False)
    table.to_csv(out / f"{tag}__budget{b1:g}.csv", index=False)
    for f in _screen37.plot_curves(rows, selected[b1], out / f"{tag}__curves.png", note):
        print(f"wrote {f}")
    print(f"wrote {out / f'{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
