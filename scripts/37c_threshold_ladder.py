#!/usr/bin/env python3
"""
Detection-threshold ladder of the reset-rule finalists, offline (Phase 4, R-D, after scripts/37b_screen_resets_stress.py).

37b chooses, per family and false-alarm budget, a configuration AND a threshold from a fixed grid. Its grid stops at the sensitive end: the
budget-1 choice of every finalist is the SMALLEST grid threshold (delta 1.0 for fast_vs_slow, h 2.0 for cusum), so nothing says that a more
sensitive threshold is not better. This script keeps each finalist's other parameters (speed or nu, warm-up W0) as 37b selected them at
`--budget` and sweeps the threshold over a finer, longer ladder, on the calibration streams (S + H) and, frozen, on the test streams (T):

    python scripts/37c_threshold_ladder.py --duo vitb16_rn50v1 --bn

Prints and writes <paths.out>/results/screen_resets_stress/threshold_ladder__<duo>.{json,csv}: per family and threshold the false-alarm rate, the
miss rate and the delay on the calibration streams and on the test streams. The ladder the live runs use (src/reset/tuning.py) is read from it:
the thresholds whose calibration false-alarm rate brackets the budget on both sides.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.reset import screening as SC
from src.reset.controller import MEMBERS, RuleSpec

_spec = importlib.util.spec_from_file_location("screen37b", Path(__file__).with_name("37b_screen_resets_stress.py"))
_s37b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s37b)

# the threshold of each detector, in the order of its sensitivity (most sensitive first); the 37b grid starts at 1.0 / 2.0
THRESHOLDS = {"fast_vs_slow": (0.25, 0.35, 0.5, 0.7, 1.0, 1.25, 1.5, 2.0), "cusum": (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
              "imm_vs_ema": (0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0), "level": (-0.5, -0.75, -1.0, -1.5, -2.0)}
FINALISTS = ("gap_fast_vs_slow", "own_fast_vs_slow", "own_cusum")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duo", required=True)
    ap.add_argument("--bn", action="store_true")
    ap.add_argument("--budget", type=float, default=1.0, help="the 37b selection whose other parameters are kept (default %(default)s)")
    ap.add_argument("--families", nargs="+", default=list(FINALISTS))
    ap.add_argument("--screen_json", default=None)
    args = ap.parse_args()

    from src.experiments.cli import setup
    from src.experiments.cli import add_common_args      # 37b's main() would also redo the whole sweep: only its stream loading is reproduced here
    ap2 = add_common_args(argparse.ArgumentParser(), tta=False, seeds=False)
    a2 = ap2.parse_args(["--duo", args.duo] + (["--bn"] if args.bn else []))
    ctx, _, out = setup(a2, "screen_resets_stress", default_seeds=[0])
    folder = out.parent / "gate_stress" / f"{args.duo}__tent{'_bn' if args.bn else ''}"
    import glob
    gate_tag = "__nnn_bt256_beta2"

    def find(name: str, heldout: bool):
        found = []
        for f in sorted(glob.glob(str(folder / f"exp-{name}__{'heldout' if heldout else 'L'}*{gate_tag}__seed*.npz"))):
            if f.endswith("__frozen.npz"):
                continue
            run, frozen = Path(f), Path(f[:-4] + "__frozen.npz")
            if frozen.is_file() and run.with_suffix(".gate.json").is_file():
                found.append((run.stem.replace(f"exp-{name}__", "").replace(gate_tag, ""), run, frozen))
        return found

    calib = [SC.ScreenStream.load(r, f, 256, f"S/{lab}") for lab, r, f in find("S", True)] + [SC.ScreenStream.load(r, f, 256, f"H/{lab}") for lab, r, f in find("H", True)]
    test = [SC.ScreenStream.load(r, f, 256, f"T/{lab}") for lab, r, f in find("T", False)]
    print(f"{len(calib)} calibration streams, {len(test)} test streams")

    path = Path(args.screen_json) if args.screen_json else out / f"tent__{args.duo}{'_bn' if args.bn else ''}__SHT__nnn_bt256__n10.json"
    chosen = json.loads(path.read_text())["selected"][str(float(args.budget))]
    rows = []
    for fam in args.families:
        base = chosen[fam]
        spec0 = RuleSpec.from_json(base["spec"])
        kind = next(iter(spec0.to_json()["parts"].values()))["kind"]
        name = _s37b_threshold_name(kind)
        for thr in THRESHOLDS[kind]:
            j = spec0.to_json()
            for part in j["parts"].values():
                part[name] = float(thr)
            spec = RuleSpec.from_json(j)
            c, t = SC.evaluate(spec, calib), SC.evaluate(spec, test)
            rows.append({"family": fam, "threshold": thr, "config": json.dumps(base["config"]), "calib_fa": c["fa"], "calib_miss": c["miss_rate"], "calib_delay": c["delay"],
                         "test_fa": t["fa"], "test_miss": t["miss_rate"], "test_delay": t["delay"],
                         "vit_delay": _member(spec, calib, "large"), "resnet_delay": _member(spec, calib, "small")})
    df = pd.DataFrame(rows)
    print(df.drop(columns="config").round(2).to_string(index=False))
    stem = out / f"threshold_ladder__{args.duo}{'_bn' if args.bn else ''}"
    df.to_csv(f"{stem}.csv", index=False)
    Path(f"{stem}.json").write_text(json.dumps({"budget": args.budget, "rows": rows}, default=float, indent=1))
    print(f"wrote {stem}.json")
    return 0


def _s37b_threshold_name(kind: str) -> str:
    return {"fast_vs_slow": "delta", "imm_vs_ema": "delta", "cusum": "h", "level": "tau", "slope": "gamma"}[kind]


def _member(spec: RuleSpec, streams, m: str) -> float:
    return _s37b.member_scores(spec, streams)[m]["delay"]


if __name__ == "__main__":
    sys.exit(main())
