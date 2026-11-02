"""
final_configs.py
================
The frozen configuration file of the FINAL TEST (scripts/42_final_test.py) of any duo, built from the held-out tuning's stored results, so that nothing in it
is chosen on the test stream. One file per duo, `<reset_tune>/[<duo>/]final_configs.json` (the main duo's predates this module and was written by hand).

    python scripts/42_final_test.py configs --duo convnextb_deits

The methods (independent signal; the names are the ones scripts/42 pairs on):

    no_reset_tuned         `tune_reset.py --baseline none`: the continual ladder's accuracy argmax per member (none__baseline2.json)
    no_reset_conservative  the continual stage's conservative rates (results.json of the continual folder)
    fixed500, fixed1000    `--baseline fixed500` / `fixed1000`: each member's rate with a reset every 500 / 1,000 steps (<cond>__rates.json)
    per_member_own_fast    the tuned rule: each member's own rate and detector setting (member__rates.json)

and the random-matched control of the rule. A piece that is not there yet is named in the error: run its tuning first (the command is in the message).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.reset.layout import Paths

SIDES = ("large", "small")
SCHEDULES = (("fixed500", "Every 500 steps"), ("fixed1000", "Every 1,000 steps"))


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def build(P: Paths) -> dict:
    """The final-test configuration of `P.duo`, or ValueError naming every missing piece."""
    if P.signal != "indep":
        raise ValueError("the final test's comparison methods (no reset, schedules) are tuned for the independent signal only")
    d, missing = P.tune_dir, []
    cont = P.continual_dir / "results.json"
    need = {"the continual stage (conservative rates)": cont, "the no-reset rates": d / "none__baseline2.json", "the tuned rule": d / "member__rates.json",
            **{f"the {c} schedule's rates": d / f"{c}__rates.json" for c, _ in SCHEDULES}}
    for what, p in need.items():
        if not p.is_file():
            hint = f"tune_reset.py --duo {P.duo} --baseline {Path(p).name.split('__')[0]}" if "__" in Path(p).name and not Path(p).name.startswith("member") else f"tune_reset.py --duo {P.duo}"
            missing.append(f"{what}: {p} (run {hint} --submit)")
    if missing:
        raise ValueError("the held-out tuning is not complete:\n  " + "\n  ".join(missing))

    def pair(chosen) -> list[float]:
        return [float(chosen[s]) for s in SIDES] if isinstance(chosen, dict) else [float(x) for x in chosen]

    sides = _read(cont)["sides"]
    for s in SIDES:
        if sides[s].get("conservative") is None:
            raise ValueError(f"the continual stage names no conservative rate for the {s} member")
    rule = _read(d / "member__rates.json")
    methods = [
        {"name": "no_reset_tuned", "cond": "none", "lrs": pair(_read(d / "none__baseline2.json")["chosen"]), "label": "No reset, accuracy-tuned rates", "kind": "baseline"},
        {"name": "no_reset_conservative", "cond": "none", "lrs": pair({s: sides[s]["conservative"] for s in SIDES}), "label": "No reset, conservative rates", "kind": "baseline"},
        *[{"name": c, "cond": c, "lrs": pair(_read(d / f"{c}__rates.json")["chosen"]), "label": label, "kind": "schedule"} for c, label in SCHEDULES],
        {"name": "per_member_own_fast", "cond": rule["cond"], "lrs": pair(rule["chosen"]), "label": "Own fast/slow, per-member settings", "kind": "rule"},
    ]
    return {"note": f"Frozen {time.strftime('%Y-%m-%d %H:%M')} from the held-out tuning of {P.duo} (src/reset/final_configs.py). Nothing here is chosen on the test stream.",
            "methods": methods, "control": [{"rule": "per_member_own_fast", "draws": [0]}]}


def write(P: Paths) -> Path:
    cfg = build(P)
    path = P.tune_dir / "final_configs.json"
    path.write_text(json.dumps(cfg, indent=1))
    return path


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        G = type("G", (), {"path": lambda self, k, _t=tmp: Path(_t)})()
        P = Paths(G, "toy")
        P.tune_dir.mkdir(parents=True)
        P.continual_dir.mkdir(parents=True)
        try:
            build(P)
            raise AssertionError("an untuned duo must be refused")
        except ValueError as e:
            assert "continual stage" in str(e) and "fixed500" in str(e) and "tuned rule" in str(e), e
        (P.continual_dir / "results.json").write_text(json.dumps({"sides": {"large": {"conservative": 1e-6}, "small": {"conservative": 1e-4}}}))
        (P.tune_dir / "none__baseline2.json").write_text(json.dumps({"chosen": {"large": 2e-6, "small": 2e-4}}))
        (P.tune_dir / "member__rates.json").write_text(json.dumps({"cond": "own_fast_vs_slow@pmL1-0.1-20-1_S1-0.1-20-1", "chosen": [3e-5, 1e-3]}))
        (P.tune_dir / "fixed500__rates.json").write_text(json.dumps({"chosen": {"large": 5e-5, "small": 5e-4}}))
        try:
            build(P)
            raise AssertionError("a missing schedule must be refused")
        except ValueError as e:
            assert "fixed1000" in str(e) and "fixed500" not in str(e).replace("fixed500__rates", ""), e
        (P.tune_dir / "fixed1000__rates.json").write_text(json.dumps({"chosen": {"large": 4e-5, "small": 4e-4}}))
        cfg = json.loads(write(P).read_text())
        by = {m["name"]: m for m in cfg["methods"]}
        assert list(by) == ["no_reset_tuned", "no_reset_conservative", "fixed500", "fixed1000", "per_member_own_fast"]
        assert by["no_reset_tuned"]["lrs"] == [2e-6, 2e-4] and by["no_reset_conservative"]["lrs"] == [1e-6, 1e-4] and by["fixed1000"]["lrs"] == [4e-5, 4e-4]
        assert by["per_member_own_fast"]["lrs"] == [3e-5, 1e-3] and by["per_member_own_fast"]["kind"] == "rule" and by["fixed500"]["cond"] == "fixed500"
        assert cfg["control"] == [{"rule": "per_member_own_fast", "draws": [0]}]
        try:
            build(Paths(G, "toy", "joint"))
            raise AssertionError("the joint signal has no comparison methods")
        except ValueError:
            pass
    print("final_configs self-test passed")
