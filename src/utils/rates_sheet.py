"""
rates_sheet.py
==============
docs/learning_rates.md: ONE sheet of every learning rate in the repo, per model x TTA method x setting x tuning method. It is a VIEW,
never a source: it is rendered from the files that hold the numbers, so it cannot drift from them.

  cfgs/duo_lrs/<duo>__<regime>.yaml                        the episodic rates (src/utils/duo_lrs.py: the one place they live)
  out/results/tune_continual_lr/<duo>__tent_bn/results.json   the continual rates of scripts/04b_tune_continual_lr.py (conservative, stress)
  out/results/gate_stress/<duo>__tent_bn/summary*.json     the rates actually run in the gate-stress experiments of scripts/04c_gate_stress.py

    python -m src.utils.rates_sheet            # rewrite docs/learning_rates.md
    python -m src.utils.rates_sheet --check    # exit 1 if the file is out of date (ignoring its timestamp)

Rerun it after any tuning (02*, 04b, 05), any `--record` / `--import_*`, and any new experiment with explicit rates. The text that explains a
regime or a tuning script is in REGIMES / TUNERS below; a script or regime that is not listed there is printed as it is, so a new one is
visible at once, but should be given a line.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "learning_rates.md"

# regime -> (order, what adapts, on which entropy)
REGIMES = {
    "single": (0, "each model alone", "its own entropy (the single-model Tent baseline)"),
    "indep": (1, "both members, never coupled", "each on its OWN entropy (Independent / Vanilla DD; engine mode both_indep)"),
    "joint": (2, "both members", "the entropy of the duo's COMBINED prediction (Joint DD; both_duo)"),
    "large_duo": (3, "the large member only", "the duo's entropy, the small member frozen (Large adapt; large_duo)"),
    "small_duo": (4, "the small member only", "the duo's entropy, the large member frozen (Small adapt; small_duo)"),
}
# tuning script -> what it did
TUNERS = {
    "02c": ("scripts/02c_tune_indep_pair.py", "episodic: each model alone (single) and the independent pair, scored on the 4 held-out shifts (speckle noise, gaussian "
            "blur, spatter, saturate; severity 5; 50k images each, each from source weights); whole-stream accuracy with a late-stream guard; sqrt-2 grid "
            "extended until interior; 3 seeds; temperatures fixed from the frozen models"),
    "02b": ("scripts/02b_tune_joint_lr.py", "episodic: the PAIR on the duo's entropy, from the single rates upward (x2 ladder, sqrt-2 refinement), scored by the static "
            "duo on the 4 held-out shifts; late-stream guard; finalists on 3 seeds"),
    "02d": ("scripts/02d_tune_large_adapt_lr.py", "episodic: the large member alone on the duo's entropy (partner frozen), 4 held-out shifts"),
    "02e": ("scripts/02e_tune_small_adapt_lr.py", "episodic: the small member alone on the duo's entropy (partner frozen), 4 held-out shifts"),
    "02": ("scripts/02_tune_tta.py", "episodic: one model alone (Tent), 4 held-out shifts, seed 0 (the older single-model tuner)"),
    "05": ("scripts/05_select_member_lr.py", "episodic: each member alone for ETA / EATA / SAR from the stored sweep, macro accuracy on the 4 held-out shifts, seed 0"),
    "default": ("pre-tuning default", "nothing tuned: the OPTIM block the duo had before the tuning scripts (never used for results)"),
    "04b": ("scripts/04b_tune_continual_lr.py", "CONTINUAL: the 4 held-out shifts x 4 rounds (each round rotated), 800k images, no reset; stability rule (late gain >= -2 points "
            "per visit against the frozen member); conservative = one rung below the stable frontier, stress = lowest rung that dies completely"),
    "trial": ("by trial", "picked by hand for one experiment, not by a tuning rule"),
}
SOURCE_KEYS = (("tune_joint_lr", "02b"), ("tune_large_adapt", "02d"), ("tune_small_adapt", "02e"), ("02c_tune_indep_pair", "02c"), ("02b_tune_joint", "02b"),
               ("05_select_member_lr", "05"), ("02_tune_tta", "02"), ("pre-tuning default", "default"), ("placeholder", "default"))


def tuner_of(source: str) -> str:
    for key, name in SOURCE_KEYS:
        if key in source:
            return name
    return source


def optim_text(o: dict) -> str:
    if o["METHOD"] == "SGD":
        return f"SGD m{o.get('MOMENTUM', 0):g}"
    return f"{o['METHOD']} b{o.get('BETA', '-')}"


def lr_text(v: float) -> str:
    mant, exp = f"{v:.3e}".split("e")
    return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"


def pct(v) -> str:
    """A percentage, or "L x / S y" for a per-member dict (None where a member has no value)."""
    if isinstance(v, dict):
        parts = [f"{s[0].upper()} {100 * float(x):.1f}" for s, x in v.items() if x is not None]
        return " / ".join(parts)
    return "" if v is None else f"{100 * float(v):.1f}"


# ------------------------------------------------------------------------------------------------------------------ episodic

def episodic_rows(root: Path) -> list[dict]:
    """One row per (duo, regime, TTA method): the large and the small member's rate side by side."""
    rows = []
    for f in sorted(glob.glob(str(root / "cfgs" / "duo_lrs" / "*.yaml"))):
        d = yaml.safe_load(open(f))
        duo, regime = d["duo"], d["regime"]
        blocks = {"(untuned default)": d.get("defaults") or {}, **{t: m for t, m in (d.get("methods") or {}).items()}}
        for tta, sides in blocks.items():
            if not sides:
                continue
            e = {s: sides[s] for s in ("large", "small") if s in sides}
            ref = next(iter(e.values()))
            what = "duo" if "duo_macro_accuracy" in ref else ("member" if "macro_accuracy" in ref else "")
            if what == "member":                                     # one score per member: both are shown
                score = {s: x.get("macro_accuracy") for s, x in e.items()}
                frozen = {s: x.get("frozen_macro_accuracy") for s, x in e.items()}
            else:
                score = ref.get("duo_macro_accuracy")
                frozen = ref.get("frozen_duo_macro_accuracy")
            optims = {optim_text(x["OPTIM"]) for x in e.values()}
            notes = [n for n in ("stale" if any(x.get("stale") for x in e.values()) else "",
                                 f"seeds {','.join(map(str, ref['seeds']))}" if ref.get("seeds") else (f"seed {ref['seed']}" if "seed" in ref else ""),
                                 "frozen BN on batch stats" if ref.get("frozen_bn_batch_stats") else "") if n]
            rows.append({"duo": duo, "regime": regime, "tta": tta, "models": {s: x["model"] for s, x in e.items()},
                         "lr": {s: x["OPTIM"]["LR"] for s, x in e.items()}, "optim": "/".join(sorted(optims)) if len(optims) == 1 else
                         ", ".join(f"{s[0].upper()}: {optim_text(x['OPTIM'])}" for s, x in e.items()),
                         "tuner": "default" if tta == "(untuned default)" else tuner_of(ref.get("source", "")), "score": score, "what": what, "frozen": frozen,
                         "notes": "; ".join(notes) + ("; " if notes and ref.get("note") else "") + (ref.get("note") or "")})
    pending = []
    for f in sorted(glob.glob(str(root / "cfgs" / "duo_lrs" / "*.yaml"))):
        d = yaml.safe_load(open(f))
        pending += [(d["duo"], d["regime"], t) for t in (d.get("pending") or {})]
    for r in rows:
        r["order"] = REGIMES.get(r["regime"], (9,))[0]
    rows.sort(key=lambda r: (r["duo"], r["order"], r["tta"] != "tent", r["tta"] == "(untuned default)", r["tta"]))
    return rows, pending


def render_episodic(rows: list[dict], pending: list[tuple]) -> list[str]:
    out = []
    for duo in sorted({r["duo"] for r in rows}, key=lambda x: (x != "vitb16_rn50v1", x)):
        mine = [r for r in rows if r["duo"] == duo]
        spec = yaml.safe_load(open(REPO / "cfgs" / "duos" / f"{duo}.yaml")) if (REPO / "cfgs" / "duos" / f"{duo}.yaml").is_file() else {}
        m = {s_: (spec.get(s_.upper()) or {}).get("NAME", "?") for s_ in ("large", "small")}
        out += [f"### {duo}" + ("  (the main duo)" if duo == "vitb16_rn50v1" else ""), "",
                f"large = {m['large']}, small = {m['small']}. A `-` is a member with no entry for that method (not tuned in this duo).", "",
                "| setting (regime) | TTA | large LR | small LR | optimizer | tuned by | score (macro acc., %; L / S = per member) | frozen | notes |", "|---|---|---|---|---|---|---|---|---|"]
        for r in mine:
            lr = {s: (lr_text(r["lr"][s]) if s in r["lr"] else "-") for s in ("large", "small")}
            if r["regime"] in ("large_duo", "small_duo"):
                lr = {s: (lr[s] if s == r["regime"].split("_")[0] else "(frozen)") for s in lr}
            out.append(f"| {r['regime']} | {r['tta']} | {lr['large']} | {lr['small']} | {r['optim']} | {r['tuner']} | "
                       f"{pct(r['score'])}{(' (' + r['what'] + ')') if r['what'] == 'duo' and r['score'] is not None else ''} | {pct(r['frozen'])} | {r['notes']} |")
        for d, reg, t in pending:
            if d == duo:
                out.append(f"| {reg} | {t} | pending | pending | | | | | not tuned: asking for it raises NotTuned |")
        out.append("")
    return out


# ------------------------------------------------------------------------------------------------------------------ continual

def continual_section(root: Path) -> list[str]:
    out = []
    for f in sorted(glob.glob(str(root / "out" / "results" / "tune_continual_lr" / "*" / "results.json"))):
        d = json.load(open(f))
        folder = Path(f).parent
        frozen = folder / "runs" / "seed0__frozen.npz"
        fm = {}
        if frozen.is_file():
            z = np.load(frozen)
            fm = {s: 100 * float(z[s].mean()) for s in ("large", "small")}
        out += [f"### {d['duo']}  (continual, 04b, seeds {','.join(map(str, d['seeds']))})", "",
                f"Stream: the 4 held-out shifts x {d['stream']['rounds']} rounds ({4 * d['stream']['rounds'] * d['stream']['images']} images), round r rotated by r. "
                f"Stable = every visit ends within {d['criteria']['delta']:g} points of the frozen member (last {int(100 * d['criteria']['late_frac'])}% of the visit); "
                f"dead = below {100 * d['criteria']['dead']['acc']:g}% for {d['criteria']['dead']['windows']} windows of {d['criteria']['dead']['window']}; "
                f"conservative = {d['criteria']['margin']} rung below the frontier. Warnings: {len(d['warnings'])}.", "",
                "| member | episodic rate (anchor) | ladder | frontier (largest stable) | **conservative** | **stress** (lowest dead) | mean acc. at conservative / at frontier | frozen |",
                "|---|---|---|---|---|---|---|---|"]
        for side in ("large", "small"):
            sel, rows = d["sides"][side], [r for r in d["rows"] if r["side"] == side]
            acc = lambda lr: next((r["acc"] for r in rows if abs(r["lr"] - lr) <= 1e-9 * lr), None) if lr else None
            lad = d["ladders"][side]
            cons, fr = acc(sel["conservative"]), acc(sel["frontier"])
            out.append(f"| {side} | {lr_text(d['episodic'][side])} | {lr_text(lad[0])} .. {lr_text(lad[-1])} ({len(lad)} rungs) | {lr_text(sel['frontier'])} | "
                       f"{lr_text(sel['conservative'])} | {lr_text(sel['stress'])} | {cons:.1f} / {fr:.1f} | {fm.get(side, float('nan')):.1f} |")
        out += ["", "Rungs below the original ladder were added (the ResNet had no stable rung down to 6.25e-5): the ResNet 1.6e-5 .. 4.4e-5, paired with throwaway ViT rungs "
                "2.2e-6 .. 6.2e-6 (a run pairs the i-th rungs of the two ladders).", ""]
    return out or ["(no 04b result yet)", ""]


# ------------------------------------------------------------------------------------------------------------------ experiments

def experiments_section(root: Path) -> list[str]:
    tuned = {}
    for f in glob.glob(str(root / "out" / "results" / "tune_continual_lr" / "*" / "results.json")):
        d = json.load(open(f))
        tuned[d["duo"]] = d
    rows = []
    for f in sorted(glob.glob(str(root / "out" / "results" / "gate_stress" / "*" / "summary*.json"))):
        duo = Path(f).parent.name.split("__")[0]
        for key, v in json.load(open(f)).items():
            m = v["meta"]
            rows.append((duo, m.get("stream", "test") + (f" x{m['rounds']} rounds" if m.get("stream") == "heldout" else ""), key.split("/")[0], m["beta"], m["b_t"], v["lrs"],
                         m.get("seed", 0), Path(f).name))
    # runs that never went through a 04c report (the reset-screening calibration runs S / H / T have no summary file): read their rates from the file names
    named = {(r[0], r[2]) for r in rows}
    unreported = {}
    for f in sorted(glob.glob(str(root / "out" / "results" / "gate_stress" / "*" / "exp-*.npz"))):
        mt = re.match(r"exp-(?P<name>[^_]+)(?P<held>__heldout)?(?:__rot(?P<rot>\d+))?__L(?P<l>[\d.e+-]+)_S(?P<s>[\d.e+-]+)(?:__nnn_bt(?P<bt>\d+)_beta(?P<beta>[\d.]+))?__seed(?P<seed>\d+)\.npz$",
                      Path(f).name)
        duo = Path(f).parent.name.split("__")[0]
        if mt is None or (duo, mt["name"]) in named:
            continue
        key = (duo, "heldout x4 rounds" if mt["held"] else "test", mt["name"], float(mt["beta"] or 0.5), int(mt["bt"] or 256), (float(mt["l"]), float(mt["s"])))
        runs = unreported.setdefault(key, [])
        runs.append((int(mt["rot"] or 0), int(mt["seed"])))
    for (duo, stream, name, beta, b_t, lrs), runs in unreported.items():
        seeds, rots = sorted({s for _, s in runs}), sorted({r for r, _ in runs})
        rows.append((duo, stream, name, beta, b_t, lrs, ",".join(map(str, seeds)) + (f" (rotations {','.join(map(str, rots))})" if stream != "test" else ""),
                     f"{len(runs)} runs, no summary (screening runs, scripts/37b_screen_resets_stress.py)"))
    # the live reset runs of scripts/38_run_reset_live.py: one row per (duo, stream, rates, gate), its conditions and seeds from the file names
    live = {}
    for f in sorted(glob.glob(str(root / "out" / "results" / "reset_live" / "*" / "live-*.npz"))):
        mt = re.match(r"live-(?P<c>[A-Za-z0-9_]+?)(?P<held>__heldout)?(?:__rot(?P<rot>\d+))?__L(?P<l>[\d.e+-]+)_S(?P<s>[\d.e+-]+)__(?P<g>[a-z]+_bt\d+_beta[\d.]+)(?P<cut>__first\d+)?__seed(?P<seed>\d+)\.npz$",
                      Path(f).name)
        if mt is None or mt["cut"]:
            continue
        duo = Path(f).parent.name.split("__")[0]
        key = (duo, "heldout x4 rounds" if mt["held"] else "test", float(mt["l"]), float(mt["s"]), mt["g"])
        d = live.setdefault(key, {"cond": set(), "seed": set(), "rot": set()})
        d["cond"].add(mt["c"]), d["seed"].add(int(mt["seed"])), d["rot"].add(int(mt["rot"] or 0))
    for (duo, stream, lr_l, lr_s, g), d in live.items():
        mg = re.match(r"([a-z]+)_bt(\d+)_beta([\d.]+)", g)
        conds = sorted(d["cond"])
        rows.append((duo, stream, f"live resets ({len(conds)} conditions: {', '.join(conds)})", float(mg[3]), int(mg[2]), (lr_l, lr_s),
                     ",".join(map(str, sorted(d["seed"]))) + (f" (rotations {','.join(map(str, sorted(d['rot'])))})" if stream != "test" else ""),
                     "scripts/38_run_reset_live.py, out/results/reset_live/"))
    out = ["| duo | stream | experiment | gate (beta, b_t) | ViT LR (how found) | ResNet LR (how found) | seed | summary file |", "|---|---|---|---|---|---|---|---|"]

    chosen_live: list = []                      # (side, lr, method name): the frozen configurations of the reset tuning (scripts/40, scripts/42)
    cfg_path = root / "out" / "results" / "reset_tune" / "final_configs.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"methods": []}
    for m in cfg["methods"]:
        chosen_live += [("large", m["lrs"][0], m["name"]), ("small", m["lrs"][1], m["name"])]

    def how(duo, side, lr, live=False):
        if live:
            names = sorted({n for sd, x, n in chosen_live if sd == side and abs(x - lr) <= 1e-6 * lr})
            return f"reset tuning, chosen for {', '.join(names)}" if names else "reset-tuning ladder rung"
        sel = tuned.get(duo, {}).get("sides", {}).get(side)
        if not sel:
            return "explicit"
        if sel["conservative"] and abs(lr - sel["conservative"]) <= 1e-9 * lr:
            return "04b conservative"
        if sel["stress"] and abs(lr - sel["stress"]) <= 1e-9 * lr:
            return "04b stress"
        return "TRIAL: above the 04b stress rate" if sel["stress"] and lr > sel["stress"] else "explicit"

    for duo, stream, name, beta, b_t, lrs, seed, fname in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
        live = name.startswith("live resets")
        out.append(f"| {duo} | {stream} | {name} | {beta:g}, {b_t} | {lr_text(lrs[0])} ({how(duo, 'large', lrs[0], live)}) | {lr_text(lrs[1])} ({how(duo, 'small', lrs[1], live)}) | {seed} | {fname} |")
    out.append("")
    if cfg["methods"]:
        out += ["**Rates tuned for resetting** (stages 2 and 3 of the reset tuning, held-out stream only; the configurations frozen for the final test in `out/results/reset_tune/final_configs.json`, "
                "chosen by `scripts/40_tune_reset.py`, run by `scripts/42_final_test.py`). A run of the table above that is not one of these is a rung of a ladder.", "",
                "| method | condition (rule and threshold) | ViT LR | ResNet LR |", "|---|---|---|---|"]
        out += [f"| {m['label']} | `{m['cond']}` | {lr_text(m['lrs'][0])} | {lr_text(m['lrs'][1])} |" for m in cfg["methods"]]
        out.append("")
    return out + [""]


# ------------------------------------------------------------------------------------------------------------------ the document

def render(root: Path = REPO, stamp: bool = True) -> str:
    rows, pending = episodic_rows(root)
    L = ["# Learning rates: one sheet", "",
         "**This file is generated** (`python -m src.utils.rates_sheet`) from the files that hold the numbers; do not edit it by hand, and rerun it after any tuning, "
         "`--record` / `--import_*`, or experiment with explicit rates. The numbers live in `cfgs/duo_lrs/*.yaml` (episodic), `out/results/tune_continual_lr/*/results.json` "
         "(continual) and the gate-stress runs; scripts read them from there only.", "",
         "All adapting members use Tent unless the TTA column says otherwise, SGD with the momentum shown, batch size 64. A *setting* is the stream a rate was "
         "tuned on (episodic: each shift from source weights; continual: one long stream, no reset) and the *regime* says which entropy drives the members.", "",
         "## 1. Reading the columns", "", "**Regimes**", "", "| regime | what adapts | on which entropy |", "|---|---|---|"]
    L += [f"| {k} | {v[1]} | {v[2]} |" for k, v in sorted(REGIMES.items(), key=lambda kv: kv[1][0])]
    L += ["", "**Tuning methods**", "", "| code | script | what it did |", "|---|---|---|"]
    L += [f"| {k} | `{v[0]}` | {v[1]} |" for k, v in TUNERS.items()]
    unknown = sorted({r["tuner"] for r in rows} - set(TUNERS))
    if unknown:
        L += ["", f"**Not described above (add a line to TUNERS): {', '.join(unknown)}**"]
    L += ["", "## 2. Episodic rates (cfgs/duo_lrs): every duo, regime and TTA method", ""]
    L += render_episodic(rows, pending)
    L += ["## 3. Continual rates (04b, held-out stream x 4 rounds)", ""] + continual_section(root)
    L += ["## 4. Rates actually run in the gate-stress experiments (04c)", "",
          "Experiments A and B use the 04b conservative / stress rates; B1 and B2 use ViT rates above the 04b stress rate, picked by trial because the 04b stress rate "
          "(3.5e-5) does not collapse the ViT on the test stream. S, H and T are the runs the reset rules are screened on offline (`scripts/37b_screen_resets_stress.py`): "
          "S = both members at a rate chosen to collapse them (ViT 1.4e-4, ResNet 5e-4; ViT above its 04b stress rate by trial, ResNet above it by trial), held-out stream, "
          "calibration; H = the 04b conservative rates, held-out stream, healthy time for the false-alarm budget; T = the S rates on the test stream, evaluation only. "
          "Every run here is Vanilla DD (both members on their own entropy, the gate outside the loop).", ""]
    L += experiments_section(root)
    L += ["## 5. Planned, not yet run", "",
          "- **Rates for a resetting scheme (Vanilla DD): not tuned yet.** The 04b rates are the largest rates that never collapse WITHOUT a reset, so they are expected to be too low "
          "once collapses are caught: on the held-out shifts a model that starts every shift from source weights (the idealised reset at every boundary) scores 45.7 for the ResNet "
          "at its episodic 5e-4 against 41.2 at the 04b conservative rate; for the ViT 51.8 against 51.6. They must be tuned WITH the reset rule running, on the held-out stream "
          "x 4 rounds (live runs, whole-stream accuracy with the resets counted), separately per rule family and for the oracle trigger.",
          "- The B1 setting (ViT 1.4e-4 (trial), ResNet 2.2e-5 (04b conservative), test stream, beta 2) stays as the qualitative stress case for the reset plots; it is not a tuned setting.", ""]
    if stamp:
        L += [f"_Generated {datetime.now().astimezone():%Y-%m-%d %H:%M %z} by src/utils/rates_sheet.py._", ""]
    return "\n".join(L)


def _strip(text: str) -> str:
    return re.sub(r"_Generated .*?_\n?", "", text).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/learning_rates.md differs from a fresh render (timestamp ignored)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    fresh = render()
    out = Path(args.out)
    if args.check:
        ok = out.is_file() and _strip(out.read_text()) == _strip(fresh)
        print("up to date" if ok else f"{out} is out of date: run python -m src.utils.rates_sheet")
        return 0 if ok else 1
    out.write_text(fresh)
    print(f"wrote {out} ({fresh.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
