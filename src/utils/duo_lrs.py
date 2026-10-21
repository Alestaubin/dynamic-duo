"""
duo_lrs.py
==========
THE one place the optimizer settings (learning rates) of the duos live: cfgs/duo_lrs/<duo>__<regime>.yaml. cfgs/duos/<duo>.yaml only
names the two models (NAME, NORM); nothing else holds a learning rate.

  indep   both members adapt on their OWN entropy (engine mode `both_indep`: Vanilla DD, the gate outside the loop).
          Tuned one model at a time: scripts/02_tune_tta.py (Tent) and scripts/05_select_member_lr.py (ETA, EATA, SAR) record the
          winner here, in EVERY duo that uses the model (`record_tuned`).
  joint   both members adapt on the entropy of the duo's COMBINED prediction (engine mode `both_duo`: the joint mode, Joint DD).
          Tuned as PAIRS by scripts/02b_tune_joint_lr.py, which only reports: `--import_joint <results.json>` copies its best
          pair in. Until then the method is `pending:` and asking for it is an ERROR, so a joint run can never silently use the
          indep rates. Only Tent has a joint mode (the ported ETA/EATA/SAR run their members independently).

File layout (every OPTIM block complete, exactly as src/tta/tent.py::setup_optimizer reads it, so a run built from it has the same
cache fingerprint as before: the OPTIM block is part of every member's key):

    duo: vitb16_rn50v1
    regime: indep
    engine_mode: both_indep
    defaults:                    # indep only: a member nothing is tuned for runs with this (the former OPTIM of cfgs/duos/<duo>.yaml)
      large: {model, norm, OPTIM, source}
      small: {...}
    methods:                     # per TTA method and per member: what was tuned, with its provenance
      tent:
        large: {model, norm, batch_size, OPTIM, source, tuned_on, macro_accuracy, ...}
        small: {...}
    pending: {tent: "why, and how to fill it"}      # joint: methods being tuned

A member is resolved on its own: its tuned entry for the method if there is one, else (indep only) its default. A `stale:` note on
an entry (tuned on the old input pipeline, say) makes every use of it warn.

    from src.utils import duo_lrs
    duo_lrs.resolve(duo_yaml_dict, "indep", "tent")                      # {"LARGE": OPTIM, "SMALL": OPTIM}
    Context.create("vitb16_rn50v1", tta="tent", regime="joint")          # goes through load_duo; NotTuned while pending

    python -m src.utils.duo_lrs --show vitb16_rn50v1 indep
    python -m src.utils.duo_lrs --check
    python -m src.utils.duo_lrs --init_joint                             # joint stubs for the duos that have none
    python -m src.utils.duo_lrs --import_joint out/results/tune_joint_lr/vitb16_rn50v1__tent_bn/results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LR_DIR = REPO_ROOT / "cfgs" / "duo_lrs"
DUO_DIR = REPO_ROOT / "cfgs" / "duos"
REGIMES = {"indep": "both_indep", "joint": "both_duo"}
SIDES = ("large", "small")

_HEADERS = {
    "indep": """\
# Learning rates of duo `{duo}`, INDEP regime: each member adapts on its OWN entropy (engine mode both_indep; Vanilla DD).
# The one place these numbers live (cfgs/duos/ only names the models). `methods`: what was tuned, per TTA method and per member,
# recorded by scripts/02_tune_tta.py and 05_select_member_lr.py (src/utils/duo_lrs.py::record_tuned); `defaults`: what a member
# runs with when nothing is tuned for the method.
""",
    "joint": """\
# Learning rates of duo `{duo}`, JOINT regime: both members adapt on the entropy of the duo's combined prediction (engine mode
# both_duo; the joint mode of scripts/03 and Joint DD). Tuned as PAIRS by scripts/02b_tune_joint_lr.py, which only reports:
# `python -m src.utils.duo_lrs --import_joint <results.json>` copies the best pair in. A method under `pending:` is being tuned:
# asking for it raises NotTuned (never the indep rates).
""",
}


class NotTuned(LookupError):
    """No learning rates for this (duo, regime, TTA method): none tuned, or still being tuned."""


# ------------------------------------------------------------------------------------------------------ reading

def _root(root) -> Path:
    return Path(root) if root is not None else LR_DIR


def path(duo: str, regime: str, root=None) -> Path:
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}, got {regime!r}")
    return _root(root) / f"{duo}__{regime}.yaml"


def load(duo: str, regime: str, root=None) -> dict:
    p = path(duo, regime, root)
    if not p.is_file():
        raise FileNotFoundError(f"{p} does not exist: the learning rates of duo {duo!r} ({regime}) live there. Create it with "
                                f"`python -m src.utils.duo_lrs {'--init_indep ' + duo if regime == 'indep' else '--init_joint'}`")
    doc = yaml.safe_load(p.read_text())
    if doc.get("duo") != duo or doc.get("regime") != regime:
        raise ValueError(f"{p} is for duo {doc.get('duo')!r}, regime {doc.get('regime')!r}")
    for key in ("defaults", "methods", "pending"):
        doc.setdefault(key, {})
    return doc


def _check_member(entry: dict, spec: dict, where: str, side: str) -> None:
    if entry["model"] != spec["NAME"]:
        raise ValueError(f"{where}: tuned for {entry['model']!r}, but the duo's {side} member is {spec['NAME']!r}")
    if entry["norm"] != spec["NORM"]:
        raise ValueError(f"{where} was tuned adapting {entry['norm']} layers, but the duo adapts {spec['NORM']} for "
                         f"{spec['NAME']}: retune (scripts/02_tune_tta.py) or fix NORM")


def resolve(duo_spec: dict, regime: str, tta: str, root=None, batch_size: int | None = None) -> dict[str, dict]:
    """{"LARGE": OPTIM, "SMALL": OPTIM} for `duo_spec` (a cfgs/duos yaml: NAME, LARGE{NAME, NORM}, SMALL{...}). Each member on its own:
    its tuned entry for `tta`, else (indep) its default. Raises NotTuned for a joint method that is pending or absent, and for an
    indep member with neither."""
    name = duo_spec["NAME"]
    doc = load(name, regime, root)
    where = f"{path(name, regime, root)}"
    methods = doc["methods"].get(tta, {})
    if regime == "joint" and not all(s in methods for s in SIDES):
        why = doc["pending"].get(tta)
        raise NotTuned(f"{name} / joint / {tta}: " + (f"not tuned yet ({why}). Pass explicit rates or wait; "
                       f"the indep rates are NOT a substitute" if why else
                       f"no tuned learning rate pair in {where} (tuned there: {sorted(doc['methods']) or 'none'})"))
    out = {}
    for side in SIDES:
        spec = duo_spec[side.upper()]
        entry = methods.get(side)
        if entry is not None:
            _check_member(entry, spec, f"{where}: {tta}/{side}", side)
            if entry.get("stale"):
                warnings.warn(f"{where}: {tta}/{side} is stale: {entry['stale']}")
            if batch_size is not None and entry.get("batch_size") not in (None, batch_size):
                warnings.warn(f"{where}: {tta}/{side} was tuned at batch size {entry['batch_size']}, the protocol uses {batch_size}")
        elif regime == "indep" and side in doc["defaults"]:
            entry = doc["defaults"][side]
            _check_member(entry, spec, f"{where}: defaults/{side}", side)
        else:
            raise NotTuned(f"{name} / {regime} / {tta} / {side}: no tuned learning rate and no default in {where}")
        out[side.upper()] = copy.deepcopy(entry["OPTIM"])
    return out


# ------------------------------------------------------------------------------------------------------ writing

def _dump(doc: dict) -> str:
    return _HEADERS[doc["regime"]].format(duo=doc["duo"]) + yaml.safe_dump(doc, sort_keys=False, width=120)


def _write(doc: dict, root=None) -> Path:
    for tta, m in doc["methods"].items():
        doc["methods"][tta] = {s: m[s] for s in SIDES if s in m}                   # large before small
    if not doc["pending"]:
        doc.pop("pending")
    if not doc["defaults"]:
        doc.pop("defaults")
    p = path(doc["duo"], doc["regime"], root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_dump(doc))
    return p


def _duo_spec(duo: str, duos_dir=None) -> dict:
    return yaml.safe_load((Path(duos_dir or DUO_DIR) / f"{duo}.yaml").read_text())


def _generic_default(model: str) -> dict:
    """SGD, momentum 0.9, at the centre of the model family's grid (src/experiments/grids.py)."""
    from src.experiments.grids import CENTRE, family, sgd_optim
    try:
        return sgd_optim(CENTRE[family(model)])
    except ValueError as e:
        raise ValueError(f"{e}; create the file with explicit defaults: python -m src.utils.duo_lrs --init_indep <duo> "
                         f"--lr_large X --lr_small Y") from None


def init_indep(duo: str, root=None, duos_dir=None, defaults: dict | None = None, source: str | None = None) -> Path | None:
    """The indep file of `duo` with its defaults (`defaults`: {"large": OPTIM, "small": OPTIM}, else SGD at the grid centre) and
    no tuned methods; None if it exists."""
    if path(duo, "indep", root).exists():
        return None
    spec = _duo_spec(duo, duos_dir)
    doc = {"duo": duo, "regime": "indep", "engine_mode": REGIMES["indep"], "defaults": {}, "methods": {}, "pending": {}}
    for side in SIDES:
        m = spec[side.upper()]
        optim = (defaults or {}).get(side) or _generic_default(m["NAME"])
        doc["defaults"][side] = {"model": m["NAME"], "norm": m["NORM"], "OPTIM": dict(optim),
                                 "source": source or ("explicit default" if (defaults or {}).get(side) else
                                                      "generic default: SGD, momentum 0.9, at the centre of the model family's grid")}
    return _write(doc, root)


def init_joint(duo: str, root=None) -> Path | None:
    """A joint file with no methods; None if the file exists (it may hold imported results)."""
    if path(duo, "joint", root).exists():
        return None
    return _write({"duo": duo, "regime": "joint", "engine_mode": REGIMES["joint"], "defaults": {}, "methods": {}, "pending": {}}, root)


def set_pending(duo: str, tta: str, reason: str, root=None) -> Path:
    doc = load(duo, "joint", root)
    if tta in doc["methods"]:
        raise ValueError(f"{duo} joint / {tta} is already tuned; not marking it pending")
    doc["pending"][tta] = reason
    return _write(doc, root)


def entry_from_record(record: dict) -> dict:
    """The entry of a tuning record (src/experiments/tuning.py::tuned_record, member_selection.py::tuned_record)."""
    e = {"model": record["model"], "norm": record["NORM"], "batch_size": record.get("batch_size"), "OPTIM": dict(record["OPTIM"]),
         "source": record.get("written_by", "unknown")}
    tuned_on = record.get("tuned_on") or {}
    if tuned_on.get("streams"):
        e["tuned_on"] = [s.replace("__", "/") for s in tuned_on["streams"]]
    if "seed" in tuned_on:
        e["seed"] = tuned_on["seed"]
    for key in ("macro_accuracy", "frozen_macro_accuracy", "worst_stream_accuracy", "frozen_bn_batch_stats", "lrs_tried", "note", "stale"):
        if key in record:
            e[key] = record[key]
    return {k: v for k, v in e.items() if v is not None}


def record_tuned(model: str, tta: str, record: dict, root=None, duos_dir=None, duos: list[str] | None = None,
                 replaced: list[dict] | None = None) -> list[Path]:
    """Record the winner of a tuning run in the indep file of EVERY duo that uses `model` (on the side(s) where it appears; `duos`
    restricts which). Returns the files written; [] if no duo uses the model. An entry that already held other optimizer settings
    is replaced, and described in `replaced` if given: {"duo", "side", "old": OPTIM, "new": OPTIM, "note": its note or None}."""
    duos_dir = Path(duos_dir or DUO_DIR)
    entry = entry_from_record(record)
    written = []
    for f in sorted(duos_dir.glob("*.yaml")):
        spec = yaml.safe_load(f.read_text())
        sides = [s for s in SIDES if spec[s.upper()]["NAME"] == model]
        if not sides or (duos is not None and spec["NAME"] not in duos):
            continue
        for s in sides:
            if spec[s.upper()]["NORM"] != record["NORM"]:
                raise ValueError(f"{f.name}: the {s} member adapts {spec[s.upper()]['NORM']} layers, the tuning record {record['NORM']}")
        init_indep(spec["NAME"], root, duos_dir)
        doc = load(spec["NAME"], "indep", root)
        for s in sides:
            old = doc["methods"].get(tta, {}).get(s)
            if old is not None and old["OPTIM"] != entry["OPTIM"] and replaced is not None:
                replaced.append({"duo": spec["NAME"], "side": s, "old": old["OPTIM"], "new": entry["OPTIM"], "note": old.get("note")})
            doc["methods"].setdefault(tta, {})[s] = copy.deepcopy(entry)
        written.append(_write(doc, root))
    return written


def import_joint(results_json, root=None, duos_dir=None) -> Path:
    """Copy the best pair of a scripts/02b_tune_joint_lr.py run (its results.json) into the duo's joint file."""
    r = json.loads(Path(results_json).read_text())
    if r.get("num_samples"):
        raise ValueError(f"{results_json} is a capped run (--num_samples {r['num_samples']}); only full runs are imported")
    duo, tta = r["duo"], r["tta"]
    spec = _duo_spec(duo, duos_dir)
    init_joint(duo, root)
    doc = load(duo, "joint", root)
    entry = {}
    for side in SIDES:
        entry[side] = {"model": spec[side.upper()]["NAME"], "norm": spec[side.upper()]["NORM"],
                       "OPTIM": {**r["optim"][side], "LR": float(r["best"][f"lr_{side}"])},
                       "source": str(results_json), "tuned_on": [s.replace("__", "/") for s in r["streams"]],
                       "duo_macro_accuracy": round(float(r["best"]["duo"]), 5),
                       "worst_stream_accuracy": round(float(r["best"]["worst"]), 5)}
    entry["temperatures"] = dict(r["temperatures"])                          # the static duo the grid was scored with
    entry["grid"] = {"lrs_large": r["lrs_large"], "lrs_small": r["lrs_small"], "bn": r.get("bn"), "seeds": r["seeds"]}
    doc["methods"][tta] = entry
    doc["pending"].pop(tta, None)
    return _write(doc, root)


# ------------------------------------------------------------------------------------------------------ checking

def check(duo: str, root=None, duos_dir=None) -> list[str]:
    """Problems of the duo's files (empty = fine)."""
    problems = []
    spec = _duo_spec(duo, duos_dir)
    for side in SIDES:
        if "OPTIM" in spec[side.upper()]:
            problems.append(f"cfgs/duos/{duo}.yaml: {side.upper()}.OPTIM is ignored; learning rates live in cfgs/duo_lrs/")
    for regime in REGIMES:
        try:
            doc = load(duo, regime, root)
        except FileNotFoundError as e:
            problems.append(str(e))
            continue
        for side in SIDES:
            if regime == "indep" and side not in doc["defaults"]:
                problems.append(f"{duo}/indep: no default for the {side} member")
            for where, e in [("defaults", doc["defaults"].get(side))] + [(f"{tta}", m.get(side)) for tta, m in doc["methods"].items()]:
                if e is None:
                    continue
                try:
                    _check_member(e, spec[side.upper()], f"{duo}/{regime}/{where}/{side}", side)
                except ValueError as ex:
                    problems.append(str(ex))
                if not {"METHOD", "LR"} <= set(e["OPTIM"]):
                    problems.append(f"{duo}/{regime}/{where}/{side}: incomplete OPTIM block")
        if regime == "joint":
            for tta, m in doc["methods"].items():
                if not all(s in m for s in SIDES):
                    problems.append(f"{duo}/joint/{tta}: a joint entry needs both members (a pair)")
    return problems


def describe(duo: str, regime: str, root=None) -> str:
    doc = load(duo, regime, root)
    lines = [f"{duo} / {regime} ({REGIMES[regime]})"]
    for side in SIDES:
        d = doc["defaults"].get(side)
        if d:
            lines.append(f"  default {side:5s} {d['model']} {d['OPTIM']['METHOD']} lr {d['OPTIM']['LR']:g}   ({d['source']})")
    for tta, m in sorted(doc["methods"].items()):
        bits = [f"{s} {m[s]['model']} lr {m[s]['OPTIM']['LR']:g}" for s in SIDES if s in m]
        lines.append(f"  {tta:5s} " + "   ".join(bits) + (f"   ({m['large']['source']})" if "large" in m else ""))
    for tta, why in sorted(doc["pending"].items()):
        lines.append(f"  {tta:5s} PENDING: {why}")
    return "\n".join(lines)


def main(argv=None) -> int:
    from src.utils.config import list_duos

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init_indep", nargs="+", metavar="DUO", help="create the indep file of these duos (defaults: --lr_large/--lr_small, "
                                                                  "else SGD at the centre of the model family's grid)")
    ap.add_argument("--lr_large", type=float)
    ap.add_argument("--lr_small", type=float)
    ap.add_argument("--init_joint", action="store_true", help="create the joint file of every duo that has none")
    ap.add_argument("--pending", nargs=3, metavar=("DUO", "TTA", "REASON"), help="mark a joint method as being tuned")
    ap.add_argument("--import_joint", metavar="RESULTS_JSON", help="copy the best pair of a 02b run into the joint file")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--show", nargs=2, metavar=("DUO", "REGIME"))
    ap.add_argument("--duo", nargs="+", default=None, help="restrict --init_joint / --check to these duos")
    a = ap.parse_args(argv)
    duos, rc = a.duo or list_duos(), 0
    if a.init_indep:
        from src.experiments.grids import sgd_optim
        for d in a.init_indep:
            dflt = {s: sgd_optim(v) for s, v in (("large", a.lr_large), ("small", a.lr_small)) if v}
            p = init_indep(d, defaults=dflt or None)
            print(f"wrote {p}" if p else f"{path(d, 'indep')} exists, kept")
    if a.init_joint:
        for d in duos:
            p = init_joint(d)
            print(f"wrote {p}" if p else f"{path(d, 'joint')} exists, kept")
    if a.pending:
        set_pending(*a.pending)
    if a.import_joint:
        print(f"wrote {import_joint(a.import_joint)}")
    if a.check:
        for d in duos:
            for p in check(d):
                print(f"PROBLEM: {p}")
                rc = 1
        print("all duo learning-rate files are consistent" if rc == 0 else "")
    if a.show:
        print(describe(*a.show))
    return rc


# ---------------------------------------------------------------------------------------------------- self-test

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())

    import tempfile

    optim = lambda lr, m=0.9: {"METHOD": "SGD", "STEPS": 1, "LR": lr, "MOMENTUM": m, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}  # noqa: E731
    adam = lambda lr: {"METHOD": "Adam", "STEPS": 1, "LR": lr, "BETA": 0.9, "WD": 0.0}                                                # noqa: E731
    rec = lambda model, tta, norm, lr: {"model": model, "tta": tta, "NORM": norm, "batch_size": 64, "OPTIM": optim(lr),           # noqa: E731
                                        "tuned_on": {"streams": ["a__s5", "b__s5"], "seed": 0}, "macro_accuracy": 0.5,
                                        "written_by": "scripts/02_tune_tta.py"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        duos, root = tmp / "duos", tmp / "lrs"
        duos.mkdir()
        mk = lambda name, large, small: (duos / f"{name}.yaml").write_text(yaml.safe_dump(                                          # noqa: E731
            {"NAME": name, "LARGE": {"NAME": large[0], "NORM": large[1]}, "SMALL": {"NAME": small[0], "NORM": small[1]}}))
        mk("main", ("vit", "LN"), ("rn", "BN"))
        mk("other", ("vit_aug", "LN"), ("rn", "BN"))                      # shares the small model with `main`
        spec = _duo_spec("main", duos)

        # no file, no learning rates: the error says where to create it
        try:
            resolve(spec, "indep", "tent", root)
            raise AssertionError("a duo without a file has no learning rates")
        except FileNotFoundError as e:
            assert "--init_indep main" in str(e)
        init_indep("main", root, duos, defaults={"large": adam(1e-4), "small": optim(2.5e-4)}, source="the pre-tuning default")
        assert init_indep("main", root, duos) is None, "an existing file is never overwritten"
        assert resolve(spec, "indep", "tent", root) == {"LARGE": adam(1e-4), "SMALL": optim(2.5e-4)}, "untuned: the defaults"
        try:
            init_indep("other", root, duos)                               # no explicit defaults and a model the grids do not know
            raise AssertionError("an unknown model has no generic default")
        except ValueError as ex:
            assert "no learning-rate grid" in str(ex)
        assert _generic_default("vit_b_16") == optim(1e-3) and _generic_default("resnet50_v1") == optim(2.5e-4), "SGD at the grid centre"
        init_indep("other", root, duos, defaults={"large": adam(1e-4), "small": optim(2.5e-4)})

        # record_tuned: the winner goes into every duo that uses the model, on its side, for that method only
        for_vit = record_tuned("vit", "tent", rec("vit", "tent", "LN", 5e-5), root, duos)
        assert [p.name for p in for_vit] == ["main__indep.yaml"]
        both = record_tuned("rn", "tent", rec("rn", "tent", "BN", 2.5e-3), root, duos)
        assert sorted(p.name for p in both) == ["main__indep.yaml", "other__indep.yaml"], "a model shared by two duos is recorded in both"
        assert resolve(spec, "indep", "tent", root) == {"LARGE": optim(5e-5), "SMALL": optim(2.5e-3)}
        assert resolve(spec, "indep", "eta", root) == {"LARGE": adam(1e-4), "SMALL": optim(2.5e-4)}, "per TTA method: eta keeps the defaults"
        assert resolve(_duo_spec("other", duos), "indep", "tent", root)["SMALL"] == optim(2.5e-3), "the shared model's rate reached the other duo"
        assert resolve(_duo_spec("other", duos), "indep", "tent", root)["LARGE"] != optim(5e-5), "a model of another duo is untouched"
        gone = []
        record_tuned("vit", "tent", {**rec("vit", "tent", "LN", 1e-4), "note": "x"}, root, duos, replaced=gone)
        assert [(g["duo"], g["side"], g["old"]["LR"], g["new"]["LR"]) for g in gone] == [("main", "large", 5e-5, 1e-4)], gone
        gone.clear()
        record_tuned("vit", "tent", {**rec("vit", "tent", "LN", 1e-4), "note": "x"}, root, duos, replaced=gone)
        assert gone == [], "recording the same settings again replaces nothing"
        record_tuned("vit", "tent", rec("vit", "tent", "LN", 5e-5), root, duos)                 # back to the value the checks below expect
        e = load("main", "indep", root)["methods"]["tent"]["large"]
        assert e["source"] == "scripts/02_tune_tta.py" and e["tuned_on"] == ["a/s5", "b/s5"] and e["macro_accuracy"] == 0.5 and e["seed"] == 0
        assert list(load("main", "indep", root)["methods"]["tent"]) == ["large", "small"]
        assert record_tuned("nobody", "tent", rec("nobody", "tent", "LN", 1e-3), root, duos) == [], "no duo uses the model: nothing written"
        try:
            record_tuned("vit", "tent", rec("vit", "tent", "BN", 1e-3), root, duos)
            raise AssertionError("a record for another NORM must be refused")
        except ValueError as ex:
            assert "adapts LN" in str(ex)
        # a half-tuned method resolves each member on its own
        record_tuned("vit", "eta", rec("vit", "eta", "LN", 5e-4), root, duos)
        assert resolve(spec, "indep", "eta", root) == {"LARGE": optim(5e-4), "SMALL": optim(2.5e-4)}
        # a file whose model is not the duo's is refused, with the NORM wording the old tuned files used
        wrong = copy.deepcopy(spec)
        wrong["SMALL"]["NORM"] = "LN"
        try:
            resolve(wrong, "indep", "tent", root)
            raise AssertionError("another NORM must be refused")
        except ValueError as ex:
            assert "tuned adapting BN layers" in str(ex)
        # stale entries warn on every use; a batch size other than the protocol's warns too
        doc = load("main", "indep", root)
        doc["methods"]["tent"]["large"]["stale"] = "old pipeline"
        doc["methods"]["tent"]["small"]["batch_size"] = 128
        _write(doc, root)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolve(spec, "indep", "tent", root, batch_size=64)
        assert any("stale: old pipeline" in str(x.message) for x in w) and any("batch size 128" in str(x.message) for x in w)

        # joint: absent / pending are errors that never fall back to the indep rates; an imported pair replaces the pending note
        init_joint("main", root)
        assert init_joint("main", root) is None
        try:
            resolve(spec, "joint", "tent", root)
            raise AssertionError("no joint pair")
        except NotTuned as ex:
            assert "no tuned learning rate pair" in str(ex)
        set_pending("main", "tent", "02b running", root)
        try:
            resolve(spec, "joint", "tent", root)
            raise AssertionError("pending")
        except NotTuned as ex:
            assert "02b running" in str(ex) and "NOT a substitute" in str(ex)
        res = tmp / "results.json"
        res.write_text(json.dumps({"duo": "main", "tta": "tent", "bn": True, "seeds": [0], "num_samples": None, "streams": ["a__s5", "b__s5"],
                                   "temperatures": {"T_l": 0.8, "T_s": 1.0, "checkpoint": "x"},
                                   "optim": {"large": optim(5e-5), "small": optim(2.5e-3)}, "lrs_large": [5e-5, 1e-4], "lrs_small": [2.5e-3, 5e-3],
                                   "best": {"lr_large": 1e-4, "lr_small": 5e-3, "duo": 0.55, "worst": 0.4}}))
        import_joint(res, root, duos)
        assert resolve(spec, "joint", "tent", root) == {"LARGE": optim(1e-4), "SMALL": optim(5e-3)}
        assert "pending" not in load("main", "joint", root) or load("main", "joint", root)["pending"] == {}
        assert resolve(spec, "indep", "tent", root)["LARGE"] == optim(5e-5), "the regimes do not mix"
        res.write_text(json.dumps(json.loads(res.read_text()) | {"num_samples": 2000}))
        try:
            import_joint(res, root, duos)
            raise AssertionError("a capped run must not be imported")
        except ValueError:
            pass
        assert check("main", root, duos) == [] and "PENDING" not in describe("main", "joint", root)
        assert "default large" in describe("main", "indep", root)
        # check() finds an OPTIM block left in a duo file, and a missing joint file
        (duos / "main.yaml").write_text(yaml.safe_dump({"NAME": "main", "LARGE": {"NAME": "vit", "NORM": "LN", "OPTIM": adam(1)},
                                                        "SMALL": {"NAME": "rn", "NORM": "BN"}}))
        assert any("LARGE.OPTIM is ignored" in p for p in check("main", root, duos))
        assert any("--init_joint" in p for p in check("other", root, duos))
    print("duo_lrs self-test passed")
