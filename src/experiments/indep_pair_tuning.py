"""
indep_pair_tuning.py
====================
Tune the Tent learning rates of the INDIVIDUAL models and of the INDEPENDENT duo (each member adapts on its OWN entropy, engine mode
`both_indep`) from per-member logits in the logit store (logit_store.py), the way scripts/02b_tune_joint_lr.py tunes the Joint pair:

  single   each model ALONE: its own accuracy at each learning rate, macro over the 4 held-out streams. The optimum must be INTERIOR to
           the grid (a lower and a higher rate both worse) and is then kept for every test.
  indep    the PAIR (lr_large, lr_small) of the independent duo: the accuracy of the static duo (z_L / T_L + z_S / T_S) / 2 with the
           Fixed TS fitted on the FROZEN models' logits (adaptation_modes.temperature_spec), macro over the held-out streams.

Criterion (the author's decisions of 2026-10-22). The score is the whole-stream online accuracy, the metric the paper reports (the
members' logits are the PRE-update ones). Next to it every candidate carries its LATE accuracy (the last 20% of each stream, whole
batches from the end, as joint_lr_tuning.late_accuracy); a candidate whose late accuracy is below the FROZEN model's (the frozen duo's
for a pair) is ruled out: an adaptation that ends worse than not adapting is drift, however good its mean. Optimizer: SGD, momentum
0.9, the rate only; the grid is log-spaced (sqrt 2), the frozen model is on every table, and the grid is extended until the optimum
is interior. Seeds (the stream order) are averaged where a candidate has several; with `min_seeds` only candidates run with at least
that many seeds compete.

The members never see each other, so a run at (a, b) is the large member at a plus the small member at b: N runs along the diagonal
(the i-th rate of each axis, scripts/04 `--lr_large a_i --lr_small b_i`) store N large and N small members and all N x N pairs are
composed offline (logit_store.check_aligned says they saw the same samples), with no model. `missing` lists what a grid still needs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from src.experiments import joint_lr_tuning as JL
from src.experiments.grids import sgd_optim
from src.experiments.logit_runs import run_member_specs, with_optim
from src.experiments.logit_store import LogitStore, MemberSpec, check_aligned
from src.experiments.runner import Context
from src.experiments.specs import RunSpec

NAN = float("nan")


# ------------------------------------------------------------------------------------------------------------ members

def member_specs(ctx: Context, tta: str, tta_kwargs: dict | None, seed: int, num_samples: int | None,
                 lrs_large: Sequence[float], lrs_small: Sequence[float], momentum: float = 0.9
                 ) -> tuple[dict[float, MemberSpec], dict[float, MemberSpec]]:
    """The stored-member identity of every rate of each axis (SGD at `momentum`; the rest of OPTIM as scripts/04 writes it)."""
    spec = RunSpec(ctx.duo["NAME"], tta, tta_kwargs or {}, "both_indep", "episodic", seed=seed, num_samples=num_samples)
    large = {lr: run_member_specs(spec, with_optim(ctx, large=sgd_optim(lr, momentum)))[0] for lr in lrs_large}
    small = {lr: run_member_specs(spec, with_optim(ctx, small=sgd_optim(lr, momentum)))[1] for lr in lrs_small}
    return large, small


def frozen_specs(ctx: Context, seed: int, num_samples: int | None) -> tuple[MemberSpec, MemberSpec]:
    """The frozen members (scripts/04 --mode no_adapt): the baseline of every table and the floor of the late-stream guard."""
    return run_member_specs(RunSpec(ctx.duo["NAME"], "tent", {}, "no_adapt", "episodic", seed=seed, num_samples=num_samples), ctx)


def missing(store: LogitStore, large: Mapping[float, MemberSpec], small: Mapping[float, MemberSpec], stems: Sequence[str],
            frozen: tuple[MemberSpec, MemberSpec] | None = None) -> dict[str, list[str]]:
    """{"large 0.0002": [stems not stored], ...} for the members the grid needs (and the frozen ones) and the store lacks."""
    out = {}
    wanted = [("large", f"{lr:g}", m) for lr, m in large.items()] + [("small", f"{lr:g}", m) for lr, m in small.items()]
    if frozen:
        wanted += [("large", "frozen", frozen[0]), ("small", "frozen", frozen[1])]
    for side, lr, m in wanted:
        if gaps := [s for s in stems if not store.has(m, s)]:
            out[f"{side} {lr}"] = gaps
    return out


# ------------------------------------------------------------------------------------------------------------ scoring

def late_correct(correct: torch.Tensor, batch_sizes: Sequence[int], frac: float = JL.LATE_FRAC) -> float:
    """Accuracy over the last `frac` of the samples of one stream, in whole batches from the end (a batch is added while the samples so
    far are fewer than `frac` of the stream): the same window as joint_lr_tuning.late_accuracy."""
    sizes = list(batch_sizes)
    need, got, i = frac * sum(sizes), 0, len(sizes)
    while i > 0 and got < need:
        i -= 1
        got += sizes[i]
    return float(correct[len(correct) - got:].float().mean())


def _acc(z: torch.Tensor, y: torch.Tensor, bs: Sequence[int]) -> tuple[float, float]:
    c = z.argmax(1) == y
    return float(c.float().mean()), late_correct(c, bs)


@torch.no_grad()
def score_seed(store: LogitStore, large: Mapping[float, MemberSpec], small: Mapping[float, MemberSpec], stems: Sequence[str],
               T_l: float, T_s: float, frozen: tuple[MemberSpec, MemberSpec] | None = None) -> dict:
    """One seed's scores, per stream: {"large": {lr: {"acc": {stem: a}, "late": {stem: l}}}, "small": {...}, "pairs": {(a, b): {"acc", "late",
    "nll"}}, "frozen": {"large": {...}, "small": {...}, "duo": {...}}}; a pair or a member with non-finite logits is {"diverged": True}.
    Each stream is read once for all members; the duo of a pair is the static duo."""
    out = {"large": {a: {"acc": {}, "late": {}} for a in large}, "small": {b: {"acc": {}, "late": {}} for b in small},
           "pairs": {(a, b): {"acc": {}, "late": {}, "nll": {}} for a in large for b in small}, "frozen": None}
    if frozen:
        out["frozen"] = {"large": {"acc": {}, "late": {}}, "small": {"acc": {}, "late": {}}, "duo": {"acc": {}, "late": {}, "nll": {}}}
    bad_l, bad_s, bad_pair = set(), set(), set()
    for stem in stems:
        zl = {a: store.read(m, stem) for a, m in large.items()}
        zs = {b: store.read(m, stem) for b, m in small.items()}
        fl, fs = (store.read(frozen[0], stem), store.read(frozen[1], stem)) if frozen else (None, None)
        ref = next(iter(zl.values()))
        y, bs = ref["labels"].long(), ref["batch_sizes"]
        for a, d in zl.items():
            check_aligned(ref, d, stem)
            if not torch.isfinite(d["z"]).all():
                bad_l.add(a)
                continue
            out["large"][a]["acc"][stem], out["large"][a]["late"][stem] = _acc(d["z"], y, bs)
        for b, d in zs.items():
            check_aligned(ref, d, stem)
            if not torch.isfinite(d["z"]).all():
                bad_s.add(b)
                continue
            out["small"][b]["acc"][stem], out["small"][b]["late"][stem] = _acc(d["z"], y, bs)
        for (a, b), per in out["pairs"].items():
            if a in bad_l or b in bad_s:
                bad_pair.add((a, b))
                continue
            z = zl[a]["z"].float() / T_l + zs[b]["z"].float() / T_s
            per["acc"][stem], per["late"][stem] = _acc(z, y, bs)
            per["nll"][stem] = float(torch.nn.functional.cross_entropy(z * 0.5, y))
        if frozen:
            check_aligned(ref, fl, stem), check_aligned(ref, fs, stem)
            out["frozen"]["large"]["acc"][stem], out["frozen"]["large"]["late"][stem] = _acc(fl["z"], y, bs)
            out["frozen"]["small"]["acc"][stem], out["frozen"]["small"]["late"][stem] = _acc(fs["z"], y, bs)
            z = fl["z"].float() / T_l + fs["z"].float() / T_s
            out["frozen"]["duo"]["acc"][stem], out["frozen"]["duo"]["late"][stem] = _acc(z, y, bs)
            out["frozen"]["duo"]["nll"][stem] = float(torch.nn.functional.cross_entropy(z * 0.5, y))
        del zl, zs
    for a in bad_l:
        out["large"][a] = {"diverged": True}
    for b in bad_s:
        out["small"][b] = {"diverged": True}
    for k in bad_pair:
        out["pairs"][k] = {"diverged": True}
    return out


def _macro(per_stream: Mapping[str, float]) -> float:
    return float(np.mean(list(per_stream.values())))


def aggregate(per_seed: Mapping[int, dict], min_seeds: int = 1) -> dict:
    """Mean over the seeds of `score_seed` results. Returns {"pairs": [points in joint_lr_tuning's format, plus n_seeds, duo_std,
    late_duo], "large": {lr: {...}}, "small": {lr: {...}}, "frozen": {...}, "seeds": [...]}. A candidate is only kept when it has at
    least `min_seeds` seeds; a candidate diverged in any seed is diverged."""
    seeds = sorted(per_seed)
    out = {"seeds": seeds, "pairs": [], "large": {}, "small": {}, "frozen": None}

    for side in ("large", "small"):
        for lr in sorted({lr for s in seeds for lr in per_seed[s][side]}):
            rows = [per_seed[s][side][lr] for s in seeds if lr in per_seed[s][side]]
            if len(rows) < min_seeds:
                continue
            if any(r.get("diverged") for r in rows):
                out[side][lr] = {"diverged": True, "acc": NAN, "late": NAN, "worst": NAN, "n_seeds": len(rows)}
                continue
            accs = [_macro(r["acc"]) for r in rows]
            streams = {k: float(np.mean([r["acc"][k] for r in rows])) for k in rows[0]["acc"]}
            out[side][lr] = {"diverged": False, "acc": float(np.mean(accs)), "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else NAN,
                             "late": float(np.mean([_macro(r["late"]) for r in rows])), "worst": min(streams.values()),
                             "worst_stream": min(streams, key=streams.get), "streams": streams, "n_seeds": len(rows)}
    pairs = sorted({k for s in seeds for k in per_seed[s]["pairs"]})
    for (a, b) in pairs:
        rows = [per_seed[s]["pairs"][a, b] for s in seeds if (a, b) in per_seed[s]["pairs"]]
        if len(rows) < min_seeds:
            continue
        if any(r.get("diverged") for r in rows):
            out["pairs"].append({"lr_large": a, "lr_small": b, "diverged": True, "duo": NAN, "worst": NAN, "worst_stream": "", "large": NAN,
                                 "small": NAN, "duo_nll": NAN, "late_duo": NAN, "streams": {}, "n_seeds": len(rows)})
            continue
        accs = [_macro(r["acc"]) for r in rows]
        streams = {k: float(np.mean([r["acc"][k] for r in rows])) for k in rows[0]["acc"]}
        la, sm = out["large"].get(a, {}), out["small"].get(b, {})
        out["pairs"].append({"lr_large": a, "lr_small": b, "diverged": False, "duo": float(np.mean(accs)),
                             "duo_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else NAN,
                             "worst": min(streams.values()), "worst_stream": min(streams, key=streams.get), "large": la.get("acc", NAN),
                             "small": sm.get("acc", NAN), "duo_nll": float(np.mean([_macro(r["nll"]) for r in rows])),
                             "late_duo": float(np.mean([_macro(r["late"]) for r in rows])), "streams": streams, "n_seeds": len(rows)})
    fz = [per_seed[s]["frozen"] for s in seeds if per_seed[s].get("frozen")]
    if fz:
        out["frozen"] = {k: {"acc": float(np.mean([_macro(f[k]["acc"]) for f in fz])), "late": float(np.mean([_macro(f[k]["late"]) for f in fz]))}
                         for k in ("large", "small", "duo")}
        out["frozen"]["duo"]["nll"] = float(np.mean([_macro(f["duo"]["nll"]) for f in fz]))
    return out


# ------------------------------------------------------------------------------------------------------------ choosing

def best_member(table: Mapping[float, dict], late_floor: float | None) -> tuple[float | None, dict]:
    """(the best rate of one model alone, a description): highest accuracy among the finite candidates, ties to the smaller rate; with
    `late_floor` those ending below it are ruled out (dropped if that would leave nothing). `interior`: a lower AND a higher rate were
    tried and both are worse (a neighbour that diverged or was ruled out by the guard counts as worse); `edge` says why not otherwise."""
    finite = {lr: r for lr, r in table.items() if not r["diverged"]}
    if not finite:
        return None, {"interior": False, "excluded": [], "guard_dropped": False}
    excluded = [lr for lr, r in finite.items() if late_floor is not None and r["late"] < late_floor]
    pool = {lr: r for lr, r in finite.items() if lr not in excluded} or finite
    best = sorted(pool, key=lambda lr: (-round(pool[lr]["acc"], 12), lr))[0]
    lrs = sorted(table)
    i = lrs.index(best)
    def worse(lr: float) -> bool:           # a neighbour that diverged, was ruled out by the guard, or scores lower brackets the optimum
        return table[lr]["diverged"] or lr in excluded or table[lr]["acc"] < pool[best]["acc"]

    lower_worse = i > 0 and worse(lrs[i - 1])
    higher_worse = i < len(lrs) - 1 and worse(lrs[i + 1])
    return best, {"interior": bool(lower_worse and higher_worse), "excluded": sorted(excluded), "guard_dropped": pool is finite and bool(excluded),
                  "edge": None if (lower_worse and higher_worse) else ("smallest" if i == 0 else "largest" if i == len(lrs) - 1 else "a neighbour is as good")}


def interior_pair(best: Mapping, lrs_large: Sequence[float], lrs_small: Sequence[float]) -> list[str]:
    """What is left of the 'optimum must be interior' rule for a pair: the axes on whose end the best pair sits (joint_lr_tuning.edge_warnings)."""
    return JL.edge_warnings(best, lrs_large, lrs_small)


# ------------------------------------------------------------------------------------------------------------ reporting

def _p(x: float) -> str:
    return "  div." if x != x else f"{100 * x:5.2f}"


def member_table(name: str, table: Mapping[float, dict], frozen: Mapping | None, best: float | None, late_floor: float | None) -> str:
    lines = [f"{name} alone (SGD momentum 0.9), macro over the held-out streams, %", f"{'lr':>10} {'acc':>7} {'std':>6} {'late':>7} {'worst':>7} {'seeds':>5}  note"]
    for lr in sorted(table):
        r = table[lr]
        note = ("<- best" if lr == best else "") + (" ruled out: late accuracy below the frozen model's" if not r["diverged"] and late_floor is not None
                                                     and r["late"] < late_floor else "")
        std = "" if r["diverged"] or r.get("std", NAN) != r.get("std", NAN) else f"{100 * r['std']:6.2f}"
        lines.append(f"{lr:>10g} {_p(r['acc']):>7} {std:>6} {_p(r['late']):>7} {_p(r.get('worst', NAN)):>7} {r['n_seeds']:>5}  {note}")
    if frozen:
        lines.append(f"{'frozen':>10} {_p(frozen['acc']):>7} {'':>6} {_p(frozen['late']):>7}")
    return "\n".join(lines)


def report(agg: Mapping, lrs_large: Sequence[float], lrs_small: Sequence[float], base: tuple[float, float] | None,
           names: tuple[str, str] = ("large member", "small member")) -> tuple[list[str], dict]:
    """The printout and the choices: ({lines}, {"large": lr, "small": lr, "pair": point, "individual_interior": {...}, "warnings": [...]})."""
    fz = agg["frozen"]
    floors = {"large": fz["large"]["late"] if fz else None, "small": fz["small"]["late"] if fz else None, "duo": fz["duo"]["late"] if fz else None}
    lines, warnings, choice = [], [], {}
    for side, name in (("large", f"{names[0]} (large member)"), ("small", f"{names[1]} (small member)")):
        best, info = best_member(agg[side], floors[side])
        choice[side] = best
        choice[f"{side}_info"] = info
        lines.append(member_table(name, agg[side], fz[side] if fz else None, best, floors[side]))
        if best is None:
            warnings.append(f"every {side} candidate diverged")
        elif not info["interior"]:
            warnings.append(f"the best {side}-member rate ({best:g}) is not interior to the grid ({info['edge']}): extend the grid and rerun")
        if info["guard_dropped"]:
            warnings.append(f"every {side} candidate ends below the frozen model's late accuracy: the guard was dropped")
        lines.append("")
    points = agg["pairs"]
    ranked = JL.rank(points, late_floor=floors["duo"])
    if not ranked:
        return lines + ["every pair diverged"], {**choice, "pair": None, "warnings": warnings + ["every pair diverged"]}
    best = ranked[0]
    excluded = JL.guard_excluded(points, floors["duo"])
    lines += [f"independent duo: accuracy (%) of the static duo, macro over the held-out streams; rows: large member's rate, columns: small "
              f"member's; * = best\n{JL.grid_table(points, lrs_large, lrs_small, 'duo', best)}",
              f"\nlate accuracy (%): the last {int(100 * JL.LATE_FRAC)}% of each stream\n{JL.grid_table(points, lrs_large, lrs_small, 'late_duo', best)}",
              f"\naccuracy (%) of the worst stream\n{JL.grid_table(points, lrs_large, lrs_small, 'worst', best)}", f"\n{JL.ranking_table(ranked)}"]
    if fz:
        lines.append(f"\nfrozen duo: {_p(fz['duo']['acc']).strip()} (late {_p(fz['duo']['late']).strip()}); frozen large {_p(fz['large']['acc']).strip()}, "
                     f"small {_p(fz['small']['acc']).strip()}; {len(excluded)} of {len([p for p in points if not p['diverged']])} pairs end below the "
                     f"frozen duo's late accuracy and are ruled out")
    lines.append(f"best pair: large {best['lr_large']:g}, small {best['lr_small']:g}: duo {100 * best['duo']:.2f}"
                 + ("" if best.get("duo_std", NAN) != best.get("duo_std", NAN) else f" +- {100 * best['duo_std']:.2f} (std over {best['n_seeds']} seeds)")
                 + f", late {100 * best['late_duo']:.2f}, worst stream {100 * best['worst']:.2f} ({best['worst_stream']})")
    if base:
        at = next((p for p in points if (p["lr_large"], p["lr_small"]) == base and not p["diverged"]), None)
        if at:
            lines.append(f"at the individually tuned rates (L {base[0]:g}, S {base[1]:g}): duo {100 * at['duo']:.2f}, late {100 * at['late_duo']:.2f}; "
                         f"the best pair adds {100 * (best['duo'] - at['duo']):+.2f} points")
    warnings += [w for w in interior_pair(best, lrs_large, lrs_small)]
    for w in warnings:
        lines.append(f"WARNING: {w}")
    return lines, {**choice, "pair": best, "warnings": warnings, "late_floor": floors}


def record_best(results_json: Path, root: Path | None = None, duos_dir: Path | None = None) -> str:
    """Record a finished full run (its results.json): the individual optima as the duo's SINGLE-regime rates and the best pair as its
    INDEP-regime rates, for that TTA method, in cfgs/duo_lrs/<duo>__single.yaml and __indep.yaml (`root`: that folder). A candidate that
    is not interior is refused. Changing these rates changes the cache key of every independent run (the temperatures do not move: they
    are fitted on the frozen models). Returns the lines to print."""
    from src.utils import duo_lrs

    r = json.loads(Path(results_json).read_text())
    if r.get("num_samples"):
        raise ValueError(f"{results_json} is a capped run (--num_samples {r['num_samples']}); only full runs are recorded")
    if r["warnings"]:
        raise ValueError(f"{results_json} has open warnings, fix them before recording: {r['warnings']}")
    duo, tta = r["duo"], r["tta"]
    spec = duo_lrs._duo_spec(duo, duos_dir)
    out = []
    for regime, lrs, metric in (("single", {"large": r["individual"]["large"], "small": r["individual"]["small"]}, "individual"),
                                ("indep", {"large": r["best"]["lr_large"], "small": r["best"]["lr_small"]}, "pair")):
        duo_lrs.init_single(duo, root) if regime == "single" else None
        doc = duo_lrs.load(duo, regime, root)
        old = doc["methods"].get(tta)
        entry = {}
        for side in ("large", "small"):
            entry[side] = {"model": spec[side.upper()]["NAME"], "norm": spec[side.upper()]["NORM"], "batch_size": r["batch_size"],
                           "OPTIM": {**r["optim"][side], "LR": float(lrs[side])}, "source": "scripts/02c_tune_indep_pair.py",
                           "tuned_on": [x.replace("__", "/") for x in r["streams"]], "seeds": r["seeds"], "temperatures": r["temperatures"]["checkpoint"],
                           "note": ("each model tuned ALONE" if metric == "individual" else "tuned as a PAIR for this duo (static-duo accuracy)")}
            if metric == "individual":
                entry[side].update({"macro_accuracy": round(float(r["individual_scores"][side]["acc"]), 5),
                                    "late_accuracy": round(float(r["individual_scores"][side]["late"]), 5),
                                    "frozen_macro_accuracy": round(float(r["frozen"][side]["acc"]), 5)})
            else:
                entry[side].update({"duo_macro_accuracy": round(float(r["best"]["duo"]), 5), "late_duo_accuracy": round(float(r["best"]["late_duo"]), 5),
                                    "worst_stream_accuracy": round(float(r["best"]["worst"]), 5)})
        doc["methods"][tta] = entry
        path = duo_lrs._write(doc, root)
        new = (float(lrs["large"]), float(lrs["small"]))
        prev = (float(old["large"]["OPTIM"]["LR"]), float(old["small"]["OPTIM"]["LR"])) if old else None
        note = "" if prev is None else " (unchanged)" if prev == new else f" (replacing large {prev[0]:g}, small {prev[1]:g})"
        out.append(f"recorded large {new[0]:g} and small {new[1]:g} as the {regime} learning rates of {duo} / {tta} in {path}{note}")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import tempfile

    import yaml

    from src.experiments._testing import make_context
    from src.utils import duo_lrs

    torch.manual_seed(0)
    n, K, T_L, T_S = 200, 30, 1.2, 0.8
    y = torch.randint(0, K, (n,))
    order = np.arange(n, dtype=np.int32)
    BS = [100, 100]

    def logits(skill: float, seed: int, tail_skill: float | None = None) -> torch.Tensor:
        """`skill` boosts the true class; `tail_skill` (if given) replaces it on the last 100 samples (the late window): a member that
        starts well and drifts."""
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, K, generator=g)
        s = torch.full((n,), float(skill))
        if tail_skill is not None:
            s[-100:] = tail_skill
        z[torch.arange(n), y] += s
        return z

    # ---- the late window --------------------------------------------------------------------------------------------
    assert late_correct(torch.tensor([1] * 5 + [0] * 5), [5, 5]) == 0.0 and late_correct(torch.tensor([1] * 5 + [0] * 5), [5, 5], 0.6) == 0.5
    assert abs(late_correct(torch.tensor([1, 1, 1, 1, 0, 0]), [2, 2, 2], 0.2) - 0.0) < 1e-12

    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_context(Path(tmp))
        lrs_l, lrs_s = [1e-4, 2e-4, 4e-4], [5e-4, 1e-3, 2e-3]
        stems = ["a__s5", "b__s5"]
        store = LogitStore(Path(tmp) / "logits", staging=None)
        # the large member improves up to 2e-4; at 4e-4 it starts best of all and then collapses (the best mean, the worst late accuracy);
        # the small one improves up to 1e-3 (2e-3 is worse)
        skill_l = {1e-4: (0.8, None), 2e-4: (1.6, None), 4e-4: (5.0, -5.0)}
        skill_s = {5e-4: 1.0, 1e-3: 2.0, 2e-3: 0.5}
        for seed in (0, 1):
            large, small = member_specs(ctx, "tent", None, seed, None, lrs_l, lrs_s)
            frozen = frozen_specs(ctx, seed, None)
            if seed == 0:
                assert len({m.key for m in large.values()}) == 3 and len({m.key for m in small.values()}) == 3
                assert all(m.model["OPTIM"]["LR"] == lr for lr, m in large.items()), "the key holds the member's own rate"
                assert set(missing(store, large, small, stems, frozen)) == {f"large {lr:g}" for lr in lrs_l} | {f"small {lr:g}" for lr in lrs_s} | {"large frozen", "small frozen"}
            for lr, m in large.items():
                for stem in stems:
                    store.write(m, stem, logits(skill_l[lr][0] + 0.1 * seed, 1, skill_l[lr][1]), y, order, BS, stream=stem)
            for lr, m in small.items():
                for stem in stems:
                    store.write(m, stem, logits(skill_s[lr] + 0.1 * seed, 2), y, order, BS, stream=stem)
            for m, sk in zip(frozen, (0.5, 0.3)):
                for stem in stems:
                    store.write(m, stem, logits(sk, 3 if sk == 0.5 else 4), y, order, BS, stream=stem)
            assert missing(store, large, small, stems, frozen) == {}
        per_seed = {}
        for seed in (0, 1):
            large, small = member_specs(ctx, "tent", None, seed, None, lrs_l, lrs_s)
            per_seed[seed] = score_seed(store, large, small, stems, T_L, T_S, frozen_specs(ctx, seed, None))
        agg = aggregate(per_seed)
        assert agg["seeds"] == [0, 1] and len(agg["pairs"]) == 9 and agg["large"][2e-4]["n_seeds"] == 2 and agg["frozen"]["duo"]["acc"] > 0
        assert aggregate({0: per_seed[0]})["large"][1e-4]["std"] != aggregate({0: per_seed[0]})["large"][1e-4]["std"], "no std from one seed"
        assert agg["large"][2e-4]["std"] >= 0 and set(agg["large"][2e-4]["streams"]) == set(stems)
        # the scores are what the logits say
        want = float((logits(1.6, 1).argmax(1) == y).float().mean())
        assert abs(per_seed[0]["large"][2e-4]["acc"]["a__s5"] - want) < 1e-9
        pair = next(p for p in agg["pairs"] if (p["lr_large"], p["lr_small"]) == (2e-4, 1e-3))
        zs0 = per_seed[0]["pairs"][2e-4, 1e-3]
        z = logits(1.6, 1) / T_L + logits(2.0, 2) / T_S
        assert abs(zs0["acc"]["a__s5"] - float((z.argmax(1) == y).float().mean())) < 1e-9, "a pair is the static duo of its two members' logits"
        assert pair["n_seeds"] == 2 and set(pair["streams"]) == set(stems) and pair["late_duo"] == pair["late_duo"]
        # the late guard: the large member at 4e-4 has the best mean accuracy among candidates but ends far below the frozen one
        floor_l = agg["frozen"]["large"]["late"]
        assert agg["large"][4e-4]["late"] < floor_l < agg["large"][2e-4]["late"], (agg["large"][4e-4]["late"], floor_l)
        best, info = best_member(agg["large"], floor_l)
        assert best == 2e-4 and info["interior"] and 4e-4 in info["excluded"] and not info["guard_dropped"], (best, info)
        assert best_member(agg["large"], None)[0] == 4e-4, "without the guard the member that collapses late wins on its mean"
        assert best_member(agg["small"], agg["frozen"]["small"]["late"])[0] == 1e-3 and best_member(agg["small"], None)[1]["interior"]
        edge = best_member({k: v for k, v in agg["large"].items() if k != 4e-4}, None)
        assert edge[0] == 2e-4 and not edge[1]["interior"] and edge[1]["edge"] == "largest", edge
        impossible, i2 = best_member(agg["large"], 2.0)
        assert impossible is not None and i2["guard_dropped"], "a guard that rules out everything is dropped, and says so"
        lines, choice = report(agg, lrs_l, lrs_s, (1e-4, 5e-4))
        text = "\n".join(lines)
        assert choice["large"] == 2e-4 and choice["small"] == 1e-3 and choice["pair"]["lr_small"] == 1e-3 and "ruled out" in text and "frozen duo:" in text
        assert not [w for w in choice["warnings"] if "interior" in w], choice["warnings"]

        # divergence: a member with non-finite logits makes its pairs diverged, the others stand
        real = store.read

        def read_nan(m, stem, raw=False):
            d = real(m, stem, raw)
            return {**d, "z": d["z"] * float("nan")} if m.key == member_specs(ctx, "tent", None, 0, None, lrs_l, lrs_s)[0][4e-4].key else d

        store.read = read_nan
        pd0 = score_seed(store, *member_specs(ctx, "tent", None, 0, None, lrs_l, lrs_s), stems, T_L, T_S)
        store.read = real
        assert pd0["large"][4e-4]["diverged"] and pd0["pairs"][4e-4, 1e-3]["diverged"] and not pd0["pairs"][2e-4, 1e-3].get("diverged")
        ag = aggregate({0: pd0})
        assert ag["large"][4e-4]["diverged"] and "div." in JL.grid_table(ag["pairs"], lrs_l, lrs_s) and best_member(ag["large"], None)[0] != 4e-4

        # min_seeds: only candidates with enough seeds compete
        only0 = {0: per_seed[0], 1: {"large": {1e-4: per_seed[1]["large"][1e-4]}, "small": {5e-4: per_seed[1]["small"][5e-4]},
                                     "pairs": {(1e-4, 5e-4): per_seed[1]["pairs"][1e-4, 5e-4]}, "frozen": None}}
        a2 = aggregate(only0, min_seeds=2)
        assert [(p["lr_large"], p["lr_small"]) for p in a2["pairs"]] == [(1e-4, 5e-4)] and list(a2["large"]) == [1e-4]

        # recording: the individual optima go to the single file, the pair to the indep file; open warnings and capped runs are refused
        duos_dir, lrs_dir = Path(tmp) / "duos", Path(tmp) / "duo_lrs"
        duos_dir.mkdir(), lrs_dir.mkdir()
        (duos_dir / "tiny.yaml").write_text(yaml.safe_dump({"NAME": "tiny", "LARGE": {"NAME": "tiny_ln", "NORM": "LN"}, "SMALL": {"NAME": "tiny_bn", "NORM": "BN"}}))
        sgd = lambda lr: sgd_optim(lr)                                                 # noqa: E731
        duo_lrs.init_indep("tiny", lrs_dir, duos_dir, defaults={"large": sgd(1e-4), "small": sgd(5e-4)})
        res = Path(tmp) / "results.json"
        payload = {"duo": "tiny", "tta": "tent", "bn": True, "seeds": [0, 1, 2], "num_samples": None, "batch_size": 64, "streams": ["a__s5", "b__s5"],
                   "temperatures": {"T_l": 0.65, "T_s": 1.0, "checkpoint": "heldout_tiny_frozen_x"}, "optim": {"large": sgd(1e-4), "small": sgd(5e-4)},
                   "individual": {"large": 5e-5, "small": 5e-4}, "individual_scores": {"large": {"acc": 0.51, "late": 0.5}, "small": {"acc": 0.45, "late": 0.44}},
                   "frozen": {"large": {"acc": 0.49}, "small": {"acc": 0.35}}, "best": {"lr_large": 1e-4, "lr_small": 5e-4, "duo": 0.57, "late_duo": 0.56, "worst": 0.38},
                   "warnings": ["the best large-member rate (5e-05) is not interior to the grid (smallest): extend the grid and rerun"]}
        res.write_text(json.dumps(payload))
        try:
            record_best(res, lrs_dir, duos_dir)
            raise AssertionError("open warnings must refuse the record")
        except ValueError as e:
            assert "open warnings" in str(e)
        res.write_text(json.dumps({**payload, "warnings": []}))
        msg = record_best(res, lrs_dir, duos_dir)
        assert "as the single learning rates" in msg and "as the indep learning rates" in msg
        tiny = {"NAME": "tiny", "LARGE": {"NAME": "tiny_ln", "NORM": "LN"}, "SMALL": {"NAME": "tiny_bn", "NORM": "BN"}}
        assert duo_lrs.resolve(tiny, "single", "tent", root=lrs_dir)["LARGE"]["LR"] == 5e-5 and duo_lrs.resolve(tiny, "indep", "tent", root=lrs_dir)["LARGE"]["LR"] == 1e-4
        assert duo_lrs.resolve(tiny, "single", "tent", root=lrs_dir)["SMALL"]["LR"] == 5e-4 and "(unchanged)" in record_best(res, lrs_dir, duos_dir)
        res.write_text(json.dumps({**payload, "warnings": [], "num_samples": 2000}))
        try:
            record_best(res, lrs_dir, duos_dir)
            raise AssertionError("a capped run must not be recorded")
        except ValueError as e:
            assert "capped" in str(e)
    print("indep_pair_tuning self-test passed")
