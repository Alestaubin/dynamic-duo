#!/usr/bin/env python3
"""
Tune Tent's learning rates for each model ALONE and for the INDEPENDENT duo (each member adapts on its own entropy), on the 4 held-out
corruptions, from per-member logits (src/experiments/indep_pair_tuning.py); no model runs here.

  single   each model alone: its own whole-stream accuracy at each rate; the optimum must be INTERIOR to the grid, then it is kept for every
           test (recorded in cfgs/duo_lrs/<duo>__single.yaml)
  indep    the PAIR (lr_large, lr_small) of the independent duo, scored by the static duo with the Fixed TS fitted on the FROZEN models
           (recorded in cfgs/duo_lrs/<duo>__indep.yaml); the joint duo's pair is tuned by 02b, upward from the single rates

A candidate whose late accuracy (the last 20% of each stream) is below the frozen model's is ruled out; the frozen model is on every
table. SGD with momentum 0.9, the rate only, on a sqrt-2 grid that is extended until the optimum is interior. Procedure:

    # 1. the diagonal runs fill the logit store (a run at (a_i, b_i) stores the large member at a_i and the small one at b_i)
    sbatch --qos=high --exclude=kn117 slurm/job.sh scripts/04_cache_logits.py --duo D --bn --streams heldout --lr_large a_i --lr_small b_i --seeds 0
    # (the axes may differ in length or in which members are already stored: the script then lists single-member runs, --mode large_indep / small_indep)
    sbatch --qos=high --exclude=kn117 slurm/job.sh scripts/04_cache_logits.py --duo D --bn --streams heldout --mode no_adapt --seeds 0   # the frozen members
    # 2. stage A: score the full grid on seed 0 (the script lists what is missing, with the commands)
    python scripts/02c_tune_indep_pair.py --duo D --bn --lrs_large a_0 ... --lrs_small b_0 ... --seeds 0
    # 3. stage B: confirm the neighbourhood of the optima on seeds 1 and 2 (the script lists the runs), then the final choice
    python scripts/02c_tune_indep_pair.py --duo D --bn --lrs_large ... --lrs_small ... --seeds 0 1 2 [--record]

--record writes the individual optima to the single file and the best pair to the indep file, and refuses while a warning is open (an
optimum on the edge of the grid, a neighbourhood that is not covered). It changes the cache key of every independent run. The
temperatures never move: they are the frozen fit, `--temps_checkpoint DIR` names another. Writes
<paths.out>/results/tune_indep_pair/<duo>__<tta>[_bn]/ranking.csv and results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments import indep_pair_tuning as IP
from src.experiments import joint_lr_tuning as JL
from src.experiments.cli import add_common_args, bn_tag, setup
from src.experiments.logit_runs import stream_segments
from src.experiments.logit_store import LogitStore
from src.experiments.results import save_rows
from src.utils import provenance

DEFAULT_MULTS = (0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 2.8, 4)


def member_commands(duo: str, bn: bool, seed: int, need: set[tuple[str, float]], lrs_large: list[float], lrs_small: list[float],
                    frozen: bool) -> list[str]:
    """The scripts/04 runs that store the members `need` ((side, rate) pairs): a diagonal both_indep run where the i-th rate of each axis is
    missing together (axes of equal length), else one run per member (large_indep / small_indep: the other member is frozen, and its optimizer
    is not part of its key, so any rate stands in for it)."""
    base = f"sbatch --qos=high --exclude=kn117 -t 00:45:00 slurm/job.sh scripts/04_cache_logits.py --duo {duo}{' --bn' if bn else ''} --streams heldout"
    out, done = [], set()
    if len(lrs_large) == len(lrs_small):
        for a, b in zip(lrs_large, lrs_small):
            if ("large", a) in need and ("small", b) in need:
                out.append(f"{base} --lr_large {a:g} --lr_small {b:g} --seeds {seed}")
                done |= {("large", a), ("small", b)}
    for side, lr in sorted(need - done):
        out.append(f"{base} --mode {side}_indep --lr_large {lr if side == 'large' else lrs_large[0]:g} "
                   f"--lr_small {lr if side == 'small' else lrs_small[0]:g} --seeds {seed}")
    return out + ([f"{base} --mode no_adapt --seeds {seed}"] if frozen else [])


def _needed(gaps: dict[str, list[str]]) -> set[tuple[str, float]]:
    return {(k.split()[0], float(k.split()[1])) for k in gaps if k.split()[1] != "frozen"}


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--mults", type=float, nargs="+", default=list(DEFAULT_MULTS), metavar="X",
                    help="the grid of each member: these multiples of its current independent rate (default sqrt-2 spaced). Default: %(default)s")
    ap.add_argument("--lrs_large", type=float, nargs="+", default=None, metavar="LR", help="explicit rates of the large member")
    ap.add_argument("--lrs_small", type=float, nargs="+", default=None, metavar="LR", help="explicit rates of the small member")
    ap.add_argument("--confirm_around", type=int, default=1, metavar="N",
                    help="seeds after the first only need the members within N grid steps of the first seed's optima (default %(default)s)")
    ap.add_argument("--momentum", type=float, default=0.9, help="SGD momentum of the stored members (scripts/04's default)")
    ap.add_argument("--temps_checkpoint", default=None, metavar="DIR", help="JointFixedTS checkpoint (default: the frozen-model fit)")
    ap.add_argument("--record", action="store_true", help="write the individual optima and the best pair to cfgs/duo_lrs (see above)")
    args = ap.parse_args()
    tag = f"{args.duo}__{args.tta}{bn_tag(args)}" + (f"__n{args.num_samples}" if args.num_samples else "")
    ctx, seeds, out = setup(args, f"tune_indep_pair/{tag}", default_seeds=[0])

    names = (provenance.model_name(ctx.duo["LARGE"]["NAME"]), provenance.model_name(ctx.duo["SMALL"]["NAME"]))
    print(provenance.footer(ctx.duo, args.tta, args.bn))
    current = {side: float(ctx.duo[side]["OPTIM"]["LR"]) for side in ("LARGE", "SMALL")}
    lrs_large = JL.lr_axis(current["LARGE"], args.mults, args.lrs_large)
    lrs_small = JL.lr_axis(current["SMALL"], args.mults, args.lrs_small)
    base = tuple(float(f"{current[side]:.6g}") for side in ("LARGE", "SMALL"))
    for side, lrs in (("LARGE", lrs_large), ("SMALL", lrs_small)):
        print(f"{side.lower()} {ctx.duo[side]['NAME']} ({ctx.duo[side]['NORM']}): grid {[f'{v:g}' for v in lrs]}")
    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, args.tta, args.tta_kwargs, args.num_samples),
                                    args.temps_checkpoint, report_only=True)
    print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}  ({temps.folder.name}; {temps.origin})")

    store = LogitStore.from_context(ctx, None)
    stems = [s.file_stem for s in stream_segments(["heldout"], ctx.g)]

    def specs(seed):
        large, small = IP.member_specs(ctx, args.tta, args.tta_kwargs, seed, args.num_samples, lrs_large, lrs_small, args.momentum)
        return large, small, IP.frozen_specs(ctx, seed, args.num_samples)

    # ---- stage A: the full grid on the first seed ------------------------------------------------------------------------
    first = seeds[0]
    large, small, frozen = specs(first)
    gaps = IP.missing(store, large, small, stems, frozen)
    if gaps:
        print(f"seed {first}: {len(gaps)} members are not stored for the held-out streams: {sorted(gaps)}")
        for c in member_commands(args.duo, args.bn, first, _needed(gaps), lrs_large, lrs_small, "large frozen" in gaps):
            print("  " + c)
        return 1
    per_seed = {first: IP.score_seed(store, large, small, stems, temps.T_l, temps.T_s, frozen)}
    agg = IP.aggregate(per_seed)
    print("\n== stage A: the full grid, seed " + str(first))
    lines, choice = IP.report(agg, lrs_large, lrs_small, base, names)
    print("\n".join(lines))
    warnings = list(choice["warnings"])

    # ---- stage B: the neighbourhood of the optima on the other seeds -----------------------------------------------------
    if len(seeds) > 1:
        centres = {"large": {lrs_large.index(choice["large"]), lrs_large.index(choice["pair"]["lr_large"])},
                   "small": {lrs_small.index(choice["small"]), lrs_small.index(choice["pair"]["lr_small"])}} if choice["pair"] and choice["large"] and choice["small"] else None
        if centres is None:
            print("no stage-A optimum to confirm; fix stage A first")
            return 1
        N = args.confirm_around
        axis = {"large": lrs_large, "small": lrs_small}
        near = {side: sorted({j for c in cs for j in range(max(0, c - N), min(len(axis[side]), c + N + 1))}) for side, cs in centres.items()}
        sub_l, sub_s = [lrs_large[i] for i in near["large"]], [lrs_small[i] for i in near["small"]]
        print(f"\n== stage B: confirming large {[f'{v:g}' for v in sub_l]} x small {[f'{v:g}' for v in sub_s]} on seeds {seeds[1:]}")
        need = {}
        for seed in seeds[1:]:
            lg, sm = IP.member_specs(ctx, args.tta, args.tta_kwargs, seed, args.num_samples, sub_l, sub_s, args.momentum)
            fz = IP.frozen_specs(ctx, seed, args.num_samples)
            gaps = IP.missing(store, lg, sm, stems, fz)
            if gaps:
                need[seed] = member_commands(args.duo, args.bn, seed, _needed(gaps), sub_l, sub_s, "large frozen" in gaps)
                continue
            per_seed[seed] = IP.score_seed(store, lg, sm, stems, temps.T_l, temps.T_s, fz)
        if need:
            for seed, cmds in need.items():
                print(f"seed {seed}: members missing; run:\n  " + "\n  ".join(cmds))
            return 1
        agg = IP.aggregate(per_seed, min_seeds=len(seeds))
        lines, choice = IP.report(agg, sub_l, sub_s, base, names)
        print("\n".join(lines))
        warnings = sorted(set(warnings) | set(choice["warnings"]))

    # ---- output ---------------------------------------------------------------------------------------------------------
    if choice["pair"] is None or choice["large"] is None or choice["small"] is None:
        return 1
    best = choice["pair"]
    rows = []
    for p in sorted(agg["pairs"], key=lambda p: (p["diverged"], -(p["duo"] if p["duo"] == p["duo"] else 0))):
        rows.append({"kind": "pair", "lr_large": p["lr_large"], "lr_small": p["lr_small"], "diverged": p["diverged"], "duo_acc": p["duo"],
                     "late_duo": p["late_duo"], "worst_acc": p["worst"], "worst_stream": p["worst_stream"], "duo_nll": p["duo_nll"], "n_seeds": p["n_seeds"],
                     "duo_std": p.get("duo_std"), **{f"acc/{k.replace('__', '/')}": v for k, v in p["streams"].items()}})
    for side in ("large", "small"):
        for lr, r in sorted(agg[side].items()):
            rows.append({"kind": side, "lr_large": lr if side == "large" else None, "lr_small": lr if side == "small" else None, "diverged": r["diverged"],
                         "duo_acc": r["acc"], "late_duo": r["late"], "worst_acc": r.get("worst"), "n_seeds": r["n_seeds"]})
    save_rows(out / "ranking.csv", pd.DataFrame(rows))
    clean = lambda v: None if isinstance(v, float) and v != v else v                              # noqa: E731
    (out / "results.json").write_text(json.dumps({
        "duo": args.duo, "tta": args.tta, "tta_kwargs": args.tta_kwargs, "bn": args.bn, "seeds": seeds, "num_samples": args.num_samples,
        "batch_size": ctx.batch_size, "streams": stems, "temperatures": {"T_l": temps.T_l, "T_s": temps.T_s, "checkpoint": temps.folder.name,
                                                                          "origin": temps.origin},
        "optim": {side.lower(): ctx.duo[side]["OPTIM"] for side in ("LARGE", "SMALL")}, "lrs_large": lrs_large, "lrs_small": lrs_small,
        "individual": {"large": choice["large"], "small": choice["small"]},
        "individual_scores": {s: {k: clean(agg[s][choice[s]][k]) for k in ("acc", "late", "worst")} for s in ("large", "small")},
        "frozen": {k: {kk: clean(vv) for kk, vv in v.items()} for k, v in agg["frozen"].items()},
        "best": {"lr_large": best["lr_large"], "lr_small": best["lr_small"], "duo": best["duo"], "late_duo": best["late_duo"], "worst": best["worst"],
                 "duo_std": clean(best.get("duo_std"))},
        "warnings": warnings, "late_floor": choice["late_floor"]}, indent=1))
    print(f"\nwrote {out / 'ranking.csv'} and {out / 'results.json'}")
    if warnings:
        print("OPEN WARNINGS (nothing can be recorded): " + "; ".join(warnings))
    if args.num_samples:
        print("capped run (--num_samples): not recorded")
    elif args.record:
        print(IP.record_best(out / "results.json", ctx.g.path("duo_lrs")))
    else:
        print(f"not recorded: --record writes large {choice['large']:g} / small {choice['small']:g} (single) and the pair "
              f"{best['lr_large']:g} / {best['lr_small']:g} (indep) to cfgs/duo_lrs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
