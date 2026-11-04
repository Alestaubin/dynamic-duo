#!/usr/bin/env python3
"""
Different ways of adapting the duo: Joint, Large / Small adapt, Independent large / small / both.

Every row is scored by the same static duo (z_L / T_L + z_S / T_S) / 2, so the rows differ only in how the members'
weights were adapted:

    Joint               both members minimize the entropy of the duo's combined prediction
    Large adapt         only the large member adapts, on the duo entropy; the small one is frozen (constant logits)
    Small adapt         only the small member adapts, on the duo entropy; the large one is frozen
    Independent large   Tent on the large member's own entropy; the small member is not adapted
    Independent small   Tent on the small member's own entropy; the large member is not adapted
    Independent both    both members run Tent on their own entropy
    (No adaptation)     frozen weights (see --bn), as the reference row

The temperatures T_L, T_S are the paper's Fixed TS, shared by every mode: fitted once on the 4 held-out corruptions
pooled, on the logits of independently adapted members (the "Held-out TS" of 01_tab_temperatures.py, whose cache
this reuses), and stored under paths.checkpoints. --temps_checkpoint DIR uses another JointFixedTS checkpoint.
One live run per (mode, seed) over the 15 test corruptions (severity 5) and the clean validation images; each stream is
cached, so a repeat only replays (and a cache from before the clean stream existed only computes that stream).

    sbatch slurm/job.sh scripts/03_tab_adaptation_modes.py --temps_only                  # once: fit T_L, T_S
    sbatch --array=0-20 slurm/job.sh scripts/03_tab_adaptation_modes.py --shard auto      # one job per (mode, seed)
    python scripts/03_tab_adaptation_modes.py --report_only                               # tables from the cache
    python scripts/03_tab_adaptation_modes.py --modes joint indep_duo --seeds 0 --num_samples 2000   # smoke test
    python scripts/03_tab_adaptation_modes.py --heldout --report_only       # the same tables on the 4 corruptions the rates were TUNED on

Learning rates. The modes adapt on different signals and each kind runs with its own rates (cfgs/duo_lrs/<duo>__<regime>.yaml):
the modes that adapt on a member's OWN entropy (Independent large / small / both) with the indep regime's rates (tuned member by member,
scripts/02_tune_tta.py), Joint (both members on the DUO entropy) with the joint regime's pair (tuned by scripts/02b_tune_joint_lr.py,
which records its winner there itself), and Large adapt / Small adapt (ONE member on the duo entropy, the other frozen) each with the
rate tuned for that mode alone (the large_duo / small_duo regimes: scripts/02d_tune_large_adapt_lr.py and 02e_tune_small_adapt_lr.py with
--record), NOT the joint pair's rate: with the partner frozen the duo entropy is another signal. A duo whose Joint pair or Large / Small
adapt rate is not tuned yet makes that mode an ERROR (NotTuned), never a silent run at other rates; --joint_lr_large LR --joint_lr_small LR
run Joint with rates given by hand (both are needed to skip the file; either one alone replaces just that member's), --anchor_lr_large LR /
--anchor_lr_small LR do it for Large / Small adapt (one rate each: the adapting member's). --lr_large / --lr_small replace the own-entropy
modes' rates the same way (the rest of each member's OPTIM stays). Independent large / small (one model adapting alone) run with the rates
tuned for that model ALONE (the single regime); Independent both with the independent duo's pair (the indep regime). Both tables' captions
list every set of rates and say which modes use which.
The rates are part of every cache key, so nothing is shared between two settings, and the temperatures (one pair for every mode, fitted
on the frozen models) never depend on them. With any of these flags the CSV and both tables are named
<tag>_lrL<large>_S<small>[_jlrL<large>_S<small>][_alrL<large>_S<small>]_<YYYYmmdd-HHMMSS> (the time of the rendering run; a shard never
renders), so they never replace the tuned run's files in the paper folder nor each other.

--heldout: every mode on the 4 held-out corruptions (the protocol's tuning set: the streams every learning rate was tuned on, no clean stream) instead of
the 15 test ones, tag `<tag>_heldout`. The member cache of the independent-segment protocols is shared with the tuners (02b, 02c, 02d, 02e), so what they
already ran is reused and only the missing (mode, seed) runs are computed (`--report_only` lists them, a `--shard auto` array computes them). These are
the numbers the rates were selected on: they say what the tuning saw, not how the modes generalize.

--gate_betas B [B ...] (DEFAULT 0.25 0.5 on the 15 test corruptions; --no_gate, --heldout and --num_samples leave the gate out): the reliability gate beside the Fixed TS rows (beta = 0),
for the two modes in which both members adapt. A beta with no Joint DD runs in --joint_dd_dir (by default `run_joint_dd_gated`, `run_joint_dd_gated__<criterion>` for another
--criterion) is left out of the Joint DD rows and the caption says so; a --gate_betas given explicitly requires them. The summary table is made from the plain modes under the plain tag.
Details: Each B > 0 adds a
Vanilla DD row after Independent both (the gate OUTSIDE the loop: replayed on that run's cached logits, no model) and a Joint DD row after Joint (the
gate INSIDE it: live runs of scripts/09_run_joint_dd.py, one per seed, read from --joint_dd_dir, default <paths.out>/results/run_joint_dd_gated). Each
beta's Joint DD runs have their own learning rates (the pair scripts/02f_tune_joint_gate.py found at that beta), recorded in the run's .meta.json and
listed in the caption, so a beta's runs go in the folder with the rest of them and nothing else:

    sbatch --array=0-2 slurm/job.sh scripts/09_run_joint_dd.py --duo vitb16_rn50v1 --bn --streams test clean --cells nnn:256 --betas 0.25 \\
        --seeds 0 1 2 --shard auto --lr_large 0.00113 --lr_small 0.00283 --eval <eval_duo_grid prefix> --out out/results/run_joint_dd_gated
    (the same with --betas 0.5 --lr_large 0.0008 --lr_small 0.00283)
    sbatch slurm/job_cpu.sh scripts/03_tab_adaptation_modes.py --bn --report_only --gate_betas 0.25 0.5

The table is written under the tag <tag>_gate_<proxy>_bt<b_t>_beta<betas> (the summary table is not: it has no gated rows). Test corruptions only.

Writes <paths.paper>/tab_adaptation_modes__<tag>.tex (accuracy per corruption: the duo under each mode, and each
member alone adapting on its own entropy / frozen; a Clean column that is not part of Avg.),
tab_adaptation_modes_summary__<tag>.tex (members' accuracy under each mode, duo accuracy / ECE / NLL, averaged over
the corruptions) and <paths.out>/results/tab_adaptation_modes/<tag>.csv (every stream, clean included). <tag> is the TTA method (plus the duo name
unless it is the default duo). A capped run (--num_samples) writes its tables to the results folder instead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments import gated_duos as GD
from src.experiments.cli import add_common_args, select_jobs, setup
from src.experiments.results import per_seed_average, save_rows
from src.utils.tables import write_table


HELDOUT_NOTE = ("These are the 4 held-out corruptions the learning rates were tuned on (speckle noise, Gaussian blur, spatter, saturate), "
                "not the test corruptions: the numbers show what the tuning saw.")


DEFAULT_GATE_BETAS = (0.25, 0.5)     # the gated rows of the test-corruption table: Vanilla DD and Joint DD at these betas, beside the Fixed TS (beta 0) rows


def print_rates(what: str, ctx) -> None:
    print(f"{what}: " + ", ".join(f"{side.lower()} {ctx.duo[side]['NAME']} {ctx.duo[side]['OPTIM']['LR']:g}"
                                  for side in ("LARGE", "SMALL")))


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    AM.add_args(ap)
    ap.add_argument("--heldout", action="store_true", help="the 4 held-out (tuning) corruptions instead of the 15 test ones; see the docstring")
    ap.add_argument("--no_gate", action="store_true", help="the table of the 15 test corruptions WITHOUT the gated rows (the former default); see --gate_betas")
    ap.add_argument("--gate_betas", type=float, nargs="+", default=None, metavar="BETA",
                    help="the reliability gate in the table (default 0.25 0.5 on the 15 test corruptions; --no_gate, --heldout and --num_samples leave it out): "
                         "for each BETA > 0, a Vanilla DD row (the gate replayed on the cached Independent "
                         "both logits) and a Joint DD row (the gate inside the loss: script 09's live runs, read from --joint_dd_dir, each at "
                         "the learning rates recorded in its .meta.json) beside the Fixed TS rows (beta = 0). The table is written under its own tag; "
                         "the summary table is not (it has no gated rows)")
    ap.add_argument("--gate_proxy", default=None, help="the gate's proxy (default: cfgs/global.yaml gate: proxy)")
    ap.add_argument("--gate_b_t", type=int, default=256, help="the gate's proxy batch size")
    ap.add_argument("--joint_dd_dir", default=None, metavar="DIR",
                    help="where script 09 wrote the Joint DD rows (default: <paths.out>/results/run_joint_dd_gated)")
    ap.add_argument("--tag_suffix", default="", metavar="TEXT",
                    help="appended to the gated table's tag (its CSV, .tex and label), for a table whose Joint DD rows come from another --joint_dd_dir (other rates) than the default's")
    args = ap.parse_args()
    gate_explicit = args.gate_betas is not None            # given: its Joint DD runs must exist; by default a beta without them is left out of the table (and said in the caption)
    if args.gate_betas and (args.heldout or args.num_samples or any(b <= 0 for b in args.gate_betas)):
        ap.error("--gate_betas takes betas > 0 (beta = 0 is the Fixed TS row) and works on the test corruptions only (no --heldout, no --num_samples)")
    if args.gate_betas is None and not (args.no_gate or args.heldout or args.num_samples):
        args.gate_betas = list(DEFAULT_GATE_BETAS)
    ctx, seeds, out = setup(args, "tab_adaptation_modes")
    segments = AM.heldout_segments(ctx) if args.heldout else None      # None: the protocol's test corruptions and the clean stream
    ctx = AM.with_learning_rates(ctx, args.lr_large, args.lr_small)       # the own-entropy modes' rates (indep regime)
    print_rates("own-entropy learning rates", ctx)

    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, args.tta, args.tta_kwargs, args.num_samples),
                                    args.temps_checkpoint, report_only=args.report_only)
    print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}  ({temps.folder})")
    if args.temps_only:
        return 0

    duo_ctx = None            # Joint's rates (joint regime); only needed when that mode runs
    if "joint" in args.modes:
        duo_ctx = AM.joint_context(ctx, args.tta, args.joint_lr_large, args.joint_lr_small)
        print_rates("Joint learning rates (both models on the duo entropy)", duo_ctx)
    # Large / Small adapt: the rate tuned for that mode alone (large_duo / small_duo regimes), never the joint pair's; NotTuned if missing
    anchor_ctx = AM.anchor_contexts(ctx, args.tta, args.modes, args.anchor_lr_large, args.anchor_lr_small)
    for key, c in anchor_ctx.items():
        side = AM.ANCHOR_SIDE[key].upper()
        print(f"{AM.MODE_BY_KEY[key].label} learning rate (the {side.lower()} member adapts on the duo entropy, the other is frozen): "
              f"{c.duo[side]['NAME']} {c.duo[side]['OPTIM']['LR']:g}")
    single_ctx = None         # Independent large / small: one model adapting alone, at the rates tuned for that model alone (single regime)
    if any(key in AM.SINGLE_MODES for key in args.modes):
        single_ctx = AM.single_context(ctx, args.tta)
        print_rates("single-model learning rates (one model adapting alone)", single_ctx)

    jobs = AM.mode_jobs(args.duo, args.tta, args.tta_kwargs, seeds, args.num_samples, temps, args.modes)
    if not args.report_only:
        AM.compute_members(ctx, select_jobs(jobs, AM.job_label, args.shard, args.only), segments, duo_ctx=duo_ctx, single_ctx=single_ctx,
                           anchor_ctx=anchor_ctx)
    gaps = AM.missing(ctx, jobs, duo_ctx, single_ctx, anchor_ctx, segments)
    if gaps:
        for label, streams in gaps.items():
            print(f"{label}: member logits missing for {len(streams)} streams (e.g. {streams[:3]}); "
                  f"run without --report_only to compute them")
        return 0 if args.shard or args.only else 1      # a shard only fills the cache; the full run renders
    if args.shard:
        return 0        # (a shard that finds the cache complete must not render too: with --lr_* the names carry a stamp)

    rows = AM.duo_rows(jobs, ctx, temps, segments, duo_ctx=duo_ctx, single_ctx=single_ctx, anchor_ctx=anchor_ctx)
    gate_kw = {}
    if args.gate_betas:
        betas = sorted(set(args.gate_betas))
        proxy = args.gate_proxy or ctx.g.gate["proxy"]
        proxy = {"nuclear_norm": "nnn", "ac_mc": "msp"}.get(proxy, proxy)           # script 09 names its files by the pure.py key
        jdir = Path(args.joint_dd_dir) if args.joint_dd_dir else ctx.g.path("out") / "results" / ("run_joint_dd_gated" + ("" if args.criterion == "mean" else f"__{args.criterion}"))
        have = {m.key for m, _ in jobs}
        spec_of = {(m.key, s.seed): s for m, s in jobs}
        rates, missing_joint = {}, []
        for beta in betas:
            if "indep_duo" in have:                  # Vanilla DD: a replay of the Independent both logits, no model
                gate = GD.vanilla_gate(ctx.g, temps, proxy=proxy, beta=beta, b_t=args.gate_b_t)
                for seed in seeds:
                    rows += GD.vanilla_rows(spec_of[("indep_duo", seed)], ctx, gate, segments, mode=GD.gate_mode(GD.VANILLA, beta))
            if "joint" in have:                      # Joint DD: script 09's live runs (the gate inside the loop)
                try:
                    rates[beta] = GD.joint_dd_rates(jdir, args.duo, proxy, args.gate_b_t, beta, seeds)
                except FileNotFoundError as e:
                    if gate_explicit:
                        raise
                    print(f"Joint DD beta {beta:g}: no runs in {jdir} ({str(e).split(' missing')[0].split('/')[-1]}): its row is left out of the table")
                    missing_joint.append(beta)
                    continue
                rows += GD.joint_dd_rows(jdir, args.duo, args.tta, rows[0]["protocol_tag"], proxy, args.gate_b_t, beta, seeds,
                                         mode=GD.gate_mode(GD.JOINT_DD, beta))
                print(f"Joint DD beta {beta:g}: large lr {rates[beta][0]:g}, small lr {rates[beta][1]:g}  ({jdir})")
        extra, relabel = GD.gated_row_defs(betas, indep="indep_duo" in have, joint="joint" in have, joint_betas=[b for b in betas if b in rates])
        gate_kw = dict(extra_rows=extra, relabel=relabel,
                       gate_text=GD.gate_caption(ctx, proxy, args.gate_b_t, betas, rates, missing_joint,
                                                 joint_rates=tuple(duo_ctx.duo[s]["OPTIM"]["LR"] for s in ("LARGE", "SMALL")) if duo_ctx else None))
    df = pd.DataFrame(rows)
    tag = AM.output_tag(args, ap.get_default("duo"), stamp=time.strftime("%Y%m%d-%H%M%S")) + ("_heldout" if args.heldout else "")
    if args.gate_betas:
        tag += f"_gate_{proxy}_bt{args.gate_b_t}_beta" + "-".join(AM._lr_text(b) for b in betas) + args.tag_suffix
    save_rows(out / f"{tag}.csv", df)
    corruptions, clean = AM.split_clean(df)
    mean_acc = lambda d: (per_seed_average(d.assign(series=d["mode"] + "/" + d["series"]))     # noqa: E731
                          .groupby("series")["accuracy"].mean().mul(100).round(2))
    print(pd.DataFrame({"avg": mean_acc(corruptions), "clean": mean_acc(clean)}).to_string())

    dest = out if args.num_samples else ctx.g.path("paper")       # a capped run (smoke test) must not overwrite the paper's
    source = "scripts/03_tab_adaptation_modes.py"
    write_table(dest / f"tab_adaptation_modes__{tag}.tex", AM.render_corruption_table(
        df, ctx, args.tta, temps, label=f"tab:adaptation_modes:{tag}", source=source, duo_ctx=duo_ctx,
        single_ctx=single_ctx, anchor_ctx=anchor_ctx, note=HELDOUT_NOTE if args.heldout else "", **gate_kw))
    plain = df[df["mode"].isin(AM.MODE_KEYS)]                      # the summary has one row per mode and none for the gated runs: it is made from the plain modes, under the plain tag
    ptag = tag.split("_gate_")[0]
    write_table(dest / f"tab_adaptation_modes_summary__{ptag}.tex", AM.render_summary_table(
        plain, ctx, args.tta, temps, label=f"tab:adaptation_modes_summary:{ptag}", source=source, duo_ctx=duo_ctx,
        single_ctx=single_ctx, anchor_ctx=anchor_ctx, note=HELDOUT_NOTE if args.heldout else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
