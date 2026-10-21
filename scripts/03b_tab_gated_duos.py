#!/usr/bin/env python3
"""
Dynamic Duos' gate against the static duo, in the two ways of adapting (Vanilla DD and Joint DD), on the 15 test corruptions
and the clean validation images. One gate setting, default: the paper's gate of cfgs/global.yaml (nuclear-norm proxy, identity
calibration, log pooling, no filter) with proxy batch size 256 and beta 0.5.

    Vanilla DD   the gate outside the adaptation loop: replayed on the logits of script 03's Independent both run (members
                 adapting on their own entropy, the indep regime's learning rates). No model runs here.
    Joint DD     the gate inside the loop (both members adapt on the entropy of the gated prediction, the joint regime's
                 learning rates): one live run per seed, made by scripts/09_run_joint_dd.py, read from --joint_dd_dir.

and, for comparison, the frozen members and the static duo (z_L / T_L + z_S / T_S) / 2 of script 03 in each regime (its Independent
both and Joint rows, read from its caches). This script computes nothing on a GPU: run it after script 03 and script 09 have finished.

    sbatch slurm/job.sh scripts/09_run_joint_dd.py --duo vitb16_rn50v1 --bn --streams test clean \\
        --eval out/results/eval_duo_grid/<name> --cells nnn:256 --betas 0.5 --seeds 0 1 2 --shard auto \\
        --out out/results/run_joint_dd_test                       # one task per seed (--array=0-2)
    sbatch --array=0-2 slurm/job.sh scripts/03_tab_adaptation_modes.py --duo vitb16_rn50v1 --bn --shard auto --modes joint ...
    sbatch slurm/job_cpu.sh scripts/03b_tab_gated_duos.py --duo vitb16_rn50v1 --bn

Writes <paths.paper>/tab_gated_duos__<tag>.tex (accuracy per corruption, with a Clean column that is not part of Avg.) and
<paths.out>/results/tab_gated_duos/<tag>.csv (every row, clean included), and prints accuracy / ECE / NLL averaged over the
corruptions. <tag> is the TTA method, the duo unless it is the default, `_bn`, and `_<proxy>_bt<b_t>_beta<beta>`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import adaptation_modes as AM
from src.experiments import gated_duos as GD
from src.experiments.cli import add_common_args, bn_tag, setup
from src.experiments.results import save_rows
from src.utils.tables import write_table


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter),
                         tta=False)
    ap.add_argument("--proxy", default=None, help="the gate's proxy, a key of src/reliability/proxies/pure.PROXIES (default: the "
                                                  "paper's gate of cfgs/global.yaml, nuclear_norm = nnn)")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--b_t", type=int, default=256, help="proxy batch size")
    ap.add_argument("--joint_dd_dir", default=None, metavar="DIR",
                    help="where script 09 wrote the Joint DD rows (default: <paths.out>/results/run_joint_dd_test)")
    ap.add_argument("--temps_checkpoint", default=None, metavar="DIR", help="as in script 03 (default: the held-out fit it uses)")
    args = ap.parse_args()
    tta = "tent"            # Joint DD (and the joint mode it is compared with) exists for Tent only

    ctx, seeds, out = setup(args, "tab_gated_duos")
    proxy = args.proxy or ctx.g.gate["proxy"]
    proxy = {"nuclear_norm": "nnn", "ac_mc": "msp"}.get(proxy, proxy)       # script 09 names runs by the pure.py key
    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, tta, None, None), args.temps_checkpoint, report_only=True)
    duo_ctx = AM.joint_context(ctx, tta)
    print(f"static duo: T_L = {temps.T_l}, T_S = {temps.T_s}  ({temps.folder})")
    for what, c in (("own-entropy", ctx), ("duo-entropy", duo_ctx)):
        print(f"{what} learning rates: " + ", ".join(f"{s.lower()} {c.duo[s]['NAME']} {c.duo[s]['OPTIM']['LR']:g}" for s in ("LARGE", "SMALL")))
    print(f"gate: {proxy}, proxy batch size {args.b_t}, beta {args.beta:g}")

    jobs = AM.mode_jobs(args.duo, tta, None, seeds, None, temps, list(GD.MODES))
    gaps = AM.missing(ctx, jobs, duo_ctx)
    if gaps:
        for label, streams in gaps.items():
            print(f"{label}: member logits missing for {len(streams)} streams (e.g. {streams[:3]}); run script 03 first")
        return 1
    rows = AM.duo_rows(jobs, ctx, temps, duo_ctx=duo_ctx)
    gate = GD.vanilla_gate(ctx.g, temps, proxy=proxy, beta=args.beta, b_t=args.b_t)
    spec_of = {(m.key, s.seed): s for m, s in jobs}
    for seed in seeds:
        rows += GD.vanilla_rows(spec_of[("indep_duo", seed)], ctx, gate)
    directory = Path(args.joint_dd_dir) if args.joint_dd_dir else out.parent / "run_joint_dd_test"
    rows += GD.joint_dd_rows(directory, args.duo, tta, rows[0]["protocol_tag"], proxy, args.b_t, args.beta, seeds)

    df = pd.DataFrame(rows)
    tag = (f"{tta}" + ("" if args.duo == ap.get_default("duo") else f"__{args.duo}") + bn_tag(args)
           + f"_{proxy}_bt{args.b_t}_beta{args.beta:g}")
    save_rows(out / f"{tag}.csv", df)
    print("\naccuracy / ECE (%) and NLL, averaged over the 15 corruptions (clean left out)" +
          (f"; mean and std over seeds {seeds}" if len(seeds) > 1 else "") + f"\n{GD.summary(df).to_string()}")
    clean = df[df["segment"] == AM.CLEAN]
    clean = clean.assign(series=clean["mode"] + "/" + clean["series"])
    print("\nclean accuracy (%)\n" + (100 * clean.groupby("series")["accuracy"].mean()).round(2).to_string())

    dest = out if args.num_samples else ctx.g.path("paper")
    write_table(dest / f"tab_gated_duos__{tag}.tex", GD.render_table(
        df, ctx, duo_ctx, tta, temps, proxy=proxy, beta=args.beta, b_t=args.b_t, label=f"tab:gated_duos:{tag}",
        source="scripts/03b_tab_gated_duos.py"))
    print(f"\nwrote {out / f'{tag}.csv'} and {dest / f'tab_gated_duos__{tag}.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
