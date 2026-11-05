#!/usr/bin/env python3
"""
50_indep_dd_beta_sweep.py
=========================
Why beta = 0.25: Independent DD (the gate outside the adaptation loop) at several betas, replayed on the cached logits of script 03's
Independent both runs (no model runs, CPU only), on the 4 held-out corruptions (where beta is chosen) and the 15 test corruptions
(episodic ImageNet-C, severity 5, clean excluded), seeds 0 1 2. beta = 0 is exactly the Fixed TS duo (every weight 0.5, log pool).

    sbatch -t 01:00:00 slurm/job_cpu.sh scripts/50_indep_dd_beta_sweep.py

Each duo is read at the criterion its adaptation table uses (vitb16_rn50v1: mean; vitb16_convnextt: worst_shift). Writes
<out>/results/indep_dd_beta_sweep/<duo>.csv (one row per beta, seed, stream) and docs/tables_sprint/indep_dd_beta_sweep.tex.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiments import adaptation_modes as AM  # noqa: E402
from src.experiments.cli import add_common_args, setup  # noqa: E402
from src.experiments.runner import replay  # noqa: E402
from src.experiments.specs import GateSpec  # noqa: E402
from src.utils import provenance  # noqa: E402

DUOS = {"vitb16_rn50v1": "mean", "vitb16_convnextt": "worst_shift"}
BETAS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
TABLE = Path("docs/tables_sprint/indep_dd_beta_sweep.tex")


def replay_store(spec, ctx, gates, segments):
    """`replay` on the tuners' per-member LogitStore (where 02c put the held-out runs of every seed) instead of script 03's cache."""
    from src.experiments.logit_runs import run_member_specs
    from src.experiments.logit_store import LogitStore, PairView
    from src.experiments.runner import _finish, _new_eval
    pv = PairView(LogitStore.from_context(ctx), *run_member_specs(spec, ctx), [g.file_stem for g in segments])
    missing = set(g.file_stem for g in segments) - set(pv.stems())
    if missing:
        raise FileNotFoundError(f"store lacks {sorted(missing)}")
    data = {g.file_stem: pv.read(g.file_stem) for g in segments}
    total = sum(d["n"] for d in data.values())
    ev = _new_eval(gates, ctx, segments)
    for g in segments:
        d = data[g.file_stem]
        ev.start_segment(g.label, g.dataset, reset=g.reset_before, n_samples=d["n"], stream_total=total)
        i = 0
        for b in d["batch_sizes"]:
            b = int(b)
            ev.update(d["z_l"][i:i + b], d["z_s"][i:i + b], d["labels"][i:i + b])
            i += b
        ev.end_segment()
    return _finish(ev, spec)


def sweep(duo: str, criterion: str, seeds, b_t: int, splits=("heldout", "test")) -> pd.DataFrame:
    ap = add_common_args(argparse.ArgumentParser(), tta=False)
    args = ap.parse_args(["--duo", duo, "--bn", "--criterion", criterion, "--seeds", *map(str, seeds)])
    ctx, seeds, out = setup(args, "indep_dd_beta_sweep")
    temps = AM.resolve_temperatures(ctx, AM.temperature_spec(duo, "tent", None, None), None, report_only=True)
    proxy = {"nuclear_norm": "nnn"}.get(ctx.g.gate["proxy"], ctx.g.gate["proxy"])
    print(f"{duo} ({criterion}): T_L {temps.T_l}, T_S {temps.T_s}; rates " +
          ", ".join(f"{s.lower()} {ctx.duo[s]['OPTIM']['LR']:g}" for s in ("LARGE", "SMALL")), flush=True)
    gates = [GateSpec.from_global(ctx.g, f"beta{b:g}", proxy=proxy, beta=b, proxy_batch_size=b_t, fixed_ts=str(temps.folder))
             for b in BETAS]
    jobs = AM.mode_jobs(duo, "tent", None, seeds, None, temps, ["indep_duo"])
    rows = []
    for _, spec in jobs:
        for split, segs in (("heldout", AM.heldout_segments(ctx)), ("test", None)):
            if split not in splits:
                continue
            try:
                res = replay_store(spec, ctx, gates, segs) if split == "heldout" else replay(spec, ctx, gates, segs)
            except FileNotFoundError:
                print(f"  seed {spec.seed} {split}: member logits not cached, skipped", flush=True)
                continue
            for r in res.rows:
                if r["series"].startswith("beta") and r["segment"] != AM.CLEAN:
                    rows.append(dict(duo=duo, split=split, seed=spec.seed, beta=float(r["series"][4:]),
                                     segment=r["segment"], accuracy=r["accuracy"]))
            print(f"  seed {spec.seed} {split}: done", flush=True)
    df = pd.DataFrame(rows)
    f = out / f"{duo}.csv"
    if f.is_file():             # keep the other split's rows from an earlier call
        old = pd.read_csv(f)
        df = pd.concat([old[~old.split.isin(list(splits))], df])
    df.to_csv(f, index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    per_seed = df.groupby(["duo", "split", "beta", "seed"])["accuracy"].mean().mul(100)
    worst = df.groupby(["duo", "split", "beta", "seed"])["accuracy"].min().mul(100)
    s = per_seed.groupby(["duo", "split", "beta"]).agg(["mean", "std"])
    s["worst"] = worst.groupby(["duo", "split", "beta"]).mean()
    return s.reset_index()


def latex(s: pd.DataFrame, names: dict) -> str:
    duos = list(DUOS)
    cols = [(d, sp) for d in duos for sp in ("heldout", "test")]
    best = {c: s[(s.duo == c[0]) & (s.split == c[1])]["mean"].max() for c in cols}
    lines = [r"\begin{table}[t]", r"    \centering", r"    \small",
             r"    \caption{Independent DD at several gate sharpnesses $\beta$: top-1 accuracy (\%) averaged over the corruptions at severity 5, "
             r"episodic, Tent, mean over SEEDS_HELDOUT. The gate $w_L = \sigma(\beta(s_L - s_S))$ (nnn, proxy batches of 256) is applied outside "
             r"the adaptation loop, so every $\beta$ is replayed on the same member logits. $\beta = 0$ is the Fixed TS duo. "
             r"\emph{Held-out}: the 4 tuning corruptions; \emph{Test}: the 15 test corruptions. Best per column in bold.}",
             r"    \label{tab:indep_dd_beta_sweep}",
             r"    \begin{tabular}{l" + "cc" * len(duos) + "}", r"        \toprule",
             "        & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{' + '.join(names[d])}}}" for d in duos) + r" \\",
             "        " + "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(duos))),
             r"        $\beta$ & " + " & ".join("Held-out & Test" for _ in duos) + r" \\", r"        \midrule"]
    for b in BETAS:
        cells = []
        for c in cols:
            v = s[(s.duo == c[0]) & (s.split == c[1]) & (s.beta == b)]["mean"].iloc[0]
            t = f"{v:.1f}"
            cells.append(rf"\textbf{{{t}}}" if f"{v:.1f}" == f"{best[c]:.1f}" else t)
        lab = r"0 (Fixed TS)" if b == 0 else f"{b:g}"
        lines.append(f"        {lab} & " + " & ".join(cells) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--b_t", type=int, default=256)
    ap.add_argument("--duos", nargs="+", default=list(DUOS), help="sweep only these (the table is written once every duo's CSV exists)")
    ap.add_argument("--splits", nargs="+", default=["heldout", "test"], choices=["heldout", "test"])
    ap.add_argument("--table_only", action="store_true", help="no replay: build the table from the duos' CSVs")
    a = ap.parse_args()
    from src.utils.config import load_duo, load_global
    g = load_global()
    if not a.table_only:
        for d in a.duos:
            sweep(d, DUOS[d], a.seeds, a.b_t, a.splits)
    out = g.path("out") / "results" / "indep_dd_beta_sweep"
    if not all((out / f"{d}.csv").is_file() for d in DUOS):
        print("not every duo swept yet: no table")
        return 0
    df = pd.concat([pd.read_csv(out / f"{d}.csv") for d in DUOS])
    print("seeds per column: " + ", ".join(f"{d} {sp}: {sorted(x.seed.unique())}" for (d, sp), x in df.groupby(["duo", "split"])))
    s = summarize(df)
    pd.set_option("display.width", 200)
    print(s.round(2).to_string(index=False))
    names = {d: tuple(provenance.model_name(load_duo(d, g)[k]["NAME"]).split(" (")[0] for k in ("LARGE", "SMALL")) for d in DUOS}
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    hs = sorted(df[df.split == "heldout"].seed.unique())
    TABLE.write_text(latex(s, names).replace("SEEDS_HELDOUT", "seed 0 (held-out) and 3 seeds (test)" if hs == [0] else f"{len(hs)} seeds"))
    print(f"wrote {TABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
