"""
49_proxy_study.py
=================
Why the reset detector watches the nuclear norm: re-run the held-out calibration streams of the reset tuning with every candidate proxy logged
(src/reset/proxy_study.py), then replay the detector's screen on each proxy's sequence.

    python scripts/49_proxy_study.py plan   --duos vitb16_rn50v1 vitb16_convnextt     # the runs, and which are done
    python scripts/49_proxy_study.py submit --duos vitb16_rn50v1 vitb16_convnextt     # sbatch the missing ones (not those already queued)
    python scripts/49_proxy_study.py run    --duo vitb16_rn50v1 --name S --rotation 0 --seed 0      # what a job runs (GPU)
    python scripts/49_proxy_study.py report --duos vitb16_rn50v1 vitb16_convnextt     # CPU: correlation table, screen, tables, figures

`--seeds 0 1` re-runs both stress seeds the tuning used (default: seed 0 only, 4 stress + 2 healthy streams per duo). Outputs:
<out>/results/proxy_study/<duo>/ (the runs, source_stats.json, screen.json, summary.csv, detect.csv, track.csv, traces_*.png) and
docs/tables_sprint/proxy_study.tex (detection screen), docs/tables_sprint/proxy_correlation.tex (correlation with accuracy, correlation.csv).
Healthy streams are run at every rotation (PS.HEALTHY_ROTATIONS; the tuning ran 0 and 2).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reset import jobs as J               # noqa: E402
from src.reset import proxy_study as PS       # noqa: E402

TABLE = Path("docs/tables_sprint/proxy_study.tex")
CORR_TABLE = Path("docs/tables_sprint/proxy_correlation.tex")
DUOS = ["vitb16_rn50v1", "vitb16_convnextt", "convnextb_deits"]


def _jobs(duo: str, seeds) -> list[dict]:
    return PS.panel_jobs(duo, seeds)


def _name(duo: str, a: dict) -> str:
    return f"ps_{duo.split('_')[-1][:8]}_{a['name']}r{a['rotation']}s{a['seed']}"


def plan(duos, seeds, go: bool, nice: int = 0) -> int:
    from src.reset.layout import Paths
    from src.utils.config import load_global
    g = load_global()
    queued = J.queued_names() if go else set()
    n = 0
    for duo in duos:
        P = Paths(g, duo)
        for a in _jobs(duo, seeds):
            out, name = PS.run_path(g, P, a), _name(duo, a)
            if out.is_file():
                print(f"done     {out.name}")
                continue
            if name in queued:
                print(f"queued   {name}")
                continue
            argv = ["sbatch", "-t", PS.TIME, "-J", name, "--qos=high", f"--exclude={J.EXCLUDE}"] + ([f"--nice={nice}"] if nice else []) + ["slurm/job.sh", "scripts/49_proxy_study.py", "run",
                    "--duo", duo, "--name", a["name"], "--rotation", str(a["rotation"]), "--seed", str(a["seed"])]
            print(("SUBMIT  " if go else "to run  ") + " ".join(argv))
            if go:
                r = subprocess.run(argv, capture_output=True, text=True)
                print("   ->", (r.stdout or r.stderr).strip())
                n += r.returncode == 0
    return n


def run(duo: str, name: str, rotation: int, seed: int, device) -> int:
    match = [a for a in _jobs(duo, (seed,)) if a["name"] == name and a["rotation"] == rotation]
    if len(match) != 1:
        raise SystemExit(f"{duo}: no calibration stream {name} rotation {rotation} seed {seed} in the reset tuning")
    PS.run_one(duo, match[0], device)
    return 0


def report(duos, budget: float) -> int:
    import pandas as pd

    from src.utils import provenance
    from src.utils.config import load_global
    from src.utils.config import load_duo
    g = load_global()
    tables, names, corr = {}, {}, {}
    for duo in duos:
        d = PS.study_dir(g, duo)
        runs = sorted(p for p in d.glob("exp-*.npz") if not p.name.endswith("__frozen.npz"))
        if not runs:
            print(f"{duo}: no panel runs in {d}")
            continue
        import numpy as np
        b_t = int(json.loads(str(np.load(runs[0])["meta"]))["b_t"])
        streams = [PS.PanelStream.load(r, b_t, r.stem.split("__seed")[0].replace("exp-", "")) for r in runs]
        cfg = load_duo(duo, g)
        names[duo] = tuple(provenance.model_name(cfg[s]["NAME"]).split(" (")[0] for s in ("LARGE", "SMALL"))
        print(f"\n== {duo}: {' + '.join(names[duo])}")
        cr = PS.correlations(streams)
        corr[duo] = cr
        pd.DataFrame(cr).to_csv(d / "correlation.csv", index=False)
        print(f"correlation with accuracy (Spearman within stream, mean over streams; {cr[0]['H_n']} healthy, {cr[0]['S_n']} collapsing):")
        print(pd.DataFrame(cr)[["label", "H_gap", "H_large", "H_small", "S_gap", "S_large", "S_small", "S_gap_min", "H_ahead", "S_ahead"]].round(2).to_string(index=False))
        res = PS.screen(streams)
        (d / "screen.json").write_text(json.dumps(res, default=float))
        pd.DataFrame(res["detect"]).to_csv(d / "detect.csv", index=False)
        pd.DataFrame(res["track"]).to_csv(d / "track.csv", index=False)
        tab = PS.summary_table(res, budget)
        pd.DataFrame(tab).to_csv(d / "summary.csv", index=False)
        tables[duo] = tab
        pd.set_option("display.width", 220)
        print(f"budget {budget:g} false alarms / 1,000 proxy batches (delay in proxy batches of {b_t} images):")
        print(pd.DataFrame(tab)[["label", "orientation", "delay", "delay_detected", "miss_rate", "fa", "delay_large", "delay_small", "delay_plus",
                                 "spearman_gap", "response", "response_falls"]].round(3).to_string(index=False))
        for st in streams:
            if st.name.startswith("S"):
                print("figure:", PS.plot_traces(st, d / f"traces_{st.name}.png", names[duo], b_t))
    if corr:
        CORR_TABLE.parent.mkdir(parents=True, exist_ok=True)
        CORR_TABLE.write_text(PS.correlation_latex(corr, names, 256))
        print(f"wrote {CORR_TABLE}")
    if tables:
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        TABLE.write_text(PS.latex_table(tables, budget, 256, names))
        print(f"wrote {TABLE}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "submit", "run", "report"])
    ap.add_argument("--duos", nargs="+", default=DUOS)
    ap.add_argument("--duo")
    ap.add_argument("--name", choices=["S", "H"])
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(PS.SEEDS))
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--nice", type=int, default=0, help="sbatch --nice (lower priority), e.g. 500")
    a = ap.parse_args()
    if a.action in ("plan", "submit"):
        n = plan(a.duos, a.seeds, a.action == "submit", a.nice)
        if a.action == "submit":
            print(f"{n} job(s) submitted")
        return 0
    if a.action == "run":
        if not (a.duo and a.name):
            ap.error("run needs --duo and --name")
        return run(a.duo, a.name, a.rotation, a.seed, a.device)
    return report(a.duos, a.budget)


if __name__ == "__main__":
    raise SystemExit(main())
