#!/usr/bin/env python3
"""
Tables and figures of the live reset runs (scripts/38_run_reset_live.py; src/reset/live_report.py). Reads the stored runs of a stream and writes, in
<paths.out>/results/reset_live/<duo>__tent_bn/report/:

    table__<stream>.csv / .json / .tex     one row per condition: whole-stream accuracy (mean, and the spread over runs) of Vanilla DD, the fixed-TS duo, both
                                           members; the gain over the frozen duo; resets per 1,000 proxy batches and per member; the share of resets that hit a
                                           member that was not below its frozen copy (the audit of live_report.py). The .tex is the tab:reset_live table.
    live__<condition>__<stream>[__rot<K>]__seed<s>.png/.pdf     the B1-style figure of one run: accuracy of the frozen duo, the fixed-TS duo, Vanilla DD and both
                                           members, the gate's weight, the proxy scores, and every reset marked (triangles on the accuracy panel, lines on the weight panel)
    overlay__<stream>[__rot<K>]__seed<s>.png/.pdf    every condition on one figure: Vanilla DD, the ViT and the ResNet over the same stream

    python scripts/39_tab_reset.py --duo vitb16_rn50v1 --bn --stream heldout --rotations 0 2 --conditions none boundary fixed1000 own_fast_vs_slow ...
    python scripts/39_tab_reset.py --duo vitb16_rn50v1 --bn --stream test --seeds 0 1 2            # every condition with a stored run

`--conditions` default: every condition that has a run for every (seed, rotation) asked for. `--figures none|first|all`: per-condition figures for no run,
the first (seed, rotation), or every run (the overlay is drawn for every (seed, rotation) either way; `none` skips it too).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.experiments import continual as RC
from src.experiments import continual_report as RR
from src.experiments import gate_stress as GS
from src.experiments.cli import add_common_args, bn_tag, setup
from src.reset import live_report as LV
from src.utils import provenance

SIDES = ("large", "small")


def rate_str(lr: float) -> str:
    return GS.rate_str(lr)


def order(conds: list[str]) -> list[str]:
    """none, the rules, the fixed schedules (by period), the references: the order of the table."""
    def key(c: str):
        if c == "none":
            return (0, 0, c)
        if c == "oracle":
            return (4, 0, c)
        if c == "boundary":
            return (3, 0, c)
        if re.fullmatch(r"fixed\d+", c):
            return (2, int(c[5:]), c)
        return (1, 0, c)
    return sorted(conds, key=key)


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter), tta=False)
    ap.add_argument("--stream", choices=("test", "heldout"), required=True)
    ap.add_argument("--rotations", type=int, nargs="+", default=[0], metavar="K")
    ap.add_argument("--conditions", nargs="+", default=None, help="default: every condition with a run for every (seed, rotation)")
    ap.add_argument("--lrs", type=float, nargs=2, default=[1.4e-4, 5e-4], metavar=("LR_L", "LR_S"))
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--b_t", type=int, default=256)
    ap.add_argument("--max_images", type=int, default=None, help="name of a smoke run (38's --max_images)")
    ap.add_argument("--figures", choices=("none", "first", "all"), default="first")
    args = ap.parse_args()
    tta = "tent"

    tag = f"{args.duo}__{tta}{bn_tag(args)}"
    ctx, seeds, out = setup(args, f"reset_live/{tag}", default_seeds=[0])
    proxy = args.proxy or ctx.g.gate["proxy"]
    proxy = {"nuclear_norm": "nnn", "ac_mc": "msp"}.get(proxy, proxy)
    report = out / "report"
    report.mkdir(parents=True, exist_ok=True)
    names = {s: provenance.model_name(ctx.duo[s.upper()]["NAME"]).split(" (")[0] for s in SIDES}
    lrs = tuple(args.lrs)
    keys = [(s, r) for r in args.rotations for s in seeds]

    def stem(cond: str, seed: int, rot: int) -> str:
        return LV.run_stem(cond, stream=args.stream, rot=rot, seed=seed, lrs=lrs, proxy=proxy, b_t=args.b_t, beta=args.beta, max_images=args.max_images)

    def path_of(cond: str, seed: int, rot: int) -> Path:
        return out / f"{stem(cond, seed, rot)}.npz"

    conds = args.conditions
    if conds is None:
        found = set()
        for f in glob.glob(str(out / "live-*.npz")):
            if f.endswith("__frozen.npz"):
                continue
            m = re.match(r"live-(?P<c>[A-Za-z0-9_]+?)(__heldout)?(__rot\d+)?(__L[\d.e+-]+_S[\d.e+-]+)?(__[a-z]+_bt\d+_beta[\d.]+)?(__first\d+)?__seed\d+\.npz$", Path(f).name)
            if m and m["c"] != "frozen":
                found.add(m["c"])
        conds = [c for c in order(sorted(found)) if all(path_of(c, s, r).is_file() for s, r in keys)]
        if not conds:
            raise SystemExit(f"no condition has a stored run for every seed/rotation {keys} in {out}: run scripts/38_run_reset_live.py first")
    conds = [c for c in conds if c != "frozen"]
    missing = [str(path_of(c, s, r)) for c in conds for s, r in keys if not path_of(c, s, r).is_file()] + \
              [str(path_of("frozen", s, r)) for s, r in keys if not path_of("frozen", s, r).is_file()]
    if missing:
        print("runs missing (see scripts/38_run_reset_live.py):\n  " + "\n  ".join(missing))
        return 1

    runs, frozen = {}, {}
    for s, r in keys:
        frozen[(s, r)] = RC.load_run(path_of("frozen", s, r))
        for c in conds:
            runs[(c, s, r)] = RC.load_run(path_of(c, s, r))
    summaries = {c: [LV.summarize(runs[(c, s, r)], frozen[(s, r)], args.b_t) for s, r in keys] for c in conds}
    rows = LV.table_rows(summaries)
    frozen_duo = [float(frozen[k]["ts"].mean() * 100) for k in keys]
    where = f"{args.stream}{' rotations ' + ','.join(map(str, args.rotations)) if args.stream == 'heldout' else ''}, seeds {','.join(map(str, seeds))}"
    print(f"\n== {where}: whole-stream accuracy (%), mean over {len(keys)} run(s); frozen duo {sum(frozen_duo) / len(frozen_duo):.1f}")
    print(f"{'condition':26s} {'duo':>6s} {'fixedTS':>8s} {'ViT':>6s} {'ResNet':>7s} {'vs frozen':>10s}   resets/1000 batches (ViT, ResNet)   false")
    for row in rows:
        sd = f" ±{row['gate_sd']:.1f}" if row.get("gate_sd") else ""
        share = row["false_share"]
        false = "-" if share != share else f"{100 * share:.0f}%"
        print(f"{LV.pretty(row['condition']):26s} {row['gate']:6.1f}{sd:6s} {row['ts']:8.1f} {row['large']:6.1f} {row['small']:7.1f} {row['gain_vs_frozen_duo']:+10.1f}"
              f"   {row['resets_per_1000_large']:6.1f}, {row['resets_per_1000_small']:6.1f}   {false}")
    stem_out = f"table__{args.stream}" + (f"__rot{''.join(map(str, args.rotations))}" if args.stream == "heldout" else "")
    pd.DataFrame(rows).to_csv(report / f"{stem_out}.csv", index=False)
    (report / f"{stem_out}.json").write_text(json.dumps({"stream": args.stream, "rotations": args.rotations, "seeds": seeds, "lrs": list(lrs), "frozen_duo": frozen_duo,
                                                         "rows": rows, "per_run": {c: s for c, s in summaries.items()}}, indent=1))
    stream_text = "held-out stream (4 shifts $\\times$ 4 rounds)" if args.stream == "heldout" else "test stream (15 corruptions)"
    spread = f", mean $\\pm$ std over {len(keys)} runs" if len(keys) > 1 else ""
    caption = (f"Vanilla DD with Tent, {names['large']} at {rate_str(lrs[0])} and {names['small']} at {rate_str(lrs[1])} (both collapse without a reset), over the "
               f"{stream_text}; mean accuracy (\\%) over the whole stream{spread}. Resets per 1,000 proxy batches; false: resets of a member not below its frozen copy.")
    (report / f"{stem_out}.tex").write_text(LV.latex_table(rows, caption, f"tab:reset_live_{args.stream}", frozen=sum(frozen_duo) / len(frozen_duo)))
    print(f"wrote {report / stem_out}.csv/.json/.tex")

    if args.figures == "none":
        return 0
    footer_base = (f"{where}; Vanilla DD, Tent; gate {proxy}, b_t {args.b_t}, beta {args.beta:g}; {names['large']} {rate_str(lrs[0])}, {names['small']} {rate_str(lrs[1])}")
    for i, (s, r) in enumerate(keys):
        fr = frozen[(s, r)]
        suffix = f"{args.stream}{f'__rot{r}' if r else ''}__seed{s}"
        footer = provenance.footer(ctx.duo, tta, args.bn, footer_base.replace(where, f"{args.stream}{f' rotation {r}' if args.stream == 'heldout' else ''}, seed {s}"))
        for c in (conds if (args.figures == "all" or i == 0) else []):
            run = runs[(c, s, r)]
            records = RR.load_gate_records(path_of(c, s, r))
            n = len(run["large"])
            an = GS.analyze(run, {k: v[:n] if hasattr(v, "__len__") and k in ("ts", "large", "small") else v for k, v in fr.items()}, records, args.b_t)
            item = {"title": f"{LV.pretty(c)}: {names['large']} at {rate_str(lrs[0])}, {names['small']} at {rate_str(lrs[1])}", "run": run, "frozen": fr, "records": records,
                    "b_t": args.b_t, "analysis": an, "names": names, "beta": args.beta, "resets": run["meta"].get("resets", [])}
            GS.plot_stress([item], report / f"live__{c}__{suffix}.png", footer=footer)
        title = f"Vanilla DD with Tent over the {args.stream} stream: {names['large']} at {rate_str(lrs[0])}, {names['small']} at {rate_str(lrs[1])}, every reset condition"
        written = LV.plot_overlay({c: runs[(c, s, r)] for c in conds}, fr, report / f"overlay__{suffix}.png", names=names, title=title, footer=footer)
        print("wrote " + ", ".join(p.name for p in written))
    print(f"figures in {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
