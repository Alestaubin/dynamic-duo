#!/usr/bin/env python3
"""
Ablation of the score scale of the gate: what does the logit buy over other ways of comparing the two members' scores?

The gate compares the members' proxy scores as w_L = sigmoid(beta (x_L - x_S)) with x = logit(clip(score, eps, 1 - eps)) (eq. 8 and 15; the
identity calibration map, no temporal filter). Here x = link(clip(score)) with the link one of `dd_eval.LINKS`:

    logit      log(a / (1 - a))          the paper's
    identity   a                         no transform: the raw score gap
    log        log a                     half of the logit (stretches low scores)
    neglog1m   -log(1 - a)               the other half (stretches high scores)
    probit     Phi^-1(a)                 normal quantile
    arcsin     arcsin(sqrt(a))           variance-stabilizing

Every link is increasing, so they all send the same member to the higher weight; they differ in how a score gap is stretched, which is
what beta has to compensate. The links therefore live on different scales and each gets its own sweep of beta over a wide geometric grid
(default 2^-4 .. 2^9, from nearly the fixed combination to hard selection); compare the links at their BEST beta, and at beta = 0.5, the
paper's, to see the cost of leaving the beta untuned.

Replay only, no model: the stored per-member logits of scripts/04_cache_logits.py (independently adapted members, the duo's tuned rates for
--tta), on the HELD-OUT streams, the only ones a choice may be made on. Same bucket rule, temperatures (the Fixed TS of the frozen models,
use --bn as the tables do) and metrics as scripts/07_eval_duo_grid.py. Per (proxy, b_t, link, beta): accuracy, ECE, NLL; the weights'
mean, spread and saturation; against the members, the fixed combination (beta = 0) and the two oracles at the same b_t; per link and
proxy-batch size, how the score gap tracks the true log-odds gap of the batch (Spearman, Pearson, pooled over the held-out buckets).

    python scripts/48_ablate_gate_link.py --duo vitb16_rn50v1 --tta tent --bn --seeds 0
    python scripts/48_ablate_gate_link.py ... --proxies nnn --b_ts 128 --links logit identity --betas 0.25 0.5 1 2 4
    python scripts/48_ablate_gate_link.py ... --lr_large 2e-4 --lr_small 5e-4     # a stress pair: this ViT collapses on spatter, the gate has work

Writes <out>/<duo>__<tta>__seed<S>{_bn}.{rows.csv,summary.csv,gaps.csv,png}; --plot_only redraws the figure from the rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from src.experiments import adaptation_modes as AM
from src.experiments import cli
from src.experiments.dd_eval import LINKS, Scorer, bucket_scores, gate_weights, load_stream, member_metrics, plan_buckets, sample_weights
from src.experiments.grids import sgd_optim
from src.experiments.logit_runs import run_member_specs, stream_segments, with_optim
from src.experiments.logit_store import LogitStore, PairView
from src.experiments.specs import RunSpec
from src.reliability.proxies.pure import PROXIES

# One colour per link, fixed (the first six slots of the reference categorical palette, light mode: the colour follows the link, never its rank).
LINK_COLOR = dict(zip(LINKS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BETAS = tuple(2.0 ** k for k in range(-4, 10))
TIE = 1e-4                                  # accuracy ties within 0.01 points go to the smaller beta (scripts/10_select_joint_dd_beta.py)
SAT = 0.05                                  # a weight within SAT of 0 or 1 is saturated
CHUNK = 24                                  # weight vectors scored at once (memory)


def lg(a: np.ndarray, eps: float) -> np.ndarray:
    a = np.clip(a, eps, 1 - eps)
    return np.log(a / (1 - a))


def evaluate_stream(sd: dict, T: tuple[float, float], proxies, b_ts, links, betas, eps: float, device, oracles: bool) -> tuple[list, list]:
    """(rows, buckets): the metric rows of one stream and, per (proxy, b_t), its per-bucket scores and accuracies."""
    z_l, z_s, y = sd["z_l"].to(device).float(), sd["z_s"].to(device).float(), sd["labels"].to(device).long()
    n, stem = len(y), sd["stem"]
    sc = Scorer(z_l, z_s, y, *T)
    rows, buckets = [], []

    def add(series, res, **kw):
        rows.append({"stream": stem, "series": series, "proxy": None, "b_t": None, "link": None, "beta": None, **kw, **res})

    add("large", member_metrics(z_l, y))
    add("small", member_metrics(z_s, y))
    fixed = sc.metrics(torch.full((1, n), 0.5, device=device))[0]
    add("fixed", fixed, beta=0.0)
    if oracles:
        w_grid = torch.linspace(0.0, 1.0, 101, device=device)
        nll_grid = sc.nll_per_sample(w_grid[:, None].expand(-1, n))
    ok_l, ok_s = (z_l.argmax(1) == y).float(), (z_s.argmax(1) == y).float()
    moving = [b for b in betas if b != 0]

    for b_t in b_ts:
        plan = plan_buckets(sd["batch_sizes"], n, b_t)
        bucket = torch.from_numpy(plan.bucket).to(device)
        size = torch.bincount(bucket, minlength=plan.n_buckets).float()
        acc_l = (torch.bincount(bucket, weights=ok_l, minlength=plan.n_buckets) / size).cpu().numpy()
        acc_s = (torch.bincount(bucket, weights=ok_s, minlength=plan.n_buckets) / size).cpu().numpy()
        if oracles:
            w_select = torch.as_tensor(acc_l >= acc_s, device=device).float()[bucket]
            nll_bucket = torch.zeros(len(w_grid), plan.n_buckets, device=device).index_add_(1, bucket, nll_grid)
            res = sc.metrics(torch.stack([w_select, w_grid[nll_bucket.argmin(0)][bucket]]))
            add("oracle_select", res[0], b_t=b_t)
            add("oracle_w", res[1], b_t=b_t)
        for proxy in proxies:
            r_l, r_s = bucket_scores(proxy, z_l, z_s, plan, None, None)
            buckets.append({"stream": stem, "proxy": proxy, "b_t": b_t, "r_l": r_l, "r_s": r_s, "acc_l": acc_l, "acc_s": acc_s})
            cells = [(link, beta) for link in links for beta in moving]
            ws = {c: gate_weights(r_l, r_s, c[1], eps, c[0]) for c in cells}
            res = []
            for i in range(0, len(cells), CHUNK):
                res += sc.metrics(np.stack([sample_weights(ws[c], plan) for c in cells[i:i + CHUNK]]))
            for (link, beta), r in zip(cells, res):
                w = ws[(link, beta)]
                add("dd", r, proxy=proxy, b_t=b_t, link=link, beta=beta, w_mean=float(w.mean()), w_sd=float(w.std()),
                    w_sat=float(((w < SAT) | (w > 1 - SAT)).mean()))
            for link in links:                                                   # beta = 0 is the fixed combination, whatever the link
                add("dd", fixed, proxy=proxy, b_t=b_t, link=link, beta=0.0, w_mean=0.5, w_sd=0.0, w_sat=0.0)
    return rows, buckets


def pooled_gap_tracking(buckets: list[dict], links, eps: float) -> pd.DataFrame:
    """Per (proxy, b_t, link): how x_L - x_S tracks the true log-odds gap of the buckets, pooled over the streams. Ordering accuracy is the
    same for every link (all increasing) and is kept as the check of that."""
    out = []
    for (proxy, b_t) in sorted({(b["proxy"], b["b_t"]) for b in buckets}):
        sel = [b for b in buckets if b["proxy"] == proxy and b["b_t"] == b_t]
        cat = {k: np.concatenate([b[k] for b in sel]) for k in ("r_l", "r_s", "acc_l", "acc_s")}
        true = lg(cat["acc_l"], eps) - lg(cat["acc_s"], eps)
        nz = cat["acc_l"] != cat["acc_s"]
        for link in links:
            f = LINKS[link]
            gap = f(np.clip(cat["r_l"], eps, 1 - eps)) - f(np.clip(cat["r_s"], eps, 1 - eps))
            ok = gap.std() > 0 and true.std() > 0
            out.append({"proxy": proxy, "b_t": b_t, "link": link, "n_buckets": len(gap), "gap_sd": float(gap.std()),
                        "spearman": float(spearmanr(gap, true).statistic) if ok else float("nan"),
                        "pearson": float(pearsonr(gap, true).statistic) if ok else float("nan"),
                        "ordering_acc": float(np.mean(np.sign(gap[nz]) == np.sign(true[nz]))) if nz.any() else float("nan")})
    return pd.DataFrame(out)


def summarize(rows: pd.DataFrame, links) -> pd.DataFrame:
    """Per (proxy, b_t, link): the best beta by macro accuracy (ties to the smaller beta), its metrics against the fixed combination
    and the oracles, the paper's beta = 0.5, and on how many streams the link beats the logit at their respective best betas."""
    dd = rows[rows.series == "dd"].astype({"b_t": int})
    macro = dd.groupby(["proxy", "b_t", "link", "beta"])[["accuracy", "nll", "ece", "w_mean", "w_sd", "w_sat"]].mean().reset_index()
    worst = dd.groupby(["proxy", "b_t", "link", "beta"]).accuracy.min().rename("worst").reset_index()
    macro = macro.merge(worst, on=["proxy", "b_t", "link", "beta"])
    ref = rows[rows.series.isin(["fixed", "large", "small"])].groupby("series")[["accuracy", "nll"]].mean()
    orc = rows[rows.series.isin(["oracle_select", "oracle_w"])].astype({"b_t": int}).groupby(["series", "b_t"]).accuracy.mean()
    best = {}
    for (proxy, b_t, link), d in macro[macro.beta > 0].groupby(["proxy", "b_t", "link"]):
        top = d.accuracy.max()
        best[(proxy, b_t, link)] = d[d.accuracy >= top - TIE].sort_values("beta").iloc[0]
    per_stream = lambda proxy, b_t, link, beta: dd[(dd.proxy == proxy) & (dd.b_t == b_t) & (dd.link == link) & (dd.beta == beta)].set_index("stream").accuracy   # noqa: E731
    out = []
    for (proxy, b_t, link), r in best.items():
        paper = macro[(macro.proxy == proxy) & (macro.b_t == b_t) & (macro.link == link) & (macro.beta == 0.5)]
        lo = best.get((proxy, b_t, "logit"))
        beat = None if lo is None else int((per_stream(proxy, b_t, link, r.beta) > per_stream(proxy, b_t, "logit", lo.beta) + 1e-12).sum())
        out.append({"proxy": proxy, "b_t": b_t, "link": link, "best_beta": r.beta, "accuracy": r.accuracy, "nll": r.nll, "ece": r.ece,
                    "worst_stream": r.worst, "gain_over_fixed": r.accuracy - ref.loc["fixed", "accuracy"],
                    "gap_to_oracle_w": orc[("oracle_w", b_t)] - r.accuracy if ("oracle_w", b_t) in orc.index else np.nan,
                    "w_sd": r.w_sd, "w_sat": r.w_sat, "streams_beating_logit": beat,
                    "acc_at_beta_0.5": float(paper.accuracy.iloc[0]) if len(paper) else np.nan,
                    "nll_at_beta_0.5": float(paper.nll.iloc[0]) if len(paper) else np.nan})
    return pd.DataFrame(out)


def plot(rows: pd.DataFrame, summary: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dd = rows[rows.series == "dd"].astype({"b_t": int})
    fixed = rows[rows.series == "fixed"].accuracy.mean()
    macro = dd.groupby(["proxy", "b_t", "link", "beta"]).accuracy.mean().reset_index()
    proxies, b_ts = sorted(macro.proxy.unique()), sorted(macro.b_t.unique())
    links = [l for l in LINKS if l in set(macro.link)]
    fig, axes = plt.subplots(len(proxies), len(b_ts), figsize=(max(9.0, 4.8 * len(b_ts)), 3.7 * len(proxies)), squeeze=False, sharex=True, facecolor=SURFACE)
    for i, proxy in enumerate(proxies):
        for j, b_t in enumerate(b_ts):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            for link in links:
                d = macro[(macro.proxy == proxy) & (macro.b_t == b_t) & (macro.link == link) & (macro.beta > 0)].sort_values("beta")
                ax.plot(d.beta, 100 * (d.accuracy - fixed), color=LINK_COLOR[link], lw=2, label=link, solid_capstyle="round", zorder=3)
                s = summary[(summary.proxy == proxy) & (summary.b_t == b_t) & (summary.link == link)]
                if len(s):
                    ax.plot([s.best_beta.iloc[0]], [100 * s.gain_over_fixed.iloc[0]], marker="o", ms=9, color=LINK_COLOR[link], mec=SURFACE, mew=2, zorder=5)
            ax.axhline(0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
            ow = rows[(rows.series == "oracle_w") & (rows.b_t == b_t)].accuracy.mean()    # NaN without oracles
            if np.isfinite(ow):
                ax.axhline(100 * (ow - fixed), color=INK2, lw=1, ls=":", zorder=2)
            ax.axvline(0.5, color=GRID, lw=1.5, zorder=1)
            ax.set_xscale("log", base=2)
            ax.set_title(f"{proxy}, b_t = {b_t}", fontsize=10, color=INK, loc="left")
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8)
            if j == 0:
                ax.set_ylabel("accuracy - fixed TS (points)", color=INK2, fontsize=9)
            if i == len(proxies) - 1:
                ax.set_xlabel("beta", color=INK2, fontsize=9)
    leg = axes[0][0].legend(fontsize=8, title="score scale", title_fontsize=8, frameon=False)
    for t in [*leg.get_texts(), leg.get_title()]:
        t.set_color(INK2)
    fig.suptitle(f"{title}: gate score scale, held-out streams (macro)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "dot = best beta of the link; dashed = fixed TS (beta 0); dotted = oracle weight at that b_t; light vertical line = beta 0.5, the paper's",
             fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = cli.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--proxies", nargs="+", default=["nnn", "msp"], choices=[p for p, d in PROXIES.items() if not d.stateful and not d.needs_duo])
    ap.add_argument("--b_ts", nargs="+", type=int, default=[128, 256])
    ap.add_argument("--links", nargs="+", default=list(LINKS), choices=list(LINKS))
    ap.add_argument("--betas", nargs="+", type=float, default=list(BETAS))
    ap.add_argument("--lr_large", type=float, default=None, help="the large member's cached learning rate (default: the duo's tuned one for --tta)")
    ap.add_argument("--lr_small", type=float, default=None, help="the small member's (default: tuned); with the other flag, a pair of different runs")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=1e-3, help="the clip of the score before the link (the live gate's)")
    ap.add_argument("--no_oracles", action="store_true", help="skip the two oracle references (saves about a third of the time)")
    ap.add_argument("--plot_only", action="store_true", help="redraw the figure from the rows that are already there")
    args = ap.parse_args()
    if "logit" not in args.links:
        ap.error("--links must include logit: it is what the others are compared with")

    ctx, seeds, out = cli.setup(args, "ablate_gate_link")
    status = 0
    for seed in seeds:
        lrs = "" if args.lr_large is None and args.lr_small is None else f"__lrL{args.lr_large or 'tuned'}_S{args.lr_small or 'tuned'}"
        name = f"{args.duo}__{args.tta}{lrs}__seed{seed}{cli.bn_tag(args)}"
        if args.plot_only:
            rows, summ = pd.read_csv(out / f"{name}.rows.csv"), pd.read_csv(out / f"{name}.summary.csv")
            plot(rows, summ, out / f"{name}.png", name)
            print(f"wrote {out / name}.png")
            continue
        store = LogitStore.from_context(ctx, None)
        spec = RunSpec(args.duo, args.tta, args.tta_kwargs or {}, "both_indep", "episodic", seed=seed, num_samples=args.num_samples)
        ctx_l = ctx if args.lr_large is None else with_optim(ctx, large=sgd_optim(args.lr_large, args.momentum))
        ctx_s = ctx if args.lr_small is None else with_optim(ctx, small=sgd_optim(args.lr_small, args.momentum))
        ml, _ = run_member_specs(spec, ctx_l)
        _, ms = run_member_specs(spec, ctx_s)
        pair = PairView(store, ml, ms)
        held = [s.file_stem for s in stream_segments(["heldout"], ctx.g)]
        if missing := [s for s in held if s not in set(pair.stems())]:
            print(f"seed {seed}: held-out streams missing from the store for large {ml.key} / small {ms.key}: {missing} (scripts/04_cache_logits.py)")
            status = 1
            continue
        frozen = AM.resolve_temperatures(ctx, AM.temperature_spec(args.duo, args.tta, args.tta_kwargs, args.num_samples), None, report_only=True)
        T = (frozen.T_l, frozen.T_s)
        print(f"seed {seed}: large {ml.key} (lr {ml.model['OPTIM']['LR']}), small {ms.key} (lr {ms.model['OPTIM']['LR']}); "
              f"Fixed TS of the frozen models T_L={T[0]}, T_S={T[1]}; held-out streams {held}", flush=True)
        device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        rows, buckets = [], []
        for stem in held:
            r, b = evaluate_stream(load_stream(pair, stem), T, args.proxies, args.b_ts, args.links, args.betas, args.eps, device, not args.no_oracles)
            rows += r
            buckets += b
            print(f"  {stem} done", flush=True)
        rows = pd.DataFrame(rows)
        gaps = pooled_gap_tracking(buckets, args.links, args.eps)
        summ = summarize(rows, args.links).merge(gaps, on=["proxy", "b_t", "link"], how="left")
        rows.to_csv(out / f"{name}.rows.csv", index=False)
        summ.to_csv(out / f"{name}.summary.csv", index=False)
        gaps.to_csv(out / f"{name}.gaps.csv", index=False)
        plot(rows, summ, out / f"{name}.png", name)

        ref = rows[rows.series.isin(["large", "small", "fixed"])].groupby("series")[["accuracy", "nll"]].mean()
        print(f"\nheld-out macro accuracy: large {100 * ref.loc['large', 'accuracy']:.2f}, small {100 * ref.loc['small', 'accuracy']:.2f}, "
              f"fixed TS (beta = 0) {100 * ref.loc['fixed', 'accuracy']:.2f}")
        show = ["link", "best_beta", "accuracy", "gain_over_fixed", "gap_to_oracle_w", "nll", "worst_stream", "w_sd", "w_sat", "streams_beating_logit",
                "acc_at_beta_0.5", "spearman", "pearson"]
        pd.set_option("display.width", 250)
        for (proxy, b_t), d in summ.groupby(["proxy", "b_t"]):
            d = d.set_index("link").loc[[l for l in LINKS if l in set(d.link)]].reset_index()
            for c in ("accuracy", "gain_over_fixed", "gap_to_oracle_w", "worst_stream", "acc_at_beta_0.5"):
                d[c] = 100 * d[c]
            print(f"\n== {proxy}, b_t = {int(b_t)}: each link at its best beta (accuracy and gaps in points)")
            print(d[show].round(3).to_string(index=False))
        print(f"\nfiles: {out / name}.{{rows.csv,summary.csv,gaps.csv,png}}")
    return status


if __name__ == "__main__":
    sys.exit(main())
