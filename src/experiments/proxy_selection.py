"""
proxy_selection.py
==================
Stage E of docs/design_section45.md: choose the proxy, beta and proxy-batch size b_t of Vanilla Dynamic Duos from the offline grid
(src/experiments/dd_eval.py, scripts/07_eval_duo_grid.py), on the HELD-OUT streams only.

The rule, as in the brief:
  1. Candidates are the Vanilla DD cells with beta > 0 (beta = 0 is the fixed combination, whatever the proxy: not a gate).
  2. The score of a cell is its top-1 accuracy averaged over the held-out streams (macro mean, one weight per stream).
  3. The winner has the best score; cells within `TIE` of it (0.01 accuracy points) are tied, and the tie goes to the higher
     ORDERING ACCURACY of the cell's (proxy, b_t), pooled over the held-out proxy batches; then the higher Spearman correlation,
     then the smaller beta, the smaller b_t, and the proxy's name, so the choice is deterministic.
  4. The same selection is repeated for a second base algorithm (ETA), and the report says whether the same proxy wins.

Nothing here reads a test stream. The evidence stored with a choice: the cell's metrics, its worst held-out stream, the baselines (members
alone, the fixed combinations, oracle selection and the batch-wise oracle weight at the same b_t), the share of the oracle gap closed, the
proxy's ordering accuracy and Spearman correlation, and the runners-up.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.experiments.dd_eval import average_over_streams, pooled_diagnostics

TIE = 1e-4          # accuracy (a fraction): 0.01 points


def load_records(npz_path: str | Path) -> dict:
    """The per-bucket records dd_eval.save_results wrote: {(stream, proxy, b_t): {field: array}}."""
    out: dict = {}
    with np.load(npz_path) as z:
        for key in z.files:
            stream, proxy, b_t, field = key.split("|")
            out.setdefault((stream, proxy, int(b_t)), {})[field] = z[key]
    return out


def select_cell(avg: pd.DataFrame, diag: pd.DataFrame, min_beta: float = 0.0) -> pd.DataFrame:
    """The Vanilla DD cells ranked by the rule above (best first), with their ordering accuracy and Spearman correlation attached.
    `avg`: average_over_streams output; `diag`: pooled_diagnostics output."""
    dd = avg[(avg.series == "dd") & (avg.beta > min_beta)].copy()
    dd = dd.merge(diag[["proxy", "b_t", "ordering_acc", "spearman", "n_buckets"]], on=["proxy", "b_t"], how="left")
    dd["_order"] = dd["ordering_acc"].fillna(-1.0)
    dd["_rho"] = dd["spearman"].fillna(-2.0)
    best = dd["accuracy"].max()
    dd["tied"] = dd["accuracy"] >= best - TIE
    # not-tied cells keep their accuracy order; the tied ones are re-ranked by ordering accuracy, then rho, then the deterministic rest
    tied = dd[dd.tied].sort_values(["_order", "_rho", "beta", "b_t", "proxy"], ascending=[False, False, True, True, True])
    rest = dd[~dd.tied].sort_values(["accuracy", "_order", "beta", "b_t", "proxy"], ascending=[False, False, True, True, True])
    return pd.concat([tied, rest]).drop(columns=["_order", "_rho"]).reset_index(drop=True)


def evidence(rows: list[dict], avg: pd.DataFrame, ranked: pd.DataFrame, heldout: Sequence[str], temps: dict) -> dict:
    """Everything that backs the winner of `ranked`: its numbers per stream, and the baselines around it."""
    w = ranked.iloc[0]
    df = pd.DataFrame(rows)
    df = df[df["stream"].isin(heldout)]
    cell = df[(df.series == "dd") & (df.proxy == w.proxy) & (df.beta == w.beta) & (df.b_t == w.b_t)]
    per_stream = {r.stream: {"accuracy": float(r.accuracy), "ece": float(r.ece), "nll": float(r.nll)} for r in cell.itertuples()}

    def macro(series, **kw):
        sel = avg[avg.series == series]
        for k, v in kw.items():
            sel = sel[sel[k] == v] if v is not None else sel[sel[k].isna()]
        return None if sel.empty else {m: float(sel.iloc[0][m]) for m in ("accuracy", "ece", "nll")}

    fixed = macro("fixed", ts="heldout")
    ow = macro("oracle_w", b_t=int(w.b_t))
    base = {"large": macro("large", ts=None, b_t=None, proxy=None), "small": macro("small", ts=None, b_t=None, proxy=None),
            "fixed_heldout_ts": fixed, "fixed_clean_ts": macro("fixed", ts="clean"),
            "oracle_select_same_b_t": macro("oracle_select", b_t=int(w.b_t)), "oracle_w_same_b_t": ow}
    gap = None if (ow is None or fixed is None or ow["accuracy"] <= fixed["accuracy"]) else \
        (w.accuracy - fixed["accuracy"]) / (ow["accuracy"] - fixed["accuracy"])
    members_best = max(base["large"]["accuracy"], base["small"]["accuracy"])
    return {"chosen": {"proxy": w.proxy, "beta": float(w.beta), "b_t": int(w.b_t), "accuracy": float(w.accuracy), "ece": float(w.ece),
                       "nll": float(w.nll), "worst_stream_accuracy": min(v["accuracy"] for v in per_stream.values()),
                       "per_stream": per_stream, "ordering_acc": None if pd.isna(w.ordering_acc) else float(w.ordering_acc),
                       "spearman": None if pd.isna(w.spearman) else float(w.spearman), "n_buckets": int(w.n_buckets)},
            "baselines": base, "temps": temps,
            "gain_over_fixed_heldout_ts_points": None if fixed is None else 100 * float(w.accuracy - fixed["accuracy"]),
            "gain_over_best_member_points": 100 * float(w.accuracy - members_best),
            "share_of_oracle_w_gap_closed": gap,
            "beats_fixed": bool(fixed is not None and w.accuracy > fixed["accuracy"]),
            "tie_broken_by_ordering": bool(ranked["tied"].sum() > 1)}


def top_table(ranked: pd.DataFrame, k: int = 10) -> list[dict]:
    cols = ["proxy", "beta", "b_t", "accuracy", "ece", "nll", "ordering_acc", "spearman", "tied"]
    return [{c: (None if (isinstance(v, float) and math.isnan(v)) else (v.item() if hasattr(v, "item") else v))
             for c, v in r.items()} for r in ranked[cols].head(k).to_dict("records")]


def best_per_proxy(ranked: pd.DataFrame) -> list[dict]:
    """Each proxy's own best cell (by accuracy; ties by ordering accuracy), best proxy first."""
    cols = ["proxy", "beta", "b_t", "accuracy", "nll", "ordering_acc", "spearman"]
    out = ranked.sort_values(["accuracy", "ordering_acc"], ascending=False).groupby("proxy", sort=False).head(1)
    return [{c: (None if (isinstance(v, float) and math.isnan(v)) else (v.item() if hasattr(v, "item") else v))
             for c, v in r.items()} for r in out[cols].to_dict("records")]


def select_algorithm(rows: list[dict], records: dict, heldout: Sequence[str], temps: dict) -> dict:
    """The whole Stage E selection for one base algorithm's evaluation (rows, per-bucket records, held-out stream stems)."""
    avg = average_over_streams(rows, heldout)
    diag = pooled_diagnostics(records, heldout)
    ranked = select_cell(avg, diag)
    ev = evidence(rows, avg, ranked, heldout, temps)
    return {**ev, "top10": top_table(ranked, 10), "best_per_proxy": best_per_proxy(ranked), "n_cells": int(len(ranked)),
            "heldout_streams": list(heldout)}


def compare_algorithms(sel: dict[str, dict]) -> dict:
    """Does the same proxy win for every algorithm? `sel`: {algorithm: select_algorithm output}."""
    algos = list(sel)
    winners = {a: sel[a]["chosen"]["proxy"] for a in algos}
    ranks = {a: {r["proxy"]: i + 1 for i, r in enumerate(sel[a]["best_per_proxy"])} for a in algos}
    first = algos[0]
    return {"winners": winners, "same_proxy_wins": len(set(winners.values())) == 1,
            "same_cell_wins": len({(sel[a]["chosen"]["proxy"], sel[a]["chosen"]["beta"], sel[a]["chosen"]["b_t"]) for a in algos}) == 1,
            "rank_of_each_winner_under_each_algorithm": {w: {a: ranks[a].get(w) for a in algos} for w in set(winners.values())},
            "top3_proxies": {a: [r["proxy"] for r in sel[a]["best_per_proxy"][:3]] for a in algos},
            "top3_overlap": sorted(set.intersection(*[set(r["proxy"] for r in sel[a]["best_per_proxy"][:3]) for a in algos])),
            "reference_algorithm": first}


if __name__ == "__main__":
    import tempfile

    from src.experiments.dd_eval import save_results

    rng = np.random.RandomState(0)
    streams = ["a__s5", "b__s5", "c__s5", "d__s5"]
    proxies = ["nnn", "msp", "doc"]

    def make_rows(best=("nnn", 1.0, 128), accs=None):
        rows = []
        for s in streams:
            for series, acc in (("large", 0.60), ("small", 0.45)):
                rows.append({"stream": s, "series": series, "proxy": None, "beta": None, "b_t": None, "ts": None,
                             "accuracy": acc, "ece": 0.05, "nll": 2.0, "n": 1000})
            for ts in ("heldout", "clean"):
                rows.append({"stream": s, "series": "fixed", "proxy": None, "beta": None, "b_t": None, "ts": ts,
                             "accuracy": 0.62 if ts == "heldout" else 0.60, "ece": 0.04, "nll": 1.9, "n": 1000})
            for b_t in (64, 128):
                for series, acc in (("oracle_select", 0.66), ("oracle_w", 0.70)):
                    rows.append({"stream": s, "series": series, "proxy": None, "beta": None, "b_t": b_t, "ts": "heldout",
                                 "accuracy": acc, "ece": 0.04, "nll": 1.8, "n": 1000})
                for p in proxies:
                    for beta in (0.0, 0.5, 1.0, 2.0):
                        acc = 0.62 if beta == 0 else 0.62 + 0.001 * beta + 0.0001 * rng.rand()
                        if (p, beta, b_t) == best:
                            acc = 0.65
                        if accs and (p, beta, b_t) in accs:
                            acc = accs[(p, beta, b_t)]
                        rows.append({"stream": s, "series": "dd", "proxy": p, "beta": beta, "b_t": b_t, "ts": "heldout",
                                     "accuracy": acc, "ece": 0.04, "nll": 1.9 - 0.1 * (acc - 0.62), "n": 1000})
        return rows

    def make_records(order_acc):
        recs = {}
        for s in streams:
            for p in proxies:
                for b_t in (64, 128):
                    n = 8
                    acc_l, acc_s = rng.uniform(0.4, 0.8, n), rng.uniform(0.3, 0.6, n)
                    good = order_acc[p]
                    r_l = np.where(rng.rand(n) < good, acc_l, acc_s)         # a proxy that orders correctly with probability `good`
                    r_s = np.where(rng.rand(n) < good, acc_s, acc_l)
                    recs[(s, p, b_t)] = {"r_l": r_l, "r_s": r_s, "acc_l": acc_l, "acc_s": acc_s}
        return recs

    recs = make_records({"nnn": 0.9, "msp": 0.6, "doc": 0.5})
    temps = {"heldout": [1.0, 1.0]}
    # 1. a clear winner, and beta = 0 is never a candidate
    sel = select_algorithm(make_rows(), recs, streams, temps)
    c = sel["chosen"]
    assert (c["proxy"], c["beta"], c["b_t"]) == ("nnn", 1.0, 128) and abs(c["accuracy"] - 0.65) < 1e-9
    assert sel["n_cells"] == 3 * 3 * 2 and all(r["beta"] > 0 for r in sel["top10"])
    assert abs(sel["gain_over_fixed_heldout_ts_points"] - 3.0) < 1e-6 and sel["beats_fixed"] and not sel["tie_broken_by_ordering"]
    assert abs(sel["gain_over_best_member_points"] - 5.0) < 1e-6
    assert abs(sel["share_of_oracle_w_gap_closed"] - (0.65 - 0.62) / (0.70 - 0.62)) < 1e-9
    assert sel["baselines"]["oracle_w_same_b_t"]["accuracy"] == 0.70 and sel["baselines"]["fixed_clean_ts"]["accuracy"] == 0.60
    assert set(c["per_stream"]) == set(streams) and c["worst_stream_accuracy"] == 0.65
    assert sel["best_per_proxy"][0]["proxy"] == "nnn" and len(sel["best_per_proxy"]) == 3
    # 2. a tie within 0.01 points goes to the better-ordering proxy, not to name or beta
    ties = {("msp", 2.0, 64): 0.66 + 5e-5, ("nnn", 0.5, 64): 0.66, ("doc", 1.0, 128): 0.66 - 5e-5}      # nnn orders best, msp is a hair higher
    sel = select_algorithm(make_rows(best=("xx", 0, 0), accs=ties), recs, streams, temps)
    assert sel["tie_broken_by_ordering"] and sel["chosen"]["proxy"] == "nnn", sel["chosen"]
    assert sel["chosen"]["accuracy"] < max(r["accuracy"] for r in sel["top10"]), "the winner is not the most accurate cell: the tie-break did it"
    ranked_p = [r["ordering_acc"] for r in sel["top10"][:3]]
    assert sel["top10"][0]["tied"] and ranked_p == sorted(ranked_p, reverse=True)
    # 3. outside the tie the accuracy decides, even against a better-ordering proxy
    sel = select_algorithm(make_rows(best=("xx", 0, 0), accs={("doc", 2.0, 64): 0.70, ("nnn", 2.0, 64): 0.69}), recs, streams, temps)
    assert sel["chosen"]["proxy"] == "doc", sel["chosen"]
    # 4. no gate beating the fixed combination is reported, not hidden
    flat = {(p, b, bt): 0.60 for p in proxies for b in (0.5, 1.0, 2.0) for bt in (64, 128)}
    sel = select_algorithm(make_rows(best=("xx", 0, 0), accs=flat), recs, streams, temps)
    assert not sel["beats_fixed"] and sel["gain_over_fixed_heldout_ts_points"] < 0
    # 5. across algorithms: same proxy, different proxies
    a = select_algorithm(make_rows(), recs, streams, temps)
    b = select_algorithm(make_rows(best=("msp", 2.0, 64)), recs, streams, temps)
    cmp_same = compare_algorithms({"tent": a, "eta": a})
    cmp_diff = compare_algorithms({"tent": a, "eta": b})
    assert cmp_same["same_proxy_wins"] and cmp_same["same_cell_wins"]
    assert not cmp_diff["same_proxy_wins"] and cmp_diff["winners"] == {"tent": "nnn", "eta": "msp"}
    assert cmp_diff["rank_of_each_winner_under_each_algorithm"]["nnn"]["eta"] >= 2 and cmp_diff["rank_of_each_winner_under_each_algorithm"]["msp"]["eta"] == 1
    # 6. the records survive a save and a load
    with tempfile.TemporaryDirectory() as tmp:
        files = save_results(tmp, "t", {"rows": make_rows(), "diag": [], "records": {k: {**v, "w_l": np.zeros((2, 8)), "betas": np.array([0., 1.])} for k, v in recs.items()}, "temps": temps, "stats": None})
        back = load_records(files["records.npz"])
        assert set(back) == set(recs) and np.allclose(back[("a__s5", "nnn", 64)]["r_l"], recs[("a__s5", "nnn", 64)]["r_l"])
    print("proxy_selection self-test passed")
