"""
proxy_study.py
==============
Why the reset rule watches the NUCLEAR NORM (nnn): the same offline screen that chose the rule (src/reset/screening.py), replayed on every candidate
proxy's sequence instead of nnn's. Driver: scripts/49_proxy_study.py (plan | submit | run | report).

**Data.** The calibration streams of the reset tuning (stress S: both members collapse; healthy H: no collapse, the false-alarm budget), on the
held-out continual stream, Independent DD (the members never see the gate, so their trajectories do not depend on the proxy), RE-RUN with a
`PanelGate`: the live gate is still nnn at the calibration beta (same run as before), and on every proxy batch it also scores the whole PANEL on
both members and stores it in the record (`record["panel"][proxy] = [r_l, r_s]`). One GPU run per stream gives every proxy. The frozen run of
the original calibration stream is reused (same images; the frozen members do not depend on anything here). By default only seed 0 is re-run
(4 stress rotations + 2 healthy per duo).

**Source statistics.** ATC, ATC-negent and energy read the source statistics of each member (`pure.fit_source_stats`): fitted once per duo on
the frozen members' RAW logits on the clean ImageNet training slice (10 images/class), the same logits the gate scores (no temperature), and
kept in `<out>/results/proxy_study/<duo>/source_stats.json`.

**Screen.** Per proxy and per ORIENTATION: `+` = the detector alarms when the proxy DROPS (every pure proxy is oriented so that higher = more
reliable), `-` = when it RISES (the rule watches logit(r), r in [0, 1], so the flipped sequence is 1 - r = a negated logit). The detector grid of screening.py is swept on the pooled calibration streams and
`select` takes, per false-alarm budget, the lowest delay; a proxy's reported result is its better orientation (so no competitor is
straw-manned by a fixed sign). Also per proxy and member: Spearman correlation over proxy batches of the proxy with the member's accuracy and
with its accuracy minus the frozen member's (the degradation the labels are built from), and the COLLAPSE RESPONSE: the proxy's mean over the
first 16 batches of each degraded stretch minus its mean over the 32 batches before it, in units of that pre-onset window's std (negative =
falls at collapse, what a drop detector needs).

The panel (and what is left out): nnn; the confidence proxies msp (AC-MC), neg_entropy (Tent's own objective) and the source-thresholded
atc (ATC, max-confidence score) / atc_negent (ATC, negative-entropy score); energy; the diversity proxies dispersity (nnn's diversity part
alone), asr_concentration (ASR's prediction concentration: entropy of the softmax of the batch-mean logits) and marginal_entropy (entropy of
the batch-mean probabilities); mutual_info (diversity minus mean entropy, the other confidence + diversity proxy); agree_duo (agreement with
the fixed-TS duo). Left out: doc (msp plus a constant: identical under a standardized fast-vs-slow detector) and cot (needs large proxy batches).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.reliability.proxies import pure
from src.reset import screening as SC
from src.reset.controller import MEMBERS, RuleSpec

PANEL = ("nnn", "msp", "neg_entropy", "atc", "atc_negent", "energy", "dispersity", "asr_concentration", "marginal_entropy", "mutual_info", "agree_duo")
LABELS = {"nnn": "Nuclear norm (ours)", "msp": "AC-MC", "neg_entropy": "Neg. entropy", "atc": "ATC (max-conf.)", "atc_negent": "ATC (neg. entropy)",
          "energy": "Energy", "dispersity": "Dispersity", "asr_concentration": "ASR concentration", "marginal_entropy": "Marginal entropy",
          "mutual_info": "Mutual information", "agree_duo": "Duo agreement"}
SEEDS = (0,)                        # the seeds of the calibration streams re-run (the tuning had 0 and 1 for S)
HEALTHY_ROTATIONS = (0, 1, 2, 3)    # healthy streams: every rotation (the tuning ran 0 and 2; 1 and 3 added for the correlation study)
RESPONSE = (32, 16)                 # collapse response: batches before the onset, batches after it
TIME = "02:15:00"


# ------------------------------------------------------------------------------------------------------------------ the gate

def _panel_gate_class():
    """PanelGate is defined lazily: the report side of this module must not need torch."""
    import torch

    from src.experiments.joint_dd import PureProxyGate

    class PanelGate(PureProxyGate):
        """A PureProxyGate (its own proxy drives the weight, exactly as before) that also scores every proxy of `panel` on both members at each
        flush and stores them in the record: record["panel"][proxy] = [r_l, r_s]. Raw logits, like the gate's own scores."""

        def __init__(self, proxy: str, *, panel: Sequence[str] = PANEL, stats_l: dict, stats_s: dict, **kw):
            super().__init__(proxy, stats_l=stats_l, stats_s=stats_s, **kw)
            unknown = [p for p in panel if p not in pure.PROXIES]
            if unknown:
                raise ValueError(f"not in pure.PROXIES: {unknown}")
            self.panel, self._stats, self._panel_last = tuple(panel), (stats_l, stats_s), None

        @torch.no_grad()
        def _proxy_scores(self, z_l, z_s, f_l, f_s, labels):
            out = super()._proxy_scores(z_l, z_s, f_l, f_s, labels)
            T_l, T_s = float(self.base_ts.Tl.item()), float(self.base_ts.Ts.item())
            duo = 0.5 * (z_l / T_l) + 0.5 * (z_s / T_s)
            panel = {}
            for p in self.panel:
                kw = {"duo_logits": duo} if pure.PROXIES[p].needs_duo else {}
                panel[p] = [pure.score(p, z_l, None, self._stats[0], **kw), pure.score(p, z_s, None, self._stats[1], **kw)]
            self._panel_last = panel
            return out

        def _flush_bucket(self) -> None:
            n = len(self.records) if self.records is not None else 0
            super()._flush_bucket()
            if self.records is not None and len(self.records) > n:
                self.records[-1]["panel"] = self._panel_last

    return PanelGate


def __getattr__(name):
    if name == "PanelGate":
        return _panel_gate_class()
    raise AttributeError(name)


# ------------------------------------------------------------------------------------------------------------------ files

def study_dir(g, duo: str) -> Path:
    return g.path("out") / "results" / "proxy_study" / duo


def run_path(g, P, a: dict) -> Path:
    """The panel run of calibration job args `a`: the calibration stream's own file name, in the study folder."""
    return study_dir(g, P.duo) / P.calibration_run(a["name"], **_key(a)).name


def frozen_path(P, a: dict) -> Path:
    return P.calibration_run(a["name"], **_key(a), frozen=True)


def _key(a: dict) -> dict:
    return dict(lrs=(float(a["lrs"][0]), float(a["lrs"][1])), seed=a["seed"], rotation=a["rotation"], beta=a["beta"], b_t=a["b_t"], proxy=a["proxy"])


# ------------------------------------------------------------------------------------------------------------------ the jobs

def panel_jobs(duo: str, seeds: Sequence[int] = SEEDS, healthy_rotations: Sequence[int] = HEALTHY_ROTATIONS) -> list[dict]:
    """The calibration-stream job args of the duo's (Independent, Tent, mean) reset tuning, for `seeds`, plus healthy (H) streams at the extra
    `healthy_rotations` the tuning did not run (same args, another rotation; they have no frozen run of their own: run_one makes one)."""
    from src.reset import pipeline as PL
    S = PL.Setup.load(duo, log=lambda *a, **k: None)
    jobs = [dict(j.args) for j in PL.calibration_jobs(S) if j.args["seed"] in seeds]
    for h in [a for a in jobs if a["name"] == "H"]:
        have = {a["rotation"] for a in jobs if a["name"] == "H" and a["seed"] == h["seed"]}
        jobs += [{**h, "rotation": r} for r in healthy_rotations if r not in have]
        have |= set(healthy_rotations)
    seen, out = set(), []
    for a in jobs:
        k = (a["name"], a["rotation"], a["seed"])
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


def fit_source_stats(ctx, tta: str, out: Path) -> tuple[dict, dict]:
    """Each frozen member's `pure.fit_source_stats` on the clean training slice (cached in `out`; the first job to finish writes it, the others read it)."""
    if out.is_file():
        d = json.loads(out.read_text())
        return d["large"], d["small"]
    import torch

    from src.experiments import continual as RC
    from src.utils.datasets.imagenet import ImageNetTrainSlice
    runner, _ = RC.make_runner(ctx, RC.RUNS["frozen"], None, tta)
    ds = ImageNetTrainSlice.from_config(ctx.g)
    zl, zs, ys = [], [], []
    for z_l, z_s, y in runner.iter_batches(ds, batch_size=ctx.batch_size, workers=ctx.g.workers, seed=0, num_samples=None):
        zl.append(z_l.float().cpu()); zs.append(z_s.float().cpu()); ys.append(y.cpu())
    y = torch.cat(ys)
    st = {"large": pure.fit_source_stats(torch.cat(zl), y), "small": pure.fit_source_stats(torch.cat(zs), y)}
    print(f"source statistics on the clean train slice ({len(y)} images): " + "; ".join(f"{s} acc {v['acc']:.3f} msp {v['msp']:.3f}" for s, v in st.items()))
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(out)
    d = json.loads(out.read_text())
    return d["large"], d["small"]


def run_one(duo: str, a: dict, device: str | None = None) -> Path:
    """Re-run one calibration stream with the PanelGate (GPU). The frozen run is the original calibration stream's."""
    import time

    from src.experiments import continual as RC
    from src.experiments import gate_stress as GS
    from src.reset import workers as W
    from src.reset.layout import Paths
    from src.utils.config import load_global

    g = load_global()
    P = Paths(g, duo)
    out, frz = run_path(g, P, a), frozen_path(P, a)
    if out.is_file():
        print(f"done already ({out})")
        return out
    if not frz.is_file() and a["name"] != "H":
        raise FileNotFoundError(f"the original calibration stream's frozen run is missing: {frz}")
    ctx = W._context(g, duo, device)
    temps = W._temps(ctx, duo)
    st_l, st_s = fit_source_stats(ctx, W.TTA, study_dir(g, duo) / "source_stats.json")
    gate = _panel_gate_class()(a["proxy"], beta=a["beta"], b_t=a["b_t"], T_l=temps.T_l, T_s=temps.T_s, stats_l=st_l, stats_s=st_s, num_classes=1000)
    t0 = time.time()
    res, _, parts, meta = GS.run_experiment(ctx, lrs=_key(a)["lrs"], watch=W.SIDES, after=a["after"], seed=a["seed"], order=None, images=a["images"],
                                            proxy=a["proxy"], beta=a["beta"], b_t=a["b_t"], T_l=temps.T_l, T_s=temps.T_s, momentum=a["momentum"],
                                            stream="heldout", rounds=a["rounds"], rotation=a["rotation"], stop_mode=a["stop_mode"], gate=gate, frozen=False)
    f = np.load(frz, allow_pickle=False) if frz.is_file() else None
    f_labels = [str(x) for x in f["part_label"]] if f is not None else []
    if f is None or int(f["part_n"].sum()) < res["seen"] or f_labels[:len(parts)] != [p.label for p in parts]:
        # no frozen run (a healthy rotation the tuning did not run), or GPU nondeterminism moved the stop past the original run's: make our own
        print(f"no frozen run covering this run ({frz.name}; this run {res['seen']} images): running the frozen members over the same images")
        from src.experiments import continual_tuning as CT
        frozen_runner, _ = RC.make_runner(ctx, RC.RUNS["frozen"], None, W.TTA)
        fr = CT.run_until(frozen_runner, parts, max_images=res["seen"], images=a["images"], batch_size=ctx.batch_size, workers=ctx.g.workers,
                          T_l=temps.T_l, T_s=temps.T_s)
        frz = out.with_name(out.stem + "__frozen.npz")
        RC.save_run(frz, fr, parts, {**meta, "run": "frozen"})
    meta.update(name=a["name"], panel=list(gate.panel), frozen=str(frz), seconds=time.time() - t0, study="proxy_study")
    RC.save_run(out, res, parts, {**meta, "run": "indep"})
    print(f"wrote {out} ({res['seen']} images, {meta['seconds'] / 60:.0f} min)")
    return out


# ------------------------------------------------------------------------------------------------------------------ the screen

@dataclass
class PanelStream:
    """One panel run: the panel values per proxy batch, the members' adapted and frozen per-batch accuracy."""
    name: str
    panel: dict[str, np.ndarray]           # proxy -> (n, 2): [r_l, r_s] per proxy batch
    acc: dict[str, np.ndarray]
    frozen: dict[str, np.ndarray]
    nnn_gate: np.ndarray                   # (n, 2) the gate's own r_l, r_s (must equal panel["nnn"])

    @classmethod
    def load(cls, run: Path, b_t: int, name: str | None = None) -> "PanelStream":
        records = json.loads(Path(run).with_suffix(".gate.json").read_text())
        d = np.load(run, allow_pickle=False)
        frz = Path(json.loads(str(d["meta"]))["frozen"])
        f = np.load(frz, allow_pickle=False)
        n = len(records)
        panel = {p: np.array([r["panel"][p] for r in records], dtype=float) for p in records[0]["panel"]}
        return cls(name or Path(run).stem, panel, {m: SC.per_batch(d[k], b_t, n) for m, k in zip(MEMBERS, ("large", "small"))},
                   {m: SC.per_batch(f[k], b_t, n) for m, k in zip(MEMBERS, ("large", "small"))},
                   np.array([[r["r_l"], r["r_s"]] for r in records], dtype=float))

    def screen_stream(self, proxy: str, sign: int = 1) -> SC.ScreenStream:
        v = self.panel[proxy] if sign > 0 else 1.0 - self.panel[proxy]      # the rule watches logit(r): 1 - r is exactly its sign flip
        return SC.ScreenStream(f"{self.name}|{proxy}{'+' if sign > 0 else '-'}", v[:, 0], v[:, 1], self.acc, self.frozen)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    sa, sb = ra.std(), rb.std()
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def collapse_response(x: np.ndarray, stretches: Sequence[tuple[int, int]], before: int = RESPONSE[0], after: int = RESPONSE[1]) -> list[float]:
    """Per stretch: (mean of x over its first `after` batches - mean over the `before` batches before the onset) / std of that pre window."""
    out = []
    for a, b in stretches:
        pre, post = x[max(a - before, 0):a], x[a:min(a + after, b)]
        if len(pre) < 4 or not len(post):
            continue
        sd = pre.std()
        out.append(float((post.mean() - pre.mean()) / max(sd, 1e-9)))
    return out


def screen(streams: Sequence[PanelStream], proxies: Sequence[str] | None = None, budgets: Sequence[float] = SC.BUDGETS, log=print) -> dict:
    """Per proxy and orientation: the screen's sweep and per-budget selection; per proxy: tracking correlations and collapse response. Returns
    {"detect": [rows], "track": [rows], "frontier": {proxy: [(fa, delay)]}}."""
    proxies = list(proxies or streams[0].panel)
    for st in streams:
        if "nnn" in st.panel and not np.allclose(st.panel["nnn"], st.nnn_gate, rtol=0, atol=1e-9):
            raise RuntimeError(f"{st.name}: the panel's nnn differs from the gate's own scores (the panel is not scored on the gate's batches)")
    events = sum(len(st.screen_stream(proxies[0]).stretches(m)) for st in streams for m in MEMBERS)
    log(f"{len(streams)} streams, {sum(len(st.acc['large']) for st in streams)} proxy batches, {events} degraded stretches (both members)")
    detect, frontier = [], {}
    for p in proxies:
        best: dict[float, dict] = {}
        for sign in (1, -1):
            ss = [st.screen_stream(p, sign) for st in streams]
            rows = SC.sweep(ss)
            frontier[f"{p}{'+' if sign > 0 else '-'}"] = SC.curve(rows, SC.FAMILY)
            for b in budgets:
                row = SC.select(rows, b)[SC.FAMILY]
                rec = {"proxy": p, "orientation": "+" if sign > 0 else "-", "budget": b}
                if row is None:
                    rec.update(config=None, delay=float("nan"), delay_detected=float("nan"), miss_rate=float("nan"), fa=float("nan"), events=events)
                else:
                    spec = RuleSpec.from_json(row["spec"])
                    per = {m: _member_delay(spec, ss, m) for m in MEMBERS}
                    rec.update(config=json.dumps({**row["config"], "thr": row["threshold"]}), delay=row["delay"], delay_detected=row["delay_detected"],
                               miss_rate=row["miss_rate"], fa=row["fa"], events=row["events"], delay_large=per["large"], delay_small=per["small"])
                detect.append(rec)
                cur = best.get(b)
                if cur is None or SC._key(_as_key(rec)) < SC._key(_as_key(cur)):
                    best[b] = rec
        for b, rec in best.items():
            rec["best"] = True
    track = []
    for p in proxies:
        for m, j in zip(MEMBERS, (0, 1)):
            rho_acc = [spearman(st.panel[p][:, j], st.acc[m]) for st in streams]
            rho_gap = [spearman(st.panel[p][:, j], st.acc[m] - st.frozen[m]) for st in streams]
            resp = [r for st in streams for r in collapse_response(st.panel[p][:, j], st.screen_stream(p).stretches(m))]
            track.append({"proxy": p, "member": m, "spearman_acc": float(np.nanmean(rho_acc)), "spearman_gap": float(np.nanmean(rho_gap)),
                          "response": float(np.mean(resp)) if resp else float("nan"), "response_falls": float(np.mean([r < 0 for r in resp])) if resp else float("nan"),
                          "stretches": len(resp)})
    return {"detect": detect, "track": track, "frontier": frontier}


def _as_key(rec: dict) -> dict:
    return {"delay": rec["delay"], "miss_rate": rec["miss_rate"], "fa": rec["fa"] if rec["fa"] == rec["fa"] else float("inf")}


def _member_delay(spec, streams: Sequence[SC.ScreenStream], member: str) -> float:
    delays = []
    for st in streams:
        delays += SC.score_member(SC.replay(spec, st)[member], st.stretches(member), st.n)["delays"]
    return float(np.mean(delays)) if delays else float("nan")


def summary_table(res: dict, budget: float = 1.0) -> list[dict]:
    """One row per proxy at `budget`: its better orientation's detection, the natural (+) orientation's, and the tracking numbers (members averaged)."""
    rows = []
    for p in dict.fromkeys(r["proxy"] for r in res["detect"]):
        at = [r for r in res["detect"] if r["proxy"] == p and r["budget"] == budget]
        best = next(r for r in at if r.get("best"))
        plus = next(r for r in at if r["orientation"] == "+")
        tr = [t for t in res["track"] if t["proxy"] == p]
        rows.append({"proxy": p, "label": LABELS.get(p, p), "family": pure.PROXIES[p].family, "orientation": best["orientation"],
                     "delay": best["delay"], "delay_detected": best["delay_detected"], "miss_rate": best["miss_rate"], "fa": best["fa"],
                     "delay_large": best.get("delay_large", float("nan")), "delay_small": best.get("delay_small", float("nan")),
                     "delay_plus": plus["delay"], "miss_plus": plus["miss_rate"],
                     "spearman_gap": float(np.nanmean([t["spearman_gap"] for t in tr])), "spearman_acc": float(np.nanmean([t["spearman_acc"] for t in tr])),
                     "response": float(np.nanmean([t["response"] for t in tr])), "response_falls": float(np.nanmean([t["response_falls"] for t in tr]))})
    return sorted(rows, key=lambda r: (r["delay"] if r["delay"] == r["delay"] else math.inf, r["miss_rate"] if r["miss_rate"] == r["miss_rate"] else math.inf))


def latex_table(tables: dict[str, list[dict]], budget: float, b_t: int, member_names: dict[str, tuple[str, str]]) -> str:
    """One booktabs table, a block per duo: proxy | orientation | delay | miss | false alarms | Spearman with the gap | response."""
    def f(x, d=1):
        return "--" if x != x else f"{x:.{d}f}"
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Which proxy should the reset detector watch? The \texttt{own\_fast\_vs\_slow} detector's grid is re-selected for every proxy (and "
             r"for both orientations: $+$ alarms on a drop, $-$ on a rise; the better is reported) on the held-out calibration streams at "
             + f"{budget:g}" + r" false alarm per 1{,}000 proxy batches. Delay in proxy batches of " + str(b_t) +
             r" images from the onset of a member's degradation (a miss is charged its whole stretch). $\rho$: Spearman correlation, over proxy batches, "
             r"of the proxy with the member's accuracy minus the frozen member's. Response: the proxy's change at a collapse onset, in pre-onset standard "
             r"deviations (negative: it falls).}", r"\label{tab:proxy_study}",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Proxy & Orient. & Delay $\downarrow$ & Miss $\downarrow$ & FA & $\rho$ $\uparrow$ & Response \\", r"\midrule"]
    for k, (duo, rows) in enumerate(tables.items()):
        if k:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\textit{" + " + ".join(member_names.get(duo, (duo,))) + r"}} \\")
        for r in rows:
            name = r["label"].replace("(ours)", "").strip()
            name = rf"\textbf{{{name}}}" if r["proxy"] == "nnn" else name
            lines.append(f"{name} & ${r['orientation']}$ & {f(r['delay'])} & {f(r['miss_rate'], 2)} & {f(r['fa'], 2)} & {f(r['spearman_gap'], 2)} & {f(r['response'], 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------ correlation with accuracy

# Row order of the correlation table: ours, the other diversity proxies, the confidence proxies, agreement.
ORDER = ("nnn", "mutual_info", "dispersity", "marginal_entropy", "asr_concentration", "msp", "neg_entropy", "atc", "atc_negent", "energy", "agree_duo")
REGIMES = (("H", "Healthy"), ("S", "Collapsing"))


def regime(st: PanelStream) -> str:
    """'H' (healthy calibration stream) or 'S' (stress: both members collapse), from the stream's name."""
    r = st.name.split("|")[0].split("__")[0].replace("exp-", "")[:1]
    if r not in ("H", "S"):
        raise ValueError(f"cannot tell the regime of stream {st.name!r}")
    return r


def correlations(streams: Sequence[PanelStream], proxies: Sequence[str] | None = None) -> list[dict]:
    """Per proxy and regime: the Spearman correlation, over the proxy batches of ONE stream, of the proxy with accuracy, averaged over the regime's
    streams. `gap` = the gate's quantity: (score_L - score_S) against (acc_L - acc_S); `large` / `small` = each member's own score against its own
    accuracy. `ahead` = the streams of the regime in which nnn's gap correlation is at least this proxy's (nnn itself: all of them).
    Within-stream (not pooled) so that neither shift difficulty nor the collapsed-vs-healthy level difference inflates it."""
    proxies = [p for p in (proxies or ORDER) if p in streams[0].panel]

    def gap(st, p):
        v = st.panel[p]
        return spearman(v[:, 0] - v[:, 1], st.acc["large"] - st.acc["small"])

    rows = []
    for p in proxies:
        row = {"proxy": p, "label": LABELS.get(p, p)}
        for r, _ in REGIMES:
            ss = [st for st in streams if regime(st) == r]
            g = np.array([gap(st, p) for st in ss])
            g_nnn = np.array([gap(st, "nnn") for st in ss]) if "nnn" in streams[0].panel else np.full(len(ss), np.nan)
            row.update({f"{r}_n": len(ss), f"{r}_gap": float(np.nanmean(g)) if len(ss) else float("nan"),
                        f"{r}_gap_min": float(np.nanmin(g)) if len(ss) else float("nan"),
                        f"{r}_gap_max": float(np.nanmax(g)) if len(ss) else float("nan"),
                        f"{r}_ahead": int(np.sum(g_nnn >= g - 1e-12)),
                        f"{r}_large": float(np.nanmean([spearman(st.panel[p][:, 0], st.acc["large"]) for st in ss])) if len(ss) else float("nan"),
                        f"{r}_small": float(np.nanmean([spearman(st.panel[p][:, 1], st.acc["small"]) for st in ss])) if len(ss) else float("nan")})
        rows.append(row)
    return rows


def correlation_latex(tables: dict[str, list[dict]], member_names: dict[str, tuple[str, str]], b_t: int = 256) -> str:
    """One booktabs table, a block per duo: proxy | per regime: gap rho, large rho, small rho | streams in which nnn's gap rho >= the proxy's.
    The best value of each column of a block is bold. The caption states the number of streams per regime (so a placeholder says what it is)."""
    def f(x):
        return "--" if x != x else f"{x:+.2f}".replace("+", "\\phantom{-}") if x >= 0 else f"{x:.2f}".replace("-", "$-$")

    cols = [f"{r}_{k}" for r, _ in REGIMES for k in ("gap", "large", "small")]
    counts = sorted({(rows[0]["H_n"], rows[0]["S_n"]) for rows in tables.values() if rows})
    n_txt = "; ".join(f"{h} healthy and {s} collapsing" for h, s in counts)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{How well each label-free proxy tracks accuracy, on the held-out continual streams of Independent DD (both members adapt with "
             r"Tent; held-out shifts only). Healthy: both members at stable learning rates; collapsing: at rates high enough that both collapse. "
             r"Spearman correlation over the proxy batches (" + str(b_t) + r" images) of one stream, averaged over the streams ("
             + n_txt + r" streams per duo). Gap: the gate's input, $s_L - s_S$, against $\mathrm{acc}_L - \mathrm{acc}_S$; "
             r"$L$ / $S$: each member's score against its own accuracy. Last column: streams in which the nuclear norm's gap correlation is at least "
             r"the proxy's. Every proxy is oriented so that higher means more reliable. Best of each column in bold.}",
             r"\label{tab:proxy_correlation}", r"\begin{tabular}{l rrr rrr r}", r"\toprule",
             r" & \multicolumn{3}{c}{Healthy} & \multicolumn{3}{c}{Collapsing} & nnn $\geq$ \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
             r"Proxy & Gap & $L$ & $S$ & Gap & $L$ & $S$ & (streams) \\", r"\midrule"]
    for k, (duo, rows) in enumerate(tables.items()):
        if k:
            lines.append(r"\midrule")
        nm = member_names.get(duo, (duo,))
        lines.append(r"\multicolumn{8}{l}{\textit{" + " + ".join(nm) + r"} \quad ($L$ = " + nm[0] + (f", $S$ = {nm[1]}" if len(nm) > 1 else "") + r")} \\")
        best = {c: max((round(r[c], 2) for r in rows if r[c] == r[c]), default=float("nan")) for c in cols}      # ties at the shown precision: all bold
        for r in rows:
            name = r["label"].replace("(ours)", "").strip()
            name = rf"\textbf{{{name}}} (ours)" if r["proxy"] == "nnn" else name
            cells = [rf"\textbf{{{f(r[c])}}}" if r[c] == r[c] and round(r[c], 2) == best[c] else f(r[c]) for c in cols]
            n = r["H_n"] + r["S_n"]
            ahead = "--" if r["proxy"] == "nnn" else f"{r['H_ahead'] + r['S_ahead']}/{n}"
            lines.append(f"{name} & " + " & ".join(cells) + f" & {ahead} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------ the figure

SHOWN = ("nnn", "msp", "atc", "asr_concentration")                  # at most four lines per panel: legend + direct labels
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")               # categorical slots 1, 2, 3, 7 (fixed order; see the dataviz palette)


def plot_traces(st: PanelStream, out: Path, member_names: tuple[str, str], b_t: int, proxies: Sequence[str] = SHOWN, w0: int = 20) -> Path:
    """Small multiples, one column per member: top, each proxy standardized on the stream's first `w0` batches; bottom, the member's accuracy minus the
    frozen member's (points). Degraded stretches shaded. One y-scale per panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    for j, m in enumerate(MEMBERS):
        top, bot = axes[0, j], axes[1, j]
        x = np.arange(len(st.acc[m]))
        for a, b in st.screen_stream(proxies[0]).stretches(m):
            for ax in (top, bot):
                ax.axvspan(a, b, color="#ecebe6", lw=0, zorder=0)
        for p, c in zip(proxies, COLORS):
            v = st.panel[p][:, j]
            ref = v[:w0]
            z = (v - ref.mean()) / max(ref.std(), 1e-9)
            top.plot(x, z, color=c, lw=1.6, label=LABELS.get(p, p))
            top.annotate(LABELS.get(p, p).replace(" (ours)", ""), (x[-1], z[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                         color=muted, va="center")
        bot.plot(x, 100 * (st.acc[m] - st.frozen[m]), color=muted, lw=1.2)
        bot.axhline(0, color=muted, lw=0.8)
        top.set_title(member_names[j], fontsize=10, color=ink, loc="left")
        top.set_ylabel("proxy (z, first %d batches)" % w0 if j == 0 else "", fontsize=8.5, color=muted)
        bot.set_ylabel("acc. − frozen (pts)" if j == 0 else "", fontsize=8.5, color=muted)
        bot.set_xlabel(f"proxy batch ({b_t} images)", fontsize=8.5, color=muted)
        for ax in (top, bot):
            ax.grid(axis="y", color=grid, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(grid)
            ax.tick_params(labelsize=7.5, colors=muted)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------------------------ self-test

if __name__ == "__main__":
    import tempfile

    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # ---- the gate: the panel is scored on the gate's own batches, nnn equals the gate's r_l / r_s, the weights are unchanged ------
    K, N = 50, 400
    y = torch.randint(0, K, (N,))
    z_l, z_s = torch.randn(N, K) * 3, torch.randn(N, K) * 2
    st_l, st_s = pure.fit_source_stats(z_l, y), pure.fit_source_stats(z_s, y)
    from src.experiments.joint_dd import PureProxyGate
    G = _panel_gate_class()
    g = G("nnn", beta=0.5, b_t=64, T_l=1.2, T_s=0.9, stats_l=st_l, stats_s=st_s, num_classes=K)
    ref = PureProxyGate("nnn", beta=0.5, b_t=64, T_l=1.2, T_s=0.9, num_classes=K)
    for gg in (g, ref):
        gg.records = []
        gg.set_corruption("s", total_samples=N)
    out_g = torch.cat([g.calibrate(z_l[i:i + 32], z_s[i:i + 32]) for i in range(0, N, 32)])
    out_r = torch.cat([ref.calibrate(z_l[i:i + 32], z_s[i:i + 32]) for i in range(0, N, 32)])
    assert torch.equal(out_g, out_r), "the panel does not change the gate"
    assert len(g.records) == len(ref.records) and all(set(r["panel"]) == set(PANEL) for r in g.records)
    assert all(r["panel"]["nnn"] == [r["r_l"], r["r_s"]] for r in g.records), "panel nnn == the gate's own score"
    assert g.records[0]["panel"]["msp"][0] == pure.score("msp", z_l[:64]), "scored on the gate's bucket (the first 64 samples)"
    # the adaptation path (calibrate_with_grad then calibrate on the same batch) does not double-log
    g2 = G("nnn", beta=0.5, b_t=64, T_l=1.2, T_s=0.9, stats_l=st_l, stats_s=st_s, num_classes=K)
    g2.records = []
    g2.set_corruption("s", total_samples=N)
    for i in range(0, N, 32):
        g2.calibrate_with_grad(z_l[i:i + 32], z_s[i:i + 32])
        g2.calibrate(z_l[i:i + 32], z_s[i:i + 32])
    assert [r["panel"] for r in g2.records] == [r["panel"] for r in g.records]

    # ---- the screen on synthetic streams: at a collapse the "confidence" proxy RISES and "diversity" falls; healthy streams have none ----
    def stream(n, onset, name, collapse=True):
        frozen = {m: np.full(n, 0.5) + 0.01 * rng.standard_normal(n) for m in MEMBERS}
        acc = {m: frozen[m] + 0.005 * rng.standard_normal(n) for m in MEMBERS}
        div = 0.8 + 0.01 * rng.standard_normal((n, 2))
        conf = 0.6 + 0.01 * rng.standard_normal((n, 2))
        if collapse:
            for j, m in enumerate(MEMBERS):
                o = onset + 40 * j
                acc[m][o:] = 0.05
                div[o:, j] -= 0.3                                 # falls at the onset
                conf[o:, j] += 0.2                                # rises at the onset
        panel = {"nnn": div, "msp": conf, "noise": 0.5 + 0.01 * rng.standard_normal((n, 2))}
        return PanelStream(name, panel, acc, frozen, div.copy())

    pure.PROXIES.setdefault("noise", pure.ProxyDef("noise", lambda z, T=None, source_stats=None: 0.0, False, "test"))
    streams = [stream(600, 200 + 30 * k, f"S{k}") for k in range(3)] + [stream(900, 0, "H0", collapse=False)]
    res = screen(streams, ["nnn", "msp", "noise"], log=lambda *a: None)
    tab = {r["proxy"]: r for r in summary_table(res)}
    assert tab["nnn"]["orientation"] == "+" and tab["nnn"]["miss_rate"] == 0 and tab["nnn"]["delay"] < 10, tab["nnn"]
    assert tab["msp"]["orientation"] == "-", "msp rises at collapse: only its flipped orientation detects"
    assert tab["msp"]["miss_plus"] == 1.0 or tab["msp"]["delay_plus"] > tab["msp"]["delay"], tab["msp"]
    assert tab["noise"]["miss_rate"] > 0.5 or tab["noise"]["delay"] > tab["nnn"]["delay"], tab["noise"]
    tr = {(t["proxy"], t["member"]): t for t in res["track"]}
    assert tr[("nnn", "large")]["response"] < -3 and tr[("msp", "large")]["response"] > 3 and tr[("nnn", "small")]["spearman_gap"] > 0.3
    assert [r["proxy"] for r in summary_table(res)][0] == "nnn"
    # the gate-consistency check catches a panel not scored on the gate's batches
    bad = stream(300, 100, "bad")
    bad.nnn_gate = bad.nnn_gate + 0.1
    try:
        screen([bad], ["nnn"], log=lambda *a: None)
        raise AssertionError("must refuse")
    except RuntimeError:
        pass
    assert spearman(np.arange(10.0), np.arange(10.0) ** 3) == 1.0 and abs(spearman(np.arange(10.0), -np.arange(10.0)) + 1) < 1e-12
    assert spearman(np.array([1.0, 1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0, 3.0])) > 0.999
    tex = latex_table({"duo": summary_table(res)}, 1.0, 256, {"duo": ("A", "B")})
    assert r"\textbf{Nuclear norm}" in tex
    assert tex.count(r"\\") >= 4 and "tab:proxy_study" in tex
    with tempfile.TemporaryDirectory() as tmp:
        p = plot_traces(streams[0], Path(tmp) / "t.png", ("A", "B"), 256, proxies=("nnn", "msp", "noise"))
        assert p.is_file() and p.stat().st_size > 1000
    # ---- correlation with accuracy: within stream, per regime; nnn (tracks the collapse) beats msp (rises at it) on the collapsing streams ----
    cor = {r["proxy"]: r for r in correlations(streams, ["nnn", "msp", "noise"])}
    assert cor["nnn"]["S_n"] == 3 and cor["nnn"]["H_n"] == 1 and cor["nnn"]["S_ahead"] == 3
    assert cor["nnn"]["S_large"] > 0.3 and cor["msp"]["S_large"] < -0.3, (cor["nnn"], cor["msp"])
    assert cor["msp"]["S_ahead"] == 3 and cor["nnn"]["S_gap"] > cor["msp"]["S_gap"]
    assert regime(PanelStream("H__heldout__rot1__L1e-05", {}, {}, {}, None)) == "H" and regime(streams[0]) == "S"
    ctex = correlation_latex({"duo": list(cor.values())}, {"duo": ("A", "B")})
    assert "tab:proxy_correlation" in ctex and "1 healthy and 3 collapsing" in ctex and r"\textbf{Nuclear norm} (ours)" in ctex
    del pure.PROXIES["noise"]
    print("proxy_study self-test passed")
