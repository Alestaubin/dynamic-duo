"""
results.py
==========
Tidy result rows and the few aggregations every table needs. Every script writes its rows to CSV first and
renders its table or figure from that CSV alone, so a table can be regenerated (or re-formatted) without
rerunning anything.

One row = one (run, segment, series):
    duo, tta, protocol, protocol_tag, seed, segment, series, n, accuracy, ece, nll, entropy
`series` is "large", "small" or a gate's name. Values are fractions in [0, 1] (tables print percentages).

Aggregation convention (matches the existing paper tables): the "Avg." of a row is the plain mean over the
segment columns, computed PER SEED; the reported mean +- std is then over seeds (sample std, ddof=1; NaN with a
single seed). So the std of an average is the spread of the seed-wise averages, not a mean of per-column stds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

VALUE_COLS = ("accuracy", "ece", "nll", "entropy")
KEY_COLS = ("duo", "tta", "protocol_tag", "seed", "segment", "series")


def annotate(rows: Iterable[dict], spec) -> list[dict]:
    """Add the run's identity to every row."""
    ident = {"duo": spec.duo, "tta": spec.tta, "protocol": spec.protocol, "protocol_tag": spec.protocol_tag,
             "seed": spec.seed}
    return [{**ident, **r} for r in rows]


def save_rows(path: str | Path, rows: Iterable[dict] | pd.DataFrame) -> Path:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_rows(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def per_seed_average(df: pd.DataFrame, segments: Sequence[str] | None = None,
                     values: Sequence[str] = VALUE_COLS) -> pd.DataFrame:
    """Mean over segments, one row per (duo, tta, protocol_tag, series, seed). `segments` restricts the
    columns averaged (e.g. the 4 'collapse' corruptions); None = all."""
    if segments is not None:
        df = df[df["segment"].isin(list(segments))]
    keys = [k for k in KEY_COLS if k not in ("segment",)]
    return df.groupby(keys, as_index=False)[list(values)].mean()


def mean_std_over_seeds(df: pd.DataFrame, group: Sequence[str], values: Sequence[str] = VALUE_COLS) -> pd.DataFrame:
    """Mean, std (ddof=1) and the seed count of `values` within each `group` (a subset of KEY_COLS)."""
    g = df.groupby(list(group))[list(values)]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(
        df.groupby(list(group))["seed"].nunique().rename("n_seeds"))
    return out.reset_index()


def fraction_of_gap(ours: float, baseline: float, oracle: float) -> float:
    """(ours - baseline) / (oracle - baseline): the share of the oracle's improvement that `ours` captures."""
    gap = oracle - baseline
    return float("nan") if abs(gap) < 1e-12 else (ours - baseline) / gap


if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace

    spec = SimpleNamespace(duo="d", tta="tent", protocol="episodic", protocol_tag="episodic", seed=0)
    rows = []
    for seed, base in ((0, 0.50), (1, 0.52), (2, 0.54)):
        for seg, off in (("fog/s5", 0.1), ("snow/s5", -0.1)):
            for series, d in (("large", 0.0), ("ours", 0.05)):
                rows += annotate([{"segment": seg, "series": series, "n": 100, "accuracy": base + off + d,
                                   "ece": 0.1, "nll": 1.0, "entropy": 1.0}], SimpleNamespace(**{**spec.__dict__, "seed": seed}))
    df = pd.DataFrame(rows)
    avg = per_seed_average(df)
    ours = avg[avg["series"] == "ours"].sort_values("seed")
    assert ours["accuracy"].round(6).tolist() == [0.55, 0.57, 0.59]       # fog and snow offsets cancel in the mean
    ms = mean_std_over_seeds(avg, ["duo", "tta", "protocol_tag", "series"])
    r = ms[ms["series"] == "ours"].iloc[0]
    assert abs(r["accuracy_mean"] - 0.57) < 1e-9 and abs(r["accuracy_std"] - 0.02) < 1e-9 and r["n_seeds"] == 3
    only_fog = per_seed_average(df, segments=["fog/s5"])
    assert abs(only_fog[only_fog["series"] == "large"]["accuracy"].mean() - 0.62) < 1e-9
    one = mean_std_over_seeds(avg[avg["seed"] == 0], ["series"])
    assert one["accuracy_std"].isna().all(), "one seed has no std"
    assert abs(fraction_of_gap(51.3, 46.7, 53.0) - (51.3 - 46.7) / (53.0 - 46.7)) < 1e-12
    assert fraction_of_gap(1.0, 2.0, 2.0) != fraction_of_gap(1.0, 2.0, 2.0), "no gap -> NaN"
    with tempfile.TemporaryDirectory() as tmp:
        p = save_rows(Path(tmp) / "a" / "r.csv", df)
        assert load_rows(p).shape == df.shape
    print("results self-test passed")
