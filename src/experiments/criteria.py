"""
criteria.py
===========
The CRITERION a tuner uses to choose a learning rate (or a pair, or a gate beta), and the key under which the choice is recorded.

  mean         the repo's default and what every recorded rate so far used: the candidate with the highest MEAN accuracy over the tuning shifts (the whole-stream
               accuracy macro-averaged over the held-out streams; in the continual tuning, over the visits of the stream). Ties go to the more conservative setting.
  worst_shift  among the candidates whose mean is within `tol` of the best mean, the one with the highest WORST-shift accuracy (the lowest per-shift accuracy:
               the worst held-out stream, or the worst visit in the continual tuning). A rate that gains on three shifts by collapsing one is thereby passed over for a
               slightly lower one that keeps every shift healthy. `tol` is an accuracy FRACTION (0.01 = one point); `--worst_tol` takes points.

What the criterion does NOT change: the guards (a candidate whose late accuracy is below the frozen model's, or that diverged or died, is out under either
criterion) and the grid (the same runs, the same cache: a second criterion only needs the extra runs its own finalists ask for).

Where the choice is recorded. cfgs/duo_lrs/<duo>__<regime>.yaml holds each regime's entries under `methods:` keyed by the TTA method; the `mean` choice keeps the plain
key (`tent`, exactly as before), another criterion's choice sits beside it under `<tta>@<criterion>` (`tent@worst_shift`). An evaluation script (`--criterion`) reads the
entries of the criterion it is asked for and says so when they are not recorded; it never falls back to the other criterion's rates (src/utils/duo_lrs.py).

    from src.experiments.criteria import Criterion, rank
    c = Criterion.parse("worst_shift", tol_points=1.0)
    ordered = rank(points, c, mean=lambda p: p["duo"], worst=lambda p: p["worst"], tie=lambda p: (p["lr_large"] * p["lr_small"],))
    ordered[0]        # the choice

Self-test: python -m src.experiments.criteria
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)

NAMES = ("mean", "worst_shift")
DEFAULT_TOL_POINTS = 1.0
_EPS = 1e-9                         # float slack of the tolerance comparison (accuracies are fractions)


@dataclass(frozen=True)
class Criterion:
    name: str = "mean"
    tol: float = DEFAULT_TOL_POINTS / 100        # accuracy fraction; only `worst_shift` reads it

    def __post_init__(self):
        if self.name not in NAMES:
            raise ValueError(f"criterion must be one of {NAMES}, got {self.name!r}")
        if not self.tol >= 0:
            raise ValueError(f"the tolerance must be >= 0, got {self.tol!r}")

    @classmethod
    def parse(cls, name: str | None = None, tol_points: float | None = None) -> "Criterion":
        """`name` (default mean) and the tolerance in accuracy POINTS (default 1.0)."""
        return cls(name or "mean", (DEFAULT_TOL_POINTS if tol_points is None else float(tol_points)) / 100)

    @classmethod
    def from_args(cls, args) -> "Criterion":
        """From argparse flags `--criterion` and `--worst_tol` (either may be absent)."""
        return cls.parse(getattr(args, "criterion", None), getattr(args, "worst_tol", None))

    @classmethod
    def from_json(cls, d: Mapping | None) -> "Criterion":
        """The inverse of `to_json` (a results.json written before criteria existed has none: mean)."""
        return cls() if not d else cls.parse(d.get("name"), d.get("tol_points"))

    def to_json(self) -> dict:
        return {"name": self.name, "tol_points": round(100 * self.tol, 6)}

    @property
    def robust(self) -> bool:
        return self.name == "worst_shift"

    @property
    def suffix(self) -> str:
        """The part of a config key after the TTA method: '' for mean, '@worst_shift'."""
        return "" if self.name == "mean" else f"@{self.name}"

    def key(self, tta: str) -> str:
        return f"{tta}{self.suffix}"

    def describe(self) -> str:
        return "mean" if not self.robust else f"worst_shift (mean within {100 * self.tol:g} points of the best)"


MEAN = Criterion()


def add_tol_arg(ap) -> None:
    """`--worst_tol POINTS` for a tuner (the tolerance of the worst_shift criterion, in accuracy points; ignored by the mean criterion). The criterion itself is
    `--criterion`, defined with the common arguments (src/experiments/cli.py)."""
    ap.add_argument("--worst_tol", type=float, default=None, metavar="POINTS",
                    help=f"with --criterion worst_shift: the candidates whose mean accuracy is within this many points of the best compete on their worst-shift "
                         f"accuracy (default {DEFAULT_TOL_POINTS:g})")


def folder_tag(criterion: "Criterion") -> str:
    """The suffix of a tuner's output folder, so that two criteria never overwrite each other's ranking.csv / results.json: '' for mean, '__worst_shift'."""
    return "" if criterion.name == "mean" else f"__{criterion.name}"


def method_key(tta: str, criterion: "Criterion | str | None" = None) -> str:
    """The `methods:` key of a TTA method's entries under a criterion (a name, a Criterion, or None = mean)."""
    if criterion is None:
        return tta
    return (criterion if isinstance(criterion, Criterion) else Criterion(criterion)).key(tta)


def split_key(key: str) -> tuple[str, str]:
    """(tta, criterion name) of a `methods:` key: 'tent' -> ('tent', 'mean'), 'tent@worst_shift' -> ('tent', 'worst_shift')."""
    tta, _, name = key.partition("@")
    return tta, name or "mean"


def feasible(items: Sequence[T], criterion: Criterion, mean: Callable[[T], float]) -> list[T]:
    """The items whose mean is within the tolerance of the best mean (all of them for the mean criterion)."""
    if not items or not criterion.robust:
        return list(items)
    top = max(mean(x) for x in items)
    return [x for x in items if mean(x) >= top - criterion.tol - _EPS]


def rank(items: Sequence[T], criterion: Criterion, *, mean: Callable[[T], float], worst: Callable[[T], float],
         tie: Callable[[T], tuple | float] = lambda x: 0.0) -> list[T]:
    """`items` best first under `criterion`; the caller has already dropped diverged, dead and guarded items.
    mean: by the mean, highest first, then `tie` (the more conservative setting first).
    worst_shift: the items within the tolerance of the best mean come first, by worst-shift accuracy (highest first), then mean, then `tie`; the rest follow by mean,
    so that the order stays informative. A NaN worst-shift accuracy ranks last among the feasible (it is a missing measurement, not a good one)."""
    def m(x):
        return round(mean(x), 12)

    if not criterion.robust:
        return sorted(items, key=lambda x: (-m(x), tie(x)))
    ok = feasible(items, criterion, mean)
    ok_ids = {id(x) for x in ok}
    def w(x):
        v = worst(x)
        return -math.inf if v != v else round(v, 12)

    first = sorted(ok, key=lambda x: (-w(x), -m(x), tie(x)))
    rest = sorted([x for x in items if id(x) not in ok_ids], key=lambda x: (-m(x), tie(x)))
    return first + rest


def pick(cands: Mapping[K, tuple[float, float]], criterion: Criterion, tie: Callable[[K], tuple | float] = lambda k: 0.0) -> K:
    """The chosen key of {key: (mean, worst)} under `criterion`."""
    keys = list(cands)
    return rank(keys, criterion, mean=lambda k: cands[k][0], worst=lambda k: cands[k][1], tie=tie)[0]


def score(criterion: Criterion, mean_: float, worst_: float, best_mean: float) -> tuple:
    """A comparable score (higher is better) of one candidate against the best mean of its set, in the same order as `rank`: used to ask whether a neighbour is strictly
    worse than the winner (the 'interior optimum' rule)."""
    if not criterion.robust:
        return (round(mean_, 12),)
    ok = mean_ >= best_mean - criterion.tol - _EPS
    return (1 if ok else 0, round(worst_, 12) if (ok and worst_ == worst_) else -math.inf if ok else round(mean_, 12))


if __name__ == "__main__":
    c = Criterion()
    assert c.name == "mean" and c.key("tent") == "tent" and not c.robust and c.suffix == ""
    w = Criterion.parse("worst_shift", tol_points=1.0)
    assert w.key("tent") == "tent@worst_shift" and w.robust and abs(w.tol - 0.01) < 1e-12 and w.to_json() == {"name": "worst_shift", "tol_points": 1.0}
    assert Criterion.from_json(w.to_json()) == w and Criterion.from_json(None) == MEAN and Criterion.parse("worst_shift").tol == 0.01
    assert method_key("tent") == "tent" and method_key("tent", "worst_shift") == "tent@worst_shift" and method_key("eata", w) == "eata@worst_shift"
    assert split_key("tent") == ("tent", "mean") and split_key("tent@worst_shift") == ("tent", "worst_shift")
    assert folder_tag(MEAN) == "" and folder_tag(w) == "__worst_shift"
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", default="mean")
    add_tol_arg(ap)
    assert Criterion.from_args(ap.parse_args([])) == MEAN and Criterion.from_args(ap.parse_args(["--criterion", "worst_shift", "--worst_tol", "0.5"])) == Criterion("worst_shift", 0.005)
    try:
        Criterion("best")
        raise AssertionError("an unknown criterion must be refused")
    except ValueError:
        pass
    try:
        Criterion("mean", -0.1)
        raise AssertionError("a negative tolerance must be refused")
    except ValueError:
        pass

    # (name, mean, worst): the best mean is A; B is within one point of it and keeps a better worst shift; C is 2 points down with the best worst of the feasible; D is far down
    P = [("A", 0.600, 0.20), ("B", 0.595, 0.35), ("C", 0.580, 0.50), ("D", 0.500, 0.55)]
    mean_, worst_ = (lambda p: p[1]), (lambda p: p[2])
    names = lambda ranked: [p[0] for p in ranked]                                         # noqa: E731
    assert names(rank(P, MEAN, mean=mean_, worst=worst_)) == ["A", "B", "C", "D"], "mean: highest mean first"
    assert names(rank(P, Criterion.parse("worst_shift", 1.0), mean=mean_, worst=worst_))[:2] == ["B", "A"], "within 1 point: B has the better worst shift"
    assert names(rank(P, Criterion.parse("worst_shift", 2.0), mean=mean_, worst=worst_))[:3] == ["C", "B", "A"], "within 2 points: C"
    assert names(rank(P, Criterion.parse("worst_shift", 0.0), mean=mean_, worst=worst_))[0] == "A", "tolerance 0 is the mean criterion"
    assert names(rank(P, Criterion.parse("worst_shift", 100.0), mean=mean_, worst=worst_))[0] == "D", "a huge tolerance is the worst-shift criterion"
    assert names(rank(P, Criterion.parse("worst_shift", 1.0), mean=mean_, worst=worst_))[-1] == "D", "the infeasible follow, by mean"
    # the tolerance edge is inclusive (floats: 0.60 - 0.01 = 0.59)
    assert names(rank([("A", 0.60, 0.1), ("E", 0.59, 0.9)], Criterion.parse("worst_shift", 1.0), mean=mean_, worst=worst_))[0] == "E"
    # ties: equal worst -> the higher mean; equal both -> `tie` (the smaller rate)
    T3 = [("x", 0.59, 0.3, 2.0), ("y", 0.60, 0.3, 3.0), ("z", 0.60, 0.3, 1.0)]
    assert [p[0] for p in rank(T3, Criterion.parse("worst_shift", 1.0), mean=lambda p: p[1], worst=lambda p: p[2], tie=lambda p: p[3])] == ["z", "y", "x"]
    assert [p[0] for p in rank(T3, MEAN, mean=lambda p: p[1], worst=lambda p: p[2], tie=lambda p: p[3])] == ["z", "y", "x"]
    # a NaN worst-shift accuracy never wins among the feasible
    assert names(rank([("A", 0.60, float("nan")), ("B", 0.60, 0.1)], Criterion.parse("worst_shift"), mean=mean_, worst=worst_))[0] == "B"
    assert pick({"a": (0.6, 0.2), "b": (0.595, 0.35)}, Criterion.parse("worst_shift")) == "b" and pick({"a": (0.6, 0.2), "b": (0.595, 0.35)}, MEAN) == "a"
    assert feasible(P, MEAN, mean_) == P and [p[0] for p in feasible(P, Criterion.parse("worst_shift", 1.0), mean_)] == ["A", "B"]
    # score(): the winner beats an infeasible neighbour and a feasible one with a lower worst shift
    s = lambda cr, p: score(cr, p[1], p[2], 0.600)                                        # noqa: E731
    assert s(w, P[1]) > s(w, P[0]) > s(w, P[2]) > s(w, P[3]) and s(MEAN, P[0]) > s(MEAN, P[1])

    # ---- the config layer: both criteria's winners side by side, each resolved on its own, never replaced by the other's -----------------------------
    import json
    import tempfile
    from pathlib import Path

    import yaml

    from src.utils import duo_lrs

    def opt(lr):
        return {"METHOD": "SGD", "STEPS": 1, "LR": lr, "MOMENTUM": 0.9, "DAMPENING": 0.0, "WD": 0.0, "NESTEROV": False}

    with tempfile.TemporaryDirectory() as tmp:
        root, duos = Path(tmp) / "lrs", Path(tmp) / "duos"
        duos.mkdir()
        (duos / "fake.yaml").write_text(yaml.safe_dump({"NAME": "fake", "LARGE": {"NAME": "ml", "NORM": "LN"}, "SMALL": {"NAME": "ms", "NORM": "BN"}}))
        duo_lrs.init_indep("fake", root, duos, defaults={"large": opt(1e-4), "small": opt(1e-3)})
        spec = duo_lrs._duo_spec("fake", duos)

        def write(name, d):
            f = Path(tmp) / name
            f.write_text(json.dumps(d))
            return f

        base = {"duo": "fake", "tta": "tent", "num_samples": None, "bn": True, "seeds": [0], "streams": ["a__s5", "b__s5"], "batch_size": 64,
                "temperatures": {"T_l": 1.0, "T_s": 1.0, "checkpoint": "/x/ckpt"}}
        for crit, (lrl, lrs) in (({}, (1e-3, 2e-3)), ({"criterion": w.to_json()}, (5e-4, 2e-3))):
            joint = {**base, **crit, "optim": {"large": opt(0), "small": opt(0)}, "lrs_large": [1e-3], "lrs_small": [2e-3],
                     "best": {"lr_large": lrl, "lr_small": lrs, "duo": 0.6, "worst": 0.4}}
            duo_lrs.import_joint(write("joint.json", joint), root=root, duos_dir=duos)
            for side, lr in (("large", 2e-3 if not crit else 1e-3), ("small", 3e-3 if not crit else 1.5e-3)):
                anchor = {**base, **crit, "side": side, "optim": {side: opt(0)}, "lrs": [lr], "best": {"lr": lr, "duo": 0.6, "worst": 0.4}}
                duo_lrs.import_anchor(write("anchor.json", anchor), root=root, duos_dir=duos)
        L = lambda regime, criterion=None: {k: v["LR"] for k, v in duo_lrs.resolve(spec, regime, "tent", root, criterion=criterion).items()}      # noqa: E731
        assert L("joint") == {"LARGE": 1e-3, "SMALL": 2e-3} and L("joint", "worst_shift") == {"LARGE": 5e-4, "SMALL": 2e-3}
        assert L("large_duo")["LARGE"] == 2e-3 and L("large_duo", "worst_shift")["LARGE"] == 1e-3
        assert L("small_duo")["SMALL"] == 3e-3 and L("small_duo", "worst_shift")["SMALL"] == 1.5e-3
        assert L("indep") == {"LARGE": 1e-4, "SMALL": 1e-3}, "mean: the indep defaults, as before"
        doc = duo_lrs.load("fake", "joint", root)
        assert sorted(doc["methods"]) == ["tent", "tent@worst_shift"] and duo_lrs.recorded_criteria(doc, "tent") == ["mean", "worst_shift"]
        assert doc["methods"]["tent@worst_shift"]["large"]["criterion"] == w.to_json() and "criterion" not in doc["methods"]["tent"]["large"]
        for regime in ("indep", "single"):
            try:
                duo_lrs.resolve(spec, regime, "tent", root, criterion="worst_shift")
                raise AssertionError(f"{regime}: the defaults must not stand in for an unrecorded criterion")
            except duo_lrs.NotTuned as e:
                assert "worst_shift" in str(e)
        try:
            duo_lrs.resolve(spec, "joint", "eata", root, criterion="worst_shift")
            raise AssertionError("another TTA method has nothing under the criterion")
        except duo_lrs.NotTuned:
            pass
        assert duo_lrs.check("fake", root, duos) == [], duo_lrs.check("fake", root, duos)
    print("criteria self-test passed")
