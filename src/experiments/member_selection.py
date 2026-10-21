"""
member_selection.py
===================
Choose each member's learning rate from the stored logits of a sweep (Phase 3, Stage D), offline, with no model.

The members adapt independently, so run i of a sweep (scripts/04_cache_logits.py --grid) holds the i-th value of EACH model's grid, and
a member's quality can be read off its own logits: the criterion is its STANDALONE mean accuracy over the held-out streams (the
same criterion scripts/02_tune_tta.py uses for Tent), computed per stream and averaged, with the worst held-out stream reported
next to it. Ties go to the smaller learning rate. When the best value is the first or the last of the values tried, the optimum
may lie outside the grid: `edge` says so (the author allowed a 5e-5 point for the ViT in that case).

`write_tuned` records the choice in the learning-rate file of every duo that uses the model (cfgs/duo_lrs/<duo>__indep.yaml,
src/utils/duo_lrs.py), which `load_duo(..., tta=<algorithm>)` reads, so every later run of that algorithm picks it up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from src.experiments.grids import sgd_optim
from src.experiments.logit_runs import run_member_specs, with_optim
from src.experiments.logit_store import LogitStore
from src.experiments.specs import RunSpec
from src.utils import duo_lrs


def member_accuracies(store: LogitStore, member, stems: Sequence[str]) -> dict[str, float]:
    """Standalone top-1 accuracy of one stored member on each stream."""
    out = {}
    for stem in stems:
        d = store.read(member, stem)                       # centered logits: the argmax is the raw logits' argmax
        out[stem] = float((d["z"].argmax(1) == d["labels"]).double().mean())
    return out


def select_side(store: LogitStore, ctx, spec: RunSpec, side: str, lrs: Sequence[float], stems: Sequence[str],
                momentum: float = 0.9) -> dict:
    """The learning-rate table of one side ("large" | "small") and the choice."""
    rows = []
    for lr in lrs:
        c = with_optim(ctx, **{side: sgd_optim(lr, momentum)})
        member = run_member_specs(spec, c)[0 if side == "large" else 1]
        missing = [s for s in stems if not store.has(member, s)]
        if missing:
            raise FileNotFoundError(f"{side} member of lr {lr:g} ({member.key}) is missing {missing} in {store.root}")
        acc = member_accuracies(store, member, stems)
        worst = min(acc, key=acc.get)
        rows.append({"lr": lr, "member_key": member.key, "accuracy": acc, "macro": sum(acc.values()) / len(acc),
                     "worst": acc[worst], "worst_stream": worst})
    best = max(rows, key=lambda r: (round(r["macro"], 12), -r["lr"]))
    ordered = sorted(r["lr"] for r in rows)
    return {"side": side, "rows": rows, "best_lr": best["lr"], "best_macro": best["macro"], "best_worst": best["worst"],
            "edge": len(ordered) > 1 and best["lr"] in (ordered[0], ordered[-1]),
            "edge_side": None if len(ordered) < 2 or best["lr"] not in (ordered[0], ordered[-1])
            else ("lowest" if best["lr"] == ordered[0] else "highest")}


def select(store: LogitStore, ctx, spec: RunSpec, lrs: dict[str, Sequence[float]], stems: Sequence[str],
           momentum: float = 0.9) -> dict:
    """{"large": select_side(...), "small": select_side(...)} for the learning rates `lrs` = {"large": [...], "small": [...]}."""
    return {side: select_side(store, ctx, spec, side, lrs[side], stems, momentum) for side in ("large", "small")}


def write_tuned(root: Path, record: dict, duos: list[str] | None = None) -> list[Path]:
    """Record the choice in the indep learning-rate file of every duo that uses the model (all of them, or `duos`)."""
    return duo_lrs.record_tuned(record["model"], record["tta"], record, root=root, duos=duos)


def tuned_record(choice: dict, *, model: str, algorithm: str, norm: str, batch_size: int, stems: Sequence[str], seed: int,
                 momentum: float = 0.9) -> dict:
    return {"model": model, "tta": algorithm, "NORM": norm, "batch_size": batch_size,
            "OPTIM": sgd_optim(choice["best_lr"], momentum),
            "tuned_on": {"streams": list(stems), "seed": seed},
            "macro_accuracy": round(choice["best_macro"], 5), "worst_stream_accuracy": round(choice["best_worst"], 5),
            "lrs_tried": sorted(r["lr"] for r in choice["rows"]), "written_by": "scripts/05_select_member_lr.py"}


if __name__ == "__main__":
    import copy
    import tempfile

    import numpy as np

    from src.experiments._testing import make_context
    from src.experiments.logit_store import MemberSpec
    from src.utils.config import GlobalConfig, load_duo, load_global

    torch.manual_seed(0)
    stems = ["a__s5", "b__s5", "c__s5", "d__s5"]
    n, K = 120, 20

    def logits_with_accuracy(acc: float, labels: torch.Tensor, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(len(labels), K, generator=g)
        right = torch.rand(len(labels), generator=g) < acc
        pred = torch.where(right, labels, (labels + 1 + torch.randint(0, K - 1, (len(labels),), generator=g)) % K)
        z[torch.arange(len(labels)), pred] += 20.0
        return z

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ctx = make_context(tmp, batch_size=32)
        store = LogitStore(tmp / "logits", staging=None)
        spec = RunSpec("tiny", seed=0)
        labels = {s: torch.randint(0, K, (n,), generator=torch.Generator().manual_seed(i)) for i, s in enumerate(stems)}
        order = np.arange(n, dtype=np.int32)
        # large: lr 0.1 -> .50, 0.2 -> .70, 0.3 -> .60 (the middle one wins; stream d is its worst)
        # small: lr 0.1 -> .30, 0.2 -> .40, 0.3 -> .55 (the last one wins: the optimum may lie beyond the grid)
        table = {"large": {0.1: 0.5, 0.2: 0.7, 0.3: 0.6}, "small": {0.1: 0.3, 0.2: 0.4, 0.3: 0.55}}
        for side, accs in table.items():
            for lr, acc in accs.items():
                m = run_member_specs(spec, with_optim(ctx, **{side: sgd_optim(lr)}))[0 if side == "large" else 1]
                for i, s in enumerate(stems):
                    a = acc - (0.2 if (side == "large" and lr == 0.2 and s == "d__s5") else 0.0)
                    store.write(m, s, logits_with_accuracy(a, labels[s], 100 * i + int(lr * 10)), labels[s], order, [32, 32, 32, 24])
        lrs = {"large": [0.1, 0.2, 0.3], "small": [0.1, 0.2, 0.3]}
        res = select(store, ctx, spec, lrs, stems)
        L, S = res["large"], res["small"]
        assert L["best_lr"] == 0.2 and S["best_lr"] == 0.3
        assert not L["edge"] and L["edge_side"] is None and S["edge"] and S["edge_side"] == "highest"
        row = next(r for r in L["rows"] if r["lr"] == 0.2)
        assert row["worst_stream"] == "d__s5" and row["worst"] < row["macro"] - 0.03 and len(row["accuracy"]) == 4
        assert abs(L["best_macro"] - 0.675) < 0.07 and L["best_worst"] <= L["best_macro"]
        # the accuracy really is the member's standalone one
        m = run_member_specs(spec, with_optim(ctx, large=sgd_optim(0.2)))[0]
        d = store.read(m, "a__s5")
        assert row["accuracy"]["a__s5"] == float((d["z"].argmax(1) == d["labels"]).double().mean())
        # ties go to the smaller learning rate
        tie = copy.deepcopy(table)
        for s in stems:
            for lr in (0.1, 0.2, 0.3):
                m = run_member_specs(spec, with_optim(ctx, small=sgd_optim(lr)))[1]
                store.write(m, s, logits_with_accuracy(0.45, labels[s], 7), labels[s], order, [32, 32, 32, 24])
        assert select_side(store, ctx, spec, "small", [0.1, 0.2, 0.3], stems)["best_lr"] == 0.1
        # a missing stream is an error that names it
        try:
            select_side(store, ctx, spec, "large", [0.1, 0.4], stems)
            raise AssertionError("a missing member must raise")
        except FileNotFoundError as e:
            assert "0.4" in str(e)
        assert select_side(store, ctx, spec, "large", [0.2], stems)["edge"] is False, "one value tried: no edge to speak of"

        # ---- the recorded choice is what load_duo reads for that algorithm ----------------------------------------------------
        raw = copy.deepcopy(load_global().raw)
        raw["paths"]["duo_lrs"] = str(tmp / "lrs")
        g = GlobalConfig(raw, Path("<test>"))
        duo_lrs.init_indep("vitb16_rn50v1", tmp / "lrs", defaults={"large": sgd_optim(1e-3), "small": sgd_optim(2.5e-4)})
        base = load_duo("vitb16_rn50v1", g, tta="eta")
        for side, name in (("large", "vit_b_16"), ("small", "resnet50_v1")):
            cfg = base[side.upper()]
            assert cfg["NAME"] == name
            write_tuned(tmp / "lrs", tuned_record(res[side], model=cfg["NAME"], algorithm="eta", norm=cfg["NORM"],
                                                  batch_size=g.batch_size, stems=stems, seed=0), duos=["vitb16_rn50v1"])
        tuned = load_duo("vitb16_rn50v1", g, tta="eta")
        assert tuned["LARGE"]["OPTIM"]["LR"] == 0.2 and tuned["SMALL"]["OPTIM"]["LR"] == 0.3
        assert tuned["LARGE"]["OPTIM"]["MOMENTUM"] == 0.9 and tuned["LARGE"]["OPTIM"]["METHOD"] == "SGD"
        assert load_duo("vitb16_rn50v1", g, tta="sar")["LARGE"]["OPTIM"] == base["LARGE"]["OPTIM"], "other algorithms are untouched"
        rec = duo_lrs.load("vitb16_rn50v1", "indep", tmp / "lrs")["methods"]["eta"]["large"]
        assert rec["lrs_tried"] == [0.1, 0.2, 0.3] and rec["tuned_on"] == [s.replace("__", "/") for s in stems] and rec["seed"] == 0
        json.dumps(res)                                   # the table is plain data
    print("member_selection self-test passed")
