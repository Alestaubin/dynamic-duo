"""
protocols.py
============
A protocol turns a RunSpec into the list of streams (Segments) a run visits, in order.

    episodic   the 15 ImageNet-C test corruptions; both members are reset to source before EACH one
    tuning     the 4 held-out corruptions, episodic (hyperparameter tuning only; never reported as results)
    natural    one natural-shift dataset (ImageNet-R/A/Sketch/V2), episodic
    continual  the 15 corruptions in one of CoTTA's 10 orders; reset ONCE at the start, never between
    ccc        one 7.5M-image CCC stream; reset once at the start

`reset_before` is the whole difference between episodic and continual: the continual loop is the episodic
loop minus the resets. Each Segment carries its own loader (one corruption per loader), so a continual
stream never mixes two corruptions inside a batch.

The gate is told about boundaries only where the members are (reset_before). In continual and CCC streams
nothing marks a shift, which is the setting the paper's reset rule has to work in.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.experiments.specs import RunSpec
from src.utils.config import GlobalConfig
from src.utils.datasets import CCC, ImageNetC, ImageNetTrainSlice, ImageNetVal, ShiftDataset, get_dataset


@dataclass
class Segment:
    label: str                  # "fog/s5", "imagenet_r", "ccc/medium/1000/43" -- also the cache file name
    dataset: ShiftDataset
    reset_before: bool          # reset the members (and tell the gate) before this segment

    @property
    def file_stem(self) -> str:
        return self.label.replace("/", "__")


def build_segments(spec: RunSpec, g: GlobalConfig) -> list[Segment]:
    a = spec.protocol_args
    severity = a.get("severity", g.severity)

    if spec.protocol in ("episodic", "tuning"):
        default = g.test_corruptions if spec.protocol == "episodic" else g.tuning_corruptions
        corruptions = a.get("corruptions") or default
        segments = [Segment(f"{c}/s{severity}", ImageNetC.from_config(g, c, severity), True) for c in corruptions]
        # `clean`: also visit the clean images first ("val": the 50,000 validation images; "train_slice": the
        # held-out clean slice of the training set). Used by the temperature studies.
        clean = a.get("clean")
        if clean == "val":
            segments.insert(0, Segment("clean", ImageNetVal.from_config(g), True))
        elif clean == "train_slice":
            segments.insert(0, Segment("clean_train_slice", ImageNetTrainSlice.from_config(g), True))
        elif clean:
            raise ValueError(f"protocol_args['clean'] must be 'val' or 'train_slice', got {clean!r}")
        return segments

    if spec.protocol == "natural":
        return [Segment(a["dataset"], get_dataset(a["dataset"], g), True)]

    if spec.protocol == "continual":
        order = g.cotta_orders[a["order"]]
        return [Segment(f"{c}/s{severity}", ImageNetC.from_config(g, c, severity), i == 0)
                for i, c in enumerate(order)]

    if spec.protocol == "ccc":
        ds = CCC.from_config(g, a["difficulty"], a["speed"], a["ccc_seed"])
        return [Segment(f"ccc/{a['difficulty']}/{a['speed']}/{a['ccc_seed']}", ds, True)]

    raise ValueError(f"unknown protocol {spec.protocol!r}")


def resets_between_segments(spec: RunSpec) -> bool:
    """True when every segment starts from source weights, so segments are independent and cacheable
    one by one. False for continual/CCC, where segment k depends on everything before it."""
    return spec.protocol in ("episodic", "tuning", "natural")


if __name__ == "__main__":
    from src.utils.config import load_global

    g = load_global()
    ep = build_segments(RunSpec("vitb16_rn50"), g)
    assert [s.label for s in ep] == [f"{c}/s5" for c in g.test_corruptions] and all(s.reset_before for s in ep)

    for i in (0, 9):
        co = build_segments(RunSpec("vitb16_rn50", protocol="continual", protocol_args={"order": i}), g)
        assert [s.label.split("/")[0] for s in co] == g.cotta_orders[i]
        assert [s.reset_before for s in co] == [True] + [False] * 14, "continual resets only once"
    assert not resets_between_segments(RunSpec("x", protocol="continual", protocol_args={"order": 0}))

    tu = build_segments(RunSpec("vitb16_rn50", protocol="tuning"), g)
    assert [s.label for s in tu] == [f"{c}/s5" for c in g.tuning_corruptions]
    assert not {s.label for s in tu} & {s.label for s in ep}, "tuning and test streams must be disjoint"

    nat = build_segments(RunSpec("vitb16_rn50", protocol="natural", protocol_args={"dataset": "imagenet_r"}), g)
    assert len(nat) == 1 and nat[0].dataset.num_classes == 200 and nat[0].file_stem == "imagenet_r"
    assert RunSpec("x", protocol="ccc", protocol_args={"difficulty": "medium", "speed": 1000, "ccc_seed": 43}).protocol_tag \
        == "ccc-medium-1000-43"

    # `clean` adds a leading clean-validation segment; a held-out corruption list is just another episodic list
    cl = build_segments(RunSpec("vitb16_rn50", protocol_args={"corruptions": ["fog"], "clean": "val"}), g)
    assert [s.label for s in cl] == ["clean", "fog/s5"] and cl[0].dataset.num_classes == 1000
    assert [s.label for s in build_segments(RunSpec("x", protocol_args={"corruptions": ["spatter"]}), g)] == ["spatter/s5"]
    try:
        build_segments(RunSpec("x", protocol_args={"clean": "nope"}), g)
        raise AssertionError("bad clean source must raise")
    except ValueError:
        pass

    # the fingerprint changes with anything that changes the members' logits, and with nothing else
    duo = {"LARGE": {"NAME": "a", "OPTIM": {"LR": 1e-4}}, "SMALL": {"NAME": "b", "OPTIM": {"LR": 1e-3}}}
    base = RunSpec("d").fingerprint(duo, 64, "masked")
    assert base == RunSpec("d").fingerprint(duo, 64, "masked")
    # independent segments share a cache whatever segment list a run asks for ...
    assert base == RunSpec("d", protocol_args={"corruptions": ["fog"], "clean": "val"}).fingerprint(duo, 64, "masked")
    assert base == RunSpec("d", protocol="tuning").fingerprint(duo, 64, "masked")
    # ... except the natural shifts: they never went through the ImageNet-C input pipeline (no resize since "native224"),
    # so their caches keep the key they had before it; every ImageNet-C / CCC key moved
    nat = RunSpec("d", protocol="natural", protocol_args={"dataset": "imagenet_r"}).fingerprint(duo, 64, "masked")
    assert nat == "e0ec2ee19e0c" and nat != base, "natural-shift caches predate the input-pipeline tag and stay valid"
    assert RunSpec("d", protocol="continual", protocol_args={"order": 0}).fingerprint(duo, 64, "masked") != nat
    assert RunSpec("d").cache_tag == RunSpec("d", protocol="tuning").cache_tag == "independent"
    # ... but a continual stream is a different run per order, and never shares with independent ones
    c0 = RunSpec("d", protocol="continual", protocol_args={"order": 0})
    assert c0.fingerprint(duo, 64, "masked") != RunSpec("d", protocol="continual", protocol_args={"order": 1}).fingerprint(duo, 64, "masked")
    assert c0.fingerprint(duo, 64, "masked") != base and c0.cache_tag == "continual-order0"
    assert base != RunSpec("d", seed=1).fingerprint(duo, 64, "masked")
    assert base != RunSpec("d", tta="eata").fingerprint(duo, 64, "masked")
    assert base != RunSpec("d").fingerprint(duo, 64, "full")
    assert base != RunSpec("d").fingerprint(duo, 32, "masked")
    assert base != RunSpec("d").fingerprint({**duo, "LARGE": {"NAME": "a", "OPTIM": {"LR": 2e-4}}}, 64, "masked")
    # the key of an independent-mode run is pinned: changing it orphans every member-logit cache on disk
    assert base == "b0b70ed382d7", f"fingerprint payload changed: every cache under cache/members is now unreachable ({base})"
    # duo-entropy modes: the static duo's temperatures are part of the key (they change what the members learn)
    j = RunSpec("d", mode="both_duo", loop_ts=(1.4, 0.9)).fingerprint(duo, 64, "masked")
    assert j != base and j != RunSpec("d", mode="both_duo", loop_ts=(1.5, 0.9)).fingerprint(duo, 64, "masked")
    assert j != RunSpec("d", mode="large_duo", loop_ts=(1.4, 0.9)).fingerprint(duo, 64, "masked")
    for bad in (dict(mode="both_duo"), dict(mode="both_indep", loop_ts=(1.0, 1.0))):
        try:
            RunSpec("d", **bad)
            raise AssertionError(f"{bad} must be rejected")
        except ValueError:
            pass
    # --bn (frozen BatchNorm members on test-batch statistics) enters the key only where it changes the logits:
    # a frozen member with NORM: BN. bn=True is the key every cache had before the flag existed.
    duo_bn = {"LARGE": {"NAME": "a", "NORM": "LN", "OPTIM": {"LR": 1e-4}}, "SMALL": {"NAME": "b", "NORM": "BN", "OPTIM": {"LR": 1e-3}}}
    fp = lambda mode, bn: RunSpec("d", mode=mode).fingerprint(duo_bn, 64, "masked", bn)     # noqa: E731
    assert fp("both_indep", False) == fp("both_indep", True), "nothing frozen: the flag is irrelevant"
    assert fp("small_indep", False) == fp("small_indep", True), "the frozen member is LayerNorm: irrelevant"
    assert fp("large_indep", False) != fp("large_indep", True) and fp("no_adapt", False) != fp("no_adapt", True)
    assert RunSpec("d", mode="no_adapt").fingerprint(duo_bn, 64, "masked", True) == "acaf911e89ab", \
        "the key of a frozen run is pinned (it was f609c5ff6bf6 until a non-adapting member lost its OPTIM block)"
    # a member that does not adapt has no optimizer: its OPTIM block is not in the key, an adapting member's is
    lr = lambda l, s: {"LARGE": {**duo_bn["LARGE"], "OPTIM": {"LR": l}}, "SMALL": {**duo_bn["SMALL"], "OPTIM": {"LR": s}}}   # noqa: E731
    key = lambda mode, d: RunSpec("d", mode=mode).fingerprint(d, 64, "masked", True)    # noqa: E731
    assert key("no_adapt", lr(1e-4, 1e-3)) == key("no_adapt", lr(5e-5, 7e-3)), "frozen: no learning rate in the key"
    assert key("large_indep", lr(1e-4, 1e-3)) == key("large_indep", lr(1e-4, 7e-3)), "the frozen SMALL side's LR is ignored"
    assert key("large_indep", lr(1e-4, 1e-3)) != key("large_indep", lr(5e-5, 1e-3)), "the adapting LARGE side's LR counts"
    assert key("small_indep", lr(1e-4, 1e-3)) == key("small_indep", lr(5e-5, 1e-3)) != key("small_indep", lr(1e-4, 7e-3))
    assert key("both_indep", lr(1e-4, 1e-3)) != key("both_indep", lr(1e-4, 7e-3)) and \
        key("both_indep", lr(1e-4, 1e-3)) != key("both_indep", lr(5e-5, 1e-3)), "both adapt: both LRs count"
    assert key("no_adapt", lr(1e-4, 1e-3)) != key("both_indep", lr(1e-4, 1e-3)), "the mode is still in the key"
    assert key("no_adapt", {**lr(1e-4, 1e-3), "SMALL": {**lr(1e-4, 1e-3)["SMALL"], "NAME": "c"}}) != key("no_adapt", lr(1e-4, 1e-3)), \
        "a frozen member's identity still counts"
    print("protocols self-test passed")
