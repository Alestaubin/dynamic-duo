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
from src.utils.datasets import CCC, ImageNetC, ShiftDataset, get_dataset


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
        return [Segment(f"{c}/s{severity}", ImageNetC.from_config(g, c, severity), True) for c in corruptions]

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

    # the fingerprint changes with anything that changes the members' logits, and with nothing else
    duo = {"LARGE": {"NAME": "a", "OPTIM": {"LR": 1e-4}}, "SMALL": {"NAME": "b", "OPTIM": {"LR": 1e-3}}}
    base = RunSpec("d").fingerprint(duo, 64, "masked")
    assert base == RunSpec("d").fingerprint(duo, 64, "masked")
    assert base != RunSpec("d", seed=1).fingerprint(duo, 64, "masked")
    assert base != RunSpec("d", tta="eata").fingerprint(duo, 64, "masked")
    assert base != RunSpec("d").fingerprint(duo, 64, "full")
    assert base != RunSpec("d").fingerprint(duo, 32, "masked")
    assert base != RunSpec("d").fingerprint({**duo, "LARGE": {"NAME": "a", "OPTIM": {"LR": 2e-4}}}, 64, "masked")
    print("protocols self-test passed")
