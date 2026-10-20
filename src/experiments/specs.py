"""
specs.py
========
Plain-data descriptions of a run and of a gate, so a result is always traceable to the exact settings that
produced it.

RunSpec  -- everything that determines the MEMBERS' logits: duo, TTA method + hyperparameters, protocol,
            seed, sample cap. Its fingerprint keys the member-logit cache, so changing a learning rate or
            the objective-logits mode can never silently reuse stale logits. (A member that does not adapt
            has no learning rate in the key: see `RunSpec.fingerprint`.)
GateSpec -- how the two members' logits are combined into one prediction. Gate settings never change the
            members' logits (they are adapted independently), so one cache serves every GateSpec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


# Protocols whose segments each start from source weights: a segment's logits do not depend on which other
# segments the run visits, so runs that differ only in the segment list share one cache directory.
INDEPENDENT_PROTOCOLS = ("episodic", "tuning", "natural")

# The input pipeline of ImageNet-C and CCC streams, part of every cache key those streams enter. The images are 224x224
# already, and since "native224" they reach the models without the models' Resize(256) (ShiftDataset.input_transform;
# before, every stream was zoomed by 14%). The natural shifts and the clean images keep the models' own transform and
# their keys, so only the streams whose logits changed were invalidated.
INPUTS_TAG = "native224"

# Modes whose adaptation loss is the entropy of a STATIC duo (both members' logits combined by a fixed
# calibrator) instead of each member's own entropy. The calibrator sits inside the adaptation loop, so its
# temperatures change the members' logits and are part of the spec (`loop_ts`).
DUO_MODES = ("both_duo", "large_duo", "small_duo")

# mode -> (adapt_large, adapt_small): which members adapt. The one table the runner and the cache key share.
MODE_SIDES = {
    "both_indep": (True, True), "large_indep": (True, False), "small_indep": (False, True), "no_adapt": (False, False),
    "both_duo": (True, True), "large_duo": (True, False), "small_duo": (False, True),
}
assert tuple(m for m in MODE_SIDES if m.endswith("_duo")) == DUO_MODES


@dataclass(frozen=True)
class RunSpec:
    duo: str                                        # cfgs/duos/<duo>.yaml
    tta: str = "tent"
    tta_kwargs: dict = field(default_factory=dict)
    mode: str = "both_indep"                        # which members adapt, on which logits; see MemberRunner
    protocol: str = "episodic"                      # episodic | tuning | natural | continual | ccc
    protocol_args: dict = field(default_factory=dict)   # episodic/tuning: {corruptions, clean}; natural: {dataset}; continual: {order}; ccc: {difficulty, speed, ccc_seed}
    seed: int = 0                                   # stream order (subset + shuffle) seed
    num_samples: int | None = None                  # cap per segment; None = the whole stream
    loop_ts: tuple[float, float] | None = None      # (T_L, T_S) of the static duo whose entropy drives a DUO_MODES run

    def __post_init__(self):
        if self.mode in DUO_MODES and self.loop_ts is None:
            raise ValueError(f"mode {self.mode!r} adapts on the duo's entropy, so the spec needs loop_ts=(T_L, T_S)")
        if self.mode not in DUO_MODES and self.loop_ts is not None:
            raise ValueError(f"loop_ts only applies to the duo-entropy modes {DUO_MODES}, not {self.mode!r}")

    @property
    def protocol_tag(self) -> str:
        """Short, filesystem-safe name of the stream, e.g. 'episodic', 'natural-imagenet_r', 'continual-order3'."""
        a = self.protocol_args
        if self.protocol in ("episodic", "tuning"):
            sel = a.get("corruptions")
            return self.protocol if not sel else f"{self.protocol}-{'+'.join(sel)}"
        if self.protocol == "natural":
            return f"natural-{a['dataset']}"
        if self.protocol == "continual":
            return f"continual-order{a['order']}"
        if self.protocol == "ccc":
            return f"ccc-{a['difficulty']}-{a['speed']}-{a['ccc_seed']}"
        raise ValueError(f"unknown protocol {self.protocol!r}")

    @property
    def independent(self) -> bool:
        return self.protocol in INDEPENDENT_PROTOCOLS

    @property
    def cache_tag(self) -> str:
        """Cache directory name: shared by every independent-segment protocol, per stream otherwise."""
        return "independent" if self.independent else self.protocol_tag

    @property
    def adapts(self) -> dict[str, bool]:
        """{"LARGE": bool, "SMALL": bool}: which members adapt in this mode."""
        return dict(zip(("LARGE", "SMALL"), MODE_SIDES[self.mode]))

    def frozen_bn_matters(self, duo_cfg: dict) -> bool:
        """True if the --bn flag changes this run's logits: a member that does not adapt uses BatchNorm (NORM: BN).
        Adapting members always use batch statistics, and LayerNorm models have none."""
        adapt = self.adapts
        return any(not adapt[side] and duo_cfg[side].get("NORM") == "BN" for side in adapt)

    def fingerprint(self, duo_cfg: dict, batch_size: int, objective_logits: str, bn: bool = False) -> str:
        """12 hex chars identifying every input that affects the members' logits. `bn`: the --bn flag (frozen
        BatchNorm members on the test batch's statistics, instead of eval() with the source ones).
        A member that does not adapt in this mode enters the key without its OPTIM block: its optimizer is never built,
        so retuning a learning rate must not orphan the frozen logits (a frozen run's key is the same whatever
        cfgs/tuned holds). An adapting member keeps it."""
        adapt = self.adapts
        payload: dict[str, Any] = {
            "models": {side: duo_cfg[side] if adapt[side] else {k: v for k, v in duo_cfg[side].items() if k != "OPTIM"}
                       for side in ("LARGE", "SMALL")},
            "tta": [self.tta, self.tta_kwargs],
            "mode": self.mode,
            "protocol": ["independent"] if self.independent else [self.protocol, self.protocol_args],
            "seed": self.seed,
            "num_samples": self.num_samples,
            "batch_size": batch_size,
            "objective_logits": objective_logits,
        }
        if self.protocol != "natural":  # ImageNet-C / CCC streams: the natural shifts keep the key they always had
            payload["inputs"] = INPUTS_TAG
        if self.mode in DUO_MODES:      # only here, so every existing (independent-mode) cache keeps its key
            payload["loop_ts"] = list(self.loop_ts)
        # Only a run whose logits depend on it, and only for the non-default value: every cache made before the
        # flag existed used test-batch statistics, which is what bn=True means, so those keys stay valid.
        if not bn and self.frozen_bn_matters(duo_cfg):
            payload["frozen_bn"] = "source"
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def describe(self) -> dict:
        d = asdict(self)
        d["protocol_tag"] = self.protocol_tag
        return d


GATE_KINDS = ("fixed_ts", "proxy_weighted")


@dataclass(frozen=True)
class GateSpec:
    """One way of combining the members' logits. `name` is the series label in results and tables."""

    name: str
    kind: str = "proxy_weighted"                    # fixed_ts | proxy_weighted
    proxy: str = "nuclear_norm"
    calib_method: str = "identity"                  # only identity is supported for now (the paper's setting)
    beta: float = 0.5
    proxy_batch_size: int = 128
    pool: str = "log"                               # log | linear
    filter: str = "none"                            # none | running_mean | ema | kalman
    filter_kwargs: dict = field(default_factory=dict)
    fixed_ts: str | None = "checkpoints/fixed_ts/default"   # T_L, T_S checkpoint (config.json); None = T_L = T_S = 1

    def __post_init__(self):
        if self.kind not in GATE_KINDS:
            raise ValueError(f"GateSpec.kind must be one of {GATE_KINDS}, got {self.kind!r}")

    @classmethod
    def from_global(cls, g, name: str = "ours", **overrides) -> "GateSpec":
        """The paper's default gate (cfgs/global.yaml `gate:`), with any field overridden."""
        d = g.gate
        base = dict(name=name, kind="proxy_weighted", proxy=d["proxy"], calib_method=d["calib_method"],
                    beta=d["beta"], proxy_batch_size=d["proxy_batch_size"], pool=d["pool"], filter=d["filter"])
        return cls(**{**base, **overrides})

    def describe(self) -> dict:
        return asdict(self)
