"""
specs.py
========
Plain-data descriptions of a run and of a gate, so a result is always traceable to the exact settings that
produced it.

RunSpec  -- everything that determines the MEMBERS' logits: duo, TTA method + hyperparameters, protocol,
            seed, sample cap. Its fingerprint keys the member-logit cache, so changing a learning rate or
            the objective-logits mode can never silently reuse stale logits.
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

# Modes whose adaptation loss is the entropy of a STATIC duo (both members' logits combined by a fixed
# calibrator) instead of each member's own entropy. The calibrator sits inside the adaptation loop, so its
# temperatures change the members' logits and are part of the spec (`loop_ts`).
DUO_MODES = ("both_duo", "large_duo", "small_duo")


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

    def fingerprint(self, duo_cfg: dict, batch_size: int, objective_logits: str) -> str:
        """12 hex chars identifying every input that affects the members' logits."""
        payload: dict[str, Any] = {
            "models": {side: duo_cfg[side] for side in ("LARGE", "SMALL")},
            "tta": [self.tta, self.tta_kwargs],
            "mode": self.mode,
            "protocol": ["independent"] if self.independent else [self.protocol, self.protocol_args],
            "seed": self.seed,
            "num_samples": self.num_samples,
            "batch_size": batch_size,
            "objective_logits": objective_logits,
        }
        if self.mode in DUO_MODES:      # only here, so every existing (independent-mode) cache keeps its key
            payload["loop_ts"] = list(self.loop_ts)
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
