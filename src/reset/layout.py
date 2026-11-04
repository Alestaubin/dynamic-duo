"""
layout.py
=========
Where the reset tuning keeps its files, for any duo (no torch: the driver and the report scripts import it cheaply).

    P = Paths(load_global(), "convnextb_deits")
    P.live_dir            <out>/results/reset_live/<duo>__tent_bn/        the live runs (one .npz + .gate.json each)
    P.continual_dir       <out>/results/tune_continual_lr/<tag>/          the continual-rate runs and results.json (conservative / stress rates)
    P.gate_stress_dir     <out>/results/gate_stress/<tag>/                the calibration streams the screens replay (stress S, healthy H)
    P.shared_screen       <out>/results/screen_resets_stress/shared_setting__<duo>.json        the screen of the shared detector setting
    P.member_screen(side) <out>/results/screen_resets_stress/member_detector__<duo>__<side>.json   the per-member candidates
    P.tune_dir            <out>/results/reset_tune/<duo>/                 config.json, the states of the search, the job specs

The run files have ONE definition here, used by the driver (is it done?) and by the workers (where to write it):

    P.continual_run(seed, lrs)           tune_continual_lr/<tag>/runs/seed<s>__L<lr>_S<lr>.npz       (lrs None: the frozen reference)
    P.calibration_run(name, ...)         gate_stress/<tag>/exp-<name>__heldout[__rot<k>]__L<lr>_S<lr>[__<gate>]__seed<s>[__frozen].npz
    P.live_run(cond, ...)                reset_live/<tag>/live-<cond>__heldout[__rot<k>]__L<lr>_S<lr>__<gate>__seed<s>.npz

The SIGNAL is the loss the members adapt on: "indep" (each member on its own entropy: Vanilla DD, the names above) or "joint" (both on the gated duo's entropy: the gate
sits inside the loss). The joint signal has its own folders (`<duo>__tent_bn_joint`, `<duo>__joint` in file names), because a run's file name does not say which it is.
The main duo's independent state files predate the per-duo folders and stay in <out>/results/reset_tune/ itself (`LEGACY_DUO`).

The CRITERION (src/experiments/criteria.py) is how the search chooses among rates and settings: `mean` (the default) or `worst_shift` (the highest worst-visit accuracy among
those whose mean is within a tolerance of the best). It changes only the DECISIONS, so only the study's own folder `tune_dir` (config.json, the states, the job specs) is
tagged with it (`<duo>[__joint]__worst_shift`); every run, calibration stream and screen is shared with the mean study, whose files they are (their names say condition and rates), so a
second study only needs the extra runs its own choices ask for. The main duo's mean study keeps its flat folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.experiments.criteria import NAMES as CRITERIA
from src.reset import live_report as LV
from src.utils.config import GlobalConfig

LEGACY_DUO = "vitb16_rn50v1"
TTA = "tent"
SIDES = ("large", "small")
SIGNALS = ("indep", "joint")


@dataclass(frozen=True)
class Paths:
    g: GlobalConfig
    duo: str
    signal: str = "indep"
    criterion: str = "mean"
    fixed_beta: float | None = None     # joint signal: a study whose gate beta is FIXED (every cell at it, no beta block); its own folder, everything else shared
    tta: str = TTA                      # the TTA method: Tent keeps every name it had; another method has its own folders, screens and study (`__<tta>`)

    def __post_init__(self):
        if self.signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}, got {self.signal!r}")
        if self.criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {self.criterion!r}")
        if self.fixed_beta is not None and self.signal != "joint":
            raise ValueError("fixed_beta is a joint-signal study (the independent members do not depend on the gate)")

    @property
    def root(self) -> Path:
        return self.g.path("out") / "results"

    @property
    def name(self) -> str:
        """The duo as file names know it: a TTA method other than Tent carries `__<tta>`, the joint signal's files `__joint`."""
        base = self.duo if self.tta == TTA else f"{self.duo}__{self.tta}"
        return base if self.signal == "indep" else f"{base}__joint"

    @property
    def study(self) -> str:
        """The name of the study (the tuning decisions): `name`, tagged with a fixed beta and with a criterion other than mean. Chain jobs and the study's folder are
        named by it."""
        beta = "" if self.fixed_beta is None else f"__beta{self.fixed_beta:g}"
        return self.name + beta + ("" if self.criterion == "mean" else f"__{self.criterion}")

    @property
    def tag(self) -> str:
        return f"{self.duo}__{self.tta}_bn" + ("" if self.signal == "indep" else "_joint")

    @property
    def continual_dir(self) -> Path:
        return self.root / "tune_continual_lr" / self.tag

    @property
    def gate_stress_dir(self) -> Path:
        return self.root / "gate_stress" / self.tag

    def live_dir(self, gates: bool = False) -> Path:
        return self.root / ("reset_gates" if gates else "reset_live") / self.tag

    @property
    def screens_dir(self) -> Path:
        return self.root / "screen_resets_stress"

    @property
    def shared_screen(self) -> Path:
        return self.screens_dir / f"shared_setting__{self.name}.json"

    def member_screen(self, side: str) -> Path:
        return self.screens_dir / f"member_detector__{self.name}__{side}.json"

    @property
    def tune_dir(self) -> Path:
        base = self.root / "reset_tune"
        return base if (self.duo == LEGACY_DUO and self.tta == TTA and self.signal == "indep" and self.criterion == "mean") else base / self.study

    # ---- run files -------------------------------------------------------------------------------------------------------
    def continual_run(self, seed: int, lrs: tuple[float, float] | None) -> Path:
        """The file of one 04b-type run: the frozen reference (`lrs` None) or the diagonal run at (lr_large, lr_small)."""
        return self.continual_dir / "runs" / (f"seed{seed}__frozen.npz" if lrs is None else f"seed{seed}__L{lrs[0]:g}_S{lrs[1]:g}.npz")

    def calibration_run(self, name: str, *, lrs: tuple[float, float], seed: int, rotation: int = 0, beta: float = 0.5, b_t: int = 256, proxy: str = "nnn",
                        frozen: bool = False) -> Path:
        """One calibration stream on the held-out stream (stress S, healthy H): the selected gate (nnn, 256, 0.5) keeps the bare name, another gate is in it."""
        gate = "" if (proxy, b_t, beta) == ("nnn", 256, 0.5) else f"__{proxy}_bt{b_t}_beta{beta:g}"
        rot = f"__rot{rotation}" if rotation else ""
        return self.gate_stress_dir / f"exp-{name}__heldout{rot}__L{lrs[0]:g}_S{lrs[1]:g}{gate}__seed{seed}{'__frozen' if frozen else ''}.npz"

    def live_run(self, cond: str, *, stream: str, rot: int, seed: int, lrs: tuple[float, float], proxy: str = "nnn", b_t: int = 256, beta: float = 0.5,
                 order: int | None = None, max_images: int | None = None, gates: bool = False) -> Path:
        stem = LV.run_stem(cond, stream=stream, rot=rot, seed=seed, lrs=lrs, proxy=proxy, b_t=b_t, beta=beta, max_images=max_images, order=order)
        return self.live_dir(gates) / f"{stem}.npz"

    @property
    def config_path(self) -> Path:
        return self.tune_dir / "config.json"

    @property
    def jobs_dir(self) -> Path:
        return self.tune_dir / "jobs"


if __name__ == "__main__":
    from src.utils.config import load_global

    g = load_global()
    a, b = Paths(g, LEGACY_DUO), Paths(g, "convnextb_deits")
    w = Paths(g, LEGACY_DUO, criterion="worst_shift")                  # a study of another criterion: its own folder, everything else shared
    assert w.tune_dir == w.root / "reset_tune" / f"{LEGACY_DUO}__worst_shift" and w.study == f"{LEGACY_DUO}__worst_shift" and a.study == LEGACY_DUO
    assert w.live_dir() == a.live_dir() and w.continual_dir == a.continual_dir and w.shared_screen == a.shared_screen and w.gate_stress_dir == a.gate_stress_dir
    fb = Paths(g, LEGACY_DUO, "joint", fixed_beta=0.25)                  # a fixed-beta joint study: its own folder, the joint runs and screens shared
    j0 = Paths(g, LEGACY_DUO, "joint")
    assert fb.tune_dir == fb.root / "reset_tune" / f"{LEGACY_DUO}__joint__beta0.25" and fb.live_dir() == j0.live_dir() and fb.shared_screen == j0.shared_screen
    assert fb.continual_dir == j0.continual_dir and fb.gate_stress_dir == j0.gate_stress_dir
    assert Paths(g, "x", "joint", "worst_shift", 0.25).study == "x__joint__beta0.25__worst_shift"
    try:
        Paths(g, "x", fixed_beta=0.25)
        raise AssertionError("an independent study with a fixed beta must be refused")
    except ValueError:
        pass
    e, ej = Paths(g, LEGACY_DUO, tta="eta_native"), Paths(g, LEGACY_DUO, "joint", tta="eta_native", fixed_beta=0.25)   # another TTA method: its own everything
    assert e.tune_dir == e.root / "reset_tune" / f"{LEGACY_DUO}__eta_native" and e.live_dir() == e.root / "reset_live" / f"{LEGACY_DUO}__eta_native_bn"
    assert ej.study == f"{LEGACY_DUO}__eta_native__joint__beta0.25" and ej.live_dir().name == f"{LEGACY_DUO}__eta_native_bn_joint" and e.shared_screen != a.shared_screen
    assert a.tag == f"{LEGACY_DUO}__tent_bn" and a.name == LEGACY_DUO and j0.name == f"{LEGACY_DUO}__joint", "Tent keeps its names"
    wj = Paths(g, "convnextb_deits", "joint", "worst_shift")
    assert wj.tune_dir == wj.root / "reset_tune" / "convnextb_deits__joint__worst_shift" and wj.name == "convnextb_deits__joint"
    try:
        Paths(g, "x", criterion="best")
        raise AssertionError("an unknown criterion must be refused")
    except ValueError:
        pass
    assert a.tune_dir == a.root / "reset_tune" and b.tune_dir == b.root / "reset_tune" / "convnextb_deits"
    assert a.live_dir().name == "vitb16_rn50v1__tent_bn" and b.live_dir(gates=True).parent.name == "reset_gates"
    assert b.member_screen("small").name == "member_detector__convnextb_deits__small.json" and b.shared_screen.name == "shared_setting__convnextb_deits.json"
    assert b.config_path.parent == b.tune_dir and b.jobs_dir.name == "jobs"
    j = Paths(g, LEGACY_DUO, "joint")                                   # the joint signal: its own folders and file names, even for the main duo
    assert j.tag == "vitb16_rn50v1__tent_bn_joint" and j.live_dir().name == j.tag and j.tune_dir == j.root / "reset_tune" / "vitb16_rn50v1__joint"
    assert j.member_screen("large").name == "member_detector__vitb16_rn50v1__joint__large.json" and j.shared_screen.name == "shared_setting__vitb16_rn50v1__joint.json"
    assert j.live_dir() != a.live_dir() and j.continual_dir != a.continual_dir and j.gate_stress_dir != a.gate_stress_dir
    try:
        Paths(g, "x", "both")
        raise AssertionError("an unknown signal must be refused")
    except ValueError:
        pass
    # the file names of the stored runs of the main duo
    assert a.continual_run(0, (1.75e-05, 3.1e-05)).name == "seed0__L1.75e-05_S3.1e-05.npz" and a.continual_run(0, None).name == "seed0__frozen.npz"
    assert a.calibration_run("S", lrs=(1.4e-4, 5e-4), seed=0, rotation=2, beta=2.0).name == "exp-S__heldout__rot2__L0.00014_S0.0005__nnn_bt256_beta2__seed0.npz"
    assert a.calibration_run("H", lrs=(1.26e-5, 2.2e-5), seed=0, frozen=True).name == "exp-H__heldout__L1.26e-05_S2.2e-05__seed0__frozen.npz"
    assert a.live_run("own_fast_vs_slow@0.7", stream="heldout", rot=2, seed=0, lrs=(5.6e-4, 8.84e-5)).name == \
        "live-own_fast_vs_slow@0.7__heldout__rot2__L0.00056_S8.84e-05__nnn_bt256_beta0.5__seed0.npz"
    print("layout self-test passed")
