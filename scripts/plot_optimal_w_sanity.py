"""
plot_optimal_w_sanity.py
========================
Compares one or more w_l-CHOOSING STRATEGIES against the TRUE per-batch
NLL-optimal w_l (src.calibrators.joint_optimal_w_oracle.optimal_w_nll), on
the same batch, so you can see exactly how much NLL each strategy leaves on
the table relative to the ceiling -- not just whether the solver itself is
correct (that check is still here too, see below).

A "strategy" is any run_cfg dict from compare_calibrators.py's
--configs_file JSON format -- calibration_mode="proxy_weighted" with any
proxy_kind/calib_map/beta/filter_kind, calibration_mode="optimal_w_oracle",
or calibration_mode="fixed_ts" (a constant w_l=0.5 reference; see
JointFixedTS.last_w_l). Built via compare_calibrators._build_calibrator, so
ANY run_cfg that works there works here, with whatever cost it naturally
implies (e.g. proxy_kind in {atc, prototype, cot} need a source-data fit
pass; fit_beta=true needs a CALIBRATOR-corruptions dev pass) -- this script
doesn't route around that, it just reuses the same builder faithfully.
calibration_mode="coca"/"oracle_ts" are NOT w_l-based strategies (COCA fits
a temperature, not a mixing weight; oracle_ts re-tunes T_l/T_s, not w_l) and
can't be compared here.

Every strategy's chosen w_l is read back via a uniform `last_w_l` property
(JointProxyWeighted, JointOptimalWOracle, JointFixedTS all expose it) --
whatever the calibrator internally decided when combine()-ing THIS exact
batch. For calibrators with set_corruption (JointProxyWeighted), this script
forces total_samples=<this batch's size> so the gate is freshly computed on
this exact batch regardless of the run_cfg's configured proxy_batch_size,
rather than possibly reusing a stale/never-refreshed prior (see
JointProxyWeighted._maybe_update_gate's stream_exhausted flush).

Correctness prerequisite: the "ceiling" every strategy is measured against
is only trustworthy if optimal_w_nll actually finds the true minimum, so
this still asserts that NLL(w_l) is CONVEX (a bowl, not concave/wiggly) on
[0, 1] and that the solver's (w_l*, nll*) matches a brute-force grid search
and beats both single-model endpoints -- see module docstring in
src/calibrators/joint_optimal_w_oracle.py for why convexity is guaranteed
(z_duo affine in w_l, cross-entropy convex in its logit argument).

Loading real logits: two formats are supported.
  --logits-l / --logits-s : the format src.utils.logits.get_model_logits
      ACTUALLY produces -- one file per model, {"logits": ..., "labels": ...}
      (see e.g. cache/logits/<model_name>/<corruption>_<severity>.pt). This
      is what you have on disk from any oracle_ts/get_model_logits run.
      Pairing two single-model caches is only safe (order-aligned labels)
      when NEITHER used tent_mode's seeded/shuffled ImageNet-C loader, or
      both used the exact same num_samples/seed -- see get_model_logits'
      own docstring. Non-tent_mode caches (plain ImageFolder, shuffle=False)
      are always safe to pair for the same corruption/severity.
  --logits : a single combined {"logits_l", "logits_s", "labels"} dict, if
      you have (or build) one -- no code in this repo currently produces
      this shape, so prefer --logits-l/--logits-s unless you made one
      yourself.
Without either, uses a synthetic random batch.

--batch-size / --offset slice a realistic ADAPTATION-batch-sized chunk out
of a big cached eval stream (e.g. the full 50000-sample ImageNet-C split) --
JointProxyWeighted/JointOptimalWOracle only ever see one adaptation batch
(config['BS'], e.g. 128) at a time, not the whole stream, so that's the
representative size to compare strategies at.

--fixed_ts supplies the canonical (T_l, T_s) both the sweep curve/true
optimum AND, by convention, every strategy's own fixed_ts_config are
expected to share (they usually do in practice) -- so the comparison is on
the same combine() scale throughout, not conflating a w_l difference with a
temperature difference.

Usage:
    python scripts/plot_optimal_w_sanity.py                      # synthetic

    # compare every strategy in a compare_runs file against the true
    # optimum, on a real cached batch:
    python scripts/plot_optimal_w_sanity.py \
        --config cfgs/dynamic_duo_config.yaml \
        --configs_file cfgs/compare_runs/calibrated.json \
        --logits-l cache/logits/resnet50/brightness_5.pt \
        --logits-s cache/logits/vit_b_16/brightness_5.pt \
        --batch-size 128

    # only a subset of that file's strategies:
    python scripts/plot_optimal_w_sanity.py \
        --config cfgs/dynamic_duo_config.yaml \
        --configs_file cfgs/compare_runs/calibrated.json \
        --strategies optimal_w_oracle proxy_weighted_oracle_acc \
        --logits-l cache/logits/resnet50/brightness_5.pt \
        --logits-s cache/logits/vit_b_16/brightness_5.pt \
        --batch-size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.calibrators.joint_fixed_TS import JointFixedTS
from src.calibrators.joint_optimal_w_oracle import combine, optimal_w_nll
from src.utils.data import load_config
from src.utils.model import get_model
from scripts.compare_calibrators import _build_calibrator, _load_run_configs

_NOT_W_L_MODES = {"coca", "oracle_ts"}


def sweep_nll(
    z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor,
    T_l: float, T_s: float, n: int = 401,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NLL at n evenly spaced w_l in [0, 1]."""
    ws = torch.linspace(0.0, 1.0, n)
    with torch.no_grad():
        nlls = torch.tensor(
            [float(F.cross_entropy(combine(z_l, z_s, float(w), T_l, T_s), labels)) for w in ws]
        )
    return ws, nlls


def min_second_difference(nlls: torch.Tensor) -> float:
    """Discrete curvature floor. Convex => every second difference >= 0
    (up to float noise)."""
    return float((nlls[:-2] - 2 * nlls[1:-1] + nlls[2:]).min())


def _run_strategy(
    run_cfg: dict, config: dict,
    large_model, large_preprocess, small_model, small_preprocess,
    device: torch.device, z_l: torch.Tensor, z_s: torch.Tensor, labels: torch.Tensor,
) -> tuple[float | None, float]:
    """Build run_cfg's calibrator (exactly as compare_calibrators.py would)
    and run it on THIS batch. Returns (w_l, nll) -- w_l is None if the
    calibrator doesn't expose last_w_l (not a w_l-based strategy)."""
    calibrator = _build_calibrator(
        run_cfg, config, large_model, large_preprocess, small_model, small_preprocess,
        device, num_samples=None, seed=None, csv_path=None,
    )
    calibrator.to(device)
    if hasattr(calibrator, "set_corruption"):
        # Force a fresh gate computation on THIS exact batch regardless of
        # the run_cfg's configured proxy_batch_size, instead of possibly
        # leaving last_w_l at an unrefreshed prior (see
        # JointProxyWeighted._maybe_update_gate's stream_exhausted flush).
        calibrator.set_corruption("plot_optimal_w_sanity", total_samples=z_l.shape[0])
    if hasattr(calibrator, "set_labels"):
        calibrator.set_labels(labels)
    with torch.no_grad():
        z_duo = calibrator.calibrate(z_l.to(device), z_s.to(device))
    nll = float(F.cross_entropy(z_duo, labels.to(z_duo.device), reduction="mean"))
    w_l = getattr(calibrator, "last_w_l", None)
    return w_l, nll


def plot(ws, nlls, w_star, nll_star, strategies: list[dict], T_l, T_s, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ws, nlls, lw=2, color="tab:blue", label="NLL$(w_l)$", zorder=2)

    ax.scatter([w_star], [nll_star], s=90, color="tab:red", zorder=5, marker="*",
               label=rf"true optimum: $w_l^*$={w_star:.4f}, NLL={nll_star:.4f}")
    ax.axvline(w_star, color="tab:red", ls="--", lw=1, alpha=0.5, zorder=1)

    # single-model endpoints, for reference
    ax.scatter([0.0, 1.0], [float(nlls[0]), float(nlls[-1])], s=40,
               facecolors="none", edgecolors="0.35", zorder=3,
               label="endpoints (small-only / large-only)")

    colors = plt.get_cmap("tab10").colors
    for i, s in enumerate(strategies):
        if s["w_l"] is None:
            continue
        color = colors[i % len(colors)]
        ax.scatter([s["w_l"]], [s["nll"]], s=70, color=color, zorder=4,
                   label=f"{s['name']}: $w_l$={s['w_l']:.4f}, regret={s['regret']:+.4f}")

    ax.set_xlabel(r"$w_l$")
    ax.set_ylabel("NLL")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(rf"NLL vs. gate weight  ($T_l$={T_l:g}, $T_s$={T_s:g})", fontsize=11)

    sec = ax.secondary_xaxis("top", functions=(lambda x: 1 - x, lambda x: 1 - x))
    sec.set_xlabel(r"$w_s = 1 - w_l$")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def _load_paired(logits_l: Path, logits_s: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load two single-model get_model_logits caches ({"logits", "labels"})
    and pair them up. Asserts the labels line up sample-for-sample -- the
    two files must be the SAME corruption/severity/split and (for
    tent_mode caches) the same num_samples/seed; see module docstring."""
    dl = torch.load(logits_l, map_location="cpu", weights_only=True)
    ds = torch.load(logits_s, map_location="cpu", weights_only=True)
    for path, d in ((logits_l, dl), (logits_s, ds)):
        assert "logits" in d and "labels" in d, (
            f"{path} doesn't look like a get_model_logits cache "
            f"(expected keys 'logits'/'labels', got {list(d.keys())})"
        )
    assert dl["labels"].shape == ds["labels"].shape and torch.equal(dl["labels"], ds["labels"]), (
        f"{logits_l} and {logits_s} have mismatched labels -- they must be the "
        f"SAME corruption/severity/split, and (for tent_mode caches) the same "
        f"num_samples/seed, or the two models' samples aren't order-aligned."
    )
    return dl["logits"].float(), ds["logits"].float(), dl["labels"]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default="cfgs/dynamic_duo_config.yaml",
                   help="duo config (LARGE/SMALL model names) -- needed to build any "
                        "strategy's calibrator, even if this script only ever combines "
                        "cached logits, never runs the models forward itself.")
    p.add_argument("--configs_file", type=Path, default=None,
                   help="compare_calibrators.py-format JSON (see cfgs/compare_runs/) "
                        "listing the strategies to compare against the optimum.")
    p.add_argument("--strategies", type=str, nargs="+", default=None,
                   help="restrict to these run_cfg names from --configs_file (default: "
                        "all of them, minus coca/oracle_ts -- not w_l-based).")
    p.add_argument("--logits", type=Path, default=None,
                   help="optional .pt with combined logits_l / logits_s / labels")
    p.add_argument("--logits-l", type=Path, default=None,
                   help="large model's get_model_logits cache ({'logits','labels'})")
    p.add_argument("--logits-s", type=Path, default=None,
                   help="small model's get_model_logits cache ({'logits','labels'})")
    p.add_argument("--batch-size", type=int, default=None,
                   help="slice this many samples (starting at --offset) out of the "
                        "loaded stream -- the representative size is the adaptation "
                        "batch size (config['BS']), not the whole cached stream.")
    p.add_argument("--offset", type=int, default=0,
                   help="start index for --batch-size slicing.")
    p.add_argument("--fixed_ts", type=str, default="checkpoints/fixed_ts/default",
                   help="JointFixedTS checkpoint supplying the canonical (T_l, T_s) for "
                        "the sweep curve/true optimum -- strategies are expected to use "
                        "the same one via their own fixed_ts_config.")
    p.add_argument("--n", type=int, default=401, help="grid points")
    p.add_argument("--out", type=Path, default=Path("optimal_w_sanity.png"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        fixed_ts = JointFixedTS.load(args.fixed_ts)
        T_l, T_s = float(fixed_ts.Tl.item()), float(fixed_ts.Ts.item())
    except FileNotFoundError:
        print(f"WARNING: --fixed_ts={args.fixed_ts!r} not found; using T_l=T_s=1.0.")
        T_l = T_s = 1.0

    if args.logits is not None:
        d = torch.load(args.logits, map_location="cpu")
        z_l, z_s, labels = d["logits_l"].float(), d["logits_s"].float(), d["labels"]
    elif args.logits_l is not None and args.logits_s is not None:
        z_l, z_s, labels = _load_paired(args.logits_l, args.logits_s)
    elif args.logits_l is not None or args.logits_s is not None:
        p.error("--logits-l and --logits-s must be given together.")
    else:
        torch.manual_seed(args.seed)
        K, B = 10, 256
        z_l = torch.randn(B, K) * 3
        z_s = torch.randn(B, K) * 3
        labels = torch.randint(0, K, (B,))

    if args.batch_size is not None:
        sl = slice(args.offset, args.offset + args.batch_size)
        z_l, z_s, labels = z_l[sl], z_s[sl], labels[sl]
    print(f"batch: {z_l.shape[0]} samples")

    ws, nlls = sweep_nll(z_l, z_s, labels, T_l, T_s, args.n)
    w_star, nll_star = optimal_w_nll(z_l, z_s, labels, T_l, T_s)

    curv = min_second_difference(nlls)
    grid_min = float(nlls.min())
    grid_argmin = float(ws[int(nlls.argmin())])
    spacing = 1.0 / (args.n - 1)

    print(f"w_l* = {w_star:.6f}  nll* = {nll_star:.6f}")
    print(f"grid argmin = {grid_argmin:.6f}  grid min = {grid_min:.6f}")
    print(f"endpoints: NLL(w_l=0) = {float(nlls[0]):.6f}  NLL(w_l=1) = {float(nlls[-1]):.6f}")
    print(f"min second difference = {curv:.3e}  (>= 0 means convex)")

    # Correctness prerequisite for trusting the ceiling every strategy below
    # is measured against -- see module docstring.
    assert curv >= -1e-6, f"NLL(w_l) is not convex on this batch: min d2 = {curv:.3e}"
    assert nll_star <= grid_min + 1e-5, (nll_star, grid_min)
    assert abs(w_star - grid_argmin) <= 2 * spacing or abs(nll_star - grid_min) < 1e-5
    assert nll_star <= float(nlls[0]) + 1e-6
    assert nll_star <= float(nlls[-1]) + 1e-6
    print("optimum ceiling check passed\n")

    strategies: list[dict] = []
    if args.configs_file is not None:
        config = load_config(args.config)
        large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
        small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
        large_model, small_model = large_model.to(device), small_model.to(device)

        run_configs = _load_run_configs(str(args.configs_file))
        if args.strategies is not None:
            wanted = set(args.strategies)
            run_configs = [c for c in run_configs if c["name"] in wanted]
            missing = wanted - {c["name"] for c in run_configs}
            if missing:
                p.error(f"Unknown strategy name(s) in --strategies: {sorted(missing)}")
        skipped = [c["name"] for c in run_configs if c["calibration_mode"] in _NOT_W_L_MODES]
        run_configs = [c for c in run_configs if c["calibration_mode"] not in _NOT_W_L_MODES]
        if skipped:
            print(f"Skipping non-w_l-based strategies: {skipped} "
                  f"(calibration_mode in {sorted(_NOT_W_L_MODES)})")

        for run_cfg in run_configs:
            print(f"Running strategy '{run_cfg['name']}' ({run_cfg['calibration_mode']})...")
            w_l, nll = _run_strategy(
                run_cfg, config, large_model, large_preprocess, small_model, small_preprocess,
                device, z_l, z_s, labels,
            )
            strategies.append({
                "name": run_cfg["name"], "w_l": w_l, "nll": nll,
                "regret": nll - nll_star,
                "w_l_gap": abs(w_l - w_star) if w_l is not None else float("nan"),
            })

    if strategies:
        print(f"\n{'strategy':<30} {'w_l':>8} {'nll':>10} {'regret':>10} {'|w_l-w_l*|':>11}")
        print("-" * 72)
        for s in strategies:
            w_l_str = f"{s['w_l']:.4f}" if s["w_l"] is not None else "n/a"
            print(f"{s['name']:<30} {w_l_str:>8} {s['nll']:>10.4f} {s['regret']:>+10.4f} "
                  f"{s['w_l_gap']:>11.4f}")

    plot(ws, nlls, w_star, nll_star, strategies, T_l, T_s, args.out)
    print("\noptimal_w sanity check passed")


if __name__ == "__main__":
    main()
