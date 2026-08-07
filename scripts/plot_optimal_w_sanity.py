"""
plot_optimal_w_sanity.py
========================
Sanity check for JointOptimalWOracle: sweep w_l across [0, 1] (equivalently
w_s = 1 - w_l across [1, 0]), plot NLL(w_l), and overlay the w_l* that
`optimal_w_nll` returns.

Expected shape is CONVEX (a bowl), not concave: z_duo is affine in w_l and
cross-entropy is convex in its logit argument, so NLL(w_l) is convex on
[0, 1]. A concave/wiggly curve would mean the premise in the module
docstring is broken and the bounded Brent solve is unjustified -- so the
script asserts convexity numerically (min second difference >= -tol)
rather than leaving it to eyeball.

Also asserts the solver's (w_l*, nll*) matches the brute-force grid
minimum, and that the minimum beats both single-model endpoints.

Note: an interior minimum is not guaranteed -- if one model dominates on
this batch the optimum sits at an endpoint. That's still convex and still
correct; only the curvature check is a real failure.

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
JointOptimalWOracle only ever sees one adaptation batch (config['BS'], e.g.
128) at a time, not the whole stream, so that's the representative size to
sanity-check. Convexity itself holds at any N (a sum of convex per-sample
terms is convex), but the NLL(w_l) curve's actual shape/sharpness is only
meaningful at real batch scale.

Usage:
    python scripts/plot_optimal_w_sanity.py                      # synthetic

    # real batch, from single-model logit caches (see --logits-l/-s above):
    python scripts/plot_optimal_w_sanity.py \
        --logits-l cache/logits/resnet50/brightness_5.pt \
        --logits-s cache/logits/vit_b_16/brightness_5.pt \
        --batch-size 128

Temperatures default to 1.0; pass --T-l/--T-s to match a frozen base_ts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.calibrators.joint_optimal_w_oracle import combine, optimal_w_nll


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


def plot(ws, nlls, w_star, nll_star, T_l, T_s, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ws, nlls, lw=2, color="tab:blue", label="NLL$(w_l)$", zorder=2)

    ax.scatter([w_star], [nll_star], s=70, color="tab:red", zorder=4,
               label=rf"optimizer: $w_l^*$={w_star:.4f}, NLL={nll_star:.4f}")
    ax.axvline(w_star, color="tab:red", ls="--", lw=1, alpha=0.6, zorder=1)

    # single-model endpoints, for reference
    ax.scatter([0.0, 1.0], [float(nlls[0]), float(nlls[-1])], s=40,
               facecolors="none", edgecolors="0.35", zorder=3,
               label="endpoints (small-only / large-only)")

    ax.set_xlabel(r"$w_l$")
    ax.set_ylabel("NLL")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(rf"NLL vs. gate weight  ($T_l$={T_l:g}, $T_s$={T_s:g})", fontsize=11)

    # mirror axis showing w_s = 1 - w_l
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
    p.add_argument("--T-l", type=float, default=1.0)
    p.add_argument("--T-s", type=float, default=1.0)
    p.add_argument("--n", type=int, default=401, help="grid points")
    p.add_argument("--out", type=Path, default=Path("optimal_w_sanity.png"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

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

    ws, nlls = sweep_nll(z_l, z_s, labels, args.T_l, args.T_s, args.n)
    w_star, nll_star = optimal_w_nll(z_l, z_s, labels, args.T_l, args.T_s)

    curv = min_second_difference(nlls)
    grid_min = float(nlls.min())
    grid_argmin = float(ws[int(nlls.argmin())])
    spacing = 1.0 / (args.n - 1)

    print(f"w_l* = {w_star:.6f}  nll* = {nll_star:.6f}")
    print(f"grid argmin = {grid_argmin:.6f}  grid min = {grid_min:.6f}")
    print(f"endpoints: NLL(w_l=0) = {float(nlls[0]):.6f}  NLL(w_l=1) = {float(nlls[-1]):.6f}")
    print(f"min second difference = {curv:.3e}  (>= 0 means convex)")

    # convexity is the real check -- everything else follows from it
    assert curv >= -1e-6, f"NLL(w_l) is not convex on this batch: min d2 = {curv:.3e}"
    # solver found the global min, not a plausible-looking nearby point
    assert nll_star <= grid_min + 1e-5, (nll_star, grid_min)
    assert abs(w_star - grid_argmin) <= 2 * spacing or abs(nll_star - grid_min) < 1e-5
    # optimum is a superset of "pick one model"
    assert nll_star <= float(nlls[0]) + 1e-6
    assert nll_star <= float(nlls[-1]) + 1e-6

    plot(ws, nlls, w_star, nll_star, args.T_l, args.T_s, args.out)
    print("optimal_w sanity check passed")


if __name__ == "__main__":
    main()
