#!/usr/bin/env python3
"""
scripts/run_tent.py
====================
Single-model TENT TTA run on ImageNet-C, producing the SAME diagnostics
plots as scripts/plot_run_diagnostics.py (batch_diagnostics.png,
proxy_diagnostics.png, per_corruption/corruption_<name>.png) but for one
model instead of a large/small duo -- no joint calibrator, no gate, no
combination. A reliability proxy (paper Sections 2-4: raw score ->
Section-3 calibration -> Section-4 temporal filter) is still tracked and
plotted against this model's own accuracy over time, since that's the
question a single-model run can still answer: does the proxy track THIS
model's own collapse/recovery under adaptation.

Unlike plot_run_diagnostics.py, the TENT hyperparameters that matter most
for a single-model sweep (--lr, --bs) are plain CLI flags rather than living
in the duo config's LARGE/SMALL.OPTIM block -- --config is only read for
TEST_DIR/VAL_DIR/WORKERS/CALIBRATOR.CORRUPTIONS (the held-out set excluded
from --corruptions' default), not for any model or optimizer settings.

--calib_config reuses the exact same JSON shape as cfgs/calib_configs/*.json
(one run_cfg dict: proxy_kind, calib_method, calib_map, filter_kind,
filter_kwargs, proxy_batch_size, proto_metric) -- an existing duo
calib_config works here unmodified. Duo-only fields in that same file
(calibration_mode, beta, pool, fixed_ts_config, prior_s, fit_beta, ...) are
simply ignored: there is no gate to combine two models with, and this
script's "prior" (both temporal filters' reset point) reads "prior" if
present, else falls back to "prior_l".

Reuse note: proxy-stats source-fitting and calibration-map fitting
(src.reliability.setup.build_proxy_weighted_calibrator) only exist in a
two-model shape (see CLAUDE.md's Reliability pipeline section). Rather than
duplicating that fitting logic for one model, this script calls it with the
SAME model/preprocess passed as both "large" and "small" and keeps only the
resulting cfg_l (a ProxyStats with the calibration map attached) --
this doubles the one-time fit-stage forward passes (a bounded cost paid
once per distinct calib_map name, then cached to disk) but changes zero
lines of the reliability pipeline itself. Everything downstream (the
per-proxy-batch score -> calibrate -> filter loop) is this script's own
_SingleModelProxyTracker, the single-model analogue of
JointProxyWeighted._flush_bucket with no gate/combine step.

Usage
-----
python scripts/run_tent.py --config cfgs/dynamic_duo_config.yaml \
    --model resnet50 --lr 0.00025 --bs 64 \
    --calib_config cfgs/calib_configs/nuclear_norm_identity_pbs128.json \
    --num_samples 2500 --severity 5 --corruptions brightness fog --norm BN
    
    # a handful of corruptions only, no wandb
    python scripts/run_tent.py --config cfgs/dynamic_duo_config.yaml \\
        --model vit_b_16 --lr 0.005 --bs 128 --norm LN \\
        --calib_config cfgs/calib_configs/nuclear_norm_identity_pbs128.json \\
        --corruptions fog snow --severity 5 --num_samples 2000 --no_wandb
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from src.utils.model import get_model, _preprocess_batch
from src.utils.data import load_config, load_imagenetC
from src.utils.metrics import get_metrics_dict
from src.tta.tent import setup_tent, softmax_entropy
from src.reliability.setup import build_proxy_weighted_calibrator
from src.reliability.proxies.stats import PROXY_KINDS, FeatureExtractor
from src.reliability.calibration.logit import to_logit
# Private helpers reused directly from the duo calibrator -- the same
# "reach into the internals of the thing that already builds this" pattern
# plot_run_diagnostics.py uses for JointProxyWeighted's _cached_* gate
# internals: no public single-model API exists for these, and duplicating
# either function would just be a second copy to keep in sync.
from src.calibrators.joint_proxy_weighted import _build_filter, _corr_stats
from src.utils.diagnostics_plots import (
    plot_batch_diagnostics, plot_single_model_proxy_diagnostics,
    plot_per_corruption_proxy_vs_accuracy, DEFAULT_EMA_WINDOW, C_LARGE,
)
from scripts._cli import (
    add_duo_config_arg, add_num_samples_arg, add_seed_arg,
    add_wandb_project_group_args, add_out_dir_run_name_args,
)

_CALIB_METHODS = {"identity", "linear", "platt", "beta", "isotonic"}
_FILTER_KINDS = {"none", "running_mean", "ema", "kalman"}

# The full 19-corruption ImageNet-C set (15 "standard" + 4 "extra" -- the
# latter are what cfgs/dynamic_duo_config*.yaml's CALIBRATOR.CORRUPTIONS
# conventionally holds out for calibration-map/gate fitting). --corruptions
# defaults to this list minus --config's CALIBRATOR.CORRUPTIONS, so a
# single-model run is screened against the same held-out-aware default as
# the rest of this codebase (see scripts/screen_duo_candidates.py's
# _CORRUPTIONS for the 15-item half of this list).
_ALL_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "shot_noise", "snow", "zoom_blur",
    "gaussian_blur", "saturate", "spatter", "speckle_noise",
]

def _load_calib_config(path: str) -> dict:
    """Load a run_cfg dict from --calib_config -- same JSON shape as one
    cfgs/compare_runs/*.json entry / cfgs/calib_configs/*.json file. Only
    proxy_kind/calib_method/calib_map/filter_kind/filter_kwargs/
    proxy_batch_size/proto_metric/prior(_l) are read; every duo-only field
    (calibration_mode, beta, pool, fixed_ts_config, prior_s, fit_beta, ...)
    is ignored if present."""
    with open(path) as f:
        run_cfg = json.load(f)
    if not isinstance(run_cfg, dict):
        raise ValueError(f"{path} must contain a single JSON object (a run_cfg dict), "
                          f"not a {type(run_cfg).__name__}.")
    if run_cfg.get("proxy_kind") not in PROXY_KINDS:
        raise ValueError(f"{path}: 'proxy_kind' must be one of {sorted(PROXY_KINDS)}, "
                          f"got {run_cfg.get('proxy_kind')!r}")
    calib_method = run_cfg.get("calib_method", "identity")
    if calib_method not in _CALIB_METHODS:
        raise ValueError(f"{path}: 'calib_method' must be one of {sorted(_CALIB_METHODS)}, "
                          f"got {calib_method!r}")
    filter_kind = run_cfg.get("filter_kind", "none")
    if filter_kind not in _FILTER_KINDS:
        raise ValueError(f"{path}: 'filter_kind' must be one of {sorted(_FILTER_KINDS)}, "
                          f"got {filter_kind!r}")
    run_cfg.setdefault("name", Path(path).stem)
    return run_cfg


class _SingleModelProxyTracker:
    """Single-model analogue of JointProxyWeighted's per-proxy-batch pipeline
    (see src/calibrators/joint_proxy_weighted.py's module docstring, steps
    1-3): raw proxy score -> Section-3 calibration -> Section-4 temporal
    filter, buffered by proxy_batch_size and flushed (see _flush) exactly
    like JointProxyWeighted._flush_bucket. No step 4 (gate) or Section-5
    combination -- there's only one model, nothing to weigh it against.
    """

    _CSV_FIELDS = ["corruption", "n_refreshes", "r", "a", "x", "acc", "n"]

    def __init__(
        self, proxy_kind: str, cfg, filter_kind: str, filter_kwargs: dict,
        prior: float, eps: float, proxy_batch_size: int,
        csv_path: str | None, verbose: bool = True,
    ):
        self.proxy_kind = proxy_kind
        self.cfg = cfg
        self.eps = eps
        self.proxy_batch_size = proxy_batch_size
        self.prior = prior
        self.verbose = verbose
        self._filter = _build_filter(filter_kind, prior, filter_kwargs)

        if verbose:
            print(
                f"\n{'#' * 78}\n"
                f"# _SingleModelProxyTracker CONFIG\n"
                f"#   proxy_kind={proxy_kind}  PROXY_BATCH_SIZE={proxy_batch_size}\n"
                f"#   filter_kind={filter_kind}  filter_kwargs={filter_kwargs}\n"
                f"#   prior={prior}  eps={eps}\n"
                f"{'#' * 78}\n"
            )

        if csv_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = Path(csv_path)
            self.csv_path: Path | None = p.parent / f"{p.name}_{ts}.csv"
        else:
            self.csv_path = None

        self._buf_z: list[torch.Tensor] = []
        self._buf_f: list[torch.Tensor] = []
        self._buf_labels: list[torch.Tensor] = []
        self._buf_n = 0
        self._current_corruption = ""
        self.n_refreshes = 0
        self._cached_r, self._cached_a, self._cached_x = 0.0, 0.5, prior
        self._stream_total_samples: int | None = None
        self._stream_samples_seen = 0
        self._corr_r: list[float] = []
        self._corr_acc: list[float] = []

    def set_corruption(self, label: str, total_samples: int | None = None) -> None:
        self._current_corruption = label
        self._filter.reset()
        self._buf_z.clear(); self._buf_f.clear(); self._buf_labels.clear()
        self._buf_n = 0
        self._stream_total_samples = total_samples
        self._stream_samples_seen = 0
        self._cached_x = self.prior

    def observe(self, z: torch.Tensor, f: torch.Tensor | None, labels: torch.Tensor) -> None:
        """Feed one adaptation batch's logits/features/labels in, flushing
        the proxy bucket (see _flush) every time it fills -- possibly
        several times if proxy_batch_size < the adaptation batch size."""
        n_total = z.shape[0]
        start = 0
        while start < n_total:
            take = min(self.proxy_batch_size - self._buf_n, n_total - start)
            sl = slice(start, start + take)
            self._buf_z.append(z[sl].detach())
            if f is not None:
                self._buf_f.append(f[sl].detach())
            self._buf_labels.append(labels[sl].detach())
            self._buf_n += take
            self._stream_samples_seen += take

            stream_exhausted = (
                self._stream_total_samples is not None
                and self._stream_samples_seen >= self._stream_total_samples
            )
            if self._buf_n >= self.proxy_batch_size or stream_exhausted:
                self._flush()
            start += take

    def _flush(self) -> None:
        agg_z = torch.cat(self._buf_z, dim=0)
        agg_f = torch.cat(self._buf_f, dim=0) if self._buf_f else None
        agg_labels = torch.cat(self._buf_labels, dim=0)
        n = agg_z.shape[0]

        r = self.cfg.score(self.proxy_kind, agg_z, agg_f, labels=agg_labels)
        a = min(max(self.cfg.predicted_acc(self.proxy_kind, r), self.eps), 1.0 - self.eps)
        x = self._filter.update(to_logit(a, self.eps))
        self._cached_r, self._cached_a, self._cached_x = r, a, x
        self.n_refreshes += 1

        self._buf_z.clear(); self._buf_f.clear(); self._buf_labels.clear()
        self._buf_n = 0

        acc = float((agg_z.argmax(1) == agg_labels).float().mean())
        self._corr_r.append(r); self._corr_acc.append(acc)

        if self.csv_path is not None:
            need_header = not self.csv_path.exists()
            with self.csv_path.open("a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._CSV_FIELDS)
                if need_header:
                    writer.writeheader()
                writer.writerow({
                    "corruption": self._current_corruption, "n_refreshes": self.n_refreshes,
                    "r": r, "a": a, "x": x, "acc": acc, "n": n,
                })

        if self.verbose:
            print(f"[proxy {self.proxy_kind} batch #{self.n_refreshes} n={n}] "
                  f"r={r:.3f} a={a:.3f} acc={acc:.3f}")

    def report_and_reset_corruption_stats(self, label: str) -> dict:
        stats = _corr_stats(self._corr_r, self._corr_acc)
        if stats["n"] > 0:
            print(f"[proxy {self.proxy_kind} {label}] n={stats['n']} proxy batches  "
                  f"R2={stats['r2']:.3f} r={stats['pearson_r']:.3f} rho={stats['spearman_rho']:.3f}")
        self._corr_r.clear(); self._corr_acc.clear()
        return stats


def _build_proxy_tracker(run_cfg: dict, cfg: dict, model, preprocess, device, args, out_dir: Path):
    """Build the ProxyStats (with calibration map attached) via the SAME
    two-model factory the duo pipeline uses -- see module docstring's Reuse
    note -- then wrap it in _SingleModelProxyTracker."""
    proxy_kind = run_cfg["proxy_kind"]
    calib_method = run_cfg.get("calib_method", "identity")
    calib_map = run_cfg.get("calib_map")
    if calib_map is None and calib_method != "identity":
        calib_map = f"{args.model}_{proxy_kind}_{calib_method}"
        print(f"No 'calib_map' in {args.calib_config} with calib_method={calib_method!r}; "
              f"auto-naming it {calib_map!r} (fit fresh if not already cached).")

    # model/preprocess doubles as both "large" and "small" here purely to
    # reuse build_proxy_weighted_calibrator's fitting logic (see module
    # docstring) -- the gate it builds (beta, prior_s, ...) is never used;
    # only the returned calibrator's .cfg_l (a ProxyStats with .calib
    # populated) is kept.
    #
    # build_proxy_weighted_calibrator (and _build_proxy_stats underneath it)
    # was written for the duo pipeline and reaches directly into
    # config["LARGE"]["NAME"]/config["SMALL"]["NAME"]/["BS"]/["WORKERS"] --
    # NOT from the large_model/small_model objects passed in -- so --config
    # here genuinely doesn't need a LARGE/SMALL/BS/WORKERS block (only
    # TEST_DIR/VAL_DIR/CALIBRATOR.*, per the module docstring); synthesize
    # the fields that call needs from what this script already has instead
    # of asking the user's config to carry duo-shaped fields it has no use
    # for otherwise.
    proxy_fit_cfg = {
        **cfg,
        "LARGE": {"NAME": args.model}, "SMALL": {"NAME": args.model},
        "BS": cfg.get("BS", args.bs), "WORKERS": cfg.get("WORKERS", 4),
    }
    # build_proxy_weighted_calibrator prints its own "JointProxyWeighted
    # CONFIG" banner on construction -- proxy_kind/filter_kwargs/prior there
    # ARE this run's real values, but PROXY_BATCH_SIZE/beta/filter_kind are
    # this throwaway object's placeholders (1 / 1.0 / "none"), NOT the
    # actual run config below (proxy_batch_size/filter_kind/prior on
    # _SingleModelProxyTracker, which prints its own accurate banner) --
    # flag that clearly so the two banners aren't read as one run's config.
    print("[run_tent] building proxy stats / calibration map only (gate/proxy_batch_size "
          "below is a throwaway placeholder -- see the _SingleModelProxyTracker CONFIG "
          "banner further down for what this run actually uses):")
    jpw_for_fitting = build_proxy_weighted_calibrator(
        proxy_kind=proxy_kind,
        proxy_cache=run_cfg.get("proxy_cache"),
        calib_map=calib_map,
        calib_method=calib_method,
        filter_kind="none",
        filter_kwargs=None,
        beta=1.0,
        prior_l=run_cfg.get("prior", run_cfg.get("prior_l", 0.5)),
        prior_s=run_cfg.get("prior", run_cfg.get("prior_l", 0.5)),
        base_ts=None,
        csv_path=None,
        config=proxy_fit_cfg,
        large_model=model, large_preprocess=preprocess,
        small_model=model, small_preprocess=preprocess,
        device=device,
        num_samples=args.num_samples, seed=args.seed,
        proto_metric=run_cfg.get("proto_metric", "cosine"),
        proxy_batch_size=1,
    )
    proxy_stats = jpw_for_fitting.cfg_l

    prior = to_logit(run_cfg.get("prior", run_cfg.get("prior_l", 0.5)))
    tracker = _SingleModelProxyTracker(
        proxy_kind=proxy_kind,
        cfg=proxy_stats,
        filter_kind=run_cfg.get("filter_kind", "none"),
        filter_kwargs=run_cfg.get("filter_kwargs") or {},
        prior=prior,
        eps=1e-3,
        proxy_batch_size=run_cfg.get("proxy_batch_size", 128),
        csv_path=str(out_dir / "proxy_log"),
        verbose=True,
    )
    return tracker


def _write_plots(
    batch_records: list[dict], corruption_boundaries: list[dict], tracker: _SingleModelProxyTracker,
    out_dir: Path, model_series: list[tuple[str, str, str]], ema_window: int,
) -> tuple[bool, bool, list[dict]]:
    """Write every diagnostics artifact from whatever has been recorded SO
    FAR, overwriting what's already on disk -- called after every corruption
    (see main()) rather than once at the very end, so a long multi-corruption
    run's plots/CSVs are visible while it's still going instead of only once
    it finishes. Idempotent for corruptions already written (per_corruption's
    own per-corruption files re-render identically); the cost is a handful
    of cheap matplotlib redraws per corruption, not extra model computation.
    """
    if batch_records:
        with (out_dir / "batch_diagnostics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(batch_records[0].keys()))
            writer.writeheader()
            writer.writerows(batch_records)

    proxy_rows: list[dict] = []
    if tracker.csv_path is not None and tracker.csv_path.exists():
        with tracker.csv_path.open() as f:
            proxy_rows = list(csv.DictReader(f))

    has_batch_plot = False
    if batch_records:
        plot_batch_diagnostics(batch_records, corruption_boundaries, out_dir / "batch_diagnostics.png",
                                ema_window=ema_window, series=model_series)
        has_batch_plot = True

    has_proxy_plot = plot_single_model_proxy_diagnostics(
        proxy_rows, out_dir / "proxy_diagnostics.png", ema_window=ema_window,
    )

    per_corruption_dir = out_dir / "per_corruption"
    per_corruption_dir.mkdir(parents=True, exist_ok=True)
    plot_per_corruption_proxy_vs_accuracy(
        batch_records, proxy_rows, per_corruption_dir, ema_window=ema_window,
        series=model_series, proxy_series=[("r", C_LARGE, "r")],
    )

    return has_batch_plot, has_proxy_plot, proxy_rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_duo_config_arg(
        p, help="Only TEST_DIR/VAL_DIR/WORKERS/CALIBRATOR.CORRUPTIONS are read from this -- "
                "not any model or optimizer settings (those come from --model/--lr/--bs below)."
    )
    p.add_argument("--model", type=str, required=True, help="Model name passed to get_model().")
    p.add_argument("--norm", type=str, required=True, default=None, choices=["BN", "LN", "GN"],
                    help="TENT norm type.")
    p.add_argument("--lr", type=float, required=True, help="TENT adaptation learning rate.")
    p.add_argument("--bs", type=int, required=True, help="Adaptation batch size.")
    p.add_argument("--steps", type=int, default=1, help="Adaptation steps per batch.")
    p.add_argument("--calib_config", type=str, required=True,
                    help="Path to a JSON run_cfg dict -- see module docstring. Same shape as "
                         "cfgs/calib_configs/*.json.")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--corruptions", type=str, nargs="+", default=None,
                    help="Defaults to every ImageNet-C corruption except --config's "
                         "CALIBRATOR.CORRUPTIONS (the usual held-out set).")
    add_num_samples_arg(p, default=None)
    add_seed_arg(p)

    add_out_dir_run_name_args(
        p, out_dir_default="out/run_diagnostics",
        run_name_help="Subdirectory name under --out_dir, and the wandb run name. Default: "
                       "auto-generated from --model/--calib_config's name/lr/bs/timestamp.",
    )
    p.add_argument("--ema_window", type=int, default=DEFAULT_EMA_WINDOW,
                    help="EMA smoothing window (in points) for every plotted line.")

    wandb_args = p.add_argument_group("wandb options")
    wandb_args.add_argument("--use_wandb", dest="use_wandb", action="store_true", default=True)
    wandb_args.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    add_wandb_project_group_args(
        wandb_args, default_project="dynamic-duos",
        group_help="Optional shared group tag. Always prefixed with --model.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  model: {args.model}")

    cfg = load_config(args.config)
    run_cfg = _load_calib_config(args.calib_config)

    norm_type = args.norm

    corruptions = args.corruptions or [
        c for c in _ALL_CORRUPTIONS if c not in set(cfg["CALIBRATOR"]["CORRUPTIONS"])
    ]

    run_name = args.run_name or (
        f"{args.model}__{run_cfg['name']}__lr{args.lr}_bs{args.bs}__"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")

    # Freshly-loaded, frozen model -- proxy-stats source-fitting and
    # calibration-map dev fitting (both potentially run inside
    # _build_proxy_tracker) must see the SAME plain-eval state the duo
    # pipeline's own build_proxy_weighted_calibrator call always sees (e.g.
    # plot_run_diagnostics.py's _run builds its calibrator before
    # setup_duo configures anything for TENT) -- not a model already
    # switched into TENT's forced live-batch-stats train mode.
    model, preprocess = get_model(args.model)
    model = model.to(device)

    tracker = _build_proxy_tracker(run_cfg, cfg, model, preprocess, device, args, out_dir)

    optim_cfg = {"METHOD": "Adam", "STEPS": args.steps, "LR": args.lr, "BETA": 0.9, "WD": 0.0}
    tented_model = setup_tent(model, norm_type=norm_type, cfg=optim_cfg)
    tented_model.eval()

    need_features = run_cfg["proxy_kind"] == "prototype"
    ext = FeatureExtractor(tented_model.model, args.model) if need_features else None

    wandb_run = None
    if args.use_wandb:
        group = f"{args.model}__{args.wandb_group}" if args.wandb_group else None
        wandb_run = wandb.init(
            project=args.wandb_project, name=run_name, group=group,
            tags=[args.model, run_cfg["proxy_kind"], norm_type],
            config={
                "model": args.model, "norm": norm_type, "lr": args.lr, "bs": args.bs,
                "steps": args.steps, "num_samples": args.num_samples, "seed": args.seed,
                "severity": args.severity, "corruptions": corruptions,
                **{f"calib/{k}": v for k, v in run_cfg.items() if k != "name"},
            },
        )
        print(f"wandb run: {wandb_run.url}")

    batch_records: list[dict] = []
    corruption_boundaries: list[dict] = []
    results_rows: list[dict] = []
    all_probs_overall, all_labels_overall = [], []
    model_series = [("model", C_LARGE, args.model)]
    has_batch_plot, has_proxy_plot, proxy_rows = False, False, []

    try:
        for corruption in corruptions:
            print(f"\n=== TENT ({args.model}) | {corruption} severity {args.severity} ===")
            tented_model.reset()

            loader = load_imagenetC(
                cfg["TEST_DIR"], args.severity, [corruption],
                device=device, batch_size=args.bs,
                num_workers=cfg.get("WORKERS", 4),
                num_samples=args.num_samples, seed=args.seed,
            )
            tracker.set_corruption(f"{corruption}/s{args.severity}", total_samples=len(loader.dataset))
            corruption_boundaries.append({"idx": len(batch_records), "label": f"{corruption}/s{args.severity}"})

            diag = {"n": 0, "acc_sum": 0.0, "nll_sum": 0.0, "ent_sum": 0.0}
            all_probs, all_labels = [], []

            for imgs, labels in tqdm(loader, desc=f"{corruption} s{args.severity}"):
                x = _preprocess_batch(imgs, preprocess, device)
                labels_dev = labels.to(device)

                z_out = tented_model.forward(x)

                probs = F.softmax(z_out.detach().cpu(), dim=1)
                acc = (probs.argmax(1) == labels).float().mean().item()
                nll = F.nll_loss(torch.log(probs.clamp(min=1e-8)), labels).item()
                ent = softmax_entropy(z_out.detach()).mean().item()
                diag["n"] += 1
                diag["acc_sum"] += acc
                diag["nll_sum"] += nll
                diag["ent_sum"] += ent

                batch_records.append({
                    "global_idx": len(batch_records), "corruption": f"{corruption}/s{args.severity}",
                    "n": x.shape[0],
                    "model_acc": acc, "model_nll": nll, "model_ent": ent,
                    "model_acc_run": diag["acc_sum"] / diag["n"],
                    "model_nll_run": diag["nll_sum"] / diag["n"],
                    "model_ent_run": diag["ent_sum"] / diag["n"],
                })

                f_feats = ext._feats.detach() if ext is not None else None
                tracker.observe(z_out.detach(), f_feats, labels_dev)

                if wandb_run is not None:
                    wandb_run.log({
                        f"{corruption}/s{args.severity}/batch_acc": acc,
                        f"{corruption}/s{args.severity}/avg_acc": diag["acc_sum"] / diag["n"],
                        f"{corruption}/s{args.severity}/batch_nll": nll,
                        f"{corruption}/s{args.severity}/batch_ent": ent,
                        f"{corruption}/s{args.severity}/avg_ent": diag["ent_sum"] / diag["n"],
                        "proxy/r": tracker._cached_r, "proxy/a": tracker._cached_a, "proxy/x": tracker._cached_x,
                    })

                all_probs.append(probs); all_labels.append(labels)
                all_probs_overall.append(probs); all_labels_overall.append(labels)

            tracker.report_and_reset_corruption_stats(f"{corruption}/s{args.severity}")

            metrics = get_metrics_dict(torch.cat(all_probs), torch.cat(all_labels))
            print(f"Results for {corruption} severity {args.severity}: {metrics}")
            if wandb_run is not None:
                wandb_run.log({f"{corruption}/s{args.severity}/{k}": v for k, v in metrics.items()})
            results_rows.append({"corruption": corruption, "severity": args.severity, **metrics})

            has_batch_plot, has_proxy_plot, proxy_rows = _write_plots(
                batch_records, corruption_boundaries, tracker, out_dir, model_series, args.ema_window,
            )
            print(f"Diagnostics through {corruption}/s{args.severity} written to {out_dir}")
    finally:
        if ext is not None:
            ext.remove()

    overall_metrics = get_metrics_dict(torch.cat(all_probs_overall), torch.cat(all_labels_overall))
    # severity (not "all") -- every row's "severity" column must stay the
    # same type for wandb.Table.add_data (int, not int|str), and this
    # script only ever runs one --severity across every corruption anyway
    # (unlike the duo pipeline's EVAL.SEVERITIES sweep), so the average
    # row's severity is genuinely just args.severity, not a separate value.
    results_rows.append({"corruption": "average", "severity": args.severity, **overall_metrics})
    print(f"\nFinal average: accuracy={overall_metrics['accuracy']:.4f}")

    # Plots/CSVs were already written after each corruption (see the
    # _write_plots call inside the loop above) -- batch_records/proxy_rows
    # haven't changed since the last one ran, so has_batch_plot/has_proxy_plot/
    # proxy_rows from that final call are already the complete, final state;
    # nothing left to (re)write here.
    if not batch_records:
        print("No batches were recorded -- nothing was plotted.")

    if wandb_run is not None:
        media = {}
        if has_batch_plot:
            media["plots/batch_diagnostics"] = wandb.Image(str(out_dir / "batch_diagnostics.png"))
        if has_proxy_plot:
            media["plots/proxy_diagnostics"] = wandb.Image(str(out_dir / "proxy_diagnostics.png"))
        if media:
            wandb_run.log(media)

        cols = list(results_rows[0].keys())
        table = wandb.Table(columns=cols)
        for row in results_rows:
            table.add_data(*[row[c] for c in cols])
        wandb_run.log({"summary/results": table})
        for k, v in overall_metrics.items():
            wandb_run.summary[k] = v
        wandb_run.finish()

    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
