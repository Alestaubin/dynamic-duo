from src.tta.dynamic_duo import setup_duo, evaluate_dynamic_duo, build_wandb_run
from src.utils.model import get_model
from src.utils.data import load_config
from src.calibrators.joint_fixed_TS import JointFixedTS, PreScaledCalibrator
from src.calibrators.joint_coca import JointCoca
from src.reliability.setup import build_proxy_weighted_calibrator, fit_beta
from src.utils.diagnostics_plots import (
    plot_batch_diagnostics, plot_proxy_diagnostics, plot_per_corruption_proxy_vs_accuracy,
    DEFAULT_EMA_WINDOW,
)
from scripts._cli import (
    add_duo_config_arg, add_num_samples_arg, add_seed_arg,
    add_proto_metric_arg, add_out_dir_run_name_args,
)

import argparse
import csv
from datetime import datetime
from pathlib import Path
import torch
import wandb

"""
source /scratch0/alxstaub/ddenv/bin/activate
export PYTHONPATH=$PYTHONPATH:~/dynamic-duo

CUDA_VISIBLE_DEVICES=0

# fixed temperature-scaling baseline:
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --seed 0 \
    --calibration_mode fixed_ts \
    --fixed_ts_config checkpoints/naive_ts/clean

# coca_ts baseline (self-adapting per-batch temperature scaling):
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --duo_calibration_mode coca \
    --seed 0

# filtered-proxy soft weighting (Sections 2-5):
python scripts/run_dynamic_duo.py \
    --config cfgs/dynamic_duo_config.yaml \
    --mode no_adapt \
    --calibration_mode proxy_weighted \
    --proxy_kind nuclear_norm \
    --calib_map resnet50_vitb16_dev \
    --calib_method isotonic \
    --filter kalman --kalman_q 1e-3 --kalman_r 1e-1 \
    --gate_beta 4.0 \
    --seed 0
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamic Duo TTA on ImageNet-C")
    add_duo_config_arg(parser, required=True)
    parser.add_argument("--mode", type=str, default="both_duo")
    parser.add_argument("--steps", type=int, default=1)
    add_num_samples_arg(parser, default=None)
    add_seed_arg(parser, default=None)

    parser.add_argument("--duo_calibration_mode", type=str, default="fixed_ts",
                        choices=["fixed_ts", "oracle_ts", "proxy_weighted", "coca"])

    parser.add_argument("--proxy_kind", type=str, default="prototype",
                        choices=["nuclear_norm", "atc", "prototype", "ac_mc", "cot", "oracle"],
                        help="Proxy kind for the proxy-weighted calibration. ")

    add_proto_metric_arg(parser)

    parser.add_argument("--proxy_cache", type=str, default=None,
                        help="Path to a .pt file containing cached proxy values. "
                             "If not provided, the proxy will be computed on-the-fly. ")

    parser.add_argument("--calib_map", type=str, default=None,
                        help="Name of a calibration-map file to load, or fit fresh "
                             "(on the config's CALIBRATOR corruptions) and save under "
                             "this name if it doesn't exist yet.")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to a CSV file to save the results (WHAT RESULTS?)")

    # Pre-processing options:
    parser.add_argument("--fixed_ts_config", type=str, default=None,
                        help="Path to a JointFixedTS checkpoint folder (see JointFixedTS.save/.load). "
                             "Required for --duo_calibration_mode fixed_ts; optional for "
                             "proxy_weighted (supplies the Section-5 base_ts T_l/T_s prior).")
    parser.add_argument("--prescale_path", type=str, default=None,
                        help="Path to a file containing the fixed temperature configuration."
                        "If set, to prescale the duo with a fixed temperature.")
    parser.add_argument("--norm_logits", action="store_true",
                        help="Whether to L2-normalize the logits before anything else. ")


    # --- filtered-proxy soft weighting ---

    parser.add_argument("--calib_method", type=str, default="isotonic",
                        choices=["identity", "linear", "platt", "beta", "isotonic"],
                        help="Section-3 calibration ladder: raw proxy -> predicted accuracy. "
                             "Only used when --calibration_mode proxy_weighted.")
    parser.add_argument("--filter", type=str, default="none",
                        choices=["none", "running_mean", "ema", "kalman"],
                        help="Section-4 temporal filter denoising the calibrated score "
                             "across batches. Only used when --calibration_mode proxy_weighted.")
    parser.add_argument("--ema_alpha", type=float, default=0.1,
                        help="EMA smoothing factor (--filter ema).")
    parser.add_argument("--kalman_q", type=float, default=1e-3,
                        help="Kalman process noise (--filter kalman).")
    parser.add_argument("--kalman_r", type=float, default=1e-1,
                        help="Kalman observation noise (--filter kalman).")
    parser.add_argument("--gate_beta", type=float, default=4.0,
                        help="Section-5 gate sharpness: w_l = sigmoid(beta * gap).")
    parser.add_argument("--prior_l", type=float, default=0.5,
                        help="Large model's clean-source accuracy, used as the "
                             "ema/kalman filter's reset prior. Defaults to a neutral 0.5.")
    parser.add_argument("--prior_s", type=float, default=0.5,
                        help="Small model's clean-source accuracy, used as the "
                             "ema/kalman filter's reset prior. Defaults to a neutral 0.5.")
    parser.add_argument("--fit_beta", action="store_true",
                        help="Grid-search --gate_beta against held-out dev-shift NLL "
                             "(config's CALIBRATOR corruptions/severities) before eval.")
    parser.add_argument("--proxy_batch_size", type=int, default=1,
                        help="Section-1 proxy batch size b_t: number of samples "
                             "aggregated into one proxy computation, independent of "
                             "the adaptation batch size (config's BS). 1 (default) "
                             "recomputes the gate every adaptation batch; larger values "
                             "trade a slower-reacting weight for less sampling noise.")

    # --- coca_ts baseline ---
    parser.add_argument("--coca_bs", type=int, default=None,
                        help="Batch size for COCA's per-batch temperature fit, "
                             "independent of the TENT batch size. Only used when "
                             "--duo_calibration_mode coca.")

    # --- diagnostics plots ---
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip the end-of-run diagnostics plots (per-batch "
                             "acc/nll/entropy for large/small/duo, plus -- for "
                             "--duo_calibration_mode proxy_weighted -- raw proxy "
                             "scores and gate weight vs. ground-truth accuracy, "
                             "the signal for whether a proxy tracks one model "
                             "collapsing during adaptation). On by default.")
    add_out_dir_run_name_args(
        parser, out_dir_default="out/run_diagnostics",
        out_dir_help="Directory to write plots (and, for proxy_weighted, "
                      "the underlying proxy CSV log) under a per-run subdir.",
        run_name_help="Subdirectory name under --out_dir. Default: "
                       "auto-generated from calibration_mode/mode/timestamp.",
    )
    parser.add_argument("--ema_window", type=int, default=DEFAULT_EMA_WINDOW,
                        help="Span (in points) of the EMA smoothing applied to every plotted "
                             "line (accuracy, NLL, entropy, proxy scores, gate weight) across "
                             "all three diagnostics plots -- alpha = 2/(window+1). Purely a "
                             "plotting knob; unrelated to --ema_alpha, the proxy_weighted "
                             "gate's own Section-4 temporal filter, which affects the logged "
                             "values themselves, not just how they're plotted.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    large_model, large_preprocess = get_model(config["LARGE"]["NAME"])
    small_model, small_preprocess = get_model(config["SMALL"]["NAME"])
    large_model = large_model.to(device)
    small_model = small_model.to(device)

    run_name = args.run_name or (
        f"{args.duo_calibration_mode}"
        f"{'_' + args.proxy_kind if args.duo_calibration_mode == 'proxy_weighted' else ''}"
        f"__{args.mode}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.out_dir) / run_name
    if not args.no_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        # proxy_weighted's proxy-batch CSV log (r_l/r_s/w_l/acc_l/acc_s/duo_acc)
        # is what the proxy-vs-collapse plot below is read back from -- point
        # it at the run's own out_dir unless the user already gave one.
        if args.duo_calibration_mode == "proxy_weighted" and args.csv_path is None:
            args.csv_path = str(out_dir / "proxy_log")

    if args.duo_calibration_mode == "proxy_weighted":
        try:
            base_ts = JointFixedTS.load(args.fixed_ts_config) if args.fixed_ts_config else None
            filter_kwargs = {
                "alpha": args.ema_alpha,
                "q": args.kalman_q,
                "r": args.kalman_r,
            }
            calibrator = build_proxy_weighted_calibrator(
                proxy_kind=args.proxy_kind,
                proxy_cache=args.proxy_cache,
                calib_map=args.calib_map,
                calib_method=args.calib_method,
                filter_kind=args.filter,
                filter_kwargs=filter_kwargs,
                beta=args.gate_beta,
                prior_l=args.prior_l,
                prior_s=args.prior_s,
                base_ts=base_ts,
                csv_path=args.csv_path,
                config=config,
                large_model=large_model, large_preprocess=large_preprocess,
                small_model=small_model, small_preprocess=small_preprocess,
                device=device,
                num_samples=args.num_samples,
                seed=args.seed,
                proto_metric=args.proto_metric,
                proxy_batch_size=args.proxy_batch_size,
            )
        except ValueError as e:
            parser.error(str(e))
        if args.fit_beta:
            fit_beta(
                calibrator,
                large_model, large_preprocess, small_model, small_preprocess,
                config, device, num_samples=args.num_samples, seed=args.seed,
            )
    elif args.duo_calibration_mode == "fixed_ts":
        if args.fixed_ts_config is None:
            parser.error("--fixed_ts_config is required when --calibration_mode is fixed_ts")
        calibrator = JointFixedTS.load(args.fixed_ts_config)
    elif args.duo_calibration_mode == "oracle_ts":
        calibrator = JointFixedTS()
    elif args.duo_calibration_mode == "coca":
        if args.coca_bs is None:
            parser.error("--coca_bs is required when --duo_calibration_mode is coca")
        calibrator = JointCoca(num_steps=10, lr=5e-2, chunk_size=args.coca_bs)
    else:
        raise ValueError(f"Invalid calibration mode: {args.duo_calibration_mode}")

    if args.prescale_path is not None:
        fixed_ts = JointFixedTS.load(args.prescale_path)
        fixed_ts.requires_grad_(False)
        calibrator = PreScaledCalibrator(fixed_ts, calibrator)
        print(f"WARNING: Pre-scaling with JointFixedTS: Tl={fixed_ts.Tl.item():.4f}  Ts={fixed_ts.Ts.item():.4f}")

    duo = setup_duo(
        large=large_model,
        large_preprocess=large_preprocess,
        small=small_model,
        small_preprocess=small_preprocess,
        mode=args.mode,
        joint_calibrator=calibrator,
        calibration_mode=args.duo_calibration_mode,
        cfg=config,
        steps=args.steps,
        norm_logits=args.norm_logits,
    )

    # Built here (rather than left for evaluate_dynamic_duo to create+finish
    # internally) so the diagnostics plots below -- only available after
    # evaluate_dynamic_duo returns -- can be logged into the SAME run instead
    # of a second, separate one. Passing wandb_run= makes evaluate_dynamic_duo
    # log into it without finishing it; we finish it ourselves at the end.
    wandb_run = build_wandb_run(duo, config, run_name=run_name) if args.wandb else None
    if wandb_run is not None:
        print(f"wandb run: {wandb_run.url}")

    # Per-batch diagnostics -- accumulated via evaluate_dynamic_duo's
    # on_corruption_start/on_batch hooks (see run_duo's docstring), read
    # straight out of DynamicDuo._diag rather than recomputed here. Only
    # collected when plotting is wanted, to avoid the bookkeeping cost on a
    # plain sweep/production run.
    batch_records, corruption_boundaries = [], []

    def _on_corruption_start(corruption, severity):
        corruption_boundaries.append({"idx": len(batch_records), "label": f"{corruption}/s{severity}"})

    def _on_batch(batch_idx, prefix, duo, outputs, z_large, z_small, labels):
        row = {"global_idx": len(batch_records), "corruption": prefix.rstrip("/")}
        for name in ("large", "small", "duo"):
            d = duo._diag[name]
            row[f"{name}_acc"] = d["acc_last"]
            row[f"{name}_nll"] = d["nll_last"]
            row[f"{name}_ent"] = d["ent_last"]
            row[f"{name}_acc_run"] = d["acc_sum"] / d["n"] if d["n"] > 0 else float("nan")
            row[f"{name}_nll_run"] = d["nll_sum"] / d["n"] if d["n"] > 0 else float("nan")
            row[f"{name}_ent_run"] = d["ent_sum"] / d["n"] if d["n"] > 0 else float("nan")
        batch_records.append(row)

    evaluate_dynamic_duo(
        duo, config, num_samples=args.num_samples, seed=args.seed, use_wandb=args.wandb,
        wandb_run=wandb_run,
        on_corruption_start=None if args.no_plots else _on_corruption_start,
        on_batch=None if args.no_plots else _on_batch,
    )

    has_batch_plot, has_proxy_plot = False, False
    if not args.no_plots:
        if batch_records:
            with (out_dir / "batch_diagnostics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(batch_records[0].keys()))
                writer.writeheader()
                writer.writerows(batch_records)
            plot_batch_diagnostics(batch_records, corruption_boundaries, out_dir / "batch_diagnostics.png",
                                    ema_window=args.ema_window)
            has_batch_plot = True
        else:
            print("No batches were recorded -- nothing to plot.")

        # proxy_kind's raw scores (r_l, r_s) and the gate weight w_l vs. each
        # proxy batch's ACTUAL accuracy -- the plot that answers "does the
        # proxy track one model collapsing during adaptation" (see
        # JointProxyWeighted's CSV logging, csv_path set above).
        proxy_csv_path = getattr(calibrator, "_csv_path", None)
        proxy_rows = []
        if proxy_csv_path is not None and Path(proxy_csv_path).exists():
            with Path(proxy_csv_path).open() as f:
                proxy_rows = list(csv.DictReader(f))
        has_proxy_plot = plot_proxy_diagnostics(proxy_rows, out_dir / "proxy_diagnostics.png",
                                                 ema_window=args.ema_window)

        # One figure per corruption (saved locally only, not sent to wandb):
        # EMA-smoothed accuracy for large/small/duo (bold, right axis) with
        # the raw proxy scores r_l/r_s (light, left axis) overlaid.
        per_corruption_dir = out_dir / "per_corruption"
        per_corruption_dir.mkdir(parents=True, exist_ok=True)
        plot_per_corruption_proxy_vs_accuracy(batch_records, proxy_rows, per_corruption_dir,
                                               ema_window=args.ema_window)

        print(f"\nDiagnostics written to {out_dir}")

    if wandb_run is not None:
        media = {}
        if has_batch_plot:
            media["plots/batch_diagnostics"] = wandb.Image(str(out_dir / "batch_diagnostics.png"))
        if has_proxy_plot:
            media["plots/proxy_diagnostics"] = wandb.Image(str(out_dir / "proxy_diagnostics.png"))
        if media:
            wandb_run.log(media)
        wandb_run.finish()
