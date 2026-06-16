#!/usr/bin/env python3
"""
Transform a W&B export CSV into LaTeX table rows for the ImageNet-C table.

Usage:
    python src/utils/csv_to_latex.py wandb_export.csv [--severity 5] [--metric accuracy]

Available metrics: accuracy, acc3, acc5, ece, nll, f1, brier
"""

import argparse
import csv
import sys

# Column order in the LaTeX table, grouped by family.
# Each entry: (CSV corruption name).
FAMILIES = {
    "Noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog", "brightness"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

# Ordered list of all corruptions following the table's column layout.
CORRUPTION_ORDER = [c for fam in FAMILIES.values() for c in fam]

# Map a model label (table) -> CSV column prefix.
MODELS = [
    ("ResNet-50", "small"),
    ("Vit-B 16",  "large"),
    ("Duo",       "duo"),
]


# Metrics that are in [0, 1] and should be displayed as percentages.
_PCT_METRICS = {"accuracy", "acc3", "acc5"}


def load_metric(path, severity, metric):
    """Return {corruption: {prefix: value}} for the given severity and metric."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row["severity"]).strip() != str(severity):
                continue
            corruption = row["corruption"].strip()
            data[corruption] = {
                prefix: float(row[f"{prefix}/{metric}"])
                for _, prefix in MODELS
            }
    return data


def format_row(label, prefix, data, metric):
    """Build one LaTeX row: metric values in corruption order, then average."""
    missing = [c for c in CORRUPTION_ORDER if c not in data]
    if missing:
        raise KeyError(f"Missing corruptions for severity: {missing}")

    scale = 100.0 if metric in _PCT_METRICS else 1.0
    values = [data[c][prefix] * scale for c in CORRUPTION_ORDER]
    avg = sum(values) / len(values)
    cells = [f"{v:.2f}" for v in values] + [f"{avg:.2f}"]
    return f"& \\textit{{{label}}}\n& " + " & ".join(cells) + r" \\"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the W&B export CSV")
    parser.add_argument("--severity", default="5", help="Severity to extract (default: 5)")
    parser.add_argument(
        "--metric", default="accuracy",
        choices=["accuracy", "acc3", "acc5", "ece", "nll", "f1", "brier"],
        help="Metric to tabulate (default: accuracy)",
    )
    args = parser.parse_args()

    data = load_metric(args.csv_path, args.severity, args.metric)
    if not data:
        sys.exit(f"No rows found for severity={args.severity}")

    scale_note = " (%)" if args.metric in _PCT_METRICS else ""
    print(f"% {args.metric}{scale_note} at severity {args.severity}")
    for label, prefix in MODELS:
        print(format_row(label, prefix, data, args.metric))


if __name__ == "__main__":
    main()
