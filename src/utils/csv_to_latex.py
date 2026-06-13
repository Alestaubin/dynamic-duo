#!/usr/bin/env python3
"""
Transform a W&B export CSV into LaTeX table rows for the ImageNet-C
top-1 accuracy table.

CSV convention:
    small/* -> ResNet-50
    large/* -> ViT-B/16
    duo/*   -> Duo (ensemble)
The `*/accuracy` field is top-1 accuracy in [0, 1].

Usage:
    python csv_to_latex.py wandb_export.csv [--severity 5]
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


def load_accuracies(path, severity):
    """Return {corruption: {prefix: accuracy}} for the given severity."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row["severity"]).strip() != str(severity):
                continue
            corruption = row["corruption"].strip()
            data[corruption] = {
                prefix: float(row[f"{prefix}/accuracy"])
                for _, prefix in MODELS
            }
    return data


def format_row(label, prefix, data):
    """Build one LaTeX row: accuracies (%) in column order, then average."""
    missing = [c for c in CORRUPTION_ORDER if c not in data]
    if missing:
        raise KeyError(f"Missing corruptions for severity: {missing}")

    accs = [data[c][prefix] * 100.0 for c in CORRUPTION_ORDER]
    avg = sum(accs) / len(accs)
    cells = [f"{a:.2f}" for a in accs] + [f"{avg:.2f}"]
    return f"& \\textit{{{label}}}\n& " + " & ".join(cells) + r" \\"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the W&B export CSV")
    parser.add_argument("--severity", default="5", help="Severity to extract (default: 5)")
    args = parser.parse_args()

    data = load_accuracies(args.csv_path, args.severity)
    if not data:
        sys.exit(f"No rows found for severity={args.severity}")

    print(f"% Top-1 accuracy (%) at severity {args.severity}")
    for label, prefix in MODELS:
        print(format_row(label, prefix, data))


if __name__ == "__main__":
    main()
