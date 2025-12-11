#!/usr/bin/env python3
"""
Create a comparison grid for LOL-pre images.

Rows: low-light input, Zero-DCE output, MDIB output, ground truth.
Columns: selected samples (default: 111, 748, 778).

The resulting figure is stored in Comparison/results_images/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LOL-pre image comparisons for a set of samples."
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["111", "748", "778"],
        help="Sample IDs to visualize (default: 111 748 778).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory containing eval15 data (default: ./eval15).",
    )
    parser.add_argument(
        "--mdib-dir",
        type=Path,
        default=None,
        help="Directory with MDIB results (default: ../MDIB/.../lol-pre).",
    )
    parser.add_argument(
        "--zerodce-dir",
        type=Path,
        default=None,
        help="Directory with Zero-DCE results (default: ../Zero-DCE/.../lol-pre).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the comparison figure (default: ./results_images).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="lol_pre_samples.png",
        help="Filename for the saved plot (default: lol_pre_samples.png).",
    )
    return parser.parse_args()


def resolve_directory(provided: Path | None, fallback: Path) -> Path:
    if provided is not None:
        return provided.expanduser().resolve()
    return fallback.resolve()


def ensure_image(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower().lstrip(".") not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path}")


def load_image(path: Path) -> np.ndarray:
    ensure_image(path)
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def collect_paths(
    samples: Iterable[str],
    low_dir: Path,
    high_dir: Path,
    mdib_dir: Path,
    zerodce_dir: Path,
) -> List[dict]:
    """Collect the required image paths for every sample."""
    results = []
    for sample in samples:
        entry = {
            "Low-light": low_dir / f"{sample}.png",
            "Zero-DCE": zerodce_dir / f"{sample}.jpg",
            "MDIB": mdib_dir / f"{sample}.jpg",
            "Ground Truth": high_dir / f"{sample}.png",
        }

        # Some files may use .jpg instead of .png inside eval; check alternates.
        for key, path in entry.items():
            if path.exists():
                continue
            alternative_png = path.with_suffix(".png")
            alternative_jpg = path.with_suffix(".jpg")
            if alternative_png.exists():
                entry[key] = alternative_png
            elif alternative_jpg.exists():
                entry[key] = alternative_jpg

        for key, path in entry.items():
            ensure_image(path)

        results.append(entry)
    return results


def plot_grid(
    samples: Sequence[str],
    image_sets: List[dict],
    output_path: Path,
) -> None:
    rows = ["Low-light", "Zero-DCE", "MDIB", "Ground Truth"]
    n_rows = len(rows)
    n_cols = len(samples)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.8 * n_cols, 2.4 * n_rows), constrained_layout=True
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, sample in enumerate(samples):
        image_dict = image_sets[col]
        for row, row_name in enumerate(rows):
            ax = axes[row][col]
            img = load_image(image_dict[row_name])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                ax.set_title(f"Sample {sample}", fontsize=12)
            if col == 0:
                ax.set_ylabel(row_name, rotation=90, fontsize=11, labelpad=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("LOL-pre Comparison", fontsize=14)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison grid to: {output_path}")


def main() -> None:
    args = parse_args()
    comparison_dir = Path(__file__).resolve().parent

    eval_dir = resolve_directory(args.eval_dir, comparison_dir / "eval15")
    low_dir = eval_dir / "low"
    high_dir = eval_dir / "high"
    mdib_dir = resolve_directory(
        args.mdib_dir,
        comparison_dir.parent
        / "MDIB"
        / "MDIB_Code"
        / "data"
        / "result"
        / "lol-pre",
    )
    zerodce_dir = resolve_directory(
        args.zerodce_dir,
        comparison_dir.parent
        / "Zero-DCE"
        / "Zero-DCE_code"
        / "data"
        / "result"
        / "lol-pre",
    )

    for directory in [low_dir, high_dir, mdib_dir, zerodce_dir]:
        if not directory.exists():
            raise FileNotFoundError(f"Required directory not found: {directory}")

    samples = [s.strip() for s in args.samples if s.strip()]
    if not samples:
        raise ValueError("No sample IDs provided.")

    image_sets = collect_paths(samples, low_dir, high_dir, mdib_dir, zerodce_dir)

    output_dir = resolve_directory(args.output, comparison_dir / "results_images")
    output_path = output_dir / args.filename
    plot_grid(samples, image_sets, output_path)


if __name__ == "__main__":
    main()
