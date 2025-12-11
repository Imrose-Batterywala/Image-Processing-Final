#!/usr/bin/env python3
"""
Compute PSNR and SSIM scores for the LOL-pre dataset.

The script compares only the enhanced MDIB and Zero-DCE outputs against the
ground-truth images stored in eval15/high and reports a single PSNR/SSIM value
per method.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate PSNR and SSIM scores for LOL-pre images."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory containing eval15/high (default: ./eval15)",
    )
    parser.add_argument(
        "--mdib-dir",
        type=Path,
        default=None,
        help=(
            "Directory with MDIB enhanced images "
            "(default: ../MDIB/MDIB_Code/data/result/lol-pre)"
        ),
    )
    parser.add_argument(
        "--zerodce-dir",
        type=Path,
        default=None,
        help=(
            "Directory with Zero-DCE enhanced images "
            "(default: ../Zero-DCE/Zero-DCE_code/data/result/lol-pre)"
        ),
    )
    return parser.parse_args()


def resolve_directory(provided: Path | None, fallback: Path) -> Path:
    """Resolve optional paths relative to the working tree."""
    if provided is not None:
        return provided.expanduser().resolve()
    return fallback.resolve()


def load_image(path: Path) -> np.ndarray:
    """Load an image and normalize it into [0, 1] float space."""
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    return arr / 255.0


def build_image_map(directory: Path) -> Dict[str, Path]:
    """Map sample IDs to image paths inside a directory."""
    image_map: Dict[str, Path] = {}
    for entry in directory.iterdir():
        if entry.is_dir():
            continue
        if entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_map[entry.stem] = entry
    return image_map


def ensure_directory(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def compute_ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Compute SSIM using the original Wang et al. (2004) formulation.

    Works on RGB inputs by converting them to luminance and evaluating the
    global statistics across the full image.
    """
    if reference.shape != estimate.shape:
        raise ValueError("SSIM requires inputs with matching shape.")

    # Convert to luminance to follow the original single-channel formulation.
    ref_luma = (
        0.299 * reference[:, :, 0]
        + 0.587 * reference[:, :, 1]
        + 0.114 * reference[:, :, 2]
    )
    est_luma = (
        0.299 * estimate[:, :, 0]
        + 0.587 * estimate[:, :, 1]
        + 0.114 * estimate[:, :, 2]
    )

    mu_x = ref_luma.mean()
    mu_y = est_luma.mean()

    sigma_x = ref_luma.var()
    sigma_y = est_luma.var()
    sigma_xy = ((ref_luma - mu_x) * (est_luma - mu_y)).mean()

    L = 1.0  # pixel range after normalization
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return float(numerator / denominator)


def compute_metrics(
    sample_ids: Iterable[str],
    ground_truth_map: Dict[str, Path],
    method_maps: MutableMapping[str, Dict[str, Path]],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[str]]]:
    """Compute PSNR/SSIM for every method using the provided sample IDs."""
    psnr_results: Dict[str, List[float]] = {method: [] for method in method_maps}
    ssim_results: Dict[str, List[float]] = {method: [] for method in method_maps}
    missing: Dict[str, List[str]] = defaultdict(list)

    for sample_id in sample_ids:
        gt_path = ground_truth_map.get(sample_id)
        if gt_path is None:
            continue
        gt_image = load_image(gt_path)

        for method, image_map in method_maps.items():
            test_path = image_map.get(sample_id)
            if test_path is None:
                missing[method].append(sample_id)
                continue

            test_image = load_image(test_path)
            psnr = float(
                peak_signal_noise_ratio(gt_image, test_image, data_range=1.0)
            )
            ssim = compute_ssim(gt_image, test_image)

            psnr_results[method].append(psnr)
            ssim_results[method].append(ssim)

    return psnr_results, ssim_results, missing


def compute_averages(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    """Return the mean value for each method (if data exists)."""
    return {
        method: float(np.mean(values))
        for method, values in metric_lists.items()
        if values
    }


def print_summary(
    method_order: Iterable[str],
    avg_psnr: Dict[str, float],
    avg_ssim: Dict[str, float],
) -> None:
    print("\nAverage metrics (LOL-pre):")
    for method in method_order:
        psnr_value = avg_psnr.get(method)
        ssim_value = avg_ssim.get(method)
        if psnr_value is None or ssim_value is None:
            print(f"  {method:12s} -> insufficient data")
            continue

        print(f"  {method:12s} -> PSNR: {psnr_value:6.2f} dB | SSIM: {ssim_value:0.4f}")


def save_metric_plot(
    metric_name: str,
    averages: Dict[str, float],
    method_order: Iterable[str],
    ylabel: str,
    output_path: Path,
) -> None:
    methods = [method for method in method_order if method in averages]
    if not methods:
        print(f"Skipping {metric_name} plot: no data to plot.")
        return

    values = [averages[method] for method in methods]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    plt.figure(figsize=(4.5, 4))
    bars = plt.bar(methods, values, color=colors[: len(methods)])

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylabel(ylabel)
    plt.title(f"LOL-pre {metric_name}")
    ylim_max = max(values)
    if metric_name.upper() == "SSIM":
        plt.ylim(0, min(1.0, ylim_max + 0.1))
    else:
        plt.ylim(0, ylim_max * 1.1 if ylim_max > 0 else 1.0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    comparison_dir = Path(__file__).resolve().parent

    eval_dir = resolve_directory(args.eval_dir, comparison_dir / "eval15")
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

    ensure_directory(high_dir, "Ground-truth eval directory")
    ensure_directory(mdib_dir, "MDIB result directory")
    ensure_directory(zerodce_dir, "Zero-DCE result directory")

    high_map = build_image_map(high_dir)
    if not high_map:
        raise RuntimeError("Could not find ground-truth images in eval15/high.")

    sample_ids = sorted(high_map.keys())
    if not sample_ids:
        raise RuntimeError("No sample IDs found in eval15/high.")

    method_maps: Dict[str, Dict[str, Path]] = {
        "MDIB": build_image_map(mdib_dir),
        "Zero-DCE": build_image_map(zerodce_dir),
    }
    method_order = list(method_maps.keys())

    psnr_results, ssim_results, missing = compute_metrics(
        sample_ids, high_map, method_maps
    )

    avg_psnr = compute_averages(psnr_results)
    avg_ssim = compute_averages(ssim_results)

    print(f"Evaluated LOL-pre samples: {', '.join(sample_ids)}")
    print_summary(method_order, avg_psnr, avg_ssim)

    results_dir = comparison_dir / "results_mean"
    save_metric_plot(
        "PSNR",
        avg_psnr,
        method_order,
        "PSNR (dB)",
        results_dir / "lol_pre_psnr.png",
    )
    save_metric_plot(
        "SSIM",
        avg_ssim,
        method_order,
        "SSIM",
        results_dir / "lol_pre_ssim.png",
    )

    for method, sample_list in missing.items():
        if sample_list:
            print(f"\nWarning: {method} missing {len(sample_list)} samples: {sample_list}")


if __name__ == "__main__":
    main()
