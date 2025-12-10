#!/usr/bin/env python3
"""
Per-image metric comparison between Zero-DCE and MDIB.
Produces one figure per dataset with subplots for each metric:
NIMA, Dark Fraction, Bright Fraction, Patch Contrast, and NIQE.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# Metrics available via pyiqa (NIMA, NIQE)
try:
    import pyiqa
except ImportError:
    pyiqa = None
    print("Warning: pyiqa not found. Install with: pip install pyiqa")


MetricsDict = Dict[str, float]
ImageMetrics = Dict[str, Dict[str, float]]

_NIMA_METRIC = None
_NIQE_METRIC = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
METRICS = [
    ("nima", "NIMA (↑)"),
    ("dark_fraction", "Dark Fraction (↓<0.2)"),
    ("bright_fraction", "Bright Fraction (↑>0.9)"),
    ("patch_contrast", "Patch Contrast"),
    ("niqe", "NIQE (↓)"),
]


def _sanitize(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return _sanitize(arr)


def compute_luminance(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def grid_patch_stats(luma: np.ndarray, grid: int = 4) -> np.ndarray:
    h, w = luma.shape
    ph, pw = h // grid, w // grid
    patches: List[float] = []
    for i in range(grid):
        for j in range(grid):
            patch = luma[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
            patches.append(float(patch.mean()))
    return np.array(patches)


def compute_nima(arr: np.ndarray, device: torch.device) -> float:
    global _NIMA_METRIC
    if pyiqa is None:
        return float("nan")
    if _NIMA_METRIC is None:
        _NIMA_METRIC = pyiqa.create_metric("nima", device=str(device))
    tensor = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(_NIMA_METRIC(tensor).item())


def compute_niqe(arr: np.ndarray, device: torch.device) -> float:
    global _NIQE_METRIC
    if pyiqa is None:
        return float("nan")
    if _NIQE_METRIC is None:
        _NIQE_METRIC = pyiqa.create_metric("niqe", device=str(device))
    luma = compute_luminance(arr)
    tensor = torch.from_numpy(luma).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(_NIQE_METRIC(tensor).item())


def compute_metrics_for_image(arr: np.ndarray, device: torch.device) -> MetricsDict:
    luma = compute_luminance(arr)
    patches = grid_patch_stats(luma)
    return {
        "nima": compute_nima(arr, device),
        "dark_fraction": float((luma < 0.2).mean()),
        "bright_fraction": float((luma > 0.9).mean()),
        "patch_contrast": float(patches.std()),
        "niqe": compute_niqe(arr, device),
    }


def collect_images(result_dir: Path) -> Dict[str, List[Path]]:
    datasets: Dict[str, List[Path]] = {}
    if not result_dir.exists():
        return datasets
    for dataset_dir in result_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        imgs = sorted([p for p in dataset_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if imgs:
            datasets[dataset_dir.name] = imgs
    return datasets


def compute_dataset_metrics(
    result_dir: Path, device: torch.device
) -> Dict[str, ImageMetrics]:
    datasets = collect_images(result_dir)
    out: Dict[str, ImageMetrics] = {}
    for dname, paths in datasets.items():
        metrics_per_image: ImageMetrics = {}
        for p in paths:
            arr = load_image(p)
            metrics_per_image[p.name] = compute_metrics_for_image(arr, device)
        out[dname] = metrics_per_image
    return out


def align_common_images(
    a: ImageMetrics, b: ImageMetrics
) -> List[Tuple[str, MetricsDict, MetricsDict]]:
    names = sorted(set(a.keys()) & set(b.keys()))
    return [(n, a[n], b[n]) for n in names]


def plot_dataset(
    dataset: str,
    pairs: List[Tuple[str, MetricsDict, MetricsDict]],
    output_dir: Path,
) -> None:
    if not pairs:
        print(f"No common images for dataset {dataset}")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    x = np.arange(len(pairs))
    width = 0.35
    image_names = [name for name, _, _ in pairs]

    for idx, (key, title) in enumerate(METRICS):
        zdce_vals = [p[1][key] for p in pairs]
        mdib_vals = [p[2][key] for p in pairs]

        ax = axes[idx]
        ax.bar(x - width / 2, zdce_vals, width, label="Zero-DCE", color="#3498db", alpha=0.85)
        ax.bar(x + width / 2, mdib_vals, width, label="MDIB", color="#e74c3c", alpha=0.85)
        ax.set_title(f"{dataset} – {title}", fontsize=12, fontweight="bold")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=90, fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if idx == 0:
            ax.legend(fontsize=10)

    # Hide unused subplot if metrics < axes
    for ax in axes[len(METRICS) :]:
        ax.axis("off")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-image metric comparison between Zero-DCE and MDIB"
    )
    parser.add_argument(
        "--zero-dce-dir",
        type=str,
        default="../Zero-DCE/Zero-DCE_code/data/result",
        help="Path to Zero-DCE results directory",
    )
    parser.add_argument(
        "--mdib-dir",
        type=str,
        default="../MDIB/MDIB_Code/data/result",
        help="Path to MDIB results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_per_image",
        help="Directory to save per-dataset plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for metric computation",
    )
    args = parser.parse_args()

    if pyiqa is None:
        print("ERROR: pyiqa is required (for NIMA/NIQE). Install with: pip install pyiqa")
        return 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    script_dir = Path(__file__).resolve().parent
    zdce_dir = (script_dir / args.zero_dce_dir).resolve()
    mdib_dir = (script_dir / args.mdib_dir).resolve()
    out_dir = (script_dir / args.output_dir).resolve()

    if not zdce_dir.exists():
        print(f"ERROR: Zero-DCE results directory not found: {zdce_dir}")
        return 1
    if not mdib_dir.exists():
        print(f"ERROR: MDIB results directory not found: {mdib_dir}")
        return 1

    print("Computing per-image metrics (this may take a while)...")
    zdce_metrics = compute_dataset_metrics(zdce_dir, device)
    mdib_metrics = compute_dataset_metrics(mdib_dir, device)

    datasets = sorted(set(zdce_metrics.keys()) & set(mdib_metrics.keys()))
    if not datasets:
        print("No common datasets found.")
        return 1

    for dset in datasets:
        pairs = align_common_images(zdce_metrics[dset], mdib_metrics[dset])
        plot_dataset(dset, pairs, out_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


