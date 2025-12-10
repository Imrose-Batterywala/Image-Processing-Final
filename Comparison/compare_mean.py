#!/usr/bin/env python3
"""
Compare metrics between Zero-DCE and DiDCE methods.
Computes mean NIMA scores, dark pixel fraction, bright pixel fraction, and patch contrast
for each dataset and creates comparison plots.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

try:
    import pyiqa
except ImportError:
    print("Warning: pyiqa not found. Please install it with: pip install pyiqa")
    pyiqa = None

_NIMA_METRIC = None
_NIQE_METRIC = None


def _sanitize_image(arr: np.ndarray) -> np.ndarray:
    """Sanitize image array to valid range [0, 1]."""
    cleaned = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(cleaned, 0.0, 1.0)


def load_image(image_path: Path) -> np.ndarray:
    """Load image and convert to numpy array in range [0, 1]."""
    img = Image.open(image_path).convert('RGB')
    arr = np.asarray(img).astype(np.float32) / 255.0
    return _sanitize_image(arr)


def compute_luminance(arr: np.ndarray) -> np.ndarray:
    """Compute luminance from RGB image.
    
    Args:
        arr: Image array in range [0, 1] with shape (H, W, 3)
    
    Returns:
        Luminance array with shape (H, W)
    """
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def grid_patch_stats(luma: np.ndarray, grid: int = 4) -> np.ndarray:
    """Compute mean luminance for each patch in a grid.
    
    Args:
        luma: Luminance array with shape (H, W)
        grid: Number of grid divisions (default 4)
    
    Returns:
        Array of patch means
    """
    h, w = luma.shape
    patch_h = h // grid
    patch_w = w // grid
    patches = []
    for i in range(grid):
        for j in range(grid):
            patch = luma[
                i * patch_h : (i + 1) * patch_h,
                j * patch_w : (j + 1) * patch_w,
            ]
            patches.append(float(patch.mean()))
    return np.array(patches)


def compute_nima_score(image: np.ndarray, device: str = "cuda") -> float:
    """Compute NIMA (Neural Image Assessment) score for an image.
    
    Args:
        image: Image array in range [0, 1] with shape (H, W, 3)
        device: Device to run computation on ('cuda' or 'cpu')
    
    Returns:
        NIMA score (higher is better, typically 0-10 range)
    """
    global _NIMA_METRIC
    
    if pyiqa is None:
        raise RuntimeError("pyiqa library is required. Install with: pip install pyiqa")
    
    if _NIMA_METRIC is None:
        _NIMA_METRIC = pyiqa.create_metric("nima", device=device)
    
    # Convert to tensor: (H, W, 3) -> (1, 3, H, W)
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)
    
    with torch.no_grad():
        score = _NIMA_METRIC(tensor)
    
    return float(score.item())


def compute_dark_pixel_fraction(image: np.ndarray) -> float:
    """Compute fraction of dark pixels (luminance < 0.2).
    
    Args:
        image: Image array in range [0, 1] with shape (H, W, 3)
    
    Returns:
        Fraction of dark pixels
    """
    luma = compute_luminance(image)
    return float((luma < 0.2).mean())


def compute_bright_pixel_fraction(image: np.ndarray) -> float:
    """Compute fraction of bright pixels (luminance > 0.9).
    
    Args:
        image: Image array in range [0, 1] with shape (H, W, 3)
    
    Returns:
        Fraction of bright pixels
    """
    luma = compute_luminance(image)
    return float((luma > 0.9).mean())


def compute_patch_contrast(image: np.ndarray, grid: int = 4) -> float:
    """Compute patch contrast as standard deviation of grid patch means.
    
    Args:
        image: Image array in range [0, 1] with shape (H, W, 3)
        grid: Number of grid divisions (default 4)
    
    Returns:
        Standard deviation of patch means (patch contrast)
    """
    luma = compute_luminance(image)
    patches = grid_patch_stats(luma, grid=grid)
    return float(patches.std())


def compute_niqe_score(image: np.ndarray, device: str = "cuda") -> float:
    """Compute NIQE (Natural Image Quality Evaluator) score for an image.
    
    Args:
        image: Image array in range [0, 1] with shape (H, W, 3)
        device: Device to run computation on ('cuda' or 'cpu')
    
    Returns:
        NIQE score (lower is better)
    """
    global _NIQE_METRIC
    
    if pyiqa is None:
        raise RuntimeError("pyiqa library is required. Install with: pip install pyiqa")
    
    if _NIQE_METRIC is None:
        _NIQE_METRIC = pyiqa.create_metric("niqe", device=device)
    
    # Compute luminance
    luma = compute_luminance(image)
    
    # Convert to tensor: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(luma).float().unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)
    
    with torch.no_grad():
        score = _NIQE_METRIC(tensor)
    
    return float(score.item())


def get_result_images(result_dir: Path) -> Dict[str, List[Path]]:
    """Get all result images organized by dataset.
    
    Args:
        result_dir: Path to result directory
    
    Returns:
        Dictionary mapping dataset names to lists of image paths
    """
    datasets = {}
    
    if not result_dir.exists():
        return datasets
    
    # Find all subdirectories (datasets)
    for dataset_dir in result_dir.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            images = []
            
            # Find all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(dataset_dir.glob(ext))
                images.extend(dataset_dir.glob(ext.upper()))
            
            if images:
                datasets[dataset_name] = sorted(images)
    
    return datasets


def compute_mean_metrics(result_dir: Path, device: str = "cuda") -> Dict[str, Dict[str, float]]:
    """Compute mean metrics for each dataset.
    
    Args:
        result_dir: Path to result directory
        device: Device to run computation on
    
    Returns:
        Dictionary mapping dataset names to dictionaries of mean metrics:
        {
            'dataset_name': {
                'nima': mean_nima_score,
                'dark_fraction': mean_dark_fraction,
                'bright_fraction': mean_bright_fraction,
                'patch_contrast': mean_patch_contrast,
                'niqe': mean_niqe_score
            }
        }
    """
    datasets = get_result_images(result_dir)
    mean_metrics = {}
    
    print(f"\nProcessing results from: {result_dir}")
    print(f"Found {len(datasets)} datasets")
    
    for dataset_name, image_paths in datasets.items():
        print(f"\n  Processing dataset: {dataset_name} ({len(image_paths)} images)")
        nima_scores = []
        dark_fractions = []
        bright_fractions = []
        patch_contrasts = []
        niqe_scores = []
        
        for img_path in image_paths:
            try:
                image = load_image(img_path)
                
                # Compute all metrics
                nima = compute_nima_score(image, device=device)
                dark_frac = compute_dark_pixel_fraction(image)
                bright_frac = compute_bright_pixel_fraction(image)
                contrast = compute_patch_contrast(image)
                niqe = compute_niqe_score(image, device=device)
                
                nima_scores.append(nima)
                dark_fractions.append(dark_frac)
                bright_fractions.append(bright_frac)
                patch_contrasts.append(contrast)
                niqe_scores.append(niqe)
            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")
        
        if nima_scores:
            mean_metrics[dataset_name] = {
                'nima': np.mean(nima_scores),
                'dark_fraction': np.mean(dark_fractions),
                'bright_fraction': np.mean(bright_fractions),
                'patch_contrast': np.mean(patch_contrasts),
                'niqe': np.mean(niqe_scores)
            }
            print(f"  Mean NIMA: {mean_metrics[dataset_name]['nima']:.4f}")
            print(f"  Mean Dark Fraction: {mean_metrics[dataset_name]['dark_fraction']:.4f}")
            print(f"  Mean Bright Fraction: {mean_metrics[dataset_name]['bright_fraction']:.4f}")
            print(f"  Mean Patch Contrast: {mean_metrics[dataset_name]['patch_contrast']:.4f}")
            print(f"  Mean NIQE: {mean_metrics[dataset_name]['niqe']:.4f}")
        else:
            print(f"  No valid scores for {dataset_name}")
    
    return mean_metrics


def create_comparison_plot(original_metrics: Dict[str, Dict[str, float]],
                          zero_dce_metrics: Dict[str, Dict[str, float]],
                          didce_metrics: Dict[str, Dict[str, float]],
                          metric_name: str,
                          metric_key: str,
                          ylabel: str,
                          title: str,
                          output_path: Path):
    """Create a comparison plot for a specific metric across datasets.
    
    Args:
        original_metrics: Metrics from original images
        zero_dce_metrics: Metrics from Zero-DCE method
        didce_metrics: Metrics from DiDCE method
        metric_name: Name of the metric (for display)
        metric_key: Key in the metrics dictionary
        ylabel: Y-axis label
        title: Plot title
        output_path: Path to save the plot
    """
    # Get common datasets
    common_datasets = sorted(set(original_metrics.keys()) & 
                            set(zero_dce_metrics.keys()) & 
                            set(didce_metrics.keys()))
    
    if not common_datasets:
        print(f"No common datasets found for plotting {metric_name}.")
        return
    
    # Prepare data
    datasets = common_datasets
    original_means = [original_metrics[d][metric_key] for d in datasets]
    zdce_means = [zero_dce_metrics[d][metric_key] for d in datasets]
    didce_means = [didce_metrics[d][metric_key] for d in datasets]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    bars1 = ax.bar(x - width, original_means, width, 
                   label='Original', alpha=0.8, color='#95a5a6')
    bars2 = ax.bar(x, zdce_means, width, 
                   label='Zero-DCE', alpha=0.8, color='#3498db')
    bars3 = ax.bar(x + width, didce_means, width,
                   label='MDIB', alpha=0.8, color='#e74c3c')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare mean NIMA scores between Zero-DCE and DiDCE methods"
    )
    parser.add_argument(
        '--zero-dce-dir',
        type=str,
        default='../Zero-DCE/Zero-DCE_code/data/result',
        help='Path to Zero-DCE results directory (relative to Comparison folder)'
    )
    parser.add_argument(
        '--didce-dir',
        type=str,
        default='../MDIB/MDIB_Code/data/result',
        help='Path to DiDCE results directory (relative to Comparison folder)'
    )
    parser.add_argument(
        '--original-dir',
        type=str,
        default='../MDIB/MDIB_Code/data/test_data',
        help='Path to original test data directory (relative to Comparison folder)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results_mean',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run NIMA computation on'
    )
    
    args = parser.parse_args()
    
    # Check if pyiqa is available
    if pyiqa is None:
        print("ERROR: pyiqa library is required for NIMA score computation.")
        print("Please install it with: pip install pyiqa")
        return 1
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Switching to CPU.")
        args.device = 'cpu'
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).resolve().parent
    zero_dce_dir = (script_dir / args.zero_dce_dir).resolve()
    didce_dir = (script_dir / args.didce_dir).resolve()
    original_dir = (script_dir / args.original_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    if not zero_dce_dir.exists():
        print(f"ERROR: Zero-DCE results directory not found: {zero_dce_dir}")
        return 1
    
    if not didce_dir.exists():
        print(f"ERROR: DiDCE results directory not found: {didce_dir}")
        return 1
    
    if not original_dir.exists():
        print(f"ERROR: Original test data directory not found: {original_dir}")
        return 1
    
    print("="*60)
    print("METRICS COMPARISON: Original vs Zero-DCE vs MDIB")
    print("="*60)
    
    # Compute mean metrics for original, Zero-DCE, and DiDCE
    print("\n" + "="*60)
    print("COMPUTING MEAN METRICS")
    print("="*60)
    
    original_metrics = compute_mean_metrics(original_dir, device=args.device)
    zero_dce_metrics = compute_mean_metrics(zero_dce_dir, device=args.device)
    didce_metrics = compute_mean_metrics(didce_dir, device=args.device)
    
    # Print summary
    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print(f"{'='*60}")
    
    common_datasets = sorted(set(original_metrics.keys()) & 
                            set(zero_dce_metrics.keys()) & 
                            set(didce_metrics.keys()))
    
    if not common_datasets:
        print("\nNo common datasets found for comparison.")
        return 1
    
    # Print summary tables for each metric
    metrics_info = [
        ('NIMA Score', 'nima', 'Higher is better'),
        ('Dark Pixel Fraction', 'dark_fraction', 'Fraction of pixels with luminance < 0.2'),
        ('Bright Pixel Fraction', 'bright_fraction', 'Fraction of pixels with luminance > 0.9'),
        ('Patch Contrast', 'patch_contrast', 'Std dev of grid patch means'),
        ('NIQE Score', 'niqe', 'Lower is better')
    ]
    
    for metric_name, metric_key, description in metrics_info:
        print(f"\n{metric_name} ({description}):")
        print(f"{'Dataset':<15} {'Original':<15} {'Zero-DCE':<15} {'MDIB':<15}")
        print("-" * 60)
        for dataset in common_datasets:
            orig = original_metrics[dataset][metric_key]
            zdce = zero_dce_metrics[dataset][metric_key]
            didce = didce_metrics[dataset][metric_key]
            print(f"{dataset:<15} {orig:<15.4f} {zdce:<15.4f} {didce:<15.4f}")
        
        # Overall means
        all_orig = [original_metrics[d][metric_key] for d in common_datasets]
        all_zdce = [zero_dce_metrics[d][metric_key] for d in common_datasets]
        all_didce = [didce_metrics[d][metric_key] for d in common_datasets]
        print("-" * 60)
        print(f"{'Overall Mean':<15} {np.mean(all_orig):<15.4f} {np.mean(all_zdce):<15.4f} {np.mean(all_didce):<15.4f}")
    
    # Create plots
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CREATING PLOTS")
    print(f"{'='*60}")
    
    # Create comparison plot for NIMA
    create_comparison_plot(
        original_metrics, zero_dce_metrics, didce_metrics,
        'NIMA Score', 'nima',
        'Mean NIMA Score',
        'Mean NIMA Scores: Original vs Zero-DCE vs MDIB',
        output_dir / 'nima.png'
    )
    
    # Create comparison plot for Dark Pixel Fraction
    create_comparison_plot(
        original_metrics, zero_dce_metrics, didce_metrics,
        'Dark Pixel Fraction', 'dark_fraction',
        'Mean Dark Pixel Fraction',
        'Mean Dark Pixel Fraction: Original vs Zero-DCE vs MDIB',
        output_dir / 'dark.png'
    )
    
    # Create comparison plot for Bright Pixel Fraction
    create_comparison_plot(
        original_metrics, zero_dce_metrics, didce_metrics,
        'Bright Pixel Fraction', 'bright_fraction',
        'Mean Bright Pixel Fraction',
        'Mean Bright Pixel Fraction: Original vs Zero-DCE vs MDIB',
        output_dir / 'bright.png'
    )
    
    # Create comparison plot for Patch Contrast
    create_comparison_plot(
        original_metrics, zero_dce_metrics, didce_metrics,
        'Patch Contrast', 'patch_contrast',
        'Mean Patch Contrast',
        'Mean Patch Contrast: Original vs Zero-DCE vs MDIB',
        output_dir / 'contrast.png'
    )
    
    # Create comparison plot for NIQE
    create_comparison_plot(
        original_metrics, zero_dce_metrics, didce_metrics,
        'NIQE Score', 'niqe',
        'Mean NIQE Score',
        'Mean NIQE Scores: Original vs Zero-DCE vs MDIB',
        output_dir / 'niqe.png'
    )
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
