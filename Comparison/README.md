# NIMA Score Comparison

This directory contains scripts to compare NIMA (Neural Image Assessment) scores between Zero-DCE and DiDCE image enhancement methods.

## Structure

```
Comparison/
├── compare_nima_scores.py  # Main comparison script with plotting
├── README.md               # This file
└── results/               # Generated results (created when script runs)
    ├── nima_comparison.csv
    ├── nima_comparison_detailed.csv
    └── plots/
        ├── nima_scores_by_dataset.png
        ├── nima_distributions.png
        ├── nima_scatter.png
        ├── win_counts.png
        └── overall_summary.png
```

## Requirements

```bash
pip install pyiqa matplotlib seaborn numpy pillow torch
```

## Usage

From the `Comparison` directory:

```bash
python compare_nima_scores.py
```

Or with custom paths:

```bash
python compare_nima_scores.py \
    --zero-dce-dir ../Zero-DCE/Zero-DCE_code/data/result \
    --didce-dir ../MDIB/MDIB_Code/data/result \
    --output results/nima_comparison.csv \
    --plots-dir results/plots \
    --device cuda
```

## Output

The script generates:

1. **CSV Reports:**
   - `nima_comparison.csv`: Summary statistics per dataset
   - `nima_comparison_detailed.csv`: Individual image scores

2. **Visualization Plots:**
   - `nima_scores_by_dataset.png`: Bar chart comparing mean scores
   - `nima_distributions.png`: Box plots showing score distributions
   - `nima_scatter.png`: Scatter plot of Zero-DCE vs DiDCE scores
   - `win_counts.png`: Bar chart showing win counts per dataset
   - `overall_summary.png`: 4-panel summary of overall comparison

## NIMA Score

NIMA (Neural Image Assessment) is a deep learning-based image quality assessment metric. Higher scores indicate better perceived image quality (typically 0-10 range).


