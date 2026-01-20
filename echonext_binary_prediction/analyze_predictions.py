#!/usr/bin/env python3
"""
Standalone Bootstrap Analysis Script for Binary Disease Prediction

Analyzes prediction CSV files and computes:
- AUROC with 95% CI (bootstrap)
- AUPRC with 95% CI (bootstrap)
- F1 score at optimal threshold (Youden's Index) with 95% CI (bootstrap)
- DOR (Diagnostic Odds Ratio) at threshold=0.5 with 95% CI (bootstrap)

Outputs:
- ROC curve PNG
- PR curve PNG
- Summary CSV with all metrics and confidence intervals

Reference: prediction_analysis.py from ECG_to_Motion_Generation project
Author: Adapted for echonext_binary_prediction
"""

import argparse
import os
from typing import Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)
from tqdm import tqdm


def load_predictions(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions from CSV file.

    Args:
        csv_path: Path to CSV with columns [eid, true_label, predicted_probability]

    Returns:
        (true_labels, predicted_probs) as numpy arrays
    """
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['true_label', 'predicted_probability']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                        f"Available columns: {list(df.columns)}")

    true_labels = df['true_label'].values
    predicted_probs = df['predicted_probability'].values

    # Validate labels are binary
    unique_labels = np.unique(true_labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")

    # Validate probabilities in [0, 1]
    if np.any(predicted_probs < 0) or np.any(predicted_probs > 1):
        raise ValueError(f"Probabilities must be in [0, 1], found range: "
                        f"[{np.min(predicted_probs):.4f}, {np.max(predicted_probs):.4f}]")

    print(f"✅ Loaded {len(true_labels)} samples")
    print(f"  Class distribution: {np.sum(true_labels == 1):.0f} diseased, "
          f"{np.sum(true_labels == 0):.0f} healthy")

    return true_labels, predicted_probs


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUROC."""
    return roc_auc_score(y_true, y_prob)


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUPRC (Average Precision)."""
    return average_precision_score(y_true, y_prob)


def find_optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find optimal threshold using Youden's Index (maximizing sensitivity + specificity - 1).

    Returns:
        Optimal threshold value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def compute_f1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """
    Compute F1 score at given threshold.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        F1 score
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def compute_f1_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Compute F1 score at optimal threshold (Youden's Index).

    Returns:
        (f1_score, optimal_threshold)
    """
    optimal_threshold = find_optimal_threshold_youden(y_true, y_prob)
    f1 = compute_f1_at_threshold(y_true, y_prob, optimal_threshold)
    return f1, optimal_threshold


def compute_dor_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Diagnostic Odds Ratio (DOR) at given threshold.

    DOR = (TP * TN) / (FP * FN)

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Threshold (default 0.5)

    Returns:
        DOR value
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Avoid division by zero by adding small constant
    if fp == 0 or fn == 0:
        dor = (tp + 0.5) * (tn + 0.5) / ((fp + 0.5) * (fn + 0.5))
    else:
        dor = (tp * tn) / (fp * fn)

    return dor


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric_func: Function that takes (y_true, y_prob) and returns metric value
        n_bootstraps: Number of bootstrap iterations (default 1000)
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_seed: Random seed for reproducibility

    Returns:
        (metric_value, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)

    n_samples = len(y_true)
    bootstrap_metrics = []

    for _ in tqdm(range(n_bootstraps), desc="Bootstrap iterations"):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        # Skip if bootstrap sample doesn't have both classes
        if len(np.unique(y_true_boot)) < 2:
            continue

        # Compute metric
        try:
            metric_val = metric_func(y_true_boot, y_prob_boot)
            bootstrap_metrics.append(metric_val)
        except:
            continue

    # Compute original metric
    metric_value = metric_func(y_true, y_prob)

    # Compute confidence interval using percentile method
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))

    return metric_value, ci_lower, ci_upper


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str,
                  auroc: float, auroc_ci_lower: float, auroc_ci_upper: float):
    """
    Plot ROC curve and save to file.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save PNG file
        auroc: AUROC value
        auroc_ci_lower: Lower bound of 95% CI
        auroc_ci_upper: Upper bound of 95% CI
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auroc:.3f} '
                   f'[{auroc_ci_lower:.3f}, {auroc_ci_upper:.3f}])')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ ROC curve saved to: {output_path}")


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str,
                 auprc: float, auprc_ci_lower: float, auprc_ci_upper: float):
    """
    Plot Precision-Recall curve and save to file.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save PNG file
        auprc: AUPRC value
        auprc_ci_lower: Lower bound of 95% CI
        auprc_ci_upper: Upper bound of 95% CI
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # Baseline (prevalence)
    baseline = np.sum(y_true == 1) / len(y_true)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUPRC = {auprc:.3f} '
                   f'[{auprc_ci_lower:.3f}, {auprc_ci_upper:.3f}])')

    # Plot baseline
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--',
             label=f'Baseline (prevalence = {baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (PPV)', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ PR curve saved to: {output_path}")


def analyze_predictions(
    predictions_csv: str,
    output_dir: str,
    n_bootstraps: int = 1000,
    random_seed: int = 42
):
    """
    Main analysis function.

    Args:
        predictions_csv: Path to predictions CSV
        output_dir: Directory to save outputs
        n_bootstraps: Number of bootstrap iterations
        random_seed: Random seed
    """
    print("\n" + "="*80)
    print("BINARY DISEASE PREDICTION ANALYSIS")
    print("="*80)
    print(f"Input: {predictions_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Bootstrap iterations: {n_bootstraps}")
    print("="*80 + "\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load predictions
    y_true, y_prob = load_predictions(predictions_csv)

    # Compute metrics with bootstrap CI
    print("\n" + "="*80)
    print("COMPUTING METRICS WITH BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*80)

    # AUROC
    print("\n1. Computing AUROC...")
    auroc, auroc_ci_lower, auroc_ci_upper = bootstrap_metric(
        y_true, y_prob, compute_auroc, n_bootstraps, random_seed=random_seed
    )
    print(f"   AUROC: {auroc:.4f} (95% CI: [{auroc_ci_lower:.4f}, {auroc_ci_upper:.4f}])")

    # AUPRC
    print("\n2. Computing AUPRC...")
    auprc, auprc_ci_lower, auprc_ci_upper = bootstrap_metric(
        y_true, y_prob, compute_auprc, n_bootstraps, random_seed=random_seed
    )
    print(f"   AUPRC: {auprc:.4f} (95% CI: [{auprc_ci_lower:.4f}, {auprc_ci_upper:.4f}])")

    # F1 at optimal threshold (Youden's Index)
    print("\n3. Computing F1 at optimal threshold (Youden's Index)...")
    def f1_with_optimal_threshold(y_true, y_prob):
        f1, _ = compute_f1_optimal_threshold(y_true, y_prob)
        return f1

    f1_optimal, f1_ci_lower, f1_ci_upper = bootstrap_metric(
        y_true, y_prob, f1_with_optimal_threshold, n_bootstraps, random_seed=random_seed
    )
    optimal_threshold = find_optimal_threshold_youden(y_true, y_prob)
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    print(f"   F1 score: {f1_optimal:.4f} (95% CI: [{f1_ci_lower:.4f}, {f1_ci_upper:.4f}])")

    # DOR at threshold 0.5
    print("\n4. Computing DOR at threshold=0.5...")
    def dor_at_half(y_true, y_prob):
        return compute_dor_at_threshold(y_true, y_prob, threshold=0.5)

    dor, dor_ci_lower, dor_ci_upper = bootstrap_metric(
        y_true, y_prob, dor_at_half, n_bootstraps, random_seed=random_seed
    )
    print(f"   DOR: {dor:.4f} (95% CI: [{dor_ci_lower:.4f}, {dor_ci_upper:.4f}])")

    # Plot ROC curve
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plot_roc_curve(y_true, y_prob, roc_path, auroc, auroc_ci_lower, auroc_ci_upper)

    # Plot PR curve
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plot_pr_curve(y_true, y_prob, pr_path, auprc, auprc_ci_lower, auprc_ci_upper)

    # Save summary CSV
    print("\n" + "="*80)
    print("SAVING SUMMARY")
    print("="*80)

    summary_df = pd.DataFrame({
        'metric': ['AUROC', 'AUPRC', 'F1_optimal_threshold', 'DOR_threshold_0.5', 'optimal_threshold'],
        'value': [auroc, auprc, f1_optimal, dor, optimal_threshold],
        'ci_lower': [auroc_ci_lower, auprc_ci_lower, f1_ci_lower, dor_ci_lower, np.nan],
        'ci_upper': [auroc_ci_upper, auprc_ci_upper, f1_ci_upper, dor_ci_upper, np.nan],
    })

    summary_path = os.path.join(output_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to: {summary_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"  - ROC curve: {roc_path}")
    print(f"  - PR curve: {pr_path}")
    print(f"  - Summary: {summary_path}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Bootstrap Analysis for Binary Disease Prediction"
    )

    parser.add_argument(
        "--predictions",
        type=str,
        default='jzheng12/Codes/ECG_MeshHeart/output/echonext_binary_prediction/checkpoints_20260115_171403lv_ef_fr0_percent45_20260115_171403/val_predictions.csv',
        help="Path to predictions CSV file (columns: eid, true_label, predicted_probability)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots and summary (default: same directory as predictions)"
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for confidence intervals (default: 1000)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set output directory to same as predictions if not specified
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.predictions)
        if args.output_dir == "":
            args.output_dir = "."

    # Run analysis
    analyze_predictions(
        predictions_csv=args.predictions,
        output_dir=args.output_dir,
        n_bootstraps=args.n_bootstraps,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
