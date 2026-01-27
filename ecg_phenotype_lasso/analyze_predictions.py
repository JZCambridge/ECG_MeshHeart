#!/usr/bin/env python3
"""
Standalone Bootstrap Analysis Script for Binary Disease Prediction

Analyzes prediction CSV files and computes:
- AUROC with 95% CI (bootstrap)
- AUPRC with 95% CI (bootstrap)
- F1 score at optimal threshold (Youden's Index) with 95% CI (bootstrap)
- TPR (Sensitivity) at optimal threshold with 95% CI (bootstrap)
- TNR (Specificity) at optimal threshold with 95% CI (bootstrap)
- FPR (Fall-out) at optimal threshold with 95% CI (bootstrap)
- FNR (Miss rate) at optimal threshold with 95% CI (bootstrap)
- DOR (Diagnostic Odds Ratio) at threshold=0.5 with 95% CI (bootstrap)

Outputs:
- ROC curve PNG
- PR curve PNG
- Confusion matrix PNG (at optimal threshold)
- Summary CSV with all metrics and confidence intervals in formatted strings

Supports both single file and batch processing modes.

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
    roc_auc_score,
    confusion_matrix
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


def compute_confusion_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """
    Compute confusion matrix metrics (TPR, TNR, FPR, FNR) at given threshold.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        Dictionary with tpr, tnr, fpr, fnr
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Calculate metrics with safe division
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity, Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Fall-out
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Miss rate

    return {
        'tpr': tpr,
        'tnr': tnr,
        'fpr': fpr,
        'fnr': fnr
    }


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


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, output_path: str,
                         threshold: float, dataset_label: str = ""):
    """
    Plot confusion matrix at given threshold and save to file.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save PNG file
        threshold: Threshold for binary predictions
        dataset_label: Label for dataset (used in title)
    """
    # Convert probabilities to binary predictions at threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics for display
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot confusion matrix as heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(xticks=[0, 1],
           yticks=[0, 1],
           xticklabels=['Predicted Negative', 'Predicted Positive'],
           yticklabels=['True Negative', 'True Positive'],
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations with counts and percentages
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percentage = count / np.sum(cm) * 100
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f'{count}\n({percentage:.1f}%)',
                   ha="center", va="center", color=text_color, fontsize=14, fontweight='bold')

    # Add title with metrics
    title = f'Confusion Matrix'
    if dataset_label:
        title += f': {dataset_label}'
    title += f'\nThreshold = {threshold:.4f}\n'
    title += f'TPR={tpr:.3f}, TNR={tnr:.3f}, FPR={fpr:.3f}, FNR={fnr:.3f}'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved to: {output_path}")


def analyze_predictions_single(
    predictions_csv: str,
    output_dir: str,
    dataset_label: str = None,
    n_bootstraps: int = 1000,
    random_seed: int = 42
):
    """
    Analyze a single prediction file and return results.

    Args:
        predictions_csv: Path to predictions CSV
        output_dir: Directory to save outputs
        dataset_label: Label for this dataset (used in filenames)
        n_bootstraps: Number of bootstrap iterations
        random_seed: Random seed

    Returns:
        Dictionary with all computed metrics and metadata
    """
    if dataset_label is None:
        dataset_label = os.path.splitext(os.path.basename(predictions_csv))[0]
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

    # TPR, TNR, FPR, FNR at optimal threshold
    print("\n4. Computing TPR, TNR, FPR, FNR at optimal threshold...")
    def tpr_at_optimal(y_true, y_prob):
        threshold = find_optimal_threshold_youden(y_true, y_prob)
        return compute_confusion_metrics_at_threshold(y_true, y_prob, threshold)['tpr']

    def tnr_at_optimal(y_true, y_prob):
        threshold = find_optimal_threshold_youden(y_true, y_prob)
        return compute_confusion_metrics_at_threshold(y_true, y_prob, threshold)['tnr']

    def fpr_at_optimal(y_true, y_prob):
        threshold = find_optimal_threshold_youden(y_true, y_prob)
        return compute_confusion_metrics_at_threshold(y_true, y_prob, threshold)['fpr']

    def fnr_at_optimal(y_true, y_prob):
        threshold = find_optimal_threshold_youden(y_true, y_prob)
        return compute_confusion_metrics_at_threshold(y_true, y_prob, threshold)['fnr']

    tpr, tpr_ci_lower, tpr_ci_upper = bootstrap_metric(
        y_true, y_prob, tpr_at_optimal, n_bootstraps, random_seed=random_seed
    )
    print(f"   TPR (Sensitivity): {tpr:.4f} (95% CI: [{tpr_ci_lower:.4f}, {tpr_ci_upper:.4f}])")

    tnr, tnr_ci_lower, tnr_ci_upper = bootstrap_metric(
        y_true, y_prob, tnr_at_optimal, n_bootstraps, random_seed=random_seed
    )
    print(f"   TNR (Specificity): {tnr:.4f} (95% CI: [{tnr_ci_lower:.4f}, {tnr_ci_upper:.4f}])")

    fpr, fpr_ci_lower, fpr_ci_upper = bootstrap_metric(
        y_true, y_prob, fpr_at_optimal, n_bootstraps, random_seed=random_seed
    )
    print(f"   FPR (Fall-out): {fpr:.4f} (95% CI: [{fpr_ci_lower:.4f}, {fpr_ci_upper:.4f}])")

    fnr, fnr_ci_lower, fnr_ci_upper = bootstrap_metric(
        y_true, y_prob, fnr_at_optimal, n_bootstraps, random_seed=random_seed
    )
    print(f"   FNR (Miss rate): {fnr:.4f} (95% CI: [{fnr_ci_lower:.4f}, {fnr_ci_upper:.4f}])")

    # DOR at threshold 0.5
    print("\n5. Computing DOR at threshold=0.5...")
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
    roc_path = os.path.join(output_dir, f"roc_curve_{dataset_label}.png")
    plot_roc_curve(y_true, y_prob, roc_path, auroc, auroc_ci_lower, auroc_ci_upper)

    # Plot PR curve
    pr_path = os.path.join(output_dir, f"pr_curve_{dataset_label}.png")
    plot_pr_curve(y_true, y_prob, pr_path, auprc, auprc_ci_lower, auprc_ci_upper)

    # Plot confusion matrix at optimal threshold
    cm_path = os.path.join(output_dir, f"confusion_matrix_{dataset_label}.png")
    plot_confusion_matrix(y_true, y_prob, cm_path, optimal_threshold, dataset_label)

    # Calculate sample counts
    n_samples = len(y_true)
    n_diseased = int(np.sum(y_true == 1))
    n_healthy = int(np.sum(y_true == 0))

    # Print summary table
    print("\n" + "="*80)
    print(f"SUMMARY OF RESULTS - {dataset_label}")
    print("="*80)
    print(f"Total samples: {n_samples} ({n_diseased} diseased, {n_healthy} healthy)")
    print(f"Optimal threshold (Youden's Index): {optimal_threshold:.4f}")
    print("-"*80)
    print(f"{'AUROC':30s}: {auroc:.4f} [{auroc_ci_lower:.4f}, {auroc_ci_upper:.4f}]")
    print(f"{'AUPRC':30s}: {auprc:.4f} [{auprc_ci_lower:.4f}, {auprc_ci_upper:.4f}]")
    print(f"{'F1_optimal_threshold':30s}: {f1_optimal:.4f} [{f1_ci_lower:.4f}, {f1_ci_upper:.4f}]")
    print(f"{'TPR_optimal_threshold':30s}: {tpr:.4f} [{tpr_ci_lower:.4f}, {tpr_ci_upper:.4f}]")
    print(f"{'TNR_optimal_threshold':30s}: {tnr:.4f} [{tnr_ci_lower:.4f}, {tnr_ci_upper:.4f}]")
    print(f"{'FPR_optimal_threshold':30s}: {fpr:.4f} [{fpr_ci_lower:.4f}, {fpr_ci_upper:.4f}]")
    print(f"{'FNR_optimal_threshold':30s}: {fnr:.4f} [{fnr_ci_lower:.4f}, {fnr_ci_upper:.4f}]")
    print(f"{'DOR_threshold_0.5':30s}: {dor:.4f} [{dor_ci_lower:.4f}, {dor_ci_upper:.4f}]")
    print("="*80 + "\n")

    # Return results dictionary
    return {
        'dataset': dataset_label,
        'metrics': {
            'AUROC': (auroc, auroc_ci_lower, auroc_ci_upper),
            'AUPRC': (auprc, auprc_ci_lower, auprc_ci_upper),
            'F1_optimal_threshold': (f1_optimal, f1_ci_lower, f1_ci_upper),
            'TPR_optimal_threshold': (tpr, tpr_ci_lower, tpr_ci_upper),
            'TNR_optimal_threshold': (tnr, tnr_ci_lower, tnr_ci_upper),
            'FPR_optimal_threshold': (fpr, fpr_ci_lower, fpr_ci_upper),
            'FNR_optimal_threshold': (fnr, fnr_ci_lower, fnr_ci_upper),
            'DOR_threshold_0.5': (dor, dor_ci_lower, dor_ci_upper),
            'optimal_threshold': (optimal_threshold, np.nan, np.nan)
        },
        'metadata': {
            'n_samples': n_samples,
            'n_diseased': n_diseased,
            'n_healthy': n_healthy,
            'optimal_threshold': optimal_threshold
        },
        'plots': {
            'roc_curve': roc_path,
            'pr_curve': pr_path,
            'confusion_matrix': cm_path
        }
    }


def analyze_predictions_batch(
    prediction_files: list,
    dataset_labels: list,
    output_dir: str,
    n_bootstraps: int = 1000,
    random_seed: int = 42
):
    """
    Analyze multiple prediction files in batch mode.

    Args:
        prediction_files: List of paths to prediction CSV files
        dataset_labels: List of labels for each dataset
        output_dir: Directory to save outputs
        n_bootstraps: Number of bootstrap iterations
        random_seed: Random seed
    """
    print("\n" + "="*80)
    print("BATCH BINARY DISEASE PREDICTION ANALYSIS")
    print("="*80)
    print(f"Number of datasets: {len(prediction_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Bootstrap iterations: {n_bootstraps}")
    print("="*80 + "\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each dataset
    all_results = []
    for pred_file, label in zip(prediction_files, dataset_labels):
        print(f"\n{'='*80}")
        print(f"Processing dataset: {label}")
        print(f"File: {pred_file}")
        print(f"{'='*80}\n")

        result = analyze_predictions_single(
            predictions_csv=pred_file,
            output_dir=output_dir,
            dataset_label=label,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed
        )
        all_results.append(result)

    # Combine all results into single CSV (long format)
    print("\n" + "="*80)
    print("SAVING COMBINED SUMMARY")
    print("="*80)

    combined_rows = []
    for result in all_results:
        dataset = result['dataset']
        metadata = result['metadata']

        for metric_name, (value, ci_lower, ci_upper) in result['metrics'].items():
            if metric_name == 'optimal_threshold':
                value_with_ci = f"{value:.4f}"
                opt_threshold_val = value
            else:
                value_with_ci = f"{value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
                opt_threshold_val = np.nan

            combined_rows.append({
                'dataset': dataset,
                'metric': metric_name,
                'value_with_ci': value_with_ci,
                'n_samples': metadata['n_samples'],
                'n_diseased': metadata['n_diseased'],
                'n_healthy': metadata['n_healthy'],
                'optimal_threshold': opt_threshold_val
            })

    combined_df = pd.DataFrame(combined_rows)
    summary_path = os.path.join(output_dir, "metrics_summary_all_datasets.csv")
    combined_df.to_csv(summary_path, index=False)
    print(f"✅ Combined summary saved to: {summary_path}")

    # Print final summary
    print("\n" + "="*80)
    print("BATCH ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"  - Combined summary: {summary_path}")
    print(f"  - Individual ROC curves: {len(all_results)} files")
    print(f"  - Individual PR curves: {len(all_results)} files")
    print(f"  - Individual confusion matrices: {len(all_results)} files")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Bootstrap Analysis for Binary Disease Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode
  python analyze_predictions.py --predictions path/to/predictions.csv --output_dir results/

  # Batch mode
  python analyze_predictions.py --batch_mode \\
    --prediction_files file1.csv file2.csv file3.csv \\
    --dataset_labels LVEF45 LVEF50 WT_Max15 \\
    --output_dir results/
        """
    )

    # Mode selection
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        default=True,
        help="Enable batch processing mode for multiple prediction files"
    )

    # Single file mode arguments
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to predictions CSV file (single file mode)"
    )
    parser.add_argument(
        "--dataset_label",
        type=str,
        default=None,
        help="Label for dataset (single file mode, default: filename)"
    )

    # Batch mode arguments
    parser.add_argument(
        "--prediction_files",
        nargs='+',
        type=str,
        default=[
                'jzheng12/Codes/ECG_MeshHeart/output/ecg_phenotype_lasso_cv/allbatches_lv_ef_fr0_percent_less_than45_cv5_20260126_150929/train_predictions.csv', # Use train_predictions.csv for samller AUC
                'jzheng12/Codes/ECG_MeshHeart/output/ecg_phenotype_lasso_cv/allbatches_lv_ef_fr0_percent_less_than50_cv5_20260126_150553/train_predictions.csv',
                'jzheng12/Codes/ECG_MeshHeart/output/ecg_phenotype_lasso_cv/sex0_female_LVEDVi_ml_m2_greater_than61_cv5_20260126_153649/val_predictions.csv',
                'jzheng12/Codes/ECG_MeshHeart/output/ecg_phenotype_lasso_cv/sex1_male_LVEDVi_ml_m2_greater_than74_cv5_20260126_153908/val_predictions.csv',
                'jzheng12/Codes/ECG_MeshHeart/output/ecg_phenotype_lasso_cv/allbatches_wt_max_greater_than13_cv5_20260126_151131/val_predictions.csv',
                'jzheng12/Codes/ECG_MeshHeart/output/ecg_phenotype_lasso_cv/allbatches_wt_max_greater_than15_cv5_20260126_151212/val_predictions.csv'
                ],
        help="List of paths to prediction CSV files (batch mode)"
    )
    parser.add_argument(
        "--dataset_labels",
        nargs='+',
        type=str,
        default=['LVEF45', 'LVEF50', 'LVEDVi_female62', 'LVEDVi_male75', 'WT_Max13', 'WT_Max15'],
        help="List of labels for each dataset (batch mode)"
    )

    # Common arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default='ukb/jz_ecg/ecg_echonext_binaryprediction_23Jan26/prediction_phenotypes_lasso_analysis',
        help="Output directory for plots and summary"
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

    if args.batch_mode:
        # Batch mode
        if args.prediction_files is None or args.dataset_labels is None:
            parser.error("--batch_mode requires --prediction_files and --dataset_labels")

        if len(args.prediction_files) != len(args.dataset_labels):
            parser.error("Number of prediction files must match number of dataset labels")

        analyze_predictions_batch(
            prediction_files=args.prediction_files,
            dataset_labels=args.dataset_labels,
            output_dir=args.output_dir,
            n_bootstraps=args.n_bootstraps,
            random_seed=args.random_seed
        )
    else:
        # Single file mode
        result = analyze_predictions_single(
            predictions_csv=args.predictions,
            output_dir=args.output_dir,
            dataset_label=args.dataset_label,
            n_bootstraps=args.n_bootstraps,
            random_seed=args.random_seed
        )

        # Save single file summary
        print("\n" + "="*80)
        print("SAVING SUMMARY")
        print("="*80)

        combined_rows = []
        dataset = result['dataset']
        metadata = result['metadata']

        for metric_name, (value, ci_lower, ci_upper) in result['metrics'].items():
            if metric_name == 'optimal_threshold':
                value_with_ci = f"{value:.4f}"
                opt_threshold_val = value
            else:
                value_with_ci = f"{value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
                opt_threshold_val = np.nan

            combined_rows.append({
                'dataset': dataset,
                'metric': metric_name,
                'value_with_ci': value_with_ci,
                'n_samples': metadata['n_samples'],
                'n_diseased': metadata['n_diseased'],
                'n_healthy': metadata['n_healthy'],
                'optimal_threshold': opt_threshold_val
            })

        summary_df = pd.DataFrame(combined_rows)
        summary_path = os.path.join(args.output_dir, "metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Summary saved to: {summary_path}")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Output directory: {args.output_dir}")
        print(f"  - ROC curve: {result['plots']['roc_curve']}")
        print(f"  - PR curve: {result['plots']['pr_curve']}")
        print(f"  - Confusion matrix: {result['plots']['confusion_matrix']}")
        print(f"  - Summary: {summary_path}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
