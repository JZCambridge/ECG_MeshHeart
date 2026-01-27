#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
ECG Phenotype LASSO Regression Training Script with 5-Fold CV - SEX-STRATIFIED VERSION
Trains LASSO classifier with cross-validation, external validation, and sex-based filtering

Key features:
- Sex stratification (filters by Sex column before training)
- 5-fold stratified cross-validation on training data
- Each fold evaluates on external validation set
- Median model selection based on external validation performance
- Binary classification (0=healthy, 1=diseased)
- Uses ONLY 16 ECG morphology features (no raw ECG, no demographics)
- Grid search for optimal alpha regularization parameter

Adapted from main_phenotype_lasso_cv.py with sex stratification
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Import LASSO model and dataloader
from model_lasso import LassoClassifierWrapper, cv_grid_search_with_external_val, compute_metrics
from loader_phenotype_lasso import PhenotypeLassoDataModule


def filter_csv_by_column(
    csv_path: str,
    filter_column: str,
    filter_value: int,
    label_column: str,
    logger: logging.Logger
) -> Tuple[str, Dict[str, int]]:
    """
    Filter CSV by column value and return path to filtered CSV.

    Args:
        csv_path: Original CSV path
        filter_column: Column name to filter by (e.g., "Sex")
        filter_value: Value to filter for (e.g., 0 for female, 1 for male)
        label_column: Name of label column for class distribution
        logger: Logger instance

    Returns:
        (filtered_csv_path, statistics_dict)
    """
    logger.info(f"\nFiltering {csv_path}...")

    # Load CSV
    df = pd.read_csv(csv_path)
    original_count = len(df)
    logger.info(f"  Original samples: {original_count}")

    # Check if filter column exists
    if filter_column not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(
            f"Filter column '{filter_column}' not found in CSV.\n"
            f"Available columns: {available_cols}"
        )

    # Handle NaN values in filter column
    if df[filter_column].isna().any():
        nan_count = df[filter_column].isna().sum()
        logger.warning(f"  Found {nan_count} NaN values in '{filter_column}'. Excluding from filtered dataset.")
        df = df.dropna(subset=[filter_column])
        logger.info(f"  After removing NaN: {len(df)} samples")

    # Filter by column value
    df_filtered = df[df[filter_column] == filter_value].copy()
    filtered_count = len(df_filtered)

    # Check if filtered dataset is empty
    if filtered_count == 0:
        unique_values = sorted(df[filter_column].unique())
        raise ValueError(
            f"No samples found with {filter_column}={filter_value}.\n"
            f"Unique values in '{filter_column}': {unique_values}\n"
            f"Hint: For Sex column, use 0 for female and 1 for male."
        )

    logger.info(f"  Filtered samples: {filtered_count} ({100*filtered_count/original_count:.1f}%)")

    # Get filtered sex distribution
    sex_dist_filtered = df_filtered[filter_column].value_counts().to_dict()
    logger.info(f"  Sex distribution in filtered data:")
    for sex_val, count in sex_dist_filtered.items():
        logger.info(f"    {filter_column}={sex_val}: {count} ({100*count/filtered_count:.1f}%)")

    # Note about class balance
    if label_column in df_filtered.columns:
        label_count = df_filtered[label_column].notna().sum()
        logger.info(f"  Samples with label data after filtering: {label_count}")
        logger.info(f"  (Class balance will be computed after threshold application during dataset creation)")

    # Create temp filtered CSV path
    base_dir = os.path.dirname(csv_path) or '.'
    base_name = os.path.basename(csv_path).replace('.csv', '')
    filtered_csv_path = os.path.join(base_dir, f"{base_name}_filtered_{filter_column}{filter_value}.csv")

    # Save filtered CSV
    df_filtered.to_csv(filtered_csv_path, index=False)
    logger.info(f"  Saved filtered CSV to: {filtered_csv_path}")

    # Return statistics
    stats = {
        'original_count': original_count,
        'filtered_count': filtered_count,
    }

    return filtered_csv_path, stats


def setup_logging(checkpoint_dir: str):
    """
    Setup logging with file and console handlers.

    Args:
        checkpoint_dir: Directory where log files will be saved

    Returns:
        logger: Configured logger instance
    """
    # Create logs directory
    log_dir = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    log_file = os.path.join(log_dir, 'training.log')

    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler with explicit flushing
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Force immediate flush after each log
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

    logger.info("="*80)
    logger.info("ECG Phenotype LASSO Regression with 5-Fold CV - SEX-STRATIFIED")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")

    # Flush immediately
    for handler in logger.handlers:
        handler.flush()

    return logger


def save_predictions(model, X, y, eids, output_path, logger, dataset_name=""):
    """
    Generate and save predictions to CSV for bootstrap analysis.

    Args:
        model: Trained LassoClassifierWrapper
        X: Feature matrix [N, D]
        y: True labels [N]
        eids: Patient EIDs [N]
        output_path: Path to save CSV
        logger: Logger instance
        dataset_name: Name for logging (e.g., 'train', 'val')

    Returns:
        Path to saved CSV file
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Generating predictions for {dataset_name} set...")
    logger.info(f"{'='*80}")

    # Get predictions
    probs = model.predict_proba(X)

    # Create DataFrame
    predictions_df = pd.DataFrame({
        'eid': eids,
        'true_label': y,
        'predicted_probability': probs
    })

    # Save to CSV
    predictions_df.to_csv(output_path, index=False)

    logger.info(f"✅ Saved {len(predictions_df)} predictions to: {output_path}")
    logger.info(f"  Class distribution: {np.sum(y == 1):.0f} diseased, {np.sum(y == 0):.0f} healthy")
    logger.info(f"  Probability range: [{np.min(probs):.4f}, {np.max(probs):.4f}]")

    return output_path


def train_lasso_model_cv_stratified(args):
    """
    Main function for training sex-stratified LASSO classifier with 5-fold CV.

    Args:
        args: Command-line arguments

    Returns:
        None
    """
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(checkpoint_dir)
    logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    logger.info("="*80)
    logger.info("ECG PHENOTYPE LASSO WITH 5-FOLD CV - SEX-STRATIFIED")
    logger.info("="*80)
    logger.info(f"Sex filter: {args.sex} ({'Female' if args.sex == 0 else 'Male'})")
    logger.info(f"Label column: {args.label_column}")
    if args.threshold is not None:
        direction_str = "< threshold" if args.threshold_direction == "less_than" else "> threshold"
        logger.info(f"Threshold: {args.threshold} (values {direction_str} → diseased)")
    else:
        logger.info("Threshold: None (using binary labels directly)")

    logger.info("\nConfiguration:")
    logger.info(f"  CV folds: {args.n_folds}")
    logger.info(f"  Alpha range: [{args.alpha_min}, {args.alpha_max}]")
    logger.info(f"  Number of alphas: {args.n_alphas}")
    logger.info(f"  Best metric: {args.best_metric}")
    logger.info(f"  Threshold direction: {args.threshold_direction}")

    # Force flush after configuration
    for handler in logger.handlers:
        handler.flush()

    # Filter CSVs by sex
    logger.info("\n" + "="*80)
    logger.info("FILTERING CSVS BY SEX")
    logger.info("="*80)

    train_csv_filtered, train_stats = filter_csv_by_column(
        args.train_csv, 'Sex', args.sex, args.label_column, logger
    )
    val_csv_filtered, val_stats = filter_csv_by_column(
        args.val_csv, 'Sex', args.sex, args.label_column, logger
    )

    logger.info(f"\nFiltering summary:")
    logger.info(f"  Train: {train_stats['filtered_count']}/{train_stats['original_count']} samples")
    logger.info(f"  Val: {val_stats['filtered_count']}/{val_stats['original_count']} samples")

    # Force flush after filtering
    for handler in logger.handlers:
        handler.flush()

    # Load data
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA")
    logger.info("="*80)

    data_module = PhenotypeLassoDataModule(
        train_csv=train_csv_filtered,
        val_csv=val_csv_filtered,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        label_column=args.label_column,
        threshold=args.threshold,
        threshold_direction=args.threshold_direction,
    )

    X_train, y_train, X_val, y_val, eids_train, eids_val = data_module.prepare_data()
    feature_names = data_module.get_feature_names()

    logger.info(f"\n✅ Data loading completed!")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  External validation samples: {len(X_val)}")
    logger.info(f"  Number of features: {X_train.shape[1]}")

    # Force flush after data loading
    for handler in logger.handlers:
        handler.flush()

    # 5-fold CV grid search with external validation
    # Create alpha grid (log-spaced)
    alpha_grid = np.logspace(np.log10(args.alpha_min), np.log10(args.alpha_max), args.n_alphas)

    best_alpha, median_model, cv_results_detailed, cv_results_summary, median_fold_info = cv_grid_search_with_external_val(
        X_train, y_train,
        X_val, y_val,
        alpha_grid,
        best_metric=args.best_metric,
        n_folds=args.n_folds,
        feature_names=feature_names,
        random_state=42
    )

    # Save CV results
    logger.info("\n" + "="*80)
    logger.info("SAVING CV RESULTS")
    logger.info("="*80)

    detailed_path = os.path.join(checkpoint_dir, "cv_results_detailed.csv")
    cv_results_detailed.to_csv(detailed_path, index=False)
    logger.info(f"✅ Detailed CV results saved to: {detailed_path}")

    summary_path = os.path.join(checkpoint_dir, "cv_results_summary.csv")
    cv_results_summary.to_csv(summary_path, index=False)
    logger.info(f"✅ Summary CV results saved to: {summary_path}")

    # Save median fold info
    median_info_path = os.path.join(checkpoint_dir, "median_fold_info.csv")
    pd.DataFrame([median_fold_info]).to_csv(median_info_path, index=False)
    logger.info(f"✅ Median fold info saved to: {median_info_path}")

    # Force flush after CV results
    for handler in logger.handlers:
        handler.flush()

    # Log feature importance from median model
    logger.info("\n" + "="*80)
    logger.info("FEATURE IMPORTANCE (Median Model)")
    logger.info("="*80)

    coeffs_df = median_model.get_coefficients_df(feature_names)
    logger.info(f"\nAll features (sorted by absolute coefficient):")
    logger.info(f"\n{coeffs_df.to_string(index=False)}")

    active_features = median_model.get_active_features(feature_names)
    logger.info(f"\nActive features (non-zero coefficients): {len(active_features)}/{len(feature_names)}")

    # Save feature importance
    feature_importance_path = os.path.join(checkpoint_dir, "feature_importance.csv")
    coeffs_df.to_csv(feature_importance_path, index=False)
    logger.info(f"\n✅ Feature importance saved to: {feature_importance_path}")

    # Force flush after feature importance
    for handler in logger.handlers:
        handler.flush()

    # Save median model
    logger.info("\n" + "="*80)
    logger.info("SAVING MEDIAN MODEL")
    logger.info("="*80)

    model_path = os.path.join(checkpoint_dir, "median_model.pkl")
    median_model.save(model_path)

    # Generate predictions from median model on FULL datasets
    logger.info("\n" + "="*80)
    logger.info("GENERATING PREDICTIONS FROM MEDIAN MODEL")
    logger.info("="*80)

    save_predictions(
        median_model, X_train, y_train, eids_train,
        os.path.join(checkpoint_dir, "train_predictions.csv"),
        logger, "train (all samples)"
    )

    save_predictions(
        median_model, X_val, y_val, eids_val,
        os.path.join(checkpoint_dir, "val_predictions.csv"),
        logger, "external validation (all samples)"
    )

    # Final evaluation on full datasets
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION (MEDIAN MODEL ON FULL DATASETS)")
    logger.info("="*80)

    train_proba = median_model.predict_proba(X_train)
    val_proba = median_model.predict_proba(X_val)

    train_metrics = compute_metrics(y_train, train_proba)
    val_metrics = compute_metrics(y_val, val_proba)

    logger.info(f"\nMedian model (alpha={best_alpha:.4f}, fold={median_fold_info['median_fold_idx']}) performance:")
    logger.info(f"  Train (all {len(X_train)} samples) - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, "
               f"AUPRC: {train_metrics['auprc']:.4f}, F1: {train_metrics['f1']:.4f}")
    logger.info(f"  Val (all {len(X_val)} samples)   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, "
               f"AUPRC: {val_metrics['auprc']:.4f}, F1: {val_metrics['f1']:.4f}")

    # Cleanup temporary filtered CSV files
    try:
        os.remove(train_csv_filtered)
        os.remove(val_csv_filtered)
        logger.info(f"\nCleaned up temporary filtered CSV files")
    except Exception as e:
        logger.warning(f"Could not remove temporary CSV files: {e}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Sex stratification: {args.sex} ({'Female' if args.sex == 0 else 'Male'})")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"\nBest alpha selected: {best_alpha:.4f}")
    logger.info(f"Median fold selected: {median_fold_info['median_fold_idx']} (out of {args.n_folds} folds)")
    logger.info(f"\nPredictions saved:")
    logger.info(f"  Train: {checkpoint_dir}/train_predictions.csv")
    logger.info(f"  Val: {checkpoint_dir}/val_predictions.csv")
    logger.info(f"\nTo analyze predictions, run:")
    logger.info(f"  python analyze_predictions.py --predictions {checkpoint_dir}/val_predictions.csv")
    logger.info("="*80)

    # Final flush before exit
    for handler in logger.handlers:
        handler.flush()

    return None


if __name__ == "__main__":
    # Date and time to string YYYYMMDD_HHMMSS
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    print("Current date and time:", dt_string)

    parser = argparse.ArgumentParser(
        description="ECG Phenotype LASSO Regression with 5-Fold CV - Sex-Stratified"
    )

    # Sex stratification
    parser.add_argument(
        "--sex",
        type=int,
        default=1,
        choices=[0, 1],
        help="Sex to filter (0=female, 1=male). REQUIRED for stratified training."
    )

    # Data paths
    parser.add_argument(
        "--train_csv",
        type=str,
        default="ukb/jz_ecg/DM_measurements/processed_wt_volume/batches/wt_volume_first_20260106.csv",
        help="Path to training CSV file with binary disease labels.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="ukb/jz_ecg/DM_measurements/processed_wt_volume/batches/wt_volume_second_20260106.csv",
        help="Path to external validation CSV file with binary disease labels.",
    )
    parser.add_argument(
        "--ecg_phenotypes_path",
        type=str,
        default="cardiac/pi514/ukbb_ecg/Final/2_Factor_ECG/data/pt_data_ecg/ecg_phenotypes.csv",
        help="Path to ECG morphology phenotypes CSV file.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="jzheng12/Codes/ECG_MeshHeart/output/ecg_phenotype_lasso_cv/",
        help="Checkpoint directory.",
    )

    # Label configuration
    parser.add_argument(
        "--label_column",
        type=str,
        default="LVEDVi_ml/m2",
        help="Name of label column in CSV (can be binary or continuous).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=74,
        help="Threshold for converting continuous variable to binary (values < threshold → diseased=1). "
             "If None, expects binary labels (0 or 1) directly. Example: --threshold 50 for LVEF.",
    )
    parser.add_argument(
        "--threshold_direction",
        type=str,
        default="greater_than",
        choices=["less_than", "greater_than"],
        help="Direction for threshold comparison. 'less_than': values < threshold → diseased "
             "(default, for LVEF). 'greater_than': values > threshold → diseased (for WT_MAX).",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="auc",
        choices=["loss", "auc"],
        help="Metric to use for selecting best model. 'loss': min validation loss. "
             "'auc': separate tracking for best train AUC and best val AUC.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="",
        help="Prefix for output files.",
    )

    # CV parameters
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )

    # LASSO hyperparameters
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.001,
        help="Minimum alpha value for grid search (default: 0.001)"
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=10.0,
        help="Maximum alpha value for grid search (default: 10.0)"
    )
    parser.add_argument(
        "--n_alphas",
        type=int,
        default=20,
        help="Number of alpha values to try in grid search (default: 20)"
    )

    args = parser.parse_args()

    # Create checkpoint directory with timestamp and sex label
    sex_label = f"sex{args.sex}_{'female' if args.sex == 0 else 'male'}"
    args.checkpoint_dir = (args.checkpoint_dir + args.output_prefix + sex_label + '_' +
                          args.label_column.replace('/', '_') + '_' + args.threshold_direction +
                          str(args.threshold) + '_cv' + str(args.n_folds) + '_' + dt_string)

    train_lasso_model_cv_stratified(args)
