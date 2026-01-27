#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
ECG Phenotype LASSO Regression Training Script
Trains LASSO classifier to predict disease status from 16 ECG morphology features only

Key features:
- Binary classification (0=healthy, 1=diseased)
- Uses ONLY 16 ECG morphology features (no raw ECG, no demographics)
- Grid search for optimal alpha regularization parameter
- Saves prediction probabilities for bootstrap analysis
- Comprehensive logging (file + console)

Adapted from echonext_binary_prediction for LASSO regression
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Import LASSO model and dataloader
from model_lasso import LassoClassifierWrapper, grid_search_lasso, compute_metrics
from loader_phenotype_lasso import PhenotypeLassoDataModule


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
    logger.info("ECG Phenotype LASSO Regression Training")
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


def train_lasso_model(args):
    """
    Main function for training LASSO classifier.

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
    logger.info("ECG PHENOTYPE LASSO REGRESSION TRAINING")
    logger.info("="*80)
    logger.info(f"Label column: {args.label_column}")
    if args.threshold is not None:
        direction_str = "< threshold" if args.threshold_direction == "less_than" else "> threshold"
        logger.info(f"Threshold: {args.threshold} (values {direction_str} → diseased)")
    else:
        logger.info("Threshold: None (using binary labels directly)")

    logger.info("\nConfiguration:")
    logger.info(f"  Alpha range: [{args.alpha_min}, {args.alpha_max}]")
    logger.info(f"  Number of alphas: {args.n_alphas}")
    logger.info(f"  Best metric: {args.best_metric}")
    logger.info(f"  Threshold direction: {args.threshold_direction}")

    # Force flush after configuration
    for handler in logger.handlers:
        handler.flush()

    # Load data
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA")
    logger.info("="*80)

    data_module = PhenotypeLassoDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        ecg_phenotypes_path=args.ecg_phenotypes_path,
        label_column=args.label_column,
        threshold=args.threshold,
        threshold_direction=args.threshold_direction,
    )

    X_train, y_train, X_val, y_val, eids_train, eids_val = data_module.prepare_data()
    feature_names = data_module.get_feature_names()

    logger.info(f"\n✅ Data loading completed!")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    logger.info(f"  Number of features: {X_train.shape[1]}")

    # Force flush after data loading
    for handler in logger.handlers:
        handler.flush()

    # Grid search for best alpha
    logger.info("\n" + "="*80)
    logger.info("GRID SEARCH FOR OPTIMAL ALPHA")
    logger.info("="*80)

    # Create alpha grid (log-spaced)
    alpha_grid = np.logspace(np.log10(args.alpha_min), np.log10(args.alpha_max), args.n_alphas)

    best_model_trainauc, best_model_valauc, grid_results = grid_search_lasso(
        X_train, y_train, X_val, y_val,
        alpha_grid,
        best_metric=args.best_metric,
        feature_names=feature_names
    )

    # Save grid search results
    grid_results_path = os.path.join(checkpoint_dir, "grid_search_results.csv")
    grid_results.to_csv(grid_results_path, index=False)
    logger.info(f"\n✅ Grid search results saved to: {grid_results_path}")

    # Force flush after grid search
    for handler in logger.handlers:
        handler.flush()

    # Log feature importance from best model
    logger.info("\n" + "="*80)
    logger.info("FEATURE IMPORTANCE (Best Val Model)")
    logger.info("="*80)

    coeffs_df = best_model_valauc.get_coefficients_df(feature_names)
    logger.info(f"\nAll features (sorted by absolute coefficient):")
    logger.info(f"\n{coeffs_df.to_string(index=False)}")

    active_features = best_model_valauc.get_active_features(feature_names)
    logger.info(f"\nActive features (non-zero coefficients): {len(active_features)}/{len(feature_names)}")

    # Force flush after feature importance
    for handler in logger.handlers:
        handler.flush()

    # Save models and generate predictions based on best_metric
    logger.info("\n" + "="*80)
    logger.info("SAVING MODELS AND PREDICTIONS")
    logger.info("="*80)

    if args.best_metric == "auc":
        # Save two models (train AUC, val AUC)
        logger.info("\nSaving best models (AUC mode)...")

        # Best train AUC model
        model_path_trainauc = os.path.join(checkpoint_dir, "best_model_trainauc.pkl")
        best_model_trainauc.save(model_path_trainauc)

        # Best val AUC model
        model_path_valauc = os.path.join(checkpoint_dir, "best_model_valauc.pkl")
        best_model_valauc.save(model_path_valauc)

        # Generate predictions from both models
        logger.info("\nGenerating predictions from best train AUC model...")
        save_predictions(
            best_model_trainauc, X_train, y_train, eids_train,
            os.path.join(checkpoint_dir, "train_predictions_trainauc.csv"),
            logger, "train"
        )
        save_predictions(
            best_model_trainauc, X_val, y_val, eids_val,
            os.path.join(checkpoint_dir, "val_predictions_trainauc.csv"),
            logger, "val"
        )

        logger.info("\nGenerating predictions from best val AUC model...")
        save_predictions(
            best_model_valauc, X_train, y_train, eids_train,
            os.path.join(checkpoint_dir, "train_predictions_valauc.csv"),
            logger, "train"
        )
        save_predictions(
            best_model_valauc, X_val, y_val, eids_val,
            os.path.join(checkpoint_dir, "val_predictions_valauc.csv"),
            logger, "val"
        )

    elif args.best_metric == "loss":
        # Save single model (validation loss)
        logger.info("\nSaving best model (Loss mode)...")

        model_path = os.path.join(checkpoint_dir, "best_model.pkl")
        best_model_valauc.save(model_path)

        # Generate predictions
        save_predictions(
            best_model_valauc, X_train, y_train, eids_train,
            os.path.join(checkpoint_dir, "train_predictions.csv"),
            logger, "train"
        )
        save_predictions(
            best_model_valauc, X_val, y_val, eids_val,
            os.path.join(checkpoint_dir, "val_predictions.csv"),
            logger, "val"
        )

    # Final evaluation
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)

    train_proba = best_model_valauc.predict_proba(X_train)
    val_proba = best_model_valauc.predict_proba(X_val)

    train_metrics = compute_metrics(y_train, train_proba)
    val_metrics = compute_metrics(y_val, val_proba)

    logger.info(f"\nBest model (alpha={best_model_valauc.alpha:.4f}) performance:")
    logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, "
               f"AUPRC: {train_metrics['auprc']:.4f}, F1: {train_metrics['f1']:.4f}")
    logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, "
               f"AUPRC: {val_metrics['auprc']:.4f}, F1: {val_metrics['f1']:.4f}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    if args.best_metric == "loss":
        logger.info(f"\nPredictions from best model (loss-based) saved to:")
        logger.info(f"  Train: {checkpoint_dir}/train_predictions.csv")
        logger.info(f"  Val: {checkpoint_dir}/val_predictions.csv")
        logger.info(f"\nTo analyze predictions, run:")
        logger.info(f"  python analyze_predictions.py --predictions {checkpoint_dir}/val_predictions.csv")
    elif args.best_metric == "auc":
        logger.info(f"\nPredictions from TWO best models saved:")
        logger.info(f"\n1. Best Train AUC model:")
        logger.info(f"  Train: {checkpoint_dir}/train_predictions_trainauc.csv")
        logger.info(f"  Val: {checkpoint_dir}/val_predictions_trainauc.csv")
        logger.info(f"\n2. Best Val AUC model:")
        logger.info(f"  Train: {checkpoint_dir}/train_predictions_valauc.csv")
        logger.info(f"  Val: {checkpoint_dir}/val_predictions_valauc.csv")
        logger.info(f"\nTo analyze predictions, run:")
        logger.info(f"  python analyze_predictions.py --predictions {checkpoint_dir}/val_predictions_valauc.csv")

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
        description="ECG Phenotype LASSO Regression Training"
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
        help="Path to validation CSV file with binary disease labels.",
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
        default="jzheng12/Codes/ECG_MeshHeart/output/ecg_phenotype_lasso/",
        help="Checkpoint directory.",
    )

    # Label configuration
    parser.add_argument(
        "--label_column",
        type=str,
        default="lv_ef_fr0_percent",
        help="Name of label column in CSV (can be binary or continuous).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=45,
        help="Threshold for converting continuous variable to binary (values < threshold → diseased=1). "
             "If None, expects binary labels (0 or 1) directly. Example: --threshold 50 for LVEF.",
    )
    parser.add_argument(
        "--threshold_direction",
        type=str,
        default="less_than",
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
        default="allbatches_",
        help="Prefix for output files.",
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

    # Create checkpoint directory with timestamp
    args.checkpoint_dir = (args.checkpoint_dir + args.output_prefix + args.label_column + '_' +
                          args.threshold_direction + str(args.threshold) + '_' + dt_string)

    train_lasso_model(args)
