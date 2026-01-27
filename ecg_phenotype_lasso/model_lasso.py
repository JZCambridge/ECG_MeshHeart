#!/usr/bin/env python
"""
LASSO model wrapper for binary classification
Includes grid search for optimal alpha selection

Adapted from echonext_binary_prediction for LASSO regression
"""

import logging
import pickle
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LassoClassifierWrapper:
    """
    Wrapper for sklearn Lasso for binary classification
    Treats LASSO regression output as probability estimates (clipped to [0,1])
    """

    def __init__(self, alpha=1.0):
        """
        Initialize LASSO classifier

        Args:
            alpha: Regularization strength (higher = more regularization)
        """
        self.lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        self.alpha = alpha
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit LASSO on training data

        Args:
            X: Feature matrix [N, D]
            y: Binary labels [N] (0 or 1)
        """
        self.lasso.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Get probability predictions from LASSO
        Clips continuous predictions to [0, 1] range

        Args:
            X: Feature matrix [N, D]

        Returns:
            Predicted probabilities [N] in range [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = self.lasso.predict(X)
        # Clip to [0, 1] to convert to probabilities
        return np.clip(predictions, 0, 1)

    def predict(self, X, threshold=0.5):
        """
        Get binary predictions based on threshold

        Args:
            X: Feature matrix [N, D]
            threshold: Classification threshold (default: 0.5)

        Returns:
            Binary predictions [N] (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_active_features(self, feature_names):
        """
        Get features with non-zero coefficients

        Args:
            feature_names: List of feature names

        Returns:
            List of (feature_name, coefficient) tuples for active features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting active features")

        active = []
        for name, coef in zip(feature_names, self.lasso.coef_):
            if abs(coef) > 1e-10:  # Non-zero threshold
                active.append((name, coef))

        # Sort by absolute coefficient value
        active.sort(key=lambda x: abs(x[1]), reverse=True)
        return active

    def get_coefficients_df(self, feature_names):
        """
        Get all coefficients as a DataFrame

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with features and coefficients, sorted by absolute value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")

        df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.lasso.coef_
        })
        df['abs_coefficient'] = df['coefficient'].abs()
        df = df.sort_values('abs_coefficient', ascending=False)
        return df[['feature', 'coefficient', 'abs_coefficient']]

    def save(self, path):
        """
        Save model using pickle

        Args:
            path: Path to save model (.pkl file)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"✅ Saved model to {path}")

    @staticmethod
    def load(path):
        """
        Load model from pickle

        Args:
            path: Path to model file (.pkl)

        Returns:
            Loaded LassoClassifierWrapper
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"✅ Loaded model from {path}")
        return model


def compute_metrics(y_true, y_pred_proba):
    """
    Compute comprehensive metrics for binary classification

    Args:
        y_true: True labels [N] (0 or 1)
        y_pred_proba: Predicted probabilities [N] in range [0, 1]

    Returns:
        Dictionary with metrics: loss, auc, auprc, accuracy, f1
    """
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true, cannot compute AUC/AUPRC")
        auc = 0.0
        auprc = 0.0
    else:
        # Clip probabilities to avoid log(0) in loss computation
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)

        # Binary cross-entropy loss
        loss = log_loss(y_true, y_pred_proba_clipped)

        # AUC and AUPRC
        auc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)

    # Binary predictions at 0.5 threshold
    y_pred = (y_pred_proba >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Use clipped probabilities for loss
    if len(np.unique(y_true)) >= 2:
        return {
            'loss': loss,
            'auc': auc,
            'auprc': auprc,
            'accuracy': accuracy,
            'f1': f1
        }
    else:
        return {
            'loss': 0.0,
            'auc': auc,
            'auprc': auprc,
            'accuracy': accuracy,
            'f1': f1
        }


def grid_search_lasso(
    X_train, y_train, X_val, y_val,
    alpha_grid,
    best_metric='auc',
    feature_names=None
) -> Tuple[LassoClassifierWrapper, LassoClassifierWrapper, pd.DataFrame]:
    """
    Grid search over alpha values to find best LASSO model

    Args:
        X_train: Training features [N, D]
        y_train: Training labels [N]
        X_val: Validation features [M, D]
        y_val: Validation labels [M]
        alpha_grid: List of alpha values to try
        best_metric: 'auc' or 'loss' for model selection
        feature_names: List of feature names for logging

    Returns:
        Tuple of (best_model_trainauc, best_model_valauc, results_df)
        - best_model_trainauc: Model with best train AUC
        - best_model_valauc: Model with best validation AUC (or loss)
        - results_df: DataFrame with all alpha results
    """
    logger.info(f"\n=== Grid Search over {len(alpha_grid)} alpha values ===")
    logger.info(f"Alpha grid: {alpha_grid}")
    logger.info(f"Best metric: {best_metric}")

    results = []
    models = []

    for alpha in alpha_grid:
        logger.info(f"\nTrying alpha={alpha:.4f}...")

        # Train model
        model = LassoClassifierWrapper(alpha=alpha)
        model.fit(X_train, y_train)

        # Evaluate on train
        train_proba = model.predict_proba(X_train)
        train_metrics = compute_metrics(y_train, train_proba)

        # Evaluate on val
        val_proba = model.predict_proba(X_val)
        val_metrics = compute_metrics(y_val, val_proba)

        # Log active features
        if feature_names is not None:
            active_features = model.get_active_features(feature_names)
            n_active = len(active_features)
            logger.info(f"  Active features: {n_active}/{len(feature_names)}")
            if n_active <= 5:
                logger.info(f"    {active_features}")

        # Log metrics
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}")

        # Store results
        results.append({
            'alpha': alpha,
            'train_loss': train_metrics['loss'],
            'train_auc': train_metrics['auc'],
            'train_auprc': train_metrics['auprc'],
            'train_accuracy': train_metrics['accuracy'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'val_auprc': val_metrics['auprc'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'n_active_features': len(active_features) if feature_names else 0
        })
        models.append(model)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Select best models based on metric
    if best_metric == 'auc':
        # Independent tracking for train AUC and val AUC
        best_train_idx = results_df['train_auc'].idxmax()
        best_val_idx = results_df['val_auc'].idxmax()

        best_model_trainauc = models[best_train_idx]
        best_model_valauc = models[best_val_idx]

        logger.info(f"\n=== Best Models (AUC mode) ===")
        logger.info(f"Best Train AUC: {results_df.loc[best_train_idx, 'train_auc']:.4f} "
                   f"(alpha={results_df.loc[best_train_idx, 'alpha']:.4f})")
        logger.info(f"Best Val AUC: {results_df.loc[best_val_idx, 'val_auc']:.4f} "
                   f"(alpha={results_df.loc[best_val_idx, 'alpha']:.4f})")

    elif best_metric == 'loss':
        # Select based on validation loss
        best_val_idx = results_df['val_loss'].idxmin()
        best_model_valauc = models[best_val_idx]  # Use same variable name for consistency

        # For loss mode, train and val best models are the same
        best_model_trainauc = best_model_valauc

        logger.info(f"\n=== Best Model (Loss mode) ===")
        logger.info(f"Best Val Loss: {results_df.loc[best_val_idx, 'val_loss']:.4f} "
                   f"(alpha={results_df.loc[best_val_idx, 'alpha']:.4f})")
    else:
        raise ValueError(f"Invalid best_metric: {best_metric}. Must be 'auc' or 'loss'.")

    return best_model_trainauc, best_model_valauc, results_df


def cv_grid_search_with_external_val(
    X_train, y_train,
    X_val_ext, y_val_ext,
    alpha_grid,
    best_metric='auc',
    n_folds=5,
    feature_names=None,
    random_state=42
) -> Tuple[float, LassoClassifierWrapper, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Performs k-fold CV with external validation at each fold

    For each alpha:
        - Performs k-fold CV on training data
        - Each fold also evaluates on external validation set
        - Tracks both internal CV metrics and external validation metrics

    Selects best alpha based on mean internal CV performance
    Then selects median-performing fold at best alpha based on external validation

    Args:
        X_train: Training features [N, D]
        y_train: Training labels [N]
        X_val_ext: External validation features [M, D]
        y_val_ext: External validation labels [M]
        alpha_grid: List of alpha values to try
        best_metric: 'auc' or 'loss' for model selection
        n_folds: Number of CV folds (default: 5)
        feature_names: List of feature names for logging
        random_state: Random seed for reproducibility

    Returns:
        - best_alpha: Selected alpha value
        - median_model: Model from median-performing fold at best alpha
        - cv_results_detailed: DataFrame with all (alpha, fold) results
        - cv_results_summary: DataFrame with mean±std for each alpha
        - median_fold_info: Dict with median fold's metrics and info
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"5-FOLD CV GRID SEARCH WITH EXTERNAL VALIDATION")
    logger.info(f"{'='*80}")
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"External validation samples: {len(X_val_ext)}")
    logger.info(f"Alpha grid: {alpha_grid}")
    logger.info(f"Number of folds: {n_folds}")
    logger.info(f"Best metric: {best_metric}")

    # Setup stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Storage for all results
    detailed_results = []
    all_models_by_alpha = {}  # {alpha: [model_fold0, model_fold1, ...]}

    # For each alpha
    for alpha_idx, alpha in enumerate(alpha_grid):
        logger.info(f"\n{'='*80}")
        logger.info(f"Alpha {alpha:.4f} ({alpha_idx+1}/{len(alpha_grid)}):")
        logger.info(f"{'='*80}")

        fold_models = []
        fold_internal_metrics = []
        fold_external_metrics = []

        # For each fold
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            # Split data
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]

            # Train model on this fold
            model = LassoClassifierWrapper(alpha=alpha)
            model.fit(X_train_fold, y_train_fold)

            # Evaluate on internal validation (1/5 of training data)
            internal_val_proba = model.predict_proba(X_val_fold)
            internal_metrics = compute_metrics(y_val_fold, internal_val_proba)

            # Evaluate on external validation (entire validation CSV)
            external_val_proba = model.predict_proba(X_val_ext)
            external_metrics = compute_metrics(y_val_ext, external_val_proba)

            # Count active features
            n_active = 0
            if feature_names is not None:
                active_features = model.get_active_features(feature_names)
                n_active = len(active_features)

            # Log fold results
            logger.info(f"  Fold {fold_idx}: Internal AUC={internal_metrics['auc']:.3f}, "
                       f"External AUC={external_metrics['auc']:.3f}, Active={n_active}")

            # Store results
            detailed_results.append({
                'alpha': alpha,
                'fold': fold_idx,
                'internal_val_loss': internal_metrics['loss'],
                'internal_val_auc': internal_metrics['auc'],
                'internal_val_auprc': internal_metrics['auprc'],
                'internal_val_accuracy': internal_metrics['accuracy'],
                'internal_val_f1': internal_metrics['f1'],
                'external_val_loss': external_metrics['loss'],
                'external_val_auc': external_metrics['auc'],
                'external_val_auprc': external_metrics['auprc'],
                'external_val_accuracy': external_metrics['accuracy'],
                'external_val_f1': external_metrics['f1'],
                'n_active_features': n_active
            })

            fold_models.append(model)
            fold_internal_metrics.append(internal_metrics)
            fold_external_metrics.append(external_metrics)

        # Store models for this alpha
        all_models_by_alpha[alpha] = fold_models

        # Compute mean ± std across folds
        mean_internal_auc = np.mean([m['auc'] for m in fold_internal_metrics])
        std_internal_auc = np.std([m['auc'] for m in fold_internal_metrics])
        mean_internal_loss = np.mean([m['loss'] for m in fold_internal_metrics])
        std_internal_loss = np.std([m['loss'] for m in fold_internal_metrics])

        mean_external_auc = np.mean([m['auc'] for m in fold_external_metrics])
        std_external_auc = np.std([m['auc'] for m in fold_external_metrics])
        mean_external_loss = np.mean([m['loss'] for m in fold_external_metrics])
        std_external_loss = np.std([m['loss'] for m in fold_external_metrics])

        logger.info(f"  Mean: Internal AUC={mean_internal_auc:.3f}±{std_internal_auc:.3f}, "
                   f"External AUC={mean_external_auc:.3f}±{std_external_auc:.3f}")

    # Convert detailed results to DataFrame
    cv_results_detailed = pd.DataFrame(detailed_results)

    # Create summary DataFrame (mean ± std for each alpha)
    summary_data = []
    for alpha in alpha_grid:
        alpha_results = cv_results_detailed[cv_results_detailed['alpha'] == alpha]

        summary_data.append({
            'alpha': alpha,
            'mean_internal_val_loss': alpha_results['internal_val_loss'].mean(),
            'std_internal_val_loss': alpha_results['internal_val_loss'].std(),
            'mean_internal_val_auc': alpha_results['internal_val_auc'].mean(),
            'std_internal_val_auc': alpha_results['internal_val_auc'].std(),
            'mean_internal_val_f1': alpha_results['internal_val_f1'].mean(),
            'std_internal_val_f1': alpha_results['internal_val_f1'].std(),
            'mean_external_val_loss': alpha_results['external_val_loss'].mean(),
            'std_external_val_loss': alpha_results['external_val_loss'].std(),
            'mean_external_val_auc': alpha_results['external_val_auc'].mean(),
            'std_external_val_auc': alpha_results['external_val_auc'].std(),
            'mean_external_val_f1': alpha_results['external_val_f1'].mean(),
            'std_external_val_f1': alpha_results['external_val_f1'].std(),
            'mean_n_active_features': alpha_results['n_active_features'].mean(),
        })

    cv_results_summary = pd.DataFrame(summary_data)

    # Select best alpha based on mean internal CV performance
    logger.info(f"\n{'='*80}")
    logger.info(f"SELECTING BEST ALPHA")
    logger.info(f"{'='*80}")

    if best_metric == 'auc':
        best_alpha_idx = cv_results_summary['mean_internal_val_auc'].idxmax()
        best_alpha = cv_results_summary.loc[best_alpha_idx, 'alpha']
        best_mean_auc = cv_results_summary.loc[best_alpha_idx, 'mean_internal_val_auc']
        logger.info(f"Best alpha: {best_alpha:.4f} (mean internal AUC={best_mean_auc:.3f})")
    elif best_metric == 'loss':
        best_alpha_idx = cv_results_summary['mean_internal_val_loss'].idxmin()
        best_alpha = cv_results_summary.loc[best_alpha_idx, 'alpha']
        best_mean_loss = cv_results_summary.loc[best_alpha_idx, 'mean_internal_val_loss']
        logger.info(f"Best alpha: {best_alpha:.4f} (mean internal loss={best_mean_loss:.3f})")
    else:
        raise ValueError(f"Invalid best_metric: {best_metric}. Must be 'auc' or 'loss'.")

    # Get models at best alpha
    fold_models_at_best = all_models_by_alpha[best_alpha]

    # Get external validation metrics for each fold at best alpha
    best_alpha_results = cv_results_detailed[cv_results_detailed['alpha'] == best_alpha]

    # Select median fold based on external validation performance
    logger.info(f"\n{'='*80}")
    logger.info(f"SELECTING MEDIAN MODEL AT BEST ALPHA")
    logger.info(f"{'='*80}")

    if best_metric == 'auc':
        # Sort by external AUC
        sorted_indices = best_alpha_results.sort_values('external_val_auc')['fold'].values
        metric_values = best_alpha_results.sort_values('external_val_auc')['external_val_auc'].values
        metric_name = 'external AUC'
    else:
        # Sort by external loss (ascending for loss)
        sorted_indices = best_alpha_results.sort_values('external_val_loss')['fold'].values
        metric_values = best_alpha_results.sort_values('external_val_loss')['external_val_loss'].values
        metric_name = 'external loss'

    # Median is at position n_folds // 2 (e.g., index 2 for 5 folds)
    median_position = n_folds // 2
    median_fold_idx = sorted_indices[median_position]
    median_model = fold_models_at_best[median_fold_idx]

    logger.info(f"Fold performances at alpha={best_alpha:.4f} (sorted by {metric_name}):")
    for i, (fold_idx, metric_val) in enumerate(zip(sorted_indices, metric_values)):
        marker = " ← MEDIAN (selected)" if i == median_position else ""
        logger.info(f"  Fold {fold_idx}: {metric_name}={metric_val:.3f}{marker}")

    # Get median fold info
    median_fold_info = best_alpha_results[best_alpha_results['fold'] == median_fold_idx].iloc[0].to_dict()
    median_fold_info['best_alpha'] = best_alpha
    median_fold_info['median_fold_idx'] = median_fold_idx

    logger.info(f"\nMedian model (Fold {median_fold_idx}) metrics:")
    logger.info(f"  Internal validation: AUC={median_fold_info['internal_val_auc']:.3f}, "
               f"Loss={median_fold_info['internal_val_loss']:.3f}, F1={median_fold_info['internal_val_f1']:.3f}")
    logger.info(f"  External validation: AUC={median_fold_info['external_val_auc']:.3f}, "
               f"Loss={median_fold_info['external_val_loss']:.3f}, F1={median_fold_info['external_val_f1']:.3f}")
    logger.info(f"  Active features: {median_fold_info['n_active_features']:.0f}/{len(feature_names) if feature_names else '?'}")

    return best_alpha, median_model, cv_results_detailed, cv_results_summary, median_fold_info


if __name__ == "__main__":
    # Test the LASSO wrapper
    print("=== Testing LassoClassifierWrapper ===")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 16

    # Create features with some correlation
    X = np.random.randn(n_samples, n_features)

    # Create labels based on a few features
    y = (0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.4 * X[:, 2] + np.random.randn(n_samples) * 0.1) > 0
    y = y.astype(float)

    # Split into train/val
    train_size = 800
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Test single model
    print("\n=== Testing Single Model ===")
    model = LassoClassifierWrapper(alpha=0.1)
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)
    val_proba = model.predict_proba(X_val)

    train_metrics = compute_metrics(y_train, train_proba)
    val_metrics = compute_metrics(y_val, val_proba)

    print(f"Train metrics: {train_metrics}")
    print(f"Val metrics: {val_metrics}")

    # Test grid search
    print("\n=== Testing Grid Search ===")
    alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    feature_names = [f"feature_{i}" for i in range(n_features)]

    best_model_trainauc, best_model_valauc, results_df = grid_search_lasso(
        X_train, y_train, X_val, y_val,
        alpha_grid,
        best_metric='auc',
        feature_names=feature_names
    )

    print(f"\nGrid search results:")
    print(results_df[['alpha', 'train_auc', 'val_auc', 'n_active_features']])

    # Test model save/load
    print("\n=== Testing Save/Load ===")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pkl")
        best_model_valauc.save(model_path)
        loaded_model = LassoClassifierWrapper.load(model_path)

        # Verify predictions match
        orig_proba = best_model_valauc.predict_proba(X_val)
        loaded_proba = loaded_model.predict_proba(X_val)
        assert np.allclose(orig_proba, loaded_proba), "Loaded model predictions don't match!"
        print("✓ Save/load test passed!")

    print("\n✓ All tests completed successfully!")
