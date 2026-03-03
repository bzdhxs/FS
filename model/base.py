"""Base class for regression models.

Provides common functionality for data loading, metric calculation,
result saving, and visualization, eliminating duplication across models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from visualizer import model_visualizer


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> Dict[str, Any]:
    """Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        label: Dataset label ("Train" or "Test")

    Returns:
        Dict with Set, R2, RMSE, RPD
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = (np.std(y_true) / rmse) if rmse > 1e-6 else 0

    return {"Set": label, "R2": r2, "RMSE": rmse, "RPD": rpd}


class BaseModel(ABC):
    """Abstract base class for all regression models.

    Provides the complete modeling pipeline:
    1. Load data
    2. Train and predict (subclass-specific)
    3. Auto-inverse transform (if Box-Cox was applied)
    4. Calculate metrics
    5. Save results
    6. Visualize

    Subclasses only need to implement:
    - train_and_predict(X_train, y_train, X_test) -> (pred_train, pred_test, best_params)
    """

    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs):
        """Initialize model.

        Args:
            logger: Optional logger instance
            **kwargs: Model-specific parameters (override defaults)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.extra_params = kwargs

    @abstractmethod
    def train_and_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Train model and generate predictions.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features

        Returns:
            (pred_train, pred_test, best_params) tuple
        """
        pass

    def run_modeling(
        self,
        train_path: str,
        test_path: str,
        selected_features: List[str],
        target_col: str,
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the complete modeling pipeline.

        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            selected_features: List of feature column names
            target_col: Target variable column name
            output_dir: Directory for output files
            **kwargs: Additional parameters

        Returns:
            Dict with train_metrics, test_metrics, best_params, n_features
        """
        self.logger.info("Starting modeling pipeline...")

        # 1. Load data
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        if not selected_features:
            feature_names = df_train.drop(
                columns=[target_col, 'Sample_ID'], errors='ignore'
            ).columns.tolist()
            self.logger.warning("No feature list provided, using all features")
        else:
            feature_names = selected_features

        self.logger.info(f"Features: {len(feature_names)}")

        X_train = df_train[feature_names].values
        X_test = df_test[feature_names].values
        y_train = df_train[target_col].values
        y_test = df_test[target_col].values

        # 2. Train and predict (subclass-specific)
        pred_train_trans, pred_test_trans, best_params = self.train_and_predict(
            X_train, y_train, X_test
        )

        self.logger.info(f"Best params: {best_params}")

        # 3. Calculate metrics
        metrics_train = calc_metrics(y_train, pred_train_trans, "Train")
        metrics_test = calc_metrics(y_test, pred_test_trans, "Test")

        self.logger.info(
            f"Results -> Train R2: {metrics_train['R2']:.4f} | "
            f"Test R2: {metrics_test['R2']:.4f} | Test RMSE: {metrics_test['RMSE']:.4f}"
        )

        # 4. Save results to results/ subdirectory
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        pd.DataFrame({
            "Measured": y_test,
            "Predicted": pred_test_trans
        }).to_csv(os.path.join(results_dir, "predictions.csv"), index=False)

        metrics_df = pd.DataFrame([metrics_train, metrics_test])
        metrics_df['Best_Params'] = str(best_params)
        metrics_df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

        # 5. Visualize to plots/ subdirectory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        model_visualizer.plot_regression_results(
            y_train, pred_train_trans,
            y_test, pred_test_trans,
            metrics_train, metrics_test,
            os.path.join(plots_dir, "prediction_scatter.png")
        )

        return {
            "train_metrics": metrics_train,
            "test_metrics": metrics_test,
            "best_params": best_params,
            "n_features": len(feature_names)
        }
