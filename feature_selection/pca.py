"""PCA (Principal Component Analysis) feature extraction.

Unlike other algorithms, PCA performs feature extraction (not selection),
generating new features (PC1, PC2, ...) from the original bands.

Default parameters are defined as class attributes.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from core.registry import register_algorithm
from feature_selection.base import BaseFeatureSelector, SelectionResult


@register_algorithm("PCA")
class PCASelector(BaseFeatureSelector):
    """PCA feature extraction adapter.

    Note: PCA is feature extraction, not selection. It generates new features
    (principal components) rather than selecting a subset of original features.
    """

    mode = "extraction"  # Override base class mode

    default_n_components = 3

    def run_selection(self, input_path: str, output_path: str, **kwargs) -> SelectionResult:
        """Run PCA feature extraction.

        Note: For PCA, input_path and output_path are expected to be train/test paths.
        This method handles both train and test data simultaneously.

        Args:
            input_path: Path to training data (or single file if test not provided)
            output_path: Path to save transformed training data
            **kwargs: Must include 'test_path' and 'test_output_path' for proper PCA

        Returns:
            SelectionResult with PC feature names
        """
        n_components = kwargs.get('n_components', self.default_n_components)
        test_path = kwargs.get('test_path')
        test_output_path = kwargs.get('test_output_path')

        if not test_path or not test_output_path:
            raise ValueError("PCA requires 'test_path' and 'test_output_path' in kwargs")

        self.logger.info(f"[PCA] Starting extraction | Components: {n_components}")
        self.logger.info(f"[PCA] Train: {Path(input_path).name}")
        self.logger.info(f"[PCA] Test: {Path(test_path).name}")

        # Load train and test data
        df_train, X_train_raw, y_train = self.load_data(input_path)
        df_test, X_test_raw, y_test = self.load_data(test_path)

        # Standardization (fit on train, transform both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # PCA (fit on train, transform both)
        pca = PCA(n_components=n_components)
        X_train_new = pca.fit_transform(X_train_scaled)
        X_test_new = pca.transform(X_test_scaled)

        n_pcs = X_train_new.shape[1]
        explained_variance = np.sum(pca.explained_variance_ratio_)
        self.logger.info(
            f"[PCA] Generated {n_pcs} features | Explained variance: {explained_variance * 100:.2f}%"
        )

        # Create PC column names
        pc_cols = [f'PC{i + 1}' for i in range(n_pcs)]

        # Save train data
        df_out_train = pd.DataFrame(X_train_new, columns=pc_cols)
        df_out_train[self.target_col] = y_train
        df_out_train.to_csv(output_path, index=False)

        # Save test data
        df_out_test = pd.DataFrame(X_test_new, columns=pc_cols)
        df_out_test[self.target_col] = y_test
        df_out_test.to_csv(test_output_path, index=False)

        self.logger.info(f"[PCA] Results saved")

        return SelectionResult(
            selected_features=pc_cols,
            selected_indices=list(range(n_pcs)),
            mode="extraction"
        )
