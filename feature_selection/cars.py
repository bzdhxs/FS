"""CARS (Competitive Adaptive Reweighted Sampling) feature selection.

Preserves the StrictCARS core logic while adapting to the unified interface.
Default parameters are defined as class attributes.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from core.registry import register_algorithm
from feature_selection.base import BaseFeatureSelector, SelectionResult


class StrictCARS:
    """CARS algorithm based on Li et al. (2009).

    Includes Strict ARS (roulette wheel sampling) with engineering robustness
    for small samples.
    """

    def __init__(self, n_iter=200, k_fold=5, n_components=5, logger=None):
        self.n_iter = n_iter
        self.k_fold = k_fold
        self.n_components = n_components
        self.logger = logger or logging.getLogger(__name__)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        RMSECV_list = []
        feat_subset_list = []

        # EDF (exponential decay function) parameters
        a = (n_features / 2) ** (1 / (self.n_iter - 1))
        k = np.log(n_features / 2) / (self.n_iter - 1)

        current_vars = np.arange(n_features)

        self.logger.info(f"[CARS] Starting | Iterations: {self.n_iter} | Init Feats: {n_features}")

        for i in range(self.n_iter):
            # 1. MCS (Monte Carlo Sampling)
            mcs_size = int(0.8 * n_samples)
            rand_idx = np.random.choice(n_samples, size=mcs_size, replace=True)

            X_mcs = X[rand_idx][:, current_vars]
            y_mcs = y[rand_idx]

            # 2. PLS modeling for weights
            curr_n_comp = min(self.n_components, len(current_vars), mcs_size - 1)
            if curr_n_comp < 1:
                curr_n_comp = 1

            model = PLSRegression(n_components=curr_n_comp)
            model.fit(X_mcs, y_mcs)

            coefs = np.abs(model.coef_).flatten()

            coef_sum = np.sum(coefs)
            if coef_sum == 0 or np.isnan(coef_sum):
                weights = np.ones(len(coefs)) / len(coefs)
            else:
                weights = coefs / coef_sum

            # 3. EDF: compute number of features to keep
            ratio = a * np.exp(-k * i)
            n_keep = int(np.round(n_features * ratio))
            n_keep = max(2, min(n_keep, len(current_vars)))

            # 4. Strict ARS (Adaptive Reweighted Sampling)
            try:
                selected_local_idx = np.random.choice(
                    np.arange(len(current_vars)),
                    size=n_keep,
                    replace=False,
                    p=weights
                )
            except ValueError:
                # Fallback to Top-K if ARS fails
                sorted_idx = np.argsort(-weights)
                selected_local_idx = sorted_idx[:n_keep]

            current_vars = current_vars[selected_local_idx]

            # 5. RMSECV via K-Fold
            X_curr = X[:, current_vars]
            kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
            cv_errors = []

            for train_idx, val_idx in kf.split(X_curr):
                cv_comp = min(self.n_components, X_curr.shape[1], len(train_idx) - 1)
                if cv_comp < 1:
                    cv_comp = 1

                model_cv = PLSRegression(n_components=cv_comp)
                model_cv.fit(X_curr[train_idx], y[train_idx])
                y_pred = model_cv.predict(X_curr[val_idx]).flatten()
                cv_errors.append(mean_squared_error(y[val_idx], y_pred))

            rmse_cv = np.sqrt(np.mean(cv_errors))

            RMSECV_list.append(rmse_cv)
            feat_subset_list.append(current_vars.copy())

            if (i + 1) % 10 == 0:
                self.logger.info(
                    f"  Iter {i + 1}/{self.n_iter} | Feats: {len(current_vars):<3} | RMSECV: {rmse_cv:.4f}"
                )

        # 6. Select best subset
        best_iter_idx = np.argmin(RMSECV_list)
        best_feats_idx = feat_subset_list[best_iter_idx]
        best_rmse = RMSECV_list[best_iter_idx]

        self.logger.info(
            f"[CARS] Best iter: {best_iter_idx + 1} | Features: {len(best_feats_idx)} | Min RMSECV: {best_rmse:.4f}"
        )

        return best_feats_idx


@register_algorithm("CARS")
class CARSSelector(BaseFeatureSelector):
    """CARS feature selection adapter."""

    default_n_iter = 200
    default_k_fold = 5
    default_n_components = 5

    def run_selection(self, input_path: str, output_path: str, **kwargs) -> SelectionResult:
        n_iter = kwargs.get('n_iter', self.default_n_iter)
        k_fold = kwargs.get('k_fold', self.default_k_fold)
        n_components = kwargs.get('n_components', self.default_n_components)

        self.logger.info(f"[CARS] Loading data: {input_path}")
        self.logger.info(f"[CARS] Config: Iter={n_iter}, K-Fold={k_fold}, Max_PC={n_components}")

        df, X_raw, y = self.load_data(input_path)

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)

        selector = StrictCARS(n_iter=n_iter, k_fold=k_fold, n_components=n_components, logger=self.logger)
        best_idx = selector.fit(X, y)

        best_feats = [self.feat_cols[i] for i in best_idx]
        self.logger.info(f"[CARS] Selected features: {len(best_feats)}")

        self.save_selection_result(df, y, best_feats, output_path)

        return SelectionResult(
            selected_features=best_feats,
            selected_indices=best_idx.tolist()
        )
