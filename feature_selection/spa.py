"""SPA (Successive Projections Algorithm) feature selection.

Preserves the three-phase SPA logic (Projection, Validation, F-test) while
adapting to the unified interface. Default parameters are defined as class attributes.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import f
from joblib import Parallel, delayed

from core.registry import register_algorithm
from feature_selection.base import BaseFeatureSelector, SelectionResult


class SPA:
    """SPA algorithm with three phases: Projection, Validation, F-test."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def _projection_fast(self, X, k, M):
        """Vectorized projection algorithm (Phase 1)."""
        X_projected = X.copy()
        n_features = X.shape[1]
        selected_indices = [k]

        for _ in range(1, M):
            v = X_projected[:, selected_indices[-1]].copy()
            norm_v_sq = np.dot(v, v)

            if norm_v_sq < 1e-10:
                break

            projection_factors = np.dot(v, X_projected) / norm_v_sq
            X_projected -= np.outer(v, projection_factors)

            norms = np.sum(np.abs(X_projected ** 2), axis=0)
            norms[selected_indices] = -1

            max_norm_idx = np.argmax(norms)
            selected_indices.append(max_norm_idx)

        return selected_indices

    def _validation(self, Xcal, ycal, var_sel, Xval=None, yval=None):
        """Validation function (Phase 2)."""
        N = Xcal.shape[0]
        var_sel = np.atleast_1d(var_sel)

        if Xval is not None:
            # Independent validation set
            Xcal_sel = Xcal[:, var_sel]
            Xval_sel = Xval[:, var_sel]

            Xcal_ones = np.hstack([np.ones((N, 1)), Xcal_sel])
            Xval_ones = np.hstack([np.ones((Xval.shape[0], 1)), Xval_sel])

            b = np.linalg.lstsq(Xcal_ones, ycal, rcond=None)[0]
            yhat = Xval_ones.dot(b)
            e = yval - yhat
        else:
            # LOO-CV
            yhat = np.zeros_like(ycal)
            X_full = np.hstack([np.ones((N, 1)), Xcal[:, var_sel]])

            for i in range(N):
                X_train = np.delete(X_full, i, axis=0)
                y_train = np.delete(ycal, i, axis=0)
                X_test = X_full[i:i + 1, :]

                b = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                yhat[i] = X_test.dot(b).item()

            e = ycal - yhat

        return yhat, e

    def _calculate_chain_error(self, k, Xcal, ycal, Xval, yval, m_min, m_max, normalization_factor):
        """Parallel subtask: calculate error for chain starting at k."""
        Xcaln = (Xcal - np.mean(Xcal, axis=0)) / normalization_factor

        sel_indices = self._projection_fast(Xcaln, k, m_max)
        sel_indices = np.array(sel_indices, dtype=int)

        press_k = []
        for m in range(m_min, m_max + 1):
            var_sel = sel_indices[:m]
            _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)
            press_val = np.dot(e.T, e).item()
            press_k.append(press_val)

        return press_k, sel_indices

    def spa(self, Xcal, ycal, m_min=1, m_max=None, Xval=None, yval=None, autoscaling=1, n_jobs=-1):
        """Main SPA function."""
        Xcal = np.array(Xcal)
        ycal = np.array(ycal).reshape(-1)
        if Xval is not None:
            Xval = np.array(Xval)
            yval = np.array(yval).reshape(-1)

        N, K = Xcal.shape

        if m_max is None:
            m_max = min(N - 1, K) if Xval is None else min(N - 2, K)

        # Normalization factor
        if autoscaling == 1:
            normalization_factor = np.std(Xcal, ddof=1, axis=0)
            normalization_factor[normalization_factor == 0] = 1.0
        else:
            normalization_factor = np.ones(K)

        # Parallel execution
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._calculate_chain_error)(
                k, Xcal, ycal, Xval, yval, m_min, m_max, normalization_factor
            ) for k in range(K)
        )

        # Parse results
        PRESS = np.zeros((m_max - m_min + 1, K))
        SEL = []

        for k, (press_vals, sel_inds) in enumerate(results):
            PRESS[:, k] = press_vals
            SEL.append(sel_inds)

        # Find global minimum PRESS
        min_press_idx_flat = np.argmin(PRESS)
        m_idx, k_sel = np.unravel_index(min_press_idx_flat, PRESS.shape)

        m_sel_count = m_idx + m_min
        var_sel_phase2 = SEL[k_sel][:m_sel_count]

        # Phase 3: F-test
        Xcal2 = np.hstack([np.ones((N, 1)), Xcal[:, var_sel_phase2]])
        b = np.linalg.lstsq(Xcal2, ycal, rcond=None)[0]
        std_deviation = np.std(Xcal2, ddof=1, axis=0)

        relev = np.abs(b * std_deviation.T)
        relev = relev[1:]  # Remove intercept

        index_decreasing_relev = np.argsort(relev)[::-1]
        sorted_indices = var_sel_phase2[index_decreasing_relev]

        PRESS_scree = np.zeros(len(sorted_indices))
        for i in range(len(sorted_indices)):
            var_sel = sorted_indices[:i + 1]
            _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)
            PRESS_scree[i] = np.dot(e.T, e).item()

        RMSEP_scree = np.sqrt(PRESS_scree / N)

        # F-test
        PRESS_min = np.min(PRESS_scree)
        dof = N - len(sorted_indices) - 1
        if dof <= 0:
            dof = 1

        alpha = 0.25
        fcrit = f.ppf(1 - alpha, dof, dof)
        PRESS_crit = PRESS_min * fcrit

        candidates = np.where(PRESS_scree <= PRESS_crit)[0]
        idx_crit = np.min(candidates) if len(candidates) > 0 else np.argmin(PRESS_scree)

        final_num_vars = max(m_min, idx_crit + 1)
        final_selected_vars = sorted_indices[:final_num_vars]

        return final_selected_vars, var_sel_phase2


@register_algorithm("SPA")
class SPASelector(BaseFeatureSelector):
    """SPA feature selection adapter."""

    default_m_min = 2
    default_m_max = 30

    def run_selection(self, input_path: str, output_path: str, **kwargs) -> SelectionResult:
        m_min = kwargs.get('m_min', self.default_m_min)
        m_max = kwargs.get('m_max', self.default_m_max)

        self.logger.info(f"[SPA] Loading data: {input_path}")
        self.logger.info(f"[SPA] Config: m_min={m_min}, m_max={m_max}, n_jobs=-1")

        df, X_raw, y = self.load_data(input_path)

        X = X_raw  # SPA doesn't require scaling

        safe_max = min(m_max, X.shape[0] - 2)

        spa_engine = SPA(logger=self.logger)
        selected_indices_res, _ = spa_engine.spa(
            Xcal=X,
            ycal=y,
            m_min=m_min,
            m_max=safe_max,
            autoscaling=1,
            n_jobs=-1
        )

        # Sort indices
        selected_indices_res = sorted(selected_indices_res)

        best_feats = [self.feat_cols[i] for i in selected_indices_res]
        self.logger.info(f"[SPA] Selected features: {len(best_feats)}")

        self.save_selection_result(df, y, best_feats, output_path)

        return SelectionResult(
            selected_features=best_feats,
            selected_indices=selected_indices_res
        )
