"""Base classes for feature selection algorithms.

This module provides abstract base classes and common functionality for all
feature selection algorithms, eliminating code duplication.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from mealpy import FloatVar

from core.constants import (
    BINARY_THRESHOLD,
    FITNESS_PENALTY_DEFAULT,
    MAX_PLS_COMPONENTS,
    INTERNAL_VAL_SIZE,
    DEFAULT_RANDOM_STATE
)
from utils.data_split import regression_stratified_split


@dataclass
class SelectionResult:
    """Result of feature selection."""
    selected_features: List[str]
    selected_indices: List[int]
    mode: str = "selection"  # "selection" or "extraction"


class BaseFeatureSelector(ABC):
    """Abstract base class for all feature selection algorithms.

    Provides common functionality:
    - Data loading with band range support
    - Result saving with target column preservation
    - Logging integration

    Subclasses must implement:
    - run_selection(): Core algorithm logic
    """

    mode = "selection"  # Override to "extraction" for PCA-like algorithms

    def __init__(self, target_col: str, band_range: Tuple[int, int], logger: Optional[logging.Logger] = None):
        """Initialize selector.

        Args:
            target_col: Target variable column name
            band_range: (start, end) band indices
            logger: Optional logger instance
        """
        self.target_col = target_col
        self.band_range = band_range
        self.feat_cols = [f'b{i}' for i in range(band_range[0], band_range[1])]
        self.logger = logger or logging.getLogger(__name__)

    def load_data(self, input_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and validate data.

        Args:
            input_path: Path to input CSV file

        Returns:
            (dataframe, X_raw, y) tuple

        Raises:
            KeyError: If required columns are missing
        """
        df = pd.read_csv(input_path)

        try:
            X_raw = df[self.feat_cols].values
            y = df[self.target_col].values
        except KeyError as e:
            raise KeyError(f"Missing required columns: {e}")

        return df, X_raw, y

    def save_selection_result(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        selected_features: List[str],
        output_path: str
    ):
        """Save selected features to CSV.

        Args:
            df: Original dataframe
            y: Target values
            selected_features: List of selected feature column names
            output_path: Output file path
        """
        df_out = df[selected_features].copy()
        df_out[self.target_col] = y
        df_out.to_csv(output_path, index=False)
        self.logger.info(f"Results saved to: {output_path}")

    @abstractmethod
    def run_selection(self, input_path: str, output_path: str, **kwargs) -> SelectionResult:
        """Run feature selection algorithm.

        Args:
            input_path: Path to input data
            output_path: Path to save results
            **kwargs: Algorithm-specific parameters

        Returns:
            SelectionResult with selected features and indices
        """
        pass


class BaseMealpySelector(BaseFeatureSelector):
    """Base class for metaheuristic algorithms using mealpy.

    Eliminates duplication across HHO, GA, GWO, MPA by providing:
    - Common fitness function (1-R² + penalty*ratio)
    - Data preprocessing (scaling, stratified split)
    - Mealpy problem setup and solving

    Subclasses only need to implement:
    - create_optimizer(): Return configured mealpy optimizer instance
    - Set default_* class attributes for parameters
    """

    # Default parameters (override in subclasses)
    default_epoch = 200
    default_pop_size = 100
    default_penalty = 0.2

    def __init__(self, target_col: str, band_range: Tuple[int, int], logger: Optional[logging.Logger] = None, **kwargs):
        """Initialize metaheuristic selector.

        Args:
            target_col: Target variable column name
            band_range: (start, end) band indices
            logger: Optional logger instance
            **kwargs: Algorithm parameters (epoch, pop_size, penalty, etc.)
        """
        super().__init__(target_col, band_range, logger)

        # Merge kwargs with defaults
        self.epoch = kwargs.get('epoch', self.default_epoch)
        self.pop_size = kwargs.get('pop_size', self.default_pop_size)
        self.penalty = kwargs.get('penalty', self.default_penalty)

        # Store extra kwargs for subclass-specific parameters (e.g., GA's pc/pm)
        self.extra_params = {k: v for k, v in kwargs.items()
                            if k not in ['epoch', 'pop_size', 'penalty']}

    @abstractmethod
    def create_optimizer(self):
        """Create and return mealpy optimizer instance.

        Returns:
            Configured mealpy optimizer (e.g., OriginalHHO, BaseGA)
        """
        pass

    def run_selection(self, input_path: str, output_path: str, **kwargs) -> SelectionResult:
        """Run metaheuristic feature selection.

        Args:
            input_path: Path to input data
            output_path: Path to save results
            **kwargs: Additional runtime parameters (ignored, use constructor params)

        Returns:
            SelectionResult with selected features
        """
        self.logger.info(f"Loading data from: {input_path}")
        self.logger.info(f"Parameters: Epoch={self.epoch}, Pop={self.pop_size}, Penalty={self.penalty}")

        # Load and preprocess data
        df, X_raw, y = self.load_data(input_path)

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)

        # Stratified split for internal validation
        X_train, X_val, y_train, y_val = regression_stratified_split(
            X, y,
            test_size=INTERNAL_VAL_SIZE,
            n_bins=5,
            random_state=DEFAULT_RANDOM_STATE
        )

        # Define unified fitness function
        def fitness_function(solution):
            sel_idx = np.where(solution > BINARY_THRESHOLD)[0]

            if len(sel_idx) == 0:
                return FITNESS_PENALTY_DEFAULT

            try:
                n_comp = min(len(sel_idx), MAX_PLS_COMPONENTS)
                model = PLSRegression(n_components=n_comp)
                model.fit(X_train[:, sel_idx], y_train)
                y_pred = model.predict(X_val[:, sel_idx])

                if y_pred.ndim > 1:
                    y_pred = y_pred.flatten()

                r2 = r2_score(y_val, y_pred)
                ratio = len(sel_idx) / X.shape[1]

                # Unified fitness: (1 - R²) + penalty * ratio
                fitness = (1 - r2) + (self.penalty * ratio)
                return fitness
            except Exception:
                return FITNESS_PENALTY_DEFAULT

        # Create optimizer (subclass-specific)
        optimizer = self.create_optimizer()

        # Setup problem
        problem_dict = {
            "obj_func": fitness_function,
            "bounds": FloatVar(lb=[0] * X.shape[1], ub=[1] * X.shape[1]),
            "minmax": "min",
            "log_to": None
        }

        # Solve
        self.logger.info("Starting optimization...")
        agent = optimizer.solve(problem_dict)

        # Extract results
        best_pos = agent.solution
        best_idx = np.where(best_pos > BINARY_THRESHOLD)[0]
        best_feats = [self.feat_cols[i] for i in best_idx]

        fit_val = getattr(agent.target, 'fitness', getattr(agent, 'fitness', 'N/A'))
        self.logger.info(f"Best fitness: {fit_val}")
        self.logger.info(f"Selected features: {len(best_feats)}")

        # Save results
        self.save_selection_result(df, y, best_feats, output_path)

        return SelectionResult(
            selected_features=best_feats,
            selected_indices=best_idx.tolist()
        )
