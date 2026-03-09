"""SG-HHO (Spectral Group-based Harris Hawks Optimization) feature selection.

This module provides a feature selection wrapper for the SG-HHO algorithm,
implementing stability-driven fitness function with cross-validation variance.

Default parameters:
- epoch: 200
- pop_size: 50
- window_size: 8
- alpha_stability: 0.2 (weight for CV standard deviation)
- beta_sparsity: 0.1 (weight for group sparsity)
"""

import numpy as np
from typing import Optional, Tuple
import logging
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

from core.registry import register_algorithm
from core.constants import (
    FITNESS_PENALTY_DEFAULT,
    MAX_PLS_COMPONENTS,
    INTERNAL_VAL_SIZE,
    DEFAULT_RANDOM_STATE
)
from feature_selection.base import BaseFeatureSelector, SelectionResult
from utils.data_split import regression_stratified_split
from improve.SGHHO import SpectralGroupHHO
from mealpy import FloatVar


@register_algorithm("SGHHO")
class SGHHOSelector(BaseFeatureSelector):
    """
    SG-HHO feature selector with stability-driven fitness.
    
    This selector implements the complete SG-HHO algorithm including:
    1. Spectral group-based search space
    2. Group-level HHO optimization
    3. Stability-driven multi-objective fitness
    4. Cross-validation variance minimization
    
    Parameters
    ----------
    target_col : str
        Target variable column name
    band_range : tuple
        (start, end) band indices
    logger : logging.Logger, optional
        Logger instance
    epoch : int, default=200
        Maximum number of iterations
    pop_size : int, default=50
        Population size
    window_size : int, default=8
        Spectral group window size
    alpha_stability : float, default=0.2
        Weight for CV standard deviation in fitness
    beta_sparsity : float, default=0.1
        Weight for group sparsity ratio in fitness
    n_cv_runs : int, default=5
        Number of cross-validation runs for stability assessment
    """
    
    # Default parameters
    default_epoch = 200
    default_pop_size = 50
    default_window_size = 8
    default_alpha_stability = 0.2
    default_beta_sparsity = 0.1
    default_n_cv_runs = 5
    
    def __init__(self, target_col: str, band_range: Tuple[int, int], 
                 logger: Optional[logging.Logger] = None, **kwargs):
        """Initialize SG-HHO selector."""
        super().__init__(target_col, band_range, logger)
        
        # Extract parameters with defaults
        self.epoch = kwargs.get('epoch', self.default_epoch)
        self.pop_size = kwargs.get('pop_size', self.default_pop_size)
        self.window_size = kwargs.get('window_size', self.default_window_size)
        self.alpha_stability = kwargs.get('alpha_stability', self.default_alpha_stability)
        self.beta_sparsity = kwargs.get('beta_sparsity', self.default_beta_sparsity)
        self.n_cv_runs = kwargs.get('n_cv_runs', self.default_n_cv_runs)
        
        # Store for fitness function
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
    
    def _stability_driven_fitness(self, solution):
        """
        Stability-driven fitness function.
        
        F = RMSE_mean + α * RMSE_std + β * GroupRatio
        
        This fitness function explicitly models:
        1. Prediction accuracy (RMSE_mean)
        2. Model stability (RMSE_std across multiple CV runs)
        3. Feature sparsity (GroupRatio)
        
        Parameters
        ----------
        solution : ndarray
            Band-level binary solution vector
            
        Returns
        -------
        float
            Fitness value (lower is better)
        """
        # Get selected band indices
        sel_idx = np.where(solution > 0.5)[0]
        
        if len(sel_idx) == 0:
            return FITNESS_PENALTY_DEFAULT
        
        try:
            # Multiple CV runs for stability assessment
            rmse_list = []
            
            for seed in range(self.n_cv_runs):
                # 5-fold cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                
                for train_idx, val_idx in kf.split(self.X_train):
                    X_tr = self.X_train[train_idx][:, sel_idx]
                    y_tr = self.y_train[train_idx]
                    X_vl = self.X_train[val_idx][:, sel_idx]
                    y_vl = self.y_train[val_idx]
                    
                    # Fit PLS model
                    n_comp = min(len(sel_idx), MAX_PLS_COMPONENTS)
                    model = PLSRegression(n_components=n_comp)
                    model.fit(X_tr, y_tr)
                    
                    # Predict
                    y_pred = model.predict(X_vl)
                    if y_pred.ndim > 1:
                        y_pred = y_pred.flatten()
                    
                    # Calculate RMSE
                    rmse = np.sqrt(mean_squared_error(y_vl, y_pred))
                    rmse_list.append(rmse)
            
            # Calculate stability metrics
            rmse_mean = np.mean(rmse_list)
            rmse_std = np.std(rmse_list)
            
            # Calculate group sparsity ratio (precise calculation)
            # Count actual number of unique groups selected
            selected_groups = set()
            for idx in sel_idx:
                group_id = idx // self.window_size
                selected_groups.add(group_id)
            
            n_bands = len(solution)
            n_total_groups = (n_bands + self.window_size - 1) // self.window_size
            group_ratio = len(selected_groups) / n_total_groups
            
            # Combined fitness (fixed parameters)
            fitness = (rmse_mean + 
                      self.alpha_stability * rmse_std + 
                      self.beta_sparsity * group_ratio)
            
            return fitness
            
        except Exception as e:
            self.logger.warning(f"Fitness evaluation error: {e}")
            return FITNESS_PENALTY_DEFAULT
    
    def run_selection(self, input_path: str, output_path: str, **kwargs) -> SelectionResult:
        """
        Run SG-HHO feature selection.
        
        Parameters
        ----------
        input_path : str
            Path to input data CSV
        output_path : str
            Path to save selected features
        **kwargs : dict
            Additional runtime parameters (ignored, use constructor params)
            
        Returns
        -------
        SelectionResult
            Result containing selected features and indices
        """
        self.logger.info(f"Loading data from: {input_path}")
        self.logger.info(f"SG-HHO Parameters:")
        self.logger.info(f"  Epoch: {self.epoch}")
        self.logger.info(f"  Population: {self.pop_size}")
        self.logger.info(f"  Window Size: {self.window_size}")
        self.logger.info(f"  Alpha (Stability): {self.alpha_stability}")
        self.logger.info(f"  Beta (Sparsity): {self.beta_sparsity}")
        self.logger.info(f"  CV Runs: {self.n_cv_runs}")
        
        # Load and preprocess data
        df, X_raw, y = self.load_data(input_path)
        
        # Scale features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)
        
        # Stratified split for internal validation
        self.X_train, self.X_val, self.y_train, self.y_val = regression_stratified_split(
            X, y,
            test_size=INTERNAL_VAL_SIZE,
            n_bins=5,
            random_state=DEFAULT_RANDOM_STATE
        )
        
        self.logger.info(f"Data split: Train={len(self.X_train)}, Val={len(self.X_val)}")
        self.logger.info(f"Feature dimensions: {X.shape[1]} bands")
        
        # Calculate expected number of groups
        n_groups = (X.shape[1] + self.window_size - 1) // self.window_size
        self.logger.info(f"Spectral groups: {n_groups} (window_size={self.window_size})")
        
        # Create SG-HHO optimizer
        optimizer = SpectralGroupHHO(
            epoch=self.epoch,
            pop_size=self.pop_size,
            window_size=self.window_size
        )
        
        # Setup optimization problem
        problem_dict = {
            "obj_func": self._stability_driven_fitness,
            "bounds": FloatVar(lb=[0] * X.shape[1], ub=[1] * X.shape[1]),
            "minmax": "min",
            "log_to": None
        }
        
        # Run optimization
        self.logger.info("Starting SG-HHO optimization...")
        self.logger.info("This may take several minutes due to stability assessment...")
        
        agent = optimizer.solve(problem_dict)
        
        # Extract results
        best_pos = agent.solution
        best_idx = np.where(best_pos > 0.5)[0]
        best_feats = [self.feat_cols[i] for i in best_idx]
        
        # Get fitness value
        fit_val = getattr(agent.target, 'fitness', getattr(agent, 'fitness', 'N/A'))
        
        self.logger.info("="*60)
        self.logger.info("SG-HHO Optimization Complete")
        self.logger.info("="*60)
        self.logger.info(f"Best fitness: {fit_val}")
        self.logger.info(f"Selected features: {len(best_feats)} / {X.shape[1]}")
        self.logger.info(f"Reduction rate: {(1 - len(best_feats)/X.shape[1])*100:.1f}%")
        
        # Estimate selected groups
        estimated_groups = len(best_feats) / self.window_size
        self.logger.info(f"Estimated selected groups: ~{estimated_groups:.1f} / {n_groups}")
        
        # Analyze spectral continuity
        if len(best_idx) > 1:
            gaps = np.diff(sorted(best_idx))
            max_gap = np.max(gaps)
            avg_gap = np.mean(gaps)
            self.logger.info(f"Spectral continuity: max_gap={max_gap}, avg_gap={avg_gap:.2f}")
        
        # Save results
        self.save_selection_result(df, y, best_feats, output_path)
        
        return SelectionResult(
            selected_features=best_feats,
            selected_indices=best_idx.tolist()
        )
