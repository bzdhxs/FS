"""Random Forest model with Optuna optimization.

Default parameters are defined as class/module attributes.
"""

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

from core.registry import register_model
from core.constants import DEFAULT_RANDOM_STATE
from model.base import BaseModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# RF hyperparameter search ranges (tuned for small samples N=68)
RF_PARAMS = {
    "n_estimators": (2, 100),
    "max_depth": (2, 10),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 4),
    "max_features": ['sqrt', 'log2', 0.3, 0.5, 0.7]
}

OPTIMIZER_CONFIG = {
    "cv_folds": 5,
    "n_trials": 200,
    "random_state": DEFAULT_RANDOM_STATE
}


@register_model("RF")
class RFModel(BaseModel):
    default_n_trials = OPTIMIZER_CONFIG["n_trials"]
    default_cv_folds = OPTIMIZER_CONFIG["cv_folds"]

    def train_and_predict(self, X_train, y_train, X_test):
        n_trials = self.extra_params.get('n_trials', self.default_n_trials)
        cv_folds = self.extra_params.get('cv_folds', self.default_cv_folds)
        params_range = self.extra_params.get('params_range', RF_PARAMS)

        self.logger.info(f"Running RF (Optuna, {n_trials} trials)...")

        def objective(trial):
            n_min, n_max = params_range["n_estimators"]
            n_estimators = trial.suggest_int('n_estimators', n_min, n_max)

            d_min, d_max = params_range["max_depth"]
            max_depth = trial.suggest_int('max_depth', d_min, d_max)

            s_min, s_max = params_range["min_samples_split"]
            min_samples_split = trial.suggest_int('min_samples_split', s_min, s_max)

            l_min, l_max = params_range["min_samples_leaf"]
            min_samples_leaf = trial.suggest_int('min_samples_leaf', l_min, l_max)

            max_features = trial.suggest_categorical('max_features', params_range["max_features"])

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=DEFAULT_RANDOM_STATE,
                n_jobs=-1
            )

            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            scores = cross_val_score(
                model, X_train, y_train, cv=cv,
                scoring='neg_root_mean_squared_error', n_jobs=-1
            )
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        self.logger.info(f"Best RMSE: {study.best_value:.4f}")

        final_model = RandomForestRegressor(
            **best_params, random_state=DEFAULT_RANDOM_STATE, n_jobs=-1
        )
        final_model.fit(X_train, y_train)

        pred_train = final_model.predict(X_train).flatten()
        pred_test = final_model.predict(X_test).flatten()

        return pred_train, pred_test, best_params
