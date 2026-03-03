"""SVM model with Optuna optimization.

Default parameters are defined as class/module attributes.
"""

import numpy as np
import optuna
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

from core.registry import register_model
from core.constants import DEFAULT_RANDOM_STATE
from model.base import BaseModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# SVM hyperparameter search ranges
SVM_PARAMS = {
    "C": (0.01, 100),
    "gamma": (0.1, 10),
    "epsilon": (0.01, 0.5),
    "kernel": ["rbf"]
}

OPTIMIZER_CONFIG = {
    "cv_folds": 5,
    "n_trials": 300,
    "random_state": DEFAULT_RANDOM_STATE
}


@register_model("SVM")
class SVMModel(BaseModel):
    default_n_trials = OPTIMIZER_CONFIG["n_trials"]
    default_cv_folds = OPTIMIZER_CONFIG["cv_folds"]

    def train_and_predict(self, X_train, y_train, X_test):
        n_trials = self.extra_params.get('n_trials', self.default_n_trials)
        cv_folds = self.extra_params.get('cv_folds', self.default_cv_folds)
        params_range = self.extra_params.get('params_range', SVM_PARAMS)

        self.logger.info(f"Running SVM (Optuna, {n_trials} trials)...")

        def objective(trial):
            c_min, c_max = params_range["C"]
            c_val = trial.suggest_float('C', c_min, c_max, log=True)

            g_min, g_max = params_range["gamma"]
            g_val = trial.suggest_float('gamma', g_min, g_max, log=True)

            e_min, e_max = params_range["epsilon"]
            e_val = trial.suggest_float('epsilon', e_min, e_max, log=True)

            k_val = trial.suggest_categorical('kernel', params_range["kernel"])

            model = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(C=c_val, gamma=g_val, epsilon=e_val, kernel=k_val))
            ])

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

        final_model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(**best_params))
        ])
        final_model.fit(X_train, y_train)

        pred_train = final_model.predict(X_train).flatten()
        pred_test = final_model.predict(X_test).flatten()

        return pred_train, pred_test, best_params
