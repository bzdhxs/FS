"""PLS Regression model with RandomizedSearchCV optimization.

Default parameters are defined as class/module attributes.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RandomizedSearchCV, KFold

from core.registry import register_model
from core.constants import DEFAULT_RANDOM_STATE
from model.base import BaseModel

# PLS optimizer config
OPTIMIZER_CONFIG = {
    "cv_folds": 5,
    "n_iter": 100,
    "scoring": "neg_root_mean_squared_error",
    "random_state": DEFAULT_RANDOM_STATE
}


@register_model("PLS")
class PLSModel(BaseModel):
    default_cv_folds = OPTIMIZER_CONFIG["cv_folds"]
    default_n_iter = OPTIMIZER_CONFIG["n_iter"]

    def train_and_predict(self, X_train, y_train, X_test):
        cv_folds = self.extra_params.get('cv_folds', self.default_cv_folds)
        n_iter = self.extra_params.get('n_iter', self.default_n_iter)

        self.logger.info("Running PLS (RandomizedSearchCV)...")

        max_components = min(X_train.shape[1], X_train.shape[0], 20)
        param_dist = {'n_components': list(range(1, max_components + 1))}

        real_n_iter = min(n_iter, len(param_dist['n_components']))

        search = RandomizedSearchCV(
            estimator=PLSRegression(),
            param_distributions=param_dist,
            n_iter=real_n_iter,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE),
            scoring=OPTIMIZER_CONFIG["scoring"],
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )

        search.fit(X_train, y_train)

        best_params = search.best_params_
        best_model = search.best_estimator_

        self.logger.info(f"Best n_components: {best_params['n_components']}")

        pred_train = best_model.predict(X_train).flatten()
        pred_test = best_model.predict(X_test).flatten()

        return pred_train, pred_test, best_params
