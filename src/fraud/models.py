"""Tree-ensemble model factories and search spaces (Phase 2).

IMPORTANT — process isolation rule: torch (see ``dnn.py``), xgboost and
pandas cannot all be loaded in one process — the combination segfaults on
macOS (conflicting bundled OpenMP runtimes; torch 2.13 / xgboost 3.3 /
Python 3.13). This module therefore contains NO torch imports, ``dnn.py``
contains no xgboost imports, and each model family is tuned in its own
process (see ``scripts/tune_*.py``). Notebooks consume only the saved
results and never fit models across families in one kernel.
"""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config


def make_rf_pipeline(seed: int = config.RANDOM_SEED, **params) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=seed, n_jobs=-1, **params)),
        ]
    )


def make_xgb_pipeline(seed: int = config.RANDOM_SEED, **params) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    random_state=seed,
                    n_jobs=-1,
                    eval_metric="aucpr",
                    tree_method="hist",
                    **params,
                ),
            ),
        ]
    )


# Search spaces for RandomizedSearchCV (parameter names target the "model"
# step inside the pipeline).
RF_SEARCH_SPACE = {
    "model__n_estimators": randint(200, 601),
    "model__max_depth": [8, 12, 16, 20, 30, None],
    "model__min_samples_leaf": randint(1, 11),
    "model__max_features": ["sqrt", 0.3, 0.4, 0.5, 0.6],
}

XGB_SEARCH_SPACE = {
    "model__n_estimators": randint(200, 801),
    "model__max_depth": randint(3, 10),
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__min_child_weight": randint(1, 11),
    "model__gamma": uniform(0, 5),
}
