"""Class-imbalance strategies as leakage-free pipelines (Phase 3).

The methodological claim this dissertation defends is that resampling must
happen *inside* cross-validation, never before the split. ``imblearn``'s
Pipeline gives that guarantee structurally rather than by convention: a
sampler's ``fit_resample`` runs only during ``fit``, and is skipped entirely
during ``predict`` / ``predict_proba``. Validation and test rows are
therefore never oversampled, never dropped, and — critically for SMOTE —
never used to synthesise neighbours.

Resampling is placed *after* scaling because SMOTE interpolates between
nearest neighbours in feature space, and ``Time``/``Amount`` are on wildly
different scales from the PCA components V1-V28.

Four strategies are compared:

===============  ============================================================
``none``         no resampling; the untreated reference
``smote``        synthesise minority neighbours up to parity
``undersample``  randomly drop majority rows down to parity
``class_weight`` reweight the loss, leaving the data untouched
                 (``class_weight`` for LR/RF, ``scale_pos_weight`` for
                 XGBoost, focal loss for the DNN — see ``dnn.py``)
===============  ============================================================

Process-isolation rule from ``models.py`` still applies: this module imports
xgboost and therefore must never be imported in a process that loads torch.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config

STRATEGIES: tuple[str, ...] = ("none", "smote", "undersample", "class_weight")

MODELS: tuple[str, ...] = ("logistic_regression", "random_forest", "xgboost")


def make_resampler(strategy: str, seed: int = config.RANDOM_SEED):
    """Return the sampler for a strategy, or None when the data is untouched.

    ``none`` and ``class_weight`` both leave the training data as-is; they
    differ only in how the model weights the loss.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if strategy == "smote":
        # k_neighbors=5 is the default and is safe here: the smallest
        # training fold still contains far more than 5 frauds.
        return SMOTE(random_state=seed, k_neighbors=5)
    if strategy == "undersample":
        return RandomUnderSampler(random_state=seed)
    return None


def scale_pos_weight(y) -> float:
    """XGBoost's class-weight equivalent: ratio of negatives to positives.

    Computed once from the training portion rather than per fold. Folds are
    stratified, so the ratio is identical to within a fraction of a percent,
    and computing it from y_train only keeps validation rows out of the
    calculation.
    """
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive samples: cannot compute scale_pos_weight.")
    return n_neg / n_pos


def make_estimator(
    model_name: str,
    strategy: str,
    params: dict | None = None,
    y_train=None,
    seed: int = config.RANDOM_SEED,
):
    """Build a scaler -> [sampler] -> model pipeline for one (model, strategy).

    Parameters
    ----------
    model_name
        One of ``MODELS``.
    strategy
        One of ``STRATEGIES``.
    params
        Hyperparameters for the estimator, *without* the ``model__`` prefix
        (as stored in ``results/tuned_params.json``).
    y_train
        Training labels, required only for ``xgboost`` + ``class_weight`` so
        ``scale_pos_weight`` can be derived. Never pass test labels.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODELS}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")

    params = dict(params or {})
    weighted = strategy == "class_weight"

    if model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced" if weighted else None,
            random_state=seed,
            **params,
        )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            # balanced_subsample reweights within each bootstrap sample,
            # which suits bagging better than a single global weighting.
            class_weight="balanced_subsample" if weighted else None,
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    else:
        if weighted:
            if y_train is None:
                raise ValueError(
                    "xgboost + class_weight needs y_train to derive scale_pos_weight."
                )
            params["scale_pos_weight"] = scale_pos_weight(y_train)
        model = XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            eval_metric="aucpr",
            tree_method="hist",
            **params,
        )

    steps = [("scaler", StandardScaler())]
    sampler = make_resampler(strategy, seed=seed)
    if sampler is not None:
        steps.append(("resampler", sampler))
    steps.append(("model", model))
    return ImbPipeline(steps)


# Phase 3 search spaces, deliberately narrower than the Phase 2 grids in
# models.py. Phase 2 tuned one model per family and could afford 500+ trees at
# depth 20; Phase 3 retunes 24 cells, and SMOTE roughly doubles every training
# fold, which made the unconstrained grid a 6-10 hour run on this machine.
#
# Capping forest size is defensible here because the question this experiment
# answers is *which imbalance strategy generalises best*, holding the learner
# roughly constant -- not what the single best-tuned model is (Phase 2 already
# answered that). Every strategy is handicapped identically, so the comparison
# stays fair. The narrower budget is a stated limitation: absolute PR-AUC in
# this table will sit slightly below the Phase 2 headline numbers.
RF_SPACE_PHASE3: dict = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 10],
    "model__min_samples_leaf": [1, 3, 5],
    "model__max_features": ["sqrt"],
}

# How far to resample, expressed as minority:majority ratio after resampling.
#
# Forcing full parity (1.0) is the textbook default but a poor one here: the
# minority is 0.167%, so 1:1 SMOTE fabricates ~226k synthetic frauds from 378
# real ones -- a 600x extrapolation that both distorts the decision boundary
# and doubles every training fold. Treating the ratio as a hyperparameter lets
# the data choose, and is the more defensible comparison. Parity stays in the
# grid so the classic setting is still represented.
SAMPLING_RATIOS: list[float] = [0.01, 0.05, 0.1, 1.0]

XGB_SPACE_PHASE3: dict = {
    "model__n_estimators": [200, 300, 400],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.03, 0.1, 0.2],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

LR_SPACE_PHASE3: dict = {
    "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
}


def search_space(model_name: str, strategy: str = "none") -> dict:
    """Randomised-search space for one (model, strategy) cell.

    Discrete lists rather than continuous distributions, so a given n_iter
    explores comparable candidates for every strategy and the search is
    reproducible. For the resampling strategies the sampling ratio is tuned
    alongside the model's own hyperparameters.
    """
    if model_name == "random_forest":
        space = dict(RF_SPACE_PHASE3)
    elif model_name == "xgboost":
        space = dict(XGB_SPACE_PHASE3)
    elif model_name == "logistic_regression":
        space = dict(LR_SPACE_PHASE3)
    else:
        raise ValueError(f"Unknown model {model_name!r}")

    if strategy in {"smote", "undersample"}:
        space["resampler__sampling_strategy"] = list(SAMPLING_RATIOS)
    return space
