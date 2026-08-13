"""Phase 3 guarantees: resampling must never touch validation or test rows.

These tests are the executable version of the dissertation's leakage claim.
They use small synthetic data rather than the real CSV so the suite stays
fast and runnable without the dataset present.

Process isolation (see ``models.py``): this module imports xgboost through
``resampling``, so it must not be run in the same process as the torch
tests. Run test files individually.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fraud import resampling


@pytest.fixture(scope="module")
def imbalanced():
    """~2% positive class, echoing the real fraud rate."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        weights=[0.98, 0.02],
        flip_y=0.0,
        random_state=42,
    )
    return X, y


def test_every_model_strategy_combination_builds(imbalanced):
    _, y = imbalanced
    for model_name in resampling.MODELS:
        for strategy in resampling.STRATEGIES:
            est = resampling.make_estimator(model_name, strategy, y_train=y)
            assert est is not None


def test_sampler_present_only_for_resampling_strategies(imbalanced):
    _, y = imbalanced
    for strategy in resampling.STRATEGIES:
        est = resampling.make_estimator("logistic_regression", strategy, y_train=y)
        has_sampler = "resampler" in dict(est.steps)
        assert has_sampler == (strategy in {"smote", "undersample"}), strategy


def test_scaler_precedes_sampler(imbalanced):
    """SMOTE interpolates by distance, so scaling must happen first."""
    _, y = imbalanced
    est = resampling.make_estimator("logistic_regression", "smote", y_train=y)
    names = [name for name, _ in est.steps]
    assert names.index("scaler") < names.index("resampler") < names.index("model")


@pytest.mark.parametrize("strategy", ["smote", "undersample"])
def test_prediction_never_resamples(imbalanced, strategy):
    """The core guarantee: predict() returns one row per input row.

    If a sampler leaked into the predict path, the output length would
    change (SMOTE would inflate it, undersampling would shrink it).
    """
    X, y = imbalanced
    est = resampling.make_estimator("logistic_regression", strategy, y_train=y)
    est.fit(X[:800], y[:800])

    held_out = X[800:]
    assert len(est.predict(held_out)) == len(held_out)
    assert len(est.predict_proba(held_out)) == len(held_out)


@pytest.mark.parametrize("strategy", resampling.STRATEGIES)
def test_cross_validation_predicts_every_row_exactly_once(imbalanced, strategy):
    """Out-of-fold predictions must align 1:1 with the original samples.

    This is the leakage check that matters for Phase 3: were SMOTE applied
    before the split (or to validation folds), the out-of-fold prediction
    array could not line up with y.
    """
    X, y = imbalanced
    est = resampling.make_estimator("logistic_regression", strategy, y_train=y)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    preds = cross_val_predict(est, X, y, cv=cv, method="predict_proba")
    assert preds.shape == (len(y), 2)


def test_smote_balances_training_data_only(imbalanced):
    """Sanity-check the sampler itself: it does balance, when asked to."""
    X, y = imbalanced
    sampler = resampling.make_resampler("smote")
    X_res, y_res = sampler.fit_resample(X, y)
    assert (y_res == 1).sum() == (y_res == 0).sum()
    assert len(X_res) > len(X)


def test_undersampler_shrinks_to_parity(imbalanced):
    X, y = imbalanced
    sampler = resampling.make_resampler("undersample")
    X_res, y_res = sampler.fit_resample(X, y)
    assert (y_res == 1).sum() == (y_res == 0).sum()
    assert len(X_res) < len(X)


def test_none_and_class_weight_leave_data_untouched():
    assert resampling.make_resampler("none") is None
    assert resampling.make_resampler("class_weight") is None


def test_class_weight_reaches_the_model(imbalanced):
    _, y = imbalanced
    lr = resampling.make_estimator("logistic_regression", "class_weight", y_train=y)
    assert lr.named_steps["model"].class_weight == "balanced"

    rf = resampling.make_estimator("random_forest", "class_weight", y_train=y)
    assert rf.named_steps["model"].class_weight == "balanced_subsample"

    xgb = resampling.make_estimator("xgboost", "class_weight", y_train=y)
    assert xgb.named_steps["model"].scale_pos_weight == pytest.approx(
        (y == 0).sum() / (y == 1).sum()
    )


def test_no_class_weight_when_strategy_is_none(imbalanced):
    _, y = imbalanced
    lr = resampling.make_estimator("logistic_regression", "none", y_train=y)
    assert lr.named_steps["model"].class_weight is None


def test_scale_pos_weight_math():
    y = np.array([0] * 90 + [1] * 10)
    assert resampling.scale_pos_weight(y) == pytest.approx(9.0)


def test_scale_pos_weight_rejects_all_negative():
    with pytest.raises(ValueError, match="No positive samples"):
        resampling.scale_pos_weight(np.zeros(10))


def test_unknown_names_are_rejected(imbalanced):
    _, y = imbalanced
    with pytest.raises(ValueError, match="Unknown strategy"):
        resampling.make_estimator("logistic_regression", "oversample_a_lot", y_train=y)
    with pytest.raises(ValueError, match="Unknown model"):
        resampling.make_estimator("catboost", "none", y_train=y)


def test_xgboost_class_weight_requires_labels():
    with pytest.raises(ValueError, match="needs y_train"):
        resampling.make_estimator("xgboost", "class_weight", y_train=None)
