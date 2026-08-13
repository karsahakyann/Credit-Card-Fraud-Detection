"""Shared helpers for the per-model tuning scripts (Phase 2).

Deliberately imports NO model libraries (no torch, no xgboost) so it is
safe in every process — see the isolation note in ``models.py``. Scripts
pass in a factory callable that builds their estimator.
"""

from __future__ import annotations

import json

import numpy as np

from . import config, data, evaluation

SCORES_DIR = config.RESULTS_DIR / "scores"
TUNED_PARAMS_PATH = config.RESULTS_DIR / "tuned_params.json"


def load_clean() -> "pd.DataFrame":  # noqa: F821 - annotation only
    return data.clean(data.load_raw())


def save_tuned_params(model_name: str, params: dict, cv_pr_auc: float, extra: dict | None = None):
    """Merge one model's best hyperparameters into results/tuned_params.json."""
    TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = {}
    if TUNED_PARAMS_PATH.exists():
        store = json.loads(TUNED_PARAMS_PATH.read_text())
    store[model_name] = {
        "params": params,
        "cv_pr_auc": cv_pr_auc,
        **(extra or {}),
    }
    TUNED_PARAMS_PATH.write_text(json.dumps(store, indent=2, default=str))


def final_evaluation(model_name: str, make_estimator, imbalance_strategy: str, notes: str):
    """Refit the tuned estimator per split protocol and log/save everything.

    For each protocol the estimator is trained on that protocol's training
    portion only, evaluated on its held-out test set, metrics appended to
    results/metrics.csv, and the continuous fraud scores saved to
    results/scores/<model>_<split>.npy so notebooks can rebuild PR curves
    without refitting models.
    """
    df = load_clean()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    splitters = {
        "stratified": data.stratified_split,
        "chronological": data.chronological_split,
    }
    results = {}
    for split_name, splitter in splitters.items():
        X_train, X_test, y_train, y_test = splitter(df)
        est = make_estimator()
        est.fit(X_train, y_train)
        metrics = evaluation.evaluate(est, X_test, y_test)
        evaluation.log_result(
            metrics, model_name=model_name, split=split_name,
            imbalance_strategy=imbalance_strategy, notes=notes,
        )
        scores = est.predict_proba(X_test)[:, 1]
        np.save(SCORES_DIR / f"{model_name}_{split_name}.npy", scores)
        results[split_name] = metrics
        print(f"[{model_name} | {split_name}] "
              f"PR-AUC={metrics['pr_auc']:.4f} MCC={metrics['mcc']:.4f} "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}")
    return results
