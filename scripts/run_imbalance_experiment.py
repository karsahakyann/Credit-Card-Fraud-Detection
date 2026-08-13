"""Phase 3 — the core class-imbalance experiment.

Compares four imbalance strategies (none / SMOTE / undersampling /
class weighting) across Logistic Regression, Random Forest and XGBoost,
under both split protocols, with every strategy retuned in its own right.

Design notes
------------
* **Leakage.** Resampling lives inside an ``imblearn`` Pipeline, so it is
  applied to training folds only — see ``fraud/resampling.py``.
* **Error bars for free.** ``RandomizedSearchCV`` already cross-validates
  every candidate, so the winning configuration's ``mean_test_score`` and
  ``std_test_score`` give the cross-validated PR-AUC and its spread without
  a second pass. Reported as ``cv_pr_auc_mean`` / ``cv_pr_auc_std``.
* **Fair tuning.** Each (model, strategy) pair is retuned separately, so no
  strategy is handicapped by hyperparameters chosen for another.
* **Protocol-local tuning.** The chronological run tunes on its own training
  portion. Reusing stratified-tuned parameters would leak future information
  into the hyperparameter choice — subtle, but exactly the kind of leakage
  this dissertation argues against.
* **CV inside each protocol.** Stratified k-fold for the random protocol;
  forward-chaining ``TimeSeriesSplit`` for the chronological one, which also
  lays the groundwork for the Phase 4 drift analysis.

Process isolation (see ``models.py``): imports xgboost, never torch. The DNN
is handled by its own script.

Usage
-----
    ./venv/bin/python scripts/run_imbalance_experiment.py            # full grid
    ./venv/bin/python scripts/run_imbalance_experiment.py --probe    # time 1 fit
    ./venv/bin/python scripts/run_imbalance_experiment.py --models random_forest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
)

from fraud import config, data, evaluation, experiment, resampling

RESULTS_CSV = config.RESULTS_DIR / "imbalance_experiment.csv"

# Search budget per (model, strategy). Deliberately small: the grid is 24
# cells and SMOTE roughly doubles every training fold. A timing probe with the
# unconstrained Phase 2 grid took >14 minutes for a single RF+SMOTE candidate,
# which put the full run at 6-10 hours; combined with the capped search spaces
# in resampling.py this brings it back to a couple of hours.
N_ITER = {
    "logistic_regression": 5,   # the whole C grid
    "random_forest": 5,
    "xgboost": 6,
}

PROTOCOLS = {
    "stratified": data.stratified_split,
    "chronological": data.chronological_split,
}


def make_cv(protocol: str, n_splits: int = config.N_CV_FOLDS):
    """Stratified folds for the random protocol, forward chaining for time."""
    if protocol == "stratified":
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED
        )
    # X_train from chronological_split is already Time-ordered, so
    # TimeSeriesSplit never trains on rows that follow its validation slice.
    return TimeSeriesSplit(n_splits=n_splits)


def run_cell(
    model_name: str,
    strategy: str,
    protocol: str,
    X_train,
    y_train,
    X_test,
    y_test,
    n_iter: int | None = None,
) -> dict:
    """Tune, cross-validate and test one (model, strategy, protocol) cell."""
    n_iter = n_iter or N_ITER[model_name]

    estimator = resampling.make_estimator(
        model_name, strategy, y_train=y_train,
    )
    space = resampling.search_space(model_name, strategy)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=n_iter,
        scoring="average_precision",   # PR-AUC, the headline metric
        cv=make_cv(protocol),
        refit=True,
        n_jobs=1,          # models already use n_jobs=-1; avoid oversubscription
        random_state=config.RANDOM_SEED,
        error_score="raise",
    )

    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    best = search.best_index_
    cv_mean = float(search.cv_results_["mean_test_score"][best])
    cv_std = float(search.cv_results_["std_test_score"][best])

    metrics = evaluation.evaluate(search.best_estimator_, X_test, y_test)
    evaluation.log_result(
        metrics,
        model_name=model_name,
        split=protocol,
        imbalance_strategy=strategy,
        notes="phase3 imbalance experiment",
    )

    scores = search.best_estimator_.predict_proba(X_test)[:, 1]
    experiment.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(
        experiment.SCORES_DIR / f"{model_name}_{strategy}_{protocol}.npy", scores
    )

    row = {
        "model": model_name,
        "strategy": strategy,
        "protocol": protocol,
        "cv_pr_auc_mean": cv_mean,
        "cv_pr_auc_std": cv_std,
        "test_pr_auc": metrics["pr_auc"],
        "test_mcc": metrics["mcc"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "n_iter": n_iter,
        "fit_seconds": round(elapsed, 1),
        "best_params": json.dumps(
            {k: str(v) for k, v in search.best_params_.items()}
        ),
    }
    print(
        f"[{protocol:13s} | {model_name:19s} | {strategy:12s}] "
        f"CV PR-AUC={cv_mean:.4f}+/-{cv_std:.4f}  "
        f"test PR-AUC={metrics['pr_auc']:.4f}  "
        f"R={metrics['recall']:.3f} P={metrics['precision']:.3f}  "
        f"({elapsed:.0f}s)",
        flush=True,
    )
    return row


def append_row(row: dict) -> None:
    """Write incrementally so a long run is never lost to an interruption."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(resampling.MODELS))
    parser.add_argument("--strategies", nargs="+", default=list(resampling.STRATEGIES))
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Time a single fit of the most expensive cell and exit.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Time ONE fit per (model, strategy) and project the full grid cost.",
    )
    args = parser.parse_args()

    df = experiment.load_clean()
    print(f"Clean dataset: {len(df):,} rows, {int(df[config.TARGET].sum())} frauds")

    if args.benchmark:
        # Measure, don't estimate: one plain fit per cell at the *slowest*
        # settings in the grid, then project the whole run from real numbers.
        X_train, _, y_train, _ = data.stratified_split(df)
        worst = {
            "random_forest": {"n_estimators": 200, "max_depth": 10},
            "xgboost": {"n_estimators": 400, "max_depth": 7},
            "logistic_regression": {},
        }
        print(f"\nSingle-fit timings on {len(X_train):,} training rows "
              f"(slowest params in each grid, worst-case sampling ratio)\n")
        per_fit: dict[tuple[str, str], float] = {}
        for model_name in args.models:
            for strategy in args.strategies:
                est = resampling.make_estimator(
                    model_name, strategy, params=worst[model_name], y_train=y_train,
                )
                if strategy in {"smote", "undersample"}:
                    est.set_params(resampler__sampling_strategy=1.0)
                t0 = time.perf_counter()
                est.fit(X_train, y_train)
                dt = time.perf_counter() - t0
                per_fit[(model_name, strategy)] = dt
                print(f"  {model_name:19s} {strategy:12s} {dt:8.1f}s", flush=True)

        folds = config.N_CV_FOLDS
        total = 0.0
        for (model_name, strategy), dt in per_fit.items():
            # n_iter candidates x folds, plus one refit on the full training set
            total += dt * (N_ITER[model_name] * folds + 1)
        total *= len(args.protocols)
        print(f"\nProjected full grid: {total / 3600:.1f} hours "
              f"({len(args.protocols)} protocols x {len(args.models)} models "
              f"x {len(args.strategies)} strategies, {folds}-fold)")
        print("This is an upper bound: every candidate is priced at the "
              "slowest hyperparameters and the widest sampling ratio.")
        return

    if args.probe:
        X_train, X_test, y_train, y_test = data.stratified_split(df)
        print("\nProbe: random_forest + smote, n_iter=1, 5-fold "
              "(the most expensive cell in the grid)\n")
        started = time.perf_counter()
        run_cell(
            "random_forest", "smote", "stratified",
            X_train, y_train, X_test, y_test, n_iter=1,
        )
        one_cell = time.perf_counter() - started
        print(f"\nOne RF+SMOTE search at n_iter=1 took {one_cell:.0f}s "
              f"({one_cell / 5:.0f}s per fold-fit).")
        print("Full grid is 24 cells; scale by each cell's n_iter to estimate.")
        return

    for protocol in args.protocols:
        X_train, X_test, y_train, y_test = PROTOCOLS[protocol](df)
        print(f"\n{'=' * 78}\n{protocol.upper()} — train {len(X_train):,} / "
              f"test {len(X_test):,} ({int(y_test.sum())} frauds)\n{'=' * 78}")
        for model_name in args.models:
            for strategy in args.strategies:
                row = run_cell(
                    model_name, strategy, protocol,
                    X_train, y_train, X_test, y_test,
                )
                append_row(row)

    print(f"\n[Saved] {RESULTS_CSV}")


if __name__ == "__main__":
    main()
