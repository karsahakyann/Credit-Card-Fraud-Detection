"""Tune XGBoost with RandomizedSearchCV on the training split (PR-AUC objective).

Runs in its own process — never import torch here (see fraud/models.py).
Usage: ./venv/bin/python scripts/tune_xgb.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from fraud import config, data, experiment
from fraud.models import XGB_SEARCH_SPACE, make_xgb_pipeline

N_ITER = 30


def main() -> None:
    df = experiment.load_clean()
    X_train, _, y_train, _ = data.stratified_split(df)

    search = RandomizedSearchCV(
        make_xgb_pipeline(),
        XGB_SEARCH_SPACE,
        n_iter=N_ITER,
        cv=StratifiedKFold(config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED),
        scoring="average_precision",
        random_state=config.RANDOM_SEED,
        n_jobs=1,       # parallelism lives inside XGBoost; keeps memory bounded
        verbose=2,
        refit=False,    # final refits happen per split protocol below
    )
    search.fit(X_train, y_train)

    best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}\nBest params: {best}")
    experiment.save_tuned_params("xgboost", best, search.best_score_,
                                 extra={"n_iter": N_ITER, "cv_folds": config.N_CV_FOLDS})

    experiment.final_evaluation(
        "xgboost",
        make_estimator=lambda: make_xgb_pipeline(**best),
        imbalance_strategy="none",
        notes="phase2 tuned",
    )


if __name__ == "__main__":
    main()
