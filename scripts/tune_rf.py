"""Tune Random Forest with RandomizedSearchCV on the training split (PR-AUC objective).

Runs in its own process. Usage: ./venv/bin/python scripts/tune_rf.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from fraud import config, data, experiment
from fraud.models import RF_SEARCH_SPACE, make_rf_pipeline

N_ITER = 15  # RF fits are the slowest of the suite; 15x5 folds is ~75 fits


def main() -> None:
    df = experiment.load_clean()
    X_train, _, y_train, _ = data.stratified_split(df)

    search = RandomizedSearchCV(
        make_rf_pipeline(),
        RF_SEARCH_SPACE,
        n_iter=N_ITER,
        cv=StratifiedKFold(config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED),
        scoring="average_precision",
        random_state=config.RANDOM_SEED,
        n_jobs=1,       # parallelism lives inside the forest; keeps memory bounded
        verbose=2,
        refit=False,
    )
    search.fit(X_train, y_train)

    best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}\nBest params: {best}")
    experiment.save_tuned_params("random_forest", best, search.best_score_,
                                 extra={"n_iter": N_ITER, "cv_folds": config.N_CV_FOLDS})

    experiment.final_evaluation(
        "random_forest",
        make_estimator=lambda: make_rf_pipeline(**best),
        imbalance_strategy="none",
        notes="phase2 tuned",
    )


if __name__ == "__main__":
    main()
