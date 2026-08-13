"""Tune the PyTorch DNN (focal loss) with Optuna on the training split.

Runs in its own process — never import xgboost here (see fraud/models.py).
Each trial trains on 75% of the training split and is scored by PR-AUC on
the remaining 25% (a single stratified validation fold rather than k-fold,
because every trial trains a full network). The best configuration is then
refit per split protocol on the full training portion.

Usage: ./venv/bin/python scripts/tune_dnn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from fraud import config, data, experiment
from fraud.dnn import TorchDNNClassifier

N_TRIALS = 20


def main() -> None:
    df = experiment.load_clean()
    X_train, _, y_train, _ = data.stratified_split(df)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train,
        random_state=config.RANDOM_SEED,
    )

    def objective(trial: optuna.Trial) -> float:
        clf = TorchDNNClassifier(
            hidden=trial.suggest_categorical("hidden", [64, 128, 256]),
            n_layers=trial.suggest_int("n_layers", 2, 3),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048]),
            focal_alpha=trial.suggest_float("focal_alpha", 0.1, 0.9),
            focal_gamma=trial.suggest_categorical("focal_gamma", [0.0, 1.0, 2.0, 3.0]),
            max_epochs=60,
            patience=5,
        )
        clf.fit(X_fit, y_fit)
        return average_precision_score(y_val, clf.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    print(f"\nBest val PR-AUC: {study.best_value:.4f}\nBest params: {best}")
    experiment.save_tuned_params("dnn", best, study.best_value,
                                 extra={"n_trials": N_TRIALS, "validation": "25% stratified holdout of train"})

    experiment.final_evaluation(
        "dnn",
        make_estimator=lambda: TorchDNNClassifier(max_epochs=60, patience=5, **best),
        imbalance_strategy="focal_loss",
        notes="phase2 tuned",
    )


if __name__ == "__main__":
    main()
