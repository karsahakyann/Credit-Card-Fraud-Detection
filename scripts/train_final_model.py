"""Train and persist the final model, plus everything the demo needs to run.

Until now every script refit from scratch and nothing was saved, so there
was no single artefact representing "the model this dissertation
recommends". A serving layer needs one, and so does anyone asking to see
the final result without re-running the experiments.

Persists to models/ (gitignored, since it is reproducible from this script):

    final_{protocol}.joblib    fitted pipeline + metadata
    replay_{protocol}.npz      held-out test features, labels, amounts

The replay bundle is the test set the model never saw during training, so
the demo streams genuinely unseen transactions rather than re-showing
training data.

Model and threshold both come from the dissertation's own results: the
Phase 3 winner (XGBoost, no resampling, protocol-tuned) at the Phase 5
cost-optimal threshold.

Usage: ./venv/bin/python scripts/train_final_model.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd

from fraud import config, data, evaluation, experiment, resampling

MODEL_DIR = config.PROJECT_ROOT / "models"
REVIEW_COST = 10.0
SPLITTERS = {"stratified": data.stratified_split,
             "chronological": data.chronological_split}


def tuned_params(protocol: str) -> dict:
    grid = pd.read_csv(config.RESULTS_DIR / "imbalance_experiment.csv")
    row = grid[(grid.model == "xgboost") & (grid.strategy == "none")
               & (grid.protocol == protocol)].iloc[0]
    out = {}
    for k, v in json.loads(row.best_params).items():
        try:
            out[k.replace("model__", "")] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.replace("model__", "")] = v
    return out


def deployed_threshold(protocol: str) -> float:
    sel = pd.read_csv(config.RESULTS_DIR / "threshold_selection.csv")
    row = sel[(sel.model == "xgboost") & (sel.protocol == protocol)
              & (sel.review_cost == REVIEW_COST)].iloc[0]
    return float(row.threshold_selected_on_train)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = experiment.load_clean()

    for protocol, splitter in SPLITTERS.items():
        X_train, X_test, y_train, y_test = splitter(df)
        y_tr, y_te = np.asarray(y_train), np.asarray(y_test)
        params = tuned_params(protocol)
        threshold = deployed_threshold(protocol)

        pipe = resampling.make_estimator(
            "xgboost", "none", params=params, y_train=y_tr,
        )
        pipe.fit(X_train, y_tr)

        metrics = evaluation.evaluate(pipe, X_test, y_te, threshold=threshold)
        bundle = {
            "pipeline": pipe,
            "protocol": protocol,
            "model_name": "XGBoost",
            "params": params,
            "threshold": threshold,
            "review_cost": REVIEW_COST,
            "feature_names": list(X_train.columns),
            "n_train": int(len(y_tr)),
            "test_metrics": metrics,
        }
        model_path = MODEL_DIR / f"final_{protocol}.joblib"
        joblib.dump(bundle, model_path)

        replay_path = MODEL_DIR / f"replay_{protocol}.npz"
        np.savez_compressed(
            replay_path,
            X=X_test.to_numpy(dtype=np.float64),
            y=y_te,
            amounts=X_test["Amount"].to_numpy(dtype=float),
            columns=np.array(list(X_test.columns), dtype=object),
        )

        print(f"\n{protocol.upper()}  threshold {threshold:.2f}")
        print(f"  PR-AUC {metrics['pr_auc']:.4f}   MCC {metrics['mcc']:.4f}")
        print(f"  precision {metrics['precision']:.3f}   recall {metrics['recall']:.3f}")
        print(f"  caught {metrics['tp']}/{metrics['tp'] + metrics['fn']}   "
              f"false alarms {metrics['fp']}")
        print(f"  [saved] {model_path.name}  ({model_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  [saved] {replay_path.name}  ({replay_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\nArtefacts in {MODEL_DIR} (gitignored; rerun this script to rebuild)")


if __name__ == "__main__":
    main()
