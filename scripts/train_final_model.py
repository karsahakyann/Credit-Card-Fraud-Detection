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

LABELS = {"xgboost": "XGBoost", "random_forest": "Random Forest",
          "logistic_regression": "Logistic Regression", "dnn": "Neural network"}

MODEL_DIR = config.PROJECT_ROOT / "models"
REVIEW_COST = 10.0
SPLITTERS = {"stratified": data.stratified_split,
             "chronological": data.chronological_split}

# Every model the console can serve. XGBoost is the project's recommendation;
# the others are persisted so the demo can switch between them live and show
# the Phase 3 comparison running rather than tabulated.
# The DNN is excluded by default. It is far slower to fit than the others
# and its loky/torch interaction stalled repeatedly here, which is not worth
# blocking artefact generation for. It still appears in the console's model
# comparison, which reads the scores Phase 2 saved, so only *live* switching
# to the network is unavailable without --with-dnn.
MODELS = ("xgboost", "random_forest", "logistic_regression")
OPTIONAL_MODELS = ("dnn",)


def tuned_params(protocol: str, model: str = "xgboost") -> dict:
    if model == "dnn":
        store = json.loads((config.RESULTS_DIR / "tuned_params.json").read_text())
        return dict(store["dnn"]["params"])
    grid = pd.read_csv(config.RESULTS_DIR / "imbalance_experiment.csv")
    row = grid[(grid.model == model) & (grid.strategy == "none")
               & (grid.protocol == protocol)].iloc[0]
    out = {}
    for k, v in json.loads(row.best_params).items():
        try:
            out[k.replace("model__", "")] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.replace("model__", "")] = v
    return out


def deployed_threshold(protocol: str, model: str = "xgboost") -> float:
    sel = pd.read_csv(config.RESULTS_DIR / "threshold_selection.csv")
    row = sel[(sel.model == model) & (sel.protocol == protocol)
              & (sel.review_cost == REVIEW_COST)].iloc[0]
    return float(row.threshold_selected_on_train)


def build_estimator(model: str, params: dict, y_train):
    """The DNN standardises internally, so it is used bare rather than piped."""
    if model == "dnn":
        from fraud.dnn import TorchDNNClassifier
        return TorchDNNClassifier(device="cpu", **params)
    return resampling.make_estimator(model, "none", params=params, y_train=y_train)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-dnn", action="store_true",
                    help="also persist the neural network (slow)")
    args = ap.parse_args()
    models = MODELS + (OPTIONAL_MODELS if args.with_dnn else ())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = experiment.load_clean()

    for protocol, splitter in SPLITTERS.items():
        X_train, X_test, y_train, y_test = splitter(df)
        y_tr, y_te = np.asarray(y_train), np.asarray(y_test)
        print(f"\n{'=' * 60}\n{protocol.upper()}\n{'=' * 60}")
        for model in models:
            params = tuned_params(protocol, model)
            threshold = deployed_threshold(protocol, model)

            fit_X = (X_train.to_numpy(dtype=np.float32) if model == "dnn"
                     else X_train)
            test_X = (X_test.to_numpy(dtype=np.float32) if model == "dnn"
                      else X_test)
            est = build_estimator(model, params, y_tr)
            est.fit(fit_X, y_tr)

            metrics = evaluation.evaluate(est, test_X, y_te, threshold=threshold)
            bundle = {
                "pipeline": est,
                "protocol": protocol,
                "model_key": model,
                "model_name": LABELS[model],
                "params": params,
                "threshold": threshold,
                "review_cost": REVIEW_COST,
                "feature_names": list(X_train.columns),
                "n_train": int(len(y_tr)),
                "test_metrics": metrics,
                "needs_float32": model == "dnn",
            }
            model_path = MODEL_DIR / f"final_{protocol}_{model}.joblib"
            joblib.dump(bundle, model_path)
            if model == "xgboost":            # default the console loads
                joblib.dump(bundle, MODEL_DIR / f"final_{protocol}.joblib")
            print(f"  {LABELS[model]:<20} t={threshold:.2f}  "
                  f"PR-AUC {metrics['pr_auc']:.4f}  "
                  f"caught {metrics['tp']}/{metrics['tp']+metrics['fn']}  "
                  f"FP {metrics['fp']:<3} [{model_path.name}]")

        params = tuned_params(protocol)
        threshold = deployed_threshold(protocol)
        metrics = joblib.load(MODEL_DIR / f"final_{protocol}.joblib")["test_metrics"]

        replay_path = MODEL_DIR / f"replay_{protocol}.npz"
        np.savez_compressed(
            replay_path,
            X=X_test.to_numpy(dtype=np.float64),
            y=y_te,
            amounts=X_test["Amount"].to_numpy(dtype=float),
            columns=np.array(list(X_test.columns), dtype=object),
        )

        print(f"  replay bundle: {replay_path.name} "
              f"({replay_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\nArtefacts in {MODEL_DIR} (gitignored; rerun this script to rebuild)")


if __name__ == "__main__":
    main()
