"""Phase 5 -- choose the operating threshold WITHOUT touching the test set.

Why this script exists
----------------------
The obvious way to do cost-sensitive evaluation is to sweep thresholds on
the test set and report the cheapest one. That number is an *oracle*: the
threshold was chosen with knowledge of the test labels, so it cannot be
achieved in deployment and it flatters the model. It is exactly the class of
optimism this dissertation criticises elsewhere, so it must not be the
headline here.

This script does it the deployable way:

1. Fit each tuned model on the training split only, collecting out-of-fold
   predictions (stratified k-fold for the random protocol, forward-chaining
   TimeSeriesSplit for the chronological one).
2. Pick the cost-minimising threshold on those out-of-fold scores, using the
   training rows' own Amounts.
3. Freeze that threshold and apply it to the held-out test scores.

The oracle threshold is still computed, but only as an upper bound, so the
gap between "achievable" and "best possible" is visible rather than hidden.

Process isolation (see models.py): torch and xgboost cannot share a process,
so ``--models dnn`` must be run separately from the tree/linear models.

Usage
-----
    ./venv/bin/python scripts/select_thresholds.py
    ./venv/bin/python scripts/select_thresholds.py --models dnn
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from fraud import config, costs, data, experiment

OUT_CSV = config.RESULTS_DIR / "threshold_selection.csv"
OOF_DIR = config.RESULTS_DIR / "oof_scores"

SPLITTERS = {
    "stratified": data.stratified_split,
    "chronological": data.chronological_split,
}
TEST_SCORE_FILES = {
    "xgboost": "xgboost_none_{p}.npy",
    "random_forest": "random_forest_none_{p}.npy",
    "logistic_regression": "logistic_regression_none_{p}.npy",
    "dnn": "dnn_{p}.npy",
}


def _cast(value: str):
    """Phase 3 stored params as strings; recover int / float / str."""
    try:
        parsed = ast.literal_eval(value)
        return parsed
    except (ValueError, SyntaxError):
        return value


def tuned_params(model: str, protocol: str) -> dict:
    """Hyperparameters that produced the saved test scores for this cell."""
    if model == "dnn":
        store = json.loads((config.RESULTS_DIR / "tuned_params.json").read_text())
        return dict(store["dnn"]["params"])

    grid = pd.read_csv(config.RESULTS_DIR / "imbalance_experiment.csv")
    row = grid[(grid.model == model) & (grid.strategy == "none")
               & (grid.protocol == protocol)]
    if row.empty:
        raise ValueError(f"No Phase 3 params for {model}/{protocol}")
    raw = json.loads(row.iloc[0].best_params)
    return {k.replace("model__", ""): _cast(v) for k, v in raw.items()}


def build_estimator(model: str, params: dict, y_train: np.ndarray):
    """Lazy imports keep torch and xgboost out of the same process."""
    if model == "dnn":
        from fraud.dnn import TorchDNNClassifier
        return TorchDNNClassifier(device="cpu", **params)
    from fraud import resampling
    return resampling.make_estimator(model, "none", params=params, y_train=y_train)


def make_cv(protocol: str, n_splits: int = config.N_CV_FOLDS):
    if protocol == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=config.RANDOM_SEED)
    # Chronological training rows are already time-ordered, so forward
    # chaining never lets a fold train on its own future.
    return TimeSeriesSplit(n_splits=n_splits)


def out_of_fold_scores(estimator, X: np.ndarray, y: np.ndarray, cv) -> np.ndarray:
    """OOF probabilities; NaN for rows no fold ever validates (TimeSeriesSplit)."""
    oof = np.full(len(y), np.nan)
    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        oof[va] = est.predict_proba(X[va])[:, 1]
        print(f"      fold {fold}: trained on {len(tr):,}, scored {len(va):,}",
              flush=True)
    return oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["xgboost", "random_forest", "logistic_regression"])
    parser.add_argument("--protocols", nargs="+", default=list(SPLITTERS))
    args = parser.parse_args()

    df = experiment.load_clean()
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for protocol in args.protocols:
        X_train, X_test, y_train, y_test = SPLITTERS[protocol](df)
        y_tr = np.asarray(y_train)
        y_te = np.asarray(y_test)
        amt_tr = X_train["Amount"].to_numpy(dtype=float)
        amt_te = X_test["Amount"].to_numpy(dtype=float)
        print(f"\n{'=' * 70}\n{protocol.upper()}  "
              f"train {len(y_tr):,} / test {len(y_te):,}\n{'=' * 70}")

        for model in args.models:
            params = tuned_params(model, protocol)
            dtype = np.float32 if model == "dnn" else np.float64
            X_arr = X_train.to_numpy(dtype=dtype)
            print(f"\n  {model}  params={params}")

            oof_path = OOF_DIR / f"{model}_{protocol}_oof.npy"
            if oof_path.exists():
                oof = np.load(oof_path)
                print(f"      [cached] {oof_path.name}")
            else:
                est = build_estimator(model, params, y_tr)
                oof = out_of_fold_scores(est, X_arr, y_tr, make_cv(protocol))
                np.save(oof_path, oof)

            scored = ~np.isnan(oof)
            test_scores = np.load(
                experiment.SCORES_DIR / TEST_SCORE_FILES[model].format(p=protocol)
            )

            for rc in costs.REVIEW_COSTS:
                # 1. threshold chosen on TRAINING out-of-fold scores only
                sel = costs.optimal_threshold(
                    y_tr[scored], oof[scored], amt_tr[scored], rc,
                )
                t_star = sel["best_threshold"]

                # 2. frozen threshold applied to the untouched test set
                deployed = costs.prediction_cost(
                    y_te, test_scores >= t_star, amt_te, rc,
                )
                # 3. reference points on the test set
                at_default = costs.prediction_cost(y_te, test_scores >= 0.5, amt_te, rc)
                oracle = costs.optimal_threshold(y_te, test_scores, amt_te, rc)
                base = costs.baseline_costs(y_te, amt_te, rc)

                rows.append({
                    "protocol": protocol,
                    "model": model,
                    "review_cost": rc,
                    "threshold_selected_on_train": t_star,
                    "cost_deployed": deployed["total_cost"],
                    "cost_at_default_0_5": at_default["total_cost"],
                    "cost_oracle_test": oracle["cost_at_best"],
                    "threshold_oracle_test": oracle["best_threshold"],
                    "optimism_gap": deployed["total_cost"] - oracle["cost_at_best"],
                    "gain_vs_default": at_default["total_cost"] - deployed["total_cost"],
                    "savings_vs_baseline": base["best_baseline"] - deployed["total_cost"],
                    "baseline_best": base["best_baseline"],
                    "tp": deployed["tp"], "fp": deployed["fp"], "fn": deployed["fn"],
                    "n_train_scored": int(scored.sum()),
                })

            r = [x for x in rows if x["model"] == model
                 and x["protocol"] == protocol and x["review_cost"] == 10.0][0]
            print(f"      rc=10  t*={r['threshold_selected_on_train']:.2f} (from train)  "
                  f"deployed {r['cost_deployed']:>8,.0f}  "
                  f"oracle {r['cost_oracle_test']:>8,.0f}  "
                  f"optimism {r['optimism_gap']:>6,.0f}")

    out = pd.DataFrame(rows)
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        keep = ~prev.set_index(["protocol", "model", "review_cost"]).index.isin(
            out.set_index(["protocol", "model", "review_cost"]).index
        )
        out = pd.concat([prev[keep], out], ignore_index=True)
    out.sort_values(["protocol", "model", "review_cost"]).to_csv(OUT_CSV, index=False)
    print(f"\n[Saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
