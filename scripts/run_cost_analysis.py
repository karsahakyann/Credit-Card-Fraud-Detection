"""Phase 5 -- cost curves and the ORACLE threshold bound.

WARNING -- this script selects thresholds on the test set. That is an oracle:
it knows the test labels, so its "optimal cost" is an upper bound on what is
achievable, NOT a deployable result. Use it for the cost curves and as a
best-case reference only.

The deployable analysis is scripts/select_thresholds.py, which picks the
threshold from training out-of-fold scores and then freezes it before
touching the test set. Headline numbers must come from there.


Prices every final-model candidate's saved test scores under the cost model
in fraud/costs.py and finds the cost-optimal threshold, per protocol and per
assumed review cost. No model is refit: scores were banked by Phases 2/3
(results/scores/), and the splits are deterministic, so y_test and the
Amount column are reconstructed exactly.

Models analysed are the no-resampling variants -- the Phase 3 winners --
plus the DNN. Resampling variants are deliberately absent: Phase 3 showed
they move the operating point, and this analysis moves it on purpose,
optimally.

Usage: ./venv/bin/python scripts/run_cost_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fraud import config, costs, data, experiment

OUT_SUMMARY = config.RESULTS_DIR / "cost_analysis.csv"
OUT_CURVES = config.RESULTS_DIR / "cost_curves.csv"

SCORE_FILES = {
    "xgboost": "xgboost_none_{p}.npy",
    "random_forest": "random_forest_none_{p}.npy",
    "logistic_regression": "logistic_regression_none_{p}.npy",
    "dnn": "dnn_{p}.npy",
}

SPLITTERS = {
    "stratified": data.stratified_split,
    "chronological": data.chronological_split,
}


def main() -> None:
    df = experiment.load_clean()
    summary_rows, curve_rows = [], []

    for protocol, splitter in SPLITTERS.items():
        X_train, X_test, y_train, y_test = splitter(df)
        y = np.asarray(y_test)
        amounts = X_test["Amount"].to_numpy()
        total_fraud_amount = float(amounts[y == 1].sum())
        print(f"\n{protocol.upper()} test set: {len(y):,} rows, "
              f"{int(y.sum())} frauds worth EUR {total_fraud_amount:,.0f}")

        for model, pattern in SCORE_FILES.items():
            path = experiment.SCORES_DIR / pattern.format(p=protocol)
            if not path.exists():
                print(f"  [skip] {path.name} missing")
                continue
            scores = np.load(path)
            if len(scores) != len(y):
                raise ValueError(
                    f"{path.name}: {len(scores)} scores vs {len(y)} test rows "
                    "-- split reconstruction mismatch, do not trust results"
                )

            for rc in costs.REVIEW_COSTS:
                res = costs.optimal_threshold(y, scores, amounts, rc)
                summary_rows.append({"protocol": protocol, "model": model, **res})

                curve = costs.threshold_cost_curve(y, scores, amounts, rc)
                curve.insert(0, "model", model)
                curve.insert(0, "protocol", protocol)
                curve.insert(0, "review_cost", rc)
                curve_rows.append(curve)

            r10 = [r for r in summary_rows
                   if r["model"] == model and r["protocol"] == protocol
                   and r["review_cost"] == 10.0][0]
            print(f"  {model:<19} rc=10: t*={r10['best_threshold']:.2f} "
                  f"cost {r10['cost_at_best']:>9,.0f} "
                  f"(default-0.5 {r10['cost_at_default_0_5']:>9,.0f}, "
                  f"tuning gain {r10['gain_from_tuning']:>7,.0f}) "
                  f"baseline {r10['baseline_best_baseline']:>9,.0f}")

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(OUT_CURVES, index=False)
    print(f"\n[Saved] {OUT_SUMMARY}")
    print(f"[Saved] {OUT_CURVES}")


if __name__ == "__main__":
    main()
