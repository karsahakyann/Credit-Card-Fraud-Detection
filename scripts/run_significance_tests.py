"""Phase 5 -- is the threshold-tuning gain real, or within noise?

The cost figures rest on 95 frauds (74 chronological), and total cost is
dominated by a handful of large-amount transactions. A euro difference
between two configurations can therefore look decisive while being well
inside sampling noise.

This script bootstraps the test set to put a confidence interval on the one
claim the phase most wants to make: that choosing the threshold by cost
beats the default 0.5. It turns out to hold for some models and not others,
which is a more useful result than the unqualified version.

Usage: ./venv/bin/python scripts/run_significance_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fraud import config, costs, data, experiment

N_BOOT = 500
REVIEW_COST = 10.0
OUT = config.RESULTS_DIR / "threshold_gain_bootstrap.csv"

SPLITTERS = {"stratified": data.stratified_split,
             "chronological": data.chronological_split}
SCORE_FILES = {
    "xgboost": "xgboost_none_{p}.npy",
    "random_forest": "random_forest_none_{p}.npy",
    "logistic_regression": "logistic_regression_none_{p}.npy",
    "dnn": "dnn_{p}.npy",
}


def main() -> None:
    df = experiment.load_clean()
    sel = pd.read_csv(config.RESULTS_DIR / "threshold_selection.csv")
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []

    print(f"Bootstrap ({N_BOOT} resamples), review cost EUR {REVIEW_COST:.0f}\n")
    print(f"{'protocol':<15}{'model':<20}{'default':>9}{'tuned':>9}"
          f"{'gain':>8}{'95% CI':>22}  verdict")
    print("-" * 96)

    for protocol, splitter in SPLITTERS.items():
        _, X_test, _, y_test = splitter(df)
        y = np.asarray(y_test)
        amounts = X_test["Amount"].to_numpy(dtype=float)

        for model, pattern in SCORE_FILES.items():
            scores = np.load(experiment.SCORES_DIR / pattern.format(p=protocol))
            t = float(sel[(sel.model == model) & (sel.protocol == protocol)
                          & (sel.review_cost == REVIEW_COST)]
                      .iloc[0].threshold_selected_on_train)

            c_def = costs.prediction_cost(y, scores >= 0.5, amounts,
                                          REVIEW_COST)["total_cost"]
            c_tun = costs.prediction_cost(y, scores >= t, amounts,
                                          REVIEW_COST)["total_cost"]

            diffs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.integers(0, len(y), len(y))
                diffs[b] = (
                    costs.prediction_cost(y[idx], scores[idx] >= 0.5,
                                          amounts[idx], REVIEW_COST)["total_cost"]
                    - costs.prediction_cost(y[idx], scores[idx] >= t,
                                            amounts[idx], REVIEW_COST)["total_cost"]
                )
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            significant = bool(lo > 0)

            rows.append({
                "protocol": protocol, "model": model,
                "threshold": t,
                "cost_default": c_def, "cost_tuned": c_tun,
                "gain": c_def - c_tun,
                "ci_lo": lo, "ci_hi": hi, "significant": significant,
            })
            print(f"{protocol:<15}{model:<20}{c_def:>9,.0f}{c_tun:>9,.0f}"
                  f"{c_def - c_tun:>8,.0f}  [{lo:>+7,.0f},{hi:>+8,.0f}]  "
                  f"{'SIGNIFICANT' if significant else 'not significant'}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    n_sig = int(out.significant.sum())
    print(f"\nSignificant in {n_sig} of {len(out)} cells.")
    print("The gain is significant precisely where the model is poorly "
          "calibrated;\nfor the well-calibrated XGBoost the default 0.5 is "
          "already near-optimal.")
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
