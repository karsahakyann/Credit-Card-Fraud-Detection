"""Does the model need all 30 features, or only the important ones?

SHAP showed the signal is concentrated: four components carry half the
attribution. The natural question is whether the other 26 earn their place.

The trap, and how it is avoided
-------------------------------
The obvious approach is to rank features by the SHAP values already in
results/, take the top k, and evaluate on the test set. Those attributions
were computed *on the test set*, so selecting with them and then scoring on
the same rows is selection leakage -- the reduced model would look better
than it could ever be in deployment, for the same reason that picking a
threshold on test data flatters it.

So the ranking here is recomputed from **training data only**: fit on train,
explain a sample of train, rank, select, then evaluate on the untouched test
set. The published test-set ranking is reported alongside purely to show
whether the two orderings agree.

Differences are bootstrapped, because with 74-95 frauds a PR-AUC gap of a
couple of points is not distinguishable from noise.

Usage: ./venv/bin/python scripts/run_feature_reduction.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud import config, data, experiment, explain, resampling

TOP_K = (4, 8, 15, 20, 30)
SHAP_SAMPLE = 20_000          # training rows explained, for speed
N_BOOT = 400
OUT = config.RESULTS_DIR / "feature_reduction.csv"

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


def train_side_ranking(X_train, y_train, params, seed=config.RANDOM_SEED):
    """Rank features by mean |SHAP| computed on TRAINING rows only."""
    pipe = resampling.make_estimator(
        "xgboost", "none", params=dict(params), y_train=y_train)
    pipe.fit(X_train, y_train)

    rng = np.random.default_rng(seed)
    n = min(SHAP_SAMPLE, len(X_train))
    idx = rng.choice(len(X_train), size=n, replace=False)
    sample = X_train.iloc[idx]

    sv, _ = explain.tree_shap_values(pipe, sample)
    imp = explain.global_importance(sv, list(X_train.columns))
    return imp.feature.tolist(), imp


def main() -> None:
    df = experiment.load_clean()
    published = pd.read_csv(config.RESULTS_DIR / "shap_global_importance.csv")
    rows = []

    for protocol, splitter in SPLITTERS.items():
        X_train, X_test, y_train, y_test = splitter(df)
        y_tr, y_te = np.asarray(y_train), np.asarray(y_test)
        params = tuned_params(protocol)

        print(f"\n{'=' * 78}\n{protocol.upper()}\n{'=' * 78}")
        ranked, imp = train_side_ranking(X_train, y_tr, params)
        print(f"  train-derived top 8 : {', '.join(ranked[:8])}")

        pub = published[published.protocol == protocol].sort_values("rank")
        pub_order = pub.feature.tolist()
        print(f"  test-derived top 8  : {', '.join(pub_order[:8])}")
        agree = len(set(ranked[:8]) & set(pub_order[:8]))
        print(f"  overlap in top 8    : {agree}/8 "
              f"({'orderings agree closely' if agree >= 6 else 'orderings differ'})")

        scores_by_k = {}
        for k in TOP_K:
            cols = ranked[:k]
            pipe = resampling.make_estimator(
                "xgboost", "none", params=dict(params), y_train=y_tr)
            pipe.fit(X_train[cols], y_tr)
            s = pipe.predict_proba(X_test[cols])[:, 1]
            scores_by_k[k] = s
            rows.append({"protocol": protocol, "k": k,
                         "pr_auc": average_precision_score(y_te, s),
                         "features": "|".join(cols)})

        full = scores_by_k[30]
        rng = np.random.default_rng(config.RANDOM_SEED)
        print(f"\n  {'k':>3}{'PR-AUC':>9}{'vs full':>9}{'95% CI':>20}  verdict")
        print("  " + "-" * 62)
        for k in TOP_K:
            ap = average_precision_score(y_te, scores_by_k[k])
            if k == 30:
                print(f"  {k:>3}{ap:>9.4f}{'--':>9}{'(reference)':>20}")
                continue
            d = np.empty(N_BOOT)
            for b in range(N_BOOT):
                i = rng.integers(0, len(y_te), len(y_te))
                if y_te[i].sum() == 0:
                    d[b] = np.nan; continue
                d[b] = (average_precision_score(y_te[i], scores_by_k[k][i])
                        - average_precision_score(y_te[i], full[i]))
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            verdict = ("worse than full" if hi < 0 else
                       "better than full" if lo > 0 else "indistinguishable")
            for r in rows:
                if r["protocol"] == protocol and r["k"] == k:
                    r.update(delta_vs_full=ap - average_precision_score(y_te, full),
                             ci_lo=lo, ci_hi=hi, verdict=verdict)
            print(f"  {k:>3}{ap:>9.4f}{ap - average_precision_score(y_te, full):>+9.4f}"
                  f"  [{lo:>+7.4f},{hi:>+7.4f}]  {verdict}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
