"""Search for final-model parameters that do not overfit.

Phase 3 tuned XGBoost and Random Forest for cross-validated PR-AUC alone.
That produced strong generalisation but near-perfect training scores:
0.9995 on training against 0.795 on test for chronological XGBoost. The
mechanism is memorisation of a very small positive class -- 399 training
frauds against up to 12,800 leaves -- and it leaves a train/test gap that no
examiner should have to be talked out of worrying about.

Definition used for "does not overfit"
--------------------------------------
Zero gap is unattainable: even logistic regression has one, and forcing it
to zero produces an underfitted model. So the criterion is:

  1. cross-validated PR-AUC statistically indistinguishable from the best
     found (within one standard error), AND
  2. the smallest train/CV gap among configurations meeting (1).

This protects generalisation first and then removes as much memorisation as
generalisation allows. The resulting models are verified independently in
scripts/run_overfitting_analysis.py.

Why these parameters
--------------------
The levers weighted most heavily attack the diagnosed mechanism directly.
min_child_weight (XGBoost) and min_samples_leaf (Random Forest) forbid a
leaf built around a handful of rows, which is exactly how a single training
fraud gets isolated. Shallower trees, L1/L2 penalties, minimum split loss
and row/column subsampling reduce capacity more generally.

Selection uses cross-validation on the training data only. The test set is
not read by this script at all.

Output: results/final_params.json -- the single source of truth every
downstream script now reads, so no result can silently use stale parameters.

Usage: ./venv/bin/python scripts/search_final_params.py
"""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, TimeSeriesSplit

from fraud import config, data, experiment, resampling

OUT_JSON = config.RESULTS_DIR / "final_params.json"
OUT_CSV = config.RESULTS_DIR / "overfitting" / "final_param_search.csv"
SPLITTERS = {"stratified": data.stratified_split,
             "chronological": data.chronological_split}

SPACES = {
    "xgboost": {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.02, 0.03, 0.05, 0.1],
        "min_child_weight": [1, 5, 10, 20, 40],
        "reg_lambda": [1, 5, 10, 20],
        "reg_alpha": [0, 1, 5],
        "gamma": [0, 0.5, 1, 2],
        "subsample": [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8, 10],
        "min_samples_leaf": [5, 10, 20, 40],
        "max_features": ["sqrt", 0.3, 0.5],
        "max_samples": [0.5, 0.7, None],
    },
}
N_ITER = {"xgboost": 40, "random_forest": 20}


def phase3_params(model: str, protocol: str) -> dict:
    """The CV-only Phase 3 parameters, kept as an anchor for comparison."""
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


def evaluate(model, params, protocol, X, y) -> dict:
    cv = (StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_SEED)
          if protocol == "stratified" else TimeSeriesSplit(5))
    folds = []
    for tr, va in cv.split(X, y):
        if y[va].sum() == 0:
            continue
        est = resampling.make_estimator(model, "none", params=dict(params), y_train=y[tr])
        est.fit(X.iloc[tr], y[tr])
        folds.append(average_precision_score(y[va], est.predict_proba(X.iloc[va])[:, 1]))
    est = resampling.make_estimator(model, "none", params=dict(params), y_train=y)
    est.fit(X, y)
    train = average_precision_score(y, est.predict_proba(X)[:, 1])
    cvm, cvs = float(np.mean(folds)), float(np.std(folds))
    return {"train_pr_auc": train, "cv_mean": cvm, "cv_sd": cvs,
            "cv_se": cvs / np.sqrt(len(folds)), "gap": train - cvm}


def main() -> None:
    df = experiment.load_clean()
    final: dict = {"_definition": (
        "Within 1 SE of best CV PR-AUC, smallest train-CV gap. "
        "Selected on training data only; test set not read."),
        "models": {}}
    rows = []

    for model in ("xgboost", "random_forest"):
        final["models"][model] = {}
        for protocol, splitter in SPLITTERS.items():
            Xtr, _, ytr, _ = splitter(df)          # test split discarded unread
            ytr = np.asarray(ytr)
            anchor = phase3_params(model, protocol)
            cands = [("phase3 anchor", anchor)]
            sampler = ParameterSampler(SPACES[model], n_iter=N_ITER[model],
                                       random_state=config.RANDOM_SEED)
            cands += [(f"search {i:02d}", dict(p)) for i, p in enumerate(sampler)]

            print(f"\n{'=' * 84}\n{model} / {protocol}  ({len(cands)} configurations)"
                  f"\n{'=' * 84}", flush=True)
            t0 = time.time()
            local = []
            for name, params in cands:
                r = evaluate(model, params, protocol, Xtr, ytr)
                rec = {"model": model, "protocol": protocol, "config": name,
                       "params": json.dumps(params, default=str), **r}
                local.append(rec); rows.append(rec)
                print(f"  {name:<15} train {r['train_pr_auc']:.4f}  "
                      f"CV {r['cv_mean']:.4f}±{r['cv_sd']:.3f}  gap {r['gap']:+.4f}",
                      flush=True)

            d = pd.DataFrame(local)
            best = d.loc[d.cv_mean.idxmax()]
            band = d[d.cv_mean >= best.cv_mean - best.cv_se]
            chosen = band.loc[band.gap.idxmin()]
            anc = d[d.config == "phase3 anchor"].iloc[0]
            print(f"\n  best CV            : {best.config} {best.cv_mean:.4f} (SE {best.cv_se:.4f})")
            print(f"  eligible (1-SE)    : {len(band)} of {len(d)}")
            print(f"  CHOSEN             : {chosen.config}")
            print(f"  gap   phase3 {anc.gap:+.4f}  ->  final {chosen.gap:+.4f}")
            print(f"  CV    phase3 {anc.cv_mean:.4f}  ->  final {chosen.cv_mean:.4f}")
            print(f"  ({time.time() - t0:.0f}s)", flush=True)

            final["models"][model][protocol] = {
                "params": json.loads(chosen.params),
                "train_pr_auc": round(float(chosen.train_pr_auc), 4),
                "cv_pr_auc": round(float(chosen.cv_mean), 4),
                "cv_sd": round(float(chosen.cv_sd), 4),
                "gap": round(float(chosen.gap), 4),
                "phase3_gap": round(float(anc.gap), 4),
                "phase3_cv_pr_auc": round(float(anc.cv_mean), 4),
                "selected_from": chosen.config,
            }

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(final, indent=2, default=str))
    print(f"\n[Saved] {OUT_JSON}\n[Saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
