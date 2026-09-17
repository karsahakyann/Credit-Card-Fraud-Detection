"""Can the train/test gap be reduced without losing generalisation?

The overfitting analysis showed XGBoost reaches near-perfect training PR-AUC
(0.9995 chronological) against 0.795 on test. The out-of-fold estimate
matched test closely, so the gap does not harm generalisation -- but "we
decided it did not matter" is a weak answer to an examiner asking whether
anything was done about it. This study actually tries.

Why the gap exists
------------------
Only 399 frauds sit in the chronological training set, while 400 trees of
depth 5 offer up to 12,800 leaves. The ensemble has ample capacity to
isolate every training fraud individually, and it does: 381 of 399 score
above every legitimate training row.

Method, and the rule that keeps it honest
-----------------------------------------
Configurations span each standard lever -- shallower trees, larger minimum
child weight, L1/L2 penalties, minimum split loss, row and column
subsampling, slower learning, and early stopping.

Selection uses cross-validation on the training data ONLY. Choosing a
configuration by its test score would leak the test set into model
selection, which is precisely the error this project has avoided throughout.
The test score is computed for every configuration but consulted only after
selection is complete.

Among configurations whose CV score is within one standard error of the
best, the most regularised -- the smallest train/CV gap -- is chosen. This
one-standard-error rule prefers the simpler model when the data cannot tell
the candidates apart, which is the right default with 74-95 test frauds.

Usage: ./venv/bin/python scripts/run_regularisation_study.py
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
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from xgboost import XGBClassifier

from fraud import config, data, experiment

OUT = config.RESULTS_DIR / "overfitting" / "regularisation_study.csv"
SPLITTERS = {"stratified": data.stratified_split,
             "chronological": data.chronological_split}

# Each entry overrides the tuned parameters. None of these were chosen by
# looking at the test set.
CONFIGS: dict[str, dict] = {
    "baseline (tuned)":         {},
    "shallow depth=3":          {"max_depth": 3},
    "depth=4":                  {"max_depth": 4},
    "min_child_weight=10":      {"min_child_weight": 10},
    "min_child_weight=25":      {"min_child_weight": 25},
    "L2 lambda=10":             {"reg_lambda": 10},
    "L1 alpha=5":               {"reg_alpha": 5},
    "gamma=2":                  {"gamma": 2},
    "subsample 0.6":            {"subsample": 0.6, "colsample_bytree": 0.6},
    "slow lr=0.01":             {"learning_rate": 0.01},
    "moderate combined":        {"max_depth": 4, "min_child_weight": 10,
                                 "reg_lambda": 5, "gamma": 1},
    "strong combined":          {"max_depth": 3, "min_child_weight": 25,
                                 "reg_lambda": 10, "gamma": 2,
                                 "subsample": 0.7, "colsample_bytree": 0.7},
    "early stopping":           {"_early_stop": True},
}


def tuned(protocol: str) -> dict:
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


def fit_predict(params: dict, X, y, X_eval, protocol: str):
    """Fit one configuration. Early stopping validates on training data only."""
    p = dict(params)
    early = p.pop("_early_stop", False)
    Xa = np.asarray(X, dtype=float); Xe = np.asarray(X_eval, dtype=float)
    if not early:
        clf = XGBClassifier(random_state=config.RANDOM_SEED, n_jobs=-1,
                            tree_method="hist", eval_metric="aucpr", **p)
        clf.fit(Xa, y)
        return clf.predict_proba(Xa)[:, 1], clf.predict_proba(Xe)[:, 1], clf.n_estimators

    # carve a validation slice out of the TRAINING rows, respecting protocol
    n = len(y)
    if protocol == "chronological":
        cut = int(n * 0.85)
        fit_idx, val_idx = np.arange(cut), np.arange(cut, n)
    else:
        rng = np.random.default_rng(config.RANDOM_SEED)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        vp = rng.choice(pos, max(1, int(len(pos) * .15)), replace=False)
        vn = rng.choice(neg, int(len(neg) * .15), replace=False)
        val_idx = np.concatenate([vp, vn])
        fit_idx = np.setdiff1d(np.arange(n), val_idx)
    p["n_estimators"] = 2000
    clf = XGBClassifier(random_state=config.RANDOM_SEED, n_jobs=-1,
                        tree_method="hist", eval_metric="aucpr",
                        early_stopping_rounds=50, **p)
    clf.fit(Xa[fit_idx], y[fit_idx],
            eval_set=[(Xa[val_idx], y[val_idx])], verbose=False)
    return (clf.predict_proba(Xa)[:, 1], clf.predict_proba(Xe)[:, 1],
            int(clf.best_iteration) + 1)


def main() -> None:
    df = experiment.load_clean()
    rows = []
    for protocol, splitter in SPLITTERS.items():
        Xtr, Xte, ytr, yte = splitter(df)
        ytr, yte = np.asarray(ytr), np.asarray(yte)
        base = tuned(protocol)
        cv = (StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_SEED)
              if protocol == "stratified" else TimeSeriesSplit(5))

        print(f"\n{'=' * 92}\n{protocol.upper()}   (selection by CV only; "
              f"test shown for reference, not used to choose)\n{'=' * 92}")
        print(f"  {'configuration':<22}{'train':>8}{'CV mean':>9}{'CV sd':>7}"
              f"{'train-CV':>10}{'trees':>7}{'test':>8}")
        print("  " + "-" * 71)
        for name, override in CONFIGS.items():
            params = {**base, **override}
            folds = []
            for tr, va in cv.split(Xtr, ytr):
                _, s_va, _ = fit_predict(params, Xtr.iloc[tr], ytr[tr],
                                         Xtr.iloc[va], protocol)
                if ytr[va].sum():
                    folds.append(average_precision_score(ytr[va], s_va))
            s_tr, s_te, n_trees = fit_predict(params, Xtr, ytr, Xte, protocol)
            a_tr = average_precision_score(ytr, s_tr)
            a_te = average_precision_score(yte, s_te)
            cvm, cvs = float(np.mean(folds)), float(np.std(folds))
            rows.append({"protocol": protocol, "config": name,
                         "train_pr_auc": a_tr, "cv_mean": cvm, "cv_sd": cvs,
                         "cv_se": cvs / np.sqrt(len(folds)),
                         "train_minus_cv": a_tr - cvm, "n_trees": n_trees,
                         "test_pr_auc": a_te})
            print(f"  {name:<22}{a_tr:>8.4f}{cvm:>9.4f}{cvs:>7.3f}"
                  f"{a_tr - cvm:>+10.4f}{n_trees:>7}{a_te:>8.4f}", flush=True)

        # one-standard-error rule, on CV only
        d = pd.DataFrame([r for r in rows if r["protocol"] == protocol])
        best = d.loc[d.cv_mean.idxmax()]
        tied = d[d.cv_mean >= best.cv_mean - best.cv_se]
        chosen = tied.loc[tied.train_minus_cv.idxmin()]
        baseline = d[d.config == "baseline (tuned)"].iloc[0]
        print(f"\n  best CV: {best.config} ({best.cv_mean:.4f}, SE {best.cv_se:.4f})")
        print(f"  tied within 1 SE: {', '.join(tied.config)}")
        print(f"  1-SE choice (most regularised of the tied): {chosen.config}")
        print(f"    train/CV gap {baseline.train_minus_cv:+.4f} -> {chosen.train_minus_cv:+.4f}")
        print(f"    test PR-AUC  {baseline.test_pr_auc:.4f} -> {chosen.test_pr_auc:.4f}"
              f"   (revealed only after choosing)")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()
