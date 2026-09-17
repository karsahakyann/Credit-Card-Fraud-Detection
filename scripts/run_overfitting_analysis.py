"""Do the models overfit? Four independent checks.

Overfitting cannot be diagnosed from one number, and the obvious number --
training score versus test score -- is actively misleading for tree
ensembles, which routinely reach a near-perfect training score while still
generalising well. So this looks from four directions, each answering a
different question:

1. Generalisation gap   train vs test PR-AUC. How much do the models
                        memorise their own training rows?
2. Honest in-sample     out-of-fold (OOF) vs test. OOF scores come from
                        folds the model never trained on, so they estimate
                        performance on the training *distribution* without
                        memorisation. OOF close to test means the model
                        generalises as well as cross-validation predicted.
3. Learning curve       train and test PR-AUC as training data grows. Curves
                        that converge indicate a well-fitted model; a test
                        curve still climbing means it is data-limited.
4. Boosting curve       XGBoost train vs held-out PR-AUC as trees are added.
                        A held-out curve that peaks and falls is overfitting
                        in its classic form; a plateau is not.

Reading the result: a large train/test gap on its own is expected for trees
and is *not* evidence of harmful overfitting. What would be evidence is OOF
materially above test (the model fits its training distribution better than
new data), or a held-out curve that turns down as capacity grows.

Usage: ./venv/bin/python scripts/run_overfitting_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from fraud import config, data, experiment, resampling, final_params

OUT = config.RESULTS_DIR / "overfitting"
MODELS = ("xgboost", "random_forest", "logistic_regression")
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
SPLITTERS = {"stratified": data.stratified_split,
             "chronological": data.chronological_split}


def params_for(model: str, protocol: str) -> dict:
    """Final parameters, from the single source of truth."""
    return final_params.params(model, protocol)


def fit(model, protocol, X, y):
    est = resampling.make_estimator(
        model, "none", params=params_for(model, protocol), y_train=y)
    est.fit(X, y)
    return est


def oof_pr_auc(model, protocol, X, y) -> float:
    cv = (StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_SEED)
          if protocol == "stratified" else TimeSeriesSplit(5))
    oof = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y):
        est = fit(model, protocol, X.iloc[tr], y[tr])
        oof[va] = est.predict_proba(X.iloc[va])[:, 1]
    m = ~np.isnan(oof)
    return float(average_precision_score(y[m], oof[m]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = experiment.load_clean()

    # ---- checks 1 and 2 -------------------------------------------------
    gap_rows = []
    print("CHECKS 1 & 2 -- generalisation gap and honest in-sample estimate\n")
    print(f"  {'protocol':<14}{'model':<21}{'train':>8}{'OOF':>8}{'test':>8}"
          f"{'train-test':>12}{'OOF-test':>10}")
    print("  " + "-" * 79)
    for protocol, splitter in SPLITTERS.items():
        Xtr, Xte, ytr, yte = splitter(df)
        ytr, yte = np.asarray(ytr), np.asarray(yte)
        for model in MODELS:
            est = fit(model, protocol, Xtr, ytr)
            tr = average_precision_score(ytr, est.predict_proba(Xtr)[:, 1])
            te = average_precision_score(yte, est.predict_proba(Xte)[:, 1])
            oof = oof_pr_auc(model, protocol, Xtr, ytr)
            gap_rows.append({"protocol": protocol, "model": model,
                             "train_pr_auc": tr, "oof_pr_auc": oof,
                             "test_pr_auc": te, "train_minus_test": tr - te,
                             "oof_minus_test": oof - te})
            print(f"  {protocol:<14}{model:<21}{tr:>8.4f}{oof:>8.4f}{te:>8.4f}"
                  f"{tr - te:>+12.4f}{oof - te:>+10.4f}", flush=True)
    pd.DataFrame(gap_rows).to_csv(OUT / "generalisation_gap.csv", index=False)

    # ---- check 3: learning curve (stratified, size effect only) ---------
    print("\nCHECK 3 -- learning curve (stratified; random subsamples isolate "
          "the effect of size)\n")
    Xtr, Xte, ytr, yte = data.stratified_split(df)
    ytr, yte = np.asarray(ytr), np.asarray(yte)
    rng = np.random.default_rng(config.RANDOM_SEED)
    lc_rows = []
    print(f"  {'model':<21}" + "".join(f"{int(f*100):>9}%" for f in FRACTIONS))
    for model in MODELS:
        tr_line, te_line = [], []
        for frac in FRACTIONS:
            if frac < 1.0:
                pos = np.flatnonzero(ytr == 1); neg = np.flatnonzero(ytr == 0)
                idx = np.concatenate([
                    rng.choice(pos, max(10, int(len(pos) * frac)), replace=False),
                    rng.choice(neg, int(len(neg) * frac), replace=False)])
            else:
                idx = np.arange(len(ytr))
            est = fit(model, "stratified", Xtr.iloc[idx], ytr[idx])
            a_tr = average_precision_score(ytr[idx], est.predict_proba(Xtr.iloc[idx])[:, 1])
            a_te = average_precision_score(yte, est.predict_proba(Xte)[:, 1])
            tr_line.append(a_tr); te_line.append(a_te)
            lc_rows.append({"model": model, "fraction": frac,
                            "n_train": len(idx), "train_pr_auc": a_tr,
                            "test_pr_auc": a_te})
        print(f"  {model + ' train':<21}" + "".join(f"{v:>10.4f}" for v in tr_line))
        print(f"  {model + ' test':<21}" + "".join(f"{v:>10.4f}" for v in te_line))
        print(f"  {'  gap':<21}" + "".join(f"{a-b:>+10.4f}" for a, b in zip(tr_line, te_line)),
              flush=True)
    pd.DataFrame(lc_rows).to_csv(OUT / "learning_curve.csv", index=False)

    # ---- check 4: XGBoost boosting curve --------------------------------
    print("\nCHECK 4 -- XGBoost train vs held-out PR-AUC as trees are added\n")
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    bc_rows = []
    for protocol, splitter in SPLITTERS.items():
        Xtr, Xte, ytr, yte = splitter(df)
        ytr, yte = np.asarray(ytr), np.asarray(yte)
        sc = StandardScaler().fit(Xtr)
        p = params_for("xgboost", protocol)
        p["n_estimators"] = 1200            # push well past the tuned value
        clf = XGBClassifier(random_state=config.RANDOM_SEED, n_jobs=-1,
                            tree_method="hist", eval_metric="aucpr", **p)
        clf.fit(sc.transform(Xtr), ytr,
                eval_set=[(sc.transform(Xtr), ytr), (sc.transform(Xte), yte)],
                verbose=False)
        ev = clf.evals_result()
        tr_curve = np.array(ev["validation_0"]["aucpr"])
        te_curve = np.array(ev["validation_1"]["aucpr"])
        tuned = params_for("xgboost", protocol)["n_estimators"]
        best = int(np.argmax(te_curve)) + 1
        for i in range(len(te_curve)):
            bc_rows.append({"protocol": protocol, "n_trees": i + 1,
                            "train_pr_auc": tr_curve[i], "heldout_pr_auc": te_curve[i]})
        print(f"  {protocol}:")
        # sorted/deduplicated: the tuned count can coincide with a checkpoint
        for n in sorted({50, 100, 200, tuned, 600, 900, 1200}):
            n = min(n, len(te_curve))
            mark = "  <- tuned" if n == tuned else ""
            print(f"    {n:>5} trees   train {tr_curve[n-1]:.4f}   "
                  f"held-out {te_curve[n-1]:.4f}{mark}")
        drop = te_curve.max() - te_curve[-1]
        print(f"    held-out peaks at {best} trees ({te_curve.max():.4f}); "
              f"at 1200 it is {te_curve[-1]:.4f} (down {drop:.4f})\n", flush=True)
    pd.DataFrame(bc_rows).to_csv(OUT / "boosting_curve.csv", index=False)
    print(f"[Saved] {OUT}/")


if __name__ == "__main__":
    main()
