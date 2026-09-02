"""Phase 5 -- SHAP interpretability for the final XGBoost model.

Refits the Phase 3 winner (XGBoost, no resampling, protocol-tuned) on the
training split, then explains its behaviour on the held-out test set:

* global ranking of feature influence (mean |SHAP|)
* a beeswarm showing direction and spread
* local explanations for three cost-relevant cases -- the most valuable
  fraud caught, the most valuable fraud missed, and the false alarm the
  model was most confident about

Local cases are selected at the *deployed* threshold from
results/threshold_selection.csv, so the explanations describe the model as
it would actually run, not at an arbitrary 0.5.

Usage: ./venv/bin/python scripts/run_shap_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ast
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud import config, data, experiment, explain, resampling

REVIEW_COST = 10.0
SPLITTERS = {
    "stratified": data.stratified_split,
    "chronological": data.chronological_split,
}


def tuned_params(protocol: str) -> dict:
    grid = pd.read_csv(config.RESULTS_DIR / "imbalance_experiment.csv")
    row = grid[(grid.model == "xgboost") & (grid.strategy == "none")
               & (grid.protocol == protocol)].iloc[0]
    raw = json.loads(row.best_params)
    out = {}
    for k, v in raw.items():
        try:
            out[k.replace("model__", "")] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.replace("model__", "")] = v
    return out


def deployed_threshold(protocol: str) -> float:
    sel = pd.read_csv(config.RESULTS_DIR / "threshold_selection.csv")
    row = sel[(sel.model == "xgboost") & (sel.protocol == protocol)
              & (sel.review_cost == REVIEW_COST)]
    return float(row.iloc[0].threshold_selected_on_train)


def bar_figure(imp: pd.DataFrame, out_path: Path, title: str, top_n: int = 15,
               direction: dict[str, float] | None = None):
    """Bars sized by mean |SHAP|, coloured by fraud-conditional direction.

    Colouring by the unconditional mean SHAP would make nearly every bar
    blue: 99.8% of rows are legitimate, so the average push is toward
    "legitimate" for almost every feature regardless of its real role.
    """
    top = imp.head(top_n).iloc[::-1]
    direction = direction or {}
    colors = ["#C44E52" if direction.get(f, 0) > 0 else "#4C72B0"
              for f in top.feature]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.feature, top.mean_abs_shap, color=colors)
    ax.set_xlabel("Mean |SHAP| (average impact on model output)")
    ax.set_title(title)
    ax.text(0.98, 0.02,
            "red = pushes toward fraud on actual frauds\nblue = pushes toward legitimate",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox={"boxstyle": "round", "fc": "white", "ec": "#cccccc"})
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def beeswarm_figure(sv, Xt, feature_names, out_path, title, max_display=15):
    import shap
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, Xt, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def waterfall_figure(contrib: pd.DataFrame, base: float, prob: float,
                     out_path: Path, title: str):
    """Local explanation as a signed contribution bar chart."""
    c = contrib.iloc[::-1]
    colors = ["#C44E52" if s > 0 else "#4C72B0" for s in c.shap]
    labels = [f"{f} = {v:,.2f}" for f, v in zip(c.feature, c.value)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(labels, c.shap, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (log-odds; + pushes toward fraud)")
    ax.set_title(f"{title}\nmodel probability {prob:.3f}  ·  base rate "
                 f"{1/(1+np.exp(-base)):.4f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = experiment.load_clean()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_imp, all_cond = [], []

    for protocol, splitter in SPLITTERS.items():
        X_train, X_test, y_train, y_test = splitter(df)
        y_te = np.asarray(y_test)
        amounts = X_test["Amount"].to_numpy(dtype=float)
        params = tuned_params(protocol)
        thr = deployed_threshold(protocol)

        print(f"\n{'=' * 74}\n{protocol.upper()}  "
              f"(deployed threshold {thr:.2f} at EUR {REVIEW_COST:.0f} review)\n{'=' * 74}")

        pipe = resampling.make_estimator(
            "xgboost", "none", params=params, y_train=np.asarray(y_train),
        )
        pipe.fit(X_train, np.asarray(y_train))
        scores = pipe.predict_proba(X_test)[:, 1]

        sv, base = explain.tree_shap_values(pipe, X_test)
        Xt = explain.transform_for_explanation(pipe, X_test)
        names = list(X_test.columns)

        imp = explain.global_importance(sv, names)
        imp.insert(0, "protocol", protocol)
        all_imp.append(imp)

        print("  Top 10 features by mean |SHAP|:")
        for _, r in imp.head(10).iterrows():
            arrow = "toward fraud" if r.mean_shap > 0 else "toward legit"
            print(f"    {int(r['rank']):>2}. {r.feature:<8} "
                  f"{r.mean_abs_shap:>7.4f}  ({r.share_pct:>4.1f}% of total)  "
                  f"mean {arrow}")

        time_rank = int(imp[imp.feature == "Time"].iloc[0]["rank"])
        time_share = float(imp[imp.feature == "Time"].iloc[0].share_pct)
        amt_rank = int(imp[imp.feature == "Amount"].iloc[0]["rank"])
        print(f"\n  Interpretable features: Time rank {time_rank} "
              f"({time_share:.1f}%), Amount rank {amt_rank}")

        cond = explain.class_conditional_importance(sv, names, y_te)
        cond.insert(0, "protocol", protocol)
        all_cond.append(cond)
        print("\n  Direction among ACTUAL frauds (top 5 by separation):")
        for _, r in cond.head(5).iterrows():
            print(f"    {r.feature:<8} fraud {r.mean_shap_fraud:>+7.3f}   "
                  f"legit {r.mean_shap_legit:>+7.3f}   sep {r.separation:>+6.3f}")

        bar_figure(imp, config.FIGURES_DIR / f"phase5_shap_bar_{protocol}.png",
                   f"Global feature influence -- XGBoost, {protocol}",
                   direction=dict(zip(cond.feature, cond.mean_shap_fraud)))
        beeswarm_figure(sv, Xt, names,
                        config.FIGURES_DIR / f"phase5_shap_beeswarm_{protocol}.png",
                        f"SHAP beeswarm -- XGBoost, {protocol}")

        cases = explain.pick_archetypes(y_te, scores, amounts, thr)
        titles = {
            "caught_fraud": "Caught fraud (highest value correctly flagged)",
            "missed_fraud": "Missed fraud (most expensive miss)",
            "false_alarm": "False alarm (most confident wrong flag)",
        }
        print()
        for case, idx in cases.items():
            contrib = explain.explain_case(sv, base, X_test, idx)
            print(f"  {titles[case]}  ->  amount EUR {amounts[idx]:,.2f}, "
                  f"score {scores[idx]:.3f}")
            for _, r in contrib.head(4).iterrows():
                print(f"      {r.feature:<8} = {r.value:>10,.2f}   "
                      f"SHAP {r.shap:>+7.3f}  {r.pushes}")
            waterfall_figure(
                contrib, base, scores[idx],
                config.FIGURES_DIR / f"phase5_shap_case_{case}_{protocol}.png",
                f"{titles[case]} -- {protocol}",
            )
            contrib.insert(0, "protocol", protocol)
            contrib.insert(0, "case", case)
            contrib.to_csv(
                config.RESULTS_DIR / f"shap_case_{case}_{protocol}.csv", index=False,
            )

    pd.concat(all_imp, ignore_index=True).to_csv(
        config.RESULTS_DIR / "shap_global_importance.csv", index=False)
    pd.concat(all_cond, ignore_index=True).to_csv(
        config.RESULTS_DIR / "shap_class_conditional.csv", index=False)
    print(f"\n[Saved] {config.RESULTS_DIR / 'shap_global_importance.csv'}")


if __name__ == "__main__":
    main()
