"""Figures for the overfitting analysis, regenerated from results/overfitting/.

Kept as a script rather than produced inline so the figures can never drift
out of step with the CSVs they depict. The first versions were made with a
one-off snippet, and regularising the final models left them showing the
pre-regularisation numbers.

Usage: ./venv/bin/python scripts/plot_overfitting.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud import config, final_params

D = config.RESULTS_DIR / "overfitting"
F = config.FIGURES_DIR
C = {"xgboost": "#175cd3", "random_forest": "#067647", "logistic_regression": "#b42318"}
L = {"xgboost": "XGBoost", "random_forest": "Random Forest",
     "logistic_regression": "Logistic Regression"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})


def learning_curve():
    lc = pd.read_csv(D / "learning_curve.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m in L:
        d = lc[lc.model == m]
        ax.plot(d.fraction * 100, d.train_pr_auc, "--", color=C[m], lw=1.3, alpha=.8)
        ax.plot(d.fraction * 100, d.test_pr_auc, "-o", color=C[m], lw=1.8, ms=4, label=L[m])
    ax.set_xlabel("Training data used (%)"); ax.set_ylabel("PR-AUC")
    ax.set_title("Learning curves (regularised) - dashed: training, solid: held-out test")
    ax.set_ylim(.6, 1.02); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(F / "overfitting_learning_curve.png", dpi=160)
    plt.close(fig)


def boosting_curve():
    bc = pd.read_csv(D / "boosting_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, p in zip(axes, ("stratified", "chronological")):
        d = bc[bc.protocol == p]
        tuned = final_params.params("xgboost", p)["n_estimators"]
        ax.plot(d.n_trees, d.train_pr_auc, color="#98a2b3", lw=1.4, label="training")
        ax.plot(d.n_trees, d.heldout_pr_auc, color="#175cd3", lw=1.8, label="held-out")
        ax.axvline(tuned, color="#101828", ls=":", lw=1)
        ax.text(tuned + 15, .755, f"selected\n{tuned}", fontsize=8, color="#475467")
        ax.set_title(p.capitalize()); ax.set_xlabel("Number of trees")
    axes[0].set_ylabel("PR-AUC"); axes[0].set_ylim(.74, 1.01)
    axes[0].legend(frameon=False)
    fig.suptitle("Regularised XGBoost: training no longer saturates, "
                 "held-out does not decline", fontsize=10)
    fig.tight_layout(); fig.savefig(F / "overfitting_boosting_curve.png", dpi=160)
    plt.close(fig)


def generalisation_gap():
    g = pd.read_csv(D / "generalisation_gap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, p in zip(axes, ("stratified", "chronological")):
        d = g[g.protocol == p].set_index("model").loc[list(L)]
        x = np.arange(3); w = .26
        ax.bar(x - w, d.train_pr_auc, w, color="#d0d5dd", label="training")
        ax.bar(x, d.oof_pr_auc, w, color="#84adff", label="out-of-fold")
        ax.bar(x + w, d.test_pr_auc, w, color="#175cd3", label="test")
        ax.set_xticks(x); ax.set_xticklabels([L[m] for m in L], fontsize=8)
        ax.set_title(p.capitalize())
    axes[0].set_ylabel("PR-AUC"); axes[0].set_ylim(.5, 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("After regularisation: training, out-of-fold and test scores sit close together",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(F / "overfitting_generalisation_gap.png", dpi=160)
    plt.close(fig)


def before_after(pre: Path | None):
    """Train/test gap before and after regularisation, if a snapshot exists."""
    if pre is None or not pre.exists():
        return
    old = pd.read_csv(pre); new = pd.read_csv(D / "generalisation_gap.csv")
    rows = []
    for _, n in new.iterrows():
        o = old[(old.protocol == n.protocol) & (old.model == n.model)]
        if o.empty or n.model == "logistic_regression":
            continue
        o = o.iloc[0]
        rows.append((f"{L[n.model]}\n{n.protocol}", o.train_minus_test, n.train_minus_test))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rows)); w = .38
    ax.bar(x - w/2, [r[1] for r in rows], w, color="#d0d5dd", label="before (Phase 3 parameters)")
    ax.bar(x + w/2, [r[2] for r in rows], w, color="#175cd3", label="after (regularised)")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("Train minus test PR-AUC"); ax.legend(frameon=False)
    ax.set_title("Regularisation roughly halves the train/test gap")
    fig.tight_layout(); fig.savefig(F / "overfitting_before_after.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    pre = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    learning_curve(); boosting_curve(); generalisation_gap(); before_after(pre)
    print("overfitting figures written")
