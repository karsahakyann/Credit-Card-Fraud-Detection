"""Standard evaluation report used by every experiment in the dissertation.

Headline metrics are the imbalance-appropriate ones: precision, recall, F1,
PR-AUC (average precision) and MCC. Accuracy and ROC-AUC are recorded for
comparison with the literature but must never be used as headline numbers —
that is one of the methodological critiques this dissertation makes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def _scores(model, X) -> np.ndarray:
    """Continuous fraud scores for ranking metrics (PR-AUC, ROC-AUC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Return the full metric report for a fitted model as a dict."""
    scores = _scores(model, X_test)
    if hasattr(model, "predict_proba"):
        y_pred = (scores >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        # headline metrics
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, scores),
        "mcc": matthews_corrcoef(y_test, y_pred),
        # reported for literature comparison only
        "roc_auc": roc_auc_score(y_test, scores),
        "accuracy": accuracy_score(y_test, y_pred),
        # confusion matrix cells
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": threshold,
        "n_test": int(len(y_test)),
        "n_test_fraud": int(y_test.sum()),
    }


def log_result(
    metrics: dict,
    model_name: str,
    split: str,
    imbalance_strategy: str = "none",
    notes: str = "",
    csv_path=None,
) -> pd.DataFrame:
    """Append one experiment result to results/metrics.csv and return the row."""
    csv_path = csv_path or config.METRICS_CSV
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "split": split,
        "imbalance_strategy": imbalance_strategy,
        "notes": notes,
        **metrics,
    }
    df = pd.DataFrame([row])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)
    return df


def report(metrics: dict, title: str = "") -> None:
    """Pretty-print a metric dict in a consistent order."""
    if title:
        print(title)
        print("-" * len(title))
    order = ["precision", "recall", "f1", "pr_auc", "mcc", "roc_auc", "accuracy"]
    for key in order:
        print(f"{key:>10}: {metrics[key]:.4f}")
    print(
        f"{'confusion':>10}: TP={metrics['tp']}  FP={metrics['fp']}  "
        f"FN={metrics['fn']}  TN={metrics['tn']}"
    )


def plot_pr_curve(model, X_test, y_test, label: str, ax=None, save_as: str | None = None):
    """Plot the precision-recall curve; optionally save under results/figures."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_test, _scores(model, X_test), name=label, ax=ax
    )
    ax.set_title("Precision-Recall curve")
    if save_as:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(config.FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return ax


def plot_confusion(model, X_test, y_test, title: str = "", save_as: str | None = None):
    """Plot the confusion matrix; optionally save under results/figures."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Legit", "Fraud"], ax=ax, colorbar=False
    )
    ax.set_title(title or "Confusion matrix")
    if save_as:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return ax
