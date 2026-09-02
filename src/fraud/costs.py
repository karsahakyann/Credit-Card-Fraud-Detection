"""Cost-sensitive evaluation (Phase 5).

PR-AUC says how well a model *ranks* transactions; it says nothing about
whether the operating threshold makes financial sense. This module prices
the confusion matrix and finds the threshold that minimises expected cost.

Cost model
----------
Every flagged transaction -- true positive or false positive -- is reviewed
by a human, at a fixed ``review_cost``. Every missed fraud costs the
transaction's own ``Amount`` (the money actually lost). Catching a fraud
therefore saves its amount but still pays one review.

    total_cost = review_cost * (TP + FP) + sum(Amount of FN)

Two degenerate baselines bracket any sensible model:

* ``flag nothing``   -- cost = sum of all fraud amounts (lose everything)
* ``flag everything``-- cost = review_cost * N        (review everything)

A model earns its keep only between those brackets, and "savings" here
always means savings against the better of the two -- a deliberately harsh
baseline, because beating the *worse* one is trivial.

The review cost is a business parameter, not a data quantity, so every
result is reported across a range of review costs rather than at one
assumed number.

Phase 3 context: resampling was shown to move the operating point along an
essentially unchanged PR curve. This module is the principled version of
that move -- pick the point by cost, not by resampling side-effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Review costs (EUR) to sweep. Spans "cheap automated check" to "expensive
# manual investigation"; results are reported at each.
REVIEW_COSTS = (2.0, 5.0, 10.0, 25.0)

# Threshold grid: fine enough that the optimum is not an artefact of step
# size, coarse enough to stay instant on 57k rows.
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def prediction_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
) -> dict:
    """Price one set of binary predictions."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    amounts = np.asarray(amounts, dtype=float)
    if not (len(y_true) == len(y_pred) == len(amounts)):
        raise ValueError("y_true, y_pred and amounts must be the same length")
    if review_cost < 0:
        raise ValueError("review_cost must be non-negative")

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    fn_loss = float(amounts[y_true & ~y_pred].sum())
    review_total = review_cost * (tp + fp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "review_cost_total": review_total,
        "fn_loss_total": fn_loss,
        "total_cost": review_total + fn_loss,
    }


def baseline_costs(
    y_true: np.ndarray, amounts: np.ndarray, review_cost: float,
) -> dict:
    """The two no-model brackets: flag nothing / flag everything."""
    y_true = np.asarray(y_true).astype(bool)
    amounts = np.asarray(amounts, dtype=float)
    flag_nothing = float(amounts[y_true].sum())
    flag_everything = review_cost * len(y_true)
    return {
        "flag_nothing": flag_nothing,
        "flag_everything": flag_everything,
        "best_baseline": min(flag_nothing, flag_everything),
    }


def threshold_cost_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
    thresholds: np.ndarray = THRESHOLDS,
) -> pd.DataFrame:
    """Total cost at every threshold, one row per threshold."""
    rows = []
    for t in thresholds:
        c = prediction_cost(y_true, scores >= t, amounts, review_cost)
        rows.append({"threshold": float(t), **c})
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
    thresholds: np.ndarray = THRESHOLDS,
) -> dict:
    """Cost-minimising threshold, with context for reporting.

    Returns the optimum plus the default-0.5 cost, the baseline bracket,
    and savings versus the *better* baseline at both thresholds.
    """
    curve = threshold_cost_curve(y_true, scores, amounts, review_cost, thresholds)
    best = curve.loc[curve.total_cost.idxmin()]
    at_default = prediction_cost(y_true, scores >= 0.5, amounts, review_cost)
    base = baseline_costs(y_true, amounts, review_cost)
    return {
        "review_cost": review_cost,
        "best_threshold": float(best.threshold),
        "cost_at_best": float(best.total_cost),
        "cost_at_default_0_5": float(at_default["total_cost"]),
        "savings_vs_baseline_at_best": base["best_baseline"] - float(best.total_cost),
        "savings_vs_baseline_at_default": base["best_baseline"] - float(at_default["total_cost"]),
        "gain_from_tuning": float(at_default["total_cost"]) - float(best.total_cost),
        **{f"baseline_{k}": v for k, v in base.items()},
        "tp_at_best": int(best.tp),
        "fp_at_best": int(best.fp),
        "fn_at_best": int(best.fn),
    }
