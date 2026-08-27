"""Phase 5 cost-model tests: the arithmetic the € claims rest on."""

from __future__ import annotations

import numpy as np
import pytest

from fraud import costs


@pytest.fixture
def tiny():
    """4 transactions: fraud of 100 and 50, legit of 10 and 20."""
    y = np.array([1, 1, 0, 0])
    amounts = np.array([100.0, 50.0, 10.0, 20.0])
    return y, amounts


def test_catch_everything_pays_reviews_only(tiny):
    y, amounts = tiny
    c = costs.prediction_cost(y, np.array([1, 1, 1, 1]), amounts, review_cost=5)
    assert c["total_cost"] == 20.0          # 4 reviews, no losses
    assert c["fn_loss_total"] == 0.0


def test_catch_nothing_loses_the_fraud_amounts(tiny):
    y, amounts = tiny
    c = costs.prediction_cost(y, np.zeros(4), amounts, review_cost=5)
    assert c["total_cost"] == 150.0         # both frauds lost, no reviews
    assert c["review_cost_total"] == 0.0


def test_mixed_prediction_prices_each_cell(tiny):
    y, amounts = tiny
    # catch the 100 fraud, miss the 50, one false alarm
    c = costs.prediction_cost(y, np.array([1, 0, 1, 0]), amounts, review_cost=5)
    assert c["tp"] == 1 and c["fp"] == 1 and c["fn"] == 1
    assert c["total_cost"] == 5 + 5 + 50.0


def test_missed_fraud_priced_by_its_own_amount(tiny):
    y, amounts = tiny
    miss_big = costs.prediction_cost(y, np.array([0, 1, 0, 0]), amounts, 0)
    miss_small = costs.prediction_cost(y, np.array([1, 0, 0, 0]), amounts, 0)
    assert miss_big["fn_loss_total"] == 100.0
    assert miss_small["fn_loss_total"] == 50.0


def test_baselines_bracket(tiny):
    y, amounts = tiny
    b = costs.baseline_costs(y, amounts, review_cost=5)
    assert b["flag_nothing"] == 150.0
    assert b["flag_everything"] == 20.0
    assert b["best_baseline"] == 20.0


def test_input_validation(tiny):
    y, amounts = tiny
    with pytest.raises(ValueError, match="same length"):
        costs.prediction_cost(y, np.array([1, 0]), amounts, 5)
    with pytest.raises(ValueError, match="non-negative"):
        costs.prediction_cost(y, np.zeros(4), amounts, -1)


def test_optimal_threshold_finds_a_separable_optimum():
    """Perfectly separable scores: any threshold inside the gap is optimal;
    the reported cost must equal review_cost per fraud caught."""
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.05).astype(int)
    scores = np.where(y == 1, 0.9, 0.1)
    amounts = np.full(n, 100.0)
    res = costs.optimal_threshold(y, scores, amounts, review_cost=5)
    assert 0.1 < res["best_threshold"] <= 0.9
    assert res["cost_at_best"] == 5.0 * y.sum()
    assert res["fn_at_best"] == 0


def test_expensive_reviews_raise_the_optimal_threshold():
    """As reviews get pricier, the optimum should flag less, never more."""
    rng = np.random.default_rng(1)
    n = 5000
    y = (rng.random(n) < 0.02).astype(int)
    scores = np.clip(0.55 * y + rng.normal(0.25, 0.18, n), 0, 1)
    amounts = np.abs(rng.gamma(2, 60, n))
    t_cheap = costs.optimal_threshold(y, scores, amounts, review_cost=1)
    t_dear = costs.optimal_threshold(y, scores, amounts, review_cost=50)
    assert t_dear["best_threshold"] >= t_cheap["best_threshold"]


def test_curve_matches_pointwise_cost(tiny):
    y, amounts = tiny
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    curve = costs.threshold_cost_curve(y, scores, amounts, review_cost=5)
    row = curve[curve.threshold == 0.50].iloc[0]
    direct = costs.prediction_cost(y, scores >= 0.5, amounts, 5)
    assert row.total_cost == direct["total_cost"]


def test_gain_from_tuning_is_consistent(tiny):
    y, amounts = tiny
    scores = np.array([0.9, 0.45, 0.6, 0.1])
    res = costs.optimal_threshold(y, scores, amounts, review_cost=5)
    assert res["gain_from_tuning"] == pytest.approx(
        res["cost_at_default_0_5"] - res["cost_at_best"]
    )
    assert res["gain_from_tuning"] >= 0   # optimum can never lose to 0.5
