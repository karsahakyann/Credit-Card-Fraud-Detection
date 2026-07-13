"""Smoke tests for the data module.

These run on a synthetic frame shaped like the real dataset, so they pass
before creditcard.csv has been downloaded. If the real dataset is present,
an extra integration test validates it too.
"""

import numpy as np
import pandas as pd
import pytest

from fraud import config, data


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """1,000 rows shaped like creditcard.csv with a 2% fraud rate."""
    rng = np.random.default_rng(0)
    n = 1_000
    df = pd.DataFrame(
        {"Time": np.sort(rng.integers(0, 172_800, n)).astype(float)}
    )
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n)
    df["Amount"] = rng.exponential(80, n).round(2)
    df["Class"] = (rng.random(n) < 0.02).astype(int)
    # inject exact duplicates that clean() must remove
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df


def test_clean_removes_duplicates_and_keeps_no_nans(synthetic_df):
    cleaned = data.clean(synthetic_df)
    assert cleaned.duplicated().sum() == 0
    assert len(cleaned) == len(synthetic_df) - 5
    assert cleaned.isna().sum().sum() == 0


def test_stratified_split_preserves_fraud_rate(synthetic_df):
    df = data.clean(synthetic_df)
    X_train, X_test, y_train, y_test = data.stratified_split(df)
    overall = df[config.TARGET].mean()
    assert y_train.mean() == pytest.approx(overall, abs=0.005)
    assert y_test.mean() == pytest.approx(overall, abs=0.005)
    assert config.TARGET not in X_train.columns


def test_chronological_split_is_time_ordered(synthetic_df):
    df = data.clean(synthetic_df)
    X_train, X_test, y_train, y_test = data.chronological_split(df)
    # every training transaction happens before (or at) every test transaction
    assert X_train["Time"].max() <= X_test["Time"].min()
    assert len(X_train) + len(X_test) == len(df)


def test_split_sizes(synthetic_df):
    df = data.clean(synthetic_df)
    _, X_test, _, _ = data.chronological_split(df, test_size=0.2)
    assert len(X_test) == pytest.approx(0.2 * len(df), rel=0.01)


@pytest.mark.skipif(
    not config.RAW_DATA_PATH.exists(), reason="real dataset not downloaded yet"
)
def test_real_dataset_sanity():
    df = data.load_raw()
    assert df.shape == (config.EXPECTED_ROWS, config.EXPECTED_COLUMNS)
    assert int(df[config.TARGET].sum()) == config.EXPECTED_FRAUDS
    cleaned = data.clean(df)
    assert len(cleaned) < len(df)  # duplicates removed
