"""Data loading, cleaning and splitting.

Two split strategies are provided deliberately:

* ``stratified_split`` — the conventional random split used by most of the
  literature; preserves the fraud rate in train and test.
* ``chronological_split`` — sorts by ``Time`` and holds out the *last*
  fraction of transactions. This is the realistic protocol (a model can only
  be trained on the past) and is the basis of the Phase 4 concept-drift
  analysis.

Scaling is intentionally NOT done here. It belongs inside model pipelines
(see ``pipelines.py``) so that scalers are fit on training folds only —
fitting them on the full dataset before splitting is a form of data leakage.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw Kaggle dataset and verify its basic shape."""
    path = path or config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place creditcard.csv in data/raw/ (see data/README.md)."
        )
    df = pd.read_csv(path)
    if df.shape != (config.EXPECTED_ROWS, config.EXPECTED_COLUMNS):
        raise ValueError(
            f"Unexpected dataset shape {df.shape}; expected "
            f"({config.EXPECTED_ROWS}, {config.EXPECTED_COLUMNS}). "
            "Is this the right file?"
        )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and assert there are no missing values.

    The dataset has no missing values but contains ~1,081 exact duplicate
    rows; keeping them would let identical transactions appear in both train
    and test sets, which inflates scores.
    """
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        raise ValueError(f"Dataset unexpectedly contains {n_missing} missing values.")
    return df.drop_duplicates().reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return X, y


def stratified_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
):
    """Random split preserving the fraud rate. Returns X_train, X_test, y_train, y_test."""
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )


def chronological_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
):
    """Time-ordered split: train on the earliest transactions, test on the latest.

    Returns X_train, X_test, y_train, y_test.
    """
    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_size))
    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    X_train, y_train = split_features_target(train)
    X_test, y_test = split_features_target(test)
    return X_train, X_test, y_train, y_test
