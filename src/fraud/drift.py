"""Temporal blocks and drift measurement (Phase 4).

The dissertation asks whether models degrade over time and whether periodic
retraining recovers the loss. Two facts about this dataset constrain how far
that question can be pushed, and both are stated here rather than buried:

1. **The data spans 48 hours.** Concept drift in the usual sense -- fraud
   tactics evolving over months -- cannot be observed in two days. What can
   be measured is temporal generalisation across a short horizon, which is a
   weaker but honest claim.
2. **The non-stationarity that does exist is largely diurnal.** Transaction
   volume swings by a factor of four between night and day, and the fraud
   *rate* swings by roughly five (about 0.46% in the quietest block against
   0.095% in the busiest). That is prior-probability shift driven by the
   day-night cycle, not evolving fraud behaviour.

Both points belong in the write-up as limitations. The machinery here is
still worth building: it is the correct protocol, it produces a real answer
for this dataset, and it would transfer unchanged to a longer series.

Leakage rule for this phase: every training row must precede every test row
in time. ``prequential_splits`` guarantees it by construction and
``tests/test_drift.py`` asserts it on real timestamps.
"""

from __future__ import annotations

from typing import Iterator, Literal

import numpy as np
import pandas as pd

from . import config

N_BLOCKS = 6            # ~8 hours each; keeps >=59 frauds per block
MIN_TRAIN_BLOCKS = 2    # first evaluation trains on 16h of history
SLIDING_WINDOW = 2      # blocks retained by the sliding-window strategy

Mode = Literal["static", "expanding", "sliding"]


def ordered_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the frame sorted by ``Time``, index reset.

    Every function here assumes positional index == temporal order.
    """
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def time_blocks(df: pd.DataFrame, n_blocks: int = N_BLOCKS) -> np.ndarray:
    """Assign each row to one of ``n_blocks`` equal-*duration* blocks.

    Equal duration rather than equal count, deliberately: the diurnal volume
    swing is part of what we are measuring, and equal-count blocks would hide
    it by stretching quiet periods.
    """
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    t = df["Time"].to_numpy()
    edges = np.linspace(t.min(), t.max(), n_blocks + 1)
    # digitize against interior edges only, so the final block is closed and
    # the maximum timestamp does not spill into a phantom block n_blocks+1.
    return np.digitize(t, edges[1:-1])


def block_summary(df: pd.DataFrame, blocks: np.ndarray) -> pd.DataFrame:
    """Per-block size, fraud count and fraud rate -- the prior-shift table."""
    y = df[config.TARGET].to_numpy()
    t = df["Time"].to_numpy()
    rows = []
    for b in np.unique(blocks):
        m = blocks == b
        n, n_fraud = int(m.sum()), int(y[m].sum())
        rows.append({
            "block": int(b),
            "n": n,
            "n_fraud": n_fraud,
            "fraud_rate_pct": 100 * n_fraud / n if n else np.nan,
            "start_hour": float(t[m].min() / 3600),
            "end_hour": float(t[m].max() / 3600),
        })
    return pd.DataFrame(rows)


def prequential_splits(
    blocks: np.ndarray,
    mode: Mode = "expanding",
    min_train_blocks: int = MIN_TRAIN_BLOCKS,
    window: int = SLIDING_WINDOW,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(test_block, train_positions, test_positions)`` forward in time.

    Retraining strategies compared by Phase 4:

    ``static``
        Train once on the first ``min_train_blocks`` and never update. This
        is the decay baseline -- the model a team deploys and forgets.
    ``expanding``
        Retrain before each block on *all* history so far.
    ``sliding``
        Retrain before each block on the most recent ``window`` blocks only,
        deliberately forgetting older data.

    In every mode the training positions come strictly from blocks earlier
    than the test block, so no future information can reach a fitted model.
    """
    if mode not in ("static", "expanding", "sliding"):
        raise ValueError(f"Unknown mode {mode!r}")

    unique_blocks = np.unique(blocks)
    if min_train_blocks >= len(unique_blocks):
        raise ValueError(
            f"min_train_blocks={min_train_blocks} leaves no block to test on "
            f"({len(unique_blocks)} blocks available)"
        )

    for test_block in unique_blocks[min_train_blocks:]:
        if mode == "static":
            train_mask = blocks < min_train_blocks
        elif mode == "expanding":
            train_mask = blocks < test_block
        else:
            train_mask = (blocks < test_block) & (blocks >= test_block - window)

        test_mask = blocks == test_block
        yield (
            int(test_block),
            np.flatnonzero(train_mask),
            np.flatnonzero(test_mask),
        )


def feature_drift(
    df: pd.DataFrame,
    blocks: np.ndarray,
    reference_block: int = 0,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Two-sample KS statistic per feature, each block against a reference.

    A large statistic means that feature's marginal distribution has moved
    away from the reference period. Computed on legitimate transactions only:
    including frauds would confound covariate shift with the prior shift that
    ``block_summary`` already reports.
    """
    from scipy.stats import ks_2samp

    if features is None:
        # "Time" is excluded deliberately: blocks are *defined* by Time, so its
        # KS statistic against any other block is 1.0 by construction. Reporting
        # it as drift would be a tautology, not a finding.
        features = [c for c in df.columns if c not in (config.TARGET, "Time")]

    legit = df[config.TARGET].to_numpy() == 0
    ref_mask = (blocks == reference_block) & legit

    rows = []
    for b in np.unique(blocks):
        if b == reference_block:
            continue
        cur_mask = (blocks == b) & legit
        for feat in features:
            stat, p = ks_2samp(
                df[feat].to_numpy()[ref_mask], df[feat].to_numpy()[cur_mask]
            )
            rows.append({
                "block": int(b),
                "feature": feat,
                "ks_stat": float(stat),
                "p_value": float(p),
            })
    return pd.DataFrame(rows)
