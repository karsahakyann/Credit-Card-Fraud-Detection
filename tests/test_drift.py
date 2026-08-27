"""Phase 4 guarantees: training data must never postdate test data.

Uses a small synthetic time series rather than the real CSV so the suite
stays fast and runnable without the dataset present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud import drift


@pytest.fixture
def timed():
    """48 simulated hours, uneven volume, ~2% fraud."""
    rng = np.random.default_rng(0)
    n = 4_000
    t = np.sort(rng.uniform(0, 172_792, size=n))
    y = (rng.random(n) < 0.02).astype(int)
    return pd.DataFrame({
        "Time": t,
        "Amount": rng.gamma(2, 30, size=n),
        "V1": rng.normal(size=n),
        "Class": y,
    })


def test_blocks_are_contiguous_and_time_ordered(timed):
    df = drift.ordered_frame(timed)
    blocks = drift.time_blocks(df, n_blocks=6)
    assert set(np.unique(blocks)) == set(range(6))
    # block index must never decrease as time increases
    assert np.all(np.diff(blocks) >= 0)


def test_block_count_respected(timed):
    df = drift.ordered_frame(timed)
    for n in (3, 6, 8):
        assert len(np.unique(drift.time_blocks(df, n_blocks=n))) == n


def test_rejects_degenerate_block_count(timed):
    with pytest.raises(ValueError, match="at least 2"):
        drift.time_blocks(timed, n_blocks=1)


@pytest.mark.parametrize("mode", ["static", "expanding", "sliding"])
def test_training_data_never_postdates_test_data(timed, mode):
    """The Phase 4 leakage guarantee, asserted on real timestamps."""
    df = drift.ordered_frame(timed)
    blocks = drift.time_blocks(df)
    t = df["Time"].to_numpy()

    n_splits = 0
    for test_block, train_idx, test_idx in drift.prequential_splits(
        blocks, mode=mode
    ):
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert t[train_idx].max() <= t[test_idx].min(), (
            f"{mode} split for block {test_block} trains on the future"
        )
        n_splits += 1
    assert n_splits == drift.N_BLOCKS - drift.MIN_TRAIN_BLOCKS


def test_static_training_set_never_grows(timed):
    df = drift.ordered_frame(timed)
    blocks = drift.time_blocks(df)
    sizes = {
        len(train)
        for _, train, _ in drift.prequential_splits(blocks, mode="static")
    }
    assert len(sizes) == 1, "static mode must reuse one fixed training set"


def test_expanding_training_set_grows_monotonically(timed):
    df = drift.ordered_frame(timed)
    blocks = drift.time_blocks(df)
    sizes = [
        len(train)
        for _, train, _ in drift.prequential_splits(blocks, mode="expanding")
    ]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_sliding_window_forgets_old_blocks(timed):
    df = drift.ordered_frame(timed)
    blocks = drift.time_blocks(df)
    for test_block, train_idx, _ in drift.prequential_splits(
        blocks, mode="sliding", window=2
    ):
        used = np.unique(blocks[train_idx])
        assert len(used) <= 2
        assert used.max() == test_block - 1


def test_test_blocks_partition_the_evaluated_period(timed):
    """Every row after the warm-up is tested exactly once."""
    df = drift.ordered_frame(timed)
    blocks = drift.time_blocks(df)
    seen = np.concatenate(
        [test for _, _, test in drift.prequential_splits(blocks, mode="expanding")]
    )
    assert len(seen) == len(np.unique(seen)), "a row was tested twice"
    assert set(seen) == set(np.flatnonzero(blocks >= drift.MIN_TRAIN_BLOCKS))


def test_rejects_impossible_warmup(timed):
    blocks = drift.time_blocks(drift.ordered_frame(timed), n_blocks=3)
    with pytest.raises(ValueError, match="no block to test"):
        list(drift.prequential_splits(blocks, min_train_blocks=3))


def test_block_summary_reports_prior_shift(timed):
    df = drift.ordered_frame(timed)
    blocks = drift.time_blocks(df)
    summary = drift.block_summary(df, blocks)
    assert len(summary) == drift.N_BLOCKS
    assert summary.n.sum() == len(df)
    assert summary.n_fraud.sum() == int(df.Class.sum())
    assert (summary.start_hour.diff().dropna() > 0).all()


def test_feature_drift_excludes_target_and_reference(timed):
    df = drift.ordered_frame(timed)
    blocks = drift.time_blocks(df)
    out = drift.feature_drift(df, blocks, reference_block=0, features=["V1"])
    assert 0 not in out.block.values
    assert set(out.feature) == {"V1"}
    assert out.ks_stat.between(0, 1).all()
