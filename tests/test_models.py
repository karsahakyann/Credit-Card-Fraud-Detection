"""Tests for the tree-ensemble factories in models.py.

The XGBoost fit runs in a clean subprocess: the pytest process loads
pandas (test_data) and torch (test_dnn), and fitting XGBoost in a process
holding all three segfaults on macOS — see the note in fraud/models.py.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from fraud.models import make_rf_pipeline


@pytest.fixture
def synthetic_imbalanced():
    rng = np.random.default_rng(0)
    n = 3_000
    y = (rng.random(n) < 0.02).astype(int)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    X[y == 1, :4] += 2.5
    return X, y


def test_rf_pipeline_fits_and_scores(synthetic_imbalanced):
    X, y = synthetic_imbalanced
    pipe = make_rf_pipeline(n_estimators=20)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    from sklearn.metrics import average_precision_score
    assert average_precision_score(y, proba[:, 1]) > 0.5


def test_xgb_pipeline_fits_in_clean_process():
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from fraud.models import make_xgb_pipeline
        from sklearn.metrics import average_precision_score

        rng = np.random.default_rng(0)
        n = 3_000
        y = (rng.random(n) < 0.02).astype(int)
        X = rng.normal(size=(n, 8)).astype(np.float32)
        X[y == 1, :4] += 2.5

        pipe = make_xgb_pipeline(n_estimators=20)
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert average_precision_score(y, proba[:, 1]) > 0.5
        print("XGB_SUBPROCESS_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, f"stderr: {result.stderr[-2000:]}"
    assert "XGB_SUBPROCESS_OK" in result.stdout
