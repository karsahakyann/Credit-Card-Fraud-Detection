"""Leakage-free model pipelines.

Every model is wrapped in a Pipeline whose preprocessing (scaling, and in
Phase 3 any resampling) is fit on training data only. Calling
``pipeline.fit(X_train, y_train)`` therefore can never see test-set
statistics — the leakage rule this dissertation enforces throughout.

Only ``Time`` and ``Amount`` need scaling: V1-V28 are PCA outputs and are
already on comparable scales, but scaling them too is harmless and keeps the
transformer simple and uniform.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def make_logreg_pipeline(class_weight=None, seed: int = config.RANDOM_SEED) -> Pipeline:
    """StandardScaler + LogisticRegression baseline.

    ``class_weight="balanced"`` is the free, model-level imbalance strategy
    that Phase 3 compares against resampling approaches.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )
