"""SHAP interpretability for the final model (Phase 5).

Answers the question a fraud analyst actually asks: *why was this
transaction flagged?* PR-AUC says the ranking is good and the cost analysis
says the threshold is right, but neither tells an investigator what to look
at, and neither would satisfy a regulator asking why a customer's card was
blocked.

What SHAP can and cannot say here
---------------------------------
V1-V28 are anonymised PCA components published by the dataset authors to
protect commercial confidentiality. SHAP can rank them and show the
direction of their effect, but nobody -- not this project, not the original
paper -- can say what "V14" corresponds to in the real world. Only ``Time``
and ``Amount`` are interpretable in plain language.

That bound is a finding about the dataset, not a shortcoming of the method:
this project can deliver *mathematical* transparency (which inputs drove
this decision, by how much) but not *semantic* transparency (what those
inputs mean). A deployment on raw, un-anonymised features would get both
from the same code.

Process isolation (see models.py): imports xgboost, never torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def transform_for_explanation(pipe, X: pd.DataFrame) -> np.ndarray:
    """Apply the pipeline's preprocessing, stopping short of the model.

    TreeExplainer must see exactly what the booster sees, so the scaler is
    applied first; the final estimator step is excluded.
    """
    Xt = X
    for name, step in pipe.steps[:-1]:
        Xt = step.transform(Xt)
    return np.asarray(Xt, dtype=float)


def tree_shap_values(pipe, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact SHAP values for a tree model inside a pipeline.

    Returns (values, base_value). TreeExplainer is exact for tree ensembles
    rather than sampled, so no approximation error enters the attributions.
    """
    import shap

    model = pipe.steps[-1][1]
    Xt = transform_for_explanation(pipe, X)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):            # older API: one array per class
        values = values[1] if len(values) > 1 else values[0]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.ravel(base)[-1])
    return np.asarray(values), float(base)


def global_importance(
    shap_values: np.ndarray, feature_names: list[str],
) -> pd.DataFrame:
    """Rank features by mean |SHAP|, keeping the signed mean for direction."""
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"{shap_values.shape[1]} SHAP columns vs {len(feature_names)} names"
        )
    out = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out["share_pct"] = 100 * out.mean_abs_shap / out.mean_abs_shap.sum()
    return out


def pick_archetypes(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Three cases worth explaining to a human, chosen by cost relevance.

    caught_fraud    the most valuable fraud correctly flagged
    missed_fraud    the most valuable fraud that slipped through -- the
                    single most expensive mistake the model made
    false_alarm     the highest-scoring legitimate transaction, i.e. the
                    case the model was most confident and most wrong about
    """
    y_true = np.asarray(y_true).astype(bool)
    flagged = scores >= threshold
    picks: dict[str, int] = {}

    tp = np.flatnonzero(y_true & flagged)
    if tp.size:
        picks["caught_fraud"] = int(tp[np.argmax(amounts[tp])])

    fn = np.flatnonzero(y_true & ~flagged)
    if fn.size:
        picks["missed_fraud"] = int(fn[np.argmax(amounts[fn])])

    fp = np.flatnonzero(~y_true & flagged)
    if fp.size:
        picks["false_alarm"] = int(fp[np.argmax(scores[fp])])

    return picks


def explain_case(
    shap_values: np.ndarray,
    base_value: float,
    X: pd.DataFrame,
    index: int,
    top_n: int = 6,
) -> pd.DataFrame:
    """Per-feature contributions for one transaction, largest effect first.

    Feature *values* are reported on their original scale so a human can
    read them; the attributions come from the scaled inputs the model saw,
    which is equivalent because scaling is monotonic per feature.
    """
    contrib = pd.DataFrame({
        "feature": list(X.columns),
        "value": X.iloc[index].to_numpy(),
        "shap": shap_values[index],
    })
    contrib["abs_shap"] = contrib.shap.abs()
    contrib = contrib.sort_values("abs_shap", ascending=False).head(top_n)
    contrib["pushes"] = np.where(contrib.shap > 0, "toward fraud", "toward legit")
    contrib.attrs["base_value"] = base_value
    return contrib.drop(columns="abs_shap").reset_index(drop=True)


def class_conditional_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Mean SHAP split by true class.

    The unconditional mean SHAP is close to useless on this dataset: 99.8%
    of rows are legitimate, so almost every feature shows a negative mean
    simply because most predictions push toward "legitimate". Splitting by
    true class shows what actually drives a fraud call.
    """
    y = np.asarray(y_true).astype(bool)
    if y.sum() == 0:
        raise ValueError("No positive samples to condition on")
    out = pd.DataFrame({
        "feature": feature_names,
        "mean_shap_fraud": shap_values[y].mean(axis=0),
        "mean_shap_legit": shap_values[~y].mean(axis=0),
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    })
    out["separation"] = out.mean_shap_fraud - out.mean_shap_legit
    return out.sort_values("separation", ascending=False).reset_index(drop=True)
