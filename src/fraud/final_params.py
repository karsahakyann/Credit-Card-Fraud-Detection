"""Single source of truth for final-model hyperparameters.

Before this module, seven scripts each read model parameters on their own --
six from the Phase 3 results table, one from tuned_params.json -- and the
drift experiment hardcoded its own copy. Changing the final model therefore
meant finding every one of those, and missing any would leave some results
computed with stale parameters while others used the new ones. A result set
built on two different models is worse than either model alone.

Every downstream script now asks this module instead.

Resolution order
----------------
1. results/final_params.json   the regularised XGBoost and Random Forest
                               parameters chosen to avoid overfitting
                               (scripts/search_final_params.py)
2. results/tuned_params.json   the neural network, which was not re-tuned
3. results/imbalance_experiment.csv
                               logistic regression, whose train/test gap was
                               already small, at its Phase 3 parameters

Phase 3's own experiment is deliberately NOT routed through here. It is a
controlled comparison of imbalance strategies and its record stands as run.
"""

from __future__ import annotations

import ast
import json
from functools import lru_cache

import pandas as pd

from . import config

FINAL_PARAMS_PATH = config.RESULTS_DIR / "final_params.json"
REGULARISED = ("xgboost", "random_forest")


@lru_cache(maxsize=None)
def _final_store() -> dict:
    if not FINAL_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{FINAL_PARAMS_PATH} not found. Run "
            "scripts/search_final_params.py to select final parameters.")
    return json.loads(FINAL_PARAMS_PATH.read_text())


def _phase3(model: str, protocol: str) -> dict:
    grid = pd.read_csv(config.RESULTS_DIR / "imbalance_experiment.csv")
    row = grid[(grid.model == model) & (grid.strategy == "none")
               & (grid.protocol == protocol)]
    if row.empty:
        raise KeyError(f"No Phase 3 parameters for {model}/{protocol}")
    out = {}
    for k, v in json.loads(row.iloc[0].best_params).items():
        try:
            out[k.replace("model__", "")] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.replace("model__", "")] = v
    return out


def params(model: str, protocol: str) -> dict:
    """Final hyperparameters for one model under one split protocol."""
    if model in REGULARISED:
        entry = _final_store()["models"][model][protocol]
        return dict(entry["params"])
    if model == "dnn":
        store = json.loads((config.RESULTS_DIR / "tuned_params.json").read_text())
        return dict(store["dnn"]["params"])
    return _phase3(model, protocol)


def source(model: str) -> str:
    """Where a model's parameters come from, for reports and the paper."""
    if model in REGULARISED:
        return "regularised (final_params.json)"
    if model == "dnn":
        return "Phase 2 tuning (tuned_params.json)"
    return "Phase 3 tuning (imbalance_experiment.csv)"


def summary(model: str, protocol: str) -> dict | None:
    """Selection diagnostics for a regularised model, else None."""
    if model not in REGULARISED:
        return None
    return {k: v for k, v in _final_store()["models"][model][protocol].items()
            if k != "params"}


# --- artefact paths ---------------------------------------------------------
# Final-model score files carry a "final_" prefix so they can never be
# confused with the Phase 3 experiment's own scores, which stay on disk as
# that experiment's record.
#
# The OOF cache names matter even more. select_thresholds.py reuses a cached
# out-of-fold file when one exists. The caches from before regularisation
# were computed with the old parameters, so reusing them under a changed
# model would silently produce thresholds for a model that no longer exists,
# with no error. A distinct name makes a stale cache impossible to pick up.

def score_file(model: str, protocol: str):
    """Held-out test scores for the final model."""
    from . import experiment
    if model == "dnn":
        # Not re-tuned: its Phase 2 scores are still the final model's scores.
        return experiment.SCORES_DIR / f"dnn_{protocol}.npy"
    return experiment.SCORES_DIR / f"final_{model}_{protocol}.npy"


def oof_file(model: str, protocol: str):
    """Out-of-fold training scores for the final model."""
    oof_dir = config.RESULTS_DIR / "oof_scores"
    if model == "dnn":
        return oof_dir / f"dnn_{protocol}_oof.npy"
    return oof_dir / f"final_{model}_{protocol}_oof.npy"
