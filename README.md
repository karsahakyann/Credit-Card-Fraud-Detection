# Credit Card Fraud Detection under Class Imbalance and Concept Drift

Code repository for the MSc dissertation *"Detecting Credit Card Fraud under
Class Imbalance and Concept Drift using Machine Learning and Explainable AI"*
(University of York / City College, MSc in Artificial Intelligence and Data
Science — Karen Sahakyan).

## What this project does

Compares Logistic Regression, Random Forest, XGBoost and a deep neural
network (focal loss) on the European credit card fraud dataset, under four
class-imbalance strategies (no resampling, SMOTE, random undersampling,
class weighting / focal loss) in **leakage-free** conditions, evaluates
robustness to **concept drift** with time-ordered validation, reports
imbalance-appropriate metrics (precision, recall, F1, **PR-AUC**, **MCC**),
and explains predictions with **SHAP**.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -e .
```

Get the dataset (see [data/README.md](data/README.md)) into
`data/raw/creditcard.csv`, then:

```bash
./venv/bin/jupyter lab notebooks/    # read the analysis
```

Run the tests one file at a time. `pytest tests/` loads torch and xgboost into
a single process, which is slow enough on some machines to look like a hang:

```bash
for t in tests/test_*.py; do ./venv/bin/python -m pytest "$t" -q; done
```

### Live demo

```bash
./venv/bin/python scripts/train_final_model.py        # build model artefacts
./venv/bin/uvicorn fraud.service:app --app-dir src    # then open 127.0.0.1:8000
```

A console that replays the held-out test set, switches between models live, and
pushes explained fraud alerts to Telegram. Telegram is optional — copy
`.env.example` to `.env` to enable it; without it alerts appear in the
dashboard only.

## Results

Final model: **XGBoost, no resampling**, regularised so that it does not overfit.
Thresholds are chosen on training data and frozen before the test set is read.

| | Stratified split | Chronological split |
|---|---|---|
| Test PR-AUC | 0.813 | 0.799 |
| Train − test gap | +0.084 | +0.082 |
| At the cost-optimal threshold | 75/95 caught, 16 false alarms | 55/74 caught, 8 false alarms |
| Cost vs doing nothing (€10 review) | €4,699 vs €14,766 | €4,265 vs €7,728 |

Accuracy and ROC-AUC are deliberately not headlined: predicting "never fraud"
scores 99.83% accuracy on this data and catches nothing.

Principal findings, one per phase:

- **Model choice dominates imbalance strategy.** Across models the PR-AUC gap
  is about 0.09; within a model the four strategies differ by at most 0.03,
  inside the error bars. The best configuration uses no resampling at all, and
  cross-validation preferred the mildest resampling ratio offered in 8 of 12
  cells.
- **Retraining helps, but less than it first appeared.** It beats a static
  model in 14 of 17 time blocks. Regularising the model halved that advantage,
  because the static model improved while the retrained one did not — part of
  the benefit originally attributed to retraining was compensation for
  overfitting. Expanding windows beat sliding ones, so the gain comes from more
  data rather than from forgetting stale patterns.
- **Threshold tuning matters only where calibration is poor.** Significant in 4
  of 8 model/protocol cells, and in neither of XGBoost's.
- **All 30 features are retained.** Cutting to the four most important costs
  0.102 PR-AUC: concentration is not redundancy.
- **Interpretability stops at the PCA boundary.** SHAP gives exact
  attributions, but `V1`–`V28` are anonymised, so the project offers
  mathematical transparency without semantic transparency.

## Layout

| Path | Purpose |
|---|---|
| `src/fraud/config.py` | Paths, random seed, split and CV settings |
| `src/fraud/data.py` | Load, clean, stratified and chronological splits |
| `src/fraud/pipelines.py`, `models.py` | Leakage-free pipelines, model factories and search spaces |
| `src/fraud/resampling.py` | The four imbalance strategies, resampling confined to training folds |
| `src/fraud/dnn.py` | PyTorch network with focal loss |
| `src/fraud/drift.py` | Time blocks and forward-walking splits |
| `src/fraud/costs.py` | Cost model and threshold search |
| `src/fraud/explain.py` | SHAP helpers and case selection |
| `src/fraud/final_params.py` | Single source of truth for final-model parameters |
| `src/fraud/evaluation.py`, `experiment.py` | Metric reports, figures, results log |
| `src/fraud/service.py`, `dashboard.html` | Demo scoring service and console |
| `src/fraud/alerts.py`, `telegram_control.py` | Alert delivery and two-way Telegram control |
| `notebooks/01`–`05` | One notebook per phase, executed with outputs |
| `scripts/tune_*.py` | Per-model hyperparameter tuning (Phase 2) |
| `scripts/run_imbalance_experiment.py` | The 24-cell imbalance grid (Phase 3) |
| `scripts/run_drift_experiment.py` | Retraining policies over time blocks (Phase 4) |
| `scripts/select_thresholds.py` | Cost-optimal thresholds from training data only (Phase 5) |
| `scripts/run_cost_analysis.py`, `run_significance_tests.py` | Cost curves and bootstrap intervals |
| `scripts/run_shap_analysis.py`, `run_feature_reduction.py` | Interpretability and feature-count study |
| `scripts/search_final_params.py`, `run_overfitting_analysis.py` | Overfitting diagnosis and the regularised selection |
| `scripts/train_final_model.py` | Persists the artefacts the demo serves |
| `results/` | Metric tables, figures, saved scores |
| `tests/` | 109 tests, including the leakage guarantees |

## Reproducing the results

`data/raw/creditcard.csv` and `models/` are not committed; everything else is.
To rebuild from scratch, in this order:

```bash
./venv/bin/python scripts/run_imbalance_experiment.py      # Phase 3 grid
./venv/bin/python scripts/search_final_params.py           # regularised parameters
./venv/bin/python scripts/select_thresholds.py             # thresholds + final scores
./venv/bin/python scripts/run_cost_analysis.py
./venv/bin/python scripts/run_significance_tests.py
./venv/bin/python scripts/run_shap_analysis.py
./venv/bin/python scripts/run_drift_experiment.py
./venv/bin/python scripts/run_overfitting_analysis.py
./venv/bin/python scripts/train_final_model.py
```

Every script reads its model parameters from `src/fraud/final_params.py`, so
no result can be built on stale parameters.

## Phase roadmap (from the project portfolio)

| Phase | Dates (2026) | Content | Status |
|---|---|---|---|
| 1 | 29 Jun – 12 Jul | Data, EDA, LR baseline on PR-AUC | ✅ done |
| 2 | 13 – 26 Jul | CV + tuning: RF, XGBoost, DNN (focal loss) | ✅ done |
| 3 | 27 Jul – 16 Aug | Core imbalance experiment (4 strategies, in-fold resampling) | ✅ done |
| 4 | 17 Aug – 6 Sep | Concept drift: chronological validation, periodic retraining | ✅ done |
| 5 | 7 – 20 Sep | Cost-sensitive evaluation + SHAP interpretability | ✅ done |
| 6–7 | 21 Sep – 31 Oct | Write-up, revision, submission | in progress |

Beyond the roadmap: a demo scoring service with Telegram alerts, and a
regularisation pass that removed the overfitting present in the first set of
final models.

## Limitations

- The dataset spans **48 hours**, so the drift analysis measures short-horizon
  temporal generalisation rather than multi-month concept drift, and the
  non-stationarity it finds is largely diurnal.
- The test sets contain only **95 and 74 frauds**, so confidence intervals are
  wide throughout and every comparison is reported with one.
- `V1`–`V28` are anonymised PCA components, bounding what SHAP can explain.
- The cost model uses a flat review fee and charges missed fraud at face value,
  with no amount floor below which review is not worth ordering.
