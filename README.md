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
./venv/bin/python -m pytest tests/   # sanity checks
./venv/bin/jupyter lab notebooks/    # run the notebooks
```

## Layout

| Path | Purpose |
|---|---|
| `src/fraud/config.py` | Paths, random seed, split/CV settings |
| `src/fraud/data.py` | Load, clean, stratified + chronological splits |
| `src/fraud/evaluation.py` | Standard metric report (PR-AUC, MCC, ...), figures, results log |
| `src/fraud/pipelines.py` | Leakage-free model pipelines |
| `notebooks/01_eda.ipynb` | Exploratory analysis (Phase 1) |
| `notebooks/02_baseline_logreg.ipynb` | Logistic Regression baseline on PR-AUC (Phase 1) |
| `results/` | Metric tables (`metrics.csv`) and dissertation figures |
| `tests/` | Split-correctness and no-leakage smoke tests |

## Phase roadmap (from the project portfolio)

| Phase | Dates (2026) | Content | Status |
|---|---|---|---|
| 1 | 29 Jun – 12 Jul | Data, EDA, LR baseline on PR-AUC | ✅ this repo |
| 2 | 13 – 26 Jul | CV + tuning: RF, XGBoost, DNN (focal loss) | next |
| 3 | 27 Jul – 16 Aug | Core imbalance experiment (4 strategies, in-fold resampling) | planned |
| 4 | 17 Aug – 6 Sep | Concept drift: chronological validation, periodic retraining | planned |
| 5 | 7 – 20 Sep | Cost-sensitive evaluation + SHAP interpretability | planned |
| 6–7 | 21 Sep – 31 Oct | Write-up, revision, submission | planned |
