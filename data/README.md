# Dataset

**European Credit Card Fraud dataset** (ULB Machine Learning Group).
284,807 transactions over 2 days in September 2013; 492 frauds (0.172%).
Features: `Time`, `V1`–`V28` (PCA-anonymised), `Amount`, `Class` (1 = fraud).

The CSV (~150 MB) is not committed to git. Place it at:

```
data/raw/creditcard.csv
```

## How to get it

**Option A — Kaggle website (simplest):**
1. Log in at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download and unzip, move `creditcard.csv` into `data/raw/`

**Option B — Kaggle API:**
1. kaggle.com → Profile → Settings → API → *Create New Token* → save the
   downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (then `chmod 600` it)
2. ```
   ./venv/bin/kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
   ```

**Option C — OpenML mirror (no account needed):** the same dataset is on
OpenML as `creditcard` (data id 1597); `scripts/download_openml.py` fetches
it and writes `data/raw/creditcard.csv`.
