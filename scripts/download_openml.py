"""Fetch the ULB credit card fraud dataset from OpenML (data id 1597).

This is the same dataset as Kaggle's mlg-ulb/creditcardfraud but requires no
account. Writes data/raw/creditcard.csv (~150 MB download).

Usage:  ./venv/bin/python scripts/download_openml.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.datasets import fetch_openml

from fraud import config


def main() -> None:
    if config.RAW_DATA_PATH.exists():
        print(f"Already present: {config.RAW_DATA_PATH}")
        return
    print("Downloading 'creditcard' from OpenML (data id 1597, ~150 MB)...")
    bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    df = bunch.frame
    # OpenML stores the label as a categorical string ('0'/'1'); Kaggle's CSV
    # is integer. Normalise so both sources produce identical files.
    df["Class"] = df["Class"].astype(int)
    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} cols -> {config.RAW_DATA_PATH}")
    print(f"Frauds: {int(df['Class'].sum())}")


if __name__ == "__main__":
    main()
