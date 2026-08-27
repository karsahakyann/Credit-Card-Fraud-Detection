"""Phase 4 -- concept drift and periodic retraining.

Splits the 48-hour series into equal-duration blocks and walks forward
through them, comparing three deployment strategies:

    static      train once on the warm-up period, never update
    expanding   retrain before each block on all history so far
    sliding     retrain before each block on the most recent blocks only

Every fit uses training rows that strictly precede the test block, so the
protocol answers "how would this model have performed in production?" rather
than "how well does it fit a random sample?".

Model is the Phase 3 winner: XGBoost with no resampling, at the
chronologically-tuned hyperparameters. Holding the model fixed isolates the
effect of the *retraining policy*, which is the Phase 4 question.

Scope caveat, repeated from drift.py because it matters: this dataset covers
two days, so what is measured is short-horizon temporal generalisation with
strong diurnal structure -- not multi-month concept drift.

Usage
-----
    ./venv/bin/python scripts/run_drift_experiment.py
    ./venv/bin/python scripts/run_drift_experiment.py --blocks 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fraud import config, drift, evaluation, experiment, resampling

def _out_paths(n_blocks: int) -> tuple[Path, Path, Path]:
    """Headline run (default blocking) writes unsuffixed files; sensitivity
    runs at other block counts write their own, so neither clobbers the other."""
    tag = "" if n_blocks == drift.N_BLOCKS else f"_{n_blocks}blocks"
    return (
        config.RESULTS_DIR / f"drift_experiment{tag}.csv",
        config.RESULTS_DIR / f"drift_blocks{tag}.csv",
        config.RESULTS_DIR / f"drift_features{tag}.csv",
    )

# Phase 3 winner: xgboost / none / chronological.
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

MODES = ("static", "expanding", "sliding")


def run_mode(
    df: pd.DataFrame,
    blocks: np.ndarray,
    mode: str,
    window: int,
    seed: int = config.RANDOM_SEED,
    verbose: bool = True,
) -> list[dict]:
    X_all = df.drop(columns=[config.TARGET])
    y_all = df[config.TARGET].to_numpy()
    rows: list[dict] = []

    for test_block, train_idx, test_idx in drift.prequential_splits(
        blocks, mode=mode, window=window
    ):
        X_tr, y_tr = X_all.iloc[train_idx], y_all[train_idx]
        X_te, y_te = X_all.iloc[test_idx], y_all[test_idx]

        if y_te.sum() == 0:
            print(f"  [{mode} | block {test_block}] no frauds in block, skipped")
            continue

        est = resampling.make_estimator(
            "xgboost", "none", params=dict(XGB_PARAMS), y_train=y_tr, seed=seed,
        )
        started = time.perf_counter()
        est.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - started

        metrics = evaluation.evaluate(est, X_te, y_te)
        # Only the reference seed is logged to metrics.csv and saved as scores;
        # the repeat seeds exist to estimate variance, not to be reported rows.
        if seed == config.RANDOM_SEED:
            evaluation.log_result(
                metrics, model_name="xgboost",
                split=f"drift_{mode}_block{test_block}",
                imbalance_strategy="none", notes="phase4 drift",
            )
            scores = est.predict_proba(X_te)[:, 1]
            experiment.SCORES_DIR.mkdir(parents=True, exist_ok=True)
            np.save(
                experiment.SCORES_DIR / f"drift_{mode}_block{test_block}.npy",
                scores,
            )

        rows.append({
            "mode": mode,
            "seed": seed,
            "test_block": test_block,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_blocks": int(len(np.unique(blocks[train_idx]))),
            "test_frauds": int(y_te.sum()),
            "train_fraud_rate_pct": 100 * float(y_tr.mean()),
            "test_fraud_rate_pct": 100 * float(y_te.mean()),
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "mcc": metrics["mcc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fit_seconds": round(elapsed, 1),
        })
        if verbose:
            print(
                f"  [{mode:9s} | block {test_block}] "
                f"PR-AUC={metrics['pr_auc']:.4f}  MCC={metrics['mcc']:.3f}  "
                f"R={metrics['recall']:.3f} P={metrics['precision']:.3f}  "
                f"(train {len(train_idx):,} rows, {elapsed:.0f}s)",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=drift.N_BLOCKS)
    parser.add_argument("--window", type=int, default=drift.SLIDING_WINDOW)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Model seeds per cell; >1 gives error bars on the mode comparison.",
    )
    args = parser.parse_args()

    RESULTS_CSV, BLOCKS_CSV, FEATURE_DRIFT_CSV = _out_paths(args.blocks)
    df = drift.ordered_frame(experiment.load_clean())
    blocks = drift.time_blocks(df, n_blocks=args.blocks)

    summary = drift.block_summary(df, blocks)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(BLOCKS_CSV, index=False)
    print(f"\nTemporal blocks ({args.blocks} x "
          f"{48 / args.blocks:.1f}h over a 48-hour dataset)\n")
    print(summary.to_string(index=False))
    print(
        f"\nFraud rate varies {summary.fraud_rate_pct.max() / summary.fraud_rate_pct.min():.1f}x "
        f"across blocks ({summary.fraud_rate_pct.min():.3f}% to "
        f"{summary.fraud_rate_pct.max():.3f}%) -- prior-probability shift."
    )

    print("\nMeasuring covariate shift (KS vs block 0, legitimate rows only) ...")
    fdrift = drift.feature_drift(df, blocks)
    fdrift.to_csv(FEATURE_DRIFT_CSV, index=False)
    worst = (
        fdrift.groupby("feature").ks_stat.max()
        .sort_values(ascending=False).head(5)
    )
    print("  Largest KS statistic by feature:")
    for feat, stat in worst.items():
        print(f"    {feat:<8} {stat:.3f}")

    seeds = [config.RANDOM_SEED + i for i in range(args.seeds)]
    all_rows: list[dict] = []
    for mode in args.modes:
        print(f"\n{'=' * 72}\n{mode.upper()} retraining"
              f"  ({len(seeds)} seed(s))\n{'=' * 72}")
        for i, seed in enumerate(seeds):
            all_rows.extend(
                run_mode(df, blocks, mode, args.window, seed=seed,
                         verbose=(i == 0))
            )

    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS_CSV, index=False)

    print(f"\n{'=' * 72}\nMean PR-AUC across evaluated blocks\n{'=' * 72}")
    for mode, grp in results.groupby("mode"):
        per_block = grp.groupby("test_block").pr_auc.mean()
        print(f"  {mode:<10} {per_block.mean():.4f} +/- {grp.pr_auc.std():.4f}  "
              f"(per block: {', '.join(f'{v:.3f}' for v in per_block)})")

    # The first evaluated block is a correctness check, not a comparison: every
    # mode trains on the identical warm-up there, so all modes must agree.
    first = results.test_block.min()
    agree = results[results.test_block == first].groupby("mode").pr_auc.mean()
    spread = agree.max() - agree.min()
    print(f"\n  Sanity check -- at block {first} all modes share one training "
          f"set; PR-AUC spread = {spread:.6f} (should be ~0)")

    # Paired per-block comparison against the static baseline. With so few
    # evaluated blocks this is the honest test: does retraining win *per
    # block*, and by how much, rather than on an average that a single block
    # could dominate.
    print(f"\n{'=' * 72}\nPaired comparison vs static (per block, excluding "
          f"the shared warm-up block)\n{'=' * 72}")
    pivot = results.groupby(["mode", "test_block"]).pr_auc.mean().unstack("mode")
    pivot = pivot.drop(index=first, errors="ignore")
    if "static" in pivot.columns:
        for mode in [m for m in pivot.columns if m != "static"]:
            delta = pivot[mode] - pivot["static"]
            wins = int((delta > 0).sum())
            print(f"  {mode:<10} mean delta {delta.mean():+.4f}  "
                  f"wins {wins}/{len(delta)} blocks  "
                  f"(per block: {', '.join(f'{v:+.3f}' for v in delta)})")
        print(f"\n  n = {len(pivot)} independent test blocks -- too few for a "
              f"significance claim;\n  report the direction and the effect "
              f"size, not a p-value.")

    print(f"\n[Saved] {RESULTS_CSV}")
    print(f"[Saved] {BLOCKS_CSV}")
    print(f"[Saved] {FEATURE_DRIFT_CSV}")


if __name__ == "__main__":
    main()
