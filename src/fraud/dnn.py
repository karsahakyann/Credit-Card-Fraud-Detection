"""PyTorch deep neural network with focal loss (Phase 2).

IMPORTANT — process isolation rule: this module must never be imported in
the same process as xgboost when pandas is also loaded (the trio segfaults
on macOS — see the note in ``models.py``). The DNN is tuned and evaluated
in its own process (``scripts/tune_dnn.py``).

Focal loss (Lin et al., 2017) down-weights easy examples so training
focuses on the hard, rare fraud cases; with ``gamma=0`` it reduces to
class-weighted binary cross-entropy — a property the unit tests verify.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn

from . import config


class FocalLoss(nn.Module):
    """Binary focal loss on logits.

    ``alpha`` weights the positive (fraud) class; ``gamma`` controls how
    strongly easy examples are down-weighted. ``gamma=0`` recovers
    alpha-weighted binary cross-entropy.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)           # prob of the true class
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width_in = n_features
        for _ in range(n_layers):
            layers += [
                nn.Linear(width_in, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            width_in = hidden
        layers.append(nn.Linear(width_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchDNNClassifier(BaseEstimator, ClassifierMixin):
    """Feed-forward net with BatchNorm, Dropout, focal loss and early stopping.

    Features are standardised internally (fit on training data only), so the
    estimator can be used directly in place of a scaler+model pipeline.
    Early stopping monitors PR-AUC on an internal stratified validation split.
    """

    def __init__(
        self,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 1024,
        max_epochs: int = 100,
        patience: int = 7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        val_fraction: float = 0.15,
        seed: int = config.RANDOM_SEED,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.val_fraction = val_fraction
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._resolve_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y).astype(int)

        # internal standardisation, fit on training data only
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.seed
        )
        X_tr = torch.tensor((X_tr - self._mean) / self._std)
        X_val = torch.tensor((X_val - self._mean) / self._std)
        y_tr_t = torch.tensor(y_tr)

        self.model_ = _MLP(X.shape[1], self.hidden, self.n_layers, self.dropout).to(device)
        loss_fn = FocalLoss(self.focal_alpha, self.focal_gamma)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(X_tr, y_tr_t)
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

        best_val, best_state, wait = -np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_scores = torch.sigmoid(self.model_(X_val.to(device))).cpu().numpy()
            val_ap = average_precision_score(y_val, val_scores)
            if self.verbose:
                print(f"epoch {epoch + 1}: val PR-AUC = {val_ap:.4f}")
            if val_ap > best_val:
                best_val, wait = val_ap, 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_pr_auc_ = float(best_val)
        self.n_epochs_ = epoch + 1
        return self

    def predict_proba(self, X) -> np.ndarray:
        device = self._resolve_device()
        X = (np.asarray(X, dtype=np.float32) - self._mean) / self._std
        self.model_.eval()
        with torch.no_grad():
            scores = []
            for i in range(0, len(X), 65536):  # chunk to bound memory
                xb = torch.tensor(X[i : i + 65536]).to(device)
                scores.append(torch.sigmoid(self.model_(xb)).cpu().numpy())
        p1 = np.concatenate(scores)
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
