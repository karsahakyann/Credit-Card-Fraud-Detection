"""Tests for dnn.py: focal loss properties and the DNN wrapper."""

import numpy as np
import pytest
import torch
from torch import nn

from fraud.dnn import FocalLoss, TorchDNNClassifier


@pytest.fixture
def synthetic_imbalanced():
    rng = np.random.default_rng(0)
    n = 3_000
    y = (rng.random(n) < 0.02).astype(int)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    X[y == 1, :4] += 2.5
    return X, y


def test_focal_loss_gamma_zero_is_weighted_bce():
    torch.manual_seed(0)
    logits = torch.randn(64)
    targets = (torch.rand(64) < 0.3).float()
    alpha = 0.25
    focal = FocalLoss(alpha=alpha, gamma=0.0)(logits, targets)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    expected = (alpha_t * bce).mean()
    assert torch.allclose(focal, expected, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    easy_logits, hard_logits = torch.tensor([4.0]), torch.tensor([0.1])
    target = torch.tensor([1.0])
    loss0, loss2 = FocalLoss(alpha=0.5, gamma=0.0), FocalLoss(alpha=0.5, gamma=2.0)
    easy_ratio = (loss2(easy_logits, target) / loss0(easy_logits, target)).item()
    hard_ratio = (loss2(hard_logits, target) / loss0(hard_logits, target)).item()
    assert easy_ratio < hard_ratio < 1.0


def test_dnn_fits_and_predicts(synthetic_imbalanced):
    X, y = synthetic_imbalanced
    clf = TorchDNNClassifier(hidden=32, n_layers=2, max_epochs=15, patience=3,
                             batch_size=256, device="cpu")
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    from sklearn.metrics import average_precision_score
    assert average_precision_score(y, proba[:, 1]) > 0.5
    assert set(np.unique(clf.predict(X))) <= {0, 1}


def test_dnn_deterministic_under_seed(synthetic_imbalanced):
    X, y = synthetic_imbalanced
    kwargs = dict(hidden=16, n_layers=2, max_epochs=5, patience=5,
                  batch_size=512, device="cpu", seed=42)
    p1 = TorchDNNClassifier(**kwargs).fit(X, y).predict_proba(X[:100])
    p2 = TorchDNNClassifier(**kwargs).fit(X, y).predict_proba(X[:100])
    assert np.allclose(p1, p2, atol=1e-6)
