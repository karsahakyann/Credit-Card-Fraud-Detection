"""Service tests using FastAPI's TestClient against the real model artefacts.

Skipped cleanly when models/ has not been built, so the suite still passes
on a fresh checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud import config

MODEL = config.PROJECT_ROOT / "models" / "final_chronological.joblib"
pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason="run scripts/train_final_model.py to build model artefacts",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from fraud.service import app
    with TestClient(app) as c:
        yield c


def test_health_reports_ready(client):
    h = client.get("/health").json()
    assert h["ready"] is True
    assert h["model"] == "XGBoost"
    assert 0 < h["threshold"] < 1
    assert h["replay_size"] > 50_000


def test_score_rejects_wrong_feature_count(client):
    assert client.post("/score", json={"features": [1.0, 2.0]}).status_code == 422


def test_score_returns_calibrated_probability(client):
    n = len(client.get("/health").json() and
            __import__("joblib").load(MODEL)["feature_names"])
    r = client.post("/score", json={"features": [0.0] * n, "notify": False})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["flagged"] == (body["probability"] >= body["threshold"])


def test_replay_advances_and_tallies(client):
    client.post("/replay/reset")
    r = client.get("/replay/next?n=50&notify=false").json()
    assert len(r["scored"]) == 50
    s = r["stats"]
    assert s["processed"] == 50
    assert s["tp"] + s["fp"] + s["fn"] + s["tn"] == 50


def test_reset_clears_state(client):
    client.get("/replay/next?n=25&notify=false")
    client.post("/replay/reset")
    s = client.get("/stats").json()
    assert s["processed"] == 0 and s["tp"] == 0 and s["total_cost"] == 0


def test_skip_to_fraud_lands_on_a_real_fraud(client):
    """The presentation aid must not distort the tallies it fast-forwards."""
    client.post("/replay/reset")
    r = client.get("/replay/skip_to_fraud").json()
    assert r["scored"], "should surface at least one transaction"
    assert r["scored"][-1]["actual_fraud"] is True
    s = r["stats"]
    # every scanned row is still counted exactly once
    assert s["processed"] == s["tp"] + s["fp"] + s["fn"] + s["tn"]
    assert s["processed"] == r["skipped"] + len(r["scored"])


def test_flagged_fraud_produces_an_explained_alert(client):
    client.post("/replay/reset")
    client.get("/replay/skip_to_fraud")
    alerts = client.get("/alerts?limit=5").json()["alerts"]
    assert alerts, "a flagged transaction should raise an alert"
    assert "FRAUD DETECTED" in alerts[0]["text"]


def test_dashboard_serves(client):
    r = client.get("/")
    assert r.status_code == 200 and "Fraud Detection" in r.text
