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
    """The presentation aid must not distort the tallies it fast-forwards.

    Uses stop_on=fraud explicitly: the default is the label-free "alert"
    mode, which stops at the model's first flag and may legitimately land
    on a false positive.
    """
    client.post("/replay/reset")
    r = client.get("/replay/skip_to_fraud?stop_on=fraud").json()
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


def test_skip_on_alert_uses_no_labels(client):
    """Default mode stops when the MODEL flags, never on ground truth."""
    client.post("/replay/reset")
    r = client.get("/replay/skip_to_fraud?stop_on=alert&notify=false").json()
    assert r["stop_on"] == "alert"
    assert r["scored"][-1]["flagged"] is True, "must stop on a model alert"


def test_skip_on_fraud_stops_at_truth(client):
    client.post("/replay/reset")
    r = client.get("/replay/skip_to_fraud?stop_on=fraud&notify=false").json()
    assert r["stop_on"] == "fraud"
    assert r["scored"][-1]["actual_fraud"] is True


def test_skip_rejects_unknown_mode(client):
    assert client.get("/replay/skip_to_fraud?stop_on=magic").status_code == 422


def test_both_skip_modes_keep_tallies_exact(client):
    """Whichever mode is used, every scanned row is counted exactly once."""
    for mode in ("alert", "fraud"):
        client.post("/replay/reset")
        r = client.get(f"/replay/skip_to_fraud?stop_on={mode}&notify=false").json()
        s = r["stats"]
        assert s["processed"] == s["tp"] + s["fp"] + s["fn"] + s["tn"]
        assert s["processed"] == r["skipped"] + len(r["scored"])


def test_whatif_reproduces_the_published_result(client):
    """The service's what-if must agree with the pipeline's own results.

    Expected values are read from threshold_selection.csv rather than
    hardcoded. The previous version pinned the pre-regularisation numbers
    (56 / 18 / 10, EUR 3,337), so changing the final model would have broken
    this test for the wrong reason -- or, worse, been "fixed" by pasting in
    whatever the service now returned. This checks the property that
    actually matters: the demo and the analysis compute the same answer.
    """
    import pandas as pd
    from fraud import config
    from fraud.service import svc
    sel = pd.read_csv(config.RESULTS_DIR / "threshold_selection.csv")
    row = sel[(sel.model == "xgboost") & (sel.protocol == svc.protocol)
              & (sel.review_cost == 10.0)].iloc[0]
    t = float(row.threshold_selected_on_train)
    r = client.get(f"/whatif?model=xgboost&threshold={t}").json()
    assert (r["tp"], r["fp"], r["fn"]) == (int(row.tp), int(row.fp), int(row.fn))
    assert abs(r["total_cost"] - row.cost_deployed) < 1


def test_whatif_rejects_bad_input(client):
    assert client.get("/whatif?model=nope").status_code == 422
    assert client.get("/whatif?model=xgboost&threshold=5").status_code == 422


def test_whatif_threshold_monotonicity(client):
    """Raising the threshold can only flag fewer transactions."""
    prev = None
    for t in (0.05, 0.2, 0.5, 0.9):
        r = client.get(f"/whatif?model=xgboost&threshold={t}").json()
        flagged = r["tp"] + r["fp"]
        if prev is not None:
            assert flagged <= prev
        prev = flagged


def test_models_lists_comparable_models(client):
    names = [m["model"] for m in client.get("/models").json()["models"]]
    assert {"xgboost", "random_forest", "logistic_regression", "dnn"} <= set(names)


def test_runtime_threshold_changes_decisions(client):
    base = client.get("/health").json()["threshold"]
    client.post("/threshold?value=0.9")
    assert client.get("/health").json()["threshold"] == 0.9
    client.post("/threshold/reset")
    assert client.get("/health").json()["threshold"] == base


def test_runtime_threshold_validates(client):
    assert client.post("/threshold?value=2").status_code == 422


def test_feedback_endpoint_starts_empty(client):
    assert "count" in client.get("/feedback").json()


def test_feedback_withholds_outcomes_by_default(client):
    """Truth must not leak into triage: it arrives later, as in practice."""
    from fraud.service import svc
    svc.feedback.clear()
    svc.feedback[1885] = "escalate"
    body = client.get("/feedback").json()
    assert body["revealed"] is False
    assert "actual_fraud" not in body["recent"][0]
    assert body["recent"][0]["disposition"] == "escalate"


def test_feedback_reveals_only_when_asked(client):
    from fraud.service import svc
    svc.feedback.clear()
    svc.feedback[1885] = "escalate"          # a genuine fraud in the replay
    body = client.get("/feedback?reveal=true").json()
    assert body["revealed"] is True
    assert body["recent"][0]["actual_fraud"] is True
    assert body["escalated"] == 1 and body["escalated_were_fraud"] == 1
    svc.feedback.clear()


def test_models_reports_which_can_serve_live(client):
    d = client.get("/models").json()
    assert "live_models" in d and "current" in d
    assert d["current"] in d["live_models"]
    for row in d["models"]:
        assert "can_serve_live" in row


def test_switching_model_changes_threshold_and_name(client):
    before = client.get("/health").json()
    r = client.post("/model?name=random_forest")
    assert r.status_code == 200
    after = client.get("/health").json()
    assert after["model_key"] == "random_forest"
    assert after["threshold"] != before["threshold"]
    client.post("/model?name=xgboost")          # restore


def test_switching_rejects_unbuilt_model(client):
    r = client.post("/model?name=not_a_model")
    assert r.status_code == 422
    assert "Available" in r.json()["detail"]


def test_switching_resets_the_replay(client):
    """A run must not mix two models' decisions."""
    client.post("/replay/reset")
    client.get("/replay/next?n=50&notify=false")
    assert client.get("/stats").json()["processed"] == 50
    client.post("/model?name=random_forest")
    assert client.get("/stats").json()["processed"] == 0
    client.post("/model?name=xgboost")


def test_each_model_restores_its_own_tuned_threshold(client):
    seen = {}
    for name in ("xgboost", "random_forest", "logistic_regression"):
        client.post(f"/model?name={name}")
        seen[name] = client.get("/health").json()["threshold"]
    client.post("/model?name=xgboost")
    assert len(set(seen.values())) == 3, f"thresholds should differ: {seen}"
