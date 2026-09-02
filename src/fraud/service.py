"""FastAPI scoring service and live demo (demo layer).

Serves the model this dissertation recommends, at the threshold its own
cost analysis selected, explaining each alert with the SHAP machinery from
Phase 5. Nothing here invents new modelling: it deploys the result.

Endpoints
---------
``GET  /``            dashboard
``POST /score``       score one transaction
``GET  /replay/...``  drive a replay of the held-out test set
``GET  /stats``       running tallies and live cost
``GET  /alerts``      recent alerts from the in-memory inbox
``GET  /health``      readiness, including whether Telegram is configured

The replay streams the **held-out test set**, which the model never saw in
training, so the demo is a real evaluation rather than a re-run of training
data.

Run:
    ./venv/bin/uvicorn fraud.service:app --app-dir src --reload
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config
from .alerts import Alert, InMemoryNotifier, build_default_notifier

MODEL_DIR = config.PROJECT_ROOT / "models"
DEFAULT_PROTOCOL = "chronological"      # the deployment-realistic one


class ScoreRequest(BaseModel):
    features: list[float] = Field(..., description="Feature vector, model column order")
    transaction_id: int | None = None
    notify: bool = True


class ScoreResponse(BaseModel):
    transaction_id: int
    probability: float
    threshold: float
    flagged: bool
    amount: float
    reasons: list[tuple[str, float]]
    notified: bool


class ReplayState:
    """Cursor and tallies for the replay, guarded for concurrent requests."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.cursor = 0
        self.tp = self.fp = self.fn = self.tn = 0
        self.fn_loss = 0.0
        self.review_cost_total = 0.0
        self.processed = 0


class Service:
    """Holds the loaded model, replay data and notifier."""

    def __init__(self, protocol: str = DEFAULT_PROTOCOL) -> None:
        self.protocol = protocol
        self.bundle: dict[str, Any] | None = None
        self.replay: dict[str, Any] | None = None
        self.state = ReplayState()
        self.notifier = build_default_notifier()
        self.inbox: InMemoryNotifier = next(
            b for b in self.notifier.backends if isinstance(b, InMemoryNotifier)
        )
        self._explainer = None

    # -- loading ---------------------------------------------------------
    def load(self) -> None:
        model_path = MODEL_DIR / f"final_{self.protocol}.joblib"
        replay_path = MODEL_DIR / f"replay_{self.protocol}.npz"
        if not model_path.exists() or not replay_path.exists():
            raise FileNotFoundError(
                f"Missing {model_path.name} / {replay_path.name}. "
                "Run: ./venv/bin/python scripts/train_final_model.py"
            )
        self.bundle = joblib.load(model_path)
        z = np.load(replay_path, allow_pickle=True)
        self.replay = {
            "X": z["X"], "y": z["y"], "amounts": z["amounts"],
            "columns": [str(c) for c in z["columns"]],
        }
        self.state.reset()

    @property
    def ready(self) -> bool:
        return self.bundle is not None and self.replay is not None

    @property
    def threshold(self) -> float:
        return float(self.bundle["threshold"]) if self.bundle else float("nan")

    # -- scoring ---------------------------------------------------------
    def _explain(self, x: np.ndarray, top_n: int = 3) -> list[tuple[str, float]]:
        """Top SHAP drivers for one row; degrades to [] if shap is absent."""
        try:
            import shap

            pipe = self.bundle["pipeline"]
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(pipe.steps[-1][1])
            xt = x.reshape(1, -1)
            for _, step in pipe.steps[:-1]:
                xt = step.transform(xt)
            values = self._explainer.shap_values(np.asarray(xt, dtype=float))
            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            row = np.asarray(values).ravel()
            names = self.bundle["feature_names"]
            order = np.argsort(-np.abs(row))[:top_n]
            return [(names[i], float(row[i])) for i in order]
        except Exception:                                    # noqa: BLE001
            return []

    def score_row(
        self, x: np.ndarray, transaction_id: int, notify: bool = True,
    ) -> ScoreResponse:
        pipe = self.bundle["pipeline"]
        prob = float(pipe.predict_proba(x.reshape(1, -1))[0, 1])
        flagged = prob >= self.threshold
        names = self.bundle["feature_names"]
        amount = float(x[names.index("Amount")]) if "Amount" in names else 0.0

        reasons: list[tuple[str, float]] = []
        notified = False
        if flagged:
            reasons = self._explain(x)
            if notify:
                notified = self.notifier.send(Alert(
                    transaction_id=transaction_id, amount=amount,
                    probability=prob, threshold=self.threshold,
                    model=self.bundle["model_name"], reasons=reasons,
                ))
        return ScoreResponse(
            transaction_id=transaction_id, probability=prob,
            threshold=self.threshold, flagged=flagged, amount=amount,
            reasons=reasons, notified=notified,
        )


svc = Service()
app = FastAPI(title="Fraud Detection Demo",
              description="Serving the dissertation's final model at its "
                          "cost-optimal threshold.",
              version="1.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        svc.load()
    except FileNotFoundError as exc:
        print(f"[startup] {exc}")


@app.get("/health")
def health() -> dict:
    return {
        "ready": svc.ready,
        "protocol": svc.protocol,
        "model": svc.bundle["model_name"] if svc.ready else None,
        "threshold": svc.threshold if svc.ready else None,
        "notifiers": svc.notifier.status(),
        "replay_size": len(svc.replay["y"]) if svc.ready else 0,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if not svc.ready:
        raise HTTPException(503, "Model not loaded; run scripts/train_final_model.py")
    expected = len(svc.bundle["feature_names"])
    if len(req.features) != expected:
        raise HTTPException(
            422, f"Expected {expected} features, got {len(req.features)}")
    tid = req.transaction_id if req.transaction_id is not None else -1
    return svc.score_row(np.asarray(req.features, dtype=float), tid, req.notify)


@app.post("/replay/reset")
def replay_reset() -> dict:
    svc.state.reset()
    svc.inbox.clear()
    return {"ok": True}


@app.get("/replay/next")
def replay_next(n: int = 1, notify: bool = True) -> dict:
    """Score the next n held-out transactions and update running tallies."""
    if not svc.ready:
        raise HTTPException(503, "Model not loaded")
    X, y, amounts = svc.replay["X"], svc.replay["y"], svc.replay["amounts"]
    out = []
    with svc.state.lock:
        for _ in range(max(1, min(n, 200))):
            i = svc.state.cursor
            if i >= len(y):
                break
            res = svc.score_row(X[i], transaction_id=int(i), notify=notify)
            actual = int(y[i])
            if res.flagged and actual == 1:
                svc.state.tp += 1
            elif res.flagged and actual == 0:
                svc.state.fp += 1
            elif not res.flagged and actual == 1:
                svc.state.fn += 1
                svc.state.fn_loss += float(amounts[i])
            else:
                svc.state.tn += 1
            if res.flagged:
                svc.state.review_cost_total += float(svc.bundle["review_cost"])
            svc.state.processed += 1
            svc.state.cursor += 1
            out.append({"transaction_id": int(i), "probability": res.probability,
                        "amount": res.amount, "flagged": res.flagged,
                        "actual_fraud": actual == 1})
    return {"scored": out, "stats": stats()}


@app.get("/replay/skip_to_fraud")
def replay_skip_to_fraud(max_scan: int = 6000, notify: bool = True) -> dict:
    """Fast-forward to the next genuine fraud, scoring everything on the way.

    Fraud is 0.13% of this stream: the first one sits about 1,900
    transactions in, and the widest gap is over 4,000. A live audience
    should not watch a minute of blank screen waiting for that.

    This is a presentation aid, not a shortcut. Every skipped transaction is
    still scored and still counted in the running tallies, so precision,
    recall and cost stay exactly what a full replay would produce -- only
    the per-row rendering is suppressed.
    """
    if not svc.ready:
        raise HTTPException(503, "Model not loaded")
    X, y, amounts = svc.replay["X"], svc.replay["y"], svc.replay["amounts"]
    shown, skipped = [], 0
    with svc.state.lock:
        for _ in range(max(1, min(max_scan, 20000))):
            i = svc.state.cursor
            if i >= len(y):
                break
            actual = int(y[i])
            res = svc.score_row(X[i], transaction_id=int(i), notify=notify)
            if res.flagged and actual == 1:
                svc.state.tp += 1
            elif res.flagged and actual == 0:
                svc.state.fp += 1
            elif not res.flagged and actual == 1:
                svc.state.fn += 1
                svc.state.fn_loss += float(amounts[i])
            else:
                svc.state.tn += 1
            if res.flagged:
                svc.state.review_cost_total += float(svc.bundle["review_cost"])
            svc.state.processed += 1
            svc.state.cursor += 1

            if actual == 1 or res.flagged:
                shown.append({"transaction_id": int(i),
                              "probability": res.probability,
                              "amount": res.amount, "flagged": res.flagged,
                              "actual_fraud": actual == 1})
                if actual == 1:
                    break
            else:
                skipped += 1
    return {"scored": shown, "skipped": skipped, "stats": stats()}


@app.get("/stats")
def stats() -> dict:
    s = svc.state
    caught = s.tp
    total_fraud = s.tp + s.fn
    return {
        "processed": s.processed,
        "remaining": (len(svc.replay["y"]) - s.cursor) if svc.ready else 0,
        "tp": s.tp, "fp": s.fp, "fn": s.fn, "tn": s.tn,
        "precision": (s.tp / (s.tp + s.fp)) if (s.tp + s.fp) else None,
        "recall": (caught / total_fraud) if total_fraud else None,
        "fraud_seen": total_fraud,
        "money_lost": round(s.fn_loss, 2),
        "review_cost": round(s.review_cost_total, 2),
        "total_cost": round(s.fn_loss + s.review_cost_total, 2),
        "threshold": svc.threshold if svc.ready else None,
    }


@app.get("/alerts")
def alerts(limit: int = 15) -> dict:
    return {"alerts": [
        {"transaction_id": a.transaction_id, "amount": a.amount,
         "probability": a.probability, "timestamp": a.timestamp,
         "reasons": a.reasons, "text": a.as_text()}
        for a in svc.inbox.recent(limit)
    ]}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return (Path(__file__).parent / "dashboard.html").read_text()
