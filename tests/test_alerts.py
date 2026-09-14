"""Alert-delivery tests: the demo must degrade, never crash."""

from __future__ import annotations

import pytest

from fraud.alerts import (
    Alert, CompositeNotifier, InMemoryNotifier, TelegramNotifier,
)


@pytest.fixture
def alert():
    return Alert(transaction_id=42, amount=847.20, probability=0.94,
                 threshold=0.26, model="XGBoost",
                 reasons=[("V14", 3.99), ("V4", 1.48), ("V12", -0.77)])


def test_alert_text_contains_the_essentials(alert):
    t = alert.as_text()
    for expected in ("FRAUD DETECTED", "847.20", "0.940", "0.26", "XGBoost", "#42"):
        assert expected in t


def test_alert_text_labels_direction(alert):
    t = alert.as_text()
    assert "V14: +3.99 toward fraud" in t
    assert "V12: -0.77 toward legitimate" in t


def test_inbox_stores_and_returns_newest_first(alert):
    inbox = InMemoryNotifier()
    for i in range(3):
        inbox.send(Alert(i, 10.0 * i, 0.9, 0.26, "XGBoost"))
    assert [a.transaction_id for a in inbox.recent()] == [2, 1, 0]


def test_inbox_is_bounded():
    inbox = InMemoryNotifier(capacity=5)
    for i in range(20):
        inbox.send(Alert(i, 1.0, 0.9, 0.26, "XGBoost"))
    assert len(inbox.alerts) == 5
    assert inbox.alerts[-1].transaction_id == 19


def test_telegram_disabled_without_credentials(monkeypatch, alert):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    t = TelegramNotifier()
    assert t.enabled is False
    assert t.send(alert) is False          # must not raise
    assert "not set" in t.last_error


def test_telegram_never_raises_on_network_failure(monkeypatch, alert):
    """A dead network must not take the scoring path down with it."""
    t = TelegramNotifier(token="x", chat_id="y")
    import fraud.alerts as mod

    class Boom:
        @staticmethod
        def post(*a, **k):
            raise ConnectionError("no route to host")

    monkeypatch.setitem(__import__("sys").modules, "requests", Boom)
    assert t.send(alert) is False
    assert "ConnectionError" in t.last_error


def test_telegram_error_never_leaks_the_token(monkeypatch, alert):
    t = TelegramNotifier(token="SECRET-TOKEN-123", chat_id="y")
    import sys

    class Resp:
        status_code = 401
        text = "Unauthorized: bot token SECRET-TOKEN-123 is invalid"

    class Fake:
        @staticmethod
        def post(*a, **k):
            return Resp()

    monkeypatch.setitem(sys.modules, "requests", Fake)
    t.send(alert)
    assert "SECRET-TOKEN-123" not in (t.last_error or "")


def test_composite_succeeds_if_any_backend_does(alert):
    class Dead:
        name = "dead"
        def send(self, a):  # noqa: D102
            return False

    inbox = InMemoryNotifier()
    c = CompositeNotifier(Dead(), inbox)
    assert c.send(alert) is True
    assert len(inbox.alerts) == 1


def test_composite_delivers_to_inbox_even_when_telegram_is_off(alert, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    inbox = InMemoryNotifier()
    c = CompositeNotifier(inbox, TelegramNotifier())
    assert c.send(alert) is True
    assert c.status() == {"inbox": True, "telegram": False}


def test_dotenv_does_not_override_real_environment(tmp_path, monkeypatch):
    from fraud.alerts import load_dotenv
    env = tmp_path / ".env"
    env.write_text('TELEGRAM_CHAT_ID="from-file"\nNEW_KEY=value\n')
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "from-environment")
    load_dotenv(str(env))
    import os
    assert os.environ["TELEGRAM_CHAT_ID"] == "from-environment"
    assert os.environ["NEW_KEY"] == "value"


def test_composite_calls_every_backend_even_after_one_succeeds(alert):
    """Regression: any() over a generator short-circuited past Telegram.

    The in-memory inbox always succeeds and is registered first, so a
    short-circuiting any() meant no later backend was ever invoked and
    alerts silently stopped reaching the phone while the UI showed success.
    """
    calls: list[str] = []

    class Recorder:
        def __init__(self, name, result):
            self.name, self.result = name, result

        def send(self, a):
            calls.append(self.name)
            return self.result

    c = CompositeNotifier(Recorder("first", True), Recorder("second", True))
    assert c.send(alert) is True
    assert calls == ["first", "second"], "every backend must be invoked"


def test_composite_reports_per_backend_outcome(alert):
    class Dead:
        name = "dead"
        def send(self, a):  # noqa: D102
            return False

    inbox = InMemoryNotifier()
    c = CompositeNotifier(inbox, Dead())
    c.send(alert)
    assert c.last_results == {"inbox": True, "dead": False}


def test_background_notifier_returns_immediately(alert):
    """Enqueueing must not wait on the network."""
    import time
    from fraud.alerts import BackgroundNotifier

    class Slow:
        name = "slow"
        enabled = True
        last_error = None
        def send(self, a):  # noqa: D102
            time.sleep(0.5)
            return True

    bg = BackgroundNotifier(Slow(), min_interval=0)
    t0 = time.time()
    assert bg.send(alert) is True
    assert time.time() - t0 < 0.1, "send() must not block on the backend"
    assert bg.flush(timeout=5)


def test_background_notifier_actually_delivers(alert):
    from fraud.alerts import BackgroundNotifier
    delivered = []

    class Recorder:
        name = "rec"
        enabled = True
        last_error = None
        def send(self, a):  # noqa: D102
            delivered.append(a.transaction_id)
            return True

    bg = BackgroundNotifier(Recorder(), min_interval=0)
    for i in range(3):
        bg.send(Alert(i, 1.0, 0.9, 0.11, "XGBoost"))
    assert bg.flush(timeout=5)
    assert sorted(delivered) == [0, 1, 2]


def test_background_notifier_survives_a_throwing_backend(alert):
    """A backend that raises must not kill the worker thread."""
    from fraud.alerts import BackgroundNotifier
    calls = []

    class Exploding:
        name = "boom"
        enabled = True
        last_error = None
        def send(self, a):  # noqa: D102
            calls.append(a.transaction_id)
            raise RuntimeError("network gone")

    bg = BackgroundNotifier(Exploding(), min_interval=0)
    bg.send(Alert(1, 1.0, 0.9, 0.11, "X"))
    bg.send(Alert(2, 1.0, 0.9, 0.11, "X"))
    assert bg.flush(timeout=5)
    assert calls == [1, 2], "worker must keep draining after an exception"


def test_background_notifier_drops_rather_than_blocks(alert):
    import time
    from fraud.alerts import BackgroundNotifier

    class Slow:
        name = "slow"
        enabled = True
        last_error = None
        def send(self, a):  # noqa: D102
            time.sleep(0.2)
            return True

    bg = BackgroundNotifier(Slow(), min_interval=0, max_queue=2)
    results = [bg.send(Alert(i, 1.0, 0.9, 0.11, "X")) for i in range(12)]
    assert results[0] is True
    assert any(r is False for r in results), "a full queue must drop, not block"
    assert "dropped" in (bg.last_error or "")


def test_telegram_names_rate_limiting(monkeypatch, alert):
    """429 must be reported as throttling, not a generic failure."""
    import sys
    class Resp:
        status_code = 429
        @staticmethod
        def json():
            return {"parameters": {"retry_after": 7}}
    class Fake:
        @staticmethod
        def post(*a, **k):
            return Resp()
    monkeypatch.setitem(sys.modules, "requests", Fake)
    t = TelegramNotifier(token="x", chat_id="y")
    assert t.send(alert) is False
    assert "rate limited" in t.last_error and "7s" in t.last_error
