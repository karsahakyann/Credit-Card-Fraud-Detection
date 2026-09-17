"""Fraud alert delivery (demo layer).

A scored transaction that clears the deployed threshold produces an alert.
Where that alert *goes* is deliberately pluggable, because the demo has to
survive a conference room with no working network as gracefully as it
demonstrates a push notification arriving on a phone.

Backends
--------
``InMemoryNotifier``
    Keeps the last N alerts in a ring buffer for the dashboard to render.
    Always available, never fails, no configuration.
``TelegramNotifier``
    Posts to the Telegram Bot API. Reads its credentials from the
    environment and *disables itself* if they are absent, so a checkout
    without a .env still runs.
``CompositeNotifier``
    Fans out to several backends; one failing never stops the others.

Credentials never appear in this file or in git. Create a bot with
@BotFather, get the numeric chat id from @userinfobot, and put both in a
gitignored .env:

    TELEGRAM_BOT_TOKEN=123456789:AA...
    TELEGRAM_CHAT_ID=987654321

Design note: alert delivery must never break scoring. Every backend
swallows its own exceptions and reports failure through the return value,
because a fraud detector that crashes when Telegram is down is worse than
one that silently stops notifying.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol


@dataclass
class Alert:
    """One fraud alert, with the explanation that justifies it."""

    transaction_id: int
    amount: float
    probability: float
    threshold: float
    model: str
    reasons: list[tuple[str, float]] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_text(self) -> str:
        """Plain-text rendering used by Telegram and the demo inbox alike."""
        lines = [
            "FRAUD DETECTED",
            f"Amount: EUR {self.amount:,.2f}",
            f"Probability: {self.probability:.3f} (threshold {self.threshold:.2f})",
            f"Model: {self.model}",
        ]
        if self.reasons:
            lines.append("")
            lines.append("Top drivers:")
            for feature, contribution in self.reasons:
                direction = "toward fraud" if contribution > 0 else "toward legitimate"
                lines.append(f"  {feature}: {contribution:+.2f} {direction}")
        lines.append("")
        lines.append(f"Transaction #{self.transaction_id} · {self.timestamp}")
        return "\n".join(lines)


class Notifier(Protocol):
    """Anything that can deliver an alert."""

    name: str

    def send(self, alert: Alert) -> bool:
        """Return True if delivered. Must never raise."""
        ...


class InMemoryNotifier:
    """Ring buffer backing the dashboard's alert feed."""

    name = "inbox"

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        if len(self.alerts) > self.capacity:
            del self.alerts[: len(self.alerts) - self.capacity]
        return True

    def recent(self, limit: int = 20) -> list[Alert]:
        return list(reversed(self.alerts[-limit:]))

    def clear(self) -> None:
        self.alerts.clear()


class TelegramNotifier:
    """Push notifications via the Telegram Bot API.

    Inert rather than fatal when unconfigured: ``enabled`` is False and
    ``send`` returns False, so the rest of the demo is unaffected.
    """

    name = "telegram"
    API = "https://api.telegram.org"

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        timeout: float = 5.0,
    ):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.timeout = timeout
        self.last_error: str | None = None
        # Optional callable(transaction_id) -> Telegram reply_markup dict.
        # Injected rather than imported so alerts.py stays free of any
        # dependency on the control layer.
        self.keyboard_factory = None

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            self.last_error = "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"
            return False
        try:
            import requests

            payload = {"chat_id": self.chat_id, "text": alert.as_text()}
            if self.keyboard_factory is not None:
                try:
                    payload["reply_markup"] = self.keyboard_factory(
                        alert.transaction_id)
                except Exception:                             # noqa: BLE001
                    pass       # a broken keyboard must not block the alert
            response = requests.post(
                f"{self.API}/bot{self.token}/sendMessage",
                json=payload, timeout=self.timeout,
            )
            if response.status_code == 429:
                # Telegram throttles per chat. A replay that hits a cluster
                # of frauds can trip this, so name it rather than reporting a
                # generic HTTP error that looks like a broken integration.
                retry = ""
                try:
                    retry = f", retry after {response.json()['parameters']['retry_after']}s"
                except Exception:                             # noqa: BLE001
                    pass
                self.last_error = f"rate limited by Telegram{retry}"
                return False
            if response.status_code != 200:
                # Never log response bodies: Telegram echoes the bot token
                # in some error payloads.
                self.last_error = f"HTTP {response.status_code}"
                return False
            self.last_error = None
            return True
        except Exception as exc:                      # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False


class BackgroundNotifier:
    """Deliver via a worker thread, so a slow backend never stalls scoring.

    A Telegram round trip takes roughly a third of a second. The replay calls
    send() from inside the request handler while holding the state lock, so a
    synchronous post adds that delay to every batch containing an alert and
    makes a live demo visibly stutter.

    Wrapping the backend moves the wait off the request path. send() enqueues
    and returns immediately; a daemon thread drains the queue, spacing sends
    to respect Telegram's per-chat throttle. A full queue drops the alert
    rather than blocking -- for a demo, a late notification is worse than a
    missing one, and the dashboard inbox has the full record either way.
    """

    def __init__(self, backend: "Notifier", min_interval: float = 1.05,
                 max_queue: int = 200):
        import queue as _queue
        import threading as _threading

        self.backend = backend
        self.name = backend.name
        self.min_interval = min_interval
        self._queue: _queue.Queue = _queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._sent = 0
        self._thread = _threading.Thread(
            target=self._worker, name=f"notifier-{self.name}", daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.backend, "enabled", True))

    @property
    def last_error(self) -> str | None:
        err = getattr(self.backend, "last_error", None)
        if self._dropped:
            return f"{self._dropped} alert(s) dropped (queue full)" + (
                f"; {err}" if err else "")
        return err

    def _worker(self) -> None:
        import time
        while True:
            alert = self._queue.get()
            try:
                if self.backend.send(alert):
                    self._sent += 1
            except Exception:                                 # noqa: BLE001
                pass                                          # never die
            finally:
                self._queue.task_done()
            time.sleep(self.min_interval)

    def send(self, alert: Alert) -> bool:
        """Enqueue. True means accepted for delivery, not yet delivered."""
        import queue as _queue
        if not self.enabled:
            return False
        try:
            self._queue.put_nowait(alert)
            return True
        except _queue.Full:
            self._dropped += 1
            return False

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until the queue drains. Used by tests and shutdown."""
        import time
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        return self._queue.empty()


class CompositeNotifier:
    """Fan out to several backends; one failure never blocks the rest."""

    name = "composite"

    def __init__(self, *backends: Notifier):
        self.backends = list(backends)
        self.last_results: dict[str, bool] = {}

    def send(self, alert: Alert) -> bool:
        """Deliver to every backend, then report whether any succeeded.

        The results are materialised into a list *before* any() sees them.
        Passing a generator here would let any() short-circuit on the first
        success, and since the in-memory inbox always succeeds and is listed
        first, every later backend -- Telegram included -- would be silently
        skipped. That bug shipped once; the ordering test below now pins it.
        """
        results = [b.send(alert) for b in list(self.backends)]
        self.last_results = dict(zip([b.name for b in self.backends], results))
        return any(results)

    def status(self) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for b in self.backends:
            out[b.name] = bool(getattr(b, "enabled", True))
        return out

    def errors(self) -> dict[str, str]:
        """Last error per backend, for surfacing silent delivery failures."""
        return {
            b.name: err for b in self.backends
            if (err := getattr(b, "last_error", None))
        }


def load_dotenv(path: str = ".env") -> int:
    """Minimal .env loader so the demo needs no extra dependency.

    Existing environment variables win, which keeps real deployment
    configuration authoritative over a checked-out file.
    """
    from pathlib import Path

    f = Path(path)
    if not f.exists():
        return 0
    loaded = 0
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def build_default_notifier(background: bool = True) -> CompositeNotifier:
    """Dashboard inbox always; Telegram too when credentials are present.

    Telegram is wrapped for background delivery by default so its network
    latency never reaches the scoring path. Tests pass background=False to
    keep delivery synchronous and assertions simple.
    """
    load_dotenv()
    telegram: Notifier = TelegramNotifier()
    if background:
        telegram = BackgroundNotifier(telegram)
    return CompositeNotifier(InMemoryNotifier(), telegram)
