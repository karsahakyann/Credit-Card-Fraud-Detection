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

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            self.last_error = "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"
            return False
        try:
            import requests

            response = requests.post(
                f"{self.API}/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": alert.as_text()},
                timeout=self.timeout,
            )
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


class CompositeNotifier:
    """Fan out to several backends; one failure never blocks the rest."""

    name = "composite"

    def __init__(self, *backends: Notifier):
        self.backends = list(backends)

    def send(self, alert: Alert) -> bool:
        return any(b.send(alert) for b in list(self.backends))

    def status(self) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for b in self.backends:
            out[b.name] = bool(getattr(b, "enabled", True))
        return out


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


def build_default_notifier() -> CompositeNotifier:
    """Dashboard inbox always; Telegram too when credentials are present."""
    load_dotenv()
    return CompositeNotifier(InMemoryNotifier(), TelegramNotifier())
