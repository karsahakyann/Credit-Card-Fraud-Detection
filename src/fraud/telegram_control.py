"""Two-way Telegram control for the demo (demo layer).

Alerts so far have been one-way: the service pushes, the phone receives.
This adds the return path, which is what makes the demo look like a
deployed system rather than a notification toy.

Two mechanisms, both over Telegram's polling API so no public webhook or
tunnel is needed -- important for demonstrating on conference wifi:

**Inline buttons on every alert.** Escalate / Dismiss / Explain.

These record a *disposition*, not a verdict, and the distinction matters. A
real analyst does not know whether an alert is fraud when it arrives -- they
investigate, and the truth only surfaces later when a chargeback is filed or
the cardholder confirms, often weeks afterwards. Asking a reviewer to label
an alert correct or incorrect on sight would be asking for a judgment the
data cannot support: the features are anonymised PCA components, so no human
can read them.

So the buttons capture what an analyst would *do*, and the ground truth is
withheld until the session summary. That mirrors the delayed-label problem
that makes supervised fraud detection hard in practice, and it keeps the
demo honest about what a person can and cannot contribute here.

**Slash commands.** ``/threshold 0.3`` re-thresholds the running service,
``/stats`` returns live tallies, ``/whatif`` compares models. Adjusting a
live system from a phone while an audience watches the dashboard react
demonstrates the Phase 5 cost analysis far better than a static table.

Design constraints, all learned the hard way earlier in this project:

* The poller must never raise. It runs in a daemon thread and swallows every
  exception, because a dead poller should degrade the demo, not crash it.
* Telegram echoes the bot token in some error payloads, so response bodies
  are never logged -- only status codes.
* getUpdates is long-polled with an offset so updates are consumed exactly
  once and the queue cannot back up.
"""

from __future__ import annotations

import threading
from typing import Callable

API = "https://api.telegram.org"


def build_alert_keyboard(transaction_id: int) -> dict:
    """Inline keyboard attached to each fraud alert.

    Escalate and dismiss are workflow actions, not claims about the truth --
    see the module docstring for why that distinction is deliberate.
    """
    return {
        "inline_keyboard": [[
            {"text": "\U0001f4cb Escalate",
             "callback_data": f"fb:{transaction_id}:escalate"},
            {"text": "\U0001f5c4 Dismiss",
             "callback_data": f"fb:{transaction_id}:dismiss"},
        ], [
            {"text": "\U0001f50d Explain",
             "callback_data": f"ex:{transaction_id}:_"},
        ]],
    }


class TelegramController:
    """Long-polls getUpdates and dispatches commands and button taps.

    Handlers are injected rather than imported so this module knows nothing
    about the service, keeping it testable without a model or a network.
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        handlers: dict[str, Callable[..., str]],
        poll_timeout: int = 25,
    ):
        self.token = token
        self.chat_id = str(chat_id)
        self.handlers = handlers
        self.poll_timeout = poll_timeout
        self.offset: int | None = None
        self.last_error: str | None = None
        self.updates_handled = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    # -- transport -------------------------------------------------------
    def _api(self, method: str, **payload):
        import requests
        try:
            r = requests.post(f"{API}/bot{self.token}/{method}",
                              json=payload, timeout=self.poll_timeout + 10)
            if r.status_code != 200:
                self.last_error = f"{method}: HTTP {r.status_code}"
                return None
            return r.json()
        except Exception as exc:                               # noqa: BLE001
            self.last_error = f"{method}: {type(exc).__name__}"
            return None

    def send(self, text: str) -> bool:
        return self._api("sendMessage", chat_id=self.chat_id, text=text) is not None

    # -- dispatch --------------------------------------------------------
    def handle_command(self, text: str) -> str:
        """Parse one slash command and run its handler."""
        parts = text.strip().split()
        cmd = parts[0].lstrip("/").split("@")[0].lower()
        args = parts[1:]

        if cmd in ("start", "help"):
            return (
                "Fraud demo controls\n\n"
                "/stats - live tallies from the replay\n"
                "/review - how your escalate/dismiss calls turned out\n"
                "/threshold <0-1> - re-threshold the running service\n"
                "/threshold - show the current threshold\n"
                "/whatif <model> [threshold] - compare a model\n"
                "/models - list comparable models\n\n"
                "Alerts carry buttons: escalate, dismiss, or ask for an "
                "explanation. Outcomes stay hidden until /review, the way a "
                "real queue works."
            )
        if cmd == "stats":
            return self.handlers["stats"]()
        if cmd == "review":
            return self.handlers["review"]()
        if cmd == "models":
            return self.handlers["models"]()
        if cmd == "threshold":
            if not args:
                return self.handlers["get_threshold"]()
            try:
                value = float(args[0])
            except ValueError:
                return f"'{args[0]}' is not a number. Try /threshold 0.3"
            if not 0.0 <= value <= 1.0:
                return "Threshold must be between 0 and 1."
            return self.handlers["set_threshold"](value)
        if cmd == "whatif":
            if not args:
                return "Usage: /whatif xgboost 0.3"
            model = args[0]
            thr = None
            if len(args) > 1:
                try:
                    thr = float(args[1])
                except ValueError:
                    return f"'{args[1]}' is not a number."
            return self.handlers["whatif"](model, thr)
        return f"Unknown command '{cmd}'. Send /help."

    def handle_callback(self, data: str) -> str:
        """Handle an inline-button tap. Format: kind:transaction_id:value."""
        try:
            kind, tid, value = data.split(":", 2)
            tid_int = int(tid)
        except (ValueError, AttributeError):
            return "Unrecognised button."
        if kind == "fb":
            return self.handlers["feedback"](tid_int, value)
        if kind == "ex":
            return self.handlers["explain"](tid_int)
        return "Unrecognised button."

    # -- polling loop ----------------------------------------------------
    def _poll_once(self) -> int:
        payload = {"timeout": self.poll_timeout}
        if self.offset is not None:
            payload["offset"] = self.offset
        data = self._api("getUpdates", **payload)
        if not data or not data.get("ok"):
            return 0

        handled = 0
        for update in data.get("result", []):
            self.offset = update["update_id"] + 1
            try:
                if "callback_query" in update:
                    cq = update["callback_query"]
                    reply = self.handle_callback(cq.get("data", ""))
                    self._api("answerCallbackQuery",
                              callback_query_id=cq["id"], text=reply[:200])
                    self.send(reply)
                elif "message" in update:
                    text = update["message"].get("text", "")
                    if text.startswith("/"):
                        self.send(self.handle_command(text))
                handled += 1
            except Exception as exc:                           # noqa: BLE001
                self.last_error = f"dispatch: {type(exc).__name__}"
        self.updates_handled += handled
        return handled

    def _loop(self) -> None:
        import time
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:                                  # noqa: BLE001
                time.sleep(2)                                  # never die

    def start(self) -> bool:
        if not self.enabled or self._thread is not None:
            return False
        self._thread = threading.Thread(
            target=self._loop, name="telegram-controller", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
