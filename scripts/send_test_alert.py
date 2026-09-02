"""Verify Telegram alerting without waiting for a fraud to appear.

Setup is easy to get subtly wrong -- most often by skipping the step where
you message your own bot first -- so this script diagnoses each failure mode
separately instead of just reporting that nothing arrived.

Usage:
    ./venv/bin/python scripts/send_test_alert.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.alerts import Alert, TelegramNotifier, load_dotenv


def mask(secret: str) -> str:
    """Never print a token in full, even to a local terminal."""
    if len(secret) < 12:
        return "***"
    return f"{secret[:6]}...{secret[-4:]}"


def main() -> int:
    loaded = load_dotenv()
    print(f"Loaded {loaded} value(s) from .env\n")

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("Not configured yet.\n")
        print("  TELEGRAM_BOT_TOKEN :", mask(token) if token else "(missing)")
        print("  TELEGRAM_CHAT_ID   :", chat_id or "(missing)")
        print("\nTo set up:")
        print("  1. Telegram -> @BotFather -> /newbot -> copy the token")
        print("  2. Open YOUR bot and press Start (a bot cannot message you first)")
        print("  3. Telegram -> @userinfobot -> copy the numeric Id")
        print("  4. cp .env.example .env  and paste both values in")
        return 1

    print(f"  token   : {mask(token)}")
    print(f"  chat_id : {chat_id}\n")

    # Check the token is valid before blaming the chat id.
    try:
        import requests

        me = requests.get(
            f"{TelegramNotifier.API}/bot{token}/getMe", timeout=10,
        ).json()
    except Exception as exc:                                   # noqa: BLE001
        print(f"Network error reaching Telegram: {type(exc).__name__}: {exc}")
        return 1

    if not me.get("ok"):
        print("Token rejected by Telegram. Re-copy it from @BotFather "
              "(it must include the digits before the colon).")
        return 1
    print(f"Bot authenticated: @{me['result'].get('username', '?')}")

    alert = Alert(
        transaction_id=0, amount=245.00, probability=0.991, threshold=0.11,
        model="XGBoost (test)",
        reasons=[("V14", 4.33), ("V10", 2.30), ("V12", 1.57)],
    )
    notifier = TelegramNotifier(token=token, chat_id=chat_id)
    if notifier.send(alert):
        print("\nTest alert sent. Check your phone.")
        return 0

    print(f"\nSend failed: {notifier.last_error}")
    print("\nMost common cause: you have not pressed Start in a chat with your")
    print("own bot yet. Telegram forbids bots from opening a conversation, so")
    print("the first message must come from you. Open the bot, send /start,")
    print("then run this again.")
    print("If it still fails, confirm TELEGRAM_CHAT_ID is your numeric id from")
    print("@userinfobot, not your @username.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
