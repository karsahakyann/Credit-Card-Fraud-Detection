"""Find your Telegram chat id using your own bot, no third-party bot needed.

@userinfobot is often recommended for this and is not always reachable. The
authoritative source is your own bot: once you have sent it a message,
Telegram's getUpdates returns the chat it came from, including the numeric
id the API needs.

Usage:
    ./venv/bin/python scripts/find_chat_id.py            # show the id
    ./venv/bin/python scripts/find_chat_id.py --write    # also save to .env
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.alerts import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def mask(secret: str) -> str:
    return f"{secret[:6]}...{secret[-4:]}" if len(secret) >= 12 else "***"


def write_chat_id(chat_id: str) -> None:
    """Set TELEGRAM_CHAT_ID in .env, preserving everything else."""
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    out, replaced = [], False
    for line in lines:
        if line.strip().startswith("TELEGRAM_CHAT_ID="):
            out.append(f"TELEGRAM_CHAT_ID={chat_id}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"TELEGRAM_CHAT_ID={chat_id}")
    ENV_PATH.write_text("\n".join(out) + "\n")


def main() -> int:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        print("No TELEGRAM_BOT_TOKEN found.\n")
        print("Get one first: Telegram -> @BotFather -> /newbot")
        print("Then put it in .env:  TELEGRAM_BOT_TOKEN=123456:AA...")
        print("(cp .env.example .env  if you have not made the file yet)")
        return 1

    print(f"Using token {mask(token)}\n")
    import requests

    try:
        me = requests.get(
            f"https://api.telegram.org/bot{token}/getMe", timeout=10).json()
    except Exception as exc:                                  # noqa: BLE001
        print(f"Could not reach Telegram: {type(exc).__name__}: {exc}")
        return 1

    if not me.get("ok"):
        print("Telegram rejected this token. Re-copy it from @BotFather --")
        print("it must include the digits before the colon.")
        return 1

    username = me["result"].get("username", "?")
    print(f"Bot authenticated: @{username}\n")

    updates = requests.get(
        f"https://api.telegram.org/bot{token}/getUpdates", timeout=10).json()
    results = updates.get("result", []) if updates.get("ok") else []

    chats: dict[str, str] = {}
    for item in results:
        msg = (item.get("message") or item.get("edited_message")
               or item.get("channel_post") or {})
        chat = msg.get("chat") or {}
        if "id" in chat:
            who = chat.get("username") or chat.get("first_name") or chat.get("title", "")
            chats[str(chat["id"])] = f"{who} ({chat.get('type', '?')})"

    if not chats:
        print("Telegram has no messages for this bot yet.\n")
        print("A bot cannot start a conversation, so you must message it first:")
        print(f"  1. Open  https://t.me/{username}")
        print("  2. Press Start, or send it any message such as 'hello'")
        print("  3. Run this script again\n")
        print("Note: Telegram only keeps recent updates. If you messaged the bot")
        print("a long time ago, or another process already consumed the update,")
        print("simply send it a fresh message and retry.")
        return 1

    print("Found chat id(s):\n")
    for cid, who in chats.items():
        print(f"  {cid}    {who}")

    chat_id = next(iter(chats))
    if "--write" in sys.argv:
        write_chat_id(chat_id)
        print(f"\nSaved TELEGRAM_CHAT_ID={chat_id} to .env")
        print("Verify with: ./venv/bin/python scripts/send_test_alert.py")
    else:
        print(f"\nAdd this line to .env:\n  TELEGRAM_CHAT_ID={chat_id}")
        print("Or rerun with --write to save it automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
