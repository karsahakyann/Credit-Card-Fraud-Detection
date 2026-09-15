"""Two-way Telegram control: command parsing and button dispatch.

No network and no model: handlers are injected, which is why the controller
takes them as a dict rather than importing the service.
"""

from __future__ import annotations

import pytest

from fraud.telegram_control import TelegramController, build_alert_keyboard


@pytest.fixture
def calls():
    return {}


@pytest.fixture
def ctrl(calls):
    def rec(name):
        def inner(*a):
            calls[name] = a
            return f"{name}:{a}"
        return inner
    return TelegramController("tok", "chat", {
        "stats": rec("stats"), "get_threshold": rec("get_threshold"),
        "set_threshold": rec("set_threshold"), "models": rec("models"),
        "whatif": rec("whatif"), "feedback": rec("feedback"),
        "explain": rec("explain"),
    })


def test_keyboard_encodes_the_transaction(ctrl):
    kb = build_alert_keyboard(1885)
    data = [b["callback_data"] for row in kb["inline_keyboard"] for b in row]
    assert "fb:1885:fraud" in data and "fb:1885:legit" in data
    assert "ex:1885:_" in data


def test_help_lists_the_commands(ctrl):
    out = ctrl.handle_command("/help")
    for c in ("/stats", "/threshold", "/whatif"):
        assert c in out


def test_threshold_with_value_sets_it(ctrl, calls):
    ctrl.handle_command("/threshold 0.3")
    assert calls["set_threshold"] == (0.3,)


def test_threshold_without_value_reads_it(ctrl, calls):
    ctrl.handle_command("/threshold")
    assert "get_threshold" in calls and "set_threshold" not in calls


def test_threshold_rejects_out_of_range(ctrl, calls):
    assert "between 0 and 1" in ctrl.handle_command("/threshold 1.5")
    assert "set_threshold" not in calls


def test_threshold_rejects_nonsense(ctrl, calls):
    assert "not a number" in ctrl.handle_command("/threshold abc")
    assert "set_threshold" not in calls


def test_command_tolerates_botname_suffix(ctrl, calls):
    """Telegram appends @botname in group chats."""
    ctrl.handle_command("/stats@CreditFraud_bot")
    assert "stats" in calls


def test_whatif_parses_model_and_threshold(ctrl, calls):
    ctrl.handle_command("/whatif random_forest 0.22")
    assert calls["whatif"] == ("random_forest", 0.22)
    ctrl.handle_command("/whatif dnn")
    assert calls["whatif"] == ("dnn", None)


def test_unknown_command_is_not_fatal(ctrl):
    assert "Unknown command" in ctrl.handle_command("/banana")


def test_button_tap_records_feedback(ctrl, calls):
    ctrl.handle_callback("fb:1885:fraud")
    assert calls["feedback"] == (1885, "fraud")
    ctrl.handle_callback("fb:42:legit")
    assert calls["feedback"] == (42, "legit")


def test_explain_button_dispatches(ctrl, calls):
    ctrl.handle_callback("ex:1885:_")
    assert calls["explain"] == (1885,)


def test_malformed_callback_is_not_fatal(ctrl):
    assert "Unrecognised" in ctrl.handle_callback("garbage")
    assert "Unrecognised" in ctrl.handle_callback("")


def test_controller_disabled_without_credentials():
    assert TelegramController("", "", {}).enabled is False
    assert TelegramController("", "", {}).start() is False
