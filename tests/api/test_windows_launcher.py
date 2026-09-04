from __future__ import annotations

from unittest.mock import patch

import run_server


def test_windows_launcher_selects_socket_event_loop_policy():
    policy = object()
    with (
        patch.object(
            run_server.asyncio,
            "WindowsSelectorEventLoopPolicy",
            create=True,
            return_value=policy,
        ) as factory,
        patch.object(run_server.asyncio, "set_event_loop_policy") as setter,
    ):
        assert run_server.configure_event_loop("win32") is True

    factory.assert_called_once_with()
    setter.assert_called_once_with(policy)


def test_non_windows_launcher_preserves_default_event_loop_policy():
    with patch.object(run_server.asyncio, "set_event_loop_policy") as setter:
        assert run_server.configure_event_loop("linux") is False
    setter.assert_not_called()
