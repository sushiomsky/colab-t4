import argparse
import subprocess
import sys

import pytest

from colab_t4 import cli, services


def test_runtime_up_returns_lifecycle_state_and_reports_progress(monkeypatch):
    options = object()
    expected = {"runtime_state": "ready", "session": "test-session"}
    events = []
    monkeypatch.setattr(cli, "lifecycle_up", lambda received: expected if received is options else None)

    result = services.runtime_up(options, lambda message, percent: events.append((message, percent)))

    assert result == expected
    assert events == [("starting runtime", None), ("runtime ready", 100)]


def test_runtime_down_returns_structured_stopped_state(monkeypatch):
    monkeypatch.setattr(cli, "lifecycle_down", lambda: None)

    assert services.runtime_down() == {"runtime_state": "stopped"}


def test_runtime_down_forwards_cancellation_callback_to_lifecycle(monkeypatch):
    received = []
    cancelled = lambda: False
    monkeypatch.setattr(cli, "lifecycle_down", lambda *, cancelled: received.append(cancelled))

    assert services.runtime_down(cancelled=cancelled) == {"runtime_state": "stopped"}
    assert received == [cancelled]


def test_runtime_restart_preserves_lifecycle_errors(monkeypatch):
    expected = RuntimeError("runtime unavailable")
    monkeypatch.setattr(cli, "lifecycle_restart", lambda options: (_ for _ in ()).throw(expected))

    with pytest.raises(RuntimeError, match="runtime unavailable"):
        services.runtime_restart(object())


def test_runtime_status_includes_existing_exit_code(monkeypatch):
    monkeypatch.setattr(cli, "_status", lambda: ({"runtime_state": "ready"}, 0))

    assert services.runtime_status() == {"runtime_state": "ready", "exit_code": 0}


def test_run_doctor_retries_after_interactive_authentication(monkeypatch):
    calls = []
    responses = iter([
        ({"checks": {"colab_auth": False}}, 1),
        ({"checks": {"colab_auth": True}}, 0),
    ])
    monkeypatch.setattr(cli, "lifecycle_doctor", lambda: next(responses))
    monkeypatch.setattr(cli, "interactive_available", lambda requested: requested)
    monkeypatch.setattr(cli, "ensure_colab_auth", lambda **kwargs: calls.append(kwargs))

    assert services.run_doctor(interactive=True) == ({"checks": {"colab_auth": True}}, 0)
    assert calls == [{"interactive": True}]


def test_cmd_doctor_requests_auth_recovery_when_effective_policy_is_true(monkeypatch):
    received = []
    monkeypatch.setattr(cli, "interactive_available", lambda requested: True)
    monkeypatch.setattr(
        cli.services,
        "run_doctor",
        lambda *, interactive: received.append(interactive) or ({"checks": {}, "auth": {}}, 0),
    )

    assert cli.cmd_doctor(argparse.Namespace(json=False, interactive=False)) == 0
    assert received == [True]


def test_cmd_doctor_passes_effective_interactive_policy_to_service(monkeypatch):
    received = []
    availability_requests = []
    monkeypatch.setattr(
        cli,
        "interactive_available",
        lambda requested: availability_requests.append(requested) or False,
    )
    monkeypatch.setattr(
        cli.services,
        "run_doctor",
        lambda *, interactive: received.append(interactive) or ({"checks": {}, "auth": {}}, 0),
    )

    assert cli.cmd_doctor(argparse.Namespace(json=False, interactive=False)) == 0
    assert availability_requests == [False]
    assert received == [False]


def test_cmd_doctor_keeps_json_noninteractive(monkeypatch):
    received = []
    monkeypatch.setattr(cli, "interactive_available", lambda requested: True)
    monkeypatch.setattr(
        cli.services,
        "run_doctor",
        lambda *, interactive: received.append(interactive) or ({"checks": {}, "auth": {}}, 0),
    )

    assert cli.cmd_doctor(argparse.Namespace(json=True, interactive=True)) == 0
    assert received == [False]


def test_services_clean_import_does_not_preload_cli():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import colab_t4.services; assert 'colab_t4.cli' not in sys.modules"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cmd_status_preserves_json_output_and_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(cli.services, "runtime_status", lambda: {"runtime_state": "ready", "exit_code": 0})

    assert cli.cmd_status(argparse.Namespace(json=True)) == 0
    assert capsys.readouterr().out == '{\n  "runtime_state": "ready"\n}\n'


def test_cmd_down_preserves_success_output_and_lifecycle_errors(monkeypatch, capsys):
    monkeypatch.setattr(cli.services, "runtime_down", lambda: {"runtime_state": "stopped"})
    assert cli.cmd_down(argparse.Namespace()) == 0
    assert capsys.readouterr().out == "runtime stopped\n"

    monkeypatch.setattr(cli.services, "runtime_down", lambda: (_ for _ in ()).throw(RuntimeError("stop failed")))
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_down(argparse.Namespace())
    assert exc_info.value.code == 1
    assert capsys.readouterr().err == "error: stop failed\n"


def test_cmd_restart_preserves_success_output_and_lifecycle_errors(monkeypatch, capsys):
    monkeypatch.setattr(cli.services, "runtime_restart", lambda options: {"runtime_state": "ready"})
    monkeypatch.setattr(cli, "print_connections", lambda: None)
    assert cli.cmd_restart(argparse.Namespace()) == 0
    assert capsys.readouterr().out == "runtime restarted\n"

    monkeypatch.setattr(cli.services, "runtime_restart", lambda options: (_ for _ in ()).throw(RuntimeError("restart failed")))
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_restart(argparse.Namespace())
    assert exc_info.value.code == 1
    assert capsys.readouterr().err == "error: restart failed\n"


def test_cmd_api_models_preserves_output_and_chat_errors(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_state", lambda: {"api_base": "http://example.test/v1"})
    monkeypatch.setattr(cli.services, "api_models", lambda: {"data": [{"id": "local"}]})
    assert cli.cmd_api(argparse.Namespace(action="models", message="ignored")) == 0
    assert capsys.readouterr().out == '{\n  "data": [\n    {\n      "id": "local"\n    }\n  ]\n}\n'

    monkeypatch.setattr(cli.services, "api_chat", lambda message: (_ for _ in ()).throw(RuntimeError("chat failed")))
    with pytest.raises(RuntimeError, match="chat failed"):
        cli.cmd_api(argparse.Namespace(action="chat", message="hello"))


def test_api_services_preserve_api_request_payloads_and_errors(monkeypatch):
    calls = []

    def request(path, *, method="GET", payload=None):
        calls.append((path, method, payload))
        if path == "models":
            return {"data": [{"id": "local"}]}
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(cli, "_api_request", request)

    assert services.api_models() == {"data": [{"id": "local"}]}
    with pytest.raises(RuntimeError, match="backend unavailable"):
        services.api_chat("hello")
    assert calls == [
        ("models", "GET", None),
        ("chat/completions", "POST", {
            "model": "local",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
        }),
    ]
