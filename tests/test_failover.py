"""Failover provisioning and per-account CLI environment tests."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from colab_t4 import lifecycle
from colab_t4.accounts import add_account, load_accounts
from colab_t4.backend import ColabCLI


class FakeCLI:
    executable = "colab"
    version = "0.6.0"

    def __init__(self, home=None, fail_new=False, dead_status=False):
        self.home = home
        self.fail_new = fail_new
        self.dead_status = dead_status
        self.calls = []

    def new_command(self, session, gpu):
        return ["colab", "new", "--session", session, "--gpu", gpu]

    def upload_command(self, session, local, remote):
        return ["colab", "upload", "--session", session, str(local), remote]

    def exec_command(self, session, local, timeout):
        return ["colab", "exec", "--session", session, "--file", str(local), "--timeout", str(timeout)]

    def download_command(self, session, remote, local):
        return ["colab", "download", "--session", session, remote, str(local)]

    def status_command(self, session):
        return ["colab", "status", "--session", session]

    def stop_command(self, session):
        return ["colab", "stop", "--session", session]

    def run(self, args, log_path, **kwargs):
        self.calls.append(list(args))
        if args[1] == "status" and self.dead_status:
            return subprocess.CompletedProcess(args, 1, "", "error: session not found")
        if args[1] == "new" and self.fail_new:
            return subprocess.CompletedProcess(args, 1, "", "error: Service Unavailable")
        if args[1] == "download" and args[-2] == "/content/.colab-t4-ready.json":
            Path(args[-1]).write_text(json.dumps({
                "ready": True,
                "gpu": "NVIDIA T4, 15360 MiB",
                "tailscale_ip": "100.64.0.2",
                "api_base": "http://100.64.0.2:8080/v1",
                "model": "/content/model/model.Q4_K_M.gguf",
                "tests": {"chat": True, "models": True},
            }))
        return subprocess.CompletedProcess(args, 0, "", "")


def options(**overrides):
    values = {
        "session": "colab-t4", "model": lifecycle.DEFAULT_MODEL, "quant": "Q4_K_M",
        "port": 8080, "ctx": 8192, "api_key": "api-test", "password": "pw-test",
        "pubkey": None, "exec_timeout": 30,
    }
    values.update(overrides)
    return type("Options", (), values)()


def test_up_fails_over_to_next_account(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TS_AUTHKEY", "tskey-test-only")
    home_a = str(tmp_path / "home-a")
    home_b = str(tmp_path / "home-b")
    Path(home_a).mkdir()
    Path(home_b).mkdir()
    add_account("a", home=home_a)
    add_account("b", home=home_b)

    discovered = []

    def fake_discover(cls, home=None):
        discovered.append(home)
        return FakeCLI(home=home, fail_new=home in (None, home_a))

    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(fake_discover))
    monkeypatch.setattr(lifecycle, "authenticate_available", lambda home=None: True)

    result = lifecycle.up(options())
    assert result["runtime_state"] == "ready"
    assert result["account"] == "b"
    accounts = {a.id: a for a in load_accounts()}
    assert accounts["default"].last_error
    assert accounts["a"].last_error
    assert accounts["b"].last_ok and not accounts["b"].last_error
    assert discovered == [None, home_a, home_b]
    err = capsys.readouterr().err
    assert "trying next account" in err


def test_up_single_account_keeps_original_error(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TS_AUTHKEY", "tskey-test-only")

    def fake_discover(cls, home=None):
        return FakeCLI(home=home, fail_new=True)

    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(fake_discover))
    monkeypatch.setattr(lifecycle, "authenticate_available", lambda home=None: True)
    with pytest.raises(Exception) as excinfo:
        lifecycle.up(options())
    assert "Colab CLI could not create the T4 session" in str(excinfo.value)


def test_up_fails_over_when_existing_session_is_lost(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TS_AUTHKEY", "tskey-test-only")
    home_a = str(tmp_path / "home-a")
    home_b = str(tmp_path / "home-b")
    Path(home_a).mkdir()
    Path(home_b).mkdir()
    add_account("a", home=home_a)
    add_account("b", home=home_b)
    # Recorded session was ready but its Colab runtime died; account a hosted it.
    lifecycle._update(session="colab-t4", runtime_state="ready", account="a")

    def fake_discover(cls, home=None):
        return FakeCLI(home=home, fail_new=home in (None, home_a), dead_status=home in (None, home_a))

    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(fake_discover))
    monkeypatch.setattr(lifecycle, "authenticate_available", lambda home=None: True)

    result = lifecycle.up(options())
    assert result["runtime_state"] == "ready"
    assert result["account"] == "b"
    accounts = {a.id: a for a in load_accounts()}
    assert accounts["a"].last_error and "did not become ready" in accounts["a"].last_error
    assert accounts["b"].last_ok and not accounts["b"].last_error
    err = capsys.readouterr().err
    assert "trying next account" in err


def test_down_uses_recorded_account_home(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    home_a = str(tmp_path / "home-a")
    Path(home_a).mkdir()
    add_account("a", home=home_a)
    lifecycle._update(session="exact-session", runtime_state="ready", account="a")

    discovered = []

    def fake_discover(cls, home=None):
        discovered.append(home)
        return FakeCLI(home=home)

    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(fake_discover))
    lifecycle.down()
    assert discovered == [home_a]


def test_wait_uses_recorded_account_home(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    home_a = str(tmp_path / "home-a")
    Path(home_a).mkdir()
    add_account("a", home=home_a)
    lifecycle._update(session="exact-session", runtime_state="creating", account="a")

    discovered = []

    def fake_discover(cls, home=None):
        discovered.append(home)
        return FakeCLI(home=home)

    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(fake_discover))
    lifecycle.wait(options(timeout=1.0))
    assert discovered == [home_a]


def test_run_uses_account_home_env(monkeypatch, tmp_path):
    captured = {}

    def fake_run(args, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    home = str(tmp_path / "acc-home")
    cli = ColabCLI("colab", "0.6.0", home=home)
    cli.run(["colab", "sessions"], tmp_path / "auth.log")
    assert captured["env"]["HOME"] == home


def test_run_without_account_keeps_real_home(monkeypatch, tmp_path):
    captured = {}

    def fake_run(args, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    original = os.environ.get("HOME")
    cli = ColabCLI("colab", "0.6.0")
    cli.run(["colab", "sessions"], tmp_path / "auth.log")
    assert captured["env"]["HOME"] == original


def test_up_without_any_account_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))

    def fake_discover(cls, home=None):
        return FakeCLI(home=home)

    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(fake_discover))
    monkeypatch.setattr(lifecycle, "authenticate_available", lambda home=None: False)
    with pytest.raises(Exception) as excinfo:
        lifecycle.up(options())
    assert "no Colab CLI authentication" in str(excinfo.value)
