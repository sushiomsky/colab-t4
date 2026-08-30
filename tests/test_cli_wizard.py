import argparse
import json
import subprocess
from pathlib import Path

import pytest

import colab_t4.cli as cli


def test_up_passes_collected_values_into_same_provision_call(monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", "/tmp/colab-t4-test-state")
    values = {
        "session": "wizard-session",
        "model": "repo/model",
        "quant": "Q4_K_M",
        "port": 8080,
        "ctx": 8192,
        "exec_timeout": 30.0,
        "ready_timeout": 60.0,
        "tailscale_authkey": "tskey-memory-only",
        "api_key": "api-memory-only",
        "ssh_password": "ssh-memory-only",
        "ssh_pubkey": "",
        "persist_secrets": False,
    }
    monkeypatch.setattr(cli, "interactive_available", lambda force=False: True)
    monkeypatch.setattr(cli.ColabCLI, "discover", classmethod(lambda cls: object()))
    monkeypatch.setattr(cli, "ensure_colab_auth", lambda **kwargs: None)
    monkeypatch.setattr(cli, "collect", lambda *args, **kwargs: dict(values))
    monkeypatch.setattr(cli, "persist", lambda _: None)
    captured = {}
    monkeypatch.setattr(cli, "lifecycle_up", lambda options: captured.update(vars(options)))
    args = argparse.Namespace(
        interactive=True, non_interactive=False, yes=True, session=None, model=None,
        quant=None, port=None, ctx=None, api_key=None, password=None, pubkey=None,
        exec_timeout=None,
    )
    assert cli.cmd_up(args) == 0
    assert captured["tailscale_authkey"] == "tskey-memory-only"
    assert captured["api_key"] == "api-memory-only"
    assert captured["ssh_password"] == "ssh-memory-only"
    assert captured["runtime_config"]["persist_secrets"] is False


def test_accounts_list_json(tmp_path, monkeypatch, capsys):
    from colab_t4.accounts import add_account

    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    add_account("work", home=str(tmp_path / "hw"))
    args = argparse.Namespace(accounts_command="list", json=True)
    assert cli.cmd_accounts(args) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["id"] for row in rows} == {"default", "work"}


def test_accounts_remove_requires_yes_outside_tty(tmp_path, monkeypatch):
    from colab_t4.accounts import add_account, load_accounts

    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    add_account("work", home=str(tmp_path / "hw"))
    args = argparse.Namespace(accounts_command="remove", account="work", yes=False)
    with pytest.raises(SystemExit):
        cli.cmd_accounts(args)
    args.yes = True
    assert cli.cmd_accounts(args) == 0
    assert [account.id for account in load_accounts()] == ["default"]


def test_accounts_auth_import_registers_profile(tmp_path, monkeypatch):
    """auth-import copies a token.json, verifies, and registers the account."""
    from colab_t4 import accounts

    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({"refresh_token": "rt", "client_id": "cid", "client_secret": "cs"}))

    captured_home = {}

    class FakeCLI:
        def __init__(self, home=None):
            self.home = home
            captured_home["home"] = home
        def sessions_command(self):
            return ["colab", "sessions"]
        def auth_log_path(self):
            return tmp_path / "auth.log"
        def run(self, args, log_path, timeout=None, **kwargs):
            return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(cli.ColabCLI, "discover", classmethod(lambda cls, home=None: FakeCLI(home)))
    monkeypatch.setattr("colab_t4.wizard.authenticate_available", lambda home=None: True)
    monkeypatch.setattr(cli, "resolve_account_email", lambda home: "imported@example.com")

    args = argparse.Namespace(
        accounts_command="auth-import", id="imp", token=str(token_file), email=None,
    )
    assert cli.cmd_accounts(args) == 0
    assert accounts.get_account("imp").home == str(tmp_path / "state" / "accounts" / "imp")
    assert accounts.get_account("imp").email == "imported@example.com"
    token_path = Path(captured_home["home"]) / ".config" / "colab-cli" / "token.json"
    assert token_path.is_file()
    assert json.loads(token_path.read_text()) == {"refresh_token": "rt", "client_id": "cid", "client_secret": "cs"}


def test_accounts_auth_import_rejects_missing_token(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    args = argparse.Namespace(
        accounts_command="auth-import", id="imp", token=str(tmp_path / "missing.json"), email=None,
    )
    with pytest.raises(SystemExit):
        cli.cmd_accounts(args)
    assert not (tmp_path / "state" / "accounts" / "imp").exists()


def test_serve_defaults_to_tailscale_ipv4_when_available(monkeypatch):
    """Removing Tailscale discovery makes the default serve bind localhost only."""
    captured = {}

    def fake_run(command, **kwargs):
        assert command == ["tailscale", "ip", "-4"]
        return subprocess.CompletedProcess(command, 0, "100.96.1.23\n", "")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "router_serve", lambda *, host, port: captured.update({"host": host, "port": port}))
    args = argparse.Namespace(host=None, port=8089, allow_non_tailnet_bind=False)

    assert cli.cmd_serve(args) == 0
    assert captured == {"host": "100.96.1.23", "port": 8089}


def test_serve_defaults_to_localhost_when_tailscale_ipv4_unavailable(monkeypatch):
    """The secure default remains usable on machines without the Tailscale CLI."""
    captured = {}

    def fake_run(command, **kwargs):
        raise FileNotFoundError("tailscale")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "router_serve", lambda *, host, port: captured.update({"host": host, "port": port}))
    args = argparse.Namespace(host=None, port=8090, allow_non_tailnet_bind=False)

    assert cli.cmd_serve(args) == 0
    assert captured == {"host": "127.0.0.1", "port": 8090}


def test_serve_rejects_non_tailnet_host_without_explicit_opt_in(monkeypatch):
    """Binding all interfaces must require an explicit policy override."""
    called = False

    def fake_serve(*, host, port):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "router_serve", fake_serve)
    args = argparse.Namespace(host="0.0.0.0", port=8089, allow_non_tailnet_bind=False)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_serve(args)

    assert exc.value.code == 1
    assert called is False


def test_serve_allows_non_tailnet_host_with_explicit_opt_in(monkeypatch):
    """Operators can still expose the router deliberately when they choose the risk."""
    captured = {}
    monkeypatch.setattr(cli, "router_serve", lambda *, host, port: captured.update({"host": host, "port": port}))
    args = argparse.Namespace(host="0.0.0.0", port=8089, allow_non_tailnet_bind=True)

    assert cli.cmd_serve(args) == 0
    assert captured == {"host": "0.0.0.0", "port": 8089}


def test_serve_parser_defaults_host_to_policy_resolution():
    """Changing the parser default back to localhost bypasses Tailscale resolution."""
    args = cli.build_parser().parse_args(["serve"])
    assert args.host is None
    assert args.allow_non_tailnet_bind is False
