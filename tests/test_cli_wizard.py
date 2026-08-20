import argparse
import json

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
    import pytest

    from colab_t4.accounts import add_account, load_accounts

    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    add_account("work", home=str(tmp_path / "hw"))
    args = argparse.Namespace(accounts_command="remove", account="work", yes=False)
    with pytest.raises(SystemExit):
        cli.cmd_accounts(args)
    args.yes = True
    assert cli.cmd_accounts(args) == 0
    assert [account.id for account in load_accounts()] == ["default"]
