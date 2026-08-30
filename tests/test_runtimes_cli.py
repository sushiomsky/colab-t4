import json

import pytest

from colab_t4 import cli, config
from colab_t4.runtimes import create_runtime, runtime_context


def _unknown_status(observed):
    observed.append(config.selected_runtime())
    return {
        "runtime_state": "unknown",
        "session": None,
        "account": None,
        "gpu": None,
        "accelerator": None,
        "tailscale": {},
        "api": {},
        "model": {},
        "last_error": None,
    }, 1


def test_explicit_runtime_selector_beats_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    create_runtime("coder", model_repo="example/coder", quant="Q4_K_M")
    create_runtime("research", model_repo="example/research", quant="Q4_K_M")
    monkeypatch.setenv("COLAB_T4_RUNTIME", "research")
    observed = []
    monkeypatch.setattr(cli, "_status", lambda: _unknown_status(observed))

    assert cli.main(["--runtime", "coder", "status", "--json"]) == 1
    json.loads(capsys.readouterr().out)
    assert observed == ["coder"]


def test_environment_selects_runtime_when_flag_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    create_runtime("research", model_repo="example/research", quant="Q4_K_M")
    monkeypatch.setenv("COLAB_T4_RUNTIME", "research")
    observed = []
    monkeypatch.setattr(cli, "_status", lambda: _unknown_status(observed))

    assert cli.main(["status", "--json"]) == 1
    json.loads(capsys.readouterr().out)
    assert observed == ["research"]


def test_runtime_create_and_list_json_never_expose_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    assert cli.main(["runtimes", "create", "coder", "--model", "example/coder", "--quant", "Q5_K_M", "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["id"] == "coder"
    assert created["session"] == "colab-t4-coder"
    assert created["api_key_configured"] is True
    assert "api_key" not in created

    assert cli.main(["runtimes", "list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["id"] for row in rows} == {"default", "coder"}
    assert all("api_key" not in row for row in rows)


def test_missing_named_runtime_fails_without_falling_back(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    with pytest.raises(SystemExit):
        cli.main(["--runtime", "missing", "status", "--json"])
    error = capsys.readouterr().err
    assert "runtime 'missing' does not exist" in error


def test_named_selector_is_rejected_for_global_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    create_runtime("coder", model_repo="example/coder", quant="Q4_K_M")
    with pytest.raises(SystemExit):
        cli.main(["--runtime", "coder", "accounts", "list", "--json"])
    assert "global command" in capsys.readouterr().err


def test_runtime_selector_preserves_raw_ssh_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    create_runtime("coder", model_repo="example/coder", quant="Q4_K_M")
    with runtime_context("coder"):
        config.save_state({"runtime_state": "ready", "session": "colab-t4-coder", "tailscale_ip": "100.64.0.7"})
    called = []
    monkeypatch.setattr(cli.os, "execvp", lambda exe, argv: called.append((exe, argv)))

    assert cli.main(["--runtime", "coder", "ssh", "--", "uname", "-a"]) == 0
    assert called[0][0] == "ssh"
    assert called[0][1][-2:] == ["uname", "-a"]
    assert "root@100.64.0.7" in called[0][1]


def test_default_runtime_selector_keeps_legacy_context(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    observed = []
    monkeypatch.setattr(cli, "_status", lambda: _unknown_status(observed))
    assert cli.main(["--runtime", "default", "status", "--json"]) == 1
    json.loads(capsys.readouterr().out)
    assert observed == ["default"]


def test_named_restart_resolves_runtime_config_before_lifecycle_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TS_AUTHKEY", "tskey-test-only")
    create_runtime("coder", model_repo="example/coder", quant="Q5_K_M")
    observed = []

    def fake_restart(args):
        observed.append({
            "runtime": config.selected_runtime(),
            "session": args.session,
            "model": args.model,
            "quant": args.quant,
            "api_key": args.api_key,
            "runtime_config": dict(args.runtime_config),
        })
        return {"runtime_state": "ready"}

    monkeypatch.setattr(cli, "lifecycle_restart", fake_restart)
    assert cli.main(["--runtime", "coder", "restart", "--non-interactive"]) == 0
    assert observed[0]["runtime"] == "coder"
    assert observed[0]["session"] == "colab-t4-coder"
    assert observed[0]["model"] == "example/coder"
    assert observed[0]["quant"] == "Q5_K_M"
    assert observed[0]["api_key"]
    assert observed[0]["runtime_config"]["api_key"] == observed[0]["api_key"]
