"""Secure web configuration helper tests."""
import json

import pytest

from colab_t4 import config


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def test_configuration_summary_reports_flags_without_secret_values(state):
    config.save_secrets({
        "tailscale_authkey": "ts-secret",
        "api_key": "api-secret",
        "hf_token": "hf-secret",
        "ssh_password": "pw-secret",
        "session": "lab-session",
        "model_repo": "repo/model",
        "quant": "Q4_K_M",
        "port": "8081",
        "ctx": "4096",
        "ssh_mode": "password",
        "tailnet": "example.ts.net",
    })

    assert hasattr(config, "configuration_summary"), "configuration_summary helper is missing"
    summary = config.configuration_summary()

    assert summary["session"] == "lab-session"
    assert summary["model"] == "repo/model"
    assert summary["port"] == 8081
    assert summary["ctx"] == 4096
    assert summary["ssh_mode"] == "password"
    assert summary["secrets"] == {
        "tailscale_authkey": True,
        "hf_token": True,
        "api_key": True,
        "ssh_password": True,
        "ssh_pubkey": False,
    }
    rendered = json.dumps(summary)
    assert "ts-secret" not in rendered
    assert "api-secret" not in rendered
    assert "hf-secret" not in rendered
    assert "pw-secret" not in rendered


def test_update_configuration_accepts_allowlisted_values_and_keeps_secret_write_only(state):
    assert hasattr(config, "update_configuration"), "update_configuration helper is missing"
    summary = config.update_configuration({
        "session": "web-session",
        "model": "new/model",
        "quant": "Q5_K_M",
        "port": 8090,
        "ctx": 8192,
        "ssh_mode": "tailscale",
        "api_key": "api-secret",
        "tailscale_authkey": "ts-secret",
    })

    saved = config.load_secrets()
    assert saved["session"] == "web-session"
    assert saved["model_repo"] == "new/model"
    assert saved["port"] == "8090"
    assert saved["ctx"] == "8192"
    assert saved["api_key"] == "api-secret"
    assert saved["tailscale_authkey"] == "ts-secret"
    assert config.secrets_path().stat().st_mode & 0o777 == 0o600
    assert summary["secrets"]["api_key"] is True
    assert "api-secret" not in json.dumps(summary)


def test_update_configuration_rejects_unknown_fields_and_empty_required_values(state):
    assert hasattr(config, "update_configuration"), "update_configuration helper is missing"
    with pytest.raises(ValueError, match="unknown configuration field"):
        config.update_configuration({"api_base": "http://internal"})
    with pytest.raises(ValueError, match="session is required"):
        config.update_configuration({"session": "   "})
    with pytest.raises(ValueError, match="port must be"):
        config.update_configuration({"port": 0})
    assert not config.secrets_path().exists()


def test_reset_configuration_requires_confirmation(state):
    config.save_secrets({"api_key": "api-secret"})

    assert hasattr(config, "reset_configuration"), "reset_configuration helper is missing"
    with pytest.raises(ValueError, match="confirmation is required"):
        config.reset_configuration(False)

    assert config.secrets_path().exists()
    config.reset_configuration(True)
    assert not config.secrets_path().exists()
