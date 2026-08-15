from pathlib import Path

from colab_t4.config import load_secrets, secrets_path
from colab_t4.wizard import collect, configure_status, persist, reset


class Options:
    session = None
    model = None
    quant = None
    port = None
    ctx = None
    exec_timeout = None
    api_key = None
    password = None
    pubkey = None
    ssh_mode = "password"

def test_first_interactive_run_keeps_secrets_in_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "cfg"))
    answers = iter(["", "n", "", "", "", "", "", "", ""])
    hidden = iter(["tskey-session", "ssh-secret", "api-secret", ""])
    values = collect(Options(), input_fn=lambda _: next(answers), getpass_fn=lambda _: next(hidden), force=True)
    assert values["tailscale_authkey"] == "tskey-session"
    assert values["ssh_password"] == "ssh-secret"
    assert values["api_key"] == "api-secret"
    assert values["persist_secrets"] is False
    assert not secrets_path().exists()


def test_auth_recovery_runs_sessions_flow_then_rechecks(monkeypatch):
    import colab_t4.wizard as wizard

    available = {"value": False}
    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    class FakeCLI:
        def run_interactive_auth(self):
            calls.append("interactive")
            available["value"] = True
            return 0
        def auth_log_path(self):
            return Path("/tmp/colab-t4-auth-test.log")
        def sessions_command(self):
            calls.append("sessions-command")
            return ["colab", "sessions"]

        def run(self, command, log_path, timeout=None):
            calls.append(("verify", command, timeout))
            return Result()

    monkeypatch.setattr(wizard, "authenticate_available", lambda: available["value"])
    wizard.ensure_colab_auth(interactive=True, input_fn=lambda _: "y", cli=FakeCLI())
    assert calls == ["interactive", "sessions-command", ("verify", ["colab", "sessions"], 60)]

def test_precedence_environment_over_saved(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "cfg"))
    persist({"persist_secrets": True, "tailscale_authkey": "saved-ts", "api_key": "saved-api", "ssh_password": "saved-pw"})
    monkeypatch.setenv("TS_AUTHKEY", "env-ts")
    monkeypatch.setenv("COLAB_T4_API_KEY", "env-api")
    monkeypatch.setenv("COLAB_T4_SSH_PASSWORD", "env-pw")
    values = collect(Options(), force=False)
    assert values["tailscale_authkey"] == "env-ts"
    assert values["api_key"] == "env-api"
    assert values["ssh_password"] == "env-pw"


def test_persisted_config_is_0600_and_show_is_redacted(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "cfg"))
    persist({"persist_secrets": True, "tailscale_authkey": "ts-secret", "api_key": "api-secret", "ssh_password": "pw-secret"})
    assert oct(secrets_path().stat().st_mode & 0o777) == "0o600"
    shown = configure_status()
    assert shown["tailscale_authkey"] is True
    assert "ts-secret" not in str(shown)


def test_reset_yes_removes_saved_config(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "cfg"))
    persist({"persist_secrets": True, "api_key": "api-secret"})
    reset(yes=True)
    assert not secrets_path().exists()
