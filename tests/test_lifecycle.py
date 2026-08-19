import json
import subprocess
from pathlib import Path

from colab_t4 import lifecycle
from colab_t4.config import load_state


class FakeCLI:
    executable = "colab"
    version = "0.6.0"

    def __init__(self):
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


def test_browserless_up_command_sequence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TS_AUTHKEY", "tskey-test-only")
    fake = FakeCLI()
    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(lambda cls, home=None: fake))
    monkeypatch.setattr(lifecycle, "authenticate_available", lambda home=None: True)
    options = type("Options", (), {
        "session": "colab-t4", "model": lifecycle.DEFAULT_MODEL, "quant": "Q4_K_M",
        "port": 8080, "ctx": 8192, "api_key": "api-test", "password": "pw-test",
        "pubkey": None, "exec_timeout": 30,
    })()
    result = lifecycle.up(options)
    assert result["runtime_state"] == "ready"
    assert fake.calls[0] == ["colab", "new", "--session", "colab-t4", "--gpu", "T4"]
    assert any(call[1] == "exec" for call in fake.calls)
    assert result["tailscale_ip"] == "100.64.0.2"
    log_text = "".join(path.read_text() for path in (tmp_path / "state" / "logs").glob("*.log"))
    assert "api-test" not in log_text


def test_down_targets_recorded_session(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    lifecycle._update(session="exact-session", runtime_state="ready")
    fake = FakeCLI()
    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(lambda cls, home=None: fake))
    lifecycle.down()
    assert fake.calls == [["colab", "stop", "--session", "exact-session"]]
    assert load_state() == {}
