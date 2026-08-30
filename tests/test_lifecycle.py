import json
import subprocess
from pathlib import Path

from colab_t4 import config, lifecycle
from colab_t4.config import load_state
from colab_t4.runtimes import create_runtime, runtime_context


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


def _options(**overrides):
    values = {
        "session": "colab-t4", "model": lifecycle.DEFAULT_MODEL, "quant": "Q4_K_M",
        "port": 8080, "ctx": 8192, "api_key": "api-test", "password": "pw-test",
        "pubkey": None, "exec_timeout": 30,
    }
    values.update(overrides)
    return type("Options", (), values)()


def test_browserless_up_command_sequence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TS_AUTHKEY", "tskey-test-only")
    fake = FakeCLI()
    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(lambda cls, home=None: fake))
    monkeypatch.setattr(lifecycle, "authenticate_available", lambda home=None: True)
    options = _options()
    result = lifecycle.up(options)
    assert result["runtime_state"] == "ready"
    assert fake.calls[0] == ["colab", "new", "--session", "colab-t4", "--gpu", "T4"]
    assert any(call[1] == "exec" for call in fake.calls)
    assert result["tailscale_ip"] == "100.64.0.2"
    assert result["model_repo"] == lifecycle.DEFAULT_MODEL
    assert result["quant"] == "Q4_K_M"
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


def test_named_down_stops_only_selected_session_and_key(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    create_runtime("coder", model_repo="example/coder", quant="Q4_K_M")
    create_runtime("research", model_repo="example/research", quant="Q5_K_M")
    with runtime_context("coder"):
        config.save_state({"runtime_state": "ready", "session": "coder-session", "tailscale_ip": "100.64.0.1"})
        coder_key_path = config.runtime_api_key_path()
        assert coder_key_path.exists()
    with runtime_context("research"):
        config.save_state({"runtime_state": "ready", "session": "research-session", "tailscale_ip": "100.64.0.2"})
        research_key_path = config.runtime_api_key_path()
        research_key = research_key_path.read_bytes()

    fake = FakeCLI()
    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(lambda cls, home=None: fake))
    with runtime_context("coder"):
        lifecycle.down()
        assert config.load_state() == {}
        assert not config.runtime_api_key_path().exists()

    assert fake.calls == [["colab", "stop", "--session", "coder-session"]]
    with runtime_context("research"):
        assert config.load_state()["session"] == "research-session"
        assert config.runtime_api_key_path().read_bytes() == research_key


def test_failed_coder_provision_does_not_mutate_research_state(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TS_AUTHKEY", "tskey-test-only")
    create_runtime("coder", model_repo="example/coder", quant="Q4_K_M")
    create_runtime("research", model_repo="example/research", quant="Q5_K_M")
    with runtime_context("research"):
        config.save_state({"runtime_state": "ready", "session": "research-session", "model_repo": "example/research"})
        research_bytes = config.state_path().read_bytes()

    class FailingCLI(FakeCLI):
        def run(self, args, log_path, **kwargs):
            self.calls.append(list(args))
            if args[1] == "new":
                return subprocess.CompletedProcess(args, 1, "", "failed")
            return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(lambda cls, home=None: FailingCLI()))
    monkeypatch.setattr(lifecycle, "authenticate_available", lambda home=None: True)
    with runtime_context("coder"):
        try:
            lifecycle.up(_options(session="colab-t4-coder", model="example/coder"))
        except Exception:
            pass

    with runtime_context("research"):
        assert config.state_path().read_bytes() == research_bytes
