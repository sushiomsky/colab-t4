import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

from colab_t4 import config, lifecycle, model


class LifecycleCLI:
    executable = "colab"
    version = "0.6.0"

    def __init__(self):
        self.calls = []

    def status_command(self, session):
        return ["colab", "status", "--session", session]

    def stop_command(self, session):
        return ["colab", "stop", "--session", session]

    def download_command(self, session, remote, local):
        return ["colab", "download", "--session", session, remote, str(local)]

    def run(self, args, log_path, **kwargs):
        self.calls.append(list(args))
        if args[1] == "download":
            Path(args[-1]).write_text(json.dumps({
                "ready": True,
                "gpu": "NVIDIA T4",
                "tailscale_ip": "100.64.0.9",
                "api_base": "http://100.64.0.9:8081/v1",
                "model": "/content/model.gguf",
                "tests": {"models": True, "chat": True},
            }), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")


class ModelCLI:
    executable = "colab"
    version = "0.6.0"

    def upload_command(self, session, local, remote):
        return ["colab", "upload", "--session", session, str(local), remote]

    def exec_command(self, session, local, timeout):
        return ["colab", "exec", "--session", session, "--file", str(local), "--timeout", str(timeout)]

    def download_command(self, session, remote, local):
        return ["colab", "download", "--session", session, remote, str(local)]

    def run(self, args, log_path, **kwargs):
        if args[1] == "download":
            target = Path(args[-1])
            if args[-2] == model.REMOTE_SWITCH_RESULT:
                target.write_text(json.dumps({"success": True}), encoding="utf-8")
            elif args[-2] == model.REMOTE_READY:
                target.write_text(json.dumps({
                    "ready": True,
                    "model": "/content/new.gguf",
                    "model_repo": "example/new-GGUF",
                    "quant": "Q4_K_M",
                    "tests": {"models": True, "chat": True, "cuda_offload": True},
                }), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")


def _lock_recorder(events):
    @contextmanager
    def locked(account_id):
        events.append(("enter", account_id))
        try:
            yield
        finally:
            events.append(("exit", account_id))
    return locked


def test_wait_holds_recorded_account_operation_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    config.save_state({"runtime_state": "creating", "session": "s", "account": "default"})
    events = []
    monkeypatch.setattr(lifecycle, "account_operation", _lock_recorder(events))
    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(lambda cls, home=None: LifecycleCLI()))

    options = type("Options", (), {"session": "s", "timeout": 1.0})()
    lifecycle.wait(options)
    assert events == [("enter", "default"), ("exit", "default")]


def test_down_holds_recorded_account_operation_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    config.save_state({"runtime_state": "ready", "session": "s", "account": "default"})
    events = []
    monkeypatch.setattr(lifecycle, "account_operation", _lock_recorder(events))
    monkeypatch.setattr(lifecycle.ColabCLI, "discover", classmethod(lambda cls, home=None: LifecycleCLI()))

    lifecycle.down()
    assert events == [("enter", "default"), ("exit", "default")]


def test_model_switch_holds_recorded_account_operation_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TS_AUTHKEY", "unused")
    config.save_state({
        "runtime_state": "ready",
        "session": "s",
        "account": "default",
        "model": "/content/old.gguf",
        "model_repo": "example/old-GGUF",
        "quant": "Q4_K_M",
        "api_base": "http://100.64.0.9:8081/v1",
        "tests": {"models": True, "chat": True, "cuda_offload": True},
    })
    config.save_runtime_api_key("api-test")
    events = []
    monkeypatch.setattr(model, "account_operation", _lock_recorder(events), raising=False)
    monkeypatch.setattr(model.ColabCLI, "discover", classmethod(lambda cls, home=None: ModelCLI()))

    model.switch_model("example/new-GGUF", "Q4_K_M", timeout=30)
    assert events == [("enter", "default"), ("exit", "default")]
