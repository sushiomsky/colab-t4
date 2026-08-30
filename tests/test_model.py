import json
import subprocess
from pathlib import Path

import pytest

from colab_t4 import model
from colab_t4.accounts import Account, save_accounts
from colab_t4.config import load_state, save_secrets, save_state


class FakeCLI:
    executable = "colab"
    version = "0.6.0"

    def __init__(self, *, result=None, ready=None, exec_returncode=0):
        self.calls = []
        self.result = result or {"success": True}
        self.ready = ready or {
            "ready": True,
            "model": "/content/models/new/model.Q4_K_M.gguf",
            "model_repo": "example/new-model-GGUF",
            "quant": "Q4_K_M",
            "tests": {"models": True, "chat": True, "cuda_offload": True},
        }
        self.exec_returncode = exec_returncode

    def upload_command(self, session, local, remote):
        return ["colab", "upload", "--session", session, str(local), remote]

    def exec_command(self, session, local, timeout):
        return ["colab", "exec", "--session", session, "--file", str(local), "--timeout", str(timeout)]

    def download_command(self, session, remote, local):
        return ["colab", "download", "--session", session, remote, str(local)]

    def run(self, args, log_path, **kwargs):
        self.calls.append(list(args))
        if args[1] == "download":
            remote = args[-2]
            target = Path(args[-1])
            if remote == model.REMOTE_SWITCH_RESULT:
                target.write_text(json.dumps(self.result), encoding="utf-8")
            elif remote == model.REMOTE_READY:
                target.write_text(json.dumps(self.ready), encoding="utf-8")
        code = self.exec_returncode if args[1] == "exec" else 0
        return subprocess.CompletedProcess(args, code, "", "")


def ready_state(tmp_path, *, account="default"):
    save_state({
        "session": "colab-t4",
        "account": account,
        "runtime_state": "ready",
        "model": "/content/model/old.Q4_K_M.gguf",
        "model_repo": "example/old-GGUF",
        "quant": "Q4_K_M",
        "api_base": "http://100.64.0.2:8081/v1",
        "tests": {"models": True, "chat": True, "cuda_offload": True},
    })
    save_secrets({"api_key": "api-secret-test", "hf_token": "hf-secret-test", "ctx": "8192", "port": "8081"})


def test_current_model_reports_recorded_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    ready_state(tmp_path)
    assert model.current_model() == {
        "session": "colab-t4",
        "account": "default",
        "runtime_state": "ready",
        "model": "/content/model/old.Q4_K_M.gguf",
        "model_repo": "example/old-GGUF",
        "quant": "Q4_K_M",
    }


def test_switch_requires_ready_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    save_state({"session": "colab-t4", "runtime_state": "stopped"})
    with pytest.raises(RuntimeError, match="ready"):
        model.switch_model("example/new-model-GGUF", "Q4_K_M")


def test_switch_uses_recorded_account_home(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    account_home = tmp_path / "account-home"
    save_accounts([Account(id="default"), Account(id="work", home=str(account_home))])
    ready_state(tmp_path, account="work")
    fake = FakeCLI()
    homes = []
    monkeypatch.setattr(model.ColabCLI, "discover", classmethod(lambda cls, home=None: homes.append(home) or fake))
    model.switch_model("example/new-model-GGUF", "Q4_K_M", timeout=30)
    assert homes == [str(account_home)]


def test_successful_switch_updates_state(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    ready_state(tmp_path)
    fake = FakeCLI()
    monkeypatch.setattr(model.ColabCLI, "discover", classmethod(lambda cls, home=None: fake))
    result = model.switch_model("example/new-model-GGUF", "Q4_K_M", timeout=30)
    state = load_state()
    assert result["model_repo"] == "example/new-model-GGUF"
    assert state["runtime_state"] == "ready"
    assert state["model"] == "/content/models/new/model.Q4_K_M.gguf"
    assert state["model_repo"] == "example/new-model-GGUF"
    assert state["quant"] == "Q4_K_M"
    assert state["tests"]["cuda_offload"] is True
    assert any(call[1] == "upload" and call[-1] == model.REMOTE_SWITCH_CONFIG for call in fake.calls)
    assert any(call[1] == "exec" for call in fake.calls)


def test_failed_switch_with_rollback_preserves_previous_model(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    ready_state(tmp_path)
    fake = FakeCLI(
        result={"success": False, "rollback_attempted": True, "rollback_success": True, "error": "new model failed"},
        ready={
            "ready": True,
            "model": "/content/model/old.Q4_K_M.gguf",
            "model_repo": "example/old-GGUF",
            "quant": "Q4_K_M",
            "tests": {"models": True, "chat": True, "cuda_offload": True},
        },
        exec_returncode=1,
    )
    monkeypatch.setattr(model.ColabCLI, "discover", classmethod(lambda cls, home=None: fake))
    with pytest.raises(RuntimeError, match="new model failed"):
        model.switch_model("example/broken-GGUF", "Q4_K_M", timeout=30)
    state = load_state()
    assert state["runtime_state"] == "ready"
    assert state["model_repo"] == "example/old-GGUF"
    assert state["model"] == "/content/model/old.Q4_K_M.gguf"


def test_failed_switch_and_failed_rollback_marks_runtime_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    ready_state(tmp_path)
    fake = FakeCLI(
        result={"success": False, "rollback_attempted": True, "rollback_success": False, "error": "switch and rollback failed"},
        exec_returncode=1,
    )
    monkeypatch.setattr(model.ColabCLI, "discover", classmethod(lambda cls, home=None: fake))
    with pytest.raises(RuntimeError, match="switch and rollback failed"):
        model.switch_model("example/broken-GGUF", "Q4_K_M", timeout=30)
    state = load_state()
    assert state["runtime_state"] == "failed"
    assert "switch and rollback failed" in state["last_error"]


def test_generated_switch_script_has_required_guards_and_no_literal_secrets():
    script = model.build_switch_script()
    lower = script.lower()
    assert "hf" in lower and "download" in lower
    assert "chat/completions" in script
    assert "cuda" in lower
    assert "rollback" in lower
    assert "os.replace" in script
    assert "api-secret-test" not in script
    assert "hf-secret-test" not in script
