import json
import subprocess
from pathlib import Path

import pytest

from colab_t4.backend import ColabCLI
from colab_t4.config import redact
from colab_t4.notebook import build_notebook


def test_cli_command_construction():
    cli = ColabCLI("colab", "0.6.0")
    assert cli.new_command("colab-t4", "T4") == ["colab", "new", "--session", "colab-t4", "--gpu", "T4"]
    assert cli.upload_command("colab-t4", Path("x.ipynb"), "/content/x.ipynb") == [
        "colab", "upload", "--session", "colab-t4", "x.ipynb", "/content/x.ipynb"
    ]
    assert cli.exec_command("colab-t4", Path("x.ipynb"), 12) == [
        "colab", "exec", "--session", "colab-t4", "--file", "x.ipynb", "--timeout", "12"
    ]
    assert cli.stop_command("colab-t4") == ["colab", "stop", "--session", "colab-t4"]


def test_notebook_is_valid_and_deterministic():
    first = json.loads(build_notebook())
    second = json.loads(build_notebook())
    assert first == second
    assert first["metadata"]["colab"]["gpuType"] == "T4"
    assert first["metadata"]["accelerator"] == "GPU"
    for cell in first["cells"]:
        source = "".join(cell["source"])
        assert "{{" not in source
        if cell["cell_type"] == "code":
            compile(source, "<cell>", "exec")
    source = "".join(first["cells"][1]["source"])
    assert "SECRET_FILE.unlink()" in source
    assert "LLAMA_CPP_RUNTIME" in source
    assert "llama-server" in source


def test_redaction():
    assert redact("prefix secret-token suffix", ["secret-token"]) == "prefix [REDACTED] suffix"


def test_tool_call_capability_classification():
    from colab_t4 import cli
    assert cli._classify_tool_response({"choices": [{"message": {"tool_calls": [{"id": "x"}]}}]}) == (True, "native")
    assert cli._classify_tool_response({"choices": [{"message": {"content": "<tool_call><function=x></function></tool_call>"}}]}) == (True, "qwen_xml_compat")
    assert cli._classify_tool_response({"choices": [{"message": {"content": "ordinary answer"}}]}) == (False, "none")


def test_discover_colab_cli(monkeypatch):
    class Result:
        returncode = 0
        stdout = "Version: 0.6.0\n"
        stderr = ""

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/colab")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Result())
    found = ColabCLI.discover()
    assert found.executable == "/usr/bin/colab"
    assert found.version == "0.6.0"


def test_ssh_forwarding(monkeypatch, tmp_path):
    import colab_t4.cli as cli
    monkeypatch.setattr(cli, "load_state", lambda: {"session": "colab-t4", "tailscale_ip": "100.64.0.2"})
    calls = []
    monkeypatch.setattr(cli.os, "execvp", lambda program, args: calls.append((program, args)))
    assert cli.cmd_ssh(["colab-t4", "--", "-L", "8080:127.0.0.1:8080"]) == 0
    assert calls[0][1] == ["ssh", "-o", "StrictHostKeyChecking=accept-new", "root@100.64.0.2", "-L", "8080:127.0.0.1:8080"]


def test_status_unknown_is_nonzero(tmp_path, monkeypatch):
    import colab_t4.cli as cli
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path))
    result, code = cli._status()
    assert result["runtime_state"] == "unknown"
    assert code != 0


def test_api_request_uses_v1_base_and_health_root(monkeypatch):
    import colab_t4.cli as cli

    monkeypatch.setattr(cli, "load_state", lambda: {"api_base": "http://100.64.0.2:8080/v1"})
    monkeypatch.setattr(cli, "load_secrets", lambda: {"api_key": "api-secret"})
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b'{"ok": true}'

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda request, timeout: requests.append((request, timeout)) or Response())
    assert cli._api_request("models") == {"ok": True}
    assert cli._api_request("health") == {"ok": True}
    assert requests[0][0].full_url == "http://100.64.0.2:8080/v1/models"
    assert requests[1][0].full_url == "http://100.64.0.2:8080/health"
    assert requests[0][0].headers["Authorization"] == "Bearer api-secret"
