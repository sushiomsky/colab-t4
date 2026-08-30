"""In-place model switching for an already provisioned Colab T4 runtime."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .accounts import get_account
from .backend import ColabCLI, ColabCLIError
from .config import load_secrets, load_state, logs_dir, redact, save_state
from .notebook import DEFAULT_CTX, DEFAULT_PORT, DEFAULT_QUANT

REMOTE_SWITCH_CONFIG = "/content/.colab-t4-model-switch.json"
REMOTE_SWITCH_RESULT = "/content/.colab-t4-model-switch-result.json"
REMOTE_READY = "/content/.colab-t4-ready.json"


SWITCH_PROVISIONING = r'''# colab-t4 in-place model switch
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

CONFIG_FILE = Path("/content/.colab-t4-model-switch.json")
RESULT_FILE = Path("/content/.colab-t4-model-switch-result.json")
READY_FILE = Path("/content/.colab-t4-ready.json")
LOG_DIR = Path("/content/colab-t4/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def write_result(**values):
    atomic_json(RESULT_FILE, values)


def run(args, *, timeout=1800, check=True, log_name=None, env=None):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, env=env)
    if log_name:
        (LOG_DIR / log_name).write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if check and result.returncode:
        raise RuntimeError("command failed: " + str(args[0]))
    return result


def request_json(url, api_key, payload=None, timeout=30):
    headers = {"Authorization": "Bearer " + api_key}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_port_free(port, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", port))
            probe.close()
            return
        except OSError:
            probe.close()
            time.sleep(0.5)
    raise RuntimeError("API port did not become free")


def stop_server(port):
    subprocess.run(["pkill", "-TERM", "-f", "llama_cpp.server"], capture_output=True)
    wait_port_free(port)


def start_server(model_path, *, api_key, port, ctx, log_name):
    log_path = LOG_DIR / log_name
    log_handle = open(log_path, "w")
    process = subprocess.Popen([
        "python3", "-m", "llama_cpp.server",
        "--model", str(model_path),
        "--model_alias", "local",
        "--n_gpu_layers", "-1",
        "--n_ctx", str(ctx),
        "--flash_attn", "true",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--api_key", api_key,
    ], stdout=log_handle, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    return process, log_path


def verify_server(process, log_path, *, api_key, port):
    models_url = "http://127.0.0.1:" + str(port) + "/v1/models"
    deadline = time.monotonic() + 480
    models = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama server exited before becoming healthy")
        try:
            models = request_json(models_url, api_key)
            if models.get("data"):
                break
        except Exception:
            pass
        time.sleep(2)
    if not models or not models.get("data"):
        raise RuntimeError("new llama server did not become healthy")
    chat = request_json(
        "http://127.0.0.1:" + str(port) + "/v1/chat/completions",
        api_key,
        payload={"model": "local", "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 8},
        timeout=120,
    )
    if not chat.get("choices"):
        raise RuntimeError("chat smoke test failed")
    cuda_offload = False
    for _ in range(30):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            cuda_offload = bool(re.search(r"offload|layers.*GPU|CUDA", text, re.IGNORECASE))
        except OSError:
            pass
        if cuda_offload:
            break
        time.sleep(1)
    if not cuda_offload:
        raise RuntimeError("server log did not prove CUDA offload")
    return {"models": True, "chat": True, "cuda_offload": True}


if not CONFIG_FILE.exists():
    raise RuntimeError("model switch configuration is missing")
config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
try:
    CONFIG_FILE.unlink()
except FileNotFoundError:
    pass
required = ["model_repo", "quant", "api_key", "port", "ctx"]
if any(not config.get(key) for key in required):
    raise RuntimeError("model switch configuration is incomplete")
if not READY_FILE.exists():
    raise RuntimeError("runtime readiness artifact is missing")
old_ready = json.loads(READY_FILE.read_text(encoding="utf-8"))
old_model = old_ready.get("model")
if not old_model:
    raise RuntimeError("current model path is missing from readiness artifact")

api_key = config["api_key"]
port = int(config["port"])
ctx = int(config["ctx"])
repo = config["model_repo"]
quant = config["quant"]
if config.get("hf_token"):
    os.environ["HF_TOKEN"] = config["hf_token"]

# Download and validate before touching the healthy old server.
safe_repo = re.sub(r"[^A-Za-z0-9._-]+", "_", repo)
target_dir = Path("/content/models") / safe_repo / quant
target_dir.mkdir(parents=True, exist_ok=True)
pattern = "*" + quant + "*.gguf"
download = run([
    "hf", "download", repo, "--include", pattern, "--local-dir", str(target_dir)
], timeout=3600, check=False, log_name="model-switch-download.log")
models = sorted(path for path in target_dir.rglob("*.gguf") if quant.lower() in path.name.lower())
if download.returncode or len(models) != 1:
    write_result(success=False, rollback_attempted=False, rollback_success=False,
                 error="requested quant did not resolve to exactly one GGUF file")
    raise RuntimeError("requested quant did not resolve to exactly one GGUF file")
new_model = models[0]
if new_model.stat().st_size > 11 * 1024**3:
    write_result(success=False, rollback_attempted=False, rollback_success=False,
                 error="selected GGUF is too large for a 16 GiB T4")
    raise RuntimeError("selected GGUF is too large for a 16 GiB T4")

rollback_attempted = False
rollback_success = False
try:
    stop_server(port)
    process, server_log = start_server(new_model, api_key=api_key, port=port, ctx=ctx, log_name="llama-server.log")
    tests = verify_server(process, server_log, api_key=api_key, port=port)
    ready = dict(old_ready)
    ready.update({
        "ready": True,
        "model": str(new_model),
        "model_repo": repo,
        "quant": quant,
        "server_pid": process.pid,
        "tests": {**(old_ready.get("tests") or {}), **tests},
    })
    atomic_json(READY_FILE, ready)
    write_result(success=True, rollback_attempted=False, rollback_success=False,
                 model=str(new_model), model_repo=repo, quant=quant)
except Exception as exc:
    rollback_attempted = True
    try:
        stop_server(port)
        old_process, old_log = start_server(old_model, api_key=api_key, port=port, ctx=ctx, log_name="llama-server.log")
        old_tests = verify_server(old_process, old_log, api_key=api_key, port=port)
        restored = dict(old_ready)
        restored["server_pid"] = old_process.pid
        restored["tests"] = {**(old_ready.get("tests") or {}), **old_tests}
        atomic_json(READY_FILE, restored)
        rollback_success = True
    except Exception:
        rollback_success = False
    write_result(success=False, rollback_attempted=rollback_attempted,
                 rollback_success=rollback_success, error=str(exc))
    raise
'''


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def current_model() -> dict[str, object]:
    """Return non-secret model metadata from the recorded local state."""
    state = load_state()
    return {
        "session": state.get("session"),
        "account": state.get("account"),
        "runtime_state": state.get("runtime_state", "unknown"),
        "model": state.get("model"),
        "model_repo": state.get("model_repo"),
        "quant": state.get("quant"),
    }


def build_switch_script() -> str:
    """Return the non-secret remote switch source for tests/auditing."""
    return SWITCH_PROVISIONING


def _build_switch_notebook() -> str:
    return json.dumps({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [SWITCH_PROVISIONING],
        }],
    }, indent=1) + "\n"


def _account_home(state: dict[str, Any]) -> str | None:
    account_id = state.get("account")
    if not account_id:
        return None
    try:
        return get_account(str(account_id)).home
    except KeyError:
        return None


def _write_temp_json(data: dict[str, Any], *, prefix: str) -> Path:
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".json")
    path = Path(name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
    except Exception:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    return path


def _write_temp_notebook() -> Path:
    fd, name = tempfile.mkstemp(prefix="colab-t4-model-switch-", suffix=".ipynb")
    os.close(fd)
    path = Path(name)
    path.write_text(_build_switch_notebook(), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def _download_json(cli: ColabCLI, session: str, remote: str, local_name: str, secrets: list[str]) -> dict[str, Any] | None:
    target = logs_dir() / local_name
    target.unlink(missing_ok=True)
    result = cli.run(
        cli.download_command(session, remote, target),
        logs_dir() / "model-switch.log",
        timeout=60,
        secrets=secrets,
    )
    if result.returncode or not target.exists():
        return None
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _apply_ready(state: dict[str, Any], ready: dict[str, Any], *, last_error: str | None = None) -> dict[str, Any]:
    state.update({
        "runtime_state": "ready",
        "model": ready.get("model", state.get("model")),
        "model_repo": ready.get("model_repo", state.get("model_repo")),
        "quant": ready.get("quant", state.get("quant")),
        "tests": ready.get("tests", state.get("tests", {})),
        "last_error": last_error,
        "updated_at": _now(),
    })
    save_state(state)
    return state


def switch_model(model_repo: str, quant: str = DEFAULT_QUANT, timeout: float = 3600) -> dict[str, object]:
    """Switch the model in the recorded ready session without recreating it."""
    state = load_state()
    session = state.get("session")
    if not session or state.get("runtime_state") != "ready":
        raise RuntimeError("a ready recorded Colab runtime is required for model switching")
    if not model_repo.strip():
        raise ValueError("model repository must not be empty")
    if not quant.strip():
        raise ValueError("quantization must not be empty")

    secrets = load_secrets()
    api_key = secrets.get("api_key", "")
    if not api_key:
        raise RuntimeError("API credentials are required for model switching")
    config = {
        "model_repo": model_repo.strip(),
        "quant": quant.strip(),
        "api_key": api_key,
        "hf_token": secrets.get("hf_token", ""),
        "port": str(secrets.get("port") or _port_from_state(state) or DEFAULT_PORT),
        "ctx": str(secrets.get("ctx") or DEFAULT_CTX),
    }
    secret_values = [value for value in config.values() if isinstance(value, str) and value]
    cli = ColabCLI.discover(home=_account_home(state))
    config_file = _write_temp_json(config, prefix="colab-t4-model-switch-config-")
    notebook = _write_temp_notebook()
    try:
        uploaded = cli.run(
            cli.upload_command(str(session), config_file, REMOTE_SWITCH_CONFIG),
            logs_dir() / "model-switch.log",
            timeout=120,
            secrets=secret_values,
        )
        if uploaded.returncode:
            raise ColabCLIError("failed to upload model switch configuration")
        executed = cli.run(
            cli.exec_command(str(session), notebook, float(timeout)),
            logs_dir() / "model-switch.log",
            timeout=float(timeout) + 120,
            secrets=secret_values,
        )
        result = _download_json(cli, str(session), REMOTE_SWITCH_RESULT, "model-switch-result.json", secret_values)
        ready = _download_json(cli, str(session), REMOTE_READY, "runtime-ready.json", secret_values)
        if result and result.get("success") and ready and ready.get("ready"):
            updated = _apply_ready(state, ready)
            return {
                **current_model(),
                "tests": updated.get("tests", {}),
            }

        error = "model switch failed"
        if result and result.get("error"):
            error = redact(str(result["error"]), secret_values)
        elif executed.returncode:
            error = "remote model switch execution failed"

        old_server_preserved = bool(result and not result.get("rollback_attempted"))
        rollback_success = bool(result and result.get("rollback_success"))
        if (old_server_preserved or rollback_success) and ready and ready.get("ready"):
            _apply_ready(state, ready, last_error="model switch failed: " + error)
        elif old_server_preserved or rollback_success:
            state["runtime_state"] = "ready"
            state["last_error"] = "model switch failed: " + error
            state["updated_at"] = _now()
            save_state(state)
        else:
            state["runtime_state"] = "failed"
            state["last_error"] = "model switch failed: " + error
            state["updated_at"] = _now()
            save_state(state)
        raise RuntimeError(error)
    finally:
        config_file.unlink(missing_ok=True)
        notebook.unlink(missing_ok=True)


def _port_from_state(state: dict[str, Any]) -> int | None:
    base = str(state.get("api_base") or "")
    if not base:
        return None
    try:
        authority = base.split("//", 1)[1].split("/", 1)[0]
        return int(authority.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None
