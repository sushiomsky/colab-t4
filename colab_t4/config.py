"""Persistent non-secret state and separately protected runtime secrets."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

PROJECT = Path("/root/colab-t4")
DEFAULT_STATE_DIR = Path(os.environ.get("COLAB_T4_STATE_DIR", Path.home() / ".config" / "colab-t4"))
_SELECTED_RUNTIME: ContextVar[str] = ContextVar("colab_t4_runtime", default="default")
_RUNTIME_DEFAULT_KEYS = {"api_key", "model_repo", "quant", "session", "port", "ctx"}


def global_state_dir() -> Path:
    path = Path(os.environ.get("COLAB_T4_STATE_DIR", DEFAULT_STATE_DIR))
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def selected_runtime() -> str:
    return _SELECTED_RUNTIME.get()


@contextmanager
def select_runtime(runtime_id: str) -> Iterator[None]:
    token = _SELECTED_RUNTIME.set(runtime_id)
    try:
        yield
    finally:
        _SELECTED_RUNTIME.reset(token)


def state_dir() -> Path:
    base = global_state_dir()
    runtime_id = selected_runtime()
    path = base if runtime_id == "default" else base / "runtimes" / runtime_id
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def logs_dir() -> Path:
    path = state_dir() / "logs"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def state_path() -> Path:
    return state_dir() / "state.json"


def secrets_path() -> Path:
    return global_state_dir() / "secrets.json"


def runtime_api_key_path() -> Path:
    return state_dir() / "runtime-api-key.json"


def _atomic_json(path: Path, data: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
        os.chmod(path, mode)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def load_state() -> dict[str, Any]:
    try:
        with state_path().open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_state(data: dict[str, Any]) -> None:
    _atomic_json(state_path(), data)


def clear_state() -> None:
    try:
        state_path().unlink()
    except FileNotFoundError:
        pass


def load_saved_secrets() -> dict[str, str]:
    try:
        with secrets_path().open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            return {str(k): str(v) for k, v in loaded.items()}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


def load_runtime_api_key() -> str:
    try:
        with runtime_api_key_path().open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and loaded.get("api_key"):
            return str(loaded["api_key"])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return ""


def load_secrets() -> dict[str, str]:
    saved = load_saved_secrets()
    runtime_id = selected_runtime()
    if runtime_id != "default":
        saved = {key: value for key, value in saved.items() if key not in _RUNTIME_DEFAULT_KEYS}
    runtime_key = load_runtime_api_key()
    if runtime_key and (runtime_id != "default" or "api_key" not in saved):
        saved["api_key"] = runtime_key
    return saved


def save_secrets(data: dict[str, str]) -> None:
    _atomic_json(secrets_path(), data)


def update_saved_secrets(values: dict[str, str]) -> None:
    saved = load_saved_secrets()
    saved.update(values)
    _atomic_json(secrets_path(), saved)


def save_runtime_api_key(api_key: str) -> None:
    _atomic_json(runtime_api_key_path(), {"api_key": api_key})


def clear_runtime_api_key() -> None:
    try:
        runtime_api_key_path().unlink()
    except FileNotFoundError:
        pass


def clear_secrets() -> None:
    for path in (secrets_path(), runtime_api_key_path()):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def generated_api_key() -> str:
    return secrets.token_urlsafe(32)


def generated_password() -> str:
    return secrets.token_urlsafe(18)


def redact(text: str, secret_values: list[str] | None = None) -> str:
    values = [v for v in (secret_values or []) if v]
    values += [
        os.environ.get("TS_AUTHKEY", ""),
        os.environ.get("HF_TOKEN", ""),
        os.environ.get("COLAB_T4_API_KEY", ""),
        os.environ.get("COLAB_T4_SSH_PASSWORD", ""),
    ]
    for value in sorted({v for v in values if len(v) >= 4}, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return text
