"""Persistent non-secret state and separately protected runtime secrets."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

PROJECT = Path("/root/colab-t4")
DEFAULT_STATE_DIR = Path(os.environ.get("COLAB_T4_STATE_DIR", Path.home() / ".config" / "colab-t4"))


def state_dir() -> Path:
    path = Path(os.environ.get("COLAB_T4_STATE_DIR", DEFAULT_STATE_DIR))
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
    return state_dir() / "secrets.json"


def runtime_api_key_path() -> Path:
    return state_dir() / "runtime-api-key.json"


def _atomic_json(path: Path, data: dict[str, Any], mode: int = 0o600) -> None:
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


def clear_secrets() -> None:
    for path in (secrets_path(), runtime_api_key_path()):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def load_secrets() -> dict[str, str]:
    value: dict[str, str] = {}
    try:
        with secrets_path().open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            value.update({str(k): str(v) for k, v in loaded.items()})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    if "api_key" not in value:
        try:
            with runtime_api_key_path().open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict) and loaded.get("api_key"):
                value["api_key"] = str(loaded["api_key"])
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
    return value


def save_secrets(data: dict[str, str]) -> None:
    _atomic_json(secrets_path(), data)


def save_runtime_api_key(api_key: str) -> None:
    _atomic_json(runtime_api_key_path(), {"api_key": api_key})


def clear_runtime_api_key() -> None:
    try:
        runtime_api_key_path().unlink()
    except FileNotFoundError:
        pass


def save_secrets(data: dict[str, str]) -> None:
    _atomic_json(secrets_path(), data)


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
