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


_SECRET_FIELDS = {"tailscale_authkey", "hf_token", "api_key", "ssh_password", "ssh_pubkey"}
_OPTIONAL_FIELDS = {"tailnet", "hf_token"}
_NON_SECRET_FIELDS = {"session", "model", "model_repo", "quant", "port", "ctx", "tailnet", "ssh_mode"}
_CONFIG_FIELDS = _SECRET_FIELDS | _NON_SECRET_FIELDS
_MODEL_STORAGE_KEY = "model_repo"
_SSH_MODES = {"tailscale", "password", "key"}


def _runtime_defaults() -> dict[str, Any]:
    from .notebook import DEFAULT_CTX, DEFAULT_HOSTNAME, DEFAULT_MODEL, DEFAULT_PORT, DEFAULT_QUANT

    return {
        "session": DEFAULT_HOSTNAME,
        "model": DEFAULT_MODEL,
        "quant": DEFAULT_QUANT,
        "port": DEFAULT_PORT,
        "ctx": DEFAULT_CTX,
        "tailnet": "",
        "ssh_mode": "tailscale",
    }


def _coerce_positive_int(name: str, value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return str(number)


def _coerce_text(name: str, value: Any, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise ValueError(f"{name} is required")
    return text


def _summary_int(saved: dict[str, str], key: str, default: int) -> int:
    try:
        return int(saved.get(key, default))
    except (TypeError, ValueError):
        return default


def configuration_summary() -> dict[str, Any]:
    """Return saved configuration without exposing secret values."""
    saved = load_secrets()
    defaults = _runtime_defaults()
    return {
        "session": saved.get("session") or defaults["session"],
        "model": saved.get("model_repo") or saved.get("model") or defaults["model"],
        "quant": saved.get("quant") or defaults["quant"],
        "port": _summary_int(saved, "port", defaults["port"]),
        "ctx": _summary_int(saved, "ctx", defaults["ctx"]),
        "tailnet": saved.get("tailnet", defaults["tailnet"]),
        "ssh_mode": saved.get("ssh_mode") or defaults["ssh_mode"],
        "secrets": {
            "tailscale_authkey": bool(saved.get("tailscale_authkey")),
            "hf_token": bool(saved.get("hf_token")),
            "api_key": bool(saved.get("api_key")),
            "ssh_password": bool(saved.get("ssh_password")),
            "ssh_pubkey": bool(saved.get("ssh_pubkey")),
        },
    }


def update_configuration(values: dict[str, Any]) -> dict[str, Any]:
    """Persist allowlisted configuration values and return a redacted summary."""
    if not isinstance(values, dict):
        raise ValueError("configuration payload must be an object")
    unknown = sorted(set(values) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError("unknown configuration field: " + unknown[0])

    saved = load_secrets()
    updated: dict[str, str] = dict(saved)
    for key, raw_value in values.items():
        storage_key = _MODEL_STORAGE_KEY if key == "model" else key
        if key in {"port", "ctx"}:
            updated[storage_key] = _coerce_positive_int(key, raw_value)
            continue
        if key == "ssh_mode":
            mode = _coerce_text(key, raw_value)
            if mode not in _SSH_MODES:
                raise ValueError("ssh_mode must be tailscale, password, or key")
            updated[storage_key] = mode
            continue
        if key in _SECRET_FIELDS:
            updated[storage_key] = _coerce_text(key, raw_value, required=key not in _OPTIONAL_FIELDS)
            continue
        updated[storage_key] = _coerce_text(key, raw_value, required=key not in _OPTIONAL_FIELDS)
    save_secrets(updated)
    return configuration_summary()


def reset_configuration(confirmed: bool) -> None:
    """Clear saved configuration only after an explicit confirmation flag."""
    if confirmed is not True:
        raise ValueError("confirmation is required")
    clear_secrets()
