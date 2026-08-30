"""Named runtime namespaces layered over the legacy default runtime."""
from __future__ import annotations

import json
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import config
from .notebook import DEFAULT_HOSTNAME, DEFAULT_MODEL, DEFAULT_QUANT

_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_RUNTIME_METADATA_KEYS = {"session", "model_repo", "quant"}


def selected_runtime() -> str:
    return config.selected_runtime()


def validate_runtime_id(runtime_id: str) -> str:
    if runtime_id == "default" or not _RUNTIME_ID_RE.fullmatch(runtime_id):
        raise ValueError(f"invalid runtime id '{runtime_id}'")
    return runtime_id


def _runtime_root(runtime_id: str) -> Path:
    return config.global_state_dir() / "runtimes" / runtime_id


def _metadata_path(runtime_id: str) -> Path:
    return _runtime_root(runtime_id) / "runtime.json"


@contextmanager
def runtime_context(runtime_id: str, *, require_exists: bool = True) -> Iterator[None]:
    if runtime_id == "default":
        with config.select_runtime("default"):
            yield
        return
    validate_runtime_id(runtime_id)
    if require_exists and not _metadata_path(runtime_id).is_file():
        raise RuntimeError(
            f"runtime '{runtime_id}' does not exist; create it with `colab-t4 runtimes create {runtime_id}`"
        )
    with config.select_runtime(runtime_id):
        yield


def load_runtime_metadata(runtime_id: str) -> dict[str, object]:
    if runtime_id == "default":
        saved = config.load_saved_secrets()
        return {
            "id": "default",
            "session": saved.get("session", DEFAULT_HOSTNAME),
            "model_repo": saved.get("model_repo", DEFAULT_MODEL),
            "quant": saved.get("quant", DEFAULT_QUANT),
        }
    validate_runtime_id(runtime_id)
    try:
        raw = json.loads(_metadata_path(runtime_id).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"runtime '{runtime_id}' does not exist; create it with `colab-t4 runtimes create {runtime_id}`"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"runtime '{runtime_id}' metadata is invalid") from exc
    if not isinstance(raw, dict) or raw.get("id") != runtime_id:
        raise RuntimeError(f"runtime '{runtime_id}' metadata is invalid")
    result = {"id": runtime_id}
    for key in _RUNTIME_METADATA_KEYS:
        value = raw.get(key)
        if value is not None:
            result[key] = value
    return result


def update_runtime_metadata(runtime_id: str, values: dict[str, object]) -> dict[str, object]:
    if runtime_id == "default":
        raise ValueError("default runtime metadata is stored by the legacy configuration flow")
    current = load_runtime_metadata(runtime_id)
    for key, value in values.items():
        if key not in _RUNTIME_METADATA_KEYS:
            raise ValueError(f"unsupported runtime metadata field '{key}'")
        current[key] = value
    config._atomic_json(_metadata_path(runtime_id), current)
    return current


def create_runtime(runtime_id: str, *, model_repo: str, quant: str) -> dict[str, object]:
    validate_runtime_id(runtime_id)
    root = _runtime_root(runtime_id)
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"runtime '{runtime_id}' already exists") from exc
    metadata: dict[str, object] = {
        "id": runtime_id,
        "session": f"colab-t4-{runtime_id}",
        "model_repo": model_repo,
        "quant": quant,
    }
    try:
        config._atomic_json(root / "runtime.json", metadata)
        with runtime_context(runtime_id):
            config.save_runtime_api_key(config.generated_api_key())
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return metadata


def _read_json_dict(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (value, True) if isinstance(value, dict) else ({}, False)
    except FileNotFoundError:
        return {}, True
    except (OSError, json.JSONDecodeError):
        return {}, False


def _inventory_row(runtime_id: str) -> dict[str, object]:
    if runtime_id == "default":
        root = config.global_state_dir()
        metadata = load_runtime_metadata("default")
        state, state_valid = _read_json_dict(root / "state.json")
        saved = config.load_saved_secrets()
        api_key_configured = bool(saved.get("api_key")) or (root / "runtime-api-key.json").is_file()
        metadata_valid = True
    else:
        root = _runtime_root(runtime_id)
        metadata_raw, metadata_valid = _read_json_dict(root / "runtime.json")
        metadata = metadata_raw if metadata_valid and metadata_raw.get("id") == runtime_id else {}
        metadata_valid = bool(metadata)
        state, state_valid = _read_json_dict(root / "state.json")
        api_key_configured = (root / "runtime-api-key.json").is_file()

    valid = metadata_valid and state_valid
    runtime_state = state.get("runtime_state", "unknown") if valid else "invalid"
    return {
        "id": runtime_id,
        "runtime_state": runtime_state,
        "session": state.get("session") or metadata.get("session"),
        "account": state.get("account"),
        "model": state.get("model"),
        "model_repo": state.get("model_repo") or metadata.get("model_repo"),
        "quant": state.get("quant") or metadata.get("quant"),
        "api_base": state.get("api_base"),
        "api_key_configured": api_key_configured,
    }


def runtime_inventory() -> list[dict[str, object]]:
    rows = [_inventory_row("default")]
    root = config.global_state_dir() / "runtimes"
    if not root.is_dir():
        return rows
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        try:
            validate_runtime_id(path.name)
        except ValueError:
            continue
        rows.append(_inventory_row(path.name))
    return rows
