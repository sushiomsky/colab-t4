"""Hermes-friendly stdio MCP adapter for :mod:`colab_t4`.

The module deliberately keeps the MCP SDK optional. Pure tool functions remain
importable and testable in the base Python 3.9 installation; constructing or
running the MCP server requires Python 3.10+ and the ``mcp`` extra.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Callable, TypeVar

from .accounts import load_accounts
from .config import load_secrets, load_state, redact, save_runtime_api_key
from .lifecycle import down as lifecycle_down
from .lifecycle import restart as lifecycle_restart
from .lifecycle import up as lifecycle_up
from .model import current_model as current_model_runtime
from .model import switch_model as switch_model_runtime
from .notebook import DEFAULT_QUANT
from .wizard import collect

try:  # MCP SDK v2 (current stable line).
    from mcp.server import MCPServer as _MCPServer  # type: ignore
except (ImportError, SyntaxError):  # pragma: no cover - exercised by MCP job / legacy installs.
    try:  # MCP SDK v1 compatibility for already-pinned environments.
        from mcp.server.fastmcp import FastMCP as _MCPServer  # type: ignore
    except (ImportError, SyntaxError):  # Base install intentionally has no MCP dependency.
        _MCPServer = None  # type: ignore

T = TypeVar("T")


def _safe_error(exc: BaseException) -> str:
    """Return an error suitable for an MCP response without leaking secrets."""
    secrets = [value for value in load_secrets().values() if isinstance(value, str)]
    return redact(str(exc), secrets)


def _safe_call(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        raise RuntimeError(_safe_error(exc)) from exc


def _safe_state(state: dict[str, Any]) -> dict[str, Any]:
    """Project runtime state onto an explicit non-secret MCP schema."""
    return {
        "runtime_state": state.get("runtime_state", "unknown"),
        "session": state.get("session"),
        "account": state.get("account"),
        "gpu": state.get("gpu"),
        "accelerator": state.get("accelerator"),
        "tailscale_ip": state.get("tailscale_ip"),
        "api_base": state.get("api_base"),
        "model": state.get("model"),
        "model_repo": state.get("model_repo"),
        "quant": state.get("quant"),
        "tests": state.get("tests") or {},
        "last_error": redact(str(state.get("last_error") or ""), list(load_secrets().values())) or None,
    }


def _runtime_options(model: str | None = None, quant: str | None = None) -> SimpleNamespace:
    """Resolve saved/env lifecycle configuration without interactive prompts."""
    options = SimpleNamespace(
        session=None,
        model=model,
        quant=quant,
        port=None,
        ctx=None,
        api_key=None,
        ssh_mode=None,
        password=None,
        pubkey=None,
        exec_timeout=None,
        interactive=False,
        non_interactive=True,
        yes=True,
    )
    values = collect(options, force=False, allow_prompt=False)
    for key, value in values.items():
        setattr(options, key, value)
    options.runtime_config = values
    if values.get("api_key"):
        save_runtime_api_key(str(values["api_key"]))
    return options


def backend_status() -> dict[str, Any]:
    """Return the recorded backend state using a fixed non-secret schema."""
    return _safe_state(load_state())


def backend_start(model: str | None = None, quant: str | None = None) -> dict[str, Any]:
    """Start or recover the Colab backend using saved/environment configuration."""
    options = _safe_call(_runtime_options, model, quant)
    return _safe_state(_safe_call(lifecycle_up, options))


def backend_stop() -> dict[str, Any]:
    """Stop the recorded Colab runtime and clear local runtime state."""
    _safe_call(lifecycle_down)
    return {"runtime_state": "stopped"}


def backend_restart(model: str | None = None, quant: str | None = None) -> dict[str, Any]:
    """Recreate the backend, optionally selecting the model for the new runtime."""
    options = _safe_call(_runtime_options, model, quant)
    return _safe_state(_safe_call(lifecycle_restart, options))


def model_current() -> dict[str, object]:
    """Return current non-secret model metadata."""
    return _safe_call(current_model_runtime)


def model_switch(model: str, quant: str = DEFAULT_QUANT) -> dict[str, object]:
    """Switch the running runtime to another GGUF model in place."""
    return _safe_call(switch_model_runtime, model, quant, 3600)


def api_info() -> dict[str, Any]:
    """Return connection information required by an OpenAI-compatible client."""
    state = load_state()
    return {
        "runtime_state": state.get("runtime_state", "unknown"),
        "api_base": state.get("api_base"),
        "model_alias": "local",
    }


def accounts_list() -> list[dict[str, Any]]:
    """Return account rotation metadata without credential or filesystem paths."""
    result: list[dict[str, Any]] = []
    for account in load_accounts():
        if account.last_error:
            status = "error"
        elif account.last_ok:
            status = "ok"
        else:
            status = "unused"
        result.append({
            "id": account.id,
            "email": account.email or "unknown",
            "status": status,
            "last_used_at": account.last_used_at or "",
            "last_ok": account.last_ok or "",
        })
    return result


_TOOLS = (
    backend_status,
    backend_start,
    backend_stop,
    backend_restart,
    model_current,
    model_switch,
    api_info,
    accounts_list,
)


def build_server():
    """Build an MCP SDK server and register the public colab-t4 tools."""
    if _MCPServer is None:
        raise RuntimeError(
            "MCP support is not installed; use Python 3.10+ and install "
            "`colab-t4[mcp]`"
        )
    server = _MCPServer("colab-t4")
    for function in _TOOLS:
        server.tool()(function)
    return server


def main() -> int:
    """Run the MCP server over stdio (the SDK default transport)."""
    if sys.version_info < (3, 10) or _MCPServer is None:
        print(
            "colab-t4 MCP requires Python 3.10+ and the optional MCP SDK; "
            "install with: pip install 'colab-t4[mcp]'",
            file=sys.stderr,
        )
        return 2
    try:
        build_server().run()
    except Exception as exc:
        print("colab-t4 MCP error: " + _safe_error(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
