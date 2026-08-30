"""Structured operations shared by command-line and management clients."""
from __future__ import annotations

import sys
from typing import Any, Callable


ProgressCallback = Callable[[str, int | None], None]
CancellationCallback = Callable[[], bool]


def _cli():
    """Import CLI helpers lazily to avoid an import cycle with CLI handlers."""
    from . import cli

    return cli


def _check_cancelled(cancelled: CancellationCallback | None) -> None:
    if cancelled and cancelled():
        raise InterruptedError("operation cancelled")


def runtime_up(
    options: Any,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> dict[str, Any]:
    _check_cancelled(cancelled)
    if progress:
        progress("starting runtime", None)
    if cancelled:
        result = _cli().lifecycle_up(options, cancelled=cancelled)
    else:
        result = _cli().lifecycle_up(options)
    _check_cancelled(cancelled)
    if progress:
        progress("runtime ready", 100)
    return result


def runtime_down(cancelled: CancellationCallback | None = None) -> dict[str, Any]:
    _check_cancelled(cancelled)
    if cancelled:
        _cli().lifecycle_down(cancelled=cancelled)
    else:
        _cli().lifecycle_down()
    _check_cancelled(cancelled)
    return {"runtime_state": "stopped"}


def runtime_restart(
    options: Any,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> dict[str, Any]:
    _check_cancelled(cancelled)
    if progress:
        progress("restarting runtime", None)
    if cancelled:
        result = _cli().lifecycle_restart(options, cancelled=cancelled)
    else:
        result = _cli().lifecycle_restart(options)
    _check_cancelled(cancelled)
    if progress:
        progress("runtime restarted", 100)
    return result


def runtime_status() -> dict[str, Any]:
    result, code = _cli()._status()
    return {**result, "exit_code": code}


def run_doctor(interactive: bool = False) -> tuple[dict[str, Any], int]:
    cli = _cli()
    result, code = cli.lifecycle_doctor()
    if interactive and not result.get("checks", {}).get("colab_auth"):
        try:
            cli.ensure_colab_auth(interactive=True)
            result, code = cli.lifecycle_doctor()
        except (RuntimeError, EOFError, KeyboardInterrupt) as exc:
            print(cli.redact(str(exc)), file=sys.stderr)
    return result, code


def api_models() -> dict[str, Any]:
    return _cli()._api_request("models")


def api_chat(message: str) -> dict[str, Any]:
    return _cli()._api_request("chat/completions", method="POST", payload={
        "model": "local",
        "messages": [{"role": "user", "content": message}],
        "max_tokens": 32,
    })
