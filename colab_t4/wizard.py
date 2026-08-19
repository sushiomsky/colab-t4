"""Interactive first-run configuration wizard.

Secrets are collected with getpass, kept in the returned dictionary for this
process, and persisted only after an explicit user choice.
"""
from __future__ import annotations

import getpass
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from .backend import ColabCLI, ColabCLIError, auth_summary, authenticate_available
from .config import generated_api_key, generated_password, load_secrets, save_secrets, clear_secrets
from .notebook import DEFAULT_CTX, DEFAULT_HOSTNAME, DEFAULT_MODEL, DEFAULT_PORT, DEFAULT_QUANT

Input = Callable[[str], str]


def interactive_available(force: bool = False) -> bool:
    # A flag may choose interactive policy, but cannot make redirected stdin
    # safe for secret prompts.
    return bool(sys.stdin.isatty() and sys.stderr.isatty())


def _ask(prompt: str, *, default: str = "", input_fn: Input = input) -> str:
    suffix = f" [{default}]" if default else ""
    value = input_fn(prompt + suffix + ": ").strip()
    return value if value else default


def _yes(prompt: str, *, default: bool = True, input_fn: Input = input) -> bool:
    marker = "Y/n" if default else "y/N"
    value = input_fn(f"{prompt} [{marker}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def _secret(prompt: str, *, allow_empty: bool = False, getpass_fn=getpass.getpass) -> str:
    while True:
        value = getpass_fn(prompt + (" (optional): " if allow_empty else ": ")).strip()
        if value or allow_empty:
            return value
        print("A value is required; press Ctrl-C to cancel.", file=sys.stderr)


def _default_pubkey() -> Path | None:
    candidates = [Path.home() / ".ssh" / "id_ed25519.pub", Path.home() / ".ssh" / "id_rsa.pub"]
    return next((path for path in candidates if path.is_file()), None)


def _auth_ok(cli: ColabCLI, home: str | None = None) -> tuple[bool, str]:
    if not authenticate_available(home):
        return False, "credentials are missing"
    result = cli.run(cli.sessions_command(), cli.auth_log_path(), timeout=60)
    text = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0:
        return True, "verified"
    lowered = text.lower()
    if "quota" in lowered or "entitlement" in lowered:
        return False, "account quota or entitlement rejected the request"
    if "401" in lowered or "unauthorized" in lowered or "credential" in lowered:
        return False, "credentials are missing, expired, or invalid"
    return False, "authentication verification failed"


def ensure_colab_auth(*, interactive: bool, input_fn: Input = input, cli: ColabCLI | None = None) -> None:
    cli = cli or ColabCLI.discover()
    home = getattr(cli, "home", None)
    ok, reason = _auth_ok(cli, home)
    if ok:
        return
    if not interactive:
        raise RuntimeError("Colab authentication is missing or invalid; run `colab sessions` interactively")
    print("Colab authentication is missing or could not be verified.")
    print("The installed CLI starts its remote OAuth flow with `colab sessions`.")
    if not _yes("Start the Colab CLI authentication flow now?", input_fn=input_fn):
        raise RuntimeError("Colab authentication is required")
    print("Running the installed command: colab sessions")
    try:
        result = cli.run_interactive_auth()
    except KeyboardInterrupt:
        raise RuntimeError("Colab authentication cancelled")
    if result != 0:
        raise RuntimeError("Colab authentication command failed; run `colab sessions` and retry")
    print("Waiting for authentication verification...")
    ok, reason = _auth_ok(cli, home)
    if not ok:
        raise RuntimeError("Colab authentication still failed: " + reason)
    print("Colab authentication verified.")


def collect(
    options: Any,
    *,
    input_fn: Input = input,
    getpass_fn=getpass.getpass,
    force: bool = False,
    allow_prompt: bool | None = None,
) -> dict[str, Any]:
    """Collect all runtime values, applying explicit/env/saved/prompt/default precedence."""
    saved = load_secrets()
    values: dict[str, Any] = {}
    values["session"] = getattr(options, "session", None) or os.environ.get("COLAB_T4_SESSION") or saved.get("session") or DEFAULT_HOSTNAME
    values["model"] = getattr(options, "model", None) or os.environ.get("COLAB_T4_MODEL") or saved.get("model_repo") or DEFAULT_MODEL
    values["quant"] = getattr(options, "quant", None) or os.environ.get("COLAB_T4_QUANT") or saved.get("quant") or DEFAULT_QUANT
    values["port"] = getattr(options, "port", None) or int(os.environ.get("COLAB_T4_PORT", saved.get("port", DEFAULT_PORT)))
    values["ctx"] = getattr(options, "ctx", None) or int(os.environ.get("COLAB_T4_CTX", saved.get("ctx", DEFAULT_CTX)))
    values["exec_timeout"] = getattr(options, "exec_timeout", None) or float(os.environ.get("COLAB_T4_EXEC_TIMEOUT", 7200))
    values["ready_timeout"] = float(os.environ.get("COLAB_T4_READY_TIMEOUT", 900))
    values["tailnet"] = os.environ.get("TS_TAILNET", saved.get("tailnet", ""))
    values["tailscale_authkey"] = os.environ.get("TS_AUTHKEY", saved.get("tailscale_authkey", ""))
    values["hf_token"] = os.environ.get("HF_TOKEN", saved.get("hf_token", ""))
    values["api_key"] = getattr(options, "api_key", None) or os.environ.get("COLAB_T4_API_KEY", saved.get("api_key", ""))
    values["ssh_mode"] = getattr(options, "ssh_mode", None) or os.environ.get("COLAB_T4_SSH_MODE", saved.get("ssh_mode", "tailscale"))
    values["ssh_password"] = getattr(options, "password", None) or os.environ.get("COLAB_T4_SSH_PASSWORD", saved.get("ssh_password", ""))
    values["ssh_pubkey"] = ""
    explicit_pubkey = getattr(options, "pubkey", None)
    candidate = Path(explicit_pubkey).expanduser() if explicit_pubkey else _default_pubkey()
    if candidate and candidate.is_file():
        values["ssh_pubkey_path"] = str(candidate)
        values["ssh_pubkey"] = candidate.read_text(encoding="utf-8").strip()
    else:
        values["ssh_pubkey_path"] = ""
    prompting = interactive_available(False) if allow_prompt is None else allow_prompt
    if not (force or prompting):
        missing = []
        if not values["tailscale_authkey"]:
            missing.append("TS_AUTHKEY")
        if not values["api_key"]:
            missing.append("COLAB_T4_API_KEY")
        if values["ssh_mode"] == "password" and not values["ssh_password"]:
            missing.append("COLAB_T4_SSH_PASSWORD")
        if values["ssh_mode"] == "key" and not values["ssh_pubkey"]:
            missing.append("SSH public key")
        if missing:
            raise RuntimeError("missing configuration: " + ", ".join(missing) + ". Use `colab-t4 configure` or environment variables.")
        values["persist_secrets"] = False
        return values

    if not values["tailscale_authkey"]:
        print("Tailscale authentication key is missing; it is needed to join your tailnet.")
        values["tailscale_authkey"] = _secret("Enter TS_AUTHKEY", getpass_fn=getpass_fn)
    if not values["tailnet"]:
        values["tailnet"] = _ask("Tailnet name is optional; enter it or press Enter for automatic discovery", input_fn=input_fn)
    if values["ssh_mode"] not in {"tailscale", "password", "key"}:
        raise RuntimeError("SSH mode must be tailscale, password, or key")
    if values["ssh_mode"] == "tailscale":
        values["ssh_password"] = ""
        values["ssh_pubkey"] = ""
    elif values["ssh_mode"] == "key":
        if values["ssh_pubkey"]:
            print(f"SSH public key detected: {values['ssh_pubkey_path']}")
            if not _yes("Use this key?", input_fn=input_fn):
                values["ssh_pubkey"] = ""
                values["ssh_pubkey_path"] = ""
        if not values["ssh_pubkey"]:
            path_text = _ask("Path to SSH public key", input_fn=input_fn)
            path = Path(path_text).expanduser()
            if not path.is_file():
                raise RuntimeError("SSH public key file does not exist")
            values["ssh_pubkey_path"] = str(path)
            values["ssh_pubkey"] = path.read_text(encoding="utf-8").strip()
    elif not values["ssh_password"]:
        values["ssh_password"] = _secret("Enter SSH password", getpass_fn=getpass_fn)
    if not values["api_key"]:
        print("No API key was provided.")
        if _yes("Generate a secure API key automatically?", input_fn=input_fn):
            values["api_key"] = generated_api_key()
            values["api_key_origin"] = "generated"
        else:
            values["api_key"] = _secret("Enter API key", getpass_fn=getpass_fn)
            values["api_key_origin"] = "entered"
    if not values["hf_token"]:
        values["hf_token"] = _secret("Enter HF_TOKEN", allow_empty=True, getpass_fn=getpass_fn)
    values["model"] = _ask("Model repository", default=values["model"], input_fn=input_fn)
    values["quant"] = _ask("Quantization", default=values["quant"], input_fn=input_fn)
    values["session"] = _ask("Colab session name", default=values["session"], input_fn=input_fn)
    values["exec_timeout"] = float(_ask("Execution timeout seconds", default=str(values["exec_timeout"]), input_fn=input_fn))
    values["ready_timeout"] = float(_ask("Readiness timeout seconds", default=str(values["ready_timeout"]), input_fn=input_fn))
    values["persist_secrets"] = _persistence_choice(input_fn)
    return values
def _persistence_choice(input_fn: Input = input) -> bool:
    print("Use entered secrets for:")
    print("  1. This run only (recommended)")
    print("  2. Save securely for future runs")
    while True:
        choice = input_fn("Choice [1]: ").strip() or "1"
        if choice in {"1", "2"}:
            return choice == "2"
        print("Enter 1 or 2.")
def persist(values: dict[str, Any]) -> None:
    if not values.get("persist_secrets"):
        return
    save_secrets({
        "tailscale_authkey": values.get("tailscale_authkey", ""),
        "tailnet": values.get("tailnet", ""),
        "hf_token": values.get("hf_token", ""),
        "api_key": values.get("api_key", ""),
        "ssh_mode": values.get("ssh_mode", "tailscale"),
        "ssh_password": values.get("ssh_password", ""),
        "ssh_pubkey": values.get("ssh_pubkey", ""),
        "model_repo": values.get("model", DEFAULT_MODEL),
        "quant": values.get("quant", DEFAULT_QUANT),
        "session": values.get("session", DEFAULT_HOSTNAME),
        "port": str(values.get("port", DEFAULT_PORT)),
        "ctx": str(values.get("ctx", DEFAULT_CTX)),
    })


def configure_status() -> dict[str, bool | str]:
    saved = load_secrets()
    return {
        "tailscale_authkey": bool(saved.get("tailscale_authkey")),
        "tailnet": bool(saved.get("tailnet")),
        "hf_token": bool(saved.get("hf_token")),
        "api_key": bool(saved.get("api_key")),
        "ssh_mode": saved.get("ssh_mode", "tailscale"),
        "ssh_password": bool(saved.get("ssh_password")),
        "ssh_pubkey": bool(saved.get("ssh_pubkey")),
    }


def reset(*, yes: bool, input_fn: Input = input) -> None:
    if not yes and interactive_available(False) and not _yes("Delete saved colab-t4 credentials?", default=False, input_fn=input_fn):
        print("Reset cancelled.")
        return
    clear_saved = __import__("colab_t4.config", fromlist=["clear_secrets"]).clear_secrets
    clear_saved()
    print("Saved credentials removed.")
