"""Browserless Colab session lifecycle orchestration."""
from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .accounts import (
    account_operation,
    candidate_accounts,
    claim_candidate_account,
    get_account,
    record_failure,
    record_success,
)
from .backend import ColabCLI, ColabCLIError, authenticate_available
from .config import (
    clear_runtime_api_key,
    generated_api_key,
    generated_password,
    load_secrets,
    load_state,
    logs_dir,
    save_secrets,
    save_state,
)
from .notebook import DEFAULT_CTX, DEFAULT_HOSTNAME, DEFAULT_MODEL, DEFAULT_PORT, DEFAULT_QUANT, build_notebook

REMOTE_SECRET = "/content/.colab-t4-secrets.json"
REMOTE_NOTEBOOK = "/content/colab-t4.ipynb"
REMOTE_READY = "/content/.colab-t4-ready.json"


class SessionUnreachable(RuntimeError):
    """The recorded Colab session cannot be reached (stopped or lost)."""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _update(**values: Any) -> dict[str, Any]:
    state = load_state()
    state.update(values)
    state["updated_at"] = now()
    save_state(state)
    return state


def _log(kind: str) -> Path:
    return logs_dir() / f"{kind}.log"


def _session_name(options: Any) -> str:
    return getattr(options, "session", None) or DEFAULT_HOSTNAME


def _account_home(state: dict[str, Any]) -> str | None:
    """HOME override for the account recorded in state, if any."""
    account_id = state.get("account")
    if not account_id:
        return None
    try:
        return get_account(account_id).home
    except KeyError:
        return None


def doctor() -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {"checks": {}, "colab_cli": None, "auth": None}
    try:
        cli = ColabCLI.discover()
        result["colab_cli"] = {"executable": cli.executable, "version": cli.version}
        result["checks"]["colab_cli"] = True
    except ColabCLIError as exc:
        result["checks"]["colab_cli"] = False
        result["error"] = str(exc)
        return result, 1
    configured = authenticate_available()
    auth = {"available": configured, "verified": False, "state": "missing" if not configured else "unverified"}
    if configured:
        checked = cli.run(cli.sessions_command(), logs_dir() / "auth-check.log", timeout=60)
        auth["verified"] = checked.returncode == 0
        auth["state"] = "verified" if checked.returncode == 0 else "expired-or-invalid"
    result["auth"] = auth
    result["checks"]["colab_auth"] = bool(auth["verified"])
    result["checks"]["tailscale_authkey"] = bool(os.environ.get("TS_AUTHKEY") or load_secrets().get("tailscale_authkey"))
    result["checks"]["ssh_client"] = bool(__import__("shutil").which("ssh"))
    result["checks"]["python"] = True
    return result, 0 if all(result["checks"].values()) else 1


def _secret_values() -> list[str]:
    return [value for value in load_secrets().values() if isinstance(value, str)]


def _write_secret_file(options: Any, session: str) -> tuple[Path, dict[str, str]]:
    configured = getattr(options, "runtime_config", None) or {}
    secrets = load_secrets()
    config = {
        "tailscale_authkey": configured.get("tailscale_authkey") or os.environ.get("TS_AUTHKEY") or secrets.get("tailscale_authkey", ""),
        "api_key": configured.get("api_key") or getattr(options, "api_key", None) or os.environ.get("COLAB_T4_API_KEY") or secrets.get("api_key", ""),
        "ssh_mode": configured.get("ssh_mode") or getattr(options, "ssh_mode", None) or os.environ.get("COLAB_T4_SSH_MODE") or secrets.get("ssh_mode", "tailscale"),
        "ssh_password": configured.get("ssh_password") or getattr(options, "password", None) or os.environ.get("COLAB_T4_SSH_PASSWORD") or secrets.get("ssh_password", ""),
        "ssh_pubkey": configured.get("ssh_pubkey", ""),
        "hf_token": configured.get("hf_token") or os.environ.get("HF_TOKEN", "") or secrets.get("hf_token", ""),
        "model_repo": configured.get("model", getattr(options, "model", DEFAULT_MODEL)),
        "quant": configured.get("quant", getattr(options, "quant", DEFAULT_QUANT)),
        "port": str(configured.get("port", getattr(options, "port", DEFAULT_PORT))),
        "ctx": str(configured.get("ctx", getattr(options, "ctx", DEFAULT_CTX))),
        "hostname": configured.get("session", session),
    }
    if not config["tailscale_authkey"]:
        raise RuntimeError("TS_AUTHKEY is required for browserless provisioning")
    if not config["api_key"]:
        raise RuntimeError("API credentials are required")
    if config["ssh_mode"] not in {"tailscale", "password", "key"}:
        raise RuntimeError("SSH mode must be tailscale, password, or key")
    if config["ssh_mode"] == "password" and not config["ssh_password"]:
        raise RuntimeError("SSH password is required for password mode")
    if config["ssh_mode"] == "key" and not config["ssh_pubkey"]:
        raise RuntimeError("SSH public key is required for key mode")
    fd, name = tempfile.mkstemp(prefix="colab-t4-secrets-", suffix=".json")
    os.close(fd)
    path = Path(name)
    path.write_text(json.dumps(config), encoding="utf-8")
    os.chmod(path, 0o600)
    return path, config


def _make_notebook(options: Any, path: Path) -> None:
    path.write_text(build_notebook(
        hostname=getattr(options, "session", DEFAULT_HOSTNAME) or DEFAULT_HOSTNAME,
        model_repo=getattr(options, "model", DEFAULT_MODEL),
        quant=getattr(options, "quant", DEFAULT_QUANT),
        port=getattr(options, "port", DEFAULT_PORT),
        ctx=getattr(options, "ctx", DEFAULT_CTX),
    ), encoding="utf-8")
    os.chmod(path, 0o600)


def _download_ready(cli: ColabCLI, session: str, secrets: list[str]) -> dict[str, Any] | None:
    target = logs_dir() / "runtime-ready.json"
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    result = cli.run(cli.download_command(session, REMOTE_READY, target), _log("runtime"), timeout=60, secrets=secrets)
    if result.returncode or not target.exists():
        return None
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def up(options: Any) -> dict[str, Any]:
    session = _session_name(options)
    current = load_state()
    if current.get("session") == session and current.get("runtime_state") in {"creating", "executing", "ready"}:
        try:
            return wait(options)
        except (SessionUnreachable, TimeoutError) as exc:
            accounts = candidate_accounts(session, load_state())
            if len(accounts) <= 1:
                raise
            active = current.get("account")
            if active:
                record_failure(active, f"existing session did not become ready: {exc}")
            print(f"existing session did not become ready ({exc}); trying next account", file=sys.stderr)
            try:
                down()
            except (RuntimeError, ColabCLIError):
                pass

    available = candidate_accounts(session, load_state())
    if not available:
        raise RuntimeError("no Colab accounts configured; run `colab-t4 accounts add`")

    account_count = len(available)
    failures: list[str] = []
    excluded: set[str] = set()
    for index in range(account_count):
        account = claim_candidate_account(session, load_state(), excluded)
        excluded.add(account.id)
        try:
            with account_operation(account.id):
                return _provision(options, session, account)
        except (ColabCLIError, RuntimeError, TimeoutError) as exc:
            if account_count == 1:
                raise
            message = str(exc)
            record_failure(account.id, message)
            failures.append(f"{account.id}: {message}")
            if index < account_count - 1:
                print(f"account '{account.id}' failed ({message}); trying next account", file=sys.stderr)
    raise RuntimeError("all accounts failed: " + " | ".join(failures))


def _provision(options: Any, session: str, account: Any) -> dict[str, Any]:
    cli = ColabCLI.discover(home=account.home)
    if not authenticate_available(account.home):
        raise RuntimeError(
            f"account '{account.id}' has no Colab CLI authentication; run `colab-t4 accounts add`"
        )
    notebook_fd, notebook_name = tempfile.mkstemp(prefix="colab-t4-", suffix=".ipynb")
    os.close(notebook_fd)
    notebook = Path(notebook_name)
    secret_file = None
    try:
        _make_notebook(options, notebook)
        secret_file, remote_config = _write_secret_file(options, session)
        secret_values = [str(value) for value in remote_config.values() if isinstance(value, str)]
        _update(session=session, account=account.id, runtime_state="creating", accelerator="T4", notebook="submitted", last_error=None, started_at=now())
        created = cli.run(cli.new_command(session, "T4"), _log("colab"), timeout=300, secrets=secret_values)
        if created.returncode:
            _update(runtime_state="failed", last_error="Colab CLI failed to create the T4 session")
            raise ColabCLIError("Colab CLI could not create the T4 session; see colab.log")
        _update(runtime_state="uploading")
        uploaded = cli.run(cli.upload_command(session, secret_file, REMOTE_SECRET), _log("colab"), timeout=120, secrets=secret_values)
        if uploaded.returncode:
            _update(runtime_state="failed", last_error="secret upload failed")
            raise ColabCLIError("failed to upload provisioning configuration")
        uploaded_nb = cli.run(cli.upload_command(session, notebook, REMOTE_NOTEBOOK), _log("colab"), timeout=120, secrets=secret_values)
        if uploaded_nb.returncode:
            _update(runtime_state="failed", last_error="notebook upload failed")
            raise ColabCLIError("failed to upload notebook")
        _update(runtime_state="executing")
        executed = cli.run(cli.exec_command(session, notebook, float(getattr(options, "exec_timeout", 7200))), _log("notebook"), timeout=float(getattr(options, "exec_timeout", 7200)) + 120, secrets=secret_values)
        if executed.returncode:
            _update(runtime_state="failed", last_error="remote notebook execution failed")
            raise ColabCLIError("remote provisioning failed; inspect `colab-t4 logs notebook`")
        ready = _download_ready(cli, session, secret_values)
        if not ready or not ready.get("ready") or not ready.get("tests", {}).get("chat"):
            _update(runtime_state="failed", last_error="readiness artifact missing or smoke test failed")
            raise ColabCLIError("remote provisioning completed without a valid readiness artifact")
        record_success(account.id)
        state = _update(
            runtime_state="ready",
            accelerator="T4",
            gpu=ready.get("gpu"),
            tailscale_ip=ready.get("tailscale_ip"),
            api_base=ready.get("api_base"),
            model=ready.get("model"),
            model_repo=remote_config["model_repo"],
            quant=remote_config["quant"],
            ssh_mode=ready.get("ssh_mode", remote_config.get("ssh_mode", "tailscale")),
            tests=ready.get("tests"),
            last_error=None,
            ready_at=now(),
        )
        return state
    except KeyboardInterrupt:
        _update(runtime_state="interrupted", last_error="interrupted by user")
        raise
    finally:
        try:
            notebook.unlink()
        except FileNotFoundError:
            pass
        if secret_file:
            try:
                secret_file.unlink()
            except FileNotFoundError:
                pass


def wait(options: Any) -> dict[str, Any]:
    state = load_state()
    cli = ColabCLI.discover(home=_account_home(state))
    session = getattr(options, "session", None) or state.get("session")
    if not session:
        raise RuntimeError("no recorded Colab session; run `colab-t4 up`")
    deadline = time.monotonic() + float(getattr(options, "timeout", 600))
    while time.monotonic() < deadline:
        state = load_state()
        state["session"] = session
        status = cli.run(cli.status_command(session), _log("status"), timeout=45, secrets=_secret_values())
        if status.returncode:
            state["runtime_state"] = "stopped"
            save_state(state)
            raise SessionUnreachable("Colab session is not reachable; inspect `colab-t4 logs status`")
        ready = _download_ready(cli, session, _secret_values())
        if ready and ready.get("ready"):
            state.update({"runtime_state": "ready", "gpu": ready.get("gpu"), "tailscale_ip": ready.get("tailscale_ip"), "api_base": ready.get("api_base"), "model": ready.get("model"), "tests": ready.get("tests"), "last_error": None, "ready_at": state.get("ready_at") or now()})
            save_state(state)
            return state
        state["runtime_state"] = "waiting"
        save_state(state)
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for Colab session '{session}'")


def down() -> None:
    state = load_state()
    session = state.get("session")
    if not session:
        clear_runtime_api_key()
        return
    cli = ColabCLI.discover(home=_account_home(state))
    result = cli.run(cli.stop_command(session), _log("colab"), timeout=120, secrets=_secret_values())
    # Colab stop is idempotent in the CLI; even an already-lost session must not
    # leave local runtime state behind.
    if result.returncode and "not found" not in (result.stdout + result.stderr).lower():
        raise ColabCLIError("failed to stop recorded Colab session; inspect colab.log")
    from .config import clear_state
    clear_state()
    clear_runtime_api_key()


def restart(options: Any) -> dict[str, Any]:
    try:
        down()
    except RuntimeError:
        pass
    return up(options)
