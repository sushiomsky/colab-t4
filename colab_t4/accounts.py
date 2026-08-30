"""Google Colab account profiles with automatic failover ordering.

Each account profile owns an isolated HOME directory under the colab-t4 state
directory. The installed Colab CLI reads its credentials from
``~/.config/colab-cli/token.json`` (and its session registry from
``~/.config/colab-cli/sessions.json``); overriding HOME for the CLI
subprocess gives every account its own independent OAuth identity.

The ``default`` account is implicit: it has no HOME override and uses the
real login home, which preserves the legacy single-account setup. Every other
account gets ``<state>/accounts/<id>`` as its HOME.
"""
from __future__ import annotations

import json
import os
import pty
import re
import shutil
import select
import signal
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import ColabCLI, colab_cli_token_path
from .config import _atomic_json, load_secrets, redact, state_dir

DEFAULT_ACCOUNT_ID = "default"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
AUTH_URL_RE = re.compile(r"https://accounts\.google\.com/o/oauth2/auth\?\S+")

ACCOUNT_SCHEMA = ("id", "email", "home", "added_at", "last_used_at", "last_error", "last_ok")


@dataclass
class Account:
    id: str
    email: str = ""
    home: str | None = None
    added_at: str = ""
    last_used_at: str = ""
    last_error: str = ""
    last_ok: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Account":
        values = {key: data.get(key, "") for key in ACCOUNT_SCHEMA}
        values["home"] = data.get("home") or None
        return cls(**values)

    def to_dict(self) -> dict[str, str]:
        return {key: getattr(self, key) or "" for key in ACCOUNT_SCHEMA}


def accounts_path() -> Path:
    return state_dir() / "accounts.json"


def _validate_account_id(account_id: str) -> str:
    if not isinstance(account_id, str):
        raise ValueError("id must be a string")
    if not _ID_RE.match(account_id):
        raise ValueError("account id must start with a letter or digit and use only letters, digits, '.', '_', '-'")
    return account_id


def _optional_string(values: dict[str, Any], key: str, default: str = "") -> str:
    if key not in values:
        return default
    value = values[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip()


def _optional_account_id(values: dict[str, Any]) -> str:
    if "id" not in values:
        return next_account_id()
    value = values["id"]
    if not isinstance(value, str):
        raise ValueError("id must be a string")
    return value or next_account_id()


def _validate_new_account_id(account_id: str) -> str:
    account_id = _validate_account_id(account_id)
    if account_id == DEFAULT_ACCOUNT_ID:
        raise ValueError("the 'default' account is implicit and cannot be created or removed")
    if any(account.id == account_id for account in load_accounts()):
        raise ValueError(f"account '{account_id}' already exists")
    return account_id


def load_accounts() -> list[Account]:
    """Load the account registry, bootstrapping the implicit ``default`` account."""
    path = accounts_path()
    accounts: list[Account] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            accounts = [Account.from_dict(item) for item in raw if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            accounts = []
    if not any(account.id == DEFAULT_ACCOUNT_ID for account in accounts):
        accounts.insert(0, Account(id=DEFAULT_ACCOUNT_ID))
        save_accounts(accounts)
    return accounts


def save_accounts(accounts: list[Account]) -> None:
    _atomic_json(accounts_path(), [account.to_dict() for account in accounts])


def get_account(account_id: str) -> Account:
    for account in load_accounts():
        if account.id == account_id:
            return account
    raise KeyError(f"account '{account_id}' is not registered")


def add_account(account_id: str, *, home: str | None, email: str = "") -> Account:
    _validate_account_id(account_id)
    accounts = load_accounts()
    if any(account.id == account_id for account in accounts):
        raise ValueError(f"account '{account_id}' already exists")
    account = Account(
        id=account_id,
        email=email,
        home=home,
        added_at=_now(),
    )
    accounts.append(account)
    save_accounts(accounts)
    return account


def remove_account(account_id: str) -> None:
    accounts = load_accounts()
    remaining = [account for account in accounts if account.id != account_id]
    if len(remaining) == len(accounts):
        raise KeyError(f"account '{account_id}' is not registered")
    if account_id == DEFAULT_ACCOUNT_ID:
        raise ValueError("the 'default' account is implicit and cannot be removed")
    save_accounts(remaining)


def account_home_dir(account_id: str) -> Path:
    return state_dir() / "accounts" / account_id


def record_success(account_id: str) -> None:
    accounts = load_accounts()
    for account in accounts:
        if account.id == account_id:
            account.last_error = ""
            account.last_ok = _now()
            account.last_used_at = _now()
    save_accounts(accounts)


def record_failure(account_id: str, error: str) -> None:
    safe_error = redact(str(error), list(load_secrets().values()))[:500]
    accounts = load_accounts()
    for account in accounts:
        if account.id == account_id:
            account.last_error = safe_error
            account.last_used_at = _now()
    save_accounts(accounts)


def _now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def candidate_accounts(session: str, state: dict[str, Any]) -> list[Account]:
    """Order accounts for a provisioning attempt.

    - Healthy accounts (no recorded last error) come before errored ones.
    - Within each group, the least-recently-used account comes first, so
      concurrent sessions round-robin across accounts (a second T4 lands on
      the next account instead of stacking on one).
    - When the requested session is already recorded, the account hosting it
      is tried first (affinity), then the rotation continues.
    """
    accounts = load_accounts()

    def sort_key(account: Account) -> tuple[int, str]:
        return (1 if account.last_error else 0, account.last_used_at or "")

    ordered = sorted(accounts, key=sort_key)
    active = state.get("account")
    if active and state.get("session") == session:
        first = next((account for account in accounts if account.id == active), None)
        if first is not None:
            ordered = [first] + [account for account in ordered if account.id != active]
    return ordered


def resolve_account_email(home: str | None) -> str | None:
    """Resolve the Google account email behind an OAuth token, if possible."""
    token_path = colab_cli_token_path(home)
    try:
        token = json.loads(token_path.read_text(encoding="utf-8"))
        refresh_token = token.get("refresh_token")
        client_id = token.get("client_id")
        client_secret = token.get("client_secret")
        if not (refresh_token and client_id and client_secret):
            return None
        body = urllib.parse.urlencode({
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
        }).encode()
        request = urllib.request.Request("https://oauth2.googleapis.com/token", data=body)
        with urllib.request.urlopen(request, timeout=30) as response:
            access_token = json.loads(response.read().decode())["access_token"]
        request = urllib.request.Request(
            "https://www.googleapis.com/oauth2/v3/userinfo?access_token=" + urllib.parse.quote(access_token)
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode()).get("email")
    except Exception:
        return None


def next_account_id() -> str:
    """Suggest a fresh account id (``account-2``, ``account-3``, ...)."""
    existing = {account.id for account in load_accounts()}
    index = 2
    while f"account-{index}" in existing:
        index += 1
    return f"account-{index}"


def _normalize_token_json(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("token_json must be a JSON object") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError("token_json must be a non-empty object")
    return value


def _write_token_json(home: str, token_json: Any) -> None:
    token = _normalize_token_json(token_json)
    target = colab_cli_token_path(home)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _atomic_json(target, token, mode=0o600)


def create_account(values: dict[str, Any]) -> dict[str, str]:
    """Create an isolated account profile, optionally importing token JSON."""
    if not isinstance(values, dict):
        raise ValueError("account payload must be an object")
    allowed = {"id", "email", "token_json"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown account field: " + unknown[0])
    account_id = _validate_new_account_id(_optional_account_id(values))
    email = _optional_string(values, "email")
    home = account_home_dir(account_id)
    try:
        home.mkdir(mode=0o700, parents=True, exist_ok=False)
        if "token_json" in values:
            _write_token_json(str(home), values["token_json"])
        add_account(account_id, home=str(home), email=email)
    except Exception:
        _remove_account_home(account_id, str(home), reject_outside=False)
        raise
    return {"account_id": account_id, "email": email, "status": "added"}


_PENDING_OAUTH: dict[str, dict[str, Any]] = {}
_PENDING_LOCK = threading.RLock()
_OAUTH_TTL_SECONDS = 900
_OAUTH_REAPER_INTERVAL_SECONDS = 5.0
_OAUTH_REAPER_STARTED = False
_OAUTH_REAPER_WAKE = threading.Event()


def _pending_key(account_id: str) -> str:
    return f"{state_dir()}::{account_id}"


def _expiry_text(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _prune_expired_oauth(now: float | None = None) -> None:
    current = time.time() if now is None else now
    expired = [key for key, pending in _PENDING_OAUTH.items() if float(pending.get("expires_at_ts", 0)) <= current]
    for key in expired:
        _cleanup_pending_oauth(_PENDING_OAUTH.pop(key), terminate=True)


def _ensure_oauth_reaper_locked() -> None:
    global _OAUTH_REAPER_STARTED
    if _OAUTH_REAPER_STARTED:
        _OAUTH_REAPER_WAKE.set()
        return
    _OAUTH_REAPER_STARTED = True
    thread = threading.Thread(target=_oauth_reaper_loop, name="colab-t4-oauth-reaper", daemon=True)
    thread.start()
    _OAUTH_REAPER_WAKE.set()


def _oauth_reaper_loop() -> None:
    while True:
        _OAUTH_REAPER_WAKE.wait(float(_OAUTH_REAPER_INTERVAL_SECONDS))
        _OAUTH_REAPER_WAKE.clear()
        with _PENDING_LOCK:
            _prune_expired_oauth()


def list_pending_oauth() -> list[dict[str, str]]:
    with _PENDING_LOCK:
        _prune_expired_oauth()
        return [
            {"account_id": pending["account_id"], "expires_at": pending["expires_at"]}
            for pending in sorted(_PENDING_OAUTH.values(), key=lambda item: item["account_id"])
            if pending.get("state_dir") == str(state_dir())
        ]


def start_oauth(values: dict[str, Any]) -> dict[str, str]:
    """Start a browserless Colab OAuth flow and keep pending state in memory."""
    if not isinstance(values, dict):
        raise ValueError("auth-start payload must be an object")
    allowed = {"id"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown auth-start field: " + unknown[0])
    account_id = _validate_new_account_id(_optional_account_id(values))
    home = account_home_dir(account_id)
    pending_key = _pending_key(account_id)
    with _PENDING_LOCK:
        _prune_expired_oauth()
        if pending_key in _PENDING_OAUTH:
            raise ValueError(f"account '{account_id}' already has a pending authentication flow")
        try:
            home.mkdir(mode=0o700, parents=True, exist_ok=False)
            process = _start_colab_oauth_process(account_id, str(home))
            expires_at_ts = time.time() + _OAUTH_TTL_SECONDS
            pending = {
                "account_id": account_id,
                "state_dir": str(state_dir()),
                "home": str(home),
                "pid": process.get("pid"),
                "fifo": process.get("fifo", ""),
                "log": process.get("log", ""),
                "expires_at_ts": expires_at_ts,
                "expires_at": _expiry_text(expires_at_ts),
            }
            _PENDING_OAUTH[pending_key] = pending
            _ensure_oauth_reaper_locked()
            return {
                "account_id": account_id,
                "authorization_url": str(process["authorization_url"]),
                "expires_at": pending["expires_at"],
            }
        except Exception:
            _remove_account_home(account_id, str(home), reject_outside=False)
            raise


def finish_oauth(values: dict[str, Any]) -> dict[str, str]:
    """Finish a pending OAuth flow without retaining or returning the code."""
    if not isinstance(values, dict):
        raise ValueError("auth-finish payload must be an object")
    allowed = {"id", "code"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown auth-finish field: " + unknown[0])
    if not isinstance(values.get("id"), str):
        raise ValueError("id must be a string")
    account_id = _validate_account_id(values["id"])
    pending_key = _pending_key(account_id)
    if not isinstance(values.get("code"), str):
        raise ValueError("code must be a string")
    code = values["code"].strip()
    if not code:
        raise ValueError("authorization code is required")
    with _PENDING_LOCK:
        _prune_expired_oauth()
        pending = _PENDING_OAUTH.get(pending_key)
        if not pending:
            raise ValueError("no matching pending Google authentication flow")
    try:
        _finish_colab_oauth_process(pending, code)
        ok, reason = _verify_account_auth(str(pending["home"]))
        if not ok:
            raise RuntimeError("authentication verification failed: " + reason)
        email = resolve_account_email(str(pending["home"])) or ""
        add_account(account_id, home=str(pending["home"]), email=email)
    except Exception:
        with _PENDING_LOCK:
            _PENDING_OAUTH.pop(pending_key, None)
        _cleanup_pending_oauth(pending, remove_home=True)
        raise
    with _PENDING_LOCK:
        _PENDING_OAUTH.pop(pending_key, None)
    _cleanup_pending_oauth(pending, remove_home=False)
    return {"account_id": account_id, "email": email, "status": "added"}


def cancel_oauth(account_id: Any) -> dict[str, str]:
    account_id = _validate_account_id(account_id)
    pending_key = _pending_key(account_id)
    with _PENDING_LOCK:
        _prune_expired_oauth()
        pending = _PENDING_OAUTH.pop(pending_key, None)
    if not pending:
        raise ValueError("no matching pending Google authentication flow")
    _cleanup_pending_oauth(pending, remove_home=True, terminate=True)
    return {"account_id": account_id, "status": "cancelled"}


def _cleanup_pending_oauth(pending: dict[str, Any], *, remove_home: bool = True, terminate: bool = False) -> None:
    if terminate and pending.get("pid"):
        try:
            os.kill(int(pending["pid"]), signal.SIGTERM)
        except (ProcessLookupError, ValueError, TypeError):
            pass
    fifo_text = str(pending.get("fifo", ""))
    if fifo_text:
        try:
            Path(fifo_text).unlink()
        except FileNotFoundError:
            pass
    log_text = str(pending.get("log", ""))
    if log_text:
        try:
            Path(log_text).unlink()
        except FileNotFoundError:
            pass
    pty_fd = pending.get("pty_fd")
    if pty_fd is not None:
        try:
            os.close(int(pty_fd))
        except OSError:
            pass
    if remove_home:
        try:
            _remove_account_home(str(pending.get("account_id", "")), str(pending.get("home", "")), reject_outside=False)
        except ValueError:
            pass


def _start_colab_oauth_process(account_id: str, home: str) -> dict[str, Any]:
    master_fd: int | None = None
    slave_fd: int | None = None
    process: subprocess.Popen[bytes] | None = None
    success = False
    try:
        cli = ColabCLI.discover(home=home)
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            cli.sessions_command(),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=cli.cli_env(),
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave_fd)
        slave_fd = None
        url = ""
        transcript = ""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master_fd], [], [], 0.2)
            if readable:
                try:
                    transcript = (transcript + os.read(master_fd, 4096).decode(errors="replace"))[-8192:]
                except OSError:
                    transcript = transcript[-8192:]
                match = AUTH_URL_RE.search(transcript)
                if match:
                    url = match.group(0)
                    break
            if process.poll() is not None:
                break
        if not url:
            process.terminate()
            raise RuntimeError("Colab CLI did not produce an authorization URL")
        success = True
        return {"authorization_url": url, "pid": process.pid, "pty_fd": master_fd}
    finally:
        if slave_fd is not None:
            try:
                os.close(slave_fd)
            except OSError:
                pass
        if not success:
            if process is not None and process.poll() is None:
                process.terminate()
            if master_fd is not None:
                try:
                    os.close(master_fd)
                except OSError:
                    pass


def _finish_colab_oauth_process(pending: dict[str, Any], code: str) -> None:
    if pending.get("pty_fd") is not None:
        try:
            os.write(int(pending["pty_fd"]), (code + "\n").encode())
        except OSError as exc:
            raise RuntimeError("could not submit authorization result") from exc
    else:
        fifo = Path(str(pending.get("fifo", "")))
        try:
            with fifo.open("w", encoding="utf-8") as handle:
                handle.write(code + "\n")
        except OSError as exc:
            raise RuntimeError("could not submit authorization result: " + str(exc)) from exc
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            os.kill(int(pending["pid"]), 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)


def _verify_account_auth(home: str) -> tuple[bool, str]:
    from .wizard import _auth_ok

    return _auth_ok(ColabCLI.discover(home=home), home)


def _expected_account_home(account_id: str) -> Path:
    return (state_dir() / "accounts").resolve(strict=False) / _validate_account_id(account_id)


def _remove_account_home(account_id: str, home: str, *, reject_outside: bool) -> None:
    if not home:
        return
    expected = _expected_account_home(account_id)
    candidate = Path(home)
    is_expected_path = candidate.parent.resolve(strict=False) == expected.parent and candidate.name == expected.name
    if not is_expected_path or candidate.is_symlink():
        if reject_outside:
            raise ValueError("account home is outside account directory")
        return
    shutil.rmtree(expected, ignore_errors=True)


def remove_account_profile(account_id: Any) -> dict[str, str]:
    account_id = _validate_account_id(account_id)
    if account_id == DEFAULT_ACCOUNT_ID:
        raise ValueError("the 'default' account is implicit and cannot be removed")
    account = get_account(account_id)
    if account.home:
        _remove_account_home(account_id, account.home, reject_outside=True)
    remove_account(account_id)
    if account.home:
        _remove_account_home(account_id, account.home, reject_outside=False)
    return {"status": "removed", "id": account_id}
