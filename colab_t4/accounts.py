"""Google Colab account profiles with automatic failover ordering.

Each account profile owns an isolated HOME directory under the global colab-t4
state directory. Named runtime namespaces deliberately share this registry so
quota/failure/LRU information coordinates all Colab sessions.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .backend import colab_cli_token_path
from .config import _atomic_json, global_state_dir

DEFAULT_ACCOUNT_ID = "default"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")

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
    return global_state_dir() / "accounts.json"


def account_home_dir(account_id: str) -> Path:
    return global_state_dir() / "accounts" / account_id


def _registry_lock_path() -> Path:
    return global_state_dir() / "accounts.lock"


def _account_lock_path(account_id: str) -> Path:
    return global_state_dir() / "account-locks" / f"{account_id}.lock"


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "a+") as handle:
            fd = -1
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if fd >= 0:
            os.close(fd)


@contextmanager
def _registry_lock() -> Iterator[None]:
    with _file_lock(_registry_lock_path()):
        yield


@contextmanager
def account_operation(account_id: str) -> Iterator[None]:
    if not _ID_RE.fullmatch(account_id):
        raise ValueError(f"invalid account id '{account_id}'")
    with _file_lock(_account_lock_path(account_id)):
        yield


def _load_accounts_unlocked() -> list[Account]:
    path = accounts_path()
    accounts: list[Account] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            accounts = [Account.from_dict(item) for item in raw if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            accounts = []
    return accounts


def _save_accounts_unlocked(accounts: list[Account]) -> None:
    _atomic_json(accounts_path(), [account.to_dict() for account in accounts])


def _ensure_default_unlocked(accounts: list[Account]) -> list[Account]:
    if not any(account.id == DEFAULT_ACCOUNT_ID for account in accounts):
        accounts.insert(0, Account(id=DEFAULT_ACCOUNT_ID))
        _save_accounts_unlocked(accounts)
    return accounts


def load_accounts() -> list[Account]:
    """Load the global account registry, bootstrapping ``default`` atomically."""
    with _registry_lock():
        return _ensure_default_unlocked(_load_accounts_unlocked())


def save_accounts(accounts: list[Account]) -> None:
    with _registry_lock():
        _save_accounts_unlocked(accounts)


def get_account(account_id: str) -> Account:
    for account in load_accounts():
        if account.id == account_id:
            return account
    raise KeyError(f"account '{account_id}' is not registered")


def add_account(account_id: str, *, home: str | None, email: str = "") -> Account:
    if not _ID_RE.fullmatch(account_id):
        raise ValueError("account id must start with a letter or digit and use only letters, digits, '.', '_', '-'")
    with _registry_lock():
        accounts = _ensure_default_unlocked(_load_accounts_unlocked())
        if any(account.id == account_id for account in accounts):
            raise ValueError(f"account '{account_id}' already exists")
        account = Account(
            id=account_id,
            email=email,
            home=home,
            added_at=_now(),
        )
        accounts.append(account)
        _save_accounts_unlocked(accounts)
        return account


def remove_account(account_id: str) -> None:
    with _registry_lock():
        accounts = _ensure_default_unlocked(_load_accounts_unlocked())
        remaining = [account for account in accounts if account.id != account_id]
        if len(remaining) == len(accounts):
            raise KeyError(f"account '{account_id}' is not registered")
        if account_id == DEFAULT_ACCOUNT_ID:
            raise ValueError("the 'default' account is implicit and cannot be removed")
        _save_accounts_unlocked(remaining)


def record_success(account_id: str) -> None:
    with _registry_lock():
        accounts = _ensure_default_unlocked(_load_accounts_unlocked())
        for account in accounts:
            if account.id == account_id:
                account.last_error = ""
                account.last_ok = _now()
                account.last_used_at = _now()
        _save_accounts_unlocked(accounts)


def record_failure(account_id: str, error: str) -> None:
    with _registry_lock():
        accounts = _ensure_default_unlocked(_load_accounts_unlocked())
        for account in accounts:
            if account.id == account_id:
                account.last_error = error[:500]
                account.last_used_at = _now()
        _save_accounts_unlocked(accounts)


def _now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ordered_accounts(accounts: list[Account], session: str, state: dict[str, Any]) -> list[Account]:
    def sort_key(account: Account) -> tuple[int, str]:
        return (1 if account.last_error else 0, account.last_used_at or "")

    ordered = sorted(accounts, key=sort_key)
    active = state.get("account")
    if active and state.get("session") == session:
        first = next((account for account in accounts if account.id == active), None)
        if first is not None and not first.last_error:
            ordered = [first] + [account for account in ordered if account.id != active]
    return ordered


def candidate_accounts(session: str, state: dict[str, Any]) -> list[Account]:
    """Order accounts for a provisioning attempt without reserving one."""
    return _ordered_accounts(load_accounts(), session, state)


def claim_candidate_account(
    session: str,
    state: dict[str, object],
    excluded: set[str] | None = None,
) -> Account:
    """Atomically select and LRU-reserve the next account for provisioning."""
    excluded = excluded or set()
    with _registry_lock():
        accounts = _ensure_default_unlocked(_load_accounts_unlocked())
        ordered = _ordered_accounts(accounts, session, state)
        chosen = next((account for account in ordered if account.id not in excluded), None)
        if chosen is None:
            raise RuntimeError("no Colab accounts available for provisioning")
        chosen.last_used_at = _now()
        _save_accounts_unlocked(accounts)
        return Account.from_dict(chosen.to_dict())


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
