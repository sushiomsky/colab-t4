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
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import colab_cli_token_path
from .config import _atomic_json, state_dir

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
    return state_dir() / "accounts.json"


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
    if not _ID_RE.match(account_id):
        raise ValueError("account id must start with a letter or digit and use only letters, digits, '.', '_', '-'")
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
    accounts = load_accounts()
    for account in accounts:
        if account.id == account_id:
            account.last_error = error[:500]
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
      is tried first (affinity) as long as it is healthy, then the rotation
      continues. A hosting account with a recorded failure never jumps the
      queue, so re-provisioning a dead session moves straight to the next
      account.
    """
    accounts = load_accounts()

    def sort_key(account: Account) -> tuple[int, str]:
        return (1 if account.last_error else 0, account.last_used_at or "")

    ordered = sorted(accounts, key=sort_key)
    active = state.get("account")
    if active and state.get("session") == session:
        first = next((account for account in accounts if account.id == active), None)
        if first is not None and not first.last_error:
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
