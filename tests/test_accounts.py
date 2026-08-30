"""Account registry and failover ordering tests."""
import argparse
import json
import threading
from pathlib import Path

import pytest

from colab_t4 import accounts
from colab_t4.accounts import (
    add_account,
    candidate_accounts,
    get_account,
    load_accounts,
    next_account_id,
    record_failure,
    record_success,
    remove_account,
    resolve_account_email,
)
from colab_t4.backend import import_token


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def test_default_account_is_bootstrapped(state):
    accounts = load_accounts()
    assert [a.id for a in accounts] == ["default"]
    assert accounts[0].home is None
    assert (state / "accounts.json").exists()


def test_add_and_remove_round_trip(state):
    add_account("work", home=str(state / "home-work"), email="work@example.com")
    add_account("lab", home=str(state / "home-lab"))
    assert [a.id for a in load_accounts()] == ["default", "work", "lab"]
    work = get_account("work")
    assert work.email == "work@example.com"
    assert work.home == str(state / "home-work")
    remove_account("work")
    assert [a.id for a in load_accounts()] == ["default", "lab"]


def test_remove_default_is_refused(state):
    with pytest.raises(ValueError):
        remove_account("default")


def test_add_rejects_duplicate_and_unsafe_ids(state):
    with pytest.raises(ValueError):
        add_account("default", home=None)
    with pytest.raises(ValueError):
        add_account("../evil", home=None)
    with pytest.raises(ValueError):
        add_account("has space", home=None)


def test_next_account_id_suggests_unused_name(state):
    assert next_account_id() == "account-2"
    add_account("account-2", home=str(state / "h2"))
    add_account("account-3", home=str(state / "h3"))
    assert next_account_id() == "account-4"


def test_record_success_clears_error_and_sets_last_ok(state):
    add_account("work", home=str(state / "h"))
    record_failure("work", "boom: Service Unavailable")
    assert get_account("work").last_error
    record_success("work")
    work = get_account("work")
    assert work.last_error == ""
    assert work.last_ok


def test_candidate_affinity_prefers_account_hosting_session(state):
    add_account("a", home=str(state / "ha"))
    add_account("b", home=str(state / "hb"))
    state_data = {"session": "qwenpaw-t4", "account": "b"}
    ordered = candidate_accounts("qwenpaw-t4", state_data)
    assert ordered[0].id == "b"
    assert [a.id for a in ordered] == ["b", "default", "a"]


def test_candidate_round_robin_for_new_session(state):
    add_account("a", home=str(state / "ha"))
    add_account("b", home=str(state / "hb"))
    record_success("b")
    # Different session: no affinity; healthy accounts first, least-used first.
    # default and a are both unused, so insertion order decides between them.
    ordered = candidate_accounts("other-t4", {"session": "qwenpaw-t4", "account": "b"})
    assert [a.id for a in ordered] == ["default", "a", "b"]


def test_candidate_tries_errored_accounts_last(state):
    add_account("a", home=str(state / "ha"))
    add_account("b", home=str(state / "hb"))
    record_failure("b", "quota exhausted")
    ordered = candidate_accounts("fresh-t4", {})
    assert [a.id for a in ordered] == ["default", "a", "b"]


def test_resolve_email_without_token_is_none(state):
    assert resolve_account_email(str(state / "missing-home")) is None


def test_accounts_file_is_mode_0600(state):
    load_accounts()
    assert (state / "accounts.json").stat().st_mode & 0o777 == 0o600
    raw = json.loads((state / "accounts.json").read_text())
    assert raw[0]["id"] == "default"


def test_auth_cancel_removes_pending_temporary_profile(state, monkeypatch):
    from colab_t4 import cli

    home = state / "accounts" / "pending"
    home.mkdir(parents=True)
    fifo = home / "oauth.stdin"
    fifo.write_text("")
    (state / "pending-auth.json").write_text(json.dumps({
        "account_id": "pending",
        "home": str(home),
        "fifo": str(fifo),
        "pid": 999999,
    }))
    monkeypatch.setattr(cli.os, "kill", lambda *_args: (_ for _ in ()).throw(ProcessLookupError()))

    assert cli._accounts_auth_cancel(argparse.Namespace(id=None)) == 0
    assert not (state / "pending-auth.json").exists()
    assert not home.exists()


def test_import_token_copies_and_masks_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    source = tmp_path / "token.json"
    source.write_text(json.dumps({"access_token": "abc", "token_type": "Bearer"}))
    assert import_token(source) is True
    target = tmp_path / "home" / ".config" / "colab-cli" / "token.json"
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert target.stat().st_mode & 0o777 == 0o600


def test_import_token_rejects_missing_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert import_token(tmp_path / "nope.json") is False
    assert not (tmp_path / "home" / ".config").exists()


def test_import_token_rejects_empty_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    source = tmp_path / "token.json"
    source.write_text("   ")
    assert import_token(source) is False


def test_import_token_targets_account_home(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("HOME", str(tmp_path / "real-home"))
    source = tmp_path / "token.json"
    source.write_text(json.dumps({"refresh_token": "r1"}))
    target_home = str(tmp_path / "acc-home")
    Path(target_home).mkdir()
    assert import_token(source, home=target_home) is True
    token = Path(target_home) / ".config" / "colab-cli" / "token.json"
    assert token.is_file()
    assert token.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_create_account_imports_token_without_echoing_secret(state):
    assert hasattr(accounts, "create_account"), "create_account helper is missing"
    response = accounts.create_account({
        "id": "web",
        "email": "web@example.com",
        "token_json": {
            "refresh_token": "refresh-secret",
            "client_id": "client-id",
            "client_secret": "client-secret",
        },
    })

    account = get_account("web")
    token_path = Path(account.home) / ".config" / "colab-cli" / "token.json"
    assert json.loads(token_path.read_text(encoding="utf-8")) == {
        "refresh_token": "refresh-secret",
        "client_id": "client-id",
        "client_secret": "client-secret",
    }
    assert token_path.stat().st_mode & 0o777 == 0o600
    rendered = json.dumps(response)
    assert "refresh-secret" not in rendered
    assert "client-secret" not in rendered
    assert response == {"account_id": "web", "email": "web@example.com", "status": "added"}


def test_create_account_rejects_non_string_id_and_email(state):
    with pytest.raises(ValueError, match="id must be a string"):
        accounts.create_account({"id": 123, "email": "web@example.com"})
    with pytest.raises(ValueError, match="email must be a string"):
        accounts.create_account({"id": "web", "email": ["web@example.com"]})


def test_start_oauth_keeps_pending_state_in_memory_without_codes(state, monkeypatch):
    assert hasattr(accounts, "start_oauth"), "start_oauth helper is missing"
    assert hasattr(accounts, "list_pending_oauth"), "list_pending_oauth helper is missing"
    monkeypatch.setattr(accounts, "_start_colab_oauth_process", lambda account_id, home: {
        "authorization_url": "https://accounts.google.com/o/oauth2/auth?client_id=test",
        "pid": 123,
        "fifo": str(Path(home) / "oauth.stdin"),
    })

    response = accounts.start_oauth({"id": "oauth"})

    assert response["account_id"] == "oauth"
    assert response["authorization_url"].startswith("https://accounts.google.com/")
    pending = accounts.list_pending_oauth()
    assert pending == [{"account_id": "oauth", "expires_at": response["expires_at"]}]
    assert "code" not in json.dumps(pending).lower()
    assert not (state / "pending-auth.json").exists()


def test_start_oauth_rejects_non_string_id(state):
    with pytest.raises(ValueError, match="id must be a string"):
        accounts.start_oauth({"id": 123})


def test_oauth_finish_deletes_transcript_even_if_code_was_written(state, monkeypatch):
    monkeypatch.setattr(accounts, "_start_colab_oauth_process", lambda account_id, home: {
        "authorization_url": "https://accounts.google.com/o/oauth2/auth?client_id=test",
        "pid": 123,
        "fifo": str(Path(home) / "oauth.stdin"),
        "log": str(Path(home) / "oauth.log"),
    })

    def fake_finish(pending, code):
        log = Path(pending["log"])
        log.write_text("authorization transcript with " + code, encoding="utf-8")
        log.chmod(0o600)

    monkeypatch.setattr(accounts, "_finish_colab_oauth_process", fake_finish)
    monkeypatch.setattr(accounts, "_verify_account_auth", lambda home: (True, "verified"))
    monkeypatch.setattr(accounts, "resolve_account_email", lambda home: "oauth@example.com")
    response = accounts.start_oauth({"id": "oauth"})
    home = state / "accounts" / "oauth"
    log = home / "oauth.log"

    assert accounts.finish_oauth({"id": "oauth", "code": "one-time-code"})["status"] == "added"

    assert response["account_id"] == "oauth"
    assert not log.exists()
    assert all("one-time-code" not in path.read_text(encoding="utf-8", errors="ignore") for path in home.rglob("*") if path.is_file())


def test_expired_oauth_prune_terminates_process_and_removes_log(state, monkeypatch):
    log = state / "accounts" / "oauth" / "oauth.log"

    def fake_start(account_id, home):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("transcript", encoding="utf-8")
        return {
            "authorization_url": "https://accounts.google.com/o/oauth2/auth?client_id=test",
            "pid": 456,
            "fifo": str(Path(home) / "oauth.stdin"),
            "log": str(log),
        }

    killed = []
    monkeypatch.setattr(accounts, "_OAUTH_TTL_SECONDS", -1)
    monkeypatch.setattr(accounts, "_start_colab_oauth_process", fake_start)
    monkeypatch.setattr(accounts.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    accounts.start_oauth({"id": "oauth"})

    assert accounts.list_pending_oauth() == []
    assert killed == [(456, accounts.signal.SIGTERM)]
    assert not log.exists()
    assert not (state / "accounts" / "oauth").exists()


def test_idle_expired_oauth_is_terminated_without_later_api_call(state, monkeypatch):
    terminated = threading.Event()

    monkeypatch.setattr(accounts, "_OAUTH_TTL_SECONDS", 0.01)
    monkeypatch.setattr(accounts, "_OAUTH_REAPER_INTERVAL_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(accounts, "_start_colab_oauth_process", lambda account_id, home: {
        "authorization_url": "https://accounts.google.com/o/oauth2/auth?client_id=test",
        "pid": 789,
        "fifo": str(Path(home) / "oauth.stdin"),
    })

    def fake_kill(pid, sig):
        if (pid, sig) == (789, accounts.signal.SIGTERM):
            terminated.set()

    monkeypatch.setattr(accounts.os, "kill", fake_kill)
    accounts.start_oauth({"id": "idle"})

    assert terminated.wait(1.0)


def test_cancel_oauth_validates_target_and_removes_pending_home(state, monkeypatch):
    assert hasattr(accounts, "start_oauth"), "start_oauth helper is missing"
    assert hasattr(accounts, "cancel_oauth"), "cancel_oauth helper is missing"
    monkeypatch.setattr(accounts, "_start_colab_oauth_process", lambda account_id, home: {
        "authorization_url": "https://accounts.google.com/o/oauth2/auth?client_id=test",
        "pid": 123,
        "fifo": str(Path(home) / "oauth.stdin"),
    })
    killed = []
    monkeypatch.setattr(accounts.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    accounts.start_oauth({"id": "oauth"})
    home = state / "accounts" / "oauth"

    with pytest.raises(ValueError, match="account id"):
        accounts.cancel_oauth("../oauth")
    response = accounts.cancel_oauth("oauth")

    assert response == {"account_id": "oauth", "status": "cancelled"}
    assert killed
    assert not home.exists()
    assert accounts.list_pending_oauth() == []


def test_cancel_oauth_deletes_temporary_log(state, monkeypatch):
    log = state / "accounts" / "oauth" / "oauth.log"

    def fake_start(account_id, home):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("authorization transcript", encoding="utf-8")
        return {
            "authorization_url": "https://accounts.google.com/o/oauth2/auth?client_id=test",
            "pid": 123,
            "fifo": str(Path(home) / "oauth.stdin"),
            "log": str(log),
        }

    monkeypatch.setattr(accounts, "_start_colab_oauth_process", fake_start)
    monkeypatch.setattr(accounts.os, "kill", lambda *_args: None)
    accounts.start_oauth({"id": "oauth"})

    accounts.cancel_oauth("oauth")

    assert not log.exists()
