"""Account registry and failover ordering tests."""
import json

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


def test_candidate_affinity_skips_failed_hosting_account(state):
    add_account("a", home=str(state / "ha"))
    add_account("b", home=str(state / "hb"))
    record_failure("a", "session died")
    state_data = {"session": "dead-t4", "account": "a"}
    ordered = candidate_accounts("dead-t4", state_data)
    assert ordered[0].id != "a"
    assert [a.id for a in ordered] == ["default", "b", "a"]


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
