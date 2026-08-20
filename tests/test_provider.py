import json

import pytest

from colab_t4.accounts import add_account
from colab_t4.provider import account_rows, resolve_model


def test_resolve_model_returns_reviewed_spec():
    spec = resolve_model("llama3.1-8b-ablit")
    assert spec.repository == "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF"
    assert spec.quant == "Q4_K_M"


def test_unknown_model_alias_is_rejected():
    with pytest.raises(ValueError, match="unknown model alias"):
        resolve_model("arbitrary-repository")


def test_account_rows_are_sanitized(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(state))
    add_account("work", home=str(tmp_path / "home"), email="work@example.com")

    rows = account_rows()

    assert {row["id"] for row in rows} == {"default", "work"}
    assert rows[1]["email"] == "work@example.com"
    assert all("token" not in json.dumps(row).lower() for row in rows)
    assert all("home" not in row for row in rows)
