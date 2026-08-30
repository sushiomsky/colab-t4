import json
import pytest
from colab_t4 import config, runtimes


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def test_default_runtime_uses_legacy_base(base):
    assert config.state_dir() == base
    with runtimes.runtime_context("default"):
        config.save_state({"runtime_state": "ready", "session": "legacy"})
    assert json.loads((base / "state.json").read_text())["session"] == "legacy"


def test_named_runtime_isolates_runtime_files_but_not_global_secrets(base):
    config.save_secrets({
        "tailscale_authkey": "ts-global",
        "api_key": "legacy-key",
        "model_repo": "legacy/model",
    })
    created = runtimes.create_runtime("coder", model_repo="example/coder-GGUF", quant="Q4_K_M")
    assert created["session"] == "colab-t4-coder"
    with runtimes.runtime_context("coder"):
        assert config.state_dir() == base / "runtimes" / "coder"
        assert config.secrets_path() == base / "secrets.json"
        assert config.logs_dir() == base / "runtimes" / "coder" / "logs"
        effective = config.load_secrets()
        assert effective["tailscale_authkey"] == "ts-global"
        assert effective["api_key"] != "legacy-key"
        assert "model_repo" not in effective
    assert (base / "runtimes" / "coder" / "runtime-api-key.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("name", ["", "default", "../evil", "has space", "/tmp/x", "a" * 33])
def test_create_rejects_reserved_or_unsafe_runtime_ids(base, name):
    with pytest.raises(ValueError):
        runtimes.create_runtime(name, model_repo="example/model", quant="Q4_K_M")


def test_runtime_context_restores_after_exception(base):
    runtimes.create_runtime("coder", model_repo="example/model", quant="Q4_K_M")
    with pytest.raises(RuntimeError):
        with runtimes.runtime_context("coder"):
            assert runtimes.selected_runtime() == "coder"
            raise RuntimeError("boom")
    assert runtimes.selected_runtime() == "default"


def test_inventory_tolerates_corrupt_named_state(base):
    runtimes.create_runtime("coder", model_repo="example/coder", quant="Q4_K_M")
    runtimes.create_runtime("broken", model_repo="example/broken", quant="Q4_K_M")
    (base / "runtimes" / "broken" / "state.json").write_text("{broken", encoding="utf-8")
    rows = {row["id"]: row for row in runtimes.runtime_inventory()}
    assert {"default", "coder", "broken"}.issubset(rows)
    assert rows["broken"]["runtime_state"] in {"invalid", "unknown"}
    assert "api_key" not in rows["coder"]
