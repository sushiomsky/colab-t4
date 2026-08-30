from types import SimpleNamespace

from colab_t4.accounts import Account
from colab_t4 import mcp_server


def test_api_info_exposes_endpoint_but_no_credentials(monkeypatch):
    monkeypatch.setattr(mcp_server, "load_state", lambda: {
        "runtime_state": "ready",
        "api_base": "http://100.64.0.2:8081/v1",
    })
    result = mcp_server.api_info()
    assert result == {
        "runtime_state": "ready",
        "api_base": "http://100.64.0.2:8081/v1",
        "model_alias": "local",
    }
    assert "api_key" not in result


def test_accounts_list_omits_home_and_oauth_details(monkeypatch):
    monkeypatch.setattr(mcp_server, "load_accounts", lambda: [
        Account(
            id="work",
            email="work@example.test",
            home="/secret/account/home",
            last_used_at="2026-08-30T10:00:00Z",
            last_ok="2026-08-30T10:00:00Z",
            last_error="quota rejected with sensitive diagnostics",
        )
    ])
    result = mcp_server.accounts_list()
    assert result == [{
        "id": "work",
        "email": "work@example.test",
        "status": "error",
        "last_used_at": "2026-08-30T10:00:00Z",
        "last_ok": "2026-08-30T10:00:00Z",
    }]
    assert "home" not in result[0]
    assert "last_error" not in result[0]


def test_model_switch_delegates(monkeypatch):
    calls = []
    expected = {"runtime_state": "ready", "model_repo": "example/new-GGUF", "quant": "Q5_K_M"}
    monkeypatch.setattr(
        mcp_server,
        "switch_model_runtime",
        lambda model, quant, timeout=3600: calls.append((model, quant, timeout)) or expected,
    )
    assert mcp_server.model_switch("example/new-GGUF", "Q5_K_M") == expected
    assert calls == [("example/new-GGUF", "Q5_K_M", 3600)]


def test_backend_start_uses_runtime_options_and_safe_state(monkeypatch):
    options = SimpleNamespace(model="example/new-GGUF", quant="Q4_K_M")
    monkeypatch.setattr(mcp_server, "_runtime_options", lambda model=None, quant=None: options)
    monkeypatch.setattr(mcp_server, "lifecycle_up", lambda received: {
        "runtime_state": "ready",
        "session": "colab-t4",
        "account": "work",
        "api_base": "http://100.64.0.2:8081/v1",
        "model_repo": received.model,
        "quant": received.quant,
        "internal_secret": "must-not-leak",
    })
    result = mcp_server.backend_start("example/new-GGUF", "Q4_K_M")
    assert result["runtime_state"] == "ready"
    assert result["model_repo"] == "example/new-GGUF"
    assert "internal_secret" not in result


def test_build_server_registers_expected_tools(monkeypatch):
    registered = []

    class FakeServer:
        def __init__(self, name):
            self.name = name

        def tool(self):
            def decorator(function):
                registered.append(function.__name__)
                return function
            return decorator

    monkeypatch.setattr(mcp_server, "_MCPServer", FakeServer)
    server = mcp_server.build_server()
    assert server.name == "colab-t4"
    assert registered == [
        "backend_status",
        "backend_start",
        "backend_stop",
        "backend_restart",
        "model_current",
        "model_switch",
        "api_info",
        "accounts_list",
    ]


def test_main_without_sdk_has_clear_install_hint(monkeypatch, capsys):
    monkeypatch.setattr(mcp_server, "_MCPServer", None)
    assert mcp_server.main() == 2
    error = capsys.readouterr().err
    assert "colab-t4[mcp]" in error
    assert "Python 3.10" in error
