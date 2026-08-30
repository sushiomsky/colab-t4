"""Tests for the local OpenAI-compatible router (Colab/Ollama backend dispatch)."""
import json
import socket
import threading
import time
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

import pytest

from colab_t4.router import Backend, Router, discover_backends, select_backend
from colab_t4.router import _handler_factory, _management_handler_factory


MANAGEMENT_TOKEN = "test-management-token"


@pytest.fixture(autouse=True)
def _management_token_env(monkeypatch):
    monkeypatch.setenv("COLAB_T4_MANAGEMENT_TOKEN", MANAGEMENT_TOKEN)


def _json_request(url, *, method="GET", payload=None, management_token=True, origin=None):
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    if method in {"POST", "PUT", "DELETE"} and management_token:
        headers["X-Colab-T4-Management-Token"] = MANAGEMENT_TOKEN
        split = urlsplit(url)
        headers["Origin"] = origin or f"{split.scheme}://{split.netloc}"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


def _json_error_body(exc):
    return json.loads(exc.value.read().decode())


def _management_headers(url):
    split = urlsplit(url)
    return {
        "X-Colab-T4-Management-Token": MANAGEMENT_TOKEN,
        "Origin": f"{split.scheme}://{split.netloc}",
    }


def _raw_http_request(port, request):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        client.sendall(request.encode())
        client.shutdown(socket.SHUT_WR)
        response = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            response += chunk
    return response


def _raw_response_status_and_headers(response):
    header_block = response.split(b"\r\n\r\n", 1)[0].decode("iso-8859-1")
    lines = header_block.split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.lower()] = value.strip()
    return status, headers


def _wait_for_job(url, job_id, expected_statuses={"succeeded", "failed", "cancelled"}):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, body = _json_request(f"{url}/api/jobs/{job_id}")
        if body["job"]["status"] in expected_statuses:
            return body["job"]
        time.sleep(0.01)
    pytest.fail(f"job {job_id} did not reach {expected_statuses}")


def test_proxy_rejects_request_bodies_over_limit():
    """Dropping the proxy request-size guard forwards oversized model payloads."""
    forwarded = False

    class SpyRouter:
        def select(self):
            return None

        def forward(self, *_args):
            nonlocal forwarded
            forwarded = True
            return 200, {"Content-Type": "application/json"}, b"{}"

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(SpyRouter()))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = (
            "POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {1024 * 1024 + 1}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        response = _raw_http_request(port, request)
        status, _headers = _raw_response_status_and_headers(response)
        assert status == 413
        assert b'{"error": "request body too large"}' in response
        assert forwarded is False
    finally:
        server.shutdown()


def test_proxy_rejects_delete_method():
    """Allowing DELETE would expose upstream verbs outside the OpenAI router contract."""
    forwarded = False

    class SpyRouter:
        def select(self):
            return None

        def forward(self, *_args):
            nonlocal forwarded
            forwarded = True
            return 200, {"Content-Type": "application/json"}, b"{}"

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(SpyRouter()))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", method="DELETE")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 405
        assert forwarded is False
    finally:
        server.shutdown()


def test_discover_backends_prioritizes_colab(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_BASE", raising=False)
    from colab_t4.config import save_state, save_secrets, save_runtime_api_key
    save_state({"api_base": "http://100.64.0.2:8080/v1"})
    save_secrets({"api_key": "secret-key"})
    backends = discover_backends()
    assert backends[0].name == "colab"
    assert backends[0].base_url == "http://100.64.0.2:8080/v1"
    assert backends[0].api_key == "secret-key"
    assert backends[1].name == "ollama"
    assert backends[1].base_url == "http://127.0.0.1:11434"


def test_discover_backends_without_colab_state(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_BASE", raising=False)
    backends = discover_backends()
    assert len(backends) == 1
    assert backends[0].name == "ollama"


def test_discover_backends_uses_ollama_env(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11435")
    backends = discover_backends()
    assert backends[0].name == "ollama"
    assert backends[0].base_url == "http://127.0.0.1:11435"


def test_select_backend_picks_first_healthy():
    backends = [
        Backend(name="colab", base_url="http://colab:8080/v1", api_key="k"),
        Backend(name="ollama", base_url="http://127.0.0.1:11434"),
    ]
    with patch("colab_t4.router._probe_backend") as mock_probe:
        mock_probe.side_effect = [False, True]
        selected = select_backend(backends)
        assert selected is not None
        assert selected.name == "ollama"


def test_select_backend_returns_none_when_all_down():
    backends = [
        Backend(name="colab", base_url="http://colab:8080/v1", api_key="k"),
        Backend(name="ollama", base_url="http://127.0.0.1:11434"),
    ]
    with patch("colab_t4.router._probe_backend", return_value=False):
        assert select_backend(backends) is None


def test_router_forward_returns_503_when_no_backend():
    router = Router(backends=[])
    status, headers, body = router.forward("POST", "/v1/chat/completions", b"{}", {})
    assert status == 503
    assert json.loads(body)["error"]["type"] == "router_error"


def test_router_forward_proxies_to_healthy_backend():
    backend = Backend(name="ollama", base_url="http://127.0.0.1:11434")
    router = Router(backends=[backend])
    captured = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["data"] = req.data
        captured["headers"] = dict(req.headers)
        return FakeResponse()

    with patch("colab_t4.router._probe_backend", return_value=True):
        with patch("colab_t4.router.urllib.request.urlopen", side_effect=fake_urlopen):
            status, headers, body = router.forward(
                "POST", "/v1/chat/completions",
                b'{"model":"local"}',
                {"Content-Type": "application/json"},
            )
    assert status == 200
    assert captured["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["data"] == b'{"model":"local"}'
    assert json.loads(body) == {"ok": True}


def test_router_forward_injects_bearer_key():
    backend = Backend(name="colab", base_url="http://colab:8080/v1", api_key="secret-key")
    router = Router(backends=[backend])
    captured = {}

    class FakeResponse:
        status = 200
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b'{}'

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.headers)
        return FakeResponse()

    with patch("colab_t4.router._probe_backend", return_value=True):
        with patch("colab_t4.router.urllib.request.urlopen", side_effect=fake_urlopen):
            router.forward("GET", "/v1/models", None, {})
    assert captured["headers"]["Authorization"] == "Bearer secret-key"


def test_router_forward_strips_hop_by_hop_headers_and_connection_tokens():
    """Forwarding hop-by-hop headers leaks transport metadata to model backends."""
    backend = Backend(name="ollama", base_url="http://127.0.0.1:11434")
    router = Router(backends=[backend])
    captured = {}

    class FakeResponse:
        status = 200
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout=None):
        captured["headers"] = {key.lower(): value for key, value in dict(req.headers).items()}
        return FakeResponse()

    headers = {
        "Host": "client.example",
        "Content-Length": "2",
        "Connection": "keep-alive, X-Debug-Hop",
        "Keep-Alive": "timeout=5",
        "Proxy-Authenticate": "Basic realm=test",
        "Proxy-Authorization": "Basic abc",
        "TE": "trailers",
        "Trailer": "Expires",
        "Transfer-Encoding": "chunked",
        "Upgrade": "websocket",
        "X-Debug-Hop": "remove-me",
        "X-Trace": "keep-me",
    }
    with patch("colab_t4.router._probe_backend", return_value=True):
        with patch("colab_t4.router.urllib.request.urlopen", side_effect=fake_urlopen):
            router.forward("POST", "/v1/chat/completions", b"{}", headers)

    assert captured["headers"]["X-trace".lower()] == "keep-me"
    for name in (
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-debug-hop",
    ):
        assert name not in captured["headers"]


def test_router_health_endpoint(tmp_path, monkeypatch):
    """The /health endpoint returns 200 with backend name when Colab is healthy."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.config import save_state, save_secrets
    save_state({"api_base": "http://colab:8080/v1"})
    save_secrets({"api_key": "k"})

    from colab_t4.router import _handler_factory, Router
    router = Router()
    handler = _handler_factory(router)

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with patch("colab_t4.router._probe_backend", return_value=True):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                body = json.loads(resp.read().decode())
                assert resp.status == 200
    finally:
        server.shutdown()
    assert body["status"] == "ok"
    assert body["backend"] == "colab"


def test_router_health_degraded_when_no_backend(tmp_path, monkeypatch):
    """The /health endpoint returns 503 when no backend is healthy."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))

    from colab_t4.router import _handler_factory, Router
    router = Router()
    handler = _handler_factory(router)

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with patch("colab_t4.router._probe_backend", return_value=False):
            with pytest.raises(urllib.request.HTTPError) as exc:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            assert exc.value.code == 503
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Management UI tests
# ---------------------------------------------------------------------------


def test_manage_ui_served(tmp_path, monkeypatch):
    """GET /manage returns the HTML management UI."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/manage", timeout=5) as resp:
            body = resp.read().decode()
            assert resp.status == 200
            assert "text/html" in resp.headers["Content-Type"]
            assert "colab-t4 Manager" in body
            assert "Dashboard" in body
            assert "Accounts" in body
            assert "Configuration" in body
    finally:
        server.shutdown()


def test_api_state_endpoint(tmp_path, monkeypatch):
    """GET /api/state returns JSON with runtime, accounts, and backends."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with patch("colab_t4.router._probe_backend", return_value=True):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=5) as resp:
                body = json.loads(resp.read().decode())
                assert resp.status == 200
                assert "runtime_state" in body
                assert "accounts" in body
                assert "backends" in body
                assert "secrets_configured" in body
                assert isinstance(body["accounts"], list)
                assert isinstance(body["backends"], list)
                assert body["proxy"]["health_url"].endswith("/health")
                assert body["proxy"]["api_url"].endswith("/v1")
                assert "ready" in body["runtime"]
                assert "updated_at" in body["runtime"]
    finally:
        server.shutdown()


def test_api_state_redacts_state_and_account_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.config import save_secrets, save_state
    from colab_t4.accounts import add_account, record_failure

    save_secrets({"api_key": "api-secret", "tailscale_authkey": "ts-secret"})
    save_state({"runtime_state": "failed", "last_error": "failed with api-secret"})
    add_account("work", home=str(tmp_path / "home"))
    record_failure("work", "quota rejected ts-secret")

    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with patch("colab_t4.router._probe_backend", return_value=True):
            status, body = _json_request(f"http://127.0.0.1:{port}/api/state")
        rendered = json.dumps(body)
        assert status == 200
        assert "api-secret" not in rendered
        assert "ts-secret" not in rendered
        assert body["last_error"] == "failed with [REDACTED]"
        work = next(account for account in body["accounts"] if account["id"] == "work")
        assert work["last_error"] == "quota rejected [REDACTED]"
    finally:
        server.shutdown()


def test_manage_ui_includes_live_dashboard_controls(tmp_path, monkeypatch):
    """The management page exposes proxy details and actionable status UI."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/manage", timeout=5) as resp:
            body = resp.read().decode()
            assert "Proxy endpoint" in body
            assert "Last updated" in body
            assert "action-status" in body
            assert "Retry" in body
    finally:
        server.shutdown()


def test_api_accounts_delete(tmp_path, monkeypatch):
    """DELETE /api/accounts/<id> removes the account."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.accounts import account_home_dir, add_account, load_accounts
    home = account_home_dir("test-account")
    home.mkdir(parents=True)
    add_account("test-account", home=str(home))
    assert any(a.id == "test-account" for a in load_accounts())

    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/api/accounts/test-account"
        req = urllib.request.Request(
            url,
            headers=_management_headers(url),
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode())
            assert resp.status == 200
            assert body["status"] == "removed"
            assert body["id"] == "test-account"
    finally:
        server.shutdown()
    assert not any(a.id == "test-account" for a in load_accounts())


def test_management_api_404_for_unknown_route(tmp_path, monkeypatch):
    """Unknown /api/ routes return 404."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(urllib.request.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/unknown", timeout=5)
        assert exc.value.code == 404
    finally:
        server.shutdown()


def test_management_mutations_require_token_and_same_origin(tmp_path, monkeypatch):
    """Removing token/origin checks lets a browser issue lifecycle mutations cross-site."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as missing_token:
            _json_request(base_url + "/api/down", method="POST", management_token=False)
        assert missing_token.value.code == 403
        assert _json_error_body(missing_token)["error"] == "management token required"

        with pytest.raises(urllib.error.HTTPError) as bad_origin:
            _json_request(base_url + "/api/down", method="POST", origin="http://evil.example")
        assert bad_origin.value.code == 403
        assert _json_error_body(bad_origin)["error"] == "management origin rejected"
    finally:
        server.shutdown()


def test_management_rejects_oversized_json_body(tmp_path, monkeypatch):
    """Removing the management body cap lets large browser payloads reach handlers."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        request = (
            "PUT /api/config HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Origin: {base_url}\r\n"
            f"X-Colab-T4-Management-Token: {MANAGEMENT_TOKEN}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {1024 * 1024 + 1}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        response = _raw_http_request(port, request)
        assert b" 413 " in response.split(b"\r\n", 1)[0]
        assert b'{"error": "request body too large"}' in response
    finally:
        server.shutdown()


def test_management_rejects_oversized_bodyless_post_before_lifecycle_route(tmp_path, monkeypatch):
    """A bodyless lifecycle mutation with huge Content-Length must fail before job creation."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        request = (
            "POST /api/down HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Origin: {base_url}\r\n"
            f"X-Colab-T4-Management-Token: {MANAGEMENT_TOKEN}\r\n"
            f"Content-Length: {1024 * 1024 + 1}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        response = _raw_http_request(port, request)
        status, _headers = _raw_response_status_and_headers(response)
        assert status == 413
        assert b'{"error": "request body too large"}' in response
        assert router.jobs.list() == []
    finally:
        server.shutdown()


def test_management_rejects_oversized_job_cancel_before_route_lookup(tmp_path, monkeypatch):
    """Oversized job cancellation declarations must not reach job-id lookup."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        request = (
            "POST /api/jobs/missing/cancel HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Origin: {base_url}\r\n"
            f"X-Colab-T4-Management-Token: {MANAGEMENT_TOKEN}\r\n"
            f"Content-Length: {1024 * 1024 + 1}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        response = _raw_http_request(port, request)
        status, _headers = _raw_response_status_and_headers(response)
        assert status == 413
        assert b'{"error": "request body too large"}' in response
    finally:
        server.shutdown()


def test_management_known_route_rejects_wrong_method(tmp_path, monkeypatch):
    """Known management actions must reject unsupported verbs instead of falling through."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/down", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 405
        assert json.loads(exc.value.read().decode())["error"] == "method not allowed"
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    ("method", "path", "allow"),
    (
        ("HEAD", "/api/down", "POST"),
        ("OPTIONS", "/api/config", "GET, PUT"),
        ("PATCH", "/api/jobs/example/cancel", "POST"),
        ("HEAD", "/api/accounts/work", "DELETE"),
    ),
)
def test_management_known_routes_reject_unsupported_methods_with_accurate_allow(tmp_path, monkeypatch, method, path, allow):
    """Route method metadata must not advertise unsupported management verbs."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        response = _raw_http_request(port, request)
        status, headers = _raw_response_status_and_headers(response)
        assert status == 405
        assert headers["allow"] == allow
    finally:
        server.shutdown()


def test_management_api_has_no_arbitrary_shell_route(tmp_path, monkeypatch):
    """Adding a shell-style management route would create remote command execution."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for path in ("/api/shell", "/api/exec", "/api/command"):
            with pytest.raises(urllib.error.HTTPError) as exc:
                _json_request(f"http://127.0.0.1:{port}{path}", method="POST", payload={"cmd": "id"})
            assert exc.value.code == 404
    finally:
        server.shutdown()


def test_management_down_routes_to_lifecycle(tmp_path, monkeypatch):
    """POST /api/down returns a pollable job backed by the shared service."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.config import save_state
    save_state({"session": "test-session", "api_base": "http://colab:8080/v1"})

    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with patch("colab_t4.services.runtime_down", return_value={"runtime_state": "stopped"}):
            status, body = _json_request(f"http://127.0.0.1:{port}/api/down", method="POST")
            assert status == 202
            assert set(body) == {"job"}
            job = _wait_for_job(f"http://127.0.0.1:{port}", body["job"]["id"])
            assert job["status"] == "succeeded"
            assert job["result"] == {"runtime_state": "stopped"}
    finally:
        server.shutdown()


def test_management_down_error_response_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.config import save_secrets

    save_secrets({"api_key": "api-secret"})
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with patch("colab_t4.services.runtime_down", side_effect=RuntimeError("api-secret failed")):
            status, body = _json_request(f"http://127.0.0.1:{port}/api/down", method="POST")
        assert status == 202
        job = _wait_for_job(f"http://127.0.0.1:{port}", body["job"]["id"])
        assert job["status"] == "failed"
        assert job["error"] == "[REDACTED] failed"
    finally:
        server.shutdown()


def test_proxy_error_response_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.config import save_secrets

    save_secrets({"api_key": "api-secret"})

    class BrokenRouter:
        def select(self):
            return None

        def forward(self, *_args):
            raise RuntimeError("proxy leaked api-secret")

    handler = _handler_factory(BrokenRouter())
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        body = _json_error_body(exc)
        assert exc.value.code == 502
        assert body["error"]["message"] == "proxy leaked [REDACTED]"
    finally:
        server.shutdown()


def test_api_config_update_and_reset_never_return_secret_values(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, body = _json_request(
            f"http://127.0.0.1:{port}/api/config",
            method="PUT",
            payload={"session": "web-session", "api_key": "api-secret"},
        )
        assert status == 200
        assert body["session"] == "web-session"
        assert body["secrets"]["api_key"] is True
        assert "api-secret" not in json.dumps(body)

        status, body = _json_request(f"http://127.0.0.1:{port}/api/config")
        assert status == 200
        assert body["session"] == "web-session"
        assert "api-secret" not in json.dumps(body)

        with pytest.raises(urllib.error.HTTPError) as exc:
            _json_request(f"http://127.0.0.1:{port}/api/config/reset", method="POST", payload={"confirmed": False})
        assert exc.value.code == 400

        status, body = _json_request(
            f"http://127.0.0.1:{port}/api/config/reset",
            method="POST",
            payload={"confirmed": True},
        )
        assert status == 200
        assert body["secrets"]["api_key"] is False
    finally:
        server.shutdown()


def test_api_accounts_import_token_response_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, body = _json_request(
            f"http://127.0.0.1:{port}/api/accounts",
            method="POST",
            payload={
                "id": "imported",
                "email": "imported@example.com",
                "token_json": {"refresh_token": "refresh-secret", "client_secret": "client-secret"},
            },
        )
        assert status == 200
        assert body == {"account_id": "imported", "email": "imported@example.com", "status": "added"}
        assert "refresh-secret" not in json.dumps(body)
        assert "client-secret" not in json.dumps(body)
    finally:
        server.shutdown()


def test_api_accounts_rejects_non_string_id_and_email(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        for payload, message in (
            ({"id": 123, "email": "typed@example.com"}, "id must be a string"),
            ({"id": "typed", "email": ["typed@example.com"]}, "email must be a string"),
        ):
            with pytest.raises(urllib.error.HTTPError) as exc:
                _json_request(f"http://127.0.0.1:{port}/api/accounts", method="POST", payload=payload)
            assert exc.value.code == 400
            assert _json_error_body(exc)["error"] == message
    finally:
        server.shutdown()


def test_api_accounts_oauth_start_finish_cancel_keep_code_out_of_responses(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4 import accounts

    finished = []
    monkeypatch.setattr(accounts, "_start_colab_oauth_process", lambda account_id, home: {
        "authorization_url": "https://accounts.google.com/o/oauth2/auth?client_id=test",
        "pid": 123,
        "fifo": str(tmp_path / "oauth.stdin"),
    })
    monkeypatch.setattr(accounts, "_finish_colab_oauth_process", lambda pending, code: finished.append(code))
    monkeypatch.setattr(accounts, "_verify_account_auth", lambda home: (True, "verified"))
    monkeypatch.setattr(accounts, "resolve_account_email", lambda home: "oauth@example.com")

    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        status, body = _json_request(
            f"http://127.0.0.1:{port}/api/accounts/auth-start",
            method="POST",
            payload={"id": "oauth"},
        )
        assert status == 200
        assert body["account_id"] == "oauth"
        assert body["authorization_url"].startswith("https://accounts.google.com/")
        assert "code" not in json.dumps(body).lower()

        status, body = _json_request(
            f"http://127.0.0.1:{port}/api/accounts/auth-finish",
            method="POST",
            payload={"id": "oauth", "code": "one-time-code"},
        )
        assert status == 200
        assert body == {"account_id": "oauth", "email": "oauth@example.com", "status": "added"}
        assert "one-time-code" not in json.dumps(body)
        assert finished == ["one-time-code"]
    finally:
        server.shutdown()


def test_api_accounts_auth_cancel_rejects_non_string_id(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _json_request(f"http://127.0.0.1:{port}/api/accounts/auth-cancel", method="POST", payload={"id": 123})
        assert exc.value.code == 400
        assert _json_error_body(exc)["error"] == "id must be a string"
    finally:
        server.shutdown()


def test_api_accounts_delete_rejects_default_and_path_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        for target in ("default", "..%2Fevil"):
            url = f"http://127.0.0.1:{port}/api/accounts/{target}"
            req = urllib.request.Request(
                url,
                headers=_management_headers(url),
                method="DELETE",
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req, timeout=5)
            assert exc.value.code == 400
    finally:
        server.shutdown()


def test_api_accounts_delete_rejects_tampered_home_without_deleting_outside_path(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.accounts import add_account, get_account

    outside = tmp_path / "outside-home"
    outside.mkdir()
    add_account("tampered", home=str(outside), email="tampered@example.com")
    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/api/accounts/tampered"
        req = urllib.request.Request(
            url,
            headers=_management_headers(url),
            method="DELETE",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 400
        assert "outside account directory" in _json_error_body(exc)["error"]
    finally:
        server.shutdown()
    assert outside.exists()
    assert get_account("tampered").home == str(outside)


def test_api_accounts_delete_rejects_symlinked_account_home_without_deleting_target(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.accounts import add_account, get_account

    outside = tmp_path / "outside-home"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("do not delete", encoding="utf-8")
    link = tmp_path / "state" / "accounts" / "linked"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    add_account("linked", home=str(link), email="linked@example.com")

    router = Router(backends=[Backend(name="ollama", base_url="http://127.0.0.1:11434")])
    handler = _management_handler_factory(router)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/api/accounts/linked"
        req = urllib.request.Request(
            url,
            headers=_management_headers(url),
            method="DELETE",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 400
        assert "outside account directory" in _json_error_body(exc)["error"]
    finally:
        server.shutdown()
    assert sentinel.read_text(encoding="utf-8") == "do not delete"
    assert get_account("linked").home == str(link)


@pytest.mark.parametrize(
    ("path", "service_name"),
    (("/api/up", "runtime_up"), ("/api/restart", "runtime_restart")),
)
def test_lifecycle_routes_return_redacted_progress_jobs(tmp_path, monkeypatch, path, service_name):
    """Lifecycle requests are accepted then expose service progress by job ID."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4 import services
    from colab_t4.config import save_secrets

    save_secrets({"api_key": "api-secret"})

    def operation(_options, progress, cancelled=None):
        progress("connecting with api-secret", 100)
        return {"detail": "ready with api-secret"}

    monkeypatch.setattr(services, service_name, operation)
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        status, body = _json_request(base_url + path, method="POST")
        assert status == 202
        assert set(body) == {"job"}
        assert body["job"]["id"]
        job = _wait_for_job(base_url, body["job"]["id"])
        rendered = json.dumps(job)
        assert job["status"] == "succeeded"
        assert job["progress"] == {"message": "connecting with [REDACTED]", "percent": 100}
        assert job["result"] == {"detail": "ready with [REDACTED]"}
        assert "api-secret" not in rendered
    finally:
        server.shutdown()


def test_api_jobs_lists_polls_cancels_and_rejects_unknown_ids(tmp_path, monkeypatch):
    """Job endpoints expose lifecycle records and reject IDs outside the registry."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4 import services

    started = threading.Event()
    release = threading.Event()

    def blocking_up(_options, progress, cancelled=None):
        progress("waiting", 20)
        started.set()
        release.wait(5)
        if cancelled():
            return {"runtime_state": "cancelled"}
        return {"runtime_state": "ready"}

    monkeypatch.setattr(services, "runtime_up", blocking_up)
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        _, created = _json_request(base_url + "/api/up", method="POST")
        job_id = created["job"]["id"]
        assert started.wait(2)

        _, listed = _json_request(base_url + "/api/jobs")
        assert any(job["id"] == job_id for job in listed["jobs"])
        _, polled = _json_request(base_url + f"/api/jobs/{job_id}")
        assert polled["job"]["progress"] == {"message": "waiting", "percent": 20}

        status, cancelled = _json_request(base_url + f"/api/jobs/{job_id}/cancel", method="POST")
        assert status == 202
        assert cancelled["job"]["id"] == job_id
        release.set()
        assert _wait_for_job(base_url, job_id)["status"] == "cancelled"

        for unknown_path, method in (("/api/jobs/missing", "GET"), ("/api/jobs/missing/cancel", "POST")):
            with pytest.raises(urllib.error.HTTPError) as exc:
                _json_request(base_url + unknown_path, method=method)
            assert exc.value.code == 404
    finally:
        release.set()
        server.shutdown()


def test_diagnostics_and_api_smoke_routes_redact_and_bound_output(tmp_path, monkeypatch):
    """Diagnostic, model, and chat payloads redact secrets and logs are bounded."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4 import services
    from colab_t4.config import logs_dir, save_secrets

    save_secrets({"api_key": "api-secret"})
    (logs_dir() / "runtime.log").write_text("x" * 20000 + " api-secret", encoding="utf-8")
    monkeypatch.setattr(services, "runtime_status", lambda: {"last_error": "api-secret status"})
    monkeypatch.setattr(services, "run_doctor", lambda: ({"detail": "api-secret doctor"}, 1))
    monkeypatch.setattr(services, "api_models", lambda: {"data": [{"id": "api-secret-model"}]})
    monkeypatch.setattr(services, "api_chat", lambda message: {"reply": f"{message}: api-secret"})

    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        _, status_body = _json_request(base_url + "/api/status")
        _, doctor_body = _json_request(base_url + "/api/doctor")
        _, logs_body = _json_request(base_url + "/api/logs")
        _, models_body = _json_request(base_url + "/api/api/models")
        _, chat_body = _json_request(base_url + "/api/api/chat", method="POST", payload={"message": "smoke"})

        rendered = json.dumps([status_body, doctor_body, logs_body, models_body, chat_body])
        assert status_body == {"status": {"last_error": "[REDACTED] status"}}
        assert doctor_body == {"doctor": {"detail": "[REDACTED] doctor"}, "exit_code": 1}
        assert models_body == {"models": {"data": [{"id": "[REDACTED]-model"}]}}
        assert chat_body == {"chat": {"reply": "smoke: [REDACTED]"}}
        assert len(logs_body["logs"]) == 1
        assert logs_body["logs"][0]["name"] == "runtime.log"
        assert logs_body["logs"][0]["truncated"] is True
        assert len(logs_body["logs"][0]["content"]) <= 16384
        assert "api-secret" not in rendered
        assert "[REDACTED]" in logs_body["logs"][0]["content"]
    finally:
        server.shutdown()


def test_api_down_cancellation_reaches_service_and_does_not_succeed(tmp_path, monkeypatch):
    """Cancelling a down job reaches the service callback and leaves no success result."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4 import services

    started = threading.Event()
    observed_cancellation = threading.Event()

    def cancellable_down(cancelled=None):
        started.set()
        while not cancelled():
            time.sleep(0.005)
        observed_cancellation.set()
        return {"runtime_state": "stopped"}

    monkeypatch.setattr(services, "runtime_down", cancellable_down)
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        _, created = _json_request(base_url + "/api/down", method="POST")
        job_id = created["job"]["id"]
        assert started.wait(2)
        _json_request(base_url + f"/api/jobs/{job_id}/cancel", method="POST")
        assert observed_cancellation.wait(2)
        job = _wait_for_job(base_url, job_id)
        assert job["status"] == "cancelled"
        assert job["result"] is None
    finally:
        server.shutdown()


def test_api_logs_rejects_symlinked_log_root(tmp_path, monkeypatch):
    """A symlinked logs directory must not expose files outside the state directory."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.config import logs_dir

    log_root = logs_dir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.log").write_text("outside-secret", encoding="utf-8")
    log_root.rmdir()
    log_root.symlink_to(outside, target_is_directory=True)

    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, body = _json_request(f"http://127.0.0.1:{port}/api/logs")
        assert body == {"logs": []}
        assert "outside-secret" not in json.dumps(body)
    finally:
        server.shutdown()


def test_api_logs_do_not_follow_file_swaps_after_validation(tmp_path, monkeypatch):
    """Replacing a listed log with a symlink cannot disclose its new target."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.config import logs_dir

    log_path = logs_dir() / "runtime.log"
    secret_path = tmp_path / "secret.txt"
    log_path.write_text("safe log", encoding="utf-8")
    secret_path.write_text("swap-secret", encoding="utf-8")
    original_open = Path.open

    def swap_before_open(path, *args, **kwargs):
        if path == log_path:
            log_path.unlink()
            log_path.symlink_to(secret_path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_before_open)
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, body = _json_request(f"http://127.0.0.1:{port}/api/logs")
        assert body == {"logs": [{"name": "runtime.log", "content": "safe log", "truncated": False}]}
        assert "swap-secret" not in json.dumps(body)
    finally:
        server.shutdown()


def test_manage_ui_lifecycle_handlers_poll_and_render_job_results(tmp_path, monkeypatch):
    """The bundled UI uses lifecycle job envelopes and renders terminal progress."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/manage", timeout=5) as response:
            body = response.read().decode()
        assert 'id="job-status"' in body
        assert "function renderJob(job)" in body
        assert "async function pollJob(jobId)" in body
        assert "data.job" in body
        assert "'/api/jobs/' + encodeURIComponent(jobId)" in body
        assert "Job failed" in body
    finally:
        server.shutdown()


def test_manage_ui_exposes_cli_parity_tabs_and_panels(tmp_path, monkeypatch):
    """Removing any CLI-parity management panel breaks the /manage contract."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/manage", timeout=5) as response:
            body = response.read().decode()
        for tab in ("Overview", "Runtime", "Accounts", "Configuration", "Diagnostics", "API Tests", "Logs", "Jobs"):
            assert f'>{tab}</button>' in body
        for panel_id in (
            "overview-panel",
            "runtime-panel",
            "accounts-panel",
            "configuration-panel",
            "diagnostics-panel",
            "api-tests-panel",
            "logs-panel",
            "jobs-panel",
        ):
            assert f'id="{panel_id}"' in body
    finally:
        server.shutdown()


def test_manage_ui_includes_secure_write_only_secrets_and_oauth_controls(tmp_path, monkeypatch):
    """Secret values stay write-only while account OAuth can be controlled from the UI."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    from colab_t4.config import save_secrets

    save_secrets({
        "api_key": "api-secret",
        "tailscale_authkey": "ts-secret",
        "hf_token": "hf-secret",
        "ssh_password": "ssh-secret",
        "ssh_pubkey": "ssh-rsa secret",
    })
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/manage", timeout=5) as response:
            body = response.read().decode()
        for field in ("tailscale_authkey", "hf_token", "api_key", "ssh_password"):
            assert f'name="{field}"' in body
            assert f'id="config-{field}"' in body
            assert 'placeholder="Leave unchanged"' in body
        assert 'name="ssh_pubkey"' in body
        assert 'id="config-ssh_pubkey"' in body
        assert 'id="oauth-start"' in body
        assert 'id="oauth-finish"' in body
        assert 'id="oauth-cancel"' in body
        assert 'id="oauth-code"' in body
        assert "api-secret" not in body
        assert "ts-secret" not in body
        assert "hf-secret" not in body
        assert "ssh-secret" not in body
    finally:
        server.shutdown()


def test_manage_ui_contract_includes_diagnostics_api_logs_jobs_and_safe_rendering(tmp_path, monkeypatch):
    """The bundled script wires API panels and avoids HTML injection for server data."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/manage", timeout=5) as response:
            body = response.read().decode()
        for helper in (
            "async function loadState()",
            "async function submitJson(",
            "async function pollJob(jobId)",
            "function renderJob(job)",
            "function showAlert(",
        ):
            assert helper in body
        for endpoint in (
            "/api/status",
            "/api/doctor",
            "/api/api/models",
            "/api/api/chat",
            "/api/logs",
            "/api/jobs",
            "/api/config",
            "/api/accounts/auth-start",
            "/api/accounts/auth-finish",
            "/api/accounts/auth-cancel",
        ):
            assert endpoint in body
        assert 'id="api-chat-message"' in body
        assert 'id="diagnostics-output"' in body
        assert 'id="logs-list"' in body
        assert 'id="jobs-list"' in body
        assert "confirm(" in body
        assert "setBusy(" in body
        assert "clearErrors()" in body
        assert ".textContent =" in body
        assert ".innerHTML" not in body
    finally:
        server.shutdown()


def test_manage_ui_lifecycle_actions_reserve_busy_state_before_post(tmp_path, monkeypatch):
    """Lifecycle actions disable controls before POST so rapid clicks cannot duplicate jobs."""
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    router = Router(backends=[])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _management_handler_factory(router))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/manage", timeout=5) as response:
            body = response.read().decode()
        runtime_start = body.index("async function runtimeAction(url, message)")
        runtime_fetch = body.index("fetch(url", runtime_start)
        runtime_busy = body.index("setBusy(true);", runtime_start)
        poll_start = body.index("async function pollJobWithOptions(jobId")
        poll_end = body.index("async function runtimeAction", poll_start)
        poll_body = body[poll_start:poll_end]

        assert runtime_busy < runtime_fetch
        assert "async function pollJob(jobId)" in body
        assert "pollJobWithOptions(data.job.id, {reserved: true})" in body
        assert "finally {" in poll_body
        assert "activeJobId = null;" in poll_body
        assert "setBusy(false);" in poll_body
    finally:
        server.shutdown()
