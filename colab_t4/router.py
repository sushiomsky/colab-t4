"""Local OpenAI-compatible router that dispatches to a Colab backend or Ollama.

The router exposes a minimal OpenAI-compatible HTTP API on localhost. Incoming
requests are forwarded, unmodified, to one of two backends:

- **Colab backend**: the provisioned T4 runtime. Its API base URL and key are
  read from the recorded runtime state (``state.json``). The Colab API is
  preferred when a runtime is ready.
- **Ollama backend**: a local Ollama instance (default ``http://127.0.0.1:11434``
  exposed as an OpenAI-compatible provider. Used when no Colab runtime is
  available or when the Colab API is unhealthy.

Routing decisions are made per-request with a bounded health probe. If the
preferred backend is unhealthy, the next is tried. This lets bots and tools
keep serving requests even when a Colab T4 is stopped or evicted.

The router also hosts a minimal web management UI (``colab-t4 serve``) that
exposes account, runtime, and configuration management through a single-page
interface.
"""
from __future__ import annotations

import json
import os
import secrets
import stat
import threading
import time
import urllib.error
import urllib.request
from hmac import compare_digest
from urllib.parse import unquote
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from . import services
from .config import configuration_summary, load_secrets, load_state, logs_dir, redact, reset_configuration, save_state, update_configuration
from .accounts import load_accounts
from .jobs import JobManager


_SECRET_FIELD_NAMES = {
    "tailscale_authkey",
    "hf_token",
    "api_key",
    "ssh_password",
    "ssh_pubkey",
}

MAX_REQUEST_BODY_BYTES = 1024 * 1024
MANAGEMENT_TOKEN_HEADER = "X-Colab-T4-Management-Token"
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class _RequestRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _secret_values() -> list[str]:
    """Return configured secret values without treating normal settings as secrets."""
    return [str(value) for key, value in load_secrets().items() if key in _SECRET_FIELD_NAMES]

# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    """A named upstream with a base URL and optional bearer token."""

    name: str
    base_url: str
    api_key: str = ""
    healthy: bool = True


def discover_backends() -> list[Backend]:
    """Build the ordered backend list from state and environment.

    Colab (from recorded runtime state) takes priority when available and
    recorded as healthy. Ollama (from ``OLLAMA_HOST`` or the default localhost
    port) follows. The Colab backend is omitted when no runtime is recorded.
    """
    backends: list[Backend] = []
    state = load_state()
    api_base = state.get("api_base")
    if api_base:
        secrets = load_secrets()
        key = secrets.get("api_key", "")
        backends.append(Backend(name="colab", base_url=api_base, api_key=key, healthy=True))
    ollama_url = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE")
    if ollama_url:
        backends.append(Backend(name="ollama", base_url=ollama_url.rstrip("/"), healthy=True))
    else:
        backends.append(Backend(name="ollama", base_url="http://127.0.0.1:11434", healthy=True))
    return backends


def _probe_backend(backend: Backend) -> bool:
    """Return True when the backend's models endpoint responds with data."""
    url = backend.base_url.rstrip("/") + "/v1/models"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    if backend.api_key:
        req.add_header("Authorization", "Bearer " + backend.api_key)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return bool(data.get("data"))
    except Exception:
        return False


def select_backend(backends: list[Backend]) -> Backend | None:
    """Return the first healthy backend, probing each in order."""
    for backend in backends:
        if _probe_backend(backend):
            backend.healthy = True
            return backend
        backend.healthy = False
    return None


class Router:
    """Stateless OpenAI-compatible request forwarder."""

    def __init__(
        self,
        backends: list[Backend] | None = None,
        jobs: JobManager | None = None,
        management_token: str | None = None,
    ):
        self._backends = backends
        self.jobs = jobs or JobManager()
        self.management_token = management_token or os.environ.get("COLAB_T4_MANAGEMENT_TOKEN") or secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._last_refresh = 0.0
        self._cached: list[Backend] = []

    def backends(self) -> list[Backend]:
        if self._backends is not None:
            return list(self._backends)
        now = time.monotonic()
        if now - self._last_refresh > 5:
            with self._lock:
                if now - self._last_refresh > 5:
                    self._cached = discover_backends()
                    self._last_refresh = now
        return list(self._cached)

    def select(self) -> Backend | None:
        return select_backend(self.backends())

    def forward(self, method: str, path: str, body: bytes | None, headers: dict[str, str]) -> tuple[int, dict[str, Any], bytes]:
        """Forward an OpenAI-compatible request to the best available backend.

        Returns ``(status, response_headers, response_body)``. On total backend
        failure returns a 503.
        """
        backend = self.select()
        if backend is None:
            return 503, {"Content-Type": "application/json"}, json.dumps(
                {"error": {"message": "no healthy backend available", "type": "router_error"}}
            ).encode()
        url = backend.base_url.rstrip("/") + path
        forward_headers: dict[str, str] = {}
        hop_by_hop = set(HOP_BY_HOP_HEADERS)
        for value in headers.get("Connection", "").split(","):
            token = value.strip().lower()
            if token:
                hop_by_hop.add(token)
        for key, value in headers.items():
            if key.lower() in {"host", "content-length"} | hop_by_hop:
                continue
            forward_headers[key] = value
        if backend.api_key and "Authorization" not in forward_headers:
            forward_headers["Authorization"] = "Bearer " + backend.api_key
        req = urllib.request.Request(url, data=body, method=method, headers=forward_headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return response.status, dict(response.headers.items()), response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers.items()), exc.read()


# ---------------------------------------------------------------------------
# HTTP request handler for the OpenAI-compatible proxy
# ---------------------------------------------------------------------------


def _handler_factory(router: Router):
    class _Handler(BaseHTTPRequestHandler):
        def _content_length(self) -> int:
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError as exc:
                raise _RequestRejected(400, "invalid content length") from exc
            if length < 0:
                raise _RequestRejected(400, "invalid content length")
            if length > MAX_REQUEST_BODY_BYTES:
                raise _RequestRejected(413, "request body too large")
            return length

        def _reject_oversized_body(self) -> bool:
            try:
                self._content_length()
                return False
            except _RequestRejected as exc:
                self._json_error_response(exc.status, exc.message)
                return True

        def _read_body(self) -> bytes:
            length = self._content_length()
            if length:
                return self.rfile.read(length)
            return b""

        def _respond(self, status: int, headers: dict[str, str], body: bytes) -> None:
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() in {"transfer-encoding", "connection", "content-length"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json_error_response(self, status: int, message: str, headers: dict[str, str] | None = None) -> None:
            response_headers = {"Content-Type": "application/json"}
            if headers:
                response_headers.update(headers)
            self._respond(status, response_headers, json.dumps({"error": message}).encode())

        def _method_not_allowed(self, allowed: list[str]) -> None:
            self._json_error_response(405, "method not allowed", {"Allow": ", ".join(allowed)})

        def _dispatch(self, method: str) -> None:
            if method not in {"GET", "POST"}:
                self._method_not_allowed(["GET", "POST"])
                return
            split = urlsplit(self.path)
            path = split.path
            forwarded = path[len("/v1"):] if path.startswith("/v1/") else path
            api_path = "/v1" + forwarded if not forwarded.startswith("/") else "/v1" + forwarded
            try:
                body = self._read_body()
                headers = {k: v for k, v in self.headers.items()}
                status, resp_headers, resp_body = router.forward(method, api_path, body, headers)
            except _RequestRejected as exc:
                self._json_error_response(exc.status, exc.message)
                return
            except Exception as exc:
                status = 502
                resp_headers = {"Content-Type": "application/json"}
                resp_body = json.dumps(
                    {"error": {"message": redact(str(exc), _secret_values()), "type": "router_error"}}
                ).encode()
            self._respond(status, resp_headers, resp_body)

        def do_GET(self) -> None:
            split = urlsplit(self.path)
            if split.path in {"/health", "/"}:
                backend = router.select()
                if backend is None:
                    self._respond(503, {"Content-Type": "application/json"},
                                  json.dumps({"status": "degraded", "backend": None}).encode())
                else:
                    self._respond(200, {"Content-Type": "application/json"},
                                  json.dumps({"status": "ok", "backend": backend.name}).encode())
                return
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._method_not_allowed(["GET", "POST"])

        def do_PUT(self) -> None:
            self._method_not_allowed(["GET", "POST"])

        def do_PATCH(self) -> None:
            self._method_not_allowed(["GET", "POST"])

        def do_HEAD(self) -> None:
            self._method_not_allowed(["GET", "POST"])

        def do_OPTIONS(self) -> None:
            self._method_not_allowed(["GET", "POST"])

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return _Handler


# ---------------------------------------------------------------------------
# Management API — exposes runtime state, accounts, and lifecycle operations
# ---------------------------------------------------------------------------


def _management_state() -> dict[str, Any]:
    """Build a snapshot of runtime + account state for the UI."""
    state = load_state()
    secrets = load_secrets()
    secret_values = _secret_values()
    accounts = load_accounts()
    runtime_state = state.get("runtime_state", "unknown")
    api_base = state.get("api_base")
    active_account = state.get("account")
    account_rows = []
    for account in accounts:
        if account.last_error:
            status = "error"
        elif account.last_ok:
            status = "ok"
        else:
            status = "unused"
        account_rows.append({
            "id": account.id,
            "email": account.email or "unknown",
            "home": account.home or "(default home)",
            "active": account.id == active_account,
            "status": status,
            "last_used": account.last_used_at or "-",
            "last_error": redact(account.last_error or "", secret_values),
        })
    backend_list = []
    active_backend = None
    for backend in discover_backends():
        is_healthy = _probe_backend(backend)
        if is_healthy and active_backend is None:
            active_backend = backend.name
        backend_list.append({
            "name": backend.name,
            "base_url": backend.base_url,
            "healthy": is_healthy,
        })
    return {
        "runtime_state": runtime_state,
        "session": state.get("session", ""),
        "account": active_account,
        "gpu": state.get("gpu", ""),
        "api_base": api_base or "",
        "model": state.get("model", ""),
        "ssh_mode": state.get("ssh_mode", "tailscale"),
        "last_error": redact(str(state.get("last_error") or ""), secret_values),
        "runtime": {
            "ready": runtime_state == "ready",
            "updated_at": state.get("updated_at", ""),
        },
        "proxy": {
            "api_url": "/v1",
            "health_url": "/health",
            "management_url": "/manage",
            "active_backend": active_backend or "none",
        },
        "accounts": account_rows,
        "backends": backend_list,
        "secrets_configured": {
            "tailscale_authkey": bool(secrets.get("tailscale_authkey")),
            "api_key": bool(secrets.get("api_key")),
            "hf_token": bool(secrets.get("hf_token")),
            "ssh_password": bool(secrets.get("ssh_password")),
        },
    }


def _lifecycle_options() -> Any:
    """Build the default options object used by lifecycle service adapters."""
    class _Options:
        session = None
        model = None
        quant = None
        port = None
        ctx = None
        api_key = None
        password = None
        pubkey = None
        exec_timeout = None
        account = None
        account_only = False

    return _Options()


def _redact_output(value: Any) -> Any:
    """Recursively redact persisted secrets from management API output."""
    secret_values = _secret_values()
    if isinstance(value, str):
        return redact(value, secret_values)
    if isinstance(value, dict):
        return {
            redact(str(key), secret_values): _redact_output(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_output(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_output(item) for item in value]
    return value


_MAX_LOG_BYTES = 16 * 1024


def _management_logs() -> list[dict[str, Any]]:
    """Return a bounded, redacted snapshot of regular files in the log directory."""
    root = logs_dir()
    entries: list[dict[str, Any]] = []
    root_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, root_flags)
    except OSError:
        return entries
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            return entries
        for name in sorted(os.listdir(root_fd)):
            file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                file_fd = os.open(name, file_flags, dir_fd=root_fd)
            except OSError:
                continue
            try:
                file_stat = os.fstat(file_fd)
                if not stat.S_ISREG(file_stat.st_mode):
                    continue
                with os.fdopen(file_fd, "rb", closefd=False) as handle:
                    if file_stat.st_size > _MAX_LOG_BYTES:
                        handle.seek(-_MAX_LOG_BYTES, 2)
                    content = handle.read(_MAX_LOG_BYTES).decode("utf-8", errors="replace")
            except OSError:
                continue
            finally:
                os.close(file_fd)
            content = _redact_output(content)[-_MAX_LOG_BYTES:]
            entries.append({
                "name": name,
                "content": content,
                "truncated": file_stat.st_size > _MAX_LOG_BYTES,
            })
    finally:
        os.close(root_fd)
    return entries


def _update_state(**values: Any) -> dict[str, Any]:
    state = load_state()
    if isinstance(values.get("last_error"), str):
        values["last_error"] = redact(values["last_error"], _secret_values())
    state.update(values)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_state(state)
    return state


# ---------------------------------------------------------------------------
# Web UI HTML
# ---------------------------------------------------------------------------

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>colab-t4 manager</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #10131a; color: #d8dbe5; }
  header { background: #1a1d25; padding: 16px 24px; border-bottom: 1px solid #2d313c; }
  header h1 { margin: 0; font-size: 20px; }
  .container { max-width: 1120px; margin: 0 auto; padding: 24px; }
  .card { background: #1a1d25; border: 1px solid #2d313c; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
  .card h2 { margin-top: 0; font-size: 16px; border-bottom: 1px solid #2d313c; padding-bottom: 8px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
  .field { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d313c; }
  .field .label { color: #8b91a7; font-size: 13px; }
  .field .value { font-family: monospace; font-size: 13px; }
  label { display: flex; flex-direction: column; gap: 5px; color: #8b91a7; font-size: 12px; }
  input, textarea, select { width: 100%; background: #10131a; color: #d8dbe5; border: 1px solid #3a4051; border-radius: 5px; padding: 8px 10px; font: inherit; font-size: 13px; }
  textarea { min-height: 88px; resize: vertical; }
  .status-ready { color: #50fa76; }
  .status-pending, .status-creating, .status-uploading, .status-executing, .status-waiting { color: #ffb86c; }
  .status-running, .status-queued { color: #ffb86c; }
  .status-succeeded { color: #50fa76; }
  .status-cancelled { color: #8be9fd; }
  .status-failed, .status-interrupted, .status-stopped { color: #ff5555; }
  .status-unknown { color: #8b91a7; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #2d313c; font-size: 13px; }
  th { color: #8b91a7; font-weight: normal; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
  .badge-ok { background: rgba(80,250,118,0.2); color: #50fa76; }
  .badge-error { background: rgba(255,85,85,0.2); color: #ff5555; }
  .badge-unused { background: rgba(139,145,167,0.2); color: #8b91a7; }
  button { background: #4c526b; color: #fff; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; font-size: 13px; }
  button:hover { background: #5c6378; }
  button.danger { background: #c44569; }
  button.danger:hover { background: #d65b82; }
  button.secondary { background: #2d313c; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .refresh-btn { background: #50fa76; color: #0f1117; }
  .refresh-btn:hover { background: #62ff8fcc; }
  .alert { padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 13px; }
  .alert-error { background: rgba(255,85,85,0.15); color: #ff5555; border: 1px solid #ff5555; }
  .alert-ok { background: rgba(80,250,118,0.15); color: #50fa76; border: 1px solid #50fa76; }
  .alert-info { background: rgba(139,145,167,0.15); color: #d4d7e0; border: 1px solid #4c526b; }
  .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
  .metric { background: #242834; border-radius: 6px; padding: 14px; }
  .metric .label { color: #8b91a7; font-size: 12px; display: block; margin-bottom: 6px; }
  .metric .value { font-family: monospace; font-size: 14px; }
  .muted { color: #8b91a7; font-size: 12px; }
  .stack { display: flex; flex-direction: column; gap: 10px; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .hidden { display: none; }
  .tabs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 16px; }
  .tab { padding: 8px 16px; background: #2d313c; border: none; cursor: pointer; font-size: 13px; border-radius: 4px 4px 0 0; }
  .tab.active { background: #4c526b; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #10131a; border: 1px solid #2d313c; border-radius: 6px; padding: 12px; max-height: 360px; overflow: auto; font-size: 12px; }
  code { background: #2d313c; padding: 2px 6px; border-radius: 3px; font-size: 12px; }
  @media (max-width: 760px) { .grid, .form-grid, .summary { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>colab-t4 Manager</h1>
  <span id="status-indicator" style="font-size:12px;color:#8b91a7;"></span>
</header>
<div class="container">
  <div id="alerts"></div>
  <div id="action-status" class="muted"></div>
  <div id="job-status" class="muted"></div>
  <button id="refresh-state" class="refresh-btn">Refresh</button>
  <button id="retry-state">Retry</button>
  <span class="muted" id="refresh-status">Auto-refreshing every 15 seconds</span>

  <div class="summary">
    <div class="metric"><span class="label">Runtime readiness</span><span class="value" id="summary-ready">-</span></div>
    <div class="metric"><span class="label">Active backend</span><span class="value" id="summary-backend">-</span></div>
    <div class="metric"><span class="label">Proxy endpoint</span><span class="value" id="summary-proxy">/v1</span></div>
    <div class="metric"><span class="label">Last updated</span><span class="value" id="summary-updated">-</span></div>
  </div>

  <div class="tabs">
    <button class="tab active" data-tab="overview-panel">Overview</button>
    <button class="tab" data-tab="runtime-panel">Runtime</button>
    <button class="tab" data-tab="accounts-panel">Accounts</button>
    <button class="tab" data-tab="configuration-panel">Configuration</button>
    <button class="tab" data-tab="diagnostics-panel">Diagnostics</button>
    <button class="tab" data-tab="api-tests-panel">API Tests</button>
    <button class="tab" data-tab="logs-panel">Logs</button>
    <button class="tab" data-tab="jobs-panel">Jobs</button>
  </div>

  <div id="overview-panel" class="tab-content active">
    <div class="card">
      <h2>Dashboard</h2>
      <div class="grid">
        <div>
          <div class="field"><span class="label">State</span><span class="value"><span id="runtime-state" class="status-unknown">unknown</span></span></div>
          <div class="field"><span class="label">Account</span><span class="value" id="runtime-account">-</span></div>
          <div class="field"><span class="label">Session</span><span class="value" id="runtime-session">-</span></div>
          <div class="field"><span class="label">GPU</span><span class="value" id="runtime-gpu">-</span></div>
          <div class="field"><span class="label">Model</span><span class="value" id="runtime-model">-</span></div>
          <div class="field"><span class="label">API base</span><span class="value" id="runtime-api">-</span></div>
        </div>
        <div class="stack">
          <button id="overview-up" data-action-button>Start runtime (up)</button>
          <button id="overview-restart" data-action-button>Restart runtime</button>
          <button id="overview-down" data-action-button class="danger">Stop runtime (down)</button>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>Backend Health</h2>
      <table id="backends-table">
        <thead><tr><th>Name</th><th>Base URL</th><th>Status</th></tr></thead>
        <tbody></tbody>
      </table>
      <p class="muted">The first healthy backend is selected for each request. Colab is preferred, with Ollama as fallback.</p>
    </div>
  </div>

  <div id="runtime-panel" class="tab-content">
    <div class="card">
      <h2>Runtime Controls</h2>
      <div class="row">
        <button id="runtime-up" data-action-button>Start</button>
        <button id="runtime-restart" data-action-button>Restart</button>
        <button id="runtime-down" data-action-button class="danger">Stop</button>
      </div>
      <p class="muted">Runtime actions return pollable jobs. Buttons stay disabled while a job is active.</p>
    </div>
    <div class="card">
      <h2>Current Runtime</h2>
      <div id="runtime-details"></div>
    </div>
  </div>

  <div id="accounts-panel" class="tab-content">
    <div class="card">
      <h2>Registered Accounts</h2>
      <table id="accounts-table">
        <thead><tr><th>ID</th><th>Email</th><th>Status</th><th>Last Used</th><th>Action</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <h2>Add Account</h2>
      <form id="account-form" class="stack">
        <div class="form-grid">
          <label>Account ID <input id="account-id" name="id" autocomplete="off" placeholder="account-2"></label>
          <label>Email <input id="account-email" name="email" type="email" autocomplete="off" placeholder="optional@example.com"></label>
        </div>
        <label>Token JSON <textarea id="account-token-json" name="token_json" autocomplete="off" placeholder="Optional write-only token JSON"></textarea></label>
        <div class="row"><button id="account-add" type="submit" data-action-button>Add account</button></div>
      </form>
    </div>
    <div class="card">
      <h2>OAuth Controls</h2>
      <div class="form-grid">
        <label>OAuth account ID <input id="oauth-account-id" autocomplete="off" placeholder="account-2"></label>
        <label>Authorization code <input id="oauth-code" type="password" autocomplete="one-time-code" placeholder="Paste code"></label>
      </div>
      <div class="row" style="margin-top:12px;">
        <button id="oauth-start" data-action-button>Start OAuth</button>
        <button id="oauth-finish" data-action-button>Finish OAuth</button>
        <button id="oauth-cancel" data-action-button class="secondary">Cancel OAuth</button>
      </div>
      <p><a id="oauth-link" class="hidden" target="_blank" rel="noopener noreferrer">Open authorization URL</a></p>
    </div>
  </div>

  <div id="configuration-panel" class="tab-content">
    <div class="card">
      <h2>Configuration</h2>
      <form id="config-form" class="stack">
        <div class="form-grid">
          <label>Session <input id="config-session" name="session" autocomplete="off"></label>
          <label>Model <input id="config-model" name="model" autocomplete="off"></label>
          <label>Quantization <input id="config-quant" name="quant" autocomplete="off"></label>
          <label>Port <input id="config-port" name="port" type="number" min="1"></label>
          <label>Context <input id="config-ctx" name="ctx" type="number" min="1"></label>
          <label>Tailnet <input id="config-tailnet" name="tailnet" autocomplete="off"></label>
          <label>SSH mode
            <select id="config-ssh_mode" name="ssh_mode">
              <option value="tailscale">tailscale</option>
              <option value="password">password</option>
              <option value="key">key</option>
            </select>
          </label>
          <label>Tailscale auth key <input id="config-tailscale_authkey" name="tailscale_authkey" type="password" autocomplete="new-password" placeholder="Leave unchanged"></label>
          <label>HF token <input id="config-hf_token" name="hf_token" type="password" autocomplete="new-password" placeholder="Leave unchanged"></label>
          <label>API key <input id="config-api_key" name="api_key" type="password" autocomplete="new-password" placeholder="Leave unchanged"></label>
          <label>SSH password <input id="config-ssh_password" name="ssh_password" type="password" autocomplete="new-password" placeholder="Leave unchanged"></label>
        </div>
        <label>SSH public key <textarea id="config-ssh_pubkey" name="ssh_pubkey" autocomplete="off" placeholder="Leave unchanged"></textarea></label>
        <div class="row">
          <button id="config-save" type="submit" data-action-button>Save configuration</button>
          <button id="config-reset" type="button" class="danger" data-action-button>Reset configuration</button>
        </div>
      </form>
    </div>
    <div class="card">
      <h2>Secrets Configuration</h2>
      <div id="secrets-status"></div>
      <p class="muted">Secret values are intentionally never sent to this page. Filled secret fields overwrite stored values; blank fields leave them unchanged.</p>
    </div>
  </div>

  <div id="diagnostics-panel" class="tab-content">
    <div class="card">
      <h2>Diagnostics</h2>
      <div class="row">
        <button id="status-run" data-action-button>Runtime status</button>
        <button id="doctor-run" data-action-button>Run doctor</button>
      </div>
      <pre id="diagnostics-output"></pre>
    </div>
  </div>

  <div id="api-tests-panel" class="tab-content">
    <div class="card">
      <h2>API Tests</h2>
      <div class="row"><button id="api-models-run" data-action-button>List models</button></div>
      <form id="api-chat-form" class="stack" style="margin-top:12px;">
        <label>Chat message <input id="api-chat-message" autocomplete="off" placeholder="smoke test"></label>
        <div class="row"><button id="api-chat-run" type="submit" data-action-button>Send chat test</button></div>
      </form>
      <pre id="api-tests-output"></pre>
    </div>
  </div>

  <div id="logs-panel" class="tab-content">
    <div class="card">
      <h2>Logs</h2>
      <button id="logs-refresh" data-action-button>Refresh logs</button>
      <div id="logs-list" class="stack" style="margin-top:12px;"></div>
    </div>
  </div>

  <div id="jobs-panel" class="tab-content">
    <div class="card">
      <h2>Jobs</h2>
      <button id="jobs-refresh" data-action-button>Refresh jobs</button>
      <table style="margin-top:12px;">
        <thead><tr><th>ID</th><th>Name</th><th>Status</th><th>Progress</th><th>Action</th></tr></thead>
        <tbody id="jobs-list"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
var activeJobId = null;
var activePollTimer = null;
var busyCount = 0;
var managementToken = "__COLAB_T4_MANAGEMENT_TOKEN__";

function byId(id) {
  return document.getElementById(id);
}

function clearNode(node) {
  node.replaceChildren();
}

function appendText(parent, tag, text, className) {
  var child = document.createElement(tag);
  if (className) child.className = className;
  child.textContent = text === undefined || text === null || text === '' ? '-' : String(text);
  parent.appendChild(child);
  return child;
}

function appendField(parent, label, value) {
  var row = document.createElement('div');
  row.className = 'field';
  appendText(row, 'span', label, 'label');
  appendText(row, 'span', value, 'value');
  parent.appendChild(row);
  return row;
}

function showTab(panelId, button) {
  document.querySelectorAll('.tab-content').forEach(function(node) { node.classList.remove('active'); });
  document.querySelectorAll('.tab').forEach(function(node) { node.classList.remove('active'); });
  byId(panelId).classList.add('active');
  if (button) button.classList.add('active');
}

function showAlert(msg, type) {
  var root = byId('alerts');
  clearNode(root);
  var div = document.createElement('div');
  div.className = 'alert alert-' + (type || 'info');
  div.textContent = msg || '';
  root.appendChild(div);
}

function clearErrors() {
  clearNode(byId('alerts'));
}

function setBusy(active) {
  busyCount += active ? 1 : -1;
  if (busyCount < 0) busyCount = 0;
  var disabled = busyCount > 0 || !!activeJobId;
  document.querySelectorAll('[data-action-button]').forEach(function(button) {
    button.disabled = disabled;
  });
}

function statusClass(state) {
  var s = (state || 'unknown').toLowerCase();
  return 'status-' + s;
}

function fillInput(id, value) {
  var node = byId(id);
  if (node) node.value = value === undefined || value === null ? '' : String(value);
}

function renderJson(targetId, value) {
  byId(targetId).textContent = JSON.stringify(value, null, 2);
}

async function submitJson(url, options) {
  options = options || {};
  clearErrors();
  var fetchOptions = {
    method: options.method || 'POST',
    headers: {'Content-Type': 'application/json', 'X-Colab-T4-Management-Token': managementToken}
  };
  if (options.body !== undefined) fetchOptions.body = JSON.stringify(options.body);
  var response = await fetch(url, fetchOptions);
  var data = await response.json();
  if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
  return data;
}

async function loadState() {
  clearErrors();
  try {
    byId('refresh-status').textContent = 'Refreshing...';
    var resp = await fetch('/api/state');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    var data = await resp.json();
    byId('status-indicator').textContent = data.runtime_state ? data.runtime_state : 'loading...';

    var rs = byId('runtime-state');
    rs.textContent = data.runtime_state || 'unknown';
    rs.className = statusClass(data.runtime_state);
    byId('runtime-account').textContent = data.account || '-';
    byId('runtime-session').textContent = data.session || '-';
    byId('runtime-gpu').textContent = data.gpu || '-';
    byId('runtime-model').textContent = data.model || '-';
    byId('runtime-api').textContent = data.api_base || '-';
    byId('summary-ready').textContent = data.runtime.ready ? 'READY' : (data.runtime_state || 'UNKNOWN').toUpperCase();
    byId('summary-backend').textContent = data.proxy.active_backend || 'none';
    byId('summary-proxy').textContent = data.proxy.api_url || '/v1';
    byId('summary-updated').textContent = data.runtime.updated_at || 'not recorded';
    byId('refresh-status').textContent = 'Updated just now';
    if (data.last_error) showAlert(data.last_error, 'error');

    var bt = document.querySelector('#backends-table tbody');
    clearNode(bt);
    data.backends.forEach(function(b) {
      var tr = document.createElement('tr');
      appendText(tr, 'td', b.name);
      var urlCell = document.createElement('td');
      appendText(urlCell, 'code', b.base_url);
      tr.appendChild(urlCell);
      var statusCell = document.createElement('td');
      appendText(statusCell, 'span', b.healthy ? 'healthy' : 'down', 'badge ' + (b.healthy ? 'badge-ok' : 'badge-error'));
      tr.appendChild(statusCell);
      bt.appendChild(tr);
    });

    var at = document.querySelector('#accounts-table tbody');
    clearNode(at);
    data.accounts.forEach(function(a) {
      var tr = document.createElement('tr');
      var cls = a.status === 'ok' ? 'badge-ok' : a.status === 'error' ? 'badge-error' : 'badge-unused';
      appendText(tr, 'td', a.id);
      appendText(tr, 'td', a.email || 'unknown');
      var statusCell = document.createElement('td');
      appendText(statusCell, 'span', a.status, 'badge ' + cls);
      if (a.active) appendText(statusCell, 'code', '*');
      if (a.last_error) appendText(statusCell, 'small', a.last_error, 'status-failed');
      tr.appendChild(statusCell);
      appendText(tr, 'td', a.last_used || '-');
      var actionCell = document.createElement('td');
      if (a.id !== 'default') {
        var remove = document.createElement('button');
        remove.className = 'danger';
        remove.dataset.actionButton = 'true';
        remove.textContent = 'Remove';
        remove.addEventListener('click', function() { deleteAccount(a.id); });
        actionCell.appendChild(remove);
      }
      tr.appendChild(actionCell);
      at.appendChild(tr);
    });

    var details = byId('runtime-details');
    clearNode(details);
    appendField(details, 'State', data.runtime_state || 'unknown');
    appendField(details, 'Account', data.account || '-');
    appendField(details, 'Session', data.session || '-');
    appendField(details, 'GPU', data.gpu || '-');
    appendField(details, 'API base', data.api_base || '-');

    var ss = byId('secrets-status');
    clearNode(ss);
    var items = data.secrets_configured;
    for (var k in items) {
      appendField(ss, k, items[k] ? 'configured' : 'not set');
    }
  } catch(e) {
    byId('refresh-status').textContent = 'Refresh failed';
    showAlert('Failed to load state: ' + e.message, 'error');
  }
}

async function startRuntime() {
  await runtimeAction('/api/up', 'Starting runtime...', true);
}

async function restartRuntime() {
  await runtimeAction('/api/restart', 'Restarting runtime...', true);
}

async function stopRuntime() {
  if (!confirm('Stop the runtime?')) return;
  await runtimeAction('/api/down', 'Stopping runtime...', true);
}

function renderJob(job) {
  var progress = job.progress || {};
  var detail = progress.message || job.status;
  if (progress.percent !== null && progress.percent !== undefined) detail += ' (' + progress.percent + '%)';
  byId('job-status').textContent = 'Job ' + job.id + ': ' + detail;
}

async function pollJob(jobId) {
  return pollJobWithOptions(jobId, {});
}

async function pollJobWithOptions(jobId, options) {
  options = options || {};
  if (!options.reserved) setBusy(true);
  activeJobId = jobId;
  try {
    while (true) {
      var response = await fetch('/api/jobs/' + encodeURIComponent(jobId));
      var data = await response.json();
      if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
      var job = data.job;
      renderJob(job);
      if (job.status === 'succeeded') {
        showAlert('Job completed', 'ok');
        return job;
      }
      if (job.status === 'failed') {
        showAlert('Job failed: ' + (job.error || 'unknown error'), 'error');
        return job;
      }
      if (job.status === 'cancelled') {
        showAlert('Job cancelled', 'info');
        return job;
      }
      await new Promise(function(resolve) { setTimeout(resolve, 1000); });
    }
  } finally {
    activeJobId = null;
    setBusy(false);
  }
}

async function runtimeAction(url, message) {
  var polling = false;
  setBusy(true);
  byId('action-status').textContent = message;
  showAlert(message, 'info');
  try {
    var resp = await fetch(url, {method: 'POST', headers: {'X-Colab-T4-Management-Token': managementToken}});
    var data = await resp.json();
    if (!resp.ok) throw new Error(data.error || ('HTTP ' + resp.status));
    if (!data.job) throw new Error('server did not return a job');
    renderJob(data.job);
    polling = true;
    var job = await pollJobWithOptions(data.job.id, {reserved: true});
    byId('action-status').textContent = 'Job ' + job.status;
    await loadState();
  } catch (e) {
    activeJobId = null;
    byId('action-status').textContent = 'Action failed';
    showAlert('Action failed: ' + e.message, 'error');
  } finally {
    if (!polling) setBusy(false);
  }
}

async function loadConfig() {
  var data = await submitJson('/api/config', {method: 'GET'});
  fillInput('config-session', data.session || '');
  fillInput('config-model', data.model || '');
  fillInput('config-quant', data.quant || '');
  fillInput('config-port', data.port || '');
  fillInput('config-ctx', data.ctx || '');
  fillInput('config-tailnet', data.tailnet || '');
  fillInput('config-ssh_mode', data.ssh_mode || 'tailscale');
}

async function saveConfig(event) {
  event.preventDefault();
  setBusy(true);
  try {
    var form = byId('config-form');
    var body = {};
    ['session', 'model', 'quant', 'port', 'ctx', 'tailnet', 'ssh_mode'].forEach(function(name) {
      body[name] = form.elements[name].value;
    });
    ['tailscale_authkey', 'hf_token', 'api_key', 'ssh_password', 'ssh_pubkey'].forEach(function(name) {
      var value = form.elements[name].value;
      if (value) body[name] = value;
    });
    await submitJson('/api/config', {method: 'PUT', body: body});
    ['tailscale_authkey', 'hf_token', 'api_key', 'ssh_password', 'ssh_pubkey'].forEach(function(name) {
      form.elements[name].value = '';
    });
    showAlert('Configuration saved', 'ok');
    await loadState();
  } catch (e) {
    showAlert('Configuration failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function resetConfig() {
  if (!confirm('Reset saved configuration and secrets?')) return;
  setBusy(true);
  try {
    await submitJson('/api/config/reset', {method: 'POST', body: {confirmed: true}});
    showAlert('Configuration reset', 'ok');
    await loadConfig();
    await loadState();
  } catch (e) {
    showAlert('Reset failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function addAccount(event) {
  event.preventDefault();
  setBusy(true);
  try {
    var body = {
      id: byId('account-id').value,
      email: byId('account-email').value
    };
    var token = byId('account-token-json').value.trim();
    if (token) body.token_json = token;
    await submitJson('/api/accounts', {method: 'POST', body: body});
    byId('account-form').reset();
    showAlert('Account added', 'ok');
    await loadState();
  } catch (e) {
    showAlert('Account add failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function deleteAccount(accountId) {
  if (!confirm('Remove account ' + accountId + '?')) return;
  setBusy(true);
  try {
    await submitJson('/api/accounts/' + encodeURIComponent(accountId), {method: 'DELETE'});
    showAlert('Account removed', 'ok');
    await loadState();
  } catch (e) {
    showAlert('Remove account failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function startOAuth() {
  setBusy(true);
  try {
    var id = byId('oauth-account-id').value;
    var data = await submitJson('/api/accounts/auth-start', {method: 'POST', body: {id: id}});
    var link = byId('oauth-link');
    link.href = data.authorization_url;
    link.textContent = 'Open authorization URL for ' + data.account_id;
    link.classList.remove('hidden');
    showAlert('OAuth started', 'info');
  } catch (e) {
    showAlert('OAuth start failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function finishOAuth() {
  setBusy(true);
  try {
    await submitJson('/api/accounts/auth-finish', {
      method: 'POST',
      body: {id: byId('oauth-account-id').value, code: byId('oauth-code').value}
    });
    byId('oauth-code').value = '';
    byId('oauth-link').classList.add('hidden');
    showAlert('OAuth account added', 'ok');
    await loadState();
  } catch (e) {
    showAlert('OAuth finish failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function cancelOAuth() {
  setBusy(true);
  try {
    await submitJson('/api/accounts/auth-cancel', {method: 'POST', body: {id: byId('oauth-account-id').value}});
    byId('oauth-code').value = '';
    byId('oauth-link').classList.add('hidden');
    showAlert('OAuth cancelled', 'info');
  } catch (e) {
    showAlert('OAuth cancel failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function runDiagnostics(kind) {
  setBusy(true);
  try {
    var url = kind === 'doctor' ? '/api/doctor' : '/api/status';
    renderJson('diagnostics-output', await submitJson(url, {method: 'GET'}));
  } catch (e) {
    showAlert('Diagnostics failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function runModelsTest() {
  setBusy(true);
  try {
    renderJson('api-tests-output', await submitJson('/api/api/models', {method: 'GET'}));
  } catch (e) {
    showAlert('Model test failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function runChatTest(event) {
  event.preventDefault();
  setBusy(true);
  try {
    renderJson('api-tests-output', await submitJson('/api/api/chat', {
      method: 'POST',
      body: {message: byId('api-chat-message').value}
    }));
  } catch (e) {
    showAlert('Chat test failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function loadLogs() {
  setBusy(true);
  try {
    var data = await submitJson('/api/logs', {method: 'GET'});
    var root = byId('logs-list');
    clearNode(root);
    data.logs.forEach(function(log) {
      var card = document.createElement('div');
      card.className = 'card';
      appendText(card, 'h2', log.name + (log.truncated ? ' (truncated)' : ''));
      appendText(card, 'pre', log.content || '');
      root.appendChild(card);
    });
    if (!data.logs.length) appendText(root, 'p', 'No logs found', 'muted');
  } catch (e) {
    showAlert('Log load failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

function renderJobs(jobs) {
  var root = byId('jobs-list');
  clearNode(root);
  jobs.forEach(function(job) {
    var tr = document.createElement('tr');
    appendText(tr, 'td', job.id);
    appendText(tr, 'td', job.name);
    appendText(tr, 'td', job.status, statusClass(job.status));
    var progress = job.progress || {};
    appendText(tr, 'td', (progress.message || '-') + (progress.percent === null || progress.percent === undefined ? '' : ' (' + progress.percent + '%)'));
    var action = document.createElement('td');
    if (['queued', 'running'].indexOf(job.status) !== -1) {
      var cancel = document.createElement('button');
      cancel.className = 'danger';
      cancel.dataset.actionButton = 'true';
      cancel.textContent = 'Cancel';
      cancel.addEventListener('click', function() { cancelJob(job.id); });
      action.appendChild(cancel);
    }
    tr.appendChild(action);
    root.appendChild(tr);
  });
}

async function loadJobs() {
  setBusy(true);
  try {
    var data = await submitJson('/api/jobs', {method: 'GET'});
    renderJobs(data.jobs || []);
  } catch (e) {
    showAlert('Job load failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function cancelJob(jobId) {
  if (!confirm('Cancel job ' + jobId + '?')) return;
  setBusy(true);
  try {
    var data = await submitJson('/api/jobs/' + encodeURIComponent(jobId) + '/cancel', {method: 'POST'});
    if (data.job) renderJob(data.job);
    await loadJobs();
  } catch (e) {
    showAlert('Cancel failed: ' + e.message, 'error');
  } finally {
    setBusy(false);
  }
}

document.querySelectorAll('.tab').forEach(function(button) {
  button.addEventListener('click', function() { showTab(button.dataset.tab, button); });
});
byId('refresh-state').addEventListener('click', loadState);
byId('retry-state').addEventListener('click', loadState);
byId('overview-up').addEventListener('click', startRuntime);
byId('overview-restart').addEventListener('click', restartRuntime);
byId('overview-down').addEventListener('click', stopRuntime);
byId('runtime-up').addEventListener('click', startRuntime);
byId('runtime-restart').addEventListener('click', restartRuntime);
byId('runtime-down').addEventListener('click', stopRuntime);
byId('config-form').addEventListener('submit', saveConfig);
byId('config-reset').addEventListener('click', resetConfig);
byId('account-form').addEventListener('submit', addAccount);
byId('oauth-start').addEventListener('click', startOAuth);
byId('oauth-finish').addEventListener('click', finishOAuth);
byId('oauth-cancel').addEventListener('click', cancelOAuth);
byId('status-run').addEventListener('click', function() { runDiagnostics('status'); });
byId('doctor-run').addEventListener('click', function() { runDiagnostics('doctor'); });
byId('api-models-run').addEventListener('click', runModelsTest);
byId('api-chat-form').addEventListener('submit', runChatTest);
byId('logs-refresh').addEventListener('click', loadLogs);
byId('jobs-refresh').addEventListener('click', loadJobs);

loadState();
loadConfig().catch(function(e) { showAlert('Failed to load configuration: ' + e.message, 'error'); });
loadJobs().catch(function(e) { showAlert('Failed to load jobs: ' + e.message, 'error'); });
setInterval(loadState, 15000);
activePollTimer = setInterval(function() {
  if (!activeJobId) loadJobs().catch(function(e) { showAlert('Job refresh failed: ' + e.message, 'error'); });
}, 3000);
</script>
</body>
</html>"""


_API_GET_EXACT = {
    "/api/state",
    "/api/config",
    "/api/jobs",
    "/api/status",
    "/api/doctor",
    "/api/logs",
    "/api/api/models",
}
_API_POST_EXACT = {
    "/api/up",
    "/api/restart",
    "/api/down",
    "/api/api/chat",
    "/api/config/reset",
    "/api/accounts",
    "/api/accounts/auth-start",
    "/api/accounts/auth-finish",
    "/api/accounts/auth-cancel",
}
_API_PUT_EXACT = {"/api/config"}


def _allowed_api_methods(path: str) -> list[str]:
    methods: list[str] = []
    if path in _API_GET_EXACT:
        methods.append("GET")
    if path in _API_POST_EXACT:
        methods.append("POST")
    if path in _API_PUT_EXACT:
        methods.append("PUT")
    if path.startswith("/api/jobs/"):
        if path.endswith("/cancel"):
            job_id = unquote(path[len("/api/jobs/"):-len("/cancel")])
            return ["POST"] if job_id and "/" not in job_id else []
        job_id = unquote(path[len("/api/jobs/"):])
        return ["GET"] if job_id and "/" not in job_id else []
    if path.startswith("/api/accounts/"):
        account_id = unquote(path[len("/api/accounts/"):])
        return ["DELETE"] if account_id and "/" not in account_id else []
    return methods


def _management_handler_factory(router: Router):
    parent_factory = _handler_factory(router)

    class _ManagementHandler(parent_factory):
        def _html(self, body: str) -> bytes:
            return body.replace("__COLAB_T4_MANAGEMENT_TOKEN__", json.dumps(router.management_token)[1:-1]).encode()

        def _same_origin(self, value: str) -> bool:
            host = self.headers.get("Host", "")
            try:
                parsed = urlsplit(value)
            except ValueError:
                return False
            return parsed.scheme in {"http", "https"} and parsed.netloc == host

        def _authorize_management_mutation(self) -> bool:
            origin = self.headers.get("Origin")
            referer = self.headers.get("Referer")
            if origin and not self._same_origin(origin):
                self._json_error_response(403, "management origin rejected")
                return False
            if not origin and referer and not self._same_origin(referer):
                self._json_error_response(403, "management origin rejected")
                return False
            provided = self.headers.get(MANAGEMENT_TOKEN_HEADER, "")
            expected = router.management_token
            if not provided or not compare_digest(provided, expected):
                self._json_error_response(403, "management token required")
                return False
            return True

        def _api_not_found_or_method(self, path: str) -> None:
            allowed = _allowed_api_methods(path)
            if allowed:
                self._method_not_allowed(allowed)
                return
            self._json(404, {"error": "not found"})

        def do_GET(self) -> None:
            split = urlsplit(self.path)
            path = split.path
            if path == "/manage" or path == "/manage/":
                body = self._html(_UI_HTML)
                self._respond(200, {"Content-Type": "text/html; charset=utf-8"}, body)
                return
            if path.startswith("/api/"):
                if self._reject_oversized_body():
                    return
                self._handle_api_get(path)
                return
            super().do_GET()

        def do_POST(self) -> None:
            split = urlsplit(self.path)
            path = split.path
            if path.startswith("/api/"):
                if self._reject_oversized_body():
                    return
                if self._authorize_management_mutation():
                    try:
                        self._handle_api_post(path)
                    except _RequestRejected as exc:
                        self._json_error_response(exc.status, exc.message)
                return
            super().do_POST()

        def do_PUT(self) -> None:
            split = urlsplit(self.path)
            path = split.path
            if path.startswith("/api/"):
                if self._reject_oversized_body():
                    return
                if self._authorize_management_mutation():
                    try:
                        self._handle_api_put(path)
                    except _RequestRejected as exc:
                        self._json_error_response(exc.status, exc.message)
                return
            self._respond(404, {"Content-Type": "application/json"},
                          json.dumps({"error": "not found"}).encode())

        def do_DELETE(self) -> None:
            split = urlsplit(self.path)
            path = split.path
            if path.startswith("/api/"):
                if self._reject_oversized_body():
                    return
                if self._authorize_management_mutation():
                    self._handle_api_delete(path)
                return
            super().do_DELETE()

        def do_HEAD(self) -> None:
            split = urlsplit(self.path)
            path = split.path
            if path.startswith("/api/"):
                if self._reject_oversized_body():
                    return
                self._api_not_found_or_method(path)
                return
            super().do_HEAD()

        def do_OPTIONS(self) -> None:
            split = urlsplit(self.path)
            path = split.path
            if path.startswith("/api/"):
                if self._reject_oversized_body():
                    return
                self._api_not_found_or_method(path)
                return
            super().do_OPTIONS()

        def do_PATCH(self) -> None:
            split = urlsplit(self.path)
            path = split.path
            if path.startswith("/api/"):
                if self._reject_oversized_body():
                    return
                self._api_not_found_or_method(path)
                return
            super().do_PATCH()

        def _read_json(self) -> dict[str, Any]:
            body = self._read_body()
            if not body:
                return {}
            try:
                value = json.loads(body.decode())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("request body must be valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def _json(self, status: int, value: dict[str, Any]) -> None:
            self._respond(status, {"Content-Type": "application/json"}, json.dumps(_redact_output(value)).encode())

        def _json_error(self, status: int, exc: Exception | str) -> None:
            message = redact(str(exc), _secret_values())
            self._json(status, {"error": message})

        def _handle_api_get(self, path: str) -> None:
            if path == "/api/state":
                self._json(200, _management_state())
                return
            if path == "/api/config":
                self._json(200, configuration_summary())
                return
            if path == "/api/jobs":
                self._json(200, {"jobs": [job.to_dict() for job in router.jobs.list()]})
                return
            if path.startswith("/api/jobs/"):
                job_id = unquote(path[len("/api/jobs/"):])
                if not job_id or "/" in job_id:
                    self._json(404, {"error": "not found"})
                    return
                job = router.jobs.get(job_id)
                if job is None:
                    self._json(404, {"error": "not found"})
                    return
                self._json(200, {"job": job.to_dict()})
                return
            if path == "/api/status":
                try:
                    self._json(200, {"status": services.runtime_status()})
                except Exception as exc:
                    self._json_error(502, exc)
                return
            if path == "/api/doctor":
                try:
                    result, exit_code = services.run_doctor()
                    self._json(200, {"doctor": result, "exit_code": exit_code})
                except Exception as exc:
                    self._json_error(502, exc)
                return
            if path == "/api/logs":
                self._json(200, {"logs": _management_logs()})
                return
            if path == "/api/api/models":
                try:
                    self._json(200, {"models": services.api_models()})
                except Exception as exc:
                    self._json_error(502, exc)
                return
            self._api_not_found_or_method(path)

        def _handle_api_put(self, path: str) -> None:
            if path == "/api/config":
                try:
                    self._json(200, update_configuration(self._read_json()))
                except ValueError as exc:
                    self._json_error(400, exc)
                return
            self._api_not_found_or_method(path)

        def _handle_api_post(self, path: str) -> None:
            if path == "/api/up":
                job = router.jobs.submit(
                    "runtime up",
                    lambda context: services.runtime_up(
                        _lifecycle_options(), context.progress, cancelled=lambda: context.cancelled,
                    ),
                )
                self._json(202, {"job": job.to_dict()})
                return
            if path == "/api/restart":
                job = router.jobs.submit(
                    "runtime restart",
                    lambda context: services.runtime_restart(
                        _lifecycle_options(), context.progress, cancelled=lambda: context.cancelled,
                    ),
                )
                self._json(202, {"job": job.to_dict()})
                return
            if path == "/api/down":
                job = router.jobs.submit(
                    "runtime down",
                    lambda context: services.runtime_down(cancelled=lambda: context.cancelled),
                )
                self._json(202, {"job": job.to_dict()})
                return
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = unquote(path[len("/api/jobs/"):-len("/cancel")])
                if not job_id or "/" in job_id:
                    self._json(404, {"error": "not found"})
                    return
                job = router.jobs.get(job_id)
                if job is None:
                    self._json(404, {"error": "not found"})
                    return
                router.jobs.cancel(job_id)
                self._json(202, {"job": job.to_dict()})
                return
            if path == "/api/api/chat":
                try:
                    message = self._read_json().get("message")
                    if not isinstance(message, str) or not message.strip():
                        raise ValueError("message must be a non-empty string")
                    self._json(200, {"chat": services.api_chat(message)})
                except ValueError as exc:
                    self._json_error(400, exc)
                except Exception as exc:
                    self._json_error(502, exc)
                return
            if path == "/api/config/reset":
                try:
                    reset_configuration(bool(self._read_json().get("confirmed") is True))
                    self._json(200, configuration_summary())
                except ValueError as exc:
                    self._json_error(400, exc)
                return
            if path == "/api/accounts":
                from .accounts import create_account
                try:
                    self._json(200, create_account(self._read_json()))
                except Exception as exc:
                    self._json_error(400, exc)
                return
            if path == "/api/accounts/auth-start":
                from .accounts import start_oauth
                try:
                    self._json(200, start_oauth(self._read_json()))
                except Exception as exc:
                    self._json_error(400, exc)
                return
            if path == "/api/accounts/auth-finish":
                from .accounts import finish_oauth
                try:
                    self._json(200, finish_oauth(self._read_json()))
                except Exception as exc:
                    self._json_error(400, exc)
                return
            if path == "/api/accounts/auth-cancel":
                from .accounts import cancel_oauth
                try:
                    body = self._read_json()
                    self._json(200, cancel_oauth(body.get("id", "")))
                except Exception as exc:
                    self._json_error(400, exc)
                return
            self._api_not_found_or_method(path)

        def _handle_api_delete(self, path: str) -> None:
            if path.startswith("/api/accounts/"):
                from .accounts import remove_account_profile
                account_id = unquote(path[len("/api/accounts/"):])
                try:
                    if not account_id or "/" in account_id:
                        raise ValueError("account id must be a single path segment")
                    response = remove_account_profile(account_id)
                except Exception as exc:
                    self._json_error(400, exc)
                    return
                self._json(200, response)
                return
            self._api_not_found_or_method(path)

    return _ManagementHandler


def serve(host: str = "127.0.0.1", port: int = 8089, backends: list[Backend] | None = None) -> None:
    """Run the router HTTP server until interrupted.

    When ``port`` is 8089 (default), the management UI is served alongside the
    proxy API. The OI-compatible proxy endpoints (``/v1/*``) and the management
    endpoints (``/api/*``, ``/manage``) are multiplexed on the same port.
    """
    router = Router(backends=backends)
    server = ThreadingHTTPServer((host, port), _management_handler_factory(router))
    print(f"colab-t4 router listening on {host}:{port}")
    print(f"  proxy: /v1/* -> OpenAI-compatible API")
    print(f"  health: /health")
    print(f"  management UI: http://{host}:{port}/manage")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
