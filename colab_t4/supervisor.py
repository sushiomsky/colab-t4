"""Keep a Colab inference runtime behind one stable local router."""
from __future__ import annotations

import threading
import time
import urllib.request
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from .accounts import load_accounts
from .backend import authenticate_available
from .config import load_secrets, load_state
from .lifecycle import down as runtime_down
from .lifecycle import up as runtime_up
from .router import start_router


def authenticated_accounts() -> list[str]:
    """Return registered profiles whose Colab CLI credentials are usable."""
    return [account.id for account in load_accounts() if authenticate_available(account.home)]


def runtime_ready() -> bool:
    state = load_state()
    if state.get("runtime_state") != "ready" or not state.get("api_base"):
        return False
    url = str(state["api_base"]).rstrip("/") + "/models"
    headers = {}
    key = load_secrets().get("api_key", "")
    # The upstream notebook exposes Ollama without bearer auth. The generated
    # llama backend remains authenticated with the managed key.
    if key and state.get("runtime") != "ollama":
        headers["Authorization"] = "Bearer " + key
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def runtime_age_seconds() -> float:
    started = load_state().get("started_at")
    if not started:
        return 0.0
    try:
        stamp = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds())
    except ValueError:
        return 0.0


def rotate_runtime(options: Any) -> None:
    """Stop before Colab's hard session quota and provision the next profile."""
    accounts = authenticated_accounts()
    current = load_state().get("account")
    next_account = next((account for account in accounts if account != current), accounts[0])
    replacement = SimpleNamespace(**vars(options))
    replacement.account = next_account
    replacement.account_only = False
    runtime_down()
    runtime_up(replacement)


def run(
    options: Any,
    *,
    host: str = "127.0.0.1",
    router_port: int = 8089,
    interval: float = 15.0,
    max_runtime_seconds: float = 17_400.0,
) -> None:
    """Run until interrupted, recovering the runtime with account rotation."""
    authenticated = authenticated_accounts()
    if not authenticated:
        raise RuntimeError("no authenticated Google account; authenticate each profile before supervise")
    if len(authenticated) != len(load_accounts()):
        raise RuntimeError("all registered Google profiles must be authenticated before supervise")

    router = start_router(host, router_port)
    threading.Thread(target=router.serve_forever, name="colab-t4-router", daemon=True).start()
    try:
        while True:
            try:
                if runtime_age_seconds() >= max_runtime_seconds:
                    rotate_runtime(options)
                elif not runtime_ready():
                    runtime_up(options)
            except Exception:
                # Keep the stable router alive while Colab is unavailable or
                # temporarily rejects an assignment. The next loop retries.
                pass
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown = getattr(router, "shutdown", None)
        if shutdown:
            shutdown()
        close = getattr(router, "server_close", None)
        if close:
            close()
