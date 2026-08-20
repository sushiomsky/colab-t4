"""Browserless colab-t4 command line application."""
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import shlex
import subprocess
import sys
import time
import re
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__
from .accounts import account_home_dir, add_account, get_account, load_accounts, next_account_id, remove_account, resolve_account_email
from .backend import ColabCLI, ColabCLIError
from .config import _atomic_json, load_secrets, load_state, logs_dir, redact, save_runtime_api_key, state_dir
from .notebook import DEFAULT_CTX, DEFAULT_MODEL, DEFAULT_PORT, DEFAULT_QUANT
from .lifecycle import doctor as lifecycle_doctor
from .lifecycle import down as lifecycle_down
from .lifecycle import restart as lifecycle_restart
from .lifecycle import up as lifecycle_up
from .lifecycle import wait as lifecycle_wait
from .wizard import _auth_ok, collect, configure_status, ensure_colab_auth, interactive_available, persist, reset


def fail(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _api_request(path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    state = load_state()
    secrets = load_secrets()
    base = state.get("api_base")
    if not base:
        raise RuntimeError("API endpoint is not recorded")
    if path == "health":
        url = base.rsplit("/v1", 1)[0] + "/health"
    else:
        url = base.rstrip("/") + "/" + path.lstrip("/")
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": "Bearer " + secrets.get("api_key", "")}
    if body:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30 if path == "health" else 120) as response:
        return json.loads(response.read().decode())


def _health() -> tuple[bool, str]:
    try:
        _api_request("models")
        return True, "reachable"
    except Exception as exc:
        return False, redact(str(exc))


def _classify_tool_response(response: dict) -> tuple[bool, str]:
    """Classify a real tool probe without treating arbitrary prose as a call."""
    choices = response.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    if isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
        return True, "native"
    content = message.get("content")
    if isinstance(content, str) and "<tool_call>" in content and "</tool_call>" in content:
        return True, "qwen_xml_compat"
    return False, "none"


def _api_tests() -> dict[str, object]:
    result: dict[str, object] = {}
    try:
        models = _api_request("models")
        result["models"] = bool(models.get("data"))
    except Exception as exc:
        result["models"] = False
        result["models_error"] = redact(str(exc))
    try:
        chat = _api_request("chat/completions", method="POST", payload={
            "model": "local",
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 8,
        })
        result["chat"] = bool(chat.get("choices"))
    except Exception as exc:
        result["chat"] = False
        result["chat_error"] = redact(str(exc))
    try:
        tool = _api_request("chat/completions", method="POST", payload={
            "model": "local",
            "messages": [{"role": "user", "content": "Use the health-check function."}],
            "tools": [{"type": "function", "function": {"name": "get_test_value", "description": "Return a health-check value.", "parameters": {"type": "object", "properties": {}}}}],
            "tool_choice": "required",
            "max_tokens": 16,
        })
        supported, mode = _classify_tool_response(tool)
        result["tool_calling"] = supported
        result["tool_calling_mode"] = mode
        result["tool_calling_tested"] = True
    except Exception as exc:
        result["tool_calling"] = False
        result["tool_calling_mode"] = "none"
        result["tool_calling_tested"] = True
        result["tool_calling_error"] = redact(str(exc))
    return result


def _status() -> tuple[dict, int]:
    state = load_state()
    result: dict[str, object] = {
        "runtime_state": state.get("runtime_state", "unknown"),
        "session": state.get("session"),
        "account": state.get("account"),
        "gpu": state.get("gpu"),
        "runtime": state.get("runtime", "unknown"),
        "llama_cpp_commit": state.get("llama_cpp_commit"),
        "llama_cpp_python_version": state.get("llama_cpp_python_version"),
        "native_error": state.get("native_error"),
        "accelerator": state.get("accelerator"),
        "notebook_execution": state.get("runtime_state"),
        "tailscale": {"ip": state.get("tailscale_ip"), "node": state.get("session")},
        "ssh": {"reachable": None},
        "api": {"base": state.get("api_base"), "reachable": None},
        "model": {"path": state.get("model"), "tests": state.get("tests", {})},
        "last_error": state.get("last_error"),
    }
    if state.get("api_base"):
        healthy, detail = _health()
        result["api"] = {"base": state.get("api_base"), "reachable": healthy, "detail": detail}
        result["model"]["tests"] = {**(state.get("tests") or {}), **_api_tests()}  # type: ignore[index]
    state_name = str(result["runtime_state"])
    if state_name == "ready" and result["api"].get("reachable") and result["model"].get("tests", {}).get("models"):  # type: ignore[union-attr]
        return result, 0
    if state_name in {"failed", "interrupted"}:
        return result, 2
    return result, 1


def print_status(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"runtime : {result.get('runtime_state')}")
    print(f"session : {result.get('session') or '-'}")
    print(f"account : {result.get('account') or '-'}")
    print(f"gpu     : {result.get('gpu') or result.get('accelerator') or '-'}")
    print(f"runtime : {result.get('runtime') or '-'}")
    print(f"commit  : {result.get('llama_cpp_commit') or '-'}")
    ts = result.get("tailscale") or {}
    print(f"tailscale: {ts.get('ip') or '-'}")
    api = result.get("api") or {}
    print(f"api     : {api.get('base') or '-'} ({api.get('detail') or ('ok' if api.get('reachable') else 'unreachable')})")
    print(f"model   : {result.get('model', {}).get('path') or '-'}")
    print(f"error   : {result.get('last_error') or '-'}")


def print_connections() -> None:
    state = load_state()
    ip = state.get("tailscale_ip")
    base = state.get("api_base")
    if not ip or not base:
        return
    print(f"ssh : ssh root@{ip}")
    print(f"api : {base}")
    print(f"curl: curl -fsS {base}/models -H 'Authorization: Bearer $COLAB_T4_API_KEY'")


def cmd_doctor(args: argparse.Namespace) -> int:
    result, code = lifecycle_doctor()
    should_interact = not args.json and interactive_available(getattr(args, "interactive", False))
    if should_interact and not result.get("checks", {}).get("colab_auth"):
        try:
            ensure_colab_auth(interactive=True)
            result, code = lifecycle_doctor()
        except (RuntimeError, EOFError, KeyboardInterrupt) as exc:
            print(redact(str(exc)), file=sys.stderr)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"colab cli: {result.get('colab_cli') or 'unavailable'}")
        print(f"auth     : {'configured' if result.get('auth', {}).get('available') else 'not configured'}")
        for name, ok in result.get("checks", {}).items():
            print(f"{name:16} {'ok' if ok else 'FAIL'}")
        if result.get("error"):
            print("error:", result["error"])
    return code


def cmd_configure(args: argparse.Namespace) -> int:
    if args.show:
        for name, configured in configure_status().items():
            print(f"{name}: {'configured' if configured else 'not configured'}")
        return 0
    if args.reset:
        try:
            reset(yes=args.yes)
        except (EOFError, KeyboardInterrupt):
            print("Reset cancelled.", file=sys.stderr)
            return 130
        return 0
    interactive = interactive_available(getattr(args, "interactive", False))
    if not interactive:
        fail("configure requires a TTY; use --show/--reset --yes or environment variables")
    try:
        ensure_colab_auth(interactive=True)
        values = collect(args, force=True)
        values["persist_secrets"] = True
        persist(values)
    except (EOFError, KeyboardInterrupt):
        print("Setup cancelled.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        fail(redact(str(exc)))
    print("Configuration saved securely under ~/.config/colab-t4/ (mode 0600).")
    return 0




def cmd_up(args: argparse.Namespace) -> int:
    interactive = interactive_available(getattr(args, "interactive", False)) and not getattr(args, "non_interactive", False)
    try:
        cli = ColabCLI.discover()
        if interactive:
            ensure_colab_auth(interactive=True, cli=cli)
        values = collect(args, force=False, allow_prompt=interactive)
        for key, value in values.items():
            setattr(args, key, value)
        args.runtime_config = values
        if interactive and not getattr(args, "yes", False) and not _confirm_create():
            print("Setup cancelled.")
            return 130
        persist(values)
        save_runtime_api_key(values["api_key"])
        lifecycle_up(args)
    except (EOFError, KeyboardInterrupt):
        print("Setup cancelled.", file=sys.stderr)
        return 130
    except (RuntimeError, ColabCLIError, TimeoutError) as exc:
        fail(redact(str(exc)))
    print("runtime ready")
    print_status(_status()[0], False)
    print_connections()
    return 0


def _confirm_create() -> bool:
    answer = input("Configuration complete. Create the Colab T4 runtime now? [Y/n]: ").strip().lower()
    return not answer or answer in {"y", "yes"}


def cmd_api(args: argparse.Namespace) -> int:
    state = load_state()
    if not state.get("api_base"):
        fail("API endpoint is not recorded")
    if args.action is None:
        print_connections()
    elif args.action == "models":
        print(json.dumps(_api_request("models"), indent=2))
    else:
        response = _api_request("chat/completions", method="POST", payload={
            "model": "local",
            "messages": [{"role": "user", "content": args.message}],
            "max_tokens": 32,
        })
        print(json.dumps(response, indent=2))
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    try:
        lifecycle_wait(args)
    except KeyboardInterrupt:
        fail("interrupted while waiting", 130)
    except (RuntimeError, ColabCLIError, TimeoutError) as exc:
        fail(redact(str(exc)))
    result, code = _status()
    print_status(result, args.json)
    if args.wait_health and not result.get("api", {}).get("reachable"):
        return 2
    return code
def cmd_ssh(raw: list[str]) -> int:
    if raw in (["-h"], ["--help"]):
        print("usage: colab-t4 ssh [SESSION] [SSH_OPTIONS_AND_REMOTE_COMMAND...]")
        print("examples: colab-t4 ssh -- uname -a")
        return 0
    tokens = list(raw)
    target = None
    if tokens and tokens[0] != "--" and not tokens[0].startswith("-"):
        target = tokens.pop(0)
    state = load_state()
    if target and target != state.get("session"):
        fail(f"session '{target}' is not the recorded session")
    ip = state.get("tailscale_ip")
    if not ip:
        fail("no recorded Tailscale address; run `colab-t4 wait`")
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]
    os.execvp("ssh", ["ssh", "-o", "StrictHostKeyChecking=accept-new", f"root@{ip}", *tokens])
    return 0




def cmd_status(args: argparse.Namespace) -> int:
    result, code = _status()
    print_status(result, args.json)
    return code
def cmd_logs(args: argparse.Namespace) -> int:
    paths = sorted(logs_dir().glob("*.log")) if args.kind == "all" else [logs_dir() / f"{args.kind}.log"]
    secrets = list(load_secrets().values())
    found = False
    for path in paths:
        if path.exists():
            found = True
            print(f"== {path.name} ==")
            print(redact(path.read_text(encoding="utf-8", errors="replace"), secrets), end="")
    if not found:
        print("no logs recorded")
    return 0


def cmd_down(_args: argparse.Namespace) -> int:
    try:
        lifecycle_down()
    except (RuntimeError, ColabCLIError) as exc:
        fail(redact(str(exc)))
    print("runtime stopped")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    try:
        lifecycle_restart(args)
    except (RuntimeError, ColabCLIError, TimeoutError) as exc:
        fail(redact(str(exc)))
    print("runtime restarted")
    print_connections()
    return 0


def _accounts_list(args: argparse.Namespace) -> int:
    accounts = load_accounts()
    state = load_state()
    active = state.get("account")
    rows = []
    for account in accounts:
        if account.last_error:
            status = "error: " + account.last_error[:60]
        elif account.last_ok:
            status = "ok"
        else:
            status = "unused"
        rows.append({
            "id": account.id,
            "email": account.email or "unknown",
            "home": account.home or "(default home)",
            "active": account.id == active,
            "status": status,
            "last_used": account.last_used_at or "-",
        })
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("no accounts")
        return 0
    for row in rows:
        marker = "*" if row["active"] else " "
        print(f"{marker}{row['id']:<16} {row['email']:<32} {row['status']:<70} home={row['home']}")
    print("(* = account hosting the recorded runtime)")
    return 0


def _accounts_add(args: argparse.Namespace) -> int:
    if not interactive_available(False):
        fail("accounts add requires a TTY for the interactive OAuth flow")
    account_id = args.id or next_account_id()
    if account_id == "default":
        fail("'default' is the implicit legacy account; choose a different id")
    home_dir = account_home_dir(account_id)
    try:
        home_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        fail(f"an account directory for '{account_id}' already exists; pick a different --id")
    cli = ColabCLI.discover(home=str(home_dir))
    print(f"Adding account '{account_id}' (isolated credentials under {home_dir}).")
    print("The Colab CLI will print an authorization URL. Open it in a browser, sign in")
    print("as the Google account to add, approve access, then paste the code back here.")
    try:
        result = cli.run_interactive_auth()
    except (EOFError, KeyboardInterrupt):
        shutil.rmtree(home_dir, ignore_errors=True)
        print("Authentication cancelled; no account was added.", file=sys.stderr)
        return 130
    if result != 0:
        shutil.rmtree(home_dir, ignore_errors=True)
        fail("Colab CLI authentication flow failed; no account was added")
    ok, reason = _auth_ok(cli, str(home_dir))
    if not ok:
        shutil.rmtree(home_dir, ignore_errors=True)
        fail("authentication verification failed: " + reason)
    email = resolve_account_email(str(home_dir)) or args.email or ""
    add_account(account_id, home=str(home_dir), email=email)
    print(f"account '{account_id}' added{(' (' + email + ')') if email else ' (email unknown)'}")
    print("The account is now part of the automatic failover rotation.")
    return 0


AUTH_URL_RE = re.compile(r"https://accounts\.google\.com/o/oauth2/auth\?\S+")


def _pending_auth_path() -> Path:
    return state_dir() / "pending-auth.json"


def _read_pending_auth() -> dict:
    try:
        value = json.loads(_pending_auth_path().read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _clear_pending_auth(pending: dict) -> None:
    path = _pending_auth_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    fifo_text = str(pending.get("fifo", ""))
    if fifo_text:
        fifo = Path(fifo_text)
        try:
            fifo.unlink()
        except FileNotFoundError:
            pass


def _accounts_auth_start(args: argparse.Namespace) -> int:
    pending = _read_pending_auth()
    if pending.get("pid"):
        try:
            os.kill(int(pending["pid"]), 0)
            fail("another Google authentication flow is already waiting")
        except ProcessLookupError:
            _clear_pending_auth(pending)
        except PermissionError:
            fail("another Google authentication flow is already waiting")
    account_id = args.id or next_account_id()
    if account_id == "default":
        fail("'default' is the implicit legacy account; choose a different id")
    home_dir = account_home_dir(account_id)
    try:
        home_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        fail(f"an account directory for '{account_id}' already exists; pick a different --id")
    fifo = home_dir / "oauth.stdin"
    log_path = home_dir / "oauth.log"
    try:
        os.mkfifo(fifo, 0o600)
        cli = ColabCLI.discover(home=str(home_dir))
        fifo_fd = os.open(fifo, os.O_RDWR)
        # The installed Colab CLI deliberately requires a TTY for OAuth.  A
        # plain FIFO is not enough: it prints the URL and then aborts.  Use
        # util-linux `script` as a tiny PTY adapter when available, while
        # retaining the direct subprocess fallback for minimal hosts.
        script = shutil.which("script")
        if script:
            command = shlex.join(cli.sessions_command())
            process = subprocess.Popen(
                [script, "-qefc", command, str(log_path)],
                stdin=fifo_fd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                env=cli.cli_env(),
                text=True,
                start_new_session=True,
            )
            log_handle = None
        else:
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                cli.sessions_command(),
                stdin=fifo_fd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=cli.cli_env(),
                text=True,
            )
        os.close(fifo_fd)
        if log_handle is not None:
            log_handle.close()
        url = ""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                text = ""
            match = AUTH_URL_RE.search(text)
            if match:
                url = match.group(0)
                break
            if process.poll() is not None:
                break
            time.sleep(0.2)
        if not url:
            process.terminate()
            _clear_pending_auth({"fifo": str(fifo)})
            shutil.rmtree(home_dir, ignore_errors=True)
            fail("Colab CLI did not produce an authorization URL")
        pending = {
            "account_id": account_id,
            "home": str(home_dir),
            "fifo": str(fifo),
            "log": str(log_path),
            "pid": process.pid,
            "created_at": time.time(),
        }
        _atomic_json(_pending_auth_path(), pending)
        print(json.dumps({"account_id": account_id, "authorization_url": url}))
        return 0
    except Exception:
        try:
            os.close(fifo_fd)
        except (NameError, OSError):
            pass
        try:
            if log_handle is not None:
                log_handle.close()
        except (NameError, OSError):
            pass
        if home_dir.exists():
            shutil.rmtree(home_dir, ignore_errors=True)
        raise


def _accounts_auth_finish(args: argparse.Namespace) -> int:
    pending = _read_pending_auth()
    if not pending or (args.id and pending.get("account_id") != args.id):
        fail("no matching pending Google authentication flow")
    code = (args.code or sys.stdin.read()).strip()
    if not code:
        fail("authorization code is required on stdin")
    fifo = Path(str(pending.get("fifo", "")))
    try:
        with fifo.open("w", encoding="utf-8") as handle:
            handle.write(code + "\n")
    except OSError as exc:
        fail("could not submit authorization result: " + str(exc))
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            os.kill(int(pending["pid"]), 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    home = str(pending["home"])
    cli = ColabCLI.discover(home=home)
    ok, reason = _auth_ok(cli, home)
    if not ok:
        _clear_pending_auth(pending)
        shutil.rmtree(home, ignore_errors=True)
        fail("authentication verification failed: " + reason)
    email = resolve_account_email(home) or ""
    add_account(str(pending["account_id"]), home=home, email=email)
    _clear_pending_auth(pending)
    print(json.dumps({"account_id": pending["account_id"], "email": email, "status": "added"}))
    return 0


def _accounts_auth_cancel(args: argparse.Namespace) -> int:
    pending = _read_pending_auth()
    if not pending or (args.id and pending.get("account_id") != args.id):
        fail("no matching pending Google authentication flow")
    pid = pending.get("pid")
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError, TypeError):
            pass
        except PermissionError as exc:
            fail("could not cancel the pending Google authentication flow: " + str(exc))
    home = str(pending.get("home", ""))
    _clear_pending_auth(pending)
    if home:
        shutil.rmtree(home, ignore_errors=True)
    print(json.dumps({"account_id": pending.get("account_id"), "status": "cancelled"}))
    return 0


def _accounts_remove(args: argparse.Namespace) -> int:
    try:
        account = get_account(args.account)
    except KeyError as exc:
        fail(str(exc))
    if not args.yes:
        if not interactive_available(False):
            fail("accounts remove requires confirmation; pass --yes")
        try:
            answer = input(f"Remove account '{account.id}'? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("Removal cancelled.", file=sys.stderr)
            return 130
        if answer not in {"y", "yes"}:
            print("Removal cancelled.")
            return 130
    try:
        remove_account(account.id)
    except ValueError as exc:
        fail(str(exc))
    if account.home:
        shutil.rmtree(account.home, ignore_errors=True)
    print(f"account '{account.id}' removed")
    return 0


def cmd_accounts(args: argparse.Namespace) -> int:
    if args.accounts_command == "list":
        return _accounts_list(args)
    if args.accounts_command == "add":
        return _accounts_add(args)
    if args.accounts_command == "auth-start":
        return _accounts_auth_start(args)
    if args.accounts_command == "auth-finish":
        return _accounts_auth_finish(args)
    if args.accounts_command == "auth-cancel":
        return _accounts_auth_cancel(args)
    if args.accounts_command == "remove":
        return _accounts_remove(args)
    return 1


def _api_action_parser(sub):
    api = sub.add_parser("api", help="exercise or print the authenticated API")
    api.add_argument("action", nargs="?", choices=["models", "chat"], default=None)
    api.add_argument("--message", default="Reply with exactly: COLAB_T4_OK")
    return api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colab-t4", description="Browserless T4 Colab runtime lifecycle manager")
    parser.add_argument("--version", action="version", version=f"colab-t4 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("up", help="create, provision, and verify a T4 runtime")
    up.add_argument("--session", default=None)
    up.add_argument("--account", default=None, help="try this registered Google account first")
    up.add_argument("--account-only", action="store_true", help="do not fail over beyond --account")
    up.add_argument("--model", default=None)
    up.add_argument("--quant", default=None)
    up.add_argument("--port", type=int, default=None)
    up.add_argument("--ctx", type=int, default=None)
    up.add_argument("--api-key")
    up.add_argument("--ssh-mode", choices=["tailscale", "password", "key"], default=None, help="SSH authentication mode (default: tailscale)")
    up.add_argument("--password")
    up.add_argument("--pubkey")
    up.add_argument("--exec-timeout", type=float, default=None)
    up.add_argument("--interactive", action="store_true")
    up.add_argument("--non-interactive", action="store_true")
    up.add_argument("--yes", action="store_true")
    sub.add_parser("status").add_argument("--json", action="store_true")
    wait = sub.add_parser("wait", help="wait for readiness")
    wait.add_argument("--session")
    wait.add_argument("--timeout", type=float, default=900)
    wait.add_argument("--wait-health", action="store_true")
    wait.add_argument("--json", action="store_true")
    sub.add_parser("down", help="stop the recorded runtime")
    restart = sub.add_parser("restart", help="stop and recreate the runtime")
    restart.add_argument("--session", default=None)
    restart.add_argument("--account", default=None, help="try this registered Google account first")
    restart.add_argument("--account-only", action="store_true", help="do not fail over beyond --account")
    restart.add_argument("--model", default=None)
    restart.add_argument("--quant", default=None)
    restart.add_argument("--port", type=int, default=None)
    restart.add_argument("--ctx", type=int, default=None)
    restart.add_argument("--api-key")
    restart.add_argument("--ssh-mode", choices=["tailscale", "password", "key"], default=None, help="SSH authentication mode (default: tailscale)")
    restart.add_argument("--password")
    restart.add_argument("--pubkey")
    restart.add_argument("--exec-timeout", type=float, default=None)
    restart.add_argument("--interactive", action="store_true")
    restart.add_argument("--non-interactive", action="store_true")
    restart.add_argument("--yes", action="store_true")
    _api_action_parser(sub)
    logs = sub.add_parser("logs", help="show redacted lifecycle logs")
    logs.add_argument("kind", nargs="?", default="all", choices=["all", "colab", "notebook", "status", "runtime", "local"])
    configure = sub.add_parser("configure", help="interactively collect and save configuration")
    configure.add_argument("--show", action="store_true")
    configure.add_argument("--reset", action="store_true")
    configure.add_argument("--yes", action="store_true")
    configure.add_argument("--interactive", action="store_true")
    doctor = sub.add_parser("doctor", help="validate dependencies and authentication")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--interactive", action="store_true")
    sub.add_parser("ssh", help="forward SSH options and remote commands")
    accounts = sub.add_parser("accounts", help="manage Google Colab account profiles (failover rotation)")
    accounts_sub = accounts.add_subparsers(dest="accounts_command", required=True)
    accounts_list = accounts_sub.add_parser("list", help="list account profiles")
    accounts_list.add_argument("--json", action="store_true")
    accounts_add = accounts_sub.add_parser("add", help="add an account via the interactive Colab OAuth flow")
    accounts_add.add_argument("--id")
    accounts_add.add_argument("--email")
    accounts_auth_start = accounts_sub.add_parser("auth-start", help="start a browserless OAuth flow and return its URL")
    accounts_auth_start.add_argument("--id")
    accounts_auth_finish = accounts_sub.add_parser("auth-finish", help="finish a pending OAuth flow; read code from stdin")
    accounts_auth_finish.add_argument("--id")
    accounts_auth_finish.add_argument("--code", help="one-time authorization code (stdin is safer and preferred)")
    accounts_auth_cancel = accounts_sub.add_parser("auth-cancel", help="cancel a pending OAuth flow and remove its temporary home")
    accounts_auth_cancel.add_argument("--id")
    accounts_remove = accounts_sub.add_parser("remove", help="remove an account profile")
    accounts_remove.add_argument("account")
    accounts_remove.add_argument("--yes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "ssh":
        return cmd_ssh(argv[1:])
    args = build_parser().parse_args(argv)
    handlers = {"up": cmd_up, "status": cmd_status, "wait": cmd_wait, "down": cmd_down, "restart": cmd_restart, "api": cmd_api, "logs": cmd_logs, "doctor": cmd_doctor, "configure": cmd_configure, "accounts": cmd_accounts}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
