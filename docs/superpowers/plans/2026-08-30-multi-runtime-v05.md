# colab-t4 v0.5.0 Native Multi-Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native, isolated management of multiple named Colab T4 runtimes while preserving the v0.4.0 default-runtime CLI and MCP behavior.

**Architecture:** Keep the existing lifecycle/model implementation and introduce a selected-runtime namespace underneath its existing config helpers. Global saved configuration and Google-account state remain under the base state directory; lifecycle state, runtime API key, and logs resolve to either the legacy base directory (`default`) or `runtimes/<id>`. CLI processes select one namespace per command, while the single-process MCP adapter serializes namespace selection with an `RLock`.

**Tech Stack:** Python 3.9+ stdlib, `argparse`, `contextvars`, `fcntl`/`flock` on the supported Unix host, existing Google Colab CLI adapter, pytest, optional MCP Python SDK v2 on Python 3.10+.

**Spec:** `docs/superpowers/specs/2026-08-30-multi-runtime-v05-design.md`

## Global Constraints

- Base package compatibility remains Python `>=3.9`; MCP remains Python `>=3.10` with `mcp>=2,<3`.
- No new runtime dependency is added.
- Commands without a runtime selector keep the legacy v0.4.0 state layout and behavior.
- `--runtime default` maps to the legacy base runtime; named runtimes live at `<base>/runtimes/<id>`.
- Runtime IDs match `^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$`; `default` is reserved.
- `secrets.json`, `accounts.json`, and account OAuth HOME directories are global; runtime `state.json`, `runtime-api-key.json`, and logs are isolated.
- Named runtimes never consume the legacy/default API key, model repository, quant, or session as an implicit fallback.
- A named runtime with no API-key file automatically generates and persists a fresh key before provisioning; this preserves the approved `down` behavior that clears the selected runtime key while allowing a later non-interactive `up`.
- Account registry/LRU mutations and per-account Colab CLI operations are coordinated across CLI processes with `flock`.
- MCP runtime operations are serialized in-process; v0.5.0 does not add parallel MCP mutation execution, routing, load balancing, or scaling.
- No API key, HF token, Tailscale key, SSH password, OAuth path, account HOME, or raw secret dictionary may appear in CLI runtime inventory or MCP responses.

---

## File Map

- Create `colab_t4/runtimes.py`: runtime ID validation, metadata CRUD, namespace context, inventory.
- Modify `colab_t4/config.py`: split global paths from selected-runtime paths; effective secret loading and runtime API-key isolation.
- Modify `colab_t4/accounts.py`: global-path use plus cross-process registry/account operation locks and account claiming.
- Modify `colab_t4/wizard.py`: named-runtime precedence, API-key regeneration, named metadata persistence without overwriting legacy defaults.
- Modify `colab_t4/cli.py`: global runtime selector, `runtimes list/create`, runtime-scoped dispatch including raw SSH.
- Modify `colab_t4/lifecycle.py`: use atomic account claims/locks and persist initial `model_repo`/`quant` in selected state.
- Modify `colab_t4/mcp_server.py`: optional runtime arguments, serialized runtime context, `runtimes_list`.
- Create `tests/test_runtimes.py`: namespace, metadata, inventory, secret isolation.
- Modify `tests/test_accounts.py`: global-path and locking/claim tests.
- Modify `tests/test_wizard.py` and `tests/test_cli_wizard.py`: named precedence/key regeneration/persistence.
- Create `tests/test_runtimes_cli.py`: selector and inventory CLI behavior.
- Modify `tests/test_lifecycle.py`, `tests/test_failover.py`, `tests/test_model.py`: cross-runtime lifecycle/model isolation.
- Modify `tests/test_mcp_server.py`: runtime-aware MCP surface, locking, redaction.
- Modify `tests/test_packaging.py`, `pyproject.toml`, `colab_t4/__init__.py`, `.github/workflows/test.yml`, `README.md`: v0.5.0 release/docs/CI.

---

### Task 1: Runtime namespace and secret-storage foundation

**Files:**
- Create: `colab_t4/runtimes.py`
- Modify: `colab_t4/config.py`
- Create: `tests/test_runtimes.py`

**Interfaces:**
- Produces: `validate_runtime_id(runtime_id: str) -> str`
- Produces: `selected_runtime() -> str`
- Produces: `runtime_context(runtime_id: str, *, require_exists: bool = True)` context manager
- Produces: `create_runtime(runtime_id: str, *, model_repo: str, quant: str) -> dict[str, object]`
- Produces: `load_runtime_metadata(runtime_id: str) -> dict[str, object]`
- Produces: `update_runtime_metadata(runtime_id: str, values: dict[str, object]) -> dict[str, object]`
- Produces: `runtime_inventory() -> list[dict[str, object]]`
- Produces from `config.py`: `global_state_dir() -> Path`, while existing `state_dir()` becomes selected-runtime aware.

- [ ] **Step 1: Write failing namespace tests**

Add `tests/test_runtimes.py` with tests that pin the contract before implementation:

```python
import json
import os

import pytest

from colab_t4 import config, runtimes


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setenv("COLAB_T4_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def test_default_runtime_uses_legacy_base(base):
    assert config.state_dir() == base
    with runtimes.runtime_context("default"):
        assert config.state_dir() == base
        config.save_state({"runtime_state": "ready", "session": "legacy"})
    assert json.loads((base / "state.json").read_text())["session"] == "legacy"


def test_named_runtime_isolates_state_key_and_logs_but_not_global_secrets(base):
    config.save_secrets({"tailscale_authkey": "ts-global", "api_key": "legacy-key", "model_repo": "legacy-model"})
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
```

Also add an inventory test with two valid namespaces and one deliberately corrupt `state.json`; assert inventory still contains `default`, `coder`, and `broken`, and the broken entry is marked `invalid`/`unknown` without raising.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest -q tests/test_runtimes.py
```

Expected: collection/import failure because `colab_t4.runtimes` and runtime-aware config helpers do not exist.

- [ ] **Step 3: Implement selected-runtime paths in `config.py`**

Add a `ContextVar` and keep global paths independent from runtime selection:

```python
from contextvars import ContextVar
from contextlib import contextmanager

_SELECTED_RUNTIME: ContextVar[str] = ContextVar("colab_t4_runtime", default="default")


def global_state_dir() -> Path:
    path = Path(os.environ.get("COLAB_T4_STATE_DIR", DEFAULT_STATE_DIR))
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def selected_runtime() -> str:
    return _SELECTED_RUNTIME.get()


@contextmanager
def select_runtime(runtime_id: str):
    token = _SELECTED_RUNTIME.set(runtime_id)
    try:
        yield
    finally:
        _SELECTED_RUNTIME.reset(token)


def state_dir() -> Path:
    base = global_state_dir()
    runtime_id = selected_runtime()
    path = base if runtime_id == "default" else base / "runtimes" / runtime_id
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path
```

Change `secrets_path()` to always return `global_state_dir() / "secrets.json"`; keep `state_path()`, `runtime_api_key_path()`, and `logs_dir()` selected-runtime scoped.

Split raw global saved secrets from effective selected-runtime secrets:

```python
_RUNTIME_DEFAULT_KEYS = {"api_key", "model_repo", "quant", "session", "port", "ctx"}


def load_saved_secrets() -> dict[str, str]:
    # Read only global secrets.json; do not merge runtime-api-key.json.
    ...


def load_secrets() -> dict[str, str]:
    saved = load_saved_secrets()
    if selected_runtime() != "default":
        saved = {k: v for k, v in saved.items() if k not in _RUNTIME_DEFAULT_KEYS}
    # Merge only the selected runtime-api-key.json.
    ...
    return saved


def update_saved_secrets(values: dict[str, str]) -> None:
    saved = load_saved_secrets()
    saved.update(values)
    _atomic_json(secrets_path(), saved)
```

Do not change `save_secrets()` semantics for the default runtime; existing callers/tests depend on replacement behavior.

- [ ] **Step 4: Implement `colab_t4/runtimes.py` minimally**

Use the validated ID only after regex checking. The module owns metadata, not generic lifecycle state:

```python
_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")


def validate_runtime_id(runtime_id: str) -> str:
    if runtime_id == "default" or not _RUNTIME_ID_RE.fullmatch(runtime_id):
        raise ValueError(f"invalid runtime id '{runtime_id}'")
    return runtime_id


def runtime_context(runtime_id: str, *, require_exists: bool = True):
    if runtime_id == "default":
        return config.select_runtime("default")
    validate_runtime_id(runtime_id)
    path = config.global_state_dir() / "runtimes" / runtime_id
    if require_exists and not (path / "runtime.json").is_file():
        raise RuntimeError(
            f"runtime '{runtime_id}' does not exist; create it with `colab-t4 runtimes create {runtime_id}`"
        )
    return config.select_runtime(runtime_id)
```

`create_runtime()` must use `mkdir(mode=0o700, parents=True, exist_ok=False)`, write `runtime.json` via `config._atomic_json`, then enter the new context and call `config.save_runtime_api_key(config.generated_api_key())`. If any write fails, remove only the just-created empty/new namespace.

`runtime_inventory()` must read state files directly by path (or enter each context sequentially), expose only: `id`, `runtime_state`, `session`, `account`, `model`, `model_repo`, `quant`, `api_base`, `api_key_configured`, and tolerate malformed JSON.

- [ ] **Step 5: Run namespace tests and the existing config-dependent suite**

Run:

```bash
pytest -q tests/test_runtimes.py tests/test_accounts.py tests/test_wizard.py tests/test_lifecycle.py tests/test_model.py
```

Expected: all PASS; any legacy default-state failure must be fixed before continuing.

- [ ] **Step 6: Commit Task 1**

```bash
git add colab_t4/config.py colab_t4/runtimes.py tests/test_runtimes.py
git commit -m "feat: add isolated runtime namespaces"
```

---

### Task 2: Global account registry locking and atomic account claiming

**Files:**
- Modify: `colab_t4/accounts.py`
- Modify: `colab_t4/lifecycle.py`
- Modify: `tests/test_accounts.py`
- Modify: `tests/test_failover.py`

**Interfaces:**
- Consumes: `config.global_state_dir()` from Task 1.
- Produces: `claim_candidate_account(session: str, state: dict[str, object], excluded: set[str] | None = None) -> Account`
- Produces: `account_operation(account_id: str)` context manager.

- [ ] **Step 1: Add RED tests for global paths and claims**

Extend `tests/test_accounts.py`:

```python
def test_accounts_remain_global_inside_named_runtime(state):
    from colab_t4.runtimes import create_runtime, runtime_context
    create_runtime("coder", model_repo="example/model", quant="Q4_K_M")
    with runtime_context("coder"):
        add_account("work", home=str(state / "accounts" / "work"))
        assert accounts.accounts_path() == state / "accounts.json"
        assert accounts.account_home_dir("work") == state / "accounts" / "work"
    assert get_account("work").id == "work"


def test_claim_updates_lru_so_next_claim_uses_another_account(state, monkeypatch):
    add_account("a", home=str(state / "ha"))
    first = accounts.claim_candidate_account("coder-session", {}, set())
    second = accounts.claim_candidate_account("research-session", {}, {first.id})
    assert second.id != first.id
    assert get_account(first.id).last_used_at
```

Add a lock test using two threads/processes around `account_operation("work")` and a barrier/event; assert the second critical section cannot enter until the first exits. The test must use the real lock files under the temporary global state directory, not a mocked lock.

- [ ] **Step 2: Run account tests and verify RED**

```bash
pytest -q tests/test_accounts.py
```

Expected: FAIL because account paths still follow selected `state_dir()` and claim/operation APIs do not exist.

- [ ] **Step 3: Refactor account file access into locked/unlocked helpers**

In `accounts.py`, switch `accounts_path()` and `account_home_dir()` to `global_state_dir()`. Add stdlib `fcntl` locking:

```python
@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a+") as handle:
        os.chmod(path, 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def accounts_lock_path() -> Path:
    return global_state_dir() / "accounts.lock"


def account_operation(account_id: str):
    safe = account_id.replace(".", "_")
    return _file_lock(global_state_dir() / "account-locks" / f"{safe}.lock")
```

Create private `_load_accounts_unlocked()` / `_save_accounts_unlocked()` so public mutation functions can hold one registry lock without recursively reopening the same lock. Keep the current JSON schema and default-account bootstrap behavior.

- [ ] **Step 4: Implement atomic claiming**

Factor current ordering logic into `_ordered_accounts(accounts, session, state, excluded)` and implement:

```python
def claim_candidate_account(session, state, excluded=None):
    excluded = excluded or set()
    with _file_lock(accounts_lock_path()):
        current = _load_accounts_unlocked(bootstrap=True)
        ordered = _ordered_accounts(current, session, state, excluded)
        if not ordered:
            raise LookupError("no untried Colab account is available")
        chosen = ordered[0]
        for account in current:
            if account.id == chosen.id:
                account.last_used_at = _now()  # reservation before releasing registry lock
                chosen = Account.from_dict(account.to_dict())
                break
        _save_accounts_unlocked(current)
        return chosen
```

Keep `candidate_accounts()` as a non-mutating compatibility/read helper for existing callers/tests.

- [ ] **Step 5: Change lifecycle provisioning to claim accounts one at a time**

Replace the one-time `accounts = candidate_accounts(...)` loop in `lifecycle.up()` with an attempted-ID set:

```python
attempted: set[str] = set()
failures: list[str] = []
while True:
    try:
        account = claim_candidate_account(session, load_state(), attempted)
    except LookupError:
        break
    attempted.add(account.id)
    try:
        with account_operation(account.id):
            return _provision(options, session, account)
    except (ColabCLIError, RuntimeError, TimeoutError) as exc:
        record_failure(account.id, str(exc))
        failures.append(f"{account.id}: {exc}")
```

For the existing-session recovery branch, use the current global account count rather than calling the mutating claim function just to decide whether failover exists.

- [ ] **Step 6: Run account/failover tests**

```bash
pytest -q tests/test_accounts.py tests/test_failover.py tests/test_lifecycle.py
```

Expected: PASS, including existing affinity/failure ordering tests and new lock/claim tests.

- [ ] **Step 7: Commit Task 2**

```bash
git add colab_t4/accounts.py colab_t4/lifecycle.py tests/test_accounts.py tests/test_failover.py
git commit -m "feat: coordinate account claims across runtimes"
```

---

### Task 3: Named-runtime configuration precedence and durable metadata

**Files:**
- Modify: `colab_t4/wizard.py`
- Modify: `colab_t4/runtimes.py`
- Modify: `tests/test_wizard.py`
- Modify: `tests/test_cli_wizard.py`

**Interfaces:**
- Consumes: `selected_runtime()`, `load_runtime_metadata()`, `update_runtime_metadata()`.
- Produces: named-runtime configuration precedence without mutating legacy/default runtime defaults.

- [ ] **Step 1: Add RED precedence tests**

Add tests that set global saved legacy values (`session=legacy`, `model_repo=legacy/model`, `quant=Q8_0`, `api_key=legacy-key`), create `coder` with its own metadata/key, enter `runtime_context("coder")`, then call `wizard.collect(..., allow_prompt=False)`.

Assert:

```python
assert values["session"] == "colab-t4-coder"
assert values["model"] == "example/coder-GGUF"
assert values["quant"] == "Q4_K_M"
assert values["api_key"] != "legacy-key"
```

Add explicit/env precedence assertions:

```python
# explicit beats env and metadata
options.model = "explicit/model"
monkeypatch.setenv("COLAB_T4_MODEL", "env/model")
assert collect(options, allow_prompt=False)["model"] == "explicit/model"
```

Add a named-runtime-after-down test: delete only `runtime-api-key.json`, call non-interactive `collect`, assert a fresh key is generated, mode `0600`, and returned without prompting.

- [ ] **Step 2: Run wizard tests and verify RED**

```bash
pytest -q tests/test_wizard.py tests/test_cli_wizard.py
```

Expected: named runtime currently inherits legacy defaults or fails on missing API key.

- [ ] **Step 3: Implement runtime-aware collection**

At the start of `collect()`:

```python
runtime_id = config.selected_runtime()
saved = config.load_saved_secrets()
metadata = load_runtime_metadata(runtime_id) if runtime_id != "default" else {}
```

Resolve named values exactly as:

```python
values["session"] = explicit_session or env_session or metadata.get("session") or DEFAULT_HOSTNAME
values["model"] = explicit_model or env_model or metadata.get("model_repo") or DEFAULT_MODEL
values["quant"] = explicit_quant or env_quant or metadata.get("quant") or DEFAULT_QUANT
```

For `default`, keep the current explicit -> env -> saved -> prompt -> default path byte-for-byte in behavior. Common Tailscale/HF/SSH settings still read global saved values for both runtime classes.

For a named API key:

```python
api_key = explicit_api_key or os.environ.get("COLAB_T4_API_KEY") or config.load_runtime_api_key()
if not api_key:
    api_key = config.generated_api_key()
    config.save_runtime_api_key(api_key)
values["api_key"] = api_key
```

Do not use `saved.get("api_key")` for a named runtime.

- [ ] **Step 4: Make persistence runtime-aware**

For default runtime, preserve current `persist()` replacement behavior. For named runtime, persist only common secret keys through `config.update_saved_secrets(...)`, persist `session/model_repo/quant` through `update_runtime_metadata()`, and always save the selected runtime API key separately.

Named persistence must not modify these legacy keys in `secrets.json`: `api_key`, `session`, `model_repo`, `quant`, `port`, `ctx`.

- [ ] **Step 5: Run wizard and namespace suites**

```bash
pytest -q tests/test_wizard.py tests/test_cli_wizard.py tests/test_runtimes.py tests/test_accounts.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add colab_t4/wizard.py colab_t4/runtimes.py tests/test_wizard.py tests/test_cli_wizard.py
git commit -m "feat: add named runtime configuration precedence"
```

---

### Task 4: CLI runtime selector and runtime inventory commands

**Files:**
- Modify: `colab_t4/cli.py`
- Create: `tests/test_runtimes_cli.py`
- Modify: `tests/test_model_cli.py`

**Interfaces:**
- Consumes: `runtime_context`, `create_runtime`, `runtime_inventory`.
- Produces CLI: `colab-t4 --runtime NAME ...`, `colab-t4 runtimes list`, `colab-t4 runtimes create NAME`.

- [ ] **Step 1: Write RED parser/dispatch tests**

Create `tests/test_runtimes_cli.py` using `cli.main([...])` and monkeypatch lifecycle boundaries. Cover:

```python
def test_runtime_selector_targets_named_status(...):
    assert cli.main(["--runtime", "coder", "status", "--json"]) in {0, 1, 2}
    # patched load_state observes selected runtime == "coder"


def test_runtime_env_fallback(monkeypatch, ...):
    monkeypatch.setenv("COLAB_T4_RUNTIME", "coder")
    cli.main(["status", "--json"])
    assert observed == ["coder"]


def test_explicit_runtime_beats_environment(monkeypatch, ...):
    monkeypatch.setenv("COLAB_T4_RUNTIME", "research")
    cli.main(["--runtime", "coder", "status", "--json"])
    assert observed == ["coder"]
```

Cover missing runtime, unsafe runtime ID, global command rejection with a named selector, `runtimes create` JSON output, duplicate creation failure, and `runtimes list` output never containing a known API key string.

Add a raw SSH test that monkeypatches `os.execvp` and calls:

```python
cli.main(["--runtime", "coder", "ssh", "--", "uname", "-a"])
```

Assert it uses coder's recorded Tailscale IP and leaves the remote argv unchanged.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
pytest -q tests/test_runtimes_cli.py tests/test_model_cli.py
```

Expected: parser rejects `--runtime`/`runtimes`.

- [ ] **Step 3: Add parser surface**

In `build_parser()`:

```python
parser.add_argument("--runtime", default=None, help="target named runtime (default: legacy runtime)")
```

Add:

```text
runtimes list [--json]
runtimes create NAME [--model REPOSITORY] [--quant QUANT] [--json]
```

Use `DEFAULT_MODEL` and `DEFAULT_QUANT` only when creating metadata, not as an implicit selector fallback for an existing named runtime.

- [ ] **Step 4: Make `main()` select the runtime around the complete command**

Because SSH currently bypasses argparse, first extract only a leading global selector (`--runtime NAME` or `--runtime=NAME`) before checking whether the command is `ssh`. Never scan arguments after `ssh`, because those are forwarded remote arguments.

Resolve runtime as:

```python
runtime_id = explicit_runtime or os.environ.get("COLAB_T4_RUNTIME") or "default"
```

For runtime-scoped commands, enter `runtime_context(runtime_id)` around the handler. For global-only `runtimes`, `accounts`, `configure`, and `doctor`, reject a selected runtime other than `default` with a clear error rather than silently ignoring it.

`runtimes list/create` must run outside a named context.

- [ ] **Step 5: Add inventory/create rendering**

JSON uses the allow-listed inventory objects from `runtime_inventory()`. Text output prints only ID/state/session/account/model/quant/API endpoint and `api-key=yes|no`; it must never load/print the key value.

- [ ] **Step 6: Run CLI regression suite**

```bash
pytest -q tests/test_runtimes_cli.py tests/test_model_cli.py tests/test_cli_wizard.py tests/test_app.py
```

Expected: PASS, including legacy command parsing.

- [ ] **Step 7: Commit Task 4**

```bash
git add colab_t4/cli.py tests/test_runtimes_cli.py tests/test_model_cli.py
git commit -m "feat: add runtime-aware CLI controls"
```

---

### Task 5: Lifecycle and model isolation across named runtimes

**Files:**
- Modify: `colab_t4/lifecycle.py`
- Modify: `tests/test_lifecycle.py`
- Modify: `tests/test_failover.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes all selected-runtime config helpers; no new public lifecycle API is required.
- Produces selected state containing initial `model_repo` and `quant` after provisioning.

- [ ] **Step 1: Add RED isolation regressions**

Use two created runtime namespaces and mocked `ColabCLI` boundaries. Write tests proving:

```python
with runtime_context("coder"):
    config.save_state({"runtime_state": "ready", "session": "coder-session", "tailscale_ip": "100.64.0.1"})
with runtime_context("research"):
    config.save_state({"runtime_state": "ready", "session": "research-session", "tailscale_ip": "100.64.0.2"})

with runtime_context("coder"):
    lifecycle.down()

with runtime_context("research"):
    assert config.load_state()["session"] == "research-session"
```

Capture the stop command and assert only `coder-session` was stopped.

Add model-switch tests with different state/account/API-key values in `coder` and `research`; switch coder and assert research's `state.json`, `runtime.json`, and API-key contents are byte-for-byte unchanged.

Add a failure test where coder provisioning/switch fails and research remains readable/ready.

- [ ] **Step 2: Run lifecycle/model tests and verify RED where metadata/isolation gaps remain**

```bash
pytest -q tests/test_lifecycle.py tests/test_failover.py tests/test_model.py
```

Expected: at least the new initial metadata assertion fails before implementation; any cross-runtime failure identifies a config helper still using a global path incorrectly.

- [ ] **Step 3: Persist requested model metadata on successful initial provisioning**

Change the successful `_update(...)` in `_provision()` to include:

```python
model_repo=remote_config["model_repo"],
quant=remote_config["quant"],
```

Do not alter the remote readiness/model path field; `model` remains the resolved GGUF path and `model_repo` remains the repository identifier.

- [ ] **Step 4: Verify down/restart/key behavior against the approved spec**

Keep `down()` clearing only the selected `state.json` and selected `runtime-api-key.json`. For a named subsequent `up`, Task 3's noninteractive collection regenerates/persists a fresh runtime key. `restart()` must collect/receive the key before `down`; its options therefore carry the same key through the stop/start operation, and the CLI/MCP startup path must call `save_runtime_api_key()` before/while provisioning so the selected key file exists again after restart.

Add explicit assertions for:

- `down(coder)` deletes coder key but not research key;
- `restart(coder)` restores a key for coder;
- default `down` still clears the legacy key exactly as v0.4.0.

- [ ] **Step 5: Run the full runtime isolation gate**

```bash
pytest -q tests/test_runtimes.py tests/test_runtimes_cli.py tests/test_accounts.py tests/test_failover.py tests/test_lifecycle.py tests/test_model.py tests/test_model_cli.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add colab_t4/lifecycle.py tests/test_lifecycle.py tests/test_failover.py tests/test_model.py
git commit -m "feat: isolate lifecycle and model state per runtime"
```

---

### Task 6: Runtime-aware Hermes MCP tools with serialized namespace selection

**Files:**
- Modify: `colab_t4/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

**Interfaces:**
- `backend_status(runtime: str | None = None)`
- `backend_start(model: str | None = None, quant: str | None = None, runtime: str | None = None)`
- `backend_stop(runtime: str | None = None)`
- `backend_restart(model: str | None = None, quant: str | None = None, runtime: str | None = None)`
- `model_current(runtime: str | None = None)`
- `model_switch(model: str, quant: str = DEFAULT_QUANT, runtime: str | None = None)`
- `api_info(runtime: str | None = None)`
- existing `accounts_list()` remains global.
- new `runtimes_list() -> list[dict[str, object]]`.

- [ ] **Step 1: Add RED MCP tests**

Extend `tests/test_mcp_server.py` so the expected registered tools become:

```python
[
    "backend_status",
    "backend_start",
    "backend_stop",
    "backend_restart",
    "model_current",
    "model_switch",
    "api_info",
    "accounts_list",
    "runtimes_list",
]
```

Add tests that create coder/research state and assert:

```python
assert mcp_server.backend_status("coder")["session"] == "coder-session"
assert mcp_server.backend_status("research")["session"] == "research-session"
assert mcp_server.backend_status()["session"] == "legacy-session"
```

Add a failure restoration test: patched `lifecycle_up` raises while targeted to coder; after the call, `selected_runtime()` must be `default`.

Add a concurrency test with two threads whose patched lifecycle function records `selected_runtime()` before/after a synchronization delay. Assert the observed operation blocks do not interleave and each operation sees only its requested runtime.

Add secret checks: known coder/research API keys must not occur in `repr(runtimes_list())`, `backend_status`, or an error returned through `_safe_call`.

- [ ] **Step 2: Run MCP tests and verify RED**

```bash
pytest -q tests/test_mcp_server.py
```

Expected: signatures/tool list fail.

- [ ] **Step 3: Add serialized runtime operation context**

In `mcp_server.py`:

```python
import threading
from contextlib import contextmanager
from .runtimes import runtime_context, runtime_inventory

_RUNTIME_LOCK = threading.RLock()


@contextmanager
def _runtime_operation(runtime: str | None):
    runtime_id = runtime or "default"
    with _RUNTIME_LOCK:
        with runtime_context(runtime_id):
            yield
```

Wrap the complete body of every runtime-scoped tool in this context. Call `_safe_call()` inside the context so `_safe_error()` can redact the selected runtime key before the namespace is restored.

- [ ] **Step 4: Preserve positional compatibility while adding optional runtime arguments**

Do not reorder existing positional parameters. For example:

```python
def model_switch(model: str, quant: str = DEFAULT_QUANT, runtime: str | None = None):
    with _runtime_operation(runtime):
        return _safe_call(switch_model_runtime, model, quant, 3600)
```

`backend_start`/`backend_restart` call `_runtime_options()` inside the selected context. `_runtime_options()` continues saving the resolved selected API key.

`runtimes_list()` runs under `_RUNTIME_LOCK` but does not enter one named runtime; return `runtime_inventory()` directly.

- [ ] **Step 5: Run MCP tests both without and with real SDK where available**

Base:

```bash
pytest -q tests/test_mcp_server.py
```

MCP environment:

```bash
python -c 'from colab_t4.mcp_server import build_server; print(type(build_server()).__name__)'
pytest -q tests/test_mcp_server.py
```

Expected: PASS and exactly nine tools registered.

- [ ] **Step 6: Commit Task 6**

```bash
git add colab_t4/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add runtime selection to MCP tools"
```

---

### Task 7: v0.5.0 documentation, packaging, CI, and final verification

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `colab_t4/__init__.py`
- Modify: `.github/workflows/test.yml`
- Modify: `tests/test_packaging.py`

**Interfaces:**
- Release version becomes `0.5.0`.
- CI retains Python 3.9/3.11/3.13 base matrix and Python 3.11 MCP-v2 job.

- [ ] **Step 1: Make packaging tests RED for v0.5.0**

Update `tests/test_packaging.py` to assert:

```python
assert project["project"]["version"] == "0.5.0"
assert init_version == "0.5.0"
```

Keep existing assertions for `requires-python >=3.9`, conditional MCP extra, and `colab-t4-mcp` entry point.

Run:

```bash
pytest -q tests/test_packaging.py
```

Expected: FAIL because metadata still says 0.4.0.

- [ ] **Step 2: Bump release metadata**

Set `pyproject.toml` and `colab_t4/__init__.py` to `0.5.0`. Update the project description only if needed to mention multi-runtime management; do not change dependency floors.

- [ ] **Step 3: Update README with concrete multi-runtime usage**

Document this exact flow:

```bash
colab-t4 runtimes create coder
colab-t4 runtimes create research \
  --model QuantFactory/Mistral-Nemo-Instruct-2407-abliterated-GGUF \
  --quant Q4_K_M

colab-t4 --runtime coder up
colab-t4 --runtime research up
colab-t4 runtimes list --json
colab-t4 --runtime coder model current --json
colab-t4 --runtime research api models
colab-t4 --runtime coder down
```

Explain:

- no selector = legacy/default runtime;
- named runtime state/key/log isolation;
- shared account registry/LRU and OAuth homes;
- a named `down` clears that runtime's key and a later `up` generates a new one if no explicit/environment key is supplied;
- Hermes tool examples using `runtime="coder"` / `runtime="research"`;
- `runtimes_list` is read-only and runtime creation is CLI-only;
- no routing/load balancing/proxy is included in v0.5.0.

- [ ] **Step 4: Extend CI smoke commands**

Change the base job smoke step to:

```yaml
- name: Smoke-test CLI metadata
  run: |
    colab-t4 --version
    colab-t4 model --help
    colab-t4 runtimes --help
    colab-t4 --runtime default status --json || test $? -eq 1
```

The status command may return 1 on a clean runner because no runtime exists; that is the expected unknown/not-ready exit contract, not a CI failure.

Keep the MCP real-SDK construction gate.

- [ ] **Step 5: Run complete local verification on the exact feature head**

Run:

```bash
pytest -q
python -m compileall -q colab_t4
colab-t4 --version
colab-t4 runtimes --help
colab-t4 --runtime default status --json || test $? -eq 1
python -c 'from colab_t4.cli import build_parser; build_parser().parse_args(["--runtime", "default", "status", "--json"])'
```

With MCP extra installed on Python 3.11+ also run:

```bash
python -c 'from colab_t4.mcp_server import build_server; print(type(build_server()).__name__)'
pytest -q tests/test_mcp_server.py
```

Expected: all tests/compile/smoke checks PASS; `colab-t4 --version` prints `0.5.0`.

- [ ] **Step 6: Scan the feature diff for secret leakage and scope creep**

Run:

```bash
git diff main...HEAD -- . ':!docs/superpowers/plans/*' ':!docs/superpowers/specs/*'
git grep -nE '(tskey-|hf_[A-Za-z0-9]|COLAB_T4_API_KEY=|refresh_token|client_secret)' -- . ':!tests/*'
```

Expected: no real credentials, no raw key logging, no routing/load-balancing subsystem, and no unrelated refactor.

- [ ] **Step 7: Commit release/docs/CI changes**

```bash
git add README.md pyproject.toml colab_t4/__init__.py .github/workflows/test.yml tests/test_packaging.py
git commit -m "release: prepare colab-t4 v0.5.0"
```

- [ ] **Step 8: Push/open PR and require fresh CI on the exact final SHA**

Push `feat/multi-runtime-v05`, open a PR against `main`, and verify the workflow run associated with the final feature SHA. Do not claim completion from an older commit's green run.

Required final gates:

```text
Python 3.9 base: PASS
Python 3.11 base: PASS
Python 3.13 base: PASS
Python 3.11 MCP v2: PASS
```

Inspect the PR diff/review comments for P1/P2 correctness/security findings. Fix findings with new RED tests where applicable, then obtain another fresh workflow run on the new head.

- [ ] **Step 9: Merge only after final-head verification**

Squash merge into `main` only when the exact final PR head is mergeable and all required jobs are green. After merge, verify the `main` push workflow on the resulting merge SHA before reporting v0.5.0 complete.
