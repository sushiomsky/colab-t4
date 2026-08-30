# colab-t4 v0.5.0 Native Multi-Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native, isolated management of multiple named Colab T4 runtimes while preserving the v0.4.0 default-runtime CLI and MCP behavior.

**Architecture:** Keep the existing lifecycle/model implementation and insert a selected-runtime namespace underneath its existing config helpers. Global saved configuration and Google-account state remain under the base state directory; lifecycle state, runtime API key, and logs resolve to either the legacy base directory (`default`) or `runtimes/<id>`. CLI processes select one namespace for a command; the single-process MCP adapter serializes namespace selection with an `RLock`.

**Tech Stack:** Python 3.9+ stdlib, `argparse`, `contextvars`, Unix `fcntl`/`flock`, the existing Google Colab CLI adapter, pytest, optional MCP Python SDK v2 on Python 3.10+.

**Spec:** `docs/superpowers/specs/2026-08-30-multi-runtime-v05-design.md`

## Global Constraints

- Base package compatibility remains Python `>=3.9`; MCP remains Python `>=3.10` with `mcp>=2,<3`.
- Add no runtime dependency.
- No runtime selector means the legacy v0.4.0 runtime and file layout.
- `--runtime default` maps to the legacy base runtime; named runtimes live at `<base>/runtimes/<id>`.
- Runtime IDs match `^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$`; `default` is reserved.
- `secrets.json`, `accounts.json`, account locks, and account OAuth HOME directories are global.
- `state.json`, `runtime-api-key.json`, and logs are selected-runtime scoped.
- Named runtimes never implicitly consume the legacy/default API key, model repository, quant, or session.
- If a named runtime has no API-key file, `up` generates and persists a fresh key before provisioning. This preserves the approved behavior where `down` clears the selected runtime key while keeping a later non-interactive `up` possible.
- Account registry/LRU mutations and per-account Colab CLI operations are coordinated across CLI processes with `flock`.
- Preserve the current single-account failure behavior: with one configured account, provisioning raises the original operation error rather than replacing it with an aggregate failover error.
- MCP runtime operations are serialized in-process. v0.5.0 does not add routing, load balancing, scaling, a proxy endpoint, or parallel MCP mutations.
- No API key, HF token, Tailscale key, SSH password, OAuth path, account HOME, or raw secret dictionary may appear in runtime inventory or MCP responses.

---

## File Map

- Create `colab_t4/runtimes.py`: runtime ID validation, metadata, namespace context, inventory.
- Modify `colab_t4/config.py`: global-vs-runtime paths, effective secret loading, selected API-key helpers.
- Modify `colab_t4/accounts.py`: global-path use, registry locks, per-account locks, atomic account claiming.
- Modify `colab_t4/wizard.py`: named-runtime precedence, key regeneration, named metadata persistence.
- Modify `colab_t4/cli.py`: global selector, `runtimes list/create`, runtime-scoped dispatch including raw SSH.
- Modify `colab_t4/lifecycle.py`: account claims/locks and initial `model_repo`/`quant` persistence.
- Modify `colab_t4/mcp_server.py`: runtime arguments, serialized context, `runtimes_list`.
- Create `tests/test_runtimes.py` and `tests/test_runtimes_cli.py`.
- Modify existing account, wizard, lifecycle, failover, model, MCP, packaging tests.
- Modify README, version metadata, and CI.

---

### Task 1: Runtime namespace and secret-storage foundation

**Files:**
- Create: `colab_t4/runtimes.py`
- Modify: `colab_t4/config.py`
- Create: `tests/test_runtimes.py`

**Interfaces:**
- `config.global_state_dir() -> Path`
- `config.selected_runtime() -> str`
- `config.select_runtime(runtime_id: str)` context manager
- `config.load_saved_secrets() -> dict[str, str]`
- `config.load_runtime_api_key() -> str`
- existing `config.state_dir()`, `state_path()`, `runtime_api_key_path()`, `logs_dir()` become selected-runtime aware
- `runtimes.validate_runtime_id(runtime_id: str) -> str`
- `runtimes.selected_runtime() -> str` delegates to config
- `runtimes.runtime_context(runtime_id: str, *, require_exists: bool = True)` context manager
- `runtimes.create_runtime(runtime_id: str, *, model_repo: str, quant: str) -> dict[str, object]`
- `runtimes.load_runtime_metadata(runtime_id: str) -> dict[str, object]`
- `runtimes.update_runtime_metadata(runtime_id: str, values: dict[str, object]) -> dict[str, object]`
- `runtimes.runtime_inventory() -> list[dict[str, object]]`

- [ ] **Step 1: Write failing namespace tests**

Create `tests/test_runtimes.py` with these concrete tests:

```python
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
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_runtimes.py
```

Expected: import/attribute failures because runtime namespace APIs do not exist.

- [ ] **Step 3: Implement selected-runtime paths in `config.py`**

Add:

```python
from contextlib import contextmanager
from contextvars import ContextVar

_SELECTED_RUNTIME: ContextVar[str] = ContextVar("colab_t4_runtime", default="default")
_RUNTIME_DEFAULT_KEYS = {"api_key", "model_repo", "quant", "session", "port", "ctx"}


def global_state_dir() -> Path:
    path = Path(os.environ.get("COLAB_T4_STATE_DIR", DEFAULT_STATE_DIR))
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
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
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path
```

Make `secrets_path()` global, while `state_path()`, `runtime_api_key_path()`, and `logs_dir()` continue to derive from `state_dir()`.

Implement complete secret readers:

```python
def load_saved_secrets() -> dict[str, str]:
    try:
        with secrets_path().open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            return {str(k): str(v) for k, v in loaded.items()}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


def load_runtime_api_key() -> str:
    try:
        with runtime_api_key_path().open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and loaded.get("api_key"):
            return str(loaded["api_key"])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return ""


def load_secrets() -> dict[str, str]:
    saved = load_saved_secrets()
    if selected_runtime() != "default":
        saved = {key: value for key, value in saved.items() if key not in _RUNTIME_DEFAULT_KEYS}
    runtime_key = load_runtime_api_key()
    if runtime_key:
        saved["api_key"] = runtime_key
    return saved


def update_saved_secrets(values: dict[str, str]) -> None:
    saved = load_saved_secrets()
    saved.update(values)
    _atomic_json(secrets_path(), saved)
```

Keep `save_secrets()` replacement semantics for legacy/default callers.

- [ ] **Step 4: Implement `runtimes.py`**

Use:

```python
_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")


def selected_runtime() -> str:
    return config.selected_runtime()


def validate_runtime_id(runtime_id: str) -> str:
    if runtime_id == "default" or not _RUNTIME_ID_RE.fullmatch(runtime_id):
        raise ValueError(f"invalid runtime id '{runtime_id}'")
    return runtime_id


@contextmanager
def runtime_context(runtime_id: str, *, require_exists: bool = True):
    if runtime_id == "default":
        with config.select_runtime("default"):
            yield
        return
    validate_runtime_id(runtime_id)
    path = config.global_state_dir() / "runtimes" / runtime_id
    if require_exists and not (path / "runtime.json").is_file():
        raise RuntimeError(
            f"runtime '{runtime_id}' does not exist; create it with `colab-t4 runtimes create {runtime_id}`"
        )
    with config.select_runtime(runtime_id):
        yield
```

`create_runtime()` must create mode-0700 `<base>/runtimes/<id>`, atomically write mode-0600 `runtime.json` with `id`, `session=colab-t4-<id>`, `model_repo`, and `quant`, then enter `runtime_context(id)` and save a generated API key. Duplicate directory creation is a hard error. If metadata/key initialization fails, remove only the just-created namespace.

`runtime_inventory()` returns only `id`, `runtime_state`, `session`, `account`, `model`, `model_repo`, `quant`, `api_base`, and `api_key_configured`. Malformed state/metadata yields `runtime_state="invalid"` without aborting other rows.

- [ ] **Step 5: Run GREEN/regression gate**

```bash
pytest -q tests/test_runtimes.py tests/test_accounts.py tests/test_wizard.py tests/test_lifecycle.py tests/test_model.py
```

Expected: all PASS and default-runtime tests unchanged.

- [ ] **Step 6: Commit**

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
- `claim_candidate_account(session: str, state: dict[str, object], excluded: set[str] | None = None) -> Account`
- `account_operation(account_id: str)` context manager

- [ ] **Step 1: Add RED account tests**

Add:

```python
def test_accounts_remain_global_inside_named_runtime(state):
    from colab_t4.runtimes import create_runtime, runtime_context
    create_runtime("coder", model_repo="example/model", quant="Q4_K_M")
    with runtime_context("coder"):
        add_account("work", home=str(state / "accounts" / "work"))
        assert accounts.accounts_path() == state / "accounts.json"
        assert accounts.account_home_dir("work") == state / "accounts" / "work"
    assert get_account("work").id == "work"


def test_claim_reserves_lru_before_next_runtime_claim(state):
    add_account("a", home=str(state / "ha"))
    first = accounts.claim_candidate_account("coder-session", {}, set())
    second = accounts.claim_candidate_account("research-session", {}, {first.id})
    assert second.id != first.id
    assert get_account(first.id).last_used_at
```

Add a real lock serialization test:

```python
def test_account_operation_serializes_threads(state):
    import threading, time
    entered = []
    first_inside = threading.Event()
    release_first = threading.Event()

    def first():
        with accounts.account_operation("default"):
            entered.append("first")
            first_inside.set()
            assert release_first.wait(2)

    def second():
        assert first_inside.wait(2)
        with accounts.account_operation("default"):
            entered.append("second")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start(); t2.start()
    assert first_inside.wait(2)
    time.sleep(0.05)
    assert entered == ["first"]
    release_first.set()
    t1.join(2); t2.join(2)
    assert entered == ["first", "second"]
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_accounts.py
```

Expected: global-path/claim/lock tests fail.

- [ ] **Step 3: Add cross-process file locks**

In `accounts.py`, switch `accounts_path()` and `account_home_dir()` to `global_state_dir()`. Add private unlocked load/save helpers and:

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


@contextmanager
def account_operation(account_id: str):
    path = global_state_dir() / "account-locks" / f"{account_id}.lock"
    with _file_lock(path):
        yield
```

Public add/remove/record/bootstrap mutations hold `accounts.lock` once and call the unlocked helpers while inside it.

- [ ] **Step 4: Implement atomic claiming**

Factor current ordering into `_ordered_accounts(...)`. Then:

```python
def claim_candidate_account(session, state, excluded=None):
    excluded = excluded or set()
    with _file_lock(accounts_lock_path()):
        current = _load_accounts_unlocked(bootstrap=True)
        ordered = _ordered_accounts(current, session, state, excluded)
        if not ordered:
            raise LookupError("no untried Colab account is available")
        chosen_id = ordered[0].id
        for account in current:
            if account.id == chosen_id:
                account.last_used_at = _now()
                chosen = Account.from_dict(account.to_dict())
                break
        _save_accounts_unlocked(current)
        return chosen
```

Keep `candidate_accounts()` non-mutating for compatibility/tests.

- [ ] **Step 5: Change lifecycle failover to claim one account at a time**

Use an attempted-ID set. Each provision attempt runs inside `account_operation(account.id)`. Store the first caught exception. If only one configured account existed, re-raise that original exception; otherwise retain the aggregate `all accounts failed: ...` error shape.

Required loop shape:

```python
attempted = set()
failures = []
first_error = None
configured_count = len(load_accounts())
while len(attempted) < configured_count:
    account = claim_candidate_account(session, load_state(), attempted)
    attempted.add(account.id)
    try:
        with account_operation(account.id):
            return _provision(options, session, account)
    except (ColabCLIError, RuntimeError, TimeoutError) as exc:
        if first_error is None:
            first_error = exc
        record_failure(account.id, str(exc))
        failures.append(f"{account.id}: {exc}")
if configured_count == 1 and first_error is not None:
    raise first_error
raise RuntimeError("all accounts failed: " + " | ".join(failures))
```

- [ ] **Step 6: Run GREEN**

```bash
pytest -q tests/test_accounts.py tests/test_failover.py tests/test_lifecycle.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add colab_t4/accounts.py colab_t4/lifecycle.py tests/test_accounts.py tests/test_failover.py
git commit -m "feat: coordinate account claims across runtimes"
```

---

### Task 3: Named-runtime configuration precedence and metadata persistence

**Files:**
- Modify: `colab_t4/wizard.py`
- Modify: `colab_t4/runtimes.py`
- Modify: `tests/test_wizard.py`
- Modify: `tests/test_cli_wizard.py`

**Interfaces:**
- Consumes Task 1 namespace/key helpers.
- Produces named precedence: explicit -> environment -> runtime metadata/key -> prompt/generation -> package default, while common global Tailscale/HF/SSH settings remain shared.

- [ ] **Step 1: Add RED precedence tests**

Create a named runtime after saving legacy defaults and assert:

```python
with runtime_context("coder"):
    values = collect(options, allow_prompt=False)
assert values["session"] == "colab-t4-coder"
assert values["model"] == "example/coder-GGUF"
assert values["quant"] == "Q4_K_M"
assert values["api_key"] != "legacy-key"
```

Add explicit-vs-env precedence:

```python
options.model = "explicit/model"
monkeypatch.setenv("COLAB_T4_MODEL", "env/model")
with runtime_context("coder"):
    assert collect(options, allow_prompt=False)["model"] == "explicit/model"
```

Add key regeneration:

```python
with runtime_context("coder"):
    config.clear_runtime_api_key()
    values = collect(options, allow_prompt=False)
    assert values["api_key"]
    assert config.load_runtime_api_key() == values["api_key"]
    assert config.runtime_api_key_path().stat().st_mode & 0o777 == 0o600
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_wizard.py tests/test_cli_wizard.py
```

Expected: named precedence or missing-key tests fail.

- [ ] **Step 3: Implement runtime-aware collection**

At `collect()` start:

```python
runtime_id = config.selected_runtime()
saved = config.load_saved_secrets()
metadata = load_runtime_metadata(runtime_id) if runtime_id != "default" else {}
```

For named runtime, resolve `session`, `model`, and `quant` from explicit option, environment, metadata, then package default. Do not read those fields from global `secrets.json`. Common Tailscale/HF/SSH fields still read global saved values.

Named API-key resolution is exactly:

```python
api_key = getattr(options, "api_key", None) or os.environ.get("COLAB_T4_API_KEY") or config.load_runtime_api_key()
if not api_key:
    api_key = config.generated_api_key()
    config.save_runtime_api_key(api_key)
values["api_key"] = api_key
```

Default runtime retains existing v0.4 behavior.

- [ ] **Step 4: Make `persist()` runtime-aware**

For default runtime, preserve current replacement semantics. For named runtime:

```python
config.update_saved_secrets({
    "tailscale_authkey": values.get("tailscale_authkey", ""),
    "tailnet": values.get("tailnet", ""),
    "hf_token": values.get("hf_token", ""),
    "ssh_mode": values.get("ssh_mode", "tailscale"),
    "ssh_password": values.get("ssh_password", ""),
    "ssh_pubkey": values.get("ssh_pubkey", ""),
})
update_runtime_metadata(runtime_id, {
    "session": values["session"],
    "model_repo": values["model"],
    "quant": values["quant"],
})
config.save_runtime_api_key(values["api_key"])
```

Never overwrite legacy `api_key`, `session`, `model_repo`, `quant`, `port`, or `ctx` in global saved configuration from a named runtime.

- [ ] **Step 5: Run GREEN**

```bash
pytest -q tests/test_wizard.py tests/test_cli_wizard.py tests/test_runtimes.py tests/test_accounts.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add colab_t4/wizard.py colab_t4/runtimes.py tests/test_wizard.py tests/test_cli_wizard.py
git commit -m "feat: add named runtime configuration precedence"
```

---

### Task 4: CLI selector and runtime inventory commands

**Files:**
- Modify: `colab_t4/cli.py`
- Create: `tests/test_runtimes_cli.py`
- Modify: `tests/test_model_cli.py`

**Interfaces:**
- `colab-t4 --runtime NAME <runtime-command>`
- `COLAB_T4_RUNTIME=NAME`
- `colab-t4 runtimes list [--json]`
- `colab-t4 runtimes create NAME [--model REPOSITORY] [--quant QUANT] [--json]`

- [ ] **Step 1: Add RED CLI tests**

Write tests that monkeypatch `cli.load_state` to record `config.selected_runtime()` and cover:

```python
assert cli.main(["--runtime", "coder", "status", "--json"]) in {0, 1, 2}
monkeypatch.setenv("COLAB_T4_RUNTIME", "research")
assert cli.main(["--runtime", "coder", "status", "--json"]) in {0, 1, 2}
```

The observed runtime must be `coder` in the second case. Add env-only `research`, missing runtime, unsafe ID, duplicate create, JSON inventory redaction, and named-selector rejection for `accounts`, `configure`, `doctor`, and `runtimes`.

Add raw SSH preservation:

```python
called = []
monkeypatch.setattr(cli.os, "execvp", lambda exe, argv: called.append((exe, argv)))
cli.main(["--runtime", "coder", "ssh", "--", "uname", "-a"])
assert called[0][1][-2:] == ["uname", "-a"]
```

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_runtimes_cli.py tests/test_model_cli.py
```

Expected: parser rejects the new surface.

- [ ] **Step 3: Add parser surface**

Add global `--runtime`. Add a `runtimes` subparser with `list` and `create`; `create` accepts `name`, `--model`, `--quant`, `--json`.

- [ ] **Step 4: Runtime-select around the complete command**

Before the existing raw-SSH shortcut, extract only a leading `--runtime NAME` or `--runtime=NAME`; do not scan remote SSH arguments. Resolve:

```python
runtime_id = explicit_runtime or os.environ.get("COLAB_T4_RUNTIME") or "default"
```

For runtime-scoped commands (`up`, `status`, `wait`, `ssh`, `logs`, `api`, `model`, `restart`, `down`), enter `runtime_context(runtime_id)` around the complete handler. For global-only commands, reject a selected ID other than `default`.

- [ ] **Step 5: Render inventory without secrets**

`runtimes list --json` prints `runtime_inventory()`. Text mode prints only ID/state/session/account/model/quant/API base and `api-key=yes|no`. `runtimes create --json` returns metadata plus `api_key_configured: true`, never the generated key.

- [ ] **Step 6: Run GREEN**

```bash
pytest -q tests/test_runtimes_cli.py tests/test_model_cli.py tests/test_cli_wizard.py tests/test_app.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add colab_t4/cli.py tests/test_runtimes_cli.py tests/test_model_cli.py
git commit -m "feat: add runtime-aware CLI controls"
```

---

### Task 5: Lifecycle and model isolation

**Files:**
- Modify: `colab_t4/lifecycle.py`
- Modify: `tests/test_lifecycle.py`
- Modify: `tests/test_failover.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- No new public lifecycle API.
- Successful provisioning state gains `model_repo` and `quant`.

- [ ] **Step 1: Add RED isolation tests**

Seed two namespaces:

```python
with runtime_context("coder"):
    config.save_state({"runtime_state": "ready", "session": "coder-session", "tailscale_ip": "100.64.0.1"})
with runtime_context("research"):
    config.save_state({"runtime_state": "ready", "session": "research-session", "tailscale_ip": "100.64.0.2"})
```

Mock the Colab stop boundary, call `lifecycle.down()` inside coder, then assert only `coder-session` was stopped and research still records `research-session`.

For model switching, save distinct coder/research state and API-key files, snapshot research `state.json`, `runtime.json`, and `runtime-api-key.json` bytes, switch coder through the existing mocked switch path, then assert all three research byte snapshots are unchanged.

Add failure isolation: force coder provisioning/model activation failure and assert research remains `runtime_state="ready"`.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_lifecycle.py tests/test_failover.py tests/test_model.py
```

Expected: initial model metadata assertion fails before the lifecycle change; any path crossover failure identifies an incorrect global path.

- [ ] **Step 3: Persist initial requested model metadata**

In successful `_provision()` state update include:

```python
model_repo=remote_config["model_repo"],
quant=remote_config["quant"],
```

Keep `model` as the resolved GGUF path from readiness.

- [ ] **Step 4: Pin down/restart key semantics with tests**

Assert:

```python
with runtime_context("coder"):
    assert config.runtime_api_key_path().exists()
    lifecycle.down()
    assert not config.runtime_api_key_path().exists()
with runtime_context("research"):
    assert config.runtime_api_key_path().exists()
```

For restart, ensure `_runtime_options`/CLI/MCP resolves the key before `down`, carries it in `runtime_config`, and saves it again for the selected runtime before provisioning completes. Default `down` must still clear the legacy runtime key.

- [ ] **Step 5: Run complete isolation gate**

```bash
pytest -q tests/test_runtimes.py tests/test_runtimes_cli.py tests/test_accounts.py tests/test_failover.py tests/test_lifecycle.py tests/test_model.py tests/test_model_cli.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add colab_t4/lifecycle.py tests/test_lifecycle.py tests/test_failover.py tests/test_model.py
git commit -m "feat: isolate lifecycle and model state per runtime"
```

---

### Task 6: Runtime-aware Hermes MCP tools

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
- existing `accounts_list()` remains global
- new `runtimes_list() -> list[dict[str, object]]`

- [ ] **Step 1: Add RED MCP tests**

Update expected tools to exactly:

```python
[
    "backend_status", "backend_start", "backend_stop", "backend_restart",
    "model_current", "model_switch", "api_info", "accounts_list", "runtimes_list",
]
```

Create default/coder/research states and assert:

```python
assert mcp_server.backend_status("coder")["session"] == "coder-session"
assert mcp_server.backend_status("research")["session"] == "research-session"
assert mcp_server.backend_status()["session"] == "legacy-session"
```

Add failure restoration: patched `lifecycle_up` raises inside coder; after the call `runtimes.selected_runtime()` is `default`.

Add a two-thread test where the patched runtime operation appends `(runtime, "enter")`, blocks briefly, then appends `(runtime, "exit")`; assert each runtime's enter/exit pair is contiguous, proving the MCP `RLock` prevents namespace interleaving.

Assert known coder/research API-key strings do not occur in `repr(mcp_server.runtimes_list())`, status results, or redacted error text.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_mcp_server.py
```

Expected: signatures/tool count fail.

- [ ] **Step 3: Add serialized runtime operation context**

```python
_RUNTIME_LOCK = threading.RLock()

@contextmanager
def _runtime_operation(runtime: str | None):
    runtime_id = runtime or "default"
    with _RUNTIME_LOCK:
        with runtime_context(runtime_id):
            yield
```

Wrap every runtime-scoped tool's entire body. `_safe_call()` must execute inside the selected context so `_safe_error()` can redact that runtime's key before context restoration.

- [ ] **Step 4: Add optional runtime parameters without breaking old positional calls**

Keep existing parameters first. Example:

```python
def model_switch(model: str, quant: str = DEFAULT_QUANT, runtime: str | None = None):
    with _runtime_operation(runtime):
        return _safe_call(switch_model_runtime, model, quant, 3600)
```

`backend_start`/`backend_restart` call `_runtime_options()` inside the selected context. `runtimes_list()` acquires `_RUNTIME_LOCK` and returns `runtime_inventory()` without selecting one named runtime.

- [ ] **Step 5: Run GREEN with and without real MCP SDK**

```bash
pytest -q tests/test_mcp_server.py
python -c 'from colab_t4.mcp_server import build_server; print(type(build_server()).__name__)'
```

In the MCP-enabled Python 3.11 environment, both commands must pass and exactly nine tools must register.

- [ ] **Step 6: Commit**

```bash
git add colab_t4/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add runtime selection to MCP tools"
```

---

### Task 7: Release metadata, docs, CI, review, and merge

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `colab_t4/__init__.py`
- Modify: `.github/workflows/test.yml`
- Modify: `tests/test_packaging.py`

**Interfaces:**
- Release version `0.5.0`.
- Keep Python 3.9/3.11/3.13 base matrix and Python 3.11 MCP-v2 job.

- [ ] **Step 1: Make packaging RED**

Change packaging assertions to version `0.5.0`, preserving all current Python/MCP/entry-point assertions.

```bash
pytest -q tests/test_packaging.py
```

Expected: FAIL while project metadata remains 0.4.0.

- [ ] **Step 2: Bump version metadata**

Set `project.version = "0.5.0"` in `pyproject.toml` and `__version__ = "0.5.0"` in `colab_t4/__init__.py`. Do not change dependency floors.

- [ ] **Step 3: Document the actual multi-runtime flow**

README must include:

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

State explicitly: no selector means legacy/default; accounts/LRU are shared; state/keys/logs are isolated; named `down` clears that runtime key and later `up` regenerates one if needed; MCP accepts `runtime="coder"`; creation is CLI-only; v0.5.0 has no routing/load balancing/proxy.

- [ ] **Step 4: Extend CI smoke commands**

Use:

```yaml
- name: Smoke-test CLI metadata
  run: |
    colab-t4 --version
    colab-t4 model --help
    colab-t4 runtimes --help
    colab-t4 --runtime default status --json || test $? -eq 1
```

Keep the real MCP SDK server-construction job.

- [ ] **Step 5: Run exact-head local verification**

```bash
pytest -q
python -m compileall -q colab_t4
colab-t4 --version
colab-t4 runtimes --help
colab-t4 --runtime default status --json || test $? -eq 1
python -c 'from colab_t4.cli import build_parser; build_parser().parse_args(["--runtime", "default", "status", "--json"])'
```

With MCP extra on Python 3.11+:

```bash
python -c 'from colab_t4.mcp_server import build_server; print(type(build_server()).__name__)'
pytest -q tests/test_mcp_server.py
```

Expected: all pass and version prints `0.5.0`.

- [ ] **Step 6: Scan diff for secrets and scope creep**

```bash
git diff main...HEAD -- . ':!docs/superpowers/plans/*' ':!docs/superpowers/specs/*'
git grep -nE '(tskey-|hf_[A-Za-z0-9]|COLAB_T4_API_KEY=|refresh_token|client_secret)' -- . ':!tests/*'
```

Expected: no real credentials, raw key logging, routing/load-balancing subsystem, or unrelated refactor.

- [ ] **Step 7: Commit release/docs/CI**

```bash
git add README.md pyproject.toml colab_t4/__init__.py .github/workflows/test.yml tests/test_packaging.py
git commit -m "release: prepare colab-t4 v0.5.0"
```

- [ ] **Step 8: Open PR and require fresh CI on the exact final feature SHA**

Required jobs:

```text
Python 3.9 base: PASS
Python 3.11 base: PASS
Python 3.13 base: PASS
Python 3.11 MCP v2: PASS
```

Inspect PR diff and review comments for correctness/security findings. A code change after review requires a fresh test run and fresh final-head workflow verification.

- [ ] **Step 9: Merge and verify `main`**

Squash merge only when the exact PR head is mergeable and required jobs are green. After merge, verify the `main` push workflow on the resulting merge SHA before reporting v0.5.0 complete.
