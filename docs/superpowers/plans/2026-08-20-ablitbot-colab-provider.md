# AblitBot Colab Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AblitBot use `colab-t4` as its managed OpenAI-compatible provider with predefined models, Google-profile management, and automatic cross-profile recovery.

**Architecture:** `colab-t4` remains authoritative for Colab accounts, credentials, lifecycle state, readiness, and runtime API keys. AblitBot keeps only provider preferences and invokes `colab-t4` with an account-specific environment; recovery tries the preferred account first and then the remaining accounts in the CLI's reported order.

**Tech Stack:** Python 3.9+, standard library, `argparse`, `subprocess`, `urllib`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-ablitbot-colab-provider-design.md`

## Global Constraints

- Model selection is restricted to a reviewed predefined allowlist.
- Google OAuth credentials remain owned by the Colab CLI and are never copied into AblitBot state.
- Runtime readiness must include GPU, API, models, chat, and CUDA-offload evidence.
- API keys must not be passed as command-line arguments or written to logs.
- Profile/model changes are committed only after the replacement runtime is ready.
- Tests must use fake CLI processes and fake HTTP endpoints; no real Colab or Telegram calls.

### Task 1: Strengthen colab-t4 readiness and account JSON operations

**Files:**
- Modify: `colab_t4/lifecycle.py:208-213,230-253`
- Modify: `colab_t4/cli.py:308-338,419-480`
- Modify: `colab_t4/notebook.py:58-63,190-212,245-257`
- Modify: `colab_t4/config.py:112-128`
- Test: `tests/test_lifecycle.py`
- Test: `tests/test_app.py`

**Interfaces:**
- `lifecycle._ready_is_valid(ready: dict) -> bool` validates all required readiness flags.
- `colab-t4 accounts list --json` remains the machine-readable account registry consumed by AblitBot.
- `colab-t4 wait --wait-health` checks the runtime `/health` endpoint rather than only `/v1/models`.

- [ ] **Step 1: Write failing tests** for readiness rejection when models, CUDA offload, or SSH evidence is false; for `/health` polling; and for JSON account rows.
- [ ] **Step 2: Run the focused tests** with `pytest tests/test_lifecycle.py tests/test_app.py -q`; confirm failures describe missing validation.
- [ ] **Step 3: Implement readiness validation** and make `up`/`wait` reject incomplete artifacts.
- [ ] **Step 4: Implement real health polling** through the existing API helper and wire `--wait-health` to it.
- [ ] **Step 5: Remove the duplicate `save_secrets` definition** and add remote secret cleanup after notebook configuration is loaded.
- [ ] **Step 6: Run the focused tests** and confirm they pass.

### Task 2: Add a stable provider-facing account/model contract

**Files:**
- Create: `colab_t4/provider.py`
- Modify: `colab_t4/__init__.py`
- Test: `tests/test_provider.py`

**Interfaces:**
- `ModelSpec(alias: str, repository: str, quant: str, description: str)` is an immutable model definition.
- `available_models() -> tuple[ModelSpec, ...]` returns the reviewed model allowlist.
- `resolve_model(alias: str) -> ModelSpec` rejects unknown aliases.
- `account_rows() -> list[dict[str, object]]` returns sanitized account metadata.

- [ ] **Step 1: Write failing tests** for known model resolution, unknown alias rejection, and secret-free account rows.
- [ ] **Step 2: Run `pytest tests/test_provider.py -q`** and verify the expected missing-interface failures.
- [ ] **Step 3: Implement the immutable model registry** using the existing abliterated model choices as the initial allowlist.
- [ ] **Step 4: Implement account-row normalization** by reading the existing account registry without exposing token paths or credentials.
- [ ] **Step 5: Run the provider tests** and confirm they pass.

### Task 3: Replace AblitBot's duplicate profile logic with colab-t4 account consumption

**Files:**
- Modify: `/root/ablitbot/ablitbot.py:120-210,265-480`
- Modify: `/root/ablitbot/.env.example`
- Test: `/root/ablitbot/tests/test_backend.py`

**Interfaces:**
- `Backend.accounts() -> list[dict]` invokes `colab-t4 accounts list --json` using the configured environment.
- `Backend.candidate_profiles() -> list[str]` returns preferred-first, then CLI-ordered account IDs.
- `Backend.provision(profile: str) -> None` provisions the current model through the selected account environment.
- `Backend.recover() -> str` returns the successful profile or raises an aggregate error.

- [ ] **Step 1: Create failing backend tests** using a fake `subprocess.run` for account JSON, profile ordering, and successful second-profile recovery.
- [ ] **Step 2: Run the backend tests** and confirm failures because AblitBot currently uses `profiles.json` and has no recovery rotation.
- [ ] **Step 3: Implement account JSON consumption** and remove automatic discovery of independent profile-home roots.
- [ ] **Step 4: Implement preferred-first recovery** while preserving the bot's selected profile if all attempts fail.
- [ ] **Step 5: Make `ensure()` call `recover()` under its existing lifecycle lock** and persist only the profile that becomes healthy.
- [ ] **Step 6: Run backend tests** and confirm they pass.

### Task 4: Make model and profile switching transactional and allowlisted

**Files:**
- Modify: `/root/ablitbot/ablitbot.py:297-303,759-824`
- Test: `/root/ablitbot/tests/test_backend.py`
- Test: `/root/ablitbot/tests/test_commands.py`

**Interfaces:**
- `Backend.set_model(alias: str) -> None` accepts only an allowlisted alias and does not persist until recovery succeeds.
- `Backend.set_profile(name: str) -> None` validates the account through `colab-t4` and does not persist until recovery succeeds.
- `/model list` and `/profile` display only sanitized provider metadata.

- [ ] **Step 1: Write failing tests** proving failed model/profile switches leave the previous preference unchanged.
- [ ] **Step 2: Run the focused command tests** and confirm current setters persist before restart.
- [ ] **Step 3: Implement staged preference changes** and commit them only after a healthy replacement runtime.
- [ ] **Step 4: Replace arbitrary repository input** with alias-only model selection.
- [ ] **Step 5: Run command tests** and confirm they pass.

### Task 5: Add local profile management commands

**Files:**
- Modify: `/root/ablitbot/ablitbot.py` CLI parser and command handlers
- Test: `/root/ablitbot/tests/test_commands.py`

**Interfaces:**
- `ablitbot profile list` prints account IDs and sanitized status.
- `ablitbot profile add --id NAME` delegates to `colab-t4 accounts add --id NAME`.
- `ablitbot profile remove NAME` delegates to `colab-t4 accounts remove NAME` with explicit confirmation handling.

- [ ] **Step 1: Write failing parser/command tests** for list, add, and remove delegation.
- [ ] **Step 2: Run them** and verify the commands are absent.
- [ ] **Step 3: Implement the local CLI surface** with non-secret subprocess output.
- [ ] **Step 4: Run the command tests** and confirm they pass.

### Task 6: Verify the complete provider workflow

**Files:**
- Modify: `README.md`
- Modify: `/root/ablitbot/.env.example`
- Test: `tests/test_lifecycle.py`
- Test: `/root/ablitbot/tests/test_backend.py`

- [ ] **Step 1: Run syntax checks** for both repositories.
- [ ] **Step 2: Run all available pytest suites** in both repositories.
- [ ] **Step 3: Run CLI help and JSON-contract smoke checks** with fake executables where needed.
- [ ] **Step 4: Audit each objective requirement**: provider use, start/stop, profile add/switch, predefined model selection, and cross-profile recovery.
- [ ] **Step 5: Document external prerequisites** for Colab OAuth, Tailscale, Telegram, and the `colab-t4` installation.
