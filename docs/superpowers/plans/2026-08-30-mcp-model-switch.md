# MCP Model Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add in-place GGUF model switching and a Hermes-compatible stdio MCP server to `colab-t4`.

**Architecture:** Keep the existing lifecycle manager authoritative for Colab/Tailscale/session state. Add a focused model-switch module that uploads and executes a remote switch script against the recorded session, and a thin MCP adapter that calls Python functions directly instead of shelling out.

**Tech Stack:** Python 3.9+, `google-colab-cli` 0.6.0 adapter, `llama_cpp.server`, Hugging Face CLI, pytest, Python MCP SDK.

**Spec:** `docs/superpowers/specs/2026-08-30-mcp-model-switch-design.md`

## Global Constraints

- Existing `up/down/wait/restart`, account rotation, Tailscale and OpenAI-compatible API behavior must remain compatible.
- Model switching must not recreate the Colab session.
- Download/validation failure must leave the old server untouched.
- Startup/health failure after server stop must attempt rollback to the old model.
- MCP responses must not expose API/HF/Tailscale/SSH/OAuth secrets.
- stdio is the only MCP transport in v0.4.0.
- Base installation remains usable without the MCP optional dependency.

---

### Task 1: Model switch core

**Files:**
- Create: `tests/test_model.py`
- Create: `colab_t4/model.py`

**Interfaces:**
- Produces: `current_model() -> dict[str, object]`
- Produces: `switch_model(model_repo: str, quant: str, timeout: float = 3600) -> dict[str, object]`
- Produces: `build_switch_script() -> str`

- [ ] **Step 1: Write failing tests** for current-model reporting, not-ready rejection, account-specific CLI HOME, success-state update, rollback-preserves-old-model, rollback-failure state, and script security/verification markers.
- [ ] **Step 2: Run** `pytest tests/test_model.py -v` and verify failures are due to the missing module/functions.
- [ ] **Step 3: Implement** `model.py` with temporary mode-0600 config/script files, `ColabCLI.upload/exec/download`, remote GGUF download/validation, delayed server stop, health/chat/CUDA verification, rollback, and atomic readiness update.
- [ ] **Step 4: Run** `pytest tests/test_model.py -v` and then the full suite.
- [ ] **Step 5: Commit** `feat: add in-place model switching`.

### Task 2: CLI model commands

**Files:**
- Create: `tests/test_model_cli.py`
- Modify: `colab_t4/cli.py`

**Interfaces:**
- Consumes: `current_model`, `switch_model`
- Produces CLI: `colab-t4 model current [--json]`
- Produces CLI: `colab-t4 model switch REPOSITORY [--quant QUANT] [--timeout SECONDS] [--json]`

- [ ] **Step 1: Write failing parser/handler tests** proving argument parsing, delegation and JSON output.
- [ ] **Step 2: Run** `pytest tests/test_model_cli.py -v` and verify the `model` command is absent.
- [ ] **Step 3: Add** `cmd_model` plus nested model subparsers and register the handler.
- [ ] **Step 4: Run** focused and full tests.
- [ ] **Step 5: Commit** `feat: expose model switching in CLI`.

### Task 3: MCP adapter

**Files:**
- Create: `tests/test_mcp_server.py`
- Create: `colab_t4/mcp_server.py`

**Interfaces:**
- Consumes lifecycle `up/down/restart`, CLI-safe status data, `current_model`, `switch_model`, `load_accounts`, `load_state`
- Produces MCP tools: `backend_status`, `backend_start`, `backend_stop`, `backend_restart`, `model_current`, `model_switch`, `api_info`, `accounts_list`
- Produces: `main() -> int`

- [ ] **Step 1: Write failing tests** for tool registration/delegation, no-secret account/API payloads, and a clear missing-MCP installation error.
- [ ] **Step 2: Run** `pytest tests/test_mcp_server.py -v` and verify failures reflect the missing adapter.
- [ ] **Step 3: Implement** a FastMCP stdio server when `mcp` is installed; keep imports guarded so `import colab_t4` works without MCP.
- [ ] **Step 4: Run** focused and full tests.
- [ ] **Step 5: Commit** `feat: add Hermes MCP server`.

### Task 4: Packaging, documentation and release metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `colab_t4/__init__.py`
- Modify: `README.md`
- Create: `.github/workflows/test.yml`

**Interfaces:**
- Produces package version `0.4.0`
- Produces optional dependency `mcp = ["mcp>=1.0"]`
- Produces script `colab-t4-mcp = "colab_t4.mcp_server:main"`

- [ ] **Step 1: Add packaging/entrypoint assertions** to existing tests before metadata changes.
- [ ] **Step 2: Verify those assertions fail.**
- [ ] **Step 3: Update** version, optional dependencies, scripts, README Hermes configuration example, model-switch examples and CI workflow.
- [ ] **Step 4: Run** `pytest -q`, `python -m compileall colab_t4`, and CLI help/version smoke tests.
- [ ] **Step 5: Commit** `release: prepare colab-t4 0.4.0`.

### Task 5: Final verification

- [ ] Compare branch against `main` and inspect every changed file for scope/security regressions.
- [ ] Verify no literal credentials or tokens appear in the diff.
- [ ] Verify all tests pass on Python 3.9+ CI.
- [ ] Create a PR describing architecture, rollback guarantees, MCP tools and verification evidence.
