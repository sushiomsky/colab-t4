# MCP Model Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add in-place GGUF model switching and a Hermes-compatible stdio MCP server to `colab-t4`.

**Architecture:** Keep the existing lifecycle manager authoritative for Colab/Tailscale/session state. Add a focused model-switch module that executes a remote switch notebook against the recorded session, and a thin MCP adapter that calls Python functions directly instead of shelling out.

**Tech Stack:** Python 3.9+ base package, MCP Python SDK v2 on Python 3.10+, `google-colab-cli` 0.6.0 adapter, `llama_cpp.server`, Hugging Face CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-mcp-model-switch-design.md`

## Global Constraints

- Existing `up/down/wait/restart`, account rotation, Tailscale and OpenAI-compatible API behavior must remain compatible.
- Model switching must not recreate the Colab session.
- Download/validation failure must leave the old server untouched.
- Startup/health failure after server stop must attempt rollback to the old model.
- MCP responses must not expose API/HF/Tailscale/SSH/OAuth secrets or account HOME paths.
- stdio is the only MCP transport in v0.4.0.
- Base installation remains Python 3.9 compatible without the MCP optional dependency.
- MCP integration targets SDK v2 (`MCPServer`) on Python 3.10+ and may accept v1 through a compatibility import fallback.

---

### Task 1: Model switch core

**Files:**
- Create: `tests/test_model.py`
- Create: `colab_t4/model.py`

**Interfaces:**
- Produces: `current_model() -> dict[str, object]`
- Produces: `switch_model(model_repo: str, quant: str, timeout: float = 3600) -> dict[str, object]`
- Produces: `build_switch_script() -> str`

- [x] **Step 1: Write failing tests** for current-model reporting, not-ready rejection, account-specific CLI HOME, success-state update, rollback-preserves-old-model, rollback-failure state, and script security/verification markers.
- [x] **Step 2: Verify RED** because `colab_t4.model` is missing.
- [x] **Step 3: Implement** `model.py` with mode-0600 switch config, `ColabCLI.upload/exec/download`, remote GGUF download/validation, delayed server stop, health/chat/CUDA verification, rollback, and atomic readiness update.
- [x] **Step 4: Verify focused/full GitHub CI passes.**
- [x] **Step 5: Commit** `feat: add in-place model switching`.

### Task 2: CLI model commands

**Files:**
- Create: `tests/test_model_cli.py`
- Modify: `colab_t4/cli.py`

**Interfaces:**
- Consumes: `current_model`, `switch_model`
- Produces CLI: `colab-t4 model current [--json]`
- Produces CLI: `colab-t4 model switch REPOSITORY [--quant QUANT] [--timeout SECONDS] [--json]`

- [x] **Step 1: Write parser/handler tests** proving argument parsing, delegation and JSON output.
- [x] **Step 2: Verify RED** because the `model` command is absent.
- [x] **Step 3: Add** `cmd_model`, nested model subparsers and handler registration.
- [x] **Step 4: Verify focused/full GitHub CI passes.**
- [x] **Step 5: Commit** `feat: expose model switching in CLI`.

### Task 3: MCP adapter

**Files:**
- Create: `tests/test_mcp_server.py`
- Create: `colab_t4/mcp_server.py`

**Interfaces:**
- Consumes lifecycle `up/down/restart`, state/model/account functions
- Produces MCP tools: `backend_status`, `backend_start`, `backend_stop`, `backend_restart`, `model_current`, `model_switch`, `api_info`, `accounts_list`
- Produces: `build_server()` and `main() -> int`

- [ ] **Step 1: Write failing tests** for safe payload helpers, lifecycle/model delegation, and no-secret/no-HOME output.
- [ ] **Step 2: Verify RED** because the MCP adapter is missing.
- [ ] **Step 3: Implement** current MCP v2 `MCPServer` stdio integration with guarded imports and an optional v1 `FastMCP` fallback.
- [ ] **Step 4: Run base tests without MCP plus MCP integration tests on Python 3.10+ with the extra installed.
- [ ] **Step 5: Commit** `feat: add Hermes MCP server`.

### Task 4: Packaging, documentation and release metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `colab_t4/__init__.py`
- Modify: `README.md`
- Modify: `.github/workflows/test.yml`

**Interfaces:**
- Produces package version `0.4.0`
- Keeps base `requires-python = ">=3.9"`
- Produces optional dependency `mcp = ["mcp>=2,<3; python_version >= '3.10'"]`
- Produces script `colab-t4-mcp = "colab_t4.mcp_server:main"`

- [ ] **Step 1: Add packaging/entrypoint assertions** before metadata changes.
- [ ] **Step 2: Verify those assertions fail.**
- [ ] **Step 3: Update** version, conditional MCP dependency, script entrypoint, README Hermes configuration/model-switch examples, and split base/MCP CI jobs.
- [ ] **Step 4: Verify** base tests on Python 3.9/3.11/3.13 and MCP integration on Python 3.11.
- [ ] **Step 5: Commit** `release: prepare colab-t4 0.4.0`.

### Task 5: Final verification

- [ ] Compare branch against `main` and inspect every changed file for scope/security regressions.
- [ ] Verify no literal credentials or tokens appear in the diff.
- [ ] Verify GitHub Actions is green across the supported matrix.
- [ ] Update the draft PR with architecture, rollback guarantees, MCP tools, Hermes configuration and verification evidence; mark ready for review.
