# Web Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the safe, useful `colab-t4` CLI surface through the Tailscale-bound web interface with shared operations, secure configuration, account authentication, diagnostics, and tracked jobs.

**Architecture:** Add a small shared service layer and in-process job manager beneath both CLI and HTTP handlers. Extend the existing `ThreadingHTTPServer` JSON API and bundled HTML UI while preserving `/v1/*`, `/health`, and existing management routes.

**Tech Stack:** Python 3.9+, standard library HTTP server/threading/json, pytest, existing `colab_t4` config/accounts/lifecycle modules.

**Spec:** `docs/superpowers/specs/2026-08-30-web-control-plane-design.md`

## Global Constraints

- The CLI remains supported, and both interfaces use the same operation and validation logic.
- Secret values are accepted write-only, persisted with existing mode `0600` protections, and never returned in API responses.
- OAuth codes/tokens are not written to job history or logs; progress and errors are redacted.
- No arbitrary browser shell endpoint is exposed.
- Existing `/api/state`, `/api/up`, `/api/restart`, `/api/down`, and account delete routes remain backward compatible.
- Existing proxy behavior and all current CLI tests must remain passing.

## File Map

- Create `colab_t4/services.py`: shared structured operations used by CLI and web handlers.
- Create `colab_t4/jobs.py`: bounded in-process job registry, polling, redaction, and cancellation.
- Modify `colab_t4/cli.py`: route existing commands through shared operations where behavior is currently duplicated.
- Modify `colab_t4/router.py`: add authenticated management API, job routes, diagnostics/config/account routes, and the expanded UI.
- Modify `colab_t4/config.py`: expose validated non-secret configuration summaries and safe update/reset helpers.
- Modify `colab_t4/accounts.py` and `colab_t4/wizard.py`: add service-friendly account/OAuth/config operations without printing secrets.
- Extend `tests/test_router.py`, `tests/test_jobs.py`, `tests/test_services.py`, `tests/test_config.py`, and account tests.
- Update `README.md` with the web API/UI capabilities and tailnet security requirements.

### Task 1: Add the shared job manager

**Files:**
- Create: `colab_t4/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- `JobManager.submit(name: str, target: Callable[[JobContext], Any]) -> Job`
- `JobManager.get(job_id: str) -> Job | None`
- `JobManager.cancel(job_id: str) -> bool`
- `Job.to_dict() -> dict[str, Any]` with `id`, `name`, `status`, `progress`, `result`, `error`, `created_at`, `updated_at`.
- `JobContext.progress(message: str, percent: int | None = None) -> None`

- [ ] **Step 1: Write failing tests** for queued/running/succeeded jobs, captured exceptions, cancellation flags, and secret redaction from progress/error/result text.
- [ ] **Step 2: Run `PYTHONPATH=. pytest -q tests/test_jobs.py` and verify the missing module/API failures.**
- [ ] **Step 3: Implement the manager with daemon worker threads, a lock-protected dictionary, bounded history of the latest 50 jobs, and redaction through `config.redact`.**
- [ ] **Step 4: Run the focused tests and verify all pass.**
- [ ] **Step 5: Commit with `git add colab_t4/jobs.py tests/test_jobs.py && git commit -m "feat: add management job manager"`.**

### Task 2: Create shared service operations

**Files:**
- Create: `colab_t4/services.py`
- Modify: `colab_t4/cli.py`
- Test: `tests/test_services.py`

**Interfaces:**
- `runtime_up(options: Any, progress: Callable[[str, int | None], None] | None = None) -> dict[str, Any]`
- `runtime_down() -> dict[str, Any]`
- `runtime_restart(options: Any, progress: Callable[[str, int | None], None] | None = None) -> dict[str, Any]`
- `runtime_status() -> dict[str, Any]`
- `run_doctor(interactive: bool = False) -> tuple[dict[str, Any], int]`
- `api_models() -> dict[str, Any]`
- `api_chat(message: str) -> dict[str, Any]`

- [ ] **Step 1: Add tests proving service functions return structured results and preserve lifecycle/API error semantics.**
- [ ] **Step 2: Run `PYTHONPATH=. pytest -q tests/test_services.py` and verify failures before implementation.**
- [ ] **Step 3: Implement thin service adapters over existing lifecycle and CLI API helpers; pass progress callbacks through lifecycle polling where available.**
- [ ] **Step 4: Update CLI handlers to call the adapters without changing command output or exit codes.**
- [ ] **Step 5: Run service tests plus the existing CLI/lifecycle tests.**
- [ ] **Step 6: Commit the shared operation layer.**

### Task 3: Add secure web configuration and account/OAuth APIs

**Files:**
- Modify: `colab_t4/config.py`
- Modify: `colab_t4/accounts.py`
- Modify: `colab_t4/wizard.py`
- Modify: `colab_t4/router.py`
- Test: `tests/test_config.py`, `tests/test_accounts.py`, `tests/test_router.py`

**Interfaces:**
- `configuration_summary() -> dict[str, Any]` returns non-secret values and boolean secret flags only.
- `update_configuration(values: dict[str, Any]) -> dict[str, Any]` validates allowlisted fields and persists with mode `0600`.
- `reset_configuration(confirmed: bool) -> None` clears saved configuration only after confirmation.
- HTTP `GET /api/config`, `PUT /api/config`, `POST /api/config/reset`.
- HTTP `POST /api/accounts`, `POST /api/accounts/auth-start`, `POST /api/accounts/auth-finish`, `POST /api/accounts/auth-cancel`.

- [ ] **Step 1: Write tests for non-secret summaries, allowlisted updates, secret non-disclosure, reset confirmation, OAuth flow state, token import redaction, and account removal target validation.**
- [ ] **Step 2: Run the focused tests and verify each new endpoint/helper fails for the intended missing behavior.**
- [ ] **Step 3: Implement config helpers using existing atomic JSON writers and permissions; reject unknown fields and empty required values.**
- [ ] **Step 4: Implement account/OAuth service adapters using existing account functions; keep pending codes in memory with expiry and never put them in job records.**
- [ ] **Step 5: Add HTTP handlers with JSON validation, explicit confirmation fields for destructive actions, and redacted errors.**
- [ ] **Step 6: Run all config/account/router tests and commit.**

### Task 4: Add jobs, diagnostics, and lifecycle API routes

**Files:**
- Modify: `colab_t4/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- HTTP `GET /api/jobs`, `GET /api/jobs/<id>`, `POST /api/jobs/<id>/cancel`.
- Existing lifecycle routes return `{ "job": { ... } }` while retaining accepted `202` responses.
- HTTP `GET /api/status`, `GET /api/doctor`, `GET /api/logs`, `GET /api/api/models`, `POST /api/api/chat`.

- [ ] **Step 1: Write HTTP tests for job polling, cancellation, lifecycle progress, doctor/status/log payloads, model listing, chat smoke tests, and redaction.**
- [ ] **Step 2: Run `PYTHONPATH=. pytest -q tests/test_router.py` and confirm the new route assertions fail.**
- [ ] **Step 3: Connect lifecycle routes to `JobManager` and service adapters; return stable JSON envelopes and `404` for unknown jobs/routes.**
- [ ] **Step 4: Add diagnostics routes with bounded log size, redacted content, and no secret file contents.**
- [ ] **Step 5: Run all HTTP tests and the full existing test suite.**
- [ ] **Step 6: Commit the API expansion.**

### Task 5: Expand the web UI to CLI parity

**Files:**
- Modify: `colab_t4/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- UI tabs: Overview, Runtime, Accounts, Configuration, Diagnostics, API Tests, Logs, Jobs.
- Browser helpers: `loadState`, `submitJson`, `pollJob`, `renderJob`, `showAlert`.

- [ ] **Step 1: Add HTML contract tests for each tab, write-only secret fields, OAuth controls, diagnostics/log panels, API test form, and job progress panel.**
- [ ] **Step 2: Run the UI tests and confirm the new labels/controls are absent.**
- [ ] **Step 3: Implement the tabs and forms using `textContent`/DOM creation for server data; never inject secret values into HTML.**
- [ ] **Step 4: Implement action handling with confirmation, disabled buttons while jobs run, polling every 1–3 seconds, retry, and clear errors.**
- [ ] **Step 5: Run router tests and manually verify `/manage` against the running Tailscale-bound server.**
- [ ] **Step 6: Commit the UI expansion.**

### Task 6: Harden tailnet exposure and document operation

**Files:**
- Modify: `colab_t4/router.py`
- Modify: `colab_t4/cli.py`
- Modify: `README.md`
- Test: `tests/test_router.py`, `tests/test_cli_wizard.py`

- [ ] **Step 1: Write tests for Tailscale-default binding resolution, explicit host override, request-size limits, method restrictions, CSRF/origin checks, and no arbitrary shell route.**
- [ ] **Step 2: Run the security-focused tests and confirm missing protections fail.**
- [ ] **Step 3: Implement safe defaults: resolve `tailscale ip -4` when available, require explicit opt-in for non-tailnet binds, cap request bodies, and validate management request origin/token.**
- [ ] **Step 4: Add README examples for configuration, account OAuth, jobs, diagnostics, and Tailscale access.**
- [ ] **Step 5: Run `PYTHONPATH=. pytest -q`, `git diff --check`, and a local health/UI smoke test.**
- [ ] **Step 6: Commit the hardening/documentation changes.**

## Final Verification

- [ ] Run the complete test suite with local socket access: `PYTHONPATH=. pytest -q`.
- [ ] Confirm no secret value appears in API responses, generated HTML, job records, or logs.
- [ ] Start the server on the Tailscale address and verify `/health`, `/manage`, `/api/state`, `/api/config`, and `/api/jobs`.
- [ ] Verify existing CLI commands still work and their exit codes/output remain compatible.
- [ ] Review `git diff --check` and report any environment-specific checks that could not run.
