# colab-t4 Web Control Plane Design

**Date:** 2026-08-30  
**Status:** Approved for implementation planning

## Goal

Expose the safe, useful surface of the `colab-t4` CLI through the existing
Tailscale-bound web interface. The CLI remains supported, and both interfaces
must use the same operation and validation logic.

## Scope

The web control plane will support:

- Runtime lifecycle: up, readiness wait, restart, down, and live progress.
- Configuration: view non-secret settings, edit settings, securely enter
  secrets, validate, persist, and reset.
- Accounts: list, add through OAuth, finish/cancel OAuth, import tokens, select
  an account for the next operation, and remove accounts.
- Diagnostics: status, doctor, redacted logs, API model listing, and a bounded
  chat smoke test.
- Proxy visibility: backend order, health, active backend, and endpoint URLs.
- Job history: operation IDs, status, progress, redacted output, errors, and
  cancellation where the underlying operation supports it.

Arbitrary browser shell execution is out of scope. SSH remains available from
the CLI; the web UI may expose constrained diagnostics only.

## Architecture

Introduce a shared service layer with structured operation results and progress
events. CLI handlers and web handlers call this layer instead of duplicating
workflow logic. Long-running operations execute in background jobs held by a
small in-process job manager; the API exposes job polling and cancellation.

The existing `ThreadingHTTPServer` remains the transport. Management routes are
added under `/api/`, while `/v1/*` and `/health` retain their proxy behavior.
The dashboard remains a bundled HTML/JavaScript page for now.

## Security

- The service binds to the host's Tailscale address by default for management
  use; binding to all interfaces is not the default.
- Secret values are accepted write-only, validated server-side, persisted with
  the existing mode `0600` protections, and never returned in API responses.
- OAuth authorization codes and imported tokens are not written to job history
  or logs. Error and progress text is redacted before storage or display.
- Destructive actions require an explicit confirmation in the UI and server
  endpoints validate exact account/runtime targets.
- No arbitrary shell command endpoint is exposed in the browser.

## API shape

The management API will provide structured endpoints for state, jobs,
configuration, accounts, OAuth flow steps, diagnostics, and lifecycle actions.
Existing `/api/state`, `/api/up`, `/api/restart`, `/api/down`, and account delete
routes remain backward compatible.

Every mutating or long-running endpoint returns either a direct result or a job
object with an ID. `GET /api/jobs/<id>` is the canonical polling endpoint.

## UI

The dashboard will be organized into Overview, Runtime, Accounts,
Configuration, Diagnostics, API Tests, Logs, and Jobs. Runtime and account
actions show progress, success, and failure states. Configuration forms clearly
distinguish ordinary values from write-only secrets and provide reset/confirm
flows without echoing secret values.

## Failure handling

Operations report actionable structured errors and preserve the last known
state. A failed background job must not make the server unavailable. In-flight
jobs are marked interrupted if the process exits. Backend health and lifecycle
state remain independently visible.

## Testing

Add unit and HTTP integration tests for:

- Shared service results and validation.
- Job creation, polling, failure, and redaction.
- Configuration updates, reset, permissions, and secret non-disclosure.
- OAuth start/finish/cancel and token import without persistence leaks.
- Runtime, diagnostics, API-test, and account endpoints.
- Existing proxy behavior and all current CLI tests.
