# MCP Model Switch Design

## Goal

Expose `colab-t4` as a Hermes-friendly MCP backend and add in-place model switching for an already-running Colab T4 runtime without recreating the Colab session.

## Scope

The release will add:

- `colab-t4 model current`
- `colab-t4 model switch <hf-repo> --quant <quant>`
- `colab-t4-mcp`, a stdio MCP server for Hermes and other MCP clients
- MCP tools for backend lifecycle, status, account listing, API information, current model, and model switching
- rollback to the previous model if a switch fails after the old server has been stopped
- state/readiness updates and smoke tests after every successful switch
- secret-safe MCP responses

Non-goals:

- running multiple models concurrently on one T4
- replacing the OpenAI-compatible `llama_cpp.server`
- HTTP/SSE MCP transport in this release
- automatic model recommendation or arbitrary Hugging Face model conversion
- changing the existing Google-account rotation semantics

## Architecture

The existing lifecycle remains authoritative for Colab session creation, account rotation, provisioning, Tailscale, and runtime state. Model switching is a separate operation against the recorded running session.

A new `colab_t4/model.py` module owns model-switch orchestration. It generates a non-secret remote notebook, uploads a separate mode-0600 switch configuration to the active Colab session, executes the notebook through the existing `ColabCLI` adapter, downloads machine-readable switch/readiness artifacts, and updates local state only after remote verification succeeds.

The remote switch code performs these steps:

1. Validate the requested Hugging Face repository and quantization selector.
2. Download the requested GGUF into a model-specific directory using `hf download`.
3. Require exactly one matching GGUF and reject files larger than 11 GiB.
4. Preserve the currently running model path and readiness metadata for rollback.
5. Stop the current `llama_cpp.server` only after the new model download has succeeded.
6. Start the new server with the same API key, port, context size, CUDA offload settings, and `local` alias.
7. Verify authenticated `/v1/models`, a chat completion, and CUDA-offload evidence in the server log.
8. If startup or verification fails, restart the previous model and report failure.
9. On success, atomically replace `/content/.colab-t4-ready.json` with the new model/quant metadata and report success.

The local model switch operation never prints API keys, Hugging Face tokens, Tailscale credentials, SSH credentials, or the uploaded secret file.

## Model switch interface

`colab_t4.model.current_model() -> dict[str, object]` returns recorded non-secret model metadata without contacting Hugging Face.

`colab_t4.model.switch_model(model_repo: str, quant: str, timeout: float = 3600) -> dict[str, object]` requires a recorded ready session. It uses the account-specific Colab CLI HOME already recorded in lifecycle state, uploads the switch configuration, executes the switch notebook, downloads the result/readiness artifacts, validates success, and persists updated model metadata.

The CLI exposes:

```text
colab-t4 model current [--json]
colab-t4 model switch REPOSITORY [--quant Q4_K_M] [--timeout 3600] [--json]
```

## MCP server

`colab_t4/mcp_server.py` uses the official Python MCP SDK through an optional dependency group. As of August 2026, MCP Python SDK v2 is the stable line, exposes `MCPServer` from `mcp.server`, and requires Python 3.10+. The implementation therefore targets v2 natively while optionally accepting an already-installed v1 SDK through a compatibility import fallback.

The base `colab-t4` package continues to support Python 3.9. The MCP extra is only installed on Python 3.10+, and invoking `colab-t4-mcp` without a supported MCP SDK produces a direct installation/runtime hint instead of a traceback.

Entry point:

```text
colab-t4-mcp
```

Transport is stdio only for v0.4.0; `mcp.run()` is invoked without a transport override because stdio is the SDK default.

Tools:

- `backend_status()` — current lifecycle/API/model status without probing secret-bearing config
- `backend_start(model: str | None, quant: str | None)` — start or recover the backend using the existing lifecycle path
- `backend_stop()` — stop the recorded runtime
- `backend_restart(model: str | None, quant: str | None)` — restart through the lifecycle manager
- `model_current()` — return current model metadata
- `model_switch(model: str, quant: str = "Q4_K_M")` — switch model in place
- `api_info()` — return API base URL and model alias, never the bearer token
- `accounts_list()` — return account IDs, email labels, health/failure metadata, and usage ordering without OAuth tokens or HOME paths

All tools return JSON-serializable dictionaries. Exceptions are converted to concise redacted errors.

## Packaging

Version becomes `0.4.0`.

`pyproject.toml` keeps `requires-python = ">=3.9"` and adds a conditional MCP extra:

```toml
[project.optional-dependencies]
mcp = ["mcp>=2,<3; python_version >= '3.10'"]
test = ["pytest>=8"]

[project.scripts]
colab-t4 = "colab_t4.cli:main"
colab-t4-mcp = "colab_t4.mcp_server:main"
```

CI verifies the base package on Python 3.9/3.11/3.13 and verifies MCP integration separately on Python 3.11 with the `mcp` extra installed.

## Error handling and rollback

A model switch must fail before touching the running server when download or GGUF validation fails.

If the new server cannot become healthy after the old server has been stopped, the switch code attempts to restart the old model. The command reports the switch as failed even when rollback succeeds; the readiness artifact continues to identify the previous model.

If rollback also fails, local state is marked `failed` with a redacted `last_error`, while preserving the Colab session/account identifiers so `restart` or `up` can recover through the existing account failover behavior.

## Security

- MCP output never contains `api_key`, `hf_token`, `tailscale_authkey`, `ssh_password`, OAuth token paths, account HOME paths, or raw secret configuration.
- Remote switch configuration is uploaded as a mode-0600 temporary JSON file and local temporary files are removed after execution.
- Existing `redact()` behavior is used for local diagnostics.
- API information exposed to Hermes contains only the OpenAI-compatible base URL and model alias `local`.

## Tests

Unit tests cover:

- current-model reporting
- refusal to switch without a ready recorded runtime
- account-specific Colab CLI selection
- successful remote switch updates local model/quant/readiness state
- failed switch preserves previous model when rollback succeeds
- failed switch marks state failed if rollback fails
- generated remote switch source keeps secrets outside source and contains health/chat/CUDA verification
- CLI parsing and JSON output for `model current` and `model switch`
- MCP tool registration and delegation
- MCP responses omit secrets and account HOME paths
- base package remains importable without MCP installed
- MCP entry point works with the current v2 SDK on Python 3.10+

Existing lifecycle, failover, account, wizard, and CLI tests must remain green.
