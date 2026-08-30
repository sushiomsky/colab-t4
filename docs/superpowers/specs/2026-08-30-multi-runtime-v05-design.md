# colab-t4 v0.5.0 Native Multi-Runtime Design

Date: 2026-08-30
Status: Approved design, implementation not started
Target branch: `feat/multi-runtime-v05`

## Goal

Add native management of multiple independent Google Colab T4 runtimes while preserving the current v0.4.0 single-runtime behavior as the default.

A user must be able to keep named runtimes such as `coder` and `research` alive at the same time, address each one explicitly from the CLI or MCP, switch models independently, and stop/restart one runtime without mutating another runtime's state, API key, logs, or Colab session.

## Non-goals

v0.5.0 does not add:

- request load balancing or model routing;
- an inference scheduler;
- automatic agent-to-runtime assignment;
- automatic scale-up/scale-down based on traffic;
- cross-runtime request failover;
- a shared OpenAI-compatible proxy endpoint;
- parallel MCP execution across runtime mutations;
- runtime deletion/garbage collection beyond explicit lifecycle `down`.

Routing remains the responsibility of Hermes, OmniRoute, ablitbot, or another caller. `colab-t4` remains a lifecycle/backend primitive.

## Compatibility contract

Running commands without `--runtime` must continue to use the legacy v0.4.0 runtime state directly under the configured base state directory. Existing installations require no migration.

For the default installation this remains:

```text
~/.config/colab-t4/
├── state.json
├── runtime-api-key.json
├── secrets.json
├── accounts.json
├── accounts/
└── logs/
```

`--runtime default` is an explicit alias for this legacy/default runtime.

Existing environment variables and CLI commands keep their current meaning when no named runtime is selected.

## Runtime namespace layout

Named runtimes live below the existing base state directory:

```text
~/.config/colab-t4/
├── state.json                     # legacy/default runtime
├── runtime-api-key.json           # legacy/default runtime API key
├── secrets.json                   # global saved configuration/secrets
├── accounts.json                  # global account registry
├── accounts/                      # global isolated Colab OAuth homes
├── logs/                          # legacy/default runtime logs
└── runtimes/
    ├── coder/
    │   ├── runtime.json           # non-secret named-runtime metadata/defaults
    │   ├── state.json
    │   ├── runtime-api-key.json
    │   └── logs/
    └── research/
        ├── runtime.json
        ├── state.json
        ├── runtime-api-key.json
        └── logs/
```

The base state directory is the value of `COLAB_T4_STATE_DIR` at process startup, or `~/.config/colab-t4` when unset. Selecting a named runtime changes only the runtime-scoped state directory inside the current process. It must never relocate the global account registry or global saved configuration.

## Global state versus runtime state

The distinction is deliberate.

### Global

These resources are shared by all runtimes:

- `secrets.json` for common Tailscale/Hugging Face/SSH/default configuration;
- `accounts.json` for account health and least-recently-used rotation;
- `accounts/<id>/` for isolated Colab CLI OAuth/session homes.

Account health and usage timestamps must remain global so provisioning `coder` changes the same LRU information later used when provisioning `research`.

### Runtime-scoped

Each runtime owns:

- lifecycle `state.json`;
- `runtime-api-key.json`;
- runtime logs;
- `runtime.json` metadata for named runtimes.

A command targeting one named runtime must not read another runtime's state/API key/logs except the explicit `runtimes list` inventory operation.

## Runtime identifiers

Named runtime IDs use the same conservative shape as account IDs:

```text
^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$
```

Reserved ID: `default`.

IDs are used only as directory names after validation. Path separators, traversal components, whitespace, empty names, and absolute paths are rejected.

## Runtime metadata

`runtimes create NAME` creates the namespace directory with mode `0700`, generates a runtime API key with mode `0600`, and writes `runtime.json` atomically.

`runtime.json` contains non-secret defaults only, initially:

```json
{
  "id": "coder",
  "session": "colab-t4-coder",
  "model_repo": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF",
  "quant": "Q4_K_M"
}
```

The session default is derived from the runtime ID so two named runtimes never accidentally request the same Colab CLI session. The default runtime continues using the current legacy default session.

`runtimes create NAME --model REPOSITORY --quant QUANT` stores the requested model defaults. Explicit `up` flags or environment variables can still override them for a run.

Creating an already-existing runtime fails without changing files.

## Configuration precedence

For runtime-specific non-secret values such as session/model/quant, precedence becomes:

```text
explicit CLI option
-> environment variable
-> named runtime metadata
-> global saved default
-> interactive prompt
-> package default
```

For common secrets such as `TS_AUTHKEY` and `HF_TOKEN`, current precedence remains:

```text
explicit/environment
-> global saved secrets
-> interactive prompt
```

Named runtimes receive an independent generated API key at `runtimes create` time. For API authentication, precedence is:

```text
explicit CLI value
-> COLAB_T4_API_KEY environment variable
-> selected runtime's runtime-api-key.json
-> global saved/default API key where legacy compatibility requires it
-> generated/interactive value
```

The default runtime keeps the exact v0.4.0 behavior.

Persisting common secrets from an interactive named-runtime command must update only global common configuration. Runtime-specific session/model/quant defaults belong in `runtime.json`, not in the global account registry.

## CLI surface

Add a global selector:

```text
colab-t4 --runtime NAME <runtime-scoped-command>
```

`COLAB_T4_RUNTIME=NAME` is the automation equivalent. Explicit `--runtime` wins over the environment.

Runtime inventory commands:

```text
colab-t4 runtimes list [--json]
colab-t4 runtimes create NAME [--model REPOSITORY] [--quant QUANT] [--json]
```

Examples:

```text
colab-t4 runtimes create coder
colab-t4 runtimes create research --model QuantFactory/Mistral-Nemo-Instruct-2407-abliterated-GGUF --quant Q4_K_M

colab-t4 --runtime coder up
colab-t4 --runtime research up
colab-t4 --runtime coder status --json
colab-t4 --runtime research model switch REPOSITORY --quant Q4_K_M
colab-t4 --runtime coder ssh -- nvidia-smi
colab-t4 --runtime research down
```

Runtime-scoped commands are:

- `up`
- `status`
- `wait`
- `ssh`
- `logs`
- `api`
- `model`
- `restart`
- `down`

Global commands remain global:

- `runtimes`
- `accounts`
- `configure`
- `doctor`

Supplying a named `--runtime` to a global-only command must fail clearly instead of silently ignoring the selector.

## `runtimes list`

The list always includes `default` plus valid named runtime directories.

Each row exposes only non-secret information:

- runtime ID;
- recorded lifecycle state;
- session;
- account;
- model repository/path;
- quant;
- API base URL;
- whether an API key is configured (boolean only).

It never returns API keys or saved credentials.

The command must tolerate a corrupt runtime `state.json`: that runtime is listed as `unknown`/invalid rather than breaking the entire inventory.

## Runtime selection implementation

The implementation should reuse the tested v0.4 lifecycle rather than duplicate it.

Introduce a small runtime namespace module responsible for:

- validating runtime IDs;
- resolving the immutable process base state directory;
- mapping `default` to the legacy base state directory;
- mapping a named runtime to `<base>/runtimes/<id>`;
- loading/writing `runtime.json`;
- enumerating named runtimes;
- entering a selected runtime context for existing state/config/lifecycle code.

The existing lifecycle/model/API/SSH/log code continues to call the normal state helpers. During a selected runtime context those helpers resolve runtime-scoped state/API-key/log paths while global account/config helpers continue resolving against the immutable base directory.

This minimizes changes to the already-tested provisioning and rollback logic.

## Concurrency model

Separate CLI processes are naturally isolated because each process selects its own runtime namespace.

The stdio MCP server is one process, so temporarily selecting a namespace through process-local configuration must be serialized by an internal re-entrant lock/context manager. The lock covers namespace selection plus the complete tool operation and restores the previous selection in `finally`.

This prevents simultaneous MCP calls from causing `coder` to read or mutate `research` state.

v0.5.0 deliberately serializes runtime-targeted MCP operations even when a generic MCP host attempts parallel calls. Per-runtime parallel MCP execution is deferred until runtime context is passed explicitly through the lower-level APIs instead of process context.

## MCP surface

Preserve existing tool names for compatibility and add an optional `runtime` selector to runtime-scoped tools:

```text
backend_status(runtime=None)
backend_start(model=None, quant=None, runtime=None)
backend_stop(runtime=None)
backend_restart(model=None, quant=None, runtime=None)
model_current(runtime=None)
model_switch(model, quant="Q4_K_M", runtime=None)
api_info(runtime=None)
accounts_list()
runtimes_list()
```

Existing calls with no runtime argument behave exactly as in v0.4.0.

`runtimes_list` exposes the same non-secret inventory as the CLI. Runtime creation remains CLI-only in v0.5.0; MCP can operate only on existing namespaces. This avoids allowing an agent to silently create unbounded Colab runtime namespaces.

MCP responses continue to use explicit allow-listed schemas. No API key, HF token, Tailscale auth key, SSH password, OAuth token path, account HOME, or raw secret configuration is returned.

## Lifecycle isolation requirements

For any named runtime `A` and `B`:

- `up(A)` changes only A's runtime state/log/API-key files plus shared account-LRU metadata;
- `status(A)` never reports B's session/API endpoint/model;
- `model_switch(A)` uses A's recorded Colab session/account/API key and rollback metadata;
- `restart(A)` stops/recreates only A's recorded session;
- `down(A)` stops exactly A's recorded session and clears only A's runtime state/API key;
- `logs(A)` reads only A's logs;
- `ssh(A)` uses only A's Tailscale address;
- B remains usable if an A lifecycle operation fails.

Shared account-LRU updates are the only expected cross-runtime writes during lifecycle operations.

## Model switching

The v0.4 in-place switch/rollback behavior remains unchanged except that all state, API-key, log, readiness, and session resolution occurs inside the selected runtime namespace.

A successful or failed switch on `coder` must not mutate `research/runtime.json`, `research/state.json`, or `research/runtime-api-key.json`.

## Error handling

Errors identify the selected runtime where useful without leaking secrets.

Examples:

```text
error: runtime 'missing' does not exist; create it with `colab-t4 runtimes create missing`
error: invalid runtime id '../x'
error: runtime 'coder' has no recorded Tailscale address
```

A missing named runtime never falls back to `default`.

A corrupt named-runtime metadata file is a hard error for mutation commands and a non-fatal `invalid` entry for `runtimes list`.

Runtime context restoration occurs in `finally`, including when lifecycle/model operations throw.

## Security

- Validate runtime IDs before any path construction.
- Runtime directories are mode `0700`.
- `runtime.json`, `state.json`, and API-key files use existing atomic mode-`0600` writes.
- Runtime inventories expose only allow-listed non-secret fields.
- Global account OAuth homes remain outside runtime namespaces.
- Named runtime selection never changes account HOME paths.
- Error redaction must include both global secrets and the selected runtime API key.
- Tests must prove no runtime API key appears in CLI JSON inventory or MCP responses.

## Testing strategy

Implementation follows TDD.

### Namespace/config tests

Cover:

- valid/invalid runtime IDs;
- `default` mapping to the legacy directory;
- named directory mapping;
- custom startup `COLAB_T4_STATE_DIR` as the base root;
- create permissions and atomic metadata;
- duplicate create rejection;
- auto-derived unique session names;
- runtime metadata precedence;
- global account/config paths remaining global while runtime state paths change;
- runtime context restoration after exceptions.

### CLI tests

Cover:

- parser support for global `--runtime`;
- `COLAB_T4_RUNTIME` fallback;
- explicit selector precedence;
- `runtimes list/create` text and JSON output;
- missing runtime failure;
- global-only command selector rejection;
- default runtime backwards compatibility.

### Lifecycle isolation tests

Using mocked Colab CLI/lifecycle boundaries, prove:

- two named states can coexist;
- independent API keys and logs;
- `down(coder)` never stops the `research` session;
- restart/model switch use the selected session/account;
- failure in one runtime preserves the other state;
- global account LRU is shared.

### MCP tests

Cover:

- old no-argument calls still target default;
- every runtime-scoped MCP tool accepts a runtime;
- `runtimes_list` registration;
- selected runtime is restored after success/failure;
- concurrent tool attempts cannot cross-contaminate namespace selection;
- safe-state allow-listing and secret redaction for each runtime;
- accounts remain global.

### Packaging/CI

Keep the existing Python 3.9/3.11/3.13 base matrix and Python 3.11 MCP-v2 job. Add CLI smoke coverage for:

```text
colab-t4 runtimes --help
colab-t4 --runtime default status --json
```

No new runtime dependency is required.

## Documentation

README updates must document:

- the default-runtime compatibility contract;
- creating/listing named runtimes;
- running multiple named T4s;
- global account sharing and LRU behavior;
- runtime-specific API endpoints/API keys;
- Hermes MCP runtime selection examples;
- the deliberate absence of routing/load balancing in v0.5.0.

## Release definition

Target version: `0.5.0`.

v0.5.0 is complete when:

1. Existing no-runtime CLI/MCP behavior remains green.
2. At least two named runtime namespaces can coexist in tests without state/API-key/log crossover.
3. Runtime-specific lifecycle/model commands target only the selected recorded Colab session.
4. Account rotation remains global across named runtimes.
5. MCP can list and operate existing named runtimes without exposing secrets.
6. The full CI matrix passes on the final feature head.
7. The feature is documented and merged only after final verification.
