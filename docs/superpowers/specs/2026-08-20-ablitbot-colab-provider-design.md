# AblitBot Colab Provider Design

## Goal

AblitBot uses `colab-t4` as its managed OpenAI-compatible model provider. It can
start and stop a T4 runtime, choose from a predefined set of abliterated or
uncensored GGUF models, manage Google Colab profiles, and automatically recover
from an unhealthy runtime by provisioning the same model with another profile.

## Current Evidence

- `colab-t4` already provisions a T4, stores runtime state, and rotates its own
  registered accounts during `up`.
- AblitBot already calls the `colab-t4` CLI, supports `/up`, `/down`, `/model`,
  and `/profile`, and has a local Ollama fallback.
- AblitBot keeps a second profile registry in `profiles.json`; this duplicates
  the account registry in `colab-t4` and prevents cross-profile recovery.
- AblitBot's `Backend.ensure()` only retries the currently selected profile.

## Architecture

`colab-t4` remains the lifecycle and Google-account authority. AblitBot calls
the CLI with an isolated `COLAB_T4_STATE_DIR` for the selected account and reads
the resulting state and runtime API key from that directory. AblitBot retains
only bot preferences (selected account, selected model, and quantization) in its
own state file.

The provider has three responsibilities:

1. Resolve the configured model alias to an allowlisted repository and quant.
2. Execute lifecycle operations against one account at a time.
3. Recover by trying candidate accounts in deterministic order and persisting
   the account that becomes healthy.

## Model Contract

Models are selected by aliases from a static allowlist. Each entry contains:

- alias;
- Hugging Face repository;
- exact GGUF quantization;
- human-readable description.

Arbitrary repository input is removed from the bot control surface. This keeps
model downloads predictable and limits the bot to reviewed abliterated or
uncensored model choices.

## Profile Contract

The bot reads profiles from `colab-t4 accounts list --json`. Profile creation
and removal remain local administrative operations because Colab OAuth requires
interactive terminal input. The bot exposes profile listing and selection; a
local `ablitbot profile add` command delegates to `colab-t4 accounts add`.

The selected profile is a preference, not a permanent pin. During recovery,
the selected profile is tried first, followed by healthy accounts in
least-recently-used order, followed by accounts with previous failures. A
successful recovery updates the bot preference to the successful account.

## Recovery Flow

For every request that needs the provider:

1. Perform a bounded health check against the recorded runtime API.
2. If healthy, send the request.
3. If unhealthy, serialize recovery so only one lifecycle operation runs.
4. Try the selected profile and current model.
5. If provisioning or readiness fails, try the next profile.
6. Persist the successful profile and only then allow chat requests.
7. If all profiles fail, return a clear unavailable response and leave the
   last failure details in status output.

Model and profile changes use the same flow. The preference is written only
after the replacement runtime is ready; failed changes do not silently point
the bot at an unusable configuration.

## CLI / Bot Surface

Existing controls remain:

- `/up` starts or recovers the provider;
- `/down` stops the selected runtime;
- `/model list` lists reviewed models;
- `/model <alias>` switches models;
- `/profile` lists profiles;
- `/profile <name>` selects a profile.

Add local commands:

- `ablitbot profile list`;
- `ablitbot profile add --id NAME`;
- `ablitbot profile remove NAME`.

Telegram control commands remain admin-only. Chat traffic never accepts an
arbitrary model repository or Google profile path.

## State and Security

- `colab-t4` owns account credentials, account health, runtime state, and API
  keys.
- AblitBot stores only non-secret preferences in its state file.
- API keys are never passed as process arguments or written into bot logs.
- Remote provisioning removes the uploaded secret file after loading it.
- Runtime readiness requires the expected GPU, API, model, chat smoke test, and
  CUDA-offload evidence.

## Testing

Add tests for:

- parsing `colab-t4 accounts list --json`;
- model alias allowlisting;
- selected-profile-first recovery ordering;
- successful failover persisting the replacement profile;
- all-profile failure preserving the previous preference;
- transactional model/profile changes;
- `/health` readiness and API failure recovery;
- local profile command construction.

The real Colab CLI and network remain integration dependencies; unit tests use
fake CLI processes and fake HTTP endpoints.
