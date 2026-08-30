# colab-t4

Browserless lifecycle manager for a Google Colab NVIDIA T4 runtime. It uses the
installed Google Colab CLI, uploads a generated provisioning notebook and
secret configuration, executes it remotely, and verifies Tailscale,
passwordless Tailscale SSH, CUDA offload, and an authenticated OpenAI-compatible
API.

## Verified Colab CLI

This project targets `google-colab-cli` **0.6.0**, executable name `colab`.
The adapter uses only commands verified from the installed CLI:

```text
colab version
colab new --session NAME --gpu T4
colab upload --session NAME LOCAL_PATH REMOTE_PATH
colab exec --session NAME --file LOCAL_PATH --timeout SECONDS
colab status --session NAME
colab download --session NAME REMOTE_PATH LOCAL_PATH
colab stop --session NAME
colab sessions
```

`colab new --gpu T4` explicitly requests a T4. The remote notebook also fails
unless `nvidia-smi` reports an NVIDIA T4 with at least 14 GiB VRAM.

The CLI's OAuth authentication is owned by `colab`, not this project. It reads
`~/.config/colab-cli/token.json` or Application Default Credentials. This tool
never copies those credentials. On a clean machine, authenticate with the
installed Colab CLI's documented OAuth flow before running `colab-t4 doctor`.

## Browserless quickstart

All source and release files live under `/root/colab-t4`.

```bash
cd /root/colab-t4
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'

export TS_AUTHKEY='tskey-...'
# Optional: export HF_TOKEN='hf_...'
# Optional: export COLAB_T4_API_KEY='your API key'

colab-t4 doctor
colab-t4 up
colab-t4 status --json
colab-t4 wait --wait-health
colab-t4 ssh -- nvidia-smi
colab-t4 ssh -- curl -fsS http://127.0.0.1:8081/v1/models
colab-t4 api
colab-t4 logs all
colab-t4 down
```

`up` performs the complete workflow without a browser:

1. Validates `colab` and local authentication.
2. Requests a named T4 session with `colab new --gpu T4`.
3. Generates and uploads a notebook plus a separately uploaded, mode-0600
   provisioning secret file.
4. Executes the notebook with `colab exec`.
5. Downloads the machine-readable readiness artifact.
6. Records the session, GPU, Tailscale IP, API endpoint, and smoke-test results
   in `~/.config/colab-t4/state.json`.
The generated API key is kept in `~/.config/colab-t4/runtime-api-key.json`
with mode `0600` so later `status`, `wait`, and `api` commands can authenticate.
`down` removes it; Tailscale and SSH credentials are not persisted unless the
wizard's save option is selected.

The default model is:

```text
HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
```

with `Q6_K`. The default context is 16K, a practical starting point for a
16 GiB Tesla T4. Set `COLAB_T4_CTX` or pass `--ctx` explicitly; 32K may work
depending on the server build, while 64K is a best-effort setting because KV
cache memory can require CPU offload or fail on a 16 GiB T4. The notebook
reports `tool_calling` separately in its readiness tests after sending an
OpenAI-compatible native function-call request.

Provisioning attempts the pinned CUDA-enabled native `ggml-org/llama.cpp`
`llama-server` first (`COLAB_T4_RUNTIME=auto`, tested commit `b10345`). The
build is cached under `/content/llama-cache` and validated for `--jinja`. If
build or validation fails, the existing CUDA-enabled `llama-cpp-python`
server remains the automatic fallback. Use `COLAB_T4_RUNTIME=native` to make
native failure explicit, or `COLAB_T4_RUNTIME=python` to skip the attempt.
Override the tested commit with `LLAMA_CPP_COMMIT=<commit>`.

`colab-t4 status --json` reports the selected runtime, commit, native fallback
reason, and tested `tool_calling_mode`.

Override deterministically:

```bash
colab-t4 up \
  --model QuantFactory/Mistral-Nemo-Instruct-2407-abliterated-GGUF \
  --quant Q4_K_M
```

The requested quant must resolve to exactly one GGUF file. No arbitrary shard
or fallback quant is selected.

## Commands

```text
colab-t4 up       create, provision, and verify a T4 runtime
colab-t4 status   show lifecycle, GPU, Tailscale, SSH/API/model state
colab-t4 wait     bounded readiness wait; --wait-health checks API health
colab-t4 ssh      forward SSH options and remote commands unchanged
colab-t4 logs     show redacted local Colab/notebook/runtime logs
colab-t4 api      print API base URL and shell-safe usage
colab-t4 restart  stop the recorded session and create a new one
colab-t4 down     stop exactly the recorded session; idempotent local cleanup
colab-t4 doctor   validate Colab CLI, auth, SSH, and Tailscale configuration
colab-t4 serve    run the local OpenAI-compatible router (Colab/Ollama dispatch)
```

For integrations that manage account selection themselves, `up` and `restart`
accept `--account NAME --account-only`. The command then provisions only that
registered Google profile; a caller can implement deterministic recovery by
trying the next profile after a failed readiness check.

SSH uses Tailscale SSH by default:

```bash
colab-t4 up --ssh-mode tailscale
colab-t4 ssh -- uname -a
```

No SSH password or public key is uploaded in this mode. Access is controlled by
the tailnet ACL's `ssh` rule and the user's Tailscale identity. Password and
public-key modes remain available with `--ssh-mode password` or
`--ssh-mode key` when a tailnet policy cannot enable Tailscale SSH.

SSH examples:

```bash
colab-t4 ssh
colab-t4 ssh colab-t4
colab-t4 ssh -p 2222
colab-t4 ssh colab-t4 -p 2222
colab-t4 ssh -- -o BatchMode=yes
colab-t4 ssh colab-t4 -- -L 8081:127.0.0.1:8081
colab-t4 ssh -- uname -a
```

`status --json` is intended for automation. Exit code 0 means ready and the
API model-list check passed; 1 means unknown/not ready; 2 means a recorded
provisioning failure.
## Interactive first-run setup

`colab-t4 up` automatically enters the wizard when it is running with a TTY.
It uses the installed authentication operation exactly as exposed by
`google-colab-cli 0.6.0`: `colab sessions` triggers the remote OAuth
flow when credentials are missing (there is no `colab auth login` command).
The CLI displays Google's authorization URL/code flow and waits for completion,
then repeats `colab sessions` as the verification operation.

The wizard collects missing Tailscale, SSH, API, model, quantization, session,
and timeout values. Secret prompts use hidden terminal input. Entered secrets
are kept in memory for the current invocation by default:

```bash
colab-t4 up
```

At the end choose:

```text
1. This run only (recommended)
2. Save securely for future runs
```

Saved credentials are mode `0600` in:

```text
~/.config/colab-t4/secrets.json
```

Configuration management never reveals values:

```bash
colab-t4 configure --show
colab-t4 configure --reset
colab-t4 configure --reset --yes
```

Precedence is deterministic:

```text
explicit option -> environment -> saved configuration -> interactive prompt -> default
```

Useful environment variables are `TS_AUTHKEY`, `TS_TAILNET`, `HF_TOKEN`,
`COLAB_T4_API_KEY`, and `COLAB_T4_SSH_PASSWORD`. Use `--non-interactive` in
automation; missing requirements are reported together and no prompt is
attempted. Redirected stdin/stderr is automatically treated as non-interactive.

For direct API verification:

```bash
colab-t4 api models
colab-t4 api chat --message 'Reply with exactly: COLAB_T4_OK'
```


## Multiple Google accounts

`up` provisions from a rotation of Google accounts. If the first account
cannot create a T4 session (quota exhausted, entitlement rejected, Google
temporarily unavailable), it automatically tries the next account, and so on,
until one succeeds. A second T4 for a different session is allocated to the
least-recently-used account, so concurrent runtimes spread across profiles
instead of stacking on one account.

Accounts are managed with the `accounts` command:

```bash
colab-t4 accounts list            # current rotation, health, and usage
colab-t4 accounts add --id work   # interactive Colab OAuth flow for a new profile
colab-t4 accounts add --id personal --email you@gmail.com
colab-t4 accounts remove work     # unregister and delete the profile
```

For automation, integrations, and tools that cannot drive an interactive browser
flow, `auth-import` registers a profile from an existing Colab CLI `token.json`:

```bash
colab-t4 accounts auth-import --id work --token ~/.config/colab-cli/token.json
colab-t4 accounts auth-import --id work --token token.json --email you@gmail.com
```

The token is copied into the account's isolated HOME, verified with
`colab sessions`, and its Google account email is resolved automatically.
If verification fails the account directory is cleaned up and no profile is
registered. This is the non-interactive counterpart to `accounts add`: use it
when a token.json was obtained from a separate browser-based OAuth session or
captured from another Colab CLI installation.

Each added account gets its own isolated HOME directory
(`~/.config/colab-t4/accounts/<id>`), so its Colab OAuth token, session
registry, and history are completely independent of the other accounts.

`accounts add` runs the installed Colab CLI's remote copy-paste OAuth flow:
it prints an authorization URL, you open it in any browser, sign in as the
Google account to add, approve, and paste the returned code back into the
terminal. The CLI then verifies the token (`colab sessions`) and resolves the
account email automatically.

The implicit `default` account is the legacy single-account setup: it uses
your real login HOME and cannot be removed. Failover ordering is:

1. The account hosting the recorded session, when re-provisioning the same
   session (affinity).
2. Healthy accounts, least-recently-used first.
3. Accounts with a recorded failure, least-recently-used first (they may have
   recovered, so they are tried last instead of being skipped).

Failures are recorded per account in `~/.config/colab-t4/accounts.json`
(mode 0600) and shown by `colab-t4 accounts list`.

## Local API router

`colab-t4 serve` runs a local OpenAI-compatible HTTP router that dispatches
requests to a Colab T4 backend (when a ready runtime is recorded) or a local
Ollama backend as a fallback. This lets bots, tools, and integrations point at
a single stable endpoint regardless of which backend is currently live.

```bash
colab-t4 serve
colab-t4 serve --host 127.0.0.1 --port 8089
```

By default, `serve` asks the local Tailscale CLI for `tailscale ip -4` and
binds to that Tailscale IPv4 address when one is available. On machines without
Tailscale, it falls back to `127.0.0.1` so local development and tests remain
usable. Explicit loopback binds and explicit Tailscale CGNAT IPv4 binds are
accepted directly:

```bash
colab-t4 serve --host 100.96.1.23 --port 8089
colab-t4 serve --host localhost --port 8089
```

Binding to a wildcard, LAN, public, or other non-tailnet address is refused
unless you explicitly acknowledge the exposure:

```bash
colab-t4 serve --host 0.0.0.0 --port 8089 --allow-non-tailnet-bind
```

On startup the router reads the recorded runtime state from
`~/.config/colab-t4/state.json`. When a runtime is ready, the Colab API base URL
and key are used. Otherwise (or when the Colab API is unhealthy) the router
falls back to Ollama, whose URL is set via `OLLAMA_HOST` / `OLLAMA_BASE` or
defaults to `http://127.0.0.1:11434`.

Each incoming OpenAI-compatible request is probed against available backends in
order; the first healthy backend receives the request. The proxy accepts the
normal OpenAI-compatible `GET` and `POST` methods, rejects oversized request
bodies, and injects the Colab bearer key when present. It does not forward
`Host`, `Content-Length`, `Connection`, `Keep-Alive`, `Proxy-Authenticate`,
`Proxy-Authorization`, `TE`, `Trailer`, `Transfer-Encoding`, `Upgrade`, or any
extension header named by the incoming `Connection` header.

The router exposes a `GET /health` endpoint that reports the active backend,
returning HTTP 200 when at least one backend is healthy and 503 otherwise.

Tailscale access examples:

```bash
TAILSCALE_IP=$(tailscale ip -4)
curl -fsS "http://$TAILSCALE_IP:8089/health"
curl -fsS "http://$TAILSCALE_IP:8089/v1/models" \
  -H "Authorization: Bearer $COLAB_T4_API_KEY"
```

### Web management interface

The router also hosts a simple web management UI. Once `colab-t4 serve` is
running, open `http://<serve-host>:8089/manage` in a browser to manage accounts,
monitor runtime status, inspect diagnostics, view redacted logs, and control
backends without the CLI.

The UI provides tabs for overview, runtime lifecycle actions, account OAuth,
configuration, diagnostics, API smoke tests, logs, and asynchronous jobs.
Secret fields are write-only: blank secret fields leave existing values
unchanged, and API responses never include secret values.

All management actions are exposed as JSON API endpoints for programmatic use:

```text
GET  /api/state             runtime + accounts + backends + secrets summary
GET  /manage                HTML management UI
GET  /api/config            redacted configuration summary
PUT  /api/config            update non-secret config and write-only secrets
POST /api/up                start provisioning (async)
POST /api/restart           stop + reprovision the runtime
POST /api/down              stop the runtime
GET  /api/jobs              list recent lifecycle jobs
GET  /api/jobs/<id>         poll one lifecycle job
POST /api/jobs/<id>/cancel  request cancellation
GET  /api/status            runtime status diagnostic
GET  /api/doctor            dependency/auth diagnostic
GET  /api/logs              bounded redacted local logs
GET  /api/api/models        run an authenticated models check
POST /api/api/chat          run an authenticated chat check
POST /api/accounts          create/import an account profile
POST /api/accounts/auth-start
POST /api/accounts/auth-finish
POST /api/accounts/auth-cancel
DELETE /api/accounts/<id>   remove a registered account
```

Management mutations (`POST`, `PUT`, `DELETE`) require same-origin browser
requests and an `X-Colab-T4-Management-Token` header. For programmatic clients,
set a token before starting the router:

```bash
export COLAB_T4_MANAGEMENT_TOKEN="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)"
colab-t4 serve
```

Then send the token on mutating management requests:

```bash
curl -fsS "http://$TAILSCALE_IP:8089/api/up" \
  -X POST \
  -H "Origin: http://$TAILSCALE_IP:8089" \
  -H "X-Colab-T4-Management-Token: $COLAB_T4_MANAGEMENT_TOKEN"
```

Lifecycle operations return job envelopes. Poll the job until it reaches
`succeeded`, `failed`, or `cancelled`:

```bash
JOB_ID=$(curl -fsS "http://$TAILSCALE_IP:8089/api/restart" \
  -X POST \
  -H "Origin: http://$TAILSCALE_IP:8089" \
  -H "X-Colab-T4-Management-Token: $COLAB_T4_MANAGEMENT_TOKEN" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["job"]["id"])')

curl -fsS "http://$TAILSCALE_IP:8089/api/jobs/$JOB_ID"
```

Configuration and account OAuth examples:

```bash
curl -fsS "http://$TAILSCALE_IP:8089/api/config" \
  -X PUT \
  -H "Origin: http://$TAILSCALE_IP:8089" \
  -H "X-Colab-T4-Management-Token: $COLAB_T4_MANAGEMENT_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"session":"colab-t4","model":"repo/model","api_key":"new-api-key"}'

curl -fsS "http://$TAILSCALE_IP:8089/api/accounts/auth-start" \
  -X POST \
  -H "Origin: http://$TAILSCALE_IP:8089" \
  -H "X-Colab-T4-Management-Token: $COLAB_T4_MANAGEMENT_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"id":"work"}'
```

Diagnostics examples:

```bash
curl -fsS "http://$TAILSCALE_IP:8089/api/status"
curl -fsS "http://$TAILSCALE_IP:8089/api/doctor"
curl -fsS "http://$TAILSCALE_IP:8089/api/logs"
curl -fsS "http://$TAILSCALE_IP:8089/api/api/models"
```

## AblitBot provider integration

The companion `/root/ablitbot/ablitbot.py` uses this CLI as its OpenAI-compatible
provider. It provisions reviewed abliterated model aliases, controls runtime
start/stop, and uses the account registry for profile selection and recovery.

```text
ablitbot profile list
ablitbot profile add --id work
ablitbot profile remove work
ablitbot up
ablitbot down
```

In Telegram, administrators can use `/up`, `/down`, `/model list`,
`/model <alias>`, `/profile`, and `/profile <account>`. If the selected
runtime becomes unhealthy, AblitBot tries the selected account first and then
other registered accounts, keeping the model choice unchanged.

## Security and state

- `TS_AUTHKEY`, `HF_TOKEN`, generated API keys, and SSH passwords are never
  embedded in the notebook source or committed to the repository.
- Provisioning secrets are uploaded separately to `/content/.colab-t4-secrets.json`
  and are not printed. Local secrets are stored mode 0600 under
  `~/.config/colab-t4/secrets.json`.
- Logs are mode 0600 and redact known secret values.
- Use a short-lived or single-use Tailscale auth key and restrict the node with
  Tailscale ACLs.
- The remote Tailscale daemon uses userspace networking because Colab does not
  provide a kernel TUN device.
- Runtime logs remain on the Colab VM until it stops. `colab-t4 down` removes
  only local runtime state; it never removes `/root/colab-t4`.

## Testing and release

```bash
cd /root/colab-t4
python3 -m venv /tmp/colab-t4-test-venv
/tmp/colab-t4-test-venv/bin/pip install '.[test]'
/tmp/colab-t4-test-venv/bin/colab-t4 --version
/tmp/colab-t4-test-venv/bin/colab-t4 --help
/tmp/colab-t4-test-venv/bin/pytest
```

The release archive is `/root/colab-t4-final.zip`. It excludes virtualenvs,
models, credentials, runtime state, logs, caches, and build output.

## External limits

Colab availability, account entitlement, OAuth validity, T4 quota, Hugging Face
availability, and Tailscale network access are external dependencies. If
Colab authentication or quota blocks `up`, the command exits nonzero and keeps
redacted diagnostics in `~/.config/colab-t4/logs/`.
