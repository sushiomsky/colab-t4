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
mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF
```

with `Q4_K_M`. Override deterministically:

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
```

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
instead of stacking on one account. The same rotation applies when the
recorded session is lost or never becomes ready again: `up` records the
failure, stops the dead session, and continues with the next account instead
of erroring out.

Accounts are managed with the `accounts` command:

```bash
colab-t4 accounts list            # current rotation, health, and usage
colab-t4 accounts add --id work   # interactive Colab OAuth flow for a new profile
colab-t4 accounts add --id personal --email you@gmail.com
colab-t4 accounts remove work     # unregister and delete the profile
```

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
   session and it has no recorded failure (affinity).
2. Healthy accounts, least-recently-used first.
3. Accounts with a recorded failure, least-recently-used first (they may have
   recovered, so they are tried last instead of being skipped).

Failures are recorded per account in `~/.config/colab-t4/accounts.json`
(mode 0600) and shown by `colab-t4 accounts list`.

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
