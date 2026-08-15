"""Generate the non-secret notebook executed by ``colab exec``."""
from __future__ import annotations

import json

DEFAULT_MODEL = "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated-GGUF"
DEFAULT_QUANT = "Q4_K_M"
DEFAULT_HOSTNAME = "colab-t4"
DEFAULT_PORT = 8081
DEFAULT_CTX = 8192

PROVISIONING = r'''# colab-t4 provisioning
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SECRET_FILE = Path("/content/.colab-t4-secrets.json")
READY_FILE = Path("/content/.colab-t4-ready.json")
LOG_DIR = Path("/content/colab-t4/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def record(name, text):
    (LOG_DIR / name).write_text(str(text), encoding="utf-8")


def run(args, *, log_name=None, timeout=1800, env=None, check=True):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, env=env)
    if log_name:
        record(log_name, (result.stdout or "") + (result.stderr or ""))
    if check and result.returncode:
        raise RuntimeError("command failed: " + str(args[0]))
    return result


def request_json(url, api_key=None, payload=None, timeout=20):
    headers = {}
    body = None
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


if not SECRET_FILE.exists():
    raise RuntimeError("provisioning secret file is missing")
secrets = json.loads(SECRET_FILE.read_text(encoding="utf-8"))
required = ["tailscale_authkey", "api_key", "model_repo", "quant", "port", "ctx"]
if any(not secrets.get(key) for key in required):
    raise RuntimeError("provisioning configuration is incomplete")

# GPU selection is requested by the local CLI and verified here.
gpu = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], log_name="gpu.log")
gpu_lines = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
if not gpu_lines or "t4" not in gpu_lines[0].lower():
    raise RuntimeError("allocated GPU is not an NVIDIA T4")
match = re.search(r"(\d+)\s*MiB", gpu_lines[0])
if not match or int(match.group(1)) < 14000:
    raise RuntimeError("allocated T4 does not expose enough VRAM")

# Non-interactive system dependencies.
run(["apt-get", "update", "-qq"], log_name="apt.log", timeout=600)
run(["apt-get", "install", "-y", "-qq", "openssh-server", "curl"], log_name="apt.log", timeout=600)

# Install Tailscale without relying on systemd (Colab runtimes do not run it).
install_script = subprocess.run(
    ["curl", "-fsSL", "https://tailscale.com/install.sh"],
    text=True, capture_output=True, timeout=120,
)
if install_script.returncode:
    raise RuntimeError("failed to download Tailscale installer")
install = subprocess.run(
    ["sh"], input=install_script.stdout, text=True, capture_output=True, timeout=600,
)
record("tailscale-install.log", (install.stdout or "") + (install.stderr or ""))
if install.returncode:
    raise RuntimeError("Tailscale installation failed")
subprocess.run(["pkill", "-x", "tailscaled"], capture_output=True)
subprocess.Popen(
    ["tailscaled", "--tun=userspace-networking", "--socks5-server=localhost:1055", "--port=41641"],
    stdout=open(LOG_DIR / "tailscaled.log", "w"), stderr=subprocess.STDOUT,
)
for _ in range(45):
    if Path("/var/run/tailscale/tailscaled.sock").exists():
        break
    time.sleep(1)
if not Path("/var/run/tailscale/tailscaled.sock").exists():
    raise RuntimeError("tailscaled did not start")
run([
    "tailscale", "up", "--authkey", secrets["tailscale_authkey"],
    "--hostname", secrets.get("hostname", "colab-t4"), "--accept-dns=false",
], log_name="tailscale.log", timeout=180)
ssh_mode = secrets.get("ssh_mode", "tailscale")
if ssh_mode == "tailscale":
    run(["tailscale", "set", "--ssh"], log_name="tailscale-ssh.log", timeout=60)
else:
    # Legacy fallback for tailnets where built-in Tailscale SSH is disallowed.
    ssh_dir = Path("/root/.ssh")
    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    pubkey = secrets.get("ssh_pubkey", "").strip()
    if pubkey:
        (ssh_dir / "authorized_keys").write_text(pubkey + "\n", encoding="utf-8")
        os.chmod(ssh_dir / "authorized_keys", 0o600)
    password = secrets.get("ssh_password", "").strip()
    if ssh_mode == "password" and password:
        proc = subprocess.run(
            ["chpasswd"], input="root:" + password + "\n",
            text=True, capture_output=True, timeout=30,
        )
        record("ssh.log", (proc.stdout or "") + (proc.stderr or ""))
        if proc.returncode:
            raise RuntimeError("failed to configure SSH credentials")
    Path("/run/sshd").mkdir(parents=True, exist_ok=True)
    subprocess.run(["pkill", "-x", "sshd"], capture_output=True)
    sshd_options = [
        "-o", "PermitRootLogin=yes",
        "-o", "PasswordAuthentication=" + ("yes" if ssh_mode == "password" else "no"),
        "-o", "PubkeyAuthentication=" + ("yes" if ssh_mode == "key" else "no"),
        "-o", "StrictModes=no",
    ]
    sshd_log = open(LOG_DIR / "sshd.log", "w")
    subprocess.Popen(
        ["/usr/sbin/sshd", "-D", *sshd_options],
        stdout=sshd_log, stderr=subprocess.STDOUT,
    )
ip_result = run(["tailscale", "ip", "-4"], log_name="tailscale.log", timeout=30)
tailnet_ip = ip_result.stdout.strip().splitlines()[0] if ip_result.stdout.strip() else ""
if not tailnet_ip:
    raise RuntimeError("Tailscale did not return an IPv4 address")

# Deterministic GGUF download: exactly one file must match the requested quant.
run(["python3", "-m", "pip", "install", "-q", "-U", "huggingface_hub"], log_name="model.log", timeout=900)
if secrets.get("hf_token"):
    os.environ["HF_TOKEN"] = secrets["hf_token"]
model_dir = Path("/content/model")
model_dir.mkdir(parents=True, exist_ok=True)
pattern = "*" + secrets["quant"] + ".gguf"
model_cmd = ["hf", "download", secrets["model_repo"], "--include", pattern, "--local-dir", str(model_dir)]
model_result = run(model_cmd, log_name="model.log", timeout=3600, check=False)
models = sorted(model_dir.rglob("*.gguf"))
models = [path for path in models if secrets["quant"] in path.name]
if model_result.returncode or len(models) != 1:
    raise RuntimeError("requested quant did not resolve to exactly one GGUF file")
model_path = models[0]
model_size = model_path.stat().st_size
if model_size > 11 * 1024**3:
    raise RuntimeError("selected GGUF is too large for a 16 GiB T4")

# Prefer the verified CUDA wheel index; source build is the bounded fallback.
variants = ["cu126", "cu124", "cu121", "cu128", "cu130"]
installed = False
for variant in variants:
    subprocess.run(["pip", "uninstall", "-y", "-q", "llama-cpp-python"], capture_output=True)
    pip_result = subprocess.run([
        "pip", "install", "-q", "--extra-index-url",
        "https://abetlen.github.io/llama-cpp-python/whl/" + variant,
        "--only-binary", ":all:", "llama-cpp-python[server]",
    ], text=True, capture_output=True, timeout=1800)
    record("llama-install.log", (pip_result.stdout or "") + (pip_result.stderr or ""))
    imported = subprocess.run(["python3", "-c", "import llama_cpp"], capture_output=True)
    if pip_result.returncode == 0 and imported.returncode == 0:
        installed = True
        break
if not installed:
    build = subprocess.run([
        "pip", "install", "-q", "llama-cpp-python[server]",
    ], env={**os.environ, "CMAKE_ARGS": "-DGGML_CUDA=on -DGGML_CUDA_F16=on"},
        text=True, capture_output=True, timeout=2400)
    record("llama-install.log", (build.stdout or "") + (build.stderr or ""))
    if build.returncode:
        raise RuntimeError("CUDA llama backend installation failed")

import llama_cpp
if not llama_cpp.llama_supports_gpu_offload():
    raise RuntimeError("llama backend was not built with CUDA GPU offload")

api_key = secrets["api_key"]
port = int(secrets["port"])
ctx = int(secrets["ctx"])
subprocess.run(["pkill", "-TERM", "-f", "llama_cpp.server"], capture_output=True)
for _ in range(20):
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
        probe.close()
        break
    except OSError:
        probe.close()
        time.sleep(1)
else:
    raise RuntimeError("configured API port is still in use")
server_log = open(LOG_DIR / "llama-server.log", "w")
server = subprocess.Popen([
    "python3", "-m", "llama_cpp.server", "--model", str(model_path),
    "--model_alias", "local", "--n_gpu_layers", "-1", "--n_ctx", str(ctx),
    "--flash_attn", "true", "--host", "0.0.0.0", "--port", str(port),
    "--api_key", api_key,
], stdout=server_log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
health_url = "http://127.0.0.1:" + str(port) + "/v1/models"
healthy = False
for _ in range(240):
    try:
        health = request_json(health_url, api_key=api_key)
        healthy = bool(health.get("data"))
    except Exception:
        pass
    if healthy:
        break
    time.sleep(2)
if not healthy:
    raise RuntimeError("llama server did not become healthy")
# Require evidence of actual GPU offload in the server log.
offload = False
for _ in range(30):
    try:
        log_text = (LOG_DIR / "llama-server.log").read_text(encoding="utf-8", errors="replace")
        offload = bool(re.search(r"offload|layers.*GPU|CUDA", log_text, re.IGNORECASE))
    except OSError:
        pass
    if offload:
        break
    time.sleep(1)
if not offload:
    raise RuntimeError("server log did not prove CUDA offloading")
models_response = request_json("http://127.0.0.1:" + str(port) + "/v1/models", api_key=api_key)
if not models_response.get("data"):
    raise RuntimeError("authenticated /v1/models returned no model")
chat_response = request_json(
    "http://127.0.0.1:" + str(port) + "/v1/chat/completions", api_key=api_key,
    payload={"model": "local", "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 8},
)
ready = {
    "ready": True,
    "gpu": gpu_lines[0],
    "tailscale_ip": tailnet_ip,
    "api_base": "http://" + tailnet_ip + ":" + str(port) + "/v1",
    "model": str(model_path),
    "quant": secrets["quant"],
    "ssh_mode": ssh_mode,
    "server_pid": server.pid,
    "tests": {"health": True, "models": True, "chat": True, "cuda_offload": True, "tailscale_ssh": ssh_mode == "tailscale"},
}
READY_FILE.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
os.chmod(READY_FILE, 0o600)
print("COLAB_T4_READY")
'''


def build_notebook(*, hostname: str = DEFAULT_HOSTNAME, model_repo: str = DEFAULT_MODEL,
                   quant: str = DEFAULT_QUANT, port: int = DEFAULT_PORT, ctx: int = DEFAULT_CTX) -> str:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# colab-t4\n", "\n", "Browserless T4 provisioning notebook.\n",
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [PROVISIONING]},
    ]
    return json.dumps({
        "nbformat": 4, "nbformat_minor": 5, "metadata": {
            "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        }, "cells": cells,
    }, indent=1) + "\n"
