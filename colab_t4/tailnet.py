"""Locate the Colab node on the tailnet.

Two backends:
- local: the ``tailscale`` CLI on this machine (``tailscale status --json``)
- api: the Tailscale HTTP API (needs an access token + tailnet name)
"""

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request


def _poll_log(log, msg):
    if log is not None:
        log(msg)


def resolve_local(prefix, timeout=600, log=None, interval=10):
    """Poll ``tailscale status --json`` for a peer whose hostname starts with ``prefix``.

    Returns ``(ipv4, hostname)``. Raises ``SystemExit`` when the local CLI is
    unavailable and ``TimeoutError`` when the node never appears.
    """
    deadline = time.time() + timeout
    while True:
        remaining = int(deadline - time.time())
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for node '{prefix}' on the tailnet")
        try:
            out = subprocess.run(
                ["tailscale", "status", "--json"], capture_output=True, text=True, timeout=15
            )
        except FileNotFoundError:
            raise SystemExit(
                "local 'tailscale' CLI not found - run from a tailnet-connected "
                "machine, or pass --api-token/--tailnet"
            )
        except Exception as e:  # transient CLI error
            _poll_log(log, f"tailscale status error: {e}")
            time.sleep(interval)
            continue
        if out.returncode == 0:
            try:
                data = json.loads(out.stdout)
            except ValueError:
                data = {}
            for _pid, peer in data.get("Peer", {}).items():
                hn = peer.get("HostName", "")
                if hn.startswith(prefix) and peer.get("Online") and peer.get("TailscaleIPs"):
                    return peer["TailscaleIPs"][0], hn
            _poll_log(log, f"[{remaining}s] node '{prefix}' not online yet")
        else:
            _poll_log(log, "tailscale status failed (are you logged in?): " + out.stderr.strip()[:200])
        time.sleep(interval)


def resolve_api(prefix, token, tailnet, timeout=600, log=None, interval=10):
    """Poll the Tailscale API for a device whose hostname starts with ``prefix``."""
    deadline = time.time() + timeout
    auth = "Basic " + base64.b64encode(f"{token}:".encode()).decode()
    while True:
        remaining = int(deadline - time.time())
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for node '{prefix}' on the tailnet")
        req = urllib.request.Request(
            f"https://api.tailscale.com/api/v2/tailnet/{tailnet}/devices"
        )
        req.add_header("Authorization", auth)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise SystemExit(
                f"Tailscale API error {e.code}: {e.read().decode(errors='replace')[:300]}"
            )
        except Exception as e:
            _poll_log(log, f"tailscale API error: {e}")
            time.sleep(interval)
            continue
        for d in data.get("devices", []):
            if d.get("hostname", "").startswith(prefix) and d.get("online"):
                addrs = [a for a in d.get("addresses", []) if ":" not in a]
                if addrs:
                    return addrs[0], d.get("hostname", "")
        _poll_log(log, f"[{remaining}s] node '{prefix}' not online yet")
        time.sleep(interval)


def wait_health(ip, port, timeout=300, log=None):
    """Poll the llama.cpp ``/health`` endpoint through the tailnet until it responds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{ip}:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        _poll_log(log, f"waiting for API at http://{ip}:{port}/health ...")
        time.sleep(5)
    return False
