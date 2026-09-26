"""Thin, testable adapter around the installed Google Colab CLI.

Verified against google-colab-cli 0.6.0:
  colab new --session NAME --gpu T4
  colab upload --session NAME LOCAL REMOTE
  colab exec --session NAME --file LOCAL --timeout SECONDS
  colab status --session NAME
  colab stop --session NAME

The adapter never invents API calls; all lifecycle operations use those CLI
commands and preserve their output in redacted local logs.
"""
from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import _atomic_json, redact

VERSION_RE = re.compile(r"Version:\s*([^\s]+)")


class ColabCLIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ColabCLI:
    executable: str
    version: str
    home: str | None = None

    @classmethod
    def discover(cls, home: str | None = None) -> "ColabCLI":
        executable = os.environ.get("COLAB_CLI") or shutil.which("colab")
        if not executable:
            raise ColabCLIError("Colab CLI executable 'colab' was not found on PATH")
        result = subprocess.run([executable, "version"], text=True, capture_output=True, timeout=30, env=cls._env(home))
        output = (result.stdout + result.stderr).strip()
        match = VERSION_RE.search(output)
        if result.returncode != 0 or not match:
            raise ColabCLIError(f"unable to identify Colab CLI version: {redact(output)}")
        return cls(executable, match.group(1), home)

    @staticmethod
    def _env(home: str | None = None) -> dict[str, str]:
        env = os.environ.copy()
        if home:
            env["HOME"] = home
        return env

    def cli_env(self) -> dict[str, str]:
        return self._env(self.home)

    def command(self, *args: str) -> list[str]:
        return [self.executable, *args]

    def new_command(self, session: str, gpu: str = "T4") -> list[str]:
        return self.command("new", "--session", session, "--gpu", gpu)

    def sessions_command(self) -> list[str]:
        return self.command("sessions")

    def prune_stale_sessions(self) -> list[str]:
        """Remove local session records whose keep-alive process is gone.

        The Colab CLI keeps a local registry in the account HOME. A crashed
        keep-alive process can leave records behind and make a later
        ``colab new`` look like an already-assigned session. Records without
        a PID are retained because they may still represent a live remote
        session managed outside this process.
        """
        base = Path(self.home) if self.home else Path.home()
        registry = base / ".config" / "colab-cli" / "sessions.json"
        try:
            value = json.loads(registry.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return []
        if not isinstance(value, dict):
            return []
        stale: list[str] = []
        for name, session in value.items():
            if not isinstance(session, dict) or session.get("keep_alive_pid") is None:
                continue
            try:
                pid = int(session["keep_alive_pid"])
                os.kill(pid, 0)
            except (TypeError, ValueError, ProcessLookupError, PermissionError):
                stale.append(str(name))
        if stale:
            for name in stale:
                value.pop(name, None)
            _atomic_json(registry, value, mode=0o600)
        return stale

    def auth_log_path(self) -> Path:
        return Path(os.environ.get("COLAB_T4_AUTH_LOG", "/tmp/colab-t4-colab-auth.log"))

    def run_interactive_auth(self) -> int:
        # `colab sessions` is the harmless CLI operation that triggers the
        # installed CLI's remote OAuth flow when credentials are absent.
        result = subprocess.run(self.command("sessions"), check=False, env=self.cli_env())
        return result.returncode

    def upload_command(self, session: str, local: Path, remote: str) -> list[str]:
        return self.command("upload", "--session", session, str(local), remote)

    def exec_command(self, session: str, local: Path, timeout: float) -> list[str]:
        return self.command("exec", "--session", session, "--file", str(local), "--timeout", str(timeout))

    def status_command(self, session: str) -> list[str]:
        return self.command("status", "--session", session)

    def stop_command(self, session: str) -> list[str]:
        return self.command("stop", "--session", session)

    def download_command(self, session: str, remote: str, local: Path) -> list[str]:
        return self.command("download", "--session", session, remote, str(local))

    def run(
        self,
        args: Sequence[str],
        log_path: Path,
        *,
        timeout: float | None = None,
        secrets: list[str] | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_path.touch(mode=0o600, exist_ok=True)
        try:
            result = subprocess.run(
                list(args),
                text=True,
                capture_output=True,
                timeout=timeout,
                env=self.cli_env(),
            )
        except subprocess.TimeoutExpired as exc:
            output = redact((exc.stdout or "") + (exc.stderr or ""), secrets)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(output)
                handle.write("\n[timeout]\n")
            raise ColabCLIError(f"Colab CLI command timed out: {args[1:]}") from exc
        output = redact((result.stdout or "") + (result.stderr or ""), secrets)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()))
            handle.write(f"$ {' '.join(args[1:])}\n")
            handle.write(output)
            if output and not output.endswith("\n"):
                handle.write("\n")
        if check and result.returncode != 0:
            raise ColabCLIError(f"Colab CLI command failed ({result.returncode}): {output[-1000:]}")
        return result


def colab_cli_token_path(home: str | None = None) -> Path:
    base = Path(home) if home else Path.home()
    return base / ".config" / "colab-cli" / "token.json"


def auth_summary(home: str | None = None) -> dict[str, object]:
    base = Path(home) if home else Path.home()
    token = base / ".config" / "colab-cli" / "token.json"
    adc = base / ".config" / "gcloud" / "application_default_credentials.json"
    return {
        "oauth_token": token.exists() and token.stat().st_size > 0,
        "adc_credentials": adc.exists() and adc.stat().st_size > 0,
        "token_path": str(token),
        "adc_path": str(adc),
    }


def authenticate_available(home: str | None = None) -> bool:
    info = auth_summary(home)
    return bool(info["oauth_token"] or info["adc_credentials"])


def import_token(source: str | Path, home: str | None = None) -> bool:
    """Import a Colab CLI ``token.json`` into a target HOME directory.

    Copies the source token to ``<home>/.config/colab-cli/token.json`` with
    mode 0600. When ``home`` is None the real user home is used. Returns True
    if a token file was written, False if the source did not exist or was
    empty.
    """
    target = colab_cli_token_path(home)
    source = Path(source)
    if not source.is_file():
        return False
    payload = source.read_bytes()
    if not payload.strip():
        return False
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".token.", dir=target.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, target)
        os.chmod(target, 0o600)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    return True
