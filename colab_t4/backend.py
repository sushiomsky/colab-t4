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
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import redact

VERSION_RE = re.compile(r"Version:\s*([^\s]+)")


class ColabCLIError(RuntimeError):
    pass


@dataclass(frozen=True)
class ColabCLI:
    executable: str
    version: str

    @classmethod
    def discover(cls) -> "ColabCLI":
        executable = os.environ.get("COLAB_CLI") or shutil.which("colab")
        if not executable:
            raise ColabCLIError("Colab CLI executable 'colab' was not found on PATH")
        result = subprocess.run([executable, "version"], text=True, capture_output=True, timeout=30)
        output = (result.stdout + result.stderr).strip()
        match = VERSION_RE.search(output)
        if result.returncode != 0 or not match:
            raise ColabCLIError(f"unable to identify Colab CLI version: {redact(output)}")
        return cls(executable, match.group(1))

    def command(self, *args: str) -> list[str]:
        return [self.executable, *args]

    def new_command(self, session: str, gpu: str = "T4") -> list[str]:
        return self.command("new", "--session", session, "--gpu", gpu)

    def sessions_command(self) -> list[str]:
        return self.command("sessions")

    def auth_log_path(self) -> Path:
        return Path(os.environ.get("COLAB_T4_AUTH_LOG", "/tmp/colab-t4-colab-auth.log"))

    def run_interactive_auth(self) -> int:
        # `colab sessions` is the harmless CLI operation that triggers the
        # installed CLI's remote OAuth flow when credentials are absent.
        result = subprocess.run(self.command("sessions"), check=False)
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
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            output = redact((exc.stdout or "") + (exc.stderr or ""), secrets)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(output)
                handle.write("\n[timeout]\n")
            raise ColabCLIError(f"Colab CLI command timed out: {args[1:]}") from exc
        output = redact((result.stdout or "") + (result.stderr or ""), secrets)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(args[1:])}\n")
            handle.write(output)
            if output and not output.endswith("\n"):
                handle.write("\n")
        if check and result.returncode != 0:
            raise ColabCLIError(f"Colab CLI command failed ({result.returncode}): {output[-1000:]}")
        return result


def auth_summary() -> dict[str, object]:
    token = Path(os.path.expanduser("~/.config/colab-cli/token.json"))
    adc = Path(os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))
    return {
        "oauth_token": token.exists() and token.stat().st_size > 0,
        "adc_credentials": adc.exists() and adc.stat().st_size > 0,
        "token_path": str(token),
        "adc_path": str(adc),
    }


def authenticate_available() -> bool:
    info = auth_summary()
    return bool(info["oauth_token"] or info["adc_credentials"])
