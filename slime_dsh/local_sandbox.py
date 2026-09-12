"""In-project local backend for slime's Sandbox protocol (vendor untouched).

Why this exists (research note 2026-09-12): pinned slime 4c193f1f and upstream
main ship exactly one Sandbox implementation, E2BSandbox (remote HTTP gateway).
Community/local alternatives were searched and none exist. The protocol is
three methods, so this is thin glue - NOT a reimplementation of any evaluator.

Contract fidelity notes:
- ExecResult = (returncode, stdout, stderr) per slime.agent.sandbox.
- exec_and_wait (the generic transport run_agent uses) issues plain shell:
  mkdir lock dirs, setsid launches, marker-file polls, tail. A faithful
  bash -c exec against a real directory satisfies it with zero special cases,
  unlike E2B which needs reverse tunnels - here the agent dials the adapter
  on 127.0.0.1 directly.
- Shared-machine discipline: commands run as the current (non-root) user,
  cwd-jailed to the task workspace, env scrubbed to a whitelist plus the
  task's own python env PATH, every call hard-timeouts and kills the process
  group (start_new_session). File IO is path-jailed under the sandbox root.

/tmp translation (local mode, because the container forbids unshare):
  slime's transport (exec_and_wait/run_agent) hardcodes /tmp/.{tag}.out/.done/
  .sh/.spawned marker paths. Concurrent sandboxes sharing the real /tmp would
  collide on tag "run" and cross-read exit-code markers. Mount namespaces are
  unavailable (no CAP_SYS_ADMIN), so we translate the transport's /tmp/.<tag>
  token forms to <root>/tmp/... on BOTH sides (write_file jail + exec text).
  Any other /tmp usage passes through to the real /tmp (documented residual;
  DSH itself gets TMPDIR pointing inside the sandbox by DshHarness).

The `user=` parameter is accepted and ignored: single-user local execution
(a system 'agent' user exists only so upstream `chown agent:agent` calls
succeed verbatim).
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
from pathlib import Path
from typing import Protocol  # noqa: F401  (documentation only)

EXEC_DEFAULT_TIMEOUT = 120
_ENV_BASE = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "",  # set per-instance
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TMPDIR": "",  # set per-instance
    "PYTHONDONTWRITEBYTECODE": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
}
# slime.agent.sandbox.exec_and_wait marker/launcher naming: /tmp/.{tag}.ext
_TMP_TOKEN = re.compile(r"/tmp/\.([A-Za-z0-9_.-]+)")


class LocalProcessSandbox:
    """One task = one root dir. Workspace = root/workspace; commands run there."""

    def __init__(self, root: Path, python_dir: Path | None = None,
                 extra_env: dict[str, str] | None = None):
        self.root = Path(root).resolve()
        self.workspace = self.root / "workspace"
        self.tmp = self.root / "tmp"
        self.home = self.root / "home"
        self.sandbox_id = f"local-{self.root.name}"
        self._python_dir = Path(python_dir).resolve() if python_dir else None
        self._extra_env = dict(extra_env or {})
        self._proc: asyncio.subprocess.Process | None = None

    async def __aenter__(self) -> "LocalProcessSandbox":
        for d in (self.root, self.workspace, self.tmp, self.home):
            d.mkdir(parents=True, exist_ok=True)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

    def _base_env(self) -> dict[str, str]:
        env = dict(_ENV_BASE)
        env["HOME"] = str(self.home)
        env["TMPDIR"] = str(self.tmp)
        if self._python_dir:
            env["PATH"] = f"{self._python_dir}:{env['PATH']}"
        env.update(self._extra_env)
        return env

    def _jail(self, sandbox_path: str) -> Path:
        p = Path(sandbox_path)
        if not p.is_absolute():
            p = self.workspace / p
        resolved = p.resolve()
        # transport marker/launcher paths live under /tmp/.{tag}.* locally too
        if str(resolved).startswith("/tmp/."):
            resolved = self.tmp / resolved.name
        if self.root not in resolved.parents and resolved != self.root:
            raise ValueError(f"path escapes sandbox root: {sandbox_path}")
        return resolved

    def _translate_tmp_tokens(self, cmd: str) -> str:
        return _TMP_TOKEN.sub(lambda m: f"{self.tmp}/.{m.group(1)}", cmd)

    async def exec(self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None,
                   timeout: int = EXEC_DEFAULT_TIMEOUT, check: bool = False,
                   idempotent: bool = True) -> tuple[int, str, str]:
        merged = self._base_env()
        if env:
            bad = set(env) & {"PATH", "HOME", "TMPDIR"}
            if bad:
                raise ValueError(f"env overrides reserved keys: {sorted(bad)}")
            merged.update(env)
        self._proc = await asyncio.create_subprocess_exec(
            "bash", "-c", self._translate_tmp_tokens(cmd),
            cwd=self.workspace, env=merged,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = await asyncio.wait_for(self._proc.communicate(), timeout)
        except asyncio.TimeoutError:
            try:  # kill the whole session (children included)
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(self._proc.wait(), 10)
            except asyncio.TimeoutError:
                pass
            return 124, "", f"command timed out after {timeout}s"
        rc = self._proc.returncode
        result = (rc if rc is not None else -1,
                  out.decode(errors="replace"), err.decode(errors="replace"))
        if check and result[0] != 0:
            raise RuntimeError(f"exec failed rc={result[0]}: {cmd[:120]}\n{result[2][:500]}")
        return result

    async def write_file(self, sandbox_path: str, content, *, user: str = "root") -> None:
        target = self._jail(sandbox_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):  # host path stream-in per the protocol docs
            shutil.copyfile(content, target)
        elif isinstance(content, bytes):
            target.write_bytes(content)
        else:
            # mirror exec's /tmp token translation: the transport's launcher
            # body writes its done-marker via /tmp/.{tag} paths that must land
            # in this sandbox's private tmp, or polls read the wrong place
            target.write_text(self._translate_tmp_tokens(content))

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        target = self._jail(sandbox_path)
        if target.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("refusing to read_file >64MiB")
        return target.read_text(errors="replace")
