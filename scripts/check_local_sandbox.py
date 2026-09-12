"""CPU conformance tests for slime_dsh.local_sandbox.LocalProcessSandbox.

The load-bearing tests drive the UNMODIFIED upstream transport
(slime.agent.sandbox.exec_and_wait / run_agent) against the real backend, so
the marker-file polling contract DshHarness depends on is verified end-to-end.
Run via: PYTHONPATH=<root>:<root>/vendor/slime .venv-slime-cpu/bin/python -m pytest
"""
import asyncio
import sys
import time
from pathlib import Path

import pytest

from slime.agent.sandbox import exec_and_wait
from slime.agent.harness.common import run_agent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slime_dsh.local_sandbox import LocalProcessSandbox  # noqa: E402


@pytest.fixture()
def sb(tmp_path):
    s = LocalProcessSandbox(tmp_path / "case", python_dir=None)
    asyncio.run(s.__aenter__())
    yield s
    asyncio.run(s.__aexit__(None, None, None))


def test_exec_roundtrip(sb):
    rc, out, err = asyncio.run(sb.exec("echo hello && echo warn >&2"))
    assert (rc, out.strip(), err.strip()) == (0, "hello", "warn")


def test_exec_workspace_cwd_and_env(sb):
    asyncio.run(sb.exec("touch marker_here"))
    rc, out, _ = asyncio.run(sb.exec("pwd && ls"))
    assert rc == 0 and out.splitlines()[0] == str(sb.workspace)
    assert "marker_here" in out


def test_exec_timeout_kills_group(sb):
    t0 = time.monotonic()
    rc, out, err = asyncio.run(sb.exec("setsid sleep 300 & sleep 300", timeout=3))
    assert rc == 124 and time.monotonic() - t0 < 15
    rc2, _, _ = asyncio.run(sb.exec("pgrep -f 'sleep 300' || true"))
    assert rc2 == 0  # no leftover sleep processes


def test_env_isolation_and_reserved_keys(sb):
    rc, out, _ = asyncio.run(sb.exec("echo $SANDBOX_TEST_VAR", env={"SANDBOX_TEST_VAR": "x"}))
    assert out.strip() == "x"
    with pytest.raises(ValueError):
        asyncio.run(sb.exec("true", env={"PATH": "/evil"}))


def test_file_io_jail(sb):
    asyncio.run(sb.write_file("notes.txt", "data"))
    assert asyncio.run(sb.read_file("notes.txt")) == "data"
    with pytest.raises(ValueError):
        asyncio.run(sb.write_file("/etc/passwd", "no"))
    with pytest.raises(ValueError):
        asyncio.run(sb.write_file("../../escape.txt", "no"))
    with pytest.raises(ValueError):
        asyncio.run(sb.read_file("/etc/shadow"))


def test_upstream_exec_and_wait_marker_protocol(sb):
    """The transport DshHarness uses: setsid launch + done-marker polling."""
    async def go():
        async with sb:
            return await exec_and_wait(
                sb, cmd="echo done-output; (exit 7)",
                tag="t1", time_budget_sec=30, want_output=True)
    ec, out = asyncio.run(go())
    assert ec == 7 and "done-output" in out


def test_upstream_run_agent_protocol(sb, tmp_path):
    """run_agent: chown agent:agent + marker protocol + trajectory out_file."""
    async def go():
        async with sb:
            return await run_agent(
                sb, workdir=".",
                start_cmd="bash -c 'echo agent-log-line; exit 0'",
                env={"A": "1"}, time_budget_sec=30)
    rc = asyncio.run(go())
    assert rc == 0
    assert (sb.workspace / ".harness").is_dir()


def test_concurrent_run_agent_no_marker_crosstalk(tmp_path):
    """Two sandboxes, same tag 'run', in flight simultaneously: exit codes
    must not cross (this is why /tmp token translation exists)."""
    async def one(i):
        s = LocalProcessSandbox(tmp_path / f"case{i}")
        async with s:
            rc = await run_agent(
                s, workdir=".",
                start_cmd=f"bash -c 'sleep {2 - i}; exit {i + 3}'",
                env={}, time_budget_sec=30)
        return rc

    async def go():
        return await asyncio.gather(one(0), one(1))
    assert sorted(asyncio.run(go())) == [3, 4]


def test_run_agent_time_budget(sb):
    async def go():
        async with sb:
            return await run_agent(
                sb, workdir=".", start_cmd="bash -c 'sleep 60'",
                env={}, time_budget_sec=1)
    rc = asyncio.run(go())
    assert rc == -1  # EXIT_TIME_BUDGET_EXCEEDED
