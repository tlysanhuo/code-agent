"""Native DSH SDK lifecycle using slime's existing sandbox/run_agent transport.

Deployment contract: this project and its isolated DSH environment must be
mounted at the same absolute path inside the sandbox. No installation, external
downloads, task preparation, model proxy, trajectory or grading logic lives here.
"""
import json
from pathlib import Path
import re
import shlex

from slime.agent.harness.common import BaseHarness, HarnessContext, run_agent
from slime.agent.sandbox import Sandbox

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-dsh/bin/python"
RUNTIME = ROOT / ".venv-dsh/lib/python3.12/site-packages/deepseek_harness_runtime/runtime/deepseek-harness-sdk-runtime-linux-x64"
WORKER = ROOT / "scripts/agent_rl_dsh_worker.py"
PATCH = ROOT / "configs/agent-rl/dsh-qwen-rl.patch.yml"


def run_directory(workdir: str) -> Path:
    # Upstream run_agent interpolates workdir into shell text without quoting.
    # Reject unsafe paths here rather than changing that transport implementation.
    path = Path(workdir)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(ROOT):
        raise ValueError("DSH workspace must be an absolute path inside the mounted project")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", workdir):
        raise ValueError("The upstream sandbox transport requires a shell-safe workspace path")
    return path / ".harness/dsh"


class DshHarness(BaseHarness):
    name = "dsh"

    async def install_cli(self, sb: Sandbox) -> None:
        # Provisioning belongs to the execution platform, not the agent adapter.
        cmd = " && ".join([
            f"test -x {shlex.quote(str(PYTHON))}",
            f"test -x {shlex.quote(str(RUNTIME))}",
            f"test -r {shlex.quote(str(WORKER))}",
            f"test -r {shlex.quote(str(PATCH))}",
        ])
        await sb.exec(cmd, user="root", check=True, timeout=30)

    async def run(self, sb: Sandbox, **kwargs) -> int:
        run_directory(kwargs["workdir"])  # before BaseHarness can mutate the sandbox
        return await super().run(sb, **kwargs)

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        directory = run_directory(ctx.workdir)
        await sb.exec(f"mkdir -p {shlex.quote(str(directory))} && chown -R agent:agent {shlex.quote(str(directory))}",
                      user="root", check=True, timeout=30)

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        directory = run_directory(ctx.workdir)
        if time_budget_sec <= 0:
            raise ValueError("DSH requires a positive time budget")
        job = {
            "dsh_bin": str(RUNTIME), "workspace": ctx.workdir,
            "dsh_home": str(directory / "home"), "patch": str(PATCH),
            "max_tokens": 4096,  # existing DSH RL deployment cap, unchanged
            "timeout": time_budget_sec, "prompt": prompt,
            "session_id": ctx.session_id, "result": str(directory / "result.json"),
        }
        job_path = directory / "job.json"
        await sb.write_file(str(job_path), json.dumps(job, ensure_ascii=False), user="agent")
        env = {
            "QWEN_BASE_URL": ctx.adapter_url.rstrip("/") + "/v1",
            "QWEN_LOCAL_API_KEY": ctx.session_id,
            "DSH_HOME": job["dsh_home"], "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": str(directory / "cache/pycache"),
            "XDG_CACHE_HOME": str(directory / "cache"),
            "XDG_CONFIG_HOME": str(directory / "config"),
            "XDG_DATA_HOME": str(directory / "data"),
            "XDG_STATE_HOME": str(directory / "state"),
            "npm_config_cache": str(directory / "cache/npm"),
            "TMPDIR": str(directory / "tmp"),
            "TMP": str(directory / "tmp"), "TEMP": str(directory / "tmp"),
        }
        await sb.exec(f"mkdir -p {shlex.quote(env['TMPDIR'])}", user="agent", check=True, timeout=30)
        return await run_agent(
            sb, workdir=ctx.workdir,
            start_cmd=shlex.join([str(PYTHON), str(WORKER), str(job_path)]),
            env=env, time_budget_sec=time_budget_sec,
        )
