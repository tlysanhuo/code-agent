"""Local-mode backend for the pinned coding_agent_rl orchestrator.

Binds the upstream four-stage flow (boot sandbox -> prepare workspace ->
harness run -> git diff -> run_evaluation) onto project-local resources:

  boot_agent_sandbox  -> LocalProcessSandbox over a PRE-PROVISIONED gym case
                         (buggy checkout already at case/sandbox/workspace,
                         qualified python env) + DshHarness.install_cli checks
  prepare_workspace   -> upstream verbatim (ensure_agent_user runs fine as
                         root; PROBLEM_STATEMENT.md lands in the workdir)
  git_diff            -> upstream verbatim (plain git in the workspace)
  run_evaluation      -> the FROZEN gym evaluator (trusted clean-index patch
                         export, oracle test restoration, F2P/P2P), the same
                         grading that produced the A/B verdict; reward 1.0 iff
                         resolved

Task metadata contract (per RL sample, carried in sample.metadata):
  instance_id, problem_statement          - task identity/prompt
  image: "local"                          - sentinel, passes the non-empty check
  workdir: absolute case/sandbox/workspace- must sit under the project root
                                          (DshHarness requirement) and under
                                          the sandbox root (jail requirement)
  local.case_dir / local.python           - pre-provisioned assets
  local.row                              - full task row for the evaluator
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

LOCAL_IMAGE = "local"


def boot_local_sandbox(image: str, instance_id: str):
    """Async-context-manager factory matching upstream boot_agent_sandbox."""
    if image != LOCAL_IMAGE:
        raise ValueError(f"local backend got non-local image: {image!r}")
    from slime_dsh.local_sandbox import LocalProcessSandbox

    class _Ctx:
        async def __aenter__(self):
            reg = registry()
            if instance_id not in reg:
                raise KeyError(f"task {instance_id} has no provisioned local case")
            entry = reg[instance_id]
            self._sb = LocalProcessSandbox(
                root=Path(entry["case_dir"]),
                python_dir=Path(entry["python"]).parent,
                workspace=Path(entry["case_dir"]) / "sandbox" / "workspace",
            )
            await self._sb.__aenter__()
            try:
                from slime_dsh.harness import DshHarness
                await DshHarness().install_cli(self._sb)
            except BaseException:
                await self._sb.__aexit__(None, None, None)
                raise
            return self._sb

        async def __aexit__(self, *exc):
            return await self._sb.__aexit__(*exc)

    return _Ctx()


_REGISTRY: dict[str, dict[str, Any]] | None = None


def registry_path() -> Path:
    return ROOT / "configs/agent-rl/local-task-registry.json"


def register_tasks(entries: dict[str, dict[str, Any]]) -> None:
    global _REGISTRY
    _REGISTRY = dict(entries)
    registry_path().write_text(json.dumps(entries, indent=1) + "\n")


def registry() -> dict[str, dict[str, Any]]:
    global _REGISTRY
    if _REGISTRY is None:
        p = registry_path()
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} missing: run the env-qualification batch first "
                "(scripts/prepare_rl_tasks.py)")
        _REGISTRY = json.loads(p.read_text())
    return _REGISTRY


async def run_evaluation_local(md: dict, *, diff_text: str, timeout_sec: int):
    """Frozen gym grading as the RL reward; runs in a worker thread.

    The evaluator derives the candidate patch from the workspace via a trusted
    clean-index export (diff_text is recorded but is not the grading source),
    restores oracle tests, and requires F2P+P2P to pass - identical to the
    stage-1 A/B verdict path.
    """
    import sys
    import types

    sys.path.insert(0, str(ROOT / "scripts"))
    if "gym_facility" not in sys.modules:  # DSH SDK lives elsewhere; unused here
        stub = types.ModuleType("gym_facility")
        stub.execute = None
        sys.modules["gym_facility"] = stub
    import gym_prepare as gp  # noqa: F401
    import gym_run as gr

    local = md.get("local") or {}
    case_dir = Path(local["case_dir"])
    python = Path(local["python"])
    row = dict(local["row"])
    (case_dir.parent / f"{case_dir.name}.diff.txt").write_text(diff_text or "")

    def grade():
        try:
            return gr.evaluate(case_dir, row, python)
        except Exception as exc:  # grading infra failure != zero reward silently
            return {"resolved": False, "category": f"grading_error:{type(exc).__name__}"}

    result = await asyncio.to_thread(grade)
    reward = 1.0 if result.get("resolved") else 0.0
    applied = bool(result.get("patch_bytes", 0)) and result.get("category") != "model_patch_application_failure"

    from examples.coding_agent_rl.swe import EvalResult
    return EvalResult(reward=reward, applied_cleanly=applied)


async def prepare_workspace_local(sb, workdir: str, md: dict) -> None:
    """Upstream prepare_workspace minus ensure_agent_user's ``git config
    --system`` (a host-global weakening we avoid on this shared machine; the
    root user's global config carries safe.directory instead)."""
    await sb.exec(f"chown -R agent:agent {workdir} 2>/dev/null || true",
                  user="root", timeout=60)
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md",
        md.get("problem_statement") or "",
        user="agent",
    )


def bind(upstream_swe_module) -> dict[str, Any]:
    """Install the local overrides; call before the orchestrator is used."""
    import examples.coding_agent_rl.generate as upstream

    upstream.boot_agent_sandbox = boot_local_sandbox
    upstream_swe_module.run_evaluation = run_evaluation_local
    upstream_swe_module.prepare_workspace = prepare_workspace_local
    return {"boot_agent_sandbox": "local",
            "run_evaluation": "frozen gym evaluator",
            "prepare_workspace": "local (no host-global git config)"}
