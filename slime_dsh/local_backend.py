"""Local-mode backend for the pinned coding_agent_rl orchestrator.

Binds the upstream four-stage flow (boot sandbox -> prepare workspace ->
harness run -> git diff -> run_evaluation) onto project-local resources:

  boot_agent_sandbox  -> PER-ROLLOUT case: a fresh copy of the pre-provisioned
                         gym case (buggy checkout + env glue via gp.make_case)
                         under <task case>/roll/<uuid>. The 8 sibling samples of
                         one prompt each get their own workspace: trajectories
                         do not interfere and grading is per-sample. A
                         ContextVar carries the roll dir from boot to
                         run_evaluation within the sample's asyncio task (the
                         upstream seams pass only image/instance_id and md).
  prepare_workspace   -> upstream semantics, but into the sandbox's own
                         workspace (sb.workspace), not the task-level md path
  git_diff            -> upstream semantics (add -N + diff, excluding
                         PROBLEM_STATEMENT/.harness) from sb.workspace
  run_evaluation      -> the FROZEN gym evaluator on the roll dir (fresh
                         export/ + independent/ inside it; the evaluator's
                         make_case copytree is single-shot per directory, which
                         silently broke re-grading shared case dirs on
                         2026-09-26 -- round1-runlog incident #3). Grading
                         evidence is extracted to <task case>/grade/ and the
                         roll dir is removed. Fallback (no roll dir in the
                         context, e.g. legacy callers): grade the task case
                         dir after clearing stale export/ and independent/.

Task metadata contract (per RL sample, carried in sample.metadata):
  instance_id, problem_statement          - task identity/prompt
  image: "local"                          - sentinel, passes the non-empty check
  workdir: absolute task case workspace   - nominal; real IO goes to sb.workspace
  local.case_dir / local.python           - pre-provisioned assets
  local.row                              - full task row for the evaluator
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

LOCAL_IMAGE = "local"

# Per-sample roll dir, set inside boot_local_sandbox and consumed by
# run_evaluation_local. asyncio tasks copy their context at creation, so
# sibling samples running as separate tasks never see each other's value.
_ROLL_DIR: ContextVar[Path | None] = ContextVar("dsh_roll_dir", default=None)

# A live rollout is bounded by SWE_AGENT_TIME_BUDGET + eval + guard (< 1h);
# anything older under roll/ is debris from an aborted sample.
_ROLLOUT_STALE_SEC = 6 * 3600

# Cap concurrent case copies: upstream gates E2B boots with this semaphore
# (SWE_BOOT_CONCURRENCY), and our boot replaced that function wholesale --
# without a gate, 64 simultaneous make_case copies storm cold Lustre and the
# 30s `git add -f .` inside make_case times out (attempt-4 aborts, ~20%).
_BOOT_SEM: asyncio.Semaphore | None = None


def _boot_sem() -> asyncio.Semaphore:
    global _BOOT_SEM
    if _BOOT_SEM is None:
        import os
        _BOOT_SEM = asyncio.Semaphore(int(os.environ.get("SWE_BOOT_CONCURRENCY", "8")))
    return _BOOT_SEM


def _make_roll_case(roll: Path, row: dict, python: Path) -> None:
    """make_case with cold-cache retry: on retry the copy is page-cache warm,
    so the 30s git-add inside succeeds. Partial copies are removed first."""
    import subprocess

    for attempt in (1, 2, 3):
        try:
            import gym_prepare as gp
            gp.make_case(roll, row, python)
            return
        except subprocess.TimeoutExpired:
            if attempt == 3:
                raise
            shutil.rmtree(roll, ignore_errors=True)
            time.sleep(2 * attempt)


def _gym():
    """Import the frozen evaluator modules (shared by boot and grading)."""
    import sys
    import types

    sys.path.insert(0, str(ROOT / "scripts"))
    if "gym_facility" not in sys.modules:  # DSH SDK lives elsewhere; unused here
        stub = types.ModuleType("gym_facility")
        stub.execute = None
        sys.modules["gym_facility"] = stub
    import gym_prepare as gp
    import gym_run as gr

    return gp, gr


def _sweep_stale(roll_root: Path, max_age_sec: int = _ROLLOUT_STALE_SEC) -> None:
    if not roll_root.is_dir():
        return
    cutoff = time.time() - max_age_sec
    for entry in roll_root.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass  # best-effort debris cleanup; never blocks a rollout


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
            task_case = Path(entry["case_dir"])
            python = Path(entry["python"])
            gp, _ = _gym()
            _sweep_stale(task_case / "roll")
            roll = task_case / "roll" / uuid.uuid4().hex[:12]
            async with _boot_sem():  # bound concurrent cold copies (see above)
                # fresh buggy checkout + env glue; ~16-67 MB per task, removed
                # after grading (run_evaluation_local) or by the stale sweep
                await asyncio.to_thread(_make_roll_case, roll, gp.task(instance_id), python)
                _ROLL_DIR.set(roll)
                self._sb = LocalProcessSandbox(
                    root=roll,
                    python_dir=python.parent,
                    workspace=roll / "sandbox" / "workspace",
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
    """Frozen gym grading as the RL reward; per-sample when a roll dir is set.

    The evaluator derives the candidate patch from the workspace via a trusted
    clean-index export (diff_text is recorded but is not the grading source),
    restores oracle tests, and requires F2P+P2P to pass - identical to the
    stage-1 A/B verdict path.
    """
    _, gr = _gym()
    local = md.get("local") or {}
    task_case = Path(local["case_dir"])
    python = Path(local["python"])
    row = dict(local["row"])
    (task_case.parent / f"{task_case.name}.diff.txt").write_text(diff_text or "")

    roll = _ROLL_DIR.get()
    if roll is not None and (roll / "sandbox" / "workspace").is_dir():
        _ROLL_DIR.set(None)

        def grade():
            try:
                # the evaluator sweeps the whole workspace into its trusted
                # export; .harness (DSH runtime, ~2.5MB node_modules) is not
                # agent work and polluted every patch (incident #5 audit)
                shutil.rmtree(roll / "sandbox" / "workspace" / ".harness",
                              ignore_errors=True)
                return gr.evaluate(roll, row, python)
            except Exception as exc:  # grading infra failure != zero reward silently
                return {"resolved": False, "category": f"grading_error:{type(exc).__name__}",
                        "grading_error": f"{type(exc).__name__}: {exc}"[:500]}

        result = await asyncio.to_thread(grade)
        # keep the per-sample grading evidence, then reclaim the ~30 MB roll dir
        try:
            grade_dir = task_case / "grade"
            grade_dir.mkdir(parents=True, exist_ok=True)
            for name in ("evaluation.json", "agent.patch"):
                src = roll / name
                if src.exists():
                    shutil.copy2(src, grade_dir / f"{roll.name}.{name}")
        except OSError:
            pass
        shutil.rmtree(roll, ignore_errors=True)
    else:
        # legacy/shared path: the evaluator's export+independent are
        # single-shot per directory, so clear them before re-grading
        def grade_shared():
            try:
                for stale in ("export", "independent"):
                    shutil.rmtree(task_case / stale, ignore_errors=True)
                return gr.evaluate(task_case, row, python)
            except Exception as exc:
                return {"resolved": False, "category": f"grading_error:{type(exc).__name__}",
                        "grading_error": f"{type(exc).__name__}: {exc}"[:500]}

        result = await asyncio.to_thread(grade_shared)

    reward = 1.0 if result.get("resolved") else 0.0
    applied = bool(result.get("patch_bytes", 0)) and result.get("category") != "model_patch_application_failure"
    if result.get("grading_error"):
        print(f"[local_backend] grading error {md.get('instance_id')}: "
              f"{result['grading_error']}", flush=True)

    from examples.coding_agent_rl.swe import EvalResult
    return EvalResult(reward=reward, applied_cleanly=applied)


async def prepare_workspace_local(sb, workdir: str, md: dict) -> None:
    """Upstream prepare_workspace, but into the sandbox's own workspace.

    With the per-rollout backend sb.workspace is the sample's private copy;
    workdir (task-level, from md) is kept only for signature compatibility.
    No host-global ``git config --system`` (we avoid weakening the shared
    machine; per-instance GIT_CONFIG_SYSTEM handles it inside the sandbox)."""
    import shlex

    ws = shlex.quote(str(sb.workspace))
    await sb.exec(f"chown -R agent:agent {ws} 2>/dev/null || true",
                  user="root", timeout=60)
    await sb.write_file(
        f"{sb.workspace}/PROBLEM_STATEMENT.md",
        md.get("problem_statement") or "",
        user="agent",
    )


async def git_diff_local(sb, workdir: str) -> str:
    """Upstream git_diff (add -N + diff, same excludes) on sb.workspace."""
    import shlex

    ws = shlex.quote(str(sb.workspace))
    _, out, _ = await sb.exec(
        f"cd {ws} && git add -N . && git diff -- . "
        f"':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'",
        user="agent", timeout=120)
    return out


def bind(upstream_swe_module) -> dict[str, Any]:
    """Install the local overrides; call before the orchestrator is used.

    boot_agent_sandbox is defined in (and called from) the generate module;
    prepare_workspace / run_evaluation / git_diff / get_metadata live in the
    swe module that generate imports. Keep that asymmetry.
    """
    import examples.coding_agent_rl.generate as upstream
    upstream.boot_agent_sandbox = boot_local_sandbox
    upstream_swe_module.run_evaluation = run_evaluation_local
    upstream_swe_module.prepare_workspace = prepare_workspace_local
    upstream_swe_module.git_diff = git_diff_local
    # upstream get_metadata projects sample.metadata onto a fixed key set and
    # drops the rest; the local evaluator needs the "local" block carried
    # through (case_dir/python/row), so wrap it to re-attach that key.
    _orig_get_metadata = upstream_swe_module.get_metadata

    def _get_metadata_with_local(sample, *args, **kwargs):
        md = _orig_get_metadata(sample, *args, **kwargs)
        local = (sample.metadata or {}).get("local")
        if local:
            md["local"] = local
        return md

    upstream_swe_module.get_metadata = _get_metadata_with_local
    return {"boot_agent_sandbox": "local per-rollout case (fresh gp.make_case copy)",
            "run_evaluation": "frozen gym evaluator on the roll dir (fallback: shared case, stale export cleared)",
            "prepare_workspace": "local sandbox workspace (no host-global git config)",
            "git_diff": "upstream semantics on sb.workspace",
            "get_metadata": "upstream projection + local block passthrough"}
