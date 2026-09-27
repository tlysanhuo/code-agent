#!/usr/bin/env python3
"""Per-rollout isolation check (round1-runlog incident #3 regression test).

Simulates the upstream generate() flow for TWO CONCURRENT sibling samples of
the same task through the real local backend: boot -> prepare -> edit ->
git_diff -> run_evaluation. Acceptance:
  1. each sample gets its own workspace (roll dirs differ);
  2. each sample's exported agent.patch contains ONLY its own edit
     (no cross-sample workspace interference);
  3. both gradings succeed (no grading_error: the evaluator's single-shot
     export/ dirs must not collide across siblings or re-gradings);
  4. roll dirs are cleaned up and per-sample grading evidence recorded.

Usage: PYTHONPATH=$PWD:$PWD/vendor/slime .venv-train-rl/bin/python \
         scripts/check_rollout_isolation.py [instance_id]
Runtime ~2-4 min (two frozen-evaluator gradings run in parallel).
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/slime"))

from slime_dsh import local_backend as lb  # noqa: E402

TASK = sys.argv[1] if len(sys.argv) > 1 else "python__mypy-14064"


async def one_sample(md: dict, tag: str) -> dict:
    """boot -> prepare -> 'agent edit' -> diff -> grade, like generate()."""
    async with lb.boot_local_sandbox("local", md["instance_id"]) as sb:
        ws = str(sb.workspace)
        await lb.prepare_workspace_local(sb, md["workdir"], md)
        # each sibling edits a file only IT knows about
        marker = f"ISOLATION_MARKER_{tag}"
        await sb.exec(f"bash -c 'echo {marker} > isolation_{tag}.py'", user="agent", check=True)
        diff_text = await lb.git_diff_local(sb, md["workdir"])
        roll = ws.split("/roll/")[-1].split("/")[0] if "/roll/" in ws else None
    ev = await lb.run_evaluation_local(md, diff_text=diff_text, timeout_sec=600)
    return {"tag": tag, "workspace": ws, "roll": roll, "diff": diff_text,
            "reward": ev.reward, "applied": ev.applied_cleanly}


async def main() -> None:
    reg = lb.registry()
    assert TASK in reg, f"{TASK} not in registry"
    entry = reg[TASK]
    gp, _ = lb._gym()
    row = gp.task(TASK)
    md = {
        "instance_id": TASK,
        "problem_statement": row.get("problem_statement") or "stub problem",
        "image": "local",
        "workdir": str(Path(entry["case_dir"]) / "sandbox" / "workspace"),
        "local": {"case_dir": entry["case_dir"], "python": entry["python"], "row": row},
    }
    task_case = Path(entry["case_dir"])

    results = await asyncio.gather(one_sample(md, "A"), one_sample(md, "B"))

    fails = []
    # 1. distinct per-rollout workspaces
    if results[0]["workspace"] == results[1]["workspace"]:
        fails.append("siblings share one workspace")
    if not results[0]["roll"] or results[0]["roll"] == results[1]["roll"]:
        fails.append("roll dir missing or identical")
    # 2. diff isolation: each diff contains its own marker and NOT the other's
    for r, other in ((results[0], "B"), (results[1], "A")):
        if f"ISOLATION_MARKER_{r['tag']}" not in r["diff"]:
            fails.append(f"{r['tag']}: own edit missing from diff")
        if f"ISOLATION_MARKER_{other}" in r["diff"]:
            fails.append(f"{r['tag']}: sibling edit leaked into diff")
    # 3. grading health: pull the recorded per-sample evaluation artifacts
    grade_files = sorted((task_case / "grade").glob("*.evaluation.json")) \
        if (task_case / "grade").is_dir() else []
    cats = []
    for f in grade_files:
        d = json.loads(f.read_text())
        cats.append(d.get("category"))
        if str(d.get("category", "")).startswith("grading_error"):
            fails.append(f"grading_error in {f.name}: {d.get('category')}")
    if len(grade_files) < 2:
        fails.append(f"expected >=2 grade artifacts, found {len(grade_files)}")
    # 4. roll dirs cleaned
    leftover = list((task_case / "roll").glob("*")) if (task_case / "roll").is_dir() else []
    if leftover:
        fails.append(f"roll dirs not cleaned: {[p.name for p in leftover]}")

    print(json.dumps({
        "task": TASK,
        "rolls": [r["roll"] for r in results],
        "rewards": [r["reward"] for r in results],
        "categories": cats,
        "grade_artifacts": [f.name for f in grade_files],
        "passed": not fails,
        "failures": fails,
    }, indent=1))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    asyncio.run(main())
