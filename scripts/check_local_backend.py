"""CPU integration test for the local RL backend (no model, real assets).

Drives boot_local_sandbox + upstream git_diff + run_evaluation_local end to
end on the validated dask__dask-6683 task: a gold patch must grade reward=1.0
and an untouched workspace must grade 0.0. This is the exact reward path the
DSH GRPO round will use.
"""
import asyncio
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

if "gym_facility" not in sys.modules:
    stub = types.ModuleType("gym_facility")
    stub.execute = None
    sys.modules["gym_facility"] = stub

import gym_prepare as gp
from slime_dsh import local_backend


def main() -> None:
    frozen = json.loads((ROOT / "runtime/gym-baseline/freeze-v2.json").read_text())
    entry = next(t for t in frozen["tasks"] if t["instance_id"] == "dask__dask-6683")
    row = gp.task("dask__dask-6683")
    python = Path(entry["python"])

    case = ROOT / "runtime/slime/local-backend-test/case-gold"
    case_empty = ROOT / "runtime/slime/local-backend-test/case-empty"
    gp.make_case(case, row, python)
    gp.make_case(case_empty, row, python)

    # apply the gold patch as if a perfect agent had worked
    import subprocess
    ws = case / "sandbox/workspace"
    subprocess.run(["git", "apply", "--binary", "-"], input=row["patch"].encode(),
                   cwd=ws, check=True)

    md = {
        "instance_id": "dask__dask-6683",
        "image": local_backend.LOCAL_IMAGE,
        "workdir": str(ws),
        "problem_statement": row["problem_statement"],
        "local": {"case_dir": str(case), "python": str(python), "row": row},
    }
    local_backend._REGISTRY = {
        "dask__dask-6683": {"case_dir": str(case), "python": str(python)},
    }

    async def run():
        # 1) boot: layout + install_cli pass
        async with local_backend.boot_local_sandbox(md["image"], md["instance_id"]) as sb:
            assert sb.workspace == ws, sb.workspace
            rc, out, _ = await sb.exec("git rev-parse --is-inside-work-tree")
            assert rc == 0 and out.strip() == "true"

            # 2) upstream git_diff sees the gold edit
            import examples.coding_agent_rl.swe as swe
            diff = await swe.git_diff(sb, md["workdir"])
            assert "diff --git" in diff, diff[:200]
            assert "+0000000" not in diff[:50]
        # 3) grading: gold -> 1.0
        ev = await local_backend.run_evaluation_local(md, diff_text=diff, timeout_sec=600)
        print("gold reward:", ev.reward, "applied:", ev.applied_cleanly)
        assert ev.reward == 1.0

        # 4) empty workspace -> 0.0
        md2 = dict(md, local={"case_dir": str(case_empty), "python": str(python), "row": row})
        ev2 = await local_backend.run_evaluation_local(md2, diff_text="", timeout_sec=600)
        print("empty reward:", ev2.reward)
        assert ev2.reward == 0.0

    asyncio.run(run())
    print("LOCAL BACKEND INTEGRATION OK")


if __name__ == "__main__":
    main()
