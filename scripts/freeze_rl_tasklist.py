#!/usr/bin/env python3
"""Freeze the stage-2 agentic-RL task pools with an independent exclusion audit.

Re-derives eligibility from the raw task parquets instead of trusting earlier
audit decisions, then applies every standing exclusion set:

  1. SWE-bench Verified 500 instance_ids (final eval set)
  2. SWE-bench full test-split instance_ids (swebench-full-test-exclusion-source)
  3. 20 held-out SWE-Gym dev-baseline tasks (configs/gym-training-exclusion.json)
  4. 3 frozen SWE-smith repair tasks (configs/swe-three-tasks.json)
  5. 81 old-audit non-eligible SWE-Gym ids (data/agent-rl/exclusion-audit.json
     decisions != identified_nonholdout)

Outputs (under data/agent-rl/):
  rl-freeze-v1-gym-eligible.txt / rl-freeze-v1-smith-eligible.txt  one id/line
  rl-freeze-v1-gym-order.txt      / rl-freeze-v1-smith-order.txt    deterministic round order
and configs/agent-rl/rl-task-freeze-v1.json (counts, hashes, per-set ledger).

Tasks whose environments were already SFT-trained on (klear66k covers
SWE-smith) are NOT excluded -- reusing the SFT task distribution for RL is
standard curriculum, not eval contamination -- but the overlap is recorded.

Env validation is NOT part of this freeze: every task still needs its
buggy/gold env qualified before any rollout, per the standing gym rules.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260911


def ids_from_parquet(path: str, col: str = "instance_id") -> set[str]:
    return set(pq.read_table(path, columns=[col])[col].to_pylist())


def sha256_of_lines(lines: list[str]) -> str:
    return hashlib.sha256(("".join(l + "\n" for l in lines)).encode()).hexdigest()


def main() -> None:
    verified = ids_from_parquet(
        "data/raw/SWE-bench__SWE-bench_Verified/data/test-00000-of-00001.parquet")
    full_test = ids_from_parquet(
        json.loads((ROOT / "data/metadata/swebench-full-test-exclusion-source.json")
                   .read_text())["path"])
    held_out = set(json.loads(
        (ROOT / "configs/gym-training-exclusion.json").read_text())["held_out_task_ids"])
    smith_three = {t["instance_id"] for t in json.loads(
        (ROOT / "configs/swe-three-tasks.json").read_text())["tasks"]}
    old_audit = json.loads((ROOT / "data/agent-rl/exclusion-audit.json").read_text())
    audit_blocked = {d["instance_id"] for d in old_audit["decisions"]
                     if d["decision"] != "identified_nonholdout"}

    ledger: dict[str, Counter] = {}

    def decide(pool: dict[str, dict], name: str) -> list[str]:
        c = Counter()
        eligible = []
        for iid, row in pool.items():
            if iid in verified:
                c["ex_verified"] += 1
            elif iid in full_test:
                c["ex_full_test"] += 1
            elif iid in held_out:
                c["ex_held_out_20"] += 1
            elif iid in smith_three:
                c["ex_smith_three"] += 1
            elif iid in audit_blocked:
                c["ex_old_audit"] += 1
            else:
                c["eligible"] += 1
                eligible.append(iid)
        ledger[name] = c
        return eligible

    gym_pool: dict[str, dict] = {}
    gym_rows = pq.read_table("data/raw/SWE-Gym__SWE-Gym/data/train-00000-of-00001.parquet",
                             columns=["instance_id", "repo"])
    for iid, repo in zip(gym_rows["instance_id"].to_pylist(), gym_rows["repo"].to_pylist()):
        gym_pool[iid] = {"repo": repo}

    smith_pool: dict[str, dict] = {}
    smith_repos = Counter()
    for path in sorted(glob.glob("data/raw/SWE-bench__SWE-smith/data/train-*.parquet")):
        rows = pq.read_table(path, columns=["instance_id", "repo", "image_name"])
        for iid, repo, img in zip(rows["instance_id"].to_pylist(),
                                  rows["repo"].to_pylist(),
                                  rows["image_name"].to_pylist()):
            smith_pool[iid] = {"repo": repo, "image_name": img}
            smith_repos[repo] += 1

    gym_eligible = sorted(decide(gym_pool, "swe_gym"))
    smith_eligible = sorted(decide(smith_pool, "swe_smith"))

    # SFT overlap (informational only, see docstring)
    klear_tasks = set()
    for path in sorted(glob.glob("data/sft-pool/sft-trajectories-00[0-7].parquet")):
        rows = pq.read_table(path, columns=["metadata"])
        for m in rows["metadata"].to_pylist():
            iid = (m or {}).get("instance_id")
            if iid:
                klear_tasks.add(iid)
    gym_sft_overlap = sum(1 for i in gym_eligible if i in klear_tasks)
    smith_sft_overlap = sum(1 for i in smith_eligible if i in klear_tasks)

    rng = random.Random(SEED)
    gym_order = list(gym_eligible)
    smith_order = list(smith_eligible)
    rng.shuffle(gym_order)
    rng.shuffle(smith_order)

    out_dir = ROOT / "data/agent-rl"
    outputs = {
        "rl-freeze-v1-gym-eligible.txt": gym_eligible,
        "rl-freeze-v1-smith-eligible.txt": smith_eligible,
        "rl-freeze-v1-gym-order.txt": gym_order,
        "rl-freeze-v1-smith-order.txt": smith_order,
    }
    for fname, lines in outputs.items():
        (out_dir / fname).write_text("".join(l + "\n" for l in lines))

    freeze = {
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "stage-2 agentic RL (GRPO, tests-pass reward) candidate pools; "
                   "env validation still required per task before any rollout",
        "seed": SEED,
        "exclusion_inputs": {
            "verified_500": len(verified),
            "swebench_full_test": len(full_test),
            "held_out_20": len(held_out),
            "smith_three": len(smith_three),
            "old_audit_blocked": len(audit_blocked),
        },
        "ledger": {k: dict(v) for k, v in ledger.items()},
        "sft_overlap_informational": {
            "note": "SWE-smith tasks already SFT-trained via klear66k are retained: "
                    "same-distribution RL curriculum is intentional, not contamination; "
                    "all eval-side exclusions applied above",
            "gym_eligible_in_klear_sft": gym_sft_overlap,
            "smith_eligible_in_klear_sft": smith_sft_overlap,
        },
        "files": {fname: {"rows": len(lines), "sha256": sha256_of_lines(lines)}
                  for fname, lines in outputs.items()},
        "round_consumption_rule": "round-N takes the first unused prefix of the "
                                  "*-order.txt lists; consumed ids are recorded per round",
        "repo_top": {"swe_smith": dict(smith_repos.most_common(10))},
    }
    (ROOT / "configs/agent-rl/rl-task-freeze-v1.json").write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: freeze[k] for k in
                      ("ledger", "sft_overlap_informational", "files")}, indent=2))


if __name__ == "__main__":
    main()
