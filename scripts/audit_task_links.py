"""Inspect task identity links without reading or exporting gold solutions."""
from collections import Counter
import json
from pathlib import Path
import time

from audit_data import rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / "data/raw"
    tasks = {}
    for _, _, r in rows(raw / "SWE-bench__SWE-smith",
                        ["instance_id", "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS"]):
        # Retain scalar summaries only, not millions of test identifier strings.
        tasks[r["instance_id"]] = {
            "has_statement": bool((r.get("problem_statement") or "").strip()),
            "test_count": len(r.get("FAIL_TO_PASS") or []) + len(r.get("PASS_TO_PASS") or []),
        }
    datasets = {}
    ids_by_source = {}
    for name in ["Kwai-Klear__SWE-smith-mini_swe_agent_plus-trajectories-66k", "SWE-bench__SWE-smith-trajectories"]:
        ids = Counter(r["instance_id"] for _, _, r in rows(raw / name, ["instance_id"]))
        ids_by_source[name] = set(ids)
        linked = set(ids) & set(tasks)
        nonempty = {i for i in linked if tasks[i]["has_statement"]}
        cheap = {i for i in nonempty if tasks[i]["test_count"] <= 200}
        datasets[name] = {"trajectory_rows": sum(ids.values()), "unique_tasks": len(ids),
                          "linked_tasks": len(linked), "unlinked_tasks": len(ids)-len(linked),
                          "linked_tasks_with_nonempty_problem": len(nonempty),
                          "linked_nonempty_tasks_at_most_200_tests": len(cheap),
                          "max_trajectories_per_task": max(ids.values()),
                          "mean_trajectories_per_task": sum(ids.values()) / len(ids),
                          "unlinked_examples": sorted(set(ids)-linked)[:10]}
    sets = list(ids_by_source.values())
    result = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "datasets": datasets, "shared_tasks_between_trajectory_sources": len(sets[0] & sets[1]),
              "note": "Identity and rough cost audit only; no container startup or success verification"}
    (ROOT / "data/audit/task_links.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
