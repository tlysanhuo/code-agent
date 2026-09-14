"""Merge the round1b qualification outputs into one registry + prompt parquet.

Inputs:
  [0..197)    logs/prepare-rl-round1b-full.log  (original batch was killed before
              writing its registry; reconstruct from qual=OK lines + case dirs)
  [197..397)  configs/agent-rl/local-task-registry-r1b-w1.json (written at worker exit)
  [397..588)  configs/agent-rl/local-task-registry-r1b-w2.json (written at worker exit)

Fail-safe: refuses to merge unless both worker registries exist and both worker
logs ended with their final summary JSON. Reconstruction cross-checks that every
qual=OK task from the original log still has its case dir + runtime-config.json.

Expected overlap: the killed original run processed one task past its stop index
([198/588] = index 197 = w1's first --skip-197 task), so exactly that iid may
appear in both the reconstructed segment and w1's registry. Overlapping iids are
deduped with worker-registry priority (w2 > w1 > reconstructed) and recorded in
the report; any python-path disagreement on an overlap is a hard failure.

Outputs (absolute paths):
  configs/agent-rl/local-task-registry-r1b-merged.json
  data/agent-rl/rl-round1b-merged-prompts.parquet
  runtime/agent-rl/round1b-merge-report.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import types as _types
if "gym_facility" not in sys.modules:
    _stub = _types.ModuleType("gym_facility")
    _stub.execute = None
    sys.modules["gym_facility"] = _stub

import gym_prepare as gp

LOG_FULL = ROOT / "logs/prepare-rl-round1b-full.log"
LOG_W1 = ROOT / "logs/prepare-rl-round1b-w1.log"
LOG_W2 = ROOT / "logs/prepare-rl-round1b-w2.log"
REG_W1 = ROOT / "configs/agent-rl/local-task-registry-r1b-w1.json"
REG_W2 = ROOT / "configs/agent-rl/local-task-registry-r1b-w2.json"
OUT_DIR_FULL = ROOT / "runtime/agent-rl/round1b"
OUT_REG = ROOT / "configs/agent-rl/local-task-registry-r1b-merged.json"
OUT_PARQUET = ROOT / "data/agent-rl/rl-round1b-merged-prompts.parquet"
OUT_REPORT = ROOT / "runtime/agent-rl/round1b-merge-report.json"

OK_LINE = re.compile(r"^\[\d+/\d+\] (\S+): qual=OK\b")


def ok_ids(log: Path) -> list[str]:
    ids = []
    for line in log.read_text().splitlines():
        m = OK_LINE.match(line.strip())
        if m:
            ids.append(m.group(1))
    return ids


def worker_finished(log: Path) -> bool:
    text = log.read_text().strip().splitlines()
    return bool(text) and text[-1].strip() == "}" and '"qualified"' in "\n".join(text[-8:])


def reconstruct_segment() -> tuple[dict, list[str]]:
    """Rebuild registry entries for the killed original batch [0..197]."""

    registry, missing = {}, []
    for iid in ok_ids(LOG_FULL):
        case = OUT_DIR_FULL / iid
        cfg = case / "runtime-config.json"
        if not cfg.is_file():
            missing.append(iid)
            continue
        python = json.loads(cfg.read_text())["task_python"]
        registry[iid] = {"case_dir": str(case), "python": str(python)}
    return registry, missing


def main() -> int:
    problems = []
    if not REG_W1.is_file() or not REG_W2.is_file():
        problems.append(f"worker registry missing: w1={REG_W1.is_file()} w2={REG_W2.is_file()}")
    for name, log in (("w1", LOG_W1), ("w2", LOG_W2)):
        if not worker_finished(log):
            problems.append(f"{name} log has no final summary (worker still running or crashed)")
    if problems:
        print(json.dumps({"merged": False, "problems": problems}, indent=1))
        return 2

    reg_full, missing = reconstruct_segment()
    reg_w1 = json.loads(REG_W1.read_text())
    reg_w2 = json.loads(REG_W2.read_text())
    if missing:
        print(json.dumps({"merged": False, "missing_case_dirs": missing[:20], "count": len(missing)}, indent=1))
        return 3

    overlap = {}
    for keep, drop in ((reg_w2, reg_w1), (reg_w2, reg_full), (reg_w1, reg_full)):
        for iid in set(keep) & set(drop):
            if keep[iid]["python"] != drop[iid]["python"]:
                problems.append(f"overlap {iid} python disagreement: {keep[iid]['python']} vs {drop[iid]['python']}")
            overlap[iid] = keep[iid]
    if problems:
        print(json.dumps({"merged": False, "problems": problems}, indent=1))
        return 4

    merged = {**reg_full, **reg_w1, **reg_w2}
    dead = [iid for iid, e in merged.items()
            if not (Path(e["python"]).is_file()
                    and (Path(e["case_dir"]) / "sandbox" / "workspace").is_dir())]
    if dead:
        print(json.dumps({"merged": False, "dead_paths": dead[:20], "count": len(dead)}, indent=1))
        return 4
    # Prompt rows rebuilt exactly like the worker writes them (same raw parquet).
    import pyarrow as pa
    import pyarrow.parquet as pq

    prompts = []
    for iid, entry in merged.items():
        row = gp.task(iid)
        case = Path(entry["case_dir"])
        meta = {
            "instance_id": iid,
            "image": "local",
            "workdir": str(case / "sandbox" / "workspace"),
            "problem_statement": row["problem_statement"],
            "local": {"case_dir": str(case), "python": entry["python"], "row": row},
        }
        prompts.append({"prompt": row["problem_statement"], "metadata": meta})

    OUT_REG.write_text(json.dumps(merged, indent=1) + "\n")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(prompts), OUT_PARQUET)

    log_counts = {
        "full_0_197": len(ok_ids(LOG_FULL)),
        "w1_197_397": len(ok_ids(LOG_W1)),
        "w2_397_588": len(ok_ids(LOG_W2)),
    }
    repo_counts: dict[str, int] = {}
    for iid in merged:
        repo_counts[iid.rsplit("__", 1)[0]] = repo_counts.get(iid.rsplit("__", 1)[0], 0) + 1
    report = {
        "merged_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "qualified_total": len(merged),
        "log_ok_counts": log_counts,
        "overlap_deduped": sorted(overlap),
        "consistency": log_counts["full_0_197"] + log_counts["w1_197_397"] + log_counts["w2_397_588"]
                       - len(overlap) == len(merged),
        "per_repo": repo_counts,
        "registry": str(OUT_REG),
        "parquet": str(OUT_PARQUET),
        "parquet_rows": len(prompts),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if report["consistency"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
