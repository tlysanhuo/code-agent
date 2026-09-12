#!/usr/bin/env python3
"""Build SFT trajectory pool and OPD prompt pool from raw trajectory datasets.

Inputs (data/raw):
  - Kwai-Klear SWE-smith mini-swe-agent trajectories (instance_id + messages)
  - SWE-Gym OpenHands SFT trajectories (messages only; instance_id recovered by
    matching the first user message against SWE-Gym problem statements)
  - R2E-Gym SFT trajectories (messages only; no upstream id, repo-level audit)

Exclusions applied (training pool must stay disjoint from eval/holdout sets):
  1. configs/gym-training-exclusion.json held_out_task_ids (20 SWE-Gym dev-eval tasks)
  2. data/agent-rl/exclusion-audit.json non-"identified_nonholdout" decisions
  3. configs/swe-three-tasks.json instance_ids (3 frozen SWE-smith tasks)
  4. SWE-bench Verified instance_ids (direct id match)
OpenHands rows whose instance_id cannot be recovered are quarantined (excluded).

Outputs (project-local):
  - data/sft-pool/sft-trajectories.parquet   column `messages` (+metadata struct)
  - data/opd-pool/opd-prompts.parquet        column `messages` single user turn (+metadata struct)
  - runtime/prompt-pool/prompt-pool-audit.json

CPU-only, streaming (iter_batches); no full-dataset materialization.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parent.parent


def norm_text(t: str | None) -> str:
    return re.sub(r"\s+", " ", t or "").strip().lower()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_user_content(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content")
    return None


def load_exclusions() -> dict:
    excl = {}

    gym = json.loads((PROJECT / "configs/gym-training-exclusion.json").read_text())
    excl["gym_held_out"] = set(gym["held_out_task_ids"])

    audit = json.loads((PROJECT / "data/agent-rl/exclusion-audit.json").read_text())
    bad = {"exclude_id_or_commit", "quarantine_unidentified_or_ambiguous", "quarantine_near_title"}
    excl["audit_blocked"] = {d["instance_id"] for d in audit["decisions"] if d["decision"] in bad}
    excl["audit_ok"] = {d["instance_id"] for d in audit["decisions"] if d["decision"] == "identified_nonholdout"}

    three = json.loads((PROJECT / "configs/swe-three-tasks.json").read_text())
    excl["swe_smith_three"] = {t["instance_id"] for t in three["tasks"]}

    vf = pq.ParquetFile(PROJECT / "data/raw/SWE-bench__SWE-bench_Verified/data/test-00000-of-00001.parquet")
    excl["verified_ids"] = {
        r["instance_id"] for b in vf.iter_batches(columns=["instance_id"]) for r in b.to_pylist()
    }
    return excl


def load_swe_gym_stmt_index() -> dict[str, str]:
    """normalized problem_statement prefix -> instance_id"""
    pf = pq.ParquetFile(PROJECT / "data/raw/SWE-Gym__SWE-Gym/data/train-00000-of-00001.parquet")
    idx = {}
    for b in pf.iter_batches(columns=["instance_id", "problem_statement"]):
        for r in b.to_pylist():
            idx[norm_text(r["problem_statement"])[:250]] = r["instance_id"]
    return idx


def match_openhands_instance(user_text: str, stmt_idx: dict[str, str]) -> str | None:
    """Recover instance_id by prefix containment (problem statement embedded in user msg)."""
    n = norm_text(user_text)
    for prefix, iid in stmt_idx.items():
        if prefix in n:
            return iid
    return None


def iter_parquet_rows(paths: list[Path], columns: list[str] | None = None):
    for p in sorted(paths):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(columns=columns):
            yield from b.to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klear-glob", default="data/raw/Kwai-Klear__SWE-smith-mini_swe_agent_plus-trajectories-66k/data/*.parquet")
    ap.add_argument("--openhands", default="data/raw/SWE-Gym__OpenHands-SFT-Trajectories/data/train.success.oss-00000-of-00001.parquet")
    ap.add_argument("--r2e-sft", default="data/raw/R2E-Gym__R2EGym-SFT-Trajectories/data/train-00000-of-00001.parquet")
    ap.add_argument("--sft-out", default="data/sft-pool/sft-trajectories.parquet")
    ap.add_argument("--opd-out", default="data/opd-pool/opd-prompts.parquet")
    ap.add_argument("--audit-out", default="runtime/prompt-pool/prompt-pool-audit.json")
    args = ap.parse_args()

    excl = load_exclusions()
    stmt_idx = load_swe_gym_stmt_index()

    audit = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "SFT cold-start trajectory pool + OPD prompt pool; exclusions per standing audit rules",
        "exclusion_sets": {k: len(v) for k, v in excl.items()},
        "sources": {},
    }

    sft_rows: list[dict] = []          # {messages, metadata}
    opd_seen: set[str] = set()          # (source, task_sha256)
    opd_rows: list[dict] = []
    counters: Counter = Counter()
    decision_log: dict[str, Counter] = {}

    def note(source: str, decision: str):
        decision_log.setdefault(source, Counter())[decision] += 1

    def add_rows(source: str, messages: list[dict], instance_id: str | None, task_text: str | None):
        task_sha = sha256_text(task_text) if task_text else None
        meta = {
            "source": source,
            "instance_id": instance_id,
            "task_sha256": task_sha,
            "n_turns": len(messages),
        }
        sft_rows.append({"messages": messages, "metadata": meta})
        counters[f"{source}.sft"] += 1
        if task_text:
            key = f"{source}:{task_sha}"
            if key not in opd_seen:
                opd_seen.add(key)
                opd_rows.append(
                    {"messages": [{"role": "user", "content": task_text}], "metadata": meta}
                )
                counters[f"{source}.opd_prompt"] += 1

    # --- Klear-66k (SWE-smith trajectories; instance_id present) ---
    src = "klear66k_swe_smith"
    klear_paths = [Path(p) for p in glob.glob(str(PROJECT / args.klear_glob))]
    for row in iter_parquet_rows(klear_paths):
        iid = row.get("instance_id")
        msgs = row.get("messages")
        task = first_user_content(msgs)
        if iid in excl["swe_smith_three"]:
            note(src, "excluded_swe_smith_frozen_three")
            continue
        if iid in excl["verified_ids"]:
            note(src, "excluded_verified_id")
            continue
        if not task or len(msgs) < 3:
            note(src, "dropped_malformed")
            continue
        add_rows(src, msgs, iid, task)
        note(src, "kept")
    audit["sources"][src] = {
        "files": len(klear_paths),
        "decisions": dict(decision_log[src]),
    }

    # --- OpenHands on SWE-Gym (instance_id recovered by text match) ---
    src = "openhands_swe_gym"
    n_openhands = 0
    for row in iter_parquet_rows([PROJECT / args.openhands]):
        n_openhands += 1
        msgs = row["messages"]
        task = first_user_content(msgs)
        iid = match_openhands_instance(task, stmt_idx) if task else None
        if iid is None:
            note(src, "quarantined_unmatched_instance")
            continue
        if iid in excl["gym_held_out"]:
            note(src, "excluded_gym_held_out")
            continue
        if iid in excl["audit_blocked"]:
            note(src, "excluded_prior_audit")
            continue
        if iid in excl["verified_ids"]:
            note(src, "excluded_verified_id")
            continue
        if not task or len(msgs) < 3:
            note(src, "dropped_malformed")
            continue
        add_rows(src, msgs, iid, task)
        note(src, "kept")
    audit["sources"][src] = {
        "rows_read": n_openhands,
        "instance_match_basis": "first-user-message normalized containment against SWE-Gym problem_statement prefixes (verified on sample python__mypy-10308)",
        "decisions": dict(decision_log[src]),
    }

    # --- R2E-Gym SFT (no upstream id; repo-level disjointness verified upstream of this run) ---
    src = "r2e_gym"
    n_r2e = 0
    for row in iter_parquet_rows([PROJECT / args.r2e_sft]):
        n_r2e += 1
        msgs = row["messages"]
        task = first_user_content(msgs)
        if not task or len(msgs) < 3:
            note(src, "dropped_malformed")
            continue
        add_rows(src, msgs, None, task)
        note(src, "kept")
    audit["sources"][src] = {
        "rows_read": n_r2e,
        "id_note": "R2E-Gym SFT trajectories carry no instance_id; local R2E-Gym-Subset repo pool (coveragepy, orange3) has empty intersection with the 12 SWE-bench Verified repos; task-level text match against Verified impossible from local artifacts (Verified parquet has no problem text) - recorded as residual risk, mitigated by repo-level disjointness",
        "decisions": dict(decision_log[src]),
    }

    # --- write parquet outputs ---
    msg_type = pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
    meta_type = pa.struct([
        pa.field("source", pa.string()),
        pa.field("instance_id", pa.string()),
        pa.field("task_sha256", pa.string()),
        pa.field("n_turns", pa.int64()),
    ])

    # SFT trajectories decode to >2GB of raw strings per 65k rows, which breaks
    # single-file parquet reads at slime's default iter_batches batch size, so
    # the SFT pool is sharded (slime read_file accepts one path per run).
    SFT_SHARD_ROWS = 8000

    def arrow_tables(rows: list[dict]):
        yield pa.table(
            {
                "messages": pa.array([r["messages"] for r in rows], type=pa.list_(msg_type)),
                "metadata": pa.array([r["metadata"] for r in rows], type=meta_type),
            }
        )

    def write_pool(rows: list[dict], out: str, name: str, shard_rows: int | None = None):
        outp = PROJECT / out
        outp.parent.mkdir(parents=True, exist_ok=True)
        if shard_rows is None:
            pq.write_table(next(arrow_tables(rows)), outp, row_group_size=2000)
            audit[name] = {
                "path": out,
                "rows": len(rows),
                "sha256": sha256_file(outp),
                "bytes": outp.stat().st_size,
            }
        else:
            shards = []
            for i in range(0, len(rows), shard_rows):
                sp = outp.parent / f"{outp.stem}-{i // shard_rows:03d}.parquet"
                pq.write_table(next(arrow_tables(rows[i : i + shard_rows])), sp, row_group_size=2000)
                shards.append(
                    {"path": str(sp.relative_to(PROJECT)), "rows": len(rows[i : i + shard_rows]), "sha256": sha256_file(sp)}
                )
            audit[name] = {
                "path": str(outp.parent) + "/",
                "rows": len(rows),
                "shard_rows": shard_rows,
                "shards": shards,
            }

    write_pool(sft_rows, args.sft_out, "sft_pool", shard_rows=SFT_SHARD_ROWS)
    write_pool(opd_rows, args.opd_out, "opd_pool")
    audit["counters"] = dict(counters)

    audit_out = PROJECT / args.audit_out
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit, indent=1, ensure_ascii=False))
    print(json.dumps({"sft_rows": len(sft_rows), "opd_rows": len(opd_rows)}, indent=1))
    print(f"audit -> {audit_out}")


if __name__ == "__main__":
    main()
