"""Read downloaded data in small batches. Never execute trajectory commands.

This is a structural audit, not execution verification or a final train split.
Lengths are Unicode characters, not tokenizer lengths. Success is upstream's
label. No gold patch or held-out test is exported into model training inputs.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/audit"


def rows(directory, columns=None):
    for path in sorted(directory.glob("data/*.parquet")):
        pf = pq.ParquetFile(path)
        cols = None if columns is None else [c for c in columns if c in pf.schema_arrow.names]
        index = 0
        for batch in pf.iter_batches(batch_size=8, columns=cols):
            for row in batch.to_pylist():
                yield path, index, row
                index += 1


def patch_hash(patch):
    # Conservative additional exact-content match; does not establish decontamination.
    text = "\n".join(line.rstrip() for line in (patch or "").splitlines()
                     if not line.startswith("index "))
    return hashlib.sha256(text.encode()).hexdigest() if text else None


def quantiles(values):
    if not values:
        return {}
    s = sorted(values)
    return {label: s[round((len(s)-1)*q)] for label, q in
            [("min", 0), ("p50", .5), ("p90", .9), ("p95", .95), ("max", 1)]}


def messages_of(row):
    m = row.get("messages", row.get("conversations"))
    if isinstance(m, str):
        m = json.loads(m)
    if not isinstance(m, list) or not all(isinstance(x, dict) for x in m):
        raise ValueError("Unsupported conversation representation")
    return m


def audit_trajectories(name):
    stats = Counter(); roles = Counter(); models = Counter(); lengths = []; turns = []
    seen_ids = set(); seen_messages = set(); candidates = []; examples = []; indicators = Counter()
    for path, index, row in rows(RAW / name):
        stats["rows"] += 1
        if "resolved" in row:
            stats["resolved_true" if row["resolved"] is True else "resolved_false_or_missing"] += 1
        models[str(row.get("model", "not_provided"))] += 1
        ident = row.get("instance_id")
        if ident:
            seen_ids.add(ident)
        try:
            messages = messages_of(row)
        except (ValueError, TypeError):
            stats["unparseable_messages"] += 1
            continue
        stats["parseable_messages"] += 1
        text = "\n".join(x.get("content", "") if isinstance(x.get("content"), str)
                         else json.dumps(x.get("content"), ensure_ascii=False) for x in messages)
        h = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if h in seen_messages:
            stats["duplicate_conversation_rows"] += 1
        seen_messages.add(h)
        lengths.append(len(text));turns.append(len(messages))
        roles.update(str(x.get("role", x.get("from", "missing"))) for x in messages)
        assistant = "\n".join(str(x.get("content", x.get("value", ""))) for x in messages
                              if x.get("role", x.get("from")) in {"assistant", "gpt"})
        for label, pattern in {
            "assistant_xml_function": r"<function=",
            "assistant_fenced_block": r"```",
            "assistant_str_replace_editor": r"str_replace_editor",
            "assistant_execute_bash": r"execute_bash",
            "assistant_think_tag": r"<think>",
            "assistant_git_command": r"\bgit\s+(?:show|log|diff|checkout|status)\b",
        }.items():
            if re.search(pattern, assistant):
                indicators[label] += 1
        patch = row.get("patch") or ""
        paths = re.findall(r"^\+\+\+ b/(.+)$", patch, re.M)
        if paths:
            stats["rows_with_patch_paths"] += 1
            if not any(p in text for p in paths):
                stats["patch_paths_absent_from_conversation_heuristic"] += 1
                if len(examples) < 5:
                    examples.append({"file": str(path.relative_to(ROOT)), "row": index,
                                     "instance_id": ident, "patch_paths": paths[:5],
                                     "note": "Heuristic only; requires upstream/replay investigation"})
        if row.get("resolved") is True:
            candidates.append({"file": str(path.relative_to(ROOT)), "row": index,
                               "instance_id": ident, "traj_id": row.get("traj_id"),
                               "characters": len(text), "messages": len(messages),
                               "conversation_sha256": h, "upstream_resolved": True})
    if candidates:
        with (OUT / (name + ".success_index.jsonl")).open("w") as f:
            for row in candidates:
                f.write(json.dumps(row) + "\n")
    return {"counts": dict(stats), "unique_instance_ids": len(seen_ids),
            "unique_conversations": len(seen_messages), "roles": dict(roles),
            "models": dict(models), "characters": quantiles(lengths),
            "message_count": quantiles(turns), "format_indicators": dict(indicators),
            "patch_mismatch_heuristic_examples": examples,
            "execution_verified_locally": False}


def audit_tasks(name, eval_ids, eval_patches):
    stats = Counter(); repos = Counter(); languages = Counter(); licenses = Counter()
    images = set(); seen_ids = set(); seen_patches = set(); test_counts = []; overlaps = []
    for path, index, row in rows(RAW / name, ["instance_id", "repo", "patch", "problem_statement",
                                           "image_name", "language", "license", "FAIL_TO_PASS", "PASS_TO_PASS"]):
        stats["rows"] += 1
        ident = row.get("instance_id")
        if ident in seen_ids:
            stats["duplicate_instance_ids"] += 1
        seen_ids.add(ident)
        repos[row.get("repo", "missing")] += 1
        languages[row.get("language", "not_provided")] += 1
        licenses[row.get("license", "not_provided")] += 1
        if row.get("image_name"):
            images.add(row["image_name"])
        if not (row.get("problem_statement") or "").strip():
            stats["empty_problem_statement"] += 1
        ft, pt = row.get("FAIL_TO_PASS") or [], row.get("PASS_TO_PASS") or []
        n = len(ft) + len(pt);test_counts.append(n)
        if not ft:
            stats["empty_fail_to_pass"] += 1
        if n > 200:
            stats["over_200_tests"] += 1
        ph = patch_hash(row.get("patch"))
        if ph:
            if ph in seen_patches:
                stats["duplicate_normalized_patch_rows"] += 1
            seen_patches.add(ph)
        if ident in eval_ids or (ph and ph in eval_patches):
            overlaps.append({"instance_id": ident, "id_match": ident in eval_ids,
                             "patch_match": ph in eval_patches, "file": str(path.relative_to(ROOT)), "row": index})
    return {"counts": dict(stats), "unique_repos": len(repos), "top_repos": repos.most_common(15),
            "languages": dict(languages), "licenses": dict(licenses), "unique_images": len(images),
            "test_count": quantiles(test_counts), "verified_overlap": overlaps,
            "overlap_check": "instance_id and normalized exact patch hash only; not semantic decontamination",
            "execution_verified_locally": False}


def main():
    OUT.mkdir(exist_ok=True)
    eval_rows = [r for _, _, r in rows(RAW / "SWE-bench__SWE-bench_Verified", ["instance_id", "patch"])]
    ids = {r["instance_id"] for r in eval_rows}
    patches = {patch_hash(r.get("patch")) for r in eval_rows} - {None}
    result = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "eval_reference_rows": len(eval_rows), "length_unit": "Unicode characters, not tokens",
              "trajectories": {}, "tasks": {}}
    for name in ["SWE-Gym__OpenHands-SFT-Trajectories", "SWE-bench__SWE-smith-trajectories",
                 "R2E-Gym__R2EGym-SFT-Trajectories", "Kwai-Klear__SWE-smith-mini_swe_agent_plus-trajectories-66k"]:
        result["trajectories"][name] = audit_trajectories(name)
        print(name, json.dumps(result["trajectories"][name]), flush=True)
    for name in ["SWE-Gym__SWE-Gym", "SWE-bench__SWE-smith", "nebius__SWE-rebench-V2"]:
        result["tasks"][name] = audit_tasks(name, ids, patches)
        print(name, json.dumps(result["tasks"][name]), flush=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
