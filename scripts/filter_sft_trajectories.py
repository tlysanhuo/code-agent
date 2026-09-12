#!/usr/bin/env python3
"""Rule-filter the Klear SFT pool before the formal run.

Filters follow Qwen3-Coder-Next TR section 3.1.2 trajectory filtering, adapted
to the mini-swe-agent trajectory contract:
  1. malformed action format: every assistant turn must contain a THOUGHT
     section and exactly one bash code fence (the harness contract);
  2. missing termination: the trajectory must end with an assistant turn
     (submission), not an environment error string;
  3. turn-count sanity: 3 <= turns <= 120;
  4. exact-trajectory dedup (messages hash).

Reads the annotated shards' metadata + messages; writes
data/sft-pool/sft-klear-filtered-16k.parquet plus a stats JSON.
CPU-only, streaming, bounded memory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parent.parent
SHARD_GLOB = str(PROJECT / "data/sft-pool/sft-trajectories-*.parquet")
OUT = PROJECT / "data/sft-pool/sft-klear-filtered-16k.parquet"
STATS = PROJECT / "runtime/prompt-pool/sft-klear-filtered-16k.json"
MAX_TOKENS = 16384

BASH_FENCE = re.compile(r"```bash\n.*?\n```", re.DOTALL)


def assistant_ok(content: str) -> bool:
    has_thought = "THOUGHT:" in content
    fences = BASH_FENCE.findall(content)
    return has_thought and len(fences) == 1


def last_turn_ok(messages: list[dict]) -> bool:
    last = messages[-1]
    if last["role"] != "assistant":
        return False
    # environment crash markers must not terminate a trajectory
    bad = ("command timed out", "Traceback (most recent call last)", "Killed")
    return not any(b in last["content"] for b in bad)


def main() -> None:
    import glob

    shards = sorted(glob.glob(SHARD_GLOB))
    msg_type = pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
    meta_type = pa.struct([
        pa.field("source", pa.string()),
        pa.field("instance_id", pa.string()),
        pa.field("task_sha256", pa.string()),
        pa.field("n_turns", pa.int64()),
        pa.field("n_tokens", pa.int64()),
    ])

    kept: list[dict] = []
    seen: set[str] = set()
    stats: Counter = Counter()
    scanned = 0
    for sp in shards:
        pf = pq.ParquetFile(sp)
        for batch in pf.iter_batches():
            for r in batch.to_pylist():
                scanned += 1
                meta = dict(r["metadata"])
                if meta["source"] != "klear66k_swe_smith":
                    stats["skip_non_klear"] += 1
                    continue
                msgs = r["messages"]
                if meta["n_tokens"] > MAX_TOKENS:
                    stats["drop_length"] += 1
                    continue
                if not (3 <= len(msgs) <= 120):
                    stats["drop_turns"] += 1
                    continue
                if not all(assistant_ok(m["content"]) for m in msgs if m["role"] == "assistant"):
                    stats["drop_format"] += 1
                    continue
                if not last_turn_ok(msgs):
                    stats["drop_no_termination"] += 1
                    continue
                h = hashlib.sha256(
                    "\x1e".join(m["role"] + "\x1f" + m["content"] for m in msgs).encode()
                ).hexdigest()
                if h in seen:
                    stats["drop_dup"] += 1
                    continue
                seen.add(h)
                kept.append({"messages": msgs, "metadata": meta})
                stats["kept"] += 1

    tbl = pa.table(
        {
            "messages": pa.array([r["messages"] for r in kept], type=pa.list_(msg_type)),
            "metadata": pa.array([r["metadata"] for r in kept], type=meta_type),
        }
    )
    pq.write_table(tbl, OUT, row_group_size=2000)
    report = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "filter_basis": "Qwen3-Coder-Next TR 3.1.2 rules adapted to mini-swe-agent contract",
        "scanned": scanned,
        "out": str(OUT.relative_to(PROJECT)),
        "rows_written": len(kept),
        "decisions": dict(stats),
    }
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
