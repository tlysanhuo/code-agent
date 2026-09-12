#!/usr/bin/env python3
"""Annotate SFT pool shards with exact full-conversation token counts.

Runs with a venv that has both pyarrow and transformers (.venv-train-rl).
Streams one shard at a time; rewrites each shard with metadata.n_tokens added
(chat-templated full conversation, matching slime sft_rollout tokenization).
Updates runtime/prompt-pool/prompt-pool-audit.json with a length histogram.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
TOK_PATH = str(PROJECT / "models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a")
SHARD_GLOB = str(PROJECT / "data/sft-pool/sft-trajectories-*.parquet")
AUDIT = PROJECT / "runtime/prompt-pool/prompt-pool-audit.json"

msg_type = pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
meta_fields = [
    pa.field("source", pa.string()),
    pa.field("instance_id", pa.string()),
    pa.field("task_sha256", pa.string()),
    pa.field("n_turns", pa.int64()),
    pa.field("n_tokens", pa.int64()),
]
meta_type = pa.struct(meta_fields)

BATCH = 64  # conversations per tokenizer call, bounded memory


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    shards = sorted(glob.glob(SHARD_GLOB))
    if not shards:
        sys.exit(f"no shards match {SHARD_GLOB}")

    hist = Counter()
    total = 0
    for sp in shards:
        pf = pq.ParquetFile(sp)
        out_rows = []
        for batch in pf.iter_batches():
            rows = batch.to_pylist()
            for i in range(0, len(rows), BATCH):
                chunk = rows[i : i + BATCH]
                outs = [tok.apply_chat_template(r["messages"], tokenize=True) for r in chunk]
                for r, out in zip(chunk, outs):
                    n = len(out["input_ids"])
                    meta = dict(r["metadata"])
                    meta["n_tokens"] = n
                    out_rows.append({"messages": r["messages"], "metadata": meta})
                    total += 1
                    hist[min(n // 4096, 16)] += 1  # buckets of 4k, capped at >=64k
        tbl = pa.table(
            {
                "messages": pa.array([r["messages"] for r in out_rows], type=pa.list_(msg_type)),
                "metadata": pa.array([r["metadata"] for r in out_rows], type=meta_type),
            }
        )
        tmp = sp + ".tmp"
        pq.write_table(tbl, tmp, row_group_size=2000)
        Path(tmp).replace(sp)
        print(f"annotated {sp} rows={len(out_rows)}", flush=True)

    audit = json.loads(AUDIT.read_text())
    audit["sft_pool"]["token_length"] = {
        "annotated_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": "models/Qwen3.5-9B chat template, full conversation",
        "total": total,
        "bucket_4k": {f"{k*4096}-{(k+1)*4096 if k < 16 else 'inf'}": v for k, v in sorted(hist.items())},
    }
    AUDIT.write_text(json.dumps(audit, indent=1, ensure_ascii=False))
    print("audit updated")


if __name__ == "__main__":
    main()
