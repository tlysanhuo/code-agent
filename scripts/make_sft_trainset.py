#!/usr/bin/env python3
"""Emit a length-bounded SFT training set from annotated pool shards.

Reads only the metadata column (n_tokens) for filtering decisions and streams
matching rows; writes data/sft-pool/train-<cap>k.parquet with the same schema
as the pool shards. CPU-only, bounded memory.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parent.parent
SHARD_GLOB = str(PROJECT / "data/sft-pool/sft-trajectories-*.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = no row cap")
    ap.add_argument("--source-filter", default="", help="keep only this metadata.source value (e.g. klear66k_swe_smith)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = Path(args.out) if args.out else PROJECT / f"data/sft-pool/train-{args.max_tokens // 1024}k.parquet"
    out = out.resolve() if out.is_absolute() else (PROJECT / out).resolve()
    shards = sorted(glob.glob(SHARD_GLOB))
    if not shards:
        raise SystemExit("no annotated shards; run annotate_sft_token_lengths.py first")

    msg_type = pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
    meta_type = pa.struct([
        pa.field("source", pa.string()),
        pa.field("instance_id", pa.string()),
        pa.field("task_sha256", pa.string()),
        pa.field("n_turns", pa.int64()),
        pa.field("n_tokens", pa.int64()),
    ])

    kept_rows = []
    stats = {"scanned": 0, "kept": 0, "per_source": {}}
    for sp in shards:
        pf = pq.ParquetFile(sp)
        for batch in pf.iter_batches():
            rows = batch.to_pylist()
            stats["scanned"] += len(rows)
            for r in rows:
                if args.source_filter and r["metadata"]["source"] != args.source_filter:
                    stats.setdefault("dropped_other_sources", 0)
                    stats["dropped_other_sources"] += 1
                    continue
                if r["metadata"]["n_tokens"] <= args.max_tokens:
                    kept_rows.append(r)
                    stats["kept"] += 1
                    src = r["metadata"]["source"]
                    stats["per_source"][src] = stats["per_source"].get(src, 0) + 1
    if args.max_rows and len(kept_rows) > args.max_rows:
        kept_rows = kept_rows[: args.max_rows]
        stats["kept"] = args.max_rows
        stats["truncated_to_max_rows"] = True

    tbl = pa.table(
        {
            "messages": pa.array([r["messages"] for r in kept_rows], type=pa.list_(msg_type)),
            "metadata": pa.array([r["metadata"] for r in kept_rows], type=meta_type),
        }
    )
    pq.write_table(tbl, out, row_group_size=2000)
    stats.update(
        {
            "built_utc": datetime.now(timezone.utc).isoformat(),
            "max_tokens": args.max_tokens,
            "out": str(out.relative_to(PROJECT)),
            "rows_written": len(kept_rows),
        }
    )
    (PROJECT / f"runtime/prompt-pool/sft-trainset-{out.stem}.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False)
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
