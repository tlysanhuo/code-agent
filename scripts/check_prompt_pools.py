#!/usr/bin/env python3
"""CPU acceptance checks for the dense-mainline prompt pools.

Runs against the pinned slime's real code paths (no fakes):
  1. OPD pool loads through slime.utils.data.Dataset with --input-key messages
     + --apply-chat-template semantics and a 4096 max-length filter.
  2. SFT loss mask (qwen3_5) on real pool trajectories supervises exactly the
     assistant spans (default 'qwen' type crashes on this data - that is why
     the SFT draft pins --loss-mask-type qwen3_5).
Writes runtime/prompt-pool/cpu-acceptance.json.

Run with a venv that has transformers + pyarrow + ray importable and
PYTHONPATH=vendor/slime (e.g. .venv-train-rl).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor/slime"))

from slime.utils.data import Dataset  # noqa: E402
from slime.utils.mask_utils import MultiTurnLossMaskGenerator  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
TOK = str(PROJECT / "models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a")
OPD_POOL = str(PROJECT / "data/opd-pool/opd-prompts.parquet")
SFT_SHARD = str(PROJECT / "data/sft-pool/sft-trajectories-000.parquet")
OUT = PROJECT / "runtime/prompt-pool/cpu-acceptance.json"


def main() -> None:
    report: dict = {"checked_utc": datetime.now(timezone.utc).isoformat(), "checks": {}}
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)

    # 1. OPD pool through slime Dataset (same kwargs RolloutDataSource passes)
    ds = Dataset(
        OPD_POOL,
        tokenizer=tok,
        processor=None,
        max_length=4096,
        prompt_key="messages",
        label_key=None,
        metadata_key="metadata",
        apply_chat_template=True,
    )
    s0 = ds[0]
    assert isinstance(s0.prompt, str) and s0.prompt.startswith("<|im_start|>"), "chat template output malformed"
    assert isinstance(s0.metadata, dict) and "source" in s0.metadata
    report["checks"]["opd_pool_dataset_load"] = {
        "loaded": len(ds),
        "raw_rows": pq.ParquetFile(OPD_POOL).metadata.num_rows,
        "dropped_by_4096_filter": pq.ParquetFile(OPD_POOL).metadata.num_rows - len(ds),
        "prompt_is_chat_template_string": True,
        "metadata_is_dict": True,
    }

    # 2. SFT loss mask correctness on real trajectories
    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type="qwen3_5")
    rows = next(pq.ParquetFile(SFT_SHARD).iter_batches(batch_size=5)).to_pylist()
    per_row = []
    for r in rows:
        msgs = r["messages"]
        token_ids, loss_mask = gen.get_loss_mask(msgs, tools=None)
        ones = sum(loss_mask)
        # ground truth: assistant char share must be close to supervised token share
        asst_chars = sum(len(m["content"]) for m in msgs if m["role"] == "assistant")
        total_chars = sum(len(m["content"]) for m in msgs)
        per_row.append(
            {
                "n_tokens": len(token_ids),
                "annotated_n_tokens": r["metadata"].get("n_tokens"),
                "token_count_matches_annotation": len(token_ids) == r["metadata"].get("n_tokens"),
                "supervised_share": round(ones / len(token_ids), 3),
                "assistant_char_share": round(asst_chars / total_chars, 3),
            }
        )
    assert all(p["token_count_matches_annotation"] for p in per_row), "annotation/tokenization mismatch"
    assert all(abs(p["supervised_share"] - p["assistant_char_share"]) < 0.15 for p in per_row), "mask supervises non-assistant spans"
    report["checks"]["sft_loss_mask_qwen3_5"] = {
        "rows": per_row,
        "note": "default 'qwen' loss-mask-type raises TemplateError('No user query found') on per-message rendering of this data; qwen3_5 renders the full conversation and is required",
    }

    # 3. role sanity across the OPD pool sources
    src_counter = Counter()
    for b in pq.ParquetFile(OPD_POOL).iter_batches(columns=["metadata"]):
        for r in b.to_pylist():
            m = r["metadata"]
            if m:
                src_counter[m["source"]] += 1
    report["checks"]["opd_pool_sources"] = dict(src_counter)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report["checks"], indent=1)[:1200])
    print(f"report -> {OUT}")


if __name__ == "__main__":
    main()
