"""Deterministically sample local trajectories and count Qwen3.5 text tokens.

Counts each message's content independently without chat-template wrappers or
separate tool-call fields: these are content-token counts, not training lengths.
The final training template and assistant-only masks still need validation.
"""
import json
from pathlib import Path
import random
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from audit_data import messages_of, quantiles

ROOT = Path(__file__).resolve().parents[1]


def parquet_sample(directory, count=128):
    files = [(p, pq.ParquetFile(p)) for p in sorted(directory.glob("data/*.parquet"))]
    total = sum(pf.metadata.num_rows for _, pf in files)
    selected = set(random.Random(20260908).sample(range(total), min(count, total)))
    offset = 0
    for path, pf in files:
        indexes = {i-offset for i in selected if offset <= i < offset+pf.metadata.num_rows}
        if indexes:
            cursor = 0
            for batch in pf.iter_batches(batch_size=32, columns=["messages"]):
                for i in indexes:
                    if cursor <= i < cursor+batch.num_rows:
                        yield path.name, i, batch.slice(i-cursor, 1).to_pylist()[0]
                cursor += batch.num_rows
        offset += pf.metadata.num_rows


def main():
    tokenizer = Tokenizer.from_file(str(ROOT / "cache/tokenizers/Qwen3.5-9B/tokenizer.json"))
    result = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "seed": 20260908, "unit": "Qwen3.5 content tokens; excludes chat template and separate tool_call fields",
              "datasets": {}}
    names = ["SWE-Gym__OpenHands-SFT-Trajectories", "SWE-bench__SWE-smith-trajectories",
             "R2E-Gym__R2EGym-SFT-Trajectories", "Kwai-Klear__SWE-smith-mini_swe_agent_plus-trajectories-66k",
             "allenai__SERA-4.6-Lite-Best-Subset"]
    for name in names:
        directory = ROOT / "data/raw" / name
        if name.startswith("allenai"):
            with (directory / "first_100.sample.jsonl").open() as f:
                sample = [("first_100.sample.jsonl", i, json.loads(line)) for i, line in enumerate(f)]
        else:
            sample = parquet_sample(directory)
        totals = []; assistants = []; records = []
        for file, index, row in sample:
            messages = messages_of(row)
            total = assistant = 0
            for message in messages:
                content = message.get("content", "")
                if not isinstance(content, str):
                    # Text blocks only; ignore image/video content in this text audit.
                    content = "\n".join(x.get("text", "") for x in content if isinstance(x, dict)) if isinstance(content, list) else ""
                n = len(tokenizer.encode(content, add_special_tokens=False).ids)
                total += n
                if message.get("role") == "assistant":
                    assistant += n
            totals.append(total);assistants.append(assistant)
            records.append({"file": file, "row": index, "content_tokens": total, "assistant_content_tokens": assistant})
        result["datasets"][name] = {"sample_rows": len(totals), "content_tokens": quantiles(totals),
                                     "assistant_content_tokens": quantiles(assistants),
                                     "content_over_16k": sum(n > 16384 for n in totals),
                                     "content_over_32k": sum(n > 32768 for n in totals),
                                     "sampling": "first 100, nonrandom" if name.startswith("allenai") else "uniform row sample across downloaded files",
                                     "records": records}
        print(name, json.dumps({k:v for k,v in result['datasets'][name].items() if k!='records'}), flush=True)
    (ROOT / "data/audit/token_lengths.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
