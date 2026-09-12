"""Held-out trajectory NLL for the stage-1 SFT checkpoint.

Feeds full multi-turn trajectories through vLLM offline with prompt_logprobs
and averages NLL over exactly the tokens the SFT objective supervised: the
same slime MultiTurnLossMaskGenerator(tokenizer_type="qwen3_5") used at
training time (vendor/slime/slime/utils/mask_utils.py, imported standalone).

Two phases, because the pinned envs split cleanly (no pip in .venv-runtime):
  prepare (CPU, .venv-train-rl: transformers 5.12.1 == training tokenizer):
    tokenizes + loss-masks, cross-checks counts vs metadata.n_tokens, writes
    a jsonl of token_ids/loss_mask rows.
  score (GPU, .venv-runtime: vLLM 0.19.1, the P1 serving stack):
    reads the jsonl, runs prompt_logprobs, writes per-row + summary NLL.
--dry-run = prepare with accounting only (still writes the jsonl).

Comparison design (score twice, once per --model-dir): base Qwen3.5-9B vs
converted SFT checkpoint (models/dense-9B-sft/formal-2card/hf-iter500).
Eval splits:
  - train-sample   rows sampled from the actual training parquet (reference)
  - held-out Klear >16k rows (excluded from training by the length filter)
  - OpenHands/R2E <=16k rows (excluded because formal SFT was Klear-only)
"""
import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MASK_UTILS = ROOT / "vendor/slime/slime/utils/mask_utils.py"


def load_mask_generator(tokenizer):
    spec = importlib.util.spec_from_file_location("slime_mask_utils", MASK_UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MultiTurnLossMaskGenerator(tokenizer, "qwen3_5")


def iter_rows(paths, limit, seed):
    rng = random.Random(seed)
    for path in paths:
        tbl = pq.read_table(path, columns=["messages", "metadata"])
        idx = list(range(tbl.num_rows))
        rng.shuffle(idx)
        taken = 0
        for i in idx:
            if taken >= limit:
                break
            meta = tbl["metadata"][i].as_py()
            yield Path(path).stem, meta, tbl["messages"][i].as_py()
            taken += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=Path, default=None,
                    help="score this prepared jsonl on GPU (vLLM env); skips prepare")
    ap.add_argument("--model-dir", default=None,
                    help="prepare: tokenizer source; score: vLLM model dir")
    ap.add_argument("--data", nargs="+", help="prepare only: parquet paths with messages+metadata")
    ap.add_argument("--limit", type=int, default=200, help="rows per file")
    ap.add_argument("--max-tokens", type=int, default=32768, help="skip trajectories longer than this")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--tag", default="nll")
    ap.add_argument("--out", type=Path, default=ROOT / "runtime/sft-formal")
    ap.add_argument("--prep", type=Path, default=None, help="prepare output jsonl path")
    ap.add_argument("--dry-run", action="store_true", help="prepare accounting only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.score is None:
        if not args.model_dir or not args.data:
            ap.error("prepare phase needs --model-dir and --data")
        run_prepare(args)
        return
    run_score(args)


def run_prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    gen = load_mask_generator(tokenizer)

    prepared = []
    n_skip_len = n_token_mismatch = 0
    for stem, meta, messages in iter_rows(args.data, args.limit, args.seed):
        token_ids, loss_mask = gen.get_loss_mask(messages)
        if len(token_ids) > args.max_tokens:
            n_skip_len += 1
            continue
        annotated = (meta or {}).get("n_tokens")
        if annotated is not None and annotated != len(token_ids):
            n_token_mismatch += 1
        n_sup = sum(loss_mask)
        if n_sup == 0:
            n_skip_len += 1
            continue
        prepared.append({"split_hint": stem, "n_tokens": len(token_ids),
                         "n_supervised": n_sup, "token_ids": token_ids,
                         "loss_mask": loss_mask, "instance_id": (meta or {}).get("instance_id")})

    counts = {}
    for p in prepared:
        counts[p["split_hint"]] = counts.get(p["split_hint"], 0) + 1
    print(json.dumps({"rows_prepared": len(prepared), "per_file": counts,
                      "skipped_long_or_empty": n_skip_len,
                      "token_count_mismatches_vs_metadata": n_token_mismatch}, indent=2))
    if n_token_mismatch:
        print("WARNING: token counts diverge from pool annotations; mask semantics changed?",
              file=sys.stderr)
    prep_path = args.prep or (args.out / f"nll-{args.tag}-prepared.jsonl")
    with prep_path.open("w") as f:
        for p in prepared:
            f.write(json.dumps(p) + "\n")
    print(f"prepared rows -> {prep_path}")


def run_score(args):
    rows = [json.loads(l) for l in args.score.open()]
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_dir, max_model_len=args.max_tokens,
              gpu_memory_utilization=0.9, enforce_eager=False)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)

    per_row = []
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]
        outs = llm.generate([p["token_ids"] for p in batch], sp)
        for p, out in zip(batch, outs):
            plp = out.prompt_logprobs
            tot, cnt = 0.0, 0
            for i, m in enumerate(p["loss_mask"]):
                if i == 0 or not m:
                    continue
                lp = plp[i]
                if lp is None:
                    continue
                tok_lp = lp.get(p["token_ids"][i])
                if tok_lp is None:
                    continue
                tot += tok_lp.logprob
                cnt += 1
            per_row.append({"split_hint": p["split_hint"],
                            "instance_id": p.get("instance_id"),
                            "n_tokens": p["n_tokens"], "n_scored": cnt,
                            "nll": -tot / cnt if cnt else None})

    summary = {"tag": args.tag, "model_dir": args.model_dir,
               "prepared_from": str(args.score), "rows": len(per_row)}
    for key in sorted({r["split_hint"] for r in per_row}):
        vals = [r["nll"] for r in per_row if r["split_hint"] == key and r["nll"] is not None]
        toks = sum(r["n_scored"] for r in per_row if r["split_hint"] == key)
        vals_sorted = sorted(vals)
        summary[f"nll/{key}"] = {
            "rows": len(vals), "mean": round(sum(vals) / len(vals), 4) if vals else None,
            "p50": round(vals_sorted[len(vals) // 2], 4) if vals else None,
            "p90": round(vals_sorted[int(len(vals) * 0.9)], 4) if vals else None,
            "scored_tokens": toks,
        }
    (args.out / f"nll-{args.tag}-rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in per_row) + "\n")
    (args.out / f"nll-{args.tag}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
