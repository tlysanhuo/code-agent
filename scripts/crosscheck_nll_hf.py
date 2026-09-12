"""Ground-truth NLL cross-check for eval_sft_nll.py using HF transformers.

Computes the same metric (mean NLL over slime qwen3_5 loss-mask positions)
with a plain HF forward pass — the standard eval-loss computation every SFT
trainer uses. Run with .venv-train-rl (transformers 5.12.1 == training stack)
on one GPU after the vLLM-based scoring finished; compares per-row NLL
against the vLLM script's rows jsonl and reports agreement.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prepared", default=ROOT / "runtime/sft-formal/nll-sft-formal-prepared.jsonl")
    ap.add_argument("--vllm-rows", required=True, help="nll-<tag>-rows.jsonl from eval_sft_nll.py")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-rows", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prepared)][: args.n_rows]
    vllm = {(r["split_hint"], r["instance_id"], r["n_tokens"]): r["nll"]
            for r in map(json.loads, open(args.vllm_rows))}

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    results = []
    with torch.no_grad():
        for r in rows:
            ids = torch.tensor([r["token_ids"]], device="cuda")
            mask = torch.tensor([r["loss_mask"]], device="cuda", dtype=torch.bool)
            logits = model(ids).logits[0]  # [L, V] bf16; one row at a time fits
            tgt = ids[0]
            m = mask[0][1:]
            # chunked log_softmax: position t scored by logits[t-1] against tgt[t]
            tot, cnt = 0.0, 0
            CH = 2048
            for s in range(0, len(tgt) - 1, CH):
                e = min(s + CH, len(tgt) - 1)
                lp = torch.log_softmax(logits[s:e].float(), dim=-1)
                sel = m[s:e]
                if not sel.any():
                    continue
                tok_lp = lp[torch.arange(e - s, device="cuda"), tgt[s + 1:e + 1]]
                tot += tok_lp[sel].sum().item()
                cnt += int(sel.sum().item())
            nll = -tot / cnt if cnt else None
            key = (r["split_hint"], r["instance_id"], r["n_tokens"])
            results.append({"key": key, "hf_nll": nll, "vllm_nll": vllm.get(key)})

    diffs = [abs(x["hf_nll"] - x["vllm_nll"]) for x in results
             if x["hf_nll"] is not None and x["vllm_nll"] is not None]
    out = {
        "tag": args.tag, "model_dir": args.model_dir, "rows_checked": len(results),
        "max_abs_diff": max(diffs) if diffs else None,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
        "verdict": "AGREE" if diffs and max(diffs) < 5e-3 else "DIVERGE",
        "rows": [{"split": k[0], "instance": k[1], "hf": h, "vllm": v}
                 for (k, h, v) in ((x["key"], x["hf_nll"], x["vllm_nll"]) for x in results)],
    }
    print(json.dumps(out, indent=2))
    (ROOT / f"runtime/sft-formal/nll-crosscheck-{args.tag}.json").write_text(
        json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
