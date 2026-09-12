"""P1 degradation measurement: HumanEvalPlus subset, base-check protocol.

Both models (original dense 9B and loopified expanded) go through the identical
server pipeline: same instruction template, temperature 0, same grader.
This is the HumanEvalPlus base-check protocol (the embedded check(candidate)
asserts), not the full evalplus plus_input harness; it is a relative
degradation metric, not an official evalplus score.

Usage:
  python scripts/eval_loopify_humaneval.py --endpoint http://127.0.0.1:18092/v1 \
      --model LoopQwen-5.6B --tag loop --limit 32
"""
import argparse
import concurrent.futures
import gzip
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/evalplus__HumanEvalPlus/v0.1.10/HumanEvalPlus.jsonl.gz"
OUTDIR = ROOT / "runtime/loop-agent"
INSTRUCTION = ("Complete the following Python function. Return exactly one Python "
               "code block containing the complete function definition and nothing else.\n\n")
CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def chat(endpoint, model, content, max_tokens):
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(f"{endpoint}/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": "Bearer local-qwen"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def extract_code(text):
    m = CODE_BLOCK.search(text)
    return m.group(1).strip() if m else text.strip()


def grade(problem, code):
    if "def " not in code:
        return "no_function"
    script = code + "\n\n" + problem["test"] + f"\ncheck({problem['entry_point']})\n"
    try:
        p = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=8)
        return "pass" if p.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    problems = [json.loads(l) for l in gzip.open(DATASET, "rt")][:args.limit]
    started = time.time()

    def one(problem):
        try:
            text = chat(args.endpoint, args.model, INSTRUCTION + problem["prompt"], args.max_tokens)
        except Exception as exc:
            return {"task_id": problem["task_id"], "status": f"request_error:{type(exc).__name__}"}
        code = extract_code(text)
        return {"task_id": problem["task_id"], "status": grade(problem, code), "code": code}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, problems))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / f"humaneval-{args.tag}.jsonl"
    with detail_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    passed = counts.get("pass", 0)
    summary = {"tag": args.tag, "model": args.model, "endpoint": args.endpoint,
               "protocol": "HumanEvalPlus base-check, first N problems, greedy, identical template",
               "limit": args.limit, "passed": passed,
               "pass_rate": round(passed / len(results), 4), "counts": counts,
               "wall_seconds": round(time.time() - started, 1),
               "detail": str(detail_path.relative_to(ROOT))}
    (OUTDIR / f"humaneval-{args.tag}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
