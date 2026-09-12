"""Stage-1 SFT value A/B: base vs SFT zero-shot SWE trajectories.

Question: did the Klear-format SFT buy agentic trajectory capability, not just
lower NLL? Protocol = the EXACT format the SFT trained on, replayed verbatim
(system prompt, <pr_description>/<instructions> first turn, THOUGHT+bash-block
turns, <returncode>/<output> observations, MINI_SWE_AGENT_FINAL_OUTPUT
submission marker) — extracted at runtime from the training parquet so the
template cannot drift from what the model saw.

Environment/grading reuse (no new evaluator): per-task workspaces and buggy/gold
qualified pythons from runtime/gym-baseline/freeze-v2.json; grading via the
frozen gym_run.evaluate (trusted clean-index patch export, oracle test
restoration, F2P/P2P run). Both models run under identical budgets.

Usage: vLLM-serve the model (P1 params + 40k ctx), then
  .venv-train-rl/bin/python scripts/eval_sft_ab_trajectories.py \
      --endpoint http://127.0.0.1:18092/v1 --model-dir <hf> --tag base
"""
import argparse
import concurrent.futures
import importlib.util
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# gym_run imports gym_facility (DSH SDK lives in another venv) but evaluate()
# never uses it; stub the module so the frozen grading path imports cleanly.
import types as _types  # noqa: E402

_stub = _types.ModuleType("gym_facility")
_stub.execute = None
sys.modules.setdefault("gym_facility", _stub)

import gym_prepare as gp  # noqa: E402
import gym_run as gr  # noqa: E402

BASH = re.compile(r"```bash\n(.*?)```", re.S)
OBS_CAP = 10000


def load_template():
    import pyarrow.parquet as pq

    row = pq.read_table(ROOT / "data/sft-pool/sft-trajectories-000.parquet",
                        columns=["messages"])["messages"][0].as_py()
    system = row[0]["content"]
    first_user = row[1]["content"]
    cut = first_user.index("</pr_description>")
    head_end = first_user.index("Consider the following PR description:\n") \
        + len("Consider the following PR description:\n")
    return {"system": system,
            "user_header": first_user[:head_end],
            "user_tail": first_user[cut:],
            "raw_example": first_user}


def edit_via_str_replace_cli(workspace: Path):
    """The helper named by the Klear instructions; behavior per its spec:
    replace only when the target appears exactly once, else error."""
    code = '''#!/usr/bin/env python
import argparse, sys
p = argparse.ArgumentParser()
p.add_argument("file"); p.add_argument("old"); p.add_argument("new")
a = p.parse_args()
t = open(a.file).read()
n = t.count(a.old)
if n != 1:
    line = t[:t.index(a.old) if a.old in t else 0].count("\\n") + 1 if a.old in t else -1
    sys.exit(f"error: target occurs {n} times (first near line {line}); refusing ambiguous replace")
open(a.file, "w").write(t.replace(a.old, a.new, 1))
print(f"replaced 1 occurrence in {a.file}")
'''
    (workspace / "edit_via_str_replace").write_text(code)


class Episode:
    def __init__(self, args, template, tokenizer, entry, index):
        self.args = args
        self.t = template
        self.tok = tokenizer
        self.entry = entry
        self.index = index
        self.case = Path(args.out) / f"task-{index:02d}"
        self.row = gp.task(entry["instance_id"])
        self.python = Path(entry["python"])

    def chat(self, messages):
        payload = {"model": self.args.model_name,
                   "messages": messages, "max_tokens": 1024, "temperature": 0}
        req = urllib.request.Request(
            f"{self.args.endpoint}/chat/completions", json.dumps(payload).encode(),
            {"Content-Type": "application/json", "Authorization": "Bearer local-qwen"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        return d["choices"][0]["message"]["content"] or "", d.get("usage", {})

    def n_tokens(self, messages):
        n = 0
        for m in messages:
            n += len(self.tok(m["content"])["input_ids"]) + 8
        return n

    def run_command(self, cmd, workspace, env):
        try:
            p = subprocess.run(["bash", "-c", cmd], cwd=workspace, env=env,
                               capture_output=True, text=True, timeout=self.args.step_timeout)
            out = p.stdout + (("\n" + p.stderr) if p.stderr.strip() else "")
            return p.returncode, out
        except subprocess.TimeoutExpired:
            return 124, "command timed out"

    def episode(self):
        gp.make_case(self.case, self.row, self.python)
        workspace = self.case / "sandbox/workspace"
        edit_via_str_replace_cli(workspace)
        env = dict(**__import__("os").environ)
        env["PATH"] = f"{self.python.parent}:{env.get('PATH', '')}"

        first_user = (self.t["user_header"] + self.row["problem_statement"]
                      + "\n" + self.t["user_tail"])
        messages = [{"role": "system", "content": self.t["system"]},
                    {"role": "user", "content": first_user}]
        rec = {"instance_id": self.row["instance_id"], "steps": 0,
               "format_errors": 0, "finished": False, "context_overflow": False,
               "wall_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        started = time.monotonic()
        consec_err = 0
        while rec["steps"] < self.args.max_steps:
            if time.monotonic() - started > self.args.wall:
                rec["stopped"] = "wall_budget"; break
            if self.n_tokens(messages) > self.args.max_ctx_tokens:
                rec["context_overflow"] = True; rec["stopped"] = "context"; break
            try:
                text, usage = self.chat(messages)
            except Exception as exc:
                rec["stopped"] = f"request_error:{type(exc).__name__}"; break
            rec["prompt_tokens"] += usage.get("prompt_tokens", 0)
            rec["completion_tokens"] += usage.get("completion_tokens", 0)
            blocks = BASH.findall(text)
            rec["steps"] += 1
            if len(blocks) != 1:
                rec["format_errors"] += 1
                consec_err += 1
                obs = ("ERROR: your response must contain exactly ONE bash code "
                       "block with ONE command; this response was rejected.")
                if consec_err >= 3:
                    messages += [{"role": "assistant", "content": text},
                                 {"role": "user", "content": obs}]
                    rec["stopped"] = "format_errors"; break
            else:
                consec_err = 0
                rc, out = self.run_command(blocks[0].strip(), workspace, env)
                if len(out) > OBS_CAP:
                    out = out[:300] + "\n... (truncated) ...\n" + out[-OBS_CAP + 320:]
                obs = f"<returncode>{rc}</returncode>\n<output>\n{out}\n</output>"
                if "MINI_SWE_AGENT_FINAL_OUTPUT" in out:
                    messages += [{"role": "assistant", "content": text},
                                 {"role": "user", "content": obs}]
                    rec["finished"] = True; rec["stopped"] = "submitted"; break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": obs}]
        else:
            rec["stopped"] = "step_budget"
        rec["wall_seconds"] = round(time.monotonic() - started, 1)
        (self.case / "episode.json").write_text(json.dumps(rec, indent=2) + "\n")
        (self.case / "messages.json").write_text(json.dumps(messages, ensure_ascii=False,
                                                            indent=1) + "\n")
        return rec

    def full(self):
        rec = self.episode()
        grade_started = time.monotonic()
        try:
            result = gr.evaluate(self.case, self.row, self.python)
            rec["resolved"] = bool(result.get("resolved"))
            rec["evaluation"] = result
        except Exception as exc:
            rec["resolved"] = False
            rec["evaluation_error"] = f"{type(exc).__name__}: {exc}"
        rec["evaluation_seconds"] = round(time.monotonic() - grade_started, 1)
        (self.case / "episode.json").write_text(json.dumps(rec, indent=2) + "\n")
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-dir", required=True, help="tokenizer source")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--step-timeout", type=int, default=120)
    ap.add_argument("--wall", type=int, default=1800)
    ap.add_argument("--max-ctx-tokens", type=int, default=36000)
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()
    args.out = args.out or (ROOT / f"runtime/sft-ab/{args.tag}")
    args.out.mkdir(parents=True, exist_ok=True)

    template = load_template()
    frozen = json.loads((ROOT / "runtime/gym-baseline/freeze-v2.json").read_text())
    entries = [{"instance_id": t["instance_id"], "python": t["python"]}
               for t in frozen["tasks"]]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    lock = threading.Lock()
    done = [0]

    def run_one(item):
        idx, entry = item
        try:
            rec = Episode(args, template, tok, entry, idx + 1).full()
        except Exception as exc:
            rec = {"instance_id": entry["instance_id"], "fatal":
                   f"{type(exc).__name__}: {exc}", "resolved": False}
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(entries)}] {entry['instance_id']}: "
                  f"resolved={rec.get('resolved')} steps={rec.get('steps')} "
                  f"stopped={rec.get('stopped')}", flush=True)
        return rec

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        recs = list(pool.map(run_one, list(enumerate(entries))))

    n = len(recs)
    resolved = sum(1 for r in recs if r.get("resolved"))
    summary = {
        "tag": args.tag, "model_dir": args.model_dir, "endpoint": args.endpoint,
        "protocol": "Klear mini-swe-agent-plus replay (verbatim template from "
                    "sft-trajectories-000.parquet); gym frozen evaluator; identical "
                    "budgets both arms",
        "budgets": {"max_steps": args.max_steps, "step_timeout": args.step_timeout,
                    "wall_sec": args.wall, "max_ctx_tokens": args.max_ctx_tokens},
        "tasks": n, "resolved": resolved,
        "resolved_rate": round(resolved / n, 4) if n else None,
        "finished_marker_rate": round(sum(1 for r in recs if r.get("finished")) / n, 4),
        "nonempty_patch_rate": round(
            sum(1 for r in recs
                if (r.get("evaluation") or {}).get("patch_bytes", 0) > 0) / n, 4),
        "format_errors_total": sum(r.get("format_errors", 0) for r in recs),
        "mean_steps": round(sum(r.get("steps", 0) for r in recs) / n, 1),
        "context_overflow_count": sum(1 for r in recs if r.get("context_overflow")),
        "wall_minutes_total": round((time.monotonic() - t0) / 60, 1),
        "cases": recs,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
