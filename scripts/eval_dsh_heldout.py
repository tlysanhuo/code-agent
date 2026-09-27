#!/usr/bin/env python3
"""Held-out DSH-path eval for RL checkpoints (mid-training offline eval).

MiMo-style checkpoint eval, our scale: run the DeepSeek Harness agent zero-shot
on the stage-1 A/B frozen held-out set (runtime/gym-baseline/freeze-v2.json,
20 tasks) against a served checkpoint, graded by the same frozen gym evaluator
that produces RL rewards. This measures the thing RL optimizes (in-harness
solve rate on unseen tasks), complementing the on-policy reward curve.

Reuses proven pieces verbatim: gp.make_case workspace prep, screen_worker.py
DSH rollout in .venv-dsh (configs/gym-dsh.patch.yml: 32k window — the
screening-proven pair), gr.evaluate grading. The prompt is the RL rollout
prompt (identical to the one DshHarness sends during training), with
PROBLEM_STATEMENT.md written into the workspace the way prepare_workspace does.

Usage (server must be up first, see scripts/eval_rl_checkpoints.sh):
  .venv-train-rl/bin/python scripts/eval_dsh_heldout.py \
      --base-url http://127.0.0.1:18097/v1 --tag iter39 [--k 1]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
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

# Verbatim from the DshHarness job prompt (RL rollouts train on this text).
RL_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the "
    "relevant tests to verify your fix passes. Do NOT modify "
    "PROBLEM_STATEMENT.md and do NOT commit. When finished, print a one-line "
    "summary and exit."
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_attempt(entry: dict, attempt: int, args) -> dict:
    iid = entry["instance_id"]
    case = Path(args.out) / iid / f"a{attempt}"
    case.mkdir(parents=True, exist_ok=True)
    row = gp.task(iid)
    python = Path(entry["python"])
    gp.make_case(case, row, python)  # fresh buggy workspace
    ws = case / "sandbox/workspace"
    (ws / "PROBLEM_STATEMENT.md").write_text(row["problem_statement"])
    (case / "screen-prompt.txt").write_text(RL_PROMPT)

    result = {"instance_id": iid, "attempt": attempt,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.monotonic()
    worker_env = {**os.environ, "QWEN_BASE_URL": args.base_url,
                  "SCREEN_ROLLOUT_TIMEOUT_S": str(args.rollout_timeout)}
    worker = subprocess.run(
        [str(ROOT / ".venv-dsh/bin/python"), str(ROOT / "scripts/screen_worker.py"), str(case)],
        env=worker_env, capture_output=True, text=True,
        timeout=args.rollout_timeout + 300, cwd=str(ROOT))
    dsh: dict = {}
    dsh_file = case / "dsh-result.json"
    if dsh_file.exists():
        try:
            dsh = json.loads(dsh_file.read_text())
        except Exception:
            pass
    result.update(dsh_status=dsh.get("status", "missing"),
                  dsh_finish_reason=dsh.get("finish_reason"),
                  dsh_seconds=dsh.get("seconds"), worker_rc=worker.returncode)
    if result["dsh_status"] != "ok":
        result["worker_stderr_tail"] = (worker.stderr or "")[-400:]

    try:
        ev = gr.evaluate(case, row, python)
        result.update({k: ev[k] for k in
                       ("resolved", "category", "f2p_passed", "f2p_total",
                        "p2p_passed", "p2p_total", "tool_calls")})
    except Exception as exc:
        result.update(resolved=False, category=f"eval_error:{type(exc).__name__}",
                      error=str(exc)[:300])
    result["seconds"] = round(time.monotonic() - t0, 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out DSH-path checkpoint eval")
    ap.add_argument("--base-url", required=True, help="vLLM OpenAI base URL")
    ap.add_argument("--tag", required=True, help="row tag, e.g. iter39 / sft500")
    ap.add_argument("--model-dir", default=None, help="recorded in the summary")
    ap.add_argument("--k", type=int, default=1, help="attempts per task (avg@k)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rollout-timeout", type=int, default=1500)
    ap.add_argument("--tasks", type=Path,
                    default=ROOT / "runtime/gym-baseline/freeze-v2.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all 20")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.out = args.out or (ROOT / f"runtime/agent-rl/heldout-evals/{args.tag}")
    args.out.mkdir(parents=True, exist_ok=True)

    frozen = json.loads(args.tasks.read_text())
    entries = [{"instance_id": t["instance_id"], "python": t["python"]}
               for t in frozen["tasks"]]
    if args.limit:
        entries = entries[: args.limit]

    # server sanity before burning time on workspace prep
    import urllib.request
    req = urllib.request.Request(args.base_url.rstrip("/") + "/models",
                                 headers={"Authorization": "Bearer local-qwen"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200, "server not reachable"
    log(f"server OK; {len(entries)} tasks x k={args.k}, concurrency {args.concurrency}")

    jobs = [(e, k) for e in entries for k in range(1, args.k + 1)]
    results_path = args.out / "results.jsonl"
    lock, done = threading.Lock(), [0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        def wrapped(job):
            entry, attempt = job
            rec = run_attempt(entry, attempt, args)
            with lock:
                done[0] += 1
                log(f"[{done[0]}/{len(jobs)}] {entry['instance_id']} a{attempt}: "
                    f"resolved={rec.get('resolved')} "
                    f"finish={rec.get('dsh_finish_reason')} {rec.get('seconds')}s")
                with results_path.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec
        recs = list(pool.map(wrapped, jobs))

    solved_tasks = {r["instance_id"] for r in recs if r.get("resolved")}
    n = len(entries)
    per_task = [(iid, sum(1 for r in recs if r["instance_id"] == iid and r.get("resolved")),
                 args.k) for iid in [e["instance_id"] for e in entries]]
    avg_k = sum(c for _, c, _ in per_task) / (n * args.k)
    any_solved = len(solved_tasks)
    summary = {
        "tag": args.tag, "model_dir": args.model_dir, "base_url": args.base_url,
        "n_tasks": n, "k": args.k, "avg_at_k": round(avg_k, 4),
        "any_solved_tasks": any_solved, "any_solved_rate": round(any_solved / n, 4),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "categories": {c: sum(1 for r in recs if r.get("category") == c)
                       for c in {r.get("category") for r in recs}},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [f"# Held-out DSH eval: {args.tag}", "",
             f"- model: `{args.model_dir}`", f"- tasks: {n} x k={args.k}",
             f"- **avg@{args.k} = {avg_k:.1%}** | any-solved: {any_solved}/{n} "
             f"({any_solved / n:.1%})", f"- categories: {summary['categories']}", "",
             "| task | solved/k |", "|---|---|"]
    lines += [f"| {iid} | {c}/{k} |" for iid, c, k in per_task]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"DONE {args.tag}: avg@{args.k}={avg_k:.1%} any-solved={any_solved}/{n} -> {args.out}")


if __name__ == "__main__":
    main()
