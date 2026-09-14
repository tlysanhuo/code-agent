#!/usr/bin/env python3
"""Difficulty screening: 27B teacher x k DSH rollouts per qualified task.

LEGO-RL protocol (arXiv 2608.17393): screen tasks with a stronger model, keep
the partial-solvability band (solved 1..k-1 of k). Unsolved-by-all and
solved-by-all tasks produce zero GRPO gradient and are dropped from round 1.

Inputs : configs/agent-rl/local-task-registry.json (from prepare_rl_tasks.py)
         data/agent-rl/rl-round1-prompts.parquet      (problem statements)
Needs  : vLLM serving Qwen3.5-27B at QWEN_BASE_URL (configs/screen-qwen-server.json,
         launched via scripts/start_qwen.sh) -- CPU envs come from the registry.
Outputs: runtime/agent-rl/round1-screen/results.jsonl        (one line per attempt)
         configs/agent-rl/rl-round1-screened.json            (band survivors + stats)
         data/agent-rl/rl-round1-screened-prompts.parquet    (training prompts)

Run with .venv-train-rl python from the project root. Resume-safe: attempts
already present in results.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_REGISTRY = ROOT / "configs/agent-rl/local-task-registry.json"
DEFAULT_PROMPTS = ROOT / "data/agent-rl/rl-round1-prompts.parquet"
DEFAULT_OUT_DIR = ROOT / "runtime/agent-rl/round1-screen"
DEFAULT_SCREENED_JSON = ROOT / "configs/agent-rl/rl-round1-screened.json"
DEFAULT_SCREENED_PARQUET = ROOT / "data/agent-rl/rl-round1-screened-prompts.parquet"

import types as _types
if "gym_facility" not in sys.modules:  # evaluate() path only; execute() unused here
    _stub = _types.ModuleType("gym_facility")
    _stub.execute = None
    sys.modules["gym_facility"] = _stub

import gym_prepare as gp
import gym_run as gr  # frozen evaluator (trusted patch export + oracle tests)
PROMPT_TMPL = (
    "Repair the issue in the repository at {workspace}. Inspect the relevant "
    "code, make a minimal source repair, and run the appropriate tests to "
    "verify your fix. Do not modify tests. You have one attempt.\n\n"
    "{problem_statement}"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_done(results_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["instance_id"], int(r["attempt"])))
            except Exception:
                continue
    return done


def record(result: dict, lock: threading.Lock, results_path: Path) -> None:
    with lock:
        with results_path.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_attempt(iid: str, row: dict, python: Path, attempt: int, args) -> dict:
    case = Path(args.out_dir) / iid / f"a{attempt}"
    case.mkdir(parents=True, exist_ok=True)
    gp.make_case(case, row, python)  # fresh buggy workspace
    ws = case / "sandbox/workspace"
    (case / "screen-prompt.txt").write_text(
        PROMPT_TMPL.format(workspace=ws, problem_statement=row["problem_statement"]))

    result = {"instance_id": iid, "attempt": attempt,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.monotonic()
    # DSH rollout in its own venv (.venv-dsh owns the SDK); wall-clock capped
    worker_env = {**os.environ, "QWEN_BASE_URL": args.base_url,
                  "SCREEN_ROLLOUT_TIMEOUT_S": str(args.rollout_timeout)}
    worker = subprocess.run(
        [str(ROOT / ".venv-dsh/bin/python"), str(ROOT / "scripts/screen_worker.py"), str(case)],
        env=worker_env, capture_output=True, text=True,
        timeout=args.rollout_timeout + 300, cwd=str(ROOT))
    dsh = {}
    dsh_result_file = case / "dsh-result.json"
    if dsh_result_file.exists():
        try:
            dsh = json.loads(dsh_result_file.read_text())
        except Exception:
            dsh = {}
    result["dsh_status"] = dsh.get("status", "missing")
    result["dsh_finish_reason"] = dsh.get("finish_reason")
    result["dsh_events"] = dsh.get("events")
    result["worker_rc"] = worker.returncode
    if result["dsh_status"] != "ok":
        result["worker_stderr_tail"] = (worker.stderr or "")[-400:]

    try:
        eval_res = gr.evaluate(case, row, python)
        result.update({k: eval_res[k] for k in
                       ("resolved", "category", "f2p_passed", "f2p_total",
                        "p2p_passed", "p2p_total", "tool_calls")})
    except Exception as exc:
        result["resolved"] = False
        result["category"] = f"screen_error:{type(exc).__name__}"
        result["error"] = str(exc)[:300]
        (case / "screen-error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
    result["seconds"] = round(time.monotonic() - t0, 1)
    return result


def serve_check(base_url: str) -> None:
    import urllib.request
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Authorization": "Bearer local-qwen"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
    log(f"server OK at {base_url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-check", action="store_true")
    ap.add_argument("--base-url", default="http://127.0.0.1:18095/v1")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="screen only first N tasks (0=all)")
    ap.add_argument("--rollout-timeout", type=float, default=1500,
                    help="per-attempt DSH wall-clock budget in seconds")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS,
                    help="source prompt parquet for screened-prompts export")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--screened-json", type=Path, default=DEFAULT_SCREENED_JSON)
    ap.add_argument("--screened-parquet", type=Path, default=DEFAULT_SCREENED_PARQUET)
    args = ap.parse_args()

    if args.serve_check:
        serve_check(args.base_url)
        return

    registry = {k: v for k, v in json.loads(
        args.registry.read_text()).items()
        if Path(v["case_dir"]).exists()}
    if args.limit:
        registry = dict(list(registry.items())[: args.limit])
    log(f"tasks in registry (with intact case): {len(registry)}")

    results_path = args.out_dir / "results.jsonl"
    if not args.aggregate_only:
        serve_check(args.base_url)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        done = load_done(results_path)
        jobs = []
        for iid, entry in registry.items():
            for a in range(1, args.k + 1):
                if (iid, a) not in done:
                    jobs.append((iid, entry))
        log(f"attempts to run: {len(jobs)} (skipping {len(done)} already done)")
        lock = threading.Lock()
        n_done = 0
        jobs = [(iid, entry, a) for iid, entry in registry.items()
                for a in range(1, args.k + 1) if (iid, a) not in done]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(run_attempt, iid, gp.task(iid),
                                Path(entry["python"]), a, args): (iid, a)
                    for iid, entry, a in jobs}
            for fut in as_completed(futs):
                iid, attempt = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # belt-and-braces; run_attempt catches its own
                    res = {"instance_id": iid, "attempt": attempt, "resolved": False,
                           "category": f"screen_error:{type(exc).__name__}",
                           "error": str(exc)[:300]}
                record(res, lock, results_path)
                n_done += 1
                log(f"[{n_done}/{len(futs)}] {iid} a{attempt}: "
                    f"resolved={res.get('resolved')} cat={res.get('category')} "
                    f"{res.get('seconds', '?')}s")

    # ---- aggregation: LEGO-RL band 1..k-1 ----
    rows = [json.loads(l) for l in results_path.read_text().splitlines()] if results_path.exists() else []
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["instance_id"], []).append(r)
    screened, stats = {}, {"solved0": [], "solved_all": [], "band": [], "incomplete": []}
    for iid, atts in by_task.items():
        if len(atts) < args.k or any(a.get("category", "").startswith("screen_error") for a in atts):
            stats["incomplete"].append(iid)
            continue
        s = sum(bool(a.get("resolved")) for a in atts)
        entry = {"solves": s, "k": args.k,
                 "categories": [a.get("category") for a in sorted(atts, key=lambda x: x["attempt"])],
                 "case_dir": registry.get(iid, {}).get("case_dir"),
                 "python": registry.get(iid, {}).get("python")}
        if s == 0:
            stats["solved0"].append(iid)
        elif s == args.k:
            stats["solved_all"].append(iid)
        else:
            stats["band"].append(iid)
            screened[iid] = entry
    summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "k": args.k, "attempts_logged": len(rows),
               "band_size": len(stats["band"]), "zero_solve": len(stats["solved0"]),
               "all_solve": len(stats["solved_all"]), "incomplete": len(stats["incomplete"]),
               "task_lists": stats}
    args.screened_json.write_text(json.dumps(summary, indent=1))
    # screened training prompts
    import pyarrow as pa, pyarrow.parquet as pq
    src = pq.read_table(args.prompts)
    src_ids = src["metadata"].to_pylist() if src.num_rows else []
    keep = []
    for row_md in src_ids:
        if row_md.get("instance_id") in screened:
            keep.append(row_md)
    if keep:
        cols = {name: [r.get(name) for r in keep] for name in src.column_names}
        pq.write_table(pa.Table.from_pydict(cols, schema=src.schema),
                       args.screened_parquet)
    log(json.dumps({k2: v2 for k2, v2 in summary.items() if k2 != "task_lists"}, indent=2))


if __name__ == "__main__":
    main()
