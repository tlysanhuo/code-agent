#!/usr/bin/env python3
"""DSH rollout worker for difficulty screening (run with .venv-dsh).

Does exactly one thing: drive the DeepSeek Harness agent on a prepared buggy
workspace and write the outcome to <case>/dsh-result.json. No pyarrow, no
evaluator imports -- the controller (screen_rl_tasks.py, .venv-train-rl)
owns orchestration and grading.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    case = Path(sys.argv[1]).resolve()
    prompt = (case / "screen-prompt.txt").read_text()
    base_url = os.environ["QWEN_BASE_URL"]
    budget_s = float(os.environ.get("SCREEN_ROLLOUT_TIMEOUT_S", "1500"))

    from deepseek_harness import DeepSeekHarness

    out = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    harness = None
    t0 = time.monotonic()
    try:
        harness = DeepSeekHarness(
            provider="qwen-local", model="Qwen3.5-27B",
            cwd=str(case / "sandbox/workspace"),
            dsh_home=str(ROOT / "runtime/dsh-home"), profile="sdk-minimal",
            patches=(str(ROOT / "configs/gym-dsh.patch.yml"),),
            env={"QWEN_LOCAL_API_KEY": "local-qwen",
                 "QWEN_BASE_URL": base_url,
                 "CUDA_VISIBLE_DEVICES": ""},
            initialize_timeout_seconds=180, request_timeout_seconds=600,
            shutdown_timeout_seconds=15,
        )
        with harness:
            run = harness.run(prompt, session_id=f"screen-{case.parent.name}-{case.name}")
            out.update(finish_reason=run.finish_reason, events=len(run.events))
            (case / "dsh-final-response.txt").write_text(run.final_response or "")
        out["status"] = "ok"
    except Exception as exc:
        out.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
    finally:
        if harness is not None:
            try:
                harness.close()
            except Exception:
                pass
    out["seconds"] = round(time.monotonic() - t0, 1)
    (case / "dsh-result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    # nonzero exit on timeout lets the controller classify wall-clock overruns
    sys.exit(0 if out.get("status") == "ok" else 3)


if __name__ == "__main__":
    main()
