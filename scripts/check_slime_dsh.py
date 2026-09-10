"""CPU protocol check: real DSH SDK -> unchanged slime -> upstream fake model.

This is a bounded test fixture, not a rollout or training implementation.
No task, reward evaluator, Qwen model weights, SGLang engine or Ray is launched.
"""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

from aiohttp.test_utils import TestClient, TestServer
import torch
from slime.agent.adapters.openai import OpenAIAdapter
from slime.utils.types import Sample
from tests.test_agent._fakes import FakeSGLangServer, FakeTokenizer

ROOT = Path(__file__).resolve().parents[1]
PIN = "4c193f1f37509cca70f0e88807a9305b70f63f4e"


async def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert torch.version.cuda is None
    assert subprocess.check_output(["git", "-C", str(ROOT / "vendor/slime"), "rev-parse", "HEAD"], text=True).strip() == PIN
    assert not subprocess.check_output(["git", "-C", str(ROOT / "vendor/slime"), "status", "--porcelain"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    out = ROOT / "runtime/slime" / ("dsh-check-" + stamp)
    out.mkdir()
    workspace = out / "workspace"
    workspace.mkdir()
    sid = "slime-cpu-" + stamp
    job = {
        "dsh_bin": str(ROOT / ".venv-dsh/lib/python3.12/site-packages/deepseek_harness_runtime/runtime/deepseek-harness-sdk-runtime-linux-x64"),
        "workspace": str(workspace), "dsh_home": str(out / "dsh-home"),
        "patch": str(ROOT / "configs/agent-rl/dsh-qwen-rl.patch.yml"),
        "max_tokens": 4096, "timeout": 45,
        "prompt": "Reply with SLIME_DSH_OK. Do not use tools.",
        "session_id": sid, "result": str(out / "worker-result.json"),
    }
    (out / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    tok = FakeTokenizer(outputs={(301,): "SLIME_DSH_OK"})
    async with FakeSGLangServer([[(-0.25, 301)]]) as model:
        adapter = OpenAIAdapter(tokenizer=tok, sglang_url=model.url, max_turns_per_sid=1)
        adapter.open_session(sid)
        async with TestClient(TestServer(adapter.app)) as client:
            env = dict(os.environ, QWEN_BASE_URL=str(client.make_url("/v1")), QWEN_LOCAL_API_KEY=sid)
            with (out / "worker.log").open("wb") as log:
                proc = await asyncio.create_subprocess_exec(
                    str(ROOT / ".venv-dsh/bin/python"), str(ROOT / "scripts/agent_rl_dsh_worker.py"),
                    str(out / "job.json"), env=env, stdout=log, stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=75)
                finally:
                    if proc.returncode is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=10)
                        except asyncio.TimeoutError:
                            os.killpg(proc.pid, signal.SIGKILL)
                            await proc.wait()
            result = json.loads((out / "worker-result.json").read_text())
            assert proc.returncode == 0 and result["completed"] and result["runtime_closed"], result
            assert "SLIME_DSH_OK" in json.dumps(result)
            samples = await adapter.finish_session(sid, base_sample=Sample(index=0, prompt=""), reward=0.0)
        assert model.routing_keys == [sid]
        assert len(samples) == 1
        sample = samples[0]
        assert sample.tokens[-1] == 301 and sample.loss_mask[-1] == 1
        assert sample.rollout_log_probs[-1] == -0.25
        (out / "sample.json").write_text(json.dumps(sample.to_dict(), indent=2) + "\n")
        (out / "generate-requests.json").write_text(json.dumps(model.requests, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "slime_commit": PIN,
        "passed": True, "real_dsh_sdk": True, "upstream_openai_adapter_unchanged": True,
        "bearer_session_routing": True, "sdk_consumed_stream_response": True,
        "synthetic_sample_ids_logprobs_preserved": True, "dsh_runtime_closed": True,
        "fake_boundaries": ["tokenizer", "SGLang model output and logprobs"],
        "real_model_requests": 0, "parameter_updates": 0,
        "limits": "One completion only; no Qwen tokenizer, task sandbox, DSH tool turn, evaluation, Megatron, weight sync or checkpoint verification.",
        "directory": str(out.relative_to(ROOT)),
        "check_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "runtime/slime/dsh-latest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
