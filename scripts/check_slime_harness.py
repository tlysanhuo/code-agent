"""Contract tests for the DSH extension; reuse upstream orchestration and fakes.

The successful case runs the real DSH SDK, a harmless fixed bash tool, and the
original upstream task/evaluation control flow. Sandbox commands and evaluator
outcomes are fake. No benchmark task or real model is run.
"""
import asyncio
import dataclasses
import json
import os
from pathlib import Path
import signal

import pytest

# Reuse the upstream CPU test's explicit transformers stub and four-edge fixture.
from tests.test_agent import test_agent_rollout_cpu as upstream_fixture
from tests.test_agent._fakes import FakeSandbox, FakeTokenizer
from slime.agent.adapters import common as adapters_common
from slime.agent.harness import common as harness_common
from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample
from slime_dsh.harness import DshHarness, ROOT, PYTHON, WORKER, run_directory
import slime_dsh.generate as extension

gen = upstream_fixture.gen
swe = upstream_fixture.swe
_REAL_SLEEP = asyncio.sleep


@pytest.mark.parametrize("path", ["/tmp/outside", str(ROOT) + "/../escape", str(ROOT) + "/bad;command"])
def test_reject_workspace_before_sandbox_mutation(path):
    async def check():
        sb = FakeSandbox()
        with pytest.raises(ValueError):
            await DshHarness().run(sb, workdir=path, session_id="s", adapter_url="http://127.0.0.1:1", time_budget_sec=1, prompt="x")
        assert not sb.exec_log and not sb.files
    asyncio.run(check())


def test_binding_reuses_upstream_function_and_rejects_live_service():
    assert extension.generate is gen.generate
    assert gen.HARNESS_CLS is DshHarness
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(SingletonMeta._instances, gen._AdapterService, object())
        with pytest.raises(RuntimeError, match="before"):
            extension.bind_upstream()


@pytest.mark.parametrize("agent_exit,eval_exit", [(0, 0), (0, 1), (7, 1)])
def test_upstream_dsh_lifecycle(tmp_path, monkeypatch, agent_exit, eval_exit):
    async def check():
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sandboxes = []
        sdk_results = []
        tok = FakeTokenizer()
        # Literal safe tool command; never derive commands from downloaded data.
        replies = [
            ('<tool_call><function=bash><parameter=command>printf SLIME_TOOL_OK</parameter></function></tool_call>', "stop", None),
            ("SLIME_DSH_TOOL_DONE", "stop", None),
        ]

        async def agent(env):
            sb = sandboxes[0]
            job_name = str(run_directory(str(workspace)) / "job.json")
            job = json.loads(sb.files[job_name])
            assert job["workspace"] == str(workspace)
            assert env["QWEN_LOCAL_API_KEY"] == job["session_id"]
            assert job["prompt"] == swe.SWE_PROMPT
            if agent_exit:
                return agent_exit  # explicit failed agent boundary; no fake success
            directory = Path(job_name).parent
            directory.mkdir(parents=True)
            Path(env["TMPDIR"]).mkdir()
            Path(job_name).write_text(json.dumps(job))
            with (directory / "sdk.log").open("wb") as log:
                proc = await asyncio.create_subprocess_exec(
                    str(PYTHON), str(WORKER), job_name, env=dict(os.environ, **env),
                    stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=70)
                finally:
                    if proc.returncode is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            await asyncio.wait_for(proc.wait(), 10)
                        except asyncio.TimeoutError:
                            os.killpg(proc.pid, signal.SIGKILL)
                            await proc.wait()
            result = json.loads(Path(job["result"]).read_text())
            sdk_results.append(result)
            assert result["completed"] and result["runtime_closed"], result
            assert "SLIME_DSH_TOOL_DONE" in json.dumps(result)
            return proc.returncode

        class RecordingSandbox(FakeSandbox):
            async def __aenter__(self):
                self.closed = False
                return self
            async def __aexit__(self, *exc):
                self.closed = True

        def sandbox_factory(image):
            # Upstream evaluator must get a distinct sandbox, not the agent one.
            sb = RecordingSandbox(image, on_launch=agent,
                responses=[("SLIME_EVAL_FIXTURE", (eval_exit, "", ""))])
            sandboxes.append(sb)
            return sb

        monkeypatch.setattr(gen, "CONFIG", dataclasses.replace(gen.CONFIG,
            adapter_public_host="127.0.0.1", adapter_bind_host="127.0.0.1", adapter_port=0,
            agent_time_budget_sec=45, eval_timeout_sec=10, rollout_guard_sec=100, boot_retries=1))
        monkeypatch.setattr(gen, "load_tokenizer", lambda *a, **k: tok)
        monkeypatch.setattr(gen, "E2BSandbox", sandbox_factory)
        monkeypatch.setattr(swe, "E2BSandbox", sandbox_factory)
        # Model output is scripted at the same boundary as the upstream CPU suite.
        monkeypatch.setattr(adapters_common, "call_sglang_generate",
            upstream_fixture.fake_call_sglang_generate(replies, tok))
        async def fast_sleep(_):
            await _REAL_SLEEP(0)
        monkeypatch.setattr(harness_common.asyncio, "sleep", fast_sleep)
        SingletonMeta.clear_instances(gen._AdapterService)
        args = upstream_fixture._args()
        sample = Sample(index=0, group_index=0, prompt="CPU fixture", metadata={
            "instance_id": "slime-dsh-cpu-fixture", "image": "fake-image",
            "workdir": str(workspace), "problem_statement": "CPU tool protocol fixture",
            "eval_cmd": "SLIME_EVAL_FIXTURE",
        })
        try:
            samples = await extension.generate(args, sample, {"max_new_tokens": 128})
            assert len(sandboxes) == 2 and all(sb.closed for sb in sandboxes)
            assert any("SLIME_EVAL_FIXTURE" in c for c, _ in sandboxes[1].exec_log)
            assert any("test -x" in c for c, _ in sandboxes[0].exec_log)
            if agent_exit:
                assert samples[0].status == Sample.Status.ABORTED
                assert samples[0].metadata["abort_reason"] == "adapter_session_empty"
                assert not sdk_results
            else:
                assert sdk_results and all(s.status == Sample.Status.COMPLETED for s in samples)
                assert sum(s.reward for s in samples) == float(eval_exit == 0)
                assert all(len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length for s in samples)
                assert any("SLIME_TOOL_OK" in json.dumps(messages) for messages, _ in tok.rendered)
                (tmp_path / "samples.json").write_text(json.dumps([s.to_dict() for s in samples], indent=2))
            (tmp_path / "sandbox-commands.json").write_text(json.dumps([s.exec_log for s in sandboxes], indent=2))
        finally:
            state = SingletonMeta._instances.get(gen._AdapterService)
            if state:
                state.app_handle.stop()
            SingletonMeta.clear_instances(gen._AdapterService)
    asyncio.run(check())
