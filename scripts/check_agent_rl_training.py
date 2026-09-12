"""Real DSH + restricted local task + native HTTP, with a scripted CPU model.

This exercises training engineering, never model ability or benchmark scoring.
All model outputs, sampled arrays and logprobs are synthetic fixtures.
"""
import asyncio
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
from types import SimpleNamespace

from agent_rl_training_task import (ROOT, LocalTrainingTask, tree_hash, write_json,
                                    CandidateRejected, TrainingEnvironmentError, run_isolated)


def prepare_fixture(root):
    private = root / 'private'
    source = private / 'source'
    source.mkdir(parents=True)
    (source / 'calc.py').write_text('def answer():\n    return 1\n')
    (private / 'gold.patch').write_text('diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def answer():\n-    return 1\n+    return 2\n')
    (private / 'tests.patch').write_text('diff --git a/tests/test_calc.py b/tests/test_calc.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/test_calc.py\n@@ -0,0 +1,3 @@\n+from calc import answer\n+def test_answer():\n+    assert answer() == 2\n')
    spec = {'task_id': 'cpu-engineering-fixture', 'protocol': 'custom-training-pytest-v1',
            'source_dir': str(source), 'source_sha256': tree_hash(source),
            'python': str(ROOT / '.venv-swe/bin/python'), 'pytest_args': ['tests/test_calc.py', '-q'],
            'gold_patch': str(private / 'gold.patch'), 'test_patch': str(private / 'tests.patch'),
            'protected_paths': ['tests/*', 'conftest.py', 'pytest.ini', 'pyproject.toml'],
            'test_timeout_seconds': 30}
    spec_path = private / 'task.json'
    write_json(spec_path, spec)
    task = LocalTrainingTask.load(spec_path)
    qualification = task.qualify(root / 'qualification')
    assert qualification['qualified'], qualification
    spec['qualification'] = str(root / 'qualification/qualification.json')
    write_json(spec_path, spec)
    registry = root / 'registry.json'
    write_json(registry, {'protocol': spec['protocol'], 'tasks': {spec['task_id']: {'task_spec': str(spec_path)}}})
    return registry


async def run_fixture(root, registry):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, _agent_loop_registry
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from agent_rl_dsh_loop import DshAgentLoop
    requests = []
    native_full = [101, 102]
    commands = [
        ('bash', {'command': 'pwd; cat calc.py; cat ' + str(root / 'private/gold.patch')}),
        ('str_replace_editor', {'command': 'view', 'path': 'calc.py'}),
        ('str_replace_editor', {'command': 'str_replace', 'path': 'calc.py', 'old_str': 'return 1', 'new_str': 'return 2'}),
        ('bash', {'command': 'python -c "import calc; print(calc.answer())"'}),
    ]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
        def do_POST(self):
            nonlocal native_full
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                ids = native_full + ([200 + len(requests)] if requests else [])
                data = json.dumps({'tokens': ids, 'count': len(ids), 'max_model_len': 8192}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            assert self.path == '/v1/chat/completions'
            n = len(requests)
            requests.append(body)
            assert body['return_token_ids'] and body['logprobs']
            assert body['chat_template_kwargs']['enable_thinking'] is True
            prompt_ids = native_full + ([200 + n] if n else [])
            sampled = [300 + n]
            native_full = prompt_ids + sampled
            if n < len(commands):
                name, arguments = commands[n]
                # DSH editor wants absolute paths; use its own cwd from prior observation.
                if name == 'str_replace_editor':
                    tool_text = next(m['content'] for m in body['messages'] if m['role'] == 'tool')
                    work = next(line.strip() for line in tool_text.splitlines() if '/sandbox/workspace' in line)
                    arguments = dict(arguments, path=work + '/calc.py')
                delta = {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': f'fixture-{n}',
                    'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
                finish = 'tool_calls'
            else:
                delta = {'role': 'assistant', 'content': 'CPU_DSH_TRAINING_OK'}
                finish = 'stop'
            def chunk(d, ids, done=None):
                return {'id': f'fixture-{n}', 'object': 'chat.completion.chunk', 'model': 'Qwen3.5-27B', 'created': 0,
                    'choices': [{'index': 0, 'delta': d, 'token_ids': ids, 'finish_reason': done,
                    'logprobs': {'content': [{'token': 'synthetic', 'logprob': -.1, 'bytes': None, 'top_logprobs': []} for _ in ids]}}]}
            first = chunk(delta, sampled)
            first['prompt_token_ids'] = prompt_ids
            last = chunk({}, [], finish)
            last['usage'] = {'prompt_tokens': len(prompt_ids), 'completion_tokens': 1, 'total_tokens': len(prompt_ids) + 1}
            data = ''.join('data: ' + json.dumps(c) + '\n\n' for c in [first, last]) + 'data: [DONE]\n\n'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(data.encode())))
            self.end_headers()
            self.wfile.write(data.encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    lifecycle = {'acquired': 0, 'released': 0, 'aborted': 0}
    async def abort():
        lifecycle['aborted'] += 1
    class FakeClient:
        async def _acquire_server(self, sid):
            lifecycle['acquired'] += 1
            return f'127.0.0.1:{server.server_port}', SimpleNamespace(abort_all_requests=SimpleNamespace(remote=abort))
        def _release_server(self, *_args, **_kwargs):
            lifecycle['released'] += 1
    try:
        with initialize_config_dir(config_dir=str(ROOT / 'configs/agent-rl'), version_base=None):
            cfg = compose(config_name='qwen35_27b_lora_2h100', overrides=['+native_http=vllm', '+training_runtime=local'])
        cfg.dsh_training.task_registry = str(registry)
        cfg.dsh_training.run_root = str(root / 'runs')
        cfg.dsh_training.trajectory_seconds = 120  # explicit shorter CPU fixture, not the training config
        tok = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
        config = OmegaConf.load(ROOT / 'configs/agent-rl/agent_loops.yaml')[0]
        loop = instantiate(config, trainer_config=DictConfigWrap(config=cfg), server_manager=FakeClient(),
            tokenizer=tok, processor=None, dataset_cls=RLHFDataset, data_config=DictConfigWrap(config=cfg.data),
            hf_model_type='qwen3_5')
        assert isinstance(loop, DshAgentLoop) and 'dsh_agent' in _agent_loop_registry
        output = await loop.run({'temperature': 1.0, 'top_p': 1.0, 'top_k': -1},
            extra_info={'instance_id': 'cpu-engineering-fixture'}, global_steps=0,
            raw_prompt=[{'role': 'user', 'content': 'Inspect calc.py. Fix answer() so it returns 2. Use the configured tools.'}])
        assert len(requests) == 5
        assert output.reward_score == 1 and sum(output.response_mask) == 5
        observations = [m['content'] for m in requests[-1]['messages'] if m['role'] == 'tool']
        assert any('Permission denied' in text or 'Operation not permitted' in text for text in observations)
        assert '2' in observations[-1]
        assert lifecycle == {'acquired': 1, 'released': 1, 'aborted': 0}
        run = next((root / 'runs').glob('20*'))
        cleanup = json.loads((run / 'cleanup.json').read_text())
        assert cleanup['worker_stopped'] and cleanup['socket_removed'] and cleanup['lease_released']
        write_json(root / 'output.json', output.model_dump())
        return {'registered_loop_instantiated': True, 'real_dsh_session': True, 'scripted_model_requests': len(requests),
                'private_gold_read_denied': True, 'shell_editor_same_workspace': True,
                'independent_training_reward': 1, 'official_benchmark_result': False, 'lifecycle': lifecycle}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        write_json(root / 'scripted-requests.json', requests)


async def check_rejections(root, registry):
    import httpx
    from agent_rl_training_runtime import NativeTrainingSession
    entry = json.loads(registry.read_text())['tasks']['cpu-engineering-fixture']
    task = LocalTrainingTask.load(entry['task_spec'])
    task.require_qualified()
    checks = []
    original = (root / 'private/gold.patch').read_bytes()
    try:
        (root / 'private/gold.patch').write_bytes(original + b'\n')
        try:
            task.require_qualified()
            raise AssertionError('Stale qualification accepted')
        except TrainingEnvironmentError:
            checks.append('changed_gold_invalidates_qualification')
    finally:
        (root / 'private/gold.patch').write_bytes(original)
    case = task.prepare(root / 'rejection-candidate')
    (case / 'sandbox/workspace/conftest.py').write_text('raise RuntimeError("candidate hook")\n')
    try:
        task.export_patch(case, root / 'rejected-patch/candidate.patch')
        raise AssertionError('Protected test edit accepted')
    except CandidateRejected:
        checks.append('protected_test_edit_rejected')
    timeout_case = task.prepare(root / 'timeout-candidate')
    try:
        await asyncio.to_thread(run_isolated, timeout_case,
            [task.spec['python'], '-c', 'import time; time.sleep(60)'], timeout=.5)
        raise AssertionError('Verifier timeout ignored')
    except TrainingEnvironmentError:
        checks.append('verifier_timeout_reaped_without_reward')
    session = NativeTrainingSession(address='127.0.0.1:1', directory=root / 'admission',
        session_id='cpu-admission', policy_version=0, sampling_params={}, tokenizer=None,
        limits={'trajectory_seconds': 10, 'requests': 0})
    # Reject before forwarding: no server, tokenizer or model connection needed.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=session), base_url='http://fixture') as client:
        response = await client.post('/v1/chat/completions', json={})
        assert response.status_code == 401 and not session.errors
        checks.append('unauthorized_request_does_not_poison_session')
        response = await client.post('/v1/chat/completions', json={},
            headers={'Authorization': 'Bearer ' + session.api_key})
        assert response.status_code == 429 and session.calls == 0
        checks.append('request_budget_rejected_before_model')
    await session.close()
    return checks


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('CODE_AGENT_ROOT') != str(ROOT):
        raise SystemExit('Use scripts/agent_rl.sh --training-check')
    root = ROOT / 'runtime/agent-rl/training-interface-checks' / secrets.token_hex(6)
    root.mkdir(parents=True)
    report = {'scope': 'CPU engineering fixture; real DSH/tools, scripted model, custom training verifier',
              'real_model_rollouts': 0, 'parameter_updates': 0, 'official_benchmark_result': False, 'passed': False}
    try:
        registry = prepare_fixture(root)
        report['checks'] = asyncio.run(run_fixture(root, registry))
        report['rejection_checks'] = asyncio.run(check_rejections(root, registry))
        import torch, ray
        assert not torch.cuda.is_initialized() and not ray.is_initialized()
        report['passed'] = True
    except BaseException as exc:
        import traceback
        report['error'] = traceback.format_exc()
        print(report['error'])
    finally:
        report['checked_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(root / 'acceptance.json', report)
        write_json(ROOT / 'runtime/agent-rl/training-interface-latest.json', {'directory': str(root), **report})
        print(json.dumps({'directory': str(root), 'passed': report['passed']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
