"""CPU-only native HTTP/trace contract checks with in-process ASGI fixtures.

No listening ports, DSH process, model engine, task environment or trainer.
The upstream HTTP proxy and training transforms execute unchanged. Backend
responses and verifier results below are explicitly synthetic test fixtures.
"""
import asyncio
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import unittest

from agent_rl_native import native_gateway_config, traces_to_agent_output, upstream

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'runtime/agent-rl/native-interface-check'
MODEL = 'Qwen3.5-27B'
TOOLS = [{'type': 'function', 'function': {
    'name': 'bash', 'description': 'Synthetic fixture; never executed',
    'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}},
}}]
MESSAGES = [{'role': 'user', 'content': 'CPU protocol fixture only'}]
EVIDENCE = {}


def fixture_chunks(turn):
    prompt_ids = [1, 2] if turn == 1 else [1, 2, 10, 11, 12, 20, 21]
    actions = [
        ({'reasoning': 'Inspect first.'}, [10], [-.1]),
        ({'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function',
                         'function': {'name': 'bash', 'arguments': '{"command":'}}]}, [11], [-.2]),
        ({'tool_calls': [{'index': 0, 'function': {'arguments': '"cat example.py"}'}}]}, [12], [-.3]),
    ] if turn == 1 else [({'content': 'Fixture finished.'}, [30, 31], [-.4, -.5])]
    def chunk(delta, ids, lps, finish=None):
        return {'id': f'chat-fixture-{turn}', 'object': 'chat.completion.chunk',
                'created': 0, 'model': MODEL, 'choices': [{
                    'index': 0, 'delta': delta, 'finish_reason': finish, 'token_ids': ids,
                    'logprobs': {'content': [{'token': str(t), 'logprob': lp,
                                             'bytes': None, 'top_logprobs': []}
                                            for t, lp in zip(ids, lps, strict=True)]}}]}
    chunks = [chunk({'role': 'assistant'}, [], [])]
    chunks[0]['prompt_token_ids'] = prompt_ids
    chunks += [chunk(*action) for action in actions]
    chunks += [chunk({}, [], [], 'tool_calls' if turn == 1 else 'stop')]
    chunks += [{'id': f'chat-fixture-{turn}', 'object': 'chat.completion.chunk',
                'created': 0, 'model': MODEL, 'choices': [], 'usage': {
                    'prompt_tokens': len(prompt_ids), 'completion_tokens': sum(len(a[1]) for a in actions),
                    'total_tokens': len(prompt_ids) + sum(len(a[1]) for a in actions)}}]
    return chunks


def output_args():
    return dict(session_id='fixture-session', model=MODEL, policy_version=7,
                reward_score=1, verifier_ref='synthetic-fixture://not-an-official-result',
                pad_token_id=0, max_prompt_tokens=16, max_response_tokens=16, max_requests=12)


class NativeInterfaceChecks(unittest.IsolatedAsyncioTestCase):
    async def test_native_http_to_upstream_training(self):
        import httpx
        import torch
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionStreamResponse
        api = upstream()
        from rllm_model_gateway.store.memory_store import MemoryTraceStore
        received = []
        config = native_gateway_config(['127.0.0.1:18080'], model=MODEL)
        app = api.gateway.create_app(config, store=MemoryTraceStore())
        def backend(request):
            self.assertEqual(str(request.url), 'http://127.0.0.1:18080/v1/chat/completions')
            body = json.loads(request.content)
            received.append(body)
            self.assertTrue(body['return_token_ids'])
            self.assertTrue(body['logprobs'])
            self.assertEqual(body['tools'], TOOLS)
            self.assertEqual(body['temperature'], 1.0)
            chunks = fixture_chunks(len(received))
            for chunk in chunks:
                ChatCompletionStreamResponse.model_validate(chunk)
            payload = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
            return httpx.Response(200, headers={'Content-Type': 'text/event-stream'}, content=payload)
        # Inject only a test transport. The real upstream proxy still forwards,
        # parses its normal native responses, and stores its normal TraceRecords.
        app.state.proxy._http = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://gateway.test') as client:
                r = await client.post('/sessions', json={'session_id': 'fixture-session',
                    'sampling_params': {'temperature': 1.0, 'top_p': 1.0, 'top_k': -1,
                                        'return_token_ids': True, 'logprobs': True}})
                self.assertEqual(r.status_code, 200)
                session_url = r.json()['url']
                await client.post('/admin/weight_version', json={'weight_version': 7})
                messages = copy.deepcopy(MESSAGES)
                for turn in (1, 2):
                    r = await client.post(session_url + '/chat/completions', json={
                        'model': MODEL, 'messages': messages, 'tools': TOOLS,
                        'stream': True, 'stream_options': {'include_usage': True}, 'max_tokens': 16})
                    self.assertEqual(r.status_code, 200)
                    chunks = [json.loads(line[6:]) for line in r.text.splitlines()
                              if line.startswith('data: ') and line != 'data: [DONE]']
                    self.assertEqual([c.get('choices', [{}])[0].get('delta') for c in chunks if c.get('choices')],
                                     [c['choices'][0]['delta'] for c in fixture_chunks(turn) if c.get('choices')])
                    (ARTIFACTS / f'native-response-{turn}.sse').write_text(r.text)
                    if turn == 1:
                        messages += [
                            {'role': 'assistant', 'content': '', 'reasoning': 'Inspect first.',
                             'tool_calls': [{'id': 'call-1', 'type': 'function',
                                             'function': {'name': 'bash', 'arguments': '{"command":"cat example.py"}'}}]},
                            {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'Synthetic observation'},
                        ]
                await client.post('/admin/flush')
                traces = (await client.get('/sessions/fixture-session/traces')).json()
                self.assertEqual(len(traces), 2)
                self.assertEqual(received[0]['messages'], MESSAGES)
                self.assertEqual(traces[0]['completion_token_ids'], [10, 11, 12])
                self.assertEqual(traces[1]['prompt_token_ids'], [1, 2, 10, 11, 12, 20, 21])
                (ARTIFACTS / 'traces.json').write_text(json.dumps(traces, indent=2) + '\n')
                out = traces_to_agent_output(traces, **output_args())
                self.assertEqual(out.prompt_ids, [1, 2])
                self.assertEqual(out.response_ids, [10, 11, 12, 20, 21, 30, 31])
                self.assertEqual(out.response_mask, [1, 1, 1, 0, 0, 1, 1])
                self.assertEqual(out.reward_score, 1)
                self.assertEqual(out.as_dict()['rollout_log_probs'].numel(), 7)
                for got, expected in zip(out.response_logprobs, [-.1, -.2, -.3, 0, 0, -.4, -.5], strict=True):
                    self.assertAlmostEqual(got, expected, places=6)
                # Exercise the trainer's actual loss aggregation, including
                # the synthetic thinking token at position 0.
                from verl.trainer.ppo.core_algos import agg_loss
                values = torch.ones(1, 7, requires_grad=True)
                loss = agg_loss(values, torch.tensor([out.response_mask]), 'token-mean')
                loss.backward()
                self.assertEqual(values.grad[0, 3:5].tolist(), [0, 0])
                self.assertGreater(values.grad[0, 0].item(), 0)
                (ARTIFACTS / 'agent-loop-output.json').write_text(out.model_dump_json(indent=2) + '\n')
                EVIDENCE['native_route'] = {'requests': 2, 'transport': 'ASGI + mock HTTP; no sockets',
                    'tool_and_reasoning_deltas_preserved': True, 'native_usage_and_arrays_aligned': True,
                    'upstream_trace_to_dataproto_to_agent_loop_output': True, 'tool_token_gradient_zero': True}
                self.check_rejections(traces)
        finally:
            await app.state.proxy.stop()

    def check_rejections(self, original):
        mutations = {
            'missing_logprobs': lambda t: t[0].update(logprobs=None),
            'short_logprobs': lambda t: t[0].update(logprobs=[-.1]),
            'mixed_policy': lambda t: t[1].update(weight_version=8),
            'missing_policy': lambda t: t[1].update(weight_version=None),
            'mixed_session': lambda t: t[1].update(session_id='different'),
            'token_prefix_drift': lambda t: t[1]['prompt_token_ids'].__setitem__(0, 99),
            'rewritten_history': lambda t: t[1]['messages'][0].update(content='rewritten'),
            'truncation': lambda t: t[1].update(finish_reason='length'),
            'infra_error': lambda t: t[1].update(finish_reason='error'),
            'usage_mismatch': lambda t: t[0]['token_counts'].update(completion=99),
            'duplicate_trace': lambda t: t[1].update(trace_id=t[0]['trace_id']),
            'changed_tools': lambda t: t[1]['raw_request'].update(tools=[]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                traces = copy.deepcopy(original)
                mutate(traces)
                with self.assertRaises(ValueError):
                    traces_to_agent_output(traces, **output_args())
        for change in ({'max_requests': 1}, {'max_prompt_tokens': 1}, {'max_response_tokens': 6}):
            with self.assertRaises(ValueError):
                traces_to_agent_output(original, **(output_args() | change))
        with self.assertRaises(ValueError):
            traces_to_agent_output([], **output_args())
        EVIDENCE['rejected_cases'] = list(mutations) + ['request_limit', 'prompt_limit', 'response_limit', 'empty']

    def test_upstream_native_config(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from verl.utils.config import validate_config
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config
        from vllm.entrypoints.cli.serve import ServeSubcommand
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        with initialize_config_dir(config_dir=str(ROOT / 'configs/agent-rl'), version_base=None):
            native = compose(config_name='qwen35_27b_lora_2h100', overrides=['+native_http=vllm'])
            base = compose(config_name='qwen35_27b_lora_2h100')
        OmegaConf.resolve(native)
        validate_config(native, need_reference_policy(native), need_critic(native))
        engine = OmegaConf.to_container(native.actor_rollout_ref.rollout.engine_kwargs.vllm, resolve=True)
        parser = FlexibleArgumentParser()
        ServeSubcommand().subparser_init(parser.add_subparsers())
        args = parser.parse_args(['serve', native.actor_rollout_ref.model.path] + build_cli_args_from_config(engine))
        self.assertTrue(args.enable_auto_tool_choice)
        self.assertEqual(args.tool_call_parser, 'qwen3_coder')
        self.assertEqual(args.reasoning_parser, 'qwen3')
        self.assertIn(MODEL, args.served_model_name)
        # The overlay changes only these four HTTP capability fields.
        merged = OmegaConf.to_container(native, resolve=True)
        for key in engine:
            if key not in OmegaConf.to_container(base.actor_rollout_ref.rollout.engine_kwargs.vllm):
                del merged['actor_rollout_ref']['rollout']['engine_kwargs']['vllm'][key]
        self.assertEqual(merged, OmegaConf.to_container(base, resolve=True))
        (ARTIFACTS / 'resolved-native-config.yaml').write_text(OmegaConf.to_yaml(native))
        EVIDENCE['configuration'] = {'upstream_validate_config': True, 'vllm_cli_parse': True,
                                     'training_parameters_unchanged': True, 'native_flags': engine}


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('CODE_AGENT_ROOT') != str(ROOT):
        raise SystemExit('Use scripts/agent_rl.sh --native-check (CPU only)')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeInterfaceChecks))
    import torch, ray
    passed = result.wasSuccessful() and not torch.cuda.is_initialized() and not ray.is_initialized()
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'passed': passed,
              'tests': result.testsRun, 'evidence': EVIDENCE,
              'cuda_initialized': torch.cuda.is_initialized(), 'ray_initialized': ray.is_initialized(),
              'synthetic_only': True, 'dsh_sessions': 0, 'task_containers': 0, 'model_requests': 0,
              'parameter_updates': 0, 'training_launch_ready': False,
              'failures': [detail for _, detail in result.failures + result.errors]}
    (ARTIFACTS / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if passed else 1)
