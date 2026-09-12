"""Native HTTP/trace interfaces; no model server, agent runner or evaluator.

Uses the frozen rLLM gateway and its existing trace-to-verl transform. Local
code validates our stricter data contract before that transform can pad missing
logprobs, split a changed history, truncate a sequence or drop an empty trace.
The caller owns DSH, the official task verifier, request budgets and lifecycle.
"""
from functools import lru_cache
import importlib
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'rllm': (ROOT / 'vendor/rllm-agent-rl', '3b40c37cf6a262cf4d28cc987ebe4f4cf797956c'),
    'verl': (ROOT / 'vendor/verl-agent-rl', '1252cc71aa5bd82e5604322064d69bfe6454c660'),
}


@lru_cache(maxsize=1)
def upstream():
    """Load selected unchanged source modules, not another training stack."""
    for path, revision in SOURCES.values():
        head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
        if head != revision or dirty:
            raise RuntimeError(f'Pinned upstream source changed: {path}')
    rllm_path = SOURCES['rllm'][0]
    for path in (rllm_path / 'rllm-model-gateway/src', rllm_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    modules = {
        'gateway': 'rllm_model_gateway',
        'trace_converter': 'rllm.engine.trace_converter',
        'types': 'rllm.types',
        'transform': 'rllm.trainer.verl.transform',
        'agent_loop': 'verl.experimental.agent_loop.agent_loop',
    }
    loaded = {key: importlib.import_module(name) for key, name in modules.items()}
    for key, module in loaded.items():
        base = SOURCES['verl' if key == 'agent_loop' else 'rllm'][0]
        if not Path(module.__file__).resolve().is_relative_to(base):
            raise RuntimeError(f'Unexpected imported module: {module.__file__}')
    return SimpleNamespace(**loaded)


def native_gateway_config(server_addresses, *, model):
    """Configure the upstream recorder for one verl TP replica's HTTP address.

    Pass LLMServerManager.get_addresses(). This constructs configuration only;
    it neither connects nor starts a service. The future owner must supply a
    durable store and enforce budgets before exposing the app to DSH.
    """
    if len(server_addresses) != 1 or not model:
        raise ValueError('This integration supports exactly one named HTTP replica')
    address = server_addresses[0]
    if '://' not in address:
        address = 'http://' + address
    from urllib.parse import urlsplit
    url = urlsplit(address)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password:
        raise ValueError('Expected a native HTTP server address without credentials')
    if url.path not in ('', '/', '/v1', '/v1/') or url.query or url.fragment:
        raise ValueError('Expected the server root or /v1 endpoint')
    return upstream().gateway.GatewayConfig(
        host='127.0.0.1', model=model,
        workers=[{'url': address, 'model_name': model}],
        cumulative_token_mode=False,
        add_return_token_ids=True, add_logprobs=True,
        strip_vllm_fields=False, sync_traces=True,
    )


def traces_to_agent_output(
    traces, *, session_id, model, policy_version, reward_score, verifier_ref,
    pad_token_id, max_prompt_tokens, max_response_tokens, max_requests,
):
    """Convert native sampled arrays with unchanged upstream training utilities.

    reward_score/verifier_ref must come from the independent task verifier;
    this function does not execute or grade a task. It only accepts completed,
    prefix-consistent text trajectories. Truncation/infra failures are rejected
    for the caller to preserve, never silently converted into a zero reward.
    """
    for limit in (max_prompt_tokens, max_response_tokens, max_requests):
        if type(limit) is not int or limit <= 0:
            raise ValueError('Positive explicit limits are required')
    if not traces or len(traces) > max_requests:
        raise ValueError('Empty trajectory or request budget exceeded')
    if type(policy_version) is not int or policy_version < 0:
        raise ValueError('A trusted sampling-phase policy version is required')
    if not session_id or not model or not verifier_ref or reward_score not in (0, 1):
        raise ValueError('Explicit identity and binary task verifier result required')
    api = upstream()
    records = [api.gateway.TraceRecord.model_validate(t) for t in traces]
    previous_full = None
    previous_messages = None
    tools = (records[0].raw_request or {}).get('tools', [])
    trace_ids = set()
    first_prompt = records[0].prompt_token_ids
    if not first_prompt or len(first_prompt) > max_prompt_tokens:
        raise ValueError('Missing or over-budget complete initial prompt')
    for index, trace in enumerate(records):
        if trace.session_id != session_id or trace.model != model or trace.weight_version != policy_version:
            raise ValueError('Mixed session, model or policy version')
        if trace.trace_id in trace_ids:
            raise ValueError('Duplicate trace')
        trace_ids.add(trace.trace_id)
        if not trace.completion_token_ids or trace.logprobs is None:
            raise ValueError('Missing sampled token IDs or logprobs')
        if len(trace.logprobs) != len(trace.completion_token_ids):
            raise ValueError('Sampled tokens and logprobs have different lengths')
        if any(not math.isfinite(lp) or lp > 0 for lp in trace.logprobs):
            raise ValueError('Invalid sampled logprob')
        if not trace.prompt_token_ids or any(t < 0 for t in trace.prompt_token_ids + trace.completion_token_ids):
            raise ValueError('Invalid native token IDs')
        expected_finish = 'stop' if index == len(records) - 1 else 'tool_calls'
        if trace.finish_reason != expected_finish:
            raise ValueError('Incomplete, truncated or unsupported trajectory termination')
        if trace.raw_request is None or trace.raw_request.get('tools', []) != tools:
            raise ValueError('Missing request provenance or changed tool schema')
        if previous_messages is not None and trace.messages[:len(previous_messages)] != previous_messages:
            raise ValueError('Conversation prefix was rewritten')
        if previous_full is not None and trace.prompt_token_ids[:len(previous_full)] != previous_full:
            raise ValueError('Native token prefix diverged; preserve trace without training it')
        previous_full = trace.prompt_token_ids + trace.completion_token_ids
        if len(previous_full) - len(first_prompt) > max_response_tokens:
            raise ValueError('Response region including observations exceeds budget')
        for name, actual in (('prompt', len(trace.prompt_token_ids)), ('completion', len(trace.completion_token_ids))):
            if name not in trace.token_counts or trace.token_counts[name] != actual:
                raise ValueError('Native usage does not cover the captured sampled arrays')
        previous_messages = trace.messages

    # Use the public upstream conversion, including its mask and padding logic.
    # Strict checks above prevent its more permissive fallback behaviors.
    steps = [api.trace_converter.trace_record_to_step(t) for t in records]
    trajectory = api.types.Trajectory(uid=session_id, name='dsh', steps=steps, reward=float(reward_score))
    episode = api.types.Episode(id=session_id + ':0', session_id=session_id, trajectories=[trajectory])
    batch = api.transform.transform_episodes_to_dataproto(
        [episode], SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=pad_token_id)),
        max_prompt_length=max_prompt_tokens, max_response_length=max_response_tokens,
    )
    if len(batch) != 1:
        raise RuntimeError('Upstream transform unexpectedly split or dropped this trajectory')
    response_len = len(previous_full) - len(first_prompt)
    output = api.agent_loop.AgentLoopOutput(
        prompt_ids=batch.batch['prompts'][0, -len(first_prompt):].tolist(),
        response_ids=batch.batch['responses'][0, :response_len].tolist(),
        response_mask=batch.batch['response_mask'][0, :response_len].tolist(),
        response_logprobs=batch.batch['rollout_log_probs'][0, :response_len].tolist(),
        reward_score=float(reward_score),
        num_turns=sum(m.get('role') != 'system' for m in records[-1].messages) + 1,
        metrics=api.agent_loop.AgentLoopMetrics(generate_sequences=sum(t.latency_ms for t in records) / 1000),
        extra_fields={'session_id': session_id, 'policy_version': policy_version,
                      'trace_ids': [t.trace_id for t in records], 'verifier_ref': verifier_ref,
                      'transport': 'native-chat-completions'},
    )
    if output.prompt_ids + output.response_ids != previous_full:
        raise RuntimeError('Upstream conversion changed native token IDs')
    return output
