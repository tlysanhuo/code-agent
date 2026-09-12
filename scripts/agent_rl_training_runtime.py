"""Owned native-HTTP recording session for the DSH training adapter.

The upstream gateway owns HTTP/SSE forwarding. This wrapper adds admission,
native /tokenize preflight, bounded requests and durable evidence, not parsing
or generation. Its Unix socket is the process sandbox's sole network route.
"""
import asyncio
import copy
from contextlib import contextmanager
import hmac
import json
import os
from pathlib import Path
import secrets
import time

import httpx
import uvicorn

from agent_rl_native import native_gateway_config, upstream
from agent_rl_training_task import ROOT, project_path, write_json

upstream()
from rllm_model_gateway.store.memory_store import MemoryTraceStore


class OwnedGatewayServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        # A library-owned gateway must not replace the trainer's signal handlers.
        yield


class DurableTraceStore(MemoryTraceStore):
    def __init__(self, directory):
        super().__init__()
        self.path = project_path(directory) / 'traces.jsonl'

    async def store_trace(self, trace_id, session_id, data):
        # Persist before acknowledging the upstream store operation.
        with self.path.open('a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        await super().store_trace(trace_id, session_id, data)


class NativeTrainingSession:
    def __init__(self, *, address, directory, session_id, policy_version,
                 sampling_params, tokenizer, limits, model='Qwen3.5-27B'):
        self.directory = project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.api_key = secrets.token_urlsafe(32)
        self.model = model
        self.limits = limits
        self.tokenizer = tokenizer
        self.sampling = copy.deepcopy(sampling_params)
        self.sampling.update(return_token_ids=True, logprobs=True,
                             chat_template_kwargs={'enable_thinking': True})
        config = native_gateway_config([address], model=model)
        self.store = DurableTraceStore(self.directory)
        self.app = upstream().gateway.create_app(config, store=self.store)
        self.app.state.sessions.create_session(session_id=session_id, sampling_params=self.sampling)
        self.app.state.proxy.weight_version = policy_version
        worker = config.workers[0]
        self.tokenize_url = worker.url.rstrip('/') + '/tokenize'
        self.socket = ROOT / '.d' / ('trl-' + secrets.token_hex(5) + '.sock')
        self.deadline = time.monotonic() + limits['trajectory_seconds']
        self.calls = 0
        self.first_prompt_length = None
        self.tool_schema = None
        self.lock = asyncio.Lock()
        self.server = self.server_task = self.http = None
        self.errors = []

    async def start(self):
        self.http = httpx.AsyncClient(trust_env=False)
        self.server = OwnedGatewayServer(uvicorn.Config(self, uds=str(self.socket),
            log_level='error', access_log=False, lifespan='on'))
        self.server_task = asyncio.create_task(self.server.serve())
        try:
            async with asyncio.timeout(min(15, self.limits['trajectory_seconds'])):
                while not self.server.started:
                    if self.server_task.done():
                        await self.server_task
                        raise RuntimeError('Native gateway exited during startup')
                    await asyncio.sleep(.02)
            self.socket.chmod(0o666)  # sandbox's unique UID; route exposes only model calls
        except BaseException:
            await self.close()
            raise

    async def close(self):
        try:
            if self.server:
                self.server.should_exit = True
            if self.server_task:
                try:
                    await asyncio.wait_for(asyncio.shield(self.server_task), 5)
                except asyncio.TimeoutError:
                    self.server.force_exit = True
                    self.server_task.cancel()
                    await asyncio.gather(self.server_task, return_exceptions=True)
        finally:
            try:
                if self.http:
                    await self.http.aclose()
            finally:
                self.socket.unlink(missing_ok=True)
                write_json(self.directory / 'session-status.json', {
                    'session_id': self.session_id, 'requests_admitted': self.calls,
                    'errors': self.errors, 'socket_removed': not self.socket.exists(),
                    'policy_version_source': 'verl synchronous rollout row, not an independent GPU weight measurement'})

    async def traces(self):
        # Streaming persistence is scheduled in the upstream generator's finally.
        pending = list(self.app.state.proxy._pending_traces)
        if pending:
            await asyncio.gather(*pending)
        await self.store.flush()
        return await self.store.get_session_traces(self.session_id)

    async def reject(self, send, reason):
        self.errors.append(reason)
        write_json(self.directory / 'rejections.json', self.errors)
        await send({'type': 'http.response.start', 'status': 429,
                    'headers': [(b'content-type', b'application/json')]})
        await send({'type': 'http.response.body', 'body': json.dumps({'error': reason}).encode()})

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        authorization = dict(scope['headers']).get(b'authorization', b'').decode()
        if not hmac.compare_digest(authorization, 'Bearer ' + self.api_key):
            await send({'type': 'http.response.start', 'status': 401, 'headers': []})
            return await send({'type': 'http.response.body', 'body': b'Unauthorized'})
        if scope['method'] != 'POST' or scope['path'] != '/v1/chat/completions':
            return await self.reject(send, 'Only native Chat Completions is exposed to the task')
        if self.lock.locked():
            return await self.reject(send, 'Parallel model requests are outside this training contract')
        async with self.lock:
            if self.errors or self.calls >= self.limits['requests'] or time.monotonic() >= self.deadline:
                return await self.reject(send, 'Session ended or request/time budget exceeded')
            raw = bytearray()
            while True:
                event = await receive()
                if event['type'] == 'http.disconnect':
                    return
                raw.extend(event.get('body', b''))
                if len(raw) > 2 * 1024 * 1024:
                    return await self.reject(send, 'Request too large')
                if not event.get('more_body', False):
                    break
            try:
                body = json.loads(raw)
                if body.get('model') != self.model or body.get('stream') is not True:
                    raise ValueError('Unexpected model or transport')
                write_json(self.directory / f'request-{self.calls + 1:02d}-original.json', body)
                schema = body.get('tools', [])
                if self.tool_schema is None:
                    self.tool_schema = schema
                elif schema != self.tool_schema:
                    raise ValueError('Tool schema changed')
                for message in body['messages']:
                    if message.get('role') == 'tool':
                        if not isinstance(message.get('content'), str):
                            raise ValueError('Only text tool observations are supported')
                        if len(self.tokenizer.encode(message['content'], add_special_tokens=False)) > self.limits['tool_tokens']:
                            raise ValueError('Tool observation exceeds the frozen token budget')
                body.update(self.sampling)
                token_request = {'model': self.model, 'messages': body['messages'], 'tools': schema,
                                 'chat_template_kwargs': body['chat_template_kwargs'],
                                 'add_generation_prompt': True, 'add_special_tokens': False}
                response = await self.http.post(self.tokenize_url, json=token_request,
                    timeout=max(.1, min(10, self.deadline - time.monotonic())))
                response.raise_for_status()
                tokenized = response.json()
                prompt_ids = tokenized['tokens']
                write_json(self.directory / f'prompt-{self.calls + 1:02d}.json', tokenized)
                if self.first_prompt_length is None:
                    if len(prompt_ids) > self.limits['prompt_tokens']:
                        raise ValueError('Complete DSH system/tools/question prompt exceeds budget')
                    self.first_prompt_length = len(prompt_ids)
                previous = await self.traces()
                if previous:
                    full = previous[-1]['prompt_token_ids'] + previous[-1]['completion_token_ids']
                    if prompt_ids[:len(full)] != full:
                        raise ValueError('Native token prefix diverged; refusing to train a rewritten history')
                remaining = self.limits['response_tokens'] - (len(prompt_ids) - self.first_prompt_length)
                if remaining <= 0 or len(prompt_ids) >= self.limits['context_tokens']:
                    raise ValueError('Trajectory context budget exhausted')
                requested = body.get('max_tokens')
                if type(requested) is not int or requested <= 0:
                    raise ValueError('A positive explicit max_tokens is required')
                body['max_tokens'] = min(requested, remaining, self.limits['context_tokens'] - len(prompt_ids))
                body['stream_options'] = {'include_usage': True}
            except Exception as exc:
                return await self.reject(send, f'{type(exc).__name__}: {exc}')
            self.calls += 1
            write_json(self.directory / f'request-{self.calls:02d}-effective.json', body)
            payload = json.dumps(body).encode()
            delivered = False
            async def replay_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': payload, 'more_body': False}
                return await receive()
            forwarded = dict(scope)
            forwarded['path'] = f'/sessions/{self.session_id}/v1/chat/completions'
            forwarded['raw_path'] = forwarded['path'].encode()
            forwarded['headers'] = [(k, v) for k, v in scope['headers'] if k.lower() != b'content-length']
            forwarded['headers'].append((b'content-length', str(len(payload)).encode()))
            with (self.directory / f'response-{self.calls:02d}.sse').open('wb') as stream:
                async def record_send(event):
                    if event['type'] == 'http.response.body':
                        stream.write(event.get('body', b''))
                        stream.flush()
                    await send(event)
                try:
                    async with asyncio.timeout(max(.1, self.deadline - time.monotonic())):
                        await self.app(forwarded, replay_receive, record_send)
                finally:
                    stream.flush()
                    os.fsync(stream.fileno())
