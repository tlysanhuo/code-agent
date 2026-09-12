"""Registered DSH AgentLoop for custom, qualified local training tasks.

The upstream trainer retains sampling/update/checkpoint ownership. This adapter
owns one task lease, one DSH worker and one native HTTP recording session.
"""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import threading
import time

from agent_rl_native import upstream, traces_to_agent_output
from agent_rl_training_task import (ROOT, DSH_PYTHON, ISOLATOR, LocalTrainingTask,
                                    TrainingEnvironmentError, CandidateRejected, project_path, write_json)
from agent_rl_training_runtime import NativeTrainingSession

api = upstream()
AgentLoopBase = api.agent_loop.AgentLoopBase
register = api.agent_loop.register
NATIVE = ROOT / '.venv-dsh/lib/python3.12/site-packages/deepseek_harness_runtime/runtime/deepseek-harness-sdk-runtime-linux-x64'


async def stop_worker(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(child.wait(), 4)
    except asyncio.TimeoutError:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await child.wait()


async def owned_thread(function, *args, **kwargs):
    """Keep cancellation from orphaning a verifier subprocess in a thread."""
    cancel = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(function, *args, cancel_event=cancel, **kwargs))
    try:
        return await asyncio.shield(task)
    except BaseException:
        cancel.set()
        await asyncio.gather(task, return_exceptions=True)
        raise


@register('dsh_agent')
class DshAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        settings = self.config.dsh_training
        task_id = kwargs['extra_info']['instance_id']
        registry = json.loads(project_path(settings.task_registry).read_text())
        entry = registry['tasks'].get(task_id)
        if not entry or not entry.get('task_spec'):
            raise TrainingEnvironmentError(f'Custom training task not prepared: {task_id}')
        task = LocalTrainingTask.load(entry['task_spec'])
        if task.spec['task_id'] != task_id:
            raise TrainingEnvironmentError('Task registry identity mismatch')
        qualification = task.require_qualified()
        step = kwargs.get('global_steps')
        if step is None:
            raise ValueError('The upstream rollout row must provide global_steps')
        step = int(step)
        limits = {
            'requests': int(self.rollout_config.multi_turn.max_assistant_turns),
            'trajectory_seconds': float(settings.trajectory_seconds),
            'prompt_tokens': int(self.config.data.max_prompt_length),
            'response_tokens': int(self.config.data.max_response_length),
            'context_tokens': int(self.rollout_config.max_model_len),
            'tool_tokens': int(self.rollout_config.multi_turn.max_tool_response_length),
        }
        prompt_messages = kwargs.get('raw_prompt', kwargs.get('prompt'))
        if prompt_messages is None or len(prompt_messages) != 1 or prompt_messages[0]['role'] != 'user':
            raise ValueError('Expected exactly one policy-visible training problem')
        problem = prompt_messages[0]['content']
        run_root = project_path(settings.run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        run = run_root / (time.strftime('%Y%m%dT%H%M%S') + '-' + secrets.token_hex(5))
        run.mkdir()
        sid = 'train-' + secrets.token_hex(8)
        started = time.monotonic()
        deadline = started + limits['trajectory_seconds']
        child = session = handle = None
        server_id = None
        lease = (run_root / '.task-lease').open('a+')
        write_json(run / 'run.json', {'task_id': task_id, 'protocol': task.spec['protocol'],
            'session_id': sid, 'policy_version': step, 'limits': limits,
            'official_benchmark_result': False, 'task_fingerprint': task.fingerprint})
        cleanup = {'worker_stopped': False, 'lease_released': False}
        try:
            # Kernel lease serializes environments across every local worker/process.
            while True:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Task environment lease deadline exceeded')
                    await asyncio.sleep(.1)
            async with asyncio.timeout(max(.1, deadline - time.monotonic())):
                # Preparation/export contain only bounded local Git/file operations.
                agent_case = task.prepare(run / 'agent')
                server_id, handle = await self.server_manager._acquire_server(sid)
                session = NativeTrainingSession(address=server_id, directory=run / 'native',
                    session_id=sid, policy_version=step, sampling_params=sampling_params,
                    tokenizer=self.tokenizer, limits=limits)
                # Include preparation time in the same end-to-end deadline.
                session.deadline = deadline
                await session.start()
                runtime_config = agent_case / 'runtime-config.json'
                runtime_settings = json.loads(runtime_config.read_text())
                runtime_settings.setdefault('task_env', {})['QWEN_LOCAL_API_KEY'] = session.api_key
                write_json(runtime_config, runtime_settings)
                patch = agent_case / 'qwen.patch.yml'
                shutil.copyfile(project_path(settings.dsh_patch), patch)
                wrapper = agent_case / 'dsh-wrapper.sh'
                command = [str(DSH_PYTHON), str(ISOLATOR), '--sandbox', str(agent_case / 'sandbox'),
                           '--socket', str(session.socket), '--', str(NATIVE)]
                wrapper.write_text('#!/usr/bin/env bash\nexec ' + shlex.join(command) + ' "$@"\n')
                wrapper.chmod(0o755)
                job = {'dsh_bin': str(wrapper), 'workspace': str(agent_case / 'sandbox/workspace'),
                    'dsh_home': str(agent_case / 'sandbox/dsh-home'), 'patch': str(patch),
                    'prompt': problem, 'session_id': sid, 'max_tokens': limits['response_tokens'],
                    'timeout': max(1, deadline - time.monotonic()), 'result': str(run / 'dsh-result.json')}
                write_json(run / 'dsh-job.json', job)
                with (run / 'dsh-worker.log').open('wb') as log:
                    child = await asyncio.create_subprocess_exec(str(DSH_PYTHON),
                        str(ROOT / 'scripts/agent_rl_dsh_worker.py'), str(run / 'dsh-job.json'),
                        stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
                    code = await child.wait()
                result = json.loads((run / 'dsh-result.json').read_text())
                if code or not result.get('completed') or session.errors:
                    raise TrainingEnvironmentError('DSH session failed or was truncated; preserving evidence without reward')
                verifier_ref = run / 'evaluation/verification.json'
                try:
                    candidate = task.export_patch(agent_case, run / 'candidate.patch')
                except CandidateRejected as exc:
                    evaluation = {'protocol': task.spec['protocol'], 'reward': 0,
                                  'candidate_rejected': str(exc), 'official_benchmark_result': False}
                    write_json(verifier_ref, evaluation)
                else:
                    evaluation = await owned_thread(task.evaluate, run / 'evaluation', patch=candidate)
                    if evaluation['test_ids'] != qualification['gold']['test_ids']:
                        raise TrainingEnvironmentError('Verifier test set changed; no reward produced')
                traces = await session.traces()
                output = traces_to_agent_output(traces, session_id=sid, model='Qwen3.5-27B',
                    policy_version=step, reward_score=evaluation['reward'],
                    verifier_ref=str(verifier_ref), pad_token_id=self.tokenizer.pad_token_id,
                    max_prompt_tokens=limits['prompt_tokens'], max_response_tokens=limits['response_tokens'],
                    max_requests=limits['requests'])
                output.extra_fields.update(training_protocol=task.spec['protocol'],
                                           official_benchmark_result=False, evidence_dir=str(run))
                write_json(run / 'agent-loop-output.json', output.model_dump())
                return output
        except BaseException as exc:
            write_json(run / 'failure.json', {'error': repr(exc), 'reward_produced': False,
                                             'elapsed_seconds': time.monotonic() - started})
            # This replica belongs to the training job, and the global task lease
            # admits only this trajectory. Do not touch unrelated model services.
            if handle is not None:
                try:
                    await asyncio.wait_for(handle.abort_all_requests.remote(), 5)
                except Exception as abort_error:
                    cleanup['abort_error'] = repr(abort_error)
            raise
        finally:
            try:
                await stop_worker(child)
                cleanup['worker_stopped'] = child is None or child.returncode is not None
                if session:
                    await session.close()
                    cleanup['socket_removed'] = not session.socket.exists()
            finally:
                try:
                    if server_id is not None:
                        self.server_manager._release_server(server_id, request_id=sid)
                finally:
                    fcntl.flock(lease, fcntl.LOCK_UN)
                    lease.close()
                    cleanup['lease_released'] = True
                    write_json(run / 'cleanup.json', cleanup)
