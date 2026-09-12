"""One bounded real repair attempt per frozen SWE-smith task; independent eval."""
import argparse
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import socketserver
import subprocess
import threading
import time
import urllib.request

from qwen_tool_suite import RelayHandler, SERVER, append_json, gpu_snapshot, tool_evidence
from qwen_connectivity_smoke import stop_owned_group
from swe_task_env import ROOT, PY, NATIVE, task_uid
from swe_prepare import BASE, SELECTION, add_acceptance, git, make_case, own, test, write


class Relay(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    def __init__(self, socket_path, case, config):
        self.case, self.config = case, config
        self.rounds = 0
        self.guard = threading.Lock()
        self.deadline = time.monotonic() + config['timeout_seconds']
        super().__init__(str(socket_path), RelayHandler)


def worker(case):
    from deepseek_harness import DeepSeekHarness
    config = json.loads((case / 'case.json').read_text())
    sandbox = case / 'sandbox'
    sock = ROOT / 'tmp' / ('swe-' + secrets.token_hex(4) + '.sock')
    relay = Relay(sock, case, config)
    os.chmod(sock, 0o666)
    thread = threading.Thread(target=relay.serve_forever, daemon=True)
    thread.start()
    wrapper = case / 'dsh-wrapper.sh'
    argv = [str(PY), str(ROOT / 'scripts/swe_task_env.py'), '--sandbox', str(sandbox),
            '--socket', str(sock), '--', str(NATIVE)]
    wrapper.write_text('#!/bin/bash\nexec ' + shlex.join(argv) + ' "$@"\n')
    wrapper.chmod(0o755)
    started = time.monotonic()
    status = {'status': 'failed', 'real_model': True, 'sandbox_uid': task_uid(sandbox)}
    h = None
    try:
        h = DeepSeekHarness(dsh_bin=str(wrapper), provider='qwen-local', model=SERVER['served_model_name'],
                max_tokens=config['max_output_tokens_per_round'], cwd=str(sandbox / 'workspace'),
                dsh_home=str(sandbox / 'dsh-home'), profile='sdk-minimal',
                patches=(str(case / 'qwen.patch.yml'),), initialize_timeout_seconds=60,
                request_timeout_seconds=590, shutdown_timeout_seconds=5)
        with h:
            answer = h.run(config['prompt'], session_id='swe-' + secrets.token_hex(6),
                           on_notification=lambda v: append_json(case / 'notifications.jsonl', dataclasses.asdict(v)))
            write(case / 'result.json', dataclasses.asdict(answer))
            status.update(status='completed' if answer.finish_reason == 'completed' else 'failed',
                          finish_reason=answer.finish_reason, final_response=answer.final_response)
    except Exception as exc:
        status['error'] = repr(exc)
    finally:
        if h:
            h.close()
            status['dsh_stderr'] = list(h.client._stderr_lines)
        relay.shutdown()
        relay.server_close()
        sock.unlink(missing_ok=True)
        status.update(elapsed_seconds=time.monotonic() - started, requests_seen=relay.rounds)
        write(case / 'worker-status.json', status)


def evaluate(case, task):
    work = case / 'sandbox/workspace'
    # Include untracked agent files without altering the baseline commit.
    git(work, 'add', '-N', '.', check=True)
    diff = git(work, 'diff', '--binary', 'HEAD', check=True).stdout
    (case / 'agent.patch').write_text(diff)
    status = git(work, 'status', '--porcelain', check=True).stdout
    (case / 'git-status.txt').write_text(status)
    independent = case / 'independent'
    if independent.exists():
        independent = case / ('independent-' + str(time.time_ns()))
    sandbox = make_case(independent, task, acceptance=False)
    fresh = sandbox / 'workspace'
    if diff.strip():
        applied = git(fresh, 'apply', '--binary', str(case / 'agent.patch'))
        application = {'returncode': applied.returncode, 'stdout': applied.stdout, 'stderr': applied.stderr}
    else:
        application = {'returncode': 0, 'empty_patch': True}
    write(case / 'patch-application.json', application)
    # The oracle comes exclusively from pinned baseline files, never from
    # candidate-created or edited tests, and was absent from the agent view.
    add_acceptance(fresh, task)
    own(sandbox)
    tested = test(independent, task, 'agent-acceptance') if application['returncode'] == 0 else {}
    outcomes = tested.get('outcomes', {}).get('tests', {})
    f2p = sum(outcomes.get(n) == 'passed' for n in task['FAIL_TO_PASS'])
    p2p = sum(outcomes.get(n) == 'passed' for n in task['PASS_TO_PASS'])
    tools, requests = tool_evidence(case)
    usage = []
    for response in sorted(case.glob('response-*.sse')):
        for line in response.read_text().splitlines():
            if line.startswith('data: ') and line != 'data: [DONE]':
                chunk = json.loads(line[6:])
                if chunk.get('usage'):
                    usage.append(chunk['usage'])
    worker_status = json.loads((case / 'worker-status.json').read_text()) if (case / 'worker-status.json').exists() else {}
    resolved = (application['returncode'] == 0 and tested.get('returncode') == 0 and
                f2p == len(task['FAIL_TO_PASS']) and p2p == len(task['PASS_TO_PASS']))
    all_seen = set(outcomes) == set(task['FAIL_TO_PASS'] + task['PASS_TO_PASS'])
    result = {'instance_id': task['instance_id'], 'resolved': resolved,
              'independent_dir': str(independent),
              'pipeline_complete': bool(requests and all_seen and application['returncode'] == 0),
              'classification': 'resolved' if resolved else 'model_failure' if requests and all_seen else 'execution_or_environment_failure',
              'f2p_passed': f2p, 'f2p_total': len(task['FAIL_TO_PASS']),
              'p2p_passed': p2p, 'p2p_total': len(task['PASS_TO_PASS']),
              'requests_recorded': len(requests), 'upstream_responses': len(list(case.glob('response-*.sse'))),
              'output_budget_per_request': 512, 'total_output_budget': 15360,
              'usage_records': usage, 'prompt_tokens': sum(u.get('prompt_tokens', 0) for u in usage),
              'completion_tokens': sum(u.get('completion_tokens', 0) for u in usage),
              'elapsed_seconds': worker_status.get('elapsed_seconds'),
              'worker_status': worker_status, 'tool_calls': len(tools),
              'tool_names': sorted({t['name'] for t in tools}),
              'patch_bytes': len(diff.encode()), 'patch_sha256': hashlib.sha256(diff.encode()).hexdigest()}
    errors = case / 'transport-errors.jsonl'
    if errors.exists():
        result['transport_errors'] = [json.loads(line) for line in errors.read_text().splitlines()]
        if not resolved and all_seen and 'maximum context length' in errors.read_text():
            result['classification'] = 'model_context_budget_failure'
            result['failure_reason'] = 'Tool outputs exceeded the fixed 8192-token context before a repair was produced; no retry or extra budget.'
    write(case / 'evaluation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--evaluate-run', type=Path,
                        help='CPU-only postprocessing of saved attempts; never starts a model')
    parser.add_argument('--task', type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker.resolve())
        return
    if args.evaluate_run:
        run = args.evaluate_run.resolve()
        assert run.is_relative_to(BASE / 'runs')
        summary = json.loads((run / 'summary.json').read_text())
        initial = run / 'initial-summary.json'
        if not initial.exists():
            shutil.copy2(run / 'summary.json', initial)
        task_map = {t['instance_id']: t for t in json.loads(SELECTION.read_text())['tasks']}
        summary['cases'] = [evaluate(case, task_map[json.loads((case / 'case.json').read_text())['instance_id']])
                            for case in sorted(run.glob('task-*'))]
        summary['status'] = 'complete' if all(c['pipeline_complete'] for c in summary['cases']) else 'incomplete'
        summary['cpu_postprocessing_note'] = 'Git safe.directory compatibility fixed using a per-copy project-local config. Re-extracted existing attempts and ran independent acceptance; zero additional model requests.'
        archive = run / 'postprocessing-scripts'
        archive.mkdir(exist_ok=True)
        for name in ['swe_three_tasks.py', 'swe_prepare.py', 'swe_task_env.py', 'audit_swe_run.py']:
            shutil.copy2(ROOT / 'scripts' / name, archive / name)
        write(run / 'summary.json', summary)
        print(json.dumps(summary, indent=2))
        return
    def terminate(signum, frame):
        raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, terminate)
    selection = json.loads(SELECTION.read_text())
    tasks = selection['tasks']
    if args.task:
        tasks = [tasks[args.task - 1]]
    checks = json.loads((BASE / 'environment-checks.json').read_text())
    assert all(any(r['instance_id'] == t['instance_id'] and r['passed'] for r in checks) for t in tasks)
    isolation = json.loads((BASE / 'isolation-check/verification.json').read_text())
    assert isolation['passed'] and not isolation['real_model']
    run = BASE / 'runs' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + secrets.token_hex(3))
    run.mkdir(parents=True)
    for index, task in enumerate(tasks, 1):
        case = run / f'task-{index}'
        sandbox = make_case(case, task)
        prompt = ('Please repair the following issue in the repository at ' + str(sandbox / 'workspace') +
                  '. Inspect the available code and tests, make the repair, and report any tests you run. '
                  'Do not modify existing tests.\n\n' + task['problem_statement'])
        write(case / 'case.json', {'instance_id': task['instance_id'], 'prompt': prompt,
                                  'max_rounds': 30, 'timeout_seconds': 600, 'max_output_tokens_per_round': 512})
    for name in ['gpu-allocation.json', 'qwen-server.json', 'swe-three-tasks.json', 'swe-requirements.lock']:
        shutil.copy2(ROOT / 'configs' / name, run / name)
    archive = run / 'scripts'
    archive.mkdir()
    for name in ['swe_three_tasks.py', 'swe_prepare.py', 'swe_task_env.py', 'swe_atomic_compat.c',
                 'qwen_tool_suite.py', 'launch_qwen.py', 'qwen_connectivity_smoke.py', 'env.sh']:
        shutil.copy2(ROOT / 'scripts' / name, archive / name)
    summary = {'status': 'running', 'evidence_dir': str(run), 'cases': [], 'selection_sha256': hashlib.sha256(SELECTION.read_bytes()).hexdigest()}
    write(run / 'summary.json', summary)
    print('Evidence directory:', run, flush=True)
    service = child = None
    stop_monitor = threading.Event()
    def monitor():
        while not stop_monitor.wait(3):
            gpu_snapshot(run / 'gpu-memory.jsonl')
    thread = threading.Thread(target=monitor, daemon=True)
    try:
        with (run / 'preflight.log').open('w') as log:
            subprocess.run(['bash', str(ROOT / 'scripts/start_qwen.sh'), '--check-only'], stdout=log, stderr=subprocess.STDOUT, check=True)
        gpu_snapshot(run / 'gpu-memory.jsonl')
        thread.start()
        with (run / 'server.log').open('w') as log:
            service = subprocess.Popen(['bash', str(ROOT / 'scripts/start_qwen.sh')], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 720
        while time.monotonic() < deadline:
            if service.poll() is not None:
                raise RuntimeError('Model server exited before readiness')
            try:
                request = urllib.request.Request(f"http://{SERVER['host']}:{SERVER['port']}/v1/models",
                                                 headers={'Authorization': 'Bearer local-qwen'})
                with urllib.request.urlopen(request, timeout=2) as response:
                    catalog = json.load(response)
                if SERVER['served_model_name'] in [x['id'] for x in catalog['data']]:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError('Server startup exceeded 12 minutes')
        shutil.copy2(ROOT / 'runtime/metadata/qwen-service-launch.json', run / 'service-launch.json')
        for index, task in enumerate(tasks, 1):
            case = run / f'task-{index}'
            with (case / 'worker.log').open('w') as log:
                child = subprocess.Popen([str(PY), str(Path(__file__).resolve()), '--worker', str(case)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                started = time.monotonic()
                try:
                    rc = child.wait(timeout=600)
                    process = {'pid': child.pid, 'returncode': rc, 'timeout': False}
                except subprocess.TimeoutExpired:
                    process = {'pid': child.pid, 'timeout': True}
                finally:
                    stop_owned_group(child)
                    child = None
                process['wall_seconds'] = time.monotonic() - started
                write(case / 'process.json', process)
            try:
                result = evaluate(case, task)
            except Exception as exc:
                result = {'instance_id': task['instance_id'], 'pipeline_complete': False,
                          'classification': 'execution_or_environment_failure', 'error': repr(exc)}
                write(case / 'evaluation-error.json', result)
            summary['cases'].append(result)
            write(run / 'summary.json', summary)
            print(task['instance_id'], result['classification'], flush=True)
        summary['status'] = 'complete' if all(r['pipeline_complete'] for r in summary['cases']) else 'incomplete'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        raise
    finally:
        if child:
            stop_owned_group(child)
        if service:
            stop_owned_group(service)
        stop_monitor.set()
        if thread.is_alive():
            thread.join(timeout=10)
        selected = json.loads((run / 'gpu-allocation.json').read_text())['gpu_uuids']
        idle = False
        for _ in range(20):
            snapshot = gpu_snapshot(run / 'gpu-memory.jsonl')
            rows = list(csv.reader(snapshot['gpus'].splitlines()))
            idle = all(any(r[1].strip() == uuid and int(r[2].strip()) < 100 for r in rows)
                       and uuid not in snapshot['processes'] for uuid in selected)
            if idle:
                break
            time.sleep(1)
        leftovers = []
        uids = {task_uid(run / f'task-{i}/sandbox') for i in range(1, len(tasks) + 1)}
        for p in Path('/proc').glob('[0-9]*'):
            try:
                uid = int(next(l.split()[1] for l in (p / 'status').read_text().splitlines() if l.startswith('Uid:')))
                if uid in uids or (p / 'cwd').resolve().is_relative_to(run):
                    leftovers.append(int(p.name))
            except (OSError, StopIteration):
                pass
        summary.update(cleanup_verified=bool(idle and not leftovers), after_cleanup=snapshot,
                       processes_remaining=leftovers, service_returncode=service.returncode if service else None,
                       finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write(run / 'summary.json', summary)
    if summary['status'] != 'complete' or not summary['cleanup_verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
