"""Real DSH tool probes with bounded requests, independent checks and owned teardown.

--prepare-only creates synthetic fixtures and checks their failing baseline on CPU.
The normal entry launches the pinned Qwen server, runs each case in its own process
and DSH home, preserves every attempt, and stops its service in finally.
"""
import argparse
import csv
import dataclasses
import difflib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

from qwen_connectivity_smoke import stop_owned_group

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv-dsh/bin/python'
LIMITS = json.loads((ROOT / 'configs/tool-smoke.json').read_text())
SERVER = json.loads((ROOT / 'configs/qwen-server.json').read_text())
SHELL_SET = 'export DSH_STATE_TOKEN="$(cat shell_seed.txt)"'
SHELL_GET = 'printf \'%s\\n\' "$DSH_STATE_TOKEN"'
BUG = 'def clamp(value, low, high):\n    """Clamp value to the inclusive bounds low and high."""\n    return min(low, max(value, high))\n'
TEST = '''import unittest
from clamp import clamp

class TestClamp(unittest.TestCase):
    def test_bounds(self):
        for value, lo, hi, expected in [(-3, 0, 10, 0), (5, 0, 10, 5),
                                        (12, 0, 10, 10), (0, 0, 10, 0),
                                        (10, 0, 10, 10), (-4, -9, -2, -4)]:
            with self.subTest(value=value):
                self.assertEqual(clamp(value, lo, hi), expected)
'''


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def append_json(path, value):
    with path.open('a') as output:
        output.write(json.dumps(value, ensure_ascii=False) + '\n')


def run_tests(workspace, destination):
    p = subprocess.run([str(PYTHON), '-m', 'unittest', '-v', 'test_clamp.py'],
                       cwd=workspace, capture_output=True, text=True, timeout=15)
    result = {'command': [str(PYTHON), '-m', 'unittest', '-v', 'test_clamp.py'],
              'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    write_json(destination, result)
    return result


def prepare(run_dir):
    for name, settings in LIMITS['cases'].items():
        case = run_dir / name
        workspace = case / 'workspace'
        workspace.mkdir(parents=True)
        token = secrets.token_hex(16)
        if name == 'read':
            (workspace / 'read_target.txt').write_text('requested_value=' + token + '\n')
            prompt = 'Use a file reading tool to read read_target.txt in this workspace. Return only the requested_value from the file. Do not guess, modify files, or use the network.'
        elif name == 'shell':
            (workspace / 'shell_seed.txt').write_text(token + '\n')
            prompt = (f'Use the persistent bash tool in two separate, sequential calls. First run exactly: {SHELL_SET}. '
                      f'After it finishes, in a second call run exactly: {SHELL_GET}. '
                      'Do not combine calls, reassign the variable, or read the file in the second call. Return the second command output only.')
        else:
            (workspace / 'clamp.py').write_text(BUG)
            (workspace / 'test_clamp.py').write_text(TEST)
            prompt = ('The clamp(value, low, high) function in clamp.py is broken. It must return value bounded by inclusive low and high; assume low <= high. '
                      'Inspect the files and run python -m unittest -v test_clamp.py to observe the failing tests. '
                      'Use str_replace_editor to edit clamp.py, then use bash to run the same tests again. '
                      'Do not change test_clamp.py or other files and do not use the network. Report the actual final test result.')
            baseline = run_tests(workspace, case / 'baseline-tests.json')
            if baseline['returncode'] == 0 or 'FAIL' not in baseline['stderr']:
                raise RuntimeError('Fixture must start with observed failing tests')
        before = {p.name: p.read_text() for p in workspace.iterdir() if p.is_file()}
        write_json(case / 'before.json', before)
        write_json(case / 'case.json', {'name': name, 'prompt': prompt, 'expected_token': token,
                   'max_rounds': LIMITS['max_rounds'], 'timeout_seconds': LIMITS['case_timeout_seconds'],
                   **settings, 'max_output_tokens_total': LIMITS['max_rounds'] * settings['max_output_tokens_per_round']})
    for name in ['gpu-allocation.json', 'qwen-server.json', 'tool-smoke.json', 'dsh-qwen.patch.yml']:
        shutil.copy2(ROOT / 'configs' / name, run_dir / name)
    source_dir = run_dir / 'scripts'
    source_dir.mkdir()
    for source in [Path(__file__), ROOT / 'scripts/launch_qwen.py', ROOT / 'scripts/qwen_connectivity_smoke.py']:
        shutil.copy2(source, source_dir / source.name)
    write_json(run_dir / 'provenance.json', {
        'dsh_source': json.loads((ROOT / 'runtime/metadata/dsh-source.json').read_text()),
        'runtime_versions': json.loads((ROOT / 'runtime/metadata/runtime-versions.json').read_text()),
        'scripts_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), ROOT / 'scripts/launch_qwen.py']},
        'fixtures_prepared_on_cpu': True})


class RecordingRelay(ThreadingHTTPServer):
    """Forward real SSE verbatim; reject an eleventh upstream request."""
    daemon_threads = True

    def __init__(self, case, config):
        self.case = case
        self.config = config
        self.rounds = 0
        self.guard = threading.Lock()
        self.deadline = time.monotonic() + config['timeout_seconds']
        super().__init__(('127.0.0.1', 0), RelayHandler)


class RelayHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        relay = self.server
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        with relay.guard:
            relay.rounds += 1
            number = relay.rounds
        append_json(relay.case / 'requests.jsonl', {'request': number, 'time': time.time(), 'body': body})
        allowed = (self.path == '/v1/chat/completions' and body.get('stream') is True
                   and body.get('model') == SERVER['served_model_name']
                   and 0 < body.get('max_tokens', 0) <= relay.config['max_output_tokens_per_round']
                   and number <= relay.config['max_rounds'] and time.monotonic() < relay.deadline)
        if not allowed:
            append_json(relay.case / 'limits.jsonl', {'request': number, 'action': 'rejected_before_upstream'})
            self.send_error(429, 'Probe budget or request validation failed')
            return
        url = f"http://{SERVER['host']}:{SERVER['port']}/v1/chat/completions"
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            'Authorization': 'Bearer local-qwen', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=max(1, min(60, relay.deadline-time.monotonic()))) as response:
                self.send_response(response.status)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'text/event-stream'))
                self.end_headers()
                with (relay.case / f'response-{number:02d}.sse').open('wb') as output:
                    while time.monotonic() < relay.deadline:
                        line = response.readline()
                        if not line:
                            break
                        output.write(line)
                        output.flush()
                        self.wfile.write(line)
                        self.wfile.flush()
                        if line.strip() == b'data: [DONE]':
                            break
        except Exception as exc:
            details = {'request': number, 'error': repr(exc)}
            if isinstance(exc, urllib.error.HTTPError):
                details['body'] = exc.read().decode(errors='replace')
            append_json(relay.case / 'transport-errors.jsonl', details)
            self.close_connection = True


def worker(case):
    from deepseek_harness import DeepSeekHarness
    config = json.loads((case / 'case.json').read_text())
    relay = RecordingRelay(case, config)
    thread = threading.Thread(target=relay.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    result = {'status': 'failed', 'real_model': True}
    harness = None
    try:
        harness = DeepSeekHarness(provider='qwen-local', model=SERVER['served_model_name'],
                  max_tokens=config['max_output_tokens_per_round'], cwd=str(case / 'workspace'),
                  dsh_home=str(case / 'dsh-home'), profile='sdk-minimal',
                  patches=(str(ROOT / 'configs/dsh-qwen.patch.yml'),),
                  env={'CUDA_VISIBLE_DEVICES': '', 'QWEN_LOCAL_API_KEY': 'local-qwen',
                       'QWEN_BASE_URL': f'http://127.0.0.1:{relay.server_port}/v1',
                       'PATH': str(PYTHON.parent) + os.pathsep + os.environ['PATH']},
                  initialize_timeout_seconds=60, request_timeout_seconds=config['timeout_seconds'],
                  shutdown_timeout_seconds=3)
        with harness:
            answer = harness.run(config['prompt'], session_id='probe-' + config['name'],
                on_notification=lambda value: append_json(case / 'notifications.jsonl', dataclasses.asdict(value)))
            write_json(case / 'result.json', dataclasses.asdict(answer))
            result.update(status='completed' if answer.finish_reason == 'completed' else 'failed',
                          finish_reason=answer.finish_reason, final_response=answer.final_response)
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        if harness is not None:
            harness.close()
        relay.shutdown()
        relay.server_close()
        thread.join(timeout=3)
        result.update(elapsed_seconds=time.monotonic()-started, requests_seen=relay.rounds)
        write_json(case / 'worker-status.json', result)
    return result['status'] == 'completed'


def tool_evidence(case):
    calls, results = {}, {}
    requests = []
    path = case / 'requests.jsonl'
    if path.exists():
        requests = [json.loads(line) for line in path.read_text().splitlines()]
    for request in requests:
        for message in request['body'].get('messages', []):
            if message.get('role') == 'assistant':
                for call in message.get('tool_calls', []):
                    calls[call['id']] = call
            if message.get('role') == 'tool':
                results[message['tool_call_id']] = message.get('content', '')
    combined = []
    for key, call in calls.items():
        try:
            arguments = json.loads(call['function']['arguments'])
        except (TypeError, ValueError):
            arguments = {}
        combined.append({'id': key, 'name': call['function']['name'], 'arguments': arguments,
                         'result': results.get(key)})
    write_json(case / 'tool-evidence.json', combined)
    return combined, requests


def verify_case(case):
    config = json.loads((case / 'case.json').read_text())
    before = json.loads((case / 'before.json').read_text())
    workspace = case / 'workspace'
    status_path = case / 'worker-status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    tools, requests = tool_evidence(case)
    checks = {'worker_completed': status.get('status') == 'completed',
              'round_budget': 0 < len(requests) <= config['max_rounds'],
              'time_budget': status.get('elapsed_seconds', float('inf')) <= config['timeout_seconds'],
              'stream_recorded': bool(list(case.glob('response-*.sse'))),
              'tool_results_present': bool(tools) and all(t['result'] is not None for t in tools)}
    if config['name'] == 'read':
        checks.update(token_from_tool=any(config['expected_token'] in str(t['result']) for t in tools),
                      answer_matches=status.get('final_response', '').strip() == config['expected_token'],
                      file_unchanged=(workspace / 'read_target.txt').read_text() == before['read_target.txt'])
    elif config['name'] == 'shell':
        bash = [t for t in tools if t['name'] == 'bash']
        checks['separate_persistent_calls'] = (len(bash) == 2
            and bash[0]['arguments'].get('command', '').strip() == SHELL_SET
            and bash[1]['arguments'].get('command', '').strip() == SHELL_GET
            and config['expected_token'] in str(bash[1]['result']))
        checks['answer_matches'] = status.get('final_response', '').strip() == config['expected_token']
    else:
        after = (workspace / 'clamp.py').read_text()
        (case / 'changes.diff').write_text(''.join(difflib.unified_diff(
            before['clamp.py'].splitlines(True), after.splitlines(True), fromfile='before/clamp.py', tofile='after/clamp.py')))
        checks['source_changed'] = before['clamp.py'] != after
        checks['tests_unchanged'] = (workspace / 'test_clamp.py').read_text() == before['test_clamp.py']
        checks['editor_used'] = any(t['name'] == 'str_replace_editor' and
            t['arguments'].get('command') in ('str_replace', 'create') for t in tools)
        test_calls = [t for t in tools if t['name'] == 'bash' and
                      'unittest' in t['arguments'].get('command', '')]
        checks['model_ran_failing_tests'] = any('FAIL' in str(t['result']) for t in test_calls)
        checks['model_ran_passing_tests'] = any('OK' in str(t['result']) for t in test_calls)
        checks['independent_tests_passed'] = run_tests(workspace, case / 'independent-tests.json')['returncode'] == 0
        # A separate process tests inputs not listed in the visible test file.
        oracle = '''import importlib.util,sys
s=importlib.util.spec_from_file_location("candidate",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
for lo,hi in [(-7,3),(0,0),(-10,-2),(2,11)]:
 for value in range(-15,16):
  expected=lo if value<lo else hi if value>hi else value
  assert m.clamp(value,lo,hi)==expected,(value,lo,hi)
print("124 independent checks passed")
'''
        p = subprocess.run([str(PYTHON), '-I', '-c', oracle, str(workspace / 'clamp.py')],
                           capture_output=True, text=True, timeout=15, cwd=case)
        write_json(case / 'independent-oracle.json', {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
        checks['independent_oracle_passed'] = p.returncode == 0
    result = {'case': config['name'], 'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'real_model_verified': all(checks.values())}
    write_json(case / 'verification.json', result)
    return result


def gpu_snapshot(path):
    values = {}
    for name, fields in [('gpus', '--query-gpu=index,uuid,memory.used,utilization.gpu'),
                         ('processes', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory')]:
        p = subprocess.run(['nvidia-smi', fields, '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=8)
        values[name] = p.stdout
    append_json(path, {'time': time.time(), **values})
    return values


def main():
    def terminate(_signum, _frame):
        raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        case = args.worker.resolve()
        assert case.is_relative_to(ROOT / 'runtime/tool-smoke')
        raise SystemExit(0 if worker(case) else 1)
    run_dir = ROOT / 'runtime/tool-smoke' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + secrets.token_hex(3))
    run_dir.mkdir(parents=True)
    prepare(run_dir)
    print('Evidence directory:', run_dir, flush=True)
    summary = {'status': 'prepared' if args.prepare_only else 'running', 'real_model_verified': False,
               'evidence_dir': str(run_dir), 'cases': [], 'gpu_service_started': False}
    write_json(run_dir / 'summary.json', summary)
    if args.prepare_only:
        return
    service = None
    child = None
    stop_monitor = threading.Event()
    def monitor():
        while not stop_monitor.wait(3):
            gpu_snapshot(run_dir / 'gpu-memory.jsonl')
    thread = threading.Thread(target=monitor, daemon=True)
    try:
        with (run_dir / 'preflight.log').open('w') as log:
            subprocess.run(['bash', str(ROOT / 'scripts/start_qwen.sh'), '--check-only'], stdout=log, stderr=subprocess.STDOUT, check=True)
        gpu_snapshot(run_dir / 'gpu-memory.jsonl')
        thread.start()
        with (run_dir / 'server.log').open('w') as log:
            service = subprocess.Popen(['bash', str(ROOT / 'scripts/start_qwen.sh')], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        summary['gpu_service_started'] = True
        deadline = time.monotonic() + 720
        while time.monotonic() < deadline:
            if service.poll() is not None:
                raise RuntimeError('Model server exited before readiness')
            try:
                url = f"http://{SERVER['host']}:{SERVER['port']}/v1/models"
                request = urllib.request.Request(url, headers={'Authorization': 'Bearer local-qwen'})
                with urllib.request.urlopen(request, timeout=2) as response:
                    catalog = json.load(response)
                if SERVER['served_model_name'] in [x['id'] for x in catalog['data']]:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError('Server startup exceeded 12 minutes')
        shutil.copy2(ROOT / 'runtime/metadata/qwen-service-launch.json', run_dir / 'service-launch.json')
        for name in LIMITS['cases']:
            case = run_dir / name
            with (case / 'worker.log').open('w') as log:
                child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--worker', str(case)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = child.wait(timeout=LIMITS['case_timeout_seconds'])
                    write_json(case / 'process.json', {'pid': child.pid, 'returncode': returncode, 'timeout': False})
                except subprocess.TimeoutExpired:
                    write_json(case / 'process.json', {'pid': child.pid, 'timeout': True})
                finally:
                    stop_owned_group(child)
                    child = None
            try:
                result = verify_case(case)
            except Exception as exc:
                result = {'case': name, 'status': 'failed', 'error': repr(exc)}
                write_json(case / 'verification.json', result)
            summary['cases'].append(result)
            write_json(run_dir / 'summary.json', summary)
            print(name, result['status'], flush=True)
        summary['status'] = 'passed' if all(c['status'] == 'passed' for c in summary['cases']) else 'failed'
        summary['real_model_verified'] = summary['status'] == 'passed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        raise
    finally:
        if child is not None:
            stop_owned_group(child)
        if service is not None:
            stop_owned_group(service)
        stop_monitor.set()
        if thread.is_alive():
            thread.join(timeout=10)
        summary['after_cleanup'] = gpu_snapshot(run_dir / 'gpu-memory.jsonl')
        selected = json.loads((run_dir / 'gpu-allocation.json').read_text())['gpu_uuids']
        for _ in range(20):
            snapshot = summary['after_cleanup']
            rows = list(csv.reader(snapshot['gpus'].splitlines()))
            idle = all(any(r[1].strip() == uuid and int(r[2].strip()) < 100 for r in rows)
                       and uuid not in snapshot['processes'] for uuid in selected)
            if idle:
                break
            time.sleep(1)
            summary['after_cleanup'] = gpu_snapshot(run_dir / 'gpu-memory.jsonl')
        leftovers = []
        for p in Path('/proc').glob('[0-9]*'):
            try:
                if (p / 'cwd').resolve().is_relative_to(run_dir):
                    leftovers.append(int(p.name))
            except OSError:
                pass
        summary['cleanup_verified'] = bool(idle and not leftovers)
        summary['workspace_processes_remaining'] = leftovers
        if not summary['cleanup_verified']:
            summary['status'] = 'failed'
            summary['real_model_verified'] = False
        summary['service_returncode'] = service.returncode if service else None
        summary['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(run_dir / 'summary.json', summary)
    if summary['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
