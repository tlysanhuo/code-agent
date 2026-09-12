"""CPU and real-32K facility gates, independent of the frozen evaluation tasks."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import socketserver
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deepseek_harness import DeepSeekHarness
from gym_relay import Relay
from gym_task_env import ROOT, PY, NATIVE, task_uid
from swe_prepare import own, write
from qwen_connectivity_smoke import stop_owned_group
from qwen_tool_suite import gpu_snapshot

SETTINGS = {'max_rounds': 30, 'timeout_seconds': 600, 'max_output_tokens_total': 16384}


def prepare(case):
    sandbox = case / 'sandbox'
    for name in ['workspace', 'home', 'tmp', 'cache', 'dsh-home', 'artifacts']:
        (sandbox / name).mkdir(parents=True)
    work = sandbox / 'workspace'
    (work / 'stress.txt').write_text(''.join(f'line {n:05d}: recorded payload abcdefghijklmnopqrstuvwxyz\n' for n in range(12000)))
    (work / 'repair.py').write_text('def add_one(n):\n    return n - 1\n')
    (work / 'test_repair.py').write_text('from repair import add_one\ndef test_add_one():\n    assert add_one(4) == 5\n')
    (work / 'bigdir').mkdir()
    for n in range(200):
        (work / 'bigdir' / f'example_file_{n:04d}.txt').write_text('directory pagination probe\n')
    shutil.copy2(ROOT / 'configs/gym-dsh-candidate.patch.yml', case / 'qwen.patch.yml')
    write(case / 'runtime-config.json', {'task_python': str(ROOT / '.venv-swe/bin/python')})
    own(sandbox)
    return sandbox


def execute(case, upstream, prompt):
    sandbox = case / 'sandbox'
    sock = ROOT / 'tmp' / ('gym-' + secrets.token_hex(4) + '.sock')
    relay = Relay(sock, case, SETTINGS, upstream)
    os.chmod(sock, 0o666)
    threading.Thread(target=relay.serve_forever, daemon=True).start()
    wrapper = case / 'dsh-wrapper.sh'
    command = [str(PY), str(ROOT / 'scripts/gym_task_env.py'), '--sandbox', str(sandbox), '--socket', str(sock), '--', str(NATIVE)]
    wrapper.write_text('#!/bin/bash\nexec ' + shlex.join(command) + ' "$@"\n')
    wrapper.chmod(0o755)
    started = time.monotonic()
    h = None
    status = {'status': 'failed'}
    try:
        h = DeepSeekHarness(dsh_bin=str(wrapper), provider='qwen-local', model='Qwen3.5-27B', max_tokens=2048,
                cwd=str(sandbox / 'workspace'), dsh_home=str(sandbox / 'dsh-home'), profile='sdk-minimal',
                patches=(str(case / 'qwen.patch.yml'),), initialize_timeout_seconds=60,
                request_timeout_seconds=590, shutdown_timeout_seconds=5)
        with h:
            result = h.run(prompt, session_id='facility-' + secrets.token_hex(4),
                on_notification=lambda v: append(case / 'notifications.jsonl', dataclasses.asdict(v)))
            write(case / 'result.json', dataclasses.asdict(result))
            status.update(status='completed' if result.finish_reason == 'completed' else 'failed',
                          finish_reason=result.finish_reason, final_response=result.final_response)
    except Exception as exc:
        status['error'] = repr(exc)
    finally:
        if h:
            h.close()
            status['dsh_stderr'] = list(h.client._stderr_lines)
        relay.shutdown()
        relay.server_close()
        sock.unlink(missing_ok=True)
        status.update(elapsed_seconds=time.monotonic()-started, requests=relay.rounds,
                      reported_output_tokens=relay.output_tokens, charged_output_tokens=relay.charged_output_tokens)
        write(case / 'worker-status.json', status)
    return status


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def mock(case):
    calls = []
    workspace = case / 'sandbox/workspace'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            n = len(calls)
            commands = {
                1: ('bash', {'command': 'cat stress.txt'}),
                2: ('str_replace_editor', {'command': 'view', 'path': str(workspace / 'bigdir')}),
                3: ('str_replace_editor', {'command': 'view', 'path': str(workspace / 'stress.txt')}),
                5: ('bash', {'command': 'python -m pytest -q test_repair.py'}),
                6: ('str_replace_editor', {'command': 'str_replace', 'path': str(workspace / 'repair.py'),
                                         'old_str': 'return n - 1', 'new_str': 'return n + 1'}),
                7: ('bash', {'command': 'python -m pytest -q test_repair.py'}),
                8: ('bash', {'command': 'sleep 45'}),
                9: ('bash', {'command': 'printf RECOVERED'}),
            }
            if n == 4:
                last = next(m['content'] for m in reversed(body['messages']) if m['role'] == 'tool')
                key = re.search(r'output=([0-9a-f]{24})', last).group(1)
                commands[n] = ('bash', {'command': f'read_output {key} 6000'})
            if n in commands:
                name, args = commands[n]
                delta = {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': f'cpu{n}', 'type': 'function',
                         'function': {'name': name, 'arguments': json.dumps(args)}}]}
                finish = 'tool_calls'
            else:
                delta, finish = {'role': 'assistant', 'content': 'CPU_GATE_DONE'}, 'stop'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for d, f, usage in [(delta, None, None), ({}, finish, None), ({}, None, {'prompt_tokens':100,'completion_tokens':10,'total_tokens':110})]:
                chunk = {'id': f'mock-{n}', 'object':'chat.completion.chunk', 'created':int(time.time()),
                         'model':'Qwen3.5-27B','choices': [] if usage else [{'index':0,'delta':d,'finish_reason':f}], 'usage':usage}
                self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status = execute(case, f'http://127.0.0.1:{server.server_port}/v1', 'Exercise facility tools.')
    finally:
        server.shutdown()
        server.server_close()
    results = {m['tool_call_id']: m['content'] for m in calls[-1]['messages'] if m['role'] == 'tool'} if calls else {}
    raw = case / 'sandbox/artifacts/shell-terminal.raw'
    pager_result = next((m['content'] for m in calls[4]['messages'] if m.get('tool_call_id')=='cpu4'), '') if len(calls)>4 else ''
    timing = [json.loads(line) for line in (case/'dsh-requests.jsonl').read_text().splitlines()]
    timeout_elapsed = timing[8]['time']-timing[7]['time'] if len(timing)>8 else 0
    timeout_raw = results.get('cpu8','')
    checks = {'completed': status['status']=='completed', 'all_9_tools_paired': len(results)==9,
              'shell_full_output_recorded': raw.exists() and 'line 11999' in raw.read_text(errors='replace'),
              'shell_paged': 'PAGED' in results.get('cpu1',''), 'directory_paged': 'PAGED' in results.get('cpu2',''),
              'file_paged': 'PAGED' in results.get('cpu3',''),
              'pager_works': 'bytes 6000:' in pager_result,
              'buggy_test_failed': 'failed' in results.get('cpu5',''),
              'editor_worked': 'edited successfully' in results.get('cpu6',''),
              'repaired_test_passed': '1 passed' in results.get('cpu7',''),
              'timeout_bounded_failure': 28<=timeout_elapsed<=38 and (
                  'timeout' in timeout_raw.lower() or 'timed out' in timeout_raw.lower()
                  or 'cannot resolve foreground process group' in timeout_raw),
              'shell_recovers': 'RECOVERED' in results.get('cpu9',''),
              'tool_view_caps': all(sum(len(str(m['content']).encode()) for m in b['messages'] if m['role']=='tool')<=16384
                                    for b in calls)}
    write(case / 'verification.json', {'real_model':False,'checks':checks,'passed':all(checks.values()),
          'timeout_elapsed_seconds':timeout_elapsed,'timeout_raw_diagnostic':timeout_raw,
          'timeout_note':'PID namespace has no remounted /proc on this platform. DSH can return its signal-resolution error at the deadline; the shell is reset and the namespace reaps all task processes on exit.'})
    print(checks, flush=True)


def real(case):
    service = None
    config = ROOT / 'configs/gym-qwen-server-candidate.json'
    allocation = ROOT / 'configs/gym-gpu-allocation.json'
    url = 'http://127.0.0.1:18081/v1'
    launcher = ['bash', str(ROOT / 'scripts/start_qwen.sh'), '--config', str(config), '--allocation', str(allocation)]
    stop_monitor = threading.Event()
    def monitor():
        while not stop_monitor.wait(3):
            gpu_snapshot(case / 'gpu-memory.jsonl')
    thread = threading.Thread(target=monitor, daemon=True)
    status = {'real_model':True,'passed':False}
    try:
        with (case/'preflight.log').open('w') as f:
            subprocess.run(launcher+['--check-only'],stdout=f,stderr=subprocess.STDOUT,check=True)
        thread.start()
        with (case/'server.log').open('w') as f:
            service = subprocess.Popen(launcher,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+720
        while time.monotonic()<deadline:
            if service.poll() is not None:
                raise RuntimeError('Service exited before readiness')
            try:
                req=urllib.request.Request(url+'/models',headers={'Authorization':'Bearer local-qwen'})
                with urllib.request.urlopen(req,timeout=2) as response:
                    assert response.status==200
                break
            except (OSError,ValueError):
                time.sleep(2)
        else:
            raise TimeoutError('Service readiness timeout')
        body={'model':'Qwen3.5-27B','messages':[{'role':'user','content':'Ignore the filler below and respond exactly LONG_CONTEXT_OK.\n'+('filler ' * 26000)+'\nRespond exactly LONG_CONTEXT_OK.'}],
              'max_tokens':32,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
        write(case/'long-context-request.json',body)
        req=urllib.request.Request(url+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer local-qwen','Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=240) as response:
            result=json.load(response)
        write(case/'long-context-response.json',result)
        long_ok=(24576<=result['usage']['prompt_tokens']<32768 and result['choices'][0]['message']['content'].strip()=='LONG_CONTEXT_OK')
        work=case/'sandbox/workspace'
        prompt=(f'Work in {work}. First use str_replace_editor to view stress.txt (it is deliberately large), then use the indicated read_output command to read a later page. '
                'Next run python -m pytest -q test_repair.py to observe the failing test. '
                'Use str_replace_editor to repair add_one in repair.py so it adds one. Run the same test again. '
                'Do not edit tests. End with FACILITY_OK only if the test passes.')
        worker=execute(case,url,prompt)
        p=subprocess.run([str(ROOT/'.venv-swe/bin/python'),'-m','pytest','-q','test_repair.py'],cwd=work,capture_output=True,text=True,timeout=30)
        write(case/'independent-tests.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        status.update(long_context_verified=long_ok,worker=worker,independent_test_passed=p.returncode==0,
                      passed=long_ok and p.returncode==0 and worker['status']=='completed')
    except BaseException as exc:
        status['error']=repr(exc)
        raise
    finally:
        if service:
            stop_owned_group(service)
        stop_monitor.set()
        if thread.is_alive():
            thread.join(timeout=10)
        status['after_cleanup']=gpu_snapshot(case/'gpu-memory.jsonl')
        status['service_returncode']=service.returncode if service else None
        write(case/'verification.json',status)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--real',action='store_true')
    args=parser.parse_args()
    case=ROOT/'runtime/gym-baseline/facility'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+('real' if args.real else 'cpu')+'-'+secrets.token_hex(3))
    prepare(case)
    print(case,flush=True)
    (real if args.real else mock)(case)


if __name__=='__main__':
    main()
