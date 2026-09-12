"""CPU-only scripted transport check; never reported as model performance."""
import dataclasses
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import shlex
import socketserver
import threading
import time

from deepseek_harness import DeepSeekHarness
from swe_task_env import ROOT, PY, NATIVE


def main():
    case = ROOT / 'runtime/swe-smoke/isolation-check'
    sandbox = case / 'sandbox'
    sock = ROOT / 'tmp/swe-isolation.sock'
    if sock.exists():
        sock.unlink()
    workspace = sandbox / 'workspace'
    commands = [
        ('bash', {'command': 'pwd; printf BEFORE > isolation.txt; cat isolation.txt; '
                            f'cat {ROOT}/configs/swe-three-tasks.json; '
                            "python -c \"import socket; socket.socket(socket.AF_UNIX)\""}),
        ('str_replace_editor', {'command': 'view', 'path': str(workspace / 'isolation.txt')}),
        ('str_replace_editor', {'command': 'str_replace', 'path': str(workspace / 'isolation.txt'),
                               'old_str': 'BEFORE', 'new_str': 'AFTER'}),
        ('bash', {'command': 'cat isolation.txt; python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect((\'1.1.1.1\',443))"'}),
    ]
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            n = len(requests) - 1
            if n < len(commands):
                name, args = commands[n]
                delta = {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': f'check{n}',
                         'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
                finish = 'tool_calls'
            else:
                delta = {'role': 'assistant', 'content': 'CPU_ISOLATION_CHECK'}
                finish = 'stop'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for d, f in [(delta, None), ({}, finish)]:
                chunk = {'id': f'mock-{n}', 'object': 'chat.completion.chunk', 'created': int(time.time()),
                         'model': 'Qwen3.5-27B', 'choices': [{'index': 0, 'delta': d, 'finish_reason': f}]}
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
    server = socketserver.ThreadingUnixStreamServer(str(sock), Handler)
    os.chmod(sock, 0o666)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    wrapper = case / 'dsh-wrapper.sh'
    argv = [str(PY), str(ROOT / 'scripts/swe_task_env.py'), '--sandbox', str(sandbox),
            '--socket', str(sock), '--', str(NATIVE)]
    wrapper.write_text('#!/bin/bash\nexec ' + shlex.join(argv) + ' "$@"\n')
    wrapper.chmod(0o755)
    result = {'real_model': False}
    try:
        h = DeepSeekHarness(dsh_bin=str(wrapper), provider='qwen-local', model='Qwen3.5-27B',
                cwd=str(workspace), dsh_home=str(sandbox / 'dsh-home'), profile='sdk-minimal',
                patches=(str(case / 'qwen.patch.yml'),), max_tokens=256,
                initialize_timeout_seconds=60, request_timeout_seconds=90, shutdown_timeout_seconds=3)
        with h:
            answer = h.run('Exercise the configured tools.', session_id='cpu-isolation')
            result['answer'] = dataclasses.asdict(answer)
        result['stderr'] = list(h.client._stderr_lines)
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        server.shutdown()
        server.server_close()
        sock.unlink(missing_ok=True)
        result['requests'] = requests
        (case / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['answer', 'requests']}, indent=2))
    print('requests', len(requests))


if __name__ == '__main__':
    main()
