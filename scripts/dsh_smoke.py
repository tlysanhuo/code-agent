"""DSH startup, mock SSE, or real localhost connectivity; always close runtime."""
import argparse
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from deepseek_harness import DeepSeekHarness

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--startup-only', action='store_true')
    group.add_argument('--mock', action='store_true')
    group.add_argument('--real', action='store_true')
    args = parser.parse_args()
    mode = 'startup' if args.startup_only else 'mock' if args.mock else 'real'
    workspace = ROOT / 'runtime/smoke-workspace'
    workspace.mkdir(exist_ok=True)
    base_url = os.environ.get('QWEN_BASE_URL', 'http://127.0.0.1:18080/v1')
    if urlparse(base_url).hostname not in {'127.0.0.1', 'localhost'}:
        raise ValueError('This probe only connects to localhost')
    server = None
    thread = None
    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests_seen.append({'path': self.path, 'body': payload})
            assert self.path == '/v1/chat/completions'
            assert payload['model'] == 'Qwen3.5-27B' and payload['stream'] is True
            assert not payload.get('tools')
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for delta, finish in [({'role': 'assistant', 'content': 'DSH_'}, None),
                                  ({'content': 'MOCK_OK'}, None), ({}, 'stop')]:
                chunk = {'id': 'mock-local', 'object': 'chat.completion.chunk',
                         'created': int(time.time()), 'model': 'Qwen3.5-27B',
                         'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
                self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')

    result_doc = {'mode': mode, 'real_qwen_verified': False, 'status': 'failed'}
    harness = None
    try:
        if args.mock:
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f'http://127.0.0.1:{server.server_port}/v1'
        harness = DeepSeekHarness(
            provider='qwen-local', model='Qwen3.5-27B', max_tokens=64,
            cwd=str(workspace), dsh_home=str(ROOT / 'runtime/dsh-home'),
            profile='sdk-minimal',
            patches=(str(ROOT / 'configs/dsh-qwen.patch.yml'),
                     str(ROOT / 'configs/dsh-smoke-no-tools.patch.yml')),
            env={'QWEN_LOCAL_API_KEY': 'local-qwen', 'QWEN_BASE_URL': base_url,
                 'CUDA_VISIBLE_DEVICES': ''},
            initialize_timeout_seconds=90, request_timeout_seconds=90,
            shutdown_timeout_seconds=5,
        )
        with harness:
            result_doc['initialized'] = True
            if not args.startup_only:
                result = harness.run('Reply exactly QWEN_DSH_OK. Do not use tools.',
                                     session_id=f'{mode}-{time.time_ns()}')
                result_doc.update(final_response=result.final_response,
                                  finish_reason=result.finish_reason,
                                  session_id=result.session_id,
                                  event_count=len(result.events))
                (ROOT / f'runtime/metadata/dsh-{mode}-events.json').write_text(
                    json.dumps(result.events, indent=2) + '\n')
                result_doc['text_delta_events'] = sum(
                    event.get('data', {}).get('chunk', {}).get('type') == 'text-delta'
                    for event in result.events)
                assert result.finish_reason == 'completed', result.finish_reason
                assert result.final_response.strip(), 'Empty assistant output'
                if args.mock:
                    assert result.final_response == 'DSH_MOCK_OK'
                    assert len(requests_seen) == 1
                else:
                    assert result.final_response.strip() == 'QWEN_DSH_OK', result.final_response
                    assert result_doc['text_delta_events'] > 0
                    result_doc['real_qwen_verified'] = True
        result_doc['status'] = 'passed'
        result_doc['runtime_closed'] = True
    finally:
        if harness is not None:
            harness.close()
        if server:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        result_doc['checked_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (ROOT / f'runtime/metadata/dsh-{mode}-check.json').write_text(
            json.dumps(result_doc, indent=2) + '\n')
        if requests_seen:
            (ROOT / 'logs/dsh-mock-requests.json').write_text(json.dumps(requests_seen, indent=2)+'\n')
        print(json.dumps(result_doc, indent=2))


if __name__ == '__main__':
    main()
