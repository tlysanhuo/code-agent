"""Recorded OpenAI relay with deterministic paged tool views and hard budgets."""
import hashlib
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import socketserver
import threading
import time
import urllib.error
import urllib.request

POLICY = ('Tool outputs are archived in full. Model-visible outputs have deterministic byte limits. '
          'When a result is paged, use bash: read_output OUTPUT_ID BYTE_OFFSET. '
          'Use str_replace_editor view_range for targeted file lines; use find/grep and sed for narrow reads. '
          'Displayed ./ paths refer to the workspace; str_replace_editor requires an absolute path. '
          'Existing tests may be run; hidden acceptance tests are not available. Do not modify existing tests.')


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def utf8_head(text, budget):
    return text.encode()[:max(0, budget)].decode('utf-8', errors='ignore')


def tool_views(body, case, per_tool=4096, all_tools=16384):
    cooked = json.loads(json.dumps(body))
    messages = [m for m in cooked['messages'] if m.get('role') == 'tool']
    budget = min(per_tool, all_tools // max(1, len(messages)))
    artifacts = case / 'sandbox/artifacts'
    artifacts.mkdir(exist_ok=True)
    private = case / 'raw-tools'
    private.mkdir(exist_ok=True)
    workspace = str(case / 'sandbox/workspace')
    for message in messages:
        raw = message['content']
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False)
        call_id = message['tool_call_id']
        key = hashlib.sha256((call_id + '\0' + raw).encode()).hexdigest()[:24]
        raw_path = private / (key + '.txt')
        if not raw_path.exists():
            raw_path.write_text(raw)
        normalized = raw.replace(workspace + '/', './').replace(workspace, '.')
        page_path = artifacts / (key + '.txt')
        if not page_path.exists():
            page_path.write_text(normalized)
        if len(normalized.encode()) <= budget:
            view = normalized
        else:
            head = utf8_head(normalized, budget - 240)
            offset = len(head.encode())
            suffix = f'\n[PAGED output={key}; bytes shown=0:{offset}/{len(normalized.encode())}; next: read_output {key} {offset}]'
            view = head + suffix
        assert len(view.encode()) <= budget
        message['content'] = view
        append(case / 'tool-view-mapping.jsonl', {'call_id': call_id, 'output_id': key,
               'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'raw_bytes': len(raw.encode()),
               'model_view_sha256': hashlib.sha256(view.encode()).hexdigest(), 'model_view_bytes': len(view.encode()),
               'raw_path': str(raw_path), 'page_path': str(page_path)})
    if cooked['messages'] and cooked['messages'][0]['role'] == 'system':
        cooked['messages'][0]['content'] += '\n\n' + POLICY
    else:
        cooked['messages'].insert(0, {'role': 'system', 'content': POLICY})
    return cooked


class Relay(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    def __init__(self, path, case, config, server):
        self.case, self.config, self.upstream = case, config, server
        self.rounds = self.output_tokens = self.charged_output_tokens = 0
        self.lock = threading.Lock()
        self.deadline = time.monotonic() + config['timeout_seconds']
        super().__init__(str(path), Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        relay = self.server
        received = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        with relay.lock:
            relay.rounds += 1
            number = relay.rounds
            remaining = relay.config['max_output_tokens_total'] - relay.charged_output_tokens
        append(relay.case / 'dsh-requests.jsonl', {'request': number, 'time': time.time(), 'body': received})
        if (self.path != '/v1/chat/completions' or number > relay.config['max_rounds']
                or remaining <= 0 or time.monotonic() >= relay.deadline):
            append(relay.case / 'limits.jsonl', {'request': number, 'reason': 'budget_exhausted'})
            self.send_error(429, 'Frozen task budget exhausted')
            return
        body = tool_views(received, relay.case)
        # pi-ai estimates against DSH's unpaged internal history and may reduce
        # max_tokens to 1. Enforce the frozen budget on the actual paged request.
        body.update(max_tokens=min(remaining, relay.config.get('max_output_tokens_per_round',2048)),
                    temperature=0, top_p=1, seed=0, presence_penalty=0, frequency_penalty=0)
        with relay.lock:
            relay.charged_output_tokens += body['max_tokens']
        append(relay.case / 'model-requests.jsonl', {'request': number, 'time': time.time(), 'body': body})
        request = urllib.request.Request(relay.upstream + '/chat/completions', data=json.dumps(body).encode(),
                  headers={'Authorization': 'Bearer local-qwen', 'Content-Type': 'application/json'})
        usage = None
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=min(180, max(1, relay.deadline-time.monotonic()))) as response:
                self.send_response(response.status)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                with (relay.case / f'response-{number:02d}.sse').open('wb') as saved:
                    while time.monotonic() < relay.deadline:
                        line = response.readline()
                        if not line:
                            break
                        saved.write(line)
                        saved.flush()
                        if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                            chunk = json.loads(line[6:])
                            if chunk.get('usage'):
                                usage = chunk['usage']
                                if usage['completion_tokens']>body['max_tokens']:
                                    raise RuntimeError('Upstream exceeded the frozen per-request output budget')
                                with relay.lock:
                                    relay.output_tokens += usage['completion_tokens']
                        self.wfile.write(line)
                        self.wfile.flush()
                        if line.strip() == b'data: [DONE]':
                            break
        except Exception as exc:
            detail = {'request': number, 'error': repr(exc)}
            if isinstance(exc, urllib.error.HTTPError):
                detail['body'] = exc.read().decode(errors='replace')
            append(relay.case / 'transport-errors.jsonl', detail)
            self.close_connection = True
        finally:
            if usage is not None:
                with relay.lock:
                    relay.charged_output_tokens -= body['max_tokens'] - usage['completion_tokens']
            append(relay.case / 'request-costs.jsonl', {'request': number, 'elapsed_seconds': time.monotonic()-started,
                                                      'usage': usage, 'cumulative_output_tokens': relay.output_tokens,
                                                      'charged_output_tokens': relay.charged_output_tokens})
