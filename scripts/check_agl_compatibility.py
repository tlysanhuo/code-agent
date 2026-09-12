"""Replay a captured DSH request against unchanged AGL; never call a model."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
COMMIT = '218f1f7c0bac0800de4d5a4e5e6f61cf7b5038b4'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert subprocess.check_output(['git', '-C', str(ROOT / 'vendor/agent-lightning'),
                                    'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(ROOT / 'vendor/agent-lightning'), 'diff', '--name-only'])
    from fastapi.testclient import TestClient
    from agentlightning.server.app import create_app
    from agentlightning.server.store import _models, _rollouts, _events, _terminal_order
    import torch
    import ray
    assert torch.version.cuda is None, 'This check requires the isolated upstream CPU environment'
    previous = json.loads((ROOT / 'runtime/agent-rl/training-interface-latest.json').read_text())
    request_path, = list(Path(previous['directory']).glob('runs/*/native/request-01-original.json'))
    body = json.loads(request_path.read_text())
    assert body['stream'] is True and body['model'] == 'Qwen3.5-27B'
    for store in (_models, _rollouts, _events, _terminal_order):
        store.clear()
    app = create_app({'key': 'cpu-fixture-key', 'default_proxy': {
        'model_name': 'Qwen3.5-27B', 'include_log_probs': True,
        'train': {'temperature': 1}, 'val': {'temperature': 1}}})
    headers = {'Authorization': 'Bearer cpu-fixture-key'}
    with TestClient(app) as client:
        # The route rejects streaming before making any backend request.
        response = client.post('/api/models', headers=headers, json=[{
            'model': 'Qwen3.5-27B', 'endpoint': 'http://127.0.0.1:9/v1', 'version': 0}])
        response.raise_for_status()
        response = client.post('/api/rollouts', headers=headers,
            json=[{'input': {'scope': 'captured DSH protocol fixture'}, 'is_train': True}])
        response.raise_for_status()
        rid = response.json()[0]['rollout_id']
        response = client.post(f'/proxy/rollout/{rid}/attempt/0/mode/train/openai/v1/chat/completions',
                               headers=headers, json=body)
        assert response.status_code == 400 and 'Streaming responses are not supported' in response.text
    assert not torch.cuda.is_initialized() and not ray.is_initialized()
    report = {
        'checked_utc': datetime.now(timezone.utc).isoformat(), 'agl_commit': COMMIT,
        'check_completed': True, 'dsh_stream_compatible': False,
        'http_status': response.status_code, 'response': response.json(),
        'request_source': str(request_path.relative_to(ROOT)),
        'request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
        'scope': 'Captured real DSH request replay into upstream ASGI app; no DSH worker or backend launched',
        'real_model_requests': 0, 'parameter_updates': 0, 'gpu_initialized': False,
        'resolution': 'Do not add a custom SSE proxy. Revisit whole upstream selection or an upstream-supported DSH non-streaming option before rollout.'}
    out = ROOT / 'runtime/agent-rl/agl-integration/dsh-stream-compatibility.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    return 2  # Expected incompatibility is deliberately not a success exit.


if __name__ == '__main__':
    raise SystemExit(main())
