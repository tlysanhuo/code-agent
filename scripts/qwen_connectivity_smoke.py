"""Bounded real-GPU connectivity test; always reap the owned service group."""
import json
import os
import secrets
import shutil
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def stop_owned_group(process):
    """Terminate only the process group created by this test."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_connectivity():
    config = json.loads((ROOT / 'configs/qwen-server.json').read_text())
    url = f"http://{config['host']}:{config['port']}/v1/models"
    subprocess.run(['bash', str(ROOT / 'scripts/start_qwen.sh'), '--check-only'], check=True)
    result = {'real_qwen_verified': False, 'status': 'failed'}
    client = None
    with (ROOT / 'logs/qwen-server-smoke.log').open('w') as log:
        process = subprocess.Popen(['bash', str(ROOT / 'scripts/start_qwen.sh')],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 720
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Server exited; inspect logs/qwen-server-smoke.log')
                try:
                    req = urllib.request.Request(url, headers={'Authorization': 'Bearer local-qwen'})
                    with urllib.request.urlopen(req, timeout=3) as response:
                        data = json.load(response)
                    if config['served_model_name'] in [x['id'] for x in data['data']]:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('Server did not become ready within 12 minutes')
            with (ROOT / 'logs/dsh-real-check.log').open('w') as output:
                client = subprocess.Popen([
                    str(ROOT / '.venv-dsh/bin/python'), str(ROOT / 'scripts/dsh_smoke.py'), '--real'],
                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'QWEN_BASE_URL': url.rsplit('/', 1)[0]})
                if client.wait(timeout=210) != 0:
                    raise RuntimeError('DSH request failed; inspect logs/dsh-real-check.log')
            result.update(status='passed', real_qwen_verified=True)
        finally:
            if client is not None:
                stop_owned_group(client)
            stop_owned_group(process)
            result['server_returncode'] = process.returncode
            result['owned_service_stopped'] = True
            result['checked_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            (ROOT / 'runtime/metadata/qwen-real-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def main():
    started = time.time()
    folder = ROOT / 'runtime/connectivity-smoke' / (
        time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + secrets.token_hex(3))
    folder.mkdir(parents=True)
    for name in ['gpu-allocation.json', 'qwen-server.json', 'dsh-qwen.patch.yml']:
        shutil.copy2(ROOT / 'configs' / name, folder / name)
    attempt = {'started_unix': started, 'status': 'failed'}
    try:
        run_connectivity()
        attempt['status'] = 'passed'
    except BaseException as exc:
        attempt['error'] = repr(exc)
        raise
    finally:
        for rel in ['logs/qwen-server-smoke.log', 'logs/dsh-real-check.log',
                    'runtime/metadata/qwen-real-check.json', 'runtime/metadata/dsh-real-check.json',
                    'runtime/metadata/dsh-real-events.json',
                    'runtime/metadata/qwen-service-launch.json']:
            path = ROOT / rel
            if path.exists() and path.stat().st_mtime >= started:
                shutil.copy2(path, folder / path.name)
        attempt['finished_unix'] = time.time()
        (folder / 'attempt.json').write_text(json.dumps(attempt, indent=2)+'\n')
        print('Connectivity evidence:', folder, flush=True)


if __name__ == '__main__':
    main()
