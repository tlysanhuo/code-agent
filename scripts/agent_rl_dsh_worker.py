"""Isolated SDK worker for the training adapter; no model server or grading."""
import dataclasses
import json
from pathlib import Path
import signal
import sys

from deepseek_harness import DeepSeekHarness


def main(path):
    job = json.loads(Path(path).read_text())
    harness = DeepSeekHarness(
        dsh_bin=job['dsh_bin'], provider='qwen-local', model='Qwen3.5-27B',
        cwd=job['workspace'], dsh_home=job['dsh_home'], profile='sdk-minimal',
        patches=(job['patch'],), max_tokens=job['max_tokens'],
        initialize_timeout_seconds=60, request_timeout_seconds=job['timeout'], shutdown_timeout_seconds=3)
    def stop(*_):
        raise KeyboardInterrupt('Owned training session cancelled')
    signal.signal(signal.SIGTERM, stop)
    result = {'completed': False}
    try:
        with harness:
            answer = harness.run(job['prompt'], session_id=job['session_id'])
            result.update(answer=dataclasses.asdict(answer), completed=answer.finish_reason == 'completed')
    except BaseException as exc:
        result['error'] = repr(exc)
    finally:
        harness.close()
        result['runtime_closed'] = harness.client._proc is None or harness.client._proc.poll() is not None
        Path(job['result']).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
