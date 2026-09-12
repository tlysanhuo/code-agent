"""Recheck preserved SWE run evidence without invoking a model or test suite."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from swe_prepare import SELECTION, base_for, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / 'summary.json').read_text())
    tasks = {t['instance_id']: t for t in json.loads(SELECTION.read_text())['tasks']}
    cases = []
    for case in sorted(run.glob('task-*')):
        config = json.loads((case / 'case.json').read_text())
        task = tasks[config['instance_id']]
        result = json.loads((case / 'evaluation.json').read_text())
        requests = [json.loads(s) for s in (case / 'requests.jsonl').read_text().splitlines()]
        response_checks = []
        for response in sorted(case.glob('response-*.sse')):
            records = [json.loads(s[6:]) for s in response.read_text().splitlines()
                       if s.startswith('data: ') and s != 'data: [DONE]']
            usage = [r['usage'] for r in records if r.get('usage')]
            response_checks.append({'file': response.name,
                                    'done': 'data: [DONE]' in response.read_text(),
                                    'usage_records': len(usage),
                                    'completion_tokens': usage[-1]['completion_tokens'] if usage else None,
                                    'prompt_tokens': usage[-1]['prompt_tokens'] if usage else None})
        tests = {n.split('::')[0] for n in task['FAIL_TO_PASS'] + task['PASS_TO_PASS']}
        independent = Path(result.get('independent_dir', str(case / 'independent'))) / 'sandbox/workspace'
        test_hashes = {n: hashlib.sha256((independent / n).read_bytes()).hexdigest() ==
                         hashlib.sha256((base_for(task) / n).read_bytes()).hexdigest() for n in tests}
        process = json.loads((case / 'process.json').read_text())
        checks = {
            'original_problem_preserved': task['problem_statement'] in config['prompt'],
            'first_request_contains_only_expected_user_prompt': any(
                m.get('role') == 'user' and config['prompt'] in str(m.get('content'))
                for m in requests[0]['body']['messages']),
            'max_30_upstream_requests': 0 < len(response_checks) <= 30,
            'output_limits_requested': all(0 < r['body'].get('max_tokens', 0) <= 512 for r in requests),
            'all_streams_complete_with_usage': all(r['done'] and r['usage_records'] == 1 for r in response_checks),
            'actual_outputs_within_limits': all(r['completion_tokens'] is not None and r['completion_tokens'] <= 512
                                              for r in response_checks),
            'context_within_8192': all(r['prompt_tokens'] is not None and r['prompt_tokens'] + r['completion_tokens'] <= 8192
                                     for r in response_checks),
            'within_10_minutes': process['wall_seconds'] <= 600 and not process['timeout'],
            'independent_oracle_unchanged': all(test_hashes.values()),
            'patch_hash_matches': hashlib.sha256((case / 'agent.patch').read_bytes()).hexdigest() == result['patch_sha256'],
            'independent_pipeline_complete': result['pipeline_complete'],
        }
        cases.append({'instance_id': task['instance_id'], 'checks': checks, 'passed': all(checks.values()),
                      'streams': response_checks, 'oracle_hash_checks': test_hashes})
    allocation = json.loads((run / 'gpu-allocation.json').read_text())
    peak = {uuid: 0 for uuid in allocation['gpu_uuids']}
    for line in (run / 'gpu-memory.jsonl').read_text().splitlines():
        sample = json.loads(line)
        for row in csv.reader(sample['gpus'].splitlines()):
            uuid = row[1].strip()
            if uuid in peak:
                peak[uuid] = max(peak[uuid], int(row[2].strip()))
    audit = {'passed': all(c['passed'] for c in cases) and summary['cleanup_verified'],
             'cases': cases, 'sampled_peak_gpu_mib': peak, 'cleanup_verified': summary['cleanup_verified'],
             'selection_hash_unchanged': summary['selection_sha256'] == hashlib.sha256(SELECTION.read_bytes()).hexdigest()}
    audit['passed'] &= audit['selection_hash_unchanged']
    write(run / 'evidence-audit.json', audit)
    print(json.dumps(audit, indent=2))
    if not audit['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
