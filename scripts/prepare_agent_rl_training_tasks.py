"""Prepare frozen, audited training/development tasks in restricted processes.

Reuses a separately installed generic dependency lock. Keeps every requested
task's qualification result; never reports this as official SWE-Gym evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from agent_rl_training_task import ROOT, LocalTrainingTask, tree_hash, write_json


def export_qualified():
    """Derive a policy-visible table from the frozen, exclusion-audited pool."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = ROOT / 'data/agent-rl'
    manifest = json.loads((data / 'manifest.json').read_text())
    assert hashlib.sha256((data / 'train.parquet').read_bytes()).hexdigest() == manifest['output_sha256']
    audit = json.loads((data / 'exclusion-audit.json').read_text())
    for record in audit['source_hashes']:
        with (ROOT / record['path']).open('rb') as source:
            assert hashlib.file_digest(source, 'sha256').hexdigest() == record['sha256'], record['path']
    registry = json.loads((ROOT / 'configs/agent-rl/training-tasks.json').read_text())
    candidates = {r['extra_info']['instance_id']: r for r in pq.read_table(data / 'train.parquet').to_pylist()}
    expansion_path = data / 'expansion-v1/manifest.json'
    if expansion_path.exists():
        expansion = json.loads(expansion_path.read_text())
        table = data / 'expansion-v1/candidates.parquet'
        assert hashlib.sha256(table.read_bytes()).hexdigest() == expansion['candidate_table_sha256']
        candidates.update({r['extra_info']['instance_id']: r for r in pq.read_table(table).to_pylist()})
    holdout = json.loads((ROOT / 'configs/agent-rl/development-holdout.json').read_text())
    held_ids = {t['instance_id'] for t in holdout['tasks']}
    held_commits = {(t['repo'], t['base_commit']) for t in holdout['tasks']}
    records, tasks, dev_records, dev_tasks = [], [], [], []
    for row in candidates.values():
        ident = row['extra_info']['instance_id']
        entry = registry['tasks'][ident]
        if entry['state'] != 'qualified':
            continue
        assert ident not in audit['excluded_ids']
        assert [row['extra_info']['repo'], row['extra_info']['base_commit']] not in audit['excluded_repo_commits']
        task = LocalTrainingTask.load(entry['task_spec'])
        qualification = task.require_qualified()
        row['extra_info'].update(environment_qualified=True,
            environment_ref=str(task.spec_path), verifier_ref=task.spec['qualification'],
            training_protocol=task.spec['protocol'], task_fingerprint=task.fingerprint)
        development = ident in held_ids or (row['extra_info']['repo'], row['extra_info']['base_commit']) in held_commits
        row['extra_info']['split'] = 'development' if development else 'train'
        (dev_records if development else records).append(row)
        (dev_tasks if development else tasks).append({'instance_id': ident, 'repo': row['extra_info']['repo'], 'fingerprint': task.fingerprint,
            'qualification': task.spec['qualification'],
            'executed_tests': len(qualification['gold']['test_ids']),
            'skipped_tests': len(qualification['gold']['skipped_test_ids'])})
    if not records:
        raise ValueError('No qualified tasks; no ready table exported')
    pq.write_table(pa.Table.from_pylist(records), data / 'train-ready.parquet')
    if dev_records:
        pq.write_table(pa.Table.from_pylist(dev_records), data / 'development-ready.parquet')
    write_json(data / 'train-ready-manifest.json', {
        'protocol': 'custom-training-pytest-v1', 'official_benchmark_result': False,
        'source_manifest': 'data/agent-rl/manifest.json', 'source_revision': manifest['revision'],
        'source_table_sha256': manifest['output_sha256'],
        'exclusion_audit_sha256': hashlib.sha256((data / 'exclusion-audit.json').read_bytes()).hexdigest(),
        'selection': 'Qualified original and frozen expansion tasks, excluding reserved development IDs/commits; no model outcomes',
        'expansion_manifest': str(expansion_path),
        'development_holdout_sha256': hashlib.sha256((ROOT / 'configs/agent-rl/development-holdout.json').read_bytes()).hexdigest(),
        'tasks': tasks, 'rows': len(records),
        'output_sha256': hashlib.sha256((data / 'train-ready.parquet').read_bytes()).hexdigest(),
        'real_model_rollouts': 0, 'parameter_updates': 0})
    write_json(data / 'development-ready-manifest.json', {
        'protocol': 'custom-training-pytest-v1', 'official_benchmark_result': False,
        'tasks': dev_tasks, 'rows': len(dev_records),
        'output_sha256': hashlib.sha256((data / 'development-ready.parquet').read_bytes()).hexdigest() if dev_records else None,
        'selection': 'Development tasks reserved before qualification; not automatically enabled for model validation',
        'real_model_rollouts': 0})
    return tasks


def prepare(instance_id, *, qualify):
    registry = json.loads((ROOT / 'configs/agent-rl/training-tasks.json').read_text())
    entry = registry['tasks'][instance_id]
    manifest = json.loads(Path(entry.get('source_manifest', ROOT / 'data/agent-rl/manifest.json')).read_text())
    identities = {t['instance_id']: t for t in manifest['tasks']}
    if instance_id not in identities:
        raise ValueError('Task must belong to the already audited training integration pool')
    source_record = Path(entry.get('private_record', ROOT / 'data/agent-rl/private' / (instance_id + '.json')))
    raw = source_record.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == identities[instance_id]['private_sha256']
    row = json.loads(raw)
    recipes = json.loads((ROOT / 'configs/agent-rl/task-recipes.json').read_text())
    recipe = recipes.get(row['repo'] + '@' + row['version'], recipes[row['repo']])
    base = ROOT / 'runtime/agent-rl/training-environments' / instance_id
    base.mkdir(exist_ok=True)
    source = base / 'source'
    archive = base / 'source.tar.gz'
    url = f"https://codeload.github.com/{row['repo']}/tar.gz/{row['base_commit']}"
    if not source.exists():
        if not archive.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'code-agent-training-preparation'})
            partial = archive.with_suffix('.gz.part')
            with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as out:
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > 100 * 1024 * 1024:
                        raise ValueError('Unexpected source archive size')
                    out.write(block)
            partial.rename(archive)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if sum(m.size for m in members) > 512 * 1024 * 1024:
                raise ValueError('Source archive exceeds bounded extraction size')
            prefix = members[0].name.split('/')[0]
            if any(m.issym() or m.islnk() or not (m.isfile() or m.isdir()) for m in members):
                raise ValueError('Source archive contains unsupported entries')
            tar.extractall(base / 'extracted', filter='data')
        (base / 'extracted' / prefix).rename(source)
    (base / 'gold.patch').write_text(row['patch'])
    (base / 'tests.patch').write_text(row['test_patch'])
    nodes = list(dict.fromkeys(row['FAIL_TO_PASS'] + row['PASS_TO_PASS']))
    spec = {'task_id': instance_id, 'protocol': 'custom-training-pytest-v1',
        'repo': row['repo'], 'base_commit': row['base_commit'], 'source_dir': str(source),
        'source_sha256': tree_hash(source), 'python': str(ROOT / recipe['python']),
        'pytest_args': [*recipe['pytest_options'], *nodes, '-q'],
        'gold_patch': str(base / 'gold.patch'),
        'test_patch': str(base / 'tests.patch'), 'test_timeout_seconds': 120, 'test_memory_bytes': 3 * 1024**3,
        'task_env': recipe['task_env'],
        'expected_fail_to_pass': row['FAIL_TO_PASS'],
        'protected_paths': ['tests/*', '*/tests/*', '*test_*.py', '*conftest.py', 'pytest.ini', 'pyproject.toml', 'setup.cfg'],
        'dependency_lock': str(ROOT / recipe['dependency_lock']),
        'test_selection_source': 'Verbatim union of dataset FAIL_TO_PASS/PASS_TO_PASS; custom local environment',
        'official_benchmark_result': False}
    spec_path = base / 'task.json'
    write_json(spec_path, spec)
    write_json(base / 'source.json', {'url': url, 'revision': row['base_commit'],
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'source_tree_sha256': spec['source_sha256'], 'private_record_sha256': hashlib.sha256(raw).hexdigest(),
        'requested_test_ids': len(nodes), 'training_only': True})
    result = {'task_id': instance_id, 'prepared': True, 'qualified': False, 'spec': str(spec_path)}
    if qualify:
        directory = base / ('qualification-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
        try:
            report = LocalTrainingTask.load(spec_path).qualify(directory)
            result['qualified'] = report['qualified']
            result['qualification'] = str(directory / 'qualification.json')
            if report['qualified']:
                spec['qualification'] = result['qualification']
                write_json(spec_path, spec)
        except Exception as exc:
            result['error'] = repr(exc)
            write_json(directory / 'failure.json', result)
    write_json(base / 'preparation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--export-qualified', action='store_true')
    parser.add_argument('--frozen-expansion', action='store_true')
    args = parser.parse_args()
    registry_path = ROOT / 'configs/agent-rl/training-tasks.json'
    registry = json.loads(registry_path.read_text())
    if args.frozen_expansion:
        frozen = json.loads((ROOT / 'data/agent-rl/expansion-v1/manifest.json').read_text())
        args.task += [t['instance_id'] for t in frozen['tasks']]
    results = []
    for task_id in args.task:
        try:
            result = prepare(task_id, qualify=args.qualify)
        except Exception as exc:
            result = {'task_id': task_id, 'prepared': False, 'qualified': False, 'error': repr(exc)}
            failure = ROOT / 'runtime/agent-rl/training-environments' / task_id / ('preparation-failure-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json')
            write_json(failure, result)
        registry['tasks'][task_id].update(task_spec=result.get('spec'), state='qualified' if result['qualified'] else 'prepared_unqualified')
        write_json(registry_path, registry)
        results.append(result)
        print(json.dumps(result), flush=True)
    if results:
        write_json(ROOT / 'runtime/agent-rl/training-environments/preparation.json', results)
    if args.export_qualified:
        print(json.dumps({'exported': export_qualified()}), flush=True)
    return 0 if all(r['qualified'] for r in results) or not args.qualify else 1


if __name__ == '__main__':
    raise SystemExit(main())
