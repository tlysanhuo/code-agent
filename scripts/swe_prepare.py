"""Frozen SWE-smith source preparation and independent CPU acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from swe_task_env import ROOT, PY, TASK_PY, task_uid

BASE = ROOT / 'runtime/swe-smoke'
SELECTION = ROOT / 'configs/swe-three-tasks.json'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def source_for(task):
    return BASE / 'sources' / task['instance_id']


def base_for(task):
    repo = task['repo'].split('/', 1)[1].rsplit('.', 1)[0]
    return BASE / 'sources' / (repo + '-base')


def full_task(task):
    return json.loads((BASE / 'private' / (task['instance_id'] + '.json')).read_text())


def own(path):
    uid = task_uid(path)
    for base, dirs, files in os.walk(path):
        os.chown(base, uid, uid)
        for name in files:
            p = Path(base) / name
            if not p.is_symlink():
                os.chown(p, uid, uid)


def git(work, *args, **kw):
    # This platform's Git 2.34 safety backport does not honor command-line
    # safe.directory. Use a project-local per-copy global config, scoped to
    # this subprocess only; never change the user's shared Git configuration.
    config = work.parent / '.operator-gitconfig'
    config.write_text('[safe]\n\tdirectory = ' + str(work) + '\n')
    env = os.environ.copy()
    env.update(GIT_CONFIG_GLOBAL=str(config), GIT_CONFIG_NOSYSTEM='1')
    return subprocess.run(['git', '-c', 'safe.directory=' + str(work), '-C', str(work), *args],
                          capture_output=True, text=True, timeout=30, env=env, **kw)


def make_case(case, task, acceptance=False):
    sandbox = case / 'sandbox'
    work = sandbox / 'workspace'
    if work.exists():
        raise FileExistsError(work)
    shutil.copytree(source_for(task), work)
    for n in ['home', 'tmp', 'cache', 'dsh-home']:
        (sandbox / n).mkdir()
    patch = (ROOT / 'configs/dsh-qwen.patch.yml').read_text().replace('maxTokens: 256', 'maxTokens: 512')
    (case / 'qwen.patch.yml').write_text(patch)
    # Only the buggy snapshot enters the single-commit repository. No remote,
    # parent history, mutation patch, task IDs, or hidden tests enter this tree.
    for args in [('init', '-q'), ('add', '-f', '.'),
                 ('-c', 'user.name=Task', '-c', 'user.email=task@local', 'commit', '-qm', 'Buggy task snapshot')]:
        git(work, *args, check=True)
    if acceptance:
        add_acceptance(work, task)
    own(sandbox)
    return sandbox


def add_acceptance(work, task):
    files = sorted({s.split('::', 1)[0] for s in task['FAIL_TO_PASS'] + task['PASS_TO_PASS']})
    for name in files:
        src = base_for(task) / name
        dst = work / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (work / 'acceptance_plugin.py').write_text('''import json
from pathlib import Path
RESULTS = {}
def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when in ("setup", "teardown") and report.failed):
        RESULTS[report.nodeid] = report.outcome
def pytest_sessionfinish(session, exitstatus):
    Path("../test-outcomes.json").write_text(json.dumps({"exitstatus": int(exitstatus), "tests": RESULTS}, indent=2))
''')


def test(case, task, label):
    sandbox = case / 'sandbox'
    nodes = task['FAIL_TO_PASS'] + task['PASS_TO_PASS']
    cmd = [str(PY), str(ROOT / 'scripts/swe_task_env.py'), '--sandbox', str(sandbox), '--',
           str(TASK_PY), '-m', 'pytest', '-q', '--tb=short', '-p', 'acceptance_plugin', *nodes]
    started = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        result = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except subprocess.TimeoutExpired as exc:
        result = {'returncode': None, 'timeout': True, 'stdout': str(exc.stdout), 'stderr': str(exc.stderr)}
    outcomes = sandbox / 'test-outcomes.json'
    result.update(command=cmd, elapsed_seconds=time.monotonic() - started,
                  outcomes=json.loads(outcomes.read_text()) if outcomes.exists() else {})
    if outcomes.exists():
        outcomes.unlink()
    write(case / (label + '.json'), result)
    return result


def validate(task):
    case = BASE / 'env-validation' / task['instance_id']
    if case.exists():
        case = case.with_name(case.name + '-' + str(time.time_ns()))
    sandbox = make_case(case, task, acceptance=True)
    work = sandbox / 'workspace'
    baseline = test(case, task, 'buggy-tests')
    data = full_task(task)
    mutation = case / 'mutation.patch'
    mutation.write_text(data['patch'])
    reverse = git(work, 'apply', '--reverse', str(mutation))
    write(case / 'gold-apply.json', {'returncode': reverse.returncode, 'stdout': reverse.stdout, 'stderr': reverse.stderr})
    gold = test(case, task, 'gold-tests') if reverse.returncode == 0 else {}
    before = baseline.get('outcomes', {}).get('tests', {})
    after = gold.get('outcomes', {}).get('tests', {})
    checks = {
        'all_expected_failures_observed': all(before.get(n) == 'failed' for n in task['FAIL_TO_PASS']),
        'all_expected_passes_preserved': all(before.get(n) == 'passed' for n in task['PASS_TO_PASS']),
        'gold_applied': reverse.returncode == 0,
        'gold_all_nodes_passed': all(after.get(n) == 'passed' for n in task['FAIL_TO_PASS'] + task['PASS_TO_PASS']),
        'gold_exit_zero': gold.get('returncode') == 0,
        'hidden_test_files_absent_from_agent_source': all(not (source_for(task) / n.split('::')[0]).exists()
                                                       for n in task['FAIL_TO_PASS'] + task['PASS_TO_PASS']),
    }
    status = {'instance_id': task['instance_id'], 'checks': checks, 'passed': all(checks.values()), 'evidence_dir': str(case),
              'baseline_seconds': baseline['elapsed_seconds'], 'gold_seconds': gold.get('elapsed_seconds')}
    write(case / 'validation.json', status)
    return status


def extract_bases():
    for f in (BASE / 'sources').glob('*.tar.gz'):
        meta = json.loads(f.with_name(f.name.removesuffix('.tar.gz') + '.json').read_text())
        if hashlib.sha256(f.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Source archive checksum mismatch: ' + str(f))
        dst = f.with_name(f.name.removesuffix('.tar.gz'))
        if dst.exists():
            continue
        dst.mkdir()
        with tarfile.open(f) as archive:
            prefix = archive.getmembers()[0].name.split('/')[0] + '/'
            for member in archive.getmembers():
                if member.name.startswith(prefix):
                    member.name = member.name[len(prefix):]
                    if member.name:
                        archive.extract(member, dst, filter='data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    extract_bases()
    selected = json.loads(SELECTION.read_text())['tasks']
    if args.task:
        selected = [selected[args.task - 1]]
    results = [validate(task) for task in selected]
    destination = BASE / 'environment-checks.json'
    previous = json.loads(destination.read_text()) if destination.exists() else []
    merged = {r['instance_id']: r for r in previous}
    merged.update({r['instance_id']: r for r in results})
    write(destination, list(merged.values()))
    print(json.dumps(results, indent=2))
    if not all(r['passed'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
