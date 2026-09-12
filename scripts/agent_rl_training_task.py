"""Project-local training tasks; deliberately not an official benchmark runner.

Reuses the existing process sandbox only. No gym benchmark controller, custom
benchmark task selector, model launcher or official score reporting is invoked.
"""
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import time
import xml.etree.ElementTree as ET

from gym_task_env import task_uid

ROOT = Path(__file__).resolve().parents[1]
ISOLATOR = ROOT / 'scripts/gym_task_env.py'
DSH_PYTHON = ROOT / '.venv-dsh/bin/python'
PROTOCOL = 'custom-training-pytest-v1'


class TrainingEnvironmentError(RuntimeError):
    pass


class CandidateRejected(TrainingEnvironmentError):
    """A completed candidate violates the declared training edit policy."""


def project_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    target = path.resolve()
    if not path.is_relative_to(ROOT) or not target.is_relative_to(ROOT) or target == ROOT:
        raise ValueError('Training artifacts and dependencies must stay in the project')
    # Keep a virtual environment's interpreter path: resolving bin/python to
    # the base interpreter would silently discard its site-packages.
    return path


def write_json(path, value):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def tree_hash(path):
    h = hashlib.sha256()
    for p in sorted(path.rglob('*')):
        if '.git' in p.relative_to(path).parts or '__pycache__' in p.parts:
            continue
        if p.is_symlink():
            raise TrainingEnvironmentError('Symlinks require a separately reviewed source policy')
        if p.is_file():
            h.update(str(p.relative_to(path)).encode() + b'\0')
            with p.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            h.update(b'\0')
    return h.hexdigest()


def git(work, *args, check=True):
    config = work.parent / '.operator-gitconfig'
    config.write_text('[safe]\n\tdirectory = ' + str(work) + '\n')
    return subprocess.run(['git', '-C', str(work), *args], check=check,
        env=os.environ | {'GIT_CONFIG_GLOBAL': str(config), 'GIT_CONFIG_NOSYSTEM': '1'},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)


def initialize_git(work):
    git(work, 'init', '-q')
    git(work, 'add', '-A')
    git(work, '-c', 'user.name=Training fixture', '-c', 'user.email=training@local.invalid',
        'commit', '-qm', 'Frozen training base')


def own_sandbox(sandbox):
    uid = task_uid(sandbox)
    for parent, dirs, files in os.walk(sandbox):
        os.chown(parent, uid, uid)
        for name in files:
            os.chown(Path(parent) / name, uid, uid, follow_symlinks=False)


def run_isolated(case, command, *, timeout, log_name='process.log', cancel_event=None, memory_bytes=None):
    """Execute with the existing user/PID/network/Landlock sandbox."""
    if memory_bytes is not None:
        command = ['/usr/bin/prlimit', f'--as={int(memory_bytes)}', '--', *command]
    argv = [str(DSH_PYTHON), str(ISOLATOR), '--sandbox', str(case / 'sandbox'), '--', *command]
    with (case / log_name).open('wb') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            deadline = time.monotonic() + timeout
            while child.poll() is None:
                if time.monotonic() >= deadline or (cancel_event and cancel_event.is_set()):
                    timed_out = True
                    break
                try:
                    child.wait(timeout=.1)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # The whole group belongs to this invocation, including sandbox children.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
    if timed_out:
        raise TrainingEnvironmentError('Training verifier timed out; no reward produced')
    return child.returncode


@dataclass
class LocalTrainingTask:
    spec: dict
    spec_path: Path

    @classmethod
    def load(cls, path):
        path = project_path(path)
        spec = json.loads(path.read_text())
        if spec.get('protocol') != PROTOCOL:
            raise TrainingEnvironmentError('Explicit custom training protocol required')
        for key in ('task_id', 'source_dir', 'source_sha256', 'python', 'pytest_args', 'gold_patch'):
            if not spec.get(key):
                raise TrainingEnvironmentError(f'Training task is not prepared: missing {key}')
        project_path(spec['python'])
        if tree_hash(project_path(spec['source_dir'])) != spec['source_sha256']:
            raise TrainingEnvironmentError('Training source snapshot changed')
        if spec.get('dependency_lock'):
            canonical = lambda name: re.sub(r'[-_.]+', '-', name).lower()
            expected = {}
            for line in project_path(spec['dependency_lock']).read_text().splitlines():
                if line.strip() and not line.startswith('#'):
                    name, version = line.split('==')
                    expected[canonical(name)] = version
            probe = subprocess.run([str(project_path(spec['python'])), '-I', '-c',
                'import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'],
                capture_output=True, text=True, check=True, timeout=10, cwd=ROOT)
            installed = {canonical(k): v for k, v in json.loads(probe.stdout).items()}
            if installed != expected:
                raise TrainingEnvironmentError('Installed task dependencies differ from the frozen lock')
        return cls(spec, path)

    @property
    def fingerprint(self):
        payload = dict(self.spec)
        payload.pop('qualification', None)
        for name in ('test_patch', 'gold_patch', 'dependency_lock'):
            if payload.get(name):
                payload[name + '_sha256'] = hashlib.sha256(project_path(payload[name]).read_bytes()).hexdigest()
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def require_qualified(self):
        report_path = self.spec.get('qualification')
        if not report_path:
            raise TrainingEnvironmentError('Task lacks buggy/gold qualification')
        report = json.loads(project_path(report_path).read_text())
        if not report.get('qualified') or report.get('fingerprint') != self.fingerprint:
            raise TrainingEnvironmentError('Task qualification missing or stale')
        return report

    def prepare(self, case, *, patch=None, with_tests=False):
        if tree_hash(project_path(self.spec['source_dir'])) != self.spec['source_sha256']:
            raise TrainingEnvironmentError('Training source changed before task preparation')
        case = project_path(case)
        case.mkdir(parents=True, exist_ok=False)
        sandbox = case / 'sandbox'
        work = sandbox / 'workspace'
        shutil.copytree(project_path(self.spec['source_dir']), work,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))
        initialize_git(work)
        for name in ('home', 'tmp', 'cache', 'dsh-home'):
            (sandbox / name).mkdir()
        if patch and project_path(patch).stat().st_size:
            git(work, 'apply', '--', str(project_path(patch)))
        if with_tests and self.spec.get('test_patch'):
            git(work, 'apply', '--', str(project_path(self.spec['test_patch'])))
        write_json(case / 'runtime-config.json', {'task_python': str(project_path(self.spec['python'])),
                    'task_env': self.spec.get('task_env', {})})
        own_sandbox(sandbox)
        return case

    def export_patch(self, agent_case, destination):
        """Diff a fresh trusted Git index; never execute the agent's Git config."""
        agent_work = agent_case / 'sandbox/workspace'
        if agent_work.is_symlink() or not agent_work.is_dir():
            raise CandidateRejected('Candidate workspace was replaced')
        destination = project_path(destination)
        reference = destination.parent / 'trusted-export'
        shutil.copytree(project_path(self.spec['source_dir']), reference,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))
        initialize_git(reference)
        for p in list(reference.iterdir()):
            if p.name == '.git':
                continue
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        # No symlink can redirect the evaluator or copy a private source.
        for p in agent_work.rglob('*'):
            if '.git' not in p.relative_to(agent_work).parts:
                mode = p.lstat().st_mode
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise CandidateRejected('Only regular candidate files/directories are supported')
        shutil.copytree(agent_work, reference, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))
        git(reference, 'add', '-A')
        changed = git(reference, 'diff', '--cached', '--name-only', '-z').stdout.decode().split('\0')
        for name in filter(None, changed):
            if any(fnmatch.fnmatch(name, pattern) for pattern in self.spec.get('protected_paths', [])):
                raise CandidateRejected(f'Protected training test/config modified: {name}')
        destination.write_bytes(git(reference, 'diff', '--cached', '--binary', 'HEAD').stdout)
        return destination

    def evaluate(self, case, *, patch=None, cancel_event=None):
        case = self.prepare(case, patch=patch, with_tests=True)
        report_path = case / 'sandbox/pytest.xml'
        code = run_isolated(case, [str(project_path(self.spec['python'])), '-m', 'pytest',
            *self.spec['pytest_args'], '--junitxml=' + str(report_path)],
            timeout=self.spec.get('test_timeout_seconds', 120), cancel_event=cancel_event,
            memory_bytes=self.spec.get('test_memory_bytes'))
        if code not in (0, 1) or not report_path.is_file():
            raise TrainingEnvironmentError(f'Test execution/collection failed ({code}); no reward')
        xml = ET.parse(report_path).getroot()
        tests = list(xml.iter('testcase'))
        if not tests or any(t.find('error') is not None for t in tests):
            raise TrainingEnvironmentError('No tests or test setup error; no reward')
        executed = [t for t in tests if t.find('skipped') is None]
        if not executed:
            raise TrainingEnvironmentError('All tests skipped; no reward')
        passed = code == 0 and all(t.find('failure') is None for t in tests)
        result = {'protocol': PROTOCOL, 'official_benchmark_result': False,
                  'task_id': self.spec['task_id'], 'reward': int(passed),
                  'test_ids': sorted(t.get('classname', '') + '::' + t.get('name', '') for t in executed),
                  'failed_test_ids': sorted(t.get('classname', '') + '::' + t.get('name', '') for t in executed if t.find('failure') is not None),
                  'skipped_test_ids': sorted(t.get('classname', '') + '::' + t.get('name', '') for t in tests if t.find('skipped') is not None),
                  'returncode': code, 'report': str(report_path), 'fingerprint': self.fingerprint}
        write_json(case / 'verification.json', result)
        return result

    def qualify(self, directory):
        directory = project_path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        buggy = self.evaluate(directory / 'buggy')
        gold = self.evaluate(directory / 'gold', patch=self.spec['gold_patch'])
        report = {'protocol': PROTOCOL, 'official_benchmark_result': False,
                  'qualified': buggy['reward'] == 0 and gold['reward'] == 1 and buggy['test_ids'] == gold['test_ids'],
                  'fingerprint': self.fingerprint, 'buggy': buggy, 'gold': gold}
        if self.spec.get('expected_fail_to_pass'):
            expected = []
            for node in self.spec['expected_fail_to_pass']:
                path, *parts = node.split('::')
                classname = path.removesuffix('.py').replace('/', '.')
                if len(parts) > 1:
                    classname += '.' + '.'.join(parts[:-1])
                expected.append(classname + '::' + parts[-1])
            report['expected_failures'] = sorted(expected)
            report['qualified'] = report['qualified'] and buggy['failed_test_ids'] == sorted(expected)
        write_json(directory / 'qualification.json', report)
        return report
