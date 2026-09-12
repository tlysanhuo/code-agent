"""Deterministic pre-model SWE-Gym source/environment validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from swe_prepare import git, own, write
from gym_task_env import ROOT, PY

BASE = ROOT / 'runtime/gym-baseline'
UV = ROOT / 'runtime/tools/uv-package/bin/uv'
BUILD_PY = ROOT / '.venv-gym-build/bin/python'


def task(instance):
    return json.loads((BASE/'private'/(instance+'.json')).read_text())


def requirements(row):
    source=BASE/'sources'/row['instance_id']
    repo=row['repo']
    old_dask=repo=='dask/dask' and row['version'].startswith('2.')
    python='3.8.20' if repo=='facebookresearch/hydra' or old_dask else '3.10.18'
    req=['pytest==7.4.4','pytest-mock==3.12.0','pytest-asyncio==0.21.1',
         'setuptools=='+('59.8.0' if old_dask else '68.2.2'),'wheel==0.42.0','packaging==23.2']
    if repo=='dask/dask':
        req+=['numpy=='+('1.19.5' if old_dask else '1.23.5'),
              'pandas=='+('1.2.5' if old_dask else '1.5.3'),
              'scipy=='+('1.5.4' if old_dask else '1.10.1'),
              'toolz==0.12.1','cloudpickle==2.2.1','fsspec==2023.6.0',
              'partd==1.4.1','PyYAML==6.0.1','click==8.1.7','importlib-metadata==6.8.0',
              'graphviz==0.20.1','jinja2==3.1.3','psutil==5.9.8']
    elif repo=='pydantic/pydantic':
        text=(source/'pyproject.toml').read_text()
        core=re.search(r'pydantic-core(?:==|>=)([0-9.]+)',text).group(1)
        req+=['pydantic-core=='+core,'typing-extensions==4.10.0','annotated-types==0.6.0',
              'dirty-equals==0.7.1','hypothesis==6.98.0','email-validator==2.1.1',
              'python-dateutil==2.9.0.post0','pytest-mock==3.12.0','pytest-examples==0.0.10']
    elif repo=='facebookresearch/hydra':
        content=(source/'requirements/requirements.txt').read_text()
        omega='2.1.0.dev25' if '2.1.0.dev25' in content else '2.1.2' if '2.1' in content else '2.2.3' if '2.2' in content else '2.0.6'
        antlr='4.9.3' if '4.9' in content else '4.8'
        req+=['omegaconf=='+omega,'antlr4-python3-runtime=='+antlr,'PyYAML==5.4.1',
              'importlib-resources==3.3.1']
    elif repo=='bokeh/bokeh':
        req+=['numpy==1.23.5','pandas==1.5.3','contourpy==1.2.0','pillow==10.2.0',
              'PyYAML==6.0.1','Jinja2==3.1.3','tornado==6.4','xyzservices==2023.10.1',
              'selenium==4.18.1','networkx==3.2.1','ipython==8.22.2','colorama==0.4.6',
              'beautifulsoup4==4.12.3','pytest-timeout==2.2.0']
    default_config='\n'.join((source/name).read_text() for name in ['pyproject.toml','setup.cfg','pytest.ini'] if (source/name).exists())
    if '--cov-config' in default_config or '--cov=' in default_config:
        req+=['pytest-cov==4.1.0','coverage==7.4.4']
    if '--benchmark-' in default_config:
        req+=['pytest-benchmark==4.0.0']
    return python, sorted(set(req))


def install(row):
    python, req=requirements(row)
    key=hashlib.sha256(json.dumps([python,req]).encode()).hexdigest()[:12]
    env=BASE/'envs'/key
    metadata=BASE/'envs'/(key+'.json')
    if metadata.exists() and json.loads(metadata.read_text()).get('installed'):
        return env/'bin/python'
    env.parent.mkdir(exist_ok=True)
    reqpath=BASE/'envs'/(key+'.in')
    reqpath.write_text('\n'.join(req)+'\n')
    logfile=BASE/'envs'/(key+'.install.log')
    with logfile.open('a') as log:
        subprocess.run([str(UV),'venv','--python',python,str(env)],stdout=log,stderr=subprocess.STDOUT,check=True)
        constraints=[]
        previous=BASE/'freeze-v1.json'
        if previous.exists():
            match=next((t for t in json.loads(previous.read_text())['tasks'] if t['instance_id']==row['instance_id']),None)
            if match:
                old=Path(match['python']).parent.parent
                constraints=['-c',str(old.parent/(old.name+'.freeze.txt'))]
        subprocess.run([str(UV),'pip','install','--python',str(env/'bin/python'),'-r',str(reqpath),*constraints],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    frozen=subprocess.check_output([str(UV),'pip','freeze','--python',str(env/'bin/python')],text=True)
    (BASE/'envs'/(key+'.freeze.txt')).write_text(frozen)
    write(metadata,{'installed':True,'python_version':python,'requirements':req,'env':str(env)})
    return env/'bin/python'


def generated(row):
    source=BASE/'sources'/row['instance_id']
    if row['repo']!='facebookresearch/hydra' or not (source/'hydra/grammar').exists():
        return None
    output=BASE/'generated'/row['instance_id']
    if (output/'complete.json').exists():
        return output/'hydra/grammar/gen'
    output.mkdir(parents=True,exist_ok=True)
    # The generators and grammars are from this exact buggy base commit.
    shutil.copytree(source/'hydra/grammar',output/'hydra/grammar',dirs_exist_ok=True)
    jars=list((source/'build_helpers/bin').glob('antlr-*-complete.jar'))
    if len(jars)!=1:
        raise RuntimeError('Expected one pinned ANTLR jar')
    java=subprocess.check_output([str(BUILD_PY),'-c','import jdk4py; print(jdk4py.JAVA)'],text=True).strip()
    with (output/'antlr.log').open('w') as log:
        for grammar in ['OverrideLexer.g4','OverrideParser.g4']:
            subprocess.run([java,'-jar',str(jars[0]),'-Dlanguage=Python3','-o',str(output/'hydra/grammar/gen'),
                            '-Xexact-output-dir','-visitor',str(output/'hydra/grammar'/grammar)],
                           cwd=output,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    write(output/'complete.json',{'base_commit':row['base_commit'],'jar':str(jars[0]),
          'jar_sha256':hashlib.sha256(jars[0].read_bytes()).hexdigest()})
    return output/'hydra/grammar/gen'


ORACLE = '''import json
from pathlib import Path
import pytest
CONFIG=json.loads(Path("../oracle.json").read_text())
RESULTS={}
ALIASES={}
COLLECTION_ERRORS=[]
def pytest_collection_modifyitems(session, config, items):
    keep=[]
    def canonical(item):
        return item.nodeid.replace(str(Path.cwd()), "/testbed")
    for expected in CONFIG["nodes"]:
        matches=[i for i in items if canonical(i)==expected]
        if not matches:
            matches=[i for i in items if canonical(i).startswith(expected)]
        if len(matches)!=1:
            COLLECTION_ERRORS.append({"expected":expected,"matches":[i.nodeid for i in matches]})
        else:
            ALIASES[matches[0].nodeid]=expected
            if matches[0] not in keep: keep.append(matches[0])
    items[:]=keep
def pytest_runtest_logreport(report):
    if report.when=="call" or (report.when in ("setup","teardown") and report.failed):
        RESULTS[ALIASES.get(report.nodeid,report.nodeid)]=report.outcome
def pytest_sessionfinish(session, exitstatus):
    Path("../test-outcomes.json").write_text(json.dumps({"exitstatus":int(exitstatus),"tests":RESULTS,"aliases":ALIASES,"collection_errors":COLLECTION_ERRORS},indent=2))
'''


def make_case(case,row,python,acceptance=False):
    sandbox=case/'sandbox'
    work=sandbox/'workspace'
    shutil.copytree(BASE/'sources'/row['instance_id'],work)
    grammar=generated(row)
    if grammar:
        shutil.copytree(grammar,work/'hydra/grammar/gen',dirs_exist_ok=True)
    if row['repo']=='bokeh/bokeh':
        # Editable source execution requires distribution metadata; do not
        # install a different release's Python sources into the task env.
        version=row['version']+'.0.dev0+g'+row['base_commit'][:8]
        info=work/'src'/('bokeh-'+version+'.dist-info')
        info.mkdir()
        (info/'METADATA').write_text('Metadata-Version: 2.1\nName: bokeh\nVersion: '+version+'\n')
        (info/'INSTALLER').write_text('project-local frozen-source adapter\n')
    if row['repo']=='facebookresearch/hydra' and (work/'hydra/extra/pytest_plugin.py').exists():
        version=re.search(r'__version__\s*=\s*[\'"]([^\'"]+)',(work/'hydra/__init__.py').read_text()).group(1)
        info=work/('hydra_core-'+version+'.dist-info');info.mkdir()
        (info/'METADATA').write_text('Metadata-Version: 2.1\nName: hydra-core\nVersion: '+version+'\n')
        (info/'entry_points.txt').write_text('[pytest11]\nhydra_pytest = hydra.extra.pytest_plugin\n')
    for name in ['home','tmp','cache','dsh-home','artifacts']:
        (sandbox/name).mkdir()
    paths=[str(work)]
    if (work/'src').exists():
        paths.append(str(work/'src'))
    runtime={'task_python':str(python),'task_env':{'PYTHONPATH':os.pathsep.join(paths), 'PYTEST_DISABLE_PLUGIN_AUTOLOAD':''}}
    write(case/'runtime-config.json',runtime)
    patch=ROOT/'configs/gym-dsh.patch.yml'
    shutil.copy2(patch if patch.exists() else ROOT/'configs/gym-dsh-candidate.patch.yml',case/'qwen.patch.yml')
    for argv in [('init','-q'),('add','-f','.'),('-c','user.name=Task','-c','user.email=task@local','commit','-qm','Buggy base snapshot')]:
        git(work,*argv,check=True)
    if acceptance:
        patch=case/'test.patch';patch.write_text(row['test_patch'])
        git(work,'apply',str(patch),check=True)
        (work/'_gym_oracle.py').write_text(ORACLE)
        write(sandbox/'oracle.json',{'nodes':row['FAIL_TO_PASS']+row['PASS_TO_PASS']})
    own(sandbox)
    return sandbox


def test(case,row,label):
    sandbox=case/'sandbox'
    python=json.loads((case/'runtime-config.json').read_text())['task_python']
    files=sorted(set(n.split('::')[0] for n in row['FAIL_TO_PASS']+row['PASS_TO_PASS']))
    plugins=['_gym_oracle']
    cmd=[str(PY),str(ROOT/'scripts/gym_task_env.py'),'--sandbox',str(sandbox),'--',python,'-m','pytest','-q','--tb=short']
    for name in plugins:cmd+=['-p',name]
    cmd+=files
    started=time.monotonic()
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        result={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except subprocess.TimeoutExpired as exc:
        result={'returncode':None,'timeout':True,'stdout':str(exc.stdout),'stderr':str(exc.stderr)}
    path=sandbox/'test-outcomes.json'
    result.update(elapsed_seconds=time.monotonic()-started,command=cmd,outcomes=json.loads(path.read_text()) if path.exists() else {})
    path.unlink(missing_ok=True)
    write(case/(label+'.json'),result)
    return result


def validate(instance):
    row=task(instance)
    case=BASE/'env-validation'/(instance+'-'+str(time.time_ns()))
    case.mkdir(parents=True)
    result={'instance_id':instance,'repo':row['repo'],'passed':False,'evidence_dir':str(case)}
    try:
        python=install(row)
        result['python']=str(python)
        sandbox=make_case(case,row,python,True)
        baseline=test(case,row,'buggy-tests')
        patch=case/'gold.patch';patch.write_text(row['patch'])
        git(sandbox/'workspace','apply',str(patch),check=True)
        gold=test(case,row,'gold-tests')
        b=baseline['outcomes'].get('tests',{});g=gold['outcomes'].get('tests',{})
        checks={'expected_buggy_failures':all(b.get(n)=='failed' for n in row['FAIL_TO_PASS']),
                'expected_buggy_passes':all(b.get(n)=='passed' for n in row['PASS_TO_PASS']),
                'gold_all_passed':all(g.get(n)=='passed' for n in row['FAIL_TO_PASS']+row['PASS_TO_PASS']),
                'gold_exit_zero':gold['returncode']==0,
                'all_nodes_collected':not baseline['outcomes'].get('collection_errors',[1]) and not gold['outcomes'].get('collection_errors',[1])}
        result.update(checks=checks,passed=all(checks.values()),baseline_seconds=baseline['elapsed_seconds'],gold_seconds=gold['elapsed_seconds'])
    except Exception as exc:
        result['error']=repr(exc)
        if isinstance(exc,subprocess.CalledProcessError):
            result.update(error_stdout=exc.stdout,error_stderr=exc.stderr)
    write(case/'validation.json',result)
    with (BASE/'validation-attempts.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('instances',nargs='+');args=parser.parse_args()
    for instance in args.instances:
        result=validate(instance);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
