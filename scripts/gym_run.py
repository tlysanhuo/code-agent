"""Frozen SWE-Gym development baseline; one attempt, trusted patch export, eval."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import threading
import time
import urllib.request

from gym_facility import execute
from gym_prepare import BASE, ROOT, PY, ORACLE, make_case, task, test
from gym_task_env import task_uid
from swe_prepare import git, own, write
from qwen_connectivity_smoke import stop_owned_group
from qwen_tool_suite import gpu_snapshot


def check_freeze():
    frozen=json.loads((ROOT/'configs/gym-baseline-freeze.json').read_text())
    for name,expected in frozen['file_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen file changed: '+name)
    allocation=json.loads((ROOT/'configs/gym-gpu-allocation.json').read_text())
    if len(allocation['gpu_uuids'])!=frozen['protocol']['tensor_parallel_size']:
        raise RuntimeError('GPU count differs from the verified/frozen parallelism')
    return frozen


def existing_test(path):
    return any(part in ('tests','test','testing') for part in Path(path).parts) or Path(path).name.startswith(('test_','conftest.'))


def evaluate(case,row,python,is_probe=False):
    started=time.monotonic()
    # Candidate .git/config, objects, HEAD and filters are untrusted. Compare
    # files against a separate operator-created repository with a clean index.
    export=case/'export'
    make_case(export,row,python)
    target=export/'sandbox/workspace'
    baseline=git(target,'rev-parse','HEAD',check=True).stdout.strip()
    originals=set(git(target,'ls-files','-z',check=True).stdout.split('\0'))-{''}
    for path in target.iterdir():
        if path.name=='.git':continue
        if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
        else:path.unlink()
    ignored={'.git','__pycache__','.pytest_cache','.hypothesis','.mypy_cache','.ruff_cache'}
    shutil.copytree(case/'sandbox/workspace',target,dirs_exist_ok=True,symlinks=True,
                    ignore=lambda directory,names:[n for n in names if n in ignored])
    git(target,'add','-A',check=True)
    diff=git(target,'diff','--cached','--binary',baseline,check=True).stdout
    changed=[p for p in git(target,'diff','--cached','--name-only','-z',baseline,check=True).stdout.split('\0') if p]
    (case/'agent.patch').write_text(diff)
    violations=[p for p in changed if p in originals and existing_test(p)]
    symlinks=[p for p in changed if (target/p).is_symlink()]
    write(case/'patch-export.json',{'method':'trusted clean Git index; candidate Git ignored','changed_paths':changed,
          'test_modification_violations':violations,'new_or_modified_symlinks':symlinks,
          'bytes':len(diff.encode()),'sha256':hashlib.sha256(diff.encode()).hexdigest()})
    independent=case/'independent'
    sandbox=make_case(independent,row,python)
    work=sandbox/'workspace'
    application={'returncode':0,'empty_patch':not bool(diff.strip())}
    if symlinks:
        application.update(returncode=1,policy_error='New/modified symlinks are not applied by the privileged evaluator')
    elif diff.strip():
        p=git(work,'apply','--binary',str(case/'agent.patch'))
        application.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    write(case/'patch-application.json',application)
    # Restore the immutable baseline versions of every oracle-affected path,
    # then apply the test patch. Candidate tests cannot replace the oracle.
    import re
    test_paths=set(re.findall(r'^\+\+\+ b/(.+)$',row['test_patch'],re.M))
    test_paths|={n.split('::')[0] for n in row['FAIL_TO_PASS']+row['PASS_TO_PASS']}
    source=BASE/'sources'/row['instance_id']
    for name in test_paths:
        dst=work/name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():shutil.rmtree(dst)
            else:dst.unlink()
        if (source/name).exists():
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dst)
    patch=independent/'test.patch';patch.write_text(row['test_patch'])
    oracle_apply=git(work,'apply',str(patch))
    if oracle_apply.returncode:
        raise RuntimeError('Frozen test patch could not be applied: '+oracle_apply.stderr)
    (work/'_gym_oracle.py').write_text(ORACLE)
    write(sandbox/'oracle.json',{'nodes':row['FAIL_TO_PASS']+row['PASS_TO_PASS']})
    oracle_hash={n:hashlib.sha256((work/n).read_bytes()).hexdigest() for n in test_paths if (work/n).is_file()}
    own(sandbox)
    tested=test(independent,row,'acceptance')
    immutable=all((work/n).is_file() and hashlib.sha256((work/n).read_bytes()).hexdigest()==sha for n,sha in oracle_hash.items())
    outcomes=tested['outcomes'].get('tests',{})
    f2p=sum(outcomes.get(n)=='passed' for n in row['FAIL_TO_PASS'])
    p2p=sum(outcomes.get(n)=='passed' for n in row['PASS_TO_PASS'])
    resolved=(bool(diff.strip()) and not violations and application['returncode']==0 and immutable
              and tested['returncode']==0 and f2p==len(row['FAIL_TO_PASS']) and p2p==len(row['PASS_TO_PASS']))
    worker=json.loads((case/'worker-status.json').read_text()) if (case/'worker-status.json').exists() else {}
    process=json.loads((case/'process.json').read_text()) if (case/'process.json').exists() else {}
    requests=[json.loads(s) for s in (case/'model-requests.jsonl').read_text().splitlines()] if (case/'model-requests.jsonl').exists() else []
    raw_requests=[json.loads(s) for s in (case/'dsh-requests.jsonl').read_text().splitlines()] if (case/'dsh-requests.jsonl').exists() else []
    calls,returns={},{}
    for request in raw_requests:
        for message in request['body']['messages']:
            if message['role']=='assistant':
                calls.update({c['id']:c for c in message.get('tool_calls',[])})
            if message['role']=='tool':returns[message['tool_call_id']]=message['content']
    write(case/'tool-evidence.json',[{'call':v,'result':returns.get(k)} for k,v in calls.items()])
    costs=[json.loads(s) for s in (case/'request-costs.jsonl').read_text().splitlines()] if (case/'request-costs.jsonl').exists() else []
    usage=[c['usage'] for c in costs if c.get('usage')]
    errors=(case/'transport-errors.jsonl').read_text() if (case/'transport-errors.jsonl').exists() else ''
    if resolved:category='resolved'
    elif process.get('timeout'):category='task_timeout'
    elif 'maximum context length' in errors:category='context_limit'
    elif (case/'limits.jsonl').exists():category='request_or_output_budget'
    elif errors:category='infrastructure_transport_failure'
    elif not requests:category='infrastructure_startup_failure'
    elif violations:category='model_modified_existing_tests'
    elif symlinks:category='model_unsafe_symlink_patch'
    elif application['returncode']!=0:category='model_patch_application_failure'
    elif not immutable:category='model_changed_oracle_during_test'
    elif tested.get('timeout'):category='acceptance_timeout'
    elif tested['outcomes'].get('collection_errors'):category='candidate_test_collection_failure'
    elif not diff.strip():category='model_empty_patch'
    else:category='model_test_failure'
    result={'instance_id':row['instance_id'],'repo':row['repo'],'resolved':resolved,'category':category,
       'f2p_passed':f2p,'f2p_total':len(row['FAIL_TO_PASS']),'p2p_passed':p2p,'p2p_total':len(row['PASS_TO_PASS']),
       'model_requests':len(requests),'dsh_request_attempts':len(raw_requests),'tool_calls':len(calls),
       'tools_paired':set(calls)==set(returns),'unpaired_call_ids':sorted(set(calls)-set(returns)),
       'orphan_return_ids':sorted(set(returns)-set(calls)),
       'tool_error_count':sum(str(v).startswith('Error:') for v in returns.values()),
       'tool_timeout_diagnostics':sum('timed out' in str(v).lower() or 'cannot resolve foreground process group' in str(v) for v in returns.values()),
       'prompt_tokens_reported':sum(u['prompt_tokens'] for u in usage),
       'output_tokens_reported':sum(u['completion_tokens'] for u in usage),'requests_missing_usage':len(requests)-len(usage),
       'unknown_output_upper_bound':sum(requests[c['request']-1]['body']['max_tokens'] for c in costs if not c.get('usage')),
       'task_seconds':process.get('wall_seconds',worker.get('elapsed_seconds')),'evaluation_seconds':time.monotonic()-started,
       'worker':worker,'oracle_unchanged':immutable,'patch_bytes':len(diff.encode()),
       'acceptance_returncode':tested['returncode'],'test_results_recorded':bool(tested['outcomes']),
       'model_attempts':0 if is_probe else 1,'real_model':not is_probe,'automatic_retries':0}
    write(case/'evaluation.json',result)
    return result


def worker(case):
    check_freeze()
    info=json.loads((case/'case.json').read_text())
    execute(case,'http://127.0.0.1:18081/v1',info['prompt'])


def summarize(run):
    results=[json.loads((p/'evaluation.json').read_text()) for p in sorted(run.glob('task-*')) if (p/'evaluation.json').exists()]
    n=len(results);r=sum(x['resolved'] for x in results);z=1.959963984540054
    if n:
        center=(r/n+z*z/(2*n))/(1+z*z/n)
        half=z*math.sqrt((r/n)*(1-r/n)/n+z*z/(4*n*n))/(1+z*z/n)
        interval=[max(0,center-half),min(1,center+half)]
    else:interval=None
    value={'completed_tasks':n,'resolved':r,'resolved_rate':r/n if n else None,'wilson_95':interval,'cases':results,
           'prompt_tokens_reported':sum(x['prompt_tokens_reported'] for x in results),
           'output_tokens_reported':sum(x['output_tokens_reported'] for x in results),
           'unknown_output_upper_bound':sum(x['unknown_output_upper_bound'] for x in results),
           'model_requests':sum(x['model_requests'] for x in results),'tool_calls':sum(x['tool_calls'] for x in results),
           'task_seconds':sum(x['task_seconds'] or 0 for x in results),'evaluation_seconds':sum(x['evaluation_seconds'] for x in results)}
    write(run/'summary.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path);parser.add_argument('--task',help='Frozen instance ID for a separately recorded rerun')
    parser.add_argument('--export-probe',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.worker.resolve());return
    if args.export_probe:
        selected=json.loads((ROOT/'configs/gym-selected-candidates.json').read_text())['tasks'][0]
        row=task(selected['instance_id']);case=BASE/'export-probes'/str(time.time_ns());make_case(case,row,Path(selected['python']))
        patch=case/'trusted-gold.patch';patch.write_text(row['patch']);git(case/'sandbox/workspace','apply',str(patch),check=True)
        shutil.rmtree(case/'sandbox/workspace/.git')
        result=evaluate(case,row,Path(selected['python']),is_probe=True);write(case/'verification.json',result)
        print(case,result['resolved']);assert result['resolved'];return
    frozen=check_freeze();selected=frozen['tasks']
    if args.task:selected=[r for r in selected if r['instance_id']==args.task];assert len(selected)==1
    registry=BASE/'formal-run.json'
    if not args.task and registry.exists():
        raise RuntimeError('The first formal run is already registered. Inspect its live process/evidence; do not silently restart. Explicit --task creates a separately labelled rerun.')
    run=BASE/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+secrets.token_hex(3));run.mkdir(parents=True)
    previous=frozen.get('supersedes_infrastructure_run')
    kind='explicit_single_task_rerun' if args.task else ('infrastructure_rerun_v2' if previous else 'original_formal_run')
    if not args.task:write(registry,{'run':str(run),'controller_pid':os.getpid(),'first_attempts_only':not bool(previous),
                                    'run_kind':kind,'supersedes_infrastructure_run':previous})
    shutil.copy2(ROOT/'configs/gym-gpu-allocation.json',run/'gpu-allocation.json')
    write(run/'run.json',{'formal_first_run':not bool(args.task or previous),'run_kind':kind,
                          'supersedes_infrastructure_run':previous,'tasks':[r['instance_id'] for r in selected],
                          'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'freeze':frozen})
    for index,entry in enumerate(selected,1):
        row=task(entry['instance_id']);case=run/f'task-{index:02d}';make_case(case,row,Path(entry['python']))
        prompt=f'Repair the issue in the repository at {case}/sandbox/workspace. Inspect relevant code, make a minimal source repair, and run appropriate tests. You have one attempt, at most 30 model requests and 600 seconds.\n\n'+row['problem_statement']
        write(case/'case.json',{'instance_id':row['instance_id'],'prompt':prompt,'python':entry['python']})
    print('Evidence directory:',run,flush=True)
    service=child=None;started=None;stop=threading.Event()
    allocated=json.loads((run/'gpu-allocation.json').read_text())['gpu_uuids']
    def monitor():
        while not stop.wait(3):
            sample=gpu_snapshot(run/'gpu-memory.jsonl')
            processes=list(csv.reader(sample['processes'].splitlines()))
            if any(sum(bool(row) and row[0].strip()==uuid for row in processes)>1 for uuid in allocated):
                write(run/'resource-contention.json',{'time':time.time(),'sample':sample,'action':'stop own controller/service to release the assigned GPU'})
                os.kill(os.getpid(),signal.SIGTERM)
                return
    thread=threading.Thread(target=monitor,daemon=True)
    def terminate(signum,frame):raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM,terminate)
    lifecycle={'cleanup_verified':False}
    try:
        launcher=['bash',str(ROOT/'scripts/start_qwen.sh'),'--config',str(ROOT/'configs/gym-qwen-server.json'),
                  '--allocation',str(ROOT/'configs/gym-gpu-allocation.json')]
        with (run/'preflight.log').open('w') as log:subprocess.run(launcher+['--check-only'],stdout=log,stderr=subprocess.STDOUT,check=True)
        thread.start();started=time.monotonic()
        with (run/'server.log').open('w') as log:service=subprocess.Popen(launcher,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+720
        while time.monotonic()<deadline:
            if service.poll() is not None:raise RuntimeError('Service exited before readiness')
            try:
                req=urllib.request.Request('http://127.0.0.1:18081/v1/models',headers={'Authorization':'Bearer local-qwen'})
                with urllib.request.urlopen(req,timeout=2) as response:assert response.status==200
                break
            except (OSError,ValueError):time.sleep(2)
        else:raise TimeoutError('Service readiness timeout')
        for index,entry in enumerate(selected,1):
            check_freeze();case=run/f'task-{index:02d}';stamp=time.monotonic()
            with (case/'worker.log').open('w') as log:
                child=subprocess.Popen([str(PY),str(Path(__file__).resolve()),'--worker',str(case)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                process={'pid':child.pid,'timeout':False}
                try:process['returncode']=child.wait(timeout=600)
                except subprocess.TimeoutExpired:process['timeout']=True
                finally:stop_owned_group(child);child=None
                process['wall_seconds']=time.monotonic()-stamp;write(case/'process.json',process)
            try:result=evaluate(case,task(entry['instance_id']),Path(entry['python']))
            except Exception as exc:
                write(case/'evaluation-error.json',{'error':repr(exc)});raise
            summarize(run);print(entry['instance_id'],result['category'],flush=True)
    except BaseException as exc:
        lifecycle['error']=repr(exc);raise
    finally:
        if child:stop_owned_group(child)
        if service:stop_owned_group(service)
        stop.set()
        if thread.is_alive():thread.join(timeout=10)
        uids={task_uid(p) for p in run.glob('task-*/sandbox')};uids|={task_uid(p) for p in run.glob('task-*/independent/sandbox')}
        leftovers=[]
        for p in Path('/proc').glob('[0-9]*'):
            try:
                uid=int(next(s.split()[1] for s in (p/'status').read_text().splitlines() if s.startswith('Uid:')))
                if uid in uids or (p/'cwd').resolve().is_relative_to(run):leftovers.append(int(p.name))
            except (OSError,StopIteration):pass
        snapshot=gpu_snapshot(run/'gpu-memory.jsonl')
        group_alive=False
        if service:
            try:os.killpg(service.pid,0);group_alive=True
            except ProcessLookupError:pass
        count=len(json.loads((ROOT/'configs/gym-gpu-allocation.json').read_text())['gpu_uuids'])
        lifecycle.update(cleanup_verified=not leftovers and not group_alive,processes_remaining=leftovers,after_cleanup=snapshot,
                         service_group_remaining=group_alive,service_pid=service.pid if service else None,
                         gpu_service_wall_seconds=time.monotonic()-started if started else 0,
                         gpu_seconds=count*(time.monotonic()-started) if started else 0,
                         service_returncode=service.returncode if service else None)
        write(run/'lifecycle.json',lifecycle)
    summarize(run)


if __name__=='__main__':main()
