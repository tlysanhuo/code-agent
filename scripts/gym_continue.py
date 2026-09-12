"""Continue only unattempted cases after a recorded resource interruption.

The frozen model worker, prompt, budgets, tool relay and evaluator are imported
unchanged. Interrupted attempts remain in the denominator and are never retried.
"""
import csv,hashlib,json,os,secrets,signal,subprocess,threading,time,urllib.request
from pathlib import Path
from gym_run import BASE,ROOT,PY,check_freeze,evaluate,summarize,task,stop_owned_group,gpu_snapshot,task_uid,write


def main():
    frozen=check_freeze();run=Path(json.loads((BASE/'formal-run.json').read_text())['run'])
    assert json.loads((run/'run.json').read_text())['freeze']==frozen
    pending=[]
    for i,row in enumerate(frozen['tasks'],1):
        case=run/f'task-{i:02d}'
        if (case/'evaluation.json').exists():continue
        assert not (case/'model-requests.jsonl').exists(),f'Unclassified partial attempt: {case}'
        assert not (case/'attempt-started.json').exists(),f'Previously started case: {case}'
        pending.append((case,row))
    assert pending,'No unattempted cases remain'
    segment=run/'continuations'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+secrets.token_hex(3));segment.mkdir(parents=True)
    code=Path(__file__).read_bytes();(segment/'gym_continue.py').write_bytes(code)
    write(segment/'protocol.json',{'script_sha256':hashlib.sha256(code).hexdigest(),'controller_pid':os.getpid(),
          'frozen_harness_sha256':hashlib.sha256((BASE/'freeze-v2.json').read_bytes()).hexdigest(),
          'tasks':[r['instance_id'] for _,r in pending],'policy':'Only never-attempted cases; completed or resource-interrupted cases are preserved without model retry. Same frozen worker and evaluator; no harness or model change.'})
    allocation=json.loads((ROOT/'configs/gym-gpu-allocation.json').read_text());allocated=allocation['gpu_uuids'];write(segment/'gpu-allocation.json',allocation)
    service=child=None;active=None;started=None;error=None;stop=threading.Event()
    def terminate(signum,frame):raise KeyboardInterrupt('Owned continuation stopped')
    signal.signal(signal.SIGTERM,terminate)
    def monitor():
        while not stop.wait(3):
            sample=gpu_snapshot(run/'gpu-memory.jsonl');processes=list(csv.reader(sample['processes'].splitlines()))
            if any(sum(bool(row) and row[0].strip()==uuid for row in processes)>1 for uuid in allocated):
                write(segment/'resource-contention.json',{'time':time.time(),'sample':sample,'action':'Stop only this continuation and its owned service'})
                os.kill(os.getpid(),signal.SIGTERM);return
    thread=threading.Thread(target=monitor,daemon=True);life={'cleanup_verified':False}
    try:
        launcher=['bash',str(ROOT/'scripts/start_qwen.sh'),'--config',str(ROOT/'configs/gym-qwen-server.json'),'--allocation',str(ROOT/'configs/gym-gpu-allocation.json')]
        with (segment/'preflight.log').open('w') as log:subprocess.run(launcher+['--check-only'],stdout=log,stderr=subprocess.STDOUT,check=True)
        thread.start();started=time.monotonic()
        with (segment/'server.log').open('w') as log:service=subprocess.Popen(launcher,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+720
        while time.monotonic()<deadline:
            if service.poll() is not None:raise RuntimeError('Service exited before readiness')
            try:
                req=urllib.request.Request('http://127.0.0.1:18081/v1/models',headers={'Authorization':'Bearer local-qwen'})
                with urllib.request.urlopen(req,timeout=2) as response:assert response.status==200
                break
            except (OSError,ValueError):time.sleep(2)
        else:raise TimeoutError('Service readiness timeout')
        for case,row in pending:
            check_freeze();active=(case,row);stamp=time.monotonic();process={'timeout':False,'returncode':None,'interrupted':False}
            write(case/'attempt-started.json',{'time':time.time(),'continuation':str(segment),'instance_id':row['instance_id'],'model_attempt_number_under_v2':1})
            with (case/'worker.log').open('w') as log:
                child=subprocess.Popen([str(PY),str(ROOT/'scripts/gym_run.py'),'--worker',str(case)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);process['pid']=child.pid
                try:process['returncode']=child.wait(timeout=600)
                except subprocess.TimeoutExpired:process['timeout']=True
                except BaseException:process['interrupted']=True;raise
                finally:
                    stop_owned_group(child);process['returncode']=child.returncode;child=None
                    process['wall_seconds']=time.monotonic()-stamp;write(case/'process.json',process)
            active=None;result=evaluate(case,task(row['instance_id']),Path(row['python']));summarize(run);print(row['instance_id'],result['category'],flush=True)
    except BaseException as exc:error=exc;life['error']=repr(exc)
    finally:
        if child:stop_owned_group(child)
        if service:stop_owned_group(service)
        stop.set()
        if thread.is_alive():thread.join(timeout=10)
        leftovers=[];uids={task_uid(p) for p in run.glob('task-*/sandbox')}|{task_uid(p) for p in run.glob('task-*/independent/sandbox')}
        for p in Path('/proc').glob('[0-9]*'):
            try:
                uid=int(next(s.split()[1] for s in (p/'status').read_text().splitlines() if s.startswith('Uid:')))
                if uid in uids or ((p/'cwd').resolve().is_relative_to(run) and int(p.name)!=os.getpid()):leftovers.append(int(p.name))
            except (OSError,StopIteration):pass
        group_alive=False
        if service:
            try:os.killpg(service.pid,0);group_alive=True
            except ProcessLookupError:pass
        elapsed=time.monotonic()-started if started else 0
        life.update(cleanup_verified=not leftovers and not group_alive,processes_remaining=leftovers,service_group_remaining=group_alive,
          service_pid=service.pid if service else None,service_returncode=service.returncode if service else None,
          gpu_service_wall_seconds=elapsed,gpu_seconds=elapsed*len(allocated),after_cleanup=gpu_snapshot(run/'gpu-memory.jsonl'))
        write(segment/'lifecycle.json',life)
    if active:
        case,row=active
        if not (case/'worker-status.json').exists():write(case/'worker-status.json',{'status':'infrastructure_stop','finish_reason':'resource_contention' if (segment/'resource-contention.json').exists() else 'controller_error'})
        result=evaluate(case,task(row['instance_id']),Path(row['python']))
        write(case/'infrastructure-classification.json',{'category':'infrastructure_resource_interruption' if (segment/'resource-contention.json').exists() else 'infrastructure_controller_interruption','error':repr(error),'segment':str(segment),'no_model_retry':True})
        summarize(run)
    print('Continuation:',segment,'cleanup verified:',life['cleanup_verified'],flush=True)
    if error:raise error
    assert life['cleanup_verified']

if __name__=='__main__':main()
