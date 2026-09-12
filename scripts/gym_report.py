"""Read-only evidence audit and cost report; never reruns an agent or a test."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def report(run):
    config=load(run/'run.json');freeze=config['freeze'];expected={r['instance_id']:r for r in freeze['tasks']}
    records=[]
    for case in sorted(run.glob('task-*')):
        if not (case/'evaluation.json').exists():continue
        result=load(case/'evaluation.json');identifier=result['instance_id'];row=load(ROOT/'runtime/gym-baseline/private'/f'{identifier}.json')
        exported=load(case/'patch-export.json');tests=load(case/'independent/acceptance.json');outcomes=tests['outcomes'].get('tests',{})
        nodes=row['FAIL_TO_PASS']+row['PASS_TO_PASS']
        actual_resolved=(bool((case/'agent.patch').read_bytes()) and not exported['test_modification_violations']
            and not exported.get('new_or_modified_symlinks') and load(case/'patch-application.json')['returncode']==0
            and tests['returncode']==0 and all(outcomes.get(n)=='passed' for n in nodes) and result['oracle_unchanged'])
        assert actual_resolved==result['resolved'],identifier
        model=lines(case/'model-requests.jsonl');raw={r['request']:r for r in lines(case/'dsh-requests.jsonl')}
        response_usage={};unfinished=[]
        for req in model:
            path=case/f"response-{req['request']:02d}.sse"
            if path.exists():
                payload=path.read_text()
                for line in payload.splitlines():
                    if line.startswith('data: ') and line!='data: [DONE]':
                        chunk=json.loads(line[6:])
                        if chunk.get('usage'):response_usage[req['request']]=chunk['usage']
                if 'data: [DONE]' not in payload:unfinished.append(req['request'])
        checks={'request_limit':len(model)<=30,'all_output_caps':all(0<r['body']['max_tokens']<=2048 for r in model),
                'sampling_frozen':all(r['body'].get('temperature')==0 and r['body'].get('top_p')==1 and r['body'].get('seed')==0 for r in model),
                'output_budget_reported':sum(u['completion_tokens'] for u in response_usage.values())<=16384,
                'patch_hash':hashlib.sha256((case/'agent.patch').read_bytes()).hexdigest()==exported['sha256'],
                'acceptance_execution_recorded':bool(tests.get('command')) and (
                    tests.get('returncode') is not None or tests.get('timeout') is True),
                'tool_pairs':result['tools_paired']}
        views_ok=True;raw_saved=True
        for req in model:
            shown=[m for m in req['body']['messages'] if m['role']=='tool']
            originals={m['tool_call_id']:m['content'] for m in raw[req['request']]['body']['messages'] if m['role']=='tool'}
            views_ok &= sum(len(str(m['content']).encode()) for m in shown)<=16384
            for m in shown:
                views_ok &= len(str(m['content']).encode())<=4096
                original=originals[m['tool_call_id']]
                if not isinstance(original,str):original=json.dumps(original,ensure_ascii=False)
                key=hashlib.sha256((m['tool_call_id']+'\0'+original).encode()).hexdigest()[:24]
                p=case/'raw-tools'/(key+'.txt')
                raw_saved &= p.exists() and p.read_text()==original
        checks.update(tool_view_limits=views_ok,raw_tool_results_saved=raw_saved)
        import re
        oracle_files=set(re.findall(r'^\+\+\+ b/(.+)$',row['test_patch'],re.M))|{n.split('::')[0] for n in nodes}
        reference=Path(expected[identifier]['evidence_dir'])/'sandbox/workspace'
        independent=case/'independent/sandbox/workspace'
        oracle_equal=True
        for name in oracle_files:
            if (reference/name).is_file():
                oracle_equal &= (independent/name).is_file() and hashlib.sha256((independent/name).read_bytes()).digest()==hashlib.sha256((reference/name).read_bytes()).digest()
        checks['oracle_matches_qualified_environment']=oracle_equal
        missing=[r for r in model if r['request'] not in response_usage]
        checks['output_budget_including_unknown_ceiling']=(sum(u['completion_tokens'] for u in response_usage.values())
            +sum(r['body']['max_tokens'] for r in missing)<=16384)
        record={k:result[k] for k in ['instance_id','repo','resolved','category','model_requests','tool_calls','tool_error_count','tool_timeout_diagnostics','task_seconds','evaluation_seconds','f2p_passed','f2p_total','p2p_passed','p2p_total','patch_bytes']}
        record.update(prompt_tokens_reported=sum(u['prompt_tokens'] for u in response_usage.values()),
                      output_tokens_reported=sum(u['completion_tokens'] for u in response_usage.values()),
                      requests_without_usage=len(missing),unknown_output_token_upper_bound=sum(r['body']['max_tokens'] for r in missing),
                      unfinished_streams=unfinished,evidence_checks=checks,evidence_checks_passed=all(checks.values()))
        process=load(case/'process.json')
        if (case/'infrastructure-classification.json').exists():
            record['runtime_category_before_infrastructure_classification']=record['category']
            record['category']=load(case/'infrastructure-classification.json')['category']
        record.update(task_timeout=process.get('timeout',False),
                      attempt_finish_reason=result.get('worker',{}).get('finish_reason','worker_interrupted_or_unavailable'))
        records.append(record)
    n=len(records);resolved=sum(r['resolved'] for r in records);z=1.959963984540054
    interval=None
    if n:
        center=(resolved/n+z*z/(2*n))/(1+z*z/n)
        half=z*math.sqrt((resolved/n)*(1-resolved/n)/n+z*z/(4*n*n))/(1+z*z/n)
        interval=[max(0,center-half),min(1,center+half)]
    allocation=load(run/'gpu-allocation.json');peaks={u:0 for u in allocation['gpu_uuids']}
    for sample in lines(run/'gpu-memory.jsonl'):
        for row in csv.reader(sample['gpus'].splitlines()):
            if row and row[1].strip() in peaks:peaks[row[1].strip()]=max(peaks[row[1].strip()],int(row[2].strip()))
    segments=[run]+sorted((run/'continuations').glob('*')) if (run/'continuations').exists() else [run]
    lifecycles=[load(p/'lifecycle.json') for p in segments if (p/'lifecycle.json').exists()]
    uncontended={u:0 for u in peaks}
    for sample in lines(run/'gpu-memory.jsonl'):
        processes=list(csv.reader(sample['processes'].splitlines()))
        for row in csv.reader(sample['gpus'].splitlines()):
            if row and row[1].strip() in uncontended and sum(bool(p) and p[0].strip()==row[1].strip() for p in processes)==1:
                uncontended[row[1].strip()]=max(uncontended[row[1].strip()],int(row[2].strip()))
    result={'scope':'SWE-Gym 20-task development subset; not an official leaderboard','formal_first_run':config['formal_first_run'],
            'reporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'run_kind':config.get('run_kind','original_formal_run'),
            'expected_tasks':len(config['tasks']),'completed_tasks':n,'resolved':resolved,'resolved_rate':resolved/n if n else None,
            'wilson_95_descriptive_interval':interval,'cases':records,'all_cases_audited':bool(n==len(config['tasks']) and all(r['evidence_checks_passed'] for r in records)),
            'input_tokens_reported':sum(r['prompt_tokens_reported'] for r in records),'output_tokens_reported':sum(r['output_tokens_reported'] for r in records),
            'requests_without_usage':sum(r['requests_without_usage'] for r in records),
            'unknown_output_tokens_upper_bound':sum(r['unknown_output_token_upper_bound'] for r in records),
            'model_requests':sum(r['model_requests'] for r in records),'tool_calls':sum(r['tool_calls'] for r in records),
            'task_wall_seconds_including_cleanup':sum(r['task_seconds'] or 0 for r in records),
            'independent_evaluation_seconds':sum(r['evaluation_seconds'] for r in records),'sampled_peak_gpu_mib':peaks,
            'sampled_peak_gpu_mib_without_concurrent_processes':uncontended,
            'gpu_memory_note':'Raw card peak can include the foreign process that triggered resource-interruption cleanup; uncontended samples are reported separately.',
            'gpu_service_wall_seconds':sum(l.get('gpu_service_wall_seconds',0) for l in lifecycles),
            'gpu_card_seconds':sum(l.get('gpu_seconds',0) for l in lifecycles),
            'service_segments':len(segments),'finished_service_segments':len(lifecycles),
            'cleanup_verified':len(lifecycles)==len(segments) and all(l.get('cleanup_verified',False) for l in lifecycles),
            'cost_note':'All completed attempts, including failures, are included. Missing usage is not counted as zero actual usage; output ceilings are reported separately. GPU card-seconds include loading, inference, idle time during independent CPU acceptance and teardown.'}
    (run/'audited-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if records:
        cols=[k for k in records[0] if k not in ['evidence_checks','unfinished_streams']]
        with (run/'results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args();r=report(args.run.resolve())
    print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
