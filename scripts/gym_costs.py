"""Summarize actual baseline and facility cost evidence, without executing tasks."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runtime/gym-baseline'
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def main():
    runs=[]
    for label,name in [('interrupted_v1','formal-run-v1.json'),('stopped_custom_protocol_v2','formal-run.json')]:
        run=Path(load(BASE/name)['run']);r=load(run/'audited-report.json')
        fields=['completed_tasks','resolved','model_requests','input_tokens_reported','output_tokens_reported','requests_without_usage','unknown_output_tokens_upper_bound','tool_calls','task_wall_seconds_including_cleanup','independent_evaluation_seconds','gpu_card_seconds','sampled_peak_gpu_mib','cleanup_verified']
        runs.append(dict(label=label,run=str(run),**{k:r[k] for k in fields}))
    facilities=[]
    for case in sorted((BASE/'facility').glob('*-real-*')):
        costs=lines(case/'request-costs.jsonl');usage=[c['usage'] for c in costs if c.get('usage')]
        usage.append(load(case/'long-context-response.json')['usage'])
        samples=lines(case/'gpu-memory.jsonl');span=samples[-1]['time']-samples[0]['time']
        peak=max(int(row[2].strip()) for s in samples for row in csv.reader(s['gpus'].splitlines()) if row[1].strip()=='GPU-fdfa4d77-7c26-ef94-8ad9-617c68b6d062')
        facilities.append({'case':str(case),'passed':load(case/'verification.json')['passed'],'model_requests':len(costs)+1,
            'input_tokens_reported':sum(u['prompt_tokens'] for u in usage),'output_tokens_reported':sum(u['completion_tokens'] for u in usage),
            'gpu_sampled_window_seconds':span,'sampled_peak_gpu_mib':peak,
            'gpu_time_note':'Observed one-card monitoring window; excludes unobserved start/stop margins. Not an exact service-lifetime measurement.',
            'service_returncode':load(case/'verification.json')['service_returncode']})
    totals={k:sum(r[k] for r in runs+facilities) for k in ['model_requests','input_tokens_reported','output_tokens_reported']}
    totals.update(requests_without_usage=sum(r['requests_without_usage'] for r in runs),unknown_output_tokens_upper_bound=sum(r['unknown_output_tokens_upper_bound'] for r in runs),
                  measured_v1_v2_gpu_card_seconds=sum(r['gpu_card_seconds'] for r in runs),additional_facility_gpu_sampled_seconds=sum(r['gpu_sampled_window_seconds'] for r in facilities))
    value={'scope':'This 20-task Goal only; V1 infrastructure interruption and both real facility trials included. Earlier separate 3-task Goal is preserved in its own report. CPU mock usage is excluded.',
           'runs':runs,'real_facilities':facilities,'totals':totals,'cost_note':'No dollar conversion: hardware billing rate was not provided. Input tokens sum repeated histories; missing usage is unknown, not zero. GPU measured service lifetime includes loading, idle time during CPU acceptance, and cleanup.'}
    (BASE/'cost-accounting.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
if __name__=='__main__':main()
