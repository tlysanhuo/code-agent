"""Development holdout identity audit and fail-closed future-training guard.

This does not emit training data. Unknown/ambiguous provenance is quarantined.
"""
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata

ROOT=Path(__file__).resolve().parents[1]


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC',html.unescape(value)).split())


def canonical_id(value):
    if not isinstance(value,str):return None
    match=re.fullmatch(r'(.+__.+)\.[0-9a-f]{8}\.pr_(\d+)',value)
    if match:return match[1]+'-'+match[2]
    return value if re.fullmatch(r'[\w.-]+__[\w.-]+(?:-\d+|\.[0-9a-f]{8}\..+)',value) else None


def classify(row, manifest):
    held=set(manifest['held_out_task_ids'])
    ids={canonical_id(row.get(k)) for k in ['instance_id','task_id'] if isinstance(row.get(k),str)}
    ids.discard(None)
    messages=row.get('messages',row.get('conversations',[]))
    if isinstance(messages,str):
        try:messages=json.loads(messages)
        except ValueError:messages=[]
    texts=[m.get('content',m.get('value','')) for m in messages if isinstance(m,dict) and m.get('role',m.get('from')) in ['user','human']]
    if row.get('problem_statement'):texts.append(row['problem_statement'])
    suspected=set()
    for text in texts:
        if not isinstance(text,str):continue
        normalized=normalize(text)
        for i,prompt in manifest['held_out_normalized_problems'].items():
            if prompt in normalized:ids.add(i)
            elif len(manifest['held_out_titles'][i])>24 and manifest['held_out_titles'][i] in normalized:
                suspected.add(i)
        blocks=re.findall(r'<(?:pr_description|github_issue|issue_description)>\s*(.*?)\s*</(?:pr_description|github_issue|issue_description)>',text,re.S)
        for block in blocks+[text]:
            digest=hashlib.sha256(normalize(block).encode()).hexdigest()
            ids.update(manifest['all_gym_problem_identity'].get(digest,[]))
    if ids & held:return {'decision':'exclude_dev_holdout','task_ids':sorted(ids&held)}
    if suspected:return {'decision':'quarantine_possible_holdout','task_ids':sorted(suspected)}
    if len(ids)==1:return {'decision':'identified_nonholdout','task_ids':sorted(ids)}
    return {'decision':'quarantine_unidentified_or_ambiguous','task_ids':sorted(ids)}


def main():
    import pyarrow.parquet as pq
    from collections import Counter
    chosen=json.loads((ROOT/'configs/gym-selected-candidates.json').read_text())['tasks']
    ids={r['instance_id'] for r in chosen}
    all_rows=pq.read_table(ROOT/'data/raw/SWE-Gym__SWE-Gym/data/train-00000-of-00001.parquet',columns=['instance_id','problem_statement','repo','base_commit']).to_pylist()
    identity={}
    for row in all_rows:
        digest=hashlib.sha256(normalize(row['problem_statement']).encode()).hexdigest()
        identity.setdefault(digest,[]).append(row['instance_id'])
    selected=[r for r in all_rows if r['instance_id'] in ids]
    manifest={'held_out_task_ids':sorted(ids),'purpose':'development evaluation only; not final test set',
              'future_training_policy':'Exclude every matching task and all associated trajectories. Quarantine unknown, ambiguous or possible held-out identities until provenance is resolved; do not treat absence of an ID match as training eligibility.',
              'held_out_normalized_problems':{r['instance_id']:normalize(r['problem_statement']) for r in selected},
              'held_out_titles':{r['instance_id']:normalize(r['problem_statement'].splitlines()[0]) for r in selected},
              'all_gym_problem_identity':identity,'task_commits':{r['instance_id']:{'repo':r['repo'],'base_commit':r['base_commit']} for r in selected}}
    (ROOT/'configs/gym-training-exclusion.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    sources=['SWE-Gym__OpenHands-SFT-Trajectories','R2E-Gym__R2EGym-SFT-Trajectories',
             'SWE-bench__SWE-smith-trajectories','Kwai-Klear__SWE-smith-mini_swe_agent_plus-trajectories-66k',
             'allenai__SERA-4.6-Lite-Best-Subset']
    report={};matches=[]
    for source in sources:
        counts=Counter();matched_tasks=Counter()
        def inspect(row,path,index):
            verdict=classify(row,manifest);counts[verdict['decision']]+=1
            if verdict['decision'] in ['exclude_dev_holdout','quarantine_possible_holdout']:
                matched_tasks.update(verdict['task_ids']);matches.append({'source':source,'path':str(path.relative_to(ROOT)),'row':index,**verdict})
        for path in sorted((ROOT/'data/raw'/source).rglob('*.parquet')):
            pf=pq.ParquetFile(path)
            columns=['instance_id'] if 'instance_id' in pf.schema_arrow.names else ['messages']
            index=0
            for batch in pf.iter_batches(batch_size=8,columns=columns):
                for row in batch.to_pylist():inspect(row,path,index);index+=1
        for path in sorted((ROOT/'data/raw'/source).glob('*.jsonl')):
            with path.open() as f:
                for index,line in enumerate(f):inspect(json.loads(line),path,index)
        report[source]={'rows':sum(counts.values()),'decisions':dict(counts),'heldout_task_matches':dict(matched_tasks)}
    verified=pq.read_table(ROOT/'data/raw/SWE-bench__SWE-bench_Verified/data/test-00000-of-00001.parquet',columns=['instance_id']).column(0).to_pylist()
    output={'sources':report,'matched_rows':matches,'verified_task_overlap':sorted(ids&set(verified)),
            'not_training_processing':'Only identity links and exclusion policy were produced; no trajectories were exported, modified or used for training.'}
    (ROOT/'runtime/gym-baseline/holdout-audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
