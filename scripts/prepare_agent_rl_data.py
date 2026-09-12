"""Prepare task references for verl; no environments, conversations or rollouts."""
import hashlib
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from gym_holdout import classify, normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/agent-rl'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hold = json.loads((ROOT / 'configs/gym-training-exclusion.json').read_text())
    sources = [ROOT / 'configs/gym-training-exclusion.json', ROOT / 'configs/swe-three-tasks.json']
    sources += sorted((ROOT / 'runtime/gym-baseline').glob('selected-candidates*.json'))
    sources += sorted((ROOT / 'runtime/gym-baseline').glob('freeze-v*.json'))
    sources += [ROOT / 'configs/gym-selected-candidates.json']
    ids = set(hold['held_out_task_ids'])
    commits = {(v['repo'], v['base_commit']) for v in hold['task_commits'].values()}

    def collect(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get('instance_id'), str):
                ids.add(obj['instance_id'])
            if obj.get('repo') and obj.get('base_commit'):
                commits.add((obj['repo'], obj['base_commit']))
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)

    for path in sources:
        collect(json.loads(path.read_text()))
    full_meta = json.loads((ROOT / 'data/metadata/swebench-full-test-exclusion-source.json').read_text())
    full = ROOT / full_meta['path']
    assert sha(full) == full_meta['sha256']
    verified = ROOT / 'data/raw/SWE-bench__SWE-bench_Verified/data/test-00000-of-00001.parquet'
    sources += [full, verified]
    excluded_rows = []
    for path in [full, verified]:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                ids.add(row['instance_id'])
                commits.add((row['repo'], row['base_commit']))
                excluded_rows.append(row)
    gym = ROOT / 'data/raw/SWE-Gym__SWE-Gym/data/train-00000-of-00001.parquet'
    rows = pq.read_table(gym).to_pylist()
    excluded_rows += [r for r in rows if r['instance_id'] in ids]
    excluded_text = {normalize(r['problem_statement']) for r in excluded_rows}
    titles = {}
    for r in excluded_rows:
        titles.setdefault(r['repo'], []).append(normalize(r['problem_statement'].splitlines()[0]))
    counts = Counter()
    eligible = []
    decisions = []
    for r in rows:
        decision = classify(r, hold)['decision']
        title = normalize(r['problem_statement'].splitlines()[0])
        if r['instance_id'] in ids or (r['repo'], r['base_commit']) in commits:
            decision = 'exclude_id_or_commit'
        elif normalize(r['problem_statement']) in excluded_text:
            decision = 'exclude_problem_duplicate'
        elif any(SequenceMatcher(None, title, t).ratio() >= .8 for t in titles.get(r['repo'], [])):
            decision = 'quarantine_near_title'
        elif not all(r.get(k) for k in ['instance_id', 'repo', 'base_commit', 'problem_statement', 'FAIL_TO_PASS', 'patch', 'test_patch']):
            decision = 'quarantine_missing_provenance'
        counts[decision] += 1
        decisions.append({'instance_id': r['instance_id'], 'decision': decision})
        if decision == 'identified_nonholdout':
            eligible.append(r)
    # Reproducible small integration pool: lexical identity order, max two per repo.
    selected = []
    per_repo = Counter()
    for r in sorted(eligible, key=lambda r: r['instance_id']):
        if per_repo[r['repo']] >= 2 or len(r['problem_statement']) > 10000:
            continue
        selected.append(r)
        per_repo[r['repo']] += 1
        if len(selected) == 8:
            break
    assert len(selected) == 8
    records = []
    original_positions = {r['instance_id']: i for i, r in enumerate(rows)}
    for r in selected:
        ident = r['instance_id']
        private = OUT / 'private' / f'{ident}.json'
        write(private, r)
        private.chmod(0o600)
        # Only problem is visible to policy. Gold/tests live behind verifier reference.
        records.append({
            'data_source': 'swe_gym_agent_rl', 'agent_name': 'dsh_agent', 'ability': 'code',
            'prompt': [{'role': 'user', 'content': r['problem_statement']}],
            'reward_model': {'style': 'rule', 'ground_truth': ''},
            'extra_info': {'index': original_positions[ident], 'instance_id': ident, 'repo': r['repo'], 'base_commit': r['base_commit'],
                           'environment_ref': 'SWE-Gym/SWE-Gym:' + ident,
                           'verifier_ref': str(private), 'split': 'train_integration',
                           'source_file': str(gym), 'source_row': original_positions[ident],
                           'environment_qualified': False},
        })
    pq.write_table(pa.Table.from_pylist(records), OUT / 'train.parquet')
    # Inspect all complete R2E shards, never partially downloaded bytes. Short repo
    # names and commit identity need a canonical mapping before cross-source use.
    r2e = []
    for path in sorted((ROOT / 'data/raw/R2E-Gym__R2E-Gym-Subset').rglob('*.parquet')):
        pf = pq.ParquetFile(path)
        sample = next(pf.iter_batches(batch_size=1)).to_pylist()[0]
        r2e.append({'file': str(path.relative_to(ROOT)), 'sha256': sha(path),
                    'rows': pf.metadata.num_rows, 'fields': pf.schema_arrow.names,
                    'example_identity': {k: sample.get(k) for k in ['repo_name', 'commit_hash', 'docker_image']},
                    'decision': 'quarantine_pending_canonical_repo_commit_and_overlap_mapping'})
    write(OUT / 'exclusion-audit.json', {'counts': dict(counts), 'decisions': decisions,
          'excluded_ids': sorted(ids), 'excluded_repo_commits': sorted(commits),
          'rule': 'existing gym_holdout.classify + full test/Verified + all frozen development identities + repo/base_commit + normalized problem + same-repo title similarity >=0.8',
          'trajectory_policy': 'All associated trajectories inherit task exclusion; no trajectory files exported or consumed. Unknown/ambiguous identities remain quarantined.',
          'limits': 'Heuristic near-duplicate screening does not establish absence of semantic or pretraining contamination.',
          'source_hashes': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in sources]})
    metadata = json.loads((ROOT / 'data/metadata/SWE-Gym__SWE-Gym.json').read_text())
    write(OUT / 'manifest.json', {'purpose': 'RL tasks for future integration; not on-policy trajectories, not qualified environments',
          'source_repo': 'SWE-Gym/SWE-Gym', 'revision': metadata.get('sha'),
          'source_file': str(gym.relative_to(ROOT)), 'source_sha256': sha(gym),
          'selection': '8 lexical IDs, <=2/repo, problem <=10000 chars after exclusions; no model outcomes',
          'tasks': [{'instance_id': r['instance_id'], 'repo': r['repo'], 'base_commit': r['base_commit'],
                     'source_row': original_positions[r['instance_id']],
                     'private_sha256': sha(OUT / 'private' / (r['instance_id'] + '.json'))} for r in selected],
          'output_sha256': sha(OUT / 'train.parquet'), 'r2e_inventory': r2e,
          'r2e_partial_shards_ignored': [str(p.relative_to(ROOT)) for p in (ROOT / 'data/raw/R2E-Gym__R2E-Gym-Subset').rglob('*.part')],
          'verifier': 'Pinned upstream SWE-Gym SWE-Bench-Fork; adapter must invoke unchanged harness in a clean environment; no local scoring implementation',
          'rollouts_generated': 0})
    print(json.dumps({'selected': [r['instance_id'] for r in selected], 'audit': dict(counts), 'r2e_complete_shards': len(r2e)}))


if __name__ == '__main__':
    main()
