"""Check the actual upstream loader/config against qualified local training tasks."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

from agent_rl_native import upstream
from agent_rl_training_task import ROOT, LocalTrainingTask, write_json


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('CODE_AGENT_ROOT') != str(ROOT):
        raise SystemExit('Use scripts/agent_rl.sh --training-data-check')
    report = {'passed': False, 'real_model_rollouts': 0, 'parameter_updates': 0}
    try:
        upstream()
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from verl.utils.config import validate_config
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.dataset.rl_dataset import RLHFDataset
        from transformers import AutoTokenizer
        with initialize_config_dir(config_dir=str(ROOT / 'configs/agent-rl'), version_base=None):
            cfg = compose(config_name='qwen35_27b_lora_2h100',
                          overrides=['+native_http=vllm', '+training_runtime=local'])
        OmegaConf.resolve(cfg)
        validate_config(cfg, need_reference_policy(cfg), need_critic(cfg))
        tok = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
        ds = RLHFDataset([cfg.data.train_files], tokenizer=tok, config=cfg.data)
        manifest = json.loads((ROOT / 'data/agent-rl/train-ready-manifest.json').read_text())
        assert hashlib.sha256(Path(cfg.data.train_files).read_bytes()).hexdigest() == manifest['output_sha256']
        assert len(ds) == manifest['rows'] and len(ds) >= cfg.data.train_batch_size
        lengths = []
        for i in range(len(ds)):
            row = ds[i]
            task = LocalTrainingTask.load(row['extra_info']['environment_ref'])
            task.require_qualified()
            assert task.fingerprint == row['extra_info']['task_fingerprint']
            assert row['agent_name'] == 'dsh_agent' and row['extra_info']['environment_qualified']
            assert not {'patch', 'test_patch', 'FAIL_TO_PASS', 'PASS_TO_PASS'} & set(row)
            assert len(row['raw_prompt']) == 1 and row['raw_prompt'][0]['role'] == 'user'
            # This upstream loader returns raw_prompt; tokenization belongs to AgentLoop.
            rendered = tok.apply_chat_template(row['raw_prompt'], add_generation_prompt=True,
                                              enable_thinking=True, return_dict=True)
            ids = rendered['input_ids']
            assert 10 < len(ids) <= cfg.data.max_prompt_length
            lengths.append(len(ids))
        import torch, ray
        assert not torch.cuda.is_initialized() and not ray.is_initialized()
        report.update(passed=True, upstream_validate_config=True, upstream_dataset_rows=len(ds),
            prompt_lengths_without_dsh_system_tools=lengths,
            qualified_fingerprints_and_installed_dependencies_match=True,
            ray_initialized=False, cuda_initialized=False)
    except Exception:
        report['error'] = traceback.format_exc()
    report['checked_utc'] = datetime.now(timezone.utc).isoformat()
    path = ROOT / 'runtime/agent-rl/training-data-check.json'
    if path.exists():
        old = json.loads(path.read_text())
        write_json(path.with_name('training-data-check-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '.json'), old)
    write_json(path, report)
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
