"""Offline CPU checks. Never init Ray, CUDA, task environments or optimizers."""
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runtime/agent-rl'
assert os.environ.get('CODE_AGENT_ROOT') == str(ROOT), 'source scripts/agent_rl_env.sh first'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation check'


def main():
    results = {}

    def check(name, fn):
        try:
            results[name] = {'ok': True, 'evidence': fn()}
        except Exception:
            results[name] = {'ok': False, 'error': traceback.format_exc()}

    def imports():
        modules = ['torch', 'transformers', 'peft', 'ray', 'vllm',
                   'verl.trainer.main_ppo', 'verl.trainer.ppo.v1.trainer_sync',
                   'verl.workers.engine.fsdp.transformer_impl',
                   'verl.experimental.agent_loop.tool_agent_loop',
                   'verl.workers.rollout.vllm_rollout.vllm_async_server',
                   'verl.workers.rollout.vllm_rollout.vllm_rollout',
                   'verl.utils.checkpoint.fsdp_checkpoint_manager']
        return {n: getattr(importlib.import_module(n), '__version__', 'imported') for n in modules}

    check('imports', imports)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT / 'configs/agent-rl'), version_base=None):
        cfg = compose(config_name='qwen35_27b_lora_2h100')
    OmegaConf.resolve(cfg)
    (OUT / 'resolved-config.yaml').write_text(OmegaConf.to_yaml(cfg))

    def configuration():
        from verl.utils.config import validate_config, omega_conf_to_dataclass
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        validate_config(cfg, need_reference_policy(cfg), need_critic(cfg))
        rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
        assert cfg.trainer.n_gpus_per_node == 2 and cfg.trainer.nnodes == 1
        assert not need_reference_policy(cfg) and not need_critic(cfg)
        assert not cfg.distillation.enabled
        assert not cfg.trainer.val_before_train and cfg.trainer.test_freq < 0
        assert rollout.max_model_len == cfg.data.max_prompt_length + cfg.data.max_response_length
        return {'hydra_resolved': True, 'upstream_validate_config': True,
                'rollout_dataclass': type(rollout).__name__, 'training_launch_ready': False}

    check('config_validation', configuration)

    def model_interface():
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig
        from verl.utils.model import get_hf_auto_model_class
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
        model_config = omega_conf_to_dataclass(cfg.actor_rollout_ref.model)
        hf = AutoConfig.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
        auto = get_hf_auto_model_class(hf)
        # Shape-only construction of the real 27B architecture, no BF16 weights read.
        with init_empty_weights():
            base = auto.from_config(hf, attn_implementation='sdpa', dtype=torch.bfloat16)
            matches = [n for n, _ in base.named_modules() if re.fullmatch(model_config.target_modules, n)]
            nbase = sum(p.numel() for p in base.parameters())
            model = FSDPEngine._build_lora_module(SimpleNamespace(model_config=model_config), base)
        trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
        assert matches and len(matches) == 3 * hf.text_config.num_hidden_layers
        assert all('lora_' in n for n, _ in trainable)
        assert all(p.device.type == 'meta' for p in model.parameters())
        return {'auto_class': auto.__name__, 'architecture': hf.architectures,
                'base_parameters': nbase, 'lora_trainable_parameters': sum(n for _, n in trainable),
                'target_count': len(matches), 'first_targets': matches[:3],
                'upstream_lora_injection': True, 'tensor_storage': 'meta only',
                'bf16_weight_bytes_estimate': nbase * 2,
                'not_verified': '27B weight load, CUDA forward/backward, memory, throughput, distributed recovery'}

    check('qwen35_meta_and_upstream_lora', model_interface)

    def data_loading():
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
        from verl.utils.dataset.rl_dataset import RLHFDataset
        tok = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
        ds = RLHFDataset([cfg.data.train_files], tokenizer=tok, config=cfg.data)
        raw = pq.read_table(cfg.data.train_files).to_pylist()
        assert len(ds) == len(raw) == 8
        assert len({r['extra_info']['index'] for r in raw}) == 8
        lengths = []
        for i in range(len(ds)):
            row = ds[i]
            assert row['agent_name'] == 'dsh_agent'
            assert all(m['role'] == 'user' for m in raw[i]['prompt'])
            assert len(raw[i]['prompt']) == 1
            assert not {'patch', 'test_patch', 'FAIL_TO_PASS', 'PASS_TO_PASS'} & set(row)
            rendered = tok.apply_chat_template(raw[i]['prompt'], add_generation_prompt=True, enable_thinking=True, return_dict=True)
            lengths.append(len(rendered['input_ids']))
        assert min(lengths) > 10 and max(lengths) <= cfg.data.max_prompt_length
        return {'upstream_loader_rows': len(ds), 'prompt_token_lengths_without_DSH_system_and_tools': lengths,
                'policy_visible_content': 'problem_statement only',
                'metadata_not_to_serialize_to_policy': 'extra_info including verifier_ref',
                'online_env_qualified': False}

    check('upstream_dataset_and_tokenizer', data_loading)

    def mask_contract():
        import torch
        from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, AgentLoopMetrics
        from verl.trainer.ppo.core_algos import agg_loss
        # Synthetic token fixture for loss plumbing, not a generated trajectory.
        out = AgentLoopOutput(prompt_ids=[1, 2], response_ids=[3, 4, 5, 6],
                              response_mask=[1, 0, 0, 1], response_logprobs=[-.3, 0., 0., -.6],
                              metrics=AgentLoopMetrics()).as_dict()
        loss_values = torch.tensor([[1., 999., 999., 3.]], requires_grad=True)
        mask = out['response_mask'].unsqueeze(0)
        loss = agg_loss(loss_values, mask, 'token-mean')
        loss.backward()
        assert loss.item() == 2.0
        assert loss_values.grad.tolist() == [[.5, 0., 0., .5]]
        assert len(out['rollout_log_probs']) == len(out['responses'])
        return {'upstream_response_mask': out['response_mask'].tolist(),
                'masked_loss': loss.item(), 'tool_token_gradient_zero': True,
                'scope': 'synthetic CPU tensor fixture; no model loss/backprop or parameter update'}

    check('tool_loss_mask_contract', mask_contract)

    def interfaces():
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
        from verl.workers.rollout.llm_server import LLMServerClient
        from verl.workers.rollout.replica import TokenOutput
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        signatures = {}
        for cls, names in [(FSDPEngine, ['get_per_tensor_param', 'save_checkpoint', 'load_checkpoint']),
                           (LLMServerClient, ['generate']),
                           (FSDPCheckpointManager, ['save_checkpoint', 'load_checkpoint'])]:
            for name in names:
                signatures[cls.__name__ + '.' + name] = str(inspect.signature(getattr(cls, name)))
        return {'signatures': signatures, 'token_output_fields': list(TokenOutput.model_fields),
                'scope': 'actual imports and signatures; no weight transfer/checkpoint roundtrip executed'}

    check('rollout_sync_checkpoint_interfaces', interfaces)

    def isolation():
        import torch, ray
        assert not torch.cuda.is_initialized() and not ray.is_initialized()
        assert Path(sys.prefix).resolve() == ROOT / '.venv-train-rl'
        assert 'include-system-site-packages = false' in (Path(sys.prefix) / 'pyvenv.cfg').read_text()
        for name in ['TMPDIR', 'HF_HOME', 'UV_CACHE_DIR', 'PIP_CACHE_DIR', 'RAY_TMPDIR',
                     'VLLM_CACHE_ROOT', 'TENSORBOARD_DIR', 'TORCH_EXTENSIONS_DIR']:
            assert Path(os.environ[name]).resolve().is_relative_to(ROOT), name
        for source in ['vendor/verl-agent-rl', 'vendor/rllm-agent-rl']:
            assert not subprocess.check_output(['git', '-C', str(ROOT / source), 'diff', '--name-only'])
        manifest = json.loads((ROOT / 'research/sources/agent-rl-prep-20260909/source.json').read_text())
        for source in manifest['sources']:
            path = str(ROOT / source['path'])
            assert subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip() == source['commit']
            assert not subprocess.check_output(['git', '-C', path, 'diff', '--name-only'])
        return {'isolated_environment': sys.prefix, 'ray_started': False, 'cuda_initialized': False}

    check('isolation_and_no_execution', isolation)
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'checks': results,
              'all_preparation_checks_pass': all(v['ok'] for v in results.values()),
              'training_launch_ready': False, 'containers_started': 0, 'rollouts_generated': 0,
              'parameter_updates': 0, 'distributed_checkpoint_roundtrip': 'not_run'}
    (OUT / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['all_preparation_checks_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
