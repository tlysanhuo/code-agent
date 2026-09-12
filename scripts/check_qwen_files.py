"""CPU-only tokenizer/config check and optional complete tensor-header audit."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import struct
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/Qwen3.5-27B/fc05daec18b0a78c049392ed2e771dde82bdf654'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    from transformers import AutoConfig, AutoTokenizer
    import torch
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    assert config.model_type == 'qwen3_5'
    assert str(config.text_config.dtype) in {'bfloat16', 'torch.bfloat16'}
    assert not getattr(config, 'quantization_config', None)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    messages = [{'role': 'user', 'content': 'Reply OK.'}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                                           enable_thinking=False)
    assert 'Reply OK.' in rendered and len(encoded) > 0
    assert encoded == tokenizer.encode(rendered, add_special_tokens=False)
    result = {'status': 'passed', 'config_class': type(config).__name__,
              'tokenizer_class': type(tokenizer).__name__, 'vocab_size': len(tokenizer),
              'sample_prompt_tokens': len(encoded), 'sample_template': rendered,
              'enable_thinking': False, 'cuda_initialized': torch.cuda.is_initialized(),
              'weights_audited': False}
    assert not result['cuda_initialized']
    if not args.metadata_only:
        manifest = json.loads((ROOT / 'runtime/metadata/qwen-download-manifest.json').read_text())
        assert manifest['complete'] and manifest['revision'] == MODEL.name
        index = json.loads((MODEL / 'model.safetensors.index.json').read_text())
        mapping = index['weight_map']
        seen = {}
        dtypes = Counter()
        parameters_by_dtype = Counter()
        total_bytes = 0
        parameters = 0
        shards = sorted(set(mapping.values()))
        for name in shards:
            path = MODEL / name
            with path.open('rb') as f:
                header_size = struct.unpack('<Q', f.read(8))[0]
                assert 0 < header_size < 32 * 1024 * 1024
                header = json.loads(f.read(header_size))
            end = 0
            for key, tensor in sorted(((k, v) for k, v in header.items() if k != '__metadata__'),
                                      key=lambda item: item[1]['data_offsets'][0]):
                assert key not in seen and mapping[key] == name
                begin, stop = tensor['data_offsets']
                assert begin == end and stop >= begin
                dtype = tensor['dtype']
                assert dtype in {'BF16', 'F32'}, (key, dtype)
                if dtype == 'F32':
                    assert key.endswith(('.linear_attn.A_log', '.linear_attn.norm.weight')), key
                n = math.prod(tensor['shape'])
                assert stop - begin == (2 if dtype == 'BF16' else 4) * n
                end = stop
                total_bytes += stop - begin
                parameters += n
                dtypes[tensor['dtype']] += 1
                parameters_by_dtype[tensor['dtype']] += n
                seen[key] = name
            assert 8 + header_size + end == path.stat().st_size
        assert seen == mapping
        assert total_bytes == index['metadata']['total_size']
        result.update(weights_audited=True, shards=len(shards), tensors=len(seen),
                      tensor_dtypes=dict(dtypes), parameters_by_dtype=dict(parameters_by_dtype),
                      tensor_bytes=total_bytes, parameters=parameters,
                      note='Official BF16 snapshot retains F32 A_log and linear-attention norm parameters; no weights were converted.')
    result['checked_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    target = ROOT / ('runtime/metadata/qwen-metadata-check.json' if args.metadata_only
                     else 'runtime/metadata/qwen-files-check.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
