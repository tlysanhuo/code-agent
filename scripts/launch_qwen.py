"""Foreground vLLM launcher restricted to explicitly allocated idle H100s."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/qwen-server.json')
    parser.add_argument('--allocation', type=Path, default=ROOT / 'configs/gpu-allocation.json')
    parser.add_argument('--model-dir', type=Path, default=None,
                        help='explicit checkpoint dir; skips the pinned-manifest check '
                             '(used for project-local loopify outputs)')
    parser.add_argument('--service-name', default='qwen')
    args = parser.parse_args()
    allocation = json.loads(args.allocation.read_text())
    devices = allocation['gpu_uuids']
    if not (1 <= len(devices) <= 2 and len(set(devices)) == len(devices)
            and allocation.get('allocated_by') and allocation.get('allocated_utc')):
        raise SystemExit('GPU allocation absent or invalid; no service started. Record the user/platform assignment in configs/gpu-allocation.json.')
    inventory = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,driver_version,index,pci.bus_id',
        '--format=csv,noheader,nounits'], text=True)
    rows = {r[0].strip(): [x.strip() for x in r[1:]] for r in csv.reader(io.StringIO(inventory))}
    # vLLM 0.19.1's NVML lookup parses CUDA_VISIBLE_DEVICES as integer indices.
    # UUIDs remain the allocation authority; resolve them on every launch.
    if [int(r[5]) for r in sorted(rows.values(), key=lambda r: r[6])] != list(range(len(rows))):
        raise SystemExit('NVML indices differ from PCI order; refusing an ambiguous GPU mapping')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'], text=True)
    for uuid in devices:
        r = rows.get(uuid)
        if not r or 'H100' not in r[0] or int(r[1]) < 79000:
            raise SystemExit(f'Allocated device is not an available H100 80GB: {uuid}')
        if int(r[2]) > 512 or int(r[3]) != 0 or uuid in active:
            raise SystemExit(f'Allocated GPU is in use: {uuid}; no process was stopped.')
    drivers = {rows[uuid][4] for uuid in devices}
    if len(drivers) != 1:
        raise SystemExit('Assigned GPUs report different driver versions')
    driver = next(iter(drivers))
    branch = int(driver.split('.')[0])
    compatibility_path = None
    if branch in (535, 570):
        compatibility_path = ROOT / 'runtime/cuda-compat/usr/local/cuda-12.9/compat'
        if not (compatibility_path / 'libcuda.so.1').is_file():
            raise SystemExit('Project-local CUDA 12.9 compatibility libraries are missing')
    elif branch < 575:
        raise SystemExit(f'CUDA 12.9 compatibility has not been prepared for driver {driver}')
    config = json.loads(args.config.read_text())
    if args.model_dir is not None:
        model = args.model_dir.resolve()
        if not (model / 'config.json').is_file() or not (model / 'model.safetensors.index.json').is_file():
            raise SystemExit(f'Not a complete HF safetensors checkpoint: {model}')
    else:
        manifest = json.loads((ROOT / 'runtime/metadata/qwen-download-manifest.json').read_text())
        if not manifest['complete'] or manifest['revision'] != config['model_revision']:
            raise SystemExit('Pinned model download and checksum verification must complete first.')
        for item in manifest['files']:
            path = ROOT / item['path']
            if not path.is_file() or path.stat().st_size != item['bytes']:
                raise SystemExit(f'Missing or changed model file: {path}')
        model = ROOT / 'models/Qwen3.5-27B' / config['model_revision']
    with socket.socket() as sock:
        # Match the HTTP server: ignore closed connections in TIME_WAIT,
        # while an actual listening service still prevents this bind.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((config['host'], config['port']))
    lock = (ROOT / f'runtime/{args.service_name}-service.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(), True)
    command = [str(ROOT / '.venv-runtime/bin/vllm'), 'serve', str(model),
               '--tokenizer', str(model), '--served-model-name', config['served_model_name'],
               '--host', config['host'], '--port', str(config['port']),
               '--api-key', 'local-qwen', '--dtype', config['dtype'],
               '--tensor-parallel-size', str(len(devices)), '--distributed-executor-backend', 'mp',
               '--max-model-len', str(config['max_model_len']),
               '--max-num-seqs', str(config['max_num_seqs']),
               '--max-num-batched-tokens', str(config['max_num_batched_tokens']),
               '--gpu-memory-utilization', str(config['gpu_memory_utilization']),
               '--enforce-eager', '--language-model-only',
               '--gdn-prefill-backend', config.get('gdn_prefill_backend', 'triton'),
               '--reasoning-parser', config['reasoning_parser'],
               '--enable-auto-tool-choice', '--tool-call-parser', config['tool_call_parser'],
               '--default-chat-template-kwargs', json.dumps({'enable_thinking': config['enable_thinking']})]
    record = {'pid': os.getpid(), 'gpu_uuids': devices, 'command': command,
              'cuda_visible_indices': [rows[uuid][5] for uuid in devices],
              'host_driver': driver, 'cuda_compat_path': str(compatibility_path) if compatibility_path else None,
              'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'check_only': args.check_only}
    print(json.dumps(record), flush=True)
    if args.check_only:
        return
    (ROOT / f'runtime/metadata/{args.service_name}-service-launch.json').write_text(json.dumps(record, indent=2)+'\n')
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(rows[uuid][5] for uuid in devices),
                      CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='4',
                      MAX_JOBS='4', NVCC_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', VLLM_NO_USAGE_STATS='1')
    if compatibility_path:
        os.environ['LD_LIBRARY_PATH'] = str(compatibility_path) + (
            ':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
