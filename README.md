# Code Agent RL

A reproducible preparation project for coding-agent reinforcement learning with **DSH + Qwen3.5-27B + slime**.

The project now uses the upstream [THUDM/slime](https://github.com/THUDM/slime) training and rollout architecture. slime owns the Ray resource flow, SGLang sampling, trajectory/loss-mask handling, Data Buffer, Megatron updates, and weight synchronization. The project adds only a thin DSH `BaseHarness` adapter and the project-local execution boundary.

## Current status

The architecture and CPU interfaces are verified; online training is deliberately not enabled yet.

- Pinned slime commit: `4c193f1f37509cca70f0e88807a9305b70f63f4e`.
- Upstream coding-agent CPU suite: **64 passed, 1 skipped**. The skipped test needs the SGLang reasoning parser.
- DSH native SDK to slime OpenAIAdapter: passed with a scripted CPU model; session routing, streaming, generated IDs, logprobs, and cleanup were checked.
- DSH through the upstream coding-agent lifecycle: **7 passed** with fake model, sandbox, and evaluator boundaries.
- Real Qwen rollout, benchmark scoring, GPU training, parameter updates, checkpoint recovery, and weight synchronization: **not run**.
- Training tasks qualified under the slime protocol: **0**. Two historical tasks were qualified only under an earlier custom protocol and are not presented as slime results.

The current Qwen3.5-27B BF16 LoRA target remains blocked by an upstream compatibility gap. slime PR [#1865](https://github.com/THUDM/slime/pull/1865) is still unmerged, and its patch does not apply to the pinned mainline. The repository therefore does not copy that patch or invent replacement training code. The project budget remains at most two H100 80GB GPUs.

See [`configs/agent-rl/readiness.json`](configs/agent-rl/readiness.json) for the machine-readable gate and [`reports/verification.json`](reports/verification.json) for sanitized evidence. Model weights, task data, caches, private logs, checkpoints, and machine-specific paths are intentionally excluded.

## Reproduce the CPU checks

Requirements: Linux, Python 3.12, Git, and [uv](https://docs.astral.sh/uv/).

```bash
bash scripts/setup.sh
bash scripts/agent_rl.sh --check
bash scripts/agent_rl.sh --dsh-check
bash scripts/agent_rl.sh --harness-check
```

All caches and environments created by the scripts stay below the checkout. The checks use synthetic model/tokenizer boundaries and do not need a GPU. `--launch` is intentionally fail-closed and does not start Ray, SGLang, DSH tasks, or training.

## Layout

- `slime_dsh/`: thin DSH adapter built on slime's existing `BaseHarness` and `run_agent` transport.
- `scripts/`: setup, bounded CPU checks, and the fail-closed project entrypoint.
- `configs/`: pinned dependency and architecture metadata.
- `reports/`: sanitized verification summaries.

This repository is an engineering preparation snapshot, not a benchmark result or a claim of completed model training.
