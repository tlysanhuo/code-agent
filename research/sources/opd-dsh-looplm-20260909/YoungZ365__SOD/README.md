<div align="center">

<h2>SOD: Step-wise On-policy Distillation for<br>Small Language Model Agents</h2>

<p>
  <a href="https://arxiv.org/abs/2605.07725">
    <img
        src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoColor=red"
      alt="Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/collections/youngzhong/sod">
    <img 
        src="https://img.shields.io/badge/Models-SOD-blue?logo=huggingface&logoColor=yellow" 
        alt="Models/SOD"
    />
  </a>
  <a href="https://huggingface.co/papers/2605.07725">
    <img 
        src="https://img.shields.io/badge/Daily%20Paper-SOD-yellow?logo=huggingface&logoColor=yellow" 
        alt="HuggingFace Daily Paper"
    />
  </a>
  <a href="https://huggingface.co/collections/Gen-Verse/open-agentrl-68eda4c05755ca5a8c663656">
    <img 
        src="https://img.shields.io/badge/Datasets-SOD-orange?logo=huggingface&logoColor=yellow" 
        alt="Datasets for SOD"
    />
  </a>
  <a href="https://www.alphaxiv.org/abs/2605.07725">
    <img 
        src="https://img.shields.io/badge/alphaXiv-2605.07725-purple?logo=arxiv&logoColor=white"
        alt="alphaXiv"
    />
  </a>
</p>

</div>

## Introduction

<p align="center">
  <img src="assets/intro.png" width="100%">
</p>

Applying On-Policy Distillation (OPD) to Tool-Integrated Reasoning (TIR) suffers from **cascading error propagation**: incorrect tool calls inject out-of-distribution observations that progressively amplify the student-teacher distribution shift, rendering the teacher's token-level supervision unreliable or even harmful.

**SOD (Step-wise On-policy Distillation)** addresses this by introducing an adaptive step-level weighting mechanism that:
- **Suppresses** distillation loss on steps where the student has drifted far from the teacher (erroneous pattern)
- **Restores** full supervision when the student recovers alignment (recovery pattern)
- **Maintains** dense token-level guidance on well-aligned steps (stable pattern)

All at **negligible additional computational cost** — the divergence metric reuses log-probabilities already computed in the OPD forward pass.

Experiments on challenging math, science, and code benchmarks show that SOD achieves up to **20.86%** improvement over the second-best baseline. **Notably, our 0.6B student achieves 26.13% on average@32 at AIME 2025.**

## Framework

<p align="center">
  <img src="assets/framework.png" width="100%">
</p>

## 📊 Main Results

Performance comparison of the Qwen3 series on 4 benchmarks. We report average@32.

| Params | Method | AIME 2024 | AIME 2025 | GPQA | LiveCodeBench | Average |
|--------|--------|:---------:|:---------:|:----:|:-------------:|:-------:|
| **0.6B** | Vanilla | 7.71 | 12.81 | 13.24 | 14.89 | 12.16 |
| | SFT | 5.67 | 5.42 | 15.20 | 9.61 | 8.97 |
| | GRPO | 4.06 | 4.90 | 20.38 | 15.95 | 11.32 |
| | OPD | 16.82 | 22.95 | 17.76 | 22.65 | 20.04 |
| | OPSD_gt | 12.63 | 17.04 | 17.32 | 16.73 | 15.93 |
| | OPSD_hint | 9.77 | 14.12 | 15.98 | 12.65 | 13.13 |
| | **SOD** | **20.84** | **26.13** | **22.19** | **27.72** | **24.22** |
| **1.7B** | Vanilla | 9.90 | 8.96 | 26.80 | 22.73 | 17.10 |
| | SFT | 26.77 | 22.40 | 29.85 | 24.63 | 25.91 |
| | GRPO | 25.63 | 21.67 | 33.55 | 20.70 | 25.39 |
| | OPD | 43.86 | 37.04 | 31.73 | 32.45 | 36.27 |
| | OPSD_gt | 33.85 | 24.69 | 35.02 | 22.73 | 29.07 |
| | OPSD_hint | 34.42 | 21.43 | 33.46 | 23.12 | 28.11 |
| | **SOD** | **50.83** | **41.72** | **38.72** | **40.63** | **42.98** |

## 🚀 Get Started
### Environment Setup

```bash
git clone https://github.com/YoungZ365/SOD.git
conda create -n SOD python=3.11
conda activate SOD
cd SOD
bash scripts/install_vllm_sglang_mcore.sh
pip install -e .[vllm]
```

### Data Preparation

Download the following datasets:

| Dataset | Link | Usage |
|---------|------|-------|
| 3K Agentic SFT Data | [🤗 HuggingFace](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K) | Cold-start SFT |
| 30K Agentic RL Data | [🤗 HuggingFace](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K) | RL / Distillation Training |
| Evaluation Benchmarks | [🤗 HuggingFace](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-Eval) | AIME2024/2025, GPQA-Diamond, LiveCodeBench |

### Sandbox Configuration

Configure [SandboxFusion](https://github.com/bytedance/SandboxFusion) for code execution:

1. **Local Deployment**: Refer to [SandboxFusion deployment docs](https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment)
2. **Cloud Service**: Use [Volcano Engine Code Sandbox](https://www.volcengine.com/docs/6662/1539235)

After obtaining an API endpoint, configure it in:
- `recipe/demystify/sandbox_fusion_tool_config.yaml`
- The function `check_correctness` in `verl/utils/reward_score/livecodebench/code_math.py`

## 🔧 Training

### Step 1: Cold-Start SFT

Configure `examples/SOD/run_sft.sh` with your paths:

- `MODEL_PATH`: Base model path (e.g., [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) or [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B))
- `TRAIN_DATA`: Path to the SFT `.parquet` file
- `SAVE_PATH`: Directory to save SFT checkpoints

```bash
bash examples/SOD/run_sft.sh
```

After SFT, merge the model checkpoint:

```bash
python3 -m verl.model_merger merge --backend fsdp \
    --local_dir <checkpoint_dir>/global_step_xxx \
    --target_dir <checkpoint_dir>/global_step_xxx/huggingface
```

### Step 2: SOD Training (Step-wise On-policy Distillation)

Configure `examples/SOD/run_sod.sh` with your paths:

- `MODEL_PATH`: Path to the SFT student model
- `TEACHER_MODEL_PATH`: Path to the teacher model (e.g., a GRPO-trained 4B model)
- `TRAIN_DATA`: Path to the RL `.parquet` file (30K dataset)
- Evaluation data paths for AIME2024/2025

```bash
bash examples/SOD/run_sod.sh
```

**Training Resources**: 8× NVIDIA H20 96GB GPUs, batch size 64.

You can monitor training dynamics and evaluation results via Weights & Biases (wandb).

## 📊 Evaluation

We support evaluation on **AIME 2024/2025**, **GPQA-Diamond**, and **LiveCodeBench-v6**.

Taking AIME as an example:

```bash
bash examples/SOD/eval/run_eval_aime.sh
```

You can observe average@32 / pass@32 / maj@32 metrics from your wandb project.

## 🤗 Model Zoo

If needed, you can download our distilled models directly:

| Model | Link | Description |
|-------|------|-------------|
| SOD-0.6B | [🤗 HuggingFace](https://huggingface.co/youngzhong/SOD-0.6B) | SOD distilled from Qwen3-4B teacher |
| SOD-1.7B | [🤗 HuggingFace](https://huggingface.co/youngzhong/SOD-1.7B) | SOD distilled from Qwen3-4B teacher |
| SOD-GRPO_teacher-4B | [🤗 HuggingFace](https://huggingface.co/youngzhong/SOD-GRPO_teacher-4B) | GRPO-trained Qwen3-4B teacher model |

All models are also available in our [HuggingFace Collection](https://huggingface.co/collections/youngzhong/sod).

## 📝 Citation

```bibtex
@article{zhong2026sod,
  title={SOD: Step-wise On-policy Distillation for Small Language Model Agents},
  author={Zhong, Qiyong and Zheng, Mao and Song, Mingyang and Lin, Xin and Sun, Jie and Jiang, Houcheng and Wang, Xiang and Fang, Junfeng},
  journal={arXiv preprint arXiv:2605.07725},
  year={2026}
}
```

## 🙏 Acknowledgements

Our implementation builds upon the excellent codebases of [VeRL](https://github.com/volcengine/verl), [Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL), and [ReTool](https://github.com/ReTool-RL/ReTool). We sincerely thank these projects for their valuable contributions to the community.
