---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- code
- code-generation
- code-reasoning
- agentic-coding
- tool-use
- instruction-tuned
- looped-transformer
- parallel-loop-transformer
- plt
---

# LoopCoder-V2

[**Paper**](https://huggingface.co/papers/2606.18023) | [**GitHub**](https://github.com/CSJianYang/LoopCoder)

LoopCoder-v2 is a 7B instruction-tuned code model based on the Parallel Loop Transformer (PLT). The model studies test-time computation scaling through repeated application of shared Transformer blocks while keeping the parameter count fixed.

The released checkpoint is the two-loop PLT variant (`plt_num_loops=2`). In the accompanying paper, [LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling](https://huggingface.co/papers/2606.18023), this setting gives the best gain-cost trade-off: the second loop provides most of the useful latent refinement, while additional loops show diminishing or unstable updates.

## Highlights

- 7B dense PLT coder trained from scratch on 18T tokens of mixed text and code data.
- Instruction-tuned with a matched supervised fine-tuning recipe.
- Uses cross-loop position offsets and shared-KV gated sliding-window attention.
- Targets code generation, multilingual code, code reasoning, agentic software engineering, and tool-use workflows.
- Strongest loop-count setting in the paper: two loops, not more.

## Model Details

| Item | Value |
| --- | --- |
| Architecture | `IQuestPLTCoderForCausalLM` |
| Parameters | Approximately 7B |
| Hidden size | 5120 |
| Layers | 14 shared layers |
| Attention heads | 40 |
| KV heads | 8 |
| Head dimension | 128 |
| Intermediate size | 27648 |
| Activation | SwiGLU |
| Normalization | RMSNorm, epsilon 1e-5 |
| Position embedding | RoPE, theta 500000 |
| Vocabulary size | 76800 |
| Max position embeddings | 131072 |
| Precision | bfloat16 |
| PLT loops | 2 |
| PLT window size | 64 |

## Evaluation Summary

Selected results from the paper are shown below. All LoopCoder-v2 variants use the same 7B shared-parameter setup and matched training, tuning, and evaluation protocols.

| Model | HumanEval+ | MultiPL-E | BigCodeBench | LiveCodeBench | SWE-bench Verified | Multi-SWE | Terminal-Bench | Terminal-Bench 2.0 | Mind2Web | BFCL V3 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, 1 loop | 81.1 | 69.5 | 40.1 | 27.4 | 43.0 | 14.0 | 26.3 | 11.2 | 35.3 | 32.2 | 38.0 |
| LoopCoder-v2, 2 loops | 84.1 | 73.9 | 46.1 | 35.4 | 64.4 | 31.0 | 34.2 | 21.0 | 34.5 | 40.1 | 46.5 |
| LoopCoder-v2, 3 loops | 75.0 | 69.8 | 43.3 | 28.6 | 27.6 | 11.0 | 30.0 | 12.2 | 35.1 | 36.3 | 36.9 |
| LoopCoder-v2, 4 loops | 76.8 | 67.3 | 40.8 | 24.5 | 22.4 | 9.3 | 26.3 | 9.0 | 41.4 | 39.5 | 34.3 |

The paper's main finding is that PLT loop-count scaling is non-monotonic. The two-loop model improves broadly over the one-loop baseline, including SWE-bench Verified from 43.0 to 64.4 and Multi-SWE from 14.0 to 31.0, while three or more loops regress on many tasks.

## Usage

This checkpoint uses a custom PLT model architecture. Load it in an environment that provides support for `IQuestPLTCoderForCausalLM` and the custom tokenizer/configuration files in this repository.

For vLLM inference, install vLLM from [yxing-bj/vllm](https://github.com/yxing-bj/vllm) and use `transformers==4.57.1`, then start the server with the following command:

```bash
vllm serve $MODEL --port 8080 \
    --max-num-batched-tokens 8192 --max-num-seqs 512 -tp 1 -dp 1 --trust-remote-code \
    --cudagraph-capture-sizes 1 2 4 8 12 16 24 32
```


## Training Data

LoopCoder-v2 was trained from scratch on an internal deduplicated mixture of text and code totaling 18T tokens, balanced at a 1:1 text-to-code token ratio. The code portion spans more than 100 programming languages. The largest language shares reported in the paper include Java, Python, JavaScript, Markdown, TypeScript, C, C++, PHP, C#, and HTML.

## Intended Use

LoopCoder-v2 is intended for code generation, code reasoning, repository-level software engineering assistance, and tool-use research. It is especially useful for studying how looped latent computation changes model behavior under fixed parameter count.

## Limitations

LoopCoder-v2 can produce incorrect, insecure, or incomplete code and should not be used without review in production systems. The released model is optimized for coding and tool-use workloads; performance on unrelated open-domain tasks may vary. The paper also shows that increasing PLT loops beyond the two-loop setting can hurt performance, so this checkpoint should be treated as the recommended loop-count configuration rather than evidence that more loops are always better.

## Citation

```bibtex
@misc{yang2026loopcoderv2loopefficienttesttime,
      title={LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling}, 
      author={Jian Yang and Shawn Guo and Wei Zhang and Tianyu Zheng and Yaxin Du and Haau-Sing Li and Jiajun Wu and Yue Song and Yan Xing and Qingsong Cai and Zelong Huang and Chuan Hao and Ran Tao and Xianglong Liu and Wayne Xin Zhao and Mingjie Tang and Weifeng Lv and Ming Zhou and Bryan Dai},
      year={2026},
      eprint={2606.18023},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.18023}, 
}
```