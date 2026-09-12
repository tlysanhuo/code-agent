---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- looped-language-model
- reasoning
- recurrent-depth
- thinking
- chain-of-thought
---

# Ouro-2.6B-Thinking

![Ouro Logo](assets/logo.png)

## Model Description


**⚠️ IMPORTANT: This model is intended for research purposes only. It is provided as-is without warranties for production use. **


**Ouro-2.6B-Thinking** is a reasoning-specialized variant of the Ouro-2.6B base model, enhanced through supervised fine-tuning on high-quality reasoning data. Please use ``transformers==4.54.1``for compatibility.

![Thinking Model Performance](assets/ouro_thinking.png)

## Key Features

- **Advanced Reasoning**: Specifically optimized for mathematical and scientific reasoning tasks
- **Compact Size**: Competitive with 4B models despite having only 2.6B parameters
- **Cross-Step Consistency**: Intermediate recurrent outputs can serve as reliable proxies for final answers
- **Explicit Thinking Process**: Trained to generate detailed reasoning steps

## Configuration

### Recurrent Steps and Adaptive Exit

The model's computational behavior can be configured through the `config.json` file:

```json
{
  "total_ut_steps": 4,
  "early_exit_threshold": 1.0
}
```

- **`total_ut_steps`**: Controls the number of recurrent steps (default: 4). You can adjust this value to trade off between performance and computation time.
- **`early_exit_threshold`**: Controls the adaptive exit mechanism (default: 1.0). Lower values encourage earlier exit, while 1.0 means always use all steps.

**Example: Modify recurrent steps**
```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("ByteDance/Ouro-2.6B-Thinking")
config.total_ut_steps = 3  # Use 3 recurrent steps instead of 4
model = AutoModelForCausalLM.from_pretrained(
    "ByteDance/Ouro-2.6B-Thinking",
    config=config,
    device_map="auto"
)
```

> **Note**: vLLM does not currently support the adaptive exit feature due to its inference optimization characteristics. When using vLLM, the model will always execute the full number of `total_ut_steps`.


## Model Architecture

Based on Ouro-2.6B with additional reasoning fine-tuning:

| Configuration | Value |
|:---|:---|
| **Parameters** | 2.6B |
| **Layers** | 24 |
| **Recurrent Steps** | 4 |
| **Hidden Size** | 2048 |
| **Attention Heads** | Multi-Head Attention (MHA) |
| **FFN Activation** | SwiGLU |
| **Position Embedding** | RoPE |
| **Vocabulary Size** | 49,152 |
| **Context Length** | 32K (SFT) |
| **Normalization** | Sandwich RMSNorm |

## Training Details

### Pre-training
- **Training Tokens**: 7.7T tokens across 4 stages
- **Base Architecture**: Ouro-2.6B

### Supervised Fine-Tuning
- **Data Size**: ~8.3M examples
- **Data Composition**:
  - Mathematics: 3.5M examples (OpenThoughts3, AceReason-1.1-SFT)
  - Code: 3.2M examples (AceReason, OpenCodeReasoning, Llama-Nemotron, OpenThoughts3)
  - Science: 808K examples (OpenThoughts3, Llama-Nemotron)
  - Chat: 767K examples (DeepWriting-20K)
- **Training**: 2 epochs, max sequence length 32K
- **Optimizer**: Adam (lr=2×10⁻⁵, β=(0.9, 0.95))
- **Scheduler**: Cosine decay


## Quick Start

**⚠️ IMPORTANT**: Please use `transformers<4.56.0` to avoid compatibility issues. We recommend `transformers==4.54.1` or earlier versions.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Bytedance/Ouro-2.6B-Thinking"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# Generate with reasoning
messages = [
    {"role": "user", "content": "Solve: If 2x + 3 = 11, what is x?"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=512, temperature=1.0, top_p=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```



## Acknowledgments

We thank [@Antizana](https://github.com/Antizana) for the KV cache fix merged from [ouro-cache-fix](https://github.com/Antizana/ouro-cache-fix), which resolved a critical compatibility issue with transformers>=4.56.0.

## Citation

```bibtex
@article{zhu2025scaling,
  title={Scaling Latent Reasoning via Looped Language Models},
  author={Zhu, Rui-Jie and Wang, Zixuan and Hua, Kai and Zhang, Tianyu and Li, Ziniu and Que, Haoran and Wei, Boyi and Wen, Zixin and Yin, Fan and Xing, He and others},
  journal={arXiv preprint arXiv:2510.25741},
  year={2025}
}


## License

This model is licensed under Apache-2.0. See the LICENSE file for details.

## Project Links

- **Paper**: [Scaling Latent Reasoning via Looped Language Models](https://huggingface.co/papers/2510.25741)
- **Project Page**: [https://ouro-llm.github.io](https://ouro-llm.github.io)
- **Code**: [https://github.com/ByteDance/Ouro](https://github.com/ByteDance/Ouro)

---