# Stage-1 SFT 验收结果（2026-09-12 UTC）

实验：dense Qwen3.5-9B，Klear 单格式 28,208 条（≤16k，规则过滤），2×H100 TP2，
1 epoch（用户钉死 = 第 500 步检查点 `iter_0000499`），官方 retool 超参。
基座快照 `models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
wandb run [46hg4r4a](https://wandb.ai/3120252125-/code-agent-dense-mainline/runs/46hg4r4a)。

## 表 1：HumanEvalPlus（P1 协议：v0.1.10 前 32 题、greedy、base-check、同模板同服务参数）

| 模型 | 通过 | pass rate | 失败题 |
|---|---|---|---|
| dense 9B base | 31/32 | 0.9688 | HumanEval/10 |
| **SFT step-500** | 31/32 | **0.9688** | HumanEval/10（与 base 同一题） |

结论：**零能力损失**（连失败题目一致）。此指标为护栏（防轨迹-SFT 损伤底座），
非增益指标——base 已 96.9% 饱和。
明细：`humaneval-sft-formal-500.jsonl` / `humaneval-dense9b.jsonl`（题级代码+判定）。

## 表 2：Held-out 轨迹 NLL（assistant-token 掩码 = 训练同款 slime `qwen3_5` mask；800 行 = 4 切片 × 200）

| 切片 | base | SFT-500 | Δ | 说明 |
|---|---|---|---|---|
| klear >16k | 0.2752 | **0.1846** | **−33%** | 同格式 held-out（构造性排除，未训过） |
| klear ≤16k 训练集参照 | 0.2860 | 0.1173 | −59% | 训练分布内 |
| OpenHands ≤16k | 0.6722 | 0.6308 | −6% | 跨格式 |
| R2E ≤16k | 0.4387 | 0.4307 | −2% | 跨格式 |

结论：同格式轨迹分布学到且泛化（train −59% vs held-out −33%，非死记）；
跨格式迁移小（与 PIPE「接口熟悉度格式特异」一致）；NLL 按行明细 `nll-*-rows.jsonl`。

## 表 3：交叉验证（HF transformers 前向 eval loss，训练栈同款 5.12.1，金标准）

| 模型 | 对拍行数 | max\|diff\| | mean\|diff\| | 判定（阈 5e-3） |
|---|---|---|---|---|
| base | 8 | 5.9e-4 | 3.0e-4 | AGREE |
| SFT-500 | 8 | 4.7e-4 | 1.7e-4 | AGREE |

结论：vLLM `prompt_logprobs` 路线数字与 HF 前向一致，**表 2 数字可信**。
背景调研：钉住的 vLLM 0.19.1 已移除官方 perplexity 脚本；lm-eval-harness 无法表达
slime 掩码多轮 assistant-NLL → prompt_logprobs 为文档化标准路径，胶水经金标准验证。

## 证据文件（本目录）

- `humaneval-sft-formal-500-summary.json` / `.jsonl`：表 1 SFT 明细
- `humaneval-dense9b-summary.json`：表 1 基线（P1 原始记录）
- `nll-base-summary.json` / `nll-sft-500-summary.json` / `nll-*-rows.jsonl`：表 2
- `nll-crosscheck-base.json` / `nll-crosscheck-sft-500.json`：表 3
- `stop-report.json`：停训+转换链路记录（检查点 135G mcore → 17G HF）
- 训练事故与 4 个评估 bug 复盘：`../PROGRESS.md` 阶段① 章节

## 遗留的决定性实验（待跑）

「SFT 买到的是 agent 轨迹能力」的直接度量：held-out 20 个 SWE-Gym 任务
（本地已验证 buggy/gold 环境）上 base vs SFT 同预算 zero-shot 对比
（Klear/mini-swe-agent 式协议），比有效轨迹率 / 非空 patch 率 / F2P 通过数。
当前表 1-3 只证明「格式学到 + 能力无损」，未证明「格式换来解决率」。
