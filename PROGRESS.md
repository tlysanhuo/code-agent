# Code-Agent 项目进度文档

> 本文档是项目进度的单一入口，每次会话结束或有状态变化时更新。
> 依据：README.md（调研与决策记录）、configs/loop-agent/plan.json（机器可读状态）。
> 新建理由（2026-09-12）：用户明确要求独立进度文档；README 已承载大量调研内容，
> 进度状态不再与之混排。README 顶部保留指针至此。

- **当前时间**：2026-09-12 ~05:10 UTC
- **项目定位**：简历面试项目。dense Qwen3.5-9B 三段管线（SFT → Agentic RL → OPD），
  LoopLM 循环层为限额加分臂（≤8 GPU·h，恢复 <50% dense 参照即止损封存）。

## 一页总览

| 阶段 | 状态 | 一句话 |
|---|---|---|
| ① SFT 冷启动 | **✅ 完成并通过验收** | 检查点已转 HF；HumanEvalPlus 持平基线 + held-out NLL 大幅改善（均经交叉验证） |
| ② Agentic RL | CPU 准备完成，环境层有缺口 | DSH→BaseHarness 接好（7 passed）；RL 任务清单冻结 v1；缺 Sandbox 本地后端 |
| ③ OPD | 配置草稿就绪 | teacher 27B SGLang 服务 smoke 未做；prompt-smoke 模式随时可跑 |
| LoopLM 臂 | 未启动（按计划最后做） | E1 资产保留 |

## 阶段 ① 验收结果（2026-09-12 评估完成，全部有证据链）

**HumanEvalPlus（P1 协议：前 32 题、greedy、同模板同服务参数）**

| 模型 | 通过 | 失败题 |
|---|---|---|
| dense 9B base（基线） | 31/32 = 96.9% | HumanEval/10 |
| **SFT step-500 检查点** | **31/32 = 96.88%** | HumanEval/10（与 base 同一题） |

→ **零能力损失**，连失败题目都一致。证据：runtime/loop-agent/humaneval-sft-formal-500.jsonl。

**Held-out NLL（assistant-token 掩码，与训练目标同款 slime `qwen3_5` mask；800 行=4 切片×200）**

| 切片 | base | SFT-500 | Δ |
|---|---|---|---|
| klear>16k（同格式 held-out，未训练） | 0.2752 | **0.1846** | **−33%** |
| 训练集参照（klear≤16k） | 0.2860 | 0.1173 | −59% |
| OpenHands≤16k（跨格式） | 0.6722 | 0.6308 | −6% |
| R2E≤16k（跨格式） | 0.4387 | 0.4307 | −2% |

→ 同格式泛化强劲（held-out 也降 33%）；train/held-out 差（0.117 vs 0.185）适中，1 epoch 未陷入死记；
跨格式迁移小——与 PIPE 的「接口熟悉度是格式特异」结论一致，DSH 接口要靠阶段②教，符合计划。
**交叉验证**：HF transformers 前向（训练栈同款 5.12.1）逐行对拍，base/SFT 各 8 行，
max|diff| = 5.9e-4 / 4.7e-4（阈 5e-3）→ vLLM prompt_logprobs 路线数字可信。
证据：runtime/sft-formal/nll-{base,sft-500}-summary.json、nll-crosscheck-*.json。

**评估工具调研结论（遵守「先查再造」纪律的补课记录）**：钉住的 vLLM 0.19.1 已移除
官方 perplexity 脚本（benchmarks 目录无 NLL 工具）；lm-eval-harness 不支持 slime 掩码的
多轮 assistant-NLL 且其 vLLM 后端对 GDN 混合架构未验证 → prompt_logprobs 是文档化标准
路径，自写胶水必须用 HF 前向金标准验证（已做，通过）。

**评估过程的 bug 复盘（4 个，全部修复并验证）**：① score 阶段顶层 import pyarrow
（.venv-runtime 无 pip）→ 改惰性导入；② 批循环步长 16 配切片 32 → 每行重复计分，
首轮数字作废重跑；③ 转换检查点的 config 是多模态声明但权重只有语言部分 → 离线 LLM
补 `language_model_only=True`；④ 248k 词表 × 16k chunk 的 logits 单块 7.6GB OOM →
chunk 4096 + gpu_mem_util 0.75 + eager。

## 阶段 ① SFT（训练全程记录）

**配置**：2×H100（GPU 0+5，UUID 记录于 configs/gpu-allocation.json）TP2/DP1 + CPU 优化器卸载；
Klear 单格式 28,208 条 ≤16k（Qwen3CN 规则过滤）；lr 1e-5 cosine、wd 0.1、bs 64；
用户钉死 1 epoch = 跨过 epoch1（441 步）的第一个检查点 = 第 500 步；
job 本身配 2 epoch（882 步），到点外部停。wandb 在线全程
（run [46hg4r4a](https://wandb.ai/3120252125-/code-agent-dense-mainline/runs/46hg4r4a)）。

**训练曲线**：loss 0.289 → 0.18（epoch1 末）→ 0.12（停前 step 533）；
grad_norm 13.4 → 0.6；~105-110 s/步；显存 ~57GB/卡 稳定。

**落盘产物**（models/dense-9B-sft/formal-2card/）：

| 产物 | 大小 | 说明 |
|---|---|---|
| `iter_0000499` | 135G | **目标检查点**（第 500 步；Megatron 0 索引命名），.metadata+latest=499 完整 |
| `iter_0000439` | 135G | 中途检查点（epoch1 边界附近） |
| `iter_0000249` | 135G | 中途检查点 |
| `hf-iter500` | 17G | **已转换**：427 tensors / 4 shards / index / tokenizer 全齐 |

**时间线与事故（如实记录）**：
- 09-11 09:46 UTC 启动（detached）；14:2x 部署 step-500 停训 watcher。
- 09-12 02:14 UTC 目标检查点落盘。**事故①**：watcher 等待目录名 `iter_0000500`
  永不出现（0 索引命名实为 `iter_0000499`），训练多跑 **63 分钟**（至 step 533）。
  多跑步数不进检查点，有效性不受影响；wandb 曲线多一段、run 显示中断，原因即此。
- 03:17-03:18 UTC 修正后停训：scoped 清理（SIGTERM 68 PID → SIGKILL 7 残留，
  仅本会话目录匹配，他人会话验证零误伤），GPU 0/5 验证 0 MiB。
- **事故②**：watcher 自动转换失败（缺 PYTHONPATH）；手工带
  `PYTHONPATH=vendor/megatron-lm-src:vendor/slime` 重跑官方转换器成功。
- 教训：状态改变类动作先报告后执行；监控命令不得含被清理路径字符串。

**未完成**：无——阶段① 验收全部通过。（后续小项：iter_249/439 共 270G 确认后可清理；cudnn/torchaudio 升级。）

## 阶段 ② Agentic RL 准备状态

- **已接好**：`slime_dsh/`（DshHarness 生命周期 + generate 钩子，vendor/slime 零修改，
  SHA 钉住）；复检 `--harness-check` 7 passed。
- **任务清单冻结 v1**（configs/agent-rl/rl-task-freeze-v1.json）：SWE-Gym 2,438→
  2,357 eligible（61 旧审计 blocked + 20 held-out，与 Verified/full-test 零交集，
  独立复算）；SWE-smith 59,133 eligible（剔 3 冻结修复任务）；确定性 round 顺序
  + sha256 在 data/agent-rl/。10,894 个与 klear SFT 重合任务按设计保留（同分布课程）。
- **缺口**：① Sandbox 本地后端（LocalProcessSandbox 实现 slime 的 Protocol，项目内
  新文件）——阶段②真正的前置；② Docker 平台层（docker-proxy 探针失败）——只有
  阶段⑤官方评估（SWE-bench Verified 官方 harness）硬需要，训练可走已验证的本地
  受限进程路（自建训练协议标注，2026-09-09 用户已授权此路线用于训练）。
- 训练不依赖 Docker；官方评估依赖 Docker。
- **下一步（等用户指令）**：调研先行的 LocalProcessSandbox（先查 slime 生态/上游
  是否已有 local/remote sandbox 实现可复用，再决定自写范围）→ 用冻结清单前缀任务
  小规模 GRPO。

## 阶段 ③ OPD 状态

- 配置草稿 `scripts/train_opd_stage.sh`（官方 on_policy_distillation 结构，
  `--use-opd --opd-type sglang`）；prompt-smoke 模式可直接跑，agent 模式依赖阶段②环境。
- teacher = Qwen3.5-27B（全程唯一，外部 SGLang 服务专职一卡）；服务 smoke 未做。

## 环境 / 资产清单

- 训练栈 `.venv-train-rl`（torch 2.11.0+cu129 / TE 2.16.1 / sglang 0.5.15.post1 /
  transformers 5.12.1 / flash-attn 2.8.3 / fla 0.4.2 / triton 3.7.1，按官方
  build_conda.sh 收敛）；评估栈 `.venv-runtime`（vLLM 0.19.1）。
- 待升级（碰现有依赖，已排队）：cudnn 9.16.0.29、torchaudio 2.11+cu129。
- GitHub：`tlysanhuo/tlysanhuo-code-agent-rl`（私有）已同步（权重/数据/日志/缓存/
  vendor/secrets 均排除）。
- 磁盘：检查点 3×135G（iter_249/439 确认后可清理入 trash）+ hf-iter500 17G + smoke 117G；卷剩余 ~6T。

## 挂起 / 等用户

1. **阶段②启动指令**（先做 Sandbox 本地后端调研）。
2. Polar B' 数据集：上游 401（gated），需用户 HF 账号接受条款+token 或等公开
   （确切段址 `nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories`）。
3. iter_249/439 清理确认；cudnn/torchaudio 升级时机。
4. 阶段②③ GPU 使用照旧逐次授权。

## 变更日志

- 2026-09-12 ~05:10 UTC — 阶段①验收完成：HumanEvalPlus 96.88% 持平基线（同题失败）；
  held-out NLL 同格式 −33%/训练集 −59%/跨格式 −6%/−2%（HF 前向交叉验证通过）；
  工具调研结论与 4 个评估 bug 复盘入档。GitHub 已同步。
- 2026-09-12 03:3x UTC — 首建本文档（用户要求）。记录阶段①完成态、两次训练事故与处置、
  评估资产就绪态、阶段②③准备态。

