# Code-Agent 项目进度文档

> 本文档是项目进度的单一入口，每次会话结束或有状态变化时更新。
> 依据：README.md（调研与决策记录）、configs/loop-agent/plan.json（机器可读状态）。
> 新建理由（2026-09-12）：用户明确要求独立进度文档；README 已承载大量调研内容，
> 进度状态不再与之混排。README 顶部保留指针至此。

- **当前时间**：2026-09-12 03:3x UTC
- **项目定位**：简历面试项目。dense Qwen3.5-9B 三段管线（SFT → Agentic RL → OPD），
  LoopLM 循环层为限额加分臂（≤8 GPU·h，恢复 <50% dense 参照即止损封存）。

## 一页总览

| 阶段 | 状态 | 一句话 |
|---|---|---|
| ① SFT 冷启动 | **训练完成+已停，检查点已转 HF，评估未跑** | 目标检查点（第 500 步）02:14 UTC 落盘；03:18 停训；HF 权重 17G 已就绪；等评估指令 |
| ② Agentic RL | CPU 准备完成，环境层有缺口 | DSH→BaseHarness 接好（7 passed）；RL 任务清单冻结 v1；缺 Sandbox 本地后端 |
| ③ OPD | 配置草稿就绪 | teacher 27B SGLang 服务 smoke 未做；prompt-smoke 模式随时可跑 |
| 评估 | 资产就绪未执行 | 一键 runbook + NLL 800 行已 prepare |
| LoopLM 臂 | 未启动（按计划最后做） | E1 资产保留 |

## 阶段 ① SFT（本次正式跑全程记录）

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
| `hf-iter500` | 17G | **已从 iter_0000499 转出**：427 tensors / 4 shards / index / tokenizer 全齐（03:22 UTC 核验） |

**时间线与事故（如实记录）**：
- 09-11 09:46 UTC 启动（detached）；14:2x UTC 部署 step-500 停训 watcher。
- 09-12 02:14 UTC 目标检查点落盘。**事故①**：watcher 等待的目录名 `iter_0000500`
  永不出现（0 索引命名实为 `iter_0000499`），训练未停，多跑 **63 分钟**（至
  step 533，epoch2 约 93 步）。多跑步数不进检查点，检查点有效性不受影响；
  wandb 曲线多出一段、run 显示被中断，原因即此。
- 09-12 03:17-03:18 UTC 修正后停训：scoped 清理（SIGTERM 68 PID → SIGKILL 7 残留，
  仅本会话 /tmp/ray-sft-smoke 匹配，他人会话验证零误伤），**GPU 0/5 验证 0 MiB**。
- **事故②**：watcher 自动转换失败（缺 PYTHONPATH）；03:2x 手工带
  `PYTHONPATH=vendor/megatron-lm-src:vendor/slime` 重跑官方转换器成功。
- 处置教训已记录：状态改变类动作先报告后执行；监控命令不得含被清理路径字符串
  （会被 scoped 匹配误杀）。

**未完成**：HumanEvalPlus 前 32 题评估（对比 dense 9B 基线 96.9%）、held-out NLL
（base vs SFT 对照）。资产全部就绪（见下），一条命令可跑，**等用户指令**。

## 评估资产（就绪，未执行）

- `scripts/eval_sft_formal.sh <gpu-uuid>`：一键 runbook——GPU 分配滚动记录（含空闲校验）
  → 以 P1 完全相同参数（:18092）serve hf-iter500 → HumanEvalPlus 前 32 题对比 96.9%
  → NLL 双模型打分 → 释放验证。
- `scripts/eval_sft_nll.py`：两阶段（prepare 已完成；score 走 .venv-runtime vLLM 0.19.1
  prompt_logprobs）；loss mask 复用训练同款 slime `qwen3_5` 生成器。
- `runtime/sft-formal/nll-sft-formal-prepared.jsonl`：800 行
  （klear>16k / OpenHands≤16k / R2E≤16k / 训练集参照 各 200，全部为训练构造性排除）。
- 已知小项：2/800 行 token 计数与 09-10 标注差 +108/+10（transformers 升 5.12.1
  模板漂移），良性，远离长度过滤边界。

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

## 阶段 ③ OPD 状态

- 配置草稿 `scripts/train_opd_stage.sh`（官方 on_policy_distillation 结构，
  `--use-opd --opd-type sglang`）；prompt-smoke 模式可直接跑，agent 模式依赖阶段②环境。
- teacher = Qwen3.5-27B（全程唯一，外部 SGLang 服务专职一卡）；服务 smoke 未做。

## 环境 / 资产清单

- 训练栈 `.venv-train-rl`（torch 2.11.0+cu129 / TE 2.16.1 / sglang 0.5.15.post1 /
  transformers 5.12.1 / flash-attn 2.8.3 / fla 0.4.2 / triton 3.7.1，按官方
  build_conda.sh 收敛）；评估栈 `.venv-runtime`（vLLM 0.19.1）。
- 待升级（碰现有依赖，已排队）：cudnn 9.16.0.29、torchaudio 2.11+cu129。
- 磁盘：检查点 3×135G（其中 iter_0000249/439 确认后可清理入 trash）+ hf-iter500 17G
  + smoke 117G；卷剩余 6.2T。

## 挂起 / 等用户

1. **评估指令**（HumanEvalPlus + NLL，runbook 一条命令）。
2. Polar B' 数据集：上游 401（gated），需用户 HF 账号接受条款+token 或等公开
   （确切段址 `nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories`，已核对论文快照）。
3. 阶段②③ GPU 使用照旧逐次授权。

## 变更日志

- 2026-09-12 03:3x UTC — 首建本文档（用户要求）。记录阶段①完成态、两次事故与处置、
  评估资产就绪态、阶段②③准备态。
