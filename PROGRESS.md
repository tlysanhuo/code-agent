# Code-Agent 项目进度文档

> 本文档是项目进度的单一入口，每次会话结束或有状态变化时更新。
> 依据：docs/research-log.md（调研流水）、configs/loop-agent/plan.json（机器可读状态）。
> **讲解主线看 [docs/decision-log.md](docs/decision-log.md)**（2026-09-12 起：D1-D14 决策
> 叙事，从项目起点记录每个选择的备选/裁决/依据/验证，PPT 按此展开）。
> 新建理由（2026-09-12）：用户明确要求独立进度文档；README 已承载大量调研内容，
> 进度状态不再与之混排。本文件与 docs/research-log.md 互补：本文件管进度，research-log 管调研全文。

- **当前时间**：2026-09-14 ~02:35 UTC
- **项目定位**：简历面试项目。dense Qwen3.5-9B 三段管线（SFT → Agentic RL → OPD），
  LoopLM 循环层为限额加分臂（≤8 GPU·h，恢复 <50% dense 参照即止损封存）。

## 一页总览

| 阶段 | 状态 | 一句话 |
|---|---|---|
| ① SFT 冷启动 | **✅ 完成并通过验收 + 价值判定** | HumanEvalPlus 持平基线；NLL −33%；**A/B：解决率 15%→30%、提交纪律 10%→100%** |
| ② Agentic RL | CPU 准备+奖励栈完成，round1b 454 待筛选（curated 87 已筛带内 12），等筛选 GPU 与 RL 启动协调 | DSH 接线+本地后端+任务冻结 v1；**奖励栈四模块落地（11/11 单测）**；round1b 扩池合并完成（454 合格）；harness-check 3 用例待改 local 契约 |
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
  SHA 钉住）；LocalProcessSandbox + local_backend（本地受限期用满 slime Sandbox 契约；
  集成检查过：金标 1.0/空 0.0）；RL 任务清单冻结 v1（configs/agent-rl/rl-task-freeze-v1.json）：
  SWE-Gym 2,438→2,357 eligible（61 旧审计 blocked + 20 held-out，与 Verified/full-test 零交集，
  独立复算）；SWE-smith 59,133 eligible（剔 3 冻结修复任务）；确定性 round 顺序
  + sha256 在 data/agent-rl/。10,894 个与 klear SFT 重合任务按设计保留（同分布课程）。
- **奖励栈已实现（2026-09-13，设计锁定 §3 执行完毕，CPU+单测）**：
  - `slime_dsh/reward.py`：二值核（金标 grading_solved）+ 未完成惩罚开关（DSH_C_UNFINISHED，
    默认 0）+ 失败四分类（environment/model/budget/none）；
  - `slime_dsh/group_repair.py`：GLM-5 组剔除+填充/丢弃（--rollout-all-samples-process-path；
    环境崩溃槽轮转重复有效槽、有效≤半整组丢）+ 组归一化钩子
    （--custom-reward-post-process-path；按 instance_id 组统计、剔除环境崩溃；
    接线时整体替换上游默认——其 reshape 在多分段轨迹下会退化为全批全局均值归一化）；
  - `slime_dsh/format_penalty.py`：Qwen3CN 规则级工具调用校验 + 字节级精确 token 定位
    （真 Qwen3.5 tokenizer 验证逐 token 字节拼接==解码 utf-8 字节）+ custom advantage
    钩子（复刻 grpo/gspo/cispo 基线后监督 token 减 DSH_C_FMT；与 loss_mask 正交）；
  - `slime_dsh/blocker.py`：Qwen3CN 防作弊拦截器（仓库链接∧网络关键字→阻断+反馈回注，
    轨迹可继续）；adapter 回复层改写机制，DSH_BLOCKER=1 显式启用（默认关）。
  - 单测 scripts/check_reward_stack.py 11/11 PASS；超参数值全部走环境变量、默认零/关，
    **数值本体=GRPO smoke 超参单待呈批**（含格式罚需同步关 --normalize-advantages 的注记）。
- **检查现状（2026-09-13 复检）**：--dsh-check passed、check_local_backend OK、
  reward-stack 11/11；**--harness-check 4 passed/3 failed——先存问题非新引入**
  （git 对照验证）：3 个 lifecycle 用例的 fixture monkeypatch 上游 E2BSandbox，而
  boot_agent_sandbox 已被 local_backend.bind 整体替换（09-11/12 本地后端接线时即失效，
  此前记录"7 passed"已过期）——待把 fixture 改造为 local 契约（image='local'+注册
  fixture 案例）。
- **缺口**：Docker 平台层（docker-proxy 探针失败）——只有阶段⑤官方评估
  （SWE-bench Verified 官方 harness）硬需要，训练走已验证的本地受限进程路
  （自建训练协议标注，2026-09-09 用户已授权此路线用于训练）。
- **下一步（等用户指令）**：难度筛选（GPU 待用户安排；合格池=round1 87+round1b 454=541）→
  任务带→GRPO smoke 超参单呈批（grpo/cispo/gspo 三选一+组大小+奖励栈开关与系数）。

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

1. **筛选 GPU 安排**：待筛=**round1b 新增 454 任务**（mypy 198+moto 256，合并一致性
   校验通过），27B×k=4（1×H100 即可；GPU 2/5 已被他人占用，起批时重看空闲卡）。
   curated 87 任务**已于 09-12/13 筛过**（带内 12/全零 69/全解 4，结果沿用
   rl-round1-screened.json，不重筛——D15 裁决扩池而非放宽预算重筛）。本批
   454×4=1,816 attempts，按上批实测 25.8 attempts/h 约 3 天单卡，可分片；筛选后
   任务带=12（curated）+round1b 产出。
2. **RL 训练启动（用户协调）**：4 卡已获批但启动必须由用户协调；前置=筛选任务带+
   **GRPO smoke 超参单已起草待批**：[configs/agent-rl/grpo-smoke-hyperparams-v1.md]
   （configs/agent-rl/grpo-smoke-hyperparams-v1.md）——estimator 三选（推荐 gspo）、
   官方配方锚点核心超参、奖励栈 v1 开关全零/关+组归一化钩子推荐接线、DAPO 动态采样、
   5 项待拍板清单。
3. Polar B' 数据集：上游 401（gated），需用户 HF token 或等公开；cudnn/torchaudio 升级时机。

## 在飞

- **round1b 454 任务难度筛选全量批（2026-09-14 04:39 UTC 起）**：GPU 0（GPU-8c4bac00，
  分配记录 configs/screen-gpu-allocation-r1b.json），27B vLLM :18095（max_num_seqs 12），
  k=4、**并发 6**（试验批 6 题×4=24 attempts 实测 57.5 attempts/h=上批 2.2×、零抢占、
  达理想吞吐 91%，故不加码），1,792 attempts 预计 **~31h（09-15 中午 UTC 前后）**收尾；
  试验 24 条结果已并入免重跑。watcher scripts/run_screen_r1b_full.sh（结束自动停服+
  验证 GPU 0 释放，报告落 runtime/agent-rl/round1b-screen/）。日志：
  logs/screen-r1b-full.log（控制器）、logs/screen-qwen-server-r1b.log（服务）、
  logs/screen-r1b-full-tail.log（watcher）。试验批结论：带内 1/全零 5/全解 0（小样本，
  全量分布待出）；服务端并发上限曾是上批瓶颈之一（max_num_seqs 4→12）。

## 变更日志

- 2026-09-14 04:4x UTC — **round1b 难度筛选起批（用户批 GPU 0 并指示调高并发、先试几题）**：
  ①screen_rl_tasks.py 参数化（--registry/--prompts/--out-dir/--screened-json/
  --screened-parquet，默认值保持 round1 原行为）；②服务端并发上限修正——发现上批
  max_num_seqs=4 是隐性瓶颈，r1b 配置提到 12（configs/agent-rl/screen-qwen-server-r1b.json，
  其余参数与上批逐字一致）；③**试验批 6 题×k=4 并发 6**（4 moto+2 mypy 跨原批/w2 段）：
  24 attempts 25.1 min=**57.5 attempts/h（上批 2.2×）**，服务零抢占、6 并发达理想吞吐
  91%（6×3600/341s），瓶颈在单 attempt 时长（p50 201s/p90 915s），再加大并发只会
  逼近 1500s 超时线——全量定并发 6；④全量批 04:39 UTC 起（1,792 attempts，试验 24 条
  并入免重跑），watcher 自动停服+释放验证。GPU 0 UUID 记录、分配文件
  configs/screen-gpu-allocation-r1b.json。
- 2026-09-14 02:3x UTC — **round1b 扩池批收尾+合并完成**：588 候选全量处理（原批 198 被
  停未写 registry + w1 200/151 OK + w2 191/154 OK）。两处要点：①原批 registry 丢失系
  prepare_rl_tasks.py 批末统一落盘+进程被停所致——从盘重建（case 目录+runtime-config
  的 task_python，与 w1 registry 逐一比对零不一致）而非重跑；②原批被停前多跑 1 任务
  （index 197=w1 首任务 mypy-15846，两边均 OK）——合并脚本修正为 python 一致性硬校验
  +worker 优先去重（顺带修复 gp 局部导入 NameError，新增 454 条全量路径存在性校验）。
  **结果：454 合格（mypy 198/moto 256，77.2%，与探针 75-83% 吻合）**；产物
  configs/agent-rl/local-task-registry-r1b-merged.json + data/agent-rl/
  rl-round1b-merged-prompts.parquet（454 行契约校验过）+ runtime/agent-rl/
  round1b-merge-report.json。**待筛=round1b 454**（curated 87 已于 09-12/13 筛过、带内 12
  沿用；初版本条误把 87 计入待筛，同日用户问询后修正口径）。research-log 第十三轮、
  plan.json 同步。
- 2026-09-13 15:0x UTC — **奖励栈 slime_dsh 实现完成（锁定设计 §3 全项）**：四模块
  （reward/group_repair/format_penalty/blocker）+ generate.py 拦截器接线（DSH_BLOCKER=1
  启用）+ scripts/check_reward_stack.py 11/11 PASS（含真实 Qwen3.5 tokenizer 字节对齐
  验证）。要点：①失败四分类枚举+二值核+未完成惩罚（DSH_C_UNFINISHED 默认 0）；
  ②GLM-5 组剔除+填充/丢弃+组归一化钩子（实例化时实测抓出并修复一个局部/全局索引
  bug——第二组起 raw_rewards 索引错位）；③格式罚 token 定位用字节级 BPE 精确对齐
  （convert_ids_to_tokens+GPT-2 逆映射，逐 token 字节拼接==utf-8 全等断言）；
  ④拦截器=adapter 回复层改写为无害 echo+反馈回注，轨迹可继续。**超参数值全部默认
  零/关走环境变量，数值本体进 smoke 超参单待批**。顺带：harness-check 3 个 lifecycle
  用例失效为**先存问题**（本地后端接线期即坏，git 对照验证非本轮引入），PROGRESS 同步
  纠正两处过期表述（"7 passed"、"缺 Sandbox 本地后端"）。**GRPO smoke 超参单 v1 起草**
  （configs/agent-rl/grpo-smoke-hyperparams-v1.md）：官方 SWE 配方锚点+奖励栈 v1 开关
  （全零/关）+组归一化钩子推荐接线+DAPO 动态采样+5 项待拍板——待用户批。文档同步：
  research-log 第十二轮、decision-log D17、plan.json、handoff 刷新。
- 2026-09-13 13:0x UTC — 扩池批提速并行化（用户问询后）：实测单任务 48s（make_case 11.4s
  +evaluate 36.9s，三次全源码拷贝+金标测试），串行均摊 96s；prepare_rl_tasks 加 --skip
  切片参数，原批 197 处后停，起 w1/w2 双 worker 分片（[197..397)/[397..588)），预计 2×
  提速至 UTC 16-17 收尾。当前合计 292/588 处理、227 合格（77.6%）。文档全面同步：
  decision-log D15/D16、research-log 第十一轮、handoff-prompt 刷新、plan.json 状态段。
- 2026-09-13 06:0x UTC — 扩池探针迭代完成（mypy 通、moto 收尾中）：
  **修复四处基础设施 bug**（如实入档）：①swe_prepare.git() 的 safe.directory 用相对路径在
  本机构建无效→改绝对路径+per-copy 配置移到 root 属主的 tmp/gitcfg/（原沙箱内配置文件被
  own() chown 后被 git 静默拒读——本轮最深的一坑）；②探针必须用绝对 --out（git apply 补丁
  路径随 -C 工作目录解析）；③起批必须 source scripts/env.sh（否则 uv 把 venv python 链到
  /root/.local，沙箱 uid 无法执行）；④install() 增加失败残骸 venv 清理（防毒化重试）。
  **新增机制**：gym_prepare 增加 --repos 仓库白名单参数；runtime-config 新增 pytest_args
  通用通道（mypy 用 -o addopts= 清掉 -nauto）。**探针结果**：mypy 12 任务 9 OK=75%（配方=
  base+mypy_extensions+black+filelock/psutil 等）；moto 配方迭代 6 轮（py-partiql-parser
  0.5.4/joserfc 解钉/boto3+botocore/responses/jinja2+werkzeug+flask/flask-cors+pytz），
  首任务已 OK，等整批出率。私有任务缓存已为新仓库批量生成（1310 个）。

- 2026-09-13 05:2x UTC — 用户裁定：①smith 环境准备+qualification 立即启动（CPU，探产批先行）；
  ②RL 训练放宽到 4 卡，但**启动必须用户协调**（不得擅自起训）；③任务挑选=探产→qual→筛选
  流程继续。筛选 GPU（②阶段 27B 服务）仍未批，等用户安排。
- 2026-09-13 05:0x UTC — 难度筛选批完成（348/348，rc=0，GPU 2 释放 3 MiB，全程 12:09→01:38
  UTC 约 13.5h）。**结果：LEGO-RL 带内仅 12 任务**（1解=6/2解=5/3解=1），27B 全程解出 35/348
  （10%）；0 解 69（79%）——远高于 LEGO-RL 的 72.7% 未筛永不解率；4/4 全解 4 个（过易）；
  2 个 bokeh 任务 incomplete（screen_error，与 qual 期 git 超时同源）待重试。产物：
  configs/agent-rl/rl-round1-screened.json + data/agent-rl/rl-round1-screened-prompts.parquet。
  **含义：curated-repo 子集对 27B+DSH+当前预算太难，首轮池太薄——需决策：直接 12 任务
  smoke vs 扩筛下一批任务（curated 剩余 ~2,060 eligible + SWE-smith 59k 池）。**
- 2026-09-12 12:1x UTC — qualification 收尾（88 OK in log / registry 87，298.6 分钟）+ 难度筛选批
  已启动：GPU 2（GPU-2e106654，分配记录 configs/screen-gpu-allocation.json），vLLM 27B BF16
  :18095，87 任务×4 rollout=348 attempts，并发 3，产物 runtime/agent-rl/round1-screen/
  results.jsonl + rl-round1-screened.json；结束自动停服+验证 0 MiB（scripts/run_screen_tail.sh）。
  途中修复三个启动坑（如实记录）：①27B 下载 manifest 历史未收尾→verify-only 补全 24 文件
  sha256 全过；②screen-qwen-server.json 手抄 revision 错一字符（e773→e771）；③编排脚本
  探活 URL 缺 /v1 前缀。超参：k=4、并发 3、rollout 墙钟 1500s+缓冲。
- 2026-09-12T11:34:38Z — 全量 qualification 完成：87/291（29.9%）合格（pydantic 32/dask 29/bokeh 15/hydra 11），registry + 87 行 prompt parquet 落盘；首轮 GRPO 规模足够。
- 2026-09-12 ~10:55 UTC — 用户三项裁定落档：①奖励设计锁定（分级方案，reward-design
  文档状态改已锁定）；②难度筛选 GPU 批准（全量，1×H100，前置=qual 收尾+脚本 CPU 验证）；
  ③KAT-Coder-V2.5 纳入权威集（六家）。decision-log D11f/D12 状态同步。
- 2026-09-12 ~11:1x UTC — 应用户要求新建决策叙事文档 docs/decision-log.md（D1-D14，
  四段式：背景/备选/裁决依据/验证，含两次奖励调研自我纠错的如实保留+PPT 骨架映射）；
  TR 分析文档降为参照附录角色；PROGRESS 顶部挂决策文档单入口。

- 2026-09-12 ~11:0x UTC — 第八轮：2026-04+ code-agent TR 综合分析文档成文（用户指令：
  4 月红线+落文档供 PPT）。research/code-agent-tr-analysis-202604plus.md：6 份入选 TR
  （KAT-V2.5/DSV4/M2/OT-Agent/AgenticQwen/LEGO-RL）分析卡+三横向对比表+共识分歧+项目
  映射+PPT 素材索引；Devstral 2(2025-12)/Qwen3.5(无文本 TR)/闭源新贵出局核查记录。
  research-log 挂指针。

- 2026-09-12 ~10:35 UTC — 第五轮调研：code-agent 项目对照（用户指令「看我们到底怎么做」）。
  发现 2026H2 harness-native RL 脉络（OpenForgeRL/LEGO-RL/EvoHarness-RL/ClawGym II），最同源
  = LEGO-RL（arXiv 2608.17393：Qwen3.5-35B 在未修改 harness 内训，任务漏斗以 27B rollout
  难度筛选收 2,699 题、未筛 72.7% 从未解出；二值奖励+GSPO；Pearson≥0.998 token 保真）；
  OpenThoughts 全开源配方（GLM-4.6 teacher 比 GPT 系好 2 倍；RL 任务 1 万筛 700）；
  DSH 训练先例核实=空（差异化成立）。**裁决：qualification 后接难度带筛选（先于奖励塑形）；
  estimator 选项扩为 grpo/cispo/gspo**。research-log 第五轮段落入档。
- 2026-09-12 ~12:0x UTC — 奖励调研三轮收敛（用户指令：只要有时效性的大厂一手内容）：
  权威证据链定稿为五家基模 TR+一篇实证标尺——新增 Kimi K2.5 TR（2602.02276：规则 outcome+
  budget-control token 效率奖+GRM 多 rubric；Toggle −25~30% token；PARL）、MiniMax M2 TR+
  M2.1 第一方博客（复合奖励含工具格式罚；CISPO；MIS+轨迹过滤治噪声；FP32 head）、Meta
  ScaleRL（2510.13786：CISPO+DAPO 动态采样+自适应 prompt 过滤在获胜配方）。时效核查：GLM-5.3/
  K2.6/Qwen3.8-Max 无训练披露；DSV4.1-Flash=配方沿用。设计文档 §1.4-1.6 逐字引文入档、
  §4.6/§5 收敛（塑形四档、噪声治理三机制、grpo/cispo 双选项）；research-log 同步（含修复
  一处编辑事故：二轮标题被吞已还原）。
- 2026-09-12 ~11:0x UTC — 奖励调研二轮修正（用户驳回「smoke 纯二值」，指出样本不足）：
  核实**基模大厂 TR 无一纯二值**——MiniMax M2 TR（arXiv 2605.26494 §6.1.5 复合奖励：过程
  奖励含工具格式错误罚+墙钟时间奖励+reward-to-go）；Meta SWE-RL（arXiv 2502.18449，代码
  开源）：补丁相似度奖励+格式罚 −1，完全不用测试；澄清 R2E-Gym「Hybrid Verifier」为推理期
  重排序非 RL 奖励、DeepSWE 博志+本地代码双重确认稀疏 0/1；补学术对照（2605.02944 pass-rate
  部分分不可靠更优；2605.05112 二值信号 ~50% 通过率最强）。设计文档 §4.1 重写为「大厂 TR
  塑形阵营 vs 开源小算力二值阵营」、§5.1 决策点改为核+塑形分级起步（a-d 选项，先测量
  格式错误率/轮数/崩溃率）；research-log 同步修正并保留首轮有效事实（slime 原生 DAPO 动态
  采样等）。
- 2026-09-12 ~10:0x UTC — 业界方案调研（用户指令「你去调研业界的方案」，校准奖励设计）：
  本地快照精读 slime 官方 coding_agent_rl（纯二值、崩溃 reward=0+remove、超时不扣分）、
  Agent Lightning swe_smith（已解轮数惩罚 t0=80/λ=0.1、prompt 膨胀惩罚、防作弊四通道+
  网络层 default-deny 立场、格式错误反馈修正）、SkyRL（二值）；Web 核实 DAPO（arXiv
  2503.14476：动态采样+软超长惩罚，verl overlong_buffer 落地）、NeMo-RL 两段 SWE 指南
  （Stage-1 pivot=单步参数匹配奖励教格式，PivotRL arXiv 2603.21383；Stage-2 纯二值；
  G=8/LR 1e-6/clip 0.2/0.28）、Klear 无公开 RL 代码。**最大发现：slime 原生 DAPO 动态采样**
  （over-sampling + dynamic-sampling-filter，vendor 可查）。设计文档新增 §4 业界对照+
  §5 决策点校准；research-log 同步。qualification 批继续在飞。
- 2026-09-12 ~09:15 UTC — 阶段②奖励设计推进（handoff 指定最高优先）：完成三份 TR 奖励
  章节精读（tmp/{qwen3cn,glm5,dsv4}_tr.txt）与钉住 slime 挂载点核对
  （remove_sample→零 loss_mask @ slime/ray/rollout.py:351；custom-rm/post-process/
  sample-filter/all-samples-process/custom-advantage/custom-loss 全查实带行号；关键发现：
  remove_sample 不影响 advantage 归一化参与 → GLM-5 组填充/丢弃语义需经
  all-samples-process/custom-reward-post-process 自写）；综合设计文档成文
  research/reward-design-stage2-grpo.md（引文+映射表+实现计划+4 决策点），research-log
  挂指针。未写任何实现代码（等用户过目）。
- 2026-09-12T08:59:01Z — 项目清理（用户指令）：删除被取代的检查点（smoke 117G、iter_249/439 270G，钉住的 iter_499+hf-iter500 保留）、smoke-hf-dryrun 17G、重复 wheel、/tmp/ray-sft-smoke 362M（项目外，训练 RAY_TMPDIR 所致；后续 RL 运行改项目内 tmp）、陈旧 DSH socket 与测试残留；models/dense-9B-sft 538G→152G。保留：27B teacher、9B base、loop-qwen（E1 资产）、四个被 configs 引用的旧 venv、cache（环境缓存）。机器级副作用（非文件）：agent 用户+/home/agent（上游 chown 需要）、root git config 的项目路径条目。
- 2026-09-12 ~07:2x UTC — A/B 完成：SFT 价值判定成立（resolved 3/20→6/20、提交率 10%→100%、格式错误 18→3、步数 −44%）；本地协议冻结评估集确立；用户决定 bench 全本地跑（不走 Docker/远端），SV 行以本地协议判分替代。
- 2026-09-12 ~05:10 UTC — 阶段①验收完成：HumanEvalPlus 96.88% 持平基线（同题失败）；
  held-out NLL 同格式 −33%/训练集 −59%/跨格式 −6%/−2%（HF 前向交叉验证通过）；
  工具调研结论与 4 个评估 bug 复盘入档。GitHub 已同步。
- 2026-09-12 03:3x UTC — 首建本文档（用户要求）。记录阶段①完成态、两次训练事故与处置、
  评估资产就绪态、阶段②③准备态。

