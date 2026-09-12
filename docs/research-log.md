Code-agent 微调实验：数据与框架调研

> **进度状态看 [PROGRESS.md](PROGRESS.md)**（2026-09-12 起为进度单一入口：阶段① SFT 已停训+转 HF、评估待指令；阶段② CPU 准备完成）。

更新：2026-09-10 UTC。项目记录首次建立于 2026-09-08；本文件作为调研、数据状态、实施路线和验收结果的统一入口。用户要求所有新增文件均在本项目目录内，见 [AGENTS.md](AGENTS.md)。

<!-- loop-agent-design:start -->
**2026-09-10 用户纠正（当前最高优先级）：此前「自研 loopify Qwen3.5-9B」主线废弃。** 用户明确：LoopLM 与 9B 模型无关，LoopLM 指真实存在的 Ouro 族模型；当前最重要的是**调研**而非自建工程。下段先前的 loopify 设计及其 P1 执行结果仅作为历史证据保留（手术/测量产物在 `models/loop-qwen/`、`runtime/loop-agent/`）；Qwen3.5-9B 下载保留（19.33GB 已校验，可作为后续任何方案的 student 候选）。

**LoopLM 与 OPD 调研结论（2026-09-10，取代此前所有选型判断）：**

*LoopLM 是什么（调研确认）：* LoopLM = 字节开源的 **Ouro 族**循环架构模型（[arXiv 2510.25741](https://arxiv.org/abs/2510.25741)，Apache-2.0，[项目页](https://ouro-llm.github.io/)）。架构：48 个共享权重的 transformer 块循环施加（`total_ut_steps=4` 可调），学习 per-token 循环深度分布 + `early_exit_threshold` 自适应退出，共享 KV cache 控制显存；1.7T→7.7T token 预训练规模，1.4B/2.6B 打平至 12B 级。Ouro-2.6B-Thinking 为推理 SFT 变体（transformers 4.54.1 custom code；**本项目安装的 vLLM 已含原生 `ouro.py` 支持**）。相关谱系：Huginn（Apple，loop-in-the-middle，对比中表现更强）、MELT（2605.07721，looped KV 显存与训练过渡）、LoopRPT（2603.19714，循环模型的 RL 预训练：EMA teacher + 步级奖励 + exit policy）、LoopCoder-v2（2606.18023，**R=2 是最优循环数**；隐式循环推理与显式 CoT 互补）、Parcae（2604.12946，循环模型缩放律）。**与 agent 的交叉已有工作：[arXiv 2608.18171](https://arxiv.org/pdf/2608.18171)《Looped LMs Improve Compositional Tool Calling》——评测了 Ouro-1.4B/2.6B 的组合工具调用，结论是循环深度确实提升该能力，但该工作只做评测、未做训练**；「循环架构 student + agent OPD 训练」的组合仍未被公开占据。

*OPD 是什么（调研确认）：* On-Policy Distillation = student 自己采样、teacher 对 student 轨迹逐 token 提供密集监督（reverse-KL）。源头为 [Thinking Machines 博客](https://thinkingmachines.ai/blog/on-policy-distillation/)；业界报道 Qwen 3.6/3.7、GLM-5.1、DeepSeek-V4 均采用。Agent 场景最新进展：**SOD**（2605.07725，teacher=GRPO 优化过的 Qwen3-4B、student=0.6B/1.7B；步级散度 d_k=步内 |Δlogπ| 均值，权重为历史散度乘积门控，工具观察 token 不计损失；比 naive OPD 相对提升 20.9%/18.5%；**SFT 与 GRPO 在 0.6B 上都打不过 vanilla**；1.7B student 恢复 teacher 的 69.8% vs OPD 的 58.9%）；**TurnOPD**（2607.05804，长程 agent 的 rollout 深度预算 + 轮次归一化损失，ALFWorld 1.7B student 86.29 vs OPD 83.00）；Reward-Gated OPD（2607.04037）。

**对本项目最关键的一条实测事实：本项目固定的 slime（`4c193f1f`）已原生支持 OPD。** `examples/on_policy_distillation/` 含 Qwen3-8B student / Qwen3-32B teacher 完整配方；`--use-opd --opd-type=sglang`（teacher 为外部 SGLang 服务，经 rm_url 取逐 token logprob）或 `--opd-type=megatron`（teacher 装进训练）；**OPD 以可加 KL 项叠加在任意 advantage estimator（GRPO/PPO）之上**，与 RL 正交而非互斥。verl 亦有对应 [OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html)与异步 OPD 配方；TRL 有 GKD trainer。**结论：OPD 不需要任何自研 trainer。**

*基于以上调研的候选方向（2026-09-10 用户已定稿：LoopLM 借鉴思想自己训练，OPD 走 slime 原生管线）：*
1. **slime 原生 OPD 主线**：student = Qwen3.5-9B（已下载校验），teacher = 项目内 Qwen3.5-27B（sglang 模式），DSH 多轮走已 CPU 验证的 custom generate 路径，OPD 与 GRPO 叠加对比。全部使用现有固定组件，无自建。
2. **Ouro-2.6B 作为 student**：字面上的「LoopLM + OPD」，rollout 可行（vLLM 原生 ouro 支持），但训练栈未验证（slime/Megatron 大概率不支持 ouro 自定义循环架构；TRL/PEFT 可行性需先探）。风险中高，需先做可行性检查再谈实施。
3. **Ouro-2.6B 作为对照基线**：在方案 1 之上，把 Ouro-2.6B-Thinking 用 vLLM 起服务、同一 DSH 管线测 zero-shot，作为评估表中的 LoopLM 对照行（引用 2608.18171 的评测结论做背景）。

**方向修订（2026-09-10 用户确认，晚于同日定稿）：主线 = 纯业界复制（dense 9B student + slime 原生 OPD + 27B teacher + DSH，即官方 Qwen3-8B←32B 配方同款）；LoopLM 循环层降级为加分实验臂**——保留 E1 全部资产，恢复训练前先跑 ≤8 GPU·h 限额试点，试点结束时循环 student 低于 dense 参照（96.9%）的 50% 即止损封存、主线照常交付，失败数据如实进报告。原方向定稿内容作为背景保留如下。** 原定稿：LoopLM 借鉴思想、自己训练；OPD 走 slime 原生管线。 具体含义：student 为借鉴 Ouro 共享权重循环思想的 Qwen3.5-9B 循环化模型（保留首尾各 4 层、中间 24 层共享为 2 循环、唯一层 20/32≈5.6B——即 P1 已完成手术并验证等价性的那套布局；GDN 状态是层内序列递归、不跨层，因此 compact 两过共享块与 expanded 32 层顺序执行在前向上严格等价，由 expanded 导出技巧与 563 tensor 核验背书）；训练不用自研 trainer，用固定 slime 的原生 `--use-opd`（teacher 走外部 SGLang 服务）+ GRPO 可叠加；DSH 多轮用已 CPU 验证的 custom generate 路径。P1 的 loopify 工作由此从「误读」转为该方向的既有地基（手术、等价性、0% 退化基线都直接复用）。

工程挂载点（2026-09-10 源码核验，全部存在于固定 slime `4c193f1f`）：`scripts/models/qwen3.5-9B.sh`（9B Megatron 模型参数现成）；`--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec` 插件机制（`slime_plugins/models/qwen3_5.py`，层定义可替换）；`--use-opd/--opd-type sglang/--opd-kl-coef/--opd-teacher-load`（OPD 参数齐全，sglang 模式 teacher 为外部服务、经 rm_url 取逐 token logprob，OPD 为叠加在任意 advantage estimator 上的可加 KL）；`freeze/only_train_params_name_list`（参数冻结选择器，`model_provider.py`）。

整体流程与数据地图（对标工业 OPD 训练循环：slime 官方配方 = Qwen3-8B student ← Qwen3-32B teacher 外部 SGLang 服务 + GRPO 叠加 OPD；SOD = 小模型上仅 OPD 有效；Thinking Machines 为方法源头）：

```
① Rollout：student 自己生成（E3: 代码/推理 prompt 单轮；E4: student 驱动 DSH 走 SWE 任务多轮）
② Teacher 打分：Qwen3.5-27B 独立 SGLang 服务（专职一张卡）对轨迹逐 token 算 logprob
③ 损失 = GRPO 优势项（E4: 测试通过=奖励）+ λ·KL(student‖teacher)   ← OPD=第②③步
④ Megatron 更新 compact 权重 → 同步回 expanded rollout 服务 → 下一轮   ← LoopLM=student 架构本身
```

LoopLM 位置 = 被训练模型（20 唯一层、共享块两过、≈5.6B；训练用 compact 形态、rollout 用 expanded 形态，等价性已验证）；OPD 位置 = 学习方式（on-policy：数据本质是 prompt/任务池，内容由 student 现生成、teacher 现打分）。数据地图：E3 prompt 池 = data/raw 的 R2E-Gym/Klear-66k/OpenHands 轨迹集（取任务描述）；E4 任务池 = SWE-Gym + SWE-smith（沿用既有环境验证与排除审计）；HumanEvalPlus = E3 恢复率标尺（只评测）；SWE-bench Verified = E5 官方成绩（只评测）；Ouro-2.6B-Thinking = E5 对照行。训练池与评测集互斥（沿用 exclusion-audit）。

实施计划（每步有验收，GPU 步骤逐次授权）：
**E1 进度（2026-09-10）：E1.a 前向等价性已通过**——`scripts/check_loop_equiv.py` 实测共享块两遍与展开 32 层 logits 逐位一致（max diff 0.0，sanity 三重通过，记录 `runtime/loop-agent/e1a-loop-equivalence.json`）；E1.b（Megatron 循环 spec/provider/权重转换）进行中。

- **E1 循环模型接入**：循环化 spec + GPT 子类（20 唯一层，共享块前向两过）；compact→mcore 权重转换；CPU 单测（compact 前向输出 == expanded 前向输出）。已知边界：循环 forward 需自定义 model provider 子类，是对 slime 的小补丁而非纯配置。
- **E2 rollout 通路**：训练态 compact 权重同步时映射回 expanded 布局（P1 的展开技巧），SGLang 当普通 Qwen3.5 serving（GPU 授权后小步 smoke）。
- **E3 OPD 恢复训练**：teacher = Qwen3.5-27B（2026-09-10 定案：E3/E4 全程唯一 teacher，外部 SGLang 服务专职一张卡；dense 9B 的 96.9% 仅作恢复率参照），目标从 P1 的 0% 恢复到该参照的 ≥90%（HumanEvalPlus base-check 同协议复测）。
- **E4 agent 阶段**：teacher 换 Qwen3.5-27B，DSH 多轮轨迹 + `--use-opd` 叠加 GRPO（SOD/TurnOPD 的步级/轮次加权作为后续改进的文献依据，首版先跑朴素 OPD）。
- **E5 评估发布**：冻结子集官方协议 + Ouro-2.6B-Thinking zero-shot 对照行（vLLM 原生 ouro 支持，本机可用）。

资源与风险（诚实清单，2026-09-10 预算更新为 2–4×H100）：分工 = 1 张 27B teacher 服务（54GB）+ 3 张 student 训练/rollout（5.6B 全参每卡 ~40GB，宽裕）；4 张卡下 27B 原版（~432GB）与 loopify-27B（~251GB）全参训练仍不可行（上游 27B RL 配方默认 32 卡），student 维持 loopify-9B；OPD 本身不增训练显存（teacher 是外部服务，KL 只是损失项）；权重同步映射与循环 provider 是两处小补丁；恢复训练效果无先例保证（RRT/LoopRPT 表明共享+恢复可行，但我们是 2×H100 缩小版）；Ouro 的自适应深度（early-exit）不在首版范围，作为文献引用的后续改进。
<!-- loop-agent-design:end -->

<!-- dense-mainline-cpuprep:start -->
**正式 SFT 在飞（2026-09-11 14:00 UTC 交接快照，详见 plan.json `formal_sft_run`）**：2×H100 TP2（GPU 0+5，UUID 已记）跑过滤后 Klear ≤16k 池（28,208 条），step 132/500，loss 0.289→0.177，~108s/步，**用户钉死 1 epoch = step 500 检查点落盘即停**（ETA ~2026-09-12 00:30 UTC），wandb [runs/46hg4r4a](https://wandb.ai/3120252125-/code-agent-dense-mainline/runs/46hg4r4a)。停后动作链已验证：scoped ray 清理→释放验证→mcore→HF 转换（官方工具，已在 smoke 检查点上全链路预演通过）→HumanEvalPlus 评估（对比 96.9% 基线）。**共享机 Ray 纪律**：独占端口 20379/28266，只按会话目录精确清理，严禁全局 ray stop/pkill。

**2026-09-11 14:36 UTC 接手会话更新（停训已自动化 + 阶段 2 CPU 队列完成）**：① **停训 watcher 已部署**（[stop_sft_formal_at_500.sh](scripts/stop_sft_formal_at_500.sh)，detached setsid + flock）：轮询 iter_0000500 完整性（`.metadata` + latest 文件 + 90s 尺寸稳定）→ scoped 清理（`/tmp/ray-sft-smoke` 会话目录匹配 + 传递子进程含 job driver——原脚本单层匹配漏 driver；实测命中 78 PID 含全部训练 actor，**他人 verl/ray 会话验证排除**）→ GPU 释放快照 → CPU 强制跑已验证的 HF 转换命令 → `runtime/sft-formal/stop-report.json`；训练早夭则记 FAILED_EARLY 不转换；GPU 评估/serving 不自动跑，留给操作者审阅 stop-report 后执行。② **held-out NLL 资产就绪**：[eval_sft_nll.py](scripts/eval_sft_nll.py) 两阶段（prepare=CPU/train-rl 环境，score=GPU/runtime vLLM 0.19.1 `prompt_logprobs`；loss mask 直接复用训练同款 slime `qwen3_5` 生成器，独立 import 不动 vendor）；评估切片 `data/sft-pool/eval-nll/`（klear>16k / openhands≤16k / r2e≤16k 各 200 行 + 训练集参照 200 行，全部为正式训练构造性排除的行）；prepare 已完成（800 行，0 mask 异常）。**发现**：2/800 行 token 计数与 09-10 标注差 +108/+10——transformers 升 5.12.1 后 `apply_chat_template` 在罕见行上的漂移；良性（标注只用于 ≤16k 过滤，两行都远离边界；训练 parquet 本身 0 不匹配）。③ **RL 任务清单冻结 v1**（[freeze_rl_tasklist.py](scripts/freeze_rl_tasklist.py) → [rl-task-freeze-v1.json](configs/agent-rl/rl-task-freeze-v1.json)）：独立复算排除——SWE-Gym 2,438→2,357 eligible（61 旧审计 blocked + 20 held-out，Verified/full-test 零交集）；SWE-smith 59,133 eligible（剔 3 冻结修复任务）；确定性 round 顺序文件 + sha256 已落 `data/agent-rl/`；**每任务环境验证仍是 rollout 前置**（本机 Docker 平台层缺口未解）。④ DSH 生命周期复检 `--harness-check` **7 passed**（钉住树未漂移）。⑤ **Polar B' 数据集重试结论：上游阻塞而非本机网络**——HF 直连已恢复（HTTP 200；平台代理反而不通），但论文（arXiv 2605.24220 快照核对）给出的确切地址 `nvidia/polar-swegym-pi-qwen35-122b-a10b-trajectories` 无认证 401（API+页面均拒），ProRL-Agent-Server 的 GitHub README 也未链接；需用户 HF 账号接受条款+token 或等上游公开，论文事实仍以快照为准。⑥ **停训后评估已收敛为一键 runbook**：[eval_sft_formal.sh](scripts/eval_sft_formal.sh) `<gpu-uuid>`——预检 stop-report → GPU 分配滚动记录 → 以 P1 完全相同的服务参数（:18092）serve hf-iter500 → HumanEvalPlus 前 32 题对比 96.9% → held-out NLL 双模型打分 → 释放验证。

**前沿基模技术报告调研（2026-09-11 第四轮，用户指令：关注前沿模型与基模技术报告）。** 三份 TR 快照于 [research/sources/base-model-trs-20260911/](research/sources/base-model-trs-20260911/manifest.json)。前沿版图：GLM-5.3（最新开源 coding SOTA，保持 5.2 底座纯靠后训练 +50% Z.ai Code Bench）、GLM-5.2（753B MoE/1M 上下文/2026-06）、Qwen3.8-Max（2026-08，尚无完整 TR）、DeepSeek-V4/V4.1-Flash、Kimi-K2.5/K2.6；公开方法论最深的正是快照的三份。

*对本项目最重要的发现（DeepSeek-V4 TR，arXiv 2606.19348，DeepSeek-AI——DSH 的东家）：*「**混合 RL 阶段被完全替换为 On-Policy Distillation**」——领域专家用 SFT+GRPO（领域定制奖励模型，含生成式奖励模型）训练，统一模型经 **OPD（reverse-KL）整合**；配套「全词表 OPD 的高效教师调度」（教师 ZeRO 式分片按需加载、**缓存最后层隐藏态而非全词表 logits**）、DSec 沙箱（全局有序轨迹日志、抢占安全恢复=缓存重放免重执行、确定性重放）、序列长度课程 4K→16K→64K→1M。**我们的三段管线（SFT→agentic GRPO→OPD）与 DSV4 的「专家 GRPO + OPD 整合」同族——OPD 作为核心卖点获得了最强业界背书。**

*GLM-5 TR（arXiv 2602.15763，Z.ai——slime 的东家）*：后训练顺序 SFT→推理 RL→Agentic RL→通用 RL→**On-Policy 跨阶段蒸馏**（防遗忘，advantage=sg[log(π_t/π_θ)]——又一个 OPD 收尾的先例）；SWE 环境 1 万+（RepoLaunch 自动建环境+从日志抽 F2P/P2P）、终端环境 Harbor 格式三段构建（Docker 成功率>90%）；奖励=测试通过二值+噪声过滤（**环境崩溃样本剔除；组内有效样本过半则重复填充、否则整组丢弃**）；**TITO Gateway token 忠实轨迹**（与 Polar、我们的 slime_dsh 三处独立印证）；GRPO+IcePop 去KL、ε 0.2/0.28；SFT 中**错误轨迹段保留但 loss 掩码**（学纠错不强化错误）；异步 RL 每 K 步推权重+重置优化器。

*Qwen3-Coder-Next TR（arXiv 2603.00729，Qwen team）*：80A3 开源 coding agent=小算力打大仗的模板——80.7 万 PR 挖掘 + 85.2 万合成可验证任务（全 Docker 化）；SFT 轨迹=480B teacher 在 **6 种 harness**（SWE-agent/mini-swe-agent/OpenHands/Claude-Code/Qwen-Code/Terminus）生成+验证器代理+成对评审过滤（多 harness 轨迹+重过滤——弱化了「单格式纯度」的必要性，过滤才是关键）；多轮 RL=完成奖励+未完成惩罚+无效工具调用 token 级惩罚+防作弊（禁止 git/curl/wget 携带 repo 链接）；262k 上下文 BFP 打包；RL 使平均轮数 50→130；SWE-bench Verified 70.6-71.3。**评估对照行现成**：GLM-4.7 74.2 / Kimi-K2.5 73.2 / DeepSeek-V3.2 70.2 / Claude-Opus-4.5 78-79。

*对我们管线的具体修订：*① OPD 阶段定位升级——从「叠加项」升为「与 DSV4 同族的核心设计」，讲法从「GRPO+KL 叠加」调整为「专家 RL+OPD 整合」的完整叙事；② SFT 轨迹过滤提权（呼应 SWE-Prime）：正式 SFT 前对 Klear 池做规则过滤（GLM-5 的错误段掩码思想 + Qwen3CN 的规则过滤：缺终止信号/失败任务/畸形工具调用剔除）；③ agent RL 奖励设计参照三家的完整清单（测试通过+未完成惩罚+无效调用惩罚+环境崩溃剔除+组内填充规则）；④ 16k 长度的业界解法是上下文课程+大卡，2 卡 16k 的工程妥协（TP2 或 8k 池）按届时卡数定。

**阶段② GRPO 奖励栈综合设计已成文（2026-09-12，待用户过目后实现）：** [research/reward-design-stage2-grpo.md](../research/reward-design-stage2-grpo.md)——三份 TR 奖励机制逐字引文（Qwen3CN 未完成惩罚+工具格式 token 级惩罚+防作弊拦截器；GLM-5 环境崩溃剔除+组填充/丢弃+只训模型 token；DSV4 专家 GRPO→OPD 整合定位）映射到钉住 slime 的原生挂载点（custom-rm / rollout-sample-filter / rollout-all-samples-process / custom-advantage 钩子，均带源码行号）；唯一实现缺口 = GLM-5 组修复语义需自写（remove_sample 不改变 advantage 归一化参与）；smoke 首轮建议纯二值奖励，惩罚项作消融臂。变更注记 2026-09-12：新建该文档并在此挂指针，因用户要求奖励综合设计单独成文过目，research-log 段落只承载调研摘要。
**同日增补（用户指令「调研业界的方案」）：** 该文档新增 §4 业界开源实现对照。仍有效的首轮事实：① **slime 原生实现了 DAPO 动态采样**（`--over-sampling-batch-size` + `--dynamic-sampling-filter-path`，自带 check_reward_nonzero_std 过滤器，vendor/slime 可查）——组卫生的零成本选项；② 格式教学的业界先例是 NeMo-RL Stage-1 pivot（PivotRL，arXiv 2603.21383，单步参数匹配奖励 0.2→0.55）与 AGL 反馈修正；③ 未完成惩罚的开源形态 = DAPO 软超长惩罚（arXiv 2503.14476，verl overlong_buffer）与 AGL 已解轮数惩罚（t0=80/λ=0.1，快照 smith_agent.py:656）；④ AGL 防作弊四通道+网络层 default-deny 为权威的立场（本地网络隔离已天然满足）。
**同日三轮收敛（用户指令：只要大厂、要有时效性的有价值内容）：** 权威源扩为五家基模 TR+一篇实证标尺，设计文档 §1.4-1.6 新增逐字引文：**Kimi K2.5 TR**（arXiv 2602.02276 §4.4.2：规则 outcome 奖+budget-control token 效率奖+GRM 细粒度多 rubric；Toggle 交替相 −25~30% 输出 token；PARL 多代理奖励 λ 退火）；**MiniMax M2 TR**（2605.26494 §6.1.5 复合奖励 α·过程+β·速度+性能，过程含工具格式错误罚）**+M2.1 第一方后训练博客**（F2P/P2P 任务类型变体；CISPO 截 IS 权重不截 token；MIS+PPO 轨迹过滤治理环境噪声长尾；FP32 LM head；多 scaffold）；**Meta ScaleRL**（2510.13786，40 万 GPU 时：PipelineRL 8 步 off-policy+CISPO+FP32 logits+DAPO 动态采样+自适应 prompt 过滤——slime 原生 cispo/dynamic-sampling 均在获胜配方内）。时效核查：GLM-5.3/K2.6/Qwen3.8-Max 无训练细节披露；DSV4.1-Flash 官方卡=沿用 DSV4 配方无算法改动。§5 决策点重排：塑形四档 a-d（b 档=效率塑形升为 2026 TR 新一等公民）、噪声治理三机制可选、estimator 给 grpo/cispo 两选项。 **同日二轮修正（用户驳回「smoke 纯二值」结论后扩大调研）：** 首轮「纯二值是开源主流」系样本不足的过度概括，**作废**。修正后的图景（设计文档 §4.1 已重写为两阵营）：**基模大厂 TR 无一是纯二值**——新核实的 MiniMax M2 TR（arXiv 2605.26494 §6.1.5）用复合奖励（过程奖励含工具格式错误罚+墙钟完成时间奖励+reward-to-go 基线）；Meta SWE-RL（arXiv 2502.18449，代码开源）完全不用测试、以补丁相似度+格式罚 −1 为奖励；DSV4 按模式施长度罚。开源小算力实现（slime 官方/DeepSWE/NeMo-RL/SkyRL）跑二值核+组卫生。澄清：R2E-Gym 论文的「Hybrid Verifier」是推理期重排序、训练为 SFT，不构成非二值 RL 证据。学术对照：2605.02944（pass-rate 部分分不可靠更优）、2605.05112（二值信号在 ~50% 通过率最强→难度过滤优先）。决策点改为「核+塑形分级起步」（文档 §5.1 选项 a-d，先测量格式错误率/轮数/崩溃率再加码）。

**第五轮：code-agent 项目对照与落地路线（2026-09-12 10:2x UTC，用户指令「调研 code agent 项目，看我们到底怎么做」）。** 核心发现：2026 下半年形成了完整的 **harness-native RL** 研究脉络，而我们正是其中一员（DSH 经 slime_dsh 适配层）：

- **LEGO-RL**（*LEGO-RL: Harness-Native Reinforcement Learning for Coding Agents*，arXiv 2608.17393，2026-08）——**与我们最同源**：未修改 harness 内训练（OpenHands SDK/Claude Code/OpenCode 分别训，Qwen3.5-35B-A3B——与我们 9B 同家族）。任务漏斗 = 36,884 OpenSWE 候选→静态规则→构建/验证器→**rollout 难度筛选（Qwen3.6-27B，只留 4 试解 1-3 次的任务带）**→2,699 任务；**未筛选任务 72.7% 从未被解出**（难度筛选决定性证据）。奖励 = 单值二值、明确声明无中间信用是局限；GRPO 式组相对 8 rollout/题 + **GSPO 序列级代理**（KL 1e-3 在 loss 内）；infra 失败轨迹权重 0 但留批内。token 保真：rollout-训练 Pearson 中位 ≥0.998（模型 API 处 in-process proxy + 消息级历史对齐）。结果：各 harness +5.8~9.4 SWE-bench Verified，且「一个控制流下获得的增益换个 harness 不保真」（KAT 在 OpenHands 下 −0.4）——**harness 内训练正当性的直接证据**。工程：agent 执行占 91.3% 墙钟、async 2.5×、预构建镜像 33×。框架/检查点/任务索引全开源。
- **OpenForgeRL**（arXiv 2607.21557）：harness-native 开源框架（proxy+veRL+K8s）；**多 harness 混训（3 个）比单 harness 在其自身 bench 上还强（48.5%）**；harness 选择本身是训练变量。
- **EvoHarness-RL**（2608.05446）、**ClawGym II**（2608.16798）同脉络。
- **OpenThoughts-Agent**（arXiv 2606.24855 + openthoughts.ai/blog/agent，2026-09-12 抓取）：全开源配方。SFT ~15k 轨迹（NL2Bash+InferredBugs；**教师发现：GLM-4.6 当 teacher 比 GPT 系好 ~2 倍**，teacher 的 bench 分数不预测数据质量）；RL 从 1 万生成任务**只留 ~700**（GPT-5-Codex 零分任务全丢）；栈 = SkyRL+Harbor；RL 增益温和（TB-Dev +2%、SWE-bench V +1%）；300+ 模型消融式工作流。
- **MiniMax Forge**：未开源（仅模型+架构文档：Gateway+DataPool 中间层解耦 agent 与训练/推理，与 Polar/slime_dsh 同模式；树合并+前缀树缓存 ~40×；reward-to-go）。
- **slime 上游动向**：钉住 commit（9/9）后仍活跃——v0.3.2「完全对齐 GLM-5 训练」、multi_agent 示例、14+ 示例；钉住策略不变，升级决策留到阶段② smoke 后。
- **DSH 训练先例核实**：检索「DeepSeek Harness + RL 训练」无发表——**DSH-harness RL 训练生态位仍空**，差异化声明成立。
- **harness 影响 20 分**（yage.ai 2026-08：同一模型在 DSH 上 bench 波动 20 点）——harness-native 训练动机背书。

**第六轮：过程奖励 vs 惩罚专项（2026-09-12，用户问题「只罚不奖对吗」）：** 确认设计=终止二值+规则化负向惩罚，无正向过程奖励。依据：① 五家 TR 中四家塑形形态就是「只罚不奖」（Qwen3CN/K2.5 budget 罚/DAPO 超长罚/DSV4 长度罚），含正向过程项的仅 MiniMax M2（系数未公开、无消融）；② 风险不对称——规则惩罚不可 hack，正向过程奖需 PRM（新 hacking 面+预算装不下）；③ 管线分工——密集正向监督已由阶段③ OPD 承担（teacher 逐 token KL，DSV4 同族）；④ ToolRL（NeurIPS'25 2504.13958）消融：format 奖励前 ~30 步饱和——格式是快技能，格式惩罚应为短命机制（设计文档 §5.1 选项 c 已加退场条件）。学术 PRM 线（AgentPRM 2511.08325 / SWE-TRACE 2604.14820 / PaTR 2607.15610 +5.0 SWE-bench / implicit step rewards ICLR26）入档为 v2/LoopLM 臂引用项，不入首轮。设计文档新增 §4.7。

**第七轮：遗漏排查收口（2026-09-12，用户指令「查有没有遗漏的技术文档」）：** 补入 **KAT-Coder-V2.5 TR**（arXiv 2607.05471，快手 Kwaipilot，SV 79.6%——设计文档 §1.7，权威级别待用户裁定）：三层规则奖励（二值核心+轨迹级行为罚+**失败轨迹部分分**：单测子集正分/检索 F2）+GRM；**PPO+GAE 替 GRPO** 以获得逐步负反馈通道+事后信息非对称 critic；沙盒可靠性工程（故障 16%→<2%、训练崩溃÷10）；网关 /generate（200 轮样本 40% 重分词漂移）；防作弊剥 git 历史+结构化解析测试输出+捷径轨迹过滤。**修正**：§4.6/§5 部分分结论从「无人采用」改为「争议项」（2605.02944 反例 vs KAT 正例），v1 仍不引入、留作方差过低备选。同轮补录 AgenticQwen（2604.21590 小模型多轮 RL）、Devstral（2509.25193）、综述两篇（2509.02547/2604.09459）。核实 Qwen3.5 无文本版 TR（仅 Omni 2604.15804）。**排查结论：基模大厂一手 TR 层无其他遗漏。**

**决策叙事文档（2026-09-12，用户指令：不要文献卡式文档，要从项目开始记录自己的选择以展示思考）：** 新建 [docs/decision-log.md](decision-log.md)——D1-D14 决策链（每条：背景→备选→裁决与依据→验证/状态），覆盖开局选型（自建→AGL→slime 的否决链）、SFT 四决策（含跨格式特异性的自我验证）、阶段②三决策（本地环境/任务冻结/直上 DSH）、奖励七轮裁决链（含两次自我纠错如实保留）、难度筛选与 estimator 现行决定、OPD 定位；附 PPT 骨架映射。code-agent-tr-analysis-202604plus.md 角色调整为参照附录（决策文档为其挂引用）。PROGRESS 顶部单入口同步。

**第八轮：2026-04+ code-agent TR 综合分析成文（用户指令：模型须 2026-04 往后、分析落文档供 PPT）：** 新建 [research/code-agent-tr-analysis-202604plus.md](../research/code-agent-tr-analysis-202604plus.md)——入选 6 份（KAT-Coder-V2.5 2607.05471 / DeepSeek-V4 2606.19348 / MiniMax M2 2605.26494 / OpenThoughts-Agent 2606.24855 / AgenticQwen 2604.21590 / LEGO-RL 2608.17393），每家一张分析卡+奖励/环境/算法三张横向对比表+共识分歧页+项目映射页（含 PPT 素材索引）。出局核查：Devstral 2=2025-12、Qwen3.5 无文本 TR、K2.6/GLM-5.3/Qwen3.8 无训练披露、闭源榜单新贵无 TR。新建文档理由：跨 TR 综合分析+演示素材是独立交付物，research-log 只管时序流水。

**对我们管线的直接裁决（我们到底怎么做）**：① **任务难度筛选是下一个必做动作**——LEGO-RL/OpenThoughts/Qwen3CN 三家独立同款（LEGO-RL 未筛任务 72.7% 从未解出）；qualification（环境有效性）完成后接难度带筛选（27B teacher 在项目内可跑筛选 rollout，或 SFT-9B 自筛更省卡更贴 on-policy），此动作在奖励塑形之前。② harness 接线模式已收敛为行业标准（模型 API 处 proxy+token 保真+消息级对齐），slime_dsh 就在其上，不改架构。③ 单 harness（DSH）训练正当（LEGO-RL harness 特异性证据），多 harness 混训记为远期消融。④ estimator 选项集更新为 grpo/cispo/**gspo**（三者都 slime 原生，分别有五家 TR 默认/ScaleRL+MiniMax/LEGO-RL 背书）。⑤ 教师质量>教师 bench 分数（OpenThoughts 发现）——阶段③ 27B teacher 选择有先例支撑。⑥ 小算力生态位（2-4 卡+OPD+DSH）仍无公开占据者。

**SFT 冷启动 smoke 完成（2026-09-11 08:41 UTC，GPU 0+6，已释放验证）。** 三段管线的第一次真实训练跑通：2×H100 TP1/DP2 + 优化器 CPU 卸载，Klear-only 8k 池 256 行切片，3 个优化器步，**loss 0.296→0.196**，torch_dist 检查点落盘（models/dense-9B-sft/smoke，117G 含优化器态），wandb 全程在线（项目 code-agent-dense-mainline / 组 dense-9b-sft-smoke，run dp0hxmty）。用后释放验证：GPU 0/6 均回 0 MiB。

**环境收敛的关键转折（用户纠偏：「先看别人有没有实现」）**：上游仓库自带官方非 Docker 环境脚本 [build_conda.sh](vendor/slime/build_conda.sh)——此前手工拼装反复踩坑的每一步它都钉死了：官方**明确卸载 flash-attn-4**（sglang 会拖入、与 TE 2.16 验证栈不兼容）改装 **flash-attn 2.8.3 指定预编译 wheel（sha256 钉死，经 gh-proxy 镜像下载并按官方摘要校验通过）**；fla 钉 0.4.2；triton 需 ≥3.7.1（Hopper 上 fla 旧核有错误结果护栏）。最终栈：torch 2.11.0+cu129（上交镜像，与 sglang 0.5.15.post1 的硬钉一致）+ TE 2.16.1（预编译 cu12 核心 + sm90 源码编译 torch 绑定）+ sglang 0.5.15.post1（PyPI wheel --no-deps，绕开其与 torch cu129 wheel 的 cuda-bindings 13 元数据冲突）+ transformers 5.12.1 + wandb 0.19.10（新版移除了 slime 用的 util.generate_id）+ numpy 1.26.4 + torch-memory-saver + flash-attn 2.8.3 + fla 0.4.2 + triton 3.7.1。**尚未装**（rollout 阶段前补）：sglang-kernel==0.4.4、sgl-deep-gemm==0.1.4、torchaudio、cudnn 9.16、apex（编译）、tilelang。

**共享机 Ray 纪律事故与修复（用户转达提醒，2026-09-11）**：此前启动脚本含全局 `ray stop --force` 且本人多次手动执行——在共享机上这会杀掉其他用户的 ray 集群（项目纪律本就写明不执行上游示例的全局 ray stop，属于违规操作，已如实记录）。修复：[run_sft_smoke.sh](scripts/run_sft_smoke.sh) 改为**独立 GCS 端口 16379 + 独立 dashboard 端口 8266 + RAY_ADDRESS**，清理只按本会话临时目录（/tmp/ray-sft-smoke）匹配 PID 精确杀——已验证清理后他人 142 个 ray 进程不受影响。当前正在跑的旧会话也以同样方式做了精确清理。

**开源项目全景调研（2026-09-10 第三轮，用户要求：调研好的开源项目）。** 七个来源快照在 [research/sources/opensource-landscape-20260910/](research/sources/opensource-landscape-20260910/manifest.json)。按「能给我们什么」组织：

*格式问题（A/B/C 决策）出现两个新解法：*
- **Polar**（*Agentic RL on Any Harness at Scale*，arXiv 2605.24220，NVIDIA）：网关代理放在 harness 与推理后端之间，逐请求记录 token 级轨迹（prompt/response token IDs、logprobs、finish reason），只把真实采样的 assistant token 设为可训练、harness 插入内容全部 loss-mask——**部署 harness 本身就是训练环境，train/test 格式天然一致**。Qwen3.5-4B 在 SWE-Gym 上 GRPO 后 SWE-bench Verified 3.8%→26.4%（Codex harness，+22.6）；同方法在 Qwen Code harness 上几乎无增益（34.6→35.2）——格式熟悉度决定增益大小，反过来印证 PIPE。开源了服务端代码（NVIDIA-NeMo/ProRL-Agent-Server，NeMo Gym 环境）与 **122B teacher 的 SWE-Gym 工具调用格式轨迹数据集**（Apache-2.0；HF 页面代理超时未快照，事实以论文快照为准，用前需下载核验）。我们的 `slime_dsh` 适配层（CPU 已验证 token/logprob/loss-mask 保全）与 Polar 网关是同一模式的项目内实现，Polar 提供了大规模参照与参考代码。
- **NeMo-RL 两段式 SWE 指南**（NVIDIA docs，Qwen3-30B-A3B）：**Stage-1「pivot」= 无沙箱的廉价单步 RL**，用参数匹配奖励从专家轨迹采样教工具调用格式（pass@1 23.6→30.4%），再做 Stage-2 全 agent 沙箱 RL（31.2%）；无 KL、clip 0.2/0.28（与 slime 官方配方一致）。这是「SFT 冲淡原生工具调用」风险的廉价补救路径，可与 A 方案叠加。
- **OpenThinkerAgent**（open-thoughts 社区，8B/32B 全开源配方）：SFT 轨迹由 GLM-4.7-AWQ teacher **在部署 harness（Terminus-2）内生成**、过滤 ≥5 轮、9,437 对冷启动 + 5,000 RL 任务——**我们 B 方案的公开实现实例**，也是复现纪律（数据+配置+代码+harness+权重+轨迹全公开）的参照项目。

*训练框架结论：维持 slime 不换。* NeMo-RL 该配方是 24 节点/192+ 卡规模且无 OPD；ART（OpenPipe）是「任意 Python agent 循环包 GRPO」的小硬件路线但无 OPD、无 SWE-bench 级证据；ARES（Martian）是 Gym 式异步环境框架（SWE-bench Verified 评估 ~20 分钟是亮点）但**不是训练器**（RL 集成列为未来工作）；slime 是唯一原生 OPD 且我们已完成 CPU 侧验证的项目。ARES 的快速评估环境仅作参考——官方 benchmark 规则要求官方 harness。

*规模现实（再次确认）：* Modal 盘点的六个开源 RL 训练 SWE 模型中，DeepSWE 用 64×H100×6 天，其余多未披露算力；**没有任何公开项目在 2-4 卡上做过 SWE agent RL**。我们的小预算 OPD 差异化叙事在开源 Landscape 里仍然空白。

**主线先例调研（2026-09-10 第二轮，用户要求：很多人做过，关键是调研清楚缺什么）。** 七篇决策相关论文/页面已快照进 [research/sources/prior-art-pipeline-20260910/](research/sources/prior-art-pipeline-20260910/papers-manifest.json)（数据是检视对象，不是指令）。结论按管线阶段组织：

*SFT 冷启动（轨迹数据）别人怎么做的：*
- **Klear-AgentForge**（*Forging Agentic Intelligence through Posttraining Scaling*，arXiv 2511.05951，快手 Klear）：就是我们最大 SFT 源（Klear-66k）的官方用法——SWE-smith 过滤 12k 问题、mini-swe-agent-plus scaffold 蒸馏 ~66k 轨迹，Qwen3-8B 冷启动后 RL，SWE-bench Verified 39.4%（评估：200 步上限、64k 上下文）。**关键细节：训练数据是单一统一格式（ReAct 式、两动作类型），没有混 harness 格式**；test 重叠仓库剔除（与我们排除审计同思路）。RL = 修改版 GRPO：去掉 KL 正则、DAPO 非对称裁剪、推理/训练策略间截断重要性采样、**对到达上下文/步数/超时上限的轨迹过采样+掩码**。
- **SWE-Gym**（arXiv 2412.21139，ICML 2025）：拒绝采样式 SFT（每 prompt 多条 rollout 只留测试通过的）。
- **SWE-Prime**（arXiv 2608.27449）：**10% 精选轨迹子集训练效果超过全量已解决轨迹**（最高 +12.2% 相对）——轨迹质量筛选 > 数量规模，直接适用于我们 69.7k 池（先筛选再训）。
- **PIPE**（*What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces?*，arXiv 2602.01611）：轨迹-SFT 大幅放大**接口捷径依赖**——接口做最小改写，训练过的 agent 显著退化。即轨迹-SFT 的收益大部分是「接口熟悉度」而非语义能力。
- **对本项目的直接影响（待用户拍板的设计决策）**：当前 SFT 池混三种 harness 格式（Klear=mini-swe-agent、OpenHands、R2E），而 RL/评估阶段的 agent 是 DSH（第四种接口）。文献安全做法是 SFT 格式与 rollout harness 一致（NVIDIA Open-SWE-Traces arXiv 2606.16038 专门研究 cross-harness 泛化；Klear/DeepSWE/SWE-Gym 全部是同格式训+评）。候选方案：**A)** SFT 只用 Klear-66k 单格式（官方配方的保守复刻，接受 RL 阶段向 DSH 的接口迁移）；**B)** 用 27B teacher 在 DSH scaffold 内对 SWE-Gym/SWE-smith 任务做拒绝采样，自产 DSH 格式轨迹再 SFT（AgentForge/SWE-Gym 的蒸馏做法；前置=任务环境就绪+teacher rollout 成本）；**C)** 维持三格式混池（无文献支持，PIPE 指出风险）。默认建议 A 起步（零额外成本、官方用法）、B 作为环境就绪后的增强。

*Agentic RL（SWE 任务 GRPO）别人怎么做的：*
- **DeepSWE**（Together AI/Agentica，arXiv 2508.17146）：Qwen3-32B 纯 GRPO 200 步 → SWE-bench Verified 59.0%；GRPO 长度偏差修正（surrogate loss 除以 max context length）；**上下文 16k→128k 消融显示 >32k 的增益只有 ~2%**——对我们 2-4×H100 预算是好消息（32k 上下文接近饱和点）。
- **长上下文 SWE agents**（arXiv 2508.03501，SWE-Gym 团队）：拒绝 SFT → RL 两段；RL 从 65k 上下文起跑；R2E-Gym 轨迹 32-64 agent 步。
- **slime 官方 SWE RL 配方**（本地 vendor/slime/examples/coding_agent_rl，8 节点×8 卡，35B-A3B）：eps-clip 0.2/0.28（即 DAPO Clip-Higher，与 AgentForge 一致）、kl-loss-coef 0（同 AgentForge 去 KL）、96k 上下文+32k 生成长度、colocate+CPU 优化器卸载+精度感知优化器、harness 侧 80k auto-compact 先于训练侧上限截断、rollout batch 8×8 采样、agent 时间预算 1800s/评估 600s、E2B sandbox。
- **规模现实（诚实）**：公开成功案例的最小配置是 8 节点×8 卡级别（8B 模型的 RL 也用 64×H800）。我们 2-4×H100 上的 9B RL 远低于公开实践，预期小 batch、少步数、长墙钟；OPD 的样本效率补偿正是我们的差异化卖点（SOD arXiv 2605.07725：0.6B 上 GRPO 打不过 vanilla、OPD 有效）。

*OPD 叠加的已知风险：*
- **OPD 综述**（*A Survey of On-Policy Distillation for LLMs*，arXiv 2604.00626）：激进 reverse-KL 蒸馏让学生 Pass@1（精度）上升但 **Pass@k（多样性/召回）灾难性下降**——与 GRPO 依赖组内奖励方差的机制直接冲突。含义：`--opd-kl-coef` 不能默认 1.0 一路到底，首轮后要按组内奖励方差监控调参；这是训练参数，任何变更先经用户同意。

*评估层硬性要求（提前暴露的缺口）：*
- **SWE-bench Verified 官方 harness 强制 Docker**（x86_64、≥120GB 存储、16GB RAM、8 核；入口 `swebench.harness.run_evaluation`）。本机 platform docker-proxy 的缓存镜像探针曾在容器创建前失败（`cannot use job_queue`，2026-09-09 记录）——**这是 GPU 阶段之前必须解决的平台层缺口**（用户规则：容器问题在执行平台层解决，不得自建替代评估器）。Epoch AI 有优化镜像 62 分钟跑全量 Verified，但用第三方镜像偏离官方协议，保守路线仍是官方 harness+冻结子集。
- slime 无官方 2-4 卡配方；单节点 8 卡可行、更小规模无公开先例——我们的草稿参数是合理推断，首次 GPU smoke 必须先验证可行性。

**主线 CPU 准备完成（2026-09-10，等 GPU 授权）。** 按已定稿三段管线（dense Qwen3.5-9B student：SFT 冷启动 → Agentic RL → OPD）完成数据池、配置草稿与论文快照，全部 CPU 侧、项目内存储：

- **SFT 轨迹池**：`data/sft-pool/sft-trajectories-000..008.parquet`（9 片×8000 行，共 69,705 条多轮轨迹）。来源=Klear-66k 65,994 + OpenHands 480 + R2E-Gym 3,231；排除审计已执行（20 个 held-out SWE-Gym 任务、旧审计 81 个 blocked ID、3 个冻结 SWE-smith 任务、SWE-bench Verified 500 个 instance_id 直查；OpenHands 轨迹无 instance_id 列，经首条 user message 与 SWE-Gym problem_statement 归一化前缀匹配全部恢复身份，5 条 held-out + 6 条旧审计排除，0 条无法识别）。schema=`messages`+`metadata(source,instance_id,task_sha256,n_turns,n_tokens)`；分片原因是单文件 messages 列解码超 parquet 2GB 数组上限，slime read_file 单文件入口需要小片。长度画像（chat template 全轨迹）：p50≈19k / p90≈30k / p99≈42k，40%≤16k、93.5%≤32k token。证据：[prompt-pool-audit.json](runtime/prompt-pool/prompt-pool-audit.json)。
- **OPD prompt 池**：`data/opd-pool/opd-prompts.parquet`（13,255 个去重任务描述，首条 user message 原文）；长度 p50=1442 / p90=1591 / max=2609 token。角色=无环境 OPD 冒烟与参照池；正式 agent-OPD 阶段仍用 SWE-Gym/SWE-smith 任务环境。
- **训练子集工具**：`scripts/make_sft_trainset.py --max-tokens 16384` 产出长度有界冷启动训练集（默认 16k 上限，`sft_rollout` 不做长度过滤，过滤必须在池侧完成）。**已产出 `data/sft-pool/train-16k.parquet`（2026-09-10）**：30,666 条 ≤16k 轨迹（Klear 28,231 / OpenHands 238 / R2E 2,197）；全池 69,705 条已逐条标注 `metadata.n_tokens`（直方图见 audit），标注计数与 slime `qwen3_5` mask 生成器的实际 tokenization 逐条相等（交叉验证）。
- **slime 配置草稿**（默认只打印、`--launch` 强制 exit 2、校验 vendor/slime pin+干净树）：`scripts/train_sft_coldstart.sh`（阶段1：官方 retool SFT 配方结构 + `slime.rollout.sft_rollout` + `--loss-type sft_loss`）；`scripts/train_opd_stage.sh`（阶段3：官方 on_policy_distillation 配方结构 + `--use-opd --opd-type sglang --opd-kl-coef 1.0`，OPD_MODE=prompt-smoke|agent 两态，agent 态注明 DSH/环境接线缺口）。超参均为官方示例默认值，正式运行前需用户复核。机器可读管线记录：[dense-mainline-pipeline.json](configs/slime/dense-mainline-pipeline.json)。
- **CPU 验收（固定 slime 真实代码路径，非 fake）**：[cpu-acceptance.json](runtime/prompt-pool/cpu-acceptance.json)——① OPD 池经 `slime.utils.data.Dataset` 以 `--input-key messages --apply-chat-template` 语义加载 13,235 条（4096 过滤 20 条），metadata 为 dict；② `MultiTurnLossMaskGenerator(qwen3_5)` 在真实轨迹上监督占比与 assistant 段字符占比一致（0.30-0.39），token 计数与池内标注逐条相等。**关键发现：SFT 必须显式 `--loss-mask-type qwen3_5`**，默认 `qwen` 的逐条消息渲染在本数据上抛 `No user query found` 模板错误（固定 slime 的 choices 里有专用 `qwen3_5` 类型，全对话渲染+offset 映射）。
- **首轮 GPU 前置缺口（2026-09-11 凌晨更新：本地 GPU 训练环境已建成，9B 转换已完成）。** 用户选定路径 b（本地建依赖）。**环境**：`.venv-train-rl`（torch 2.10.0+cu129）上补齐——TE 2.16.1 预编译 cu12 核心 wheel（自 tuna 镜像下载，sha256 对 PyPI 元数据校验通过；公共 PyPI 直连仅 13KB/s 不可用）+ `transformer_engine_torch` sdist 源码编译（仅 sm90 目标 1.5 分钟完成；TE 2.16.1 的 CUDA 12 构建目标恰为 12.3 = 系统 nvcc，无版本问题）+ flash-linear-attention 0.5.2 + numpy 降级 1.26.4（megatron 断言要求）+ [env-train-gpu.sh](scripts/env-train-gpu.sh)（venv 的 nvidia 库目录前置 LD_LIBRARY_PATH，防系统 /usr/local/cuda-12.3 的 cudart 遮蔽 12.9）。import 全链路通过（TE 绑定/fla/补丁版 megatron/slime/qwen3_5 插件）。apex 未装（mcore 有 torch 回退；TE norm 不依赖 apex；`--no-gradient-accumulation-fusion` 关掉对应断言，纯训练期优化不影响检查点）。**转换**：GPU 0（GPU-8c4bac00，UUID 已记 [gpu-allocation.json](configs/gpu-allocation.json)、用后验证释放 0 MiB）上以未修改上游转换器完成，exit 0，17G torch_dist release 检查点落盘。**DCP 元数据核验**：477 键、8×`linear_qkv.layer_norm_weight`（full-attention 层）、24×`self_attention.input_layernorm`+216×GDN 参数（线性注意力层）、32×`mlp.linear_fc1.layer_norm_weight`——TE 训练布局逐层对账无误（8+24=32）。WANDB_API_KEY 仍待用户提供；Polar B' 数据集待网络恢复。
- **论文快照**：探索章两篇新论文已快照进 [research/sources/looplm-exploratory-20260910/](research/sources/looplm-exploratory-20260910/papers-manifest.json)——*Training-Free Looped Transformers*（arXiv 2605.23872，Lizhang Chen 等；HTML 快照未含机构信息，引用前需补核）与 *Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers*（arXiv 2609.02702，Xu Zou、Jie Tang，Z.ai/清华）。
<!-- dense-mainline-cpuprep:end -->

<!-- loopify-historical:start -->
**[历史] 已废弃的 loopify 主线（2026-09-10 定稿、同日被用户否决）：** 此前设计为对 Qwen3.5-9B 做 RRT 式层共享手术（保留首尾各 4 层、中间 24 层 2 循环、唯一层 20/32≈5.6B 有效参数）。P1 已执行完毕：下载校验 19.33GB；`scripts/loopify_qwen.py` 手术产出 compact/expanded 双形态；563 tensor 逐位核验通过；expanded 被未改动 vLLM 0.19.1 直接 serving；零样本退化（HumanEvalPlus 前 32 题 base-check、贪心、同管线）：**dense 9B 31/32=96.9% vs 硬绑 loopified 0/32=0%**。该结果作为「硬绑层共享在无恢复训练时完全崩塌」的证据保留（与 RRT/LoopRPT 结论一致），证据链 [p1-degradation.json](runtime/loop-agent/p1-degradation.json)。废弃原因：该方向把 LoopLM 误解为「需要自建的东西」，用户澄清 LoopLM 指 Ouro 族现成模型，且当前优先级是调研而非自建工程。
<!-- loopify-historical:end -->

<!-- slime-migration:start -->
**当前主线：已按用户要求切换为 slime（2026-09-09 UTC）。** 默认工程入口 `bash scripts/agent_rl.sh` 已转到 [slime_rl.sh](scripts/slime_rl.sh)，不再进入原有 verl 自建 AgentLoop/数据验收流程。官方源码固定在 [vendor/slime](vendor/slime)，commit `4c193f1f37509cca70f0e88807a9305b70f63f4e`，工作树未修改。原入口与 readiness 原文保存在 [runtime/slime/legacy](runtime/slime/legacy)，旧依赖、数据和失败证据保留。下面先前的选型建议均为历史记录，以本段和 [architecture.json](configs/slime/architecture.json) 为准。

目标流程为 **上游任务数据/环境 → DSH → slime 原生 OpenAIAdapter → SGLang 采样 → slime 轨迹与奖励处理 → Megatron LoRA 更新 → 权重回传 SGLang**；Ray 的资源调度、样本调度、Sample/loss mask、任务编排和评分均采用 slime 原有实现。DSH 保留原 SDK/provider；项目接入范围限于 agent 生命周期和执行平台边界。该图表达目标职责，尚不表示端到端训练已运行。BF16、LoRA 和最多 2×H100 的约束没有改成上游示例的默认全参/32 卡配置。

已经执行的检查：

- 新建独立 `.venv-slime-cpu`（Python 3.12.12、Torch 2.13.0+cpu），按固定源码的 GitHub `agent-test` CPU 依赖清单安装并记录完整 [cpu-requirements.lock](configs/slime/cpu-requirements.lock)。上游 CI 用 Python 3.10，本地用 3.12 的原生 `asyncio.timeout`。未安装 Megatron/SGLang GPU 栈，也未复用 AGL/verl 作为训练依赖。
- 原版 `tests/test_agent` **64 passed、1 skipped**：[JUnit](runtime/slime/upstream-check-1/junit.xml)。覆盖 adapter、轨迹分支、harness、sandbox 执行协议及示例编排；模型、tokenizer、sandbox、CLI 边界使用上游 fake。跳过的是依赖 SGLang 的 Qwen reasoning parser 测试，不能据此声称 thinking 解析已验收。
- [真实 DSH SDK 协议检查](runtime/slime/dsh-latest.json)通过：DSH 向未修改的 slime OpenAIAdapter 请求并消费流式回复，Bearer session 路由正确，Sample 保留脚本模型的生成 ID、logprob 与 loss mask，DSH runtime 已关闭。该检查只有一次完成回复；tokenizer 和模型为上游 CPU fake，未运行真实 Qwen、多轮工具任务或训练。检查实现仅是 [check_slime_dsh.py](scripts/check_slime_dsh.py) 测试夹具，没有新增生产代理或调度器。

仍未完成的接入必须分开看：**DSH 尚未注册到上游 `BaseHarness`/`coding_agent_rl` 完整生命周期**；原版示例注册 Claude Code/Codex，使用 E2B-compatible sandbox，项目内工作区/受限进程执行后端尚未接入。下一步应通过其既有 Sandbox 接口解决环境，而非恢复旧自建任务编排或复制评分流程。完整 Qwen tokenizer、SGLang reasoning/tool parser、多轮生成 token 对齐也需另行验证。

**LoRA 训练仍是明确缺口。** [上游 PR #1865](https://github.com/THUDM/slime/pull/1865)在此次核对时仍为 open、未合并，本项目未采用该分支。当前固定主线没有经本项目验证的 Qwen3.5-27B BF16 LoRA 两卡配方；不会编造可用 CLI 参数或自动改全参训练。GPU 启动、实际更新、显存、权重同步与保存恢复均未验证，`--launch` 必须返回 2。完整缺口在 [readiness.json](configs/agent-rl/readiness.json)。

数据随架构切换重新区分资格：slime 协议下已验证的训练题为 **0**；此前 2 道 Dask 仅有历史自建协议资格，24 道扩池候选只有源码准备、未完成新增环境验收，6 道开发候选继续预留。历史 SWE-smith/SWE-Gym 和相关轨迹排除继续生效。首轮仍只是“2 题×2 轨迹、最多 1 次 LoRA 更新”的建议规模；未冻结为训练配置、未运行参数更新。

复核入口：`bash scripts/agent_rl.sh --status` 查看实际缺口，`--config` 只打印架构清单，`--check` 运行上游 CPU 测试，`--dsh-check` 运行上述真实 SDK/脚本模型检查。`--native-*`、`--training-*` 旧选项不再由默认入口接受。验收及文件哈希：[acceptance.json](runtime/slime/acceptance.json)。不执行上游示例启动脚本中的全局 `pkill`/`ray stop`；本轮未占 GPU、启动任务环境或运行 benchmark。
<!-- slime-migration:end -->

**DSH BaseHarness 接入检查（2026-09-10）。** 新增 `slime_dsh/` 作为上游 `BaseHarness` 的薄适配层，并通过 slime 原版 `coding_agent_rl.generate` 的 `--custom-generate-function-path` 选择 DSH；它只负责 SDK 生命周期、session/base URL 和项目内运行目录，轨迹、任务准备、评分和权重流程仍由上游负责。`scripts/check_slime_harness.py` 在 fake sandbox/model/evaluator 边界下 **6 passed**，覆盖 agent/eval sandbox 分离、成功/失败奖励和路径拒绝。该测试没有启动真实任务容器。

**上游入口核对更新（2026-09-10）。** 已核对固定 slime 的 `sglang_rollout.py`：它通过 `load_function(custom_generate_function_path)` 直接调用自定义 generate，并把 per-sample path 传入标准 rollout；`slime_dsh.generate.generate` 的实际 import/bind smoke 已通过。项目没有复制 slime 的训练或 rollout 调度逻辑。

**LoRA 复核更新。** 对当前固定主线执行 PR #1865 的只读 `git apply --check` 失败：补丁引用的 `hf_weight_iterator_bridge.py` 在主线不存在，`model.py`/`model_provider.py` 上下文也不匹配；因此没有把未合并代码复制进项目。GitHub API 当前还列出 #1140（FSDP LoRA）为 open，但它不是本项目固定 Megatron＋Qwen3.5 路径。证据与校验见 [lora-compatibility.json](runtime/slime/lora-compatibility.json) 和 [slime-integration-20260910](research/sources/slime-integration-20260910/source.json)。

<!-- upstream-pipeline-review:start -->
**以下为切换前的调研与历史证据，非当前默认选型。**
**slime 专项适用性核对（2026-09-09）。** slime 发布于 [THUDM](https://github.com/THUDM) 组织（主页标明 Zhipu.ai/Z.ai 与清华 KEG）；项目 citation 列出 Zilin Zhu、Chengxing Xie、Xin Lv 和社区贡献者。Z.ai 的 [GLM-5 官方说明](https://docs.z.ai/guides/llm/glm-5)也将 slime 列为异步 RL 基础设施。它位于训练系统层：Ray 管理资源，SGLang＋router 负责 rollout，Data Buffer 承接轨迹/奖励，Megatron 更新策略并将权重回传 SGLang；DSH 可作为外围 agent/harness 接入，Qwen 仍是被训练的模型。

固定主线 `4c193f1f…` 已有 `scripts/models/qwen3.5-27B.sh` 和 `scripts/run-qwen3.5-27B.sh`；后者默认 4 节点×8 卡并启用 colocate，这只是脚本默认规模，不是最低 GPU 数的证明。coding_agent_rl 例子有现成 harness、OpenAI/Anthropic adapter、独立评分 sandbox 和 token 轨迹处理；`slime/agent/adapters/openai.py` 有流式分支。当前例子注册 Claude Code/Codex，未提供本项目 DSH 的已验证组合；环境用 E2B-compatible 服务，采用它会切换当前 verl/vLLM 主栈。

**两卡 LoRA 是更直接的限制。** 本轮源码树、主线 arguments/model 模块未找到上述 LoRA 接入实现；GitHub API 与 PR 页面确认 [#1865](https://github.com/THUDM/slime/pull/1865)（Megatron-Bridge LoRA GRPO actor training）仍为 `open`、`merged=false`。PR 自述覆盖 dense GRPO/colocate，并有其他限制；不能把未合并的 CPU 测试声明当作主线已支持 Qwen3.5-27B LoRA 或两卡闭环。没有已有可靠配方时，采用该分支意味着额外维护和验证，违背当前“尽量不自己造训练管线”的优先目标。因此判断是：slime 对大规模 coding-agent RL 的架构适合且有实际项目使用证据，但对本项目固定的 2×H100＋27B BF16 LoRA，暂不能优先于有明确 LoRA 路径的候选。若存在团队内部已验证、可提供版本和运行证据的 slime LoRA 配方，应据此重新判断，不能凭空假定存在。

此轮保存了上述模型脚本、主线参数/模型模块、OpenAI adapter 和 PR 元数据，仅审阅未安装、未运行、未采用未合并分支；来源仍见下方统一 source manifest。

**AGL 流式不兼容后的替代筛选（2026-09-09）。** 优先比较 rLLM 整体管线与 AReaL，不再根据“OpenAI-compatible”字样决定接入。rLLM `3b40c37c…` 的 native gateway 有真实 SSE 转发实现，且本项目以前的真实 DSH＋脚本模型 CPU 联调通过过这一层；这仍不能证明其 UnifiedTrainer/verl 后端全栈已通过。若保留 verl，先验证 rLLM 的整体框架，避免继续抽接口自行拼装。

AReaL `f289b989…` 的 `proxy_rollout_server.py:chat_completions` 明确对流式请求返回 `StreamingResponse`，`tests/experimental/openai/test_streaming_chat_completions.py` 有配套端点测试；本轮已读源码并留存，未运行这些测试或安装 AReaL。其 LoRA 支持矩阵列出 FSDP2＋vLLM/SGLang；SWE 示例还依赖 SWEAgent 与 AEnvironment，迁移涉及完整训练和环境栈，不能声称 DSH＋27B 两卡已经适配。Uni-Agent 有 SSE 组装路径，但 DSH thinking/tool/跨轮一致性仍需检查；slime 有完整 coding-agent 例子，采用 Megatron/SGLang 和 E2B-compatible 环境，保留为较大迁移的备选。

新增核对 OpenRLHF `3c3be623…`：它确实有多轮 AgentExecutor 和 token/logprob 记录，但当前 `examples/python/agent_func_openai_server_executor.py` 的 chat 路由返回普通 JSON、仅 content 输出，未实现 SSE/tool-call 响应分支，不能直接用这个例子对接当前 DSH。此判断仅针对该示例，不代表整个 OpenRLHF 不支持其他扩展。

下一次比较应让候选框架接受同一份 DSH 流式请求，验证工具/thinking 内容、原始生成 IDs/logprobs、观察 mask、取消/终止及训练数据输出，再决定使用哪套整体管线。本轮只更新来源与判断，无依赖切换、GPU 或新训练任务。追加的固定源码与测试文件仍登记在 [source.json](research/sources/upstream-pipeline-review-20260909/source.json)。

**首轮实施检查（2026-09-09 16:46 UTC）：AGL 原版 CPU 测试通过，但当前 DSH 流式接入不通过；撤回上一段“AGL 原生转发更适合 DSH”作为已充分验证的判断。** 已将官方 AGL `218f1f7c…` 检出到 `vendor/agent-lightning`，按照其原版 `uv.lock` 安装 `.venv-agl-cpu`（Python 3.12.12、AGL 1.0.1、verl 0.8.0、Torch 2.13.0+cpu，145 个包）。源码没有改动，也没有在旧 `.venv-train-rl` 混装。它的 CPU 依赖组不包含可用于本项目 GPU 训练的完整推理栈。

实际运行上游 server/controller/verl/SWE-smith agent 测试：**90 passed，0 skipped**，报告 [upstream-tests-2.xml](runtime/agent-rl/agl-integration/upstream-tests-2.xml)。首次受宿主 `LD_LIBRARY_PATH` 注入的 Python 3.10 Torch 动态库干扰，出现 69 passed/3 skipped；未将这次跳过称为完整通过。已在独立 CPU 检查入口限定动态库路径，保留原始报告和日志。当前依赖 `uv pip check` 通过。

关键阻断来自实际源码与请求重放：`agentlightning/server/proxy.py:forward_request` 在任何后端请求前，对 `stream=true` 返回 HTTP 400、`Streaming responses are not supported`。DSH 当前 `llm-pi-ai` 走 `streamSimple()`，此前真实 DSH 请求也明确是 `stream=true`。已将该保存请求送入原版 AGL ASGI 路由，得到上述 400：[dsh-stream-compatibility.json](runtime/agent-rl/agl-integration/dsh-stream-compatibility.json)。这次检查没有监听模型服务、执行 DSH worker 或初始化 GPU，也没有修改请求为非流式掩盖问题。HTTP 形状兼容和 token/logprob 字段支持并不等于流式兼容；上一轮选型结论遗漏了这个直接条件。

首轮规模建议明确为两阶段：**先完成无 GPU 的协议接入；通过后再做 2 道经所选上游任务流程验证的训练题、每题 2 条轨迹、最多 1 次 LoRA 更新，验证权重回传及保存/恢复。** 原有两道 Dask 的 custom-training-pytest 通过记录不能直接冒充 AGL SWE-smith 的环境资格；正式首轮题目须使用对应上游协议并重新核对排除。四条轨迹只用于工程检查，不用于声称效果提升。如果同题两条奖励相同，GRPO 没有组内区分信号，不应为了制造一次“成功更新”修改奖励。

现阶段不安装更多 GPU 栈、不消耗 H100、不开始批量任务。下一步先核对上游支持的 DSH 非流式选项或改验现成支持流式的完整框架；没有证据时不自建 SSE 代理、不修改 AGL 核心、不把之前抽取 rLLM 接口的 CPU 成功当作 rLLM 全栈训练成功。2×2/1-step 是待兼容性通过后的建议边界，训练参数和数据规模未被实际改动。复核命令：`bash scripts/agl_cpu.sh --upstream-check`；`bash scripts/agl_cpu.sh --compatibility-check` 预期返回 exit 2，表示当前组合不兼容。版本和验收清单见 [acceptance.json](runtime/agent-rl/agl-integration/acceptance.json)。

**修改方案收敛（2026-09-09，后于下表调研）：建议先验证 Agent Lightning＋verl 完整管线，Uni-Agent 作为备选。** 先前按“verl 官方扩展”优先 Uni-Agent；进一步对照本项目明确要求保留 DSH 原生工具/thinking 流，AGL 的 `server/proxy.py` 直接向 vLLM 请求 token IDs/logprobs 并转发响应，更贴近既有连通路径。Uni-Agent 的 `MessageCodec.decode_response` 当前构造 content/tool_calls，OpenAI adapter 则从完整 outcome 组装 SSE；不能仅凭 adapter 有 reasoning_content 分支就认为 codec 已产生该字段。这是基于源码的接入优先级判断，不是对两个框架整体质量的排名，也不是已经完成迁移。

具体改法：训练控制、rollout 调度、轨迹聚合和 verl 交互交给 AGL 的现成实现；项目只通过它的 agent/runner 入口启动 DSH、注入每次 rollout 的 endpoint/model/workspace，并提交上游任务验收结果。DSH 继续使用现有 SDK/provider/工具，Qwen 使用原生模板及解析器；不直接复制 SWE-smith 示例的文本 bash agent 或 Hermes 模板。任务数据/schema、环境初始化和验证器优先配套复用上游，已有排除审计继续生效；本地受限执行要求作为单独的环境接入条件，不通过重写整个训练管线解决。之前的自建 AgentLoop、gateway session 和 pytest 批次管理只保留证据，迁移完成后退出正式运行路径。

验证顺序为：固定并核对 AGL 支持的 verl/vLLM 版本与当前 pin 差异 → 不改上游流程的 CPU/协议检查 → 在后续授权的两卡检查真实 DSH rollout、LoRA 参数更新、权重回传、保存/恢复 → 按同一上游数据环境协议扩池。每一步都以实际证据决定是否继续；如 AGL 存在无法配置解决的具体不兼容，回到 Uni-Agent 备选，先记录缺口，不默认新增自研框架。当前 `implementation_hold`、GPU/训练启动保护均未解除。

“新”主要指 **DSH＋Qwen3.5-27B LoRA 在线 Agent RL 的目标组合及两卡配方**。DSH 官方仍标注 developer preview；Qwen3.5-27B 官方已有 Transformers/vLLM/SGLang 接入，LoRA 也有成熟 PEFT 实现。目标组合缺少本轮找到的完整复现证据，尤其是多轮采样 token 对齐、混合注意力模型训练、LoRA 权重回传和两卡内存；这不构成重写模型内核或训练流程的理由。核对来源：[DSH 官方](https://www.deepseek.com/harness/en/)、[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.5-27B)、[PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora)。

**最新方向纠正（2026-09-09）：暂停自建管线，以成熟上游的完整训练方案为选型对象。** 此前仅复用 gateway/转换函数，却继续自行组织 AgentLoop、工作区、资格检查和奖励流程，选型调研不足；CPU 接口成功不构成继续扩大这套架构的理由。下面保留的自建实现和 2 题验证属于历史工程证据，不再作为已选定的正式训练方案。

本轮在线核对 6 个项目的官方资料，保存固定 commit 的源码/配方快照，未安装或执行这些上游方案。主比较按“数据入口→agent/环境→轨迹→奖励→更新/同步/恢复”整条路径进行。机器可读来源及逐文件 SHA-256：[source.json](research/sources/upstream-pipeline-review-20260909/source.json)。

| 方案 | 完整上游能力与实际入口 | 对当前项目的适用性和边界 |
|---|---|---|
| **verl Uni-Agent** `bb96ecab…` | `AgentFrameworkRolloutAdapter`、Task/Agent/Sandbox、会话 gateway、token/mask/logprob、奖励及 TransferQueue 训练接入均已存在；`examples/quickstart/training/train_qwen3p5_dense.sh` 与 mini-swe-agent 黑盒训练例子 | **保留 verl 时优先验证**。已有 Qwen3.5 dense 配方，但该脚本默认 4B、Megatron、8×8 卡；不是本项目 27B LoRA 两卡配方。它捆绑 verl `fefb0802…`，不同于本地 `1252cc71…`。源码树未找到 DSH 命名适配器，需要核对其公开 Agent/runner 扩展点，不能默认无改动接入。见[上游训练入口](https://github.com/verl-project/uni-agent/blob/bb96ecab183bdf0c3dba6958fd81d3bd96a024cd/examples/quickstart/training/train_qwen3p5_dense.sh)。 |
| **Agent Lightning v1** `218f1f7c…` | Gateway→Controller→Trainer；`examples/swe_smith/run.sh`、任务 agent、预分训练/验证集及防奖励作弊措施一并发布；透明代理请求原生 token IDs/logprobs | 最值得对照的同模型系列完整案例。[官方 coding-agent 配方](https://microsoft.github.io/agent-lightning/latest/75-example-coding-agent/)是 Qwen3.5-9B、4×B200、K8s。源码使用 FSDP、定制 Hermes 模板，不能直接宣称保持本地 DSH/Qwen 模板。`setup_verl.sh` 的 0.8.0 路线固定 vLLM 0.20.2，与本地不同；Local Controller 存在，但不是隔离任务镜像的替代品。 |
| **rLLM 整体框架** `3b40c37c…` | UnifiedTrainer、Workflow Engine、Gateway、sandbox 调度和 CLI harness 已串起来；`BaseCliHarness` 负责安装/配置/启动外部 agent，轨迹由上游收集；支持 verl 后端 | 应整体评估，不能只摘录 trace converter 后自行重组。当前 [Harbor SWE 示例](https://github.com/rllm-org/rllm/blob/3b40c37cf6a262cf4d28cc987ebe4f4cf797956c/examples/harbor_swe/train_harbor.sh)默认 Tinker＋Daytona＋mini-swe-agent，并非本地 verl。其 pyproject 的 verl extra 固定 verl 0.8.0/vLLM 0.22.1；文档仍有旧版依赖说明，安装以固定源码为准。 |
| **SkyRL＋mini-swe-agent** `ba3487ae…` | 提供 SWE-Gym 预处理、Podman 任务环境、生成/干净副本验收与训练脚本 | [现成配方](https://github.com/NovaSky-AI/SkyRL/blob/ba3487ae66917d37d82d4c4482c44f4e252f45b0/examples/train/mini_swe_agent/README.md)完整，但示例为 8B/8×H100、30B/16×H100。当前 generator 从 messages 构造 response IDs，返回 `rollout_logprobs=None`；这一个示例未满足我们保留原始采样概率的要求，不据此推断整个 SkyRL 不支持。 |
| **slime coding_agent_rl** | 成熟训练主栈 Megatron＋SGLang；共享 harness/adapter，独立干净 sandbox 评分，分支轨迹处理；通过标准 custom-generate 扩展接入 | [SWE 示例](https://github.com/THUDM/slime/tree/main/examples/coding_agent_rl)已经实现此前我们自行组织的许多环节，当前代码有 Claude Code/Codex，环境为 E2B-compatible。若采用它会涉及训练/推理主栈变更，不能冒充对现有 verl/vLLM 的小补丁。 |
| **AReaL＋AReaL-SWEAgent** | 自带 SWE GRPO 入口和 OpenAI proxy，外部 SWEAgent 执行工具/评分；LoRA 文档列出 FSDP2＋vLLM 支持 | [SWE 配方](https://github.com/areal-project/AReaL/tree/main/examples/swe)依赖独立 SWEAgent 与 AEnvironment 服务；任务环境还需要部署方提供，示例为 Qwen3-30B-A3B。值得作为完整备选，但不是简单切换 backend 即可接管现有工程。 |

**复用边界。** 正式路线应由选中的一个上游拥有任务生命周期、会话/轨迹、奖励传递、批次组织、训练更新和权重同步。本项目优先只维护固定配置、数据排除/来源清单，以及上游明确缺少的 DSH 启动适配或平台 sandbox provider。DSH 本身继续负责 agent 工具与行为；不因它缺少某个现成 runner 就重写训练管线。已有自建网关会话/AgentLoop/pytest 管理代码暂停发展，后续评估哪些可以退出运行路径；现阶段未删除证据或迁移依赖。

**为什么目前优先 Uni-Agent、同时认真比较 Agent Lightning。** Uni-Agent 已提供 trainer 侧的正式扩展入口，并把黑盒 agent 和 SWE task/environment 分开，适合保留 verl；AGL 的 native HTTP 透传路径更贴近本地已有 vLLM/DSH 接入，且同系列模型有完整 coding 训练配方。最终取舍取决于固定版本的两卡 LoRA 兼容与 DSH 工具/thinking 回放，不能用项目名气、README 的兼容宣称或默认模型路径判断。这是调研优先级，不是已经批准更换主栈。

**源码核对发现的具体差距。**

- Uni-Agent 的 gateway 自己使用 MessageCodec 和后端 tool parser，返回 OpenAI 形状的数据；其 SSE 适配是对完整 generation outcome 的封装，不等于逐字节透传 vLLM 原生流。DSH 已能直连 vLLM，不能因此直接推出这条训练 gateway 的 thinking/tool/session 重放已经验证。框架现成能力应先用自身检查验证，具体不兼容才能成为最小改动的依据。
- Uni-Agent [训练文档](https://uni-agent.readthedocs.io/en/latest/quickstart/rl-training.html)提到已处理的 1,150 题集；但固定 `swe_rebench/preprocess.py` 实际加载 `nebius/SWE-rebench` 的 `filtered` split，未固定 revision，输出文件名也不带 `_1150`，与 dense launcher 默认文件名不同。正式采用时必须先对齐版本和数据来源，不能把本地 SWE-rebench-V2 或 SWE-Gym 表直接当同一 schema。
- AGL 公开的数据配方约 6,000 训练/400 验证，筛选用过 Qwen3.5-9B 的四次难度探测。可以复用其数据和环境流程，但该难度筛选不是本地 27B 实测结果，仍要执行项目既定的 holdout 排除；本轮没有下载这份预处理集，也没有进行难度 rollout。
- “有 local backend”不表示已有隔离的仓库训练环境。Uni-Agent 文档明确 local 是宿主执行；rLLM 的 LocalSandbox 也直接运行宿主 subprocess。不能为绕开 DockerProxy 就把这些直接用于不受信任任务。环境可交给成熟的上游 provider 或平台独立任务实例，平台接入和训练管线选型分开解决；本轮不探测容器权限、不部署云服务。
- 六条路线中，本轮没有查到可直接证明 **DSH＋Qwen3.5-27B BF16 LoRA＋2×H100** 已跑通的完整配方。缺的是目标组合的兼容性与资源证据，不是必须再造一套通用管线。GPU、正式训练和依赖切换均未执行。

**任务池暂停状态。** 上一轮已冻结 24 个扩展候选（Dask/Bokeh/Hydra 各 8），其中 6 个开发任务在测试前预留；24 份源码已准备，依赖安装在用户纠正前启动并已完成。尚未对这 24 题运行新的 buggy/gold qualification，也未导出扩大的训练集；`train-ready.parquet` 仍为原有 2 道 Dask。新增准备/导出脚本改动处于未完成、未验收状态，不应直接恢复批量运行。先按所选上游数据格式及环境协议重新审视这些候选，已有来源与排除信息可以保留。

当前 research hold 见 [readiness.json](configs/agent-rl/readiness.json) 和 [pipeline-review-hold.json](runtime/agent-rl/pipeline-review-hold.json)。下方 `training-engineering.json` 与 `training-engineering-final-check.json` 是此前 CPU 交付时的历史快照，后续准备脚本及文档发生了变化，不应把旧哈希/通过记录当作当前全部文件的验收。`--launch` 的无条件拒绝仍存在。此次只新增研究快照并更新状态文档；没有继续写训练管线或运行新训练任务。
<!-- upstream-pipeline-review:end -->

<!-- local-agent-rl-engineering:start -->
**当前训练工程状态（2026-09-09 15:00 UTC）：已接入项目内独立工作区＋受限进程后端，完成真实 DSH 的 CPU 接口联调，准备两道经过 buggy/gold 验证的训练任务；尚未启动真实模型 rollout 或参数更新。** 本节取代下方历史记录中“AgentLoop 未注册、DSH session/持久化未实现、训练必须先解决官方容器”的现状描述。官方 benchmark 仍须使用其官方环境和 evaluator；自建训练协议不会产生官方成绩，历史 SWE-smith 0/3 和已停止的 SWE-Gym facility smoke 均保持原样。

实际连接为 `verl AgentLoop → DSH 原生 SDK/pi-ai → rLLM 原生 HTTP gateway → verl 管理的 vLLM`；DSH 的 Shell/编辑器共用一个受限工作区，结束后仅导出候选 patch，交给独立干净副本运行训练测试，再经上游 trace converter/DataProto 转换为 `AgentLoopOutput`。保留 Qwen3.5-27B BF16＋LoRA、固定 verl `1252cc71…` 和现有 vLLM 0.19.1；未改写上游训练循环或 SSE 解析器，未升级推理环境。

- [agent_rl_dsh_loop.py](scripts/agent_rl_dsh_loop.py) 已按上游接口注册 `dsh_agent`，管理工作区租约、DSH worker、模型 replica 获取/释放和终局奖励。项目文件锁限制并发环境为 1；每条轨迹最多 12 次请求、600 秒，初始完整提示上限 4096 tokens、后续生成及工具观察合计 4096、上下文 8192、单次工具观察 1024。超限直接留痕拒绝，不静默裁剪训练轨迹。
- [agent_rl_training_runtime.py](scripts/agent_rl_training_runtime.py) 复用原版 rLLM native Chat Completions gateway，增加 session 鉴权、原生 `/tokenize` 预算/前缀预检与耐久请求、SSE、TraceRecord 记录。模型生成 token 不重新分词；loss mask/logprobs 使用已验证的上游转换。policy version 来自同步 trainer row，仍不是 GPU 权重同步的独立证据。
- [agent_rl_training_task.py](scripts/agent_rl_training_task.py) 只复用原有 `gym_task_env.py` 的底层进程隔离，不调用旧 benchmark runner。当前主机已实测独立 UID、user/PID/network namespace 与 Landlock 隔离；gold/test 资料对 DSH 不可读，Shell 和编辑器读写同一工作区。候选 patch 用独立可信 Git 索引导出，拒绝测试配置修改、符号链接/特殊文件。新环境也必须具备这些内核/权限条件；普通工作目录本身不构成隔离。
- 验收使用明确命名的 **`custom-training-pytest-v1`**。固定 source tree、原始测试节点、依赖锁、gold/test patch 的指纹；加载时检查实际安装包与锁一致。buggy 必须恰好失败在指定 FAIL_TO_PASS 项，gold 必须通过，实际执行测试集合须一致。完成且通过的候选 reward=1，测试失败或违反编辑规则 reward=0；超时、采集/初始化错误、token 缺失和截断均保留失败且不生成训练 reward。

**真实训练任务环境（仅 CPU pytest；没有模型尝试）**

| 任务 | 请求测试数 | 实际执行 / 跳过 | Buggy 失败项 | Gold |
|---|---:|---:|---:|---|
| `dask__dask-10009` | 365 | 364 / 1 | 1，等于指定 FAIL_TO_PASS | 执行项全部通过 |
| `dask__dask-10027` | 560 | 532 / 28 | 2，等于指定 FAIL_TO_PASS | 执行项全部通过 |

可选依赖导致的 skip 已保留，不宣称完整官方测试覆盖。两题来自原先通过排除审计的 8 题训练工程池，无模型结果筛选；其余 6 题尚未配环境。已导出 [train-ready.parquet](data/agent-rl/train-ready.parquet) 与 [train-ready-manifest.json](data/agent-rl/train-ready-manifest.json)，`+training_runtime=local` 仅加载这 2 题，足够承接候选 batch=2、每题 n=2 的后续联调，不能代表生产训练集规模或多样性。原始 8 题文件与排除审计保留，历史测试题/关联轨迹不进入训练。训练依赖位于独立 `.venv-train-dask`（Python 3.10.18、33 个固定包）；锁和源码下载 SHA 在 [training-environments](runtime/agent-rl/training-environments/dependency-source.json)。依赖最初复用通用锁，未读取旧 holdout 的任务代码或轨迹。

**实际验收证据**：[training-interface-latest.json](runtime/agent-rl/training-interface-latest.json) 记录真实 DSH 发出 5 次脚本模型请求，完成 Shell→编辑器→Shell 修复，私有 gold 读取被拒绝，独立训练 verifier 得到 reward=1，原始采样数组转为上游训练输出；worker/socket/租约已清理。模型响应、token IDs/logprobs 是明确的 CPU fixture，**不证明 Qwen 修复能力、真实采样概率或训练成功**。另有 5 项拒绝/清理检查：gold 变更使资格过期、保护测试修改拒绝、verifier 超时清理且无 reward、未授权请求不破坏合法 session、请求预算在访问模型前拒绝。原有 native 接口测试覆盖 16 类轨迹拒绝条件及观察 token 零梯度；上游 Hydra 配置解析和真实两题数据加载检查见 [training-data-check.json](runtime/agent-rl/training-data-check.json)。

首轮 fixture 因解释器符号链接被 resolve 成基解释器而找不到 pytest；首轮 Dask 环境缺少显式 pytest 插件并遇到线程资源限制。已修复解释器路径保留、显式固定插件和 Dask/BLAS 双线程配置，所有失败与最终 qualification 原样保留。最终 verifier 单进程地址空间上限 3 GiB、每次 120 秒，未使用 GPU。

复核入口（CPU 工程检查，不会启动训练）：

```bash
source scripts/env.sh
bash scripts/agent_rl.sh --training-check
bash scripts/agent_rl.sh --training-config
bash scripts/agent_rl.sh --training-data-check
bash scripts/agent_rl.sh --native-check
# 已配置环境的资格重验；保留每次结果，最多两题顺序运行。
.venv-train-rl/bin/python scripts/prepare_agent_rl_training_tasks.py \
  --task dask__dask-10009 --task dask__dask-10027 --qualify --export-qualified
```

**训练前仍需完成的验证**：在另行授权、重新分配的最多两张空闲 H100 上，验证真实 Qwen thinking/tool 采样 token 概率覆盖和跨轮前缀；验证 FSDP2 BF16＋LoRA 的 forward/backward、vLLM 合并权重同步、sleep/wake、峰值内存和 checkpoint 恢复。原生 HTTP 路径的取消与训练器权重更新周期仍须实测。上游 colocated transfer 有硬编码 `/tmp` IPC，未来执行环境须把该路径落在项目内；现有 TMPDIR 不能替代该检查。持续训练按父目录要求走平台 Training Task，需可用资源与显式参数更新授权。**DockerProxy 权限不再是此自建训练后端的必需条件**，官方 benchmark 的平台问题单独保留。

`scripts/agent_rl.sh --launch` 继续无条件返回 exit 2，不提供绕过开关。本轮没有容器/队列探测、真实模型 rollout、GPU 初始化或参数更新。当前状态见 [readiness.json](configs/agent-rl/readiness.json)，版本、文件哈希与检查证据见 [training-engineering.json](configs/agent-rl/training-engineering.json)。旧 `runtime/agent-rl/artifact-manifest.json` 仍为历史快照，后续改动以新清单为准。
<!-- local-agent-rl-engineering:end -->

本轮主线：**原始 Qwen3.5-27B BF16 + LoRA 参数更新 + 多轮 Agent RL，保留 DSH**。本轮仅准备训练工程，不做 benchmark、容器排查、rollout、参数更新，不自动转为 SFT，不实施 LoopLM/OPD。下文较早的研究路线、历史命令及预算保留为记录；当前状态与范围以下节为准。既有 SWE-smith 0/3 和被停止的自定义 SWE-Gym 设施记录保持原样。


<!-- native-agent-rl-interfaces:start -->
**原生接入与上游训练接口复用（2026-09-09；已完成第一批 CPU 接口实现，在线 RL 尚未就绪）**

按用户“先复用原生接入和上游训练接口”的要求，新增 [agent_rl_native.py](scripts/agent_rl_native.py) 与 [native_http/vllm.yaml](configs/agent-rl/native_http/vllm.yaml)。固定 verl 的 `vLLMHttpServer.run_server` 本来就调用 vLLM `build_app/init_app_state`，`LLMServerManager.get_addresses()` 明确返回原生 Chat Completions 地址；无需另写模型服务。配置组通过 `+native_http=vllm` 加入原有训练候选，仅增加模型服务别名、auto tool choice、`qwen3_coder` 工具解析器与 `qwen3` reasoning 解析器这四个 HTTP 能力字段，训练参数完全继承原候选。

实际复用链路：**原生 Chat Completions → 固定 rLLM gateway 的会话/TraceRecord → 上游 `trace_record_to_step` 与 `transform_episodes_to_dataproto` → 固定 verl `AgentLoopOutput`**。本地代码只生成配置、验证更严格的输入合同并接合这两个上游输出结构；不改 DSH、vLLM、rLLM、verl 源码，不重写解析器、SSE、训练循环或评分器。rLLM gateway 保持 `cumulative_token_mode=false`，原生工具/正文/thinking delta 继续由服务端产生。选中的上游模块直接从干净固定源码导入；未安装完整 rLLM、未升级 0.19.1，也未改原依赖锁。当前仅验证已安装环境能导入和使用这些模块，不声称完整 rLLM 包的全部可选后端依赖均已安装。

`traces_to_agent_output` 在调用上游转换前，要求每轮有原始 prompt/completion IDs、等长有限 logprobs、匹配的 usage、相同 session/model/policy version、固定工具 schema 和连续 token/消息前缀。它拒绝空轨迹、重复请求记录、超过预算、截断和基础设施错误，避免上游转换中的补零、分段或丢弃兜底把失败静默变成训练样本。reward 与 verifier_ref 必须由调用方提供，函数本身不执行或判分任务。当前只接受完整终止的工具轨迹；截断如何用于学习仍未定义，失败应由调用方保留。

**已实际通过的检查**：[check_agent_rl_native.py](scripts/check_agent_rl_native.py) 使用进程内 ASGI + mock HTTP 传输，运行原版上游网关和转换函数。两轮合成流保留 thinking、分片工具调用及最终正文；返回原始数组 `[10,11,12,20,21,30,31]`，上游生成 mask `[1,1,1,0,0,1,1]`、对应 logprobs，真实 `agg_loss` CPU 张量检查确认两个观察 token 梯度为零、thinking token 参与 loss。另覆盖 16 类拒绝条件。原生 vLLM 响应 schema、Hydra 配置、上游 `validate_config` 与 vLLM CLI 参数解析均通过，测试确认除上述四个 HTTP 字段外，配置与原候选相同。两次初始 Hydra 配置组合失败已保存，最终改用配置组；没有修改上游 Hydra 或训练配置规避问题。

本次 **没有监听端口、DSH session、真实模型请求、任务容器、Ray/CUDA 初始化、rollout 或参数更新**。测试的响应、token、logprob、policy version 和 reward 是明确标记的合成 fixture，不能作为真实采样或官方任务成功证据。结果及失败记录在 [acceptance.json](runtime/agent-rl/native-interface-check/acceptance.json) 和同目录；逐文件版本与哈希见 [native-interfaces.json](configs/agent-rl/native-interfaces.json)。

复核命令（均不会启动训练）：

```bash
bash scripts/agent_rl.sh --native-check
bash scripts/agent_rl.sh --native-config
```

**后续在线连接仍需完成**：将已验证接口接入真实 DSH session 与已注册 AgentLoop；由训练阶段管理 HTTP admission、policy version、请求/时间预算和正在执行的请求取消，不能把 `/admin/weight_version` 的手工标签当成权重同步证据。原生 HTTP 请求绕过 `LLMServerClient.generate` 的部分 Python 层控制，必须核对它与上游 sleep/wake、abort/resume 和权重更新的协同。上游 gateway 的流式 TraceRecord 摘要不保存原始 SSE 或 reasoning 文本、工具参数仍可能是分片，不能将摘要当完整 agent 日志；真实运行前需将 DSH 会话事件/原生流证据持久化，并接入实际耐久存储、环境租约及官方验证器。CPU 检查仅使用内存 store，未为训练启动记录服务。真实 Qwen 请求的 thinking/tool token 概率覆盖及跨轮前缀一致性仍需实证。

`--launch` 仍实测返回 exit 2，`training_launch_ready=false`；官方环境和最多两卡 GPU 联调未执行。

变更记录：2026-09-09 14:18 UTC — 实施原生 HTTP 配置组及上游轨迹转换接口复用，增加 CPU 接口验收和 16 类输入拒绝检查，保留配置失败证据；保持训练参数、依赖、上游源码和启动保护不变。早先下表中拟自行构造模型桥/轨迹模块的方案，以本节已实现的上游复用方式为准。
<!-- native-agent-rl-interfaces:end -->

<!-- code-agent-implementation-review:start -->
**开源 code-agent 实现对照（2026-09-09 13:18 UTC；源码审阅，未接入运行）**

本次按用户要求直接核对 agent 循环、执行环境、训练轨迹与验证器的源码。工程判断：保留已固定的 DSH + verl 主栈，优先复用现成接口；缺失的工作集中在 DSH 会话/模型桥和官方执行环境连接，不需要另写训练循环。下表是实现参考，不是更换框架或启动实验的决定。

**vLLM/DSH 兼容性与修改方向纠正（2026-09-09 14:03 UTC）**：用户指出 vLLM 已支持 DSH；复核确认，DSH 的[官方 provider 文档](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/providers.md)支持 OpenAI-compatible 自托管 endpoint，vLLM 提供原生 Chat Completions、流式输出和工具调用。项目 `runtime/metadata/dsh-real-check.json` 已记录 vLLM 0.19.1 + Qwen + DSH 的真实连通成功；不能把 RL 数据接入未完成说成 DSH 推理不受支持。GitHub releases/latest 与 PyPI 在 14:02 UTC 均返回最新版本 **0.29.0**（GitHub 发布于当天 08:54:49 UTC），[发布页](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)；本地仍为 0.19.1，未升级。下一步应先审查固定 verl 管理的原生 HTTP 服务与训练生命周期，优先直接复用原生流和返回的 token IDs/logprobs，必要时加透明记录层；**撤回“先写自定义模型桥、重新解析并封装 SSE”为默认路线**。只有证明原生路径缺少具体训练能力后才设计最小补充，不能以某个 rLLM 示例局限推断整个 vLLM/DSH 不支持。已有 0.19.1 protocol 也声明了 `return_token_ids`、prompt/completion token IDs 与 logprobs；多轮前缀对齐、thinking/tool token 概率覆盖、policy version 和权重同步仍需验证，不从声明直接推出训练闭环已完成。

| 参考实现与固定版本 | 源码中实际存在的能力 | 对本项目的具体用法 |
|---|---|---|
| [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent/blob/04d809ceab9df28f9adaed044884180159172930/src/minisweagent/agents/default.py) `04d809ce…` | `DefaultAgent.run → step → query/execute_actions`，Model 与 Environment 分离；逐轮保存 messages、退出状态、请求数和费用，模型调用前检查预算 | 借鉴明确的任务生命周期及失败留痕。其调用前的 wall-time 检查不等于能中断正在阻塞的请求；DSH 桥仍需整体 deadline 与取消传播。保留 DSH 原生工具协议，不将文本 bash 轨迹直接当 DSH 训练轨迹 |
| [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk/blob/7a018907c9b6cac2ee7b205a1b4a5e3e0228e8e4/examples/02_remote_agent_server/02_convo_with_docker_sandboxed_server.py) `7a018907…` | `Conversation(agent, workspace)` 绑定远端执行；`DockerWorkspace` 管理生命周期，`RemoteWorkspaceMixin` 将命令、文件上传/下载和 Git 操作送到 agent-server | 借鉴“一个任务对应同一个执行工作区”。该示例 agent-server 镜像不是 SWE-Gym 官方题目镜像；不能直接拿来替代已固定 benchmark 环境，也不意味着安装 SDK 就解决平台执行权限 |
| [历史 DeepSWE/rLLM](https://github.com/rllm-org/rllm/blob/6458960c2478dc348c10b23e67f2789d7c1745d1/examples/swe/train_deepswe_agent.py) `6458960c…` | `AgentTrainer(agent_class=SWEAgent, env_class=SWEEnv)`；`SWEEnv` 复用 R2E `RepoEnv.reset/step/compute_reward/close`，终局奖励单独计算 | 借鉴训练器、agent 和环境分工。当前 SWE-Gym 记录不符合 R2E schema，不能把历史环境类直接套过来；历史实现与当前 rLLM 网关分开看 |
| [当前 rLLM model gateway](https://github.com/rllm-org/rllm/blob/3b40c37cf6a262cf4d28cc987ebe4f4cf797956c/rllm-model-gateway/src/rllm_model_gateway/proxy.py) `3b40c37c…` | 按 `/sessions/{sid}/v1` 路由，注入 token/logprob 请求参数；保存流式原始 token、概率和权重版本；`TokenAccumulator` 维护跨轮前缀 | 最接近 DSH 所缺的模型桥。可参考独立网关包，不必安装整个 rLLM。**不是即插即用**：累积模式改走 completions，响应转换只生成 content，未恢复 Qwen reasoning/tool-call 结构；历史分叉或 renderer 不支持时会 reset 回 chat 路径。首个 DSH 训练接入必须明确拒绝不支持的轨迹，不能默默合并。该包实际 pyproject 已依赖 Transformers/renderers，README 的“无 Transformers”描述滞后 |
| [SkyRL + mini-SWE-agent](https://github.com/NovaSky-AI/SkyRL/blob/ba3487ae66917d37d82d4c4482c44f4e252f45b0/examples/train/mini_swe_agent/mini_swe_generator.py) `ba3487ae…` | 外部 agent 经模型 HTTP 入口生成补丁，再在新环境应用补丁验收；Ray 按轨迹调度 | 借鉴生成/验收分离的组织方式。此示例从 messages 重建 response IDs，`assistant_logprobs=None`、`rollout_logprobs=None`，不满足本项目原始采样 token/logprob 合同。其 [mini_swe_utils.py](https://github.com/NovaSky-AI/SkyRL/blob/ba3487ae66917d37d82d4c4482c44f4e252f45b0/examples/train/mini_swe_agent/mini_swe_utils.py) 依据 eval_script 退出码设置 resolved，没有直接调用本项目固定官方 grading/report 入口；也不直接复制其错误回零、latest 镜像和丢弃空轨迹行为 |

**在现有栈上可复用的连接点**

- DSH 已有 [fs-e2b](https://github.com/deepseek-ai/deepseek-harness/blob/5dda764ed3aa172535a7967b06ff95d9cbfe536a/packages/e2b/fs-e2b/src/index.ts) 与 [subprocess-e2b](https://github.com/deepseek-ai/deepseek-harness/blob/5dda764ed3aa172535a7967b06ff95d9cbfe536a/packages/e2b/subprocess-e2b/src/index.ts)，都取得同一个 `ctx.e2b.getSandbox()`。这是文件与进程共同切换执行位置的现成实现参考；不将其视为已有 Docker 适配，也不在本轮启用 E2B。只重定向 Shell、编辑器仍读写宿主的方案不成立。
- 当前固定 [verl AgentLoopBase](https://github.com/verl-project/verl/blob/1252cc71aa5bd82e5604322064d69bfe6454c660/verl/experimental/agent_loop/agent_loop.py) 已提供 `ct_build_initial_tokens`、`ct_merge_assistant_token`、`ct_merge_context_msg`；[ToolAgentLoop](https://github.com/verl-project/verl/blob/1252cc71aa5bd82e5604322064d69bfe6454c660/verl/experimental/agent_loop/tool_agent_loop.py) 展示了调用 `LLMServerClient.generate(prompt_ids, sampling_params)` 后，用原始 `TokenOutput.token_ids/log_probs` 更新轨迹及 mask 的路径。DSH 的 `dsh_agent` 应复用这些接口，DSH 自己继续驱动工具调用；普通 ToolAgentLoop 不冒充 DSH。
- 模型桥需要两份对齐输出：面向 DSH 的 reasoning/content/tool_calls 流，以及面向训练器的原始 token/logprob、mask、request/session ID 与 policy version。保存文本供查看，保存采样数组供训练；Qwen 的工具和 thinking 解析需复用已固定后端解析器并验证，不能把 XML/特殊 token 原样当成普通 content 就声称接通原生工具。
- 任务结束时只导出候选 patch，交给固定 `SWE-Gym/SWE-Bench-Fork` 官方环境和验证器；gold/test 资料保留在验证侧。基础设施失败单列，不伪造 reward=0，不丢弃失败改变分母。平台权限及项目内存储仍是后续执行前提。

后续接入的验收顺序建议：先用无 GPU 的固定协议样本检查 DSH 流、token/logprob/mask 等长、thinking/tool-call 还原、前缀变化拒绝、超时取消和预算计数；随后在可用的官方任务环境检查 Shell→编辑器→Shell 的状态一致性与 buggy/gold；最后在另行授权的最多两张 H100 上检查真实 rollout→LoRA 更新→权重同步→checkpoint 恢复。前者只能证明协议适配，后两者通过后才可能形成实际 RL 闭环。没有新增训练参数或将两卡候选称为上游验证配方。

**来源判断纠正（2026-09-09 14:00 UTC）**：上一轮错误地拿主线参考 `5dda764e…` 与本地 release 作比较，将正常版本差异当成了来源问题。实际 `runtime/metadata/dsh-source.json` 和 `runtime-versions.json` 已明确记录 vendor 属于 **`a66e4702047846cdaa10c66c9d3df3951f5ea70d` / `dsh-v0.1.2-rc.1`**。本轮将之前三个不同文件以及实际启用的 pi-ai adapter/context/stream 共 6 个文件与该 release 重新在线逐字节核对，全部一致。已修正 `configs/agent-rl/dsh-interface.json` 的错误 commit。运行配置还明确禁用了 `llm-deepseek`，Qwen 实際走 **`llm-pi-ai → qwen-local → QWEN_BASE_URL`**；接口清单同步改为这条路径。主线源码快照继续作为版本明确的研究资料，不作为本地运行包的同版源码。

**针对当前工程的修改清单（设计已核对，以下新增模块尚未实现）**

| 拟修改位置 | 具体职责与复用点 | 完成标准 |
|---|---|---|
| 原生 vLLM HTTP 接入与必要的轨迹记录层（具体模块待接口核对） | 继续使用 DSH pi-ai 的 `QWEN_BASE_URL`，优先连接固定 verl 管理的 vLLM Chat Completions 服务；先核对原生 `return_token_ids/logprobs`、session 关联与版本标记。解析及 SSE 交给 vLLM，不默认新增 `model_bridge.py`；rLLM 网关可作为透明记录层参考 | 验证训练时 HTTP endpoint 可用且随 verl 生命周期工作，工具/thinking 输出与原始采样数组完整对齐，取消和预算有效。只有原生路径存在已证实缺口时，再比较 token-in 接口或最小适配；之前 CPU 手动解析样本只说明解析器存在，不构成另写服务的理由 |
| `code_agent_rl/trajectory.py`（拟新增） | 包装固定 verl 的 Continuous Token builder，只追加新工具观察和真实生成 token；记录同一轨迹的 policy version、会话/请求 ID、生成区 mask 和终止原因 | 模型 token 不重新分词，mask/logprob 与 token 等长；拒绝历史改写、压缩、工具 schema 变化、缺失概率、混合权重版本。完整首轮 system/tools/question 应在传入可能截断的上游 helper 前检查长度；不能仅验证问题 prompt |
| `code_agent_rl/dsh_loop.py` + `configs/agent-rl/agent_loops.yaml`（拟新增） | 薄 `AgentLoopBase` 适配：接收任务引用，创建 DSH session，调用模型桥，结束后输出 `AgentLoopOutput`；使用上游 agent_loop_config_path 注册，不重写训练循环 | CPU 注入测试先验证预算与生命周期；DSH run 为同步 SDK，不能阻塞模型桥所在 event loop。并发环境数须有实际租约/信号量，整体 deadline 能取消模型与工具，退出后资源释放；注册类本身不能作为 online ready |
| 官方环境连接与 `configs/agent-rl/dsh-qwen-rl.patch.yml`（拟新增） | 优先评估将完整 DSH 运行进程放入同一个上游创建的官方任务容器，复用其本地 Shell/编辑器；SDK 的 dsh_bin 执行入口可承接受控容器 exec，训练器与 gold/verifier 保持外部。无需先写 Docker 版文件系统/进程全套 provider，也不新造任务镜像或评分代码 | 固定官方镜像与工作树、受控只读挂载 DSH 必需运行文件、独立 DSH_HOME/日志，验证 glibc/运行包兼容与模型端点可达；不挂载整个项目而暴露私有 gold。Shell→编辑器→Shell 状态一致，官方 buggy/gold 通过，候选 patch 经同一官方 evaluator 验收。此路径尚未运行，若平台存储/容器条件不满足则仍阻止启动 |

RL patch 应独立于已验证 smoke 配置。后者当前 `reasoningEfforts=false`、maxTokens=256；RL 候选要求 thinking 与完整 8192 上下文，不能直接复制后称为同样配置，也不在设计阶段修改既有推理协议。现有候选预算只是待审配置，实施必须逐项对应 provenance，并明确基础设施失败与截断轨迹的训练处理。

第一批工作应先核对原生 vLLM/verl HTTP 训练接口，确定实际所需的最小轨迹适配，再交付 CPU 验收；随后才注册带完整环境连接的 loop。官方环境权限解决后联调，最后另行授权最多两卡的真实 rollout/更新/恢复检查。当前 `--launch` 的无条件拒绝继续保留；不以文件存在或 registry 导入成功解除保护。日志方面，`agent_rl_env.sh` 强制 offline 与父级规则不一致，拟恢复为不覆盖调用方模式的配置并审阅 trainer logger，但本轮不调整日志模式或训练参数。

**本轮实际验证**：现有 vLLM 0.19.1 的两个 Qwen 解析器，在真实固定 tokenizer + 合成字符串的 CPU 检查中，正确分离 thinking、提取原生 bash 工具及字符串参数，并把未结束的 thinking 保留为 reasoning。未执行工具、DSH session、GPU 生成或训练；这只验证解析器复用点，不等于 SSE/token 桥完成。证据：[parser-interface-check.json](runtime/agent-rl/parser-interface-check.json)。

新增上游源码快照、读取的本地文件及 SHA-256 统一记在 [source.json](research/sources/code-agent-implementation-20260909/source.json)。旧 `runtime/agent-rl/artifact-manifest.json` 是此前交付时的快照，保持原样；现行 README 与 DSH 接口元数据已后续更新，历史哈希不再代表这两个文件。既有训练执行代码和训练参数未改。

变更记录：2026-09-09 13:21 UTC — 按用户要求参考开源 code-agent 的真实实现，补充网关、统一工作区及训练轨迹接口映射，明确 SkyRL/rLLM 示例不能直接照搬的细节，记录 DSH 本地源码与声明版本的局部不一致。仅更新研究来源与本文；未安装依赖、修改训练/agent 代码、启动模型、容器、benchmark、rollout 或参数更新。`training_launch_ready=false` 保持不变。

变更记录：2026-09-09 14:00 UTC — 按用户“看看怎么改”核对实际调用链，纠正上一条记录中的 DSH 来源误判和接口清单错误版本，确认 release 的 6 个源码文件一致；给出具体文件/接口/验收的修改设计，并通过无 GPU 的 Qwen 解析器样本检查。新适配模块尚未实现，未改训练参数或放开启动保护。

变更记录：2026-09-09 14:03 UTC — 按用户关于 vLLM 已支持 DSH 的纠正，区分已验证的推理兼容与待验证的 RL 轨迹接入；在线核对最新发布 0.29.0，改为优先复用原生 Chat Completions 和训练数据字段，撤回默认重建模型流式协议的方案。未升级依赖或启动服务。
<!-- code-agent-implementation-review:end -->

<!-- agent-rl-preparation:start -->
**Agent RL 训练工程准备（2026-09-09，独立准备完成；在线训练启动尚未就绪）**

交付目录：`.venv-train-rl`、`vendor/verl-agent-rl`、`configs/agent-rl/`、`data/agent-rl/`、`runtime/agent-rl/`。本轮 GPU/Ray 服务、任务容器、rollout、benchmark、参数更新均为 **0**。没有修改已有 DSH、推理环境、共享环境或上游训练源码。GPU 未分配，也未检查/使用其他任务的卡。

**实现选型与来源区分**

| 对象 | 核对结果 | 本工程决定 |
|---|---|---|
| Together AI / Agentica 2025 年 DeepSWE-Preview | [发布说明](https://www.together.ai/blog/deepswe)是 Qwen3-32B + R2E-Gym 多轮 RL；公开规模 64 H100、约 4,500 任务。已存 2025-07-03 发布窗口源码 `rllm@6458960c2478dc348c10b23e67f2789d7c1745d1`，其 gitlink 固定 `agentica-project/verl@777704aa64c5745c4ccb250ae8977469579ebfc0`。历史 `train_deepswe_32b.sh` 实际写 `adv_estimator=loop`、n=8、4096/32768、SP/TP=8、2×8 GPU、lr=1e-6、50 步、5400 秒 | 作为历史实现参考。脚本的 16 卡设置与发布实验 64 卡规模分开记录；未找到能证明该快照正是公开权重完整训练配置的证据。这里算法名字 `loop` 不是本项目内部 LoopLM 架构改造；本工程也没有照搬它 |
| 当前 rLLM | 固定 `3b40c37cf6a262cf4d28cc987ebe4f4cf797956c`，新版 UnifiedTrainer/VerlEngine、轨迹转换、gateway 与多后端，依赖声明包含 `verl==0.8.0`、`vllm==0.22.1`、Transformers≥5.5.3、FlashAttention 2.8.3 | 仅源码对照，未安装 rLLM 的全部后端。未发现现成 DSH 接入；不能将最新框架当历史 DeepSWE 配方 |
| 当前 rLLM 的 `deepswe` 数据入口 | `rllm/data/deepswe_builder.py` 拉取 **datacurve-ai/deep-swe 的 113 个 Harbor 格式测试任务** | 与 Agentica DeepSWE 训练模型及 R2E-Gym 训练集不同，未运行该数据入口、未将其当训练数据 |
| 主训练栈 verl | **固定 `1252cc71aa5bd82e5604322064d69bfe6454c660`**；[Qwen3.5-27B FSDP2 示例](https://github.com/verl-project/verl/blob/1252cc71aa5bd82e5604322064d69bfe6454c660/examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh)、LoRA/FSDP2、AgentLoop、V1 同步训练器、checkpoint 与 merged-LoRA 同步均有真实源码 | 包版本为 **0.10.0.dev0（固定开发 commit，非稳定发布版）**；采用上游 `verl.trainer.main_ppo` → V1 `PPOTrainerSync`，算法为上游 **GRPO**。不重写训练循环/算法/评分器。该组合是本项目候选，**不是 DeepSWE 复现，也不是双卡验证配方** |

选型期间保留两类失败：verl v0.8.0 的 `numpy<2` 与 vLLM 0.19.1 的 OpenCV≥4.13/NumPy≥2 不能同时解析，见 `runtime/agent-rl/dependency-conflict-v080.log`；因此改为已修复 NumPy 约束的上述 commit，没有 `--no-deps` 强装。v0.8.0 的 `[vllm]` extra 上限还停在 0.12，不能用于 Qwen3.5。当前 commit 也包含 FSDP2 CPU offload 与 merged-LoRA 同步修正。vLLM 固定 0.19.1 是已有推理 wheel 家族的独立安装；上游 27B 示例注释使用 0.18.0，故本组合仍需 GPU 实证。

完整源码 commit、研究文件 SHA-256、历史子模块信息见 [source.json](research/sources/agent-rl-prep-20260909/source.json)。未运行仓库附带的 benchmark/下载/训练脚本。

**独立环境与检查证据**

- Python 3.12.12，Torch 2.10.0+cu129，vLLM 0.19.1，Transformers 5.5.4，PEFT 0.18.1，Ray 2.55.1，NumPy 2.2.6，TensorDict 0.10.0。全部 **229 个包**已安装并通过依赖检查；`include-system-site-packages=false`，使用复制安装而非修改其他环境。
- 上游 V1 训练器需要 TransferQueue，固定 `d45bfe84c9940ee7650e624dba3d60c5dbf95c62`，作为本地训练数据传输依赖安装。它与平台 DockerProxy 的 job_queue 无关；本轮没有启动任何队列服务。
- 输入版本锁：[agent-rl-requirements.in](configs/agent-rl-requirements.in)；解析结果：[agent-rl-requirements.lock](configs/agent-rl-requirements.lock)；带 wheel URL/大小/SHA-256 的 [pylock.agent-rl.toml](configs/pylock.agent-rl.toml)。editable 上游源码另由 commit/文件清单固定。离线重建 dry-run 结果为 `Would make no changes`，不是在另一台机器完成重建的声明。
- 模型复用 `models/Qwen3.5-27B/fc05daec18b0a78c049392ed2e771dde82bdf654` 和同目录 tokenizer。既有完整性证据 `runtime/metadata/qwen-files-check.json` 保留，本轮没有重新下载/改写权重。上游 `get_hf_auto_model_class` 正确选择 **AutoModelForImageTextToText → Qwen3_5ForConditionalGeneration**，不是误用普通 AutoModelForCausalLM。
- 对真实 27B 配置做 **meta-only** 构造，并调用上游 `FSDPEngine._build_lora_module`：27,356,728,560 个 base 参数，192 个文本 FFN 线性层匹配，69,206,016 个 LoRA 可训练参数，其他参数冻结。无真实权重存储、无模型 forward/backward、无优化器更新；不能当作 27B 实际加载或显存测量。
- 已通过训练器/V1/FSDP/vLLM adapter/checkpoint 导入、Hydra 解析、上游 `validate_config` 与 RolloutConfig 构造、实际 RLHFDataset/tokenizer 读取。CPU 合成 token fixture 用上游 `AgentLoopOutput` 和 `agg_loss` 验证工具返回 mask=0、对应梯度为 0；不代表真实 DSH 轨迹 mask 已打通。
- 初次检查中发现 V1 所需 TransferQueue 缺失与新版 `skip` 配置位置变化，均已修复；失败证据保存在 `runtime/agent-rl/acceptance-attempt-1.json`。一次 PyPI 下载超时保留在 `install-attempt-1.log`，延长单次下载超时后安装完成。

最终验收见 [acceptance.json](runtime/agent-rl/acceptance.json)。**未验证**：27B BF16 GPU 训练前向/反向、FSDP2 分布式 LoRA 权重同步、vLLM sleep/wake 数值一致性、真实 checkpoint 保存恢复、显存和吞吐。当前以 SDPA/原生 Torch GDN 路径作为候选，没有安装 FlashAttention/FLA/causal-conv1d 训练快核；其性能不能由已有推理 smoke 外推。

**训练任务入口与去污染**

SWE-Gym 固定 HF revision `bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`，本地 2,438 行。字段包含 instance_id、repo、base_commit、problem_statement、patch/test_patch、FAIL_TO_PASS/PASS_TO_PASS。RL 输入只把 **problem_statement** 放进 user prompt；任务/环境/验证器引用保存在 extra_info，原始 gold/tests 留在 `data/agent-rl/private/`，未来不得挂载到 agent 工作树或序列化到模型上下文。Parquet 的空 ground_truth 只是上游 schema 兼容字段，**不提供默认正确答案或伪奖励**；需要环境验收后由 AgentLoopOutput.reward_score 返回上游结果。

已用原有 `gym_holdout.classify`，叠加完整 SWE-bench test、Verified、三个 SWE-smith 题、各版冻结开发身份、repo/base_commit、标准化问题和同仓库标题相似度≥0.8 筛查。2,438 行中：2,357 行通过当前身份规则，26 行按 ID/commit 排除，54 行身份未知/歧义隔离，1 行近重复标题隔离。不是对语义近重复或底座预训练污染的完全保证。所有关联轨迹继承任务排除规则；未知轨迹不因无 ID 命中而被视为合格。本轮 **没有消费/转换已有 SFT 对话**。

按 ID 字典序、每仓库≤2题、问题≤10,000字符，确定 8 题供未来联调（不是代表性训练集或已通过环境检查的任务）：

`Project-MONAI__MONAI-1010`、`Project-MONAI__MONAI-1011`、`bokeh__bokeh-12779`、`bokeh__bokeh-12841`、`conan-io__conan-10213`、`conan-io__conan-10686`、`dask__dask-10009`、`dask__dask-10027`。

实际 tokenizer 问题 prompt 长度为 53–2,864 tokens，尚不包括 DSH system/tools，在线接入后必须重新检查完整前缀。清单、源文件 SHA-256/原始行号/私有记录 SHA-256 见 [manifest.json](data/agent-rl/manifest.json)，逐行排除结果见 [exclusion-audit.json](data/agent-rl/exclusion-audit.json)。当前 train.parquet 已由上游 RLHFDataset 读入 8 行，每题 index 独立。

R2E-Gym-Subset 固定 revision `2e8108ff942f24fcb5686badfaf7f9a8808566d5`，本轮检查已有 **7 个完整分片**，保留缺失/未完成分片状态。字段为 repo_name、commit_hash、docker_image、problem_statement、parsed_commit_content、execution_result_content、expected_output_json 等；其中 `prompt` 是上游构造 issue 的提示，不能当 RL 任务 prompt，parsed_commit_content 含 oracle 信息，不能直接发给模型。短仓库名与 commit 还缺跨源 canonical repo/任务身份映射，故 **全部暂隔离、未导出为训练样本**。逐分片字段、行数及哈希已写入同一 manifest。既有 R2E-Gym/OpenHands SFT trajectories 与这些任务记录分开处理。

选定 SWE-Gym 任务未来调用已固定 `SWE-Gym/SWE-Bench-Fork@242429c188fcfd06aad13fce9a54d450470bf0ac` 的环境/验证器；R2E 参考代码为 `agentica-project/R2E-Gym@353348a0ff690f2592025eff41b3fef4201a4d8b`。没有重写任务环境、测试选择或评分，也没有把 SWE-Gym 行塞入 R2E 的不兼容 schema。当前任务环境均 `environment_qualified=false`，不代表未来训练 reward 已可用。

**两张 H100 的首个配置候选（供审阅）**

配置：[qwen35_27b_lora_2h100.yaml](configs/agent-rl/qwen35_27b_lora_2h100.yaml)。所有 128 个覆盖字段的上游路径、调整理由和预算状态见 [provenance.json](configs/agent-rl/provenance.json)；继承默认值固定在当前上游 commit 和 `runtime/agent-rl/resolved-config.yaml`。

| 项目 | 候选设置 |
|---|---|
| GPU /阶段分配 | 一节点最多2张 H100 80GB；同两卡 FSDP2 分片训练与 TP2 vLLM rollout 交替，无额外 reference/critic/teacher 卡 |
| 上游切换机制 | V1 `PPOTrainerSync.on_sample_end` 调 `sleep_replicas`；`on_step_end` 调 `update_weights`；FSDP2 param/optimizer/offload_policy=true；vLLM free_cache_engine=true，merged-LoRA 路径使用 level-2 sleep 后完整权重恢复 |
| 模型/显存策略 | base BF16、梯度 checkpoint、文本 FFN LoRA rank16/alpha32/bias=none/dropout=0；merge=true，使用上游合并/恢复和权重传输；不量化，不改模型结构。SDPA、无 packing/SP、禁用 torch compile/CUDA graphs，优先减少首轮变量 |
| 上下文/多轮 | 总8192=初始prompt最多4096+response区域最多4096；后者包含工具观察与模板 token，不是每轮都可输出4096。最多12 assistant轮/12请求、单工具输出最多1024 tokens；最终由待接入桥严格执行 |
| batch/GRPO | 2个不同任务×每组2条轨迹=每step 4条；minibatch为2个原始任务，microbatch每GPU 1、ppo_epochs=1；temperature=1/top_p=1/top_k=-1、seed42；GRPO标准化，clip=0.2，entropy=0，无KL loss/reward |
| 更新 | 上游 AdamW，lr=1e-6、betas=(.9,.999)、weight_decay=.01、grad_clip=1、constant LR、warmup=0；首个后续联调仅1个更新step，**本轮0步** |
| checkpoint | save_freq=1，保留最多2个；model/optimizer/extra由上游保存，trainer保存data.pt等状态。初始resume=disable；后续指定真实global_step目录并resume_path，不能仅加载adapter冒充优化器/RNG/数据位置恢复 |
| 日志/评测 | 项目内 console/tensorboard 与 rollout日志；val_before_train=false/test_freq=-1，禁用benchmark。val_files仅指同一小训练文件满足loader构造，不进行验证，不以训练任务充当独立指标 |
| 运行预算候选 | task环境并发1、每轨迹600秒、4条轨迹、总job 3600秒（若落实则≤7200 GPU卡秒）；8 CPU、192GiB主机RAM作为平台申请候选；这些wall-time/任务并发上限尚须桥/平台入口执行，不能把num_workers=1/max_num_seqs=1误当任务容器并发限制 |

BF16 base 仅参数约54.7GB十进制；两卡总显存160GB并不证明可训练。仍需考虑全词表 logits、长序列激活、GDN状态、合并权重时的CPU备份、FSDP all-gather/保存峰值和vLLM缓存。LoRA节省更新状态，**不消除base模型或rollout副本**。不承诺双卡8K可行或吞吐。两条轨迹同为成功/失败时GRPO缺乏组内相对信号，必须保留并报告，不能筛掉零奖励组冒充训练有效。

**DSH真实接口和尚缺适配**

完整输入输出及源码位置见 [dsh-interface.json](configs/agent-rl/dsh-interface.json)，状态为 `adapter_not_implemented`：

1. 复用已存在的 `DeepSeekHarness.run(..., session_id, on_notification)`/`close()`、DSH工具schema与结果协议。shell和editor必须绑定同一上游任务执行环境；仅设置宿主cwd不足以隔离任务，且不能把gold/test信息发给policy。
2. 在上游 `AgentLoopBase.run` 接入一个最小 DSH session/model bridge，注册 `dsh_agent`。上游 `LLMServerClient.generate(request_id, prompt_ids, sampling_params)` 已返回 `TokenOutput.token_ids/log_probs/stop_reason`，应直接保留并关联任务、请求、policy global_step。DSH当前 `WireChoice` 只有delta/finish_reason，usage只是计数，**已有SSE日志不能恢复真实采样token/logprob**。
3. 桥输出 `AgentLoopOutput(prompt_ids,response_ids,response_mask,response_logprobs,reward_score,num_turns,metrics)`：generated含thinking token的mask=1，工具/插入上下文/padding=0，数组长度严格对应，复用上游continuous-token builder。DSH若压缩/重写历史导致非连续前缀，应显式拒绝/结束该短联调轨迹，不静默拼接成错误on-policy样本。
4. 从候选patch调用固定上游验证器，在干净环境验收；未修复/gold的环境资格检查留给后续在线阶段。基础设施失败独立记录并终止有界联调，不伪装成测试失败的0奖励。完成/超时/context截断/tool格式错误分别记录并清理自有session/环境。

现成 `BaseTool.create/execute/calc_reward/release` 和 `ToolAgentLoop` 可作工具接入参考；只跑它就称 **verl参考agent**，不能称DSH。没有新增harness策略或自写RL框架。DockerProxy权限问题仅记录为未来在线执行条件，本轮未测试队列/容器、未修改开发机、未索要队列信息。

另外查到上游 colocated weight-transfer 在 `vllm_rollout.py`/`utils.py` **硬编码 `/tmp` IPC socket**，只设置TMPDIR无法覆盖。未来平台job必须在自己的mount namespace把`/tmp`映射到项目内临时目录，同时满足容器数据的项目内存储约束；本轮未修改挂载或上游代码。正式训练仍遵守父AGENTS的Training Task规则。

**已验证的入口与未就绪状态**

```bash
cd /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
source scripts/env.sh
# 使用已固定源码/项目缓存复核安装，不启动训练。
bash scripts/install_agent_rl.sh --offline
# CPU、无GPU/无Ray启动的依赖/模型meta/数据/配置检查。
bash scripts/agent_rl.sh --check
# 调用真实 verl.trainer.main_ppo 的 Hydra --cfg job --resolve；仅解析。
bash scripts/agent_rl.sh --config
# 明确返回exit 2，列出真实缺项；不会启动训练。
bash scripts/agent_rl.sh --launch
```

`--launch` 目前没有可绕过的执行开关。真实训练入口属于上游 `verl.trainer.main_ppo`，但本项目 `dsh_agent`尚未注册，因此**尚不存在已验证能启动DSH RL的训练命令**；没有用占位模块或普通聊天奖励冒充。缺项见 [readiness.json](configs/agent-rl/readiness.json)：最小DSH token/session桥、统一任务执行与原验证器连接、完整prompt/budgets联调、GPU分配和双卡数值/显存/保存恢复短检、项目内临时挂载与在线执行权限、后续参数更新授权。容器问题没有阻止本轮任何独立准备。

变更记录：2026-09-09 13:02 UTC — 按本轮Goal完成独立Agent RL训练工程准备，固定上游与依赖、导出8个有排除审计的任务引用、检查LoRA/mask/配置接口；准确区分CPU验收与尚未实现的DSH在线训练。未执行benchmark、容器、rollout或参数更新。
<!-- agent-rl-preparation:end -->

<!-- industry-agent-rl:start -->
**工业界训练路线修正（2026-09-09）：Agent RL 为主线，SFT 是条件分支**

用户要求参考工业界方案。此前建议被当前容器阻塞和两卡预算过度牵引，将方便先做的离线 SFT 写成了必经顺序；现修正。SFT、RL、OPD 是不同的学习信号/训练目标；LoRA 是参数更新方式，可以用于 SFT、RL 或 OPD；内部 LoopLM 是模型架构改造，不等于上述任何一种训练方法。保留最终 DSH 目标，不因已有 Klear 文本命令轨迹就自动将主线切成 mini-swe-agent-plus。

| 一手来源 | 公开的真实方法 | 本项目应如何参考 |
|---|---|---|
| [Together AI + Agentica / DeepSWE](https://www.together.ai/blog/deepswe) | 在已有 Qwen3-32B 上直接做多轮 RL，使用 R2E-Gym 训练环境和任务完成的可验证奖励，训练框架 rLLM；公开资源为约4,500任务、64×H100、6天 | 作为可复现 RL 实现的首要参考。这里“RL-only”指该次后训练，不代表底座从未SFT；证明额外SFT不是普遍前提。其64卡结果不是本项目两卡可行性证明 |
| [Qwen3-Coder-Next 技术报告 v1](https://arxiv.org/html/2603.00729v1) | continued pretraining/mid-training → SFT → 专家训练（含多轮 SWE RL）→ 专家蒸馏；SWE 专家依靠环境反馈，筛选有用难度，分离 SFT/RL prompts | 作为工业流程参考：任务与执行环境质量、轨迹反馈、难度分布都重要，不能仅抄GRPO名字。报告的专家蒸馏章节没有给出足以认定其必为本项目所说OPD的细节 |
| [Thinking Machines / On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | student 自己生成轨迹，teacher 在 student 访问的上下文上提供逐token分布监督；与仅模仿teacher离线答案不同 | OPD 是另一种学习信号，可在方法明确时与环境奖励配合；不能等同于测试通过奖励，也不能用普通聊天API返回一段答案冒充token概率评分。其数学/助手任务结果不证明代码agent组合已验证 |
| [Thinking Machines / LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) | 实际包含LoRA的策略梯度RL实验 | 说明LoRA和RL不冲突；数学实验中LoRA与全参的比较不能外推成27B长轨迹SWE任务相同效果或双卡显存保证 |

**对当前27B + DSH的建议，以下是基于资料的工程判断，而非某家未公开配方的复现声明：**

1. 先接通上游任务环境/验证器，并确认DSH所有工具操作同一任务工作树。用独立于最终测试集的训练任务检查原始27B：工具格式是否可靠、是否能完成完整轨迹、是否存在成功与失败的奖励差异。当前本地工具smoke不能回答这些问题。容器基础设施错误单独记录，不当成模型能力失败或正常训练奖励。
2. 若已有可学习的成功/失败信号，优先试 **直接 LoRA Agent RL**；不为走流程强行增加SFT。若主要是工具协议错误或在有效环境中几乎全部失败，先诊断协议/任务难度，再考虑少量、匹配DSH的成功轨迹做SFT冷启动。纯终局0/1奖励的GRPO在同组全成或全败时缺少组内相对优势；不能靠无限增加失败rollout解决。
3. 实现首先审查DeepSWE的rLLM训练入口及其后端，而非自行编写训练循环。已有参考 `rllm-org/rllm@3b40c37cf6a262cf4d28cc987ebe4f4cf797956c` 的依赖声明包含 `verl==0.8.0`；这不保证它就是历史DeepSWE发布配方，也不证明支持当前Qwen3.5/LoRA/DSH组合。下一步需要核对发布配方commit、模型后端、工具返回loss mask、rollout实际采样token/logprob、权重同步和adapter恢复；DSH到训练栈的连接尚未验证。若只跑通上游R2E agent，必须明确标为参考agent，不冠以DSH。
4. 第一阶段模型结构保持原始27B，建立真实RL对照后分别加OPD、内部LoopLM；这样能区分环境奖励、teacher监督和架构改造的收益。OPD需先确定teacher、评分接口、tokenizer对齐和额外成本；有可用teacher再设计，不能默认两卡同时常驻多个27B模型。LoopLM作为后续独立研究，不在首个工业式RL闭环里同时改动。
5. 训练使用SWE-Gym/R2E-Gym等获选训练集的上游环境和验收；SWE-bench测试集只作独立官方评测，不作为RL训练奖励来源。固定数据、环境、agent和预算，保留基础设施失败及全部模型失败。训练奖励与公开benchmark判分是两个用途，不能修改官方判分来迎合训练奖励。

两卡约束影响可运行的规模和部署，不应改写研究目标。27B BF16多轮LoRA-RL在两张H100上的显存、轨迹长度、吞吐，以及框架是否支持交替rollout/更新或卸载，均未验证；优先查上游现成能力并做受控可行性检查，不承诺把64卡配方简单缩为两卡即可复现。正式持续训练仍按平台训练任务规则执行。DockerProxy当前队列权限问题是本地在线Agent RL的实际执行阻塞；离线SFT可独立准备，只作为条件分支，不用它替代RL目标。

资料快照与SHA-256：`research/sources/code-agent-training-start-20260909/industry-rl/source.json`。本轮仅调研及修正文档，无依赖安装、数据加工、模型调用、容器启动、SFT/RL/OPD训练或checkpoint生成。

变更记录：2026-09-09 12:13 UTC — 根据用户“仍要RL、参考工业界方案”的澄清，将SFT必经顺序改为依冷启动能力决定，明确DeepSWE实现参考、Qwen工业流程、LoRA/OPD/LoopLM的不同角色与未验证接口。
<!-- industry-agent-rl:end -->

<!-- code-agent-training-start:start -->
**开源 code-agent 训练路线核对（2026-09-09；调研完成，训练未开始）**

此前提出的 **现成轨迹 → ms-swift 的 Qwen3.5 BF16 LoRA SFT → 官方 SWE-bench 环境与评测器对照** 保留为可选的离线参考实验，已不再作为必经主线；最新工业界 RL 路线见上一节。离线 SFT 读取已存在的对话轨迹，不依赖任务容器运行；当前 DockerProxy 队列错误阻塞在线执行、采集和官方评测，但不必阻塞离线训练准备。不能把训练 loss 降低或本地工具连通称为 agent 能力提升。本轮仅阅读源码/配置、抽查已有数据格式、保存参考来源并更新本文，没有安装训练栈、导出训练集、启动 GPU、训练或评测。

| 上游项目 | 实际开放内容与已读配置 | 对本项目的用法与边界 |
|---|---|---|
| [Klear / mini-swe-agent-plus](https://github.com/Kwai-Klear/mini-swe-agent-plus) | 公开约 66k SWE-smith 轨迹，README 明确使用 ms-swift 训练 Qwen3-8B；开放 agent 工具和运行配置 | 最接近当前已有数据和轻量 SFT 起点；未发现可据此直接认定已验证的 Qwen3.5-27B、双 H100 配方 |
| [SWE-Gym / OpenHands](https://github.com/SWE-Gym/SWE-Gym/blob/b681068ca20628c6987b7416cc4cf03f06b77ba5/docs/OpenHands.md) | 成功轨迹 SFT；torchtune policy 配置为 Qwen2.5-Coder-32B 全参数、32768 长度、BF16、assistant-only loss，文档训练入口 N_GPUS=8 | 借鉴完整轨迹训练和评测闭环；不是直接适用于本项目两卡的 LoRA 配置，YAML 头部残留 7B/两卡示例注释不能当作实际32B资源要求 |
| [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym/blob/0d94c4eb9431cd195c55a7ea3abd54006c9a1735/train/train_r2egym_32B_agent.yaml) | 提供真实 LLaMA-Factory SFT YAML：Qwen2.5-Coder-32B、full、ZeRO-3 offload、20480 长度、BF16、2 epochs | 可参考训练方法；原配方不等于 Qwen3.5 双卡可运行，也不将现有未确认任务身份的 R2E 轨迹直接混入训练 |
| [SWE-RL](https://github.com/facebookresearch/swe-rl) | 当前 README 明确代码主要为提示模板与序列相似度奖励；reward.py 对比模型修改与 oracle 修改 | 不是完整可直接启动的在线 agent RL 训练栈，不能用该仓库名替代缺失的训练实现 |
| [SkyRL + mini-SWE-agent](https://github.com/NovaSky-AI/SkyRL/tree/ba3487ae66917d37d82d4c4482c44f4e252f45b0/examples/train/mini_swe_agent) | 提供生成轨迹、干净环境评测、Ray 调度的在线训练集成；示例使用 Podman，8B 标注 8×H100、30B 标注 16×H100 | 适合后续研究在线 RL；上游示例超出当前两卡预算，这不证明所有小规模 RL 都不可行 |
| [ms-swift Qwen3.5 文档](https://swift.readthedocs.io/en/v4.0/BestPractices/Qwen3_5-Best-Practice.html) | 明确支持 Qwen3.5 Dense/MoE；Dense 推荐 Transformers 后端，有 LoRA SFT、adapter 保存/加载示例 | 首选训练器。所读 v4.0 文档的 Dense 实例是 4B、2048 长度，不能外推为27B长轨迹显存验证；源码快照与文档均为参考，尚非已安装/验证的依赖锁 |

**数据与 DSH 协议是必须解决的具体接口问题。** 本地 Klear 已有 65,994 行、10,894 个不同任务，不能把行数当独立题数。只读抽查首条 parquet 记录：字段为 instance_id/messages，消息只有 role/content；system 要求 THOUGHT 和一个 bash 代码块，执行返回作为 user 消息，首条共171条消息。因此不能仅改 role 或工具名称就称为 DSH 原生 function-call 数据。该抽查不是逐行格式审计，且现有数据没有逐行 resolved/teacher 字段，不能称全部已经过本项目成功验证。

如需要独立的离线 SFT 对照，可用 **Klear 原协议的离线 SFT 作为明确命名的参考实验**，使用原 mini-swe-agent-plus 做其 agent 对照；这是与最终 DSH 方案区分开的参考路线，未擅自切换当前 DSH 环境。最终若坚持 DSH 原生工具协议，需先选到匹配的真实轨迹，或在官方任务环境内用 DSH 采集并通过上游测试；不能将文本命令轨迹伪装为 DSH 轨迹。已有 DSH 本地连通记录不等于大规模、成功验证的训练语料。训练器与评测器复用上游，仅需固定配置及必要的数据格式适配，不再编写替代训练循环或评分逻辑。

**可选 SFT 分支的实施建议（尚未执行、不是论文原配方或已冻结参数；不作为 RL 的强制前置）：**

1. 建立项目内独立 `.venv-train`，以 ms-swift 对 Qwen3.5 的实际支持固定 trainer、Transformers、PEFT、CUDA 扩展及依赖；使用已有固定 revision 的27B BF16权重。先核对 tokenizer/chat template、thinking 格式、assistant-only loss mask、LoRA 注入模块和视觉分支冻结。推理 `.venv-runtime` 保持独立。上游 v4.0 文档提示 Transformers GatedDeltaNet 的 packing/padding_free 限制，必须按选定版本验证，不能照搬旧 Qwen2.5 配置。
2. 拟用不超过32条完整短轨迹、约20个 optimizer steps 做训练设施检查，最长上下文先以不超过8K作资源探针；验收反向传播、非零有效监督 token、有限梯度/损失、只有预期 LoRA 参数更新、checkpoint 保存/恢复及 adapter 重新加载。具体 batch、长度、rank、学习率和分布式配置先形成可审阅配置，运行前再确定。这些步数仅是训练设施检查预算，与 benchmark 的官方测试规则无关。
3. 通过后，拟扩大至约1,000个可追溯、去重的独立训练任务做首个 SFT 对照。按任务身份划分训练/验证，执行既有 `configs/gym-training-exclusion.json` 排除规则，隔离官方测试集和所有用于调试/调参的题目；不以随机行切分掩盖同题多轨迹泄漏。原始 Klear 抽样内容长度中位数约17.7K、P90约32K，短轨迹8K smoke 不代表正式数据分布，不静默截掉轨迹结尾；最终长度与显存实测后再定。可追溯身份不等于成功标签，筛选须记录所依据的来源/验证证据。
4. 官方环境可用后，以固定题目、同一 agent 协议和推理预算，对比原始27B与SFT adapter；补丁统一交给固定版本官方 SWE-bench 评测器。可以先小子集验证通路，但小样本不作为可靠的整体提升结论。原协议参考实验的成绩不得冠以 DSH；DSH 迁移效果另行验证。
5. 本分支若实施，作为与直接 RL 路线比较的 SFT 对照；不要求先完成 SFT 才允许研究在线 RL。内部 LoopLM 改造与 OPD 分别加入对照。两张 H100 下同时放 student、teacher、rollout 服务的资源布局尚未验证；本轮没有现成 DSH + Qwen3.5-27B + LoopLM/OPD 可直接复制的训练配方证据。

资源边界：最多两张明确获分配的 H100 80GB。已有单卡约65GB的32K推理实测不证明训练可行；27B BF16权重本身约54GB十进制，训练还需要激活、梯度等。双卡 LoRA 的可用长度和并行策略必须实测，不承诺32K训练可行。持续训练按父目录 AGENTS 的开发机规则使用平台 Training Tasks；所有环境、缓存、checkpoint、日志仍放项目个人卷内，不能把平台单worker默认8卡当作8卡授权。离线SFT不需要额外运行SWE任务容器；正式评测/在线采集仍需解决容器入口。训练期间同两卡上的既有推理服务应退出。

研究快照与逐文件 SHA-256：`research/sources/code-agent-training-start-20260909/source.json`。所读源码 commit：Klear `3dfa5e26831306978ff3cfa2da15b49113ded0e6`；SWE-Gym `b681068ca20628c6987b7416cc4cf03f06b77ba5`；R2E-Gym `0d94c4eb9431cd195c55a7ea3abd54006c9a1735`；SWE-RL `5aa10d67f1db07ecd3a20483d63fe2d5029ac9c9`；SkyRL `ba3487ae66917d37d82d4c4482c44f4e252f45b0`；ms-swift `ddd16cb15ec90d2b122fbf1e57d7fd2a259427fc`。ms-swift 主线旧文档路径返回404，改读并保存上述版本化在线文档；没有把404记录当成已读取源码。

变更记录：2026-09-09 12:08 UTC — 按用户要求从开源训练项目出发明确 SFT 起点、数据/DSH 协议边界、两卡资源验证和后续官方对照；纠正将 Docker 视为离线训练先决条件的方向偏差。当前训练步骤为0，训练环境、训练集导出、LoRA checkpoint 均未完成。
<!-- code-agent-training-start:end -->

<!-- glm-docker-docs:start -->
**平台 Docker 说明核对（2026-09-09）**

用户提供的两份 GLM 官方说明已读取，原始 HTML、UTF-8 正文和哈希保存在 `research/sources/GLM-Docker/`。文档明确：不启用 DockerProxy 时，平台支持在开发机/训练任务容器内运行原生 Docker-in-Docker；启用 DockerProxy 时，请求转发至外部 Docker 集群。此前在已启用 Proxy 的当前开发容器内手动启动第二个 dockerd 失败，不能推导出平台正式提供的原生 Docker 模式也不可用。

DockerProxy 文档将 `cannot use job_queue: %s` 明确列为目标队列使用权限检查失败，与本次实际错误完全一致。创建开发机/训练任务时选择的 DockerProxy 队列会自动注入容器标签；文档也支持通过 `platform.glm.ai/job-queue=<获授权队列ID>` 指定队列。因此修复代理路径可以是选择可使用的队列或开通现有队列权限，不一定需要复制开发机。没有获授权队列 ID 时不猜测或尝试其他命名 socket。

另一条平台支持的路径是：若决定新建/复制用于本地小子集评测的开发机，在平台配置中不启用 DockerProxy，使用平台提供的原生 Docker。此选项是对本地调试的建议，尚未创建、验证；不能只在当前进程里取消 DOCKER_HOST 就宣称原生模式已启用。个人卷必须保留原路径并允许写入，Docker 镜像层、容器存储与日志的项目内位置仍需核对落实。原开发机不在新环境验证前停止或删除。参考：[DockerProxy 快速开始](https://platform.glm.ai/docs/platform/dockerproxy/proxy-start)、[原生 Docker](https://platform.glm.ai/docs/platform/docker/dind)。

用户再次确认当前开发机已启用 DockerProxy。继续排查当前代理的队列授权，不据此要求重建开发机。只读查询检查发现当前 PATH 无 `platformctl`；镜像内 `/ml_platform/bin/mlpctl --help` 依赖缺失的内部 admin.yaml 而退出，不能当作可用的用户队列查询工具。已读取官方 `platformctl <CLUSTER> jobqueue list` 与队列管理说明，未修改或猜测授权；下一次容器探针需要已获授权的 DockerProxy 队列 ID，或由队列管理员给当前队列开通权限。
<!-- glm-docker-docs:end -->

<!-- shared-docker-probe:start -->
**平台共享 Docker 实测（2026-09-09 11:11 UTC）**

按用户“你测试一下”的明确要求，实际测试当前 `DOCKER_HOST=unix:///ml_platform/docker/docker.sock`。复用已有 `ubuntu:22.04` 镜像 ID `sha256:ce941a2a18bbb922e434d6d6d2b31e571a5c3826eaf6ada0a41dcc905bd2d906`，未拉取镜像；测试不挂载项目、不使用 GPU、不联网，要求只读根目录、关闭容器日志驱动，限制 CPU/内存，并准备在结束后删除唯一测试容器。

结果：`docker create` 返回 `Error response from daemon: cannot use job_queue: queue-20260414132934-1amtr`，退出码 1。容器未创建，容器内命令没有执行；随后按唯一名称检查无残留容器。日志和全部参数在 `runtime/official-swe/metadata/shared-docker-probe-97c483fb55.json`。

`docker version` 显示该入口的 Server Platform 为 **docker-proxy**，不是直接操作本机普通 Docker daemon；其 Engine 信息报告 28.0.4，客户端为 20.10.18。当前能查询镜像和 Docker 信息，但默认平台任务队列不可用或无权使用，具体是队列状态还是授权配置仍需平台侧核实。此故障与前述“项目内嵌套 Docker 无权创建 mount namespace”不同；不能再把共享入口描述为“仅未测试”。需要为平台 Docker 代理配置可用且获授权的任务队列。没有改用其他命名 socket 规避队列限制，没有修改共享 Docker 或启动训练/评测。
<!-- shared-docker-probe:end -->

<!-- swebench-requirements:start -->
**SWE-bench 评测需求调研（2026-09-09；本轮未安装、未运行评测、未训练）**

本次对象是 `SWE-bench/SWE-bench` 主线，区别于此前 `SWE-Gym/SWE-Bench-Fork`。已读取官方文档并保存主线 commit `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` 的参考源码，版本字段为 **5.0.2**；仅保存在 `research/sources/SWE-bench__SWE-bench/`，没有替换已安装环境。官方任务定义仓库参考 commit 为 `SWE-bench/swe-bench-tasks@3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d`。来源、路径和逐文件 SHA-256 在两个 `source.json` 中。README 徽章、安装页和实际代码存在版本滞后：本次以固定 commit 的 `pyproject.toml` 为准，控制端 Python 要求 **>=3.10**；不能照抄旧网页的 3.8+/3.9+。

主流程是：Qwen + DSH 在官方缺陷仓库环境中生成补丁 → 将补丁按官方 JSONL 格式交给 SWE-bench → 官方容器执行测试并由官方解析器判分。生成补丁阶段需要模型推理资源；Verified/Lite 的官方任务判分主要消耗 CPU、RAM 和磁盘，不需要为每个测试容器分配 H100。官方评测不要求先训练模型，官方基础权重可以直接建立 baseline。当前没有 LoRA/OPD 训练、LoopLM 模型改造或训练 checkpoint。

| 必要条件 | 官方要求/本项目需要固定的内容 | 本项目当前状态 |
|---|---|---|
| 任务版本 | 数据集、split、revision、具体 instance IDs；Verified test 为 500 题，Lite test 为 300 题 | 已有 Verified revision `78f471bf655a3137b2e8a75af1501690ec009ec3` 的 500 行数据；已核对包含 `image`、`eval_script`、`log_parser`、`eval_type`。尚未选定 SWE-bench smoke 题目 |
| 官方评测软件 | 固定主线源码/安装依赖；v5 默认拉取任务镜像，`--task-repo` 则读取官方任务树构建 | 主线仅完成源码研究；`.venv-official-swe` 内是此前 SWE-Gym fork，不能称为已安装本次主线 5.0.2 |
| 官方任务环境 | 对应仓库 base commit、官方镜像或官方 Dockerfile、测试脚本及判分逻辑；运行时记录实际镜像 digest | 官方 SWE-bench 任务镜像未下载、未运行；先前确认的 xingyaoww 镜像属于 SWE-Gym，不能混用 |
| 可用容器执行节点 | 本地 Docker 能实际拉取/解包/创建并运行容器，或使用官方 Modal 云后端 | 共享 Docker 可连接但 data-root 位于项目外；项目独立 Docker 能启动但因挂载命名空间权限拒绝无法创建容器。两个事实不能混为“所有 Docker 都不可用” |
| DSH 的任务环境接入 | 读文件、编辑、Shell 必须操作同一官方任务工作树；问题可见，gold/test patch 与未来修复历史不暴露给模型 | Qwen + DSH 本地工具连通已验证；官方任务容器中的完整工具闭环未接通、未验证。不能只将 Shell 放入容器而让编辑器操作宿主 |
| Agent 协议 | 固定模型、提示、工具、上下文、thinking、采样、token/时间/尝试预算，并如实公开 | 原自定义 smoke 的 30 轮/600 秒不是官方标准，本次不自动沿用；复现某篇论文还必须匹配那篇论文的 agent 和预算 |
| 证据与报告 | 官方预测文件、patch、test_output、report、运行日志及真实 agent 轨迹；分开统计未提交、空补丁、未解决和基础设施错误 | 历史自定义记录保留但不充当本次成绩；尚无 SWE-bench 官方判分结果 |

官方基础资源建议为 **x86_64、8 CPU 核、16 GB RAM、至少 120 GB 可用磁盘**，属于运行环境建议，不代表每题恰需这些资源或 500 题镜像的精确总量。镜像、构建层、日志和并发会增加实际占用。当前 `nproc` 显示 112，但容器 CPU quota 为约 33 核且机器共享，不能据此占满；平台已有规则仍以项目内存储、受控并发和避免接近 100 GB 峰值 RAM 为约束。个人 Lustre 挂载实际剩余约 2.6 TB，并不代表本项目独占这些空间。初始 smoke 建议并发 1，具体资源预算应在真正执行前复核。27B 推理已有单张 H100 的 32K 连通实测，显存约 65,443 MiB；这只是现有推理布局的实测，不是官方 benchmark 的 GPU 要求。来源：[官方安装/资源说明](https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/README.md)、[实际 Python 要求](https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/pyproject.toml)。

数据规模以固定数据仓库的 split 元信息为准，而非旧网页汇总数。此次查询 Lite revision `b0dde1093fe417d83b7184254edf8199c1f0dff5` 的 test=300、dev=23；官方介绍页的 Lite=534 与当前元信息不一致，不用该旧数字计算分母。元信息快照在 `dataset-metadata.json`。模型对照需固定所选题目，不因成绩或环境错误静默换题；用于设施调试/调参的题目不能再充当未见最终测试集。

官方预测文件一行对应一个任务，核心字段如下；这里是格式示意，没有生成实际预测：

```json
{"instance_id":"sympy__sympy-20590","model_name_or_path":"Qwen3.5-27B+DSH","model_patch":"diff --git ..."}
```

官方判分核心是缺陷相关测试达到所要求的解决状态，同时回归测试维持通过。具体 skipped、xfail、测试 ID 解析及失败分类，以固定版本 `grading.py` 为准，不能自行写成“pytest 退出码为 0 就算成功”或修改测试选择。失败及基础设施问题保留，报告固定评测集合的分母。`--timeout` 默认 **1800 秒**是每个实例的测试执行超时，不是模型生成阶段的总时限。官方并未统一规定所有 agent 都必须 30/50/100 轮；与别人比较时，scaffold、尝试次数、是否 best-of-k 和预算同样影响结论。来源：[官方预测与执行说明](https://www.swebench.com/SWE-bench/guides/evaluation/)、[固定 CLI 参数](https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/swebench/cli/evaluate.py)、[固定判分源码](https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/swebench/harness/grading.py)。

**官方生成入口不等于 DSH**：`swebench infer` 默认封装 mini-SWE-agent。可以将其作为明确标记的参考 agent，但不能运行它之后把成绩写为 Qwen + DSH。保留 DSH 时，应只对接官方任务环境和标准补丁输出，再交给 `swebench eval`；不重写 benchmark 评测器。已安装 DSH 的文档也把本地进程 sandbox 与容器/远程执行区分开，所以“本地工具已通”不能推出“官方容器工具已通”。来源：[官方 CLI](https://www.swebench.com/SWE-bench/reference/cli/)、本地 `vendor/deepseek-harness/docs/subsystems/sandbox.md`。

下面仅展示核对过参数的 **v5 官方调用形式，未执行**。真正运行时必须先安装固定主线、准备固定 revision 的本地任务数据、确认镜像 digest/构建版本及容器权限；不能用当前旧 fork 环境直接冒充执行。`sympy__sympy-20590` 是官方 README 的示例，不代表已选择本项目报告集：

```bash
# 容器权限通过后的官方 gold 校验示例
swebench eval verified --gold -i sympy__sympy-20590 \
  -j 1 -t 1800 --run-id gold-smoke-01
# 模型补丁交给同一官方评测器
swebench eval verified -p preds.jsonl -i sympy__sympy-20590 \
  -j 1 -t 1800 --run-id qwen-dsh-smoke-01
# 要从固定官方任务树构建，再添加：--task-repo /项目内/固定任务仓库
```

相同 `run_id` 会复用已有结果，换补丁必须换 run_id。gold 检查不调用模型，可先独立完成。未修复基线也应使用该版本上游支持的负对照方式；空预测常被官方入口跳过，不能把“没有运行测试”写成 buggy 验证成功。建议执行顺序：先解决容器节点 → 单题官方 gold/负对照 → 同题 DSH 工具及补丁闭环 → 冻结小子集和 agent 预算 → 小子集执行及审计 → 再决定正式训练与保留集评测。这是调研建议，本轮没有启动其中任何执行步骤。

官方 Modal 后端能把判分放在云上，但需要账户、连通和计算预算；固定源码明确禁止同时传 `--modal` 与 `--task-repo`，不能假设远端会执行本地 Dockerfile。它也不自动解决 DSH 在解题阶段的远程工作空间接入。此前查到的 OpenHands Apptainer 是另一个上游维护的适配，不能直接等同本次 SWE-bench 主线 Docker 运行已验证。现有全部数据/缓存/日志留在项目内的要求，对外部执行同样需要落实。

本地研究评测和提交榜单是两件事。提交榜单另需公开预测、测试日志、轨迹和方法说明，并满足官方 experiments 仓库届时的资格政策；本轮不提交、不发布。来源：[官方提交要求](https://github.com/SWE-bench/experiments)。可比性上还应记录计算资源：厂商实测表明基础设施配置能改变 SWE-bench 得分，不能将微小变化直接归因于训练。Verified 也存在公开题污染及测试缺陷争议；OpenAI 在 2026-02-23 公布相关分析，所以它可作为公开对照之一，但不应独自支撑泛化能力提升结论。来源：[基础设施影响分析](https://www.anthropic.com/engineering/infrastructure-noise)、[Verified 局限性分析](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)。
<!-- swebench-requirements:end -->

<!-- lightweight-bench-options:start -->
**较轻量的模型评测选项（仅调研，尚未切换或执行；2026-09-09）**

截至目前没有启动训练，没有 LoRA/OPD 更新、LoopLM 模型改造或训练 checkpoint；实际运行一直使用固定的官方 Qwen3.5-27B 权重。此前工作为环境准备、连通性检查及已停止的自定义设施 smoke。官方 SWE-Gym 容器执行仍受当前平台权限限制。

用户希望了解更方便的 benchmark：优先考虑 [EvalPlus 的 MBPP+](https://github.com/evalplus/evalplus) 作为快速模型代码生成基线，其官方 CLI 支持连接本地 OpenAI-compatible 服务并提供 Python 本地评测方式，无需逐仓库构建环境；仍需核对生成代码执行隔离，不把“没有强制 Docker”当作共享宿主可直接执行的保证。更有区分度的后续模型对照可选 [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)，使用官方 checker，固定 release、时间范围、提示与采样参数；官方默认 code_generation_lite 对测试做了加速。不要因为代码文件名为 custom_evaluator 就自行重写检查逻辑，其可直接接收模型输出是官方提供的入口。[BigCodeBench](https://github.com/bigcode-project/bigcodebench/blob/main/ADVANCED_USAGE.md) 更偏库调用与复杂指令，但本地运行依赖较多，不是当前最省环境准备工作的首选。

以上主要测模型代码生成/算法能力，不能替代 Qwen + DSH 的仓库修复 agent 评测。MBPP+ 为长期公开题，不能只凭其涨分断言泛化提升；任何训练前后对照都应固定官方协议、排除评测题及近重复训练样本。当前仅提供选项，没有自动更换 benchmark、安装这些工具或开始训练。
<!-- lightweight-bench-options:end -->

<!-- official-swe:start -->
**官方 SWE-Gym 环境接入（2026-09-09，评测器就绪，容器权限阻塞）**

执行原则按用户最新要求固定：使用未经修改的官方任务环境、评测入口与判分代码；小子集只缩小题目数量。不继续自定义评测，不编写替代环境、测试筛选器或判分器。以下历史自定义结果仅作设施 smoke。

已获取官方 `SWE-Gym/SWE-Bench-Fork` commit `242429c188fcfd06aad13fce9a54d450470bf0ac`，源码在 `vendor/SWE-Bench-Fork/`，安装于项目独立 `.venv-official-swe/`（Python 3.12.12）。97 个原始源码文件与固定归档一致，39 个已安装 Python 模块逐文件一致，未修改上游代码。官方 `python -m swebench.harness.run_evaluation --help` 和 64 个安装包的依赖检查通过；依赖版本及哈希在 `runtime/official-swe/requirements.freeze.txt`、`requirements.lock`，源码来源与核验在 `metadata/source.json`。

小子集准备为原冻结列表中每个仓库的第一题：Dask 6683、Pydantic 5529、Hydra 1560、Bokeh 13636，尚未运行官方容器测试。数据直接从已固定 SWE-Gym revision `bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb` 导出，字段未改写，文件 `runtime/official-swe/smoke-tasks.json`；选择清单及哈希在 `metadata/smoke-selection.json`。`official-generated/` 内的 Dockerfile、环境脚本与 eval.sh 均由上游 `make_test_spec` 原样生成，仅供核对，没有自行重写。

依据官方 OpenHands 源码的 `__` → `_s_` 镜像命名规则，Pydantic、Hydra、Bokeh 三题的公开预构建镜像 manifest 已获取并固定 amd64 digest，见 `metadata/task-images.json`；合计约 5.34 GB 压缩层尚未下载。Dask 6683 的匿名 registry 请求返回 401，尚未确认可拉取，保留官方构建配方，不静默替换环境或题目。参考源码 commit 与原始文件在 `metadata/openhands-source.json`、`upstream-reference/`。

**实际阻塞证据**：共享 Docker 的 data root 是 `/var/lib/docker`，不符合所有新增文件留在项目内的约束，因此未用于拉取、构建或启动任务。项目独立 Docker 20.10.18 / vfs 的 data root 为 `runtime/od/data`，exec root 为 `.d/`，socket 为 `runtime/od/docker.sock`，能够启动。但最小官方 hello-world 镜像在创建挂载命名空间时失败：`Error creating mount namespace before pivot: operation not permitted`。随后实际调用下列官方 gold 评测入口，也在基础镜像构建阶段以相同错误退出；**官方测试执行数为 0，不能称为 gold 验收失败或模型失败**。日志分别在 `runtime/official-swe/logs/docker-container-probe.log`、`logs/official-gold-preflight.log`、`logs/build_images/base/sweb.base.x86_64__latest/build_image.log`。独立 Docker 及其 containerd 已退出，未启用 GPU，未修改共享 Docker 或平台安全策略。

继续执行需要：一台具备 Docker 容器权限、挂载当前 Lustre 项目目录的执行节点，以及数据根目录同样位于项目内的专用 Docker daemon；也可以提供满足相同存储约束的项目专用 Docker endpoint。当前受限开发容器无法自行授予宿主的挂载命名空间权限。不会以自定义隔离程序替代官方环境。

资源到位后的官方入口（仅直接调用上游 CLI；首先单题 gold，成功后再扩大 smoke，并检查 buggy 基线）：

```bash
cd /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
source scripts/env.sh
export DOCKER_CONFIG="$CODE_AGENT_ROOT/runtime/od/client"
# DOCKER_HOST 必须指向具备容器权限且 data-root 在本项目内的专用 daemon。
# 本机已停止的探针 daemon 不满足容器执行权限，不能用来完成评测。
docker info --format '{{.DockerRootDir}}'
cd "$CODE_AGENT_ROOT/runtime/official-swe"
"$CODE_AGENT_ROOT/.venv-official-swe/bin/python" -m swebench.harness.run_evaluation \
  --dataset_name "$CODE_AGENT_ROOT/runtime/official-swe/smoke-tasks.json" \
  --split train --instance_ids pydantic__pydantic-5529 \
  --predictions_path gold --max_workers 1 --timeout 1800 \
  --cache_level instance --run_id official-gold-smoke-01
```

`--predictions_path gold` 是上游内置功能；1800 秒是该版本官方评测器的测试超时默认值，1 个 worker 只限制并发。两者均不代表模型 agent 的轮数/token 预算。模型生成阶段尚未启动，DSH 预算不会冒称官方规定。机器状态见 `runtime/official-swe/metadata/acceptance.json`。官方依据：[SWE-Gym 项目](https://github.com/SWE-Gym/SWE-Gym#reproducing-results)、[官方评测入口](https://github.com/SWE-Gym/SWE-Bench-Fork/blob/242429c188fcfd06aad13fce9a54d450470bf0ac/swebench/harness/run_evaluation.py)。

补查现成运行后端：OpenHands/benchmarks 已提供 `--workspace apptainer` 与 `swebench-eval --apptainer`，用官方 SWE-bench 镜像及测试脚本做本地评测，无需 Docker daemon；这是 OpenHands 维护的适配，并非 SWE-Gym 当前固定评测器已验证的直接替代。本机未发现 apptainer/singularity 或 `/dev/fuse`，尚未安装，也未验证当前平台权限、SWE-Gym 任务及 DSH 接入兼容性。SWE-Gym 官方复现文档另提供 OpenHands RemoteRuntime；SWE-bench 官方新 CLI 提供 `--modal` 云端评测，但未确认其对本轮 SWE-Gym 的直接兼容。云端运行需要账户及执行/存储条件，不符合现有所有文件必须在项目内的约束时不能直接采用。本次仅核对上游文档，没有更换 benchmark 或启动外部服务。来源：[OpenHands Apptainer 文档](https://github.com/OpenHands/benchmarks/blob/main/benchmarks/swebench/README.md#apptainer-workspace-for-hpc-clusters)、[SWE-Gym RemoteRuntime](https://github.com/SWE-Gym/SWE-Gym/blob/main/docs/OpenHands.md)、[SWE-bench CLI](https://www.swebench.com/SWE-bench/reference/cli/)。
<!-- official-swe:end -->

<!-- gym-baseline:start -->
**SWE-Gym 自定义设施 smoke（已停止；不作为官方协议模型基线）**

用户于 2026-09-09 明确纠正：可以缩小测试子集，但不能自行替换 benchmark 的任务环境、评测入口和判分语义。以下 V1/V2 仅保留为设施 smoke 历史，撤回其“正式模型基线”定位，不能用于官方协议下的模型能力或训练收益结论。当前自定义评测已停止，禁止按旧入口自动续行。固定 27B 权重、独立推理环境、DSH 连通性和分页检查仍是可复用的工程资产。

后续评测先固定 SWE-Gym 提供的任务镜像 digest（或原始官方构建配方）、官方 SWE-Bench-Fork 评测 commit、测试执行与结果解析；Qwen + DSH 作为被测系统，只负责生成补丁，再交给官方评测器判分。先做小子集 buggy/gold smoke，再进行模型评测。官方未统一规定的 agent 请求次数、时间/token 限制需单独标明来源与对照配置；不能称 30 次请求为官方标准，也不能直接将 DSH 请求数等同 OpenHands 迭代数。依据：[官方环境与镜像说明](https://github.com/SWE-Gym/SWE-Gym#reproducing-results)、[官方复现步骤](https://github.com/SWE-Gym/SWE-Gym/blob/main/docs/OpenHands.md)。

最终方向已明确为 Qwen3.5-27B + 内部 LoopLM 改造 + LoRA/OPD + DSH。本轮只建立可复现评测基线，复用固定 27B、DSH 和推理环境，不进行模型结构改造或训练。以下既有 SWE-smith 0/3 保持原始记录。

已读取 2,438 个 SWE-Gym 任务与 500 个 Verified 任务，完整任务 ID 交集为 0；候选仓库与 Verified 仓库也无交集。SWE-Gym revision 为 `bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`，Verified 对照 revision 为 `78f471bf655a3137b2e8a75af1501690ec009ec3`。`configs/gym-candidate-policy.json` 事先规定：Dask、Pydantic、Hydra、Bokeh 各取 5 题，要求问题、FAIL_TO_PASS、PASS_TO_PASS 均非空，仓库内按（测试总数、instance ID）升序验证，取前 5 个环境合格者。唯一被替换的候选是 `pydantic__pydantic-9023`：gold patch 与指定 base commit 的源码上下文不匹配；按顺序改取 `pydantic__pydantic-5138`。两者均未在替换前进行模型尝试，具体证据在 `runtime/gym-baseline/candidate-rejection-evidence.json`。安装和验收过程中的其他错误、修正、复验均保留于 `validation-attempts.jsonl` 与各自证据目录。

最终 20 题全部通过最终环境配置下的 buggy/gold 复验；`configs/gym-baseline-freeze.json` 固定任务 ID、base commit 来源、环境、评测协议和执行代码/配置哈希。实际首轮目录为 `runtime/gym-baseline/runs/20260909T090911Z-4fa3d3/`。每题使用单独的缺陷工作副本和 DSH 会话，Git 历史只含该缺陷快照；gold/test patch、环境验收副本和未来 Git 历史不对模型开放。补丁通过操作员新建的可信 Git 索引导出，不信任任务内 `.git`；再应用到另一干净缺陷副本，用原始固定验收测试检查。空补丁不会计为修复，修改既有测试或新建/修改 symlink 的补丁计为失败；原始补丁仍保存。

冻结配置采用 **32768 上下文、每题 30 个模型请求/600 秒/累计生成 16384 tokens、并发 1**；单次上限 2048，temperature=0、top_p=1、seed=0、thinking=false，TP=1、单张 H100。工具每条模型可见结果最多 4096 UTF-8 字节，全部历史工具结果合计最多 16384 字节，按消息数量确定性分配；完整 DSH 返回、原始终端字节流、逐请求模型实际输入分别保存。`read_output ID BYTE_OFFSET` 可读取后续 3072 字节分页，文件也保留原生 `view_range`。目录路径在模型视图中相对化，原始记录不改。pi-ai 原本按未分页的内部历史缩小输出上限，现由转发层对**实际分页后的请求**执行冻结预算；原始 DSH 请求和实际模型请求的差异完整保存。累计预算先预留请求上限，只有收到真实 usage 才退回未用预算，流中断不释放未知消耗。预算拒绝、上下文超限和超时均保留在结果中，不换题、不挑最好一次。

`runtime/gym-baseline/facility/20260909T082552Z-cpu-681cce/` 的 CPU 配置探针通过了 9 个工具调用与返回配对、大目录/长文件/Shell 输出分页、完整终端输出、后续页读取、编辑、失败及通过测试、约 32 秒超时失败和 Shell 重置恢复。它使用显式模拟传输，不计模型成绩。平台无法在命名空间内重新挂载 `/proc`，超时可能返回 DSH 的进程组解析错误；原始诊断保留，PID 命名空间负责最终清理。前四次 CPU 配置检查和修正记录也保留。

真实 GPU 设施复验 `20260909T083250Z-real-329627/` 已通过：32K 服务处理 **26,032 个输入 tokens** 并正确输出指定内容；DSH 用 **8 次请求、753 个输出 tokens、65.48 秒** 完成分页、编辑及测试修复，测试文件未变，独立重跑通过。GPU 3 采样峰值 **65,443 MiB**，设施检查结束后服务正常退出。首次真实检查 `20260909T082933Z-real-2f8975/` 也通过长输入生成，但工具请求被旧的 1-token 上限截断，失败记录没有删除。CPU 预算探针另验证了累计上限耗尽后请求在到达模型前被拒绝；补丁导出探针用受控 gold fixture 验证了任务 `.git` 被删除时仍能导出并独立验收，均不计正式任务成绩。

本轮复用 Qwen revision `fc05daec18b0a78c049392ed2e771dde82bdf654`、DSH `0.1.2rc1` / commit `a66e4702047846cdaa10c66c9d3df3951f5ea70d`、vLLM `0.19.1`、torch `2.10.0+cu129`。任务依赖环境（第二版共 14 个，包含第一版历史环境）使用项目内 Python **3.8.20 / 3.10.18** 和 pytest **7.4.4**，按 base commit 需要固定 NumPy、pydantic-core、OmegaConf 等兼容版本。完整路径、Python 实际解析路径及依赖锁见 `runtime/gym-baseline/environment-runtime-manifest.json`、`runtime/gym-baseline/envs/*.lock`；已通过版本及依赖一致性检查。Hydra parser 从固定源码自带 ANTLR jar 生成，JDK `17.0.9.2` 仅装于项目构建环境；Graphviz Ubuntu 包仅解包到项目目录，没有安装到宿主。Bokeh 直接运行固定源码，并添加明确标为本地构建的 distribution metadata，没有安装另一个版本的 Bokeh 源码替代任务代码。这里是经 buggy/gold 验证的最小任务环境，不冒充官方 Docker 镜像环境。

SWE-Gym 官方环境参考 commit 为 `242429c188fcfd06aad13fce9a54d450470bf0ac`，保存于 `research/sources/SWE-Gym__SWE-Bench-Fork/`。来源：[SWE-Gym 官方项目](https://github.com/SWE-Gym/SWE-Gym)、[固定环境参考源码](https://github.com/SWE-Gym/SWE-Bench-Fork/tree/242429c188fcfd06aad13fce9a54d450470bf0ac)。所有新增环境、缓存、源码、容器替代隔离空间、日志和临时文件均在本项目内；没有修改共享 Docker 或宿主环境。

开发集训练排除已建立：`configs/gym-training-exclusion.json` 保留 20 个任务及身份指纹；`runtime/gym-baseline/holdout-audit.json` 定位到 **72 条关联轨迹**（OpenHands 5、SWE-smith 51、Klear 16），全部排除出后续训练。身份未确认或有歧义的记录先隔离，不能因没有匹配到 ID 就默认可训练。`scripts/gym_holdout.py` 提供该准入检查，没有输出、修改训练数据或运行训练。这 20 题是开发评测，允许研究时观察结果，不能再当最终测试集。

历史入口记录（协议纠正后不得继续运行旧模型评测；下列单题命令仅作历史记录）：

```bash
cd /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
source scripts/env.sh
# 只读检查任务环境；需要恢复时另加 --restore，可选 --offline。
.venv-dsh/bin/python scripts/gym_restore_envs.py
# 单题单独重跑，使用同一冻结协议，并自行启动/退出模型服务。
# 已停止，不再执行：.venv-dsh/bin/python scripts/gym_run.py --task dask__dask-6683
# 从第二版现有证据重新审计统计，不调用模型或重跑测试。
.venv-dsh/bin/python scripts/gym_report.py \
  runtime/gym-baseline/runs/20260909T094817Z-bf4601
```

后续四组对照协议（本 Goal 不训练）：原版 27B、普通 LoRA、Loop+LoRA、Loop+LoRA+OPD 共用这 20 个任务、固定 base commit/验收测试、同一 harness/提示/工具分页及上述外部预算。LoRA 与 Loop+LoRA 共用经过排除检查的训练划分和训练 token 预算，固定并报告 LoRA rank、target modules、优化器、训练种子及 checkpoint 选择规则；Loop 层范围、重复次数和退出规则须在运行前登记。OPD 的 teacher、数据来源及额外采样/训练成本单独报告。保留每个任务的配对结果、所有失败与重跑，不取多次最好值；同时报告成功率和 GPU 卡秒，因为内部循环会增加计算成本。若硬件卡数或运行协议需要变化，应对所有对照组重建一套共同基线，不能直接把成本与本次单卡结果混比。20 题样本小且按环境成本选择，Wilson 95% 区间仅作开发子集的描述；比较时报告配对胜/负和不确定性，不宣称小幅差异显著。最终结论另用未参与调参的测试集。

<!-- gym-results:start -->
首次运行 `runtime/gym-baseline/runs/20260909T090911Z-4fa3d3/` 已因明确的测试依赖缺口停止，自有服务已清理。前三题完成（1 题独立验收成功），第四题在停止时保留了部分轨迹并单独验收；原始记录不改写。Dask 10380 的仓库默认 pytest 参数需要 pytest-cov，而首次资格验证清除了 addopts，未暴露该缺口；静态核对同时发现 Pydantic 9082 需要解析 pytest-benchmark 参数。第二版仅补齐这些依赖，并让 buggy/gold 和独立验收保留仓库默认参数；pytest-benchmark 按仓库默认 --benchmark-disable 禁用实际 benchmark。所有原有依赖版本、20 题、模型、工具分页和预算保持不变，随后建立独立的完整重跑记录，不挑选两轮最好结果。

第二版于 `2026-09-09T09:48:16Z` 冻结，20/20 题保留默认参数后全部重新通过 buggy/gold 验证，两个受影响环境另通过真实 DSH Shell 的 CPU 模拟驱动检查（不调用模型、不计修复成绩）。运行目录：`runtime/gym-baseline/runs/20260909T094817Z-bf4601`。当前目录内共有 14 个带哈希锁定的任务环境（含第一版历史环境）；复核旧依赖版本未变。冻结快照为 `runtime/gym-baseline/freeze-v2.json` 与 `frozen-sources-v2/`；第一版独立保留在 `freeze-v1.json` 与 `frozen-sources/`。第一版已发生 92 次模型请求，报告输入 620,466、输出 13,444 tokens，另 1 次中断请求缺 usage、输出上限 2,048 tokens；GPU 占用 1,296.21 卡秒，所有成本另计，不从最终报告隐藏。

第二版第一题执行到第 19 次请求时，其他任务短暂占用 GPU 3，监测器按规则主动退出本轮自有服务，释放后显存回到空闲值。该题的空补丁已独立验收并计失败，明确分类为资源冲突导致的基础设施中断，不做模型重试。保留原始 `resource-contention.json`、`lifecycle.json` 和部分请求；`scripts/gym_continue.py` 只继续同一运行目录内尚未尝试的 19 题，导入未修改的冻结 worker/提示/预算/评测器，每段开始前保存自身源码与哈希，结束后保存清理及 GPU 时间。该续行方式不增加任何题目的模型尝试次数，重启成本累加。

**协议纠正后的停止记录**：第二版第 2–5 题完成完整尝试，第 1 题因资源冲突中断，第 6 题在用户指出方法问题后停止并保留部分轨迹；剩余 14 题未尝试。自定义验收下保留 1 个成功记录，仅用于设施诊断，不给出“resolved/20”模型基线。自有服务/任务进程清理已验证，GPU 3 释放后为 3 MiB。当前审计记录为 `runtime/gym-baseline/runs/20260909T094817Z-bf4601/audited-report.json`，停止原因见 `user-direction-stop.json` 与 `runtime/gym-baseline/protocol-hold.json`。第二版所有已发生请求（包括中断）为 127 次，已报告输入 745,418 / 输出 17,381 tokens；另有 2 次请求缺 usage，未知输出上限合计 4,096 tokens；两个服务段累计 1730.66 GPU 卡秒。第一版成本仍单列保留。
<!-- gym-results:end -->
<!-- gym-baseline:end -->

<!-- swe-three:start -->
**三个 SWE-smith 真实仓库合成任务（2026-09-09，执行与独立验收闭环完成，修复 0/3）**

三题清单在模型运行前冻结于 `configs/swe-three-tasks.json`：SQLGlot 的 `func_pm_ctrl_invert_if__2jq6jsa6`、`func_pm_op_change__atmgadcu`，以及 Schedule 的 `func_basic__1d2lxayf`。按非空问题、纯 Python 环境、测试数量和 instance ID 排序选择，不按模型结果换题。固定任务 commit 分别为 `6d541d8155d3d9ea4fe33c1e482e7db1b598c9e7`、`7891963a9b745fd3ae3cb1f065e3859bb4e02469`、`f9ac18ede1e40b05576741a264c6cb38fc97fbeb`。

| 任务 | 缺陷基线：失败 / 通过 | gold 后通过 | 请求数（成功生成） | 尝试耗时 | 输出 tokens | 空补丁独立验收 |
|---|---:|---:|---:|---:|---:|---|
| SQLGlot / `2jq6jsa6` | 1 / 1 | 2 / 2 | 3（2） | 28.00 秒 | 216 | F2P 0/1，P2P 1/1 |
| SQLGlot / `atmgadcu` | 1 / 1 | 2 / 2 | 3（2） | 24.71 秒 | 216 | F2P 0/1，P2P 1/1 |
| Schedule / `1d2lxayf` | 2 / 79 | 81 / 81 | 5（4） | 45.67 秒 | 473 | F2P 0/2，P2P 79/79 |

三题均在修复前因工具输出超过固定 **8192 tokens** 上下文而结束，归类为 `model_context_budget_failure`，没有重试或换题。前两题的目录展开返回 13,471 字符；第三题随后读取源码返回 16,240 字符。pi-ai 将超限请求的输出预算降至 1，服务仍因输入过长返回 HTTP 400。共 11 次真实请求，其中 8 次成功生成的 usage 合计输入 22,314、输出 **905 tokens**；3 次拒绝请求没有生成或 usage，不包含在输入 token 合计中。这是本次固定上下文下的尝试失败，不能据此判断模型看过相关源码后能否修复。

三份 `agent.patch` 均为 **0 字节**；在三个新的缺陷副本上应用空补丁、重新补入原始验收测试后，观察到与缺陷基线一致的失败。环境的 buggy/gold 检查和模型尝试结果分别保存，没有把环境通过当作模型修复通过。Git 2.34 的所有权检查最初阻止导出；已用只对子进程生效的项目内配置修复，并仅重做 CPU 后处理，**没有增加模型请求**。

完整实际记录在 `runtime/swe-smoke/runs/20260909T071918Z-19d82c/`：

- `summary.json`、`evidence-audit.json`：最终三题结果，以及原始请求、SSE usage、预算、补丁哈希、验收测试未变和清理的复核；`initial-summary.json` 保留 Git 导出失败时的原始状态。
- 各 `task-N/` 的 `requests.jsonl`、`response-*.sse`、`transport-errors.jsonl`、`notifications.jsonl`、`sandbox/dsh-home/`、`tool-evidence.json`：真实交互、工具返回和上下文超限错误。
- `agent.patch`、`patch-application.json`、`independent/agent-acceptance.json`、`evaluation.json`：空补丁提取、独立副本的实际测试和分类。独立验收文件哈希与官方基线一致。
- `scripts/` 保存运行开始时的源码，`postprocessing-scripts/` 保存 Git 兼容修复后的 CPU 后处理源码；配置与来源也有快照。

最终实际执行只使用 **GPU 3 一张 H100**；采样峰值 **65,259 MiB（约 63.73 GiB）**，权重加载报告 50.22 GiB。服务于 07:25:15 UTC 正常退出；结束后 GPU 3 回到 3 MiB、无计算进程。`final-process-check.json` 再次确认没有自有模型/DSH/任务进程，18080 端口已关闭。没有运行 Docker 任务容器，也没有修改共享 Docker。汇总入口：`runtime/metadata/swe-three-acceptance.json`。

数据的 `patch` 是注入缺陷的变更；已在独立副本中反向应用并验证 gold 修复。任务分支缺失的验收文件仅从固定原始基线补入验收副本，基线 commit 为 SQLGlot `036601ba9cbe4d175d6a9d38bc27587eab858968`、Schedule `82a43db1b938d8fdf60103bd41f329e06c8d3651`。每次基线/gold 测试约 2–3 秒，所有 F2P、P2P 节点均检查，不仅检查退出码。机器证据在 `runtime/swe-smoke/environment-checks.json` 和对应 `env-validation/` 目录。

复用已验证的 Qwen、vLLM、DSH 和 8192 上下文。每题一次尝试，最多 30 个上游模型请求、600 秒、每次输出 512 tokens（总输出上限 15,360），并发 1。传给模型的是原始问题和缺陷工作树；工作树重新建立为单个缺陷基线 commit，不含 remote、未来 Git 历史、gold patch、测试清单或隐藏验收文件。请求、真实 SSE、DSH 通知和工具结果由隔离环境外的记录器保存。

环境使用项目内 `.venv-swe/`（Python 3.12.12），pytest 8.3.5、pytz 2025.2、python-dateutil 2.9.0.post0 及其固定传递依赖，共 7 个包。完整版本/hash 在 `configs/swe-requirements.lock`，wheel 在 `runtime/wheelhouse/swe/`；依赖一致性和带 hash 的离线重装检查通过。这里只重建所选测试需要的源码环境，**没有拉取官方 Docker 镜像**；官方 Python profile 默认 3.10，本次实际为 3.12.12。环境一致性依据是冻结节点的 buggy/gold 验证，不能称为官方 benchmark 环境或榜单成绩。

隔离使用每个副本独立的宿主 UID、用户/PID/网络命名空间、Landlock 文件访问限制和 seccomp。DSH 原生运行时及其 Shell、编辑器子进程共享同一个限制与工作目录；CPU 配置探针已实测 Shell 写入、编辑器修改、Shell 再读到修改。模型仅能通过字节转发器连接本地 Qwen；外部网络、宿主 Unix socket、其他任务和私有验收资料的访问被限制。模型缓存、环境、源码、临时目录和日志全部在本项目。

当前内核仅提供 Landlock ABI 1，原生编辑器的跨目录原子替换会返回 EXDEV。`scripts/swe_atomic_compat.c` 与执行包装器仅为任务目录内的原子 rename/link 提供兼容代理，保持原子发布语义；代理通过目录 fd 限定目标，实测拒绝目录外目标。没有修改 DSH 源码、插件组合或推理策略。探针是明确标注的模拟模型传输，用于验证实际工具与隔离，不计作模型成绩。证据：`runtime/swe-smoke/isolation-check/verification.json`、`result.json`、`failed-cross-dir-rename.json`。

重跑单题（1、2、3 对应冻结清单顺序；每次创建新证据目录）：

```bash
cd /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
source scripts/env.sh
# 需要重建任务依赖时，使用已下载 wheel 离线安装。
bash scripts/install_swe_env.sh
# CPU：独立重做所选题的缺陷基线与 gold 验收。
.venv-dsh/bin/python scripts/swe_prepare.py --task 1
# GPU：启动服务、一次修复尝试、提取补丁、独立验收并退出服务。
.venv-dsh/bin/python scripts/swe_three_tasks.py --task 1
# 三题完整重跑时省略 --task；这会产生新的尝试记录。

# 仅对已保存的三次尝试重新提取补丁、独立验收；不启动 GPU、不调用模型。
.venv-dsh/bin/python scripts/swe_three_tasks.py --evaluate-run \
  runtime/swe-smoke/runs/20260909T071918Z-19d82c
.venv-dsh/bin/python scripts/audit_swe_run.py \
  runtime/swe-smoke/runs/20260909T071918Z-19d82c
```

服务启动仍检查 `configs/gpu-allocation.json` 的授权与当前空闲状态。模型运行前的三个中止均保留，且均为 **0 个模型修复请求**：`20260909T071203Z-4fb9ff/` 因 GPU 2 已被其他任务占用而中止；`20260909T071322Z-ad8fde/` 的就绪探针漏带 API key，收到 401，修正前已清理服务；`20260909T071648Z-5e0159/` 中 GPU 5 在预检后、初始化期间被其他任务占用，显存检查中止。随后按已有“1–2 张、优先空闲”授权，仅使用持续空闲的 GPU 3 完成实际运行。没有停止其他任务的进程。

源码/任务/依赖来源与哈希在 `runtime/swe-smoke/provenance.json`、`sources/*.json`、`dependency-manifest.json`；数据 revision 为 `ea6d7173829c7ec8fa16c22055699ff2e9188091`。`gold-source-check.json` 确认反向修复后的三个源码文件哈希均与固定官方基线相同。SWE-smith profile 参考固定官方 commit `9b74ac08118a85c39c356802f7961893af73e07f`，源文件在 `research/sources/SWE-bench__SWE-smith/`。依据：[SWE-smith 官方仓库](https://github.com/SWE-bench/SWE-smith/tree/9b74ac08118a85c39c356802f7961893af73e07f)、[Linux Landlock 文档](https://docs.kernel.org/userspace-api/landlock.html)。本轮只执行这三个真实仓库上的合成任务，不跑完整 benchmark、不训练；没有为提高本轮成绩调整工具输出策略或重新尝试。
<!-- swe-three:end -->

**真实 GPU 连通与工具闭环（2026-09-09，已通过并退出服务）**

复用固定 Qwen3.5-27B、DSH 0.1.2rc1 和独立环境，本轮没有重新安装依赖或下载模型。按用户“1–2 张、优先空闲”的授权，使用 H100 GPU 2、3；UUID、分配来源/时间和服务释放时间在 `configs/gpu-allocation.json`。真实无工具连通返回 `QWEN_DSH_OK`，记录了 6 个流式文本片段；三个真实工具用例均通过独立验收。未运行任务容器、benchmark、长上下文吞吐测试、训练数据处理或训练，未修改 DSH 架构/策略或共享环境。

| 真实模型用例 | 模型请求轮数 | 用例耗时 | 实际输出 tokens | 执行证据与结果 |
|---|---:|---:|---:|---|
| 读取文件 | 2 | 12.65 秒 | 60 | `bash` 实际读取新生成的随机值文件，工具返回和最终回答均与文件一致；文件未修改 |
| 持久 Shell | 3 | 14.84 秒 | 101 | 两次独立 `bash` 调用：先从文件设置变量，随后只打印变量，得到同一随机值；第二次未重新赋值或读取文件 |
| 编辑 / 测试 | 10 | 76.18 秒 | 859 | 模型先运行可见测试得到 4 个失败，再通过 `str_replace_editor` 将 `min(low, max(value, high))` 改为 `max(low, min(value, high))`，随后实际运行测试得到 `OK`；独立重跑测试及 124 项额外输入检查通过，测试文件未改 |

修复用例包含两次编辑器路径错误（相对路径 `.`、不存在的 `/clamp.py`）；模型随后用 `pwd` 定位工作目录并完成修改。错误和全部后续尝试保留，10 轮正好达到上限，没有删除失败工具调用或追加预算。这里是三个合成功能用例的一次成功运行，不能外推为真实仓库修复能力或 benchmark 结果。

所有用例各自最多 10 个实际模型请求、5 分钟。`configs/tool-smoke.json` 中逐轮输出上限为读取 128、Shell 128、修复 256 tokens，总上限分别 1,280 / 1,280 / 2,560 tokens。记录器原样转发真实 SSE，保存请求及响应并在第 11 次请求到达模型前拒绝；不会生成模型回复。模型/工具本身使用官方 `sdk-minimal` 的正常 bash/editor 配置。无工具探针单独禁用工具。

成功工具运行的完整证据在 `runtime/tool-smoke/20260909T062642Z-0f7088/`：

- `summary.json`：三例结果与清理确认；`independent-evidence-audit.json`：逐份原始 SSE、实际请求上限、usage 和显存采样的复核。
- 各用例下的 `requests.jsonl`、`response-*.sse`、`notifications.jsonl`、`dsh-home/`、`tool-evidence.json`、`worker-status.json`：全部真实请求、流式响应、工具返回与会话事件。
- `repair/changes.diff`、`baseline-tests.json`、`independent-tests.json`、`independent-oracle.json`、`before.json`、`verification.json`：源码差异、失败基线、实际测试及独立检查。
- `gpu-memory.jsonl`：3 秒间隔显存/进程采样；GPU 2、3 采样峰值分别为 **66,470 / 66,469 MiB**（约各 64.91 GiB）。结束后分别回到 **4 / 3 MiB**，分配卡无计算进程，临时工作目录无残留进程。
- `scripts/`、配置快照、`provenance.json`：运行时源码、模型/DSH/环境版本和配置。后续每次运行都会创建新的证据目录，不覆盖之前的尝试。

最终机器可读验收为 `runtime/metadata/gpu-tool-acceptance.json`。无工具成功记录为 `runtime/metadata/dsh-real-check.json`、`dsh-real-events.json` 及 `runtime/connectivity-smoke/` 下对应目录；`runtime/dsh-home/sessions/` 保留真实会话。所有本轮启动的 GPU、DSH 和编译进程已清理；没有停止其他任务。

启动/配置问题及保留的失败记录：

| 未通过尝试 | 发现与处理 | 证据 |
|---|---|---|
| 初次服务启动 | vLLM 0.19.1 将 `CUDA_VISIBLE_DEVICES` UUID 按整数解析。启动器现按分配 UUID 查询实时卡号，核对 PCI 顺序后设置整数可见卡号 | `runtime/gpu-smoke-attempts/20260909-01/` |
| FlashInfer 首次 JIT | 默认 32 个 CUDA 编译进程造成过高主存使用，主动停止自有服务/编译进程；改用 vLLM 已支持的 Triton GDN 后端，并设置 `MAX_JOBS=4`、`NVCC_THREADS=1` | `runtime/gpu-smoke-attempts/20260909-02/` |
| 停止期间的预检 | 旧 API 进程尚在退出，端口仍被占用，未启动第二个服务；确认原进程后完成清理 | `runtime/gpu-smoke-attempts/20260909-03/` |
| 64-token 无工具探针 | 实际请求被 pi-ai 缩为 `max_tokens=1`，只输出 `Q` 后结束。该依赖固定预留 4096 上下文 tokens；服务和 DSH 由 4096 统一改为 **8192**，显式小输出预算不变 | `runtime/gpu-smoke-attempts/20260909-04/`，含上游固定版本源文件引用 |
| 工具套件首次预检 | 已退出连接的 TIME_WAIT 导致裸 bind 误报端口占用。预检改为与服务器一致的 SO_REUSEADDR，仍拒绝实际监听者 | `runtime/tool-smoke/20260909T062553Z-606f1a/` |

以上均为部署兼容性处理，没有更换模型、依赖版本或 harness 策略。8192 上下文用于容纳 pi-ai 的固定预留空间和短工具请求，没有进行长上下文吞吐测试。FlashInfer 后端的完整执行没有通过；本轮成功使用的是 Triton GDN + FlashAttention 后端。CUDA 12.9 兼容库和编译器继续使用项目内既有版本，宿主驱动未改。

重跑命令（每次启动仍会核对分配 UUID、卡型、空闲状态和端口）：

```bash
cd /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
source scripts/env.sh
.venv-dsh/bin/python scripts/qwen_connectivity_smoke.py
.venv-dsh/bin/python scripts/qwen_tool_suite.py
```

每个命令自行启动和退出其模型/DSH 服务，成功、失败、超时都保存结果并清理自有进程。仅检查合成用例准备时可使用 `qwen_tool_suite.py --prepare-only`，该模式不调用模型，也不代表真实验收通过。启动器和请求限额检查已验证，新增的失败退出码/源码归档逻辑通过 CPU 准备入口检查；成功运行时的确切脚本快照也已保留。

变更记录：2026-09-09 06:33 UTC — 完成真实 27B/DSH 无工具连通及读取、持久 Shell、编辑/测试闭环，保留所有失败历史、实际请求和独立检查，验证服务与显存已释放。正式训练和基准评测不在本轮范围。

**历史：最小环境准备（2026-09-08，当时尚未执行 GPU 验证）**

以下是 2026-09-08 的环境准备记录；当时官方模型下载、安装和 CPU 检查完成，尚无 GPU 分配，真实连通未执行。当前 GPU 结果见文首。本轮没有拉取任务容器、跑 benchmark、修改 harness 架构、处理训练数据或训练。以下既有研究与完整路线保留为后续规划。

| 项目 | 固定版本 / 路径 | 已完成检查 |
|---|---|---|
| 官方 Qwen3.5-27B BF16 | `fc05daec18b0a78c049392ed2e771dde82bdf654`；`models/Qwen3.5-27B/<revision>/` | 24 个文件共 55,586,167,982 字节，均通过上游 SHA256/Git blob 校验；11 个权重分片、1,199 个张量通过索引、大小和类型审计 |
| DSH 源码 | `dsh-v0.1.2-rc.1`，commit `a66e4702047846cdaa10c66c9d3df3951f5ea70d`；`vendor/deepseek-harness/` | Git 传输失败后取得官方 commit tarball；8,854 个 Git blob 与官方 tree 哈希一致，源码未改动 |
| DSH SDK / runtime wheel | 均为 `0.1.2rc1`；`.venv-dsh/` | CLI 报告 `0.1.2-rc.1`；SDK 的 5 个 Python 源文件与固定 commit 一致；7 个依赖一致性检查通过 |
| 推理环境 | `.venv-runtime/`；vLLM `0.19.1`、torch `2.10.0+cu129`、Transformers `5.5.4`、FlashInfer `0.6.6` | 179 个依赖一致性检查通过；导入、Qwen 模型注册、`qwen3_coder` 工具解析器和 `qwen3` 推理解析器检查通过 |
| 下载环境 | `.venv-download/`；huggingface-hub `1.30.0`、hf-xet `1.6.0`、requests `2.32.5` | 18 个依赖检查通过；实际模型传输使用 HTTP Range 脚本；Xet 性能探测未完成，其临时文件已清理 |
| CUDA 编译组件 | `runtime/cuda-toolchain/usr/local/cuda-12.9/`；NVCC `12.9.86` | 7 个 NVIDIA 官方包校验后仅解包到项目；sm_90 PTX 编译通过，没有执行 GPU kernel；`scripts/env.sh` 指向此编译器 |
| Python / 安装工具 | CPython `3.12.12` 在 `runtime/python/`；uv `0.12.10` 在 `runtime/tools/uv-package/` | 三个环境均不继承系统 site-packages；通过项目内 wheelhouse 离线重装检查 |
| NVIDIA CUDA 兼容库 | `cuda-compat-12-9` / `575.57.08-0ubuntu1`，仅解包至 `runtime/cuda-compat/`，不进行系统安装 | 当前宿主驱动 `535.230.02`；包 SHA256、库依赖及 CPU 加载检查通过，`cuDriverGetVersion=12090`，未调用 `cuInit`；GPU 内核兼容性未实测 |

官方运行依据：[Qwen 模型仓库](https://huggingface.co/Qwen/Qwen3.5-27B/tree/fc05daec18b0a78c049392ed2e771dde82bdf654)、[vLLM Qwen3.5-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-27B)、[vLLM 0.19.1 的 CUDA 二进制说明](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/)、[NVIDIA CUDA 前向兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)。依赖之间已通过安装检查，不将其等同于已在 GPU 上验证。

所有以下路径均相对于本项目根目录 `/lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent`：

- `configs/{runtime,dsh,downloader}-requirements.lock`：固定所有 Python 依赖版本及发行包 SHA256；`configs/pylock.runtime.toml` 保留本平台 wheel 来源。`runtime/wheelhouse/` 保存已校验的依赖包，可离线重装；清单为 `runtime/metadata/wheelhouse-manifest.json`。
- `runtime/metadata/qwen-model.json`：冻结模型 revision、文件大小和上游 SHA256/Git blob；`qwen-download-manifest.json` 保存逐文件实际 SHA256 和 `complete` 状态。只有 `complete: true` 才表示全量下载校验完成。
- `scripts/download_qwen.py`：Range 续传至 `.part` 及尾段文件，大分片最多三个连接、总计最多 22 个；合并和校验后才正式命名。重跑跳过通过完整哈希复验的文件。`download-resume-check.json` 记录预存分段、连接中断和合并中断的恢复测试。
- `configs/dsh-qwen.patch.yml`：在官方 `sdk-minimal` 上通过配置启用官方 `llm-pi-ai` 的 `qwen-local` provider，替换 DeepSeek 路由；没有新增插件或改 harness 源码。正常 profile 保留持久 bash 与编辑器；连通探针附加 `dsh-smoke-no-tools.patch.yml` 禁用工具。
- `configs/qwen-server.json`：`127.0.0.1:18080/v1`，模型别名 `Qwen3.5-27B`，本地占位 key `local-qwen`；BF16、原 4096 上下文（9 月 9 日已修正为 8192）、并发 1、输出探针 64 tokens、`enable_thinking=false`、eager 模式。该配置只用于最小连通，未验证 32K 上下文、吞吐、GPU 工具调用或训练。
- `runtime/dsh-home/`：profile、插件代理和 JSONL 会话；`runtime/smoke-workspace/`：独立探针工作目录；所有缓存、临时文件和日志分别在 `cache/`、`tmp/`、`logs/`。

复现与检查命令：

```bash
cd /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
source scripts/env.sh

# 已准备的 wheelhouse 可直接离线重建环境；不改共享 Python。
bash scripts/install_runtime.sh

# 查看传输和已校验状态；下载中断后运行同一下载命令续传。
.venv-dsh/bin/python scripts/model_download_status.py
# --verify-only 全量重验，不下载。
.venv-download/bin/python scripts/download_qwen.py
.venv-download/bin/python scripts/download_qwen.py --verify-only
.venv-runtime/bin/python scripts/check_qwen_files.py

# 不需要 GPU：真实 DSH 启动/握手，与本地模拟 SSE 通信。
.venv-dsh/bin/python scripts/dsh_smoke.py --startup-only
.venv-dsh/bin/python scripts/dsh_smoke.py --mock

# 只有获得明确 GPU 分配后，将 UUID、分配来源和时间记录在
# configs/gpu-allocation.json。脚本检查 1–2 张闲置 H100，空分配直接退出。
# 同时启动 Qwen 与 DSH，最多等待模型启动 12 分钟，结束/失败均清理自有服务。
.venv-dsh/bin/python scripts/qwen_connectivity_smoke.py

# 需要前台单独启动时（本轮未执行）；结束后 Ctrl-C 并检查自有进程。
bash scripts/start_qwen.sh

# 模型运行时，在另一个终端启动 DSH stdio JSON-RPC 服务供 SDK 使用。
QWEN_LOCAL_API_KEY=local-qwen .venv-dsh/bin/dsh --profile sdk-minimal \
  --patch "$CODE_AGENT_ROOT/configs/dsh-qwen.patch.yml"
```

模型细节：官方 BF16 仓库原样含 1,103 个 BF16 张量，以及 96 个 FP32 张量（48 个 `linear_attn.A_log`、48 个 `linear_attn.norm.weight`，合计仅 8,448 个参数）。没有重写、量化或转换任何权重。总张量数据字节数 `55,562,872,800` 与官方 index 一致；没有遗留模型 `.part` 或尾段文件。最终证据为 `runtime/metadata/{qwen-download-manifest,qwen-files-check,qwen-non-bf16-tensors,acceptance}.json` 及 `logs/qwen-files-check.log`。

已验证：DSH 版本与 profile 配置展开；Qwen provider 初始化；模拟 SSE 请求返回 `DSH_MOCK_OK` 并正常结束；三个环境依赖一致性与离线重装；Qwen config/tokenizer/template（248,077 个 tokenizer 条目，示例完整模板 15 tokens）；vLLM 模型/解析器注册与配置中的 CLI 参数存在。证据为 `logs/{dsh-startup-check,dsh-mock-check,runtime-import-check,qwen-metadata-check,vllm-serve-help,reinstall-check}.log` 和 `runtime/metadata/*check*.json`。这些检查没有初始化 CUDA。

截至 2026-09-08 的历史未验证项：真实 Qwen API 响应、DSH 与真实模型端到端连通、GPU 内核/显存、生成质量、真实工具调用、32K 上下文和训练。当时 `configs/gpu-allocation.json` 为空；9 月 9 日的分配与释放记录见文首。可见 GPU 不能作为分配依据。启动保护已实测会拒绝空分配，没有启动本项目的 GPU 服务，也没有停止其他任务。

出处细节：官方 DSH runtime wheel 的内部 JSON 写着 `0.0.0-dev`，但发行包版本为 `0.1.2rc1`、CLI 报告 `0.1.2-rc.1`；两者分别保存在 `runtime/metadata/runtime-versions.json`，没有将内部字段擅自改成 release 版本。源码 archive 与 wheel 哈希、SDK 源码对照、CPU/模拟检查均各自记录；这不构成对整个上游构建链的可复现构建证明。

变更记录：2026-09-08 UTC — 按本轮授权建立三个项目内独立环境、固定官方模型/DSH/依赖、准备本地 Qwen 启动配置，完成安装和 CPU/模拟检查；全部权重下载和审计已完成，真实 GPU 连通因未分配 GPU 而保留未验证。

**DSH + OPD + LoopLM 调研（2026-09-09，研究与源码检查，尚未实验）**

结论：三者可以构成一个研究系统，但作用在不同层级。DSH 负责外部工具与环境交互；OPD 用当前 student 自己走到的状态获得 teacher 监督；LoopLM 在模型内部重复共享层、增加隐状态计算深度。本轮公开检索尚未找到完整、可直接复现的“DSH + OPD + 内部 LoopLM”项目。已有工作分别覆盖关键组成部分，不能据此声称组合的创新性或有效性已经得到证明。

建议将问题收敛为：**在相同任务和计算预算下，增加哪些 agent 决策的内部循环深度，能减少无效工具调用；这种行为能否通过 OPD 转移到较少循环的 student？** 先用现成小型循环模型验证机制，保留 Qwen3.5-27B + DSH 作为已有工程基线。此前“暂不加入 LoopLM”是旧路线约束；用户本轮已明确要求重新研究它，下面的方案优先于旧路线中的研究排除项，但不是执行训练的授权。

| 最相关的工作 | 已有证据与可借鉴部分 | 对本项目的边界 |
|---|---|---|
| [Ouro / Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)；[官方 2.6B Thinking 模型](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking) | 已公开小型共享层循环模型、权重与 HF forward 实现，适合研究不同内部深度 | 是 LoopLM 底座候选；本轮没有验证它的 DSH 工具能力、训练兼容性或仓库修复表现 |
| [Relaxed Recursive Transformers，ICLR 2025](https://arxiv.org/html/2410.20672v3) | 从预训练模型转换共享循环层，用深度专属 LoRA 放松共享约束，再 uptraining 和蒸馏；是“已有模型 + 循环 + LoRA + KD”的直接先例 | 实验包括 15B/60B tokens uptraining，使用 8×H100；并非少量 agent 轨迹上的 LoRA 微调。未证明适用于 Qwen3.5 的混合注意力架构 |
| [LoopCoder-v2](https://arxiv.org/html/2606.18023v1)；[官方项目](https://github.com/CSJianYang/LoopCoder) | 7B PLT 代码模型，作者报告两循环模型 SWE-bench Verified 64.4，对照一循环 43.0；三、四循环反而退化 | 各循环数变体经过对应训练，不能解读为同一 checkpoint 改个开关就涨分。18T tokens 预训练，全系列约 100 万 GPU 小时；我们只能考虑复用权重，不能复刻训练规模。分数未在本项目复现 |
| [LoopRPT](https://arxiv.org/html/2603.19714v1) | Ouro 上使用 EMA teacher 和带噪声的隐状态 rollout，把学习信号分配到内部循环；Omni-Math 4,428 题中留出 200 题验证 | 最接近“小模型 + teacher + 循环后训练”的规模参照。论文每次运行使用 8×A100 80GB，1.4B/2.6B 约 2/4 小时；不是两卡实测，也不是 DSH 多轮 agent OPD。本轮未定位到作者官方训练仓库 |
| [SOD v3](https://arxiv.org/abs/2605.07725v3)；[官方代码](https://github.com/YoungZ365/SOD) | 对工具错误引发的分布偏移按步骤调整 OPD 权重；有基于 verl 的实现、teacher 和冷启动数据入口 | 数学、科学及 LiveCodeBench 的工具推理，不是仓库修复。示例资源为 8×H20 96GB；不能原封不动套到两卡。其成绩有 average@32 等口径，不能与单次 SWE 成功率比较 |
| [TurnOPD](https://arxiv.org/html/2607.05804v1) | 同时调节外部 rollout 轮数和逐轮 loss 分配，解释为何长轨迹尾部可能花费很多但监督很弱 | 实验是 ALFWorld、WebShop、多跳搜索；这里的 depth 是外部交互深度，不是 LoopLM 的内部循环深度。可作为 agent OPD 的强方法对照 |
| [Reward-Gated OPD](https://arxiv.org/html/2607.04037v1) | 根据 verifier 结果与 teacher/student 概率差是否一致，筛选蒸馏信号，避免盲目跟随错误 teacher | 可借鉴到有训练测试的代码任务；不能把终局奖励当作每个工具动作都正确的证据。论文 student 两卡训练、teacher TP=2 的安排并非总共两卡 |
| [ShortOPD v2](https://arxiv.org/abs/2607.13124v2) | 用压缩前模型监督压缩后模型自己生成的序列，以自适应长度减少退化重复段的训练成本 | 为“架构压缩后做 OPD 恢复”提供相邻证据；对象是结构化剪枝，不能当作 Qwen3.5 转 LoopLM 已验证。当前只核对 v2 摘要，不引用未审计的实现细节 |
| [MELT](https://arxiv.org/html/2605.07721v1) | 将逐循环独立 KV 改为共享缓存，并通过过渡训练与注意力对齐蒸馏恢复模型 | 解释 LoopLM 的显存风险及可能方向；主训练 1,040 H100 GPU 小时，研究总量约 20,000 GPU 小时，不适合首轮直接复刻 |

这些工作支持“值得做机制实验”，不支持“27B 上加一个 LoopLM 参数就能涨 bench”。RRT 中的 layer-wise LoRA 是架构转换的一部分，不等同于保持原 27B 结构、冻结底座后的普通 PEFT。Ouro 的第三方 `rkstgr/LoopLM` 是重实现；首轮以官方模型源码为依据。社区 DSH 的记忆/skill “distill”插件属于文本提炼，也不能代替参数训练中的 OPD。

**本机源码核对的关键结果**

1. **现有 Qwen3.5-27B 不是这里讨论的深度循环底座。** 固定 revision 的配置是 64 层，每组 3 个 linear-attention 层加 1 个 full-attention 层。其沿序列的线性注意力状态不等于重复共享层的深度循环。若研究 RRT 式转换，需要处理层共享、深度专属 adapter、GatedDeltaNet 状态及 KV 索引，先在小模型验证；不能仅包装 DSH 循环或改 `num_hidden_layers`。
2. **已安装 vLLM 0.19.1 包含 `OuroForCausalLM`，并声明 `SupportsLoRA`。** `ouro.py` 确实重复共享层，按 loop×layer 创建独立 attention/cache 槽；当前 forward 执行固定 `total_ut_steps`，不做逐 token 动态退出。它只是代码层面的支持证据：未加载 Ouro 权重，未验证本机内核、PEFT、训练器或 DSH 连通。源码快照和哈希见 `implementation-inspection.json`。
3. **HF Ouro 的 early-exit 选项不自动节省内部计算。** 固定官方源码先运行所有 `total_ut_steps`，再从保存的 hidden states 选 `exit_at_step` 或阈值对应输出。以“选了第一步”报告一循环成本会失真。该源码带 labels 时默认使用退出概率加权 logits，而已安装 vLLM 使用末循环输出；训练与 rollout 必须明确统一输出语义。真正缩减计算需核对 `total_ut_steps`、梯度路径和缓存的一致性。依据：[固定官方源码](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/f1edd81e7ac41355db670500ceaf204e0f73af68/modeling_ouro.py)。
4. **小参数模型也可能有大 KV。** 对所查 Ouro-2.6B 配置，BF16、batch=1、48 层、16 KV heads、head_dim=128、4 循环且各循环独立缓存时，8,192 tokens 的纯 KV 理论值为 12 GiB，32,768 为 48 GiB。公式为 `2(K,V) × 2 bytes × 48 × loops × 16 × 128 × context × batch`，不含约 5.34 GB 权重、激活、workspace 和分配器开销；不是显存实测。循环不能解决先前三题的输入上下文超限。
5. **LoopCoder-v2 有公开推理路径，训练路径仍需补齐。** 模型仓库指定专用 `yxing-bj/vllm` 和 Transformers 4.57.1；本轮固定该 fork commit，确认 registry 将 `IQuestPLTCoderForCausalLM` 映射到 `iquest_loopcoder.py`。官方主仓当前仅 README 和两张图；HF 仓库包含配置与 tokenizer 自定义代码，未含对应 HF 模型 forward。所查 fork 模型类也未声明 `SupportsLoRA`。因此它是代码能力候选，但离接入常规 LoRA/OPD 训练器比 Ouro 多一步；不能把模型可下载写成可训练。依据：[模型说明](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2)、[固定推理实现](https://github.com/yxing-bj/vllm/blob/2608cc7dd5462bcb27c3f7acf3b187a29902d965/vllm/model_executor/models/iquest_loopcoder.py)。

**建议复用什么、自己实现什么**

复用 DSH 的会话、工具和环境交互，复用模型的循环实现，优先评估 ms-swift 的多轮 GKD，verl/SOD 作为训练接口与 loss 的参考。我们自己实现 DSH 到训练器的状态/轨迹桥接、teacher 评分接入、循环预算记录及实验对照。当前 ms-swift 的固定示例是 Qwen3.5-2B→9B 的四卡数学工具任务；它证明存在多轮接口，不证明 Ouro + DSH 已兼容。[固定多轮 GKD 示例](https://github.com/modelscope/ms-swift/blob/3f80ae012b80a438446a378279c1acc16fa0a3d4/examples/train/rlhf/gkd/multi_turn.sh)。不从头写完整 harness 或分布式训练器。

可检验的系统流程为：`当前 student 在 DSH 中执行 → 保存实际前缀、动作与工具返回 → frozen teacher 对这些 student 前缀评分 → 在 assistant token 上计算 OPD loss → 更新 LoRA → 下一批重新采样`。teacher 看到的是 student 真正走到的状态；工具返回、系统提示不作为预测目标。teacher 只生成一条自己的成功轨迹再训练属于离线蒸馏/SFT；已下载 Klear 和 SWE-smith 轨迹可用来做冷启动与格式审计，但不是更新后 student 的 on-policy 数据。

内部循环的首个假设可设为：**同一 Ouro 底座，固定四循环 teacher，训练一或两循环 student 的共享 LoRA**。这样 tokenizer 一致，且 teacher 权重可以冻结。必须先验证四循环在选定任务上更好；不能以“更深”代替 teacher 质量证据，也不能让完全相同的 teacher/student 在同一前缀产生零差异监督。Qwen27B 与 Ouro 的词表不同，不可直接对 token ID 做 KL；27B 若提供文本示范，先按 SFT 使用。跨 tokenizer 蒸馏需要另设计对齐，不进入最小实验。

先固定每个请求的循环数。若后续在同一会话切换循环深度，需要重建更深循环的历史缓存或证明其他缓存策略正确，不能继续使用缺失深层历史的 cache。按失败反馈动态分配深度可作为后续 harness 研究点；LoRA、深度预算和逐轮 loss 不应第一轮同时修改，否则无法归因。LoopRPT、RRT、TurnOPD 已覆盖相关概念，论文贡献应来自可重复的 agent 机制发现，不能只以三个组件拼接命名。

**两张 H100 下的选择与下一步范围**

| 路线 | 建议 | 当前主要缺口 |
|---|---|---|
| 原版 Qwen3.5-27B + DSH，后接 OPD | 保留为已有工程主线和后续强基线 | 先解决实际上下文预算；27B teacher 质量、长序列训练显存和评分方式仍未验证。这条路线自身不包含内部 LoopLM |
| Ouro-2.6B + DSH + 同底座跨循环 OPD | **建议先做机制可行性实验**，复用已预训练的循环能力 | 工具格式冷启动、HF/vLLM logits 一致性、较少循环的可用性、训练和 rollout 权重同步 |
| LoopCoder-v2 7B + DSH + OPD | 若目标重代码能力，列为下一候选 | 专用推理栈隔离、可反传模型实现和 LoRA 支持，需要更多移植工作 |
| 将 Qwen3.5-27B 转换为循环网络 | 保留为更长期的架构研究支线 | 模型转换与恢复训练量不确定，不能承诺两卡短期完成或保留原模型能力 |

Ouro 首轮可考虑 teacher 与 student 分卡，student rollout 与更新分阶段；这只是资源方案，实际批量和长度须由显存/吞吐短测决定。32K 多轮 OPD 不因权重很小就自动可行。长轨迹数据也不能任意截尾，否则训练的策略与执行状态会改变。保留已下载数据，优先按任务划分小型训练/开发集，并先审计工具语义；不需要现在再拉一批大数据。

下一小 Goal 建议只做 **LoopLM 可训练性和 DSH 接入验证**：固定一个候选，完成实际减循环 forward、缓存与无缓存 logits 对照、有限梯度及 adapter 保存/加载检查、一个短工具闭环、显存和耗时记录；明确此阶段不要求涨分。若这些检查通过，再安排一组普通 LoRA SFT 对照与一组 vanilla OPD，小规模检查是否值得继续。TurnOPD/SOD 式加权和动态循环放到后续消融。

可行性与小型闭环按 2–4 个有效工程工作日规划；若要补 LoopCoder 训练实现或重新处理循环缓存，可能延长至一到两周。这是工程估计，不是运行承诺。论文级证据还需要固定开发/评测划分、足够任务数及配对重复。至少比较同底座低循环基线、低循环+SFT、低循环+OPD、固定高循环 teacher；除成功率外，记录真实循环执行次数、输入/输出 tokens、工具次数、耗时和显存，以相同总成本比较。现有三题 0/3 主要反映 8K 上下文设施限制，不能当作这套研究的能力基线。

来源快照统一位于 `research/sources/opd-dsh-looplm-20260909/`：`papers-manifest.json` 保存论文/元信息 URL、获取时间和 SHA-256；每个 HF/GitHub 子目录的 `source.json` 保存固定 revision 与文件哈希；`implementation-inspection.json` 保存本机源码核对与显存公式。Ouro revision 为 `f1edd81e7ac41355db670500ceaf204e0f73af68`，LoopCoder-v2 为 `b87cf3aa2186937b0d0362a684d7d30f234543e3`。本轮仅下载论文、元信息与少量源码作静态检查，未下载这些候选权重、未运行远程代码、未占用 GPU。

变更记录：2026-09-09 UTC — 按用户要求展开 DSH + OPD + LoopLM 调研，补充相关论文、固定公开实现、核对本机 Ouro 支持与循环/cache 语义，提出两卡上的小型机制验证路线；未修改模型、harness 配置或三题原始结果，未启动训练。

**此前主线结论（2026-09-08；LoopLM/OPD 的最新研究范围以上文为准）**

用户已选择主线：基于 [DeepSeek Harness (DSH)](https://github.com/deepseek-ai/deepseek-harness)，使用 Qwen3.5-27B 推理研究 harness 架构，架构稳定后先做 BF16 LoRA SFT。9B 保留为快速调试和跨规模对照；mini-SWE-agent/plus 保留为框架对照。先固定原版 27B 比较 harness 配置，再固定 harness 比较原模型与 LoRA 模型。OPD 是后续独立评估项，全参数微调不作为当前主线。

已有数据资产仍有效，但迁移到 DSH 时必须重新核对持久 shell、编辑工具、角色及工具调用格式。SFT 优先评估 ms-swift。若后续开展 OPD，再评估其多轮 GKD 接口及 teacher 资源；verl、SkyRL 和 Agent Lightning 是集成备选及实现参考。

**DeepSeek Harness 架构候选（2026-09-08 追加）**

已下载并阅读固定 commit `c389f96bf3a9b6807cb71ed6bdad5849be0df6d8` 的架构、模型接入、Python SDK、日志导出和 compaction 文档，来源记录在 `research/sources/deepseek-ai__deepseek-harness/source.json`。这是此前调研的文档/源文件快照；本轮安装的是文首另行固定的 release commit。9 月 9 日已验证最小工具闭环，见文首；完整任务环境仍未验证。

- Cordis 负责插件装卸与依赖；模型 adapter、工具、会话、agent loop、上下文处理和调度可独立组合。适合将 harness 策略做成可替换实验模块。[架构文档](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md)
- 自定义 provider 支持自建 Chat Completions endpoint，因而可规划接本地 vLLM/SGLang 的 Qwen 服务；实际还需验证 tool calling、thinking 字段和模板匹配。[模型接入文档](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/providers.md)
- 官方提供 headless、SDK 和 sdk-minimal profile，以及 Python SDK。可以从 Python 实验程序控制运行，不必把 Python trainer 改写为 TypeScript。[SDK 文档](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/python-sdk.md)
- 官网的 Minimal mode 保留持久 bash 与 str_replace_editor；适合作为首个低复杂度配置。注意运行 preset 的 Minimal 与启动 profile 的 sdk-minimal 属于不同配置层，不能仅凭名称视作相同配置。[官网](https://www.deepseek.com/harness/)
- 追加式事件日志有利于追踪上下文注入与工具执行，但日志不等于 OPD 训练数据：仍需在模型请求边界记录实际输入、生成 token IDs、采样 logprobs、loss mask 和权重版本。

建议实施顺序：DSH 最小工具配置 + 原版 Qwen 的连通/执行测试；与 mini/plus 在相同任务和预算下对照；依次实验上下文压缩、文件检索/输出裁剪、测试失败恢复策略；稳定后从最终 harness 采集训练数据并接 SFT/OPD。它仍处于 developer preview，实验必须固定 commit、依赖和最终插件配置。所有 DSH home、配置、插件缓存与日志都必须留在本项目下。

上述是基于公开资料和代码阅读的工程建议；尚未安装 GPU 训练栈、运行模型、复放任务、训练或评测，也没有承诺两卡能够直接运行上游默认配置。两张 H100 是用户给出的资源边界；teacher 暂按同样两卡内的开源模型规划，未假定额外 API 预算。

**已经下载的数据**

| 数据 | 本地范围 | 用途及限制 |
|---|---|---|
| [Klear mini-swe-agent-plus 66k](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) | 全部 47 个 Parquet 分片，65,994 条 | 首选 SFT 候选。具有 instance_id，可连接 SWE-smith 任务；没有逐行 resolved 字段，需复放抽检。 |
| [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | ticks split，25,826 条，8 个分片 | 结构/质量对照。未下载 tool/xml 版本或未被当前 config 引用的 train 文件，避免重复格式与历史文件混用。 |
| [SWE-smith tasks](https://huggingface.co/datasets/SWE-bench/SWE-smith) | 全部 59,136 个任务记录 | 第一批 rollout 任务池；记录本身不包含已下载的容器镜像。 |
| [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | 2,438 个任务 | 真实 issue 的补充任务池，镜像需根据配套环境映射获取。 |
| [SWE-Gym OpenHands SFT](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | 491 条 | 小规模对照，只有 messages 字段，OpenHands 工具协议。 |
| [R2E-Gym SFT](https://huggingface.co/datasets/R2E-Gym/R2EGym-SFT-Trajectories) | 3,231 条 | 对照，只有 messages 字段，需恢复任务关联和工具语义。 |
| [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2) | 32,079 个任务，其中 Python 7,243 条 | 后续真实仓库扩展；任务级镜像有 32,079 个，不能按元数据大小估计执行存储。 |
| [SERA Best Subset](https://huggingface.co/datasets/allenai/SERA-4.6-Lite-Best-Subset) | 仅前 100 条，约 17.8 MB | 非随机审计样本，不代表全量分布；原始 JSONL 约 8.52 GB。 |
| [SWE-bench Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | 500 个评测记录 | 仅用于重叠检查及未来最终评测，禁止混入训练。 |

初始数据审计仅下载 Qwen3.5-9B 的 tokenizer、配置和模板，没有下载其权重；本轮 Qwen3.5-27B 准备状态见文首。数据文件使用固定 Hugging Face commit URL，保存字节数和 SHA256；对于 LFS 文件同时对照上游 SHA256。下载成功只表示传输完整，不表示标签或任务内容正确。

本轮完整下载清单共 80 个文件、3,420,686,547 字节，加上 SERA 样本共 3,438,470,099 字节（约 3.44GB，不含 tokenizer、代码快照和缓存）。全部文件通过下载校验，Parquet 行数已核对，未留下 `.part` 文件。

机器可读记录：

- [主下载清单](data/metadata/download_manifest.json)、[Klear 下载清单](data/metadata/klear_download_manifest.json)、[补充下载清单](data/metadata/supplement_download_manifest.json)、[SERA 样本清单](data/metadata/sera_sample_manifest.json)。
- [结构审计](data/audit/summary.json)、[内容 token 抽样](data/audit/token_lengths.json)。token 抽样使用固定种子，每个全量轨迹数据集抽 128 行；SERA 使用已下载的前 100 行。
- [Klear 全量审计](data/audit/klear_summary.json)、[任务关联审计](data/audit/task_links.json)。
- [上游代码索引](research/sources/code_manifest.json) 及 `research/sources/<owner>__<repo>/source.json` 保存所读文件的固定 commit 和来源。它们是只读研究快照，未作为安装环境或运行代码。

**实际数据检查改变了选型**

Klear 的 65,994 行实际对应 10,894 个不同任务；messages 有 4 行完全重复，全部可解析。文件没有 teacher 名称或逐行成功标签。不能将 66k 解释成 66k 个独立任务，也不能给本地未复放的数据贴“已验证成功”的标签。

任务连接实查：Klear 的 10,894 个任务全部能连接到此次下载的 SWE-smith，且题目非空，其中 5,273 个任务的测试总数不超过 200，适合作为执行成本探测池。原版 ticks 的 13,500 个任务中有 1,111 个无法连接到当前任务版本。两份轨迹数据共享 7,820 个任务，必须跨数据源按 task ID 一起划分，避免一份作训练、另一份作验证造成泄漏。

Qwen3.5 tokenizer 抽样结果如下。它逐条计算 message content 的 token 数后相加，未包含 chat template 和独立 tool_calls 字段，因此是正文长度参考，不是最终训练序列长度。

| 数据 | 样本数 | 正文 token 中位数 | P90 | 正文超过 16k | 正文超过 32k |
|---|---:|---:|---:|---:|---:|
| Klear | 128 | 17,718 | 31,979 | 73 | 10 |
| SWE-smith ticks | 128 | 23,331 | 48,570 | 87 | 39 |
| SWE-Gym SFT | 128 | 15,950 | 36,797 | 61 | 17 |
| R2E-Gym SFT | 128 | 13,645 | 24,201 | 48 | 2 |
| SERA（前 100 条，非随机） | 100 | 23,059 | 30,954 | 90 | 5 |

这使 32K 成为首轮值得验证的长度，而不是直接套用通用 4K/8K SFT 配置。不要为了适配短上下文而无记录地截掉轨迹结尾；数据子集选择与最终模板确定后再定训练预算。

SWE-smith ticks 中只有 11,430 行标记 resolved=true，14,396 行为 false；25,826 行对应 13,500 个 instance_id，存在 2,255 行完全重复的 messages。全部 messages 能被 JSON 解码，但不能由此认定轨迹正确。

更明显的问题是 patch 的关联：18,544 行有可解析补丁路径，其中 15,412 行的补丁路径在对话中完全没有出现。这是启发式异常指标，不是证明错误的比例。具体样本中 boltons 的 tableutils 修复却附带 dask/parquet 补丁，sqlparse 的任务附带 feedparser 补丁。因此暂不使用该版本的 patch 作为监督或自动重放依据；resolved 标签也需单独复核。原始文件保留供追查，不改写“修正”。

SWE-smith 任务实际有 18,033 条空 problem_statement，39,208 条的 FAIL_TO_PASS 与 PASS_TO_PASS 合计超过 200 个测试。记录中有 222 个不同 repo/镜像标识，这些包含仓库版本，不能等同于论文所述独立仓库数。故“59k 条都能立即用于训练”不成立。空题目、缺失分支、镜像无法启动和测试时间过长是不同问题，要分别记录。

SWE-Gym 与 R2E 的 SFT 文件只含 messages，缺少显式任务 ID；R2E 有 7 行重复 messages。不能把同一任务的不同轨迹随机分到训练和开发两侧。Klear 显式 instance_id 的价值不仅是方便读取，更在于可追踪环境和按任务分组。

对已下载的 SWE-Gym、SWE-smith、SWE-rebench V2 任务进行了 instance_id 和去除 index 行后的精确 patch 哈希检查，未发现与这版 Verified 的直接重叠。这不等于语义去污染，也不能证明底座预训练没有见过评测内容。后续还要查近重复补丁、同一 PR 的 backport、任务描述近重复和仓库版本泄漏。

SERA 数据卡明确说明主实验不做验证，不能将“高质量”直接理解为测试通过。其 Best Subset 卡头标记 Apache-2.0，而正文写 ODC-By，来源元数据存在不一致，已保存原文，暂作为审计/方法对照。SWE-rebench 的数据许可和每个源仓库的许可分别记录，不把集合许可当作覆盖所有代码。

**框架和项目比较**

| 项目 | 已核对内容 | 对本项目的判断 |
|---|---|---|
| [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) | agent loop、Docker 环境、动作解析及评测配置 | 执行骨架首选；固定版本和完整配置，不能只记录模型名。 |
| [mini-swe-agent-plus / Klear](https://github.com/Kwai-Klear/mini-swe-agent-plus) | 在 mini 上加入唯一匹配的字符串替换脚本，提供同框架轨迹；README 指出使用 ms-swift 训练 | 最贴近首轮数据。优先验证固定 fork，后续可将必要工具移植到上游版本。 |
| [ms-swift](https://github.com/modelscope/ms-swift) | Qwen3.5 SFT 文档；GKD trainer；多轮 GKD 示例和 scheduler 的 token/mask 接口 | 首选轻量训练接入。多轮官方例子是数学工具场景，不是已完成的 SWE 两卡方案；仍需环境适配和 loss mask 验证。 |
| [SWE-smith](https://swesmith.com/getting_started/assets/) | 任务、轨迹、环境资产 | 首轮任务池和数据构造思路；以当前实际文件审计为准，不能只用旧论文数字。 |
| [SWE-Gym / OpenHands LM](https://github.com/SWE-Gym/SWE-Gym) | 真实任务、成功轨迹及 OpenHands 训练路线 | 对照和扩展。OpenHands 的持久 shell、编辑工具等行为不能靠改 role 名无损转换到 mini。 |
| [R2E-Gym / DeepSWE](https://github.com/R2E-Gym/R2E-Gym) | SFT 轨迹、环境、验证器、DeepSWE 复现指南 | 学习执行验证和 RL 设计。指南含推理/评测步骤，不将其误称为两卡训练配方，也不混淆单次成功率和多候选选择结果。 |
| [rLLM](https://github.com/rllm-org/rllm) | mini harness adapter、distillation 示例、SFT 设计文档 | 通用 agent 训练备选。部分例子走托管 backend；设计文档、已实现功能和已跑通配置要分开判断。 |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL/tree/main/examples/train/mini_swe_agent) | 现成 mini-SWE-agent + SWE-Gym + Podman 集成、独立 OPD 示例 | 最有价值的完整集成参考之一；所读 Qwen3-8B 示例要求 8×H100，30B 示例要求 16×H100，需缩减及验证。 |
| [Agent Lightning](https://microsoft.github.io/agent-lightning/latest/75-example-coding-agent/) | Qwen3.5-9B + verl 的 SWE-smith 训练脚本和任务筛选流程 | 同底座的强参考；官方示例 4×B200、K8s，默认长轨迹/多 rollout，不适合照搬。优先借鉴任务探测和轨迹记录方法。 |
| [verl OPD](https://github.com/verl-project/verl/tree/main/examples/on_policy_distillation_trainer) | teacher resource pool、logprob 接口、Qwen3.5 示例 | 若 ms-swift 接入不合适，作为训练后端备选；官方 Qwen3.5 示例不构成仓库级 agent 兼容性证明。 |
| [SERA](https://github.com/allenai/SERA) | 两阶段数据生成、软验证、SWE/mini 框架和 SFT 处理 | 数据生成与 SFT 对照，不把 patch 相似度当成代码测试正确率。 |
| [Spider](https://github.com/collinear-ai/spider) | SWE OPD 配置和执行脚本 | 值得参考 teacher/工具接法，但公开 SWE 示例学生后端是 Tinker，不能当作本地两卡训练开箱即用。 |
| [SWE-World](https://github.com/RUCAIBox/SWE-World) | 用 learned transition/reward models 替代执行反馈 | 暂不选；两卡下还要承担额外世界模型和模拟误差验证。最终仍应真实执行评测。 |
| [Meta SWE-RL](https://github.com/facebookresearch/swe-rl) | 软件演化数据、相似度奖励和 Agentless 路线 | 方法对照；与多轮工具 agent OPD 的目标/训练接口不同。 |

补充数据候选 [Nebius OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) 有 67,074 条，来自 Qwen3-Coder-480B，包含 tools、trajectory_id、resolved 等字段。已保存元数据和卡片；约 2.08GB 的完整文件尚未下载。若 Klear 复放质量不好，或选择 OpenHands 框架，这是优先候选。

**我们自己写的部分**

1. 数据入口：固定版本、任务关联、去重、仓库分组、质量标记、训练模板和 assistant-only loss mask。工具输出参与上下文，不作为模型输出监督。
2. 环境接入：连接 task ID、任务镜像、工作目录、编辑工具、停止规则和测试结果；主候选通过 DSH 执行动作，mini/plus 作为外部对照。训练和评测共享动作解析/工具语义。
3. 训练桥接：把逐轮状态、真实生成的 token IDs、logprobs、loss masks 和模型版本接到 ms-swift scheduler 或备选训练后端。上下文压缩后尤其不能凭拼接后的字符串假装是原 rollout token 序列。
4. 结果记录：任务级成功/失败、环境错误、超时、格式错误、token 消耗和墙钟时间。用官方 SWE-bench harness 评估补丁。

不新增多 agent 编排、UI 或完整训练引擎。OPD/RL 接口是后续待实现工作，初始调研阶段只做下载、CPU 审计与源码研究；当前最小环境 Goal 见文首。

**实施计划：先验证 harness，再做模型适配（2026-09-08，尚未实施）**

研究问题：在两张 H100 的条件下，为 Qwen3.5-27B 设计怎样的上下文与反馈机制，能够提高仓库修复成功率或降低完成同类任务的成本？主研发对象是基于 DSH 的 harness；后续先用 LoRA SFT 适配这个 harness，OPD 单独评估。9B 仅用于快速调试和跨规模对照。首轮只处理 Python 仓库修复，暂不加入内部 LoopLM、多 agent 编排、网页搜索或新 UI。

目标架构：Python 任务调度/评测程序 → DSH SDK 和固定插件 profile → Qwen 服务与隔离的任务环境；DSH 的上下文插件和失败恢复插件是实验变量。逐轮日志进入项目内的数据记录层，后续由 Python trainer 使用。使用官方 SWE-bench 验收组件，独立于 agent 的工作目录和可见上下文。

| 阶段 | 工作与产物 | 验收标准 |
|---|---|---|
| 0. 固定配置与数据边界 | 固定 DSH commit、模型 revision、依赖、工具 schema、thinking 模式；确定容器存储和 GPU 分配；建立不可重叠的任务清单 | 所有新增状态位于本项目；每个运行有配置快照、任务 ID、模型/框架版本；不能依赖宿主默认目录或额外 GPU |
| 1. 执行闭环 | 接入本地 Qwen3.5-27B，先跑文件读取/编辑/测试小例，再选约 20 个任务验证实际环境；保存逐轮输入输出和退出原因 | 流式响应、工具调用、持久 shell、编辑器、终止及超时正常；干净 checkout 上的 gold patch 可通过验收、未修复版本有预期失败；这 20 个任务只用于设施调试 |
| 2. Harness 基线 | 固定原始 Qwen3.5-27B，比较 DSH 最小工具配置与 mini/plus；用约 50 个开发任务定位主要失败类型 | 相同任务、上下文上限、总生成预算及运行次数，完整记录差异；先查环境/协议错误，再把失败归因于模型；小样本结果只用于筛选 |
| 3. Harness 改进 | 按顺序实验“上下文管理”和“失败恢复”，并做单独及组合消融；对入围方案扩大到约 200 个开发任务 | 报告逐任务成功差异、输入/生成 token、耗时、工具错误及循环次数；成功率提升或成功率接近时的成本下降须可复现；没有收益就不保留额外机制 |
| 4. SFT 适配 | 冻结候选 harness；先整理约 1k 个不同任务的轨迹，验证模板和 assistant-only mask 后进行 BF16 LoRA；根据开发集结果再扩大 | 训练和推理动作/工具语义一致，工具结果不进入输出 loss；能恢复 checkpoint；同 harness 下与原模型对照，退化则回到数据或模板诊断 |
| 5. 最终评测与后续 OPD 评估 | 先对已完成的 harness/LoRA 方案做冻结评测；若后续 teacher 能力和两卡资源可行，再在 100–300 个训练任务验证 OPD | 最终评测不依赖 OPD 成功；若开展 OPD，teacher 真正对 student 前缀评分，记录 token/mask/模型版本；评测仅比较入围模型与必要基线 |

阶段 3 的两个具体插件方向：

1. 上下文管理：根据 token 预算保留任务描述、最近修改、当前测试失败和关键文件发现；较长工具输出落盘并用索引按需取回。分别比较固定裁剪、确定性结构化记录、模型摘要等方案，摘要模型产生的推理也计入总成本。原始会话事件保持完整，能追查某个请求实际看到的内容。
2. 失败恢复：记录重复无效命令、测试报错与修改进展；在连续无进展时触发一次明确的重新定位/重新检查动作。改变执行策略前先在开发集观察失败分布，避免使用笼统的“再想一遍”提示替代可检验机制。新增测试只能作为开发反馈，不能替代独立验收。

消融矩阵：A=原始 Qwen3.5-27B+DSH 最小工具；B=A+上下文管理；C=A+失败恢复；D=A+两者；E=最佳 harness+27B LoRA SFT。F=最佳 harness+SFT+OPD 属于后续可选实验，需先验证资源与 teacher。mini/plus 仅作为额外框架参照，它与 DSH 工具语义不同，因此其分数差异归为整体 harness 差异，不能都算作某个插件的收益。

数据划分：20 个设施调试任务不进入报告集；开发集从训练资源池预留，首批 50 个是约 200 个开发任务的子集；按标准化的 task ID 跨所有数据源分组，随后加入仓库隔离检查。Klear 与原版 ticks 共享的任务必须归到同一侧。按任务选择 SFT 样本，再控制仓库占比和同任务轨迹数。Verified 500 题保留为最终公开对照，并明确无法排除底座预训练污染。所有近重复检查和清理规则在看最终结果前冻结。

预算原则：区分每次请求的上下文上限、单次输出上限和整条轨迹累计生成预算；不能用“32K 上下文”代替总推理成本。32K 是当前数据长度支持的首选实测目标，尚不是已运行的训练配置。所有摘要、重试及辅助模型调用都计入成本；同时报告累计输入、输出 token 和实际墙钟时间。入围方案用多个随机种子做配对复核，并报告有限样本的不确定性，不因 50 题的微小波动宣称提升。

两卡安排：设施/基线阶段优先实测 27B 的两卡推理及 32K 上下文，在测量显存和吞吐后确定并行方案与并发；SFT 阶段使用两卡 BF16 LoRA，并与 rollout 分阶段运行。9B 不默认与 27B 同时驻留。后续 27B OPD 需要另行确认更强 teacher、评分能力和资源安排，不能沿用此前“9B student + 27B teacher”的配置，也不假定存在额外 GPU 或付费 API。容器从少量并发开始，按 CPU 配额、内存、测试耗时增加。模型和训练器兼容性需单独验证，不能从 API 连通推导训练已可用。

OPD 数据合同：在模型调用边界保存确切 prompt、工具 schema、采样设置、生成 token IDs、student logprobs、loss mask 和模型版本。压缩后的请求必须保存压缩后的实际输入；不把完整日志直接重新分词后当作原始 rollout。teacher 需要能够对指定 student token/前缀返回概率；若只有文本 API，改做纠错轨迹再 SFT并如实命名。每批固定 student 版本，更新后重新采样。先选一种可解释的蒸馏目标，不同时混入多个 RL/reward 技巧。

需要处理的集成风险：DSH developer preview 的接口变化（固定 commit/profile）；SDK 最小配置默认 DeepSeek adapter（增加并验证 Qwen provider）；Klear 与 DSH 的持久 shell/编辑器协议不同（优先在目标 harness 重新采样，逐条验证可转换数据）；上下文压缩破坏训练重放（记录每次请求）；测试状态和源码泄漏（干净环境独立验收，隐藏 gold patch、验收测试和未来 Git 历史）。

目前只准备上述阶段 0/1 的环境子集，完整任务执行和实验阶段尚未开始。正式训练和持续 rollout 按现有平台执行规则准备；下一次实施从阶段 0/1 开始，不先启动长训练。第一份可交付结果应是 DSH+Qwen3.5-27B 的可运行小闭环、20 个任务的环境检查报告和冻结的基线配置。现有 tokenizer 审计使用 9B tokenizer；训练前需核对 27B tokenizer 和最终模板，不能把此前统计直接当作已验证的 27B 训练长度。

**第三个 Goal 建议：三个真实 SWE 任务的执行与验收闭环（2026-09-09，待启动）**

真实 GPU 和合成工具用例已通过，下一步验证现成仓库任务的环境接入与独立验收。本段只是下一轮建议，不代表本轮已开始下载任务镜像或运行任务。

```text
在 /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
复用已通过真实连通检查的 DSH + Qwen3.5-27B，完成三个真实代码仓库
SWE 任务的执行与独立验收闭环。

1. 阅读 AGENTS.md、README 文首和已有检查脚本。复用固定模型、
   依赖、Triton GDN 后端及部署修复，不重做环境搭建。
2. 从已下载且可关联的 SWE-smith 任务中选择三个题目非空、测试成本
   较低的 Python 修复任务，尽量覆盖两个仓库。先按环境条件选择，
   冻结任务清单再调用模型，不按模型是否做对来换题。
3. 仅获取这些任务所需的镜像/仓库，固定镜像 digest 与源码版本。
   容器存储必须在本项目内，不迁移、重启或修改共享 Docker daemon。
   优先复用现有执行组件，仅实现必要的任务/环境适配。确认 DSH 的
   bash 和编辑器都作用于同一隔离任务环境，不能一个改宿主、一个跑容器。
4. 对每个任务先检查未修复版本的预期失败，以及 gold patch 后的测试
   通过；这两种验证分别在干净环境进行。环境有问题时记录具体原因，
   最多检查十个候选以取得三个有效任务，无法取得时如实报告。
5. 给 DSH 原始问题和待修复仓库，让真实 27B 执行一次修复尝试。
   不暴露 gold patch、隐藏验收测试或未来 Git 历史。
   每题最多 30 次模型请求、10 分钟，并发 1；明确冻结输出预算。
   初始沿用已验证的 8192 上下文；若超出，记录为上下文限制，
   本轮不扩展成长上下文优化或追分实验。
6. 提取 agent 的最终补丁，在独立干净环境应用并验收，报告
   FAIL_TO_PASS、PASS_TO_PASS、补丁应用情况以及模型/环境失败。
   保留每次请求、实际工具调用、退出原因、token、耗时和全部失败记录。
7. 更新现有 README，提供单任务可重跑入口和三题结果表；结束或失败
   后清理自己启动的模型服务、DSH 进程和任务容器，保留项目内证据。

资源：所有新增状态留在本项目。最多两张 H100，启动前按当前授权
重新核对空闲情况与 UUID；不得把上轮已释放的卡当作仍被独占。
只运行有界短测试，遵守已有平台规则，不影响其他任务。

本轮不跑完整 benchmark、不计算可宣传的 benchmark 分数、不改
harness 策略、不增加新能力、不做训练数据转换或 LoRA/OPD 训练。
验收标准是任务可加载、工具在正确环境工作、补丁可独立验收、失败
可定位和命令可重跑；不要求模型把三题全部做对。清楚区分真实仓库
上的合成修复任务与自然 GitHub issue，本轮只验证环境集成。
```

变更记录：2026-09-09 UTC — 根据已通过的真实 GPU/工具检查，提出第三个小范围 Goal：三个 SWE 任务的环境、执行与独立验收；未启动该目标。

**第二个 Goal：真实 GPU 连通与最小工具闭环（已完成，原始 prompt 留档）**

第一个 Goal 的文件下载、安装和 CPU/模拟检查已完成。下一步先验证真实 GPU 内核、模型响应和工具执行，不进入 benchmark 或 harness 研发。当前仅编写下一轮目标，不代表 GPU 已获分配或已启动运行。

```text
在 /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
复用已准备的环境，完成 DSH + Qwen3.5-27B 的真实 GPU 连通与最小工具闭环。

1. 阅读 AGENTS.md、README 文首和现有启动/检查脚本。复用固定版本的
   模型、DSH、vLLM 和独立环境，不重新搭建或重复下载。
2. 先取得明确的 1–2 张 H100 GPU 分配，将 UUID、来源与时间记录到
   configs/gpu-allocation.json。未分配时只完成 CPU 侧准备并报告缺项，
   不自行选择看似空闲的卡，也不绕过启动保护。
3. source scripts/env.sh，使用现有 qwen_connectivity_smoke.py 验证
   真实模型加载、GPU kernel、流式输出和 DSH 无工具连通。
   保留原始 4K 最小探针，先定位 CUDA 兼容层和模型执行方面的问题。
4. 在独立临时工作目录内增加三个不依赖外网的微型功能检查：
   a. 让模型读取一个文件并返回其中指定内容；
   b. 在一个多轮工具任务中，验证 Shell 的目录或变量状态被保留；
   c. 让模型修复一个带失败测试的小 Python 函数，使用实际编辑工具
      写入文件并运行测试，最后由独立检查确认源码修改与测试通过。
   使用真实模型和实际工具，不以模拟回复或模型自述代替执行证据。
5. 复用正常 DSH bash/editor 配置；无工具探针的禁用工具 patch 仅用于
   第 3 步。为工具检查单独设置并记录有限的轮数、输出和超时预算，
   不将 64-token 无工具探针预算直接沿用到工具任务。必要调整仅针对
   连通性/兼容性，记录原因和配置差异，不开展架构改进。
6. 保存请求、工具参数与结果、最终文件差异、测试输出、峰值显存、
   启动时间和退出原因。更新现有 README，给出可重复的运行命令。
7. 成功、失败或超时均清理本轮自己启动的 DSH 和模型服务，核对进程
   与 GPU 占用。独立工作目录与日志保留在项目内供复查。

边界：本轮只做有超时和轮数上限的短测试；每个工具用例最多 10 轮、
5 分钟。复用既有模型启动等待上限。最多一份模型服务，GPU 数不超过
明确分配数量。所有新增文件、缓存、DSH home 与工作目录在本项目内；
不改宿主驱动、共享环境或共享 Docker，不影响其他任务。
不拉 SWE 任务容器、不跑 benchmark、不测长上下文吞吐、不改 harness
策略、不处理训练数据、不训练模型。遵守已有平台执行规则。

验收：真实 27B 能经 DSH 完成文件读取、持久 Shell 和编辑/测试闭环；
证据可追溯、命令可重跑、服务已退出。记录所有尝试和模型失败，不通过
无限重试挑成功样本。GPU 未分配或真实测试失败时，明确标记对应阶段
未验证/未通过，不能把 CPU/模拟检查当成真实闭环成功。
```

变更记录：2026-09-09 UTC — 根据已完成的最小环境准备，编写第二个小范围 Goal；新增范围仅为真实 GPU 连通和三个微型工具用例，未启动 Goal 或 GPU 服务。

**第一个 Goal：环境、模型与 DSH 准备（2026-09-08，用户缩小范围）**

当前第一个 Goal 只做以下准备工作；后面的完整项目目标保留为后续路线，不是本轮执行范围。以下为本轮已启动的 Goal 范围，当前执行状态见文首环境准备记录。

```text
在 /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
完成 DSH + Qwen3.5-27B 的最小运行环境准备。

1. 阅读项目及父目录 AGENTS.md、现有 README，检查资源与依赖。
2. 在项目内建立独立环境，固定可兼容的依赖版本，不修改共享环境。
3. 下载 Qwen/Qwen3.5-27B 官方 BF16 权重、tokenizer 和配置；
   固定 revision，校验文件完整性，支持中断后继续下载。
4. 拉取 DeepSeek 官方 deepseek-ai/deepseek-harness，固定 commit，
   安装最小运行入口所需依赖，准备接本地 Qwen 服务的配置。
5. 完成依赖检查和 DSH 启动检查。若有已明确分配的 GPU，
   进行一次本地 Qwen 与 DSH 的最小连通测试，然后退出服务；
   若 GPU 或平台条件未满足，完成独立准备并明确标记未验证部分。
6. 更新现有 README，记录版本、目录、启动命令和检查结果。

所有新增文件、环境、缓存、模型、DSH home 和日志均放在本项目内。
最多使用两张已获分配的 H100，不占用其他任务，不改宿主驱动、
共享 Docker 或已有训练环境。遵守现有平台执行规则。

本轮不下载任务容器、不跑 benchmark、不写 harness 改进插件，
不处理训练数据，也不启动 LoRA、OPD 或全参训练。
交付是可追踪的模型与 DSH 源码、独立运行环境、启动配置和简短验收记录。
必须区分“文件已下载”“依赖已安装”和“实际连通已验证”。
```

变更记录：2026-09-08 UTC — 按用户要求，将第一个 Goal 缩小到搭环境、下载 27B 和 DSH，只保留最小检查；评测、插件研发和训练移入后续目标。

**后续完整项目 Goal 参考（当前不执行）**

以下是待用户启动的执行目标文本；编写该文本不代表已经创建或启动 Goal。

```text
在 /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent 内，完成一个可复现的 DSH + Qwen3.5-27B 仓库修复 agent 实验项目：先研究 harness 架构，再用 BF16 LoRA SFT 适配。交付可运行代码、冻结配置、模型适配器、逐任务评测结果和有证据的结论。目标是检验改进是否有效；即使没有涨分，完成预定实验并如实解释也是有效交付，不得为了达到某个分数无限调参或污染评测。

约束与现有资产：
- 先阅读项目及父目录 AGENTS.md、项目 README.md，以及 data/metadata 和 data/audit。复用已经下载和校验的数据，不重复下载已有文件，不另建竞争性的计划文档。
- 所有新增代码、环境、缓存、模型、DSH home、容器存储、临时文件、日志和 checkpoint 都放在本项目目录内；执行会写缓存的命令前 source scripts/env.sh。
- GPU 预算为两张 H100 80GB；先核对分配与可用性，可见八卡不代表可用八卡。不得占用或终止别人的任务，不修改共享 Docker daemon 或共享训练环境。
- 主模型是 Qwen3.5-27B；9B 仅用于必要的快速调试或对照。当前不做全参数微调、OPD、LoopLM、多 agent 编排、付费 teacher 调用或新 UI；OPD 留到后续目标评估。
- 固定 DSH 源码版本和插件 profile，复用其模型/工具/会话/执行能力；自行实现的重点是上下文策略、失败恢复、任务适配、轨迹记录和训练接口。
- 遵守已有平台执行规则。准备正式运行需要的环境、入口脚本、挂载、资源和配置后，再处理必须由用户/UI 完成的平台步骤。外部条件未满足时继续完成独立准备工作，如实标注未执行部分，不能把准备完成报告成训练完成。

按以下里程碑推进：
1. 接通 DSH 与本地 Qwen3.5-27B。核对模型/tokenizer revision、thinking 和工具调用格式；验证读取、编辑、持久 shell、流式结果、停止和超时。保存实际生效的完整配置与每轮模型请求。先测显存和吞吐，再确定两卡推理布局及并发。
2. 接入现成的 SWE 任务环境和官方验收组件。用约 20 个设施调试任务核对未修复版本、gold patch、独立验收及环境恢复；这些任务不进入报告集。只按需获取任务镜像。gold patch、验收测试和未来 Git 历史不得暴露给 agent。
3. 在任何训练前冻结任务分组和评测协议。跨数据源按 task ID 与近重复关系去重，分开训练、开发和最终评测。开发集首批约 50 题，扩大后约 200 题；SWE-bench Verified 500 题只作最终公开对照，不据其结果调参。记录哪些限制无法排除底座预训练污染。
4. 固定原始 27B，建立 DSH 最小工具配置基线。依次实现并比较：A 原始 harness；B 仅上下文管理；C 仅失败恢复；D 两者组合。上下文策略优先验证确定性的记录/裁剪/按需取回，按失败分析决定是否需要额外摘要模型。先在 50 题筛选，再对入围方案扩大到 200 题。所有组统一预算；辅助调用、摘要和重试也计费。mini/plus 只在需要框架参照时加入，不把整体框架差异归因于单个插件。
5. 冻结选中的 harness，整理约 1000 个不同训练任务的可追踪轨迹。优先使用已下载 Klear 数据；核对 DSH 工具语义，不能靠重命名 role 假装兼容。转换无法忠实保留行为时，在目标 harness 上重新采样，单独统计采样成本。核对完整聊天模板、token 长度和 assistant-only loss mask，保留失败后恢复成功的有效过程，禁止训练工具返回内容。
6. 用成熟训练器做一次有边界的 27B BF16 LoRA SFT 实验：先验证短训练、保存/恢复、适配器加载和显存，再正式运行。32K 是长度实测目标，不是已经验证可用的配置。根据短测记录确定并冻结训练参数，不为跑通或追分暗中更换模型、数据或预算。开发集退化时先查数据/模板；允许一次有明确诊断依据的修正，不开启无限参数搜索。
7. 在同一 harness 下比较原模型和 LoRA 模型。对入围的 harness 改进做必要的随机种子配对复核；最终冻结两个主要系统，在 Verified 500 题上对照：原始 27B+基线 DSH，以及入围最终系统。若 LoRA 或插件无收益，保留更好的原始组件并如实报告，仍保留 LoRA 实验产物。

评测与完成标准：
- 每个任务记录成功/失败、环境故障、超时、格式错误、工具轮数、输入/输出 token、实际耗时、模型与配置版本。基础设施错误不得被默默删出分母或当成模型能力结论。
- 区分每次请求的上下文上限、单次输出上限和整条轨迹累计生成预算；训练公平性比较保持推理条件一致。
- 交付依赖锁定/源码版本、启动及评测入口、DSH 插件和配置、数据来源/校验/切分清单、LoRA checkpoint、实验表格、逐任务结果和复现说明。没有实际运行的项目必须明确标注。
- 最终回答：harness 哪些改动有效；LoRA 是否有额外收益；收益是否伴随更多 token/时间；当前瓶颈是什么；后续 OPD 值不值得做。给出样本规模和不确定性，不把目标改善写成已经实现的结果。
- 实现、验证和修复已获授权范围内的问题时持续推进；保存可恢复进度，按里程碑更新 README。涉及必要的外部资源信息或平台用户操作时，先准备具体可审查的材料，再提出最小必要问题。
```

**工作量估算（规划值，非实测性能）**

以熟悉 Python/TypeScript、LLM 部署和训练栈的工程师估算，完整首轮约 8–15 个有效工程工作日；若计入串行 GPU 运行、平台排队及兼容性返工，按 2–4 周日历时间规划。该估算编写时已有数据与初步审计；27B/DSH 环境的当前状态见文首。任务容器、训练栈与基线仍未实施。自动化能减少重复操作，不保证消除模型内核、工具协议、环境和存储方面的调试。

| 工作包 | 有效工程工作日估计 |
|---|---:|
| DSH/Qwen 接入、独立环境、资源与存储配置 | 1–3 |
| 任务环境、验收、轨迹记录及冻结基线 | 2–3 |
| 两类 harness 策略、消融与失败分析 | 2–4 |
| 数据适配、LoRA、恢复与推理验证 | 2–3 |
| 最终结果检查及复现交付 | 1–2 |

GPU 时间必须在阶段 1 后重新估算，不能从参数量或“能放下”推出吞吐。量级示例：如果 1000 条整理后的训练轨迹合计 20–30M tokens，两卡实测总训练吞吐分别落在 300–1000 tokens/s，则一轮训练约 6–28 小时；这些吞吐数值是假设情景，不是当前机器测量。最终两个系统各 500 题共 1000 条轨迹；若平均每条 10 分钟、有效并发 4，总墙钟约 42 小时，不含镜像准备/重试，且更高并发可能降低单条速度。开发集消融、配对复核和重新采样还需另计。

因此可先把前两个执行里程碑作为首个检查点，预计约 3–6 个工程工作日交付“可运行闭环+可靠基线”；完整目标仍包含后续 harness 实验和 LoRA。资源瓶颈优先级是长轨迹推理/评测、任务环境与协议一致性、训练内核适配，而不是原始数据条数。

变更记录：2026-09-08 UTC — 按用户要求编写可复制的完整 Goal Prompt，并估算工程与运行工作量；仅生成目标文本，未调用工具启动 Goal 或执行训练。

**环境与复现**

所有新增内容位于项目内。默认系统 Python 用于 requests/pyarrow 数据操作；`.venv-audit` 仅新增 tokenizers 0.22.2 并复用系统 CPU 数据依赖，不是 GPU 训练环境。因为系统缺少 ensurepip，该环境使用 `venv --without-pip --system-site-packages` 创建，并通过 pip 的项目内 target 安装 tokenizer。未改动共享训练环境。

```bash
cd /lustre/prod_glm_volumes/volume-20260201002229-o7c51/code-agent
source scripts/env.sh
python scripts/download_data.py --profile primary
python scripts/download_data.py --profile klear
python scripts/download_data.py --profile supplement
python scripts/audit_data.py
python scripts/audit_task_links.py
.venv-audit/bin/python scripts/audit_token_lengths.py
```

下载脚本以 `data/metadata/` 下已经冻结的元信息为输入；重新运行会校验已有文件。SERA 是仅前 100 行的独立样本，其来源和边界写在 sample manifest 中。`data/audit/*.success_index.jsonl` 只保存上游成功标记对应的定位索引，不是已经去污染或可直接训练的数据集。

当前 Docker daemon 的 data root 是 `/var/lib/docker`，与用户要求的项目内存储不一致；本轮没有拉镜像。后续先配置项目专用容器存储，并验证该 Lustre 挂载支持所选存储驱动，不迁移或重启共享 Docker daemon。父目录开发机执行规则仍适用：本轮是下载准备/CPU 审计，正式训练和持续 rollout 的执行位置需按已有平台规则落实。

变更记录：2026-09-08 UTC — 下载固定版本数据和研究源文件；建立传输校验、结构审计和 Qwen tokenizer 内容长度抽样；根据框架资源要求及数据实查，形成 mini/plus + ms-swift 的首轮建议。

变更记录：2026-09-08 UTC — 根据用户对 harness 架构目标的澄清，补查 DeepSeek 官方 DSH；将其列为主架构候选，保留 mini/plus 对照和此前数据，分开设计 harness 与模型微调实验。

变更记录：2026-09-08 UTC — 用户选择“27B 推理研究 DSH harness，训练先 LoRA”；将主模型从 9B 调整为 27B，9B 保留作对照，OPD 改为资源与 teacher 单独验证后的可选阶段。本次只更新计划，没有启动模型下载、服务或训练。

变更记录：2026-09-08 18:26 UTC — 完成本轮最小环境验收：官方模型 24/24 文件完整、固定 DSH 和推理环境就绪、CUDA 12.9 用户库/编译器在项目内、CPU 与模拟通信检查通过。最终状态见文首及 `runtime/metadata/acceptance.json`；未分配 GPU，因此没有启动真实模型服务。
