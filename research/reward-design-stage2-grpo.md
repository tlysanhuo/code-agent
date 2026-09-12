# 阶段② Agentic RL（GRPO）奖励栈设计 —— 基模大厂 TR 综合

> 状态：**已锁定（用户批准 2026-09-12 ~10:55 UTC）：按分级方案执行**——v1=二值核+原生 DAPO
> 动态采样+上游 abort 通道起步，格式罚带退场条件，部分分挂重访条件（组内方差过低再议），
> 过程正奖归阶段③ OPD。实现进 slime_dsh/（带单测），超参单另呈。
> 授权边界（用户裁定 2026-09-11/12，2026-09-12 三轮扩大）：奖励设计只认**基模大厂一手来源**
> ——完整 TR/第一方后训练披露：DeepSeek-V4（2606.19348）、Qwen3-Coder-Next（2603.00729）、
> GLM-5（2602.15763）、**Kimi K2.5（2602.02276）**、**MiniMax M2（2605.26494）+M2.1 后训练
> 博客**；实证标尺补 Meta ScaleRL（2510.13786）。Klear 等仅作数据来源，不作为配方权威；
> **不许自创规则**（2026-09-12 用户裁定：**KAT-Coder-V2.5 纳入权威集**——模型线 TR 与
> Klear 数据线区分对待，权威集现为六家）。时效核查（2026-09-12）：GLM-5.3/K2.6/Qwen3.8-Max 无训练细节披露
> （能力发布）；DSV4.1-Flash 官方卡确认沿用 DSV4 后训练（SFT→RL→OPD，无算法改动）——
> 上述 TR 即当前可得的全部权威层。
> 证据：TR 全文快照 `research/sources/base-model-trs-20260911/`；奖励章节纯文本提取
> `tmp/{qwen3cn,glm5,dsv4}_tr.txt`（本文所有引文均出自提取文本，标注行号）；K2.5/MiniMax
> 引文为 2026-09-12 在线抓取（一手 arXiv HTML/官方博客）。
> 本文新建理由：research-log 的 TR 段是调研摘要，用户要求奖励综合设计成文后单独过目；
> research-log 已加指针（见其 2026-09-12 变更注记）。

## 0. 适用范围

阶段② = dense Qwen3.5-9B SFT 检查点（hf-iter500）在 DSH harness + 本地受限进程环境上的
GRPO 训练。任务池 = 冻结 v1 清单（configs/agent-rl/rl-task-freeze-v1.json）。
阶段③（OPD，teacher=Qwen3.5-27B）只在此文定位、不展开设计。

## 1. 三份 TR 的奖励机制（逐字引文）

### 1.1 Qwen3-Coder-Next §4.2.4 Software Engineering Expert（qwen3cn_tr.txt:288-300）

**任务侧**
> "RL queries are derived from real-world software engineering tasks … To prevent information
> leakage between training stages, SFT and RL prompts are fully disjoint. In addition, we
> estimate the pass-rate distribution of each training instance and filter out both overly
> easy examples and noisy failure cases."（:288-289）

**奖励主体与两个惩罚**
> "Trajectory-level rewards are assigned based on final task completion. However, correct final
> outcomes do not necessarily imply high-quality intermediate reasoning or tool usage (Shum et
> al., 2025). To address this, we introduce additional trajectory-level and token-level
> penalties."（:291）
> "First, we apply an unfinished trajectory penalty. When the number of interaction turns
> exceeds a predefined maximum, the trajectory reward is penalized to discourage excessively
> long rollouts and failure to terminate."（:292）
> "Second, we apply a turn-level tool-format penalty. At each interaction step, we perform
> rule-based validation of tool-call format correctness. During optimization, tokens associated
> with invalid tool calls receive token-level penalties, preventing the model from learning
> malformed tool invocation patterns."（:293）

**防作弊拦截器（两届）**
> "Prior work has shown that GitHub-based environments may unintentionally leak future commit
> information, which agents can exploit to recover ground-truth fixes (e.g., via git log
> --all). To mitigate this, we adopt standard protections including removing remotes,
> branches, and tags."（:296-298）
> "During later RL stages, however, many new ways of reward hacking emerge. Agents attempt to
> reconnect local repositories to GitHub using commands such as git remote add, or retrieve
> commit history through git clone, curl, or similar tools … Fully disabling network access is
> not reasonable …"（:299）
> "To address this, we introduce a heuristic blocking rule. Any tool call containing both a
> repository link (e.g. github.com/{repo}) and network-access keywords (e.g. git, curl, wget)
> is blocked, and the agent receives explicit feedback indicating the prohibited action."（:300）

效果（Figure 7 说明，:294）：有拦截器时 RL 持续提升、平均轮数 50→130（长程能力涌现）；
无拦截器时随能力上升 exploit 自发增长。

### 1.2 GLM-5 §3.2/§4.1（glm5_tr.txt）

**算法骨干（Reasoning RL，:416-431）**
GRPO + IcePop（pop 算子把训练/推理分布失配比 ρ 超出 [1/β, β] 的 token 置零，β=2），
**去掉 KL 正则**，非对称裁剪 ε_low=0.2 / ε_high=0.28，group size 32、batch 32，全程 on-policy；
奖励为领域专属 judge/评测系统给出的**二值 outcome reward**（:438）。

**Agentic RL（§4.1，:482-487）**
组内策略优化 L(θ)=E_x[1/K Σ (r(x,y_i) − r̄(x))]；并且：
> "It is noted that only model-generated tokens are used for optimization, and the environment
> feedback is ignored in loss computation."（:487）

**噪声样本剔除与组修复（§4.1.2，:526-528）**
> "…coding-agent sandboxes can be inherently unstable and may fail for reasons unrelated to
> the model (e.g., environment crashes). Such failures introduce noisy training signals because
> they reflect environment instability rather than the model's capability. To mitigate this, we
> record the failure reason for each sample and exclude samples that fail due to environment
> collapse. For group-based sampling methods such as GRPO, removing failed samples can leave an
> incomplete group. In that case, we pad the group by repeating valid samples if the number of
> valid samples exceeds half of the group size; otherwise, we drop the entire group."（:528）

（同节另有异步版本陈旧度丢弃 w′−w₀>τ，:527——属异步基础设施，我们同步小规模首轮不涉及。）

**SFT 侧错误段掩码（§3.1，:415）**——阶段①已完成设计哲学对齐，列为背景。

### 1.3 DeepSeek-V4 §5.1/§5.1.1/§5.1.2（dsv4_tr.txt）

**管线定位（:2003-2011）**
> "…a critical methodological substitution was made: the mixed Reinforcement Learning (RL)
> stage was entirely replaced by On-Policy Distillation (OPD…)."（:2003）
领域专家 = fine-tuning + GRPO（超参沿用自家前作），统一模型经多教师 OPD（reverse-KL）整合。

**易验证/难验证任务分流 + GRM（:2118-2125）**
> "Typically, easy-to-verify tasks can be effectively optimized using simple rule-based
> verifiers or test cases."（:2118）
难验证任务才用 rubric 数据 + GRM（actor 自身即 GRM，联合优化判别与生成）。

**OPD 技术要点（:2336-2362）**：多教师 reverse-KL；token 级
sg[log π_E/π_θ] 作 per-token advantage 是省资源的简化（高方差、不稳），DSV4 用
**全词表 logit 蒸馏**；配套教师调度（缓存最后层隐藏态、按教师索引排序 dispatch、TileLang KL 核）。

**长度惩罚先例（:2020）**：不同 reasoning mode 在 RL 中施加不同 length penalty 与上下文窗口。

### 1.4 Kimi K2.5 TR §4.4.2/§3（arXiv 2602.02276，Moonshot AI；2026-09-12 在线抓取）

**核心奖励三信号（§4.4.2）**
> "We apply a rule-based outcome reward for tasks with verifiable solutions, such as reasoning and
> agentic tasks." + "To optimize resource consumption, we also incorporate a budget-control reward
> aimed at enhancing token efficiency." + GRM（生成式奖励模型）用于开放任务。
GRM 定位："not as binary adjudicators, but as fine-grained evaluators"；反 hacking 手段 =
多 rubric 分工（"multiple alternative GRM rubrics tailored to different task contexts"）。

**Token 效率专项（Toggle）**：交替「预算受限相/标准相」——预算 = 正确响应长度的 ρ 分位，
且只在平均准确率超阈值 λ 的问题上生效；K2 Thinking 上 **输出 token −25~30%，精度近零损失**。

**Agent Swarm 奖励（PARL，§3）**：r = λ₁·r_parallel + λ₂·r_finish + r_perf——r_parallel 反「串行
塌缩」、r_finish（子代理完成率）反「虚假并行」（一种 reward hacking），**λ₁/λ₂ 随训练退火到
零**；预算用 critical steps（主代理步数+每阶段最大子代理步数）计量。〔单代理项目，列为
2026 多智能体塑形的存在性证据，不采纳。〕

### 1.5 MiniMax M2 TR §6.1.5 + M2.1 后训练博客（arXiv 2605.26494 + minimax.io，2026-09-12 抓取）

**复合奖励（TR §6.1.5，动机：轨迹可达 192K token/数千中间动作）**：每步
r = α·过程奖励 + β·速度奖励 + 性能——过程奖励含「语言混杂罚+**工具调用格式错误罚**+
良构中间推理奖」；速度奖励=墙钟完成时间比值过单调递减塑形（激励并行工具调用）；
reward-to-go+轨迹基线降方差。α/β 未公开数值，无奖励消融。

**M2.1 第一方后训练经验（工程细节更实）**：① F2P/P2P 按任务类型变体（修 bug 用 F2P+P2P；
加功能提取新增测试点；性能优化提取能验证稳定性能差的 P2P；代码评审用 LLM 一致性≈可验证）；
② 算法=CISPO（截断重要性采样**权重**而非 token——修 PPO/DAPO 把 'wait' 这类词永久剪没梯度
的问题）；③ agent 噪声治理=**MIS（多重重要性采样）+ 基于 PPO 的轨迹过滤**（滤长尾异常轨迹）；
④ FP32 LM head（训练-推理一致性）；⑤ 单 scaffold 训练泛化差，多 scaffold 拒绝采样。

### 1.7 KAT-Coder-V2.5 TR（arXiv 2607.05471，Kwaipilot/快手，2026-07；SWE-bench Verified
79.6%；2026-09-12 第七轮遗漏排查新增。**2026-09-12 用户裁定：纳入权威集**——与被裁定「仅作数据」的
Klear 同属快手系，但这是其模型线 TR、非数据管线论文）

**三层规则奖励 + GRM**：
- **Tier 1 核心分**：二值——F2P 全过+P2P 不回归才满分（"all-or-nothing outcome criterion"）；
- **Tier 2 行为约束（轨迹级惩罚，与结果无关）**：think/response 内容重复、乱码输出、
  `<tool_use_error>` 类工具调用错误（缺参/错参）、工具调用误置于推理字段、单轮内重复
  同工具调用、超阈并行、遗留调试产物；
- **Tier 3 失败轨迹激励（部分分）**：文件检索 F2 分（precision+recall 联合）+ 单测通过
  子集正分——"assigns positive rewards to meaningful progress within failed trajectories"；
- **GRM**：RL 训练的 judge 按 rubric 维度打分（故障诊断/复现、修复后验证、执行策略——
  罚复杂 python -c、缺复现、缺回归测试），其自身训练含 λ 加权假阳性罚。

**算法与稳定性工程**（对我们最有 transfer 价值的部分）：
- **PPO+GAE 而非 GRPO**，理由原话：为了"penalize localized bad behaviors such as erroneous
  tool calls or regressive intermediate patches"（critic 提供逐步负反馈通道）；事后信息
  增强的非对称 critic（终局奖励/测试结果/patch diff 只注入 value 估计，不进策略）；
- **沙盒可靠性决定训练成败**：早期 ~16% 轨迹的失败源自沙盒本身；修复后（镜像管理重设计、
  环境变量验证器 bug 修复）沙盒反馈错误 16%→<2%、**训练崩溃降 ~10 倍**——GLM-5 噪声
  剔除方向的最强实证；
- **网关 /generate（TITO）**：~200 轮样本中 ~40% 观察到重分词漂移——再证 token 保真必要性；
- **防作弊**：环境去污剥除 git 历史与元数据；测试结果**解析结构化输出而非 exit code**、
  要求 >90% 测试收集率与可复现判定；过程感知的轨迹过滤剔除「通过但走捷径」（硬编码/
  篡改测试/绕过机制）。

**同类补录**：AgenticQwen（arXiv 2604.21590，8B/30B-A3B 小模型多轮 GRPO——我们的量级；
reasoning RL 二值 outcome，agentic RL 用 235B judge 按 rubric[0,1] 子目标部分分，全模拟
环境）；Devstral（arXiv 2509.25193，Mistral，2025-09）/Devstral 2（72.2% SV）；综述锚点：
*The Landscape of Agentic RL for LLMs*（2509.02547）、*From Reasoning to Agentic: Credit
Assignment*（2604.09459，47 种信用分配方法谱系）。

### 1.6 Meta ScaleRL（*The Art of Scaling Reinforcement Learning Compute for LLMs*，arXiv
2510.13786，2025-10，40 万 GPU 时的系统实证；MiniMax M2.1 亲引为 scaling 标尺）

不是奖励设计论文，但给出**什么才有效**的实证配方：sigmoidal scaling law；PipelineRL
8 步 off-policy；**CISPO**；FP32 logits；**DAPO 动态采样+自适应 prompt 过滤**。
对我们的背书：slime 原生的 dynamic-sampling（§2.3）与 cispo（advantage_estimator 三选项之一）
都在 ScaleRL 获胜配方里。

## 2. 综合设计（本项目阶段②奖励栈）

三家共同核心：**SWE 任务是易验证任务 → 规则验证器（测试通过）二值奖励**（DSV4 :2118 明说
易验证走规则；Qwen3CN :291 终局完成奖励；GLM-5 :438 二值 outcome）。在此基础上叠加
Qwen3CN 的两个惩罚与拦截器、GLM-5 的噪声剔除与组修复。分层如下：

### 2.1 奖励公式（v1 提案）

```
R(trajectory) = 1.0                                  # 测试通过（F2P 全过 + P2P 不回归，金标判定）
              − c_unfinished · 1[turns > T_max]      # Qwen3CN 未完成惩罚（轨迹级）
              − c_fmt · (invalid_toolcall_tokens /   # Qwen3CN 工具格式惩罚（token 级）
                         supervised_tokens)          #   实现层级见 §3，归一方式待定
环境崩溃/基础设施失败: 不进奖励域 —— 剔除样本 + 组修复（GLM-5 规则）
```

- **c_unfinished、c_fmt、T_max、组大小 K：TR 均未给数值** → 全部列为训练超参，
  按项目纪律需用户批准后才进配置；**smoke 首轮建议 c_unfinished=c_fmt=0**（即纯二值，
  三家共同基线，也最接近「不自创」），惩罚项作为紧随其后的消融臂逐个打开。
- 防作弊不属于奖励域：拦截器在执行层（§2.4）。
- **不做**：GRM/ORM 混合奖励（GLM-5 §3.4 General RL 范畴，DSV4 明说易验证任务不需要）、
  DSV4 长度惩罚按 mode 分档（其对象是 reasoning token 长度，不是 agent 轮数；Qwen3CN 的
  未完成惩罚已覆盖我们的轮数预算需求）。

### 2.2 数据侧（rollout 前）

| TR 机制 | 我们的对应 | 状态 |
|---|---|---|
| SFT/RL prompts 完全不相交（Qwen3CN :289） | 冻结清单 v1 已做排除审计（61 blocked + 20 held-out 等，与 Verified/full-test 零交集） | ✅ 已冻结 |
| pass-rate 分布过滤：去过易 + 噪声失败（Qwen3CN :289） | qualification 批（在飞，logs/prepare-rl-round1-full.log）先保证环境有效（buggy 失败 F2P、gold 通过）；过易过滤在首轮 rollout 后按实测 pass-rate 复核（Qwen3CN 也是按实例估计） | 环境有效性在飞；过易过滤 = 二步 |
| 环境就绪（GLM-5 10k 环境 / Qwen3CN 80 万 Docker 任务） | 本地 venv 环境 per-task qualification（自建训练协议，已授权路线） | 在飞 |

### 2.3 训练侧（advantage/loss）

| TR 机制 | 我们的对应 | slime 原生支持 |
|---|---|---|
| 二值 outcome → 组内归一化（三家） | advantage_estimator=grpo，`--rewards-normalization`（均值减除）+ `--grpo-std-normalization`（除 std） | ✅ slime/ray/rollout.py:285-303 |
| 只训模型生成 token，环境反馈不进 loss（GLM-5 :487） | slime_dsh 生成器产 token 级 loss_mask（工具观察=0）——已 CPU 验证（TITO 同族，Polar/GLM-5/slime_dsh 三处印证） | ✅ rollout.py:344-358 逐 token mask |
| 未完成惩罚（Qwen3CN） | `--custom-rm-path`：`custom_rm(args, sample) -> float` 内按 DSH 元数据轮数判超额扣分 | ✅ arguments.py:1367-1376 |
| 工具格式 token 级惩罚（Qwen3CN） | advantage 在 slime 里是逐 token 张量：`--custom-advantage-function-path` 在 GRPO 归一化后对无效调用 token 的 advantage 减常数 c_fmt | ✅ loss.py:704-743（custom 钩子在 KL 计算后、可写 per-token advantages）；备选近似=折算进标量 custom_rm（非 token 级，列偏差） |
| 环境崩溃剔除（GLM-5 :528） | `--rollout-sample-filter-path` 置 `sample.remove_sample=True` → loss_mask 全零 | ✅ rollout.py:351-352 |
| **组填充/丢弃**（GLM-5 :528：有效>半则重复填充，否则整组丢） | ⚠️ 半缺口：`remove_sample` **不改变 advantage 归一化参与**（arguments.py:1442-1444 明示）；上游 coding_agent_rl 的 abort 处理是 reward=0+remove（examples/coding_agent_rl/generate.py:315-338），GLM-5 语义要求崩溃样本不进组统计 | 需 `--rollout-all-samples-process-path`（arguments.py:1446-1453）或 `--custom-reward-post-process-path`（rollout.py:279-281 整体替换组归一化）实现剔除+填充/丢弃 —— 新增自写代码，带单测 |
| ε 0.2/0.28 非对称裁剪、去 KL（GLM-5 :431；slime 官方 SWE 配方同款） | slime 参数现成（eps-clip low/high、kl-loss-coef 0） | ✅（数值=训练超参，smoke 前给用户批） |
| 零梯度组过滤 + 过采样保有效批（**DAPO 动态采样**，arXiv 2503.14476 ByteDance Seed） | `--over-sampling-batch-size` + `--dynamic-sampling-filter-path`，自带 `check_reward_nonzero_std[_with_fallback]`（组内奖励 std≤1e-6 即丢弃该组，fallback 版防采样死循环） | ✅ **原生**（arguments.py:440-462；filter_hub/dynamic_sampling_filters.py）——见 §4 业界对照 |
| IcePop pop(ρ)（GLM-5 :426-428） | slime 有 `--use-rollout-logprobs`/`--use-tis`/mismatch metrics（loss.py:1097）但语义非 IcePop | 首轮不用（同步 on-policy 小步距下失配有限）；记录为后续项 |
| 异步版本陈旧度丢弃（GLM-5 :527） | 不适用（首轮同步） | — |

### 2.4 执行侧（防作弊，Qwen3CN :296-300）

1. **标准保护**：任务工作区准备时移除 git remotes/branches/tags（SWE-bench PR#471 泄漏
   防护，Qwen3CN 称 standard protections）。→ 环境准备脚本加检查项 + qualification 断言。
2. **启发式拦截器**：slime_dsh 生成层拦截「同时含仓库链接（github.com/{repo} 等）与
   网络访问关键字（git clone/remote add、curl、wget）」的工具调用，**阻断并给 agent 明确
   反馈**（Qwen3CN 原文行为，agent 可继续任务而非直接终止）。
3. 我们的本地环境无外网（网络命名空间隔离），拦截器是纵深防御第二层而非唯一防线；
   保留它是因为 Qwen3CN 观察到 exploit 随能力增长而涌现（Figure 7 右）。

### 2.5 阶段③ OPD 的定位（只定位，不设计）

- DSV4（:2003）：混合 RL 阶段被 OPD 完全取代，作为专家整合手段；GLM-5（:453-458）用
  on-policy 跨阶段蒸馏防遗忘，advantage=sg[log π_teacher/π_train]、组大小可降到 1。
- slime 原生 `--use-opd --opd-type sglang`：teacher logprob 经 rm_url 取回、存
  `sample.teacher_log_probs`、KL 项进 compute_advantages_and_returns；
  `post_process_rewards` 明示「若有任务奖励可在此叠加」（slime/rollout/on_policy_distillation.py:26-78）。
  即 **阶段③ = 阶段② 奖励栈 + OPD KL 项**，管线同族 DSV4「专家 GRPO→OPD 整合」。
- DSV4 的全词表蒸馏 vs token 级简化之争（:2358-2361）：slime 实现是 token 级
  （teacher_log_probs 逐 token）——与 GLM-5 跨阶段蒸馏公式同款；DSV4 认为高方差。
  2-4 卡预算下全词表不可行（248k 词表 × 27B teacher），接受 token 级并监控，
  作为已知偏差记录（OPD 综述 2604.00626 的 Pass@k 风险同此监控）。

## 3. 实现计划（批准后执行；slime_dsh/ 内，vendor/slime 零修改）

1. `slime_dsh/reward.py`：custom_rm（二值 + 未完成惩罚开关）+ 崩溃/失败原因分类枚举
   （environment / model / budget，GLM-5「record the failure reason」的字段化）。
2. `slime_dsh/group_repair.py`：组级剔除+填充/丢弃（GLM-5 规则），挂
   `--rollout-all-samples-process-path`；**单测**：全崩组丢弃、半崩组填充、无崩组直通、
   剔除后组统计不含崩溃样本 reward。
3. `slime_dsh/format_penalty.py`：DSH 工具调用格式校验器（规则级，Qwen3CN
   rule-based validation）+ token 定位；接 `--custom-advantage-function-path` 的 per-token
   减法；单测：无效调用 token 恰好被罚、观察 token 不受影响（与 loss_mask 正交）。
4. `slime_dsh/blocker.py`：拦截器规则 + 反馈注入；单测：链接+关键字组合命中、纯链接
   或纯关键字放行、反馈后轨迹可继续。
5. 环境准备脚本：git 标准保护断言。
6. 以上全部 CPU 单测先行；GRPO smoke 超参（组大小、lr、c_* 等）另列清单给用户批。

## 4. 业界开源实现对照（2026-09-12 增补调研，用户指令「调研业界的方案」）

> 范围说明：§1-§3 的配方权威仍按用户裁定限于三份基模 TR；本节是**开源实现的工程参照**，
> 用来校准决策点与超参取值——即「别人实际怎么落地的」。引用带标题+arXiv号/仓库+机构。

### 4.1 奖励形态的两个阵营（2026-09-12 二轮调研修正：首轮「纯二值是主流」结论作废）

> 修正记录：首轮调研只检视了手里 4 个开源实现就下「纯二值主流」结论，被用户指出后扩大调研，
> 发现**基模大厂 TR 无一是纯二值**——正确图景是「可验证结果信号为核 + 塑形项叠加」，
> 塑形密度与轨迹长度/算力规模正相关。以下分阵营如实列出。

**阵营 A：大厂 TR（复合/塑形奖励，全部非纯二值）**

| TR | 奖励构成（除测试通过主信号外） | 出处 |
|---|---|---|
| Qwen3CN（2603.00729） | +未完成惩罚（轮数超额）+工具格式 token 级惩罚+防作弊拦截器 | §1.1 逐字引文 |
| GLM-5（2602.15763） | 二值 outcome + 崩溃剔除/组填充丢弃（数据卫生层塑形） | §1.2 |
| DSV4（2606.19348） | 专家 GRPO 按推理模式施不同长度惩罚与上下文窗口；难验证任务 rubric+GRM | §1.3 :2020 |
| **MiniMax M2**（*The MiniMax-M2 Series*，arXiv 2605.26494，MiniMax——本轮新增） | **复合奖励**：① 过程奖励（每步：语言混杂罚+**工具调用格式错误罚**+良构中间推理奖）；② 任务完成时间奖励（墙钟比值过单调递减塑形函数——刻意激励并行工具调用）；③ reward-to-go+轨迹基线。每步 r = α·过程 + β·速度 + 性能；α/β 未给数值、无奖励消融。动机原话：轨迹可达 192K token/数千中间动作，纯 outcome 不够 | TR §6.1.5（2026-09-12 抓取）；配套开源框架 Forge（MiniMax-AI HF blog） |
| **SWE-RL**（*SWE-RL: Advancing LLM Reasoning via RL on Open Software Evolution*，arXiv 2502.18449，Meta，NeurIPS'25——本轮新增） | **完全不用测试**：奖励 = 模型补丁与金标补丁的序列相似度（unified diff 编辑相似度），格式错误 → **−1 罚**，超长补丁按长度罚缩放——为免环境执行的规模化路线 | github.com/facebookresearch/swe-rl（src/swerl/core） |
| Kimi K2.5（2602.02276，Moonshot） | agentic RL 用终端/SWE 可验证奖励（细节在 TR agentic 章，未逐字核到公式，列为待补） | arXiv 2602.02276 |

**阵营 B：开源 SWE-agent RL 实现（纯二值 + 组卫生）**

| 实现 | agentic 阶段奖励 | 出处（本地可查） |
|---|---|---|
| slime 官方 coding_agent_rl | 1.0 iff 解决，无部分分；agent 超时只记 warning 不扣分；崩溃 abort=reward 0+remove | vendor/slime/examples/coding_agent_rl/{generate.py,swe.py} |
| DeepSWE（rllm，arXiv 2508.17146） | **博客+本地代码双重确认稀疏 0/1 ORM**（"To keep things simple, our reward function employs a sparse ORM"）；R2E-Gym runtime 三路奖励源码均严格 0/1；GRPO++（clip-higher/去KL/去std/长度归一）+ compact filtering 防奖励塌缩 | research/sources/prior-art-pipeline-20260910/deepswe_blog.html + deepswe-coldstart-20260909/ 代码快照 |
| NeMo-RL SWE-2（NVIDIA） | 二值测试通过；G=8、恒定 LR 1e-6、clip 0.2/0.28 | docs.nvidia.com/nemo/rl（2026-09-12 抓取） |
| SkyRL mini-swe | `reward = int(resolved)` | 本地快照 mini_swe_generator.py:114 |
| SWE-Gym | resolved 定义（F2P 全过+P2P 不回归）即二值 | arXiv 2412.21139 |

**⚠️ 一条以讹传讹的澄清（本轮核实）**：R2E-Gym 论文（arXiv 2504.07164）标题的「Hybrid Verifier」
是**推理期重排序**（执行分+免执行分 Top-n 融合），其训练是拒绝采样 SFT 而非 RL——**不构成**
非二值 RL 奖励的证据；其 runtime 开源代码的 RL 奖励是严格 0/1。

**学术对照实验（奖励形态的直接证据，2025-2026）**：
- *Exploring Pass-Rate Reward in Reinforcement Learning for Code*（arXiv 2605.02944）：受控实验中
  **测试通过率（pass-rate）部分分并不能可靠地胜过二值**——部分分不是自动更优。
- *Rollout Pass-Rate Control*（arXiv 2605.05112）：二值奖励的 RL 信号在 **~50% rollout 通过率**
  附近最强——**任务难度过滤比奖励复杂化更关键**（与 Qwen3CN 的 pass-rate 分布过滤一致）。
- SWE-RM（arXiv 2512.21919，Qwen team，即 Qwen3CN TR §4.2.4 引的 Shum et al. 2025）：二值信号
  稀疏、难区分轨迹质量——Qwen3CN 加过程惩罚的动机来源。
- ToolRL（NeurIPS 2025）：工具学习的细粒度奖励设计研究。

**修正后的对照结论**：两个阵营不矛盾，而是**同一设计在不同约束下的两端**——大厂 TR 轨迹长
（192K/千步）、算力大，必须加过程/速度/格式塑形对抗稀疏与奖励塌缩；小算力开源复刻以
二值+组卫生（DAPO 动态采样）+难度过滤拿到大部分收益。**没有「纯二值是业界标准」这回事**，
首轮该表述作废；我们的设计应落在两阵营之间按预算定位。

### 4.2 未完成/长度惩罚的开源形态：DAPO 软超长惩罚 + AGL 已解轮数惩罚

- **DAPO**（*DAPO: An Open-Source, Large-Scale Reinforcement Learning System*，arXiv 2503.14476，
  ByteDance Seed）**Soft Overlong Punishment**（式 13）：截断样本奖励 = −1·min(L_超额/L_cache, 1)，
  在 [L_max−L_cache, L_max] 线性爬升、到硬顶饱和 −1；原设置 L_max=16384 + 软缓冲 4096。
  verl 落地为 `overlong_buffer`（docs: verl.readthedocs.io/en/latest/algo/dapo.html）。**这是
  「未完成惩罚」的 token-长度版开源实现**（Qwen3CN 是轮数版）。
- **Agent Lightning swe_smith**（Microsoft，本地快照 research/sources/microsoft__agent-lightning）：
  `length_penalized_reward`（smith_agent.py:656-677）——**只罚已解决(solved)的训练轨迹**：
  `1.0 − λ·clip((n_turns−t0)/(max_turns−t0), 0, 1)`，默认 t0=80、λ=0.1；另叠 prompt 膨胀惩罚
  （软起 50k / 硬顶 64k / 最大罚 0.1）。**关键设计：验证集奖励永不整形**（驱动检查点选择），
  未解决轨迹不罚（保留探索空间）——与 Qwen3CN「罚未完成」互补的另一面。
- **Klear-AgentForge**（arXiv 2511.05951，快手 Kwai-Klear）：论文称全开源，但截至本调研
  **未找到公开 RL 训练代码仓库**（HF 仅有模型+SFT 数据）；其截断过采样+掩码只有论文描述，
  按用户裁定本就只作数据参照、非配方权威。

### 4.3 格式问题的开源形态：廉价 pivot 阶段 / 反馈修正，而非 token 级惩罚

- **NeMo-RL Stage-1 pivot**（NVIDIA；方法论文 *PivotRL: High Accuracy Agentic Post-Training at
  Low Compute Cost*，arXiv 2603.21383）：从专家轨迹抽单步决策点做**无沙箱单步 RL**，奖励 =
  与专家动作的弹性参数匹配（graded，实测 0.2→0.55 爬升 = 格式学习曲线）；Stage-2 再上全
  agent 二值奖励。配套数据集模式 Nemotron-RL-...-Tool-Use-Pivot-v1（每 assistant 步一个
  单步任务）。**这是「SFT 冲淡格式、RL 初期格式错误率高」的业界标准补救**——research-log
  2026-09-10 轮已记录为可与主线叠加的选项，当时 pass@1 23.6→30.4%。
- **Agent Lightning**：格式错误走**反馈修正**通道（format_error_message 带示例回注，让模型
  自纠）而非奖励惩罚（smith_agent.py 的 format_error 分支）。
- **开源常规**：轨迹级 format reward（open-r1 `rewards.py` 的 format_reward 是被广泛复制的
  模板；NousResearch hermes-agent 文档化 incremental/strict/correctness 三层奖励）。
- **开源代码层未找到**：Qwen3CN 式「无效工具调用 token 级惩罚」的公开**实现**——但
  **TR 层有双背书**：Qwen3CN（token 级）与 MiniMax M2（步级过程奖励内含工具调用格式
  错误罚，§4.1）都把格式罚放进奖励本身。开源无现成代码只说明要自写（slime custom
  advantage 钩子可表达，§2.3），不说明方案存疑。

### 4.4 组卫生：DAPO 动态采样（slime 原生）+ GLM-5 崩溃剔除（TR 独有）

- **DAPO 动态采样**：过采样提示、丢弃组内奖励全同（std=0，零梯度）的组、直至凑满有效批——
  开源界组卫生的事实标准，且 **slime 已原生实现**（§2.3 表新增行）。NeMo-RL 文档同款
  （std>0 过滤+跨批累积凑批）。
- **GLM-5 崩溃剔除+填充/丢弃**（§1.2）：未在检视的开源实现中见到同款；slime 的 abort 通道
  （reward=0+remove）是最近似物，差异在崩溃样本以 0 奖励参与组归一化。**两机制正交可叠**：
  动态采样管零梯度组，崩溃剔除管噪声样本；本地环境不稳（qualification 日志的 startup
  failure 率）放大了后者的必要性。
- 已知实现陷阱（NeMo-RL issue #2431）：动态采样若按**整形后**奖励过滤，长度类奖励不对称
  渗入过滤决策——若启用 4.2 的软超长惩罚，过滤应基于**原始**奖励。

### 4.5 防作弊：基础设施层为权威、代码拦截为兜底

- **Agent Lightning**（本地快照，最完整开源参照）：①relocate .git 出工作树（O(1) 改名，
  关闭 git 历史/`git show`/`git log -p`/checkout 前置 sha 泄漏——SWE-smith 注入型任务的
  特有泄漏面）；②四通道动作拒绝：git 调用 / 网络抓取（curl/wget/python-http）/ 包安装
  （pip/conda）/ 测试篡改（conftest/pytest.ini/sitecustomize 等），**每条拒绝都带指路反馈**
  （告诉模型该怎么干活）；③原文立场：「Network/install/tamper blocks are a code backstop;
  the authoritative fix is a default-deny egress NetworkPolicy」。
- **Qwen3CN**（§1.1）：标准保护（去 remotes/branches/tags）+ 启发式拦截器（仓库链接∧网络
  关键字→阻断+反馈）。保留网络是刻意设计（合法用途），靠拦截器区分。
- **我们本地路线的现实**：受限进程沙箱网络命名空间隔离 = 天然的 default-deny egress，
  已满足 AGL 的「权威层」；git 元数据处置需按任务来源定（SWE-Gym 环境带完整 git 历史，
  relocate 或去 remotes/branches/tags 二选一，实现时按 AGL/Qwen3CN 两参照执行）。

### 4.6 对设计的净影响（供 §5 决策点引用，经三轮调研收敛）

1. **奖励核心不是「二选一」而是「核+塑形」**：五家大厂 TR（§1.1-1.5）全部是可验证结果核 +
   塑形项；开源小算力实现常只跑二值核。我们 DSH 格式零训练起步、初期解决率低，稀疏问题
   比典型开源复刻更严重，塑形优先级应高于「照抄 slime 官方示例」。
2. **2026 的新一等公民是 token/时间效率塑形**：K2.5 budget-control+Toggle（−25~30% token）、
   MiniMax M2 墙钟速度奖励、Qwen3CN 未完成惩罚、DSV4 按模式长度罚——连 GLM-5.3 的发布
   卖点都是 fewer output tokens。我们 A/B 已见 SFT 使步数 −44%，此塑形有直接先例可依。
3. 部分分（pass-rate）**有争议**：反方 2605.02944（受控实验不可靠更优），正方 KAT-Coder-V2.5（Tier 3 失败轨迹单测子集正分+检索 F2，SV 79.6%）——v1 不引入、标注为 v1 后重访项（若难度筛选后仍大量 0/4 组再考虑）；
   难度过滤（~50% 通过率甜点，2605.05112；ScaleRL 自适应 prompt 过滤同族）收益更确定。
4. **格式塑形有两家大厂 TR 背书**：Qwen3CN（token 级）与 MiniMax M2（步级过程奖励）；
   NeMo-RL pivot 是前置教学路线。开源无现成代码=需自写（slime 钩子可表达），不等于存疑。
5. **环境噪声治理有三个业界机制可选**：GLM-5 崩溃剔除+组填充/丢弃；MiniMax M2.1 的
   MIS+轨迹过滤（滤长尾异常轨迹）；DAPO 动态采样（slime 原生，ScaleRL 获胜配方成分）。
6. **算法层**：CISPO 同时被 MiniMax（生产）与 Meta ScaleRL（实证）背书，slime 原生支持
   （advantage_estimator=cispo）；GRPO 亦是全部五家的公开默认起点。
7. GRM/ORM：仅用于不可验证任务（DSV4/K2.5/GLM-5 一致）；SWE 走规则验证器，不建 GRM。

### 4.7 过程奖励 vs 惩罚的专项对照（2026-09-12 第六轮，用户问题「只罚不奖对吗」）

**设计现状确认**：本设计 = 终止结果奖励（二值测试通过）+ 规则化负向惩罚（未完成/格式）+
数据卫生；**无正向过程奖励**。该结构与业界实践的对照：

| 层 | 谁在做 | 内容 |
|---|---|---|
| 纯 outcome（无任何过程项） | DeepSWE、slime 官方、NeMo-RL SWE2、LEGO-RL（明示局限）、SWE-RL（相似度 outcome） | 生产/开源主流 |
| outcome + **只罚不奖** | **Qwen3CN、K2.5（budget 罚）、DAPO（超长罚）、DSV4（长度罚）** | 五家 TR 中四家的塑形形态——**与本设计同构** |
| outcome + 含正向过程项 | 仅 MiniMax M2（良构中间推理**奖**；α/β 未公开、无消融） | 生产中唯一，证据最弱 |
| 学术正向过程奖励线 | ToolRL（NeurIPS'25 2504.13958：细粒度正奖——工具名/参数名/参数值分项打分）；AgentPRM（2511.08325）；SWE-TRACE（2604.14820 rubric PRM）；PaTR（2607.15610，PRM 树 rollout，SWE-bench +5.0）；implicit step rewards（ICLR 2026） | 有效但需训 PRM/额外 rollout |

**三个支撑「只罚不奖」的论据**：
1. **风险不对称**：规则化惩罚不可 hack（少做坏事即少罚），正向过程奖励需要 PRM/rubric——
   新的 reward-hacking 面 + 训练/服务预算（2-4 卡装不下第二个奖励模型）。
2. **管线分工**：我们的三段设计已把「密集正向监督」分配给了阶段③ OPD（teacher 逐 token
   reverse-KL——不依赖可 hack 的奖励函数的密集信号）。GRPO 阶段保持 outcome+惩罚、OPD
   阶段提供密集正信号，与 DSV4「专家 GRPO→OPD 整合」同族分工。
3. **时间线证据**（ToolRL 消融）：format 奖励在**前 ~30 步内饱和**、之后由 correctness 驱动
   ——格式是快技能。我们的 DSH 格式冷启动问题同理：格式惩罚/pivot 应为**短命机制**，
   饱和后撤除，不长期占用奖励通道。

**修订**：§5.1 选项 c（格式惩罚）增加退场条件「格式错误率连续 N 步低于阈值后关闭」；
学术 PRM 线（AgentPRM/SWE-TRACE/PaTR，SWE-bench +5）作为阶段② v2 或 LoopLM 臂的
引用性后续项，不入首轮。

## 5. 待用户拍板的决策点（经 §4 业界对照校准）

1. **奖励核心与塑形起点**（三轮收敛后）：核心=测试通过可验证信号（五家 TR 一致）；
   塑形分四档：**a)** 二值核+组卫生（开源小算力同款）；**b)** a + 效率塑形（Qwen3CN 未完成
   惩罚 / K2.5 budget-control——2026 TR 新一等公民，我们有 A/B 步数 −44% 的既有观测）；**c)** b +
   工具格式惩罚（token 级贴 Qwen3CN / 步级贴 MiniMax M2，双 TR 背书，需自写）；**d)** 格式走
   pivot 前置教学（NeMo-RL/PivotRL 完整先例）后再 a/b。建议「先测量后加码」：首轮 a +
   严格记录格式错误率/轮数分布/崩溃率，再定 b/c/d。测试比例部分分 v1 不引入（争议项：2605.02944 反例 vs KAT-Coder-V2.5 正例，留作组内方差过低时的备选）。
2. **环境噪声治理三选（可叠）**：slime 原生 DAPO 动态采样（零代码，ScaleRL 获胜配方成分）；
   GLM-5 崩溃剔除+组填充/丢弃（小自写）；MiniMax M2.1 MIS+轨迹过滤（小自写）。首轮用原生
   动态采样+上游 abort 通道，按实测崩溃率决定是否加后两者。
3. **advantage estimator**：grpo（五家默认起点）vs cispo（MiniMax 生产+Meta 实证双背书，
   slime 原生 `--advantage-estimator cispo`）——超参单里给用户选。
4. 各超参数值：业界锚点已集齐（NeMo-RL：G=8、恒定 LR 1e-6、clip 0.2/0.28、KL 0、turns 200、
   超时 1800s；AGL：t0=80/λ=0.1、prompt 软 50k/硬 64k；DAPO：软缓冲 4096 饱和 −1；K2.5 Toggle：
   预算=正确响应 ρ 分位、仅在准确率>λ 时生效；slime 官方 SWE 配方：batch 8×8、96k 上下文、
   1800s/600s 预算）——smoke 配置单按此起草另呈。
