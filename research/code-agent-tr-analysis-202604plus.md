# Code-Agent 技术报告分析(2026-04 以后)——面向项目讲解的对照文档

> 建档:2026-09-12。用途与边界(用户指令):①只收 **2026 年 4 月及以后**的 code agent
> 相关模型技术报告(此前的 GLM-5/K2.5/Qwen3CN 等在奖励设计文档中,不在此重复);
> ②本项目的讲解以 PPT 呈现,本文按「每 TR 一张分析卡+横向对比表+对我们项目的映射」
> 组织,可直接转 PPT 页。新建文档理由:research-log 管时序流水,reward-design 文档管
> 奖励单一主题;跨 TR 的综合分析+演示素材是独立交付物,research-log 已挂指针。
> 入选核查(2026-09-12):Devstral 2 = 2025-12 发布→出局;Qwen3.5 无文本 TR;K2.6/
> GLM-5.3/Qwen3.8-Max 无训练披露;榜单新贵(Claude Mythos/Fable 5/GPT-5.3 Codex)为闭源
> 无 TR。**入选 6 份**:KAT-Coder-V2.5、DeepSeek-V4、MiniMax M2、OpenThoughts-Agent、
> AgenticQwen、LEGO-RL(方法学)。

## 卡 1:KAT-Coder-V2.5(快手 Kwaipilot,arXiv 2607.05471,2026-07)

| 维度 | 内容 |
|---|---|
| 基本盘 | SWE-bench Verified **79.6%**(开源第一梯队);专业 coding agent 模型线 |
| 奖励 | **三层规则+GRM**:①二值核心(F2P 全过+P2P 不回归);②轨迹级行为罚(think/response 重复、乱码、`<tool_use_error>` 缺参错参、工具调用误入推理字段、单轮重复同工具、超阈并行、遗留调试产物);③**失败轨迹部分分**(文件检索 F2+单测子集正分);GRM=RL 训练的 rubric judge(带假阳性罚) |
| 算法 | **PPO+GAE(弃 GRPO)**——理由:critic 提供逐步负反馈通道以"惩罚局部坏行为";事后信息非对称 critic(终局奖励/测试结果只注入 value,不进策略) |
| 环境工程 | 沙盒可靠性决定成败:早期 **16% 轨迹败于沙盒本身**→修复后 <2%,**训练崩溃降 10 倍**;网关 /generate TITO(~200 轮样本 **40% 重分词漂移**) |
| 防作弊 | 剥 git 历史+元数据;测试结果解析结构化输出(非 exit code)+>90% 收集率;捷径轨迹过程过滤 |
| 启示(我们) | 噪声治理优先级的最强实证(印证 GLM-5 方向);部分分是争议项的正方;"惩罚局部坏行为"可用 token 级 advantage 实现,无需引入 PPO |

## 卡 2:DeepSeek-V4(DeepSeek-AI,arXiv 2606.19348,2026-06;DSH 作者)

| 维度 | 内容 |
|---|---|
| 基本盘 | 后训练范式声明:**混合 RL 阶段被 OPD 完全取代** |
| 管线 | 领域专家(精细调+GRPO,领域奖励)→ 多教师**全词表 OPD**(reverse-KL)整合成统一模型 |
| 奖励 | 专家 GRPO:领域定制奖励+按推理模式(length penalty/上下文窗口分档)+难验证任务 rubric+GRM(actor 自身即 GRM 联合优化) |
| 环境工程 | DSec 沙箱:全局有序轨迹日志、抢占安全恢复(WAL 缓存重放免重执行——**从头重生成会引入长度偏差**)、四种执行基底统一接口 |
| 效率 | 全词表蒸馏的教师调度:ZeRO 式分片按需加载、缓存最后层隐藏态、按教师索引排序 dispatch、TileLang KL 核 |
| 启示(我们) | 我们三段管线(SFT→GRPO→OPD)与此同族——阶段③ OPD 的核心叙事来源;token 级 vs 全词表的取舍已记录在奖励文档 §2.5 |

## 卡 3:MiniMax M2(MiniMax,arXiv 2605.26494 + M2.1 第一方博客,2026-05)

| 维度 | 内容 |
|---|---|
| 基本盘 | 通用 agent 模型,M2.1 在 SWE/工具使用第一梯队;开源权重,框架 Forge 不开源 |
| 奖励 | **复合每步奖励** r = α·过程(语言混杂罚+**工具格式错误罚**+良构推理奖)+β·速度(墙钟比值单调递减塑形——激励并行工具调用)+性能;reward-to-go+轨迹基线;α/β 未公开 |
| 任务构造 | F2P/P2P 按**任务类型变体**:修 bug=F2P+P2P;加功能=提取新增测试点;性能优化=提取能验证性能差的 P2P;代码评审=LLM 一致性近似可验证 |
| 算法 | **CISPO**(截 IS 权重不截 token——修 PPO/DAPO 把 'wait' 类词永久剪掉梯度);**MIS+PPO 轨迹过滤**治理环境噪声长尾;FP32 LM head(训练-推理一致性) |
| 工程 | 多 scaffold 拒绝采样(单 scaffold 泛化差);Forge=Gateway+DataPool 中间层解耦 agent 与训练/推理 |
| 启示(我们) | CISPO 与 DAPO 动态采样同在 Meta ScaleRL 获胜配方——slime 原生都有;格式罚进过程奖励的正方 TR;环境噪声的第三种治理机制(MIS+过滤) |

## 卡 4:OpenThoughts-Agent(open-thoughts 社区,arXiv 2606.24855+博客,2026-06)

| 维度 | 内容 |
|---|---|
| 基本盘 | 全开源配方(数据/管线/模型);Qwen3-32B 微调,7 个 agent 基准平均 **44.8%**(超 Nemotron-Terminal-32B 的 40.9%);100+ 受控消融 |
| SFT 配方 | 轨迹数据(NL2Bash+InferredBugs);**教师选择>一切**:GLM-4.6 当 teacher 比 GPT 系好 ~2 倍(teacher 的 bench 分不预测数据质量);论文终版 100K 样本 |
| RL 配方 | 从 1 万生成任务**只留 ~700**(GPT-5-Codex 零分任务全丢);奖励=pytest 验证(Harbor 式);栈=SkyRL+Harbor;**RL 增益温和**(TB-Dev +2%、SV +1%)——价值主要在 SFT 数据配方 |
| 启示(我们) | 「任务先筛再用」的独立印证;teacher 质量对蒸馏/数据的关键性(阶段③ 27B 的选型逻辑);社区级小团队工作流参照(我们的项目体量同量级) |

## 卡 5:AgenticQwen(arXiv 2604.21590,2026-04)

| 维度 | 内容 |
|---|---|
| 基本盘 | **小模型 agentic RL 代表**(8B/30B-A3B):8B 的 TAU-2 类基准翻倍(23.8→47.4),30B-A3B 逼近 235B |
| 奖励 | reasoning RL=二值 outcome;agentic RL=**rubric[0,1] 子目标部分分**(235B judge 逐子目标核验) |
| 数据 | 双飞轮:失败自扩展(3×一致性过滤)+行为树扩展(条件分支/对抗性 mock 用户导向陷阱分支);**全模拟环境**(用户与工具均 LLM 模拟,免真实 API) |
| 启示(我们) | 小模型+judge 部分分路线存在且有效,但依赖 235B 级 judge 与模拟环境——我们走真实环境+规则验证器,不采;「多轮数据飞轮」思想可供后续扩池参考 |

## 卡 6:LEGO-RL(arXiv 2608.17393,2026-08,方法学+开源框架)

| 维度 | 内容 |
|---|---|
| 定位 | **harness-native RL**(与我们最同源):未修改 harness(OpenHands/Claude Code/OpenCode)内训练 Qwen3.5-35B-A3B |
| 任务漏斗 | 36,884 OpenSWE→静态规则→构建/验证器→**27B rollout 难度筛选(留 4 试解 1-3 次)**→2,699 题;**未筛选任务 72.7% 从未被解出** |
| 奖励 | 单值二值(明示无中间信用是局限);infra 失败权重 0 但留批内;GSPO+loss 内 KL 1e-3 |
| 保真 | rollout-训练 Pearson 中位 **≥0.998**(模型 API 处 proxy+消息级历史对齐) |
| 结果 | 各 harness SV +5.8~9.4;**增益跨 harness 不保真**(KAT 在 OpenHands 下 −0.4)——harness 内训练正当性 |
| 工程 | agent 执行占 91.3% 墙钟;async 2.5×;预构建镜像 33× |
| 启示(我们) | 难度筛选协议的直接模板(我们照此执行);gspo 先例;单 harness(DSH)训练的正当性证据 |

## 横向对比

### 奖励设计

| TR | 核心 | 过程塑形 | 部分分 | 学习型评估器 |
|---|---|---|---|---|
| KAT-V2.5 | 二值 | 轨迹级行为罚(7 项) | ✅ 失败轨迹(F2/单测子集) | GRM(rubric) |
| DSV4 | 领域奖励 | 按模式长度罚 | — | GRM(仅难验证任务) |
| MiniMax M2 | 测试通过 | **步级过程奖罚**+速度奖 | — | — |
| OpenThoughts | pytest 二值 | — | — | — |
| AgenticQwen | 二值/子目标 | — | ✅(judge rubric) | 235B judge |
| LEGO-RL | 二值 | — | —(明示局限) | — |

### 环境与任务工程(各家投入最大的共同点)

| TR | 任务构造 | 环境可靠性措施 |
|---|---|---|
| KAT-V2.5 | PR 挖掘+AutoBuilder 去污 | 沙盒故障 16%→<2%(崩溃÷10) |
| DSV4 | 序列长度课程 4K→1M | DSec 轨迹日志+WAL 抢占恢复 |
| MiniMax M2 | 10 万+可运行 PR、按类型提测试 | MIS+轨迹过滤滤长尾 |
| OpenThoughts | 合成+强模型零分过滤 | Daytona 沙箱 |
| LEGO-RL | 27B 难度筛选(72.7% 淘汰) | 预构建镜像 33×/Nydus 懒加载 |

### 算法选择

| TR | 算法 | 一句话理由 |
|---|---|---|
| KAT-V2.5 | PPO+GAE+非对称 critic | 要逐步负反馈通道 |
| DSV4 | GRPO(专家期)→OPD(整合期) | 密集监督交给蒸馏 |
| MiniMax M2 | CISPO | 保全 token 梯度+截权重 |
| LEGO-RL | GSPO | 序列级比率适配长轨迹 |
| AgenticQwen/OpenThoughts | GRPO | 通用默认 |

## 共识与分歧(PPT 核心页)

**共识(全部 6 家)**:①可验证结果信号是核心;②环境/任务工程投入大于奖励技巧;③
token 级保真(网关/proxy)是基础设施标配;④难度/质量筛选决定有效样本率。

**分歧**:①部分分:KAT 正方 vs LEGO-RL 明示局限+受控实验反方(2605.02944)——我们 v1
不采用、留重访;②算法:GRPO 家族 vs PPO+critic vs CISPO——我们在 slime 原生三选
(grpo/cispo/gspo),零自写;③过程塑形密度:MiniMax 步级奖罚 vs 其余只有罚/没有——
我们采"只罚不奖"(4/6 家形态)。

## 对我们项目的映射(讲解逻辑线)

1. 我们的管线 = 这 6 份报告的交集路线:真实环境+可验证二值核心+难度筛选+token 保真
   (slime_dsh)+噪声治理;差异化 = 小算力(2-4 卡)+DSH harness(发表记录为空)+OPD
   整合(DSV4 同族)。
2. 每个设计决策的出处:奖励=[奖励设计文档](reward-design-stage2-grpo.md);难度筛选=
   LEGO-RL;噪声治理=GLM-5+KAT 实证;算法=三选项全原生;阶段③=DSV4。
3. PPT 素材索引:卡 1-6 → 每家一页;横向对比三表 → 三页;共识分歧 → 一页;映射 → 一页。
