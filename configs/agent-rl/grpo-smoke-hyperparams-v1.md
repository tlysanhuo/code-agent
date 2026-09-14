# GRPO smoke 超参单 v1(待呈批;批准前不进任何启动配置)

> 状态:**草案待用户批准**(2026-09-13 起草)。依据:锁定设计 research/reward-design-stage2-grpo.md
> §5 决策点 1-4;数值锚点=slime 官方 SWE 配方(vendor/slime/examples/coding_agent_rl/
> run_qwen36_35b_a3b_swe_8nodes.sh,8 节点 64 卡)+第九轮 RL 用量实证(DeepSWE/LEGO-RL/
> NeMo-RL)。所有旗标名已对照钉住 slime 源码逐一核实。训练启动=4 卡已批但必须用户协调(D16)。
> 本单只呈参数,不启动任何东西。

## 0. 已定项(不在本单讨论范围)

- student=models/dense-9B-sft/formal-2card/hf-iter500(阶段① 验收检查点);
- rollout/格式=DSH(slime_dsh 接线,D9);环境=本地受限进程+冻结 gym 评测器(D7,自建协议标注);
- 任务池=难度筛选后任务带(等 round1b 收尾+27B 筛选,挂起①);curated 带内 12 任务够 smoke;
- 奖励核=二值(金标 F2P+P2P),塑形分级起步(D11f);WANDB online 不许改;
- 并行拓扑=colocate+optimizer-cpu-offload(官方 SWE 配方同款,SFT 期已在同模型验证过 2 卡版)。

## 1. estimator 三选(决策点 3;D13 倾向 gspo)

| 选项 | 出处背书 | 备注 |
|---|---|---|
| grpo(官方默认) | 五家 TR 默认起点;官方 SWE 配方即 grpo | 最保守,与官方配方零差异 |
| cispo | MiniMax M2 生产+Meta ScaleRL 获胜配方成分 | 截断重要性采样权重;长轨迹 token 保护 |
| **gspo(推荐)** | Qwen 自家(TR 系);LEGO-RL 同款先例(序列级比率,适配长 agent 轨迹) | student 同族+最同源先例;不赌死,smoke 可 A/B |

slime 旗标:`--advantage-estimator grpo|gspo|cispo`(原生,arguments.py:954 choices 核实)。

## 2. 训练核心超参(官方 SWE 配方锚点,4 卡缩放)

| 参数 | 提议值 | 锚点/依据 |
|---|---|---|
| lr | 1e-6,constant | 官方配方+NeMo-RL SWE2 完全一致(恒定 1e-6) |
| eps-clip / eps-clip-high | 0.2 / 0.28 | 官方配方+GLM-5+NeMo-RL 三处一致(非对称裁剪) |
| kl-loss-coef / kl-coef / entropy-coef | 0 / 0 / 0 | 官方配方+GLM-5「去掉 KL 正则」 |
| weight-decay / adam β1,β2 | 0.1 / 0.9,0.98 | 官方配方 |
| rollout-temperature | 1.0 | 官方配方 |
| n-samples-per-prompt(组大小 K) | **8** | NeMo-RL G=8、LEGO-RL G=8、DeepSWE 8(三家一致) |
| rollout-batch-size | **8**(smoke 可降 4) | 官方 8;每步 8×8=64 rollout(NeMo-RL 同量级) |
| global-batch-size | 64(K=8 时) | 官方 64;含义=每训练步 rollout 数 |
| num-rollout(smoke 总步数) | **20 起,上限 50** | 第九轮:业界 126-200 步档;smoke 先验证可行性 |
| rollout-max-context-len / response | 32768 / 8192 | DeepSWE:32k 外增益 ~2%(预算现实上限);SFT 16k 同族 |
| micro-batch-size / 动态批 | 1 / use-dynamic-batch-size | 官方配方(长轨迹防 OOM) |
| 并行 | TP2×DP2(4 卡)+colocate+cpu-offload | SFT 期 2 卡验证过 TP2;4 卡可行性须 smoke 实测(已知风险:无官方 2-4 卡配方) |
| DSH 每轮 max_tokens | 4096(现有部署上限,harness.py 不变) | DSH RL 部署契约 |
| agent 时间预算 / 评测超时 | 1800s / 600s | 官方配方;上一轮筛选实测墙钟 1500s+缓冲全链路可用 |

## 3. 奖励栈开关(本轮实现的 slime_dsh;决策点 1/2 的「先测量后加码」)

**v1 smoke 推荐(=锁定设计的分级起步)**:

| 开关 | v1 值 | 依据 |
|---|---|---|
| DSH_C_UNFINISHED | **0**(关) | 纯二值起步(D11f 分级方案;三家共同基线) |
| DSH_C_FMT | **0**(关) | 同上;格式罚为紧随消融臂 |
| DSH_GROUP_REPAIR | 0(关) | D11d:v1 用原生 DAPO+上游 abort 通道;**加装条件=实测崩溃率** |
| DSH_BLOCKER | **1(开)** | 执行层防作弊(非奖励超参;本地无外网已是最权威层,此为纵深第二层) |
| DSH_TOOL_NAMES | 不设 | 格式罚启用时再配(白名单校验可选) |

**v1 必须接线的一个钩子(推荐,需用户点头)**:
`--custom-reward-post-process-path slime_dsh.group_repair.reward_post_process`
理由:实现期读源码发现上游默认组归一化的 reshape 依赖样本总数==K×batch,而 DSH 多分段
轨迹恒不满足→**退化为全批全局均值归一化**(决策 D17 裁决 2)。锁定设计 §2.3 明确要求
「二值 outcome → 组内归一化(三家)」——接线此钩子才是设计意图的忠实实现(无崩溃样本时
与 slime 语义等价,含 std 除法一致性单测)。不加=接受全局均值近似。
`--normalize-advantages` 保持默认(源码默认 False;v2 开格式罚时同样必须保持关——
后置白化会吃掉罚项);`--grpo-std-normalization` 保持默认开(Dr.GRPO 除 std;钩子内
含同款一致性单测)。

**消融臂(首轮测量后,决策点 1 的 b/c 档)**:
- DSH_C_UNFINISHED=0.1 起(AGL λ=0.1 长度罚锚点;退场=未完成率降档后关);
- DSH_C_FMT=0.05~0.1 起(无 TR 数值锚点,区间供选;退场=格式错误率连续 N 步<阈值,
  ToolRL ~30 步饱和证据);两者一次只开一个。

## 4. 动态采样(DAPO,决策点 2;零代码原生)

| 参数 | 提议值 | 依据 |
|---|---|---|
| --over-sampling-batch-size | 12(1.5×batch) | DAPO 过采样;ScaleRL 获胜配方成分 |
| --dynamic-sampling-filter-path | `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std_with_fallback` | 官方示例路径(fallback 防采样死循环);零方差组丢弃 |

注(实现陷阱,NeMo-RL issue 教训):若未来启用长度/格式类奖励,过滤必须基于**原始**
奖励——v1 纯二值无此问题。

## 5. 监控与验收

- wandb:online,project code-agent-dense-mainline,group stage2-grpo-smoke(WANDB_DIR 在卷上);
- 首要观测(为消融臂定值):组内奖励方差分布、0/8 全零组占比、abort 分类占比
  (environment/budget)、格式错误 token 率(rollout dump 离线统计)、rollout 墙钟/步;
- smoke 通过标准(建议):≥20 步无 NaN/无 OOM/loss 下降趋势+组方差非退化(具体阈值用户可改);
- 每步 GPU·h 与 rollout 数入账(简历项目成本披露惯例)。

## 6. 待用户拍板清单

1. estimator:grpo / cispo / **gspo(推荐)**;
2. v1 是否接线组归一化钩子(推荐接,理由见 §3);
3. rollout-batch-size smoke 用 8 还是 4;
4. 消融臂系数区间认可(DSH_C_UNFINISHED 0.1 起 / DSH_C_FMT 0.05~0.1 起);
5. 其余 §2 数值默认接受或逐项修改。
