# OPD 整合机制(示意)

> 本页为阶段③机制说明示意(非实验数据)。

## 机制

- **student 采样,teacher 打分**:RL rollout 的每一步仍由 9B student 生成;27B teacher
  对 student 采样出的 token 序列逐 token 给出 logprob,以反向 KL
  `KL(student ‖ teacher)` 作为附加训练信号叠加在 RL 奖励之上。

- **信号密度对比**:

| 信号 | 密度 | 到达时机 |
|---|---|---|
| 可验证奖励(测试判分) | 稀疏:每条轨迹一次(0/1) | 轨迹结束 |
| teacher 反向 KL | 稠密:每个 token 一个 | 每步即时 |

- **为什么是反向 KL 而不是正向**:优化 `student ‖ teacher` 允许 student 在 teacher
  高概率区域外探索(保持 RL 的探索能力),而正向 `teacher ‖ student` 会把分布硬拉向
  teacher、抹掉 RL 学到的策略差异。

## 与管线的分工

- 正向密集监督 → 归 teacher KL(dense)
- 行为对错判定 → 归可验证奖励(sparse but grounded)
- 两者叠加:奖励防"学偏",蒸馏提"样本效率"(同性能所需 rollout 减半)

## 工程要点

- teacher 与 student 词表严格对齐(248,320),token 级打分才能成立
- teacher 外置 SGLang 服务(不占训练卡),`--use-opd --opd-type sglang` 官方结构
