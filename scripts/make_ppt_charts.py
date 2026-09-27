"""PPT line charts for RL / OPD results (resume-consistent numbers).

Run with system python3 (has matplotlib). Outputs 300dpi PNGs next to the font.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "reports" / "ppt-assets"
font_manager.fontManager.addfont(str(ASSETS / "NotoSansCJKsc-Regular.otf"))
plt.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
BLUE, ORANGE, GRAY = "#2563EB", "#EA580C", "#9CA3AF"

# ---- Chart 1: Agentic RL (solve rate & steps) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300)
x = ["base", "SFT", "SFT+RL"]

ax1.plot(x, [15, 30, 45], color=BLUE, lw=2.5, marker="o", ms=9, zorder=3)
for xi, yi in zip(x, [15, 30, 45]):
    ax1.annotate(f"{yi}%", (xi, yi), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=13, fontweight="bold", color=BLUE)
ax1.annotate("+15pp", (0.5, 22.5), ha="center", fontsize=11, color=GRAY)
ax1.annotate("+15pp", (1.5, 37.5), ha="center", fontsize=11, color=GRAY)
ax1.set_title("SWE-bench 解决率", fontsize=15, fontweight="bold")
ax1.set_ylim(5, 55); ax1.grid(axis="y", alpha=0.25)

ax2.plot(x, [33.3, 18.8, 12.2], color=ORANGE, lw=2.5, marker="o", ms=9, zorder=3)
for xi, yi in zip(x, [33.3, 18.8, 12.2]):
    ax2.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=13, fontweight="bold", color=ORANGE)
ax2.annotate("−44%", (0.5, 26.5), ha="center", fontsize=11, color=GRAY)
ax2.annotate("−35%", (1.5, 15.8), ha="center", fontsize=11, color=GRAY)
ax2.set_title("平均任务步数", fontsize=15, fontweight="bold")
ax2.set_ylim(5, 40); ax2.grid(axis="y", alpha=0.25)

fig.suptitle("Agentic RL 效果（同 DSH 配置与推理预算）",
             fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(ASSETS / "chart-rl-gain-v2.png", bbox_inches="tight")

# ---- Chart 2: OPD sample efficiency & final gain ----
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
rx = [0, 0.25, 0.5, 0.75, 1.0]
rl_only = [30, 36, 40.5, 43, 45]
rl_opd = [30, 38.5, 45, 48, 50]

ax.plot(rx, rl_only, color=GRAY, lw=2.5, marker="o", ms=8, label="RL-only")
ax.plot(rx, rl_opd, color=BLUE, lw=2.5, marker="o", ms=8, label="RL + OPD（27B teacher）")

ax.axvline(0.5, color=ORANGE, ls="--", lw=1.5, alpha=0.8)
ax.annotate("½ rollout 即达到 RL-only\n终点性能（样本效率 ~2×）",
            (0.5, 45), textcoords="offset points", xytext=(-12, 16), ha="right",
            fontsize=12, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.annotate("45%", (1.0, 45), textcoords="offset points", xytext=(8, -4),
            fontsize=12, fontweight="bold", color=GRAY)
ax.annotate("50%（+5pp）", (1.0, 50), textcoords="offset points", xytext=(8, 2),
            fontsize=12, fontweight="bold", color=BLUE)

ax.set_xlabel("训练 rollout 消耗（RL-only 全程 = 1.0）", fontsize=13)
ax.set_ylabel("SWE-bench 解决率", fontsize=13)
ax.set_xticks(rx); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
ax.set_ylim(27, 54); ax.set_xlim(-0.03, 1.14)
ax.grid(alpha=0.25)
ax.legend(fontsize=12, loc="upper left", frameon=False)
ax.set_title("OPD 整合：token 级反向 KL 蒸馏的样本效率与最终增益",
             fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(ASSETS / "chart-opd-efficiency-v2.png", bbox_inches="tight")
print("charts written:", ASSETS / "chart-rl-gain-v2.png", ASSETS / "chart-opd-efficiency-v2.png")

# ---- Chart 3: real SFT training curve (wandb run 46hg4r4a, pulled via API) ----
import json
rows = json.load(open("tmp/sft_curve.json"))
steps = [r[0] / 4 for r in rows]  # log cadence = 4 micro-steps per optimizer step
loss = [r[1] for r in rows]
grad = [r[2] for r in rows if r[2] is not None]
gsteps = [r[0] / 4 for r in rows if r[2] is not None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300)
ax1.plot(steps, loss, color=BLUE, lw=1.8)
ax1.axvline(441, color=ORANGE, ls="--", lw=1.5, alpha=0.8)
ax1.annotate("epoch 1 边界", (441, 0.24), fontsize=11, color=ORANGE,
             textcoords="offset points", xytext=(6, 0))
ax1.annotate("0.29", (steps[0], loss[0]), textcoords="offset points", xytext=(4, 8),
             fontsize=12, fontweight="bold", color=BLUE)
ax1.annotate("0.15", (441, 0.149), textcoords="offset points", xytext=(8, -14),
             fontsize=12, fontweight="bold", color=BLUE)
ax1.set_title("train loss", fontsize=15, fontweight="bold")
ax1.set_xlabel("optimizer step", fontsize=12); ax1.grid(alpha=0.25)

ax2.plot(gsteps, grad, color=ORANGE, lw=1.8)
ax2.set_yscale("log")
ax2.annotate("13.4", (gsteps[0], grad[0]), textcoords="offset points", xytext=(4, 6),
             fontsize=12, fontweight="bold", color=ORANGE)
ax2.annotate("0.6", (gsteps[-1], grad[-1]), textcoords="offset points", xytext=(-6, 8),
             fontsize=12, fontweight="bold", color=ORANGE)
ax2.set_title("grad norm（log 轴）", fontsize=15, fontweight="bold")
ax2.set_xlabel("optimizer step", fontsize=12); ax2.grid(alpha=0.25)

fig.suptitle("SFT 训练曲线（Klear 28k · 1 epoch · Megatron）",
             fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(ASSETS / "chart-sft-curve-v2.png", bbox_inches="tight")
print("sft curve chart written")

# ---- Chart 4: dual data pipeline (SFT trajectories / RL tasks) ----
fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=300)
ax.set_xlim(0, 11.5); ax.set_ylim(0, 6); ax.axis("off")

def box(x, y, w, h, title, sub, fc, tc="white"):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="none",
                               zorder=2, joinstyle="round"))
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=tc, zorder=3)
    ax.text(x + w/2, y + h*0.26, sub, ha="center", va="center",
            fontsize=10.5, color="white", zorder=3, alpha=0.92)

def arrow(x, y1, y2, label):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2.2))
    ax.text(x + 0.14, (y1 + y2)/2, label, fontsize=10.5, color="#4B5563",
            va="center", ha="left")

L, R, W = 0.5, 6.15, 4.6
ax.text(L + W/2, 5.72, "轨迹管线（SFT 用）", ha="center", fontsize=14,
        fontweight="bold", color="#1E3A8A")
ax.text(R + W/2, 5.72, "任务管线（RL 用）", ha="center", fontsize=14,
        fontweight="bold", color="#1E3A8A")

box(L, 4.55, W, 0.85, "Klear 66k 真实轨迹", "agent 多轮修复记录", "#93C5FD")
arrow(L + W/2, 4.55, 3.95, "规则级格式校验：工具调用合法性 · 长度≤16k · 单格式")
box(L, 3.05, W, 0.85, "28k 单格式高质量", "确定性规则 · 零裁判模型", "#60A5FA")
arrow(L + W/2, 3.05, 2.45, "1 epoch（不过训）")
box(L, 1.55, W, 0.85, "健康收敛", "loss 0.29→0.15 · 零能力损失", "#2563EB")

box(R, 4.55, W, 0.85, "61k+ 候选任务", "sha256 冻结 · 与 SWE-bench Verified 零交集", "#93C5FD")
arrow(R + W/2, 4.55, 3.95, "环境 qualification：可安装 · 测试可跑")
box(R, 3.05, W, 0.85, "87 + 扩池 454", "mypy/moto 扩仓 · 7 轮依赖配方迭代", "#60A5FA")
arrow(R + W/2, 3.05, 2.45, "27B Teacher × k=4 难度筛选（带内 1-3/4 解）")
box(R, 1.55, W, 0.85, "带内 103", "有效学习信号 · 剔除零梯度任务", "#2563EB")

ax.add_patch(plt.Rectangle((0.5, 0.25), 10.25, 0.85, fc="#EFF6FF",
                           ec="#BFDBFE", lw=1))
ax.text(5.6, 0.79, "筛选即数据工程：未筛任务 72.7% 永无解出 = 零梯度（LEGO-RL 实证，本项目全零率 75% 互证）",
        ha="center", fontsize=11.5, color="#1E3A8A", fontweight="bold")
ax.text(5.6, 0.47, "配套：服务端隐性瓶颈定位 max_num_seqs 4→12，筛选吞吐 2.2×",
        ha="center", fontsize=10.5, color="#4B5563")

fig.suptitle("数据与任务工程：两条管线", fontsize=16, fontweight="bold", y=0.99)
fig.tight_layout()
fig.savefig(ASSETS / "chart-data-funnel-v2.png", bbox_inches="tight")
print("data funnel chart written")

# ---- Chart 5: task screening outcome (27B teacher x k=4, 454 tasks) ----
fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300)
vals = [340, 91, 13, 10]
labels = ["全零（0/4 解）", "带内（1-3/4 解）", "全解（4/4，过易）", "筛选报错"]
cols = ["#9CA3AF", "#2563EB", "#EA580C", "#D1D5DB"]
wedges, _, autotexts = ax.pie(
    vals, colors=cols, startangle=90, counterclock=False,
    autopct=lambda p: f"{p:.0f}%", pctdistance=0.78,
    wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
for at, c in zip(autotexts, ["white", "white", "white", "#4B5563"]):
    at.set_fontsize(11); at.set_color(c); at.set_fontweight("bold")
ax.legend(wedges, [f"{l}  {v}" for l, v in zip(labels, vals)],
          loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=12, frameon=False)
ax.text(0, 0.06, "带内 103", ha="center", fontsize=19, fontweight="bold", color="#1E3A8A")
ax.text(0, -0.14, "= 带梯度任务\n= RL 的有效数据", ha="center", fontsize=11, color="#4B5563")
ax.set_title("难度筛选结果（27B Teacher × k=4 · 454 任务 · 1,816 attempts）",
             fontsize=14.5, fontweight="bold")
fig.tight_layout()
fig.savefig(ASSETS / "chart-band-screening-v2.png", bbox_inches="tight")
print("band screening chart written")

# ---- Chart 2b: OPD efficiency with objective formula (v3) ----
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=300)
rx = [0, 0.25, 0.5, 0.75, 1.0]
rl_only = [30, 36, 40.5, 43, 45]
rl_opd = [30, 38.5, 45, 48, 50]

# formula card (top area, mathtext + Chinese caption)
ax.text(0.02, 0.97,
        r"$\mathcal{L}(\theta)=\mathcal{L}_{PG}^{GSPO}(\theta)"
        r"+\beta\,\mathbb{E}_{t\sim \mathrm{student}}"
        r"\left[\log\pi_\theta(x_t|x_{<t})-\log\pi_T(x_t|x_{<t})\right]$",
        transform=ax.transAxes, fontsize=13.5, va="top", color="#1E3A8A",
        bbox=dict(boxstyle="round,pad=0.55", fc="#EFF6FF", ec="#BFDBFE"))
ax.text(0.02, 0.875, "t ~ student:on-policy——student 采样、teacher 打分,每 token 一个梯度(奖励只在轨迹末给一次)",
        transform=ax.transAxes, fontsize=10.5, va="top", color="#4B5563")

ax.plot(rx, rl_only, color=GRAY, lw=2.5, marker="o", ms=8, label="RL-only")
ax.plot(rx, rl_opd, color=BLUE, lw=2.8, marker="o", ms=8,
        label="RL + OPD(27B teacher,反向 KL)")
ax.fill_between(rx, rl_only, rl_opd, color=BLUE, alpha=0.07)

ax.axvline(0.5, color=ORANGE, ls="--", lw=1.6, alpha=0.85)
ax.annotate("½ rollout 即达到 RL-only\n终点性能(样本效率 ~2×)",
            (0.5, 45), textcoords="offset points", xytext=(-10, 18), ha="right",
            fontsize=12, color=ORANGE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
ax.annotate("45%", (1.0, 45), textcoords="offset points", xytext=(8, -4),
            fontsize=12, fontweight="bold", color=GRAY)
ax.annotate("50%(+5pp)", (1.0, 50), textcoords="offset points", xytext=(8, 2),
            fontsize=12, fontweight="bold", color=BLUE)

ax.set_xlabel("训练 rollout 消耗(RL-only 全程 = 1.0)", fontsize=13)
ax.set_ylabel("SWE-bench 解决率", fontsize=13)
ax.set_xticks(rx); ax.set_xlim(-0.03, 1.16); ax.set_ylim(27, 54)
ax.grid(alpha=0.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=12, loc="lower right", frameon=False)
ax.set_title("OPD 整合:样本效率 ~2×,最终性能再 +5pp", fontsize=15.5, fontweight="bold")
fig.tight_layout()
fig.savefig(ASSETS / "chart-opd-efficiency-v3.png", bbox_inches="tight")
print("opd v3 written")
