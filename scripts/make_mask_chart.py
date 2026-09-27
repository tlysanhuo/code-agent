"""Gradient-coverage before/after chart (PPT). Both bars are REAL samples:

- before: attempt-6 dump — 32,768-token trajectory, loss_mask covers the
  final 151 tokens only (the incident-#5 pathology);
- after: attempt-17 dump — 51-turn native-stitched trajectory, 6,233 of
  29,727 tokens trainable, real run-length structure.

Mask arrays are pre-exported to tmp/mask_data.json by the train venv (torch);
this script only needs matplotlib, so it runs on system python3.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "reports" / "ppt-assets"
font_manager.fontManager.addfont(str(ASSETS / "NotoSansCJKsc-Regular.otf"))
plt.rcParams.update({
    "font.family": "Noto Sans CJK SC", "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
BLUE, ORANGE, GRAY, DARK = "#2563EB", "#EA580C", "#D1D5DB", "#4B5563"

D = json.load(open("tmp/mask_data.json"))
b_tot, b_resp = D["b_tot"], D["b_resp"]
a_tot, a_resp, a_mask = D["a_tot"], D["a_resp"], D["a_mask"]

fig, ax = plt.subplots(figsize=(11, 4.4), dpi=300)
ax.set_xlim(-3400, 37500)
ax.set_ylim(0, 3.6)
ax.axis("off")

# ---- BEFORE bar: true scale, lit tail magnified by the annotation ----
y0, h = 2.3, 0.5
ax.broken_barh([(0, b_tot - b_resp)], (y0, h), facecolors=GRAY)
ax.broken_barh([(b_tot - b_resp, 900)], (y0, h), facecolors=ORANGE)  # tail widened to 900 for visibility
ax.text(-700, y0 + h / 2, "修复前", fontsize=14, fontweight="bold", color=DARK, ha="right", va="center")
ax.text(b_tot / 2, y0 - 0.32,
        f"轨迹全长 {b_tot:,} tokens,历史与工具返回全在 prompt(mask = 0)",
        fontsize=10.5, color=DARK, ha="center")
ax.annotate(f"梯度仅覆盖末轮 {b_resp} token(0.5%)\n模型在学「写总结」",
            (b_tot, y0 + h), xytext=(27200, y0 + h + 0.72), fontsize=11.5,
            color=ORANGE, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))

# ---- AFTER bar: real run-length structure of the stitched mask ----
y1 = 0.62
ax.broken_barh([(0, a_tot - a_resp)], (y1, h), facecolors="#E5E7EB")
x = a_tot - a_resp
for start, width, lit in D["a_runs"]:
    ax.broken_barh([(x + start, width)], (y1, h), facecolors=BLUE if lit else GRAY)
ax.text(-700, y1 + h / 2, "修复后", fontsize=14, fontweight="bold", color=DARK, ha="right", va="center")
ax.text(a_tot / 2, y1 - 0.32,
        f"51 轮 assistant 全部进梯度:可训练 {a_mask:,} / {a_tot:,} tokens(其余为工具返回,正确不进 loss)",
        fontsize=10.5, color=DARK, ha="center")
ax.annotate("在学「解题」", (a_tot * 0.6, y1 + h / 2), xytext=(33800, y1 + h / 2),
            fontsize=11.5, color=BLUE, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))

ax.set_title("梯度覆盖:修复前后对比(两根条均为真实 rollout 样本的 loss_mask)",
             fontsize=15, fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig(ASSETS / "chart-mask-coverage-v2.png", bbox_inches="tight")
print("mask chart written")
