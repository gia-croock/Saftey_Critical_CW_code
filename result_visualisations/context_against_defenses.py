"""
Generate ICML-style horizontal bar chart comparing remaking strategies
against jailbreak attacks (ASR_e metric, lower is better).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────
strategies = [
    "Greedy",
    "Stochastic Annealing",
    "Random",
    "Budget-Constrained Random",
    "SPD (Hybrid AR/MD)",
]
asr_values = [75.74, 70.21, 62.98, 52.34, 48.51]
# Mark which are novel contributions
is_ours = [False, False, False, True, True]

# ── ICML-compatible style ────────────────────────────────────────────
# Use LaTeX if available, otherwise fall back gracefully
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.0,
        "xtick.direction": "out",
        "figure.dpi": 300,
    }
)

# Uncomment the following lines if you have LaTeX installed for best results:
# plt.rcParams.update({
#     "text.usetex": True,
#     "text.latex.preamble": r"\usepackage{times}",
# })
USE_TEX = False

# ── Colors ───────────────────────────────────────────────────────────
COLOR_OURS = "#C05621"  # rust orange — our methods
COLOR_BASELINE = "#2B6693"  # steel blue — baselines
COLOR_DEFAULT = "#2B6693"  # steel blue — no-defense default

# ── Figure (ICML single-column width ≈ 3.25 in) ─────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.5))

y_pos = np.arange(len(strategies))
bar_colors = []
for i, ours in enumerate(is_ours):
    if ours:
        bar_colors.append(COLOR_OURS)
    elif strategies[i] == "Default (Greedy)":
        bar_colors.append(COLOR_DEFAULT)
    else:
        bar_colors.append(COLOR_BASELINE)

bars = ax.barh(
    y_pos,
    asr_values,
    height=0.58,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.3,
    zorder=3,
)

# ── Value labels ─────────────────────────────────────────────────────
for bar, val in zip(bars, asr_values):
    ax.text(
        val + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}",
        va="center",
        ha="left",
        fontsize=8.5,
        color="#334155",
    )

# ── Axes ─────────────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels(strategies, fontweight="bold")
ax.set_xlabel(
    r"ASR$_e$ (\%) $\downarrow$" if USE_TEX else r"ASR$_e$ (%) $\downarrow$",
    labelpad=4,
)
ax.set_xlim(0, 82)
ax.invert_yaxis()

# Light gridlines on x only
ax.set_axisbelow(True)
ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.65, color="#94A3B8")
ax.yaxis.grid(False)

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

# ── Legend ────────────────────────────────────────────────────────────
legend_ours = mpatches.Patch(facecolor=COLOR_OURS, edgecolor="none", label="Ours")
legend_base = mpatches.Patch(
    facecolor=COLOR_BASELINE, edgecolor="none", label="Baseline"
)
ax.legend(
    handles=[legend_ours, legend_base],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=False,
    fontsize=7.5,
    handlelength=1.2,
    handletextpad=0.4,
    columnspacing=1.5,
)

# ── Save ─────────────────────────────────────────────────────────────
fig.tight_layout(pad=0.4)
fig.savefig("fig_remaking_strategies.pdf", bbox_inches="tight", dpi=300)
fig.savefig("fig_remaking_strategies.png", bbox_inches="tight", dpi=300)
print("Saved: fig_remaking_strategies.pdf / .png")
