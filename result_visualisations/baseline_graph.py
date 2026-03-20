import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── ICML style config ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": False,           # set True if LaTeX is installed
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
})

# ── Data ───────────────────────────────────────────────────────────────────────
attacks = ["Dija +\nHarmBench", "Context Nesting +\nHarmBench", "Vanilla\nHarmBench"]
asr_e   = [77.87, 75.74,  3.40]
asr_k   = [97.45, 88.51,  6.81]

x      = np.arange(len(attacks))
width  = 0.32

# ── Colors (colorblind-friendly) ───────────────────────────────────────────────
col_e = "#C04828"   # muted red
col_k = "#3B6EA5"   # muted blue

# ── Figure size: ICML single-column = 3.25 in; double = 6.75 in ───────────────
fig, ax = plt.subplots(figsize=(3.25, 2.6))

bars_e = ax.bar(x - width/2, asr_e, width, color=col_e, edgecolor="white",
                linewidth=0.3, zorder=3, label=r"$\mathrm{ASR}_e$")
bars_k = ax.bar(x + width/2, asr_k, width, color=col_k, edgecolor="white",
                linewidth=0.3, zorder=3, label=r"$\mathrm{ASR}_k$")

# Value labels on bars
for bar, val in zip(list(bars_e) + list(bars_k),
                    asr_e + asr_k):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.1f}",
            ha="center", va="bottom",
            fontsize=6.5, color="#222222")

ax.set_xticks(x)
ax.set_xticklabels(attacks, linespacing=1.3)
ax.set_ylabel("Attack Success Rate (%)")
ax.set_ylim(0, 112)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

# Legend inside, upper right, away from tall bars
ax.legend(loc="upper right", frameon=False,
          handlelength=1.2, handletextpad=0.4, borderpad=0)

fig.tight_layout()
fig.savefig("baseline_asr.pdf")
fig.savefig("baseline_asr.png")
print("Saved: baseline_asr.pdf  /  baseline_asr.png")