"""
Two-panel figure: within-session modality hierarchy (a) → cross-session reversal (b).

Panel (a): Single-modality LOSO bars + fusion weight annotations; chance line at 50%.
Panel (b): Ablation ladder (cross-within delta) with SE bars + individual subject
           dots behind the full-model bar. Significance markers above each bar.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = "/Users/pranmodu/Projects/columbia/liinc_eye/data/results/main"
OUT_PNG = "/Users/pranmodu/Projects/columbia/liinc_eye/data/results/main/statistical_analyses/figure_modality_dissociation.png"
OUT_PDF = "/Users/pranmodu/Projects/columbia/liinc_eye/data/results/main/statistical_analyses/figure_modality_dissociation.pdf"

# ── Panel (a) data ─────────────────────────────────────────────────────────
modalities_a  = ["Behavior", "Gaze", "EEG", "Pupillometry"]
accuracies_a  = [71.1, 62.2, 56.6, 51.4]   # LOSO % accuracy
fusion_weights = [85.2, 13.9, 0.2, 0.6]    # % weight in full fusion model

# ── Panel (b) data ─────────────────────────────────────────────────────────
ablation = pd.read_csv(f"{BASE}/cross_visit/modality_ablation_summary.csv")
order = ["behavior_only", "gaze+behavior", "all_features"]
ablation["_ord"] = ablation["modality"].map({k: i for i, k in enumerate(order)})
ablation = ablation.sort_values("_ord").reset_index(drop=True)

deltas_b   = ablation["delta_mean"].values * 100
sems_b     = ablation["delta_sem"].values  * 100
p_values_b = ablation["p"].values
labels_b   = ["Behavior\nonly", "Gaze +\nBehavior", "Full model\n(+ Physio/EEG)"]


# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":          9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype":      42,
})

COLORS_A     = ["#2C5F8A", "#5A9EC7", "#A8C4DA", "#C8DAEC"]
COLORS_B     = ["#A8C4DA", "#5A9EC7", "#2C5F8A"]

# Wider figure; slightly more space for panel b
fig = plt.figure(figsize=(8.2, 3.8))
ax_a = fig.add_axes([0.07, 0.19, 0.38, 0.73])  # left, bottom, width, height
ax_b = fig.add_axes([0.55, 0.19, 0.43, 0.73])

# ═══════════════════════════════════════════════════════════════════════════
# Panel (a) — within-session modality hierarchy
# ═══════════════════════════════════════════════════════════════════════════
ax = ax_a
x = np.arange(len(modalities_a))
bars = ax.bar(x, accuracies_a, width=0.55, color=COLORS_A,
              edgecolor="white", linewidth=0.4, zorder=3)

# Chance line — 50% included as a y-tick, labeled "chance" in gray
ax.axhline(50, color="#888888", linewidth=0.9, linestyle="--", zorder=2)

# Fusion-weight labels above each bar
for bar, w in zip(bars, fusion_weights):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            f"wt: {w:.1f}%",
            ha="center", va="bottom", fontsize=6.3, color="#444444", zorder=5)

# Accuracy labels inside bars
for bar, acc in zip(bars, accuracies_a):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 1.6,
            f"{acc:.1f}%", ha="center", va="top",
            fontsize=7.5, color="white", fontweight="bold", zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(modalities_a, fontsize=8.5)
ax.set_ylabel("LOSO accuracy (%)", fontsize=8.5)
ax.set_ylim(44, 83)
ax.set_xlim(-0.65, 3.55)
ax.set_yticks([50, 60, 70, 80])
ax.set_yticklabels(["50\n(chance)", "60", "70", "80"], fontsize=7.5)
# Color the 50% tick label gray to match the dashed line
ax.get_yticklabels()[0].set_color("#888888")
ax.get_yticklabels()[0].set_fontsize(6.3)
ax.set_title("(a)  Within-session", fontsize=9.5, fontweight="bold",
             loc="left", pad=6)
# Footnote
ax.text(0.5, -0.10,
        '"wt:" = weight in full fusion model',
        transform=ax.transAxes, fontsize=5.8, color="#888888",
        va="top", ha="center", style="italic")

# ═══════════════════════════════════════════════════════════════════════════
# Panel (b) — ablation ladder
# ═══════════════════════════════════════════════════════════════════════════
ax = ax_b
x3 = np.arange(3)

# Zero line — 0 tick labeled "within=cross" in gray
ax.axhline(0, color="#888888", linewidth=0.9, linestyle="--", zorder=1)

# Bars
bars_b = ax.bar(x3, deltas_b, width=0.5, color=COLORS_B,
                edgecolor="white", linewidth=0.4, zorder=3)

# SE error bars
ax.errorbar(x3, deltas_b, yerr=sems_b,
            fmt="none", ecolor="#333333", elinewidth=1.2,
            capsize=3.5, capthick=1.2, zorder=4)

# Significance markers
sig_map  = {True: "**", False: None}
sig_strs = []
for p in p_values_b:
    if   p < 0.01: sig_strs.append("**")
    elif p < 0.05: sig_strs.append("*")
    elif p < 0.10: sig_strs.append("†")
    else:          sig_strs.append("ns")

sig_colors = {"**": "#cc2222", "*": "#cc2222", "†": "#e07700", "ns": "#888888"}
for i, (delta, sem, sig) in enumerate(zip(deltas_b, sems_b, sig_strs)):
    # For full model bar (i==2), push marker above the delta label
    y_top = delta + sem + (1.0 if i == 2 else 0.18)
    fs    = 9.5 if sig in ("**", "*") else 7.5
    fw    = "bold" if sig in ("**", "*") else "normal"
    ax.text(i, y_top, sig, ha="center", va="bottom",
            fontsize=fs, color=sig_colors[sig], fontweight=fw, zorder=6)

# Delta labels — bars 0 & 1 inside bar (white), bar 2 (full model) above bar
# to stay clear of the jitter dots
for i, (bar, delta) in enumerate(zip(bars_b, deltas_b)):
    sign = "+" if delta >= 0 else ""
    if i == 2:                           # full model — place above bar + SE
        y_pos = delta + sems_b[i] + 0.55
        va, fc = "bottom", "#cc2222"
    elif delta > 0.7:
        y_pos, va, fc = delta - 0.12, "top", "white"
    else:
        y_pos, va, fc = delta + 0.13, "bottom", "#333333"
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            f"{sign}{delta:.1f} pp",
            ha="center", va=va, fontsize=7.0, color=fc,
            fontweight="bold", zorder=7)

# p-value callout — points to the ** marker
ax.annotate("p = 0.008\nt(30) = 2.85",
            xy=(2, deltas_b[2] + sems_b[2] + 1.3),
            xytext=(2.38, 4.6),
            fontsize=6.4, color="#cc2222", ha="left", va="center", zorder=8,
            arrowprops=dict(arrowstyle="->", color="#cc2222", lw=0.75))


ax.set_xticks(x3)
ax.set_xticklabels(labels_b, fontsize=8.2, linespacing=1.3)
ax.set_ylabel("Cross − within accuracy (pp)", fontsize=8.5)
ax.set_ylim(-2.2, 5.8)
ax.set_xlim(-0.5, 2.75)
ax.set_yticks([-2, -1, 0, 1, 2, 3, 4, 5])
ax.set_yticklabels(["-2", "-1", "0\n(within\n=cross)", "1", "2", "3", "4", "5"], fontsize=7.5)
ax.get_yticklabels()[2].set_color("#888888")
ax.get_yticklabels()[2].set_fontsize(6.0)
ax.tick_params(labelsize=8)
ax.set_title("(b)  Cross-session generalization", fontsize=9.5,
             fontweight="bold", loc="left", pad=6)
# Significance legend footnote — below x-tick labels
ax.text(0.5, -0.18,
        "ns: p ≥ 0.10;  †: p < 0.10;  **: p < 0.01",
        transform=ax.transAxes, fontsize=5.8, color="#555555",
        va="top", ha="center", style="italic")

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"Saved:\n  {OUT_PNG}\n  {OUT_PDF}")
plt.close()
