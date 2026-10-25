import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, "performance_data.csv"))

selected = [
    ("ORB+InceptionV4",  "Early",        "Acc",         93.4),
    ("LMF Low-Rank",     "Intermediate", "Acc-2",       76.4),
    ("H-DenseFormer",    "Intermediate", "DSC",         73.9),
    ("HAIM",             "Intermediate", "F1",          90.0),
    ("DF-DM",            "Intermediate", "Macro F1",    92.0),
    ("Multi-level FGA",  "Intermediate", "Acc-2",       85.4),
    ("Breast DL",        "Late",         "AUC",         96.8),
    ("MFTN",             "Attention",    "mAP@50",      98.3),
    ("AttnHybrid+SVM",   "Attention",    "Acc",         98.5),
    ("DynMM",            "Adaptive",     "Cost Red.",   46.5),
    ("Hybrid ECG",       "Hybrid",       "Acc",         93.6),
    ("Pathomic Fusion",  "Hybrid",       "C-idx(x100)", 82.6),
]

methods = [s[0] for s in selected]
strategies = [s[1] for s in selected]
metric_labels = [s[2] for s in selected]
values = [s[3] for s in selected]

strategy_colors = {
    "Early": "#2196F3",
    "Intermediate": "#4CAF50",
    "Late": "#FF9800",
    "Attention": "#9C27B0",
    "Adaptive": "#F44336",
    "Hybrid": "#607D8B",
}

bar_colors = [strategy_colors[s] for s in strategies]

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(methods))
bars = ax.barh(y_pos, values, height=0.65, color=bar_colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=10)

for bar, metric in zip(bars, metric_labels):
    if bar.get_width() > 25:
        ax.text(3, bar.get_y() + bar.get_height() / 2,
                f"[{metric}]", va="center", ha="left", fontsize=8,
                color="white", fontstyle="italic")

ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Performance Metric Value (%, C-index x100, or Cost Reduction %)", fontsize=11)
ax.set_xlim(0, 108)
ax.axvline(x=90, color="gray", linestyle=":", alpha=0.4)
ax.axvline(x=80, color="gray", linestyle=":", alpha=0.3)

legend_patches = [mpatches.Patch(color=c, label=s) for s, c in strategy_colors.items()]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
          frameon=True, fancybox=True, shadow=True,
          title="Fusion Strategy", title_fontsize=10)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

ax.text(0.5, -0.09,
        "Note: DynMM (46.5%) = computational cost reduction. C-index values scaled x100.\n"
        "Different metrics used per task. Direct comparison requires caution.",
        transform=ax.transAxes, fontsize=8, ha="center", va="top",
        style="italic", color="gray")

plt.tight_layout()

out_pdf = os.path.join(base_dir, "fig_performance.pdf")
out_png = os.path.join(base_dir, "fig_performance.png")
fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
