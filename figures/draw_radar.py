import os
from math import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, "radar_scores.csv"))

categories = df["Dimension"].tolist()
N = len(categories)

strategies = {
    "Early Fusion": df["Early_Fusion"].values,
    "Intermediate Fusion": df["Intermediate_Fusion"].values,
    "Late Fusion": df["Late_Fusion"].values,
    "Attention-Based": df["Attention_Fusion"].values,
    "Adaptive Fusion": df["Adaptive_Fusion"].values,
}

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

colors = {
    "Early Fusion": "#2196F3",
    "Intermediate Fusion": "#4CAF50",
    "Late Fusion": "#FF9800",
    "Attention-Based": "#9C27B0",
    "Adaptive Fusion": "#F44336",
}

markers = {
    "Early Fusion": "o",
    "Intermediate Fusion": "s",
    "Late Fusion": "^",
    "Attention-Based": "D",
    "Adaptive Fusion": "p",
}

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
ax.set_facecolor("#FAFAFA")
fig.patch.set_facecolor("white")

ax.set_rlabel_position(30)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=9, color="gray")
ax.set_ylim(0, 5.5)

for name, values in strategies.items():
    vals = values.tolist() + values[:1].tolist()
    ax.plot(angles, vals, "-", linewidth=2.2, label=name,
            color=colors[name], marker=markers[name], markersize=8)
    ax.fill(angles, vals, alpha=0.08, color=colors[name])

wrapped_labels = [
    "Cross-Modal\nInteraction",
    "Missing Modality\nRobustness",
    "Computational\nEfficiency",
    "Interpretability",
    "Implementation\nSimplicity",
]
ax.set_xticks(angles[:-1])
ax.set_xticklabels(wrapped_labels, fontsize=11, fontweight="bold")

ax.set_title(
    "Comparative Radar Chart of Five Fusion Strategies\n"
    "(Scored 1-5 based on evidence from 36 reviewed studies)",
    fontsize=13, fontweight="bold", pad=25, y=1.08,
)

ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
          fontsize=10, frameon=True, fancybox=True, shadow=True)
ax.grid(color="lightgray", linewidth=0.5, linestyle="--")

plt.tight_layout()

out_pdf = os.path.join(base_dir, "fig_radar_comparison.pdf")
out_png = os.path.join(base_dir, "fig_radar_comparison.png")
fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
