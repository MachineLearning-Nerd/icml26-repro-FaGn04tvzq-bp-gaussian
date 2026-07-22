"""Render the four figures used by the reproduction report.

The values below are transcribed from the immutable Hugging Face run log for
the six-claim baseline.  Keeping this script beside the report makes every
visual inspectable without rerunning belief propagation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

CHAIN = np.array([0.6117, 0.1743, 0.0667, 0.0327, 0.0212, 0.0173, 0.0161])
TREE = np.array([0.6117, 0.1743, 0.0667, 0.0327, 0.0212, 0.0173, 0.0161])
LOOPY = np.array([0.5940, 0.1494, 0.0555, 0.0251, 0.0189])
EXCLUSION = np.array([0.3474, 0.2708, 0.2122, 0.1676, 0.1336, 0.1078, 0.0880])
DEGREES = np.array([2, 4, 6])
DEGREE_KL = np.array([0.3941, 0.7203, 0.9234])

COLORS = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "red": "#dc2626",
    "ink": "#172033",
    "muted": "#64748b",
}


def finish(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


plt.style.use("seaborn-v0_8-whitegrid")

# Headline: Gaussianization across the three tested topologies.
fig, ax = plt.subplots(figsize=(9.2, 5.1))
ax.plot(range(len(CHAIN)), CHAIN, "o-", lw=2.6, color=COLORS["blue"], label="Chain")
ax.plot(range(len(TREE)), TREE, "s--", lw=2.1, color=COLORS["orange"], label="Tree (symmetric simplification)")
ax.plot(range(len(LOOPY)), LOOPY, "^-", lw=2.6, color=COLORS["green"], label="Loopy graph")
ax.axhline(0.02, color=COLORS["red"], ls=":", lw=2, label="Paper Gaussian threshold (0.02)")
ax.scatter([3], [CHAIN[3]], s=95, facecolor="white", edgecolor=COLORS["red"], lw=2.2, zorder=6)
ax.annotate("chain hop 3 = 0.0327", (3, CHAIN[3]), xytext=(3.3, 0.105), arrowprops={"arrowstyle": "->", "color": COLORS["muted"]}, color=COLORS["ink"])
ax.set_yscale("log")
ax.set_xlabel("Topological distance / sampled node index")
ax.set_ylabel("KL(belief || best-fit Gaussian), log scale")
ax.set_title("Beliefs move sharply toward Gaussian shape")
ax.legend(frameon=True, ncol=2)
finish(fig, "headline_topologies.png")

# Mechanism control: multiplication at a high-degree star is not Gaussianizing.
fig, ax = plt.subplots(figsize=(8.2, 4.8))
bars = ax.bar(DEGREES, DEGREE_KL, width=1.15, color=["#93c5fd", "#60a5fa", "#2563eb"])
for bar, value in zip(bars, DEGREE_KL):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.4f}", ha="center", fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_xticks(DEGREES)
ax.set_xlabel("Star center degree")
ax.set_ylabel("Center-belief KL")
ax.set_title("More neighbors increased non-Gaussianity without path depth")
finish(fig, "degree_control.png")

# Boundary diagnostic: the alternative prior remains farther from Gaussian.
fig, ax = plt.subplots(figsize=(8.8, 4.9))
ax.plot(range(len(CHAIN)), CHAIN, "o-", lw=2.5, color=COLORS["blue"], label="Normal bimodal prior")
ax.plot(range(len(EXCLUSION)), EXCLUSION, "o-", lw=2.5, color=COLORS["red"], label="Wide bimodal diagnostic")
ax.axhline(0.02, color=COLORS["muted"], ls=":", lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("Hop")
ax.set_ylabel("KL(belief || Gaussian), log scale")
ax.set_title("The alternative prior leaves a thicker non-Gaussian boundary")
ax.legend()
finish(fig, "exclusion_diagnostic.png")

# Synthetic analogue of the paper's textureless-vs-edge split.
fig, ax = plt.subplots(figsize=(7.6, 4.7))
values = [0.0173, 0.6117]
bars = ax.bar(["Smooth / far (proxy)", "Edge / near (proxy)"], values, color=[COLORS["green"], COLORS["red"]], width=0.62)
ax.axhline(0.02, color=COLORS["ink"], ls=":", lw=1.8, label="Paper Gaussian threshold")
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.4f}", ha="center", fontweight="bold")
ax.set_ylim(0, 0.70)
ax.set_ylabel("KL(belief || Gaussian)")
ax.set_title("Synthetic proxy preserves the smooth-versus-edge split")
ax.legend(loc="upper left")
finish(fig, "stereo_proxy.png")
