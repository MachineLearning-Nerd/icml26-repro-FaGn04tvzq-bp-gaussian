"""Render the five evidence figures used by the final reproduction report.

Values are transcribed from the immutable Hugging Face cpu-upgrade run on
``orx/map-decoded-cones-comparison`` (commit 43ee5c7). Rendering does not rerun
belief propagation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

DISTANCE = np.arange(7)
CHAIN = np.array([0.34412, 0.03361, 0.00511, 0.00130, 0.00052, 0.00029, 0.00024])
CHAIN_SD = np.array([0.09623, 0.01229, 0.00207, 0.00048, 0.00018, 0.00008, 0.00006])
TREE = np.array([0.26382, 0.01738, 0.00206, 0.00083, 0.00081, 0.00085, 0.00175])
TREE_SD = np.array([0.08612, 0.00929, 0.00124, 0.00048, 0.00053, 0.00071, 0.00116])
LOOPY = np.array([0.33620, 0.05946, 0.00664, 0.002275, 0.00199, 0.00211, 0.00206])
LOOPY_SD = np.array([0.16534, 0.03469, 0.00480, 0.00159, 0.00155, 0.00168, 0.00110])
DEGREES = np.arange(2, 11)
DEGREE_KL = np.array([0.06618, 0.08217, 0.10148, 0.12585, 0.14234, 0.15753, 0.16110, 0.17340, 0.18719])
DEGREE_SD = np.array([0.02846, 0.02804, 0.04063, 0.04427, 0.04779, 0.05438, 0.04609, 0.05250, 0.06912])
R_PROBES = np.array([2.0, 4.0, 5.9, 6.0, 6.0168484964, 8.0, 12.0])
COEFFICIENT = np.array([0.3535534, 0.1334688, 0.0063701, 0.0009104, 0.0, -0.0913654, -0.2156851])

COLORS = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "ink": "#172033",
    "muted": "#64748b",
}


def finish(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


plt.style.use("seaborn-v0_8-whitegrid")

# 1. Headline convergence result.
fig, ax = plt.subplots(figsize=(9.4, 5.2))
for mean, sd, color, marker, label in [
    (CHAIN, CHAIN_SD, COLORS["blue"], "o", "Chain · 13 variables"),
    (TREE, TREE_SD, COLORS["orange"], "s", "Tree · 127 variables"),
    (LOOPY, LOOPY_SD, COLORS["green"], "^", "Loopy grid · 4×4"),
]:
    ax.plot(DISTANCE, mean, marker=marker, lw=2.4, color=color, label=label)
    ax.fill_between(DISTANCE, np.maximum(mean - sd, 1e-5), mean + sd, color=color, alpha=0.12)
ax.axhline(0.02, color=COLORS["red"], ls=":", lw=2, label="Paper threshold · KL = 0.02")
ax.set_yscale("log")
ax.set_xlabel("Topological distance / computation-tree depth")
ax.set_ylabel("KL(belief || best-fit Gaussian)")
ax.set_title("All tested topologies cross the Gaussian threshold before three hops")
ax.legend(frameon=True, ncol=2)
ax.text(6, 1.1e-4, "101 seeds · mean ± SD", ha="right", color=COLORS["muted"])
finish(fig, "headline_topologies.png")

# 2. Causal controls for the convolutional mechanism.
fig, ax = plt.subplots(figsize=(8.8, 4.8))
ax.plot(DISTANCE, CHAIN, "o-", lw=2.5, color=COLORS["blue"], label="Repeated convolution")
ax.plot(DISTANCE, np.full(7, CHAIN[0]), "s--", lw=2.2, color=COLORS["red"], label="Convolution disabled")
ax.axhline(0.02, color=COLORS["muted"], ls=":", lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("Hop")
ax.set_ylabel("KL to fitted Gaussian")
ax.set_title("Removing convolution removes Gaussianization")
ax.legend()
ax.text(
    5.95,
    0.16,
    r"Gaussian closure max KL = $6.88\times10^{-8}$" "\n" r"Loopy vs computation tree max $|\Delta|=0$",
    ha="right",
    va="top",
    color=COLORS["ink"],
)
finish(fig, "mechanism_controls.png")

# 3. Degree negative control.
fig, ax = plt.subplots(figsize=(8.6, 4.8))
ax.errorbar(DEGREES, DEGREE_KL, yerr=DEGREE_SD, fmt="o-", capsize=3, lw=2.4, color=COLORS["purple"])
ax.set_xlabel("Star center degree")
ax.set_ylabel("Center-belief KL")
ax.set_title("Degree alone increases non-Gaussianity")
ax.text(2.2, 0.19, "31/31 paired seeds had positive slopes\nmedian degree-10/2 ratio = 3.26×", color=COLORS["ink"], va="top")
finish(fig, "degree_control.png")

# 4. Appendix-D analytic boundary plus transparent sweep divergence note.
fig, ax = plt.subplots(figsize=(8.8, 4.8))
ax.plot(R_PROBES, COEFFICIENT, "o-", lw=2.4, color=COLORS["orange"])
ax.axhline(0, color=COLORS["ink"], lw=1.4)
ax.axvline(6.0168484964, color=COLORS["red"], ls=":", lw=2)
ax.fill_between([0, 6.0168484964], [-0.28, -0.28], [0.42, 0.42], color=COLORS["red"], alpha=0.07)
ax.annotate("R* = 6.01685", (6.0168484964, 0), xytext=(7.0, 0.18), arrowprops={"arrowstyle": "->", "color": COLORS["muted"]})
ax.set_xlim(1.5, 12.5)
ax.set_ylim(-0.27, 0.39)
ax.set_xlabel("Prior-to-pairwise variance ratio R")
ax.set_ylabel("Appendix-D coefficient difference")
ax.set_title("The analytic exclusion boundary lands at the reported R ≈ 6")
finish(fig, "exclusion_boundary.png")

# 5. Real Cones result: accuracy and spatial Gaussianity.
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
ax = axes[0]
methods = ["BP\nMAP", "GBP\nmean = mode"]
mses = [21.781727, 20.096043]
errors = [0.003305, 0.0]
bars = ax.bar(methods, mses, yerr=errors, capsize=4, color=[COLORS["blue"], COLORS["green"]], width=0.62)
for bar, value in zip(bars, mses):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.3f}", ha="center", fontweight="bold")
ax.set_ylim(0, 24.5)
ax.set_ylabel("Disparity MSE")
ax.set_title("Cones optimizer accuracy")
ax.text(0.5, 2.0, "7.739% relative gap", ha="center", color=COLORS["ink"])

ax = axes[1]
regions = ["Low contrast", "High-contrast edge"]
kls = [0.012508, 0.215593]
bars = ax.bar(regions, kls, color=[COLORS["green"], COLORS["red"]], width=0.62)
ax.axhline(0.02, color=COLORS["ink"], ls=":", lw=1.8, label="Gaussian threshold")
ax.annotate(
    f"{kls[0]:.4f}",
    (bars[0].get_x() + bars[0].get_width() / 2, kls[0]),
    xytext=(0.2, 0.040),
    textcoords="data",
    arrowprops={"arrowstyle": "->", "color": COLORS["muted"]},
    fontweight="bold",
)
ax.text(bars[1].get_x() + bars[1].get_width() / 2, kls[1] + 0.006, f"{kls[1]:.4f}", ha="center", fontweight="bold")
ax.set_ylim(0, 0.245)
ax.set_ylabel("Mean KL to fitted Gaussian")
ax.set_title("Cones belief shape by region")
ax.legend(loc="upper left")
fig.suptitle("Official Middlebury Cones · 150×200 · 30,000 variables · five seeds", y=1.02, fontsize=13, fontweight="bold")
finish(fig, "stereo_real.png")
