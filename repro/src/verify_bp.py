"""Paper-scale CPU reproduction of synthetic results in arXiv:2601.21935.

All sizes, seed ranges, discretizations, and distribution widths below come
from Appendix I.  The script prints a compact evidence block because remote
run logs are the canonical experiment record.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
import time
import numpy as np

import paper_scale as P


OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUT, exist_ok=True)
START = time.perf_counter()
PAPER_KL_THRESHOLD = 0.02
SEEDS_101 = range(42, 143)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def curve_summary(curve: dict[int, list[float]]) -> dict[str, dict[str, float]]:
    return {
        str(distance): dict(zip(("mean", "std"), P.mean_std(values)))
        for distance, values in sorted(curve.items())
    }


def print_curve(label: str, summary: dict[str, dict[str, float]]) -> None:
    means = {key: round(value["mean"], 5) for key, value in summary.items()}
    stds = {key: round(value["std"], 5) for key, value in summary.items()}
    print(f"  {label} mean KL by distance: {means}")
    print(f"  {label} std  KL by distance: {stds}")


def random_kernels(rng: np.random.Generator, edges: list[tuple[int, int]], width: int):
    return {
        (min(a, b), max(a, b)): P.random_compact_pdf(rng, width)
        for a, b in edges
    }


results: dict[str, object] = {
    "paper": "arXiv:2601.21935",
    "implementation": "clean-room NumPy sum-product BP with compact shift-invariant convolutions",
}


# Claims 1--3 / Figure 4a: exact Appendix-I graph sizes and randomized seeds.
section("CLAIMS 1-3 / FIGURE 4a: PAPER-SCALE TOPOLOGY CONVERGENCE")
x1024 = np.linspace(-32.0, 31.0, 1024)
chain_curve: dict[int, list[float]] = defaultdict(list)
tree_curve: dict[int, list[float]] = defaultdict(list)
grid_curve: dict[int, list[float]] = defaultdict(list)

for seed in SEEDS_101:
    rng = np.random.default_rng(seed)

    # 13-variable chain, two prior factors (one at each endpoint).
    chain_edges = [(node, node + 1) for node in range(12)]
    chain_unaries = {
        0: P.compact_to_grid(P.random_compact_pdf(rng, 16), 1024),
        12: P.compact_to_grid(P.random_compact_pdf(rng, 16), 1024),
    }
    chain_beliefs = P.exact_tree_bp(
        1024, chain_edges, chain_unaries, random_kernels(rng, chain_edges, 12), root=0
    )
    for node, belief in chain_beliefs.items():
        chain_curve[min(node, 12 - node)].append(P.kl_to_fit_gaussian(belief, x1024))

    # Genuine full binary tree: 127 variables, 64 independent leaf priors.
    tree_edges = [((node - 1) // 2, node) for node in range(1, 127)]
    tree_unaries = {
        leaf: P.compact_to_grid(P.random_compact_pdf(rng, 16), 1024)
        for leaf in range(63, 127)
    }
    tree_beliefs = P.exact_tree_bp(
        1024, tree_edges, tree_unaries, random_kernels(rng, tree_edges, 12), root=0
    )
    for node, belief in tree_beliefs.items():
        level = int(np.floor(np.log2(node + 1)))
        distance_to_leaf = 6 - level
        tree_curve[distance_to_leaf].append(P.kl_to_fit_gaussian(belief, x1024))

    # 4x4 loopy grid, one prior at a corner, 30 synchronous iterations.
    grid_edges = []
    for row in range(4):
        for col in range(4):
            node = row * 4 + col
            if col < 3:
                grid_edges.append((node, node + 1))
            if row < 3:
                grid_edges.append((node, node + 4))
    grid_prior = P.compact_to_grid(P.random_compact_pdf(rng, 16), 1024)
    grid_beliefs = P.loopy_grid_bp(
        1024, 4, 4, {0: grid_prior}, random_kernels(rng, grid_edges, 12), iterations=30
    )
    for node, belief in grid_beliefs.items():
        distance = node // 4 + node % 4
        grid_curve[distance].append(P.kl_to_fit_gaussian(belief, x1024))

chain_summary = curve_summary(chain_curve)
tree_summary = curve_summary(tree_curve)
grid_summary = curve_summary(grid_curve)
print("  Appendix-I settings: 1024 bins [-32,31], 12-bin random factors, 16-bin random priors")
print("  seeds: 42..142 inclusive (n=101)")
print_curve("chain (13 variables, 2 priors)", chain_summary)
print_curve("tree (127 variables, 64 distinct leaf priors)", tree_summary)
print_curve("loopy grid (4x4, 1 prior)", grid_summary)

c1 = chain_summary["3"]["mean"] < PAPER_KL_THRESHOLD
c2 = tree_summary["3"]["mean"] < PAPER_KL_THRESHOLD
c3 = grid_summary["3"]["mean"] < PAPER_KL_THRESHOLD
print(f"  paper threshold at distance 3: chain={chain_summary['3']['mean']:.6f}, tree={tree_summary['3']['mean']:.6f}, grid={grid_summary['3']['mean']:.6f}")
print(f"  assessments: C1={'ALIGNED' if c1 else 'DIVERGENT'}; C2={'ALIGNED' if c2 else 'DIVERGENT'}; C3={'ALIGNED' if c3 else 'DIVERGENT'}")
results["claim_1_chain"] = {"aligned": bool(c1), "curve": chain_summary}
results["claim_2_tree"] = {
    "aligned": bool(c2),
    "curve": tree_summary,
    "distinct_leaf_priors": 64,
    "branch_multiplication": True,
}
results["claim_3_loopy"] = {"aligned": bool(c3), "curve": grid_summary, "iterations": 30}


# Claim 5 / Figure 4b: degree is isolated with zero convolutional depth.
section("CLAIM 5 / FIGURE 4b: NODE-DEGREE CONTROL")
degree_values = list(range(2, 11))
degree_kls: dict[int, list[float]] = defaultdict(list)
for seed in range(42, 73):
    for degree in degree_values:
        rng = np.random.default_rng(seed * 100 + degree)
        edges = [(0, leaf) for leaf in range(1, degree + 1)]
        unaries = {
            leaf: P.compact_to_grid(P.random_compact_pdf(rng, 30), 1024)
            for leaf in range(1, degree + 1)
        }
        beliefs = P.exact_tree_bp(
            1024, edges, unaries, random_kernels(rng, edges, 12), root=0
        )
        degree_kls[degree].append(P.kl_to_fit_gaussian(beliefs[0], x1024))

degree_summary = curve_summary(degree_kls)
print("  Appendix-I settings: 3..11 total variables, 31 seeds, 1024 bins, 30-bin priors, 12-bin factors")
print_curve("star center", degree_summary)
degree_means = np.asarray([degree_summary[str(d)]["mean"] for d in degree_values])
degree_slope = float(np.polyfit(degree_values, degree_means, 1)[0])
c5 = degree_slope > 0.0 and degree_means[-1] > degree_means[0]
print(f"  OLS slope KL/neighbor={degree_slope:.6f}; endpoints {degree_means[0]:.6f}->{degree_means[-1]:.6f}")
print(f"  assessment: C5 degree effect={'ALIGNED' if c5 else 'DIVERGENT'}")
results["claim_5_degree"] = {
    "aligned": bool(c5),
    "curve": degree_summary,
    "ols_slope": degree_slope,
}


# Claim 4 / Figure 4c: normalized variance sweep on a 20-variable chain.
section("CLAIM 4 / FIGURE 4c: NORMALIZED PRIOR-VARIANCE EXCLUSION ZONE")
x128 = np.linspace(0.0, 63.0, 128)
widths = list(range(1, 129))
prior_sweep: dict[int, list[float]] = defaultdict(list)
ratio_sweep: dict[int, list[float]] = defaultdict(list)
chain20_edges = [(node, node + 1) for node in range(19)]
for seed in SEEDS_101:
    rng = np.random.default_rng(seed)
    kernels20 = random_kernels(rng, chain20_edges, 8)
    kernel_vars = []
    offsets = (np.arange(8, dtype=np.float64) - 3.5) * (x128[1] - x128[0])
    for kernel in kernels20.values():
        kernel_vars.append(P.pmf_variance(kernel, offsets))
    pairwise_variance = float(np.mean(kernel_vars))
    for width in widths:
        prior = P.bounded_uniform_pdf(128, width)
        beliefs = P.exact_tree_bp(
            128,
            chain20_edges,
            {node: prior for node in range(20)},
            kernels20,
            root=0,
        )
        center = beliefs[10]
        prior_sweep[width].append(P.kl_to_fit_gaussian(center, x128))
        ratio_sweep[width].append(P.pmf_variance(prior, x128) / pairwise_variance)

width_mean_kl = {width: P.mean_std(prior_sweep[width])[0] for width in widths}
width_std_kl = {width: P.mean_std(prior_sweep[width])[1] for width in widths}
width_mean_ratio = {width: P.mean_std(ratio_sweep[width])[0] for width in widths}
low_r = [width_mean_kl[w] for w in widths if width_mean_ratio[w] <= 6.0 and w > 1]
high_r = [width_mean_kl[w] for w in widths if width_mean_ratio[w] > 6.0]
low_r_mean = float(np.mean(low_r))
high_r_mean = float(np.mean(high_r))
crossings = [w for w in widths if width_mean_kl[w] < PAPER_KL_THRESHOLD]
first_crossing = crossings[0] if crossings else None
c4 = low_r_mean > high_r_mean and bool(crossings)
selected_widths = [1, 2, 4, 8, 16, 32, 64, 128]
selected = {
    str(w): {
        "R": width_mean_ratio[w],
        "kl_mean": width_mean_kl[w],
        "kl_std": width_std_kl[w],
    }
    for w in selected_widths
}
print("  Appendix-I settings: 20 variables, 128 bins [0,63], 8-bin random factors, widths 1..128")
print("  seeds: 42..142 inclusive (n=101); center belief after exact converged tree BP (=20 synchronous iterations)")
print(f"  selected sweep points (width: R, mean KL, std): {json.dumps(selected, sort_keys=True)}")
print(f"  mean KL for R<=6 (excluding degenerate width=1): {low_r_mean:.6f}")
print(f"  mean KL for R>6: {high_r_mean:.6f}")
print(f"  first width with mean KL<0.02: {first_crossing}; R={width_mean_ratio[first_crossing] if first_crossing else None}")
print(f"  assessment: C4 exclusion boundary={'ALIGNED' if c4 else 'DIVERGENT'}")
results["claim_4_exclusion"] = {
    "aligned": bool(c4),
    "low_R_mean_kl": low_r_mean,
    "high_R_mean_kl": high_r_mean,
    "first_width_below_0.02": first_crossing,
    "selected_points": selected,
    "full_curve": [
        {
            "width": width,
            "R": width_mean_ratio[width],
            "kl_mean": width_mean_kl[width],
            "kl_std": width_std_kl[width],
        }
        for width in widths
    ],
}


section("PAPER-SCALE SYNTHETIC SUMMARY")
checks = {
    "claim_1_chain": c1,
    "claim_2_tree": c2,
    "claim_3_loopy": c3,
    "claim_4_exclusion": c4,
    "claim_5_degree": c5,
}
for name, aligned in checks.items():
    print(f"  {name}: {'ALIGNED' if aligned else 'DIVERGENT IN THIS RUN'}")
print(f"  aligned: {sum(checks.values())}/5")
print("  Claim 6 is intentionally separate: it requires real Middlebury Cones BP/GBP, not a proxy.")
print(f"  elapsed_seconds={time.perf_counter() - START:.3f}")
results["summary"] = {"aligned": int(sum(checks.values())), "total": 5}
with open(os.path.join(OUT, "paper_scale_synthetic.json"), "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2)
print("  wrote outputs/paper_scale_synthetic.json")
