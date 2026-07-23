"""Paper-scale discrete belief-propagation experiments from arXiv:2601.21935.

The paper uses shift-invariant pairwise factors, so a factor-to-variable update
is a short one-dimensional convolution.  Keeping kernels in their compact
support makes the Appendix-I experiments practical on CPU without changing the
sum-product calculation.
"""
from __future__ import annotations

from collections import defaultdict, deque
import numpy as np


TINY = 1e-300


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(p, 1.0 / p.size)
    return p / total


def multiply_pmfs(parts: list[np.ndarray], size: int) -> np.ndarray:
    """Normalized pointwise product, evaluated in log space."""
    if not parts:
        return np.full(size, 1.0 / size)
    logp = np.zeros(size, dtype=np.float64)
    for p in parts:
        logp += np.log(np.maximum(p, TINY))
    logp -= float(logp.max())
    return normalize(np.exp(logp))


def random_compact_pdf(rng: np.random.Generator, width: int) -> np.ndarray:
    """The paper's random-noise distribution on a compact support."""
    return normalize(rng.random(width))


def bounded_uniform_pdf(size: int, width: int) -> np.ndarray:
    """Centered bounded-uniform prior over ``width`` of ``size`` bins."""
    width = int(np.clip(width, 1, size))
    p = np.zeros(size, dtype=np.float64)
    start = (size - width) // 2
    p[start : start + width] = 1.0
    return normalize(p)


def compact_to_grid(compact: np.ndarray, size: int) -> np.ndarray:
    p = np.zeros(size, dtype=np.float64)
    start = (size - compact.size) // 2
    p[start : start + compact.size] = compact
    return normalize(p)


def convolve_message(cavity: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Shift-invariant factor update, cropped to the variable domain."""
    return normalize(np.convolve(cavity, kernel, mode="same"))


def kl_to_fit_gaussian(p: np.ndarray, x: np.ndarray) -> float:
    p = normalize(p)
    mu = float(np.dot(p, x))
    variance = float(np.dot(p, (x - mu) ** 2))
    spacing = float(abs(x[1] - x[0]))
    variance = max(variance, (spacing / 8.0) ** 2)
    g = normalize(np.exp(-0.5 * (x - mu) ** 2 / variance))
    mask = p > 0.0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.maximum(g[mask], TINY)))))


def pmf_variance(p: np.ndarray, x: np.ndarray) -> float:
    p = normalize(p)
    mu = float(np.dot(p, x))
    return float(np.dot(p, (x - mu) ** 2))


def _kernel_for_direction(
    kernels: dict[tuple[int, int], np.ndarray], source: int, target: int
) -> np.ndarray:
    key = (min(source, target), max(source, target))
    kernel = kernels[key]
    return kernel if source < target else kernel[::-1]


def exact_tree_bp(
    size: int,
    edges: list[tuple[int, int]],
    unaries: dict[int, np.ndarray],
    kernels: dict[tuple[int, int], np.ndarray],
    root: int = 0,
) -> dict[int, np.ndarray]:
    """Exact sum-product BP for a pairwise tree, including both directions."""
    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    nodes = sorted(neighbours)
    uniform = np.full(size, 1.0 / size)
    unary = {node: normalize(unaries.get(node, uniform)) for node in nodes}

    parent = {root: -1}
    order: list[int] = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        order.append(node)
        for other in neighbours[node]:
            if other == parent[node]:
                continue
            parent[other] = node
            queue.append(other)

    messages: dict[tuple[int, int], np.ndarray] = {}
    for node in reversed(order[1:]):
        par = parent[node]
        incoming = [unary[node]]
        incoming.extend(messages[(other, node)] for other in neighbours[node] if other != par)
        cavity = multiply_pmfs(incoming, size)
        messages[(node, par)] = convolve_message(
            cavity, _kernel_for_direction(kernels, node, par)
        )

    for node in order:
        for child in neighbours[node]:
            if parent.get(child) != node:
                continue
            incoming = [unary[node]]
            incoming.extend(messages[(other, node)] for other in neighbours[node] if other != child)
            cavity = multiply_pmfs(incoming, size)
            messages[(node, child)] = convolve_message(
                cavity, _kernel_for_direction(kernels, node, child)
            )

    beliefs = {}
    for node in nodes:
        beliefs[node] = multiply_pmfs(
            [unary[node], *(messages[(other, node)] for other in neighbours[node])], size
        )
    return beliefs


def loopy_grid_bp(
    size: int,
    rows: int,
    cols: int,
    unaries: dict[int, np.ndarray],
    kernels: dict[tuple[int, int], np.ndarray],
    iterations: int,
) -> dict[int, np.ndarray]:
    """Synchronous sum-product BP on a four-neighbour grid."""
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col + 1 < cols:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + cols))
    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    uniform = np.full(size, 1.0 / size)
    unary = {node: normalize(unaries.get(node, uniform)) for node in range(rows * cols)}
    messages = {
        direction: uniform.copy()
        for a, b in edges
        for direction in ((a, b), (b, a))
    }

    for _ in range(iterations):
        updated = {}
        for source, target in messages:
            incoming = [unary[source]]
            incoming.extend(
                messages[(other, source)] for other in neighbours[source] if other != target
            )
            cavity = multiply_pmfs(incoming, size)
            updated[(source, target)] = convolve_message(
                cavity, _kernel_for_direction(kernels, source, target)
            )
        messages = updated

    return {
        node: multiply_pmfs(
            [unary[node], *(messages[(other, node)] for other in neighbours[node])], size
        )
        for node in range(rows * cols)
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0
