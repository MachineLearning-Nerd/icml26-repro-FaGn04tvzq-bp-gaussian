"""Clean-room belief propagation on pairwise factor graphs, from
"Belief Propagation Converges to Gaussian Distributions in Sparsely-Connected Factor Graphs"
(arXiv 2601.21935). numpy, CPU (discretized continuous domain).

Pairwise chain: variables x_0..x_L on a grid. Prior (unary, non-Gaussian) at x_0. Pairwise factors
psi(x_i,x_j)=exp(-(x_i-x_j)^2/(2 sig_p^2)) (Gaussian smoothing). BP message i->i+1 is the psi-
convolution of the incoming product; repeated convolution drives beliefs to Gaussian (CLT).
"""
from __future__ import annotations
import numpy as np


def grid(lo, hi, G):
    return np.linspace(lo, hi, G)


def gaussian_kernel(X, sig):
    d2 = (X[:, None] - X[None, :]) ** 2
    return np.exp(-d2 / (2 * sig ** 2))


def bimodal_prior(X, means=(-2.5, 2.5), sd=0.7):
    """Strongly non-Gaussian unary prior (bimodal mixture)."""
    p = np.zeros_like(X)
    for m in means:
        p += np.exp(-(X - m) ** 2 / (2 * sd ** 2))
    return p / p.sum()


def uniform_unary(X):
    return np.ones_like(X) / len(X)


def chain_bp(X, prior, psi, L):
    """Forward BP on a chain x_0..x_L with `prior` at x_0; returns beliefs (L+1, G)."""
    G = len(X); beliefs = np.zeros((L + 1, G))
    b = prior.copy(); b /= b.sum()
    beliefs[0] = b
    for i in range(L):
        # message i->i+1 = psi^T @ (unary_i * belief_i); unary is prior at 0, uniform elsewhere
        msg = psi.T @ beliefs[i]
        msg /= msg.sum()
        beliefs[i + 1] = msg
    return beliefs


def kl_to_gaussian(b, X):
    """Discrete KL( b || Gaussian) on the grid (both as PMFs over grid points). Always >= 0."""
    b = np.maximum(b / b.sum(), 1e-12)
    mu = float(np.sum(X * b)); var = float(np.sum((X - mu) ** 2 * b))
    var = max(var, 1e-6)
    g = np.exp(-(X - mu) ** 2 / (2 * var))
    g = np.maximum(g / g.sum(), 1e-12)
    return float(np.sum(b * (np.log(b) - np.log(g))))


def tree_bp(X, prior, psi, depth, branching=2):
    """BP on a balanced tree of given depth with prior at the root; returns beliefs by level (depth+1, G)."""
    G = len(X); beliefs = np.zeros((depth + 1, G))
    b = prior.copy(); b /= b.sum(); beliefs[0] = b
    for d in range(depth):
        msg = psi.T @ beliefs[d]; msg /= msg.sum()
        beliefs[d + 1] = msg          # all nodes at level d+1 have the same (symmetric tree) belief
    return beliefs


def loopy_bp(X, priors, psi, edges, iters=50):
    """Loopy sum-product BP on a general graph (priors: dict node->unary; edges: list). Returns beliefs dict."""
    G = len(X); nodes = sorted(set([n for e in edges for n in e]) | set(priors.keys()))
    unary = {n: (priors[n] if n in priors else np.ones(G) / G) for n in nodes}
    msg = {(i, j): np.ones(G) / G for i, j in edges} | {(j, i): np.ones(G) / G for i, j in edges}
    nbr = {n: [] for n in nodes}
    for i, j in edges: nbr[i].append(j); nbr[j].append(i)
    for _ in range(iters):
        new = {}
        for i, j in msg:
            prod = unary[i].copy()
            for k in nbr[i]:
                if k != j: prod = prod * msg[(k, i)]
            m = psi.T @ prod; m /= m.sum(); new[(i, j)] = m
        msg = new
    beliefs = {}
    for n in nodes:
        b = unary[n].copy()
        for k in nbr[n]: b = b * msg[(k, n)]
        beliefs[n] = b / b.sum()
    return beliefs
