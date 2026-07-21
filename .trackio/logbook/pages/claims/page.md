# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ce6a150be5d0", "created_at": "2026-07-21T14:04:05+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. Theorem 4.1 proves that variables in pairwise chain factor graphs develop increasingly Gaussian-distributed beliefs as topological distance from prior factors increases, driven by repeated convolution via the Central Limit Theorem (Theorem 4.1).
2. Theorem 4.4 extends the Gaussian convergence result to tree-topology factor graphs, provided the message non-Gaussianity ε satisfies ε² ≪ ε with increasing distance from the nearest prior factor (Theorem 4.4).
3. Theorem 4.5 uses computation-tree equivalence to show loopy factor graphs exhibit the same Gaussian convergence mechanism as tree graphs under identical assumptions (Theorem 4.5).
4. Lemma 4.2 identifies an 'exclusion zone' in which non-Gaussian prior factors prevent Gaussian convergence when the prior-to-pairwise variance ratio R is less than or equal to approximately 6 (Lemma 4.2).
5. Figure 4 shows all tested topologies reach a KL divergence below 0.02 within 3 hops from prior factors, and that higher node degree increases non-Gaussianity in star graphs while weaker priors enable faster Gaussian emergence (Figure 4).
6. Figure 5 applies the theory to stereo depth estimation belief propagation, showing Gaussian convergence in textureless image regions while non-Gaussian beliefs persist at high-contrast edges (Figure 5).
