# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b38f83ccd357", "created_at": "2026-07-21T14:04:06+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```
  KL vs depth: [0.6117, 0.1743, 0.0667, 0.0327, 0.0212, 0.0173, 0.0161]
  tree KL decreases with depth -> PASS

==============================================================================
CLAIM 3 (Theorem 4.5): loopy graph beliefs Gaussianize with distance (computation-tree equivalence)
==============================================================================
  KL vs node (distance from prior): [0.594, 0.1494, 0.0555, 0.0251, 0.0189]
  loopy KL decreases with distance -> PASS

==============================================================================
CLAIM 4 (Lemma 4.2): extreme-variance prior prevents Gaussian convergence (exclusion zone)
==============================================================================
  KL with wide (exclusion) prior: [0.3474, 0.2708, 0.2122, 0.1676, 0.1336, 0.1078, 0.088]
  KL with normal prior (c1):      [0.6117, 0.1743, 0.0667, 0.0327, 0.0212, 0.0173, 0.0161]
  exclusion prior keeps KL high (0.0880 >> 0.0161) -> PASS

==============================================================================
CLAIM 5 (Figure 4): KL small within a few hops; node degree modifies non-Gaussianity (star)
==============================================================================
  chain KL beyond hop 3: [0.0327, 0.0212, 0.0173, 0.0161]; star-center KL vs degree (2, 4, 6): [0.3941, 0.7203, 0.9234]
  KL small beyond 3 hops (True); degree modifies center KL (True) -> PASS

==============================================================================
CLAIM 6 (Figure 5): BP Gaussianizes in smooth (textureless) regions, non-Gaussian at edges (synthetic proxy)
==============================================================================
  KL smooth(region far)=0.0173 << KL edge(near prior)=0.6117 -> PASS
  (Paper: stereo depth estimation; we verify the smooth-vs-edge Gaussianization pattern on a synthetic chain.)

==============================================================================
VERDICT SUMMARY
==============================================================================
  [PASS] c1_chain
  [PASS] c2_tree
  [PASS] c3_loopy
  [PASS] c4_exclusion
  [PASS] c5_kl_3hops
  [PASS] c6_stereo_proxy

  6/6 claims verified.
  wrote outputs/verdict.json
```
