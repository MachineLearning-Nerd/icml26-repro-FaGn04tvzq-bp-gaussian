# Conclusion

## Executive assessment

The final clean-room CPU run supplies direct evidence for all six claim groups and reports **6/6 aligned**. It specifically closes the previous audit gaps: 1,024-bin multi-seed tests replace the 90-bin toy; the tree genuinely branches; loopy BP is checked against an explicit computation tree; the `R≈6` boundary is solved directly and swept empirically; and Claim 6 uses a real 30,000-variable Middlebury Cones graph with both BP and GBP.

The strongest numbers are hop-3 KL `0.001303/0.000826/0.002275`, exact computation-tree difference `0`, analytic `R*=6.016848`, 31/31 positive degree slopes, and Cones low-contrast/edge KL `0.012508/0.215593`.

The assessment is qualified in two places. The secondary finite-grid Figure-4c sweep did not show the paper's transition. The Cones BP/GBP gap is `7.739%`, which meets the registered 10% similarity criterion but is not literally negligible. A source-exact replication still needs the unavailable supplementary factor implementation and original data-processing convention.

| Scope | Compute | Outcome |
|---|---|---|
| Six claim groups; synthetic paper scale plus official Cones | Hugging Face `cpu-upgrade`; final run 4m19s | 6/6 aligned with disclosed divergence and substitutions |
