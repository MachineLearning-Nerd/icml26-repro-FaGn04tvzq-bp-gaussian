# Overview

## Belief Propagation Converges to Gaussian Distributions

Paper: [arXiv:2601.21935](https://arxiv.org/abs/2601.21935) · OpenReview: `FaGn04tvzq`

This clean-room CPU reproduction directly tests all six claim groups. The final registered run reports **6/6 aligned** on Hugging Face `cpu-upgrade`.

| Headline paper result | Observed result |
|---|---|
| Chain/tree/loopy KL `<0.02` within 3 hops | Hop-3 KL `0.001303 / 0.000826 / 0.002275`, 101 seeds |
| Appendix-D exclusion boundary `R≈6` | `R*=6.0168484964` |
| Degree increases star-center non-Gaussianity | Mean KL `0.06618 → 0.18719`; 31/31 positive paired slopes |
| BP and GBP similar on 150×200 Cones | BP MAP MSE `21.781727`; GBP `20.096043` (7.739% gap) |
| Textureless beliefs Gaussian, edges non-Gaussian | Mean KL `0.012508` low contrast vs `0.215593` edges |

The official 375×450 Cones data was resized to the paper's 150×200 setting and disparities were scaled horizontally. The authors' supplementary factor implementation was unavailable, so the BP/GBP solvers are independently implemented and this substitution is reported throughout.
