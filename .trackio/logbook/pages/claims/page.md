# Claims

## Claim-by-claim assessment

| # | Claim | Direct evidence | Assessment |
|---:|---|---|---|
| 1 | Chain beliefs become Gaussian with path distance. | 1,024 bins, random factors, seeds 42–142: mean KL `0.34412 → 0.001303` by hop 3 and `0.00024` by hop 6. Disabling convolution leaves KL constant; Gaussian closure max KL `6.877e-08`. | **Aligned** |
| 2 | Convergence extends to genuine branching trees. | 127 variables and 64 distinct leaf priors: mean KL `0.26382 → 0.000826` by distance 3 over 101 seeds. Branch messages are multiplied at internal nodes. | **Aligned** |
| 3 | Loopy graphs inherit the computation-tree mechanism. | 4×4 loopy grid, 101 seeds: mean KL `0.33620 → 0.002275` by depth 3. Direct loopy and depth-4 unwrapped messages have maximum absolute difference `0`. | **Aligned** |
| 4 | Confident priors create an exclusion boundary near `R≤6`. | Appendix-D numerical solve: `u*=0.4991595572`, `R*=6.0168484964`; sign probes flip across the root. A separate 101-seed normalized-prior-variance sweep did not show the reported low-KL transition. | **Analytic claim aligned; finite-grid sweep divergent in this implementation** |
| 5 | All topologies reach KL `<0.02` within 3 hops; degree alone increases star non-Gaussianity. | Hop-3 KL `0.001303/0.000826/0.002275`. Star mean KL `0.06618 → 0.18719` from degree 2→10; 31/31 paired slopes positive; Gaussian-incoming control max KL `8.764e-09`. | **Aligned** |
| 6 | Cones GBP tracks BP, and textureless pixels Gaussianize while edges do not. | Official Cones, 150×200, 30,000 variables, five seeds: BP MAP MSE `21.781727±0.003305`, GBP `20.096043±0`, 7.739% gap. Mean KL `0.012508` low contrast vs `0.215593` edges. | **Aligned under registered <10% MSE criterion** |

The aggregate label does not erase the Claim-4 finite-grid divergence or the posterior-decoder sensitivity: decoding BP by posterior expectation gives MSE `25.116087`, which is reported as a secondary result.
