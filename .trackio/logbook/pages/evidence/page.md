# Evidence

## Paper-scale topology results

Mean KL to a moment-matched Gaussian over seeds 42–142:

| Distance | Chain | Tree | Loopy grid |
|---:|---:|---:|---:|
| 0 | 0.34412 | 0.26382 | 0.33620 |
| 1 | 0.03361 | 0.01738 | 0.05946 |
| 2 | 0.00511 | 0.00206 | 0.00664 |
| 3 | 0.00130 | 0.00083 | 0.00227 |
| 4 | 0.00052 | 0.00081 | 0.00199 |
| 5 | 0.00029 | 0.00085 | 0.00211 |
| 6 | 0.00024 | 0.00175 | 0.00206 |

The paper threshold is 0.02. The direct computation-tree message difference
was 0.000e+00.

## Controls and boundary

- Disabled convolution leaves KL constant.
- Gaussian closure residual is at most 6.877e-08.
- Star degree 2→10 raises mean center KL from 0.06618 to 0.18719.
- All 31 paired degree slopes are positive.
- Analytic Appendix-D boundary: u*=0.4991595572 and R*=6.0168484964.
- The substituted finite-grid sweep has mean KL 0.080769 below normalized
  variance 0.5 and 0.081358 above it; no width falls below 0.02.

## Real Middlebury Cones

Settings: 150×200, 30,000 variables, 28 disparity labels, 5×5 patches,
lambda=0.002, edge threshold 3, seeds 42–46, maximum 2,000 iterations.

| Metric | BP | GBP |
|---|---:|---:|
| Registered disparity MSE | 21.781727 ± 0.003305 (MAP) | 20.096043 ± 0.000000 |
| Iterations to stability | about 580 | about 260 |

Relative MSE gap is 7.7390%. BP posterior-expectation sensitivity MSE is
25.116087.

| Region | Mean KL | Fraction KL below 0.02 |
|---|---:|---:|
| Low contrast | 0.012508 | 83.79% |
| High-contrast edge | 0.215593 | 23.63% |
