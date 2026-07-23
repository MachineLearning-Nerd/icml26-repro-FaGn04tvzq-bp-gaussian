# Evidence

## Paper-scale topology results

All values are mean KL to a moment-matched Gaussian over seeds 42–142.

| Distance | Chain (13 variables) | Tree (127 variables) | Loopy grid (4×4) |
|---:|---:|---:|---:|
| 0 | 0.34412 | 0.26382 | 0.33620 |
| 1 | 0.03361 | 0.01738 | 0.05946 |
| 2 | 0.00511 | 0.00206 | 0.00664 |
| 3 | **0.00130** | **0.00083** | **0.00227** |
| 4 | 0.00052 | 0.00081 | 0.00199 |
| 5 | 0.00029 | 0.00085 | 0.00211 |
| 6 | 0.00024 | 0.00175 | 0.00206 |

The paper's Gaussian threshold is `0.02`. The exact computation-tree message difference is `0.000e+00`.

## Mechanism, degree, and boundary controls

- Destructive control: disabled-convolution KL stays exactly constant.
- Closure control: Gaussian inputs remain Gaussian, maximum KL `6.877e-08`.
- Star degree 2→10: mean center KL `0.06618 → 0.18719`; 31/31 paired slopes positive; median endpoint ratio `3.260×`.
- Gaussian-incoming star control: maximum KL `8.764e-09`.
- Appendix-D boundary: `u*=0.4991595572`, `R*=6.0168484964`; coefficient is positive at `R=6` and negative at `R=8`.
- Transparent divergence: the substituted Figure-4c finite-grid sweep had mean KL `0.080769` below normalized variance 0.5 and `0.081358` above; no width fell below `0.02`.

## Real Middlebury Cones

Settings: official Cones stereo pair, 150×200, 30,000 variables, 28 disparity labels, 5×5 patches, `lambda=0.002`, edge threshold `3`, seeds 42–46, maximum 2,000 iterations.

| Metric | BP | GBP |
|---|---:|---:|
| Registered disparity MSE | `21.781727 ± 0.003305` (MAP) | `20.096043 ± 0.000000` (Gaussian mean/mode) |
| Iterations to stability | about 580 | about 260 |

Relative MSE gap: `7.7390%`. BP posterior-expectation sensitivity MSE: `25.116087`.

| Region | Mean KL | Median KL | Fraction KL `<0.02` |
|---|---:|---:|---:|
| Low contrast | `0.012508` | `0.001876` | `83.79%` |
| High-contrast edge | `0.215593` | `0.103099` | `23.63%` |
