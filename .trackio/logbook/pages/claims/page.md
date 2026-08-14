# Claims

## Claim-by-claim assessment

| # | Claim | Producer and evidence | Assessment |
|---:|---|---|---|
| 1 | Chain beliefs become Gaussian with path distance. | verify_bp.py and paper_scale.py; 1,024 bins, seeds 42–142, convolution and Gaussian-closure controls | **VERIFIED_SCOPED** |
| 2 | Convergence extends to genuine branching trees. | verify_bp.py and paper_scale.py; 127 variables and 64 distinct leaf priors | **VERIFIED_SCOPED** |
| 3 | Loopy graphs inherit the computation-tree mechanism. | verify_bp.py and paper_scale.py; 4×4 grid and explicit unwrapped-message comparison | **VERIFIED_SCOPED** |
| 4 | Confident priors create an exclusion boundary near R≈6. | Analytic root/sign probes align; a separate 101-seed finite-grid sweep shows no low-KL transition | **MIXED_RESULTS** |
| 5 | Degree alone increases star-center non-Gaussianity. | Paired-prefix degree 2–10 sweep over 31 seeds and Gaussian-incoming control | **VERIFIED_SCOPED** |
| 6 | Cones GBP tracks BP and textureless beliefs are more Gaussian. | stereo_cones.py and verify_bp.py; five seeds, MAP decoder, contrast-region KL | **VERIFIED_SCOPED** under registered <10% MSE criterion |

The original output label 6/6 aligned is retained in verdict.json as
historical run metadata. It is not a substitute for the current claim
boundaries above.
