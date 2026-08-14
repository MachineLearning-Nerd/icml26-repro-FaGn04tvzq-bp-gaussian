# Overview

## Belief Propagation Converges to Gaussian Distributions

Paper: [arXiv:2601.21935](https://arxiv.org/abs/2601.21935) · OpenReview:
FaGn04tvzq

The registered clean-room CPU run labeled all six claim groups aligned. The
repository audit preserves two qualifications: Claim 4 has a divergent
finite-grid sweep, and Claim 6 uses clean-room GBP and preprocessing.

| Audit item | Status |
| --- | --- |
| Overall | MIXED_RESULTS |
| Evidence-release gate | PASSED |
| Strict universal paper-claim gate | NOT_READY |
| C1, C2, C3, C5 | VERIFIED_SCOPED |
| C4 | MIXED_RESULTS |
| C6 | VERIFIED_SCOPED under registered <10% MSE criterion |

Headline evidence: hop-3 chain/tree/loopy KL is 0.001303 / 0.000826 /
0.002275; the analytic exclusion boundary is R*=6.0168484964; star-center
KL rises from 0.06618 to 0.18719; Cones BP MAP and GBP MSE are 21.781727 and
20.096043.
