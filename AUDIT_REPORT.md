# Reproduction audit report

## Executive assessment

The committed final run provides finite evidence for all six claim groups, but
the defensible repository status is MIXED_RESULTS:

- C1, C2, C3, and C5 are VERIFIED_SCOPED under their declared synthetic
  contracts.
- C4 is mixed: the analytic Appendix-D boundary aligns, while the substituted
  finite-grid Figure-4c sweep does not recover the reported low-KL transition.
- C6 is VERIFIED_SCOPED under the registered <10% BP/GBP MSE criterion, with
  an explicit clean-room implementation and preprocessing boundary.
- The evidence-release gate is PASSED; the strict universal paper-claim gate
  is NOT_READY.

The original run recorded 6/6 aligned. That label is preserved as historical
run metadata, not used to hide the C4 divergence or the C6 substitutions.

## Claim-to-evidence map

| Claim | Producer | Key control or check | Result |
| ---: | --- | --- | --- |
| C1 | verify_bp.py + paper_scale.py chain path | Disabled-convolution control and Gaussian-closure control | Hop-3 mean KL 0.001303; VERIFIED_SCOPED |
| C2 | verify_bp.py + paper_scale.py tree path | 127 variables and 64 distinct leaf priors | Distance-3 mean KL 0.000826; VERIFIED_SCOPED |
| C3 | verify_bp.py + paper_scale.py grid path | Explicit depth-4 computation-tree comparison | Hop-3 mean KL 0.002275; max message difference 0; VERIFIED_SCOPED |
| C4 | verify_bp.py analytic and sweep paths | Root/sign probes plus a 101-seed finite-grid sweep | R*=6.0168484964 analytically; finite sweep divergent; MIXED_RESULTS |
| C5 | verify_bp.py paired star path | Paired prefixes and Gaussian-incoming negative control | 31/31 positive slopes; VERIFIED_SCOPED |
| C6 | verify_bp.py + stereo_cones.py | Five-seed MSE comparison and contrast-region KL split | 7.739% MSE gap; VERIFIED_SCOPED under registered criterion |

## Evidence details

### Topology and mechanism

The chain, branching tree, and loopy grid use 1,024 bins and seeds 42–142.
At hop or depth three, the mean KL values are 0.001303, 0.000826, and
0.002275, all below the paper's 0.02 threshold. The grid messages match the
explicitly expanded computation-tree messages to floating-point zero in the
registered run.

The mechanism controls are causal within this finite implementation:
disabling convolution leaves the initial KL unchanged, while convolving
Gaussian inputs produces a maximum residual KL of 6.877e-08. The degree
control uses the same seed-specific star prefixes as degree increases; its
mean center KL rises from 0.06618 at degree 2 to 0.18719 at degree 10, with
all 31 paired slopes positive.

### Claim-4 boundary and divergence

The Appendix-D coefficient equation gives u*=0.4991595572 and
R*=6.0168484964, close to the paper's stated R≈6. However, the separate
101-seed, 128-width finite-grid sweep gives mean KL 0.080769 below normalized
prior variance 0.5 and 0.081358 above it; no tested width falls below 0.02.
This is reported as a real implementation divergence, not converted into a
passing claim by the analytic result.

### Cones

The registered five-seed run uses 30,000 variables, 28 disparity labels,
5×5 patches, lambda=0.002, edge threshold 3, and up to 2,000 synchronous
iterations. MAP-decoded BP has MSE 21.781727±0.003305; mode/mean-decoded GBP
has MSE 20.096043; the relative gap is 7.739%. Low-contrast mean KL is
0.012508, compared with 0.215593 on high-contrast edges.

The BP posterior-expectation sensitivity is 25.116087, so the decoder is a
material part of the result. The registered criterion allows a gap below 10%;
the paper does not provide an exact MSE table against which the numeric value
can be source-matched.

## Controls and negative evidence

- Removing convolution prevents the chain from Gaussianizing.
- Gaussian inputs remain Gaussian up to the discretization residual.
- The computation-tree identity is checked directly rather than inferred from
  visually similar curves.
- The Gaussian-incoming star control remains Gaussian as degree increases.
- The finite-grid Figure-4c divergence is preserved in the verdict and report.
- Posterior-expectation BP MSE is retained as a sensitivity result.

## Reproduction boundaries

This is not a source-exact rerun. The supplementary factor code was
unavailable, so the repository uses clean-room NumPy implementations for
photometric factors, mode-centered Laplace GBP, preprocessing, and decoding.
The Cones result is therefore evidence for the registered implementation and
criterion, not a universal guarantee that every source-exact implementation
has identical numbers.

No independent CSV checker is committed. The verifier, output files, report
figures, and controls form the declared evidence chain. A future stronger
audit should add the authors' supplementary definitions, original 375×450
processing, additional scenes, and an independent checker.
