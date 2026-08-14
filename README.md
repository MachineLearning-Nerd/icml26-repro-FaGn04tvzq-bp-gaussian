# ICML 2026 reproduction: Belief Propagation and Gaussian convergence

This repository contains a clean-room NumPy reproduction audit for:

> **Belief Propagation Converges to Gaussian Distributions in Sparsely-Connected Factor Graphs**

- Paper: [arXiv:2601.21935](https://arxiv.org/abs/2601.21935)
- Review record: [OpenReview FaGn04tvzq](https://openreview.net/forum?id=FaGn04tvzq)
- Authors: Tom Yates, Yuzhou Cheng, Ignacio Alzugaray, Danyal Akarca, Pedro A. M. Mediano, and Andrew J. Davison
- Repository owner: [MachineLearning-Nerd](https://github.com/MachineLearning-Nerd)

The arXiv record identifies this as a preprint under review. This repository
does not claim acceptance or authorship; it records which finite experiments
were run, how each result was produced, and where the implementation differs
from the paper's source-exact setup.

## Status at a glance

| Gate | Status |
| --- | --- |
| Evidence-release gate | **PASSED** |
| Overall reproduction status | **MIXED_RESULTS** |
| Strict universal paper-claim gate | **NOT_READY** |
| Claim groups | C1, C2, C3, C5: **VERIFIED_SCOPED**; C4: **MIXED_RESULTS**; C6: **VERIFIED_SCOPED** |
| Registered snapshot | 6/6 claim groups labeled aligned by the original run; this README preserves the disclosed C4 divergence and C6 implementation boundary |

The aggregate `6/6 aligned` label must not be read as six source-exact
proofs. Claim 4 has an aligned analytic boundary but a divergent substituted
finite-grid sweep. Claim 6 meets the registered `<10%` BP/GBP MSE criterion,
but its factor implementation and preprocessing are clean-room substitutions.

## What the paper studies

The paper asks when repeated message updates in sparse, loopy factor graphs
make non-Gaussian beliefs approximately Gaussian. Its proposed mechanism is
that pairwise BP updates convolve distributions along paths, while branching
multiplies incoming messages and can preserve non-Gaussian structure. The
paper combines theoretical conditions with synthetic topology experiments and
a stereo-depth application.

This audit covers the six claim groups represented in the repository:

1. chain messages become Gaussian with path distance;
2. the effect extends to a genuinely branching tree;
3. loopy BP follows an equivalent computation-tree construction;
4. confident priors create the Appendix-D exclusion boundary near `R≈6`;
5. increasing star degree without path depth increases center non-Gaussianity;
6. GBP and BP have similar Cones disparity error, while low-contrast beliefs
   are more Gaussian than edge beliefs.

## Claim ledger

| Claim | Paper target | How the result is produced | Observed result and status |
| ---: | --- | --- | --- |
| C1 | Chain Gaussianization and the `KL < 0.02` threshold by hop 3 | `repro/src/verify_bp.py` calls the 1,024-bin chain routines in `repro/src/paper_scale.py` for seeds 42–142; destructive convolution and Gaussian-closure controls run in the same verifier | Mean KL `0.34412 → 0.001303` by hop 3 and `0.00024` by hop 6; **VERIFIED_SCOPED** |
| C2 | Gaussianization in a branching tree | The verifier builds a 127-variable binary tree with 64 distinct leaf priors and computes distance-binned KL values | Mean KL at distance 3 is `0.000826`; **VERIFIED_SCOPED** |
| C3 | Computation-tree explanation for loopy BP | The verifier runs a 4×4 synchronous grid and explicitly compares messages with a depth-4 unwrapped computation tree | Hop-3 mean KL `0.002275`; maximum direct/unwrapped message difference `0`; **VERIFIED_SCOPED** |
| C4 | Appendix-D confident-prior boundary and Figure 4c transition | `verify_bp.py` solves the analytic coefficient equation and runs a separate 101-seed, 128-width normalized-variance sweep | Analytic `u*=0.4991595572`, `R*=6.0168484964` aligns; finite-grid means `0.080769` vs `0.081358` show no `<0.02` transition; **MIXED_RESULTS** |
| C5 | Degree increases non-Gaussianity when path depth is zero | The verifier uses paired-prefix star graphs for degrees 2–10 over 31 seeds and a Gaussian-incoming control | Mean center KL `0.06618 → 0.18719`; `31/31` paired slopes positive; **VERIFIED_SCOPED** |
| C6 | Cones BP/GBP accuracy and spatial Gaussianity | `repro/src/stereo_cones.py` constructs the 150×200 clean-room stereo audit; `verify_bp.py` runs five seeds and computes MSE/KL summaries | BP MAP MSE `21.781727±0.003305`, GBP `20.096043`, gap `7.739%`; low/high-contrast mean KL `0.012508/0.215593`; **VERIFIED_SCOPED under the registered `<10%` criterion** |

## Claim producers and evidence paths

The canonical end-to-end producer is:

```bash
python repro/src/verify_bp.py
```

It writes the registered snapshot to `outputs/verdict.json` and the textual
run record to `outputs/verify_run.log`. The main production paths are:

- `repro/src/paper_scale.py` — normalized-grid probability vectors,
  convolutional messages, exact tree BP, loopy-grid BP, KL diagnostics, and
  synthetic controls for C1–C5.
- `repro/src/stereo_cones.py` — photometric priors, edge-masked pairwise
  factors, non-parametric BP, information-form GBP, and Cones region metrics
  for C6.
- `repro/src/verify_bp.py` — fixed seeds, thresholds, claim predicates,
  summary serialization, and the final six-claim verdict.
- `reports/bp-gaussian/make_figures.py` — report figures regenerated from
  the recorded evidence.
- `notebooks/bp_gaussian_tutorial.py` — reader-facing explanation; it is not
  the authoritative verifier.

There is no separate source-exact supplementary implementation or independent
CSV checker in this repository. The verifier, committed outputs, report, and
explicit controls are the evidence boundary; `AUDIT_REPORT.md` records that
limitation instead of treating the clean-room implementation as the authors'
code.

## Reproduce the registered run

The original evidence run used a Hugging Face `cpu-upgrade` CPU runner,
Python 3, and NumPy `2.3.2`:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --disable-pip-version-check numpy==2.3.2
python repro/src/verify_bp.py
```

The registered run took about 4 minutes 19 seconds. The real-data path uses
the official Middlebury Cones pair, resized from 375×450 to 150×200 with
horizontally scaled disparity labels. Reproducing C6 therefore requires the
data acquisition described in `SOURCE_MANIFEST.md`; do not silently replace
it with a proxy scene.

The committed historical experiment snapshot was
`43ee5c73417181e0faf29be9223745d06e0af19a` on the old
`orx/map-decoded-cones-comparison` ref. After normalization, that snapshot is
represented by `release/map-decoded-cones`; the branch audit records the final
public vocabulary.

## Final branch guide

The final remote vocabulary contains seven purpose-based branches. Historical
names are retained only as provenance:

| Final branch | Historical source ref | Responsibility |
| --- | --- | --- |
| [main](https://github.com/MachineLearning-Nerd/icml26-belief-propagation-gaussian-convergence/tree/main) | `master` | Canonical README, reports, gate metadata, and public snapshot |
| [baseline/clean-room](https://github.com/MachineLearning-Nerd/icml26-belief-propagation-gaussian-convergence/tree/baseline/clean-room) | `orx/six-claim-clean-room-baseline` | Initial six-claim clean-room mechanism baseline |
| [research/paper-scale-synthetic](https://github.com/MachineLearning-Nerd/icml26-belief-propagation-gaussian-convergence/tree/research/paper-scale-synthetic) | `orx/paper-scale-synthetic-claims-1-5` | Appendix-I scale synthetic claims and controls |
| [research/normalized-prior-variance](https://github.com/MachineLearning-Nerd/icml26-belief-propagation-gaussian-convergence/tree/research/normalized-prior-variance) | `orx/normalized-prior-variance-gaussian-fit` | Claim-4 normalized-variance sweep |
| [research/middlebury-cones-control](https://github.com/MachineLearning-Nerd/icml26-belief-propagation-gaussian-convergence/tree/research/middlebury-cones-control) | `orx/real-middlebury-cones-bp-and-gbp` | Static-moment Cones negative control |
| [research/mode-centered-gbp](https://github.com/MachineLearning-Nerd/icml26-belief-propagation-gaussian-convergence/tree/research/mode-centered-gbp) | `orx/mode-centered-laplace-gbp` | Mode-centered Laplace GBP ablation |
| [release/map-decoded-cones](https://github.com/MachineLearning-Nerd/icml26-belief-propagation-gaussian-convergence/tree/release/map-decoded-cones) | `orx/map-decoded-cones-comparison` | Registered Cones comparison and release snapshot |

Branch responsibilities and the final remote checks are repeated in
`BRANCH_AUDIT.md`.

## Repository map

- `repro/src/` — source implementation and the fixed verifier.
- `outputs/` — result snapshot plus run log; provenance is documented in
  `outputs/README.md`.
- `reports/bp-gaussian/` — claim-by-claim report and figures.
- `notebooks/` — self-contained tutorial.
- `.trackio/logbook/` — reader-facing experiment log.
- `SOURCE_MANIFEST.md` — paper, data, environment, and source anchors.
- `AUDIT_REPORT.md` — claim boundaries, controls, substitutions, and open gaps.
- `STATUS.md` — short current-status card.
- `publication_gate.json` and `GATE_READY.md` — machine-readable and human
  readable release-gate records.

## Citation

```bibtex
@misc{yates2026belief,
  title         = {Belief Propagation Converges to Gaussian Distributions in Sparsely-Connected Factor Graphs},
  author        = {Tom Yates and Yuzhou Cheng and Ignacio Alzugaray and Danyal Akarca and Pedro A. M. Mediano and Andrew J. Davison},
  year          = {2026},
  eprint        = {2601.21935},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC}
}
```

Please cite the paper using the official arXiv record above. This repository
is a reproduction audit, not a substitute for the paper or its authors'
supplementary materials.

## Thank you

Thank you to Tom Yates, Yuzhou Cheng, Ignacio Alzugaray, Danyal Akarca,
Pedro A. M. Mediano, and Andrew J. Davison for making the question, theory,
figures, and experimental targets available for independent study. The
clean-room implementation and the qualifications in this repository are
intended to make the work easier to inspect and reproduce, while preserving
credit for the original ideas.

Maintained by `MachineLearning-Nerd` with authorship and claim boundaries
kept explicit.
