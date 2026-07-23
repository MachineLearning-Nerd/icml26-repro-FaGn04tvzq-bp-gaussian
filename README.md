# Reproduction: Gaussian emergence in sparse belief propagation

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/blob/master/notebooks/bp_gaussian_tutorial.py)

We tested all six claim groups from [*Belief Propagation Converges to Gaussian Distributions in Sparsely-Connected Factor Graphs* (arXiv:2601.21935)](https://arxiv.org/abs/2601.21935). The final Hugging Face `cpu-upgrade` run assessed **6/6 claims as aligned**.

The headline paper threshold is `KL < 0.02` within three hops. At hop three, observed mean KL was **0.001303** for the chain, **0.000826** for a real 127-node tree, and **0.002275** for a 4×4 loopy grid over 101 seeds. The Appendix-D boundary was **R*=6.016848** versus the paper's `R≈6`. On official Middlebury Cones at 150×200, BP MAP MSE was **21.781727 ± 0.003305** and GBP was **20.096043 ± 0.000000** (a **7.739%** gap); low-contrast mean KL was **0.012508**, versus **0.215593** at high-contrast edges.

The implementation is clean-room. Cones was resized from 375×450 to the paper's 150×200 setting and disparities were scaled horizontally; GBP used a mode-centered local Laplace projection because the authors' supplementary factor code was unavailable. The final comparison uses standard posterior-mode decoding for discrete BP and the equal mean/mode for GBP. A secondary 128-width Figure-4c sweep did not show the paper's low-KL transition and is reported as divergent in this implementation, separately from the aligned analytic boundary.

- [Read the illustrated claim-by-claim report](reports/bp-gaussian/report.md)
- [Explore the self-contained marimo tutorial](notebooks/bp_gaussian_tutorial.py)
- [Inspect the final immutable experiment branch](https://github.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/tree/orx/map-decoded-cones-comparison)

### Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/six-claim-clean-room-baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/tree/orx/six-claim-clean-room-baseline) | Frozen clean-room mechanism baseline | `python -m venv .venv && . .venv/bin/activate && python -m pip install --disable-pip-version-check numpy==2.3.2 && python repro/src/verify_bp.py` | Toy-scale baseline; established experiment contract | HF `cpu-upgrade`, 26s |
| [`orx/paper-scale-synthetic-claims-1-5`](https://github.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/tree/orx/paper-scale-synthetic-claims-1-5) | Appendix-I scale, 101 seeds, genuine tree, loopy computation tree, degree control | `python -m venv .venv && . .venv/bin/activate && python -m pip install --disable-pip-version-check numpy==2.3.2 && python repro/src/verify_bp.py` | Claims 1, 2, 3, and 5 aligned; empirical Claim-4 sweep divergent | HF `cpu-upgrade`, 47s |
| [`orx/real-middlebury-cones-bp-and-gbp`](https://github.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/tree/orx/real-middlebury-cones-bp-and-gbp) | Official Cones with static global-moment GBP projection | `python -m venv .venv && . .venv/bin/activate && python -m pip install --disable-pip-version-check numpy==2.3.2 && python repro/src/verify_bp.py` | Negative control: BP/GBP MSE `25.116/30.445`; low-contrast spatial KL aligned | HF `cpu-upgrade`, 5m02s |
| [`orx/mode-centered-laplace-gbp`](https://github.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/tree/orx/mode-centered-laplace-gbp) | Preserve the dominant photometric mode in GBP | `python -m venv .venv && . .venv/bin/activate && python -m pip install --disable-pip-version-check numpy==2.3.2 && python repro/src/verify_bp.py` | GBP MSE improved to `20.096`; exposed posterior-mean/MAP decoder asymmetry | HF `cpu-upgrade`, 4m56s |
| [`orx/map-decoded-cones-comparison`](https://github.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/tree/orx/map-decoded-cones-comparison) | Standard MAP decoding for discrete BP; expectation retained as sensitivity | `python -m venv .venv && . .venv/bin/activate && python -m pip install --disable-pip-version-check numpy==2.3.2 && python repro/src/verify_bp.py` | **6/6 aligned**; Cones BP/GBP relative MSE gap `7.739%` | HF `cpu-upgrade`, 4m19s |
| `master` | Public README, report, figures, notebook, and final source snapshot | Not run as an experiment (publication surface) | Presentation-only | None |

The provider's dollar charge was not available in the run logs.

---

## Upstream reproduction scaffold

OpenReview `FaGn04tvzq`. arXiv `2601.21935`. 6 claims/12 pts. Owner: loop12pt.
