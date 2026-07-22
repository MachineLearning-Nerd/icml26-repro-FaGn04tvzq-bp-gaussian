# Reproduction: Gaussian emergence in sparse belief propagation

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/blob/master/notebooks/bp_gaussian_tutorial.py)

We tested all six claim groups from [*Belief Propagation Converges to Gaussian Distributions in Sparsely-Connected Factor Graphs* (arXiv:2601.21935)](https://arxiv.org/abs/2601.21935) with a clean-room, discretized NumPy implementation. On Hugging Face `cpu-upgrade`, the chain's KL divergence to its best-fit Gaussian fell from **0.6117 to 0.0161** over six hops and the verifier reported **6/6 checks passed**.

The scientific assessment is **partially aligned overall**. The central direction is clear, but the paper's headline `KL < 0.02 within 3 hops` was not reached by the chain at hop 3 (`0.0327`); it first crossed at hop 5. The run used a 90-bin deterministic synthetic graph instead of the paper's 1,024-bin multi-seed setup, its tree is a symmetric path simplification, and Figure 5 was tested only through a synthetic smooth-versus-edge proxy—not the 30,000-variable Middlebury Cones BP/GBP comparison.

- [Read the illustrated claim-by-claim report](reports/bp-gaussian/report.md)
- [Explore the self-contained marimo tutorial](notebooks/bp_gaussian_tutorial.py)
- [Inspect the immutable experiment branch](https://github.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/tree/orx/six-claim-clean-room-baseline)

### Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/six-claim-clean-room-baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-FaGn04tvzq-bp-gaussian/tree/orx/six-claim-clean-room-baseline) | Frozen clean-room suite for chain, tree, loopy, exclusion, degree/rate, and stereo-proxy claims | `python -m venv .venv && . .venv/bin/activate && python -m pip install --disable-pip-version-check numpy==2.3.2 && python repro/src/verify_bp.py` | 6/6 programmed checks; claim-level assessment: 2 aligned, 3 partially aligned, 1 inconclusive under this setup | Hugging Face `cpu-upgrade`, Python 3.12, 26 s |
| `master` | README, report, figures, and notebook | Not run as an experiment (publication surface) | Presentation-only | None |

The provider's dollar charge was not available in the run log. The report separates paper evidence, observed evidence, substitutions, and work still needed for a full-scale reproduction.

---

## Upstream reproduction scaffold

OpenReview `FaGn04tvzq`. arXiv `2601.21935`. 6 claims/12 pts. Owner: loop12pt.
