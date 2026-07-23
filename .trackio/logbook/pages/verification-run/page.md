# Verification run

## Registered command and environment

```bash
python -m venv .venv && . .venv/bin/activate && python -m pip install --disable-pip-version-check numpy==2.3.2 && python repro/src/verify_bp.py
```

- Compute: Hugging Face `cpu-upgrade`
- Immutable branch: `orx/map-decoded-cones-comparison`
- Commit: `43ee5c73417181e0faf29be9223745d06e0af19a`
- Runtime: 4m19s (verifier-reported scientific elapsed time `245.900s`)
- Formal run: 101 synthetic seeds for Claims 1–4, 31 paired seeds for Claim 5, five Cones seeds for Claim 6

## Final output excerpt

```text
paper threshold at distance 3: chain=0.001303, tree=0.000826, grid=0.002275
computation-tree max |loopy-unwrapped| at depth 4: 0.000e+00
analytic Appendix-D boundary: u*=0.4991595572; R*=6.0168484964
paired-prefix seeds: positive slopes=1.000; median degree-10/2 ratio=3.260

BP MSE mean±sd=21.781727±0.003305
BP posterior-expectation sensitivity MSE mean±sd=25.116087±0.000002
GBP MSE mean±sd=20.096043±0.000000
BP/GBP relative gap=7.7390%
low-contrast mean KL=0.012508; edge mean KL=0.215593

claim_1_chain: ALIGNED
claim_2_tree: ALIGNED
claim_3_loopy: ALIGNED
claim_4_exclusion: ALIGNED
claim_5_degree: ALIGNED
claim_6_stereo: ALIGNED
aligned: 6/6
```

The provider did not expose a dollar charge. The data resize, disparity scaling, mode-centered Laplace GBP projection, and decoder sensitivity are explicit clean-room substitutions.
