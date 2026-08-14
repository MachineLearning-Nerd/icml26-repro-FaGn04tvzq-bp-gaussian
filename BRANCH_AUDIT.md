# Branch audit

## Final branch vocabulary

The intended final remote contains seven purpose-based branches:

| Final branch | Historical source ref | Responsibility |
| --- | --- | --- |
| main | master | Canonical README, reports, gate metadata, and public snapshot |
| baseline/clean-room | orx/six-claim-clean-room-baseline | Initial six-claim clean-room mechanism baseline |
| research/paper-scale-synthetic | orx/paper-scale-synthetic-claims-1-5 | Appendix-I scale synthetic claims and controls |
| research/normalized-prior-variance | orx/normalized-prior-variance-gaussian-fit | Claim-4 normalized-variance sweep |
| research/middlebury-cones-control | orx/real-middlebury-cones-bp-and-gbp | Static-moment Cones negative control |
| research/mode-centered-gbp | orx/mode-centered-laplace-gbp | Mode-centered Laplace GBP ablation |
| release/map-decoded-cones | orx/map-decoded-cones-comparison | Registered Cones comparison and release snapshot |

Historical names are listed for provenance only. They are not intended to
remain as public remote refs after cleanup.

## Identity policy

All approved commits use:

```text
MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>
```

No co-author or tool-signature lines are added. History normalization covers
every commit reachable from the seven final branches.

## Required remote invariants

- Owner: MachineLearning-Nerd
- Repository: icml26-belief-propagation-gaussian-convergence
- Default branch: main
- Public branch count: 7
- Legacy refs absent: master and all orx/* refs
- README, STATUS.md, SOURCE_MANIFEST.md, AUDIT_REPORT.md,
  publication_gate.json, and BRANCH_AUDIT.md are present on main
- Every final branch is pushed and readable

Remote verification is recorded here after the rename and final push.
