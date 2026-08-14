# Output provenance

verdict.json and verify_run.log are the committed output snapshot from the
registered final experiment. They are evidence artifacts, not a promise that
the run is source-exact or that a later rerun will produce bit-identical
floating-point values.

- Historical source branch: orx/map-decoded-cones-comparison
- Historical source commit: 43ee5c73417181e0faf29be9223745d06e0af19a
- Normalized public branch: release/map-decoded-cones
- Backend: Hugging Face cpu-upgrade
- Environment: Python 3 with NumPy 2.3.2
- Canonical producer: python repro/src/verify_bp.py

The output keeps the original 6/6 aligned run label. Read it together with
AUDIT_REPORT.md: C4 has a finite-grid divergence, and C6 is bounded by the
registered criterion and clean-room substitutions.
