# Final validation test report

Generated: 2026-09-28T16:08:06.262205+00:00

| Test | Status | Major | Detail |
|---|---|---:|---|
| required inputs found | PASS | True |  |
| temporal split rows disjoint | PASS | True |  |
| section IDs separated in held-section protocols | PASS | True |  |
| split-local dense reconstruction enabled | PASS | True |  |
| preprocessing fitted on training only | PASS | True |  |
| no test target used in fitting, imputation, corruption, or calibration | PASS | True |  |
| bootstrap is clustered and multiplicity preserved | PASS | True |  |
| same corruption realizations used across compared models | PASS | True |  |
| severity truth physical bounds | PASS | True |  |
| truth total equals disjoint sum | PASS | True |  |
| prediction physical bounds | PASS | True |  |
| target dates after anchors | PASS | True |  |
| first anchors are first survey | PASS | True |  |
| rolling anchors immediately precede target | PASS | True |  |
| stored and recomputed metrics agree | PASS | False |  |

Major failures: **0**.

The full experiment is permitted only when this count is zero. Metric reproduction uses the saved 277-row temporal prediction sample and a tolerance of 2×10⁻⁵.