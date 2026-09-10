# Offset versus task-direction alignment

This new adaptation-outcome study follows the completed geometry analysis at
`6826b67`: that study found associations, but associations cannot show that forcing
alignment improves predictions.  This experiment tests that engineering hypothesis
without changing the frozen Riemannian source classifier.

The initial policy is [`config/domain_vs_direction_alignment.json`](../config/domain_vs_direction_alignment.json).
Subjects 1–20 are used for development only; the 86 eligible evaluation subjects
remain locked until the final adaptation specification is committed.  One balanced
calibration run supplies its first seven fists and first seven feet trials.  The
other two runs are always the test set.

`B` translates tangent vectors by `z - m_target + m_source`, a label-free
recentering.  `C` additionally applies the unique minimal-plane orthogonal rotation
that maps the normalized labeled target task direction to the frozen source
direction.  Rotation preserves lengths and inner products; it does not change task
effect magnitude.  Neither transform may use test-run EEG to estimate its parameters.

## Development and final freeze

Development used leave-one-subject-out source models on subjects 1–20 and the exact
three calibration-run scenarios.  Its median BA was 0.579 for A, 0.568 for B, and
0.558 for C.  These results selected no transformation: they only verified the
implementation and fixed the practical success gate (median gain ≥0.02 and ≥60%
improved).  The final method was committed at `e346074` before evaluation outcomes
were calculated.

## Held-out adaptation results

All 86 eligible participants were evaluated from the existing covariance
checkpoints.  The source tangent reference, StandardScaler, and shrinkage LDA were
fitted once from development data and never changed.  Each condition uses identical
held-out trials and each participant contributes the equal mean of three scenarios.

| Condition | Median BA (IQR) | ≥0.60 | ≥0.70 | Median fists / feet recall |
| --- | ---: | ---: | ---: | ---: |
| A: frozen zero-shot | 0.571 (0.519–0.634) | 31.4% | 11.6% | 0.590 / 0.600 |
| B: offset-only | 0.588 (0.535–0.661) | 44.2% | 18.6% | 0.600 / 0.571 |
| C: offset + direction rotation | 0.540 (0.494–0.589) | 23.3% | 8.1% | 0.548 / 0.583 |

| Paired comparison | Median gain (IQR) | Improved / worsened / tied | Holm p |
| --- | ---: | ---: | ---: |
| B − A | +0.003 (−0.024 to +0.063) | 45 / 38 / 3 | 0.125 |
| C − B | −0.039 (−0.109 to +0.023) | 30 / 56 / 0 | <0.001 |
| C − A | −0.017 (−0.096 to +0.037) | 35 / 50 / 1 | 0.125 |

Offset correction increases the descriptive median but misses both frozen practical
criteria and corrected inference.  Direction correction is reliably harmful.  The
conclusion is not that geometry was estimated incorrectly: translation reduced the
held-out source-midpoint distance from 7.235 to 3.046.  In contrast, the held-out
test-run direction alignment barely changed (median 0.140 to 0.142), because a
single calibration-run direction does not reliably represent the other runs.

Calibration-run medians are A: 0.567/0.571/0.567, B: 0.567/0.599/0.600, and C:
0.545/0.536/0.536 for calibration runs 6/10/14.  This scenario variability further
argues against selecting a favorable run.  Existing QC had been assessed in the
frozen decoder work; no outcome-driven additional QC subgroup was introduced.

## Decision gate

This selects **E: neither geometry nor decoding improves reliably**.  Recentring
does what it is designed to do geometrically but does not translate into a practical
decoding benefit, and one-run supervised direction rotation harms performance.  The
next study should test calibration quantity and run diversity under a new frozen
protocol; it is not implemented here.

## Reproduction

```bash
python scripts/develop_domain_vs_direction_alignment.py
python scripts/evaluate_domain_vs_direction_alignment.py
python -m unittest tests.test_domain_vs_direction_alignment tests.test_domain_vs_direction_alignment_artifacts -v
```

## What you should understand now

- Tangent-space translation removes a mean location difference but cannot guarantee that a source classifier boundary transfers.
- A minimal orthogonal rotation preserves geometry, yet can still harm predictions when its calibration direction is not stable across runs.
- The prior alignment–accuracy correlation was an observational association; this intervention tested and rejected its simple causal-engineering interpretation.
- A participant-level paired comparison, not individual trials, determines whether the adaptation is useful.
