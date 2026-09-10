# Calibration quantity versus run diversity

This new outcome study follows `913d043`.  Earlier participants appear in historical
studies, but this exact fixed target-CSP comparison is newly frozen.  It asks whether
personalization benefits from more labels, more recording contexts, or both.

The fixed model is four-component Ledoit–Wolf CSP, target-training-only standard
scaling, and equal-prior shrinkage LDA.  `A` has 14 labels from one run, `B` has the
same 14 labels split across both available runs (4/3 and 3/4 variants), and `C` uses
all labels from both runs.  The held-out third run is never used for fitting.

More observations reduce covariance-estimation noise; more runs also sample changed
recording contexts.  Those are distinct sources of information even when their label
counts match.  The complete protocol is
[`config/calibration_quantity_and_diversity.json`](../config/calibration_quantity_and_diversity.json).

## Development and held-out results

Development subjects 1–20 verified the subset and variant-averaging mechanics
(median A/B/C BA: 0.602/0.569/0.658).  The final specification was committed at
`d85295b` before calculating the new outcomes on the same 86 eligible participants
used in earlier studies.  Those prior studies are historical context; only this
quantity/diversity comparison was newly locked.

| Condition | Median BA (IQR) | ≥0.60 | ≥0.70 | Fists / feet recall |
| --- | ---: | ---: | ---: | ---: |
| A: one run, 14 labels | 0.596 (0.525–0.680) | 47.7% | 20.9% | 0.595 / 0.643 |
| B: two runs, 14 labels | 0.595 (0.536–0.711) | 45.3% | 27.9% | 0.604 / 0.655 |
| C: two runs, all labels | 0.661 (0.560–0.778) | 66.3% | 43.0% | 0.690 / 0.688 |

| Comparison | Median gain (IQR) | Improved / worsened / tied | Holm p |
| --- | ---: | ---: | ---: |
| B − A | +0.005 (−0.035 to +0.055) | 48 / 35 / 3 | 0.143 |
| C − B | +0.036 (−0.010 to +0.092) | 57 / 27 / 2 | <0.001 |
| C − A | +0.048 (0.008–0.106) | 66 / 19 / 1 | exploratory total |

Thus diversity alone fails the frozen practical gate, while additional quantity
after diversity meets it (gain ≥0.02; 66.3% improved).  Calibration-to-held-out-run
direction cosine is low for all conditions (A/B/C median 0.132/0.102/0.138); C only
slightly exceeds A.  Direction magnitude is 2.198/2.881/2.148.  The performance gain
therefore is not compelling evidence that two-run calibration recovers a stable task
direction.  It is most consistent with using more labeled observations to improve
the participant-specific CSP covariance and LDA estimates.

## Decision

Choose **B: quantity matters after diversity.**  A future study should characterize
a larger multi-run calibration learning curve under a separately frozen protocol;
none is implemented here.

## Reproduction

```bash
python scripts/develop_calibration_quantity_and_diversity.py
python scripts/evaluate_calibration_quantity_and_diversity.py
python scripts/diagnose_calibration_quantity_geometry.py
```

## What you should understand now

- Fourteen labels from two runs do not automatically outperform fourteen labels from one run.
- More labels can improve CSP because its class covariance estimates and the LDA boundary become less variable.
- A better score does not by itself prove a more stable task direction; the held-out direction diagnostic remained low.
