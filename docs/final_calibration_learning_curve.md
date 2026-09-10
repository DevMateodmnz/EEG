# Final calibration learning curve

This is the final empirical study for project v1.0.  It follows the completed
quantity-versus-diversity result and changes only the number of balanced labeled
target trials: 14, 18, 22, 26, and 28.  Each subset is deterministically nested
within two available calibration runs; complementary allocations are averaged before
participant-level inference.  The fixed four-component Ledoit–Wolf CSP, target
scaler, and shrinkage LDA are unchanged.  The policy is
[`config/final_calibration_learning_curve.json`](../config/final_calibration_learning_curve.json).

The smallest near-full burden must be within 0.01 of the 28-label cohort median,
no worse than −0.01 participant-matched, have a subsequent median increment below
0.01, retain reasonably balanced recall, and not be minority-driven.  This rule was
written before evaluation outcomes.

## Held-out curve

The freeze commit `bcbd7a9` precedes all 86 evaluation outcomes. Median BA (IQR)
was 0.595 (0.536–0.711), 0.623 (0.548–0.740), 0.622 (0.571–0.769), 0.647
(0.576–0.770), and 0.650 (0.563–0.774) at 14/18/22/26/28 labels. The 18−14
median gain was +0.018 (54/28/4 improved/worsened/tied; Holm `p<0.001`). Later
increments were +0.005, 0.000, and 0.000.

No pre-28 size meets every near-full rule: 18 is within 0.01 of the 28-label
median and has a sub-0.01 next increment, but its matched median difference versus
28 is −0.0104, slightly below the frozen −0.01 boundary. The conservative decision
is **B: late saturation**—approximately 26–28 balanced labels/two substantial runs
are supported; no shorter sufficient burden is established.
