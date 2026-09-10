# Riemannian covariance representation for motor-imagery decoding

[Back to the project README](../README.md) ·
[Zero-shot CSP decoder](cross_subject_decoding.md) ·
[Spatial-personalization study](spatial_personalization.md)

## Initial methodology freeze

This study starts from completed spatial-personalization commit `51dad94`. It
preserves the held-out within-subject CSP+LDA median of 0.658, zero-shot CSP+LDA
median of 0.570, one-run target-CSP median of 0.596, and target-CSP subspace
stability of 0.131. Those results, their predictions, and their methods are frozen.

The new question is:

> Does representing every trial by its full regularized 64-channel covariance
> geometry transfer better to an unseen participant than the frozen source CSP?

The pre-development policy is
[`config/riemannian_decoding.json`](../config/riemannian_decoding.json). Subjects
1–20 are the only method-development cohort. The 86 technically eligible subjects
21–109 remain locked until the development selection is separately committed.
Their recordings and several historical outcomes are not broadly untouched, but no
Riemannian zero-shot decoding outcome has been used to choose this method.

## One trial as a covariance matrix

The fixed input to this study is one 8–30 Hz task interval with shape `(64 channels,
320 samples)`, spanning +1 to +3 seconds. A covariance matrix summarizes how the
64 channel time series vary together:

\[
C = \frac{1}{T-1}(X-\bar X)(X-\bar X)^\top,
\]

where `X` is `(64, 320)`, `T = 320`, and `C` is `(64, 64)`. Diagonal entries describe
channel variance; off-diagonals describe co-fluctuation. Thus one covariance retains
the complete pairwise spatial pattern, rather than selecting four supervised CSP
components.

Average reference can make an ordinary empirical covariance singular because the
referenced channels sum to zero. The frozen Ledoit–Wolf estimator shrinks each trial
covariance toward a scaled identity matrix, making it symmetric positive-definite
(SPD). Fixed trace normalization removes only each trial's global covariance scale;
it preserves its full relative spatial structure and does not estimate any parameter
from another trial or participant.

## Why covariance needs its own geometry

SPD matrices do not form an unrestricted Euclidean vector space: adding or linearly
interpolating two valid covariances can leave the SPD set or distort their natural
relative differences. The affine-invariant Riemannian distance between SPD matrices
`A` and `B` is

\[
d(A,B)=\left\|\log\left(A^{-1/2}BA^{-1/2}\right)\right\|_F.
\]

It compares one covariance after expressing the other relative to it. A training-only
Riemannian (Karcher) mean supplies a reference point `R`. Each covariance maps to a
local linear tangent coordinate through

\[
S=\log\left(R^{-1/2} C R^{-1/2}\right).
\]

The upper triangle of `S` becomes 2,080 features; off-diagonal values receive a
`\sqrt{2}` weight so ordinary dot products match the matrix Frobenius inner product.
A conventional linear classifier can then operate in this local coordinate system.
This is the tangent-space approach described for motor-imagery BCI by Barachant and
colleagues ([2012](https://pubmed.ncbi.nlm.nih.gov/22010143/)); the implementation
uses the documented [`pyRiemann` tangent-space estimator](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html).

## How this differs from CSP

CSP uses class labels to learn a small set of sensor filters that maximize class
variance contrast. It is supervised spatial compression. The Riemannian pipeline
first constructs an unsupervised SPD covariance for every trial, keeps all 64-channel
relationships, then uses labels only in the final linear classifier. The reference
mean is still learned, so it must be fitted solely from training participants.

```text
8–30 Hz trial EEG (64 × 320)
        ↓ Ledoit–Wolf covariance + fixed trace normalization
SPD covariance (64 × 64)
        ↓ training-only affine-invariant Riemannian mean and log map
tangent vector (2,080 features)
        ↓ training-only scaling and linear classifier
fists / feet prediction
```

For every held-out participant, no EEG—labeled or unlabeled—may affect the covariance
reference, tangent basis, scaler, classifier, or classifier selection. Tangent-space
updating and target covariance alignment are explicitly disabled.

## Frozen development and evaluation design

Development is leave-one-subject-out over subjects 1–20. Each fold fits all learned
stages using 855 trials from 19 people and predicts all 45 trials from the twentieth.
The only development choice is between training-only scaled shrinkage LDA and fixed
L2 logistic regression. Logistic regression is selected only if its participant-
matched median gain is at least +0.01 and it wins for at least 60% of development
participants; otherwise the simpler shrinkage LDA is retained.

After a final freeze, one model will be fitted once on 900 trials from subjects 1–20
and applied independently to each of the 86 eligible evaluation participants. Each
participant contributes the unweighted mean of run 6, 10, and 14 balanced accuracy;
trials are not treated as independent group observations.

A clear improvement over historical zero-shot CSP requires a Riemannian median of at
least 0.60, median paired gain at least +0.03, at least 60% improved, supporting
one-sided exact-sign `p ≤ 0.05`, balanced median class recall, QC robustness, and no
materially worse participant run range. Smaller but directionally consistent gains
have a separately frozen modest-improvement category. The full criteria are in the
machine-readable policy.

At this freeze, no Riemannian classifier result has been calculated for either
development or evaluation subjects. Development evidence, final selection, held-out
results, diagnostics, limitations, and reproduction commands will be added only after
their corresponding scientific boundaries are respected.

## Development evidence and classifier selection

After the initial protocol and participant-isolated implementation were committed,
subjects 1–20 completed the planned 20-fold leave-one-subject-out comparison. Each
held-out person supplied 45 predictions per candidate and one unweighted mean of
their three run balanced accuracies. The 19-person training side supplied 855 trial
covariances for the Riemannian reference, tangent map, scaler, and classifier.

| Linear head | Median balanced accuracy | 95% bootstrap median CI | IQR | >0.50 | ≥0.60 | Median fists/feet recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shrinkage LDA | 0.580 | 0.536–0.619 | 0.532–0.624 | 17/20 | 8/20 | 0.639 / 0.568 |
| L2 logistic regression | 0.595 | 0.540–0.618 | 0.522–0.629 | 16/20 | 8/20 | 0.652 / 0.557 |

The participant-matched logistic-minus-LDA median is +0.012 (IQR −0.024 to
+0.042), but logistic wins for only 10/20 participants, LDA wins for nine, and one
ties. The pre-frozen rule requires both a +0.01 median gain and logistic wins for at
least 60% of participants. That consistency condition fails (50%), so the simpler
**training-only StandardScaler plus shrinkage LDA** is selected. No evaluation EEG or
Riemannian outcome informed this choice.

The machine-readable [model summary](assets/riemannian_decoder_development_model_summary.csv),
[participant scores](assets/riemannian_decoder_development_subject_scores.csv),
[fit audit](assets/riemannian_decoder_development_fit_audit.csv), and
[method selection](assets/riemannian_decoder_development_method_selection.csv)
record the development boundary and result.

## Final method freeze before evaluation

The final configuration is
[`config/riemannian_decoding_final.json`](../config/riemannian_decoding_final.json).
It freezes the initial policy, Riemannian implementation, all development artifacts,
and the shrinkage-LDA selection before any Riemannian prediction for subjects 21–109.
The model will fit one 900-trial training-only Riemannian reference and tangent space,
then predict each eligible participant independently. No source CSP, historical model,
or target EEG adaptation will be refitted.

## Final held-out evaluation

The final freeze was committed at `c2f132f` before the new outcomes were opened. One
Riemannian reference, scaler, and shrinkage LDA were fitted on 900 trials from
subjects 1–20, then applied unchanged to all 3,840 trials from 86 eligible unseen
participants. Every trial has exactly one preserved prediction.

| Model | Median BA (95% bootstrap CI) | IQR | Range | >0.50 | ≥0.60 | ≥0.70 | Median fists/feet recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Riemannian Ledoit–Wolf tangent LDA | 0.573 (0.549–0.577) | 0.519–0.631 | 0.372–0.863 | 65/86 | 28/86 | 10/86 | 0.591 / 0.600 |
| Historical zero-shot CSP+LDA | 0.570 | 0.505–0.641 | 0.414–1.000 | 65/86 | 32/86 | 14/86 | 0.762 / 0.600 |
| Historical spectral-LDA | 0.536 | 0.474–0.598 | 0.393–0.786 | 55/86 | 21/86 | 4/86 | 0.542 / 0.571 |

Riemannian minus CSP is **−0.024** (IQR −0.094 to +0.048): 33 participants improve,
52 worsen, and one ties (one-sided exact-sign `p = 0.985`). The QC-sensitive matched
difference is −0.027. Thus Riemannian covariance geometry does not meet even the
frozen modest-improvement gate. It is descriptively better than spectral-LDA by
+0.030 (49 improved, 35 worsened, two ties), but that does not rescue the primary
comparison against CSP.

Run medians are 0.562 (run 6), 0.571 (run 10), and 0.576 (run 14). Riemannian
participant run range is 0.174 versus historical CSP 0.138: variability is not
reduced.

## Post-primary diagnostics and decision

Among 21 participants with historical CSP BA at or below 0.50, the Riemannian-minus-
CSP median is +0.065 and 17 improve. This pre-defined poor-performance subgroup is
descriptive and susceptible to regression to the mean; it cannot justify target
selection or revise the failed cohort-wide primary gate.

The strongest supported interpretation is **no representation improvement**: retaining
the full regularized covariance did not outperform CSP's supervised spatial contrast
in strict zero-shot transfer, and it increased participant run variability. The
next decision is **C: no improvement and substantial inter-participant/domain
variability; study domain variability before additional decoding methods.** No target
alignment, calibration, or new model family is implemented here.

## Reproduction and verification

```bash
python scripts/develop_riemannian_decoder.py
python scripts/evaluate_riemannian_decoder.py
python -m unittest tests.test_riemannian_decoding_freeze tests.test_riemannian_decoding tests.test_riemannian_decoding_development_artifacts tests.test_riemannian_decoding_final_freeze tests.test_riemannian_decoding_evaluation_artifacts -v
```

The evaluation reuses validated 8–30 Hz target tensors, stores compact `(trials,64,64)`
SPD covariance checkpoints under `outputs/riemannian_decoder_checkpoints/`, validates
all source/config/code/array hashes, and leaves historical CSP and spectral outputs
unchanged. The main machine-readable artifacts are the [group summary](assets/riemannian_decoder_evaluation_group_summary.csv), [paired results](assets/riemannian_decoder_evaluation_paired_summary.csv), [predictions](assets/riemannian_decoder_evaluation_predictions.csv), and [fit audit](assets/riemannian_decoder_evaluation_fit_audit.csv).

## What you should understand now

- A covariance matrix describes a trial's complete channel-by-channel variation, not individual neurons or a brain source.
- SPD geometry uses a training-only Riemannian reference before a linear classifier sees tangent coordinates.
- A geometry-aware representation can be valid and still fail to transfer better than a supervised CSP representation.
- The poor-CSP subgroup result is exploratory; the participant-wide matched comparison is the decision evidence.
