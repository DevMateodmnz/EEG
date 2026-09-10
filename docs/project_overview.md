# Project overview and scientific chronology

This is the canonical English overview. Spanish readers can use the [adapted overview](project_overview_es.md); frozen figures and tables are shared.

## Scope

The project investigates reproducible, sensor-level motor-imagery EEG analysis using public data. It separates physiological task/rest analysis from predictive decoding: an ERD-like power reduction and a classifier score answer different questions.

## Completed / frozen evidence

The primary completed evidence is the PhysioNet EEG Motor Movement/Imagery Database work using the motor-imagery fists-versus-feet runs. Development began with subject 1, was replicated in participants 2–20, and expanded to the 2–109 cohort. Subjects 88, 92, and 100 are recorded as **EXCLUDED — TECHNICAL** because their sampling/annotation structure was incompatible with the frozen pipeline; they were not excluded for outcomes.

The frozen full-cohort report documents task-versus-rest 12–13 Hz reductions at C3/C4. The completed decoding work evaluates spectral and CSP+LDA pipelines with run-level held-out data, then studies cross-subject transfer, calibration, spatial personalization, and Riemannian covariance features. The integrated account is in the [final scientific report](final_scientific_report.md); individual studies retain their own protocol/result chronology.

## Secondary and negative evidence

Individualized-frequency, calibration, geometry, spatial, and Riemannian analyses are retained because they constrain interpretation. In particular, individualized frequency did not automatically improve the fixed marker, cross-participant decoding was weaker than participant-specific decoding, and small adaptations did not eliminate that gap. These are not hidden failures: they identify the limits of the fixed classical pipeline.

## External validation — IN PROGRESS

The Lee2019/OpenBMI study is a pre-outcome, participant-specific left-versus-right cross-session protocol. It uses only labelled offline MI EEG, trains session 1→2 and 2→1, and forbids target-session calibration. NEMAR source acquisition is resumable and SHA-256-gated. There are no Lee2019 performance outcomes, exclusions, or result figures yet. See [external validation](external_validation.md).
