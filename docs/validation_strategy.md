# Validation strategy

The project uses layered validation rather than treating a high classifier score as sufficient evidence.

## Data and physiology

- Raw inspection precedes filtering and epoching; invalid recording tails and incompatible annotations are explicit technical conditions.
- Trial-quality and feature-stability diagnostics are recorded rather than converted into undocumented visual decisions.
- Cohort replication distinguishes a predeclared primary task/rest result from later secondary analyses.
- Participant-level summaries and the documented sign/multiple-comparison procedures avoid using trials as independent participants.

## Prediction

- Held-out runs or held-out participants define the evaluation units for the completed decoder studies.
- CSP, scalers, LDA, and related learned stages are fitted on training data only; fit-audit CSVs store train/test identities.
- Frozen configs and artifact hashes test that parent methods and completed outputs have not silently drifted.
- Calibration studies preserve development/freeze/evaluation separation.

## External validation

The Lee2019 source gate requires all 108 files, expected sizes, SHA-256 checks, canonical adapter readability, two intended sessions, and valid labelled offline MI structure before any outcome is computed. The protocol prohibits target-session calibration and partial-cohort performance inspection. These safeguards reduce—but do not eliminate—risks from public historical data, offline evaluation, or unmeasured confounders.
