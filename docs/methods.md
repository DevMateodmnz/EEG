# Methods

This canonical overview points to the frozen, study-specific methods rather than replacing them. The completed primary work uses PhysioNet EEGMMIDB 64-channel, 160 Hz recordings and the relevant imagery runs documented in [Dataset](dataset.md), [preprocessing](preprocessing.md), [epoching](epoching.md), [event-related spectral analysis](event_related_spectral_analysis.md), and [full-cohort replication](full_cohort_replication.md).

## Signal and trial pipeline

Recordings are inspected before transformation. The pipeline identifies the recording-valid interval, handles the documented constant tail rather than treating it as EEG, applies the frozen average-reference/FIR settings, maps task annotations to epochs, and records trial quality rather than silently discarding observations. FIR choices, passbands, timing, and edge handling are configuration-controlled; see the linked methods documents for exact units and parameters.

Physiological summaries compare task and reference periods with power expressed as documented by each frozen study. C3/C4 are emphasized because they are conventional central sensor locations relevant to hand motor imagery, not because scalp sensors identify a unique neural source.

## Decoding

Completed decoder studies use predeclared spectral baselines and CSP+LDA variants. Learned transforms are fit only on the training portion of the relevant held-out-run/person split; fit-audit artifacts record source and evaluation identities. Balanced accuracy is used when class balance matters. Calibration studies keep development, method freeze, and evaluation distinct. Exact study designs belong in [within-subject decoding](within_subject_decoding.md), [cross-subject decoding](cross_subject_decoding.md), and [minimal target calibration](minimal_target_calibration.md).

## External protocol

Lee2019 remains methodological only: native-rate 62-channel EEG, labelled offline left/right MI, +1 to +3 s after the cue, physical 1–40 Hz input and 8–30 Hz CSP band, and source-session-only fitting. See [external validation](external_validation.md).
