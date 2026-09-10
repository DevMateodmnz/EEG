# Collaboration

This independent project welcomes scientific review, reproducibility checks, methodological criticism, comparisons against alternative EEG pipelines, and discussion with EEG/BCI researchers. It is particularly useful to challenge the scope of sensor-level ERD interpretation, decoder split design, calibration conclusions, and external-validation plan.

Completed frozen results must not be silently overwritten. New hypotheses, preprocessing choices, model classes, or datasets should be introduced as separately versioned analyses with a clear development/freeze/evaluation chronology.

## Good first scientific extensions

1. Complete the already frozen Lee2019 cross-session source gate and evaluate it without changing the decoder.
2. After that result, choose a new, explicitly separated question: calibration burden versus session adaptation, depending on the external transfer evidence.
3. Reproduce selected completed analyses in another public motor-imagery corpus while preserving the same provenance and leakage audits.

This repository does not claim institutional affiliation or clinical expertise. Contact and attribution should use the project metadata in [CITATION.cff](../CITATION.cff).
