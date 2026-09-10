# External Cross-Session Replication on Lee2019 Motor Imagery

## Pre-outcome protocol

This `EXTERNAL_CROSS_SESSION_REPLICATION` tests transportability of the accepted
participant-specific left-versus-right CSP+LDA architecture on a new corpus, not a
new model-development exercise. The frozen policy is
[`config/external_cross_session_lee2019.json`](../config/external_cross_session_lee2019.json).

The source is Lee et al.'s OpenBMI motor-imagery data (GigaScience 2019,
DOI: `10.1093/gigascience/giz002`; data DOI: `10.5524/100542`) accessed only through
MOABB's canonical `Lee2019_MI` adapter. Acquisition is pinned to its NEMAR
`nm000338` `sourcedata/` provider, which preserves the original upstream MAT
filenames and records an upstream SHA-256 and byte count for each source. This is an
acquisition-route decision made before outcomes, not a preprocessing or model choice.
It has 54 healthy participants, two sessions
on different days, 62 EEG and four EMG channels at 1000 Hz, and 100 balanced labelled
offline left/right hand MI trials per session. The separate online MI phase has no
labels through MOABB and is forbidden from formal evaluation.

The two folds train session 1 and test session 2, then reverse that direction. The
event marks the visual-arrow imagery cue after a 3-second fixation; the frozen
semantic interval is +1 to +3 seconds after that cue. EEG only is average-referenced
and filtered at native rate. Physical frequency boundaries remain 1–40 Hz for the
input and 8–30 Hz for CSP; native-rate FIR length and 2-second Welch sample counts
are necessary adapters, not tuning.

Source-only acquisition is resumable with
`python scripts/acquire_external_cross_session_lee2019.py`. It accepts a session
source only after its local byte count and SHA-256 match the NEMAR provenance
manifest; `.part` files are never treated as data. The command does not evaluate a
classifier or create scientific results.

The inherited pipelines are four-component empirical CSP with fold-local scaler and
equal-prior shrinkage LDA, and the C3/Cz/C4 six-feature 8–13/14–30 Hz log-PSD
baseline. Every learned stage is fit exclusively on the source session. Participants,
not directional evaluations, are the inferential unit. Results will be added only
after this frozen protocol, adapter tests, and a predeclared smoke subset succeed.
