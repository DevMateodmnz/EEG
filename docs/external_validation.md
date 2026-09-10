# External cross-session validation — IN PROGRESS

This is a protocol/status document, not a results report. The frozen study asks whether the historical participant-specific left-versus-right CSP+LDA decoder transfers between two independent recording-day sessions in the Lee2019/OpenBMI corpus without target-session calibration.

The protocol fixes: MOABB `Lee2019_MI`; 54 candidate participants; two sessions; labelled offline MI only; 62 EEG channels (no EMG features); native-rate processing; the +1 to +3 s post-cue interval; a 1–40 Hz input range; 8–30 Hz CSP; the historical four-component CSP+LDA and C3/Cz/C4 spectral-LDA baseline; balanced accuracy; and participant-level inference. Each direction trains 1→2 and 2→1. All learned stages must fit the source session only.

The pre-outcome NEMAR provider is `nm000338`. Every required source must pass manifest byte-size and SHA-256 validation before canonical loading, smoke evaluation, or cohort scoring. The resumable source-only command is:

```bash
MOABB_DOWNLOAD_PROVIDER=nemar python scripts/acquire_external_cross_session_lee2019.py --data-dir data
```

The current source set is incomplete, therefore no external scientific outcome, exclusion, figure, or claim exists. The implementation and tests are retained to make the future evaluation auditable. See the detailed [frozen protocol](external_cross_session_lee2019.md).
