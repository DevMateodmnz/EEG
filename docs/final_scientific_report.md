# Replicated motor-imagery EEG physiology and limits of cross-participant decoding

## Abstract

PhysioNet EEGMMIDB runs 6/10/14 replicated central fists-related 12–13 Hz
task-versus-rest reductions (87/105 C4 and 91/105 C3 participants negative) and
supported leakage-safe fists-versus-feet decoding. Within-person CSP+LDA reached
median BA 0.658 (n=86), while strict zero-shot CSP reached 0.570. Small adaptations
and Riemannian covariance decoding did not solve this gap. Participant-specific CSP
reached 0.650 at 28 balanced labels. The evidence supports personalized calibration,
not a universal zero-shot decoder.

## Integrated interpretation

The immutable raw EDF pipeline uses tail handling where required, average reference,
and frozen filtering; subjects 88, 92, and 100 remain technically incompatible.
Task/rest ERD is distinct from fists-versus-feet prediction. The fixed-frequency
marker replicated while individualized frequency did not improve it. Geometry shows
large between-person variation and weak, unstable task-direction sharing, but does
not establish a biological cause. Recentring improved geometry without practical BA
benefit and direction rotation harmed decoding.

At fixed 14 labels, two-run diversity did not help. More labels did: the frozen
14/18/22/26/28 curve was 0.595/0.623/0.622/0.647/0.650. Conservatively, no shorter
burden meets every near-full criterion; roughly 26–28 balanced labels over two runs
is supported for this fixed classical pipeline.

## Limitations

This public legacy, task-specific dataset has no EOG, source localization, external
dataset, online BCI test, clinical validation, or session/day generalization. Later
participants were reused for post-primary mechanistic studies. Runs and trial counts
are limited; claims remain sensor-level and classical-model-specific.
