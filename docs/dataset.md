# Dataset and experimental protocol

[Back to the project README](../README.md)

## Dataset identity

This project uses version 1.0.0 of the PhysioNet EEG Motor Movement/Imagery Dataset. The dataset contains 64-channel EEG recordings acquired with the BCI2000 system during baseline, motor-execution, and motor-imagery tasks.

- Authoritative dataset page: <https://physionet.org/content/eegmmidb/1.0.0/>
- Dataset DOI: <https://doi.org/10.13026/C28G6P>
- File format: EDF+ with annotations
- Sampling frequency: 160 Hz for the frozen compatible cohort; 128 Hz for subjects
  88, 92, and 100 in the selected runs
- Electrode organization: 64 scalp EEG signals based on the international 10-10 system

The raw data was collected by the dataset contributors. This repository performs a secondary analysis; it did not recruit participants or acquire EEG.

## Experimental runs

Each participant completed 14 runs:

| Runs | Condition |
| --- | --- |
| 1 | Baseline, eyes open |
| 2 | Baseline, eyes closed |
| 3, 7, 11 | Motor execution: left fist versus right fist |
| 4, 8, 12 | Motor imagery: left fist versus right fist |
| 5, 9, 13 | Motor execution: both fists versus both feet |
| 6, 10, 14 | Motor imagery: both fists versus both feet |

The development experiment selects subject 1 and runs 6, 10, and 14. The first
replication applies the unchanged selection to subjects 2–20; the second requests
subjects 21–109. Among the second cohort's 267 requested runs, 258 retain the frozen
160 Hz and 30-annotation contract, with `T1/T2` split `8/7` in 131 and `7/8` in 127.
The nine files belonging to subjects 88, 92, and 100 load but use 128 Hz and different
annotation counts, so those subjects are transparently excluded without replacement.
See [the first replication](multi_subject_replication.md) and
[the full-cohort replication](full_cohort_replication.md).

## Why these runs were selected

Runs 6, 10, and 14 are three repetitions of the same bilateral motor-imagery condition. During task periods, the participant imagined repeatedly opening and closing either both fists or both feet without intentionally executing the movement.

Starting with one participant limited the initial engineering and scientific scope
while the analysis was learned and validated. Using all three matching runs provided
repeated observations of the same task. That first within-subject experiment could
not support between-person claims, so the methodology was then frozen and replicated
in subjects 2–20 and then independently in eligible subjects 21–109. These replications
add evidence across people but still do not make this fixed dataset cohort a random or
representative population sample.

The initial classification target is **both-fists imagery versus both-feet imagery**. It is not left-hand versus right-hand imagery.

## Annotation meanings

PhysioNet uses three annotation descriptions:

| Annotation | Meaning in runs 6, 10, and 14 |
| --- | --- |
| `T0` | Rest |
| `T1` | Onset of imagined movement of both fists |
| `T2` | Onset of imagined movement of both feet |

This mapping is run-dependent. In unilateral hand runs, `T1` and `T2` instead refer to left and right fist conditions. Code and documentation must therefore never interpret `T1` or `T2` without knowing the run type.

Annotations provide timing and condition information. They are now converted to event-aligned task and rest/context epochs with run-specific semantic metadata; they have not been converted into machine-learning features or predictions. See [Event extraction, epoching, and trial-quality assessment](epoching.md).

## Local acquisition

The project uses MNE's `eegbci.load_data` utility and the official PhysioNet source
to acquire the selected subject/run files:

- Implementation: [`scripts/inspect_raw_data.py`](../scripts/inspect_raw_data.py)
- Local generated-data directory: `data/MNE-eegbci-data/`
- Version-control policy: `data/` is excluded from Git

The full 327-file local collection occupies approximately 796 MB. This preserves
reproducibility without duplicating PhysioNet's raw data in the repository. Every
analysis input is identified by SHA-256 in the full-cohort metadata.

## Scientific scope and limitations

- Subject 1 is a development participant, not a representative population. Eligible
  subjects 2–109 form fixed replication cohorts, not a random population sample.
- The completed confirmatory method is not portable to subjects 88, 92, and 100
  without a separately specified 128-Hz/protocol adaptation.
- The project currently analyzes existing recordings and cannot control the original acquisition procedure.
- Event annotations describe the protocol but do not guarantee clean, artifact-free EEG.
- The dataset page and inspected EDF metadata do not identify the named physical acquisition reference; the later average-reference choice is therefore a project methodology decision rather than dataset-provided metadata.
- Motor imagery is inferred from the instructed experimental condition; EEG does not independently prove the participant's subjective mental strategy.
- Future predictive results must remain separate from biological or causal interpretation.

## Sources

- Schalk, G. (2009). [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (version 1.0.0). PhysioNet. <https://doi.org/10.13026/C28G6P>
- MNE-Python contributors. [`mne.datasets.eegbci.load_data`](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html).
