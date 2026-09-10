# Reproducibility

[Back to the project README](../README.md)

This document describes how to reproduce the completed environment through the
held-out within-subject decoding analysis.

> Public-release note: this guide documents executed commands and frozen artifacts,
> but the current repository still needs a deliberate migration of absolute local
> paths embedded in hash-linked historic provenance artifacts before a public push.
> See [the release audit](public_release_audit.md).

## Ongoing Lee2019 acquisition (do not evaluate yet)

The external cross-session study is not a completed reproduction target. Its
source-only, resumable command is:

```bash
MOABB_DOWNLOAD_PROVIDER=nemar \
  python scripts/acquire_external_cross_session_lee2019.py --data-dir data
```

It downloads only missing Lee2019 sources, validates them against NEMAR manifest
bytes and SHA-256 values, and writes only a local ignored acquisition journal. Do
not run `evaluate_external_cross_session_lee2019.py` until all 108 sources satisfy
the frozen gate described in [external validation](external_validation.md).

## v1.1 individualized-mu reliability package

After the PhysioNet EEGBCI data are present, run:

```bash
python scripts/analyze_individual_mu_reliability.py
```

The command reads each eligible run as an immutable input, validates the frozen v1.0
individual-mu configuration hash, and writes independent-rest tables, figures,
summary, and a deterministic provenance manifest under `docs/assets/`. It does not
recompute or alter the v1.0 held-out task-effect artifacts.

## Frozen unilateral motor-imagery generalization

After official EEGMMIDB runs 4, 8, and 12 are locally available for the historical
86-participant decoder cohort, run:

```bash
python scripts/evaluate_unilateral_motor_imagery_generalization.py
```

The command acquires only missing official files through MNE, validates each EDF's
identity and structure, then evaluates the frozen `spectral_baseline_6` and
`csp_4_empirical` architectures using participant-local leave-one-run-out folds. It
writes tables, figures, source hashes, package versions, and the deterministic
10,000-resample bootstrap provenance to `docs/assets/`. Raw EDFs remain excluded
from Git. See [the unilateral report](unilateral_motor_imagery_generalization.md)
for the frozen protocol and expected artifacts.

## Verified environment

The milestone was executed successfully with:

| Component | Version |
| --- | --- |
| Python | 3.14.7 |
| MNE-Python | 1.12.1 |
| NumPy | 2.5.2 |
| Matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |
| SciPy (transitive dependency used by MNE) | 1.18.1 |

The direct Python dependencies are pinned in [`requirements.txt`](../requirements.txt). The complete transitive dependency graph and operating-system packages are not locked yet, so this is version-controlled reproducibility rather than a fully frozen computational environment.

## Create the environment

From the repository root, create a project-local virtual environment:

```bash
python3 -m venv .venv
```

Activate it with Fish:

```fish
source .venv/bin/activate.fish
```

Or activate it with Bash/Zsh:

```bash
source .venv/bin/activate
```

Install the pinned direct dependencies:

```bash
python -m pip install -r requirements.txt
```

Using `python -m pip` ensures that packages are installed through the Python interpreter from the active environment.

## Verify dependency imports

Run:

```bash
python check_environment.py
```

Expected output for the environment used to complete this milestone:

```text
MNE: 1.12.1
NumPy: 2.5.2
Matplotlib: 3.11.1
scikit-learn: 1.9.0
```

## Acquire and inspect the data

Run:

```bash
python scripts/inspect_raw_data.py
```

MNE downloads these files on first execution:

```text
data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R06.edf
data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R10.edf
data/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R14.edf
```

The total observed download was approximately 7.4 MB. The `data/` directory is intentionally excluded by [`.gitignore`](../.gitignore); rerunning the command reuses existing valid files.

Expected metadata for every run:

```text
Data shape (channels, samples): (64, 20000)
Channel types: {'eeg': 64}
Sampling frequency: 160 Hz
Time between samples: 6.25 ms
Approximate duration: 125.00 s
```

Annotation order may differ between runs, but every selected file contains 15 `T0` annotations, 7 `T1` annotations, and 8 `T2` annotations.

## Generate raw-visualization evidence

Run the documented subject 1/run 6 visualization with default options:

```bash
python scripts/visualize_raw_eeg.py
```

This command:

- loads run 6 without filtering or re-referencing;
- standardizes EEGBCI channel names;
- plots `Fp1`, `C3`, `Cz`, `C4`, `Pz`, and `Oz` from 8–17 s;
- verifies that MNE's microvolt values equal its internal volt values multiplied by $10^6$;
- exports all annotation onsets, durations, descriptions, conditions, and sample indices;
- attaches `standard_1005` template coordinates only for the electrode-location figure.

Expected report artifacts:

```text
docs/assets/subject01_run06_raw_selected_channels_008-017s.png
docs/assets/subject01_run06_annotation_timeline.png
docs/assets/subject01_run06_electrode_locations_standard1005.png
docs/assets/subject01_run06_annotations.csv
```

The default selected window is samples `1280:2720`, corresponding to 8–17 s at 160 Hz. The verified selected-channel amplitude range is approximately −369 to +416 µV.

To inspect another project run or time window, use the CLI options:

```bash
python scripts/visualize_raw_eeg.py --help
```

## Assess quantitative signal quality and reference

Run the complete subject 1 assessment:

```bash
python scripts/assess_signal_quality.py
```

The default command:

- loads runs 6, 10, and 14 without filtering;
- verifies `(64, 20,000)` finite µV arrays for each run;
- calculates channel-level mean, standard deviation, variance, extrema, peak-to-peak amplitude, median, MAD, IQR, maximum adjacent-sample step, exact-repeat measures, and median peer correlation;
- applies deterministic modified robust z-score candidate rules with threshold 3.5;
- calculates 2-second non-overlapping temporal metrics;
- detects intervals where all channels remain constant;
- applies average reference only to an in-memory copy of report run 6;
- checks MNE's result against the explicit average-subtraction equation;
- verifies that the original MNE `Raw` object is unchanged.

Expected report artifacts:

```text
docs/assets/subject01_runs06-10-14_channel_quality.csv
docs/assets/subject01_runs06-10-14_temporal_quality.csv
docs/assets/subject01_run06_channel_quality.png
docs/assets/subject01_run06_temporal_quality.png
docs/assets/subject01_run06_average_reference_comparison_008-017s.png
```

The channel table contains 192 rows: 64 channels × 3 runs. The temporal table contains 189 rows: 63 windows × 3 runs. Every selected run contains a common constant segment at 124.5–125.0 s. No channel is marked bad or removed.

Inspect configuration options:

```bash
python scripts/assess_signal_quality.py --help
```

An alternate run can be written outside the tracked report:

```bash
python scripts/assess_signal_quality.py \
  --runs 10 \
  --output-dir outputs/run10_quality
```

The raw EDF files are read-only inputs to this workflow. The reference comparison is not saved as a processed EEG file; it is regenerated from a copied in-memory object.

## Characterize the raw frequency spectrum

Run:

```bash
python scripts/analyze_spectrum.py
```

The default command:

- loads subject 1, runs 6, 10, and 14;
- detects and excludes samples 19,920–19,999 from a copied derived view;
- confirms that no earlier sample is zero across all 64 channels;
- compares the stored representation with a 64-channel average-reference copy;
- computes mean Welch PSD using 3-second/480-sample Hann windows, 240-sample overlap, per-segment DC removal, and `n_fft=480` without zero-padding;
- verifies an exact 0–80 Hz NumPy DFT grid with 0.333333 Hz spacing;
- verifies 82 Welch windows and exact use of all 19,920 valid samples;
- independently checks mean aggregation and a synthetic 10 Hz sine with expected integrated power 0.5 V²;
- integrates 1–45 Hz into five adjacent bands and checks their sum;
- evaluates exact 50 and 60 Hz bins against flanking frequencies;
- exports full-precision linear-unit tables and readable figures.

Expected report artifacts and data-row counts:

```text
docs/assets/subject01_runs06-10-14_channel_psd_summary.csv       384
docs/assets/subject01_runs06-10-14_band_power.csv               1920
docs/assets/subject01_runs06-10-14_line_frequency_analysis.csv  768
docs/assets/subject01_runs06-10-14_reference_effects.csv        192
docs/assets/subject01_runs06-10-14_psd_curves.csv                1434
docs/assets/subject01_runs06-10-14_broad_psd.png
docs/assets/subject01_runs06-10-14_sensorimotor_psd.png
docs/assets/subject01_runs06-10-14_line_frequency_analysis.png
```

The PSD output contains no NaN, infinity, or negative power values. The five adjacent relative-band fractions sum to one per channel/reference/run to floating-point precision. `C3`, `Cz`, and `C4` are checked before analysis.

Inspect the CLI or write an isolated analysis outside tracked report artifacts:

```bash
python scripts/analyze_spectrum.py --help
python scripts/analyze_spectrum.py \
  --runs 10 \
  --output-dir outputs/run10_spectrum
```

The default analysis is deterministic and contains no random operation. Custom segment/overlap values must cover the valid sample range exactly; custom line frequencies must be on an exact DFT bin below Nyquist.

The verified raw-file SHA-256 digests after this milestone are:

```text
5369364f2c4e81ca141679d6dd2ba6ece61c7eb53d7fae31241b308876e1b6b3  S001R06.edf
20de1c7746c2349d16bda5e9f1b0ac7b7ad1581102a2e30dd2ac422696f62fb1  S001R10.edf
2110c48e3106898e3dbca47e39b330637afd3d3b8bc2da3ba1e44f4ac1118137  S001R14.edf
```

These hashes establish byte identity for the locally verified inputs; they do not substitute for the PhysioNet source citation.

## Reproduce controlled preprocessing

The production policy is versioned separately from implementation:

```text
config/preprocessing.json
```

Run all three recordings:

```bash
python scripts/preprocess_eeg.py
```

This command imports reusable operations from `eeg_project/preprocessing.py` and:

- reloads each immutable EDF and records its SHA-256;
- excludes the detected 80-sample zero tail before padding or filtering;
- confirms no earlier all-channel zero sample exists;
- retains all 64 reviewed channels and applies average reference to a copy;
- independently verifies the reference subtraction equation;
- designs the exact symmetric 529-coefficient FIR;
- applies the filter continuously with 1/40 Hz passband edges, 1/10 Hz transitions, Hamming `firwin`, compensated zero phase, and `reflect_limited` padding;
- verifies shape `(64, 19,920)`, finite values, 160 Hz sampling, and unchanged annotations;
- evaluates no/0.5/1 Hz high-pass candidates and 40/45 Hz low-pass candidates;
- evaluates and rejects a post-low-pass 60 Hz notch;
- generates frequency/impulse responses, real-EEG comparisons, a synthetic impulse/step test, metrics, and deterministic provenance.

Expected artifacts and data rows:

```text
docs/assets/subject01_runs06-10-14_preprocessing_metrics.csv          576
docs/assets/subject01_runs06-10-14_filter_candidate_comparison.csv    15
docs/assets/subject01_runs06-10-14_filter_response.csv                 84
docs/assets/subject01_runs06-10-14_time_window_metrics.csv            384
docs/assets/subject01_runs06-10-14_preprocessing_metadata.json
docs/assets/subject01_runs06-10-14_filter_responses.png
docs/assets/subject01_runs06-10-14_preprocessing_psd.png
docs/assets/subject01_runs06-10-14_synthetic_transient_response.png
docs/assets/subject01_run06_preprocessing_normal_008-012s.png
docs/assets/subject01_run06_preprocessing_transient_013-017s.png
```

Inspect the CLI or isolate one run:

```bash
python scripts/preprocess_eeg.py --help
python scripts/preprocess_eeg.py \
  --runs 10 \
  --output-dir outputs/run10_preprocessing
```

When run 6 is omitted, the run-6-only time-window tables and figures are not generated. No `.fif` is persisted: the derived continuous EEG is regenerated in memory from raw data and configuration.

Run the focused preprocessing tests:

```bash
python -m unittest discover -s tests -v
```

These tests independently check the production response, FIR symmetry, zero-tail detection, deterministic output, annotation/sample preservation, average-reference mathematics, and equivalence of reference-before-filter versus filter-before-reference for identical linear per-channel filtering.

## Reproduce event extraction and epoching

The epoch policy is versioned in:

```text
config/epoching.json
```

Run the verified subject 1 report:

```bash
python scripts/create_epochs.py --subject 1 --runs 6 10 14
```

The command reloads each immutable EDF, executes the production continuous
preprocessing, validates the bilateral motor-imagery run mapping, independently
audits annotation-to-sample conversion, extracts task and rest/context epochs,
checks boundaries, and computes deterministic exploratory trial-quality
metrics. It does not save epoch FIF arrays or reject statistical candidates.

Expected console summary:

```text
Task X shape: (45, 64, 961); event-relative time -2 to 4 s; baseline=None
Rest/context shape: (45, 64, 641); event-relative time 0 to 4 s; not a task class
Task counts: {'both_feet_imagery': 24, 'both_fists_imagery': 21}
Statistical candidate task epochs: 6; confirmed exclusions: 0
```

Expected report artifacts and data rows:

```text
docs/assets/subject01_runs06-10-14_annotation_structure.csv       90
docs/assets/subject01_runs06-10-14_task_trials.csv                45
docs/assets/subject01_runs06-10-14_rest_context_epochs.csv        45
docs/assets/subject01_runs06-10-14_trial_quality.csv               45
docs/assets/subject01_runs06-10-14_epoching_metadata.json
docs/assets/subject01_runs06-10-14_annotation_structure.png
docs/assets/subject01_runs06-10-14_representative_task_epochs.png
docs/assets/subject01_runs06-10-14_c3_trial_heatmap.png
docs/assets/subject01_runs06-10-14_trial_quality.png
docs/assets/subject01_runs06-10-14_candidate_epoch_comparison.png
```

The real annotation audit finds 15 `T0`, 7 `T1`, and 8 `T2` annotations in
each run, with 4.2-second rest and 4.1-second task intervals. All annotations
alternate contiguously, all 45 requested task epochs are valid, and no sample
from the excluded 19,920–19,999 tail enters an epoch.

Inspect options or isolate one run under ignored disposable output:

```bash
python scripts/create_epochs.py --help
python scripts/create_epochs.py --runs 10 --output-dir outputs/run10_epoching
```

The isolated run-10 output should contain task shape `(15, 64, 961)`,
rest/context shape `(15, 64, 641)`, 7 both-fists trials, 8 both-feet trials,
and one statistical candidate with no confirmed rejection.

Run all focused numerical tests:

```bash
python -m unittest discover -s tests -v
```

The epoch tests cover explicit run semantics, failure for unsupported run
families, annotation rounding, inclusive sample counts, boundary detection,
real counts and shapes, exact first/middle/final event timing, source hashes,
annotation preservation, finite data, candidate-only QC, and repeated-array
identity.

## Reproduce event-related spectral analysis

Run:

```bash
python scripts/analyze_event_related_spectrum.py --subject 1 --runs 6 10 14
```

Expected core output:

```text
Task/rest TFR: (45, 64, 30, 241) / (45, 64, 30, 161)
Morlet: 6-35 Hz, 1-Hz centers, n_cycles=f/2, 127 samples
Paired T0 reference: 1.1-3.1 s; task summary: 1-3 s
Primary task trials: 45; automatic exclusions: 0
```

The command produces 8,640 provenance-preserving trial/channel/band rows,
condition/run/window/time-frequency summaries, metadata, and six report
figures. Full TFR arrays are regenerated in memory rather than persisted.

The focused tests verify the frequency grid and non-overlapping bands, actual
127-sample wavelets, exact percent-change equation, synthetic 10-Hz
localization, a synthetic −75% power reduction, real T0/task pairing, TFR
dimensions, provenance, QC identity, and non-mutation of all 45 primary epochs.

See [Event-related spectral analysis](event_related_spectral_analysis.md) for
artifact names, actual findings, and interpretation limits.

## Reproduce artifact policy and feature stability

Run:

```bash
python scripts/analyze_trial_quality.py --subject 1 --runs 6 10 14
```

Expected core output:

```text
Master trials retained: 45/45; permanent rejections: 0
Task QC candidates: 6; paired-rest candidates: 5
both_fists_imagery C3 central_12_13: -55.57% | 6/7 moderate
both_fists_imagery C4 central_12_13: -52.28% | 7/7 high
```

The command reuses preprocessing, epoching, task/rest pairing, and Morlet
operations. It regenerates full `(45, 64, 30, 241)` task and
`(45, 64, 30, 161)` rest TFRs in memory; applies label-independent paired-side
metrics; reviews the six existing candidates against systematic same-run,
same-condition controls; computes percent/dB, timing, first-pair, QC, candidate,
and reference sensitivities; and writes the predefined seven-criterion grades.

Expected report artifacts:

```text
docs/assets/subject01_runs06-10-14_trial_quality_detailed.csv
docs/assets/subject01_runs06-10-14_paired_reference_quality.csv
docs/assets/subject01_runs06-10-14_candidate_trial_review.csv
docs/assets/subject01_runs06-10-14_normalization_sensitivity.csv
docs/assets/subject01_runs06-10-14_feature_stability.csv
docs/assets/subject01_runs06-10-14_candidate_influence.csv
docs/assets/subject01_runs06-10-14_rest_pair_influence.csv
docs/assets/subject01_runs06-10-14_artifact_evidence.csv
docs/assets/subject01_runs06-10-14_quality_timing_sensitivity.csv
docs/assets/subject01_runs06-10-14_trial_quality_metadata.json
docs/assets/subject01_runs06-10-14_candidate_trial_gallery.png
docs/assets/subject01_runs06-10-14_quality_metric_distributions.png
docs/assets/subject01_runs06-10-14_normalization_diagnostics.png
docs/assets/subject01_runs06-10-14_feature_stability_matrix.png
```

The metadata records unchanged condition counts, exact task-candidate identities,
the stronger-review subset, zero permanent rejections, paired-reference candidate
count, source hashes, and that QC did not use labels. The focused synthetic tests
independently check `10 log10(0.5) = -3.0103 dB`, robust outlier identity,
trimmed means, Spearman ranks, leave-one-out median influence, and deterministic
high/moderate/low grading. Real-data tests preserve all 45 pair identities, 21/24
condition counts, all three hashes, and the exact six candidates.

See [Artifact policy and spectral feature stability](trial_quality_and_feature_stability.md)
for the complete scientific method, numerical result, ICA decision, final policy,
and decision gate.

## Reproduce the subjects 1–20 replication

Run the technical audit before computing spectral outcomes:

```bash
python scripts/analyze_replication_cohort.py --audit-only
```

This requests subjects 1–20 and runs 6, 10, and 14, validates sampling, standardized
channels, annotation protocol, boundaries, finiteness, and flat sensors, and records
all 60 source hashes. The verified audit finds 60/60 compatible runs and 20/20
eligible subjects. It also preserves the observed `T1/T2` counterbalancing and crops
an 80-sample all-channel-zero tail only in the 9 files where one exists.

Then regenerate the complete report:

```bash
python scripts/analyze_replication_cohort.py
python -m unittest discover -s tests -v
```

After initial acquisition (about 137 MB), the full report takes about 20 seconds in
the verified environment. Its essential expected output is:

```text
Requested/eligible subjects: 20/20
Replication subjects: 19
Replication trials: 855
C4 fists 12–13 Hz: 17/19 negative, median -30.73%, strong replication
C3 fists 12–13 Hz: 16/19 negative, median -29.83%, strong replication
```

The command computes all-channel time-domain QC but only C3/Cz/C4 cohort TFRs. For
each subject the task shape is `(45, 3, 30, 241)` and the paired-rest shape is
`(45, 3, 30, 161)`. Arrays are regenerated in memory and not persisted. Compact
tracked output includes:

```text
docs/assets/subjects01-20_replication_cohort_audit.csv
docs/assets/subjects01-20_replication_subject_eligibility.csv
docs/assets/subjects01-20_replication_trial_spectral_measurements.csv
docs/assets/subjects01-20_replication_subject_feature_summary.csv
docs/assets/subjects01-20_replication_subject_run_summary.csv
docs/assets/subjects01-20_replication_subject_qc_summary.csv
docs/assets/subjects01-20_replication_replication_primary_features.csv
docs/assets/subjects01-20_replication_cohort_feature_summary.csv
docs/assets/subjects01-20_replication_central_peak_summary.csv
docs/assets/subjects01-20_replication_peak_feature_relationship.csv
docs/assets/subjects01-20_replication_metadata.json
```

Five PNG figures use those compact summaries. The metadata records frozen-config
hashes, the bootstrap seed, source hashes, subject/run/trial counts, primary criteria,
and the artifact inventory. Subject 1 must reproduce its earlier C4/C3 primary values
exactly. The focused cohort tests cover policy validation, one-value-per-subject
aggregation, development-subject separation, retention of reversed but valid results,
technical exclusion reasons, sign/Wilson/Holm calculations, replication categories,
and a real subject-2 audit.

See [Multi-subject replication and pipeline generalization](multi_subject_replication.md)
for the complete method, evidence, interpretation, and decision gate.

## Reproduce the full-cohort replication

The second-replication policy and all five required scientific configuration hashes
are recorded in:

```text
config/full_cohort_replication.json
```

The complete 109-subject × 3-run local EDF collection occupies approximately 796 MB
under `data/MNE-eegbci-data/files/eegmmidb/1.0.0/`. Raw data remain Git-ignored and
read-only to the analysis. Existing complete files are reused. Acquisition speed or
the use of an official PhysioNet transfer mirror changes only how identical bytes
arrive locally; the audit and SHA-256 checks establish the analysis inputs.

First reproduce the outcome-independent cohort-2 audit:

```bash
python scripts/analyze_full_cohort_replication.py --audit-only
```

Expected audit summary:

```text
Requested subjects: 89
Technically eligible subjects: 86
Requested runs: 267
Compatible/usable runs: 258
Technically incompatible subjects: 88, 92, 100
```

All 267 files load. The three excluded subjects have 128 Hz recordings and
nonstandard annotation counts under the frozen 160 Hz/30-annotation protocol. The
audit records every source hash, zero-tail length, task counterbalance, and failure
reason in:

```text
docs/assets/subjects01-109_full_replication_replication2_audit.csv
docs/assets/subjects01-109_full_replication_replication2_eligibility.csv
```

Then generate outcomes and the complete report:

```bash
python scripts/analyze_full_cohort_replication.py
```

Expected primary console summary:

```text
C4 cohort 2: 70/86 negative, median -26.51%, strong replication
C3 cohort 2: 75/86 negative, median -21.07%, strong replication
C4 combined: 87/105 negative, median -26.67%
C3 combined: 91/105 negative, median -23.90%
```

The first eligible run processes one subject at a time and writes 86 compact JSON
checkpoints under `outputs/full_cohort_checkpoints/`. Each checkpoint is reused only
when its subject, completion state, frozen configs, processing-code hashes, and source
EDF hashes match. Full TFR arrays are never persisted. A valid rerun uses the
checkpoints and deterministically rebuilds the tracked CSV, JSON, and PNG report.

Force representative subject recomputation and compare it with the stored checkpoint:

```bash
python scripts/analyze_full_cohort_replication.py --verify-subjects 21 65 109
```

Run the complete numerical and real-data regression suite:

```bash
python -m unittest discover -s tests -v
python -m compileall -q eeg_project scripts tests
```

The regression suite requires subject 1 to retain its exact C4/C3 primary results,
requires a historical cohort-1 subject to reproduce stored rows, checks disjoint
cohort identities, rejects stale checkpoints, preserves outcome-independent
eligibility, and verifies annotation pairing when EDF duration fields are rounded.

The full report's principal artifacts are:

```text
docs/assets/subjects01-109_full_replication_metadata.json
docs/assets/subjects01-109_full_replication_cohort_comparison.csv
docs/assets/subjects01-109_full_replication_subject_primary_table.csv
docs/assets/subjects01-109_full_replication_subject_run_summary.csv
docs/assets/subjects01-109_full_replication_subject_qc_summary.csv
docs/assets/subjects01-109_full_replication_replication2_trial_spectral_measurements.csv
docs/assets/subjects01-109_full_replication_central_peak_summary.csv
docs/assets/subjects01-109_full_replication_peak_relationship.csv
docs/assets/subjects01-109_full_replication_primary_subjects.png
docs/assets/subjects01-109_full_replication_cohort_comparison.png
docs/assets/subjects01-109_full_replication_run_consistency.png
docs/assets/subjects01-109_full_replication_narrow_vs_mu.png
docs/assets/subjects01-109_full_replication_central_peak_evidence.png
```

The trial table has 34,560 data rows (3,840 trials × 3 channels × 3 reported band
definitions). The metadata records 327 immutable EDF hashes, 86 checkpoint files,
zero outcome-based exclusions, and explicit `false` values for full-TFR persistence,
CSP, and classification. See [Full-cohort EEG replication](full_cohort_replication.md)
for the full scientific interpretation.

## Reproduce the individualized-frequency methodology study

The study policy, outcome-independent even/odd participant partition, candidate
space, historical hashes, final rest-only estimator, and pre-outcome decision rules
are recorded in:

```text
config/individual_mu_frequency.json
```

Git chronology is part of reproducibility:

```text
d322c58  completed fixed-band full cohort
63b6025  initial IMF partition and candidate methodology freeze
cd2376d  reusable estimator implementation
9eb3989  final method and success-rule freeze
<later>  held-out odd-ID individualized outcomes
```

The development command reads only the 51 even-ID participants' T0 rest epochs and
rebuilds the two candidate-method tables. It never computes individualized task
outcomes:

```bash
python scripts/analyze_individual_mu_frequency.py --develop-method
```

The evaluation command requires the committed final-freeze configuration. For every
odd-ID held-out participant and run, it estimates the peak from only the other two
runs' T0 epochs, computes the five-frequency Morlet band for that held-out run, and
joins exact historical fixed-band trial rows by participant, run, trial, channel, and
raw source hash:

```bash
python scripts/evaluate_individual_mu_frequency.py
```

Expected console summary:

```text
Coverage: 50/54 (92.6%)
C4: fixed 43/50 vs individualized 39/50; paired median difference 0.96 points; no improvement
C3: fixed 45/50 vs individualized 40/50; paired median difference 2.44 points; no improvement
Overall frozen category: no improvement
```

Principal machine-readable outputs are:

```text
docs/assets/individual_mu_heldout_evaluation_metadata.json
docs/assets/individual_mu_heldout_evaluation_peak_estimates.csv
docs/assets/individual_mu_heldout_evaluation_peak_subject_summary.csv
docs/assets/individual_mu_heldout_evaluation_trial_comparison.csv
docs/assets/individual_mu_heldout_evaluation_subject_comparison.csv
docs/assets/individual_mu_heldout_evaluation_group_comparison.csv
docs/assets/individual_mu_heldout_evaluation_peak_distance_relationship.csv
```

The test suite checks the synthetic 10-Hz recovery, boundary and flat-spectrum
policies, exact 0.25-Hz Welch grid and five-center individualized band, partition and
hash freezes, held-out-run separation, exact matched trial identities, subject-level
aggregation, fixed-artifact hashes, and current raw EDF hashes:

```bash
python -m unittest discover -s tests -v
python -m compileall -q eeg_project scripts tests
```

The evaluator persists compact tables and figures only. Full Morlet arrays are
discarded after aggregation. See
[Individualized central alpha/mu-frequency methodology](individual_mu_frequency.md)
for the method, evidence, and interpretation.

## Reproduce the within-subject decoding study

The initial policy, selected final method, and outcome-independent retained-trial
correction are independently frozen in:

```text
config/within_subject_decoding.json
config/within_subject_decoding_final.json
config/within_subject_decoding_correction.json
```

Git chronology protects the evaluation boundary:

```text
14e32ae  initial classifier-study freeze before any decoding outcome
49a51bc  leakage-safe implementation and synthetic tests
bf27724  subjects 1–20 development result
7bbd249  final method freeze before subjects 21–109 outcomes
be524e0  outcome-independent retained-trial validation correction
8cc5300  activate the corrective freeze before evaluation restart
```

The development command reproduces the already completed five-candidate comparison;
it must not be used to revise the frozen method after seeing evaluation results:

```bash
python scripts/develop_within_subject_decoder.py
```

The final evaluator validates and reuses one compact JSON checkpoint per eligible
participant, then deterministically rebuilds the report without refitting when all
86 checkpoints remain valid:

```bash
python scripts/evaluate_within_subject_decoding.py
```

Expected summary:

```text
spectral_baseline_6: median=0.585; IQR=0.501–0.664; >0.5=74.4%
csp_4_empirical: median=0.658; IQR=0.565–0.769; >0.5=87.2%
CSP minus baseline median=0.086; clear added value
Frozen decoding category: useful decoding signal
```

The evaluator produces 3,840 primary predictions per model and 1,032 fold-fit audit
rows. Checkpoints are accepted only when the active configuration hash, frozen core
hash, completion marker, participant identity, and all three source EDF hashes match.
The copied QC pipeline is refitted separately. Historical ERD and frequency tables
are joined only after primary summaries; unavailable peak values remain blank.

Primary report files have prefix
`docs/assets/within_subject_decoder_evaluation_`. The shortest machine-readable
entry points are `group_summary.csv`, `model_comparison_summary.csv`,
`confusion_matrices.csv`, `error_analysis.csv`, and `metadata.json`; predictions and
fit audits retain the trial-level leakage evidence. Compact checkpoints live under
`outputs/within_subject_decoder_checkpoints/` and remain excluded from Git.

Focused and complete verification commands are:

```bash
python -m unittest tests.test_decoding_evaluation tests.test_decoding_evaluation_artifacts -v
python -m unittest discover -s tests -v
python -m compileall -q eeg_project scripts tests
```

The artifact regression checks require all 86 eligible subjects, one prediction per
retained trial and model, exact checkpoint/config/core/source hashes, disjoint fit and
test keys for CSP/scaler/LDA, correct confusion totals, legitimate missing historical
peaks, and unchanged historical EEG artifacts.

## Reproduce the decoding reliability study

This post-hoc diagnostic milestone starts from the completed decoder commit
`4a39b4f`. Its policy was committed separately at `68a75a6`, before calculating the
new covariance-shift outcomes, in:

```text
config/decoding_reliability.json
```

The report command validates the immutable historical prediction, fold-score,
subject-score, evaluation-metadata, decoder-config, decoder-core, and physiology
file hashes. It derives run, class, and QC summaries from those existing files and
does not fit CSP, a scaler, or LDA:

```bash
python scripts/analyze_decoding_reliability.py
```

Only the label-independent shift diagnostic reloads EEG. It regenerates the same
8–30 Hz task-trial representation, estimates Ledoit–Wolf covariance separately for
the two pooled training runs and held-out run, and stores three compact distances
per participant under `outputs/decoding_reliability_checkpoints/`. A checkpoint is
reused only when the reliability-policy hash and all three source EDF hashes match.
No class labels are accepted by the distance function and no classifier is refitted.

Expected summary:

```text
Runs above chance: 3/3=55, 2/3=15, 1/3=11, 0/3=5
Run medians: run 6=0.670, run 10=0.621, run 14=0.670
Friedman matched-run p=0.103; no Holm-adjusted pairwise p < 0.05
Median within-participant range=0.147
Centered covariance-distance/accuracy r=-0.015; permutation p=0.819
Decision gate: A, separately frozen cross-subject decoding
```

Generated report files use prefix `docs/assets/decoding_reliability_`. The main
machine-readable outputs are `subject_reliability.csv`, `run_summary.csv`,
`run_comparisons.csv`, `class_recalls.csv`, `quality_relationships.csv`,
`run_shift_relationships.csv`, `historical_relationships.csv`, and `metadata.json`.
The metadata records historical hashes, checkpoint provenance, no classifier refit,
and hashes for every other reliability artifact.

Focused verification is:

```bash
python -m unittest tests.test_decoding_reliability tests.test_decoding_reliability_artifacts -v
python -m unittest discover -s tests -v
python -m compileall -q eeg_project scripts tests
```

Artifact checks require all 86 subjects and 258 held-out folds, unchanged historical
classifier files, missing historical peak values preserved as missing, label-free
shift checkpoints, valid source/config hashes, and exactly reproducible CSV/JSON/PNG
bytes on a checkpoint-only rerun in the verified environment.

## Reproduce the zero-shot cross-subject decoding study

The completed reliability milestone at `c52dd77` is the immutable starting point.
Cross-subject chronology preserves the outcome boundary:

```text
74b49d0  initial cross-subject policy before any cross-subject outcome
58e87e1  participant-isolated implementation and synthetic/unit tests
5da3b37  subjects 1–20 LOSO development evidence
f66b299  final exact models and success criteria before evaluation EEG was opened
53a23d6  checkpointed held-out evaluation implementation
```

The two machine-readable policies are:

```text
config/cross_subject_decoding.json
config/cross_subject_decoding_final.json
```

The development command performs 20 leave-one-subject-out folds using subjects 1–20
only. It is a reproduction command, not permission to select a new method after
evaluation outcomes have been opened:

```bash
python scripts/develop_cross_subject_decoder.py
```

The final command fits each frozen model once on all 900 development trials and
predicts each eligible subject 21–109 without target calibration. When the 86 compact
checkpoints are valid, it loads their saved predictions and rebuilds the report in
about seconds without reloading EEG or refitting a classifier:

```bash
python scripts/evaluate_cross_subject_decoding.py
```

Expected primary summary:

```text
spectral_baseline_6: median=0.536; IQR=0.474–0.598; >0.5=64.0%
csp_4_empirical: median=0.570; IQR=0.505–0.641; >0.5=75.6%
CSP minus spectral median=+0.039; CSP transfers better
Both model categories: no convincing zero-shot transfer
No evaluation-participant adaptation was used
```

Report files use prefix `docs/assets/cross_subject_decoder_evaluation_`. The main
tables are `group_summary.csv`, `subject_scores.csv`, `predictions.csv`,
`fit_audit.csv`, `paired_summary.csv`, `run_summary.csv`,
`within_cross_summary.csv`, `exploratory_relationships.csv`, and `metadata.json`.
Compact per-target checkpoints live under
`outputs/cross_subject_decoder_checkpoints/` and are excluded from Git.

Focused and full verification commands are:

```bash
python -m unittest tests.test_cross_subject_decoding tests.test_cross_subject_development_artifacts tests.test_cross_subject_final_freeze tests.test_cross_subject_evaluation tests.test_cross_subject_evaluation_artifacts -v
python -m unittest discover -s tests -v
python -m compileall -q eeg_project scripts tests
```

The focused artifact checks require exactly 86 eligible subjects, 3,840 unique target
trials and one prediction per trial per model, participant-level summaries, exact
frozen config/core/source hashes, all 20 training participants—and no target
participant—at CSP/scaler/LDA fit, no target adaptation, and unchanged within-subject
and historical EEG files. Two consecutive checkpoint-only report rebuilds produced
all 17 tracked outputs byte-for-byte identically in the verified environment.

## Reproduce the minimal target-participant calibration study

The completed zero-shot result at `6dca561` is the immutable starting point. The
calibration chronology preserves both the development/evaluation boundary and the
final method freeze:

```text
d4cff9a  initial calibration policy and success criteria before outcomes
c81388c  leakage-safe threshold and target-LDA implementation
b918cb4  subjects 1–20 LOSO development and global method selection
80e5ebc  threshold method frozen before subjects 21–109 calibration outcomes
a498974  checkpointed held-out evaluation implementation
```

The initial and final machine-readable policies are:

```text
config/minimal_target_calibration.json
config/minimal_target_calibration_final.json
```

The development command compares threshold calibration with the existing
shrinkage-LDA head using subjects 1–20 only. It reproduces a completed method choice;
it must not be used to revise that choice after the evaluation curve was opened:

```bash
python scripts/develop_target_calibration.py
```

The evaluation command fits the source CSP/scaler/LDA once on 900 trials from
subjects 1–20. For each of 86 targets, it transforms 42–45 trials with that frozen
source model and saves compact features, labels, QC flags, trial identities, source
scores, and hashes. A valid checkpoint-only rebuild does not reload EEG or refit the
source model:

```bash
python scripts/evaluate_target_calibration.py
```

Expected primary curve:

```text
labels:                    0      4      8     12     14
median balanced accuracy: 0.570  0.606  0.617  0.615  0.613
median matched gain:      +0.000 +0.007 +0.002 +0.007 +0.007
improved participants:       0     46     44     45     45
convincing benefit:        no     no     no     no     no
```

Each nonzero calibration subset contains equal fists/feet counts selected in
annotation chronology from one run. The other two runs supply test trials only. A
trial is tested under the two calibration-run contexts in which its own run is not
the calibration run, but inference and curve summaries retain exactly one value per
participant per size. Size zero reproduces the historical zero-shot participant
scores exactly.

Report files use prefix `docs/assets/target_calibration_evaluation_`. The principal
tables are `curve_summary.csv`, `participant_scores.csv`,
`matched_gain_subjects.csv`, `predictions.csv`, `fit_audit.csv`,
`incremental_gains.csv`, `subgroup_gains.csv`, `calibration_run_summary.csv`, and
`metadata.json`. Compact target-feature checkpoints live under
`outputs/target_calibration_checkpoints/` and are excluded from Git.

Focused and full verification commands are:

```bash
python -m unittest tests.test_target_calibration tests.test_target_calibration_development_artifacts tests.test_target_calibration_final_freeze tests.test_target_calibration_evaluation tests.test_target_calibration_evaluation_artifacts -v
python -m unittest discover -s tests -v
python -m compileall -q eeg_project scripts tests
```

The artifact guards require 86 source/config-validated checkpoints, 3,840 unique
target trials, 38,400 prediction appearances, exactly two test appearances per trial
at each size, balanced nested chronological calibration sets, and no target EEG in
source CSP/scaler fitting. Historical zero-shot, within-subject, reliability, and
physiology files must retain their frozen hashes. Two consecutive checkpoint-only
rebuilds produced all 15 report files byte-for-byte identically in the verified
environment.

## Reproduce the decision-layer versus spatial-personalization study

The completed calibration result at `5d11ed1` is the immutable starting point. The
new chronology preserves development, final method selection, and held-out target-
CSP evaluation as separate boundaries:

```text
4ceea62  initial A/B/C protocol and success criteria before development outcomes
fb9f55d  leakage-safe spatial-personalization implementation
51fab24  subjects 1–20 development and global method selection
5d43e46  source-scaler B and Ledoit–Wolf C frozen before evaluation outcomes
f44f7ec  checkpointed held-out evaluation implementation
```

The initial and final machine-readable policies are:

```text
config/spatial_personalization.json
config/spatial_personalization_final.json
```

Development uses subjects 1–20 only. It compares source versus target scaling for B
and empirical versus Ledoit–Wolf covariance for C, then reproduces the completed
global selections:

```bash
python scripts/develop_spatial_personalization.py
```

Expected development selections are source scaling for B and Ledoit–Wolf covariance
for C. The source/target scaler candidates tie for every participant. Ledoit–Wolf
minus empirical C has median paired gain +0.011 and wins for 11/20 participants.

Evaluation reuses the 86 validated source-feature checkpoints rather than refitting
the source model. On the first pass it preprocesses each target's three EDFs and
saves one aligned 8–30 Hz array plus provenance; later passes validate all hashes and
reuse those arrays:

```bash
python scripts/evaluate_spatial_personalization.py
```

Expected held-out summary:

```text
method:                         A source     B target LDA   C target CSP
median balanced accuracy:        0.570          0.565          0.596
median paired gain vs A:         0.000         -0.014         +0.010
median C-minus-B gain:                                          +0.014
participants improved C vs B:                                   53/86
convincing frozen transition:       no             no              no
interpretation: one-run personalization is insufficient
```

Each target-supervised scenario uses exactly the first seven trials per class from
one calibration run. The other two runs remain test-only. A predicts with the frozen
source pipeline, B fits only target LDA on source CSP/source-scaled features, and C
fits target Ledoit–Wolf CSP, scaler, and LDA. All summaries and inference reduce the
repeated scenario/test-run observations to one value per participant.

Report files use prefix `docs/assets/spatial_personalization_evaluation_`; there are
16 tables/figures plus one metadata file. Raw target arrays and provenance live under
`outputs/spatial_personalization_checkpoints/`, occupy approximately 601 MB locally,
and are excluded from Git. Focused and full verification commands are:

```bash
python -m unittest tests.test_spatial_personalization_freeze tests.test_spatial_personalization tests.test_spatial_personalization_development_artifacts tests.test_spatial_personalization_final_freeze tests.test_spatial_personalization_evaluation tests.test_spatial_personalization_evaluation_artifacts -v
python -m unittest discover -s tests -v
python -m compileall -q eeg_project scripts tests
```

The artifact guards require all 86 participants, 86 source-feature and 86 raw-array
checkpoints with matching source/config/code/EDF/array hashes, 3,840 unique target
trials, and two identical test contexts per trial and method. Fit audits prove that
test EEG and labels never enter CSP, scaler, or LDA fitting. Historical physiology,
within-subject, reliability, zero-shot, and threshold-calibration inputs retain their
frozen hashes. Two consecutive checkpoint-only rebuilds produced all 17 report files
byte-for-byte identically.

## Reproduce the Riemannian covariance-decoding study

The Riemannian protocol and final method were separately frozen before evaluation:

```text
131e601  initial covariance/tangent-space protocol
edcea46  leakage-safe implementation
8adad9c  subjects 1–20 LOSO development
c2f132f  final Ledoit–Wolf tangent-space LDA freeze
908921a  held-out evaluator implementation
```

```bash
python scripts/develop_riemannian_decoder.py
python scripts/evaluate_riemannian_decoder.py
```

The evaluator fits exactly one training-only affine-invariant Riemannian reference,
tangent transform, scaler, and shrinkage LDA on 900 trials from subjects 1–20. It
then predicts 3,840 trials from 86 participants independently. It reuses validated
8–30 Hz tensors, stores compact SPD covariance checkpoints under
`outputs/riemannian_decoder_checkpoints/`, and regenerates the tracked
`docs/assets/riemannian_decoder_evaluation_` report. Expected primary median is
0.573; matched Riemannian-minus-CSP median is −0.024.

## Generated and excluded content

| Path | Purpose | Git policy |
| --- | --- | --- |
| `.venv/` | Local Python environment | Excluded |
| `data/` | Reproducibly downloaded PhysioNet data | Excluded |
| `outputs/` | Validated but regenerable full-cohort checkpoints and disposable verification output | Excluded |
| `__pycache__/` | Python bytecode cache | Excluded |
| `docs/assets/` | Small reproducible figures/tables used by the public report | Tracked |

The tracked report assets can be regenerated with the visualization, signal-quality,
spectral, preprocessing, epoching, event-related, trial-quality, and cohort commands
above. Bulk or disposable intermediate outputs should remain under `outputs/` rather
than being added to the report.

## Determinism and current limits

The decoder uses fixed leave-one-run-out splits and no randomized model
initialization. Cohort bootstrap and plotting jitter use fixed seeds. Repeated
CSV/JSON bytes are stable in
the verified environment; PNG metadata/rendering may be library-
or platform-sensitive even when plotted values are unchanged.

Current reproducibility limitations:

- only direct dependencies are pinned;
- the operating system and system libraries are not captured in a container or lock file;
- the script depends on PhysioNet availability for the initial download;
- exact download timing and log order can vary without affecting the metadata result.

The fitted classifiers and full feature arrays are intentionally not persisted; exact
reconstruction therefore also depends on the pinned direct library versions and the
validated raw EDF bytes.

## Authoritative data source

Schalk, G. (2009). [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (version 1.0.0). PhysioNet. <https://doi.org/10.13026/C28G6P>
