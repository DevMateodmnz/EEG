# Raw-data inspection

[Back to the project README](../README.md)

## Purpose

The first data milestone verifies what the selected recordings actually contain before applying signal processing. This prevents later code from relying on untested assumptions about channel count, sample count, timing, channel types, or annotations.

No preprocessing is performed during this milestone.

## Implementation

The executable inspection is in [`scripts/inspect_raw_data.py`](../scripts/inspect_raw_data.py).

The script:

1. Defines subject 1 and runs 6, 10, and 14 explicitly.
2. Downloads the corresponding EDF files with `mne.datasets.eegbci.load_data`.
3. Opens each file with `mne.io.read_raw_edf(..., preload=False)`.
4. Reads the number of channels, samples, channel types, and sampling frequency.
5. Calculates the sample interval and approximate recording duration.
6. Counts each annotation description.

`preload=False` means the entire signal matrix is not copied into memory during metadata inspection. MNE can read samples from disk later when needed. Operations such as filtering will require data to be loaded, but filtering is outside this milestone.

## Calculations

For sampling frequency $f_s$, the sample interval is:

\[
\Delta t = \frac{1}{f_s}
\]

With $f_s=160\ \text{Hz}$:

\[
\Delta t = 6.25\ \text{ms}
\]

For $N=20{,}000$ samples, the approximate amount of recorded data is:

\[
\frac{N}{f_s} = \frac{20{,}000}{160} = 125\ \text{s}
\]

## Executed results

The script was executed successfully against locally downloaded PhysioNet files.

| Run | File | Shape | Channel types | $f_s$ | Sample interval | Approx. duration | Annotation counts |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 6 | `S001R06.edf` | `(64, 20,000)` | 64 EEG | 160 Hz | 6.25 ms | 125 s | `T0`: 15, `T1`: 7, `T2`: 8 |
| 10 | `S001R10.edf` | `(64, 20,000)` | 64 EEG | 160 Hz | 6.25 ms | 125 s | `T0`: 15, `T1`: 7, `T2`: 8 |
| 14 | `S001R14.edf` | `(64, 20,000)` | 64 EEG | 160 Hz | 6.25 ms | 125 s | `T0`: 15, `T1`: 7, `T2`: 8 |

Aggregated across the selected runs:

| Annotation | Count | Interpretation |
| --- | ---: | --- |
| `T0` | 45 | Rest annotations |
| `T1` | 21 | Both-fists imagery onsets |
| `T2` | 24 | Both-feet imagery onsets |

The difference between 21 and 24 is a small class imbalance. Later evaluation must use stratification and class-aware metrics where appropriate instead of assuming identical class counts.

## What these results establish

The executed inspection establishes that:

- all three files are accessible and readable with the installed MNE version;
- each file contains 64 channels classified by MNE as EEG;
- each file contains 20,000 samples per channel at 160 Hz;
- annotations for rest, fists imagery, and feet imagery are present;
- the two future task classes have unequal annotation counts.

## What these results do not establish

Metadata does not show whether:

- the signals have appropriate amplitudes;
- channels contain artifacts, flat segments, or excessive noise;
- channel locations and referencing are configured correctly for later spatial analysis;
- every annotation will produce a usable epoch;
- the two task conditions are statistically or predictively distinguishable.

Raw waveform, annotation, reference, and montage questions are continued in [Raw EEG visualization and annotation timing](raw_visualization.md). Signal-quality, preprocessing, epoching, and evaluation questions require later milestones.

## Run the inspection

From an activated project environment:

```bash
python scripts/inspect_raw_data.py
```

The first run downloads approximately 7.4 MB. A successful run prints one metadata block for each selected recording.

## Sources

- Schalk, G. (2009). [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (version 1.0.0). PhysioNet.
- MNE-Python contributors. [`mne.datasets.eegbci.load_data`](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html).
- MNE-Python contributors. [Algorithms and implementation details](https://mne.tools/stable/documentation/implementation.html): memory-efficient raw-data access.
