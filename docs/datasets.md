# Datasets and provenance

## Completed primary data: PhysioNet EEG Motor Movement/Imagery Database

The repository uses public EEGMMIDB recordings through MNE's `eegbci` utilities. The project documentation specifies the selected motor-imagery runs, annotation meanings, 64-channel/160 Hz structure, and raw-data checks in [Dataset](dataset.md) and [raw-data inspection](data_inspection.md). Raw EDF files are downloaded under `data/`, ignored by Git, and remain the responsibility of their original provider/authors. This repository distributes code, configurations, and derived frozen artifacts—not a replacement copy of the source dataset.

The completed full-cohort workflow records subjects 88, 92, and 100 as technical exclusions because they violate the frozen sampling/annotation assumptions. That is a compatibility statement, not a data-quality or participant-performance judgment.

## Ongoing external data: Lee2019/OpenBMI

The external protocol uses MOABB `Lee2019_MI`, data DOI `10.5524/100542`, and NEMAR deposit `nm000338`. It requires 54 participants × two labelled offline MI sessions (108 MAT sources), uses 62 EEG channels while excluding EMG from decoder features, and retains native 1000 Hz sampling. NEMAR's sourcedata manifest supplies upstream filenames, byte counts, and SHA-256 values. Partial files are not accepted as sources.

The provider change from rate-limited GigaDB to NEMAR occurred before outcome evaluation. Acquisition is resumable with `scripts/acquire_external_cross_session_lee2019.py`; no Lee2019 raw data, cache, or partial source belongs in Git.

## Redistribution and release boundary

`LICENSE` applies to Mateo's original code and original project documentation unless otherwise indicated. It does not relicense source datasets, third-party software, papers, or externally authored materials. Check the original providers' terms before redistributing raw data or making legal claims about derived artifacts. The current repository still contains historic absolute local source paths inside some hash-linked derived artifacts; see [release audit note](public_release_audit.md).
