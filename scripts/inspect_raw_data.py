"""Download and inspect the raw EEG recordings used in the first experiment.

This script intentionally performs no preprocessing. Its purpose is to verify
what the dataset actually contains before any scientific decisions are made.
"""

from collections import Counter
from pathlib import Path

import mne
from mne.datasets import eegbci


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
SUBJECT = 1
RUNS = (6, 10, 14)


def describe_recording(run: int, file_path: Path, raw: mne.io.BaseRaw) -> None:
    """Print the metadata needed to understand one continuous EEG run."""
    sampling_frequency = float(raw.info["sfreq"])
    sample_interval_ms = 1_000 / sampling_frequency
    duration_seconds = raw.n_times / sampling_frequency
    channel_type_counts = Counter(raw.get_channel_types())
    annotation_counts = Counter(str(description) for description in raw.annotations.description)

    print(f"\nRun {run}")
    print(f"  File: {file_path.name}")
    print(f"  Data shape (channels, samples): ({len(raw.ch_names)}, {raw.n_times})")
    print(f"  Channel types: {dict(channel_type_counts)}")
    print(f"  Sampling frequency: {sampling_frequency:g} Hz")
    print(f"  Time between samples: {sample_interval_ms:.2f} ms")
    print(f"  Approximate duration: {duration_seconds:.2f} s")
    print(f"  Annotation counts: {dict(annotation_counts)}")


def main() -> None:
    """Download the selected files and report each run's raw metadata."""
    DATA_DIRECTORY.mkdir(exist_ok=True)
    downloaded_files = eegbci.load_data(
        subjects=SUBJECT,
        runs=list(RUNS),
        path=DATA_DIRECTORY,
        update_path=False,
    )

    print(f"Subject: {SUBJECT}")
    print(f"Runs: {list(RUNS)} (motor imagery: both fists vs both feet)")

    for run, downloaded_file in zip(RUNS, downloaded_files, strict=True):
        file_path = Path(downloaded_file)
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose="error")
        describe_recording(run, file_path, raw)


if __name__ == "__main__":
    main()
