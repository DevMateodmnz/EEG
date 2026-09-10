"""Create reproducible raw-EEG and annotation visualizations.

The workflow is intentionally exploratory: it standardizes EEGBCI channel
names for readability but does not filter, re-reference, reject artifacts, or
otherwise modify the recorded voltage signals.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import Patch
from mne.datasets import eegbci


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
MOTOR_IMAGERY_RUNS = (6, 10, 14)
DEFAULT_CHANNELS = ("Fp1", "C3", "Cz", "C4", "Pz", "Oz")

ANNOTATION_LABELS = {
    "T0": "Rest",
    "T1": "Both fists imagery",
    "T2": "Both feet imagery",
}
ANNOTATION_COLORS = {
    "T0": "#b8b8b8",
    "T1": "#4c78a8",
    "T2": "#f58518",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for one reproducible visualization run."""
    parser = argparse.ArgumentParser(
        description="Visualize unfiltered EEGBCI motor-imagery data and annotations."
    )
    parser.add_argument("--subject", type=int, default=1, help="EEGBCI subject (1-109).")
    parser.add_argument(
        "--run",
        type=int,
        choices=MOTOR_IMAGERY_RUNS,
        default=6,
        help="Hands-versus-feet motor-imagery run.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=8.0,
        help="Start of the selected raw-data window in seconds.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=9.0,
        help="Duration of the selected raw-data window in seconds.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=list(DEFAULT_CHANNELS),
        help="Standardized EEG channel names to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for static figures and the annotation CSV.",
    )
    return parser.parse_args()


def load_recording(subject: int, run: int) -> tuple[mne.io.BaseRaw, Path]:
    """Download one recording if needed and open it without preloading samples."""
    if not 1 <= subject <= 109:
        raise ValueError(f"Subject must be between 1 and 109; received {subject}.")

    (downloaded_file,) = eegbci.load_data(
        subjects=subject,
        runs=run,
        path=DATA_DIRECTORY,
        update_path=False,
    )
    file_path = Path(downloaded_file)
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose="error")

    # EEGBCI EDF labels contain capitalization differences and trailing periods.
    # This official helper makes names such as "C3.." readable as "C3" without
    # changing the recorded voltage values.
    eegbci.standardize(raw)
    return raw, file_path


def validate_window_and_channels(
    raw: mne.io.BaseRaw,
    start_seconds: float,
    duration_seconds: float,
    channels: Sequence[str],
) -> tuple[float, int, int]:
    """Validate plotting choices and return end time plus sample boundaries."""
    if start_seconds < 0:
        raise ValueError("Window start must be zero or greater.")
    if duration_seconds <= 0:
        raise ValueError("Window duration must be greater than zero.")
    if not channels:
        raise ValueError("At least one channel must be selected.")

    missing_channels = sorted(set(channels) - set(raw.ch_names))
    if missing_channels:
        raise ValueError(f"Channels not present after standardization: {missing_channels}")

    end_seconds = start_seconds + duration_seconds
    if end_seconds > raw.duration:
        raise ValueError(
            f"Requested window ends at {end_seconds:.3f} s, "
            f"but the recording duration is {raw.duration:.3f} s."
        )

    start_sample, stop_sample = raw.time_as_index(
        [start_seconds, end_seconds], use_rounding=True
    )
    if stop_sample <= start_sample:
        raise ValueError("The requested time window contains no samples.")

    return end_seconds, int(start_sample), int(stop_sample)


def annotation_rows(raw: mne.io.BaseRaw) -> list[dict[str, int | float | str]]:
    """Return annotations with dataset labels and recording-relative samples."""
    rows: list[dict[str, int | float | str]] = []
    for index, annotation in enumerate(raw.annotations):
        onset_seconds = float(annotation["onset"])
        duration_seconds = float(annotation["duration"])
        description = str(annotation["description"])
        sample_index = int(raw.time_as_index([onset_seconds], use_rounding=True)[0])
        rows.append(
            {
                "index": index,
                "description": description,
                "condition": ANNOTATION_LABELS.get(description, "Unknown"),
                "onset_seconds": onset_seconds,
                "duration_seconds": duration_seconds,
                "sample_index": sample_index,
            }
        )
    return rows


def save_annotation_csv(
    rows: Sequence[dict[str, int | float | str]], output_path: Path
) -> None:
    """Save the complete annotation timing table as reproducible evidence."""
    fieldnames = (
        "index",
        "description",
        "condition",
        "onset_seconds",
        "duration_seconds",
        "sample_index",
    )
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotation_legend() -> list[Patch]:
    """Create a consistent legend for all annotation visualizations."""
    return [
        Patch(
            facecolor=ANNOTATION_COLORS[description],
            alpha=0.35,
            label=f"{description}: {label}",
        )
        for description, label in ANNOTATION_LABELS.items()
    ]


def plot_selected_channels(
    raw: mne.io.BaseRaw,
    subject: int,
    run: int,
    channels: Sequence[str],
    start_seconds: float,
    end_seconds: float,
    start_sample: int,
    stop_sample: int,
    output_path: Path,
) -> tuple[float, float]:
    """Plot unfiltered raw voltages in microvolts with annotation intervals."""
    data_volts, times = raw.get_data(
        picks=list(channels),
        start=start_sample,
        stop=stop_sample,
        return_times=True,
    )
    data_microvolts = raw.get_data(
        picks=list(channels),
        start=start_sample,
        stop=stop_sample,
        units="uV",
    )

    # Verify the public MNE unit conversion explicitly: 1 V = 1,000,000 µV.
    if not np.allclose(data_microvolts, data_volts * 1_000_000):
        raise RuntimeError("MNE voltage-to-microvolt conversion check failed.")

    amplitude_limit = float(np.max(np.abs(data_microvolts)) * 1.05)
    figure, axes = plt.subplots(
        len(channels),
        1,
        figsize=(13, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)

    for axis, channel, values in zip(axes_array, channels, data_microvolts, strict=True):
        axis.plot(times, values, color="#222222", linewidth=0.75)
        axis.axhline(0, color="#777777", linewidth=0.5, linestyle="--")
        axis.set_ylabel(channel, rotation=0, ha="right", va="center")
        axis.set_ylim(-amplitude_limit, amplitude_limit)
        axis.grid(axis="x", alpha=0.2)

        for annotation in raw.annotations:
            onset = float(annotation["onset"])
            offset = onset + float(annotation["duration"])
            if onset < end_seconds and offset > start_seconds:
                description = str(annotation["description"])
                axis.axvspan(
                    max(onset, start_seconds),
                    min(offset, end_seconds),
                    color=ANNOTATION_COLORS.get(description, "#cccccc"),
                    alpha=0.14,
                    linewidth=0,
                )

    axes_array[-1].set_xlabel("Time from recording start (s)")
    figure.supylabel("Raw amplitude (µV; stored reference)")
    figure.suptitle(
        f"Subject {subject}, run {run} — unfiltered raw EEG with annotation intervals\n"
        f"Window {start_seconds:g}–{end_seconds:g} s; common amplitude scale across channels",
        fontsize=13,
    )
    figure.legend(
        handles=annotation_legend(),
        loc="outside upper right",
        frameon=False,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return float(data_microvolts.min()), float(data_microvolts.max())


def plot_annotation_timeline(
    raw: mne.io.BaseRaw,
    subject: int,
    run: int,
    rows: Sequence[dict[str, int | float | str]],
    output_path: Path,
) -> None:
    """Plot every annotated interval across the complete recording."""
    figure, axis = plt.subplots(figsize=(15, 3.2), constrained_layout=True)

    for row in rows:
        onset = float(row["onset_seconds"])
        duration = float(row["duration_seconds"])
        description = str(row["description"])
        axis.broken_barh(
            [(onset, duration)],
            (0, 1),
            facecolors=ANNOTATION_COLORS.get(description, "#cccccc"),
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )
        axis.text(
            onset + duration / 2,
            0.5,
            description,
            ha="center",
            va="center",
            fontsize=6,
            color="white" if description != "T0" else "#333333",
        )

    axis.set_xlim(0, raw.duration)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xlabel("Time from recording start (s)")
    axis.set_title(f"Subject {subject}, run {run} — complete annotation timeline")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(handles=annotation_legend(), loc="upper center", ncol=3, frameon=False)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_template_electrode_locations(raw: mne.io.BaseRaw, output_path: Path) -> None:
    """Plot standard template locations, never claiming measured coordinates."""
    raw_with_template = raw.copy()
    standard_montage = mne.channels.make_standard_montage("standard_1005")
    missing_channels = sorted(set(raw_with_template.ch_names) - set(standard_montage.ch_names))
    if missing_channels:
        raise RuntimeError(
            "Cannot attach standard_1005 template; missing channel names: "
            f"{missing_channels}"
        )
    raw_with_template.set_montage(standard_montage, on_missing="raise")

    figure = raw_with_template.plot_sensors(
        kind="topomap",
        ch_type="eeg",
        show_names=True,
        show=False,
    )
    figure.set_size_inches(8, 8)
    figure.suptitle(
        "EEGBCI channel labels on MNE standard_1005 template\n"
        "Template positions — not subject-digitized coordinates",
        fontsize=12,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_annotation_table(rows: Sequence[dict[str, int | float | str]]) -> None:
    """Print exact annotation timing and summarize observed durations."""
    print("\nAnnotations")
    print(" idx  code  onset (s)  duration (s)  sample  condition")
    for row in rows:
        print(
            f"{int(row['index']):>4}  "
            f"{str(row['description']):<4}  "
            f"{float(row['onset_seconds']):>9.4f}  "
            f"{float(row['duration_seconds']):>12.4f}  "
            f"{int(row['sample_index']):>6}  "
            f"{row['condition']}"
        )

    durations_by_description: dict[str, set[float]] = defaultdict(set)
    for row in rows:
        durations_by_description[str(row["description"])].add(
            float(row["duration_seconds"])
        )
    print("\nObserved durations by description:")
    for description in ANNOTATION_LABELS:
        durations = sorted(durations_by_description[description])
        print(f"  {description}: {durations} s")


def time_token(value: float) -> str:
    """Format a time for descriptive, filesystem-safe figure names."""
    if value.is_integer():
        return f"{int(value):03d}"
    return f"{value:05.1f}".replace(".", "p")


def display_path(path: Path) -> Path:
    """Prefer a project-relative path, while supporting external output folders."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main() -> None:
    """Generate all static evidence for one raw-visualization run."""
    arguments = parse_arguments()
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    raw, file_path = load_recording(arguments.subject, arguments.run)
    end_seconds, start_sample, stop_sample = validate_window_and_channels(
        raw,
        arguments.start,
        arguments.duration,
        arguments.channels,
    )
    rows = annotation_rows(raw)

    prefix = f"subject{arguments.subject:02d}_run{arguments.run:02d}"
    window_token = f"{time_token(arguments.start)}-{time_token(end_seconds)}s"
    raw_figure_path = output_directory / (
        f"{prefix}_raw_selected_channels_{window_token}.png"
    )
    timeline_figure_path = output_directory / f"{prefix}_annotation_timeline.png"
    electrode_figure_path = output_directory / (
        f"{prefix}_electrode_locations_standard1005.png"
    )
    annotation_csv_path = output_directory / f"{prefix}_annotations.csv"

    minimum_uv, maximum_uv = plot_selected_channels(
        raw,
        arguments.subject,
        arguments.run,
        arguments.channels,
        arguments.start,
        end_seconds,
        start_sample,
        stop_sample,
        raw_figure_path,
    )
    plot_annotation_timeline(
        raw,
        arguments.subject,
        arguments.run,
        rows,
        timeline_figure_path,
    )
    plot_template_electrode_locations(raw, electrode_figure_path)
    save_annotation_csv(rows, annotation_csv_path)

    print(f"Loaded: {file_path}")
    print(f"Shape (channels, samples): ({len(raw.ch_names)}, {raw.n_times})")
    print(f"Sampling frequency: {float(raw.info['sfreq']):g} Hz")
    print("MNE internal EEG unit: volts; plotted unit: microvolts")
    print(f"Custom MNE reference applied: {bool(raw.info['custom_ref_applied'])}")
    print("Named physical reference in accessible metadata: not identified")
    print("Subject-digitized electrode coordinates in EDF: not present")
    print("Electrode figure coordinates: MNE standard_1005 template")
    print(f"Selected channels: {list(arguments.channels)}")
    print(
        f"Selected window: {arguments.start:g}-{end_seconds:g} s "
        f"(samples {start_sample}:{stop_sample})"
    )
    print(f"Observed selected-window amplitude range: {minimum_uv:.2f} to {maximum_uv:.2f} µV")
    print_annotation_table(rows)
    print("\nGenerated report artifacts:")
    for generated_path in (
        raw_figure_path,
        timeline_figure_path,
        electrode_figure_path,
        annotation_csv_path,
    ):
        print(f"  {display_path(generated_path)}")


if __name__ == "__main__":
    main()
