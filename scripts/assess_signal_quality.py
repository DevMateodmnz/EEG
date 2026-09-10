"""Assess raw EEG signal quality and compare an average-reference representation.

This is an exploratory quality-control workflow, not an automatic channel
rejection system. It never modifies the downloaded EDF files, marks channels
bad, filters data, or removes artifacts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
MOTOR_IMAGERY_RUNS = (6, 10, 14)
DEFAULT_RUNS = (6, 10, 14)
DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_ROBUST_Z_THRESHOLD = 3.5
REFERENCE_COMPARISON_CHANNELS = ("Fp1", "C3", "Cz", "C4", "Pz", "Oz")
TEMPORAL_CHANNELS = (
    "Fp1",
    "Fpz",
    "Fp2",
    "AF7",
    "AF3",
    "AFz",
    "AF4",
    "AF8",
    "C3",
    "Cz",
    "C4",
)
MODIFIED_Z_SCALE = 0.6744897501960817

ANNOTATION_COLORS = {
    "T0": "#b8b8b8",
    "T1": "#4c78a8",
    "T2": "#f58518",
}

def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for a reproducible quality assessment."""
    parser = argparse.ArgumentParser(
        description=(
            "Quantify unfiltered EEGBCI signal quality and compare average reference."
        )
    )
    parser.add_argument("--subject", type=int, default=1, help="EEGBCI subject (1-109).")
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        choices=MOTOR_IMAGERY_RUNS,
        default=list(DEFAULT_RUNS),
        help="One or more hands-versus-feet motor-imagery runs.",
    )
    parser.add_argument(
        "--report-run",
        type=int,
        choices=MOTOR_IMAGERY_RUNS,
        help="Run used for figures; defaults to the first requested run.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Length of non-overlapping temporal-quality windows.",
    )
    parser.add_argument(
        "--robust-z-threshold",
        type=float,
        default=DEFAULT_ROBUST_Z_THRESHOLD,
        help="Exploratory modified robust-z candidate threshold.",
    )
    parser.add_argument(
        "--comparison-start",
        type=float,
        default=8.0,
        help="Start time for the reference-comparison figure.",
    )
    parser.add_argument(
        "--comparison-duration",
        type=float,
        default=9.0,
        help="Duration of the reference-comparison figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for report figures and CSV tables.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> int:
    """Validate scientific configuration and return the selected report run."""
    if not 1 <= arguments.subject <= 109:
        raise ValueError(f"Subject must be between 1 and 109; received {arguments.subject}.")
    if arguments.window_seconds <= 0:
        raise ValueError("Temporal window length must be greater than zero.")
    if arguments.robust_z_threshold <= 0:
        raise ValueError("Robust-z threshold must be greater than zero.")
    if arguments.comparison_start < 0 or arguments.comparison_duration <= 0:
        raise ValueError("Reference-comparison start/duration must define a positive window.")
    if len(set(arguments.runs)) != len(arguments.runs):
        raise ValueError("Run numbers must not be duplicated.")

    report_run = arguments.report_run or arguments.runs[0]
    if report_run not in arguments.runs:
        raise ValueError("--report-run must also be included in --runs.")
    return report_run


def load_recording(subject: int, run: int) -> tuple[mne.io.BaseRaw, Path]:
    """Load one authoritative EDF and standardize labels without preprocessing."""
    (downloaded_file,) = eegbci.load_data(
        subjects=subject,
        runs=run,
        path=DATA_DIRECTORY,
        update_path=False,
    )
    file_path = Path(downloaded_file)
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose="error")
    eegbci.standardize(raw)
    return raw, file_path


def read_edf_signal_header(file_path: Path) -> dict[str, list[str]]:
    """Read per-signal EDF header fields needed for transparent metadata QC."""
    with file_path.open("rb") as edf_file:
        fixed_header = edf_file.read(256)
        if len(fixed_header) != 256:
            raise RuntimeError(f"Incomplete EDF fixed header: {file_path}")
        signal_count = int(fixed_header[252:256].decode("ascii").strip())
        signal_header = edf_file.read(signal_count * 256)

    field_widths = (
        ("label", 16),
        ("transducer", 80),
        ("physical_dimension", 8),
        ("physical_minimum", 8),
        ("physical_maximum", 8),
        ("digital_minimum", 8),
        ("digital_maximum", 8),
        ("prefiltering", 80),
        ("samples_per_record", 8),
        ("reserved", 32),
    )
    fields: dict[str, list[str]] = {}
    offset = 0
    for name, width in field_widths:
        fields[name] = [
            signal_header[offset + index * width : offset + (index + 1) * width]
            .decode("ascii")
            .strip()
            for index in range(signal_count)
        ]
        offset += width * signal_count
    return fields


def robust_z_scores(values: np.ndarray) -> np.ndarray:
    """Calculate modified z-scores from the cross-channel median and MAD."""
    center = float(np.median(values))
    dispersion = float(np.median(np.abs(values - center)))
    if np.isclose(dispersion, 0.0):
        return np.zeros_like(values, dtype=float)
    return MODIFIED_Z_SCALE * (values - center) / dispersion


def longest_identical_run(samples: np.ndarray) -> int:
    """Return the largest number of consecutive exactly equal samples."""
    if samples.size == 0:
        return 0
    boundaries = np.flatnonzero(np.r_[True, samples[1:] != samples[:-1], True])
    return int(np.max(np.diff(boundaries)))


def calculate_channel_metrics(
    subject: int,
    run: int,
    raw: mne.io.BaseRaw,
    data_uV: np.ndarray,
    threshold: float,
) -> list[dict[str, int | float | str]]:
    """Calculate interpretable whole-recording metrics for every EEG channel."""
    differences = np.diff(data_uV, axis=1)
    medians = np.median(data_uV, axis=1)
    means = np.mean(data_uV, axis=1)
    standard_deviations = np.std(data_uV, axis=1, ddof=0)
    peak_to_peak = np.ptp(data_uV, axis=1)
    maximum_steps = np.max(np.abs(differences), axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = np.corrcoef(data_uV)
    np.fill_diagonal(correlations, np.nan)
    median_correlations = np.nanmedian(correlations, axis=1)

    robust_scores = {
        "mean": robust_z_scores(means),
        "std": robust_z_scores(standard_deviations),
        "peak_to_peak": robust_z_scores(peak_to_peak),
        "max_abs_step": robust_z_scores(maximum_steps),
        "median_peer_correlation": robust_z_scores(median_correlations),
    }

    rows: list[dict[str, int | float | str]] = []
    for index, channel in enumerate(raw.ch_names):
        evidence: list[str] = []
        if abs(robust_scores["mean"][index]) > threshold:
            evidence.append(f"unusual mean offset (rz={robust_scores['mean'][index]:.2f})")
        if robust_scores["std"][index] > threshold:
            evidence.append(f"high standard deviation (rz={robust_scores['std'][index]:.2f})")
        elif robust_scores["std"][index] < -threshold:
            evidence.append(f"low standard deviation (rz={robust_scores['std'][index]:.2f})")
        if robust_scores["peak_to_peak"][index] > threshold:
            evidence.append(
                f"high peak-to-peak amplitude (rz={robust_scores['peak_to_peak'][index]:.2f})"
            )
        if robust_scores["max_abs_step"][index] > threshold:
            evidence.append(
                f"large sample step (rz={robust_scores['max_abs_step'][index]:.2f})"
            )
        if robust_scores["median_peer_correlation"][index] < -threshold:
            evidence.append(
                "low median peer correlation "
                f"(rz={robust_scores['median_peer_correlation'][index]:.2f})"
            )

        channel_values = data_uV[index]
        channel_differences = differences[index]
        longest_run_samples = longest_identical_run(channel_values)
        rows.append(
            {
                "subject": subject,
                "run": run,
                "channel": channel,
                "mean_uV": float(means[index]),
                "std_uV": float(standard_deviations[index]),
                "variance_uV2": float(np.var(channel_values, ddof=0)),
                "minimum_uV": float(np.min(channel_values)),
                "maximum_uV": float(np.max(channel_values)),
                "peak_to_peak_uV": float(peak_to_peak[index]),
                "median_uV": float(medians[index]),
                "mad_uV": float(np.median(np.abs(channel_values - medians[index]))),
                "iqr_uV": float(
                    np.percentile(channel_values, 75)
                    - np.percentile(channel_values, 25)
                ),
                "max_abs_step_uV": float(maximum_steps[index]),
                "identical_step_fraction": float(np.mean(channel_differences == 0)),
                "longest_identical_run_samples": longest_run_samples,
                "longest_identical_run_seconds": float(
                    longest_run_samples / raw.info["sfreq"]
                ),
                "median_peer_correlation": float(median_correlations[index]),
                "mean_robust_z": float(robust_scores["mean"][index]),
                "std_robust_z": float(robust_scores["std"][index]),
                "peak_to_peak_robust_z": float(
                    robust_scores["peak_to_peak"][index]
                ),
                "max_abs_step_robust_z": float(
                    robust_scores["max_abs_step"][index]
                ),
                "median_peer_correlation_robust_z": float(
                    robust_scores["median_peer_correlation"][index]
                ),
                "candidate_status": (
                    "statistically unusual" if evidence else "not flagged by heuristic"
                ),
                "candidate_evidence": "; ".join(evidence),
            }
        )
    return rows


def calculate_temporal_metrics(
    subject: int,
    run: int,
    raw: mne.io.BaseRaw,
    data_uV: np.ndarray,
    window_seconds: float,
) -> list[dict[str, int | float | str]]:
    """Summarize channel amplitude and common flatness in short windows."""
    window_samples = int(round(window_seconds * float(raw.info["sfreq"])))
    if window_samples < 2:
        raise ValueError("Temporal windows must contain at least two samples.")

    selected_indices = {
        channel: raw.ch_names.index(channel)
        for channel in TEMPORAL_CHANNELS
        if channel in raw.ch_names
    }
    rows: list[dict[str, int | float | str]] = []
    for window_index, start_sample in enumerate(
        range(0, raw.n_times, window_samples)
    ):
        stop_sample = min(start_sample + window_samples, raw.n_times)
        window = data_uV[:, start_sample:stop_sample]
        channel_peak_to_peak = np.ptp(window, axis=1)
        channel_standard_deviation = np.std(window, axis=1, ddof=0)
        maximum_index = int(np.argmax(channel_peak_to_peak))
        simultaneous_identical_fraction = float(
            np.mean(np.all(np.diff(window, axis=1) == 0, axis=0))
        )
        row: dict[str, int | float | str] = {
            "subject": subject,
            "run": run,
            "window_index": window_index,
            "start_seconds": float(start_sample / raw.info["sfreq"]),
            "end_seconds": float(stop_sample / raw.info["sfreq"]),
            "sample_count": int(stop_sample - start_sample),
            "median_peak_to_peak_uV": float(np.median(channel_peak_to_peak)),
            "q75_peak_to_peak_uV": float(np.percentile(channel_peak_to_peak, 75)),
            "maximum_peak_to_peak_uV": float(channel_peak_to_peak[maximum_index]),
            "maximum_channel": raw.ch_names[maximum_index],
            "median_std_uV": float(np.median(channel_standard_deviation)),
            "simultaneous_identical_step_fraction": simultaneous_identical_fraction,
        }
        for channel, index in selected_indices.items():
            row[f"{channel}_peak_to_peak_uV"] = float(channel_peak_to_peak[index])
        rows.append(row)

    for channel in selected_indices:
        values = np.array(
            [float(row[f"{channel}_peak_to_peak_uV"]) for row in rows]
        )
        scores = robust_z_scores(values)
        for row, score in zip(rows, scores, strict=True):
            row[f"{channel}_peak_to_peak_robust_z"] = float(score)
    return rows


def find_common_constant_segments(
    raw: mne.io.BaseRaw, data_uV: np.ndarray
) -> list[tuple[int, int, float, float]]:
    """Locate intervals where every EEG channel stays constant through time."""
    all_channels_constant = np.all(np.diff(data_uV, axis=1) == 0, axis=0)
    # A constant difference at index k connects samples k and k+1.
    boundaries = np.flatnonzero(
        np.r_[True, all_channels_constant[1:] != all_channels_constant[:-1], True]
    )
    segments: list[tuple[int, int, float, float]] = []
    for start_difference, stop_difference in zip(
        boundaries[:-1], boundaries[1:], strict=True
    ):
        if not all_channels_constant[start_difference]:
            continue
        start_sample = int(start_difference)
        stop_sample = int(stop_difference + 1)
        segments.append(
            (
                start_sample,
                stop_sample,
                float(start_sample / raw.info["sfreq"]),
                float(stop_sample / raw.info["sfreq"]),
            )
        )
    return segments


def write_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    """Write dictionaries using stable field order from the first row."""
    if not rows:
        raise ValueError(f"Cannot write an empty table: {output_path}")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_channel_comparison(
    subject: int,
    run: int,
    rows: Sequence[dict[str, int | float | str]],
    threshold: float,
    output_path: Path,
) -> None:
    """Plot four complementary channel-level quality metrics."""
    channels = [str(row["channel"]) for row in rows]
    candidate_mask = np.array(
        [row["candidate_status"] == "statistically unusual" for row in rows]
    )
    x_positions = np.arange(len(channels))
    metrics = (
        ("peak_to_peak_uV", "Peak-to-peak (µV)"),
        ("std_uV", "Standard deviation (µV)"),
        ("max_abs_step_uV", "Maximum adjacent-sample step (µV)"),
        ("median_peer_correlation", "Median correlation with other channels"),
    )

    figure, axes = plt.subplots(4, 1, figsize=(18, 13), sharex=True, constrained_layout=True)
    for axis, (field, label) in zip(axes, metrics, strict=True):
        values = np.array([float(row[field]) for row in rows])
        colors = np.where(candidate_mask, "#d1495b", "#4c78a8")
        axis.bar(x_positions, values, color=colors, width=0.82)
        axis.axhline(np.median(values), color="#222222", linestyle="--", linewidth=1)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x_positions, channels, rotation=90, fontsize=7)
    axes[-1].set_xlabel("Standardized EEG channel")
    figure.suptitle(
        f"Subject {subject}, run {run} — unfiltered whole-recording channel quality\n"
        f"Red = candidate from at least one modified robust-z rule (|threshold| {threshold:g})",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_temporal_quality(
    subject: int,
    run: int,
    rows: Sequence[dict[str, int | float | str]],
    comparison_start: float,
    comparison_end: float,
    output_path: Path,
) -> None:
    """Show short-window amplitude, frontal behavior, and common flatness."""
    starts = np.array([float(row["start_seconds"]) for row in rows])
    ends = np.array([float(row["end_seconds"]) for row in rows])
    centers = (starts + ends) / 2
    median_values = np.array([float(row["median_peak_to_peak_uV"]) for row in rows])
    q75_values = np.array([float(row["q75_peak_to_peak_uV"]) for row in rows])
    maximum_values = np.array([float(row["maximum_peak_to_peak_uV"]) for row in rows])
    common_flatness = np.array(
        [float(row["simultaneous_identical_step_fraction"]) for row in rows]
    )

    figure, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True, constrained_layout=True)
    axes[0].plot(centers, maximum_values, label="Maximum channel", color="#d1495b")
    axes[0].plot(centers, q75_values, label="75th percentile", color="#f58518")
    axes[0].plot(centers, median_values, label="Channel median", color="#4c78a8")
    axes[0].set_ylabel("Window peak-to-peak (µV)")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.2)

    frontal_channels = ("Fp1", "Fpz", "Fp2", "AF7", "AF3")
    for channel in frontal_channels:
        field = f"{channel}_peak_to_peak_uV"
        if field not in rows[0]:
            continue
        values = np.array([float(row[field]) for row in rows])
        axes[1].plot(
            centers,
            values,
            label=channel,
            linewidth=2.2 if channel == "Fp1" else 1.0,
        )
    axes[1].set_ylabel("Frontal peak-to-peak (µV)")
    axes[1].legend(frameon=False, ncol=5)
    axes[1].grid(alpha=0.2)

    axes[2].bar(centers, common_flatness, width=ends - starts, color="#79706e")
    axes[2].set_ylabel("Fraction of simultaneous\nidentical adjacent steps")
    axes[2].set_xlabel("Time from recording start (s)")
    axes[2].set_ylim(0, max(0.55, float(common_flatness.max()) * 1.1))
    axes[2].grid(axis="y", alpha=0.2)

    for axis in axes:
        axis.axvspan(
            comparison_start,
            comparison_end,
            color="#72b7b2",
            alpha=0.14,
            label=None,
        )
    figure.suptitle(
        f"Subject {subject}, run {run} — temporal quality in non-overlapping windows\n"
        f"Green span marks the {comparison_start:g}–{comparison_end:g} s comparison region",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def shade_annotations(
    axis: plt.Axes,
    raw: mne.io.BaseRaw,
    start_seconds: float,
    end_seconds: float,
) -> None:
    """Add condition intervals without interpreting them as signal causes."""
    for annotation in raw.annotations:
        onset = float(annotation["onset"])
        offset = onset + float(annotation["duration"])
        if onset < end_seconds and offset > start_seconds:
            axis.axvspan(
                max(onset, start_seconds),
                min(offset, end_seconds),
                color=ANNOTATION_COLORS.get(str(annotation["description"]), "#cccccc"),
                alpha=0.11,
                linewidth=0,
            )


def plot_reference_comparison(
    subject: int,
    run: int,
    raw: mne.io.BaseRaw,
    start_seconds: float,
    duration_seconds: float,
    output_path: Path,
) -> tuple[float, float]:
    """Compare stored and average references on copies with identical scaling."""
    end_seconds = start_seconds + duration_seconds
    if end_seconds > raw.duration:
        raise ValueError(
            f"Reference-comparison window ends at {end_seconds:g} s, "
            f"after the {raw.duration:g} s recording."
        )
    missing = sorted(set(REFERENCE_COMPARISON_CHANNELS) - set(raw.ch_names))
    if missing:
        raise ValueError(f"Reference-comparison channels are missing: {missing}")

    start_sample, stop_sample = raw.time_as_index(
        [start_seconds, end_seconds], use_rounding=True
    )
    original_all_uV = raw.get_data(units="uV")
    original_snapshot = original_all_uV.copy()
    average_reference_uV = np.mean(original_all_uV, axis=0)

    average_referenced = raw.copy().load_data(verbose="error")
    average_referenced.set_eeg_reference(
        ref_channels="average", projection=False, verbose="error"
    )
    derived_all_uV = average_referenced.get_data(units="uV")

    if not np.array_equal(raw.get_data(units="uV"), original_snapshot):
        raise RuntimeError("The original Raw object changed during re-referencing.")
    if not np.allclose(
        derived_all_uV,
        original_all_uV - average_reference_uV,
        rtol=1e-12,
        atol=1e-9,
    ):
        raise RuntimeError("MNE average reference does not match the explicit equation.")
    maximum_channel_mean = float(np.max(np.abs(np.mean(derived_all_uV, axis=0))))
    if maximum_channel_mean > 1e-9:
        raise RuntimeError("Average-referenced channel mean is not numerically zero.")

    picks = [raw.ch_names.index(channel) for channel in REFERENCE_COMPARISON_CHANNELS]
    original_window = original_all_uV[picks, start_sample:stop_sample]
    derived_window = derived_all_uV[picks, start_sample:stop_sample]
    times = raw.times[start_sample:stop_sample]
    amplitude_limit = float(
        max(np.max(np.abs(original_window)), np.max(np.abs(derived_window))) * 1.05
    )

    figure, axes = plt.subplots(
        len(picks),
        2,
        figsize=(16, 12),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for row_index, channel in enumerate(REFERENCE_COMPARISON_CHANNELS):
        for column_index, (values, color) in enumerate(
            ((original_window[row_index], "#222222"), (derived_window[row_index], "#4c78a8"))
        ):
            axis = axes[row_index, column_index]
            axis.plot(times, values, color=color, linewidth=0.7)
            axis.axhline(0, color="#777777", linewidth=0.5, linestyle="--")
            axis.set_ylim(-amplitude_limit, amplitude_limit)
            axis.grid(axis="x", alpha=0.2)
            shade_annotations(axis, raw, start_seconds, end_seconds)
        axes[row_index, 0].set_ylabel(channel, rotation=0, ha="right", va="center")

    axes[0, 0].set_title("Raw EDF values — stored reference")
    axes[0, 1].set_title("Derived copy — 64-channel average reference")
    axes[-1, 0].set_xlabel("Time from recording start (s)")
    axes[-1, 1].set_xlabel("Time from recording start (s)")
    figure.supylabel("Amplitude (µV; identical scale in both columns)")
    figure.suptitle(
        f"Subject {subject}, run {run} — controlled reference comparison\n"
        f"Window {start_seconds:g}–{end_seconds:g} s; no filtering or artifact removal",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return float(np.min(average_reference_uV)), float(np.max(average_reference_uV))


def format_runs(runs: Sequence[int]) -> str:
    """Create a compact deterministic run token for artifact names."""
    return "-".join(f"{run:02d}" for run in runs)


def time_token(value: float) -> str:
    """Format a time for descriptive, filesystem-safe artifact names."""
    if value.is_integer():
        return f"{int(value):03d}"
    return f"{value:05.1f}".replace(".", "p")


def display_path(path: Path) -> Path:
    """Prefer a project-relative path while supporting external output paths."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main() -> None:
    """Run quality assessment, evidence export, and reference comparison."""
    arguments = parse_arguments()
    report_run = validate_arguments(arguments)
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    all_channel_rows: list[dict[str, int | float | str]] = []
    all_temporal_rows: list[dict[str, int | float | str]] = []
    raw_by_run: dict[int, mne.io.BaseRaw] = {}
    file_by_run: dict[int, Path] = {}
    common_segments_by_run: dict[int, list[tuple[int, int, float, float]]] = {}

    for run in arguments.runs:
        raw, file_path = load_recording(arguments.subject, run)
        raw_by_run[run] = raw
        file_by_run[run] = file_path
        data_volts = raw.get_data()
        data_uV = raw.get_data(units="uV")
        if not np.allclose(data_uV, data_volts * 1_000_000):
            raise RuntimeError(f"Voltage conversion failed for run {run}.")
        if data_uV.shape != (64, 20_000):
            raise RuntimeError(f"Unexpected run {run} shape: {data_uV.shape}")
        if not np.isfinite(data_uV).all():
            raise RuntimeError(f"Run {run} contains unexpected NaN or infinite values.")

        all_channel_rows.extend(
            calculate_channel_metrics(
                arguments.subject,
                run,
                raw,
                data_uV,
                arguments.robust_z_threshold,
            )
        )
        all_temporal_rows.extend(
            calculate_temporal_metrics(
                arguments.subject,
                run,
                raw,
                data_uV,
                arguments.window_seconds,
            )
        )
        common_segments_by_run[run] = find_common_constant_segments(raw, data_uV)

    run_token = format_runs(arguments.runs)
    prefix = f"subject{arguments.subject:02d}"
    channel_csv_path = output_directory / (
        f"{prefix}_runs{run_token}_channel_quality.csv"
    )
    temporal_csv_path = output_directory / (
        f"{prefix}_runs{run_token}_temporal_quality.csv"
    )
    report_prefix = f"{prefix}_run{report_run:02d}"
    channel_figure_path = output_directory / f"{report_prefix}_channel_quality.png"
    temporal_figure_path = output_directory / f"{report_prefix}_temporal_quality.png"
    comparison_end = arguments.comparison_start + arguments.comparison_duration
    comparison_token = (
        f"{time_token(arguments.comparison_start)}-{time_token(comparison_end)}s"
    )
    reference_figure_path = output_directory / (
        f"{report_prefix}_average_reference_comparison_{comparison_token}.png"
    )

    write_csv(all_channel_rows, channel_csv_path)
    write_csv(all_temporal_rows, temporal_csv_path)
    report_channel_rows = [
        row for row in all_channel_rows if int(row["run"]) == report_run
    ]
    report_temporal_rows = [
        row for row in all_temporal_rows if int(row["run"]) == report_run
    ]
    plot_channel_comparison(
        arguments.subject,
        report_run,
        report_channel_rows,
        arguments.robust_z_threshold,
        channel_figure_path,
    )
    plot_temporal_quality(
        arguments.subject,
        report_run,
        report_temporal_rows,
        arguments.comparison_start,
        comparison_end,
        temporal_figure_path,
    )
    reference_minimum, reference_maximum = plot_reference_comparison(
        arguments.subject,
        report_run,
        raw_by_run[report_run],
        arguments.comparison_start,
        arguments.comparison_duration,
        reference_figure_path,
    )

    print("Quantitative raw EEG signal-quality assessment")
    print(f"Subject: {arguments.subject}; runs: {arguments.runs}")
    print(f"Temporal window: {arguments.window_seconds:g} s (non-overlapping)")
    print(
        "Candidate heuristic: modified robust z-score, "
        f"exploratory threshold {arguments.robust_z_threshold:g}"
    )
    print("No channels were marked bad, removed, interpolated, or excluded.")
    print("No filtering or notch filtering was applied.")

    for run in arguments.runs:
        raw = raw_by_run[run]
        header = read_edf_signal_header(file_by_run[run])
        eeg_count = len(raw.ch_names)
        candidate_rows = [
            row
            for row in all_channel_rows
            if int(row["run"]) == run
            and row["candidate_status"] == "statistically unusual"
        ]
        print(f"\nRun {run}")
        print(f"  File: {file_by_run[run]}")
        print(f"  Shape: ({eeg_count}, {raw.n_times}); sfreq: {raw.info['sfreq']:g} Hz")
        print(f"  Existing raw.info['bads']: {raw.info['bads']}")
        print(f"  Custom MNE reference applied: {bool(raw.info['custom_ref_applied'])}")
        print("  Named physical reference: not identified in accessible metadata")
        print(
            "  EDF physical units/range: "
            f"{sorted(set(header['physical_dimension'][:eeg_count]))}; "
            f"{sorted(set(header['physical_minimum'][:eeg_count]))} to "
            f"{sorted(set(header['physical_maximum'][:eeg_count]))}"
        )
        print(
            "  EDF digital range: "
            f"{sorted(set(header['digital_minimum'][:eeg_count]))} to "
            f"{sorted(set(header['digital_maximum'][:eeg_count]))}"
        )
        print(
            "  EDF prefilter text: "
            f"{sorted(set(header['prefiltering'][:eeg_count]))}"
        )
        print(f"  Statistically unusual candidates ({len(candidate_rows)}):")
        for row in candidate_rows:
            print(f"    {row['channel']}: {row['candidate_evidence']}")
        print(f"  Common constant segments: {common_segments_by_run[run]}")

    report_fp1_windows = [
        row
        for row in report_temporal_rows
        if float(row["start_seconds"]) <= 14.0 < float(row["end_seconds"])
    ]
    if report_run == 6 and report_fp1_windows:
        row = report_fp1_windows[0]
        print("\nRun 6 Fp1 14-16 s evidence:")
        print(f"  Peak-to-peak: {float(row['Fp1_peak_to_peak_uV']):.2f} µV")
        print(
            "  Across-window modified robust z: "
            f"{float(row['Fp1_peak_to_peak_robust_z']):.2f}"
        )
        print(f"  Maximum channel in window: {row['maximum_channel']}")

    print("\nAverage-reference comparison")
    print("  Applied only to an in-memory copy; raw EDF and Raw object unchanged")
    print("  Explicit subtraction equation matched MNE output")
    print("  Across-channel mean after re-reference: numerically zero at every sample")
    print(
        "  Report-run average-reference signal range: "
        f"{reference_minimum:.2f} to {reference_maximum:.2f} µV"
    )
    print("  Future use remains conditional on deliberate candidate-channel review")

    print("\nGenerated report artifacts:")
    for generated_path in (
        channel_csv_path,
        temporal_csv_path,
        channel_figure_path,
        temporal_figure_path,
        reference_figure_path,
    ):
        print(f"  {display_path(generated_path)}")


if __name__ == "__main__":
    main()
