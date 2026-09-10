"""Analyze paired-rest event-related Morlet power without classification."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.epoching import (  # noqa: E402
    DEFAULT_EPOCHING_CONFIG_PATH,
    RunEpochs,
    calculate_trial_quality,
    extract_run_epochs,
    load_epoching_config,
)
from eeg_project.preprocessing import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_preprocessing_config,
)
from eeg_project.time_frequency import (  # noqa: E402
    DEFAULT_EVENT_RELATED_CONFIG_PATH,
    EventRelatedSpectralConfig,
    FrequencyBand,
    SpectralEpochDataset,
    TimeFrequencyPower,
    assemble_spectral_epoch_dataset,
    band_frequency_mask,
    band_percent_time_courses,
    compute_morlet_power,
    descriptive_statistics,
    interval_mask,
    load_event_related_spectral_config,
    normalize_task_tfr_percent,
    percent_power_change,
    trial_band_measurements,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_RUNS = (6, 10, 14)
CONDITIONS = ("both_fists_imagery", "both_feet_imagery")
CONDITION_COLORS = {
    "both_fists_imagery": "#4c78a8",
    "both_feet_imagery": "#f58518",
}
ANALYSIS_SETS = (
    "all_trials",
    "exclude_qc_candidates",
    "exclude_first_rest_pair_per_run",
    "exclude_qc_and_first_rest_pair",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired-rest event-related Morlet power and descriptive "
            "fists/feet comparisons without rejecting trials or training a model."
        )
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument(
        "--runs", type=int, nargs="+", choices=DEFAULT_RUNS, default=list(DEFAULT_RUNS)
    )
    parser.add_argument(
        "--preprocessing-config", type=Path, default=DEFAULT_CONFIG_PATH
    )
    parser.add_argument("--epoching-config", type=Path, default=DEFAULT_EPOCHING_CONFIG_PATH)
    parser.add_argument(
        "--spectral-config", type=Path, default=DEFAULT_EVENT_RELATED_CONFIG_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not 1 <= arguments.subject <= 109:
        raise ValueError("Subject must be between 1 and 109.")
    if len(set(arguments.runs)) != len(arguments.runs):
        raise ValueError("Run numbers must not be duplicated.")


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path.resolve()


def artifact_prefix(subject: int, runs: Sequence[int]) -> str:
    return f"subject{subject:02d}_runs" + "-".join(f"{run:02d}" for run in runs)


def write_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {output_path}.")
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def belongs_to_analysis_set(row: Mapping[str, object], analysis_set: str) -> bool:
    is_candidate = row["quality_status"] == "statistical candidate"
    is_first = int(row["run_trial_index"]) == 0
    if analysis_set == "all_trials":
        return True
    if analysis_set == "exclude_qc_candidates":
        return not is_candidate
    if analysis_set == "exclude_first_rest_pair_per_run":
        return not is_first
    if analysis_set == "exclude_qc_and_first_rest_pair":
        return not is_candidate and not is_first
    raise ValueError(f"Unknown analysis set: {analysis_set}.")


def summarize_trial_rows(
    trial_rows: Sequence[dict[str, object]],
    group_fields: Sequence[str],
    analysis_sets: Sequence[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for analysis_set in analysis_sets:
        grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
        for row in trial_rows:
            if belongs_to_analysis_set(row, analysis_set):
                key = tuple(row[field] for field in group_fields)
                grouped[key].append(float(row["change_percent"]))
        for key in sorted(grouped, key=lambda values: tuple(str(value) for value in values)):
            output.append(
                {
                    "analysis_set": analysis_set,
                    **dict(zip(group_fields, key, strict=True)),
                    **descriptive_statistics(np.array(grouped[key], dtype=float)),
                }
            )
    return output


def build_condition_differences(
    condition_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (
            row["analysis_set"],
            row["channel"],
            row["frequency_band"],
            row["semantic_condition"],
        ): row
        for row in condition_rows
    }
    output: list[dict[str, object]] = []
    for analysis_set in ANALYSIS_SETS:
        channels = sorted(
            {
                str(row["channel"])
                for row in condition_rows
                if row["analysis_set"] == analysis_set
            }
        )
        bands = sorted(
            {
                str(row["frequency_band"])
                for row in condition_rows
                if row["analysis_set"] == analysis_set
            }
        )
        for channel in channels:
            for band in bands:
                fists = lookup[(analysis_set, channel, band, CONDITIONS[0])]
                feet = lookup[(analysis_set, channel, band, CONDITIONS[1])]
                output.append(
                    {
                        "analysis_set": analysis_set,
                        "channel": channel,
                        "frequency_band": band,
                        "fists_trial_count": fists["trial_count"],
                        "feet_trial_count": feet["trial_count"],
                        "fists_median_change_percent": fists["median_change_percent"],
                        "feet_median_change_percent": feet["median_change_percent"],
                        "feet_minus_fists_median_percentage_points": (
                            float(feet["median_change_percent"])
                            - float(fists["median_change_percent"])
                        ),
                        "fists_iqr_percentage_points": fists[
                            "iqr_change_percentage_points"
                        ],
                        "feet_iqr_percentage_points": feet[
                            "iqr_change_percentage_points"
                        ],
                    }
                )
    return output


def build_window_sensitivity_rows(
    task_power: TimeFrequencyPower,
    rest_power: TimeFrequencyPower,
    dataset: SpectralEpochDataset,
    config: EventRelatedSpectralConfig,
) -> list[dict[str, object]]:
    labels = np.array([pair.task.semantic_condition for pair in dataset.pairs])
    output: list[dict[str, object]] = []
    for rest_interval in config.rest_window_sensitivity_seconds:
        rest_mask = interval_mask(rest_power.times_seconds, rest_interval)
        for task_interval in config.task_window_sensitivity_seconds:
            task_mask = interval_mask(task_power.times_seconds, task_interval)
            for band in config.bands:
                frequency_mask = band_frequency_mask(task_power.frequencies_hz, band)
                reference = np.mean(
                    rest_power.power_v2[:, :, frequency_mask][:, :, :, rest_mask],
                    axis=(2, 3),
                )
                task = np.mean(
                    task_power.power_v2[:, :, frequency_mask][:, :, :, task_mask],
                    axis=(2, 3),
                )
                changes = percent_power_change(task, reference)
                for channel in config.sensorimotor_channels:
                    channel_index = dataset.channel_names.index(channel)
                    for condition in CONDITIONS:
                        values = changes[labels == condition, channel_index]
                        output.append(
                            {
                                "task_interval_start_seconds": task_interval[0],
                                "task_interval_stop_seconds": task_interval[1],
                                "rest_interval_start_seconds": rest_interval[0],
                                "rest_interval_stop_seconds": rest_interval[1],
                                "selected_primary_windows": bool(
                                    task_interval == config.task_analysis_interval_seconds
                                    and rest_interval
                                    == config.paired_rest_reference_interval_seconds
                                ),
                                "frequency_band": band.name,
                                "channel": channel,
                                "semantic_condition": condition,
                                **descriptive_statistics(values),
                            }
                        )
    return output


def build_time_course_rows(
    task_power: TimeFrequencyPower,
    rest_power: TimeFrequencyPower,
    dataset: SpectralEpochDataset,
    config: EventRelatedSpectralConfig,
    quality_rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    labels = np.array([pair.task.semantic_condition for pair in dataset.pairs])
    candidate = np.array(
        [row["quality_status"] == "statistical candidate" for row in quality_rows]
    )
    sensor_indices = [dataset.channel_names.index(c) for c in config.sensorimotor_channels]
    rows: list[dict[str, object]] = []
    course_by_band: dict[str, np.ndarray] = {}
    for band in config.bands:
        courses = band_percent_time_courses(
            task_power,
            rest_power,
            band,
            config.paired_rest_reference_interval_seconds,
        )[:, sensor_indices]
        course_by_band[band.name] = courses
        for analysis_set in ("all_trials", "exclude_qc_candidates"):
            allowed = np.ones(labels.size, dtype=bool)
            if analysis_set == "exclude_qc_candidates":
                allowed &= ~candidate
            for condition in CONDITIONS:
                trial_mask = allowed & (labels == condition)
                for channel_index, channel in enumerate(config.sensorimotor_channels):
                    values = courses[trial_mask, channel_index]
                    q25, median, q75 = np.percentile(values, [25, 50, 75], axis=0)
                    mean = np.mean(values, axis=0)
                    for time_index, time in enumerate(task_power.times_seconds):
                        rows.append(
                            {
                                "analysis_set": analysis_set,
                                "semantic_condition": condition,
                                "channel": channel,
                                "frequency_band": band.name,
                                "time_seconds": float(time),
                                "trial_count": int(values.shape[0]),
                                "mean_change_percent": float(mean[time_index]),
                                "median_change_percent": float(median[time_index]),
                                "q25_change_percent": float(q25[time_index]),
                                "q75_change_percent": float(q75[time_index]),
                            }
                        )
    return rows, course_by_band


def build_sensorimotor_tfr_rows(
    normalized_sensor_tfr: np.ndarray,
    task_power: TimeFrequencyPower,
    dataset: SpectralEpochDataset,
    config: EventRelatedSpectralConfig,
) -> list[dict[str, object]]:
    labels = np.array([pair.task.semantic_condition for pair in dataset.pairs])
    display_mask = interval_mask(task_power.times_seconds, config.display_interval_seconds)
    display_indices = np.flatnonzero(display_mask)
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        values = normalized_sensor_tfr[labels == condition]
        q25, median, q75 = np.percentile(values, [25, 50, 75], axis=0)
        mean = np.mean(values, axis=0)
        for channel_index, channel in enumerate(config.sensorimotor_channels):
            for frequency_index, frequency in enumerate(task_power.frequencies_hz):
                for time_index in display_indices:
                    rows.append(
                        {
                            "semantic_condition": condition,
                            "channel": channel,
                            "frequency_hz": float(frequency),
                            "time_seconds": float(task_power.times_seconds[time_index]),
                            "trial_count": int(values.shape[0]),
                            "mean_change_percent": float(
                                mean[channel_index, frequency_index, time_index]
                            ),
                            "median_change_percent": float(
                                median[channel_index, frequency_index, time_index]
                            ),
                            "q25_change_percent": float(
                                q25[channel_index, frequency_index, time_index]
                            ),
                            "q75_change_percent": float(
                                q75[channel_index, frequency_index, time_index]
                            ),
                        }
                    )
    return rows


def plot_sensorimotor_tfr(
    normalized_sensor_tfr: np.ndarray,
    task_power: TimeFrequencyPower,
    dataset: SpectralEpochDataset,
    config: EventRelatedSpectralConfig,
    output_path: Path,
) -> None:
    labels = np.array([pair.task.semantic_condition for pair in dataset.pairs])
    display_mask = interval_mask(task_power.times_seconds, config.display_interval_seconds)
    medians = np.stack(
        [np.median(normalized_sensor_tfr[labels == condition], axis=0) for condition in CONDITIONS]
    )
    displayed = medians[..., display_mask]
    limit = max(25.0, float(np.percentile(np.abs(displayed), 98)))
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, sharey=True,
                               constrained_layout=True)
    image = None
    times = task_power.times_seconds[display_mask]
    for row_index, channel in enumerate(config.sensorimotor_channels):
        for column_index, condition in enumerate(CONDITIONS):
            axis = axes[row_index, column_index]
            image = axis.pcolormesh(
                times,
                task_power.frequencies_hz,
                displayed[column_index, row_index],
                shading="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            axis.axvline(0, color="black", linestyle=":", linewidth=1)
            axis.axvline(1, color="#555555", linestyle="--", linewidth=0.8)
            axis.axvline(3, color="#555555", linestyle="--", linewidth=0.8)
            if row_index == 0:
                axis.set_title(condition.replace("_", " "))
            if column_index == 0:
                axis.set_ylabel(f"{channel}\nFrequency (Hz)")
            if row_index == 2:
                axis.set_xlabel("Time relative to task onset (s)")
    assert image is not None
    figure.colorbar(image, ax=axes, label="Median paired-rest power change (%)", shrink=0.85)
    figure.suptitle(
        "Sensorimotor time–frequency power — trial medians; one shared symmetric scale",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_band_time_courses(
    courses: np.ndarray,
    band: FrequencyBand,
    task_times: np.ndarray,
    dataset: SpectralEpochDataset,
    config: EventRelatedSpectralConfig,
    output_path: Path,
) -> None:
    labels = np.array([pair.task.semantic_condition for pair in dataset.pairs])
    display_mask = interval_mask(task_times, config.display_interval_seconds)
    figure, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                               constrained_layout=True)
    for channel_index, (axis, channel) in enumerate(
        zip(axes, config.sensorimotor_channels, strict=True)
    ):
        for condition in CONDITIONS:
            values = courses[labels == condition, channel_index]
            q25, median, q75 = np.percentile(values, [25, 50, 75], axis=0)
            color = CONDITION_COLORS[condition]
            axis.plot(task_times[display_mask], median[display_mask], color=color,
                      label=condition.replace("_", " "))
            axis.fill_between(task_times[display_mask], q25[display_mask], q75[display_mask],
                              color=color, alpha=0.18)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.axvline(0, color="black", linestyle=":")
        axis.axvspan(1, 3, color="#777777", alpha=0.08)
        axis.set_ylabel(f"{channel}\nChange (%)")
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Time relative to task onset (s)")
    figure.suptitle(
        f"{band.name.replace('_', ' ')} ({band.lower_hz:g}–{band.upper_hz:g} Hz) "
        "paired-rest change — median and IQR",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trial_distributions(
    trial_rows: Sequence[dict[str, object]],
    config: EventRelatedSpectralConfig,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), sharey="row",
                               constrained_layout=True)
    for row_index, band_name in enumerate(("mu", "beta")):
        for column_index, channel in enumerate(config.sensorimotor_channels):
            axis = axes[row_index, column_index]
            for condition_index, condition in enumerate(CONDITIONS):
                selected = [
                    row for row in trial_rows
                    if row["frequency_band"] == band_name
                    and row["channel"] == channel
                    and row["semantic_condition"] == condition
                ]
                values = np.array([float(row["change_percent"]) for row in selected])
                offsets = np.linspace(-0.13, 0.13, len(values))
                candidates = np.array(
                    [row["quality_status"] == "statistical candidate" for row in selected]
                )
                x = condition_index + offsets
                axis.scatter(x[~candidates], values[~candidates], s=24,
                             color=CONDITION_COLORS[condition], alpha=0.7)
                axis.scatter(x[candidates], values[candidates], s=65, facecolors="none",
                             edgecolors="#d62728", linewidths=1.2)
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                axis.vlines(condition_index, q25, q75, color="black", linewidth=4)
                axis.scatter([condition_index], [median], marker="_", s=180,
                             color="white", linewidths=2, zorder=4)
            axis.axhline(0, color="#555555", linewidth=0.8)
            axis.set_xticks([0, 1], ["fists", "feet"])
            axis.set_title(channel)
            axis.grid(axis="y", alpha=0.18)
            if column_index == 0:
                axis.set_ylabel(f"{band_name} change (%)")
    figure.suptitle(
        "Trial-level paired-rest changes — black bars are IQR; red rings are QC candidates",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_run_and_qc_sensitivity(
    run_rows: Sequence[dict[str, object]],
    condition_rows: Sequence[dict[str, object]],
    config: EventRelatedSpectralConfig,
    output_path: Path,
) -> None:
    measurements = [
        (channel, band) for band in ("mu", "beta")
        for channel in config.sensorimotor_channels
    ]
    primary_run = {
        (int(row["run"]), row["semantic_condition"], row["channel"], row["frequency_band"]):
        float(row["median_change_percent"])
        for row in run_rows if row["analysis_set"] == "all_trials"
    }
    run_labels = [
        (run, condition) for run in DEFAULT_RUNS for condition in CONDITIONS
        if (run, condition, "C3", "mu") in primary_run
    ]
    run_matrix = np.array([
        [primary_run[(run, condition, channel, band)] for channel, band in measurements]
        for run, condition in run_labels
    ])
    summary = {
        (row["analysis_set"], row["semantic_condition"], row["channel"], row["frequency_band"]):
        float(row["median_change_percent"])
        for row in condition_rows
    }
    qc_matrix = np.array([
        [
            summary[("exclude_qc_candidates", condition, channel, band)]
            - summary[("all_trials", condition, channel, band)]
            for channel, band in measurements
        ]
        for condition in CONDITIONS
    ])
    limit = max(25.0, float(np.percentile(np.abs(run_matrix), 98)))
    qc_limit = max(5.0, float(np.max(np.abs(qc_matrix))))
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    image = axes[0].imshow(run_matrix, cmap="RdBu_r", vmin=-limit, vmax=limit,
                           aspect="auto")
    axes[0].set_yticks(np.arange(len(run_labels)),
                       [f"run {run} {cond.split('_')[1]}" for run, cond in run_labels])
    axes[0].set_xticks(np.arange(len(measurements)),
                       [f"{channel} {band}" for channel, band in measurements])
    axes[0].set_title("Run-level median paired-rest change (%)")
    figure.colorbar(image, ax=axes[0], label="Change (%)")
    image_qc = axes[1].imshow(qc_matrix, cmap="PiYG", vmin=-qc_limit, vmax=qc_limit,
                              aspect="auto")
    axes[1].set_yticks([0, 1], ["both fists", "both feet"])
    axes[1].set_xticks(np.arange(len(measurements)),
                       [f"{channel} {band}" for channel, band in measurements])
    axes[1].set_title("Median shift after excluding six QC candidates (percentage points)")
    figure.colorbar(image_qc, ax=axes[1], label="Non-candidate minus all-trial median")
    figure.suptitle("Run repeatability and QC-candidate sensitivity", fontsize=14)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_band_topographies(
    trial_rows: Sequence[dict[str, object]],
    run_results: Mapping[int, RunEpochs],
    config: EventRelatedSpectralConfig,
    output_path: Path,
) -> None:
    first = run_results[next(iter(run_results))]
    info = first.task_epochs.info.copy()
    info.set_montage(mne.channels.make_standard_montage("standard_1005"),
                     on_missing="raise")
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for band in ("mu", "beta"):
        for condition in CONDITIONS:
            values = []
            for channel in first.task_epochs.ch_names:
                selected = [
                    float(row["change_percent"]) for row in trial_rows
                    if row["frequency_band"] == band
                    and row["semantic_condition"] == condition
                    and row["channel"] == channel
                ]
                values.append(float(np.median(selected)))
            lookup[(band, condition)] = np.array(values)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row_index, band in enumerate(("mu", "beta")):
        fists = lookup[(band, CONDITIONS[0])]
        feet = lookup[(band, CONDITIONS[1])]
        maps = (fists, feet, feet - fists)
        limit = max(10.0, max(float(np.max(np.abs(values))) for values in maps))
        for column_index, (values, title) in enumerate(
            zip(maps, ("both fists", "both feet", "feet − fists"), strict=True)
        ):
            image, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=axes[row_index, column_index],
                show=False,
                cmap="RdBu_r",
                vlim=(-limit, limit),
                contours=6,
                sensors=True,
            )
            axes[row_index, column_index].set_title(f"{band}: {title}")
        figure.colorbar(image, ax=axes[row_index], label="Median change (%)",
                        shrink=0.75)
    figure.suptitle(
        "Sensor-level band-power maps — standard_1005 template; not source localization",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    preprocessing_path = arguments.preprocessing_config.expanduser().resolve()
    epoching_path = arguments.epoching_config.expanduser().resolve()
    spectral_path = arguments.spectral_config.expanduser().resolve()
    preprocessing_config = load_preprocessing_config(preprocessing_path)
    epoching_config = load_epoching_config(epoching_path)
    config = load_event_related_spectral_config(spectral_path)
    if not np.isclose(
        config.continuous_filter_half_support_seconds,
        epoching_config.filter_half_support_seconds,
    ):
        raise RuntimeError("Spectral and epoching filter-support records differ.")
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    run_results = {
        run: extract_run_epochs(
            arguments.subject, run, preprocessing_config, epoching_config
        )
        for run in arguments.runs
    }
    dataset = assemble_spectral_epoch_dataset(run_results, arguments.runs)
    task_records = [pair.task for pair in dataset.pairs]
    quality_rows = calculate_trial_quality(
        dataset.task_data_volts,
        dataset.task_times,
        dataset.channel_names,
        task_records,
        epoching_config.trial_quality,
    )
    quality_by_identity = {
        (int(row["run"]), int(row["run_trial_index"])): row for row in quality_rows
    }

    task_power = compute_morlet_power(
        dataset.task_data_volts,
        dataset.task_times,
        dataset.sampling_frequency_hz,
        config,
    )
    rest_power = compute_morlet_power(
        dataset.rest_data_volts,
        dataset.rest_times,
        dataset.sampling_frequency_hz,
        config,
    )
    trial_rows = trial_band_measurements(
        task_power, rest_power, dataset, config, quality_by_identity
    )
    for row in trial_rows:
        row["source_file"] = str(display_path(Path(str(row["source_file"]))))
    condition_rows = summarize_trial_rows(
        trial_rows,
        ("semantic_condition", "channel", "frequency_band"),
        ANALYSIS_SETS,
    )
    run_rows = summarize_trial_rows(
        trial_rows,
        ("run", "semantic_condition", "channel", "frequency_band"),
        ("all_trials", "exclude_qc_candidates"),
    )
    difference_rows = build_condition_differences(condition_rows)
    window_rows = build_window_sensitivity_rows(
        task_power, rest_power, dataset, config
    )
    time_course_rows, course_by_band = build_time_course_rows(
        task_power, rest_power, dataset, config, quality_rows
    )
    sensor_indices = [dataset.channel_names.index(c) for c in config.sensorimotor_channels]
    task_sensor = TimeFrequencyPower(
        task_power.power_v2[:, sensor_indices],
        task_power.frequencies_hz,
        task_power.times_seconds,
        task_power.n_cycles,
        task_power.wavelet_lengths_samples,
    )
    rest_sensor = TimeFrequencyPower(
        rest_power.power_v2[:, sensor_indices],
        rest_power.frequencies_hz,
        rest_power.times_seconds,
        rest_power.n_cycles,
        rest_power.wavelet_lengths_samples,
    )
    normalized_sensor = normalize_task_tfr_percent(
        task_sensor, rest_sensor, config.paired_rest_reference_interval_seconds
    )
    tfr_rows = build_sensorimotor_tfr_rows(
        normalized_sensor, task_power, dataset, config
    )

    prefix = artifact_prefix(arguments.subject, arguments.runs)
    paths = {
        "trial": output_directory / f"{prefix}_event_trial_band_power.csv",
        "condition": output_directory / f"{prefix}_event_condition_band_summary.csv",
        "run": output_directory / f"{prefix}_event_run_band_summary.csv",
        "difference": output_directory / f"{prefix}_event_condition_difference.csv",
        "window": output_directory / f"{prefix}_event_window_sensitivity.csv",
        "time_course": output_directory / f"{prefix}_event_band_time_courses.csv",
        "tfr": output_directory / f"{prefix}_event_sensorimotor_tfr.csv",
        "metadata": output_directory / f"{prefix}_event_spectral_metadata.json",
        "tfr_figure": output_directory / f"{prefix}_event_sensorimotor_tfr.png",
        "mu_figure": output_directory / f"{prefix}_event_mu_time_courses.png",
        "beta_figure": output_directory / f"{prefix}_event_beta_time_courses.png",
        "distribution_figure": output_directory / f"{prefix}_event_trial_distributions.png",
        "sensitivity_figure": output_directory / f"{prefix}_event_run_qc_sensitivity.png",
        "topography_figure": output_directory / f"{prefix}_event_band_topographies.png",
    }
    for key, rows in (
        ("trial", trial_rows),
        ("condition", condition_rows),
        ("run", run_rows),
        ("difference", difference_rows),
        ("window", window_rows),
        ("time_course", time_course_rows),
        ("tfr", tfr_rows),
    ):
        write_csv(rows, paths[key])

    plot_sensorimotor_tfr(
        normalized_sensor, task_power, dataset, config, paths["tfr_figure"]
    )
    band_lookup = {band.name: band for band in config.bands}
    plot_band_time_courses(
        course_by_band["mu"], band_lookup["mu"], task_power.times_seconds,
        dataset, config, paths["mu_figure"]
    )
    plot_band_time_courses(
        course_by_band["beta"], band_lookup["beta"], task_power.times_seconds,
        dataset, config, paths["beta_figure"]
    )
    plot_trial_distributions(trial_rows, config, paths["distribution_figure"])
    plot_run_and_qc_sensitivity(
        run_rows, condition_rows, config, paths["sensitivity_figure"]
    )
    plot_band_topographies(
        trial_rows, run_results, config, paths["topography_figure"]
    )

    sensor_summary = [
        row for row in condition_rows
        if row["analysis_set"] == "all_trials"
        and row["channel"] in config.sensorimotor_channels
    ]
    wavelet_half_support = float(
        np.max(task_power.wavelet_lengths_samples - 1)
        / (2 * dataset.sampling_frequency_hz)
    )
    metadata = {
        "schema_version": 1,
        "subject": arguments.subject,
        "runs": list(arguments.runs),
        "input_task_shape": list(dataset.task_data_volts.shape),
        "input_rest_shape": list(dataset.rest_data_volts.shape),
        "task_tfr_shape": list(task_power.power_v2.shape),
        "rest_tfr_shape": list(rest_power.power_v2.shape),
        "full_tfr_arrays_persisted": False,
        "frequency_centers_hz": task_power.frequencies_hz.tolist(),
        "task_tfr_times_seconds": {
            "start": float(task_power.times_seconds[0]),
            "stop": float(task_power.times_seconds[-1]),
            "step": float(np.median(np.diff(task_power.times_seconds))),
            "count": int(task_power.times_seconds.size),
        },
        "wavelet": {
            "n_cycles_formula": "frequency_hz * 0.5",
            "minimum_cycles": float(task_power.n_cycles.min()),
            "maximum_cycles": float(task_power.n_cycles.max()),
            "wavelet_length_samples_unique": sorted(
                set(int(value) for value in task_power.wavelet_lengths_samples)
            ),
            "wavelet_half_support_seconds": wavelet_half_support,
            "continuous_filter_half_support_seconds": (
                config.continuous_filter_half_support_seconds
            ),
            "combined_nominal_half_support_seconds": (
                wavelet_half_support + config.continuous_filter_half_support_seconds
            ),
        },
        "reference": {
            "strategy": "trial-paired immediately preceding T0",
            "interval_seconds_from_t0_onset": list(
                config.paired_rest_reference_interval_seconds
            ),
            "normalization": "100 * (task_power - reference_power) / reference_power",
        },
        "task_interval_seconds": list(config.task_analysis_interval_seconds),
        "bands_hz_inclusive_centers": {
            band.name: [band.lower_hz, band.upper_hz] for band in config.bands
        },
        "trial_counts": dict(
            sorted(Counter(pair.task.semantic_condition for pair in dataset.pairs).items())
        ),
        "quality": {
            "statistical_candidates": sum(
                row["quality_status"] == "statistical candidate" for row in quality_rows
            ),
            "confirmed_exclusions": 0,
            "primary_analysis_uses_all_trials": True,
        },
        "sensorimotor_primary_summary": sensor_summary,
        "software": {
            "python": sys.version.split()[0],
            "mne": mne.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "configuration_paths": {
            "preprocessing": str(display_path(preprocessing_path)),
            "epoching": str(display_path(epoching_path)),
            "event_related_spectral": str(display_path(spectral_path)),
        },
        "report_artifacts": [path.name for path in paths.values()],
    }
    with paths["metadata"].open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    print("Event-related spectral analysis (descriptive, single subject)")
    print(f"Subject: {arguments.subject}; runs: {arguments.runs}")
    print(f"Task/rest inputs: {dataset.task_data_volts.shape} / {dataset.rest_data_volts.shape}")
    print(f"Task/rest TFR: {task_power.power_v2.shape} / {rest_power.power_v2.shape}")
    print(
        f"Morlet: {task_power.frequencies_hz[0]:g}-{task_power.frequencies_hz[-1]:g} Hz, "
        f"1-Hz centers, n_cycles=f/2, {task_power.wavelet_lengths_samples[0]} samples"
    )
    print(
        "Paired T0 reference: 1.1-3.1 s; task summary: 1-3 s; "
        "normalization: percent change"
    )
    print(
        f"Nominal half-support: wavelet ±{wavelet_half_support:.3f} s + "
        f"FIR ±{config.continuous_filter_half_support_seconds:.2f} s = "
        f"±{wavelet_half_support + config.continuous_filter_half_support_seconds:.3f} s"
    )
    printed_counts = Counter(pair.task.semantic_condition for pair in dataset.pairs)
    print(
        f"Primary task trials: {len(dataset.pairs)} "
        f"({printed_counts['both_fists_imagery']} fists, "
        f"{printed_counts['both_feet_imagery']} feet); automatic exclusions: 0"
    )
    print("QC sensitivity excludes six candidates only in a copied comparison mask")
    print("\nC3/Cz/C4 all-trial medians (%):")
    for row in sensor_summary:
        print(
            f"  {row['semantic_condition']} {row['channel']} "
            f"{row['frequency_band']}: {float(row['median_change_percent']):+.2f}% "
            f"(IQR {float(row['q25_change_percent']):+.2f} to "
            f"{float(row['q75_change_percent']):+.2f})"
        )
    print("\nGenerated report artifacts:")
    for path in paths.values():
        print(f"  {display_path(path)}")


if __name__ == "__main__":
    main()
