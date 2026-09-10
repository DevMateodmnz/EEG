"""Audit paired task/rest quality and spectral feature stability without ML."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

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
    assemble_spectral_epoch_dataset,
    band_frequency_mask,
    compute_morlet_power,
    descriptive_statistics,
    interval_mask,
    load_event_related_spectral_config,
    percent_power_change,
    trial_band_measurements,
)
from eeg_project.trial_quality import (  # noqa: E402
    DEFAULT_TRIAL_QUALITY_CONFIG_PATH,
    QUALITY_METRICS,
    add_normalization_diagnostics,
    build_candidate_influence_rows,
    build_feature_stability_rows,
    classify_metric_candidates,
    is_strong_review_candidate,
    load_trial_quality_config,
    paired_side_quality_metrics,
    systematic_matched_controls,
)


DEFAULT_RUNS = (6, 10, 14)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
CONDITIONS = ("both_fists_imagery", "both_feet_imagery")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate label-independent paired quality diagnostics and a predefined "
            "spectral feature-stability report. No epochs are deleted."
        )
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument(
        "--runs", type=int, nargs="+", choices=DEFAULT_RUNS, default=list(DEFAULT_RUNS)
    )
    parser.add_argument("--preprocessing-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--epoching-config", type=Path, default=DEFAULT_EPOCHING_CONFIG_PATH)
    parser.add_argument(
        "--spectral-config", type=Path, default=DEFAULT_EVENT_RELATED_CONFIG_PATH
    )
    parser.add_argument(
        "--quality-config", type=Path, default=DEFAULT_TRIAL_QUALITY_CONFIG_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not 1 <= arguments.subject <= 109:
        raise ValueError("Subject must be between 1 and 109.")
    if len(set(arguments.runs)) != len(arguments.runs):
        raise ValueError("Run numbers must not be duplicated.")


def artifact_prefix(subject: int, runs: Sequence[int]) -> str:
    return f"subject{subject:02d}_runs" + "-".join(f"{run:02d}" for run in runs)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_timing_rows(task_power, rest_power, dataset, spectral_config) -> list[dict[str, object]]:
    """Evaluate only the 12 timing combinations fixed in spectral configuration."""
    labels = np.array([pair.task.semantic_condition for pair in dataset.pairs])
    output: list[dict[str, object]] = []
    for rest_interval in spectral_config.rest_window_sensitivity_seconds:
        rest_mask = interval_mask(rest_power.times_seconds, rest_interval)
        for task_interval in spectral_config.task_window_sensitivity_seconds:
            task_mask = interval_mask(task_power.times_seconds, task_interval)
            for band in spectral_config.bands:
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
                for channel in spectral_config.sensorimotor_channels:
                    channel_index = dataset.channel_names.index(channel)
                    for condition in CONDITIONS:
                        values = changes[labels == condition, channel_index]
                        output.append(
                            {
                                "task_interval_start_seconds": task_interval[0],
                                "task_interval_stop_seconds": task_interval[1],
                                "rest_interval_start_seconds": rest_interval[0],
                                "rest_interval_stop_seconds": rest_interval[1],
                                "frequency_band": band.name,
                                "channel": channel,
                                "semantic_condition": condition,
                                **descriptive_statistics(values),
                            }
                        )
    return output


def nearest_channel_neighbors(channel_names: Sequence[str], count: int = 4) -> dict[str, list[str]]:
    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    output: dict[str, list[str]] = {}
    for channel in channel_names:
        distances = sorted(
            (
                (float(np.linalg.norm(positions[channel] - positions[other])), other)
                for other in channel_names
                if other != channel
            ),
            key=lambda item: (item[0], item[1]),
        )
        output[channel] = [other for _, other in distances[:count]]
    return output


def neighborhood_coherence(
    trial_data: np.ndarray,
    times: np.ndarray,
    channel_names: Sequence[str],
    channels: Sequence[str],
    neighbors: Mapping[str, Sequence[str]],
) -> str:
    """Summarize zero-lag coherence with four template-nearest sensors for review."""
    mask = (times >= 0) & (times <= 4)
    summaries: list[str] = []
    for channel in channels:
        own = trial_data[channel_names.index(channel), mask]
        correlations = [
            float(np.corrcoef(own, trial_data[channel_names.index(other), mask])[0, 1])
            for other in neighbors[channel]
        ]
        summaries.append(f"{channel}={np.median(correlations):.3f}")
    return ";".join(summaries)


def likely_interpretation(row: Mapping[str, object]) -> str:
    evidence = str(row["quality_evidence"])
    channels = str(row["channel_specific_p2p_candidate_channels"])
    if "frontal" in evidence or any(name in channels.split("/") for name in ("Fp1", "Fpz", "Fp2")):
        return "frontal/ocular-compatible transient; physical source unconfirmed"
    if "precontext" in evidence and "task" not in evidence.replace("precontext", ""):
        return "unusual pre-task context; not evidence of task-period failure"
    if "abs_step" in evidence:
        return "abrupt non-physiological-compatible transient; source unconfirmed"
    if channels:
        return "spatially distributed amplitude/variability anomaly; source unconfirmed"
    return "statistically unusual scalar metric; physical source unconfirmed"


def build_paired_rows(
    dataset,
    task_quality_rows: Sequence[Mapping[str, object]],
    task_pair_metrics: Sequence[Mapping[str, float]],
    rest_pair_metrics: Sequence[Mapping[str, float]],
    task_pair_evidence: Sequence[Sequence[str]],
    rest_pair_evidence: Sequence[Sequence[str]],
    rest_candidates: np.ndarray,
    quality_config,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (pair, original) in enumerate(zip(dataset.pairs, task_quality_rows, strict=True)):
        task_candidate = original["quality_status"] == "statistical candidate"
        rest_candidate = bool(rest_candidates[index])
        driver = (
            "both"
            if task_candidate and rest_candidate
            else "task_candidate"
            if task_candidate
            else "rest_candidate"
            if rest_candidate
            else "neither"
        )
        rows.append(
            {
                "subject": pair.task.subject,
                "run": pair.task.run,
                "run_trial_index": pair.task.run_trial_index,
                "semantic_condition": pair.task.semantic_condition,
                "task_annotation": pair.task.annotation,
                "task_event_sample": pair.task.event_sample,
                "task_event_time_seconds": pair.task.event_time_seconds,
                "rest_annotation": pair.rest.annotation,
                "rest_event_sample": pair.rest.event_sample,
                "rest_event_time_seconds": pair.rest.event_time_seconds,
                "task_quality_status": original["quality_status"],
                "task_quality_evidence": original["quality_evidence"],
                "strong_task_review_candidate": is_strong_review_candidate(
                    original, quality_config
                ),
                "paired_task_metric_status": (
                    "statistical candidate" if task_pair_evidence[index] else "not flagged"
                ),
                "paired_task_metric_evidence": ";".join(task_pair_evidence[index]),
                "paired_rest_quality_status": (
                    "statistical candidate" if rest_candidate else "not flagged"
                ),
                "paired_rest_quality_evidence": ";".join(rest_pair_evidence[index]),
                "pair_quality_driver": driver,
                **{f"task_{name}": value for name, value in task_pair_metrics[index].items()},
                **{f"rest_{name}": value for name, value in rest_pair_metrics[index].items()},
                "primary_retained": True,
                "source_file": display_path(pair.source_path),
                "source_sha256": pair.source_sha256,
            }
        )
    return rows


def build_candidate_review_rows(
    dataset,
    task_quality_rows,
    paired_rows,
    diagnostic_rows,
    matches,
) -> list[dict[str, object]]:
    quality_lookup = {
        (int(row["run"]), int(row["run_trial_index"])): row for row in task_quality_rows
    }
    paired_lookup = {
        (int(row["run"]), int(row["run_trial_index"])): row for row in paired_rows
    }
    pair_index = {
        (pair.task.run, pair.task.run_trial_index): index
        for index, pair in enumerate(dataset.pairs)
    }
    spectral_lookup = {
        (
            int(row["run"]),
            int(row["run_trial_index"]),
            str(row["channel"]),
            str(row["frequency_band"]),
        ): row
        for row in diagnostic_rows
    }
    neighbors = nearest_channel_neighbors(dataset.channel_names)
    output: list[dict[str, object]] = []
    for identity, matched_identity in matches.items():
        row = quality_lookup[identity]
        matched = quality_lookup[matched_identity]
        index = pair_index[identity]
        candidate_channels = [
            channel
            for channel in str(row["channel_specific_p2p_candidate_channels"]).split("/")
            if channel
        ]
        if not candidate_channels:
            task_mask = (dataset.task_times >= 0) & (dataset.task_times <= 4)
            p2p = np.ptp(dataset.task_data_volts[index, :, task_mask], axis=1)
            candidate_channels = [dataset.channel_names[int(np.argmax(p2p))]]
        coherence = neighborhood_coherence(
            dataset.task_data_volts[index],
            dataset.task_times,
            dataset.channel_names,
            candidate_channels,
            neighbors,
        )
        output.append(
            {
                "subject": row["subject"],
                "run": identity[0],
                "run_trial_index": identity[1],
                "semantic_condition": row["semantic_condition"],
                "event_time_seconds": row["event_time_seconds"],
                "quality_status": row["quality_status"],
                "quality_evidence": row["quality_evidence"],
                "max_full_epoch_p2p_uV": row["max_full_epoch_p2p_uV"],
                "max_task_p2p_uV": row["max_task_p2p_uV"],
                "median_task_p2p_uV": row["median_task_p2p_uV"],
                "max_precontext_p2p_uV": row["max_precontext_p2p_uV"],
                "frontal_max_task_p2p_uV": row["frontal_max_task_p2p_uV"],
                "max_task_abs_step_uV": row["max_task_abs_step_uV"],
                "median_task_channel_std_uV": row["median_task_channel_std_uV"],
                "driving_channels": "/".join(candidate_channels),
                "median_correlation_with_four_nearest_channels": coherence,
                "paired_rest_quality_status": paired_lookup[identity][
                    "paired_rest_quality_status"
                ],
                "paired_rest_max_p2p_uV": paired_lookup[identity]["rest_max_p2p_uV"],
                "paired_rest_frontal_max_p2p_uV": paired_lookup[identity][
                    "rest_frontal_max_p2p_uV"
                ],
                "paired_rest_max_abs_step_uV": paired_lookup[identity][
                    "rest_max_abs_step_uV"
                ],
                "paired_rest_median_channel_std_uV": paired_lookup[identity][
                    "rest_median_channel_std_uV"
                ],
                "paired_task_max_p2p_uV": paired_lookup[identity]["task_max_p2p_uV"],
                "paired_task_frontal_max_p2p_uV": paired_lookup[identity][
                    "task_frontal_max_p2p_uV"
                ],
                "paired_task_max_abs_step_uV": paired_lookup[identity][
                    "task_max_abs_step_uV"
                ],
                "pair_quality_driver": paired_lookup[identity]["pair_quality_driver"],
                "c3_12_13_change_percent_sensitivity_context_only": spectral_lookup[
                    (*identity, "C3", "central_12_13")
                ]["change_percent"],
                "c4_12_13_change_percent_sensitivity_context_only": spectral_lookup[
                    (*identity, "C4", "central_12_13")
                ]["change_percent"],
                "interpretation": likely_interpretation(row),
                "evidence_strength": (
                    "strong statistical review evidence; source unconfirmed"
                    if paired_lookup[identity]["strong_task_review_candidate"]
                    else "limited statistical review evidence; source unconfirmed"
                ),
                "current_decision": "retain primary; flag metadata; sensitivity only",
                "matched_run": matched_identity[0],
                "matched_run_trial_index": matched_identity[1],
                "matched_quality_status": matched["quality_status"],
                "matched_max_task_p2p_uV": matched["max_task_p2p_uV"],
                "matched_frontal_max_task_p2p_uV": matched[
                    "frontal_max_task_p2p_uV"
                ],
                "matched_max_task_abs_step_uV": matched["max_task_abs_step_uV"],
                "matching_method": (
                    "nearest robust-scaled QC vector; same run and condition; "
                    "deterministic trial-index tie break"
                ),
            }
        )
    return output


def consolidated_evidence_rows(candidate_count: int) -> list[dict[str, object]]:
    return [
        {
            "evidence_scope": "continuous boundary",
            "observation": "all 64 channels are exactly zero from 124.5 to 125.0 s in all three runs",
            "possible_explanation": "recording or file boundary padding",
            "evidence_strength": "strong for invalid non-physiological segment; exact cause unknown",
            "current_decision": "deterministically crop copied analysis data at 124.5 s",
        },
        {
            "evidence_scope": "continuous frontal channels",
            "observation": "run 6 Fp1/FPz/Fp2/AF7/AF3 have unusually high whole-run variability",
            "possible_explanation": "ocular or another widespread frontal source",
            "evidence_strength": "moderate pattern compatibility; no dedicated EOG",
            "current_decision": "retain channels; document and review",
        },
        {
            "evidence_scope": "temporal transient",
            "observation": "run 6, 14-16 s Fp1 reaches about 785 uV peak-to-peak with frontal neighbors involved",
            "possible_explanation": "blink, eye movement, movement, or electrode transient",
            "evidence_strength": "strong unusual-signal evidence; weak source identification",
            "current_decision": "retain uncertain event; no unvalidated correction",
        },
        {
            "evidence_scope": "run channel candidates",
            "observation": "5, 10, and 14 statistical channel candidates in runs 6, 10, and 14",
            "possible_explanation": "run-relative amplitude/variability differences or sensor behavior",
            "evidence_strength": "statistical candidate evidence only; no confirmed bad sensor",
            "current_decision": "retain; do not interpolate",
        },
        {
            "evidence_scope": "task trials",
            "observation": f"{candidate_count} of 45 task trials exceed predefined robust scalar criteria",
            "possible_explanation": "transient, widespread variability, or ordinary distribution tail",
            "evidence_strength": "reproducible candidate classification; source varies and remains uncertain",
            "current_decision": "retain primary; exclude only from copied sensitivity subsets",
        },
        {
            "evidence_scope": "event-related spectrum",
            "observation": "C3/C4 fists 12-13 Hz paired-rest reduction survives six-candidate exclusion",
            "possible_explanation": "main direction is not created by the flagged task trials",
            "evidence_strength": "strong sensitivity evidence, not artifact-source identification",
            "current_decision": "use only to assess robustness; never to assign QC status",
        },
    ]


def build_reference_influence_rows(diagnostic_rows, quality_config) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    features = set(quality_config.primary_features)
    for row in diagnostic_rows:
        feature = (
            str(row["semantic_condition"]),
            str(row["channel"]),
            str(row["frequency_band"]),
        )
        if feature not in features:
            continue
        reference_z = float(row["reference_log_power_robust_z"])
        task_z = float(row["task_log_power_robust_z"])
        threshold = quality_config.robust_z_threshold
        if reference_z < -threshold and task_z > threshold:
            driver = "low_reference_and_high_task"
        elif reference_z < -threshold:
            driver = "low_reference"
        elif task_z > threshold:
            driver = "high_task"
        elif reference_z > threshold and task_z < -threshold:
            driver = "high_reference_and_low_task"
        elif reference_z > threshold:
            driver = "high_reference"
        elif task_z < -threshold:
            driver = "low_task"
        else:
            driver = "neither_robust_outlier"
        output.append(
            {
                "subject": row["subject"],
                "run": row["run"],
                "run_trial_index": row["run_trial_index"],
                "semantic_condition": row["semantic_condition"],
                "task_event_time_seconds": row["task_event_time_seconds"],
                "quality_status": row["quality_status"],
                "channel": row["channel"],
                "frequency_band": row["frequency_band"],
                "reference_mean_wavelet_power_v2": row["reference_mean_wavelet_power_v2"],
                "task_mean_wavelet_power_v2": row["task_mean_wavelet_power_v2"],
                "change_percent": row["change_percent"],
                "change_db": row["change_db"],
                "reference_log_power_robust_z": reference_z,
                "task_log_power_robust_z": task_z,
                "spectral_extreme_driver": driver,
                "primary_retained": True,
            }
        )
    return output


def plot_candidate_gallery(dataset, candidate_reviews, output_path: Path) -> None:
    index_lookup = {
        (pair.task.run, pair.task.run_trial_index): index
        for index, pair in enumerate(dataset.pairs)
    }
    figure, axes = plt.subplots(len(candidate_reviews), 2, figsize=(16, 18), sharex=True,
                               constrained_layout=True)
    for row_index, review in enumerate(candidate_reviews):
        identities = (
            (int(review["run"]), int(review["run_trial_index"])),
            (int(review["matched_run"]), int(review["matched_run_trial_index"])),
        )
        candidate_channels = str(review["driving_channels"]).split("/")[:2]
        channels = list(dict.fromkeys([*candidate_channels, "C3", "C4"]))[:4]
        arrays = [dataset.task_data_volts[index_lookup[identity]] * 1e6 for identity in identities]
        limit = max(float(np.percentile(np.abs(array[[dataset.channel_names.index(c) for c in channels]]), 99.5)) for array in arrays)
        limit = max(limit, 50.0)
        for column, (identity, array, title) in enumerate(
            zip(identities, arrays, ("candidate", "systematic matched trial"), strict=True)
        ):
            axis = axes[row_index, column]
            for channel in channels:
                axis.plot(dataset.task_times, array[dataset.channel_names.index(channel)],
                          linewidth=0.7, label=channel)
            axis.axvline(0, color="black", linestyle=":", linewidth=0.8)
            axis.axvspan(0, 4, color="#777777", alpha=0.05)
            axis.set_ylim(-limit, limit)
            axis.set_title(f"run {identity[0]} trial {identity[1]} — {title}")
            axis.set_ylabel("uV")
            axis.grid(alpha=0.15)
            if row_index == 0:
                axis.legend(frameon=False, ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("Time relative to task onset (s)")
    axes[-1, 1].set_xlabel("Time relative to task onset (s)")
    figure.suptitle("Six statistical candidates and deterministic matched ordinary trials", fontsize=14)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_qc_distributions(task_quality_rows, paired_rows, output_path: Path) -> None:
    metrics = (
        "max_task_p2p_uV_robust_z",
        "max_task_abs_step_uV_robust_z",
        "median_task_p2p_uV_robust_z",
        "median_task_channel_std_uV_robust_z",
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics, strict=True):
        values = np.array([float(row[metric]) for row in task_quality_rows])
        candidates = np.array(
            [row["quality_status"] == "statistical candidate" for row in task_quality_rows]
        )
        x = np.arange(values.size)
        axis.scatter(x[~candidates], values[~candidates], color="#4c78a8", s=28)
        axis.scatter(x[candidates], values[candidates], facecolors="none", edgecolors="#d62728", s=75)
        axis.axhline(3.5, color="#d62728", linestyle="--", linewidth=0.8)
        axis.axhline(-3.5, color="#d62728", linestyle="--", linewidth=0.8)
        axis.set_title(metric.replace("_robust_z", "").replace("_", " "))
        axis.set_xlabel("Master trial array index")
        axis.set_ylabel("Modified robust z")
        axis.grid(alpha=0.15)
    figure.suptitle(
        "Task QC distributions — red rings mark candidates; labels did not define status",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_normalization(diagnostic_rows, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    selected = [
        row for row in diagnostic_rows
        if row["channel"] in {"C3", "C4"}
        and row["frequency_band"] == "central_12_13"
    ]
    colors = {
        "both_fists_imagery": "#4c78a8",
        "both_feet_imagery": "#f58518",
    }
    for condition in CONDITIONS:
        rows = [row for row in selected if row["semantic_condition"] == condition]
        axes[0].scatter(
            [float(row["change_percent"]) for row in rows],
            [float(row["change_db"]) for row in rows],
            color=colors[condition], alpha=0.72, label=condition.replace("_", " "),
        )
        axes[1].scatter(
            [float(row["reference_log_power_robust_z"]) for row in rows],
            [abs(float(row["change_percent"])) for row in rows],
            color=colors[condition], alpha=0.72,
        )
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].axvline(0, color="#555555", linewidth=0.8)
    axes[0].set(xlabel="Percent change", ylabel="dB ratio", title="Same ratio, different scale")
    axes[0].legend(frameon=False)
    axes[1].axvline(-3.5, color="#d62728", linestyle="--", linewidth=0.8)
    axes[1].set(
        xlabel="Paired-rest log-power modified robust z",
        ylabel="Absolute percent change",
        title="Denominator diagnostic",
    )
    for axis in axes:
        axis.grid(alpha=0.15)
    figure.suptitle("C3/C4 12-13 Hz normalization behavior")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_stability_matrix(stability_rows, output_path: Path) -> None:
    criterion_fields = [field for field in stability_rows[0] if field.startswith("criterion_")]
    matrix = np.array(
        [[bool(row[field]) for field in criterion_fields] for row in stability_rows], dtype=float
    )
    labels = [
        f"{row['semantic_condition'].replace('both_', '').replace('_imagery', '')} "
        f"{row['channel']} {row['frequency_band']} ({row['stability_grade']})"
        for row in stability_rows
    ]
    figure, axis = plt.subplots(figsize=(14, 9), constrained_layout=True)
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xticks(
        np.arange(len(criterion_fields)),
        [field.replace("criterion_", "").replace("_", " ") for field in criterion_fields],
        rotation=35,
        ha="right",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, "pass" if matrix[row_index, column_index] else "fail",
                      ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, ticks=[0, 1], label="Predefined criterion")
    axis.set_title("Spectral feature stability under the predefined seven-criterion policy")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    preprocessing_config = load_preprocessing_config(arguments.preprocessing_config)
    epoching_config = load_epoching_config(arguments.epoching_config)
    spectral_config = load_event_related_spectral_config(arguments.spectral_config)
    quality_config = load_trial_quality_config(arguments.quality_config)
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    run_results = {
        run: extract_run_epochs(
            arguments.subject, run, preprocessing_config, epoching_config
        )
        for run in arguments.runs
    }
    dataset = assemble_spectral_epoch_dataset(run_results, arguments.runs)
    master_snapshot = dataset.task_data_volts.copy()
    task_records = [pair.task for pair in dataset.pairs]
    task_quality_rows = calculate_trial_quality(
        dataset.task_data_volts,
        dataset.task_times,
        dataset.channel_names,
        task_records,
        epoching_config.trial_quality,
    )
    quality_by_identity = {
        (int(row["run"]), int(row["run_trial_index"])): row
        for row in task_quality_rows
    }

    task_metrics = paired_side_quality_metrics(
        dataset.task_data_volts,
        dataset.task_times,
        dataset.channel_names,
        epoching_config.trial_quality.frontal_channels,
        (0.0, 4.0),
    )
    rest_metrics = paired_side_quality_metrics(
        dataset.rest_data_volts,
        dataset.rest_times,
        dataset.channel_names,
        epoching_config.trial_quality.frontal_channels,
        (0.0, 4.0),
    )
    task_metric_rows, task_metric_evidence, _ = classify_metric_candidates(
        task_metrics, quality_config.robust_z_threshold
    )
    rest_metric_rows, rest_metric_evidence, rest_candidates = classify_metric_candidates(
        rest_metrics, quality_config.robust_z_threshold
    )
    paired_rows = build_paired_rows(
        dataset,
        task_quality_rows,
        task_metric_rows,
        rest_metric_rows,
        task_metric_evidence,
        rest_metric_evidence,
        rest_candidates,
        quality_config,
    )

    task_power = compute_morlet_power(
        dataset.task_data_volts,
        dataset.task_times,
        dataset.sampling_frequency_hz,
        spectral_config,
    )
    rest_power = compute_morlet_power(
        dataset.rest_data_volts,
        dataset.rest_times,
        dataset.sampling_frequency_hz,
        spectral_config,
    )
    trial_rows = trial_band_measurements(
        task_power, rest_power, dataset, spectral_config, quality_by_identity
    )
    diagnostic_rows = add_normalization_diagnostics(
        trial_rows, quality_config.robust_z_threshold
    )
    for row in diagnostic_rows:
        row["source_file"] = display_path(Path(str(row["source_file"])))
    timing_rows = build_timing_rows(
        task_power, rest_power, dataset, spectral_config
    )
    stability_rows = build_feature_stability_rows(
        diagnostic_rows, timing_rows, quality_by_identity, quality_config
    )
    influence_rows = build_candidate_influence_rows(diagnostic_rows, quality_config)
    reference_rows = build_reference_influence_rows(diagnostic_rows, quality_config)

    match_metrics = (
        "max_task_p2p_uV",
        "median_task_p2p_uV",
        "frontal_max_task_p2p_uV",
        "max_task_abs_step_uV",
        "minimum_task_channel_std_uV",
        "median_task_channel_std_uV",
    )
    matches = systematic_matched_controls(task_quality_rows, match_metrics)
    candidate_reviews = build_candidate_review_rows(
        dataset,
        task_quality_rows,
        paired_rows,
        diagnostic_rows,
        matches,
    )
    evidence_rows = consolidated_evidence_rows(len(candidate_reviews))

    if not np.array_equal(master_snapshot, dataset.task_data_volts):
        raise RuntimeError("Sensitivity analysis mutated the master task epochs.")
    if Counter(pair.task.semantic_condition for pair in dataset.pairs) != {
        "both_fists_imagery": 21,
        "both_feet_imagery": 24,
    }:
        raise RuntimeError("Primary task balance changed unexpectedly.")
    if len(dataset.pairs) != 45 or not all(row["primary_retained"] for row in paired_rows):
        raise RuntimeError("Primary trial identity or retention changed unexpectedly.")

    prefix = artifact_prefix(arguments.subject, arguments.runs)
    paths = {
        "detailed": output_directory / f"{prefix}_trial_quality_detailed.csv",
        "paired": output_directory / f"{prefix}_paired_reference_quality.csv",
        "review": output_directory / f"{prefix}_candidate_trial_review.csv",
        "normalization": output_directory / f"{prefix}_normalization_sensitivity.csv",
        "stability": output_directory / f"{prefix}_feature_stability.csv",
        "influence": output_directory / f"{prefix}_candidate_influence.csv",
        "reference": output_directory / f"{prefix}_rest_pair_influence.csv",
        "evidence": output_directory / f"{prefix}_artifact_evidence.csv",
        "timing": output_directory / f"{prefix}_quality_timing_sensitivity.csv",
        "metadata": output_directory / f"{prefix}_trial_quality_metadata.json",
        "gallery_figure": output_directory / f"{prefix}_candidate_trial_gallery.png",
        "qc_figure": output_directory / f"{prefix}_quality_metric_distributions.png",
        "normalization_figure": output_directory / f"{prefix}_normalization_diagnostics.png",
        "stability_figure": output_directory / f"{prefix}_feature_stability_matrix.png",
    }
    detailed_rows = []
    paired_lookup = {
        (int(row["run"]), int(row["run_trial_index"])): row for row in paired_rows
    }
    for row in task_quality_rows:
        identity = (int(row["run"]), int(row["run_trial_index"]))
        detailed_rows.append({**row, **{f"pair_{key}": value for key, value in paired_lookup[identity].items() if key not in row}})
    for key, rows in (
        ("detailed", detailed_rows),
        ("paired", paired_rows),
        ("review", candidate_reviews),
        ("normalization", diagnostic_rows),
        ("stability", stability_rows),
        ("influence", influence_rows),
        ("reference", reference_rows),
        ("evidence", evidence_rows),
        ("timing", timing_rows),
    ):
        write_csv(rows, paths[key])

    plot_candidate_gallery(dataset, candidate_reviews, paths["gallery_figure"])
    plot_qc_distributions(task_quality_rows, paired_rows, paths["qc_figure"])
    plot_normalization(diagnostic_rows, paths["normalization_figure"])
    plot_stability_matrix(stability_rows, paths["stability_figure"])

    metadata = {
        "schema_version": 1,
        "subject": arguments.subject,
        "runs": list(arguments.runs),
        "master_task_shape": list(dataset.task_data_volts.shape),
        "master_rest_shape": list(dataset.rest_data_volts.shape),
        "task_tfr_shape": list(task_power.power_v2.shape),
        "rest_tfr_shape": list(rest_power.power_v2.shape),
        "condition_counts": dict(Counter(pair.task.semantic_condition for pair in dataset.pairs)),
        "task_candidate_identities": sorted(
            [list(identity) for identity, row in quality_by_identity.items() if row["quality_status"] == "statistical candidate"]
        ),
        "strong_review_candidate_identities": sorted(
            [list(identity) for identity, row in quality_by_identity.items() if is_strong_review_candidate(row, quality_config)]
        ),
        "paired_rest_candidate_count": int(rest_candidates.sum()),
        "primary_trial_count": len(dataset.pairs),
        "permanent_trial_rejection_count": 0,
        "quality_decisions_used_condition_labels": False,
        "ica_policy": quality_config.ica_policy,
        "source_sha256_by_run": {
            str(run): result.preprocessing.source_sha256 for run, result in run_results.items()
        },
        "config_paths": {
            "preprocessing": display_path(arguments.preprocessing_config),
            "epoching": display_path(arguments.epoching_config),
            "spectral": display_path(arguments.spectral_config),
            "trial_quality": display_path(arguments.quality_config),
        },
        "artifacts": {key: display_path(path) for key, path in paths.items() if key != "metadata"},
    }
    with paths["metadata"].open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    print("Artifact policy and spectral feature-stability report complete.")
    print(f"Master trials retained: {len(dataset.pairs)}/45; permanent rejections: 0")
    print(f"Task QC candidates: {len(candidate_reviews)}; paired-rest candidates: {int(rest_candidates.sum())}")
    for row in stability_rows:
        print(
            f"  {row['semantic_condition']} {row['channel']} {row['frequency_band']}: "
            f"{float(row['median_change_percent']):.2f}% | "
            f"{row['criteria_passed']}/7 {row['stability_grade']}"
        )
    print(f"Metadata: {display_path(paths['metadata'])}")


if __name__ == "__main__":
    main()
