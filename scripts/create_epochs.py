"""Create, audit, visualize, and report motor-imagery trial epochs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

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
    EpochRecord,
    RunEpochs,
    calculate_trial_quality,
    extract_run_epochs,
    load_epoching_config,
)
from eeg_project.preprocessing import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_preprocessing_config,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_RUNS = (6, 10, 14)
DISPLAY_CHANNELS = ("Fp1", "C3", "Cz", "C4")
CONDITION_COLORS = {
    "both_fists_imagery": "#4c78a8",
    "both_feet_imagery": "#f58518",
    "rest": "#b8b8b8",
}


def parse_arguments() -> argparse.Namespace:
    """Parse the reproducible epoch-report CLI."""
    parser = argparse.ArgumentParser(
        description="Create traceable task and rest epochs without rejecting candidates."
    )
    parser.add_argument("--subject", type=int, default=1, help="EEGBCI subject (1-109).")
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        choices=DEFAULT_RUNS,
        default=list(DEFAULT_RUNS),
        help="Supported bilateral motor-imagery runs.",
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Validated continuous-preprocessing JSON.",
    )
    parser.add_argument(
        "--epoching-config",
        type=Path,
        default=DEFAULT_EPOCHING_CONFIG_PATH,
        help="Event/epoch policy JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for report CSV, JSON, and PNG artifacts.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject invalid subject or duplicate-run requests."""
    if not 1 <= arguments.subject <= 109:
        raise ValueError("Subject must be between 1 and 109.")
    if len(set(arguments.runs)) != len(arguments.runs):
        raise ValueError("Run numbers must not be duplicated.")


def write_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    """Write deterministic rows using the ordered union of columns."""
    if not rows:
        raise ValueError(f"Cannot write an empty table: {output_path}.")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> Path:
    """Prefer a project-relative path for console output."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def artifact_prefix(subject: int, runs: Sequence[int]) -> str:
    """Create a stable artifact prefix."""
    return f"subject{subject:02d}_runs" + "-".join(f"{run:02d}" for run in runs)


def choose_representative_indices(
    quality_rows: Sequence[dict[str, object]],
) -> dict[str, int]:
    """Select the non-flagged trial nearest each condition's median amplitude."""
    result: dict[str, int] = {}
    for condition in ("both_fists_imagery", "both_feet_imagery"):
        indices = [
            index
            for index, row in enumerate(quality_rows)
            if row["semantic_condition"] == condition
            and row["quality_status"] == "not flagged"
        ]
        if not indices:
            raise RuntimeError(f"No non-flagged representative exists for {condition}.")
        values = np.array(
            [float(quality_rows[index]["max_task_p2p_uV"]) for index in indices]
        )
        median = float(np.median(values))
        result[condition] = indices[int(np.argmin(np.abs(values - median)))]
    return result


def plot_annotation_structure(
    runs: Sequence[int],
    annotation_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    """Plot the complete measured annotation sequence for every run."""
    figure, axes = plt.subplots(
        len(runs), 1, figsize=(15, 2.6 * len(runs)), sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, run in zip(axes_array, runs, strict=True):
        rows = [row for row in annotation_rows if int(row["run"]) == run]
        for row in rows:
            onset = float(row["onset_seconds"])
            duration = float(row["duration_seconds"])
            semantic = str(row["semantic_condition"])
            axis.broken_barh(
                [(onset, duration)],
                (0, 1),
                facecolors=CONDITION_COLORS[semantic],
                edgecolors="white",
                linewidth=0.4,
            )
        axis.set_ylim(0, 1)
        axis.set_yticks([0.5], [f"Run {run}"])
        axis.grid(axis="x", alpha=0.2)
    axes_array[-1].set_xlim(0, 124.5)
    axes_array[-1].set_xlabel("Absolute time from recording start (s)")
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=8, label=label)
        for label, color in (
            ("T0 rest", CONDITION_COLORS["rest"]),
            ("T1 both-fists imagery", CONDITION_COLORS["both_fists_imagery"]),
            ("T2 both-feet imagery", CONDITION_COLORS["both_feet_imagery"]),
        )
    ]
    axes_array[0].legend(handles=handles, frameon=False, ncol=3, loc="upper center")
    figure.suptitle(
        "Measured annotations — contiguous T0/task alternation; zero tail begins at 124.5 s",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_representative_epochs(
    task_data_volts: np.ndarray,
    times: np.ndarray,
    channel_names: Sequence[str],
    quality_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    """Show deterministic representative T1/T2 trials in event-relative time."""
    selected = choose_representative_indices(quality_rows)
    conditions = ("both_fists_imagery", "both_feet_imagery")
    figure, axes = plt.subplots(
        len(DISPLAY_CHANNELS), 2, figsize=(15, 11), sharex=True,
        constrained_layout=True,
    )
    for row_index, channel in enumerate(DISPLAY_CHANNELS):
        channel_index = channel_names.index(channel)
        values_by_condition = [
            task_data_volts[selected[condition], channel_index] * 1_000_000
            for condition in conditions
        ]
        lower = min(float(np.min(values)) for values in values_by_condition)
        upper = max(float(np.max(values)) for values in values_by_condition)
        margin = max(1.0, 0.05 * (upper - lower))
        for column, (condition, values) in enumerate(
            zip(conditions, values_by_condition, strict=True)
        ):
            axis = axes[row_index, column]
            axis.axvspan(-2, 0, color=CONDITION_COLORS["rest"], alpha=0.12)
            axis.axvspan(0, 4, color=CONDITION_COLORS[condition], alpha=0.08)
            axis.axvline(0, color="#222222", linestyle=":")
            axis.plot(times, values, color=CONDITION_COLORS[condition], linewidth=0.9)
            axis.set_ylim(lower - margin, upper + margin)
            axis.grid(alpha=0.18)
            if row_index == 0:
                qrow = quality_rows[selected[condition]]
                axis.set_title(
                    f"{condition.replace('_', ' ')}\n"
                    f"run {qrow['run']}, event {float(qrow['event_time_seconds']):g} s"
                )
            if column == 0:
                axis.set_ylabel(f"{channel}\nµV")
            if row_index == len(DISPLAY_CHANNELS) - 1:
                axis.set_xlabel("Time relative to task onset (s)")
    figure.suptitle(
        "Representative non-flagged task epochs — context is retained; baseline correction is off",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trial_heatmap(
    task_data_volts: np.ndarray,
    times: np.ndarray,
    channel_names: Sequence[str],
    quality_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    """Show C3 trial variability without averaging away non-phase-locked activity."""
    c3_index = channel_names.index("C3")
    data_uv = task_data_volts[:, c3_index] * 1_000_000
    limit = float(np.percentile(np.abs(data_uv), 99))
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, constrained_layout=True)
    last_image = None
    for axis, condition in zip(
        axes, ("both_fists_imagery", "both_feet_imagery"), strict=True
    ):
        indices = [
            index
            for index, row in enumerate(quality_rows)
            if row["semantic_condition"] == condition
        ]
        values = data_uv[indices]
        last_image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], -0.5, len(indices) - 0.5],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.axvline(0, color="#222222", linestyle=":")
        axis.set_title(f"{condition.replace('_', ' ')} ({len(indices)} trials)")
        axis.set_xlabel("Time relative to task onset (s)")
        axis.set_ylabel("Trial index within condition")
    assert last_image is not None
    figure.colorbar(last_image, ax=axes, label="C3 voltage (µV)", shrink=0.85)
    figure.suptitle(
        "C3 trial variability — individual trials, not an ERP or condition effect",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trial_quality(
    quality_rows: Sequence[dict[str, object]], output_path: Path
) -> None:
    """Plot a trial-level amplitude metric and exploratory candidate identities."""
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    indices = np.arange(len(quality_rows))
    for condition in ("both_fists_imagery", "both_feet_imagery"):
        condition_indices = [
            index
            for index, row in enumerate(quality_rows)
            if row["semantic_condition"] == condition
        ]
        axes[0].scatter(
            condition_indices,
            [float(quality_rows[index]["max_task_p2p_uV"]) for index in condition_indices],
            color=CONDITION_COLORS[condition],
            label=condition.replace("_", " "),
            alpha=0.8,
        )
    candidate_indices = [
        index
        for index, row in enumerate(quality_rows)
        if row["quality_status"] == "statistical candidate"
    ]
    axes[0].scatter(
        candidate_indices,
        [float(quality_rows[index]["max_task_p2p_uV"]) for index in candidate_indices],
        facecolors="none",
        edgecolors="#d62728",
        s=100,
        linewidths=1.5,
        label="statistical candidate",
    )
    for boundary in (14.5, 29.5):
        axes[0].axvline(boundary, color="#777777", linestyle=":")
    axes[0].set_ylabel("Maximum channel task-period peak-to-peak (µV)")
    axes[0].set_xlabel("Combined task-epoch array index (runs 6 → 10 → 14)")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.2)

    score_fields = (
        "max_task_p2p_uV_robust_z",
        "median_task_p2p_uV_robust_z",
        "frontal_max_task_p2p_uV_robust_z",
        "max_task_abs_step_uV_robust_z",
    )
    score_matrix = np.array(
        [[float(row[field]) for row in quality_rows] for field in score_fields]
    )
    image = axes[1].imshow(
        score_matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-7,
        vmax=7,
        interpolation="nearest",
    )
    axes[1].set_yticks(
        np.arange(len(score_fields)),
        [field.replace("_uV_robust_z", "").replace("_", " ") for field in score_fields],
    )
    axes[1].set_xlabel("Combined task-epoch array index")
    axes[1].set_title("Selected modified robust z-scores; ±3.5 is exploratory")
    figure.colorbar(image, ax=axes[1], label="Modified robust z-score")
    figure.suptitle(
        "Trial-quality review — candidates are preserved, not automatically rejected",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_candidate_comparison(
    task_data_volts: np.ndarray,
    times: np.ndarray,
    channel_names: Sequence[str],
    quality_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    """Compare the strongest deterministic candidate with a typical same-class trial."""
    candidate_indices = [
        index
        for index, row in enumerate(quality_rows)
        if row["quality_status"] == "statistical candidate"
    ]
    if not candidate_indices:
        raise RuntimeError("Candidate comparison requested but no candidates exist.")
    robust_fields = [field for field in quality_rows[0] if field.endswith("_robust_z")]
    candidate_index = max(
        candidate_indices,
        key=lambda index: max(abs(float(quality_rows[index][field])) for field in robust_fields),
    )
    condition = str(quality_rows[candidate_index]["semantic_condition"])
    typical_pool = [
        index
        for index, row in enumerate(quality_rows)
        if row["semantic_condition"] == condition and row["quality_status"] == "not flagged"
    ]
    typical_values = np.array(
        [float(quality_rows[index]["max_task_p2p_uV"]) for index in typical_pool]
    )
    typical_index = typical_pool[
        int(np.argmin(np.abs(typical_values - np.median(typical_values))))
    ]
    selected = (candidate_index, typical_index)
    labels = ("Strongest statistical candidate", "Typical same-condition trial")

    figure, axes = plt.subplots(
        len(DISPLAY_CHANNELS), 2, figsize=(15, 11), sharex=True,
        constrained_layout=True,
    )
    for row_index, channel in enumerate(DISPLAY_CHANNELS):
        channel_index = channel_names.index(channel)
        values_list = [task_data_volts[index, channel_index] * 1_000_000 for index in selected]
        lower = min(float(np.min(values)) for values in values_list)
        upper = max(float(np.max(values)) for values in values_list)
        margin = max(1.0, 0.05 * (upper - lower))
        for column, (index, label, values) in enumerate(
            zip(selected, labels, values_list, strict=True)
        ):
            axis = axes[row_index, column]
            axis.axvspan(-2, 0, color=CONDITION_COLORS["rest"], alpha=0.12)
            axis.axvline(0, color="#222222", linestyle=":")
            axis.plot(times, values, color=CONDITION_COLORS[condition], linewidth=0.9)
            axis.set_ylim(lower - margin, upper + margin)
            axis.grid(alpha=0.18)
            if row_index == 0:
                row = quality_rows[index]
                axis.set_title(
                    f"{label}\nrun {row['run']}, {row['annotation']} at "
                    f"{float(row['event_time_seconds']):g} s"
                )
            if column == 0:
                axis.set_ylabel(f"{channel}\nµV")
            if row_index == len(DISPLAY_CHANNELS) - 1:
                axis.set_xlabel("Time relative to task onset (s)")
    figure.suptitle(
        f"Exploratory quality comparison — {condition.replace('_', ' ')}; neither trial is rejected",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Execute event conversion, epoching, quality review, and reporting."""
    arguments = parse_arguments()
    validate_arguments(arguments)
    preprocessing_path = arguments.preprocessing_config.expanduser().resolve()
    epoching_path = arguments.epoching_config.expanduser().resolve()
    preprocessing_config = load_preprocessing_config(preprocessing_path)
    epoching_config = load_epoching_config(epoching_path)
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    run_results: dict[int, RunEpochs] = {}
    annotation_rows: list[dict[str, object]] = []
    all_task_records: list[EpochRecord] = []
    all_rest_records: list[EpochRecord] = []
    task_arrays: list[np.ndarray] = []
    rest_arrays: list[np.ndarray] = []
    for run in arguments.runs:
        result = extract_run_epochs(
            arguments.subject,
            run,
            preprocessing_config,
            epoching_config,
        )
        run_results[run] = result
        for row in result.annotation_rows:
            annotation_rows.append(
                {
                    "subject": arguments.subject,
                    **row,
                    "source_file": str(display_path(result.preprocessing.source_path)),
                    "source_sha256": result.preprocessing.source_sha256,
                }
            )
        valid_task_records = [record for record in result.task_records if record.valid]
        valid_rest_records = [record for record in result.rest_records if record.valid]
        all_task_records.extend(valid_task_records)
        all_rest_records.extend(valid_rest_records)
        task_arrays.append(result.task_epochs.get_data(copy=True))
        rest_arrays.append(result.rest_epochs.get_data(copy=True))

    task_data = np.concatenate(task_arrays, axis=0)
    rest_data = np.concatenate(rest_arrays, axis=0)
    first_result = run_results[arguments.runs[0]]
    task_times = first_result.task_epochs.times.copy()
    rest_times = first_result.rest_epochs.times.copy()
    channel_names = list(first_result.task_epochs.ch_names)
    if not np.isfinite(task_data).all() or not np.isfinite(rest_data).all():
        raise RuntimeError("Extracted epoch arrays contain non-finite values.")

    quality_rows = calculate_trial_quality(
        task_data,
        task_times,
        channel_names,
        all_task_records,
        epoching_config.trial_quality,
    )
    for index, row in enumerate(quality_rows):
        row["epoch_array_index"] = index
        row["source_file"] = str(
            display_path(run_results[int(row["run"])].preprocessing.source_path)
        )
        row["source_sha256"] = run_results[int(row["run"])].preprocessing.source_sha256

    quality_by_identity = {
        (int(row["run"]), int(row["run_trial_index"])): row for row in quality_rows
    }
    task_rows: list[dict[str, object]] = []
    array_index = 0
    record_index = 0
    for run in arguments.runs:
        for record in run_results[run].task_records:
            quality = quality_by_identity.get((run, record.run_trial_index))
            task_rows.append(
                {
                    "trial_record_index": record_index,
                    "epoch_array_index": array_index if record.valid else "",
                    **record.to_dict(),
                    "source_file": str(display_path(run_results[run].preprocessing.source_path)),
                    "source_sha256": run_results[run].preprocessing.source_sha256,
                    "quality_status": quality["quality_status"] if quality else "not assessed",
                    "quality_evidence": quality["quality_evidence"] if quality else "",
                    "confirmed_exclusion": False,
                }
            )
            record_index += 1
            if record.valid:
                array_index += 1

    rest_rows: list[dict[str, object]] = []
    rest_array_index = 0
    for run in arguments.runs:
        for record in run_results[run].rest_records:
            rest_rows.append(
                {
                    "rest_record_index": len(rest_rows),
                    "rest_epoch_array_index": rest_array_index if record.valid else "",
                    **record.to_dict(),
                    "source_file": str(display_path(run_results[run].preprocessing.source_path)),
                    "source_sha256": run_results[run].preprocessing.source_sha256,
                    "role": "separate experimental rest/context; not a primary task class",
                }
            )
            if record.valid:
                rest_array_index += 1

    prefix = artifact_prefix(arguments.subject, arguments.runs)
    annotation_path = output_directory / f"{prefix}_annotation_structure.csv"
    task_path = output_directory / f"{prefix}_task_trials.csv"
    rest_path = output_directory / f"{prefix}_rest_context_epochs.csv"
    quality_path = output_directory / f"{prefix}_trial_quality.csv"
    metadata_path = output_directory / f"{prefix}_epoching_metadata.json"
    annotation_figure_path = output_directory / f"{prefix}_annotation_structure.png"
    representative_path = output_directory / f"{prefix}_representative_task_epochs.png"
    heatmap_path = output_directory / f"{prefix}_c3_trial_heatmap.png"
    quality_figure_path = output_directory / f"{prefix}_trial_quality.png"
    candidate_path = output_directory / f"{prefix}_candidate_epoch_comparison.png"

    write_csv(annotation_rows, annotation_path)
    write_csv(task_rows, task_path)
    write_csv(rest_rows, rest_path)
    write_csv(quality_rows, quality_path)
    plot_annotation_structure(arguments.runs, annotation_rows, annotation_figure_path)
    plot_representative_epochs(
        task_data, task_times, channel_names, quality_rows, representative_path
    )
    plot_trial_heatmap(task_data, task_times, channel_names, quality_rows, heatmap_path)
    plot_trial_quality(quality_rows, quality_figure_path)
    plot_candidate_comparison(
        task_data, task_times, channel_names, quality_rows, candidate_path
    )

    generated_paths = [
        annotation_path,
        task_path,
        rest_path,
        quality_path,
        metadata_path,
        annotation_figure_path,
        representative_path,
        heatmap_path,
        quality_figure_path,
        candidate_path,
    ]
    task_counts = Counter(record.semantic_condition for record in all_task_records)
    candidate_counts = Counter(
        int(row["run"])
        for row in quality_rows
        if row["quality_status"] == "statistical candidate"
    )
    run_summaries = []
    for run in arguments.runs:
        result = run_results[run]
        run_task_records = [record for record in result.task_records if record.valid]
        annotations = Counter(str(value) for value in result.preprocessing.filtered.annotations.description)
        run_summaries.append(
            {
                "run": run,
                "annotation_counts": dict(sorted(annotations.items())),
                "task_condition_counts": dict(
                    sorted(Counter(record.semantic_condition for record in run_task_records).items())
                ),
                "valid_task_epochs": len(run_task_records),
                "invalid_task_records": sum(not record.valid for record in result.task_records),
                "valid_rest_epochs": sum(record.valid for record in result.rest_records),
                "statistical_candidate_task_epochs": candidate_counts[run],
                "confirmed_task_exclusions": 0,
                "source_file": str(display_path(result.preprocessing.source_path)),
                "source_sha256": result.preprocessing.source_sha256,
            }
        )

    metadata = {
        "schema_version": 1,
        "subject": arguments.subject,
        "runs": list(arguments.runs),
        "preprocessing_config_path": str(display_path(preprocessing_path)),
        "epoching_config_path": str(display_path(epoching_path)),
        "protocol": {
            "run_family": "bilateral both-fists/both-feet motor imagery",
            "T0": "rest",
            "T1": "both_fists_imagery",
            "T2": "both_feet_imagery",
            "mapping_scope": "runs 6, 10, and 14 only",
            "source": "PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0",
        },
        "task_epoch": {
            "tmin_seconds": float(task_times[0]),
            "tmax_seconds": float(task_times[-1]),
            "n_samples": int(task_times.size),
            "baseline": None,
            "shape": list(task_data.shape),
            "condition_counts": dict(sorted(task_counts.items())),
            "invalid_boundary_records": sum(not row["valid"] for row in task_rows),
            "confirmed_exclusions": 0,
        },
        "rest_context_epoch": {
            "tmin_seconds": float(rest_times[0]),
            "tmax_seconds": float(rest_times[-1]),
            "n_samples": int(rest_times.size),
            "baseline": None,
            "shape": list(rest_data.shape),
            "role": "separate context/rest collection, not a third primary task class",
        },
        "sampling_frequency_hz": float(first_result.task_epochs.info["sfreq"]),
        "channel_count": len(channel_names),
        "channel_names": channel_names,
        "filter_half_support_seconds": epoching_config.filter_half_support_seconds,
        "provisional_later_task_analysis_interval_seconds": list(
            epoching_config.provisional_task_analysis_interval_seconds
        ),
        "baseline_decision": "No time-domain baseline correction; preserve preceding rest context for later spectral decisions.",
        "quality": {
            "modified_robust_z_threshold": (
                epoching_config.trial_quality.modified_robust_z_threshold
            ),
            "statistical_candidates": sum(
                row["quality_status"] == "statistical candidate" for row in quality_rows
            ),
            "confirmed_exclusions": 0,
            "channel_specific_flags_are_supporting_only": True,
        },
        "run_summaries": run_summaries,
        "persistence_policy": (
            "Regenerate Epochs deterministically in memory; persist only report, "
            "quality, and provenance artifacts."
        ),
        "software": {
            "python": sys.version.split()[0],
            "mne": mne.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "report_artifacts": [path.name for path in generated_paths],
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    print("Event extraction, epoching, and trial-quality assessment")
    print(f"Subject: {arguments.subject}; runs: {arguments.runs}")
    print("Verified semantics: T0=rest; T1=both-fists imagery; T2=both-feet imagery")
    print("Mapping scope: bilateral motor-imagery runs 6/10/14 only")
    print(
        f"Task X shape: {task_data.shape}; event-relative time "
        f"{task_times[0]:g} to {task_times[-1]:g} s; baseline=None"
    )
    print(
        f"Rest/context shape: {rest_data.shape}; event-relative time "
        f"{rest_times[0]:g} to {rest_times[-1]:g} s; not a task class"
    )
    print(f"Task counts: {dict(sorted(task_counts.items()))}")
    print(
        "Statistical candidate task epochs: "
        f"{sum(row['quality_status'] == 'statistical candidate' for row in quality_rows)}; "
        "confirmed exclusions: 0"
    )
    for run_summary in run_summaries:
        print(
            f"  Run {run_summary['run']}: annotations={run_summary['annotation_counts']}; "
            f"tasks={run_summary['task_condition_counts']}; "
            f"boundary exclusions={run_summary['invalid_task_records']}; "
            f"candidates={run_summary['statistical_candidate_task_epochs']}"
        )
    print("No epoch FIF was saved; arrays are regenerated from raw data and configuration")
    print("\nGenerated report artifacts:")
    for path in generated_paths:
        print(f"  {display_path(path)}")


if __name__ == "__main__":
    main()
