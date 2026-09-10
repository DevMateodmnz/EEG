"""Reusable subject-at-a-time execution for EEG replication cohorts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .cohort import (
    central_peak_summary,
    summarize_subject_features,
    summarize_subject_runs,
)
from .epoching import calculate_trial_quality, extract_run_epochs
from .time_frequency import (
    SpectralEpochDataset,
    assemble_spectral_epoch_dataset,
    compute_morlet_power,
    trial_band_measurements,
)
from .trial_quality import (
    add_normalization_diagnostics,
    classify_metric_candidates,
    paired_side_quality_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def display_path(path: Path) -> str:
    """Use a repository-relative path when possible."""
    try:
        return str(path.expanduser().resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.expanduser().resolve())


def subject_qc_summary(
    subject: int,
    role: str,
    dataset,
    run_results,
    task_quality_rows,
    rest_candidate_count: int,
    audit_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Create one compact, label-independent QC row for a processed subject."""
    condition_counts = Counter(pair.task.semantic_condition for pair in dataset.pairs)
    channel_names = sorted(
        {
            channel
            for row in audit_rows
            if int(row["subject"]) == subject
            for channel in str(row["channel_candidate_names"]).split("/")
            if channel
        }
    )
    subject_audit = [row for row in audit_rows if int(row["subject"]) == subject]
    return {
        "subject": subject,
        "subject_role": role,
        "runs_available": "/".join(str(run) for run in sorted(run_results)),
        "run_count": len(run_results),
        "task_trials": len(dataset.pairs),
        "rest_pairs": len(dataset.pairs),
        "fists_trials": condition_counts["both_fists_imagery"],
        "feet_trials": condition_counts["both_feet_imagery"],
        "qc_task_candidates": sum(
            row["quality_status"] == "statistical candidate"
            for row in task_quality_rows
        ),
        "qc_rest_candidates": rest_candidate_count,
        "channel_candidate_run_records": sum(
            int(row["channel_candidate_count"]) for row in subject_audit
        ),
        "unique_channel_candidate_count": len(channel_names),
        "unique_channel_candidate_names": "/".join(channel_names),
        "confirmed_channel_failures": sum(
            int(row["confirmed_flat_or_nonfinite_channel_count"])
            for row in subject_audit
        ),
        "invalid_tail_runs": sum(
            int(row["trailing_zero_samples"]) > 0 for row in subject_audit
        ),
        "trailing_zero_samples_total": sum(
            int(row["trailing_zero_samples"]) for row in subject_audit
        ),
        "deterministic_invalid_task_epochs": sum(
            not record.valid
            for result in run_results.values()
            for record in result.task_records
        ),
        "deterministic_invalid_rest_epochs": sum(
            not record.valid
            for result in run_results.values()
            for record in result.rest_records
        ),
        "permanent_artifact_trial_exclusions": 0,
        "processing_status": "processed_frozen_pipeline",
        "outcome_used_for_inclusion": False,
    }


def process_subject(
    subject: int,
    role: str,
    cohort_config,
    preprocessing_config,
    epoching_config,
    spectral_config,
    quality_config,
    data_directory: Path,
    audit_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Run the frozen pipeline for one eligible subject and return compact rows."""
    run_results = {
        run: extract_run_epochs(
            subject,
            run,
            preprocessing_config,
            epoching_config,
            data_directory=data_directory,
        )
        for run in cohort_config.runs
    }
    dataset = assemble_spectral_epoch_dataset(run_results, cohort_config.runs)
    master_snapshot = dataset.task_data_volts.copy()
    task_quality_rows = calculate_trial_quality(
        dataset.task_data_volts,
        dataset.task_times,
        dataset.channel_names,
        [pair.task for pair in dataset.pairs],
        epoching_config.trial_quality,
    )
    quality_by_identity = {
        (int(row["run"]), int(row["run_trial_index"])): row
        for row in task_quality_rows
    }

    sensor_indices = [
        dataset.channel_names.index(channel) for channel in ("C3", "Cz", "C4")
    ]
    sensor_dataset = SpectralEpochDataset(
        task_data_volts=dataset.task_data_volts[:, sensor_indices].copy(),
        rest_data_volts=dataset.rest_data_volts[:, sensor_indices].copy(),
        task_times=dataset.task_times.copy(),
        rest_times=dataset.rest_times.copy(),
        channel_names=["C3", "Cz", "C4"],
        sampling_frequency_hz=dataset.sampling_frequency_hz,
        pairs=dataset.pairs.copy(),
    )
    task_power = compute_morlet_power(
        sensor_dataset.task_data_volts,
        sensor_dataset.task_times,
        sensor_dataset.sampling_frequency_hz,
        spectral_config,
    )
    rest_power = compute_morlet_power(
        sensor_dataset.rest_data_volts,
        sensor_dataset.rest_times,
        sensor_dataset.sampling_frequency_hz,
        spectral_config,
    )
    trial_rows = trial_band_measurements(
        task_power,
        rest_power,
        sensor_dataset,
        spectral_config,
        quality_by_identity,
    )
    diagnostic_rows = add_normalization_diagnostics(
        trial_rows, quality_config.robust_z_threshold
    )
    for row in diagnostic_rows:
        row["subject_role"] = role
        row["source_file"] = display_path(Path(str(row["source_file"])))

    rest_metrics = paired_side_quality_metrics(
        dataset.rest_data_volts,
        dataset.rest_times,
        dataset.channel_names,
        epoching_config.trial_quality.frontal_channels,
        (0.0, 4.0),
    )
    _, _, rest_candidates = classify_metric_candidates(
        rest_metrics, quality_config.robust_z_threshold
    )
    feature_rows = summarize_subject_features(
        diagnostic_rows,
        subject,
        role,
        cohort_config.all_features,
        quality_config.trimmed_mean_fraction_each_tail,
    )
    run_rows = summarize_subject_runs(
        diagnostic_rows, subject, role, cohort_config.all_features
    )
    peak_row = central_peak_summary(
        subject,
        role,
        [
            (
                run,
                result.preprocessing.referenced.get_data(),
                result.preprocessing.referenced.ch_names,
                float(result.preprocessing.referenced.info["sfreq"]),
            )
            for run, result in sorted(run_results.items())
        ],
        cohort_config.central_peak_exploration,
    )
    qc_row = subject_qc_summary(
        subject,
        role,
        dataset,
        run_results,
        task_quality_rows,
        int(rest_candidates.sum()),
        audit_rows,
    )
    if not np.array_equal(dataset.task_data_volts, master_snapshot):
        raise RuntimeError(f"Subject {subject}: sensitivity analysis mutated master epochs.")
    if not all(row["primary_retained"] for row in diagnostic_rows):
        raise RuntimeError(f"Subject {subject}: a primary spectral trial was removed.")
    return {
        "trial_rows": diagnostic_rows,
        "feature_rows": feature_rows,
        "run_rows": run_rows,
        "peak_row": peak_row,
        "qc_row": qc_row,
        "task_tfr_shape": list(task_power.power_v2.shape),
        "rest_tfr_shape": list(rest_power.power_v2.shape),
    }
