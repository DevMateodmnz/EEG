"""Held-out individualized-frequency evaluation after the final method freeze."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
from mne.time_frequency import morlet, tfr_array_morlet
from scipy.stats import wilcoxon

from .cohort import holm_adjust
from .provenance import canonical_source_id
from .time_frequency import (
    EventRelatedSpectralConfig,
    SpectralEpochDataset,
    TimeFrequencyPower,
    interval_mask,
    percent_power_change,
)
from .trial_quality import trimmed_mean


def compute_custom_morlet_power(
    data_volts: np.ndarray,
    times_seconds: np.ndarray,
    sampling_frequency_hz: float,
    frequencies_hz: np.ndarray,
    config: EventRelatedSpectralConfig,
) -> TimeFrequencyPower:
    """Compute frozen-wavelet power at a pre-estimated individualized grid."""
    data = np.asarray(data_volts, dtype=float)
    times = np.asarray(times_seconds, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if data.ndim != 3 or data.shape[2] != times.size:
        raise ValueError("Expected data shaped (trials, channels, epoch times).")
    if frequencies.ndim != 1 or frequencies.size != 5:
        raise ValueError("The frozen individualized band requires five frequency centers.")
    if not np.isfinite(data).all() or not np.isfinite(frequencies).all():
        raise ValueError("Morlet data and frequency centers must be finite.")
    if not np.allclose(np.diff(frequencies), 0.5):
        raise ValueError("Individualized Morlet centers must be spaced by 0.5 Hz.")
    measured_sfreq = 1.0 / float(np.median(np.diff(times)))
    if not np.isclose(measured_sfreq, sampling_frequency_hz):
        raise ValueError("Time vector and sampling frequency are inconsistent.")
    cycles = frequencies * config.wavelet.cycles_per_hz
    wavelets = morlet(
        sampling_frequency_hz,
        frequencies,
        n_cycles=cycles,
        zero_mean=config.wavelet.zero_mean,
    )
    lengths = np.array([len(wavelet) for wavelet in wavelets], dtype=int)
    if lengths.max() > data.shape[2]:
        raise ValueError("An individualized wavelet is longer than its epoch.")
    power = tfr_array_morlet(
        data,
        sampling_frequency_hz,
        frequencies,
        n_cycles=cycles,
        zero_mean=config.wavelet.zero_mean,
        use_fft=config.wavelet.use_fft,
        decim=config.wavelet.decimation,
        output="power",
        n_jobs=1,
        verbose="error",
    )
    output_times = times[:: config.wavelet.decimation]
    expected = (data.shape[0], data.shape[1], 5, output_times.size)
    if power.shape != expected or not np.isfinite(power).all() or np.any(power < 0):
        raise RuntimeError(f"Unexpected individualized Morlet output: {power.shape}.")
    return TimeFrequencyPower(power, frequencies, output_times, cycles, lengths)


def individualized_trial_measurements(
    task_power: TimeFrequencyPower,
    rest_power: TimeFrequencyPower,
    dataset: SpectralEpochDataset,
    config: EventRelatedSpectralConfig,
    peak_frequency_hz: float,
) -> list[dict[str, object]]:
    """Return trial-level individualized percent and dB measurements."""
    if task_power.power_v2.shape[:3] != rest_power.power_v2.shape[:3]:
        raise ValueError("Individualized task/rest TFR dimensions differ.")
    if not np.array_equal(task_power.frequencies_hz, rest_power.frequencies_hz):
        raise ValueError("Individualized task/rest frequency grids differ.")
    if task_power.power_v2.shape[0] != len(dataset.pairs):
        raise ValueError("Individualized TFR trials and provenance pairs differ.")
    task_mask = interval_mask(
        task_power.times_seconds, config.task_analysis_interval_seconds
    )
    rest_mask = interval_mask(
        rest_power.times_seconds, config.paired_rest_reference_interval_seconds
    )
    task_values = np.mean(task_power.power_v2[..., task_mask], axis=(2, 3))
    rest_values = np.mean(rest_power.power_v2[..., rest_mask], axis=(2, 3))
    changes = percent_power_change(task_values, rest_values)
    db_changes = 10.0 * np.log10(task_values / rest_values)
    rows: list[dict[str, object]] = []
    for trial_index, pair in enumerate(dataset.pairs):
        for channel_index, channel in enumerate(dataset.channel_names):
            rows.append(
                {
                    "subject": pair.task.subject,
                    "run": pair.task.run,
                    "run_trial_index": pair.task.run_trial_index,
                    "task_annotation_index": pair.task.annotation_index,
                    "rest_annotation_index": pair.rest.annotation_index,
                    "semantic_condition": pair.task.semantic_condition,
                    "channel": channel,
                    "peak_frequency_hz": peak_frequency_hz,
                    "band_low_hz": float(task_power.frequencies_hz[0]),
                    "band_high_hz": float(task_power.frequencies_hz[-1]),
                    "frequency_center_count": int(task_power.frequencies_hz.size),
                    "frequency_centers_hz": "/".join(
                        f"{value:g}" for value in task_power.frequencies_hz
                    ),
                    "task_interval_start_seconds": config.task_analysis_interval_seconds[0],
                    "task_interval_stop_seconds": config.task_analysis_interval_seconds[1],
                    "rest_interval_start_seconds": config.paired_rest_reference_interval_seconds[0],
                    "rest_interval_stop_seconds": config.paired_rest_reference_interval_seconds[1],
                    "reference_mean_wavelet_power_v2": float(rest_values[trial_index, channel_index]),
                    "task_mean_wavelet_power_v2": float(task_values[trial_index, channel_index]),
                    "individualized_change_percent": float(changes[trial_index, channel_index]),
                    "individualized_change_db": float(db_changes[trial_index, channel_index]),
                    "source_file": canonical_source_id(str(pair.source_path)),
                    "source_sha256": pair.source_sha256,
                    "primary_retained": True,
                }
            )
    return rows


def peak_reliability_category(span_hz: float) -> str:
    """Apply the pre-registered three-fold peak-span categories."""
    if not np.isfinite(span_hz) or span_hz < 0:
        raise ValueError("Peak span must be finite and nonnegative.")
    if span_hz <= 0.5:
        return "stable"
    if span_hz <= 1.0:
        return "moderate"
    return "unstable"


def summarize_subject_methods(
    trial_rows: Sequence[Mapping[str, object]],
    peak_subject_rows: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate held-out runs/trials to one paired method row per person/channel."""
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in trial_rows:
        if row["semantic_condition"] == "both_fists_imagery":
            grouped[(int(row["subject"]), str(row["channel"]))].append(row)
    output: list[dict[str, object]] = []
    for (subject, channel), rows in sorted(grouped.items()):
        if subject not in peak_subject_rows or not peak_subject_rows[subject]["method_available"]:
            raise RuntimeError("Trial output contains a subject without complete peak availability.")
        individualized = np.array(
            [float(row["individualized_change_percent"]) for row in rows]
        )
        fixed = np.array([float(row["fixed_narrow_change_percent"]) for row in rows])
        broad = np.array([float(row["fixed_broad_change_percent"]) for row in rows])
        individualized_db = np.array(
            [float(row["individualized_change_db"]) for row in rows]
        )
        fixed_db = np.array([float(row["fixed_narrow_change_db"]) for row in rows])
        broad_db = np.array([float(row["fixed_broad_change_db"]) for row in rows])
        candidate = np.array(
            [row["quality_status"] == "statistical candidate" for row in rows], dtype=bool
        )
        if candidate.all():
            raise RuntimeError("QC sensitivity cannot remove every matched trial.")
        run_metrics: dict[int, tuple[float, float, float]] = {}
        for run in (6, 10, 14):
            run_rows = [row for row in rows if int(row["run"]) == run]
            if not run_rows:
                raise RuntimeError(f"Subject {subject} lacks run {run} method rows.")
            run_metrics[run] = (
                float(np.median([float(row["individualized_change_percent"]) for row in run_rows])),
                float(np.median([float(row["fixed_narrow_change_percent"]) for row in run_rows])),
                float(np.median([float(row["fixed_broad_change_percent"]) for row in run_rows])),
            )
        individualized_run_values = np.array([run_metrics[run][0] for run in (6, 10, 14)])
        fixed_run_values = np.array([run_metrics[run][1] for run in (6, 10, 14)])
        broad_run_values = np.array([run_metrics[run][2] for run in (6, 10, 14)])
        peak = peak_subject_rows[subject]
        output.append(
            {
                "subject": subject,
                "study_role": "held_out_method_evaluation",
                "channel": channel,
                "trial_count": len(rows),
                "peak_frequency_median_hz": peak["peak_frequency_median_hz"],
                "peak_distance_from_12_5_hz": abs(float(peak["peak_frequency_median_hz"]) - 12.5),
                "three_fold_peak_span_hz": peak["three_fold_peak_span_hz"],
                "peak_reliability": peak["peak_reliability"],
                "individualized_median_percent": float(np.median(individualized)),
                "fixed_narrow_median_percent": float(np.median(fixed)),
                "fixed_broad_median_percent": float(np.median(broad)),
                "individualized_minus_fixed_narrow_median_percent": float(
                    np.median(individualized) - np.median(fixed)
                ),
                "individualized_minus_fixed_broad_median_percent": float(
                    np.median(individualized) - np.median(broad)
                ),
                "individualized_median_db": float(np.median(individualized_db)),
                "fixed_narrow_median_db": float(np.median(fixed_db)),
                "fixed_broad_median_db": float(np.median(broad_db)),
                "individualized_minus_fixed_narrow_median_db": float(
                    np.median(individualized_db) - np.median(fixed_db)
                ),
                "individualized_qc_excluded_median_percent": float(
                    np.median(individualized[~candidate])
                ),
                "fixed_narrow_qc_excluded_median_percent": float(np.median(fixed[~candidate])),
                "qc_paired_difference_percent": float(
                    np.median(individualized[~candidate]) - np.median(fixed[~candidate])
                ),
                "individualized_negative_trial_fraction": float(np.mean(individualized < 0)),
                "fixed_narrow_negative_trial_fraction": float(np.mean(fixed < 0)),
                "low_individualized_reference_power_candidate_count": int(
                    sum(bool(row["individualized_low_reference_power_candidate"]) for row in rows)
                ),
                "individualized_run_6_median_percent": individualized_run_values[0],
                "individualized_run_10_median_percent": individualized_run_values[1],
                "individualized_run_14_median_percent": individualized_run_values[2],
                "fixed_narrow_run_6_median_percent": fixed_run_values[0],
                "fixed_narrow_run_10_median_percent": fixed_run_values[1],
                "fixed_narrow_run_14_median_percent": fixed_run_values[2],
                "fixed_broad_run_6_median_percent": broad_run_values[0],
                "fixed_broad_run_10_median_percent": broad_run_values[1],
                "fixed_broad_run_14_median_percent": broad_run_values[2],
                "individualized_negative_run_count": int(np.sum(individualized_run_values < 0)),
                "fixed_narrow_negative_run_count": int(np.sum(fixed_run_values < 0)),
                "fixed_broad_negative_run_count": int(np.sum(broad_run_values < 0)),
                "individualized_run_range_percentage_points": float(np.ptp(individualized_run_values)),
                "fixed_narrow_run_range_percentage_points": float(np.ptp(fixed_run_values)),
                "fixed_broad_run_range_percentage_points": float(np.ptp(broad_run_values)),
                "subject_is_group_observational_unit": True,
            }
        )
    return output


def classify_method_comparison(
    coverage_fraction: float,
    negative_fraction_gain: float,
    supporting_run_fraction_gain: float,
    median_paired_difference: float,
    qc_median_paired_difference: float,
    db_median_paired_difference: float,
    leave_one_out_maximum_paired_median: float,
) -> str:
    """Apply the final pre-outcome clear/mixed/no-improvement criteria."""
    clear = (
        coverage_fraction >= 0.8
        and negative_fraction_gain >= 0.05
        and supporting_run_fraction_gain >= 0.05
        and median_paired_difference < 0
        and qc_median_paired_difference < 0
        and db_median_paired_difference < 0
        and leave_one_out_maximum_paired_median < 0
    )
    if clear:
        return "clear improvement"
    consistency_gains = (negative_fraction_gain, supporting_run_fraction_gain)
    mixed = (
        coverage_fraction >= 0.6
        and median_paired_difference < 0
        and max(consistency_gains) >= 0
        and min(consistency_gains) >= -0.05
        and qc_median_paired_difference < 0
        and db_median_paired_difference < 0
    )
    return "mixed result" if mixed else "no improvement"


def paired_group_summaries(
    subject_rows: Sequence[Mapping[str, object]],
    eligible_evaluation_count: int,
) -> list[dict[str, object]]:
    """Summarize matched fixed/individualized evidence at the subject level."""
    output: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for channel in ("C4", "C3"):
        rows = [row for row in subject_rows if row["channel"] == channel]
        individualized = np.array([float(row["individualized_median_percent"]) for row in rows])
        fixed = np.array([float(row["fixed_narrow_median_percent"]) for row in rows])
        broad = np.array([float(row["fixed_broad_median_percent"]) for row in rows])
        differences = individualized - fixed
        differences_db = np.array(
            [float(row["individualized_minus_fixed_narrow_median_db"]) for row in rows]
        )
        qc_differences = np.array([float(row["qc_paired_difference_percent"]) for row in rows])
        loo = np.array(
            [float(np.median(np.delete(differences, index))) for index in range(len(rows))]
        )
        fixed_negative = int(np.sum(fixed < 0))
        individualized_negative = int(np.sum(individualized < 0))
        fixed_run_support = int(
            sum(int(row["fixed_narrow_negative_run_count"]) >= 2 for row in rows)
        )
        individualized_run_support = int(
            sum(int(row["individualized_negative_run_count"]) >= 2 for row in rows)
        )
        negative_gain = individualized_negative / len(rows) - fixed_negative / len(rows)
        run_gain = individualized_run_support / len(rows) - fixed_run_support / len(rows)
        if np.allclose(differences, 0):
            p_value = 1.0
        else:
            p_value = float(
                wilcoxon(
                    individualized,
                    fixed,
                    alternative="less",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        raw_p_values.append(p_value)
        summary = {
            "channel": channel,
            "eligible_evaluation_subject_count": eligible_evaluation_count,
            "matched_individualizable_subject_count": len(rows),
            "coverage_fraction": len(rows) / eligible_evaluation_count,
            "fixed_narrow_negative_subject_count": fixed_negative,
            "fixed_narrow_negative_subject_fraction": fixed_negative / len(rows),
            "individualized_negative_subject_count": individualized_negative,
            "individualized_negative_subject_fraction": individualized_negative / len(rows),
            "negative_subject_fraction_gain": negative_gain,
            "fixed_narrow_group_median_percent": float(np.median(fixed)),
            "individualized_group_median_percent": float(np.median(individualized)),
            "fixed_broad_group_median_percent": float(np.median(broad)),
            "median_paired_individualized_minus_fixed_narrow_percent": float(np.median(differences)),
            "q25_paired_difference_percent": float(np.percentile(differences, 25)),
            "q75_paired_difference_percent": float(np.percentile(differences, 75)),
            "minimum_paired_difference_percent": float(np.min(differences)),
            "maximum_paired_difference_percent": float(np.max(differences)),
            "trimmed_mean_paired_difference_percent": trimmed_mean(differences, 0.1),
            "fixed_narrow_at_least_two_negative_runs_count": fixed_run_support,
            "fixed_narrow_at_least_two_negative_runs_fraction": fixed_run_support / len(rows),
            "individualized_at_least_two_negative_runs_count": individualized_run_support,
            "individualized_at_least_two_negative_runs_fraction": individualized_run_support / len(rows),
            "at_least_two_negative_runs_fraction_gain": run_gain,
            "median_fixed_narrow_run_range_percentage_points": float(
                np.median([float(row["fixed_narrow_run_range_percentage_points"]) for row in rows])
            ),
            "median_individualized_run_range_percentage_points": float(
                np.median([float(row["individualized_run_range_percentage_points"]) for row in rows])
            ),
            "median_qc_paired_difference_percent": float(np.median(qc_differences)),
            "median_paired_difference_db": float(np.median(differences_db)),
            "percent_and_db_paired_difference_direction_agreement_count": int(
                np.sum(np.sign(differences) == np.sign(differences_db))
            ),
            "leave_one_subject_out_minimum_paired_median_percent": float(np.min(loo)),
            "leave_one_subject_out_maximum_paired_median_percent": float(np.max(loo)),
            "one_sided_paired_wilcoxon_p": p_value,
        }
        summary["method_category"] = classify_method_comparison(
            summary["coverage_fraction"],
            negative_gain,
            run_gain,
            summary["median_paired_individualized_minus_fixed_narrow_percent"],
            summary["median_qc_paired_difference_percent"],
            summary["median_paired_difference_db"],
            summary["leave_one_subject_out_maximum_paired_median_percent"],
        )
        output.append(summary)
    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(output, adjusted, strict=True):
        row["holm_adjusted_paired_wilcoxon_p"] = adjusted_p
    return output
