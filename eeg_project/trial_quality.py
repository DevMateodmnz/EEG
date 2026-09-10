"""Reusable artifact-policy and spectral feature-stability operations.

Quality classification in this module is deliberately label-independent.  Condition
labels are used only after QC status is fixed, when scientific sensitivity summaries
are calculated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .epoching import modified_robust_z


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIAL_QUALITY_CONFIG_PATH = PROJECT_ROOT / "config" / "trial_quality.json"


@dataclass(frozen=True)
class StabilityCriteria:
    """Predefined criteria used to grade one spectral measurement."""

    minimum_timing_combinations_same_direction: int
    minimum_trial_direction_fraction: float
    maximum_absolute_denominator_spearman_rho: float
    maximum_leave_one_candidate_median_shift_percentage_points: float


@dataclass(frozen=True)
class TrialQualityConfig:
    """Compact, versioned project policy for quality and stability decisions."""

    schema_version: int
    quality_metrics: tuple[str, ...]
    robust_z_threshold: float
    strong_review_robust_z_threshold: float
    strong_review_minimum_independent_scalar_flags: int
    candidate_policy: str
    absolute_amplitudes_are_review_evidence_only: bool
    normalizations: tuple[str, ...]
    sensitivity_sets: tuple[str, ...]
    trimmed_mean_fraction_each_tail: float
    stability_criteria: StabilityCriteria
    high_minimum_passed_criteria: int
    moderate_minimum_passed_criteria: int
    primary_features: tuple[tuple[str, str, str], ...]
    quality_decisions_must_not_use_condition_labels: bool
    ica_policy: str


QUALITY_METRICS = (
    "max_p2p_uV",
    "median_channel_p2p_uV",
    "frontal_max_p2p_uV",
    "max_abs_step_uV",
    "minimum_channel_std_uV",
    "median_channel_std_uV",
)
HIGH_OUTLIER_METRICS = QUALITY_METRICS[:4] + (QUALITY_METRICS[5],)
LOW_OUTLIER_METRICS = (QUALITY_METRICS[4],)


def load_trial_quality_config(
    path: Path = DEFAULT_TRIAL_QUALITY_CONFIG_PATH,
) -> TrialQualityConfig:
    """Load and strictly validate the policy frozen before the analysis."""
    with path.expanduser().resolve().open(encoding="utf-8") as config_file:
        payload: dict[str, Any] = json.load(config_file)
    criteria = payload["stability_criteria"]
    grades = payload["stability_grades"]
    config = TrialQualityConfig(
        schema_version=int(payload["schema_version"]),
        quality_metrics=tuple(payload["quality_metrics"]),
        robust_z_threshold=float(payload["robust_z_threshold"]),
        strong_review_robust_z_threshold=float(
            payload["strong_review_robust_z_threshold"]
        ),
        strong_review_minimum_independent_scalar_flags=int(
            payload["strong_review_minimum_independent_scalar_flags"]
        ),
        candidate_policy=str(payload["candidate_policy"]),
        absolute_amplitudes_are_review_evidence_only=bool(
            payload["absolute_amplitudes_are_review_evidence_only"]
        ),
        normalizations=tuple(payload["normalizations"]),
        sensitivity_sets=tuple(payload["sensitivity_sets"]),
        trimmed_mean_fraction_each_tail=float(
            payload["trimmed_mean_fraction_each_tail"]
        ),
        stability_criteria=StabilityCriteria(
            minimum_timing_combinations_same_direction=int(
                criteria["minimum_timing_combinations_same_direction"]
            ),
            minimum_trial_direction_fraction=float(
                criteria["minimum_trial_direction_fraction"]
            ),
            maximum_absolute_denominator_spearman_rho=float(
                criteria["maximum_absolute_denominator_spearman_rho"]
            ),
            maximum_leave_one_candidate_median_shift_percentage_points=float(
                criteria[
                    "maximum_leave_one_candidate_median_shift_percentage_points"
                ]
            ),
        ),
        high_minimum_passed_criteria=int(
            grades["high_minimum_passed_criteria"]
        ),
        moderate_minimum_passed_criteria=int(
            grades["moderate_minimum_passed_criteria"]
        ),
        primary_features=tuple(tuple(values) for values in payload["primary_features"]),
        quality_decisions_must_not_use_condition_labels=bool(
            payload["quality_decisions_must_not_use_condition_labels"]
        ),
        ica_policy=str(payload["ica_policy"]),
    )
    validate_trial_quality_config(config)
    return config


def validate_trial_quality_config(config: TrialQualityConfig) -> None:
    """Reject ambiguous settings or policies that permit circular QC."""
    if config.schema_version != 1:
        raise ValueError(f"Unsupported trial-quality schema: {config.schema_version}.")
    if config.robust_z_threshold <= 0:
        raise ValueError("Robust-z threshold must be positive.")
    if config.strong_review_robust_z_threshold <= config.robust_z_threshold:
        raise ValueError("Strong-review threshold must exceed the candidate threshold.")
    if config.normalizations != ("percent_change", "db_ratio"):
        raise ValueError("The reviewed normalization variants are percent and dB.")
    expected_quality_metrics = (
        "maximum_peak_to_peak",
        "median_channel_peak_to_peak",
        "frontal_maximum_peak_to_peak",
        "maximum_absolute_sample_step",
        "minimum_channel_standard_deviation",
        "median_channel_standard_deviation",
    )
    if config.quality_metrics != expected_quality_metrics:
        raise ValueError("Unexpected label-independent quality metric policy.")
    if config.sensitivity_sets != (
        "all_trials",
        "exclude_all_task_candidates",
        "exclude_strong_review_candidates",
        "exclude_first_pair_per_run",
        "configured_timing_grid",
    ):
        raise ValueError("Unexpected predefined sensitivity sets.")
    if not 0 <= config.trimmed_mean_fraction_each_tail < 0.5:
        raise ValueError("Trim fraction must be in [0, 0.5).")
    if not config.quality_decisions_must_not_use_condition_labels:
        raise ValueError("Condition labels must never define quality status.")
    if config.candidate_policy != "retain_primary_flag_metadata_sensitivity_only":
        raise ValueError("Primary trials must retain statistical QC candidates.")
    if len(config.primary_features) == 0 or len(set(config.primary_features)) != len(
        config.primary_features
    ):
        raise ValueError("Primary feature definitions must be non-empty and unique.")


def db_power_ratio(task_power: np.ndarray, reference_power: np.ndarray) -> np.ndarray:
    """Return ``10 log10(task/reference)`` with strict positive-power checks."""
    task = np.asarray(task_power, dtype=float)
    reference = np.asarray(reference_power, dtype=float)
    if (
        np.any(task <= 0)
        or np.any(reference <= 0)
        or not np.isfinite(task).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Task and reference power must be finite and strictly positive.")
    ratio_db = 10.0 * np.log10(task / reference)
    if not np.isfinite(ratio_db).all():
        raise RuntimeError("dB normalization produced NaN or infinity.")
    return ratio_db


def trimmed_mean(values: np.ndarray, fraction_each_tail: float) -> float:
    """Return a deterministic symmetric trimmed mean without an extra dependency."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Trimmed-mean input must be a non-empty finite vector.")
    if not 0 <= fraction_each_tail < 0.5:
        raise ValueError("Trim fraction must be in [0, 0.5).")
    trim_count = int(np.floor(values.size * fraction_each_tail))
    ordered = np.sort(values)
    retained = ordered[trim_count : values.size - trim_count]
    if retained.size == 0:
        raise ValueError("Trim setting removed every observation.")
    return float(np.mean(retained))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign one-based average ranks, including deterministic tie handling."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman rank correlation; constant inputs yield zero association."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("Spearman inputs must be equal one-dimensional vectors.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman inputs must be finite.")
    x_ranks = _average_ranks(x)
    y_ranks = _average_ranks(y)
    if np.std(x_ranks) == 0 or np.std(y_ranks) == 0:
        return 0.0
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def paired_side_quality_metrics(
    data_volts: np.ndarray,
    times_seconds: np.ndarray,
    channel_names: Sequence[str],
    frontal_channels: Sequence[str],
    interval_seconds: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Calculate the already-justified compact metrics for one side of each pair."""
    data = np.asarray(data_volts, dtype=float)
    times = np.asarray(times_seconds, dtype=float)
    if data.ndim != 3 or data.shape[1] != len(channel_names):
        raise ValueError("Expected data shaped (trials, channels, times).")
    if data.shape[2] != times.size or not np.isfinite(data).all():
        raise ValueError("Data/time dimensions differ or contain non-finite values.")
    missing = sorted(set(frontal_channels) - set(channel_names))
    if missing:
        raise ValueError(f"Missing frontal channels: {missing}.")
    mask = (times >= interval_seconds[0] - 1e-12) & (
        times <= interval_seconds[1] + 1e-12
    )
    if not mask.any():
        raise ValueError("Quality interval contains no samples.")
    data_uv = data[:, :, mask] * 1_000_000.0
    p2p = np.ptp(data_uv, axis=2)
    standard_deviation = np.std(data_uv, axis=2)
    frontal_indices = [channel_names.index(channel) for channel in frontal_channels]
    return {
        "max_p2p_uV": np.max(p2p, axis=1),
        "median_channel_p2p_uV": np.median(p2p, axis=1),
        "frontal_max_p2p_uV": np.max(p2p[:, frontal_indices], axis=1),
        "max_abs_step_uV": np.max(np.abs(np.diff(data_uv, axis=2)), axis=(1, 2)),
        "minimum_channel_std_uV": np.min(standard_deviation, axis=1),
        "median_channel_std_uV": np.median(standard_deviation, axis=1),
    }


def classify_metric_candidates(
    metrics: Mapping[str, np.ndarray], threshold: float
) -> tuple[list[dict[str, float]], list[tuple[str, ...]], np.ndarray]:
    """Classify robust outliers without receiving or inspecting condition labels."""
    if tuple(metrics) != QUALITY_METRICS:
        raise ValueError(f"Expected compact quality metrics in order: {QUALITY_METRICS}.")
    lengths = {np.asarray(values).size for values in metrics.values()}
    if len(lengths) != 1:
        raise ValueError("Quality metric lengths differ.")
    scores = {name: modified_robust_z(values) for name, values in metrics.items()}
    count = next(iter(lengths))
    evidence: list[tuple[str, ...]] = []
    rows: list[dict[str, float]] = []
    candidate = np.zeros(count, dtype=bool)
    for index in range(count):
        flags: list[str] = []
        for name in HIGH_OUTLIER_METRICS:
            if scores[name][index] > threshold:
                flags.append(f"high {name}")
        for name in LOW_OUTLIER_METRICS:
            if scores[name][index] < -threshold:
                flags.append(f"low {name}")
        candidate[index] = bool(flags)
        evidence.append(tuple(flags))
        rows.append(
            {
                **{name: float(metrics[name][index]) for name in QUALITY_METRICS},
                **{
                    f"{name}_robust_z": float(scores[name][index])
                    for name in QUALITY_METRICS
                },
            }
        )
    return rows, evidence, candidate


def scalar_evidence_count(quality_evidence: str) -> int:
    """Count independent scalar flags, excluding channel-specific support text."""
    return sum(
        bool(item.strip()) and not item.strip().startswith("channel-specific")
        for item in quality_evidence.split(";")
    )


def is_strong_review_candidate(
    quality_row: Mapping[str, object], config: TrialQualityConfig
) -> bool:
    """Apply the predefined strong-review rule to QC evidence only."""
    robust_scores = [
        abs(float(value))
        for key, value in quality_row.items()
        if key.endswith("_robust_z") and np.isfinite(float(value))
    ]
    return bool(
        quality_row["quality_status"] == "statistical candidate"
        and (
            (robust_scores and max(robust_scores) >= config.strong_review_robust_z_threshold)
            or scalar_evidence_count(str(quality_row["quality_evidence"]))
            >= config.strong_review_minimum_independent_scalar_flags
        )
    )


def systematic_matched_controls(
    quality_rows: Sequence[Mapping[str, object]],
    metric_names: Sequence[str],
) -> dict[tuple[int, int], tuple[int, int]]:
    """Match each candidate to the nearest non-candidate in its run and condition."""
    matrix = np.array(
        [[float(row[name]) for name in metric_names] for row in quality_rows], dtype=float
    )
    center = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - center), axis=0)
    scale[scale == 0] = 1.0
    standardized = (matrix - center) / scale
    matches: dict[tuple[int, int], tuple[int, int]] = {}
    for index, row in enumerate(quality_rows):
        if row["quality_status"] != "statistical candidate":
            continue
        eligible = [
            other_index
            for other_index, other in enumerate(quality_rows)
            if other["quality_status"] == "not flagged"
            and int(other["run"]) == int(row["run"])
            and str(other["semantic_condition"]) == str(row["semantic_condition"])
        ]
        if not eligible:
            raise RuntimeError("A candidate has no same-run, same-condition control.")
        distances = [
            (float(np.linalg.norm(standardized[index] - standardized[other])), other)
            for other in eligible
        ]
        _, matched_index = min(
            distances,
            key=lambda item: (
                item[0],
                int(quality_rows[item[1]]["run_trial_index"]),
            ),
        )
        matches[(int(row["run"]), int(row["run_trial_index"]))] = (
            int(quality_rows[matched_index]["run"]),
            int(quality_rows[matched_index]["run_trial_index"]),
        )
    return matches


def leave_one_out_median_influence(
    values: np.ndarray, identities: Sequence[tuple[int, int]]
) -> list[dict[str, float | int]]:
    """Return the change in median after omitting each identified observation."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != len(identities) or values.size < 2:
        raise ValueError("Values and identities must align and contain at least two rows.")
    baseline = float(np.median(values))
    rows: list[dict[str, float | int]] = []
    for index, (run, trial) in enumerate(identities):
        without = float(np.median(np.delete(values, index)))
        rows.append(
            {
                "run": run,
                "run_trial_index": trial,
                "all_trial_median": baseline,
                "median_without_trial": without,
                "median_shift": without - baseline,
                "absolute_median_shift": abs(without - baseline),
            }
        )
    return rows


def direction(value: float) -> str:
    """Describe a signed result without treating tiny non-zero values as zero."""
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"


def same_nonzero_direction(values: Sequence[float], expected: str) -> bool:
    """Return whether every value has the specified non-zero direction."""
    if expected not in {"negative", "positive"}:
        return False
    return bool(values) and all(direction(float(value)) == expected for value in values)


def grade_stability(
    criteria_passed: Mapping[str, bool], config: TrialQualityConfig
) -> tuple[int, str]:
    """Apply the predefined 7-criterion high/moderate/low grading rule."""
    if len(criteria_passed) != 7:
        raise ValueError("Stability grading requires exactly seven predefined criteria.")
    passed = sum(bool(value) for value in criteria_passed.values())
    if passed >= config.high_minimum_passed_criteria:
        grade = "high"
    elif passed >= config.moderate_minimum_passed_criteria:
        grade = "moderate"
    else:
        grade = "low"
    return passed, grade


def add_normalization_diagnostics(
    trial_rows: Sequence[Mapping[str, object]], robust_threshold: float
) -> list[dict[str, object]]:
    """Add dB ratios and label-independent denominator/task robust scores."""
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(trial_rows):
        grouped.setdefault((str(row["channel"]), str(row["frequency_band"])), []).append(
            index
        )
    output = [dict(row) for row in trial_rows]
    for indices in grouped.values():
        references = np.array(
            [float(trial_rows[index]["reference_mean_wavelet_power_v2"]) for index in indices]
        )
        tasks = np.array(
            [float(trial_rows[index]["task_mean_wavelet_power_v2"]) for index in indices]
        )
        # Log power makes multiplicative deviations comparable before robust scoring.
        reference_scores = modified_robust_z(np.log10(references))
        task_scores = modified_robust_z(np.log10(tasks))
        db_values = db_power_ratio(tasks, references)
        for local_index, row_index in enumerate(indices):
            output[row_index].update(
                {
                    "change_db": float(db_values[local_index]),
                    "reference_log_power_robust_z": float(reference_scores[local_index]),
                    "task_log_power_robust_z": float(task_scores[local_index]),
                    "low_reference_power_candidate": bool(
                        reference_scores[local_index] < -robust_threshold
                    ),
                }
            )
    return output


def build_candidate_influence_rows(
    diagnostic_rows: Sequence[Mapping[str, object]],
    config: TrialQualityConfig,
) -> list[dict[str, object]]:
    """Quantify candidate influence after QC identities have already been fixed."""
    output: list[dict[str, object]] = []
    for condition, channel, band in config.primary_features:
        selected = [
            row
            for row in diagnostic_rows
            if row["semantic_condition"] == condition
            and row["channel"] == channel
            and row["frequency_band"] == band
        ]
        values = np.array([float(row["change_percent"]) for row in selected])
        identities = [
            (int(row["run"]), int(row["run_trial_index"])) for row in selected
        ]
        influence = {
            (int(row["run"]), int(row["run_trial_index"])): row
            for row in leave_one_out_median_influence(values, identities)
        }
        for row in selected:
            if row["quality_status"] != "statistical candidate":
                continue
            identity = (int(row["run"]), int(row["run_trial_index"]))
            output.append(
                {
                    "subject": row["subject"],
                    "run": identity[0],
                    "run_trial_index": identity[1],
                    "semantic_condition": condition,
                    "task_event_time_seconds": row["task_event_time_seconds"],
                    "quality_status": row["quality_status"],
                    "quality_evidence": row["quality_evidence"],
                    "channel": channel,
                    "frequency_band": band,
                    "candidate_change_percent": row["change_percent"],
                    **influence[identity],
                }
            )
    return output


def build_feature_stability_rows(
    diagnostic_rows: Sequence[Mapping[str, object]],
    timing_rows: Sequence[Mapping[str, object]],
    quality_by_identity: Mapping[tuple[int, int], Mapping[str, object]],
    config: TrialQualityConfig,
) -> list[dict[str, object]]:
    """Evaluate the predefined small sensitivity grid for each primary feature."""
    criteria = config.stability_criteria
    strong_ids = {
        identity
        for identity, row in quality_by_identity.items()
        if is_strong_review_candidate(row, config)
    }
    output: list[dict[str, object]] = []
    for condition, channel, band in config.primary_features:
        selected = [
            row
            for row in diagnostic_rows
            if row["semantic_condition"] == condition
            and row["channel"] == channel
            and row["frequency_band"] == band
        ]
        if not selected:
            raise RuntimeError(f"No trial rows for {(condition, channel, band)}.")
        percent = np.array([float(row["change_percent"]) for row in selected])
        db = np.array([float(row["change_db"]) for row in selected])
        primary_median = float(np.median(percent))
        expected_direction = direction(primary_median)
        identities = [(int(row["run"]), int(row["run_trial_index"])) for row in selected]
        candidate_mask = np.array(
            [row["quality_status"] == "statistical candidate" for row in selected]
        )
        strong_mask = np.array([identity in strong_ids for identity in identities])
        first_mask = np.array([int(row["run_trial_index"]) == 0 for row in selected])
        run_values: dict[int, np.ndarray] = {}
        for run in sorted({int(row["run"]) for row in selected}):
            run_values[run] = percent[
                np.array([int(row["run"]) == run for row in selected])
            ]
        run_medians = [float(np.median(values)) for values in run_values.values()]
        same_direction_count = int(
            np.sum(
                [
                    direction(float(row["median_change_percent"])) == expected_direction
                    for row in timing_rows
                    if row["semantic_condition"] == condition
                    and row["channel"] == channel
                    and row["frequency_band"] == band
                ]
            )
        )
        timing_total = sum(
            row["semantic_condition"] == condition
            and row["channel"] == channel
            and row["frequency_band"] == band
            for row in timing_rows
        )
        if timing_total == 0:
            raise RuntimeError(f"No timing rows for {(condition, channel, band)}.")
        fraction_in_direction = float(
            np.mean([direction(value) == expected_direction for value in percent])
        )
        reference = np.array(
            [float(row["reference_mean_wavelet_power_v2"]) for row in selected]
        )
        denominator_rho = spearman_rho(np.abs(percent), 1.0 / reference)
        candidate_influences = build_candidate_influence_rows(
            selected,
            replace(config, primary_features=((condition, channel, band),)),
        )
        max_influence = max(
            (float(row["absolute_median_shift"]) for row in candidate_influences),
            default=0.0,
        )
        qc_median = float(np.median(percent[~candidate_mask]))
        strong_median = float(np.median(percent[~strong_mask]))
        db_median = float(np.median(db))
        passed_map = {
            "same_direction_all_three_runs": same_nonzero_direction(
                run_medians, expected_direction
            ),
            "qc_sensitivity_preserves_direction": same_nonzero_direction(
                [qc_median, strong_median], expected_direction
            ),
            "timing_sensitivity": same_direction_count
            >= criteria.minimum_timing_combinations_same_direction,
            "trial_direction_fraction": fraction_in_direction
            >= criteria.minimum_trial_direction_fraction,
            "normalization_preserves_direction": direction(db_median)
            == expected_direction,
            "denominator_stability": abs(denominator_rho)
            < criteria.maximum_absolute_denominator_spearman_rho,
            "candidate_influence": max_influence
            <= criteria.maximum_leave_one_candidate_median_shift_percentage_points,
        }
        passed_count, grade = grade_stability(passed_map, config)
        q25, q75 = np.percentile(percent, [25, 75])
        row: dict[str, object] = {
            "semantic_condition": condition,
            "channel": channel,
            "frequency_band": band,
            "trial_count": len(selected),
            "direction": expected_direction,
            "median_change_percent": primary_median,
            "mean_change_percent": float(np.mean(percent)),
            "trimmed_mean_change_percent": trimmed_mean(
                percent, config.trimmed_mean_fraction_each_tail
            ),
            "iqr_change_percentage_points": float(q75 - q25),
            "median_change_db": db_median,
            "mean_change_db": float(np.mean(db)),
            "exclude_all_candidates_median_percent": qc_median,
            "exclude_strong_candidates_median_percent": strong_median,
            "exclude_first_pair_median_percent": float(np.median(percent[~first_mask])),
            "run_medians_percent": ";".join(
                f"run{run}={np.median(values):.6f}" for run, values in run_values.items()
            ),
            "run_iqrs_percent": ";".join(
                f"run{run}={np.subtract(*np.percentile(values, [75, 25])):.6f}"
                for run, values in run_values.items()
            ),
            "run_trial_counts": ";".join(
                f"run{run}={values.size}" for run, values in run_values.items()
            ),
            "run_mads_percent": ";".join(
                f"run{run}={np.median(np.abs(values - np.median(values))):.6f}"
                for run, values in run_values.items()
            ),
            "trial_count_in_primary_direction": int(
                sum(direction(value) == expected_direction for value in percent)
            ),
            "trial_fraction_in_primary_direction": fraction_in_direction,
            "timing_same_direction_count": same_direction_count,
            "timing_combination_count": timing_total,
            "denominator_spearman_rho_abs_change_vs_inverse_reference": denominator_rho,
            "maximum_candidate_median_shift_percentage_points": max_influence,
            **{f"criterion_{name}": passed for name, passed in passed_map.items()},
            "criteria_passed": passed_count,
            "criteria_total": 7,
            "stability_grade": grade,
            "feature_worthiness": (
                "strong scientific candidate"
                if grade == "high"
                else "possible candidate"
                if grade == "moderate"
                else "unstable / not currently justified"
            ),
        }
        output.append(row)
    return output
