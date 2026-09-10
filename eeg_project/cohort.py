"""Reusable cohort audit, subject aggregation, and replication operations."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
from mne.datasets import eegbci
from mne.time_frequency import psd_array_welch
import numpy as np

from .epoching import modified_robust_z
from .preprocessing import DEFAULT_DATA_DIRECTORY, find_valid_stop, sha256_file
from .time_frequency import descriptive_statistics
from .trial_quality import db_power_ratio, trimmed_mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLICATION_CONFIG_PATH = PROJECT_ROOT / "config" / "replication_cohort.json"

EXPECTED_EEGBCI_CHANNELS = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7",
    "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7",
    "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2",
    "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
    "Iz",
)


@dataclass(frozen=True)
class TechnicalEligibility:
    """Frozen rules that determine compatibility without inspecting outcomes."""

    required_sampling_frequency_hz: float
    required_channel_count: int
    required_annotation_count_per_run: int
    required_rest_annotation_count_per_run: int
    required_total_task_annotation_count_per_run: int
    allowed_T1_counts_per_run: tuple[int, ...]
    allowed_T2_counts_per_run: tuple[int, ...]
    required_all_runs_for_primary_replication: bool
    allow_deterministic_channel_name_standardization: bool
    confirmed_flat_channel_standard_deviation_uV: float
    scientifically_unusual_outcomes_remain_eligible: bool
    outcome_based_exclusion_forbidden: bool


@dataclass(frozen=True)
class ReplicationSuccessCriteria:
    """Outcome categories frozen before replication subjects are processed."""

    predicted_direction: str
    strong_minimum_subject_fraction_in_direction: float
    strong_requires_cohort_median_in_direction: bool
    strong_minimum_fraction_with_at_least_two_of_three_runs_in_direction: float
    strong_requires_qc_sensitivity_direction: bool
    strong_requires_leave_one_subject_out_direction: bool
    mixed_minimum_subject_fraction_in_direction: float
    mixed_requires_cohort_median_in_direction: bool


@dataclass(frozen=True)
class CentralPeakSettings:
    """Predefined exploratory continuous-PSD peak measurement."""

    channels: tuple[str, ...]
    integer_frequency_centers_hz: tuple[int, ...]
    welch_window_seconds: float
    welch_overlap_fraction: float
    minimum_peak_prominence_db_over_band_median: float
    may_redefine_primary_replication_band: bool


@dataclass(frozen=True)
class GroupInferenceSettings:
    """Small, predefined subject-level inferential policy."""

    primary_test: str
    null_negative_direction_probability: float
    multiple_comparison_adjustment: str
    direction_proportion_interval: str
    bootstrap_subject_median_resamples: int
    bootstrap_seed: int


@dataclass(frozen=True)
class ReplicationCohortConfig:
    """Complete cohort definition and frozen replication policy."""

    schema_version: int
    development_subject: int
    replication_subjects: tuple[int, ...]
    runs: tuple[int, ...]
    frozen_configuration_paths: dict[str, str]
    primary_replication_features: tuple[tuple[str, str, str], ...]
    secondary_features: tuple[tuple[str, str, str], ...]
    technical_eligibility: TechnicalEligibility
    replication_success_criteria: ReplicationSuccessCriteria
    central_peak_exploration: CentralPeakSettings
    group_inference: GroupInferenceSettings
    cache_policy: str
    persist_full_tfr_arrays: bool

    @property
    def all_subjects(self) -> tuple[int, ...]:
        return (self.development_subject, *self.replication_subjects)

    @property
    def all_features(self) -> tuple[tuple[str, str, str], ...]:
        return (*self.primary_replication_features, *self.secondary_features)


def load_replication_cohort_config(
    path: Path = DEFAULT_REPLICATION_CONFIG_PATH,
) -> ReplicationCohortConfig:
    """Load and strictly validate the frozen cohort policy."""
    with path.expanduser().resolve().open(encoding="utf-8") as config_file:
        payload: dict[str, Any] = json.load(config_file)
    eligibility = payload["technical_eligibility"]
    success = payload["replication_success_criteria"]
    peak = payload["central_peak_exploration"]
    inference = payload["group_inference"]
    config = ReplicationCohortConfig(
        schema_version=int(payload["schema_version"]),
        development_subject=int(payload["development_subject"]),
        replication_subjects=tuple(int(value) for value in payload["replication_subjects"]),
        runs=tuple(int(value) for value in payload["runs"]),
        frozen_configuration_paths={
            str(name): str(value)
            for name, value in payload["frozen_configuration_paths"].items()
        },
        primary_replication_features=tuple(
            tuple(str(value) for value in feature)
            for feature in payload["primary_replication_features"]
        ),
        secondary_features=tuple(
            tuple(str(value) for value in feature)
            for feature in payload["secondary_features"]
        ),
        technical_eligibility=TechnicalEligibility(
            required_sampling_frequency_hz=float(
                eligibility["required_sampling_frequency_hz"]
            ),
            required_channel_count=int(eligibility["required_channel_count"]),
            required_annotation_count_per_run=int(
                eligibility["required_annotation_count_per_run"]
            ),
            required_rest_annotation_count_per_run=int(
                eligibility["required_rest_annotation_count_per_run"]
            ),
            required_total_task_annotation_count_per_run=int(
                eligibility["required_total_task_annotation_count_per_run"]
            ),
            allowed_T1_counts_per_run=tuple(
                int(value) for value in eligibility["allowed_T1_counts_per_run"]
            ),
            allowed_T2_counts_per_run=tuple(
                int(value) for value in eligibility["allowed_T2_counts_per_run"]
            ),
            required_all_runs_for_primary_replication=bool(
                eligibility["required_all_runs_for_primary_replication"]
            ),
            allow_deterministic_channel_name_standardization=bool(
                eligibility["allow_deterministic_channel_name_standardization"]
            ),
            confirmed_flat_channel_standard_deviation_uV=float(
                eligibility["confirmed_flat_channel_standard_deviation_uV"]
            ),
            scientifically_unusual_outcomes_remain_eligible=bool(
                eligibility["scientifically_unusual_outcomes_remain_eligible"]
            ),
            outcome_based_exclusion_forbidden=bool(
                eligibility["outcome_based_exclusion_forbidden"]
            ),
        ),
        replication_success_criteria=ReplicationSuccessCriteria(
            predicted_direction=str(success["predicted_direction"]),
            strong_minimum_subject_fraction_in_direction=float(
                success["strong_minimum_subject_fraction_in_direction"]
            ),
            strong_requires_cohort_median_in_direction=bool(
                success["strong_requires_cohort_median_in_direction"]
            ),
            strong_minimum_fraction_with_at_least_two_of_three_runs_in_direction=float(
                success[
                    "strong_minimum_fraction_with_at_least_two_of_three_runs_in_direction"
                ]
            ),
            strong_requires_qc_sensitivity_direction=bool(
                success["strong_requires_qc_sensitivity_direction"]
            ),
            strong_requires_leave_one_subject_out_direction=bool(
                success["strong_requires_leave_one_subject_out_direction"]
            ),
            mixed_minimum_subject_fraction_in_direction=float(
                success["mixed_minimum_subject_fraction_in_direction"]
            ),
            mixed_requires_cohort_median_in_direction=bool(
                success["mixed_requires_cohort_median_in_direction"]
            ),
        ),
        central_peak_exploration=CentralPeakSettings(
            channels=tuple(peak["channels"]),
            integer_frequency_centers_hz=tuple(
                int(value) for value in peak["integer_frequency_centers_hz"]
            ),
            welch_window_seconds=float(peak["welch_window_seconds"]),
            welch_overlap_fraction=float(peak["welch_overlap_fraction"]),
            minimum_peak_prominence_db_over_band_median=float(
                peak["minimum_peak_prominence_db_over_band_median"]
            ),
            may_redefine_primary_replication_band=bool(
                peak["may_redefine_primary_replication_band"]
            ),
        ),
        group_inference=GroupInferenceSettings(
            primary_test=str(inference["primary_test"]),
            null_negative_direction_probability=float(
                inference["null_negative_direction_probability"]
            ),
            multiple_comparison_adjustment=str(
                inference["multiple_comparison_adjustment"]
            ),
            direction_proportion_interval=str(
                inference["direction_proportion_interval"]
            ),
            bootstrap_subject_median_resamples=int(
                inference["bootstrap_subject_median_resamples"]
            ),
            bootstrap_seed=int(inference["bootstrap_seed"]),
        ),
        cache_policy=str(payload["cache_policy"]),
        persist_full_tfr_arrays=bool(payload["persist_full_tfr_arrays"]),
    )
    validate_replication_cohort_config(config)
    return config


def validate_replication_cohort_config(config: ReplicationCohortConfig) -> None:
    """Enforce the preregistered-like cohort and scientific constraints."""
    if config.schema_version != 1:
        raise ValueError(f"Unsupported replication schema: {config.schema_version}.")
    if config.development_subject != 1 or config.replication_subjects != tuple(range(2, 21)):
        raise ValueError("The frozen cohort must be subject 1 plus subjects 2-20.")
    if config.runs != (6, 10, 14):
        raise ValueError("Replication runs must remain 6, 10, and 14.")
    if config.primary_replication_features != (
        ("both_fists_imagery", "C4", "central_12_13"),
        ("both_fists_imagery", "C3", "central_12_13"),
    ):
        raise ValueError("Unexpected primary replication hypotheses.")
    if len(config.all_features) != 12 or len(set(config.all_features)) != 12:
        raise ValueError("The frozen cohort must contain 12 unique predefined features.")
    eligibility = config.technical_eligibility
    if (
        eligibility.required_annotation_count_per_run != 30
        or eligibility.required_rest_annotation_count_per_run != 15
        or eligibility.required_total_task_annotation_count_per_run != 15
        or eligibility.allowed_T1_counts_per_run != (7, 8)
        or eligibility.allowed_T2_counts_per_run != (7, 8)
    ):
        raise ValueError("Unexpected annotation-count policy.")
    if not (
        eligibility.required_all_runs_for_primary_replication
        and eligibility.scientifically_unusual_outcomes_remain_eligible
        and eligibility.outcome_based_exclusion_forbidden
    ):
        raise ValueError("Technical eligibility must be complete and outcome-independent.")
    success = config.replication_success_criteria
    if success.predicted_direction != "negative":
        raise ValueError("Primary replication direction must remain negative.")
    if config.central_peak_exploration.may_redefine_primary_replication_band:
        raise ValueError("Exploratory peaks must not redefine the primary band.")
    if config.persist_full_tfr_arrays:
        raise ValueError("Full cohort TFR arrays must not be persisted.")


def file_sha256(path: Path) -> str:
    """Alias the source hash operation for explicit cohort provenance."""
    return sha256_file(path)


def configuration_hashes(config: ReplicationCohortConfig) -> dict[str, str]:
    """Hash every frozen scientific configuration."""
    hashes: dict[str, str] = {}
    for name, configured_path in config.frozen_configuration_paths.items():
        path = (PROJECT_ROOT / configured_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen configuration: {path}.")
        hashes[name] = file_sha256(path)
    return hashes


def channel_quality_candidates(data_volts: np.ndarray) -> tuple[list[str], list[str]]:
    """Return statistical candidates and deterministic flat/non-finite failures."""
    data = np.asarray(data_volts, dtype=float)
    if data.ndim != 2 or data.shape[0] != len(EXPECTED_EEGBCI_CHANNELS):
        raise ValueError("Expected continuous data shaped (64, samples).")
    failures = [
        EXPECTED_EEGBCI_CHANNELS[index]
        for index in range(data.shape[0])
        if not np.isfinite(data[index]).all() or np.std(data[index]) == 0
    ]
    if not np.isfinite(data).all():
        return [], failures
    data_uv = data * 1_000_000.0
    means = np.mean(data_uv, axis=1)
    standard_deviations = np.std(data_uv, axis=1)
    peak_to_peak = np.ptp(data_uv, axis=1)
    maximum_steps = np.max(np.abs(np.diff(data_uv, axis=1)), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = np.corrcoef(data_uv)
    np.fill_diagonal(correlations, np.nan)
    peer_correlation = np.nanmedian(correlations, axis=1)
    scores = {
        "mean": modified_robust_z(means),
        "std": modified_robust_z(standard_deviations),
        "p2p": modified_robust_z(peak_to_peak),
        "step": modified_robust_z(maximum_steps),
        "peer": modified_robust_z(peer_correlation),
    }
    candidates = [
        EXPECTED_EEGBCI_CHANNELS[index]
        for index in range(data.shape[0])
        if abs(scores["mean"][index]) > 3.5
        or abs(scores["std"][index]) > 3.5
        or scores["p2p"][index] > 3.5
        or scores["step"][index] > 3.5
        or scores["peer"][index] < -3.5
    ]
    return candidates, failures


def audit_recording(
    subject: int,
    run: int,
    config: ReplicationCohortConfig,
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
) -> dict[str, object]:
    """Download/load one run and audit technical compatibility before outcomes."""
    base: dict[str, object] = {
        "subject": subject,
        "subject_role": "development" if subject == config.development_subject else "replication",
        "run": run,
        "file_exists": False,
        "load_success": False,
        "source_file": "",
        "source_sha256": "",
        "edf_format": "",
        "sampling_frequency_hz": "",
        "channel_count": "",
        "channel_names_compatible": False,
        "annotation_count": "",
        "T0_count": "",
        "T1_count": "",
        "T2_count": "",
        "annotation_descriptions_compatible": False,
        "annotation_alternation_compatible": False,
        "duration_seconds": "",
        "original_samples": "",
        "valid_samples": "",
        "trailing_zero_samples": "",
        "earlier_all_channel_zero_samples": "",
        "channel_candidate_count": "",
        "channel_candidate_names": "",
        "confirmed_flat_or_nonfinite_channel_count": "",
        "confirmed_flat_or_nonfinite_channels": "",
        "protocol_compatible": False,
        "technical_category": "technically incompatible",
        "failure_reason": "",
    }
    try:
        (downloaded_file,) = eegbci.load_data(
            subjects=subject,
            runs=run,
            path=data_directory,
            update_path=False,
        )
        source_path = Path(downloaded_file).resolve()
        base["file_exists"] = source_path.is_file()
        base["source_file"] = str(source_path)
        base["source_sha256"] = file_sha256(source_path)
        base["edf_format"] = source_path.suffix.lower().lstrip(".")
        raw = mne.io.read_raw_edf(source_path, preload=False, verbose="error")
        raw_names_before = tuple(raw.ch_names)
        eegbci.standardize(raw)
        standardized_names = tuple(raw.ch_names)
        data = raw.get_data()
        valid_stop, tail_samples, earlier_zeros = find_valid_stop(data)
        valid_data = data[:, :valid_stop]
        channel_candidates, confirmed_failures = channel_quality_candidates(valid_data)
        descriptions = [str(value) for value in raw.annotations.description]
        counts = Counter(descriptions)
        alternates = all(
            description == "T0" if index % 2 == 0 else description in {"T1", "T2"}
            for index, description in enumerate(descriptions)
        )
        sfreq = float(raw.info["sfreq"])
        eligibility = config.technical_eligibility
        failures: list[str] = []
        if base["edf_format"] != "edf":
            failures.append("unexpected_file_format")
        if not np.isclose(sfreq, eligibility.required_sampling_frequency_hz):
            failures.append("unexpected_sampling_frequency")
        if len(raw.ch_names) != eligibility.required_channel_count:
            failures.append("unexpected_channel_count")
        if standardized_names != EXPECTED_EEGBCI_CHANNELS:
            failures.append("incompatible_standardized_channel_names")
        if set(counts) != {"T0", "T1", "T2"}:
            failures.append("unexpected_annotation_descriptions")
        if (
            len(descriptions) != eligibility.required_annotation_count_per_run
            or counts["T0"] != eligibility.required_rest_annotation_count_per_run
            or counts["T1"] + counts["T2"]
            != eligibility.required_total_task_annotation_count_per_run
            or counts["T1"] not in eligibility.allowed_T1_counts_per_run
            or counts["T2"] not in eligibility.allowed_T2_counts_per_run
        ):
            failures.append("unexpected_annotation_counts")
        if not alternates:
            failures.append("unexpected_annotation_alternation")
        if earlier_zeros:
            failures.append("all_channel_zero_samples_before_tail")
        if valid_stop < 2:
            failures.append("insufficient_valid_samples")
        if confirmed_failures:
            failures.append("confirmed_flat_or_nonfinite_channel")
        standardized_changed = raw_names_before != standardized_names
        category = (
            "adaptable_without_scientific_retuning"
            if not failures and standardized_changed
            else "eligible"
            if not failures
            else "technically_incompatible"
        )
        base.update(
            {
                "load_success": True,
                "sampling_frequency_hz": sfreq,
                "channel_count": len(raw.ch_names),
                "channel_names_compatible": standardized_names == EXPECTED_EEGBCI_CHANNELS,
                "annotation_count": len(descriptions),
                "T0_count": counts["T0"],
                "T1_count": counts["T1"],
                "T2_count": counts["T2"],
                "annotation_descriptions_compatible": set(counts) == {"T0", "T1", "T2"},
                "annotation_alternation_compatible": alternates,
                "duration_seconds": raw.n_times / sfreq,
                "original_samples": raw.n_times,
                "valid_samples": valid_stop,
                "trailing_zero_samples": tail_samples,
                "earlier_all_channel_zero_samples": earlier_zeros,
                "channel_candidate_count": len(channel_candidates),
                "channel_candidate_names": "/".join(channel_candidates),
                "confirmed_flat_or_nonfinite_channel_count": len(confirmed_failures),
                "confirmed_flat_or_nonfinite_channels": "/".join(confirmed_failures),
                "protocol_compatible": not failures,
                "technical_category": category,
                "failure_reason": ";".join(failures),
            }
        )
    except Exception as error:  # audit must record rather than silently drop failures
        base["failure_reason"] = f"{type(error).__name__}: {error}"
    return base


def audit_cohort(
    config: ReplicationCohortConfig,
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
) -> list[dict[str, object]]:
    """Audit every predefined subject/run, retaining failures as explicit rows."""
    return [
        audit_recording(subject, run, config, data_directory=data_directory)
        for subject in config.all_subjects
        for run in config.runs
    ]


def subject_eligibility_rows(
    audit_rows: Sequence[Mapping[str, object]],
    config: ReplicationCohortConfig,
) -> list[dict[str, object]]:
    """Collapse run audits into deterministic subject eligibility decisions."""
    output: list[dict[str, object]] = []
    for subject in config.all_subjects:
        rows = [row for row in audit_rows if int(row["subject"]) == subject]
        compatible_runs = [int(row["run"]) for row in rows if row["protocol_compatible"]]
        failure_details = [
            f"run{row['run']}:{row['failure_reason']}"
            for row in rows
            if not row["protocol_compatible"]
        ]
        fully_eligible = set(compatible_runs) == set(config.runs)
        output.append(
            {
                "subject": subject,
                "subject_role": "development" if subject == config.development_subject else "replication",
                "requested_runs": "/".join(str(run) for run in config.runs),
                "compatible_runs": "/".join(str(run) for run in compatible_runs),
                "compatible_run_count": len(compatible_runs),
                "technical_eligibility": "eligible" if fully_eligible else "technically_incompatible",
                "primary_replication_included": bool(
                    fully_eligible and subject in config.replication_subjects
                ),
                "development_summary_included": bool(
                    fully_eligible and subject == config.development_subject
                ),
                "outcome_used_for_eligibility": False,
                "failure_reason": ";".join(failure_details),
            }
        )
    return output


def summarize_subject_features(
    trial_rows: Sequence[Mapping[str, object]],
    subject: int,
    subject_role: str,
    features: Sequence[tuple[str, str, str]],
    trim_fraction: float,
) -> list[dict[str, object]]:
    """Return one row per subject/feature; trials never become group units."""
    output: list[dict[str, object]] = []
    for condition, channel, band in features:
        selected = [
            row
            for row in trial_rows
            if row["semantic_condition"] == condition
            and row["channel"] == channel
            and row["frequency_band"] == band
        ]
        values = np.array([float(row["change_percent"]) for row in selected])
        db_values = np.array([float(row["change_db"]) for row in selected])
        candidate = np.array(
            [row["quality_status"] == "statistical candidate" for row in selected]
        )
        if values.size == 0:
            raise RuntimeError(f"Missing subject feature: {(subject, condition, channel, band)}.")
        stats = descriptive_statistics(values)
        q25_db, median_db, q75_db = np.percentile(db_values, [25, 50, 75])
        noncandidate_median = float(np.median(values[~candidate]))
        output.append(
            {
                "subject": subject,
                "subject_role": subject_role,
                "semantic_condition": condition,
                "channel": channel,
                "frequency_band": band,
                **stats,
                "trimmed_mean_change_percent": trimmed_mean(values, trim_fraction),
                "median_change_db": float(median_db),
                "iqr_change_db": float(q75_db - q25_db),
                "negative_trial_count": int(np.sum(values < 0)),
                "negative_trial_fraction": float(np.mean(values < 0)),
                "positive_trial_count": int(np.sum(values > 0)),
                "zero_trial_count": int(np.sum(values == 0)),
                "qc_candidate_trial_count": int(candidate.sum()),
                "exclude_qc_candidates_median_change_percent": noncandidate_median,
                "qc_sensitivity_preserves_primary_direction": bool(
                    np.sign(noncandidate_median) == np.sign(np.median(values))
                ),
                "subject_is_group_observational_unit": True,
            }
        )
    return output


def summarize_subject_runs(
    trial_rows: Sequence[Mapping[str, object]],
    subject: int,
    subject_role: str,
    features: Sequence[tuple[str, str, str]],
) -> list[dict[str, object]]:
    """Return run-specific subject summaries with trial counts and direction."""
    output: list[dict[str, object]] = []
    for condition, channel, band in features:
        for run in sorted({int(row["run"]) for row in trial_rows}):
            selected = [
                row
                for row in trial_rows
                if int(row["run"]) == run
                and row["semantic_condition"] == condition
                and row["channel"] == channel
                and row["frequency_band"] == band
            ]
            if not selected:
                continue
            values = np.array([float(row["change_percent"]) for row in selected])
            median = float(np.median(values))
            q25, q75 = np.percentile(values, [25, 75])
            output.append(
                {
                    "subject": subject,
                    "subject_role": subject_role,
                    "run": run,
                    "semantic_condition": condition,
                    "channel": channel,
                    "frequency_band": band,
                    "trial_count": values.size,
                    "median_change_percent": median,
                    "iqr_change_percentage_points": float(q75 - q25),
                    "negative_trial_count": int(np.sum(values < 0)),
                    "negative_trial_fraction": float(np.mean(values < 0)),
                    "run_direction": "negative" if median < 0 else "positive" if median > 0 else "zero",
                }
            )
    return output


def run_direction_category(run_medians: Sequence[float], predicted_direction: str) -> str:
    """Classify within-subject support as 3/3, 2/3, 1/3, or 0/3."""
    values = np.asarray(run_medians, dtype=float)
    if values.size != 3 or not np.isfinite(values).all():
        raise ValueError("Run consistency requires exactly three finite run medians.")
    supporting = int(np.sum(values < 0) if predicted_direction == "negative" else np.sum(values > 0))
    return f"{supporting}/3_{predicted_direction}"


def build_primary_replication_rows(
    subject_rows: Sequence[Mapping[str, object]],
    run_rows: Sequence[Mapping[str, object]],
    config: ReplicationCohortConfig,
) -> list[dict[str, object]]:
    """Attach three-run consistency to each subject-level primary measurement."""
    output: list[dict[str, object]] = []
    for subject_row in subject_rows:
        feature = (
            str(subject_row["semantic_condition"]),
            str(subject_row["channel"]),
            str(subject_row["frequency_band"]),
        )
        if feature not in config.primary_replication_features:
            continue
        selected_runs = sorted(
            [
                row
                for row in run_rows
                if int(row["subject"]) == int(subject_row["subject"])
                and (
                    str(row["semantic_condition"]),
                    str(row["channel"]),
                    str(row["frequency_band"]),
                ) == feature
            ],
            key=lambda row: int(row["run"]),
        )
        run_medians = [float(row["median_change_percent"]) for row in selected_runs]
        if len(run_medians) != 3:
            raise RuntimeError(f"Primary feature lacks three runs: {subject_row}.")
        output.append(
            {
                **dict(subject_row),
                "run_6_median_change_percent": run_medians[0],
                "run_10_median_change_percent": run_medians[1],
                "run_14_median_change_percent": run_medians[2],
                "run_direction_consistency": run_direction_category(
                    run_medians, config.replication_success_criteria.predicted_direction
                ),
                "supporting_run_count": int(sum(value < 0 for value in run_medians)),
            }
        )
    return output


def exact_sign_test_one_sided(
    negative_count: int,
    nonzero_count: int,
    null_probability: float = 0.5,
) -> float:
    """Exact P(K>=observed) for a prespecified negative-direction sign test."""
    if not 0 <= negative_count <= nonzero_count or nonzero_count < 1:
        raise ValueError("Sign-test counts are invalid.")
    if not 0 < null_probability < 1:
        raise ValueError("Null probability must lie strictly between zero and one.")
    probability = sum(
        math.comb(nonzero_count, count)
        * null_probability**count
        * (1 - null_probability) ** (nonzero_count - count)
        for count in range(negative_count, nonzero_count + 1)
    )
    return float(min(1.0, probability))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a directional subject fraction."""
    if not 0 <= successes <= total or total < 1:
        raise ValueError("Wilson interval counts are invalid.")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return float(center - half_width), float(center + half_width)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm inputs must be p-values.")
    order = np.argsort(values, kind="mergesort")
    adjusted_sorted = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (values.size - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(values.size, dtype=float)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()


def bootstrap_median_interval(
    values: np.ndarray, resamples: int, seed: int
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval across subjects."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must contain finite subject values.")
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(resamples, values.size))
    medians = np.median(values[indices], axis=1)
    lower, upper = np.percentile(medians, [2.5, 97.5])
    return float(lower), float(upper)


def replication_category(
    subject_fraction: float,
    cohort_median: float,
    fraction_with_two_supporting_runs: float,
    qc_direction_preserved: bool,
    leave_one_out_direction_preserved: bool,
    criteria: ReplicationSuccessCriteria,
) -> tuple[str, dict[str, bool]]:
    """Apply strong/mixed/failed rules frozen before cohort outcomes."""
    median_supports = cohort_median < 0 if criteria.predicted_direction == "negative" else cohort_median > 0
    passed = {
        "subject_fraction": subject_fraction
        >= criteria.strong_minimum_subject_fraction_in_direction,
        "cohort_median": median_supports,
        "run_support": fraction_with_two_supporting_runs
        >= criteria.strong_minimum_fraction_with_at_least_two_of_three_runs_in_direction,
        "qc_sensitivity": qc_direction_preserved,
        "leave_one_subject_out": leave_one_out_direction_preserved,
    }
    if all(passed.values()):
        category = "strong replication"
    elif (
        subject_fraction >= criteria.mixed_minimum_subject_fraction_in_direction
        and median_supports
    ):
        category = "mixed replication"
    else:
        category = "failed replication"
    return category, passed


def cohort_feature_summaries(
    subject_rows: Sequence[Mapping[str, object]],
    primary_rows: Sequence[Mapping[str, object]],
    config: ReplicationCohortConfig,
    included_subject_ids: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    """Aggregate one value per replication subject for each frozen feature."""
    included = (
        set(config.replication_subjects)
        if included_subject_ids is None
        else {int(subject) for subject in included_subject_ids}
    )
    if not included:
        raise ValueError("At least one replication subject must be included.")
    output: list[dict[str, object]] = []
    raw_primary_p_values: list[float] = []
    primary_output_indices: list[int] = []
    for feature_index, (condition, channel, band) in enumerate(config.all_features):
        selected = [
            row
            for row in subject_rows
            if int(row["subject"]) in included
            and row["semantic_condition"] == condition
            and row["channel"] == channel
            and row["frequency_band"] == band
        ]
        values = np.array([float(row["median_change_percent"]) for row in selected])
        qc_values = np.array(
            [float(row["exclude_qc_candidates_median_change_percent"]) for row in selected]
        )
        contributing_subjects = {int(row["subject"]) for row in selected}
        if (
            values.size != len(contributing_subjects)
            or values.size == 0
            or contributing_subjects != included
        ):
            raise RuntimeError(
                "Feature must contain exactly one row per technically eligible "
                f"replication subject: {(condition, channel, band)}."
            )
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        negative = int(np.sum(values < 0))
        positive = int(np.sum(values > 0))
        zero = int(np.sum(values == 0))
        nonzero = negative + positive
        lower_fraction, upper_fraction = wilson_interval(negative, nonzero)
        bootstrap_lower, bootstrap_upper = bootstrap_median_interval(
            values,
            config.group_inference.bootstrap_subject_median_resamples,
            config.group_inference.bootstrap_seed + feature_index,
        )
        primary_selected = [
            row
            for row in primary_rows
            if int(row["subject"]) in included
            and row["semantic_condition"] == condition
            and row["channel"] == channel
            and row["frequency_band"] == band
        ]
        run_support_fraction = (
            float(np.mean([int(row["supporting_run_count"]) >= 2 for row in primary_selected]))
            if primary_selected
            else float("nan")
        )
        leave_one_medians = np.array(
            [float(np.median(np.delete(values, index))) for index in range(values.size)]
        )
        qc_median = float(np.median(qc_values))
        qc_direction_preserved = np.sign(qc_median) == np.sign(median)
        leave_one_preserved = bool(np.all(np.sign(leave_one_medians) == np.sign(median)))
        is_primary = (condition, channel, band) in config.primary_replication_features
        if is_primary:
            category, criteria_passed = replication_category(
                negative / nonzero,
                float(median),
                run_support_fraction,
                qc_direction_preserved,
                leave_one_preserved,
                config.replication_success_criteria,
            )
            raw_p = exact_sign_test_one_sided(
                negative,
                nonzero,
                config.group_inference.null_negative_direction_probability,
            )
        else:
            category = "secondary / exploratory; not replication-graded"
            criteria_passed = {}
            raw_p = float("nan")
        row: dict[str, object] = {
            "semantic_condition": condition,
            "channel": channel,
            "frequency_band": band,
            "analysis_role": "primary" if is_primary else "secondary_or_exploratory",
            "replication_subject_count": values.size,
            "negative_subject_count": negative,
            "positive_subject_count": positive,
            "zero_subject_count": zero,
            "negative_subject_fraction": negative / nonzero,
            "negative_fraction_wilson_95_lower": lower_fraction,
            "negative_fraction_wilson_95_upper": upper_fraction,
            "median_across_subject_medians_percent": float(median),
            "q25_across_subject_medians_percent": float(q25),
            "q75_across_subject_medians_percent": float(q75),
            "iqr_across_subject_medians_percentage_points": float(q75 - q25),
            "minimum_subject_median_percent": float(np.min(values)),
            "maximum_subject_median_percent": float(np.max(values)),
            "mad_across_subject_medians_percent": float(
                np.median(np.abs(values - median))
            ),
            "bootstrap_median_95_lower_percent": bootstrap_lower,
            "bootstrap_median_95_upper_percent": bootstrap_upper,
            "qc_excluded_cohort_median_percent": qc_median,
            "qc_sensitivity_direction_preserved": qc_direction_preserved,
            "fraction_subjects_with_at_least_two_supporting_runs": run_support_fraction if is_primary else "",
            "leave_one_subject_out_minimum_cohort_median_percent": float(np.min(leave_one_medians)),
            "leave_one_subject_out_maximum_cohort_median_percent": float(np.max(leave_one_medians)),
            "leave_one_subject_out_direction_preserved": leave_one_preserved,
            "one_sided_exact_sign_test_p": raw_p if is_primary else "",
            "holm_adjusted_primary_p": "",
            "replication_category": category,
            **{f"strong_criterion_{name}": value for name, value in criteria_passed.items()},
        }
        output.append(row)
        if is_primary:
            raw_primary_p_values.append(raw_p)
            primary_output_indices.append(len(output) - 1)
    adjusted = holm_adjust(raw_primary_p_values)
    for index, adjusted_p in zip(primary_output_indices, adjusted, strict=True):
        output[index]["holm_adjusted_primary_p"] = adjusted_p
    return output


def central_peak_summary(
    subject: int,
    subject_role: str,
    referenced_run_data: Sequence[tuple[int, np.ndarray, Sequence[str], float]],
    settings: CentralPeakSettings,
) -> dict[str, object]:
    """Measure a fixed-grid central 8-13 Hz peak without changing replication bands."""
    run_psds: list[np.ndarray] = []
    frequency_vector: np.ndarray | None = None
    for _, data, channel_names, sfreq in referenced_run_data:
        indices = [channel_names.index(channel) for channel in settings.channels]
        n_per_segment = int(round(settings.welch_window_seconds * sfreq))
        n_overlap = int(round(n_per_segment * settings.welch_overlap_fraction))
        psd, frequencies = psd_array_welch(
            data[indices],
            sfreq,
            fmin=min(settings.integer_frequency_centers_hz),
            fmax=max(settings.integer_frequency_centers_hz),
            n_fft=n_per_segment,
            n_per_seg=n_per_segment,
            n_overlap=n_overlap,
            average="mean",
            window="hann",
            remove_dc=True,
            verbose="error",
        )
        run_psds.append(np.mean(psd, axis=0))
        if frequency_vector is None:
            frequency_vector = frequencies
        elif not np.array_equal(frequency_vector, frequencies):
            raise RuntimeError("Central PSD frequency grids differ between runs.")
    assert frequency_vector is not None
    integer_power = np.array(
        [
            np.mean(
                [
                    run_psd[int(np.argmin(np.abs(frequency_vector - frequency)))]
                    for run_psd in run_psds
                ]
            )
            for frequency in settings.integer_frequency_centers_hz
        ]
    )
    peak_index = int(np.argmax(integer_power))
    peak_frequency = settings.integer_frequency_centers_hz[peak_index]
    band_median = float(np.median(integer_power))
    prominence_db = float(db_power_ratio(integer_power[peak_index], band_median))
    well_defined = prominence_db >= settings.minimum_peak_prominence_db_over_band_median
    return {
        "subject": subject,
        "subject_role": subject_role,
        "central_channels": "/".join(settings.channels),
        "peak_frequency_hz": peak_frequency if well_defined else "",
        "peak_category": str(peak_frequency) if well_defined else "poorly_defined",
        "peak_prominence_db_over_8_13_band_median": prominence_db,
        "well_defined_peak": well_defined,
        **{
            f"power_at_{frequency}_hz_v2_per_hz": float(integer_power[index])
            for index, frequency in enumerate(settings.integer_frequency_centers_hz)
        },
        "primary_replication_band_redefined": False,
    }


def frozen_config_fingerprint(
    config: ReplicationCohortConfig,
    cohort_config_path: Path = DEFAULT_REPLICATION_CONFIG_PATH,
) -> str:
    """Hash cohort JSON plus referenced configuration hashes for cache provenance."""
    digest = hashlib.sha256()
    cohort_bytes = cohort_config_path.expanduser().resolve().read_bytes()
    digest.update(cohort_bytes)
    for name, value in sorted(configuration_hashes(config).items()):
        digest.update(name.encode())
        digest.update(value.encode())
    return digest.hexdigest()
