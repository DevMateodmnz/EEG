"""Held-out group summaries for the final-frozen within-subject decoders."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .decoding import (
    PROJECT_ROOT,
    RUNS,
    TARGET_LABELS,
    SubjectDecodingData,
    TrialIdentity,
    _filter_continuous_csp_copy,
    bootstrap_median_interval,
    exact_sign_test_above_chance,
    load_decoding_config,
    paired_wilcoxon,
)
from .epoching import calculate_trial_quality, extract_run_epochs, load_epoching_config
from .preprocessing import DEFAULT_DATA_DIRECTORY, load_preprocessing_config


DEFAULT_FINAL_DECODING_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "within_subject_decoding_final.json"
)
DEFAULT_ACTIVE_DECODING_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "within_subject_decoding_correction.json"
)


def file_sha256(path: Path) -> str:
    """Return a compact file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_final_decoding_config(
    path: Path = DEFAULT_ACTIVE_DECODING_CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the final freeze and its exact immutable initial policy."""
    active_path = path.expanduser().resolve()
    active: dict[str, Any] = json.loads(active_path.read_text(encoding="utf-8"))
    if active.get("study_stage") == "technical_validation_correction_frozen":
        parent_record = active["parent_final_policy"]
        final_path = PROJECT_ROOT / parent_record["path"]
        if file_sha256(final_path) != parent_record["sha256"]:
            raise RuntimeError("Parent final decoder policy changed after correction freeze.")
        correction_record = active["corrected_evaluation_implementation"]
        correction_path = PROJECT_ROOT / correction_record["path"]
        if file_sha256(correction_path) != correction_record["sha256"]:
            raise RuntimeError("Corrected evaluation implementation drifted after freeze.")
        if active["model_or_metric_setting_changed"]:
            raise RuntimeError("Technical correction must not change a model or metric setting.")
        final: dict[str, Any] = json.loads(final_path.read_text(encoding="utf-8"))
        final["active_technical_correction"] = active
    else:
        final_path = active_path
        final = active
    if final["study_stage"] != "final_decoder_method_frozen":
        raise RuntimeError("Evaluation requires the final decoder-method freeze.")
    if final["evaluation_classifier_outcomes_inspected_at_final_freeze"]:
        raise RuntimeError("Final decoder chronology says evaluation outcomes were already open.")
    if not final["evaluation_unlocked_after_this_freeze_commit"]:
        raise RuntimeError("Final decoder configuration does not unlock evaluation.")
    initial_record = final["initial_policy"]
    initial_path = PROJECT_ROOT / initial_record["path"]
    if file_sha256(initial_path) != initial_record["sha256"]:
        raise RuntimeError("Initial decoder policy changed after its freeze.")
    initial = load_decoding_config(initial_path)
    for record in final["development_evidence"]["artifacts"].values():
        artifact = PROJECT_ROOT / record["path"]
        if file_sha256(artifact) != record["sha256"]:
            raise RuntimeError(f"Decoder development artifact drift: {artifact}.")
    for record in final["frozen_method_implementation"].values():
        implementation = PROJECT_ROOT / record["path"]
        if file_sha256(implementation) != record["sha256"]:
            raise RuntimeError(f"Frozen decoder implementation drift: {implementation}.")
    if final["cohorts"] != {
        key: initial["cohorts"][key]
        for key in (
            "decoder_development_subjects",
            "decoder_evaluation_requested_subjects",
            "decoder_evaluation_eligible_subjects",
            "technically_incompatible_subjects",
        )
    }:
        raise RuntimeError("Final and initial decoder cohorts differ.")
    if final["selected_spectral_baseline"]["candidate_name"] != "spectral_baseline_6":
        raise RuntimeError("Unexpected final spectral baseline.")
    if final["selected_csp_decoder"]["candidate_name"] != "csp_4_empirical":
        raise RuntimeError("Unexpected final CSP decoder.")
    return final, initial


def load_evaluation_subject_data(
    subject: int,
    config: Mapping[str, Any],
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
) -> SubjectDecodingData:
    """Construct evaluation data while accepting frozen 15- or 14-trial runs.

    The frozen core's transform is reproduced exactly. The only correction is the
    validation of previously established deterministic boundary exclusions: a run
    may retain balanced 7/7 classes instead of the ordinary 7/8 or 8/7 pattern.
    """
    allowed = set(config["cohorts"]["decoder_evaluation_eligible_subjects"])
    if subject not in allowed:
        raise ValueError(f"Subject {subject} is outside the eligible evaluation cohort.")
    preprocessing = load_preprocessing_config()
    epoching = load_epoching_config()
    interval = tuple(float(value) for value in config["prediction_target"]["task_interval_seconds"])
    csp_filter = config["fixed_feature_methods"]["csp_feature_filter"]
    fixed_arrays: list[np.ndarray] = []
    csp_arrays: list[np.ndarray] = []
    labels: list[int] = []
    runs: list[int] = []
    identities: list[TrialIdentity] = []
    qc_candidates: list[bool] = []
    channel_names: list[str] | None = None
    sampling_frequency: float | None = None
    for run in RUNS:
        result = extract_run_epochs(
            subject, run, preprocessing, epoching, data_directory=data_directory
        )
        epochs = result.task_epochs
        sfreq = float(epochs.info["sfreq"])
        if sampling_frequency is None:
            sampling_frequency = sfreq
            channel_names = list(epochs.ch_names)
        elif not np.isclose(sampling_frequency, sfreq) or channel_names != list(epochs.ch_names):
            raise RuntimeError("Evaluation run sampling frequency or channel order drifted.")
        valid_records = [record for record in result.task_records if record.valid]
        full_epoch_data = epochs.get_data(copy=True)
        if full_epoch_data.shape[0] != len(valid_records):
            raise RuntimeError("Evaluation task epochs and valid records differ.")
        start_index = int(round((interval[0] - float(epochs.times[0])) * sfreq))
        sample_count = int(round((interval[1] - interval[0]) * sfreq))
        selected_fixed = full_epoch_data[:, :, start_index : start_index + sample_count]
        if selected_fixed.shape[2] != 320:
            raise RuntimeError("Evaluation interval must contain 320 samples.")
        feature_raw = _filter_continuous_csp_copy(result.preprocessing.filtered, csp_filter)
        run_csp: list[np.ndarray] = []
        for record in valid_records:
            start_sample = record.event_sample + int(round(interval[0] * sfreq))
            trial = feature_raw.get_data(start=start_sample, stop=start_sample + sample_count)
            if trial.shape != (len(feature_raw.ch_names), sample_count):
                raise RuntimeError("Evaluation CSP interval has the wrong shape.")
            run_csp.append(trial)
        selected_csp = np.stack(run_csp)
        quality_rows = calculate_trial_quality(
            full_epoch_data,
            epochs.times.copy(),
            epochs.ch_names,
            valid_records,
            epoching.trial_quality,
        )
        for record, quality_row in zip(valid_records, quality_rows, strict=True):
            if record.semantic_condition not in TARGET_LABELS:
                raise RuntimeError("A non-target condition entered evaluation data.")
            labels.append(TARGET_LABELS[record.semantic_condition])
            runs.append(run)
            identities.append(
                TrialIdentity(
                    subject,
                    run,
                    record.run_trial_index,
                    record.annotation_index,
                    record.semantic_condition,
                    str(result.preprocessing.source_path),
                    result.preprocessing.source_sha256,
                )
            )
            qc_candidates.append(quality_row["quality_status"] == "statistical candidate")
        fixed_arrays.append(selected_fixed)
        csp_arrays.append(selected_csp)
    assert channel_names is not None and sampling_frequency is not None
    dataset = SubjectDecodingData(
        subject,
        np.concatenate(fixed_arrays),
        np.concatenate(csp_arrays),
        np.asarray(labels, dtype=int),
        np.asarray(runs, dtype=int),
        identities,
        np.asarray(qc_candidates, dtype=bool),
        channel_names,
        sampling_frequency,
        interval[0],
        interval[1],
    )
    count = dataset.labels.size
    if (
        dataset.fixed_task_data_volts.shape != dataset.csp_task_data_volts.shape
        or dataset.fixed_task_data_volts.shape[0] != count
        or len(dataset.identities) != count
        or dataset.qc_candidates.size != count
        or not np.isfinite(dataset.fixed_task_data_volts).all()
        or not np.isfinite(dataset.csp_task_data_volts).all()
        or len({identity.key for identity in dataset.identities}) != count
    ):
        raise RuntimeError("Corrected evaluation arrays or identities are invalid.")
    for run in RUNS:
        selected = dataset.labels[dataset.runs == run]
        class_counts = sorted(Counter(selected).values())
        if class_counts not in ([7, 7], [7, 8]):
            raise RuntimeError(
                f"Run {run} retained unsupported class counts {class_counts}; "
                "the correction permits only historical 7/7 or 7/8 patterns."
            )
    return dataset


def paired_model_rows(
    subject_rows: Sequence[Mapping[str, object]],
    baseline_name: str,
    csp_name: str,
) -> list[dict[str, object]]:
    """Return one exact primary/QC paired model row per evaluation participant."""
    lookup = {
        (int(row["subject"]), str(row["model"]), bool(row["qc_sensitivity"])): row
        for row in subject_rows
    }
    subjects = sorted({int(row["subject"]) for row in subject_rows})
    output: list[dict[str, object]] = []
    for subject in subjects:
        baseline = lookup[(subject, baseline_name, False)]
        csp = lookup[(subject, csp_name, False)]
        baseline_qc = lookup[(subject, baseline_name, True)]
        csp_qc = lookup[(subject, csp_name, True)]
        output.append(
            {
                "subject": subject,
                "baseline_model": baseline_name,
                "csp_model": csp_name,
                "baseline_balanced_accuracy": baseline["primary_mean_fold_balanced_accuracy"],
                "csp_balanced_accuracy": csp["primary_mean_fold_balanced_accuracy"],
                "csp_minus_baseline_balanced_accuracy": float(
                    csp["primary_mean_fold_balanced_accuracy"]
                )
                - float(baseline["primary_mean_fold_balanced_accuracy"]),
                "baseline_qc_balanced_accuracy": baseline_qc[
                    "primary_mean_fold_balanced_accuracy"
                ],
                "csp_qc_balanced_accuracy": csp_qc[
                    "primary_mean_fold_balanced_accuracy"
                ],
                "qc_csp_minus_baseline_balanced_accuracy": float(
                    csp_qc["primary_mean_fold_balanced_accuracy"]
                )
                - float(baseline_qc["primary_mean_fold_balanced_accuracy"]),
                "csp_folds_above_chance_count": csp["folds_above_chance_count"],
                "csp_fold_range": csp["fold_balanced_accuracy_range"],
                "csp_fists_recall": csp["out_of_fold_fists_recall"],
                "csp_feet_recall": csp["out_of_fold_feet_recall"],
                "qc_candidate_count": csp["qc_candidate_count"],
                "matched_subject_and_trials": True,
            }
        )
    return output


def model_comparison_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize paired CSP minus spectral performance across participants."""
    differences = np.array(
        [float(row["csp_minus_baseline_balanced_accuracy"]) for row in rows]
    )
    qc_differences = np.array(
        [float(row["qc_csp_minus_baseline_balanced_accuracy"]) for row in rows]
    )
    baseline = np.array([float(row["baseline_balanced_accuracy"]) for row in rows])
    csp = np.array([float(row["csp_balanced_accuracy"]) for row in rows])
    return {
        "matched_subject_count": len(rows),
        "median_csp_minus_baseline_balanced_accuracy": float(np.median(differences)),
        "q25_csp_minus_baseline_balanced_accuracy": float(np.percentile(differences, 25)),
        "q75_csp_minus_baseline_balanced_accuracy": float(np.percentile(differences, 75)),
        "minimum_csp_minus_baseline_balanced_accuracy": float(np.min(differences)),
        "maximum_csp_minus_baseline_balanced_accuracy": float(np.max(differences)),
        "csp_better_count": int(np.sum(differences > 0)),
        "baseline_better_count": int(np.sum(differences < 0)),
        "tie_count": int(np.sum(differences == 0)),
        "fraction_csp_better": float(np.mean(differences > 0)),
        "qc_median_csp_minus_baseline_balanced_accuracy": float(np.median(qc_differences)),
        "two_sided_paired_wilcoxon_p": paired_wilcoxon(csp, baseline),
        "group_unit": "participant",
    }


def summarize_model_group(
    subject_rows: Sequence[Mapping[str, object]],
    model: str,
    final: Mapping[str, Any],
) -> dict[str, object]:
    """Describe one frozen model with one primary score per participant."""
    primary = sorted(
        [row for row in subject_rows if row["model"] == model and not row["qc_sensitivity"]],
        key=lambda row: int(row["subject"]),
    )
    sensitivity = sorted(
        [row for row in subject_rows if row["model"] == model and row["qc_sensitivity"]],
        key=lambda row: int(row["subject"]),
    )
    if [row["subject"] for row in primary] != [row["subject"] for row in sensitivity]:
        raise RuntimeError("Primary and QC model subjects differ.")
    scores = np.array([float(row["primary_mean_fold_balanced_accuracy"]) for row in primary])
    qc_scores = np.array(
        [float(row["primary_mean_fold_balanced_accuracy"]) for row in sensitivity]
    )
    fists = np.array([float(row["out_of_fold_fists_recall"]) for row in primary])
    feet = np.array([float(row["out_of_fold_feet_recall"]) for row in primary])
    loo = np.array(
        [float(np.median(np.delete(scores, index))) for index in range(scores.size)]
    )
    bootstrap = final["group_inference"]["bootstrap_median_confidence_interval"]
    ci_low, ci_high = bootstrap_median_interval(
        scores,
        resamples=int(bootstrap["resamples"]),
        confidence_level=float(bootstrap["confidence_level"]),
        random_seed=int(bootstrap["random_seed"]),
    )
    sign = exact_sign_test_above_chance(scores)
    return {
        "model": model,
        "evaluation_subject_count": len(primary),
        "median_subject_balanced_accuracy": float(np.median(scores)),
        "q25_subject_balanced_accuracy": float(np.percentile(scores, 25)),
        "q75_subject_balanced_accuracy": float(np.percentile(scores, 75)),
        "minimum_subject_balanced_accuracy": float(np.min(scores)),
        "maximum_subject_balanced_accuracy": float(np.max(scores)),
        "bootstrap_median_95_lower": ci_low,
        "bootstrap_median_95_upper": ci_high,
        "fraction_subjects_above_0_5": float(np.mean(scores > 0.5)),
        "fraction_subjects_at_or_above_0_6": float(np.mean(scores >= 0.6)),
        "fraction_subjects_at_or_above_0_7": float(np.mean(scores >= 0.7)),
        "fraction_subjects_with_at_least_two_folds_above_0_5": float(
            np.mean([int(row["folds_above_chance_count"]) >= 2 for row in primary])
        ),
        "median_fold_balanced_accuracy_range": float(
            np.median([float(row["fold_balanced_accuracy_range"]) for row in primary])
        ),
        "median_fists_recall": float(np.median(fists)),
        "median_feet_recall": float(np.median(feet)),
        "absolute_median_class_recall_difference": float(
            abs(np.median(fists) - np.median(feet))
        ),
        "qc_cohort_median_balanced_accuracy": float(np.median(qc_scores)),
        "qc_cohort_median_minus_primary_cohort_median": float(
            np.median(qc_scores) - np.median(scores)
        ),
        "median_paired_qc_minus_primary_balanced_accuracy": float(
            np.median(qc_scores - scores)
        ),
        "leave_one_subject_out_minimum_cohort_median": float(np.min(loo)),
        "leave_one_subject_out_maximum_cohort_median": float(np.max(loo)),
        **sign,
        "group_unit": "participant",
    }


def classify_decoding_signal(
    csp_summary: Mapping[str, object],
    paired_median_difference: float,
    final: Mapping[str, Any],
) -> str:
    """Apply the final-frozen useful/mixed/no-convincing thresholds."""
    useful = final["evaluation_success_criteria"]["useful_decoding_signal"]
    if (
        float(csp_summary["median_subject_balanced_accuracy"]) >= useful["minimum_cohort_median_balanced_accuracy"]
        and float(csp_summary["fraction_subjects_above_0_5"]) >= useful["minimum_fraction_subjects_above_0_5"]
        and float(csp_summary["fraction_subjects_with_at_least_two_folds_above_0_5"]) >= useful["minimum_fraction_subjects_with_at_least_two_folds_above_0_5"]
        and float(csp_summary["median_fists_recall"]) >= useful["minimum_median_recall_each_class"]
        and float(csp_summary["median_feet_recall"]) >= useful["minimum_median_recall_each_class"]
        and float(csp_summary["absolute_median_class_recall_difference"]) <= useful["maximum_absolute_difference_between_median_class_recalls"]
        and float(csp_summary["qc_cohort_median_minus_primary_cohort_median"]) >= useful["minimum_qc_cohort_median_minus_primary_cohort_median"]
        and paired_median_difference >= useful["minimum_csp_minus_baseline_paired_median"]
        and float(csp_summary["leave_one_subject_out_minimum_cohort_median"]) >= useful["minimum_leave_one_subject_out_cohort_median"]
    ):
        return "useful decoding signal"
    mixed = final["evaluation_success_criteria"]["mixed_decoding_evidence"]
    if (
        float(csp_summary["median_subject_balanced_accuracy"]) >= mixed["minimum_cohort_median_balanced_accuracy"]
        and float(csp_summary["fraction_subjects_above_0_5"]) >= mixed["minimum_fraction_subjects_above_0_5"]
        and float(csp_summary["median_fists_recall"]) >= mixed["minimum_median_recall_each_class"]
        and float(csp_summary["median_feet_recall"]) >= mixed["minimum_median_recall_each_class"]
        and float(csp_summary["absolute_median_class_recall_difference"]) <= mixed["maximum_absolute_difference_between_median_class_recalls"]
        and float(csp_summary["qc_cohort_median_minus_primary_cohort_median"]) >= mixed["minimum_qc_cohort_median_minus_primary_cohort_median"]
    ):
        return "mixed decoding evidence"
    return "no convincing decoding"


def classify_csp_added_value(
    comparison: Mapping[str, object], final: Mapping[str, Any]
) -> str:
    """Apply frozen paired CSP-versus-baseline added-value thresholds."""
    clear = final["csp_added_value_criteria"]["clear_added_value"]
    if (
        float(comparison["median_csp_minus_baseline_balanced_accuracy"]) >= clear["minimum_paired_median_difference"]
        and float(comparison["fraction_csp_better"]) >= clear["minimum_fraction_csp_better"]
        and float(comparison["qc_median_csp_minus_baseline_balanced_accuracy"]) >= clear["minimum_qc_paired_median_difference"]
    ):
        return "clear added value"
    modest = final["csp_added_value_criteria"]["modest_added_value"]
    if (
        float(comparison["median_csp_minus_baseline_balanced_accuracy"]) > modest["minimum_paired_median_difference_exclusive"]
        and float(comparison["fraction_csp_better"]) > modest["minimum_fraction_csp_better_exclusive"]
    ):
        return "modest added value"
    return "no added value"


def aggregate_confusions(
    prediction_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate exact confusion counts by frozen model and QC set."""
    output: list[dict[str, object]] = []
    for model in sorted({str(row["model"]) for row in prediction_rows}):
        for qc in (False, True):
            rows = [
                row for row in prediction_rows
                if row["model"] == model and bool(row["qc_sensitivity"]) == qc
            ]
            counts = Counter(
                (str(row["true_condition"]), str(row["predicted_condition"])) for row in rows
            )
            ff = counts[("both_fists_imagery", "both_fists_imagery")]
            ft = counts[("both_fists_imagery", "both_feet_imagery")]
            tf = counts[("both_feet_imagery", "both_fists_imagery")]
            tt = counts[("both_feet_imagery", "both_feet_imagery")]
            if ff + ft + tf + tt != len(rows):
                raise RuntimeError("Aggregate confusion counts differ from predictions.")
            output.append(
                {
                    "model": model,
                    "qc_sensitivity": qc,
                    "true_fists_predicted_fists": ff,
                    "true_fists_predicted_feet": ft,
                    "true_feet_predicted_fists": tf,
                    "true_feet_predicted_feet": tt,
                    "fists_recall": ff / (ff + ft),
                    "feet_recall": tt / (tf + tt),
                    "trial_count": len(rows),
                }
            )
    return output


def error_analysis_rows(
    prediction_rows: Sequence[Mapping[str, object]],
    fold_rows: Sequence[Mapping[str, object]],
    subject_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return predefined post-evaluation descriptive error summaries."""
    output: list[dict[str, object]] = []
    for model in sorted({str(row["model"]) for row in prediction_rows}):
        primary_predictions = [
            row for row in prediction_rows
            if row["model"] == model and not row["qc_sensitivity"]
        ]
        primary_folds = [
            row for row in fold_rows if row["model"] == model and not row["qc_sensitivity"]
        ]
        for run in (6, 10, 14):
            scores = np.array(
                [float(row["balanced_accuracy"]) for row in primary_folds if int(row["test_run"]) == run]
            )
            output.append(
                {
                    "model": model,
                    "analysis": "held_out_run_distribution",
                    "group": f"run_{run}",
                    "n": len(scores),
                    "value": float(np.median(scores)),
                    "secondary_value": float(np.percentile(scores, 25)),
                    "tertiary_value": float(np.percentile(scores, 75)),
                    "units": "balanced_accuracy_median_q25_q75",
                }
            )
        for candidate in (False, True):
            rows = [row for row in primary_predictions if bool(row["qc_candidate"]) == candidate]
            output.append(
                {
                    "model": model,
                    "analysis": "prediction_accuracy_by_qc_status",
                    "group": "statistical_candidate" if candidate else "not_flagged",
                    "n": len(rows),
                    "value": float(np.mean([bool(row["correct"]) for row in rows])) if rows else "",
                    "secondary_value": "",
                    "tertiary_value": "",
                    "units": "ordinary_prediction_accuracy",
                }
            )
        grouped: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
        for row in primary_predictions:
            grouped[(int(row["subject"]), int(row["run"]))].append(row)
        for fists_count in (7, 8):
            scores = []
            for rows in grouped.values():
                if sum(row["true_condition"] == "both_fists_imagery" for row in rows) != fists_count:
                    continue
                fists_rows = [row for row in rows if row["true_condition"] == "both_fists_imagery"]
                feet_rows = [row for row in rows if row["true_condition"] == "both_feet_imagery"]
                scores.append(
                    (np.mean([row["correct"] for row in fists_rows]) + np.mean([row["correct"] for row in feet_rows])) / 2
                )
            output.append(
                {
                    "model": model,
                    "analysis": "trial_counterbalance",
                    "group": f"{fists_count}_fists_{15-fists_count}_feet",
                    "n": len(scores),
                    "value": float(np.median(scores)),
                    "secondary_value": "",
                    "tertiary_value": "",
                    "units": "fold_balanced_accuracy_median",
                }
            )
        selected_subjects = [
            row for row in subject_rows
            if row["model"] == model and not row["qc_sensitivity"]
            and float(row["primary_mean_fold_balanced_accuracy"]) <= 0.5
        ]
        driven = sum(
            int(row["folds_above_chance_count"]) >= 2 for row in selected_subjects
        )
        output.append(
            {
                "model": model,
                "analysis": "below_or_at_chance_subject_run_pattern",
                "group": "at_least_two_above_chance_folds_despite_subject_score_le_0_5",
                "n": len(selected_subjects),
                "value": driven,
                "secondary_value": len(selected_subjects) - driven,
                "tertiary_value": "",
                "units": "driven_by_one_low_run_count_vs_not_count",
            }
        )
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def exploratory_relationship_rows(
    csp_subject_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Join frozen historical physiology only after decoder scores are finalized."""
    score_lookup = {
        int(row["subject"]): float(row["primary_mean_fold_balanced_accuracy"])
        for row in csp_subject_rows
        if not row["qc_sensitivity"]
    }
    primary = {
        int(row["subject"]): row
        for row in _read_csv(PROJECT_ROOT / "docs/assets/subjects01-109_full_replication_subject_primary_table.csv")
        if row["technical_eligibility"] == "eligible"
    }
    peaks = {
        int(row["subject"]): row
        for row in _read_csv(PROJECT_ROOT / "docs/assets/subjects01-109_full_replication_central_peak_summary.csv")
    }
    imf = {
        int(row["subject"]): row
        for row in _read_csv(PROJECT_ROOT / "docs/assets/individual_mu_heldout_evaluation_peak_subject_summary.csv")
    }
    joined: list[dict[str, object]] = []
    for subject, score in sorted(score_lookup.items()):
        historical = primary[subject]
        peak = peaks[subject]
        imf_row = imf.get(subject)
        joined.append(
            {
                "subject": subject,
                "csp_balanced_accuracy": score,
                "c4_fixed_fists_erd_median_percent": float(historical["c4_median_percent_change"]),
                "c3_fixed_fists_erd_median_percent": float(historical["c3_median_percent_change"]),
                "historical_central_peak_frequency_hz": float(peak["peak_frequency_hz"]),
                "historical_central_peak_well_defined": peak["well_defined_peak"] == "True",
                "imf_study_subject": imf_row is not None,
                "imf_method_available": imf_row is not None and imf_row["method_available"] == "True",
                "imf_peak_frequency_median_hz": (
                    float(imf_row["peak_frequency_median_hz"])
                    if imf_row is not None and imf_row["method_available"] == "True" else ""
                ),
                "imf_three_fold_peak_span_hz": (
                    float(imf_row["three_fold_peak_span_hz"])
                    if imf_row is not None and imf_row["method_available"] == "True" else ""
                ),
                "analysis_role": "post_evaluation_exploratory_only",
            }
        )
    relationships: list[dict[str, object]] = []
    for field, label in (
        ("c4_fixed_fists_erd_median_percent", "C4_fixed_ERD"),
        ("c3_fixed_fists_erd_median_percent", "C3_fixed_ERD"),
        ("historical_central_peak_frequency_hz", "historical_central_peak_frequency"),
    ):
        x = np.array([float(row[field]) for row in joined])
        y = np.array([float(row["csp_balanced_accuracy"]) for row in joined])
        statistic, p_value = spearmanr(x, y)
        relationships.append(
            {
                "relationship": label,
                "subject_count": len(x),
                "spearman_rho": float(statistic),
                "two_sided_unadjusted_p": float(p_value),
                "analysis_role": "post_evaluation_exploratory_only",
            }
        )
    available = [row for row in joined if row["imf_method_available"]]
    for field, label in (
        ("imf_peak_frequency_median_hz", "IMF_peak_frequency"),
        ("imf_three_fold_peak_span_hz", "IMF_peak_span_reliability"),
    ):
        x = np.array([float(row[field]) for row in available])
        y = np.array([float(row["csp_balanced_accuracy"]) for row in available])
        statistic, p_value = spearmanr(x, y)
        relationships.append(
            {
                "relationship": label,
                "subject_count": len(x),
                "spearman_rho": float(statistic),
                "two_sided_unadjusted_p": float(p_value),
                "analysis_role": "post_evaluation_exploratory_only",
            }
        )
    imf_overlap = [row for row in joined if row["imf_study_subject"]]
    available_scores = [float(row["csp_balanced_accuracy"]) for row in imf_overlap if row["imf_method_available"]]
    unavailable_scores = [float(row["csp_balanced_accuracy"]) for row in imf_overlap if not row["imf_method_available"]]
    relationships.append(
        {
            "relationship": "IMF_availability_group_medians",
            "subject_count": len(imf_overlap),
            "spearman_rho": "",
            "two_sided_unadjusted_p": "",
            "available_count": len(available_scores),
            "available_median_balanced_accuracy": float(np.median(available_scores)),
            "unavailable_count": len(unavailable_scores),
            "unavailable_median_balanced_accuracy": (
                float(np.median(unavailable_scores)) if unavailable_scores else ""
            ),
            "analysis_role": "post_evaluation_exploratory_only",
        }
    )
    return joined, relationships
