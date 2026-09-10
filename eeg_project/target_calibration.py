"""Leakage-safe decision-layer calibration for a frozen cross-subject CSP model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from .cross_subject_decoding import FittedCrossSubjectModel, file_sha256
from .decoding import (
    PROJECT_ROOT,
    RUNS,
    SubjectDecodingData,
    bootstrap_median_interval,
)


DEFAULT_CALIBRATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "minimal_target_calibration.json"
)
INITIAL_CALIBRATION_FREEZE_COMMIT = "d4cff9a"
CALIBRATION_METHODS = ("threshold", "target_lda")
CALIBRATION_SIZES = (0, 4, 8, 12, 14)


@dataclass(frozen=True)
class TargetSourceFeatures:
    """One target represented only through frozen source CSP and scaling."""

    subject: int
    features: np.ndarray
    source_decision_scores: np.ndarray
    source_predictions: np.ndarray
    labels: np.ndarray
    runs: np.ndarray
    identities: tuple[object, ...]
    qc_candidates: np.ndarray


@dataclass
class CalibrationScenarioResult:
    """Predictions, test-run metrics, and an audit for one calibration scenario."""

    prediction_rows: list[dict[str, object]]
    test_run_rows: list[dict[str, object]]
    audit_row: dict[str, object]


def load_calibration_config(
    path: Path = DEFAULT_CALIBRATION_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the initial calibration freeze and verify historical evidence."""
    resolved = path.expanduser().resolve()
    config: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    validate_calibration_config(config)
    return config


def validate_calibration_config(config: Mapping[str, Any]) -> None:
    """Reject cohort, curve, leakage, method-selection, or historical drift."""
    if config["schema_version"] != 1:
        raise ValueError("Unsupported target-calibration schema.")
    if config["study_stage"] != "initial_minimal_target_calibration_methodology_frozen":
        raise ValueError("Expected the initial target-calibration freeze.")
    cohorts = config["cohorts"]
    if cohorts["development_subjects"] != list(range(1, 21)):
        raise ValueError("Calibration development cohort drifted.")
    requested = list(range(21, 110))
    if cohorts["evaluation_requested_subjects"] != requested:
        raise ValueError("Calibration evaluation cohort drifted.")
    if cohorts["technically_incompatible_subjects"] != [88, 92, 100]:
        raise ValueError("Calibration technical exclusions drifted.")
    if cohorts["evaluation_eligible_subjects"] != [
        subject for subject in requested if subject not in {88, 92, 100}
    ]:
        raise ValueError("Calibration eligible cohort is inconsistent.")
    sizes = config["calibration_sizes"]
    if tuple(sizes["ordered_total_trials"]) != CALIBRATION_SIZES:
        raise ValueError("Calibration sizes drifted.")
    if sizes["trials_per_class"] != {
        "0": 0,
        "4": 2,
        "8": 4,
        "12": 6,
        "14": 7,
    }:
        raise ValueError("Balanced calibration counts drifted.")
    representation = config["immutable_source_representation"]
    if representation["model"] != "csp_4_empirical":
        raise ValueError("Calibration must preserve the frozen CSP representation.")
    for field in (
        "CSP_fit_target_EEG",
        "source_scaler_fit_target_EEG",
        "unlabeled_target_alignment",
        "target_covariance_alignment",
        "target_specific_preprocessing",
    ):
        if representation[field]:
            raise ValueError(f"Forbidden target operation enabled: {field}.")
    scenarios = config["simulated_calibration_session"]["scenarios"]
    expected = [
        {"calibration_run": run, "test_runs": [other for other in RUNS if other != run]}
        for run in RUNS
    ]
    if scenarios != expected:
        raise ValueError("Calibration/test run scenarios drifted.")
    selection = config["development_method_selection"]
    if selection["otherwise"] != "threshold":
        raise ValueError("The frozen simplicity preference drifted.")
    if config["primary_evaluation"]["group_observational_unit"] != "participant":
        raise ValueError("Calibration group inference must use participants.")
    for record in config["historical_inputs"].values():
        path = PROJECT_ROOT / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Historical calibration input drift: {path}.")


def calibration_indices(
    target: SubjectDecodingData,
    calibration_run: int,
    calibration_size: int,
) -> np.ndarray:
    """Select the exact first N trials per class by annotation chronology."""
    if calibration_run not in RUNS or calibration_size not in CALIBRATION_SIZES:
        raise ValueError("Unknown calibration run or size.")
    per_class = calibration_size // 2
    selected: list[int] = []
    for label in (0, 1):
        candidates = [
            index
            for index, identity in enumerate(target.identities)
            if int(target.runs[index]) == calibration_run
            and int(target.labels[index]) == label
            and int(identity.run) == calibration_run
        ]
        candidates.sort(key=lambda index: int(target.identities[index].annotation_index))
        if len(candidates) < per_class:
            raise RuntimeError(
                f"S{target.subject:03d} run {calibration_run} has only "
                f"{len(candidates)} trials for class {label}; {per_class} required."
            )
        selected.extend(candidates[:per_class])
    ordered = np.asarray(
        sorted(selected, key=lambda index: int(target.identities[index].annotation_index)),
        dtype=int,
    )
    if ordered.size != calibration_size:
        raise RuntimeError("Calibration subset has the wrong size.")
    if calibration_size and set(np.bincount(target.labels[ordered], minlength=2)) != {
        per_class
    }:
        raise RuntimeError("Calibration subset is not class balanced.")
    return ordered


def validate_nested_calibration_subsets(
    target: SubjectDecodingData,
) -> None:
    """Prove all runs support the frozen nested 0/4/8/12/14 curve."""
    for run in RUNS:
        previous: set[int] = set()
        for size in CALIBRATION_SIZES:
            current = set(calibration_indices(target, run, size).tolist())
            if not previous.issubset(current):
                raise RuntimeError("Calibration subsets are not nested.")
            previous = current


def transform_target_with_source(
    fitted: FittedCrossSubjectModel,
    target: SubjectDecodingData,
) -> TargetSourceFeatures:
    """Apply frozen source CSP/scaler once without fitting on target EEG."""
    if fitted.model_name != "csp_4_empirical":
        raise ValueError("Target calibration is frozen to csp_4_empirical.")
    if target.subject in fitted.training_subjects:
        raise ValueError("Target participant occurs in source-model fitting.")
    csp = fitted.pipeline.named_steps["csp"]
    scaler = fitted.pipeline.named_steps["scaler"]
    lda = fitted.pipeline.named_steps["lda"]
    csp_features = np.asarray(csp.transform(target.csp_task_data_volts), dtype=float)
    features = np.asarray(scaler.transform(csp_features), dtype=float)
    scores = np.asarray(lda.decision_function(features), dtype=float)
    predictions = np.asarray(lda.predict(features), dtype=int)
    count = target.labels.size
    if (
        features.shape != (count, 4)
        or scores.shape != (count,)
        or predictions.shape != (count,)
        or not np.isfinite(features).all()
        or not np.isfinite(scores).all()
    ):
        raise RuntimeError("Frozen target source features are invalid.")
    return TargetSourceFeatures(
        target.subject,
        features,
        scores,
        predictions,
        target.labels.copy(),
        target.runs.copy(),
        tuple(target.identities),
        target.qc_candidates.copy(),
    )


def select_balanced_accuracy_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Choose a deterministic threshold using labeled calibration scores only."""
    score_values = np.asarray(scores, dtype=float)
    label_values = np.asarray(labels, dtype=int)
    if score_values.ndim != 1 or score_values.size < 4:
        raise ValueError("Threshold calibration requires at least four scores.")
    if score_values.shape != label_values.shape or set(label_values) != {0, 1}:
        raise ValueError("Threshold calibration requires aligned binary classes.")
    unique = np.unique(score_values)
    candidates = [np.nextafter(unique[0], -np.inf)]
    candidates.extend(((unique[:-1] + unique[1:]) / 2.0).tolist())
    candidates.append(np.nextafter(unique[-1], np.inf))
    ranked = []
    for threshold in candidates:
        predictions = (score_values >= threshold).astype(int)
        score = balanced_accuracy_score(label_values, predictions)
        ranked.append((-float(score), abs(float(threshold)), float(threshold)))
    return min(ranked)[2]


def fit_target_lda(features: np.ndarray, labels: np.ndarray) -> LinearDiscriminantAnalysis:
    """Fit the exact existing shrinkage-LDA head on labeled calibration rows only."""
    feature_values = np.asarray(features, dtype=float)
    label_values = np.asarray(labels, dtype=int)
    if (
        feature_values.ndim != 2
        or feature_values.shape[0] < 4
        or feature_values.shape[0] != label_values.size
        or set(label_values) != {0, 1}
    ):
        raise ValueError("Target LDA requires at least four aligned binary trials.")
    lda = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto", priors=np.asarray([0.5, 0.5])
    )
    lda.fit(feature_values, label_values)
    if set(lda.classes_) != {0, 1}:
        raise RuntimeError("Target LDA did not fit both classes.")
    return lda


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    if y_true.size == 0 or set(np.unique(y_true)) != {0, 1}:
        raise RuntimeError("Calibration scoring requires both classes.")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ff, ft, tf, tt = (int(value) for value in matrix.ravel())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fists_recall": ff / (ff + ft),
        "feet_recall": tt / (tf + tt),
        "true_fists_predicted_fists": ff,
        "true_fists_predicted_feet": ft,
        "true_feet_predicted_fists": tf,
        "true_feet_predicted_feet": tt,
        "trial_count": int(y_true.size),
    }


def evaluate_calibration_scenario(
    source: TargetSourceFeatures,
    training_subjects: Sequence[int],
    calibration_run: int,
    calibration_size: int,
    method: str,
) -> CalibrationScenarioResult:
    """Calibrate on one run and predict only the other two target runs."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method: {method}.")
    if calibration_run not in RUNS:
        raise ValueError(f"Unknown calibration run: {calibration_run}.")
    test_runs = tuple(run for run in RUNS if run != calibration_run)
    calibration = calibration_indices_from_features(source, calibration_run, calibration_size)
    calibration_keys = [source.identities[index].key for index in calibration]
    test_indices = np.flatnonzero(np.isin(source.runs, test_runs))
    if set(calibration).intersection(test_indices.tolist()):
        raise RuntimeError("Calibration and test trials overlap.")

    threshold: float | None = None
    target_lda: LinearDiscriminantAnalysis | None = None
    if calibration_size == 0:
        predictions = source.source_predictions[test_indices]
    elif method == "threshold":
        threshold = select_balanced_accuracy_threshold(
            source.source_decision_scores[calibration], source.labels[calibration]
        )
        predictions = (
            source.source_decision_scores[test_indices] >= threshold
        ).astype(int)
    else:
        target_lda = fit_target_lda(
            source.features[calibration], source.labels[calibration]
        )
        predictions = np.asarray(target_lda.predict(source.features[test_indices]), dtype=int)

    prediction_rows: list[dict[str, object]] = []
    for index, prediction in zip(test_indices, predictions, strict=True):
        identity = source.identities[index]
        prediction_rows.append(
            {
                "subject": source.subject,
                "method": method,
                "calibration_size": calibration_size,
                "calibration_run": calibration_run,
                "test_run": int(source.runs[index]),
                "trial_key": identity.key,
                "run_trial_index": identity.run_trial_index,
                "annotation_index": identity.annotation_index,
                "true_condition": identity.semantic_condition,
                "true_label": int(source.labels[index]),
                "predicted_label": int(prediction),
                "predicted_condition": (
                    "both_fists_imagery" if prediction == 0 else "both_feet_imagery"
                ),
                "correct": bool(prediction == source.labels[index]),
                "qc_candidate": bool(source.qc_candidates[index]),
                "source_decision_score_feet": float(
                    source.source_decision_scores[index]
                ),
                "calibrated_threshold": threshold if threshold is not None else "",
                "calibration_trial_keys": "/".join(calibration_keys),
                "training_subjects": "/".join(str(value) for value in training_subjects),
                "source_CSP_frozen": True,
                "source_scaler_frozen": True,
                "test_run_used_for_calibration": False,
                "source_file": identity.source_file,
                "source_sha256": identity.source_sha256,
            }
        )

    test_run_rows: list[dict[str, object]] = []
    for test_run in test_runs:
        for qc_sensitivity in (False, True):
            row_indices = np.asarray(
                [
                    row_index
                    for row_index, row in enumerate(prediction_rows)
                    if row["test_run"] == test_run
                    and (not qc_sensitivity or not row["qc_candidate"])
                ],
                dtype=int,
            )
            true = np.asarray(
                [prediction_rows[index]["true_label"] for index in row_indices],
                dtype=int,
            )
            predicted = np.asarray(
                [prediction_rows[index]["predicted_label"] for index in row_indices],
                dtype=int,
            )
            metrics = _metric_row(true, predicted)
            test_run_rows.append(
                {
                    "subject": source.subject,
                    "method": method,
                    "calibration_size": calibration_size,
                    "calibration_run": calibration_run,
                    "test_run": test_run,
                    "qc_sensitivity": qc_sensitivity,
                    **metrics,
                }
            )

    audit = {
        "subject": source.subject,
        "method": method,
        "calibration_size": calibration_size,
        "calibration_run": calibration_run,
        "test_runs": "/".join(str(run) for run in test_runs),
        "training_subjects": "/".join(str(value) for value in training_subjects),
        "source_CSP_fit_subjects": "/".join(str(value) for value in training_subjects),
        "source_scaler_fit_subjects": "/".join(str(value) for value in training_subjects),
        "calibration_trial_keys": "/".join(calibration_keys),
        "calibration_trial_count": calibration_size,
        "calibration_fists_count": int(np.sum(source.labels[calibration] == 0)),
        "calibration_feet_count": int(np.sum(source.labels[calibration] == 1)),
        "test_trial_keys": "/".join(source.identities[index].key for index in test_indices),
        "test_runs_absent_from_calibration": bool(
            all(source.runs[index] == calibration_run for index in calibration)
        ),
        "target_EEG_used_for_source_CSP_fit": False,
        "target_EEG_used_for_source_scaler_fit": False,
        "unlabeled_test_EEG_used_for_adaptation": False,
        "target_CSP_refit": False,
        "target_scaler_refit": False,
        "decision_layer_fitted_from_calibration_only": calibration_size > 0,
        "calibrated_threshold": threshold if threshold is not None else "",
        "target_LDA_training_trial_count": (
            calibration_size if target_lda is not None else 0
        ),
    }
    return CalibrationScenarioResult(prediction_rows, test_run_rows, audit)


def calibration_indices_from_features(
    source: TargetSourceFeatures,
    calibration_run: int,
    calibration_size: int,
) -> np.ndarray:
    """Select calibration rows from a frozen-feature bundle."""
    proxy = SubjectDecodingData(
        source.subject,
        np.empty((source.labels.size, 0, 0)),
        np.empty((source.labels.size, 0, 0)),
        source.labels,
        source.runs,
        list(source.identities),
        source.qc_candidates,
        [],
        0.0,
        0.0,
        0.0,
    )
    return calibration_indices(proxy, calibration_run, calibration_size)


def participant_calibration_rows(
    test_run_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    expected_subjects: Sequence[int],
    methods: Sequence[str],
) -> list[dict[str, object]]:
    """Aggregate repeated scenario predictions to one participant value per size."""
    output: list[dict[str, object]] = []
    for subject in expected_subjects:
        for method in methods:
            for size in CALIBRATION_SIZES:
                for qc_sensitivity in (False, True):
                    scores = [
                        row
                        for row in test_run_rows
                        if int(row["subject"]) == subject
                        and row["method"] == method
                        and int(row["calibration_size"]) == size
                        and bool(row["qc_sensitivity"]) == qc_sensitivity
                    ]
                    if len(scores) != 6:
                        raise RuntimeError("Participant calibration score lacks six test runs.")
                    if size == 0:
                        unique_run_scores: list[float] = []
                        for run in RUNS:
                            repeated = [
                                float(row["balanced_accuracy"])
                                for row in scores
                                if int(row["test_run"]) == run
                            ]
                            if len(repeated) != 2 or repeated[0] != repeated[1]:
                                raise RuntimeError(
                                    "Zero-shot repeated run scores are not identical."
                                )
                            unique_run_scores.append(repeated[0])
                        primary_score = float(np.mean(unique_run_scores))
                    else:
                        primary_score = float(
                            np.mean(
                                [float(row["balanced_accuracy"]) for row in scores]
                            )
                        )
                    selected_predictions = [
                        row
                        for row in prediction_rows
                        if int(row["subject"]) == subject
                        and row["method"] == method
                        and int(row["calibration_size"]) == size
                        and (not qc_sensitivity or not bool(row["qc_candidate"]))
                    ]
                    if not selected_predictions:
                        raise RuntimeError("Participant calibration predictions are absent.")
                    true = np.asarray(
                        [int(row["true_label"]) for row in selected_predictions], dtype=int
                    )
                    predicted = np.asarray(
                        [int(row["predicted_label"]) for row in selected_predictions],
                        dtype=int,
                    )
                    pooled = _metric_row(true, predicted)
                    output.append(
                        {
                            "subject": subject,
                            "method": method,
                            "calibration_size": size,
                            "qc_sensitivity": qc_sensitivity,
                            "primary_mean_test_run_balanced_accuracy": primary_score,
                            "pooled_accuracy": pooled["accuracy"],
                            "fists_recall": pooled["fists_recall"],
                            "feet_recall": pooled["feet_recall"],
                            "trial_prediction_appearances": pooled["trial_count"],
                            "group_observational_unit": "participant",
                        }
                    )
    return output


def development_method_comparison(
    participant_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Apply the frozen global threshold-versus-target-LDA selection rule."""
    rows = [row for row in participant_rows if not bool(row["qc_sensitivity"])]
    subjects = sorted({int(row["subject"]) for row in rows})
    paired: list[dict[str, object]] = []
    for subject in subjects:
        utilities: dict[str, float] = {}
        for method in CALIBRATION_METHODS:
            values = [
                float(row["primary_mean_test_run_balanced_accuracy"])
                for row in rows
                if int(row["subject"]) == subject
                and row["method"] == method
                and int(row["calibration_size"]) > 0
            ]
            if len(values) != 4:
                raise RuntimeError("Development method utility lacks four nonzero sizes.")
            utilities[method] = float(np.mean(values))
        paired.append(
            {
                "subject": subject,
                "threshold_mean_nonzero_size_accuracy": utilities["threshold"],
                "target_LDA_mean_nonzero_size_accuracy": utilities["target_lda"],
                "target_LDA_minus_threshold": (
                    utilities["target_lda"] - utilities["threshold"]
                ),
            }
        )
    differences = np.asarray([float(row["target_LDA_minus_threshold"]) for row in paired])
    rule = config["development_method_selection"]["select_target_LDA_only_if"]
    median = float(np.median(differences))
    target_better = int(np.sum(differences > 0))
    threshold_better = int(np.sum(differences < 0))
    fraction = target_better / len(differences)
    selected = (
        "target_lda"
        if median >= float(rule["minimum_median_paired_difference"])
        and fraction >= float(rule["minimum_fraction_subjects_target_LDA_better"])
        else "threshold"
    )
    summary = {
        "participant_count": len(differences),
        "median_target_LDA_minus_threshold_utility": median,
        "q25_target_LDA_minus_threshold_utility": float(np.percentile(differences, 25)),
        "q75_target_LDA_minus_threshold_utility": float(np.percentile(differences, 75)),
        "target_LDA_better_count": target_better,
        "threshold_better_count": threshold_better,
        "tie_count": int(np.sum(differences == 0)),
        "target_LDA_better_fraction": fraction,
        "selected_method": selected,
        "selection_rule_applied_without_evaluation_outcomes": True,
    }
    return paired, summary


def exact_sign_test_positive(differences: np.ndarray) -> dict[str, object]:
    """One-sided exact sign test of positive participant-matched gains."""
    values = np.asarray(differences, dtype=float)
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    ties = int(np.sum(values == 0))
    p_value = (
        float(binomtest(positive, positive + negative, 0.5, alternative="greater").pvalue)
        if positive + negative
        else 1.0
    )
    return {"improved_count": positive, "worsened_count": negative, "tie_count": ties, "one_sided_exact_sign_p": p_value}


def calibration_curve_summary(
    participant_rows: Sequence[Mapping[str, object]],
    method: str,
    config: Mapping[str, Any],
    *,
    expected_subject_count: int,
) -> list[dict[str, object]]:
    """Summarize one participant score per size with matched zero-shot gains."""
    primary = [
        row
        for row in participant_rows
        if row["method"] == method and not bool(row["qc_sensitivity"])
    ]
    qc = {
        (int(row["subject"]), int(row["calibration_size"])): row
        for row in participant_rows
        if row["method"] == method and bool(row["qc_sensitivity"])
    }
    zero = {
        int(row["subject"]): float(row["primary_mean_test_run_balanced_accuracy"])
        for row in primary
        if int(row["calibration_size"]) == 0
    }
    if len(zero) != expected_subject_count:
        raise RuntimeError("Calibration curve lacks matched zero-shot participants.")
    bootstrap = config["group_inference"]["bootstrap_median_confidence_interval"]
    output: list[dict[str, object]] = []
    for size in CALIBRATION_SIZES:
        selected = sorted(
            (row for row in primary if int(row["calibration_size"]) == size),
            key=lambda row: int(row["subject"]),
        )
        if len(selected) != expected_subject_count:
            raise RuntimeError("Calibration curve lacks one row per participant.")
        scores = np.asarray(
            [float(row["primary_mean_test_run_balanced_accuracy"]) for row in selected]
        )
        gains = np.asarray(
            [score - zero[int(row["subject"])] for score, row in zip(scores, selected, strict=True)]
        )
        fists = np.asarray([float(row["fists_recall"]) for row in selected])
        feet = np.asarray([float(row["feet_recall"]) for row in selected])
        qc_scores = np.asarray(
            [
                float(qc[(int(row["subject"]), size)]["primary_mean_test_run_balanced_accuracy"])
                for row in selected
            ]
        )
        lower, upper = bootstrap_median_interval(
            scores,
            resamples=int(bootstrap["resamples"]),
            confidence_level=float(bootstrap["confidence_level"]),
            random_seed=int(bootstrap["random_seed"]) + size,
        )
        sign = exact_sign_test_positive(gains)
        output.append(
            {
                "method": method,
                "calibration_size": size,
                "participant_count": expected_subject_count,
                "median_balanced_accuracy": float(np.median(scores)),
                "bootstrap_median_ci_lower": lower,
                "bootstrap_median_ci_upper": upper,
                "q25_balanced_accuracy": float(np.percentile(scores, 25)),
                "q75_balanced_accuracy": float(np.percentile(scores, 75)),
                "above_0_5_count": int(np.sum(scores > 0.5)),
                "above_0_5_fraction": float(np.mean(scores > 0.5)),
                "at_or_above_0_6_count": int(np.sum(scores >= 0.6)),
                "at_or_above_0_6_fraction": float(np.mean(scores >= 0.6)),
                "at_or_above_0_7_count": int(np.sum(scores >= 0.7)),
                "at_or_above_0_7_fraction": float(np.mean(scores >= 0.7)),
                "median_fists_recall": float(np.median(fists)),
                "median_feet_recall": float(np.median(feet)),
                "median_gain_over_matched_zero": float(np.median(gains)),
                "q25_gain_over_matched_zero": float(np.percentile(gains, 25)),
                "q75_gain_over_matched_zero": float(np.percentile(gains, 75)),
                **sign,
                "improved_fraction": sign["improved_count"] / expected_subject_count,
                "QC_cohort_median_balanced_accuracy": float(np.median(qc_scores)),
                "median_paired_QC_minus_primary_balanced_accuracy": float(
                    np.median(qc_scores - scores)
                ),
                "group_observational_unit": "participant",
            }
        )
    return output


def convincing_benefit(row: Mapping[str, object], config: Mapping[str, Any]) -> bool:
    """Apply only the pre-outcome minimum-calibration success criteria."""
    if int(row["calibration_size"]) == 0:
        return False
    criteria = config["convincing_benefit_criteria"]
    fists = float(row["median_fists_recall"])
    feet = float(row["median_feet_recall"])
    return bool(
        float(row["median_balanced_accuracy"])
        >= criteria["minimum_median_balanced_accuracy"]
        and float(row["median_gain_over_matched_zero"])
        >= criteria["minimum_median_gain_over_matched_zero"]
        and float(row["improved_fraction"])
        >= criteria["minimum_fraction_subjects_improved"]
        and min(fists, feet) >= criteria["minimum_median_recall_each_class"]
        and abs(fists - feet)
        <= criteria["maximum_absolute_median_class_recall_difference"]
        and float(row["median_paired_QC_minus_primary_balanced_accuracy"])
        >= criteria["minimum_median_paired_QC_minus_primary_balanced_accuracy"]
        and float(row["one_sided_exact_sign_p"])
        <= criteria["maximum_one_sided_exact_sign_test_p_improvement"]
    )
