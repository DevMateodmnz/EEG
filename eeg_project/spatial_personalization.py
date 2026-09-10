"""Leakage-safe decision-layer and target-CSP personalization operations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mne.decoding import CSP
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cross_subject_decoding import file_sha256
from .decoding import PROJECT_ROOT, RUNS, SubjectDecodingData, bootstrap_median_interval
from .target_calibration import (
    TargetSourceFeatures,
    calibration_indices,
    exact_sign_test_positive,
)


DEFAULT_SPATIAL_PERSONALIZATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "spatial_personalization.json"
)
INITIAL_SPATIAL_PERSONALIZATION_FREEZE_COMMIT = "4ceea62"
A_METHOD = "source_zero_shot"
B_METHODS = (
    "source_csp_target_lda_source_scaler",
    "source_csp_target_lda_target_scaler",
)
C_METHODS = ("target_csp_empirical", "target_csp_ledoit_wolf")
DEVELOPMENT_METHODS = (A_METHOD, *B_METHODS, *C_METHODS)


@dataclass
class PersonalizationScenarioResult:
    """Predictions, metrics, fit audit, and optional fitted CSP subspace."""

    prediction_rows: list[dict[str, object]]
    test_run_rows: list[dict[str, object]]
    audit_row: dict[str, object]
    csp_filter_subspace: np.ndarray | None


PipelineFactory = Callable[[str, Mapping[str, Any]], Any]


def load_spatial_personalization_config(
    path: Path = DEFAULT_SPATIAL_PERSONALIZATION_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the initial spatial-personalization freeze."""
    resolved = path.expanduser().resolve()
    config: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    validate_spatial_personalization_config(config)
    return config


def validate_spatial_personalization_config(config: Mapping[str, Any]) -> None:
    """Reject cohort, geometry, candidate, leakage, gate, or history drift."""
    if config["schema_version"] != 1:
        raise ValueError("Unsupported spatial-personalization schema.")
    if config["study_stage"] != "initial_spatial_personalization_methodology_frozen":
        raise ValueError("Expected the initial spatial-personalization freeze.")
    cohorts = config["cohorts"]
    if cohorts["development_subjects"] != list(range(1, 21)):
        raise ValueError("Spatial-personalization development cohort drifted.")
    requested = list(range(21, 110))
    if cohorts["evaluation_requested_subjects"] != requested:
        raise ValueError("Spatial-personalization evaluation cohort drifted.")
    if cohorts["technically_incompatible_subjects"] != [88, 92, 100]:
        raise ValueError("Spatial-personalization exclusions drifted.")
    if cohorts["evaluation_eligible_subjects"] != [
        subject for subject in requested if subject not in {88, 92, 100}
    ]:
        raise ValueError("Spatial-personalization eligible cohort is inconsistent.")
    geometry = config["calibration_geometry"]
    if geometry["runs"] != list(RUNS) or geometry["total_calibration_trials"] != 14:
        raise ValueError("Complete-run calibration geometry drifted.")
    if geometry["trials_per_class"] != 7 or not geometry["balanced"]:
        raise ValueError("Calibration must contain seven trials per class.")
    scenarios = {
        int(row["calibration_run"]): set(int(run) for run in row["test_runs"])
        for row in geometry["scenarios"]
    }
    if scenarios != {run: set(RUNS) - {run} for run in RUNS}:
        raise ValueError("Calibration/test scenarios drifted.")
    for guard in (
        "test_labels_enter_fitting",
        "unlabeled_test_EEG_enters_adaptation",
        "test_covariance_alignment",
    ):
        if geometry[guard]:
            raise ValueError(f"Forbidden test adaptation enabled: {guard}.")
    fixed = config["fixed_signal_representation"]
    if (
        fixed["passband_hz"] != [8.0, 30.0]
        or fixed["task_interval_seconds"] != [1.0, 3.0]
        or fixed["target_CSP_components"] != 4
    ):
        raise ValueError("Frozen CSP representation drifted.")
    if set(config["representation_B_source_CSP_target_classifier"]["development_scaler_candidates"]) != {
        "retain_source_scaler",
        "fit_target_scaler",
    }:
        raise ValueError("Decision-layer scaling candidates drifted.")
    if set(config["representation_C_target_CSP_target_classifier"]["covariance_candidates"]) != {
        "empirical",
        "ledoit_wolf",
    }:
        raise ValueError("Target-CSP covariance candidates drifted.")
    if config["primary_comparisons"] != ["B_minus_A", "C_minus_B", "C_minus_A"]:
        raise ValueError("Primary personalization comparisons drifted.")
    if config["participant_aggregation"]["group_observational_unit"] != "participant":
        raise ValueError("The group observational unit must remain the participant.")
    if not config["CSP_stability"]["sign_order_and_within_subspace_rotation_invariant"]:
        raise ValueError("CSP stability must remain subspace based.")
    for record in config["historical_inputs"].values():
        path = PROJECT_ROOT / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Historical spatial-personalization input drift: {path}.")


def complete_run_calibration_indices(
    dataset: SubjectDecodingData, calibration_run: int
) -> np.ndarray:
    """Return the first seven chronological trials per class from one run."""
    indices = calibration_indices(dataset, calibration_run, 14)
    if indices.size != 14:
        raise RuntimeError("Complete-run calibration must contain 14 trials.")
    if np.bincount(dataset.labels[indices], minlength=2).tolist() != [7, 7]:
        raise RuntimeError("Complete-run calibration is not balanced 7/7.")
    annotations = [dataset.identities[index].annotation_index for index in indices]
    if annotations != sorted(annotations):
        raise RuntimeError("Calibration trials are not in annotation chronology.")
    return indices


def build_personalization_pipeline(
    method: str, config: Mapping[str, Any]
) -> Pipeline | LinearDiscriminantAnalysis:
    """Construct one unfitted downstream or target-CSP pipeline."""
    lda_settings = config["representation_B_source_CSP_target_classifier"]["target_LDA"]
    lda = LinearDiscriminantAnalysis(
        solver=str(lda_settings["solver"]),
        shrinkage=lda_settings["shrinkage"],
        priors=np.asarray(lda_settings["priors"], dtype=float),
    )
    if method == B_METHODS[0]:
        return lda
    if method == B_METHODS[1]:
        scaler = config["representation_B_source_CSP_target_classifier"][
            "target_scaler_when_candidate_enabled"
        ]
        return Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(
                        with_mean=bool(scaler["with_mean"]),
                        with_std=bool(scaler["with_std"]),
                    ),
                ),
                ("lda", lda),
            ]
        )
    if method in C_METHODS:
        fixed = config["fixed_signal_representation"]
        reg = (
            config["representation_C_target_CSP_target_classifier"][
                "covariance_candidates"
            ]["empirical" if method == C_METHODS[0] else "ledoit_wolf"]
        )
        c_scaler = config["representation_C_target_CSP_target_classifier"]["scaler"]
        return Pipeline(
            [
                (
                    "csp",
                    CSP(
                        n_components=int(fixed["target_CSP_components"]),
                        reg=reg,
                        log=bool(fixed["CSP_log"]),
                        cov_est=str(fixed["CSP_cov_est"]),
                        transform_into=str(fixed["CSP_transform_into"]),
                        norm_trace=bool(fixed["CSP_norm_trace"]),
                        component_order=str(fixed["CSP_component_order"]),
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(
                        with_mean=bool(c_scaler["with_mean"]),
                        with_std=bool(c_scaler["with_std"]),
                    ),
                ),
                ("lda", lda),
            ]
        )
    raise ValueError(f"Unknown personalization method: {method}.")


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Calculate the compact binary metric set with explicit class accounting."""
    if y_true.size == 0 or set(np.unique(y_true)) != {0, 1}:
        raise RuntimeError("Personalization scoring requires both classes.")
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


def _decision_function(model: Any, values: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(values), dtype=float)
    if scores.ndim != 1 or scores.shape[0] != values.shape[0]:
        raise RuntimeError("Personalized binary model returned invalid decision scores.")
    return scores


def evaluate_personalization_scenario(
    dataset: SubjectDecodingData,
    source: TargetSourceFeatures,
    training_subjects: Sequence[int],
    calibration_run: int,
    method: str,
    config: Mapping[str, Any],
    *,
    pipeline_factory: PipelineFactory = build_personalization_pipeline,
) -> PersonalizationScenarioResult:
    """Fit only allowed target stages on one run and predict the other two."""
    if method not in DEVELOPMENT_METHODS:
        raise ValueError(f"Unknown personalization method: {method}.")
    if dataset.subject != source.subject:
        raise ValueError("Raw target and source-feature target subjects differ.")
    if [identity.key for identity in dataset.identities] != [
        identity.key for identity in source.identities
    ]:
        raise RuntimeError("Raw and source-feature target trial identities differ.")
    if not np.array_equal(dataset.labels, source.labels) or not np.array_equal(
        dataset.runs, source.runs
    ):
        raise RuntimeError("Raw and source-feature target labels or runs differ.")
    if dataset.subject in set(training_subjects):
        raise ValueError("Target participant occurs in source training subjects.")

    calibration = complete_run_calibration_indices(dataset, calibration_run)
    test_runs = tuple(run for run in RUNS if run != calibration_run)
    test_indices = np.flatnonzero(np.isin(dataset.runs, test_runs))
    if set(calibration).intersection(test_indices.tolist()):
        raise RuntimeError("Calibration and test trials overlap.")
    calibration_keys = [dataset.identities[index].key for index in calibration]
    test_keys = [dataset.identities[index].key for index in test_indices]

    model: Any | None = None
    csp_subspace: np.ndarray | None = None
    if method == A_METHOD:
        predictions = source.source_predictions[test_indices]
        decision_scores = source.source_decision_scores[test_indices]
    elif method in B_METHODS:
        model = pipeline_factory(method, config)
        model.fit(source.features[calibration], dataset.labels[calibration])
        predictions = np.asarray(model.predict(source.features[test_indices]), dtype=int)
        decision_scores = _decision_function(model, source.features[test_indices])
    else:
        model = pipeline_factory(method, config)
        model.fit(dataset.csp_task_data_volts[calibration], dataset.labels[calibration])
        predictions = np.asarray(
            model.predict(dataset.csp_task_data_volts[test_indices]), dtype=int
        )
        decision_scores = _decision_function(model, dataset.csp_task_data_volts[test_indices])
        filters = np.asarray(model.named_steps["csp"].filters_, dtype=float)
        component_count = int(config["fixed_signal_representation"]["target_CSP_components"])
        if filters.ndim != 2 or filters.shape[0] < component_count:
            raise RuntimeError("Fitted target CSP filters have invalid shape.")
        csp_subspace = filters[:component_count].T.copy()
        if csp_subspace.shape != (dataset.csp_task_data_volts.shape[1], component_count):
            raise RuntimeError("Target CSP filter subspace has invalid dimensions.")

    if predictions.shape != test_indices.shape or not np.isfinite(decision_scores).all():
        raise RuntimeError("Personalization predictions are invalid.")
    prediction_rows: list[dict[str, object]] = []
    for index, prediction, score in zip(
        test_indices, predictions, decision_scores, strict=True
    ):
        identity = dataset.identities[index]
        prediction_rows.append(
            {
                "subject": dataset.subject,
                "method": method,
                "calibration_run": calibration_run,
                "test_run": int(dataset.runs[index]),
                "trial_key": identity.key,
                "run_trial_index": identity.run_trial_index,
                "annotation_index": identity.annotation_index,
                "true_condition": identity.semantic_condition,
                "true_label": int(dataset.labels[index]),
                "predicted_label": int(prediction),
                "predicted_condition": (
                    "both_fists_imagery" if prediction == 0 else "both_feet_imagery"
                ),
                "decision_score_feet": float(score),
                "correct": bool(prediction == dataset.labels[index]),
                "qc_candidate": bool(dataset.qc_candidates[index]),
                "calibration_trial_keys": "/".join(calibration_keys),
                "training_subjects": "/".join(str(value) for value in training_subjects),
                "test_run_used_for_fitting": False,
                "source_file": identity.source_file,
                "source_sha256": identity.source_sha256,
            }
        )

    test_run_rows: list[dict[str, object]] = []
    for test_run in test_runs:
        for qc_sensitivity in (False, True):
            selected = [
                row
                for row in prediction_rows
                if int(row["test_run"]) == test_run
                and (not qc_sensitivity or not bool(row["qc_candidate"]))
            ]
            metrics = _metric_row(
                np.asarray([int(row["true_label"]) for row in selected]),
                np.asarray([int(row["predicted_label"]) for row in selected]),
            )
            test_run_rows.append(
                {
                    "subject": dataset.subject,
                    "method": method,
                    "calibration_run": calibration_run,
                    "test_run": test_run,
                    "qc_sensitivity": qc_sensitivity,
                    **metrics,
                }
            )

    target_csp = method in C_METHODS
    target_scaler = method == B_METHODS[1] or target_csp
    target_lda = method != A_METHOD
    audit = {
        "subject": dataset.subject,
        "method": method,
        "calibration_run": calibration_run,
        "test_runs": "/".join(str(run) for run in test_runs),
        "training_subjects": "/".join(str(value) for value in training_subjects),
        "calibration_trial_keys": "/".join(calibration_keys),
        "calibration_trial_count": len(calibration),
        "calibration_fists_count": int(np.sum(dataset.labels[calibration] == 0)),
        "calibration_feet_count": int(np.sum(dataset.labels[calibration] == 1)),
        "test_trial_keys": "/".join(test_keys),
        "same_test_trial_count": len(test_indices),
        "source_CSP_fit_target_EEG": False,
        "target_CSP_fit_trial_keys": "/".join(calibration_keys) if target_csp else "not_applicable",
        "target_scaler_fit_trial_keys": "/".join(calibration_keys) if target_scaler else "not_applicable",
        "target_LDA_fit_trial_keys": "/".join(calibration_keys) if target_lda else "not_applicable",
        "test_EEG_used_for_CSP_fit": False,
        "test_EEG_used_for_scaler_fit": False,
        "test_EEG_used_for_LDA_fit": False,
        "test_labels_used_for_fitting": False,
        "unlabeled_test_EEG_used_for_adaptation": False,
        "test_covariance_alignment": False,
    }
    return PersonalizationScenarioResult(
        prediction_rows, test_run_rows, audit, csp_subspace
    )


def participant_personalization_rows(
    test_run_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    expected_subjects: Sequence[int],
    methods: Sequence[str],
) -> list[dict[str, object]]:
    """Aggregate six matched scenario/run scores to one value per participant."""
    output: list[dict[str, object]] = []
    for subject in expected_subjects:
        for method in methods:
            for qc_sensitivity in (False, True):
                selected_scores = [
                    row
                    for row in test_run_rows
                    if int(row["subject"]) == subject
                    and row["method"] == method
                    and bool(row["qc_sensitivity"]) == qc_sensitivity
                ]
                if len(selected_scores) != 6:
                    raise RuntimeError("Participant/method lacks six scenario-test scores.")
                if method == A_METHOD:
                    unique_run_scores: list[float] = []
                    for run in RUNS:
                        repeated = [
                            float(row["balanced_accuracy"])
                            for row in selected_scores
                            if int(row["test_run"]) == run
                        ]
                        if len(repeated) != 2 or repeated[0] != repeated[1]:
                            raise RuntimeError(
                                "Repeated zero-shot run scores are not identical."
                            )
                        unique_run_scores.append(repeated[0])
                    primary_score = float(np.mean(unique_run_scores))
                else:
                    primary_score = float(
                        np.mean(
                            [float(row["balanced_accuracy"]) for row in selected_scores]
                        )
                    )
                appearances = [
                    row
                    for row in prediction_rows
                    if int(row["subject"]) == subject
                    and row["method"] == method
                    and (not qc_sensitivity or not bool(row["qc_candidate"]))
                ]
                metrics = _metric_row(
                    np.asarray([int(row["true_label"]) for row in appearances]),
                    np.asarray([int(row["predicted_label"]) for row in appearances]),
                )
                output.append(
                    {
                        "subject": subject,
                        "method": method,
                        "qc_sensitivity": qc_sensitivity,
                        "primary_mean_scenario_test_run_balanced_accuracy": primary_score,
                        "scenario_test_run_balanced_accuracy_range": float(
                            np.ptp(
                                [float(row["balanced_accuracy"]) for row in selected_scores]
                            )
                        ),
                        "pooled_accuracy": metrics["accuracy"],
                        "fists_recall": metrics["fists_recall"],
                        "feet_recall": metrics["feet_recall"],
                        "trial_prediction_appearances": metrics["trial_count"],
                        "group_observational_unit": "participant",
                    }
                )
    return output


def select_development_methods(
    participant_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Apply frozen global choices for B scaling and C covariance."""
    primary = [row for row in participant_rows if not bool(row["qc_sensitivity"])]
    subjects = sorted({int(row["subject"]) for row in primary})
    lookup = {
        (int(row["subject"]), str(row["method"])): float(
            row["primary_mean_scenario_test_run_balanced_accuracy"]
        )
        for row in primary
    }
    paired: list[dict[str, object]] = []
    b_differences: list[float] = []
    c_differences: list[float] = []
    for subject in subjects:
        b_difference = lookup[(subject, B_METHODS[1])] - lookup[(subject, B_METHODS[0])]
        c_difference = lookup[(subject, C_METHODS[1])] - lookup[(subject, C_METHODS[0])]
        b_differences.append(b_difference)
        c_differences.append(c_difference)
        paired.extend(
            [
                {
                    "subject": subject,
                    "selection": "B_scaler",
                    "reference_method": B_METHODS[0],
                    "candidate_method": B_METHODS[1],
                    "candidate_minus_reference": b_difference,
                },
                {
                    "subject": subject,
                    "selection": "C_covariance",
                    "reference_method": C_METHODS[0],
                    "candidate_method": C_METHODS[1],
                    "candidate_minus_reference": c_difference,
                },
            ]
        )

    b_values = np.asarray(b_differences)
    c_values = np.asarray(c_differences)
    b_rule = config["representation_B_source_CSP_target_classifier"][
        "scaler_selection_rule"
    ]
    b_median = float(np.median(b_values))
    b_fraction = float(np.mean(b_values > 0))
    selected_b = (
        B_METHODS[1]
        if b_median
        >= float(b_rule["select_target_scaler_only_if_median_difference_at_least"])
        and b_fraction
        >= float(b_rule["and_fraction_participants_target_scaler_better_at_least"])
        else B_METHODS[0]
    )
    c_median = float(np.median(c_values))
    tie = float(
        config["representation_C_target_CSP_target_classifier"][
            "covariance_selection_rule"
        ]["practical_tie_absolute_median_difference"]
    )
    selected_c = C_METHODS[1] if c_median >= -tie else C_METHODS[0]
    summary = {
        "participant_count": len(subjects),
        "B_target_scaler_minus_source_scaler_median": b_median,
        "B_target_scaler_minus_source_scaler_q25": float(np.percentile(b_values, 25)),
        "B_target_scaler_minus_source_scaler_q75": float(np.percentile(b_values, 75)),
        "B_target_scaler_better_count": int(np.sum(b_values > 0)),
        "B_source_scaler_better_count": int(np.sum(b_values < 0)),
        "B_tie_count": int(np.sum(b_values == 0)),
        "selected_B_method": selected_b,
        "C_ledoit_wolf_minus_empirical_median": c_median,
        "C_ledoit_wolf_minus_empirical_q25": float(np.percentile(c_values, 25)),
        "C_ledoit_wolf_minus_empirical_q75": float(np.percentile(c_values, 75)),
        "C_ledoit_wolf_better_count": int(np.sum(c_values > 0)),
        "C_empirical_better_count": int(np.sum(c_values < 0)),
        "C_tie_count": int(np.sum(c_values == 0)),
        "selected_C_method": selected_c,
        "selection_used_evaluation_target_CSP_outcomes": False,
    }
    return paired, summary


def personalization_group_summary(
    participant_rows: Sequence[Mapping[str, object]],
    methods: Sequence[str],
    config: Mapping[str, Any],
    *,
    expected_subject_count: int,
) -> list[dict[str, object]]:
    """Summarize one primary and one QC value per participant and method."""
    bootstrap = config["group_inference"]
    output: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        primary = sorted(
            (
                row
                for row in participant_rows
                if row["method"] == method and not bool(row["qc_sensitivity"])
            ),
            key=lambda row: int(row["subject"]),
        )
        qc = {
            int(row["subject"]): row
            for row in participant_rows
            if row["method"] == method and bool(row["qc_sensitivity"])
        }
        if len(primary) != expected_subject_count or len(qc) != expected_subject_count:
            raise RuntimeError("Group summary lacks complete participant coverage.")
        scores = np.asarray(
            [float(row["primary_mean_scenario_test_run_balanced_accuracy"]) for row in primary]
        )
        qc_scores = np.asarray(
            [
                float(qc[int(row["subject"])]["primary_mean_scenario_test_run_balanced_accuracy"])
                for row in primary
            ]
        )
        fists = np.asarray([float(row["fists_recall"]) for row in primary])
        feet = np.asarray([float(row["feet_recall"]) for row in primary])
        lower, upper = bootstrap_median_interval(
            scores,
            resamples=int(bootstrap["bootstrap_resamples"]),
            confidence_level=float(bootstrap["bootstrap_confidence_level"]),
            random_seed=int(bootstrap["bootstrap_random_seed"]) + method_index,
        )
        output.append(
            {
                "method": method,
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
                "median_scenario_test_run_range": float(
                    np.median(
                        [
                            float(row["scenario_test_run_balanced_accuracy_range"])
                            for row in primary
                        ]
                    )
                ),
                "QC_cohort_median_balanced_accuracy": float(np.median(qc_scores)),
                "median_paired_QC_minus_primary_balanced_accuracy": float(
                    np.median(qc_scores - scores)
                ),
                "group_observational_unit": "participant",
            }
        )
    return output


def paired_personalization_comparison(
    participant_rows: Sequence[Mapping[str, object]],
    candidate: str,
    reference: str,
    config: Mapping[str, Any],
    *,
    expected_subject_count: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return participant-matched gains and the frozen paired summary."""
    lookup = {
        (int(row["subject"]), str(row["method"]), bool(row["qc_sensitivity"])): row
        for row in participant_rows
    }
    subjects = sorted(
        {
            int(row["subject"])
            for row in participant_rows
            if row["method"] == candidate and not bool(row["qc_sensitivity"])
        }
    )
    if len(subjects) != expected_subject_count:
        raise RuntimeError("Paired comparison lacks complete participant coverage.")
    rows: list[dict[str, object]] = []
    primary_gains: list[float] = []
    qc_gains: list[float] = []
    candidate_scores: list[float] = []
    fists: list[float] = []
    feet: list[float] = []
    for subject in subjects:
        candidate_row = lookup[(subject, candidate, False)]
        reference_row = lookup[(subject, reference, False)]
        candidate_score = float(
            candidate_row["primary_mean_scenario_test_run_balanced_accuracy"]
        )
        reference_score = float(
            reference_row["primary_mean_scenario_test_run_balanced_accuracy"]
        )
        gain = candidate_score - reference_score
        qc_gain = float(
            lookup[(subject, candidate, True)][
                "primary_mean_scenario_test_run_balanced_accuracy"
            ]
        ) - float(
            lookup[(subject, reference, True)][
                "primary_mean_scenario_test_run_balanced_accuracy"
            ]
        )
        rows.append(
            {
                "subject": subject,
                "candidate_method": candidate,
                "reference_method": reference,
                "reference_balanced_accuracy": reference_score,
                "candidate_balanced_accuracy": candidate_score,
                "candidate_minus_reference": gain,
                "QC_candidate_minus_reference": qc_gain,
            }
        )
        candidate_scores.append(candidate_score)
        primary_gains.append(gain)
        qc_gains.append(qc_gain)
        fists.append(float(candidate_row["fists_recall"]))
        feet.append(float(candidate_row["feet_recall"]))
    gains = np.asarray(primary_gains)
    sign = exact_sign_test_positive(gains)
    summary: dict[str, object] = {
        "comparison": f"{candidate}_minus_{reference}",
        "candidate_method": candidate,
        "reference_method": reference,
        "participant_count": expected_subject_count,
        "median_paired_difference": float(np.median(gains)),
        "q25_paired_difference": float(np.percentile(gains, 25)),
        "q75_paired_difference": float(np.percentile(gains, 75)),
        **sign,
        "improved_fraction": sign["improved_count"] / expected_subject_count,
        "candidate_median_balanced_accuracy": float(np.median(candidate_scores)),
        "candidate_median_fists_recall": float(np.median(fists)),
        "candidate_median_feet_recall": float(np.median(feet)),
        "QC_median_paired_difference": float(np.median(qc_gains)),
    }
    summary["convincing_paired_gain"] = convincing_paired_gain(summary, config)
    return rows, summary


def convincing_paired_gain(
    summary: Mapping[str, object], config: Mapping[str, Any]
) -> bool:
    """Apply every frozen practical criterion to one paired transition."""
    criteria = config["convincing_paired_gain_criteria"]
    fists = float(summary["candidate_median_fists_recall"])
    feet = float(summary["candidate_median_feet_recall"])
    return bool(
        float(summary["candidate_median_balanced_accuracy"])
        >= float(criteria["minimum_candidate_median_balanced_accuracy"])
        and float(summary["median_paired_difference"])
        >= float(criteria["minimum_median_paired_gain"])
        and float(summary["improved_fraction"])
        >= float(criteria["minimum_fraction_participants_improved"])
        and float(summary["one_sided_exact_sign_p"])
        <= float(criteria["maximum_one_sided_exact_sign_test_p"])
        and min(fists, feet) >= float(criteria["minimum_median_recall_each_class"])
        and abs(fists - feet)
        <= float(criteria["maximum_absolute_median_class_recall_difference"])
        and float(summary["QC_median_paired_difference"])
        >= float(criteria["minimum_QC_median_paired_gain"])
    )


def classify_personalization_geometry(
    b_minus_a: Mapping[str, object],
    c_minus_b: Mapping[str, object],
    c_minus_a: Mapping[str, object],
) -> str:
    """Apply the frozen decision/spatial/both/insufficient interpretation gate."""
    b = bool(b_minus_a["convincing_paired_gain"])
    c = bool(c_minus_b["convincing_paired_gain"])
    combined = bool(c_minus_a["convincing_paired_gain"])
    if b and c:
        return "both decision layer and spatial representation matter"
    if b:
        return "decision-boundary bottleneck"
    if c:
        return "spatial-representation bottleneck"
    if (
        combined
        and float(b_minus_a["median_paired_difference"]) > 0
        and float(c_minus_b["median_paired_difference"]) > 0
    ):
        return "both decision layer and spatial representation matter"
    return "one-run personalization is insufficient"


def csp_subspace_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Compare CSP filter spaces invariantly to sign, order, and basis rotation."""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.ndim != 2 or right.shape != left.shape or left.shape[1] != 4:
        raise ValueError("CSP subspaces must have matching (channels, 4) shape.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("CSP subspaces contain non-finite values.")
    angles = subspace_angles(left, right)
    return float(np.mean(np.cos(angles) ** 2))


def participant_csp_stability_row(
    subject: int,
    method: str,
    subspaces_by_calibration_run: Mapping[int, np.ndarray],
) -> dict[str, object]:
    """Aggregate the three pairwise target-CSP subspace similarities."""
    if method not in C_METHODS or set(subspaces_by_calibration_run) != set(RUNS):
        raise ValueError("CSP stability requires one target subspace per run.")
    pair_values: dict[tuple[int, int], float] = {}
    for first, second in ((6, 10), (6, 14), (10, 14)):
        pair_values[(first, second)] = csp_subspace_similarity(
            subspaces_by_calibration_run[first], subspaces_by_calibration_run[second]
        )
    mean = float(np.mean(list(pair_values.values())))
    return {
        "subject": subject,
        "method": method,
        "similarity_run_6_vs_10": pair_values[(6, 10)],
        "similarity_run_6_vs_14": pair_values[(6, 14)],
        "similarity_run_10_vs_14": pair_values[(10, 14)],
        "mean_pairwise_subspace_similarity": mean,
        "minimum_pairwise_subspace_similarity": float(min(pair_values.values())),
        "sign_order_and_rotation_invariant": True,
        "analysis_role": "post_primary_exploratory_stability",
    }
