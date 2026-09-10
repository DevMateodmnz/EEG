"""Leakage-safe zero-shot cross-subject motor-imagery decoding."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import Pipeline

from .decoding import (
    PROJECT_ROOT,
    RUNS,
    SubjectDecodingData,
    bootstrap_median_interval,
    build_candidate_pipeline,
    exact_sign_test_above_chance,
    load_decoding_config,
    load_subject_decoding_data,
    paired_wilcoxon,
    spectral_log_psd_features,
)
from .decoding_evaluation import load_evaluation_subject_data, load_final_decoding_config
from .preprocessing import DEFAULT_DATA_DIRECTORY
from .provenance import resolve_source_id


DEFAULT_CROSS_SUBJECT_CONFIG_PATH = PROJECT_ROOT / "config" / "cross_subject_decoding.json"
INITIAL_CROSS_SUBJECT_FREEZE_COMMIT = "74b49d0"
MODEL_NAMES = ("spectral_baseline_6", "csp_4_empirical")


@dataclass(frozen=True)
class SubjectFold:
    """One development split with a completely unseen test participant."""

    test_subject: int
    training_subjects: tuple[int, ...]


@dataclass
class FittedCrossSubjectModel:
    """A pipeline fitted without access to any future target participant."""

    model_name: str
    pipeline: Pipeline
    training_subjects: tuple[int, ...]
    training_trial_count: int
    training_trial_counts_by_subject: dict[int, int]
    training_source_hashes: dict[str, str]
    feature_names: tuple[str, ...]


@dataclass
class CrossSubjectTargetResult:
    """Trial predictions and participant/run summaries for one unseen target."""

    prediction_rows: list[dict[str, object]]
    run_rows: list[dict[str, object]]
    subject_row: dict[str, object]
    fit_audit_row: dict[str, object]


PipelineFactory = Callable[[str, Mapping[str, Any]], Pipeline]


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for immutable-method and artifact guards."""
    if not path.is_file() and str(path).replace("\\", "/").startswith("physionet/"):
        path = resolve_source_id(str(path), DEFAULT_DATA_DIRECTORY)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one tracked CSV artifact."""
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def load_cross_subject_config(
    path: Path = DEFAULT_CROSS_SUBJECT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the initial cross-subject policy and validate immutable boundaries."""
    resolved = path.expanduser().resolve()
    payload: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    validate_cross_subject_config(payload)
    return payload


def validate_cross_subject_config(config: Mapping[str, Any]) -> None:
    """Reject cohort, method, leakage, metric, or historical-file drift."""
    if int(config["schema_version"]) != 1:
        raise ValueError("Unsupported cross-subject decoding schema.")
    if config["study_stage"] != "initial_cross_subject_methodology_frozen":
        raise ValueError("Expected the initial cross-subject methodology freeze.")
    cohorts = config["cohorts"]
    development = list(cohorts["development_subjects"])
    requested = list(cohorts["evaluation_requested_subjects"])
    incompatible = list(cohorts["technically_incompatible_subjects"])
    eligible = list(cohorts["evaluation_eligible_subjects"])
    if development != list(range(1, 21)) or requested != list(range(21, 110)):
        raise ValueError("Cross-subject cohort identities drifted.")
    if incompatible != [88, 92, 100]:
        raise ValueError("Historical technical exclusions drifted.")
    if eligible != [subject for subject in requested if subject not in incompatible]:
        raise ValueError("Eligible evaluation participants are inconsistent.")
    if tuple(config["candidate_models"]) != MODEL_NAMES:
        raise ValueError("Cross-subject candidates must remain the two inherited models.")
    if config["development_cross_validation"]["method"] != "leave_one_subject_out":
        raise ValueError("Cross-subject development must be leave-one-subject-out.")
    if config["metrics"]["group_observational_unit"] != "participant":
        raise ValueError("Cross-subject group summaries must use participants.")
    if config["metrics"]["primary"] != (
        "unweighted_mean_of_three_run_balanced_accuracies_per_evaluation_subject"
    ):
        raise ValueError("Cross-subject primary metric drifted.")
    forbidden = config["zero_shot_leakage_guards"]
    if not all(
        forbidden[key] is False
        for key in (
            "target_subject_feature_selection",
            "target_subject_hyperparameter_selection",
            "target_subject_centering",
            "target_subject_scaling",
            "target_subject_covariance_alignment",
            "target_subject_CSP_adaptation",
            "target_subject_transductive_normalization",
            "target_subject_unlabeled_adaptation",
        )
    ):
        raise ValueError("A target-participant adaptation route was enabled.")
    if not all(
        forbidden[key] is True
        for key in (
            "CSP_fit_training_subjects_only",
            "scaler_fit_training_subjects_only",
            "LDA_fit_training_subjects_only",
        )
    ):
        raise ValueError("A learned stage is not restricted to training participants.")
    for record in config["historical_inputs"].values():
        historical_path = PROJECT_ROOT / record["path"]
        if file_sha256(historical_path) != record["sha256"]:
            raise RuntimeError(f"Historical cross-subject input drift: {historical_path}.")


def leave_one_subject_out_folds(subjects: Sequence[int]) -> list[SubjectFold]:
    """Return exact participant-isolated development folds."""
    ordered = tuple(int(subject) for subject in subjects)
    if ordered != tuple(range(1, 21)) or len(set(ordered)) != 20:
        raise ValueError("Development LOSO requires ordered unique subjects 1–20.")
    folds = [
        SubjectFold(
            test_subject=test_subject,
            training_subjects=tuple(subject for subject in ordered if subject != test_subject),
        )
        for test_subject in ordered
    ]
    if any(
        fold.test_subject in fold.training_subjects
        or len(fold.training_subjects) != 19
        or set(fold.training_subjects) | {fold.test_subject} != set(ordered)
        for fold in folds
    ):
        raise RuntimeError("Participant overlap or omission in LOSO folds.")
    return folds


def load_cross_subject_data(
    subject: int,
    cross_config: Mapping[str, Any],
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
) -> SubjectDecodingData:
    """Load exact inherited features for one development or evaluation participant."""
    development = set(cross_config["cohorts"]["development_subjects"])
    evaluation = set(cross_config["cohorts"]["evaluation_eligible_subjects"])
    if subject in development:
        initial = load_decoding_config()
        return load_subject_decoding_data(subject, initial, data_directory=data_directory)
    if subject in evaluation:
        _, initial = load_final_decoding_config()
        return load_evaluation_subject_data(subject, initial, data_directory=data_directory)
    raise ValueError(f"Subject {subject} is outside frozen cross-subject cohorts.")


def _validate_dataset_collection(datasets: Sequence[SubjectDecodingData]) -> None:
    if not datasets:
        raise ValueError("Cross-subject fitting requires training datasets.")
    subjects = [dataset.subject for dataset in datasets]
    if len(subjects) != len(set(subjects)):
        raise ValueError("A participant occurs more than once in training data.")
    reference = datasets[0]
    for dataset in datasets:
        if (
            dataset.channel_names != reference.channel_names
            or not np.isclose(dataset.sampling_frequency_hz, reference.sampling_frequency_hz)
            or dataset.fixed_task_data_volts.shape[1:] != reference.fixed_task_data_volts.shape[1:]
            or dataset.csp_task_data_volts.shape[1:] != reference.csp_task_data_volts.shape[1:]
        ):
            raise RuntimeError("Cross-subject channel, sampling, or trial shapes differ.")
        if set(np.unique(dataset.labels)) != {0, 1} or set(np.unique(dataset.runs)) != set(RUNS):
            raise RuntimeError("A cross-subject dataset lacks a target class or run.")
        if any(identity.subject != dataset.subject for identity in dataset.identities):
            raise RuntimeError("Trial provenance contains the wrong participant.")


def _model_input(
    dataset: SubjectDecodingData,
    model_name: str,
    within_config: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if model_name == "spectral_baseline_6":
        features, names = spectral_log_psd_features(
            dataset,
            within_config["candidate_models"][model_name],
            within_config["fixed_feature_methods"]["spectral"],
        )
        return features, tuple(names)
    if model_name == "csp_4_empirical":
        return dataset.csp_task_data_volts, tuple(
            f"CSP{index + 1}_log_average_power"
            for index in range(
                int(within_config["candidate_models"][model_name]["csp_components"])
            )
        )
    raise ValueError(f"Unknown frozen cross-subject model: {model_name}.")


def fit_cross_subject_model(
    training_datasets: Sequence[SubjectDecodingData],
    model_name: str,
    within_config: Mapping[str, Any],
    *,
    pipeline_factory: PipelineFactory = build_candidate_pipeline,
) -> FittedCrossSubjectModel:
    """Fit CSP/scaler/LDA using only the explicitly supplied participants."""
    _validate_dataset_collection(training_datasets)
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Model {model_name} is outside the frozen candidate family.")
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    trial_counts: dict[int, int] = {}
    source_hashes: dict[str, str] = {}
    for dataset in training_datasets:
        model_input, names = _model_input(dataset, model_name, within_config)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Cross-subject feature names differ between participants.")
        arrays.append(model_input)
        labels.append(dataset.labels)
        trial_counts[dataset.subject] = int(dataset.labels.size)
        for identity in dataset.identities:
            previous = source_hashes.setdefault(identity.source_file, identity.source_sha256)
            if previous != identity.source_sha256:
                raise RuntimeError("One training source path has inconsistent hashes.")
    training_input = np.concatenate(arrays, axis=0)
    training_labels = np.concatenate(labels)
    if set(training_labels) != {0, 1} or not np.isfinite(training_input).all():
        raise RuntimeError("Cross-subject training arrays are invalid.")
    pipeline = pipeline_factory(model_name, within_config)
    pipeline.fit(training_input, training_labels)
    scaler = pipeline.named_steps["scaler"]
    sample_count = int(training_labels.size)
    if int(np.max(np.atleast_1d(scaler.n_samples_seen_))) != sample_count:
        raise RuntimeError("Scaler did not fit the complete training-participant trial set.")
    if set(pipeline.named_steps["lda"].classes_) != {0, 1}:
        raise RuntimeError("LDA training classes are incomplete.")
    return FittedCrossSubjectModel(
        model_name=model_name,
        pipeline=pipeline,
        training_subjects=tuple(sorted(trial_counts)),
        training_trial_count=sample_count,
        training_trial_counts_by_subject=trial_counts,
        training_source_hashes=source_hashes,
        feature_names=feature_names or (),
    )


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fists_fists, fists_feet, feet_fists, feet_feet = (int(value) for value in matrix.ravel())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fists_recall": float(
            recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
        ),
        "feet_recall": float(
            recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
        ),
        "true_fists_predicted_fists": fists_fists,
        "true_fists_predicted_feet": fists_feet,
        "true_feet_predicted_fists": feet_fists,
        "true_feet_predicted_feet": feet_feet,
        "trial_count": int(y_true.size),
    }


def predict_cross_subject_target(
    fitted: FittedCrossSubjectModel,
    target: SubjectDecodingData,
    within_config: Mapping[str, Any],
) -> CrossSubjectTargetResult:
    """Predict one unseen participant without fitting or target-distribution adaptation."""
    if target.subject in fitted.training_subjects:
        raise ValueError("Target participant occurs in the fitted training cohort.")
    _validate_dataset_collection([target])
    target_input, feature_names = _model_input(target, fitted.model_name, within_config)
    if feature_names != fitted.feature_names:
        raise RuntimeError("Target feature definitions differ from fitted definitions.")
    predictions = np.asarray(fitted.pipeline.predict(target_input), dtype=int)
    decision_scores = np.asarray(fitted.pipeline.decision_function(target_input), dtype=float)
    if (
        predictions.shape != target.labels.shape
        or decision_scores.shape != target.labels.shape
        or not np.isfinite(decision_scores).all()
    ):
        raise RuntimeError("Zero-shot target predictions are invalid.")
    training_subject_text = "/".join(str(subject) for subject in fitted.training_subjects)
    prediction_rows: list[dict[str, object]] = []
    for index, (prediction, score) in enumerate(
        zip(predictions, decision_scores, strict=True)
    ):
        identity = target.identities[index]
        true_label = int(target.labels[index])
        prediction_rows.append(
            {
                "subject": target.subject,
                "run": identity.run,
                "run_trial_index": identity.run_trial_index,
                "annotation_index": identity.annotation_index,
                "trial_key": identity.key,
                "true_condition": identity.semantic_condition,
                "true_label": true_label,
                "predicted_condition": (
                    "both_fists_imagery" if prediction == 0 else "both_feet_imagery"
                ),
                "predicted_label": int(prediction),
                "decision_score_feet": float(score),
                "correct": bool(prediction == true_label),
                "model": fitted.model_name,
                "training_subjects": training_subject_text,
                "zero_shot": True,
                "target_adaptation_used": False,
                "source_file": identity.source_file,
                "source_sha256": identity.source_sha256,
            }
        )
    run_rows: list[dict[str, object]] = []
    for run in RUNS:
        selected = target.runs == run
        metrics = _metric_row(target.labels[selected], predictions[selected])
        run_rows.append(
            {
                "subject": target.subject,
                "model": fitted.model_name,
                "test_run": run,
                **metrics,
                "training_subjects": training_subject_text,
                "target_adaptation_used": False,
            }
        )
    pooled = _metric_row(target.labels, predictions)
    primary = float(np.mean([float(row["balanced_accuracy"]) for row in run_rows]))
    subject_row: dict[str, object] = {
        "subject": target.subject,
        "model": fitted.model_name,
        "primary_mean_run_balanced_accuracy": primary,
        "balanced_accuracy_run_6": run_rows[0]["balanced_accuracy"],
        "balanced_accuracy_run_10": run_rows[1]["balanced_accuracy"],
        "balanced_accuracy_run_14": run_rows[2]["balanced_accuracy"],
        "pooled_balanced_accuracy": pooled["balanced_accuracy"],
        "pooled_accuracy": pooled["accuracy"],
        "fists_recall": pooled["fists_recall"],
        "feet_recall": pooled["feet_recall"],
        "true_fists_predicted_fists": pooled["true_fists_predicted_fists"],
        "true_fists_predicted_feet": pooled["true_fists_predicted_feet"],
        "true_feet_predicted_fists": pooled["true_feet_predicted_fists"],
        "true_feet_predicted_feet": pooled["true_feet_predicted_feet"],
        "trial_count": pooled["trial_count"],
        "group_observational_unit": "participant",
        "zero_shot": True,
        "target_adaptation_used": False,
    }
    counts_text = "/".join(
        f"S{subject:03d}:{fitted.training_trial_counts_by_subject[subject]}"
        for subject in fitted.training_subjects
    )
    fit_audit = {
        "target_subject": target.subject,
        "model": fitted.model_name,
        "training_subjects": training_subject_text,
        "training_subject_count": len(fitted.training_subjects),
        "training_trial_count": fitted.training_trial_count,
        "training_trial_counts_by_subject": counts_text,
        "target_subject_absent_from_training": target.subject not in fitted.training_subjects,
        "csp_fit_subjects": (
            training_subject_text if fitted.model_name.startswith("csp_") else "not_applicable"
        ),
        "scaler_fit_subjects": training_subject_text,
        "lda_fit_subjects": training_subject_text,
        "target_centering_used": False,
        "target_scaling_fit_used": False,
        "target_covariance_alignment_used": False,
        "target_CSP_adaptation_used": False,
        "target_transductive_normalization_used": False,
        "target_unlabeled_adaptation_used": False,
        "feature_names": "/".join(fitted.feature_names),
    }
    return CrossSubjectTargetResult(prediction_rows, run_rows, subject_row, fit_audit)


def summarize_cross_subject_group(
    subject_rows: Sequence[Mapping[str, object]],
    model_name: str,
    config: Mapping[str, Any],
    *,
    expected_subject_count: int,
) -> dict[str, object]:
    """Summarize exactly one primary balanced accuracy per unseen participant."""
    selected = [row for row in subject_rows if row["model"] == model_name]
    if (
        len(selected) != expected_subject_count
        or len({int(row["subject"]) for row in selected}) != expected_subject_count
        or any(row["group_observational_unit"] != "participant" for row in selected)
    ):
        raise RuntimeError("Cross-subject summary lacks one row per participant.")
    scores = np.asarray(
        [float(row["primary_mean_run_balanced_accuracy"]) for row in selected]
    )
    fists = np.asarray([float(row["fists_recall"]) for row in selected])
    feet = np.asarray([float(row["feet_recall"]) for row in selected])
    bootstrap = config["group_inference"]["bootstrap_median_confidence_interval"]
    lower, upper = bootstrap_median_interval(
        scores,
        resamples=int(bootstrap["resamples"]),
        confidence_level=float(bootstrap["confidence_level"]),
        random_seed=int(bootstrap["random_seed"]),
    )
    sign = exact_sign_test_above_chance(scores)
    return {
        "model": model_name,
        "participant_count": expected_subject_count,
        "median_balanced_accuracy": float(np.median(scores)),
        "bootstrap_median_ci_lower": lower,
        "bootstrap_median_ci_upper": upper,
        "q25_balanced_accuracy": float(np.percentile(scores, 25)),
        "q75_balanced_accuracy": float(np.percentile(scores, 75)),
        "minimum_balanced_accuracy": float(np.min(scores)),
        "maximum_balanced_accuracy": float(np.max(scores)),
        "above_0_5_count": int(np.sum(scores > 0.5)),
        "above_0_5_fraction": float(np.mean(scores > 0.5)),
        "at_or_above_0_6_count": int(np.sum(scores >= 0.6)),
        "at_or_above_0_6_fraction": float(np.mean(scores >= 0.6)),
        "at_or_above_0_7_count": int(np.sum(scores >= 0.7)),
        "at_or_above_0_7_fraction": float(np.mean(scores >= 0.7)),
        "median_fists_recall": float(np.median(fists)),
        "median_feet_recall": float(np.median(feet)),
        **sign,
        "group_observational_unit": "participant",
    }


def paired_cross_subject_models(
    subject_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return matched participant rows and one supporting paired comparison."""
    lookup = {
        (int(row["subject"]), str(row["model"])): row for row in subject_rows
    }
    subjects = sorted({int(row["subject"]) for row in subject_rows})
    rows: list[dict[str, object]] = []
    for subject in subjects:
        baseline = lookup[(subject, MODEL_NAMES[0])]
        csp = lookup[(subject, MODEL_NAMES[1])]
        baseline_score = float(baseline["primary_mean_run_balanced_accuracy"])
        csp_score = float(csp["primary_mean_run_balanced_accuracy"])
        rows.append(
            {
                "subject": subject,
                "spectral_balanced_accuracy": baseline_score,
                "csp_balanced_accuracy": csp_score,
                "csp_minus_spectral": csp_score - baseline_score,
            }
        )
    differences = np.asarray([float(row["csp_minus_spectral"]) for row in rows])
    baseline_values = np.asarray([float(row["spectral_balanced_accuracy"]) for row in rows])
    csp_values = np.asarray([float(row["csp_balanced_accuracy"]) for row in rows])
    summary = {
        "participant_count": len(rows),
        "median_CSP_minus_spectral": float(np.median(differences)),
        "q25_CSP_minus_spectral": float(np.percentile(differences, 25)),
        "q75_CSP_minus_spectral": float(np.percentile(differences, 75)),
        "CSP_better_count": int(np.sum(differences > 0)),
        "spectral_better_count": int(np.sum(differences < 0)),
        "tie_count": int(np.sum(differences == 0)),
        "CSP_better_fraction": float(np.mean(differences > 0)),
        "spectral_better_fraction": float(np.mean(differences < 0)),
        "paired_wilcoxon_two_sided_p": paired_wilcoxon(csp_values, baseline_values),
        "group_observational_unit": "participant",
    }
    return rows, summary


def classify_transfer(summary: Mapping[str, object], config: Mapping[str, Any]) -> str:
    """Apply the outcome-independent zero-shot transfer categories."""
    median = float(summary["median_balanced_accuracy"])
    fraction = float(summary["above_0_5_fraction"])
    fists = float(summary["median_fists_recall"])
    feet = float(summary["median_feet_recall"])
    criteria = config["evaluation_success_criteria"]
    clear = criteria["clear_zero_shot_transfer"]
    if (
        median >= float(clear["minimum_median_balanced_accuracy"])
        and fraction >= float(clear["minimum_fraction_subjects_above_0_5"])
        and min(fists, feet) >= float(clear["minimum_median_recall_each_class"])
        and abs(fists - feet)
        <= float(clear["maximum_absolute_median_class_recall_difference"])
    ):
        return "clear zero-shot transfer"
    modest = criteria["modest_zero_shot_transfer"]
    if (
        median >= float(modest["minimum_median_balanced_accuracy"])
        and fraction >= float(modest["minimum_fraction_subjects_above_0_5"])
        and min(fists, feet) >= float(modest["minimum_median_recall_each_class"])
        and abs(fists - feet)
        <= float(modest["maximum_absolute_median_class_recall_difference"])
    ):
        return "modest zero-shot transfer"
    return "no convincing zero-shot transfer"


def classify_paired_transfer(summary: Mapping[str, object], config: Mapping[str, Any]) -> str:
    """Apply the pre-outcome CSP-versus-spectral transfer categories."""
    difference = float(summary["median_CSP_minus_spectral"])
    csp_fraction = float(summary["CSP_better_fraction"])
    spectral_fraction = float(summary["spectral_better_fraction"])
    criteria = config["paired_model_criteria"]
    if (
        difference >= float(criteria["CSP_transfers_better"]["minimum_median_CSP_minus_spectral"])
        and csp_fraction >= float(criteria["CSP_transfers_better"]["minimum_fraction_CSP_better"])
    ):
        return "CSP transfers better"
    if (
        difference <= float(criteria["spectral_transfers_better"]["maximum_median_CSP_minus_spectral"])
        and spectral_fraction
        >= float(criteria["spectral_transfers_better"]["minimum_fraction_spectral_better"])
    ):
        return "spectral transfers better"
    if abs(difference) <= float(criteria["comparable"]["maximum_absolute_median_difference"]):
        return "comparable transfer"
    return "mixed model transfer"
