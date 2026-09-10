"""Strict zero-shot Riemannian covariance decoding operations.

Each trial is converted independently into a regularized SPD covariance.  All
subsequent learned geometry--the Riemannian reference, tangent map, scaling, and
linear classifier--is fitted only from the explicitly supplied training people.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pyriemann.tangentspace import TangentSpace
from scipy.stats import binomtest
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .decoding import (
    PROJECT_ROOT,
    RUNS,
    SubjectDecodingData,
    TrialIdentity,
    bootstrap_median_interval,
)


DEFAULT_RIEMANNIAN_CONFIG_PATH = PROJECT_ROOT / "config" / "riemannian_decoding.json"
INITIAL_RIEMANNIAN_FREEZE_COMMIT = "131e601"
MODEL_NAMES = ("shrinkage_lda", "l2_logistic_regression")


@dataclass(frozen=True)
class SubjectFold:
    """One development split with one entirely unseen participant."""

    test_subject: int
    training_subjects: tuple[int, ...]


@dataclass
class SubjectCovarianceData:
    """Trial-level SPD covariances and immutable provenance for one participant."""

    subject: int
    covariances: np.ndarray
    labels: np.ndarray
    runs: np.ndarray
    identities: list[TrialIdentity]
    qc_candidates: np.ndarray


@dataclass
class FittedRiemannianModel:
    """A tangent-space pipeline fitted without target-participant input."""

    model_name: str
    pipeline: Pipeline
    training_subjects: tuple[int, ...]
    training_trial_count: int
    training_trial_counts_by_subject: dict[int, int]
    training_source_hashes: dict[str, str]


@dataclass
class RiemannianTargetResult:
    """Predictions, participant/run summaries, and learned-stage provenance."""

    prediction_rows: list[dict[str, object]]
    run_rows: list[dict[str, object]]
    subject_rows: list[dict[str, object]]
    fit_audit_row: dict[str, object]


def file_sha256(path: Path) -> str:
    """Return the digest used to guard frozen inputs and artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_riemannian_config(
    path: Path = DEFAULT_RIEMANNIAN_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the pre-development Riemannian policy."""
    resolved = path.expanduser().resolve()
    config: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    validate_riemannian_config(config)
    return config


def validate_riemannian_config(config: Mapping[str, Any]) -> None:
    """Reject cohort, geometry, leakage, metric, or historical-result drift."""
    if int(config["schema_version"]) != 1:
        raise ValueError("Unsupported Riemannian decoding schema.")
    if config["study_stage"] != "initial_riemannian_methodology_frozen":
        raise ValueError("Expected the initial Riemannian methodology freeze.")
    cohorts = config["cohorts"]
    if list(cohorts["development_subjects"]) != list(range(1, 21)):
        raise ValueError("Riemannian development subjects drifted.")
    if list(cohorts["evaluation_requested_subjects"]) != list(range(21, 110)):
        raise ValueError("Riemannian requested evaluation subjects drifted.")
    if list(cohorts["technically_incompatible_subjects"]) != [88, 92, 100]:
        raise ValueError("Technical exclusions drifted.")
    expected_eligible = [subject for subject in range(21, 110) if subject not in {88, 92, 100}]
    if list(cohorts["evaluation_eligible_subjects"]) != expected_eligible:
        raise ValueError("Riemannian eligible evaluation subjects drifted.")
    if not config["evaluation_locked"]:
        raise ValueError("Development requires a locked evaluation cohort.")
    target = config["target"]
    if target["runs"] != list(RUNS) or target["task_interval_seconds"] != [1.0, 3.0]:
        raise ValueError("Target runs or time interval drifted.")
    representation = config["frozen_EEG_representation"]
    if representation["passband_hz"] != [8.0, 30.0] or representation["channels"] != 64:
        raise ValueError("Frozen EEG representation drifted.")
    covariance = config["covariance"]
    if covariance["estimator"] != "sklearn.covariance.LedoitWolf" or not covariance["trace_normalization"]:
        raise ValueError("The frozen covariance estimator drifted.")
    tangent = config["tangent_space"]
    if (
        tangent["library"] != "pyriemann==0.12"
        or tangent["reference_mean_metric"] != "riemann"
        or tangent["mapping_metric"] != "riemann"
        or int(tangent["feature_count"]) != 2080
        or bool(tangent["tsupdate"])
    ):
        raise ValueError("The frozen tangent-space method drifted.")
    if tuple(config["candidate_linear_classifiers"]) != MODEL_NAMES:
        raise ValueError("Riemannian classifier candidates drifted.")
    if config["development_cross_validation"]["method"] != "leave_one_subject_out":
        raise ValueError("Riemannian development must be leave-one-subject-out.")
    if config["metrics"]["group_observational_unit"] != "participant":
        raise ValueError("Riemannian group summaries must use participants.")
    guards = config["zero_shot_leakage_guards"]
    if not all(guards[key] for key in (
        "reference_mean_fit_training_subjects_only",
        "tangent_space_basis_fit_training_subjects_only",
        "scaler_fit_training_subjects_only",
        "classifier_fit_training_subjects_only",
    )):
        raise ValueError("A learned Riemannian stage is not training-only.")
    if any(guards[key] for key in (
        "target_subject_feature_selection",
        "target_subject_hyperparameter_selection",
        "target_subject_centering",
        "target_subject_scaling",
        "target_subject_covariance_alignment",
        "target_subject_transductive_normalization",
        "target_subject_unlabeled_adaptation",
        "target_tangent_space_update",
    )):
        raise ValueError("A prohibited target-adaptation route was enabled.")
    for record in config["historical_inputs"].values():
        path = PROJECT_ROOT / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Historical Riemannian input drift: {path}.")


def leave_one_subject_out_folds(subjects: Sequence[int]) -> list[SubjectFold]:
    """Return exact participant-isolated development folds."""
    ordered = tuple(int(subject) for subject in subjects)
    if ordered != tuple(range(1, 21)) or len(set(ordered)) != 20:
        raise ValueError("Riemannian LOSO requires ordered unique subjects 1–20.")
    return [
        SubjectFold(subject, tuple(candidate for candidate in ordered if candidate != subject))
        for subject in ordered
    ]


def trial_covariances(trials: np.ndarray) -> np.ndarray:
    """Estimate independently trace-normalized Ledoit--Wolf SPD covariances.

    Parameters
    ----------
    trials
        EEG in volts with shape ``(trials, channels, samples)``.  Each covariance is
        fitted from its own 320 temporal observations; no participant or label is
        involved in this fixed representation transformation.
    """
    values = np.asarray(trials, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (64, 320) or not np.isfinite(values).all():
        raise ValueError("Riemannian trial input must be finite (trials, 64, 320).")
    output = np.empty((values.shape[0], 64, 64), dtype=float)
    for index, trial in enumerate(values):
        covariance = LedoitWolf(assume_centered=False, store_precision=False).fit(trial.T).covariance_
        covariance = (covariance + covariance.T) / 2.0
        trace = float(np.trace(covariance))
        if not np.isfinite(trace) or trace <= 0.0:
            raise RuntimeError("Ledoit--Wolf covariance has invalid trace.")
        covariance /= trace
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] <= 0.0 or not np.isfinite(eigenvalues).all():
            raise RuntimeError("Ledoit--Wolf covariance is not SPD after normalization.")
        output[index] = covariance
    return output


def covariance_data_from_dataset(dataset: SubjectDecodingData) -> SubjectCovarianceData:
    """Convert a frozen 8--30 Hz trial tensor into aligned SPD trial covariances."""
    covariances = trial_covariances(dataset.csp_task_data_volts)
    count = dataset.labels.size
    if (
        covariances.shape != (count, 64, 64)
        or len(dataset.identities) != count
        or dataset.runs.size != count
        or dataset.qc_candidates.size != count
        or len({identity.key for identity in dataset.identities}) != count
    ):
        raise RuntimeError("Riemannian covariance provenance is misaligned.")
    return SubjectCovarianceData(
        dataset.subject,
        covariances,
        np.asarray(dataset.labels, dtype=int).copy(),
        np.asarray(dataset.runs, dtype=int).copy(),
        list(dataset.identities),
        np.asarray(dataset.qc_candidates, dtype=bool).copy(),
    )


def _validate_covariance_collection(datasets: Sequence[SubjectCovarianceData]) -> None:
    if not datasets:
        raise ValueError("Riemannian fitting requires at least one training participant.")
    subjects = [dataset.subject for dataset in datasets]
    if len(subjects) != len(set(subjects)):
        raise ValueError("A Riemannian training participant occurs more than once.")
    for dataset in datasets:
        count = dataset.labels.size
        if (
            dataset.covariances.shape != (count, 64, 64)
            or dataset.runs.size != count
            or len(dataset.identities) != count
            or dataset.qc_candidates.size != count
            or set(np.unique(dataset.labels)) != {0, 1}
            or set(np.unique(dataset.runs)) != set(RUNS)
            or any(identity.subject != dataset.subject for identity in dataset.identities)
        ):
            raise RuntimeError("Riemannian training data are invalid or misaligned.")


def _classifier(model_name: str, config: Mapping[str, Any]) -> object:
    if model_name == "shrinkage_lda":
        settings = config["candidate_linear_classifiers"][model_name]
        return LinearDiscriminantAnalysis(
            solver=settings["solver"], shrinkage=settings["shrinkage"], priors=settings["priors"]
        )
    if model_name == "l2_logistic_regression":
        settings = config["candidate_linear_classifiers"][model_name]
        return LogisticRegression(
            C=float(settings["C"]), solver=settings["solver"],
            max_iter=int(settings["max_iter"]), class_weight=settings["class_weight"],
        )
    raise ValueError(f"Unknown frozen Riemannian model: {model_name}.")


def fit_riemannian_model(
    training_datasets: Sequence[SubjectCovarianceData],
    model_name: str,
    config: Mapping[str, Any],
) -> FittedRiemannianModel:
    """Fit reference, tangent transform, scaling, and classifier on training people only."""
    _validate_covariance_collection(training_datasets)
    if model_name not in MODEL_NAMES:
        raise ValueError("Model is outside the frozen Riemannian candidate set.")
    arrays = [dataset.covariances for dataset in training_datasets]
    labels = [dataset.labels for dataset in training_datasets]
    training_covariances = np.concatenate(arrays, axis=0)
    training_labels = np.concatenate(labels)
    if training_covariances.shape[0] != training_labels.size or not np.isfinite(training_covariances).all():
        raise RuntimeError("Riemannian training arrays are invalid.")
    tangent = TangentSpace(metric="riemann", tsupdate=False)
    pipeline = Pipeline([
        ("tangent", tangent),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("classifier", _classifier(model_name, config)),
    ])
    pipeline.fit(training_covariances, training_labels)
    fitted_tangent = pipeline.named_steps["tangent"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    if (
        fitted_tangent.reference_.shape != (64, 64)
        or not np.isfinite(fitted_tangent.reference_).all()
        or int(np.max(np.atleast_1d(scaler.n_samples_seen_))) != training_labels.size
        or set(classifier.classes_) != {0, 1}
    ):
        raise RuntimeError("A learned Riemannian stage has invalid training state.")
    source_hashes: dict[str, str] = {}
    counts: dict[int, int] = {}
    for dataset in training_datasets:
        counts[dataset.subject] = int(dataset.labels.size)
        for identity in dataset.identities:
            previous = source_hashes.setdefault(identity.source_file, identity.source_sha256)
            if previous != identity.source_sha256:
                raise RuntimeError("One Riemannian source path has inconsistent hashes.")
    return FittedRiemannianModel(
        model_name,
        pipeline,
        tuple(sorted(counts)),
        int(training_labels.size),
        counts,
        source_hashes,
    )


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fists_fists, fists_feet, feet_fists, feet_feet = (int(value) for value in matrix.ravel())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fists_recall": float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "feet_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "true_fists_predicted_fists": fists_fists,
        "true_fists_predicted_feet": fists_feet,
        "true_feet_predicted_fists": feet_fists,
        "true_feet_predicted_feet": feet_feet,
        "trial_count": int(y_true.size),
    }


def _summary_rows_for_mask(
    target: SubjectCovarianceData,
    predictions: np.ndarray,
    fitted: FittedRiemannianModel,
    *,
    qc_sensitivity: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mask = ~target.qc_candidates if qc_sensitivity else np.ones(target.labels.size, dtype=bool)
    if not np.any(mask):
        raise RuntimeError("QC sensitivity removed every target trial.")
    training_subject_text = "/".join(str(subject) for subject in fitted.training_subjects)
    run_rows: list[dict[str, object]] = []
    for run in RUNS:
        selected = mask & (target.runs == run)
        if set(target.labels[selected]) != {0, 1}:
            raise RuntimeError("QC sensitivity leaves a run without both target classes.")
        run_rows.append({
            "subject": target.subject,
            "model": fitted.model_name,
            "test_run": run,
            **_metric_row(target.labels[selected], predictions[selected]),
            "qc_sensitivity": qc_sensitivity,
            "training_subjects": training_subject_text,
            "target_adaptation_used": False,
        })
    pooled = _metric_row(target.labels[mask], predictions[mask])
    run_scores = np.asarray([float(row["balanced_accuracy"]) for row in run_rows])
    subject_row = {
        "subject": target.subject,
        "model": fitted.model_name,
        "primary_mean_run_balanced_accuracy": float(np.mean(run_scores)),
        "balanced_accuracy_run_6": run_rows[0]["balanced_accuracy"],
        "balanced_accuracy_run_10": run_rows[1]["balanced_accuracy"],
        "balanced_accuracy_run_14": run_rows[2]["balanced_accuracy"],
        "participant_run_range": float(np.max(run_scores) - np.min(run_scores)),
        "pooled_balanced_accuracy": pooled["balanced_accuracy"],
        "pooled_accuracy": pooled["accuracy"],
        "fists_recall": pooled["fists_recall"],
        "feet_recall": pooled["feet_recall"],
        "true_fists_predicted_fists": pooled["true_fists_predicted_fists"],
        "true_fists_predicted_feet": pooled["true_fists_predicted_feet"],
        "true_feet_predicted_fists": pooled["true_feet_predicted_fists"],
        "true_feet_predicted_feet": pooled["true_feet_predicted_feet"],
        "trial_count": pooled["trial_count"],
        "qc_sensitivity": qc_sensitivity,
        "group_observational_unit": "participant",
        "zero_shot": True,
        "target_adaptation_used": False,
    }
    return run_rows, subject_row


def predict_riemannian_target(
    fitted: FittedRiemannianModel,
    target: SubjectCovarianceData,
) -> RiemannianTargetResult:
    """Apply an already-fitted pipeline to one unseen participant exactly once per trial."""
    if target.subject in fitted.training_subjects:
        raise ValueError("Target participant occurs in the fitted Riemannian training cohort.")
    _validate_covariance_collection([target])
    predictions = np.asarray(fitted.pipeline.predict(target.covariances), dtype=int)
    scores = np.asarray(fitted.pipeline.decision_function(target.covariances), dtype=float)
    if predictions.shape != target.labels.shape or scores.shape != target.labels.shape:
        raise RuntimeError("Riemannian target predictions have invalid shape.")
    training_subject_text = "/".join(str(subject) for subject in fitted.training_subjects)
    prediction_rows: list[dict[str, object]] = []
    for index, (prediction, score) in enumerate(zip(predictions, scores, strict=True)):
        identity = target.identities[index]
        prediction_rows.append({
            "subject": target.subject,
            "run": identity.run,
            "run_trial_index": identity.run_trial_index,
            "annotation_index": identity.annotation_index,
            "trial_key": identity.key,
            "true_condition": identity.semantic_condition,
            "true_label": int(target.labels[index]),
            "predicted_condition": "both_fists_imagery" if prediction == 0 else "both_feet_imagery",
            "predicted_label": int(prediction),
            "decision_score_feet": float(score),
            "correct": bool(prediction == target.labels[index]),
            "model": fitted.model_name,
            "training_subjects": training_subject_text,
            "zero_shot": True,
            "target_adaptation_used": False,
            "qc_candidate": bool(target.qc_candidates[index]),
            "source_file": identity.source_file,
            "source_sha256": identity.source_sha256,
        })
    primary_runs, primary_subject = _summary_rows_for_mask(target, predictions, fitted, qc_sensitivity=False)
    qc_runs, qc_subject = _summary_rows_for_mask(target, predictions, fitted, qc_sensitivity=True)
    counts_text = "/".join(
        f"S{subject:03d}:{fitted.training_trial_counts_by_subject[subject]}"
        for subject in fitted.training_subjects
    )
    audit = {
        "target_subject": target.subject,
        "model": fitted.model_name,
        "training_subjects": training_subject_text,
        "training_subject_count": len(fitted.training_subjects),
        "training_trial_count": fitted.training_trial_count,
        "training_trial_counts_by_subject": counts_text,
        "target_subject_absent_from_training": True,
        "covariance_estimator": "LedoitWolf_per_trial_only",
        "reference_mean_fit_subjects": training_subject_text,
        "tangent_basis_fit_subjects": training_subject_text,
        "scaler_fit_subjects": training_subject_text,
        "classifier_fit_subjects": training_subject_text,
        "test_EEG_used_for_reference_fit": False,
        "test_EEG_used_for_tangent_basis_fit": False,
        "test_EEG_used_for_scaler_fit": False,
        "test_EEG_used_for_classifier_fit": False,
        "test_labels_used_for_fitting": False,
        "target_centering_used": False,
        "target_scaling_fit_used": False,
        "target_covariance_alignment_used": False,
        "target_transductive_normalization_used": False,
        "target_tangent_space_update_used": False,
        "target_unlabeled_adaptation_used": False,
        "tangent_feature_count": 2080,
    }
    return RiemannianTargetResult(prediction_rows, primary_runs + qc_runs, [primary_subject, qc_subject], audit)


def summarize_riemannian_group(
    subject_rows: Sequence[Mapping[str, object]],
    model_name: str,
    config: Mapping[str, Any],
    *,
    expected_subject_count: int,
    qc_sensitivity: bool = False,
) -> dict[str, object]:
    """Summarize one primary score per participant, never pooled trials."""
    selected = [
        row for row in subject_rows
        if row["model"] == model_name and bool(row["qc_sensitivity"]) == qc_sensitivity
    ]
    if (
        len(selected) != expected_subject_count
        or len({int(row["subject"]) for row in selected}) != expected_subject_count
        or any(row["group_observational_unit"] != "participant" for row in selected)
    ):
        raise RuntimeError("Riemannian summary lacks one row per participant.")
    scores = np.asarray([float(row["primary_mean_run_balanced_accuracy"]) for row in selected])
    fists = np.asarray([float(row["fists_recall"]) for row in selected])
    feet = np.asarray([float(row["feet_recall"]) for row in selected])
    run_ranges = np.asarray([float(row["participant_run_range"]) for row in selected])
    bootstrap = config["group_inference"]["bootstrap_median_confidence_interval"]
    lower, upper = bootstrap_median_interval(
        scores, resamples=int(bootstrap["resamples"]),
        confidence_level=float(bootstrap["confidence_level"]), random_seed=int(bootstrap["random_seed"]),
    )
    return {
        "model": model_name,
        "qc_sensitivity": qc_sensitivity,
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
        "median_participant_run_range": float(np.median(run_ranges)),
        "group_observational_unit": "participant",
    }


def exact_sign_test_positive(differences: Sequence[float]) -> dict[str, object]:
    """One-sided exact sign test, excluding exact participant ties."""
    values = np.asarray(differences, dtype=float)
    improved = int(np.sum(values > 0.0))
    worsened = int(np.sum(values < 0.0))
    compared = improved + worsened
    return {
        "improved_count": improved,
        "worsened_count": worsened,
        "tie_count": int(np.sum(values == 0.0)),
        "one_sided_exact_sign_p": float(binomtest(improved, compared, 0.5, alternative="greater").pvalue) if compared else 1.0,
    }


def paired_riemannian_comparison(
    riemannian_rows: Sequence[Mapping[str, object]],
    historical_rows: Sequence[Mapping[str, object]],
    *,
    historical_model: str,
    expected_subject_count: int,
    qc_riemannian_rows: Sequence[Mapping[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare Riemannian and historical models with one matched row per person."""
    riemannian = {int(row["subject"]): row for row in riemannian_rows}
    historical = {
        int(row["subject"]): row for row in historical_rows if row["model"] == historical_model
    }
    if len(riemannian) != expected_subject_count or set(riemannian) != set(historical):
        raise RuntimeError("Riemannian historical comparison lacks exact participant matching.")
    qc_lookup = {int(row["subject"]): row for row in qc_riemannian_rows or []}
    rows: list[dict[str, object]] = []
    gains: list[float] = []
    qc_gains: list[float] = []
    for subject in sorted(riemannian):
        candidate = float(riemannian[subject]["primary_mean_run_balanced_accuracy"])
        reference = float(historical[subject]["primary_mean_run_balanced_accuracy"])
        gain = candidate - reference
        row: dict[str, object] = {
            "subject": subject,
            "riemannian_balanced_accuracy": candidate,
            "historical_model": historical_model,
            "historical_balanced_accuracy": reference,
            "riemannian_minus_historical": gain,
        }
        if subject in qc_lookup:
            qc_gain = float(qc_lookup[subject]["primary_mean_run_balanced_accuracy"]) - reference
            row["QC_riemannian_minus_historical"] = qc_gain
            qc_gains.append(qc_gain)
        rows.append(row)
        gains.append(gain)
    values = np.asarray(gains)
    summary: dict[str, object] = {
        "comparison": f"riemannian_minus_{historical_model}",
        "participant_count": expected_subject_count,
        "median_paired_difference": float(np.median(values)),
        "q25_paired_difference": float(np.percentile(values, 25)),
        "q75_paired_difference": float(np.percentile(values, 75)),
        **exact_sign_test_positive(values),
        "QC_median_paired_difference": float(np.median(qc_gains)) if qc_gains else None,
        "group_observational_unit": "participant",
    }
    return rows, summary


def select_development_classifier(
    subject_rows: Sequence[Mapping[str, object]], config: Mapping[str, Any]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Apply the frozen global LDA-versus-logistic development choice."""
    lookup = {
        (int(row["subject"]), str(row["model"])): row
        for row in subject_rows
        if not bool(row["qc_sensitivity"])
    }
    subjects = sorted({int(row["subject"]) for row in subject_rows if not bool(row["qc_sensitivity"])})
    rows: list[dict[str, object]] = []
    differences: list[float] = []
    for subject in subjects:
        lda = float(lookup[(subject, "shrinkage_lda")]["primary_mean_run_balanced_accuracy"])
        logistic = float(lookup[(subject, "l2_logistic_regression")]["primary_mean_run_balanced_accuracy"])
        difference = logistic - lda
        differences.append(difference)
        rows.append({
            "subject": subject,
            "shrinkage_lda_balanced_accuracy": lda,
            "l2_logistic_regression_balanced_accuracy": logistic,
            "logistic_minus_lda": difference,
        })
    values = np.asarray(differences)
    policy = config["candidate_policy"]
    logistic_fraction = float(np.mean(values > 0.0))
    selected = "l2_logistic_regression" if (
        float(np.median(values)) >= float(policy["select_logistic_only_if_median_paired_gain_at_least"])
        and logistic_fraction >= float(policy["and_fraction_logistic_better_at_least"])
    ) else "shrinkage_lda"
    summary: dict[str, object] = {
        "participant_count": len(rows),
        "median_logistic_minus_lda": float(np.median(values)),
        "q25_logistic_minus_lda": float(np.percentile(values, 25)),
        "q75_logistic_minus_lda": float(np.percentile(values, 75)),
        "logistic_better_count": int(np.sum(values > 0.0)),
        "lda_better_count": int(np.sum(values < 0.0)),
        "tie_count": int(np.sum(values == 0.0)),
        "logistic_better_fraction": logistic_fraction,
        "selected_classifier": selected,
        "selection_rule": "logistic_requires_median_gain_at_least_0.01_and_wins_at_least_60_percent;_otherwise_simpler_shrinkage_lda",
        "evaluation_outcome_used": False,
        "group_observational_unit": "participant",
    }
    return rows, summary


def classify_riemannian_result(
    group_summary: Mapping[str, object],
    csp_comparison: Mapping[str, object],
    csp_group_summary: Mapping[str, object],
    config: Mapping[str, Any],
) -> str:
    """Apply the frozen clear/modest/no-improvement representation gate."""
    median = float(group_summary["median_balanced_accuracy"])
    gain = float(csp_comparison["median_paired_difference"])
    improved = float(csp_comparison["improved_count"]) / float(csp_comparison["participant_count"])
    fists = float(group_summary["median_fists_recall"])
    feet = float(group_summary["median_feet_recall"])
    qc_gain = float(csp_comparison["QC_median_paired_difference"])
    range_increase = float(group_summary["median_participant_run_range"]) - float(csp_group_summary["median_participant_run_range"])
    clear = config["success_criteria"]["clear_representation_improvement"]
    if (
        median >= float(clear["minimum_Riemannian_median_balanced_accuracy"])
        and gain >= float(clear["minimum_median_Riemannian_minus_CSP"])
        and improved >= float(clear["minimum_fraction_Riemannian_better_than_CSP"])
        and float(csp_comparison["one_sided_exact_sign_p"]) <= float(clear["maximum_one_sided_exact_sign_p"])
        and min(fists, feet) >= float(clear["minimum_median_recall_each_class"])
        and abs(fists - feet) <= float(clear["maximum_absolute_median_class_recall_difference"])
        and qc_gain >= float(clear["minimum_QC_sensitive_median_Riemannian_minus_CSP"])
        and range_increase <= float(clear["maximum_increase_in_median_participant_run_range_vs_CSP"])
    ):
        return "clear representation improvement"
    modest = config["success_criteria"]["modest_improvement"]
    if (
        median >= float(modest["minimum_Riemannian_median_balanced_accuracy"])
        and gain >= float(modest["minimum_median_Riemannian_minus_CSP"])
        and improved >= float(modest["minimum_fraction_Riemannian_better_than_CSP"])
        and min(fists, feet) >= float(modest["minimum_median_recall_each_class"])
        and abs(fists - feet) <= float(modest["maximum_absolute_median_class_recall_difference"])
        and qc_gain >= float(modest["minimum_QC_sensitive_median_Riemannian_minus_CSP"])
    ):
        return "modest improvement"
    return "no improvement"
