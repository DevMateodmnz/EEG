"""Leakage-safe within-subject motor-imagery decoding operations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import mne
from mne.decoding import CSP
import numpy as np
from scipy.signal import welch
from scipy.stats import binomtest, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .epoching import calculate_trial_quality, extract_run_epochs, load_epoching_config
from .preprocessing import DEFAULT_DATA_DIRECTORY, load_preprocessing_config
from .provenance import canonical_source_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECODING_CONFIG_PATH = PROJECT_ROOT / "config" / "within_subject_decoding.json"
TARGET_LABELS = {"both_fists_imagery": 0, "both_feet_imagery": 1}
RUNS = (6, 10, 14)


@dataclass(frozen=True)
class TrialIdentity:
    """Stable identity and provenance for one task trial."""

    subject: int
    run: int
    run_trial_index: int
    annotation_index: int
    semantic_condition: str
    source_file: str
    source_sha256: str

    @property
    def key(self) -> str:
        return f"S{self.subject:03d}_R{self.run:02d}_T{self.run_trial_index:02d}"


@dataclass
class SubjectDecodingData:
    """Aligned fixed and CSP task arrays for one participant."""

    subject: int
    fixed_task_data_volts: np.ndarray
    csp_task_data_volts: np.ndarray
    labels: np.ndarray
    runs: np.ndarray
    identities: list[TrialIdentity]
    qc_candidates: np.ndarray
    channel_names: list[str]
    sampling_frequency_hz: float
    interval_start_seconds: float
    interval_stop_seconds: float


@dataclass(frozen=True)
class FoldIndices:
    """One run-aware train/test partition after optional QC exclusion."""

    test_run: int
    training_runs: tuple[int, int]
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class CandidateResult:
    """Predictions, fold metrics, subject summary, and fit audit for one model."""

    prediction_rows: list[dict[str, object]]
    fold_rows: list[dict[str, object]]
    subject_row: dict[str, object]
    fit_audit_rows: list[dict[str, object]]


PipelineFactory = Callable[[str, Mapping[str, Any]], Any]


def sha256(path: Path) -> str:
    """Return a file digest for frozen-method guards."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_decoding_config(
    path: Path = DEFAULT_DECODING_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the visible decoder methodology."""
    resolved = path.expanduser().resolve()
    payload: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    validate_decoding_config(payload)
    return payload


def validate_decoding_config(payload: Mapping[str, Any]) -> None:
    """Reject cohort, target, fold, or leakage-policy drift."""
    if int(payload["schema_version"]) != 1:
        raise ValueError("Unsupported within-subject decoding schema.")
    cohorts = payload["cohorts"]
    development = list(cohorts["decoder_development_subjects"])
    requested = list(cohorts["decoder_evaluation_requested_subjects"])
    eligible = list(cohorts["decoder_evaluation_eligible_subjects"])
    incompatible = list(cohorts["technically_incompatible_subjects"])
    if development != list(range(1, 21)) or requested != list(range(21, 110)):
        raise ValueError("Decoder cohort identities differ from the initial freeze.")
    if incompatible != [88, 92, 100]:
        raise ValueError("Technical incompatibility identities have drifted.")
    if eligible != [subject for subject in requested if subject not in incompatible]:
        raise ValueError("Eligible decoder-evaluation subjects are inconsistent.")
    target = payload["prediction_target"]
    if target["classes"] != TARGET_LABELS or target["excluded_class"] != "T0_rest":
        raise ValueError("Decoder target must be fists versus feet with T0 excluded.")
    if tuple(target["runs"]) != RUNS or target["task_interval_seconds"] != [1.0, 3.0]:
        raise ValueError("Decoder runs or task interval have drifted.")
    cross_validation = payload["cross_validation"]
    if cross_validation["method"] != "leave_one_run_out":
        raise ValueError("Primary decoding must use leave-one-run-out folds.")
    if not cross_validation["random_trial_split_forbidden"]:
        raise ValueError("Random trial splitting must remain forbidden.")
    expected = {run: set(RUNS) - {run} for run in RUNS}
    actual = {
        int(fold["test_run"]): set(int(run) for run in fold["training_runs"])
        for fold in cross_validation["folds"]
    }
    if actual != expected:
        raise ValueError("Configured leave-one-run-out folds are incorrect.")
    if payload["metrics"]["primary"] != "balanced_accuracy":
        raise ValueError("Balanced accuracy must remain the primary metric.")
    if payload["metrics"]["group_observational_unit"] != "participant":
        raise ValueError("The group observational unit must remain the participant.")
    candidates = payload["candidate_models"]
    if set(candidates) != {
        "spectral_baseline_6",
        "spectral_baseline_9",
        "csp_4_empirical",
        "csp_4_ledoit_wolf",
        "csp_6_ledoit_wolf",
    }:
        raise ValueError("Candidate decoder family differs from the initial freeze.")
    for record in payload["historical_files"].values():
        path = PROJECT_ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Historical file drift: {path}.")


def leave_one_run_out_folds(
    runs: np.ndarray,
    *,
    include: np.ndarray | None = None,
) -> list[FoldIndices]:
    """Return exact run-aware folds; never randomly partition trials."""
    run_values = np.asarray(runs, dtype=int)
    if run_values.ndim != 1 or set(np.unique(run_values)) != set(RUNS):
        raise ValueError("Every subject must contain runs 6, 10, and 14.")
    allowed = np.ones(run_values.size, dtype=bool) if include is None else np.asarray(include, dtype=bool)
    if allowed.shape != run_values.shape:
        raise ValueError("Fold inclusion mask shape differs from runs.")
    folds: list[FoldIndices] = []
    for test_run in RUNS:
        training_runs = tuple(run for run in RUNS if run != test_run)
        train_indices = np.flatnonzero(allowed & np.isin(run_values, training_runs))
        test_indices = np.flatnonzero(allowed & (run_values == test_run))
        if train_indices.size == 0 or test_indices.size == 0:
            raise RuntimeError(f"Fold holding run {test_run} has no train or test trials.")
        if np.intersect1d(train_indices, test_indices).size:
            raise RuntimeError("Train and test trial indices overlap.")
        if set(run_values[train_indices]) != set(training_runs):
            raise RuntimeError("A training fold lacks one of its two runs.")
        if set(run_values[test_indices]) != {test_run}:
            raise RuntimeError("A test fold contains the wrong run.")
        folds.append(FoldIndices(test_run, training_runs, train_indices, test_indices))
    return folds


def _filter_continuous_csp_copy(raw: mne.io.BaseRaw, method: Mapping[str, Any]) -> mne.io.BaseRaw:
    """Create the frozen 8–30 Hz CSP copy without mutating the 1–40 Hz input."""
    original = raw.get_data().copy()
    feature_raw = raw.copy().filter(
        l_freq=float(method["passband_hz"][0]),
        h_freq=float(method["passband_hz"][1]),
        picks="eeg",
        filter_length=int(method["filter_length_samples"]),
        l_trans_bandwidth=float(method["transition_bandwidth_hz"][0]),
        h_trans_bandwidth=float(method["transition_bandwidth_hz"][1]),
        method=str(method["method"]),
        phase=str(method["phase"]),
        fir_window=str(method["fir_window"]),
        fir_design=str(method["fir_design"]),
        pad=str(method["pad"]),
        verbose="error",
    )
    if not np.array_equal(raw.get_data(), original):
        raise RuntimeError("CSP feature filtering mutated the historical 1–40 Hz run.")
    if not np.isfinite(feature_raw.get_data()).all():
        raise RuntimeError("CSP feature-filtered continuous data are non-finite.")
    return feature_raw


def load_subject_decoding_data(
    subject: int,
    config: Mapping[str, Any],
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
) -> SubjectDecodingData:
    """Load one person's immutable runs and construct aligned task-only arrays."""
    allowed = set(config["cohorts"]["decoder_development_subjects"]) | set(
        config["cohorts"]["decoder_evaluation_eligible_subjects"]
    )
    if subject not in allowed:
        raise ValueError(f"Subject {subject} is outside technically eligible decoder cohorts.")
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
            raise RuntimeError("Decoder run sampling frequency or channel order drifted.")
        valid_records = [record for record in result.task_records if record.valid]
        full_epoch_data = epochs.get_data(copy=True)
        if full_epoch_data.shape[0] != len(valid_records):
            raise RuntimeError("Task epochs and valid provenance records differ.")
        start_index = int(round((interval[0] - float(epochs.times[0])) * sfreq))
        sample_count = int(round((interval[1] - interval[0]) * sfreq))
        stop_index = start_index + sample_count
        selected_fixed = full_epoch_data[:, :, start_index:stop_index]
        if selected_fixed.shape[2] != 320:
            raise RuntimeError("Frozen +1 to +3 s half-open interval must contain 320 samples.")
        feature_raw = _filter_continuous_csp_copy(result.preprocessing.filtered, csp_filter)
        run_csp: list[np.ndarray] = []
        for record in valid_records:
            start_sample = record.event_sample + int(round(interval[0] * sfreq))
            stop_sample = start_sample + sample_count
            trial = feature_raw.get_data(start=start_sample, stop=stop_sample)
            if trial.shape != (len(feature_raw.ch_names), sample_count):
                raise RuntimeError("Continuous CSP interval extraction returned the wrong shape.")
            run_csp.append(trial)
        selected_csp = np.stack(run_csp)
        quality_rows = calculate_trial_quality(
            full_epoch_data,
            epochs.times.copy(),
            epochs.ch_names,
            valid_records,
            epoching.trial_quality,
        )
        if len(quality_rows) != len(valid_records):
            raise RuntimeError("Decoder QC rows and task records differ.")
        for record, quality_row in zip(valid_records, quality_rows, strict=True):
            if record.semantic_condition not in TARGET_LABELS:
                raise RuntimeError("A non-target condition entered decoder task data.")
            labels.append(TARGET_LABELS[record.semantic_condition])
            runs.append(run)
            identities.append(
                TrialIdentity(
                    subject=subject,
                    run=run,
                    run_trial_index=record.run_trial_index,
                    annotation_index=record.annotation_index,
                    semantic_condition=record.semantic_condition,
                    source_file=canonical_source_id(str(result.preprocessing.source_path)),
                    source_sha256=result.preprocessing.source_sha256,
                )
            )
            qc_candidates.append(quality_row["quality_status"] == "statistical candidate")
        fixed_arrays.append(selected_fixed)
        csp_arrays.append(selected_csp)
    assert channel_names is not None and sampling_frequency is not None
    dataset = SubjectDecodingData(
        subject=subject,
        fixed_task_data_volts=np.concatenate(fixed_arrays),
        csp_task_data_volts=np.concatenate(csp_arrays),
        labels=np.asarray(labels, dtype=int),
        runs=np.asarray(runs, dtype=int),
        identities=identities,
        qc_candidates=np.asarray(qc_candidates, dtype=bool),
        channel_names=channel_names,
        sampling_frequency_hz=sampling_frequency,
        interval_start_seconds=interval[0],
        interval_stop_seconds=interval[1],
    )
    validate_subject_decoding_data(dataset)
    return dataset


def validate_subject_decoding_data(dataset: SubjectDecodingData) -> None:
    """Validate array alignment, identities, semantics, and finite values."""
    count = dataset.labels.size
    if dataset.fixed_task_data_volts.shape[0] != count or dataset.csp_task_data_volts.shape[0] != count:
        raise ValueError("Decoder trial-array counts differ from labels.")
    if len(dataset.identities) != count or dataset.runs.size != count or dataset.qc_candidates.size != count:
        raise ValueError("Decoder provenance arrays are not aligned.")
    if dataset.fixed_task_data_volts.shape[1:] != dataset.csp_task_data_volts.shape[1:]:
        raise ValueError("Fixed and CSP arrays must have identical channel/time shapes.")
    if dataset.fixed_task_data_volts.shape[1] != len(dataset.channel_names):
        raise ValueError("Decoder channel dimension and names differ.")
    if not np.isfinite(dataset.fixed_task_data_volts).all() or not np.isfinite(dataset.csp_task_data_volts).all():
        raise ValueError("Decoder data contain NaN or infinity.")
    if set(np.unique(dataset.labels)) != {0, 1} or set(np.unique(dataset.runs)) != set(RUNS):
        raise ValueError("Decoder labels or runs are incomplete.")
    keys = [identity.key for identity in dataset.identities]
    if len(keys) != len(set(keys)):
        raise ValueError("Decoder trial identities are not unique.")
    for run in RUNS:
        selected = dataset.labels[dataset.runs == run]
        counts = sorted(Counter(selected).values())
        if counts != [7, 8]:
            raise RuntimeError(f"Run {run} does not retain the expected 7/8 class balance.")


def spectral_log_psd_features(
    dataset: SubjectDecodingData,
    candidate: Mapping[str, Any],
    method: Mapping[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Calculate fixed task-only channel/band log-Welch features per trial."""
    channel_indices = [dataset.channel_names.index(name) for name in candidate["channels"]]
    data = dataset.fixed_task_data_volts[:, channel_indices]
    frequencies, power = welch(
        data,
        fs=dataset.sampling_frequency_hz,
        window="hann",
        nperseg=int(method["window_samples_at_160_hz"]),
        noverlap=int(method["overlap_samples"]),
        detrend=str(method["detrend"]),
        scaling="density",
        axis=-1,
    )
    if not np.isclose(np.median(np.diff(frequencies)), method["frequency_resolution_hz"]):
        raise RuntimeError("Spectral decoder frequency resolution drifted.")
    if np.any(power <= 0) or not np.isfinite(power).all():
        raise RuntimeError("Spectral decoder PSD is nonpositive or non-finite.")
    feature_columns: list[np.ndarray] = []
    names: list[str] = []
    log_power = 10.0 * np.log10(power)
    for channel_index, channel in enumerate(candidate["channels"]):
        for band_name, bounds in candidate["bands_hz"].items():
            mask = (frequencies >= float(bounds[0])) & (frequencies <= float(bounds[1]))
            if not mask.any():
                raise RuntimeError(f"No Welch frequency bins in decoder band {band_name}.")
            feature_columns.append(np.mean(log_power[:, channel_index, mask], axis=1))
            names.append(f"{channel}_{band_name}_mean_log_psd_db")
    features = np.column_stack(feature_columns)
    if features.shape != (dataset.labels.size, int(candidate["feature_count"])):
        raise RuntimeError("Spectral decoder feature shape differs from policy.")
    return features, names


def build_candidate_pipeline(
    candidate_name: str,
    config: Mapping[str, Any],
) -> Pipeline:
    """Build an unfitted fold-local pipeline for one frozen candidate."""
    candidate = config["candidate_models"][candidate_name]
    lda_settings = config["supervised_pipeline"]["lda"]
    lda = LinearDiscriminantAnalysis(
        solver=lda_settings["solver"],
        shrinkage=lda_settings["shrinkage"],
        priors=np.asarray(lda_settings["priors"], dtype=float),
    )
    steps: list[tuple[str, Any]] = []
    if candidate_name.startswith("csp_"):
        csp_settings = config["supervised_pipeline"]["csp"]
        steps.append(
            (
                "csp",
                CSP(
                    n_components=int(candidate["csp_components"]),
                    reg=candidate["csp_covariance_regularization"],
                    log=bool(csp_settings["log"]),
                    cov_est=csp_settings["cov_est"],
                    transform_into=csp_settings["transform_into"],
                    norm_trace=bool(csp_settings["norm_trace"]),
                    component_order=csp_settings["component_order"],
                ),
            )
        )
    scaler_settings = config["supervised_pipeline"]["scaler"]
    steps.extend(
        [
            (
                "scaler",
                StandardScaler(
                    with_mean=bool(scaler_settings["with_mean"]),
                    with_std=bool(scaler_settings["with_std"]),
                ),
            ),
            ("lda", lda),
        ]
    )
    return Pipeline(steps)


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    """Return the compact predefined binary metric set."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if matrix.shape != (2, 2) or int(matrix.sum()) != y_true.size:
        raise RuntimeError("Confusion matrix does not account for every prediction.")
    fists_fists, fists_feet, feet_fists, feet_feet = (int(value) for value in matrix.ravel())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "fists_recall": float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "feet_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "true_fists_predicted_fists": fists_fists,
        "true_fists_predicted_feet": fists_feet,
        "true_feet_predicted_fists": feet_fists,
        "true_feet_predicted_feet": feet_feet,
        "trial_count": int(y_true.size),
    }


def evaluate_candidate(
    dataset: SubjectDecodingData,
    candidate_name: str,
    config: Mapping[str, Any],
    *,
    qc_excluded: bool = False,
    pipeline_factory: PipelineFactory = build_candidate_pipeline,
) -> CandidateResult:
    """Fit all supervised operations inside each run-aware training fold."""
    if candidate_name not in config["candidate_models"]:
        raise ValueError(f"Unknown decoder candidate: {candidate_name}.")
    candidate = config["candidate_models"][candidate_name]
    if candidate_name.startswith("spectral_"):
        model_input, feature_names = spectral_log_psd_features(
            dataset, candidate, config["fixed_feature_methods"]["spectral"]
        )
    else:
        model_input = dataset.csp_task_data_volts
        feature_names = [f"CSP{index + 1}_log_average_power" for index in range(int(candidate["csp_components"]))]
    include = ~dataset.qc_candidates if qc_excluded else np.ones(dataset.labels.size, dtype=bool)
    folds = leave_one_run_out_folds(dataset.runs, include=include)
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    predicted_indices: list[int] = []
    for fold_index, fold in enumerate(folds, start=1):
        train_indices = fold.train_indices
        test_indices = fold.test_indices
        y_train = dataset.labels[train_indices]
        y_test = dataset.labels[test_indices]
        if set(y_train) != {0, 1} or set(y_test) != {0, 1}:
            raise RuntimeError("Each decoder fold must retain both classes.")
        pipeline = pipeline_factory(candidate_name, config)
        pipeline.fit(model_input[train_indices], y_train)
        predictions = np.asarray(pipeline.predict(model_input[test_indices]), dtype=int)
        decision_scores = np.asarray(pipeline.decision_function(model_input[test_indices]), dtype=float)
        if decision_scores.ndim != 1:
            raise RuntimeError("Binary decoder must produce one decision score per trial.")
        if not np.isfinite(decision_scores).all() or predictions.shape != y_test.shape:
            raise RuntimeError("Decoder predictions or decision scores are invalid.")
        metrics = _metric_row(y_test, predictions, decision_scores)
        fold_rows.append(
            {
                "subject": dataset.subject,
                "model": candidate_name,
                "qc_sensitivity": qc_excluded,
                "fold": fold_index,
                "test_run": fold.test_run,
                "training_runs": "/".join(str(run) for run in fold.training_runs),
                **metrics,
            }
        )
        train_keys = [dataset.identities[index].key for index in train_indices]
        test_keys = [dataset.identities[index].key for index in test_indices]
        audit_rows.append(
            {
                "subject": dataset.subject,
                "model": candidate_name,
                "qc_sensitivity": qc_excluded,
                "fold": fold_index,
                "test_run": fold.test_run,
                "training_runs": "/".join(str(run) for run in fold.training_runs),
                "train_trial_count": len(train_indices),
                "test_trial_count": len(test_indices),
                "train_trial_keys": "/".join(train_keys),
                "test_trial_keys": "/".join(test_keys),
                "csp_fit_trial_keys": "/".join(train_keys) if candidate_name.startswith("csp_") else "not_applicable",
                "scaler_fit_trial_keys": "/".join(train_keys),
                "lda_fit_trial_keys": "/".join(train_keys),
                "test_run_absent_from_all_fit_stages": not any(
                    dataset.identities[index].run == fold.test_run for index in train_indices
                ),
                "subject_isolation_preserved": all(
                    dataset.identities[index].subject == dataset.subject
                    for index in np.concatenate((train_indices, test_indices))
                ),
                "feature_names": "/".join(feature_names),
            }
        )
        for local_index, prediction, score in zip(test_indices, predictions, decision_scores, strict=True):
            identity = dataset.identities[int(local_index)]
            true_label = int(dataset.labels[local_index])
            prediction_rows.append(
                {
                    "subject": identity.subject,
                    "run": identity.run,
                    "run_trial_index": identity.run_trial_index,
                    "annotation_index": identity.annotation_index,
                    "trial_key": identity.key,
                    "true_condition": identity.semantic_condition,
                    "true_label": true_label,
                    "predicted_condition": "both_fists_imagery" if prediction == 0 else "both_feet_imagery",
                    "predicted_label": int(prediction),
                    "decision_score_feet": float(score),
                    "correct": bool(prediction == true_label),
                    "fold": fold_index,
                    "test_run": fold.test_run,
                    "training_runs": "/".join(str(run) for run in fold.training_runs),
                    "model": candidate_name,
                    "qc_sensitivity": qc_excluded,
                    "qc_candidate": bool(dataset.qc_candidates[local_index]),
                    "source_file": identity.source_file,
                    "source_sha256": identity.source_sha256,
                }
            )
        predicted_indices.extend(int(index) for index in test_indices)
    expected_indices = np.flatnonzero(include).tolist()
    if sorted(predicted_indices) != expected_indices or len(predicted_indices) != len(set(predicted_indices)):
        raise RuntimeError("Every included trial must receive exactly one out-of-run prediction.")
    ordered_predictions = sorted(prediction_rows, key=lambda row: (int(row["run"]), int(row["run_trial_index"])))
    y_true = np.array([int(row["true_label"]) for row in ordered_predictions])
    y_pred = np.array([int(row["predicted_label"]) for row in ordered_predictions])
    scores = np.array([float(row["decision_score_feet"]) for row in ordered_predictions])
    combined_metrics = _metric_row(y_true, y_pred, scores)
    fold_balanced = np.array([float(row["balanced_accuracy"]) for row in fold_rows])
    subject_row: dict[str, object] = {
        "subject": dataset.subject,
        "model": candidate_name,
        "qc_sensitivity": qc_excluded,
        "primary_mean_fold_balanced_accuracy": float(np.mean(fold_balanced)),
        "fold_balanced_accuracy_run_6": float(fold_balanced[0]),
        "fold_balanced_accuracy_run_10": float(fold_balanced[1]),
        "fold_balanced_accuracy_run_14": float(fold_balanced[2]),
        "folds_above_chance_count": int(np.sum(fold_balanced > 0.5)),
        "fold_balanced_accuracy_range": float(np.ptp(fold_balanced)),
        **{f"out_of_fold_{name}": value for name, value in combined_metrics.items()},
        "qc_candidate_count": int(np.sum(dataset.qc_candidates)),
        "included_trial_count": len(ordered_predictions),
        "group_observational_unit": "participant",
    }
    return CandidateResult(prediction_rows, fold_rows, subject_row, audit_rows)


def summarize_development_candidates(
    subject_rows: Sequence[Mapping[str, object]],
    *,
    qc_sensitivity: bool = False,
) -> list[dict[str, object]]:
    """Summarize candidates with exactly one score per development participant."""
    output: list[dict[str, object]] = []
    for candidate_name in sorted({str(row["model"]) for row in subject_rows}):
        selected = [
            row
            for row in subject_rows
            if row["model"] == candidate_name
            and bool(row["qc_sensitivity"]) == qc_sensitivity
        ]
        scores = np.array([float(row["primary_mean_fold_balanced_accuracy"]) for row in selected])
        if len(selected) != 20 or len({int(row["subject"]) for row in selected}) != 20:
            raise RuntimeError(f"Development candidate {candidate_name} lacks 20 unique subjects.")
        output.append(
            {
                "model": candidate_name,
                "development_subject_count": len(scores),
                "median_subject_balanced_accuracy": float(np.median(scores)),
                "q25_subject_balanced_accuracy": float(np.percentile(scores, 25)),
                "q75_subject_balanced_accuracy": float(np.percentile(scores, 75)),
                "minimum_subject_balanced_accuracy": float(np.min(scores)),
                "maximum_subject_balanced_accuracy": float(np.max(scores)),
                "fraction_subjects_above_0_5": float(np.mean(scores > 0.5)),
                "fraction_subjects_at_or_above_0_6": float(np.mean(scores >= 0.6)),
                "fraction_subjects_with_at_least_two_folds_above_0_5": float(
                    np.mean([int(row["folds_above_chance_count"]) >= 2 for row in selected])
                ),
                "median_fold_range": float(
                    np.median([float(row["fold_balanced_accuracy_range"]) for row in selected])
                ),
                "selection_uses_one_score_per_subject": True,
                "evaluation_subject_outcome_used": False,
            }
        )
    return output


def select_development_model(
    summary_rows: Sequence[Mapping[str, object]],
    family_prefix: str,
    config: Mapping[str, Any],
) -> str:
    """Apply frozen median/tolerance/preference selection within one family."""
    selected = [row for row in summary_rows if str(row["model"]).startswith(family_prefix)]
    if not selected:
        raise ValueError(f"No development candidates for prefix {family_prefix}.")
    best = max(float(row["median_subject_balanced_accuracy"]) for row in selected)
    tolerance = float(config["development_selection"]["practical_tie_tolerance_balanced_accuracy"])
    near_best = {
        str(row["model"])
        for row in selected
        if best - float(row["median_subject_balanced_accuracy"]) <= tolerance + 1e-12
    }
    preference_key = "spectral_tie_preference" if family_prefix == "spectral_" else "csp_tie_preference"
    for name in config["development_selection"][preference_key]:
        if name in near_best:
            return name
    raise RuntimeError("Frozen model preference does not cover near-best candidates.")


def majority_baseline_metrics(labels: np.ndarray) -> dict[str, float]:
    """Return trivial majority-class accuracy and balanced accuracy."""
    y = np.asarray(labels, dtype=int)
    counts = Counter(y)
    majority = max(sorted(counts), key=counts.get)
    predictions = np.full_like(y, majority)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
    }


def bootstrap_median_interval(
    values: Sequence[float],
    *,
    resamples: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[float, float]:
    """Return a fixed-seed percentile bootstrap interval over participants."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise ValueError("Bootstrap values must be a finite one-dimensional sample.")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    medians = np.median(data[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return float(np.quantile(medians, alpha)), float(np.quantile(medians, 1.0 - alpha))


def exact_sign_test_above_chance(values: Sequence[float], chance: float = 0.5) -> dict[str, object]:
    """Test whether above-chance participant counts exceed below-chance counts."""
    data = np.asarray(values, dtype=float)
    above = int(np.sum(data > chance))
    below = int(np.sum(data < chance))
    ties = int(np.sum(data == chance))
    p_value = float(binomtest(above, above + below, p=0.5, alternative="greater").pvalue) if above + below else 1.0
    return {"above_count": above, "below_count": below, "tie_count": ties, "one_sided_exact_sign_p": p_value}


def paired_wilcoxon(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Return the frozen supporting two-sided paired signed-rank p-value."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Paired model scores must be aligned vectors.")
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
