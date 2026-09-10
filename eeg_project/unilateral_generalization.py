"""Frozen bilateral-decoder architecture applied to unilateral imagery runs only.

This module is deliberately separate from ``decoding.py``: that file is hash-frozen
as part of the accepted bilateral study. Shared public preprocessing and the exact
frozen model builder are reused without changing historical decoder behavior.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
from mne.datasets import eegbci
import numpy as np
from scipy.signal import welch
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from .decoding import TrialIdentity, build_candidate_pipeline, spectral_log_psd_features
from .epoching import EpochRecord, calculate_trial_quality, load_epoching_config
from .preprocessing import DEFAULT_DATA_DIRECTORY, load_preprocessing_config, preprocess_recording, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "unilateral_motor_imagery_generalization.json"
UNILATERAL_RUNS = (4, 8, 12)
UNILATERAL_LABELS = {"left_fist_imagery": 0, "right_fist_imagery": 1}


@dataclass(frozen=True)
class MotorImageryTask:
    identifier: str
    runs: tuple[int, ...]
    labels: Mapping[str, int]
    annotation_to_label: Mapping[str, int]


@dataclass(frozen=True)
class FoldIndices:
    test_run: int
    training_runs: tuple[int, int]
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class UnilateralDataset:
    subject: int
    fixed_task_data_volts: np.ndarray
    csp_task_data_volts: np.ndarray
    labels: np.ndarray
    runs: np.ndarray
    identities: list[TrialIdentity]
    qc_candidates: np.ndarray
    channel_names: list[str]
    sampling_frequency_hz: float


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_definition_for_run(run: int) -> MotorImageryTask:
    """Resolve context-sensitive imagery semantics; reject execution contexts."""
    if run in UNILATERAL_RUNS:
        return MotorImageryTask("left_vs_right_fist_imagery", UNILATERAL_RUNS, UNILATERAL_LABELS, {"T1": 0, "T2": 1})
    if run in (6, 10, 14):
        return MotorImageryTask("bilateral_historical_only", (6, 10, 14), {"both_fists_imagery": 0, "both_feet_imagery": 1}, {"T1": 0, "T2": 1})
    raise ValueError(f"Run {run} is not an approved unilateral imagery run (4/8/12).")


def validate_source_identity(subject: int, run: int, source_name: str) -> str:
    """Confirm a downloaded EDF filename belongs to the requested unilateral run."""
    if run not in UNILATERAL_RUNS:
        raise ValueError("Only unilateral runs may enter this source manifest.")
    expected = f"S{subject:03d}R{run:02d}.edf"
    if Path(source_name).name != expected:
        raise ValueError(f"Expected {expected}; received {Path(source_name).name}.")
    return expected


def load_study_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    parent = payload["parent_bilateral_decoder"]
    for field, expected in (("config", parent["sha256"]), ("metadata", parent["metadata_sha256"])):
        observed = file_sha256(PROJECT_ROOT / parent[field])
        if observed != expected:
            raise RuntimeError(f"Frozen bilateral parent drift: {parent[field]}.")
    if tuple(payload["task"]["runs"]) != UNILATERAL_RUNS or payload["task"]["labels"] != UNILATERAL_LABELS:
        raise ValueError("Unilateral task mapping drifted.")
    if payload["frozen_architecture"]["parent_models"] != ["spectral_baseline_6", "csp_4_empirical"]:
        raise ValueError("The unilateral model set must equal the frozen bilateral models.")
    return payload


def load_parent_decoder_config(config: Mapping[str, Any]) -> dict[str, Any]:
    final = json.loads((PROJECT_ROOT / config["parent_bilateral_decoder"]["config"]).read_text(encoding="utf-8"))
    initial = final["initial_policy"]
    path = PROJECT_ROOT / initial["path"]
    if file_sha256(path) != initial["sha256"]:
        raise RuntimeError("Frozen bilateral initial decoder policy has drifted.")
    return json.loads(path.read_text(encoding="utf-8"))


def expected_edf_paths(subjects: Sequence[int], data_directory: Path = DEFAULT_DATA_DIRECTORY) -> list[Path]:
    root = data_directory / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
    return [root / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf" for subject in sorted(subjects) for run in UNILATERAL_RUNS]


def acquire_unilateral_edfs(subjects: Sequence[int], data_directory: Path = DEFAULT_DATA_DIRECTORY) -> dict[str, str]:
    """Fetch only requested official EEGBCI EDFs and return immutable source hashes."""
    def fetch(identity: tuple[int, int]) -> tuple[str, str]:
        subject, run = identity
        last_error: Exception | None = None
        for _ in range(3):
            try:
                (downloaded,) = eegbci.load_data(subjects=subject, runs=run, path=data_directory, update_path=False, verbose=False)
                source = Path(downloaded).resolve()
                validate_source_identity(subject, run, str(source))
                raw = mne.io.read_raw_edf(source, preload=False, verbose="error")
                if raw.info["nchan"] != 64 or not np.isclose(raw.info["sfreq"], 160.0):
                    raise RuntimeError(f"Unexpected EEGMMIDB structure in {source}.")
                descriptions = set(str(value) for value in raw.annotations.description)
                if descriptions != {"T0", "T1", "T2"}:
                    raise RuntimeError(f"Unexpected annotations in {source}: {descriptions}.")
                return str(source), sha256_file(source)
            except Exception as error:  # Preserve the requested identity in a retry-safe failure.
                last_error = error
        raise RuntimeError(f"Official EDF acquisition failed for subject {subject}, run {run}: {last_error}")
    identities = [(subject, run) for subject in sorted(subjects) for run in UNILATERAL_RUNS]
    hashes: dict[str, str] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch, identity): identity for identity in identities}
        for future in as_completed(futures):
            try:
                path, digest = future.result()
                hashes[path] = digest
            except Exception as error:
                failures.append(str(error))
    if failures:
        raise RuntimeError("; ".join(sorted(failures)))
    return dict(sorted(hashes.items()))


def leave_one_run_out_folds(runs: np.ndarray, include: np.ndarray | None = None) -> list[FoldIndices]:
    values = np.asarray(runs, dtype=int)
    if values.ndim != 1 or set(np.unique(values)) != set(UNILATERAL_RUNS):
        raise ValueError("Unilateral data must contain exactly runs 4, 8, and 12.")
    allowed = np.ones(values.size, dtype=bool) if include is None else np.asarray(include, dtype=bool)
    folds: list[FoldIndices] = []
    for test_run in UNILATERAL_RUNS:
        training = tuple(run for run in UNILATERAL_RUNS if run != test_run)
        train = np.flatnonzero(allowed & np.isin(values, training))
        test = np.flatnonzero(allowed & (values == test_run))
        if not train.size or not test.size or np.intersect1d(train, test).size:
            raise RuntimeError("Unilateral train/test partition is invalid.")
        folds.append(FoldIndices(test_run, training, train, test))
    return folds


def validate_unilateral_class_balance(labels: np.ndarray) -> None:
    """Permit only source-verified retained-trial patterns with both targets."""
    counts = sorted(Counter(np.asarray(labels, dtype=int)).values())
    if counts not in ([6, 6], [6, 7], [7, 7], [7, 8]):
        raise RuntimeError(
            "Unilateral class balance must retain both targets as 7/8, 7/7, 7/6, or 6/6."
        )


def _continuous_csp_copy(raw: mne.io.BaseRaw, parent: Mapping[str, Any]) -> mne.io.BaseRaw:
    settings = parent["fixed_feature_methods"]["csp_feature_filter"]
    snapshot = raw.get_data().copy()
    output = raw.copy().filter(l_freq=float(settings["passband_hz"][0]), h_freq=float(settings["passband_hz"][1]), picks="eeg", filter_length=int(settings["filter_length_samples"]), l_trans_bandwidth=float(settings["transition_bandwidth_hz"][0]), h_trans_bandwidth=float(settings["transition_bandwidth_hz"][1]), method=str(settings["method"]), phase=str(settings["phase"]), fir_window=str(settings["fir_window"]), fir_design=str(settings["fir_design"]), pad=str(settings["pad"]), verbose="error")
    if not np.array_equal(raw.get_data(), snapshot) or not np.isfinite(output.get_data()).all():
        raise RuntimeError("CSP feature copy was invalid or mutated historical preprocessing.")
    return output


def _task_records(subject: int, run: int, raw: mne.io.BaseRaw, events: np.ndarray) -> list[EpochRecord]:
    sfreq = float(raw.info["sfreq"])
    records: list[EpochRecord] = []
    for annotation_index, event in enumerate(events):
        code = int(event[2])
        if code not in (2, 3):
            continue
        start, stop = int(event[0] - 2 * sfreq), int(event[0] + 4 * sfreq)
        valid = start >= raw.first_samp and stop < raw.first_samp + raw.n_times
        annotation = "T1" if code == 2 else "T2"
        semantic = "left_fist_imagery" if code == 2 else "right_fist_imagery"
        records.append(EpochRecord(subject, run, len(records), annotation_index, annotation, semantic, code, int(event[0]), (int(event[0])-raw.first_samp)/sfreq, 0.0, "", "", -2.0, 4.0, start, stop, (start-raw.first_samp)/sfreq, (stop-raw.first_samp)/sfreq, stop-start+1, valid, "" if valid else "epoch_crosses_valid_recording_boundary"))
    return records


def load_unilateral_dataset(subject: int, parent: Mapping[str, Any], data_directory: Path = DEFAULT_DATA_DIRECTORY) -> UnilateralDataset:
    """Extract independent unilateral task trials using frozen bilateral processing."""
    preprocessing = load_preprocessing_config()
    quality_config = load_epoching_config()
    fixed_parts: list[np.ndarray] = []; csp_parts: list[np.ndarray] = []; labels: list[int] = []; run_values: list[int] = []; identities: list[TrialIdentity] = []; candidates: list[bool] = []
    channel_names: list[str] | None = None
    for run in UNILATERAL_RUNS:
        task_definition_for_run(run)
        recording = preprocess_recording(subject, run, preprocessing, data_directory=data_directory)
        raw = recording.filtered
        events, event_id = mne.events_from_annotations(raw, event_id={"T0": 1, "T1": 2, "T2": 3}, verbose="error")
        if event_id != {"T0": 1, "T1": 2, "T2": 3}:
            raise RuntimeError("Unilateral annotation coding drifted.")
        records = [row for row in _task_records(subject, run, raw, events) if row.valid]
        event_rows = np.array([[row.event_sample, 0, row.event_code] for row in records], dtype=int)
        epochs = mne.Epochs(raw, event_rows, event_id={"left_fist_imagery": 2, "right_fist_imagery": 3}, tmin=-2.0, tmax=4.0, baseline=None, picks="eeg", preload=True, proj=False, reject=None, flat=None, detrend=None, reject_by_annotation=False, verbose="error")
        all_data = epochs.get_data(copy=True)
        sfreq = float(epochs.info["sfreq"]); start = int(round((1.0 - epochs.times[0]) * sfreq)); stop = start + int(2 * sfreq)
        if all_data.shape[2] != 961 or stop - start != 320:
            raise RuntimeError("Frozen unilateral task interval dimensions drifted.")
        if channel_names is None: channel_names = list(epochs.ch_names)
        elif channel_names != list(epochs.ch_names): raise RuntimeError("Channel order differs across unilateral runs.")
        csp_raw = _continuous_csp_copy(recording.filtered, parent)
        csp_trials = [csp_raw.get_data(start=row.event_sample + int(sfreq), stop=row.event_sample + int(3 * sfreq)) for row in records]
        quality = calculate_trial_quality(all_data, epochs.times, epochs.ch_names, records, quality_config.trial_quality)
        for record, quality_row in zip(records, quality, strict=True):
            labels.append(UNILATERAL_LABELS[record.semantic_condition]); run_values.append(run)
            identities.append(TrialIdentity(subject, run, record.run_trial_index, record.annotation_index, record.semantic_condition, str(recording.source_path), recording.source_sha256))
            candidates.append(quality_row["quality_status"] == "statistical candidate")
        fixed_parts.append(all_data[:, :, start:stop]); csp_parts.append(np.stack(csp_trials))
    assert channel_names is not None
    dataset = UnilateralDataset(subject, np.concatenate(fixed_parts), np.concatenate(csp_parts), np.asarray(labels), np.asarray(run_values), identities, np.asarray(candidates), channel_names, 160.0)
    if set(dataset.labels) != {0, 1}:
        raise RuntimeError("Unilateral target labels are incomplete.")
    for run in UNILATERAL_RUNS:
        validate_unilateral_class_balance(dataset.labels[dataset.runs == run])
    return dataset


def _metrics(y_true: np.ndarray, predicted: np.ndarray, score: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1]); left_left, left_right, right_left, right_right = (int(x) for x in matrix.ravel())
    return {"balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)), "accuracy": float(accuracy_score(y_true, predicted)), "macro_f1": float(f1_score(y_true, predicted, average="macro")), "roc_auc": float(roc_auc_score(y_true, score)), "left_recall": float(recall_score(y_true, predicted, pos_label=0)), "right_recall": float(recall_score(y_true, predicted, pos_label=1)), "true_left_predicted_left": left_left, "true_left_predicted_right": left_right, "true_right_predicted_left": right_left, "true_right_predicted_right": right_right, "trial_count": int(y_true.size)}


def evaluate_unilateral(dataset: UnilateralDataset, parent: Mapping[str, Any], model: str, qc_excluded: bool = False) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run exact frozen model settings with only two unilateral runs in every fit."""
    if model not in {"spectral_baseline_6", "csp_4_empirical"}: raise ValueError("Model is outside frozen unilateral architecture.")
    # The frozen feature function only needs aligned task arrays and channel metadata.
    proxy = type("FrozenFeatureProxy", (), {"fixed_task_data_volts": dataset.fixed_task_data_volts, "channel_names": dataset.channel_names, "sampling_frequency_hz": dataset.sampling_frequency_hz, "labels": dataset.labels})()
    inputs = spectral_log_psd_features(proxy, parent["candidate_models"][model], parent["fixed_feature_methods"]["spectral"])[0] if model.startswith("spectral") else dataset.csp_task_data_volts
    include = ~dataset.qc_candidates if qc_excluded else np.ones(dataset.labels.size, dtype=bool)
    predictions: list[dict[str, object]]=[]; folds: list[dict[str, object]]=[]; audits: list[dict[str, object]]=[]; predicted_indices: list[int]=[]
    for fold_number, fold in enumerate(leave_one_run_out_folds(dataset.runs, include), 1):
        pipeline = build_candidate_pipeline(model, parent)
        pipeline.fit(inputs[fold.train_indices], dataset.labels[fold.train_indices])
        predicted = np.asarray(pipeline.predict(inputs[fold.test_indices]), dtype=int); score = np.asarray(pipeline.decision_function(inputs[fold.test_indices]), dtype=float)
        metric = _metrics(dataset.labels[fold.test_indices], predicted, score)
        folds.append({"subject":dataset.subject,"task":"left_vs_right_fist_imagery","model":model,"qc_sensitivity":qc_excluded,"fold":fold_number,"test_run":fold.test_run,"training_runs":"/".join(map(str,fold.training_runs)),**metric})
        train_keys=[dataset.identities[int(i)].key for i in fold.train_indices]; test_keys=[dataset.identities[int(i)].key for i in fold.test_indices]
        audits.append({"subject":dataset.subject,"task":"left_vs_right_fist_imagery","model":model,"fold":fold_number,"test_run":fold.test_run,"training_runs":"/".join(map(str,fold.training_runs)),"train_trial_keys":"/".join(train_keys),"test_trial_keys":"/".join(test_keys),"csp_fit_trial_keys":"/".join(train_keys) if model.startswith("csp") else "not_applicable","scaler_fit_trial_keys":"/".join(train_keys),"lda_fit_trial_keys":"/".join(train_keys),"test_run_absent_from_all_fit_stages":True})
        for index, value, decision in zip(fold.test_indices, predicted, score, strict=True):
            identity=dataset.identities[int(index)]; true=int(dataset.labels[index])
            predictions.append({"subject":identity.subject,"task":"left_vs_right_fist_imagery","run":identity.run,"run_trial_index":identity.run_trial_index,"trial_key":identity.key,"true_condition":identity.semantic_condition,"true_label":true,"predicted_condition":"left_fist_imagery" if value==0 else "right_fist_imagery","predicted_label":int(value),"decision_score_right":float(decision),"correct":bool(value==true),"fold":fold_number,"test_run":fold.test_run,"training_runs":"/".join(map(str,fold.training_runs)),"model":model,"qc_sensitivity":qc_excluded,"source_file":identity.source_file,"source_sha256":identity.source_sha256})
        predicted_indices.extend(fold.test_indices.tolist())
    if sorted(predicted_indices) != np.flatnonzero(include).tolist(): raise RuntimeError("Each retained unilateral trial needs exactly one out-of-run prediction.")
    fold_scores=np.array([float(row["balanced_accuracy"]) for row in folds]); ordered=sorted(predictions,key=lambda row:(int(row["run"]),int(row["run_trial_index"])))
    all_metrics=_metrics(np.array([int(row["true_label"]) for row in ordered]),np.array([int(row["predicted_label"]) for row in ordered]),np.array([float(row["decision_score_right"]) for row in ordered]))
    subject={"subject":dataset.subject,"task":"left_vs_right_fist_imagery","model":model,"qc_sensitivity":qc_excluded,"primary_mean_fold_balanced_accuracy":float(np.mean(fold_scores)),"fold_balanced_accuracy_run_4":float(fold_scores[0]),"fold_balanced_accuracy_run_8":float(fold_scores[1]),"fold_balanced_accuracy_run_12":float(fold_scores[2]),"fold_balanced_accuracy_range":float(np.ptp(fold_scores)),"folds_above_chance_count":int(np.sum(fold_scores>.5)),**{f"out_of_fold_{key}":value for key,value in all_metrics.items()},"included_trial_count":len(ordered),"group_observational_unit":"participant"}
    return predictions, folds, audits, subject


def bootstrap_median(values: Sequence[float], resamples: int, seed: int) -> tuple[float, float]:
    data=np.asarray(values,float); rng=np.random.default_rng(seed); samples=np.array([np.median(rng.choice(data,data.size,replace=True)) for _ in range(resamples)])
    return float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def paired_summary(left: Sequence[float], right: Sequence[float], seed: int, resamples: int) -> dict[str, object]:
    differences=np.asarray(left,float)-np.asarray(right,float)
    statistic,p_value=wilcoxon(differences,alternative="two-sided",zero_method="wilcox") if np.any(differences) else (0.0,1.0)
    return {"paired_n":int(differences.size),"median_difference":float(np.median(differences)),"bootstrap_median_95":list(bootstrap_median(differences,resamples,seed)),"wilcoxon_statistic":float(statistic),"wilcoxon_p":float(p_value)}


def sign_test_above_chance(scores: Sequence[float]) -> dict[str, object]:
    values=np.asarray(scores,float); above=int(np.sum(values>.5)); below=int(np.sum(values<.5)); ties=int(np.sum(values==.5))
    return {"above_count":above,"below_count":below,"tie_count":ties,"one_sided_exact_sign_p":float(binomtest(above,above+below,.5,alternative="greater").pvalue) if above+below else 1.0}
