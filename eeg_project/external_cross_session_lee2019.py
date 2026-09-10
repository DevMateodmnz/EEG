"""Frozen PhysioNet decoder architecture on labelled Lee2019 MI sessions only."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
import numpy as np
from scipy.signal import welch
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from .decoding import build_candidate_pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "external_cross_session_lee2019.json"
DEFAULT_DATA_DIRECTORY = ROOT / "data"
LEE_LABELS = {"left_hand": 0, "right_hand": 1}
SESSION_IDS = ("1", "2")
LEE2019_NEMAR_ID = "nm000338"
LEE2019_DATASET_CODE = "Lee2019-MI"


@dataclass(frozen=True)
class SessionFold:
    train_session: str
    test_session: str


@dataclass(frozen=True)
class ExternalTrialIdentity:
    subject: int
    session: str
    trial_index: int
    source_file: str
    source_sha256: str

    @property
    def key(self) -> str:
        return f"S{self.subject:02d}_S{self.session}_T{self.trial_index:03d}"


@dataclass
class SessionTrials:
    fixed: np.ndarray
    csp: np.ndarray
    labels: np.ndarray
    identities: list[ExternalTrialIdentity]
    channels: list[str]
    sfreq: float


@dataclass
class ExternalSubjectData:
    subject: int
    sessions: dict[str, SessionTrials]
    source_manifest: list[dict[str, object]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def lee2019_nemar_provider(data_directory: Path):
    """Pin one load/download to the official MOABB NEMAR source mirror.

    This changes acquisition infrastructure only.  It deliberately uses
    process-local environment variables so an external-study command cannot
    alter the user's global MNE or MOABB configuration.
    """
    keys = {
        "MOABB_DOWNLOAD_PROVIDER": "nemar",
        "MNE_DATASETS_LEE2019-MI_PATH": str(data_directory.resolve()),
    }
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def lee2019_nemar_sources(subject: int, data_directory: Path, *, verify_hashes: bool = True) -> list[dict[str, object]]:
    """Return the two validated original-source records for one participant.

    The NEMAR provenance manifest supplies upstream filenames, bytes, and
    SHA-256 values.  A `.part` file is never accepted as a source record.
    """
    manifest_path = data_directory / "NEMAR" / LEE2019_NEMAR_ID / "sourcedata" / "sourcedata_provenance.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Lee2019 NEMAR provenance manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [entry for entry in manifest.get("files", []) if str(entry.get("subject")) == str(subject)]
    if len(entries) != 2:
        raise ValueError(f"NEMAR manifest does not contain exactly two Lee2019 source files for subject {subject}.")
    records: list[dict[str, object]] = []
    for entry in sorted(entries, key=lambda item: str(item["file"])):
        relative = Path(str(entry["file"]))
        source = manifest_path.parent / relative
        expected_bytes, expected_hash = int(entry["bytes"]), str(entry["sha256"])
        if not source.is_file() or source.stat().st_size != expected_bytes:
            raise FileNotFoundError(f"Lee2019 source is incomplete or missing: {source}")
        observed_hash = sha256_file(source) if verify_hashes else expected_hash
        if observed_hash != expected_hash:
            raise RuntimeError(f"Lee2019 source hash mismatch: {source}")
        session = "1" if relative.parts[0] == "session1" else "2" if relative.parts[0] == "session2" else None
        if session is None:
            raise ValueError(f"Unexpected Lee2019 source session path: {relative}")
        records.append({"subject": subject, "session": session, "run": "1train", "source_file": str(source.relative_to(data_directory)), "source_sha256": expected_hash, "source_bytes": expected_bytes, "provider": "MOABB_NEMAR_sourcedata", "validation": "sha256_valid"})
    if tuple(record["session"] for record in records) != SESSION_IDS:
        raise ValueError(f"NEMAR source sessions drifted for subject {subject}.")
    return records


def load_study_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload["study_id"] != "external_cross_session_lee2019" or payload["classification"] != "EXTERNAL_CROSS_SESSION_REPLICATION":
        raise ValueError("External replication identity drifted.")
    parent = payload["parent_unilateral_study"]
    for field, expected in (("config", parent["sha256"]), ("summary", parent["summary_sha256"])):
        if sha256_file(ROOT / parent[field]) != expected:
            raise RuntimeError(f"Frozen unilateral parent drifted: {parent[field]}.")
    external = payload["external_dataset"]
    if external["adapter"] != "moabb.datasets.Lee2019_MI" or external["candidate_subjects"] != list(range(1, 55)):
        raise ValueError("Lee2019 dataset identity or cohort drifted.")
    if tuple(external["sessions"]) != SESSION_IDS or external["formal_run"] != "1train" or not external["online_test_run_forbidden"]:
        raise ValueError("External session/run policy drifted.")
    if external.get("download_provider") != "nemar" or external.get("nemar_deposit") != LEE2019_NEMAR_ID:
        raise ValueError("External source-provider policy drifted.")
    if payload["task"]["labels"] != LEE_LABELS or payload["dataset_adapter"]["task_interval_seconds_after_cue"] != [1.0, 3.0]:
        raise ValueError("External task mapping or epoch drifted.")
    return payload


def validate_formal_run(run: str) -> None:
    if run != "1train":
        raise ValueError("Only labelled offline Lee2019 MI training run '1train' is permitted.")


def validate_channel_names(channels: Sequence[str], *, required: Sequence[str] = ("C3", "Cz", "C4")) -> None:
    missing = [channel for channel in required if channel not in channels]
    if missing:
        raise ValueError(f"Required external EEG channels missing: {', '.join(missing)}.")


def validate_labels(labels: Sequence[str]) -> np.ndarray:
    try:
        encoded = np.asarray([LEE_LABELS[label] for label in labels], dtype=int)
    except KeyError as error:
        raise ValueError("Only authoritative left_hand/right_hand labels are permitted.") from error
    if set(encoded) != {0, 1}:
        raise ValueError("Both external motor-imagery classes are required.")
    return encoded


def epoch_sample_bounds(sfreq: float) -> tuple[int, int]:
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Sampling frequency must be finite and positive.")
    return int(round(sfreq)), int(round(3 * sfreq))


def cross_session_folds() -> tuple[SessionFold, SessionFold]:
    return (SessionFold("1", "2"), SessionFold("2", "1"))


def participant_mean_score(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("A participant needs exactly two finite directional scores.")
    return float(np.mean(values))


def validate_source_only_fit(train_session: str, test_session: str, train_keys: Sequence[str], test_keys: Sequence[str]) -> None:
    if train_session == test_session or set(train_keys) & set(test_keys):
        raise ValueError("Cross-session train/test provenance overlaps.")
    if not train_keys or not test_keys:
        raise ValueError("Source and target sessions must both contain trials.")


def validate_fit_audits(audits: Sequence[Mapping[str, object]]) -> None:
    """Reject any audit which cannot prove every learned stage was source-only."""
    if not audits:
        raise ValueError("External evaluation produced no fit audits.")
    seen: set[tuple[int, str, str, str]] = set()
    for audit in audits:
        subject = int(audit["subject"])
        model = str(audit["model"])
        train_session, test_session = str(audit["train_session"]), str(audit["test_session"])
        train_keys = str(audit["train_trial_keys"]).split("/")
        test_keys = str(audit["test_trial_keys"]).split("/")
        validate_source_only_fit(train_session, test_session, train_keys, test_keys)
        if not bool(audit["source_session_only"]):
            raise ValueError("External audit does not certify source-only fitting.")
        if str(audit["scaler_fit_trial_keys"]) != "/".join(train_keys) or str(audit["lda_fit_trial_keys"]) != "/".join(train_keys):
            raise ValueError("Scaler or LDA audit provenance is not exactly the source session.")
        csp_keys = str(audit["csp_fit_trial_keys"])
        if model.startswith("csp") and csp_keys != "/".join(train_keys):
            raise ValueError("CSP audit provenance is not exactly the source session.")
        if model.startswith("spectral") and csp_keys != "not_applicable":
            raise ValueError("Spectral baseline must not declare a CSP fit.")
        key = (subject, model, train_session, test_session)
        if key in seen:
            raise ValueError("Duplicate external fit-audit fold.")
        seen.add(key)


def _moabb_dataset() -> Any:
    try:
        from moabb.datasets import Lee2019_MI
    except ImportError as error:
        raise RuntimeError("MOABB 1.7.1 is required for the canonical Lee2019 adapter.") from error
    return Lee2019_MI(train_run=True, test_run=False, resting_state=False)


def _preprocess_session(raw: mne.io.BaseRaw, config: Mapping[str, Any]) -> mne.io.BaseRaw:
    picks = mne.pick_types(raw.info, eeg=True, emg=False, stim=False, exclude=[])
    if len(picks) != int(config["external_dataset"]["expected_eeg_channel_count"]):
        raise ValueError("Lee2019 adapter did not expose the expected 62 EEG channels.")
    eeg = raw.copy().pick(picks).set_eeg_reference("average", verbose="error")
    primary = config["dataset_adapter"]["primary_filter"]
    return eeg.filter(l_freq=float(primary["passband_hz"][0]), h_freq=float(primary["passband_hz"][1]),
                      l_trans_bandwidth=float(primary["transition_bandwidth_hz"][0]), h_trans_bandwidth=float(primary["transition_bandwidth_hz"][1]),
                      filter_length="auto", method=str(primary["method"]), phase=str(primary["phase"]),
                      fir_window=str(primary["fir_window"]), fir_design=str(primary["fir_design"]), pad=str(primary["pad"]), verbose="error")


def _csp_copy(primary: mne.io.BaseRaw, config: Mapping[str, Any]) -> mne.io.BaseRaw:
    settings = config["dataset_adapter"]["csp_filter"]
    before = primary.get_data().copy()
    result = primary.copy().filter(l_freq=float(settings["passband_hz"][0]), h_freq=float(settings["passband_hz"][1]),
                                   l_trans_bandwidth=float(settings["transition_bandwidth_hz"][0]), h_trans_bandwidth=float(settings["transition_bandwidth_hz"][1]),
                                   filter_length="auto", method=str(settings["method"]), phase=str(settings["phase"]),
                                   fir_window=str(settings["fir_window"]), fir_design=str(settings["fir_design"]), pad=str(settings["pad"]), verbose="error")
    if not np.array_equal(primary.get_data(), before):
        raise RuntimeError("External CSP filter mutated its primary input.")
    return result


def _session_trials(subject: int, session: str, raw: mne.io.BaseRaw, source: Path, config: Mapping[str, Any]) -> SessionTrials:
    primary = _preprocess_session(raw, config)
    csp_raw = _csp_copy(primary, config)
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose="error")
    labels: list[str] = []
    selected_events: list[np.ndarray] = []
    for event in events:
        label = {2: "left_hand", 1: "right_hand"}.get(int(event[2]))
        if label is not None:
            labels.append(label); selected_events.append(event)
    y = validate_labels(labels)
    sfreq = float(primary.info["sfreq"])
    start, stop = epoch_sample_bounds(sfreq)
    length = stop - start
    fixed: list[np.ndarray] = []; csp: list[np.ndarray] = []; identities: list[ExternalTrialIdentity] = []
    source_hash = sha256_file(source)
    for index, event in enumerate(selected_events):
        onset = int(event[0]); left, right = onset + start, onset + stop
        if left < primary.first_samp or right > primary.first_samp + primary.n_times:
            raise ValueError(f"Truncated Lee2019 MI epoch for subject {subject}, session {session}, trial {index}.")
        fixed.append(primary.get_data(start=left, stop=right))
        csp.append(csp_raw.get_data(start=left, stop=right))
        identities.append(ExternalTrialIdentity(subject, session, index, str(source), source_hash))
    arrays = np.stack(fixed), np.stack(csp)
    if arrays[0].shape != (y.size, 62, length) or not np.isfinite(arrays[0]).all() or not np.isfinite(arrays[1]).all():
        raise RuntimeError("External session epoch schema is invalid.")
    validate_channel_names(primary.ch_names)
    return SessionTrials(arrays[0], arrays[1], y, identities, list(primary.ch_names), sfreq)


def load_subject_data(subject: int, config: Mapping[str, Any], data_directory: Path = DEFAULT_DATA_DIRECTORY) -> ExternalSubjectData:
    if subject not in config["external_dataset"]["candidate_subjects"]:
        raise ValueError("Subject lies outside frozen Lee2019 cohort.")
    dataset = _moabb_dataset()
    # get_data prefetches only this participant's NEMAR sourcedata and serves
    # the adapter through a hardlink/copy from that validated source store.
    with lee2019_nemar_provider(data_directory):
        records = dataset.get_data(subjects=[subject])[subject]
    sources = lee2019_nemar_sources(subject, data_directory)
    sessions: dict[str, SessionTrials] = {}; manifest: list[dict[str, object]] = []
    for session, source_record in zip(SESSION_IDS, sources, strict=True):
        validate_formal_run("1train")
        if session not in records or set(records[session]) != {"1train"}:
            raise ValueError("MOABB supplied a non-training or missing Lee2019 session.")
        source = data_directory / str(source_record["source_file"])
        sessions[session] = _session_trials(subject, session, records[session]["1train"], source, config)
        manifest.append(source_record)
    return ExternalSubjectData(subject, sessions, manifest)


def _spectral_features(trials: SessionTrials, config: Mapping[str, Any]) -> np.ndarray:
    inherited = config["inherited_scientific_config"]; channels = inherited["spectral_channels"]
    indices = [trials.channels.index(channel) for channel in channels]
    data = trials.fixed[:, indices]
    nperseg = int(round(float(inherited["spectral_welch_window_seconds"]) * trials.sfreq))
    frequencies, power = welch(data, fs=trials.sfreq, window="hann", nperseg=nperseg, noverlap=0, detrend="constant", scaling="density", axis=-1)
    columns: list[np.ndarray] = []
    for channel_index in range(len(channels)):
        for bounds in inherited["spectral_bands_hz"].values():
            mask = (frequencies >= float(bounds[0])) & (frequencies <= float(bounds[1]))
            columns.append(np.mean(10 * np.log10(power[:, channel_index, mask]), axis=1))
    features = np.column_stack(columns)
    if features.shape != (trials.labels.size, 6) or not np.isfinite(features).all():
        raise RuntimeError("External frozen spectral feature schema is invalid.")
    return features


def _metrics(y: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    left_left, left_right, right_left, right_right = (int(value) for value in matrix.ravel())
    return {"balanced_accuracy": float(balanced_accuracy_score(y, prediction)), "accuracy": float(accuracy_score(y, prediction)), "macro_f1": float(f1_score(y, prediction, average="macro")), "roc_auc": float(roc_auc_score(y, score)), "left_recall": float(recall_score(y, prediction, pos_label=0)), "right_recall": float(recall_score(y, prediction, pos_label=1)), "true_left_predicted_left": left_left, "true_left_predicted_right": left_right, "true_right_predicted_left": right_left, "true_right_predicted_right": right_right, "trial_count": int(y.size)}


def evaluate_subject(data: ExternalSubjectData, parent_decoder: Mapping[str, Any], config: Mapping[str, Any], model: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if model not in {"spectral_baseline_6", "csp_4_empirical"}:
        raise ValueError("Model is outside the frozen external transfer set.")
    predictions: list[dict[str, object]]=[]; scores: list[dict[str, object]]=[]; audits: list[dict[str, object]]=[]
    for fold in cross_session_folds():
        source, target = data.sessions[fold.train_session], data.sessions[fold.test_session]
        validate_source_only_fit(fold.train_session, fold.test_session, [x.key for x in source.identities], [x.key for x in target.identities])
        x_train = _spectral_features(source, config) if model.startswith("spectral") else source.csp
        x_test = _spectral_features(target, config) if model.startswith("spectral") else target.csp
        pipeline = build_candidate_pipeline(model, parent_decoder); pipeline.fit(x_train, source.labels)
        predicted = np.asarray(pipeline.predict(x_test), dtype=int); decision = np.asarray(pipeline.decision_function(x_test), dtype=float)
        metric = _metrics(target.labels, predicted, decision)
        scores.append({"subject": data.subject, "model": model, "train_session": fold.train_session, "test_session": fold.test_session, **metric})
        train_keys = [x.key for x in source.identities]; test_keys = [x.key for x in target.identities]
        audits.append({"subject": data.subject, "model": model, "train_session": fold.train_session, "test_session": fold.test_session, "train_trial_keys": "/".join(train_keys), "test_trial_keys": "/".join(test_keys), "csp_fit_trial_keys": "/".join(train_keys) if model.startswith("csp") else "not_applicable", "scaler_fit_trial_keys": "/".join(train_keys), "lda_fit_trial_keys": "/".join(train_keys), "source_session_only": True})
        for identity, true, value, confidence in zip(target.identities, target.labels, predicted, decision, strict=True):
            predictions.append({"subject": data.subject, "model": model, "train_session": fold.train_session, "test_session": fold.test_session, "trial_key": identity.key, "true_class": "left_hand" if true == 0 else "right_hand", "true_label": int(true), "predicted_class": "left_hand" if value == 0 else "right_hand", "predicted_label": int(value), "decision_score_right": float(confidence), "source_file": identity.source_file, "source_sha256": identity.source_sha256})
    direction_scores = [float(row["balanced_accuracy"]) for row in scores]
    subject = {"subject": data.subject, "model": model, "primary_mean_cross_session_balanced_accuracy": participant_mean_score(direction_scores), "session1_to_session2_balanced_accuracy": direction_scores[0], "session2_to_session1_balanced_accuracy": direction_scores[1], "group_observational_unit": "participant"}
    return predictions, scores, audits, subject


def bootstrap_median(values: Sequence[float], resamples: int, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    samples = [np.median(rng.choice(data, data.size, replace=True)) for _ in range(resamples)]
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def sign_test_above_chance(values: Sequence[float]) -> dict[str, object]:
    data = np.asarray(values, dtype=float); above=int(np.sum(data>.5)); below=int(np.sum(data<.5)); ties=int(np.sum(data==.5))
    return {"above_count": above, "below_count": below, "tie_count": ties, "one_sided_exact_sign_p": float(binomtest(above, above + below, .5, alternative="greater").pvalue) if above + below else 1.0}


def paired_summary(left: Sequence[float], right: Sequence[float], seed: int, resamples: int) -> dict[str, object]:
    diff = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    statistic, p_value = wilcoxon(diff, alternative="two-sided", zero_method="wilcox") if np.any(diff) else (0., 1.)
    return {"paired_n": int(diff.size), "median_difference": float(np.median(diff)), "bootstrap_median_95": list(bootstrap_median(diff, resamples, seed)), "wilcoxon_statistic": float(statistic), "wilcoxon_p": float(p_value)}
