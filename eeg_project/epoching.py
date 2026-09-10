"""Reusable annotation, event, epoch, and trial-quality operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np

from .preprocessing import (
    DEFAULT_DATA_DIRECTORY,
    PreprocessedRecording,
    PreprocessingConfig,
    preprocess_recording,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPOCHING_CONFIG_PATH = PROJECT_ROOT / "config" / "epoching.json"
BILATERAL_MOTOR_IMAGERY_RUNS = (6, 10, 14)


@dataclass(frozen=True)
class EpochWindow:
    """An inclusive event-relative epoch interval with baseline policy."""

    tmin_seconds: float
    tmax_seconds: float
    baseline: None


@dataclass(frozen=True)
class TrialQualitySettings:
    """Small exploratory robust-quality policy."""

    modified_robust_z_threshold: float
    frontal_channels: tuple[str, ...]


@dataclass(frozen=True)
class EpochingConfig:
    """Versioned scientific policy for event and epoch construction."""

    schema_version: int
    supported_runs: tuple[int, ...]
    event_id: dict[str, int]
    task_semantics: dict[str, str]
    rest_semantic: str
    task_epoch: EpochWindow
    rest_epoch: EpochWindow
    provisional_task_analysis_interval_seconds: tuple[float, float]
    filter_half_support_seconds: float
    trial_quality: TrialQualitySettings
    persist_epoch_binaries: bool


@dataclass(frozen=True)
class EpochRecord:
    """Trace one prospective epoch back to its annotation and source run."""

    subject: int
    run: int
    run_trial_index: int
    annotation_index: int
    annotation: str
    semantic_condition: str
    event_code: int
    event_sample: int
    event_time_seconds: float
    annotation_duration_seconds: float
    previous_annotation: str
    next_annotation: str
    epoch_tmin_seconds: float
    epoch_tmax_seconds: float
    epoch_start_sample: int
    epoch_stop_sample_inclusive: int
    epoch_start_seconds: float
    epoch_stop_seconds: float
    n_samples: int
    valid: bool
    exclusion_reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a CSV/JSON-ready dictionary."""
        return asdict(self)


@dataclass
class RunEpochs:
    """Preprocessed continuous input plus task/rest epochs and provenance."""

    preprocessing: PreprocessedRecording
    task_epochs: mne.Epochs
    rest_epochs: mne.Epochs
    annotation_rows: list[dict[str, object]]
    task_records: list[EpochRecord]
    rest_records: list[EpochRecord]
    event_id: dict[str, int]


def load_epoching_config(
    path: Path = DEFAULT_EPOCHING_CONFIG_PATH,
) -> EpochingConfig:
    """Load and validate the machine-readable epoching policy."""
    with path.expanduser().resolve().open(encoding="utf-8") as config_file:
        payload: dict[str, Any] = json.load(config_file)
    task_window = EpochWindow(**payload["task_epoch"])
    rest_window = EpochWindow(**payload["rest_epoch"])
    quality = TrialQualitySettings(
        modified_robust_z_threshold=float(
            payload["trial_quality"]["modified_robust_z_threshold"]
        ),
        frontal_channels=tuple(payload["trial_quality"]["frontal_channels"]),
    )
    config = EpochingConfig(
        schema_version=int(payload["schema_version"]),
        supported_runs=tuple(int(run) for run in payload["supported_runs"]),
        event_id={key: int(value) for key, value in payload["event_id"].items()},
        task_semantics=dict(payload["task_semantics"]),
        rest_semantic=str(payload["rest_semantic"]),
        task_epoch=task_window,
        rest_epoch=rest_window,
        provisional_task_analysis_interval_seconds=tuple(
            float(value)
            for value in payload["provisional_task_analysis_interval_seconds"]
        ),
        filter_half_support_seconds=float(payload["filter_half_support_seconds"]),
        trial_quality=quality,
        persist_epoch_binaries=bool(payload["persist_epoch_binaries"]),
    )
    validate_epoching_config(config)
    return config


def validate_epoching_config(config: EpochingConfig) -> None:
    """Reject unsupported semantics or internally inconsistent windows."""
    if config.schema_version != 1:
        raise ValueError(f"Unsupported epoching schema: {config.schema_version}.")
    if config.supported_runs != BILATERAL_MOTOR_IMAGERY_RUNS:
        raise ValueError("Current epoching policy must explicitly support runs 6/10/14.")
    if set(config.event_id) != {"T0", "T1", "T2"}:
        raise ValueError("Event mapping must contain exactly T0, T1, and T2.")
    if len(set(config.event_id.values())) != 3 or min(config.event_id.values()) <= 0:
        raise ValueError("Event IDs must be distinct positive integers.")
    if config.task_semantics != {
        "T1": "both_fists_imagery",
        "T2": "both_feet_imagery",
    }:
        raise ValueError("Unexpected bilateral motor-imagery semantics.")
    for window in (config.task_epoch, config.rest_epoch):
        if window.tmax_seconds <= window.tmin_seconds:
            raise ValueError("Epoch tmax must be greater than tmin.")
        if window.baseline is not None:
            raise ValueError("This milestone deliberately uses baseline=None.")
    if (config.task_epoch.tmin_seconds, config.task_epoch.tmax_seconds) != (-2.0, 4.0):
        raise ValueError("The current task storage window must be -2 to +4 seconds.")
    if (config.rest_epoch.tmin_seconds, config.rest_epoch.tmax_seconds) != (0.0, 4.0):
        raise ValueError("The current rest/context window must be 0 to +4 seconds.")
    analysis_start, analysis_stop = config.provisional_task_analysis_interval_seconds
    if not 0 <= analysis_start < analysis_stop <= config.task_epoch.tmax_seconds:
        raise ValueError("Provisional task-analysis interval lies outside the task period.")
    if config.filter_half_support_seconds <= 0:
        raise ValueError("Filter half-support must be positive.")
    if config.trial_quality.modified_robust_z_threshold <= 0:
        raise ValueError("Trial-quality robust-z threshold must be positive.")
    if not config.trial_quality.frontal_channels:
        raise ValueError("At least one frontal trial-quality channel is required.")
    if config.persist_epoch_binaries:
        raise ValueError("Epoch binary persistence is not approved at this stage.")


def semantic_mapping_for_run(run: int, config: EpochingConfig) -> dict[str, str]:
    """Return run-family semantics or fail instead of silently mislabeling."""
    if run not in config.supported_runs:
        raise ValueError(
            f"Run {run} is not supported by the bilateral motor-imagery mapping; "
            "T1/T2 semantics differ across EEGBCI run families."
        )
    return {"T0": config.rest_semantic, **config.task_semantics}


def expected_epoch_samples(window: EpochWindow, sampling_frequency: float) -> int:
    """Calculate MNE's inclusive endpoint sample count independently."""
    return int(round((window.tmax_seconds - window.tmin_seconds) * sampling_frequency)) + 1


def annotations_to_events(
    raw: mne.io.BaseRaw,
    run: int,
    config: EpochingConfig,
) -> tuple[np.ndarray, dict[str, int], list[dict[str, object]]]:
    """Convert all supported annotations to events and audit every mapping."""
    semantics = semantic_mapping_for_run(run, config)
    descriptions = set(str(value) for value in raw.annotations.description)
    if descriptions != set(config.event_id):
        raise RuntimeError(f"Unexpected run {run} annotation descriptions: {descriptions}.")
    events, event_id = mne.events_from_annotations(
        raw,
        event_id=config.event_id,
        use_rounding=True,
        verbose="error",
    )
    if event_id != config.event_id:
        raise RuntimeError(f"MNE event mapping differs from policy: {event_id}.")
    if len(events) != len(raw.annotations):
        raise RuntimeError("MNE event count differs from annotation count.")
    if raw.first_samp != 0 or raw.annotations.orig_time != raw.info["meas_date"]:
        raise RuntimeError(
            "Independent sample audit currently expects first_samp=0 and matching origins."
        )

    sfreq = float(raw.info["sfreq"])
    annotations = list(raw.annotations)
    rows: list[dict[str, object]] = []
    for index, annotation in enumerate(annotations):
        description = str(annotation["description"])
        onset = float(annotation["onset"])
        duration = float(annotation["duration"])
        end = onset + duration
        manual_sample = int(np.rint(onset * sfreq)) + raw.first_samp
        event_sample = int(events[index, 0])
        if int(events[index, 2]) != config.event_id[description]:
            raise RuntimeError(
                f"Run {run} annotation {index}: event code does not match {description}."
            )
        if event_sample != manual_sample:
            raise RuntimeError(
                f"Run {run} annotation {index}: MNE sample {event_sample} differs "
                f"from round(onset*fs)={manual_sample}."
            )
        next_onset = (
            float(annotations[index + 1]["onset"])
            if index + 1 < len(annotations)
            else np.nan
        )
        gap = next_onset - end if np.isfinite(next_onset) else np.nan
        rows.append(
            {
                "run": run,
                "annotation_index": index,
                "description": description,
                "semantic_condition": semantics[description],
                "onset_seconds": onset,
                "duration_seconds": duration,
                "end_seconds": end,
                "event_code": int(events[index, 2]),
                "event_sample": event_sample,
                "manual_event_sample": manual_sample,
                "previous_annotation": (
                    str(annotations[index - 1]["description"]) if index else ""
                ),
                "next_annotation": (
                    str(annotations[index + 1]["description"])
                    if index + 1 < len(annotations)
                    else ""
                ),
                "gap_to_next_seconds": float(gap) if np.isfinite(gap) else "",
                "overlaps_next": bool(np.isfinite(gap) and gap < -1e-9),
                "inside_valid_duration": bool(end <= raw.n_times / sfreq + 1e-12),
            }
        )
    return events, dict(event_id), rows


def build_epoch_records(
    subject: int,
    run: int,
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    annotation_rows: Sequence[dict[str, object]],
    annotations: tuple[str, ...],
    window: EpochWindow,
    config: EpochingConfig,
) -> list[EpochRecord]:
    """Construct boundary-audited metadata for task or rest events."""
    semantics = semantic_mapping_for_run(run, config)
    sfreq = float(raw.info["sfreq"])
    start_offset = int(round(window.tmin_seconds * sfreq))
    stop_offset = int(round(window.tmax_seconds * sfreq))
    expected_samples = expected_epoch_samples(window, sfreq)
    records: list[EpochRecord] = []

    for annotation_row, event in zip(annotation_rows, events, strict=True):
        annotation = str(annotation_row["description"])
        if annotation not in annotations:
            continue
        event_sample = int(event[0])
        start_sample = event_sample + start_offset
        stop_sample = event_sample + stop_offset
        valid = bool(
            start_sample >= raw.first_samp
            and stop_sample < raw.first_samp + raw.n_times
        )
        reason = "" if valid else "epoch_crosses_valid_recording_boundary"
        n_samples = stop_sample - start_sample + 1
        if n_samples != expected_samples:
            raise RuntimeError("Independent epoch sample count is inconsistent.")
        records.append(
            EpochRecord(
                subject=subject,
                run=run,
                run_trial_index=len(records),
                annotation_index=int(annotation_row["annotation_index"]),
                annotation=annotation,
                semantic_condition=semantics[annotation],
                event_code=int(event[2]),
                event_sample=event_sample,
                event_time_seconds=(event_sample - raw.first_samp) / sfreq,
                annotation_duration_seconds=float(annotation_row["duration_seconds"]),
                previous_annotation=str(annotation_row["previous_annotation"]),
                next_annotation=str(annotation_row["next_annotation"]),
                epoch_tmin_seconds=window.tmin_seconds,
                epoch_tmax_seconds=window.tmax_seconds,
                epoch_start_sample=start_sample,
                epoch_stop_sample_inclusive=stop_sample,
                epoch_start_seconds=(start_sample - raw.first_samp) / sfreq,
                epoch_stop_seconds=(stop_sample - raw.first_samp) / sfreq,
                n_samples=n_samples,
                valid=valid,
                exclusion_reason=reason,
            )
        )
    return records


def _create_mne_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    records: Sequence[EpochRecord],
    event_id: dict[str, int],
    window: EpochWindow,
) -> mne.Epochs:
    """Create preloaded epochs only for deterministically valid records."""
    valid_codes = {record.event_code for record in records if record.valid}
    valid_event_rows = np.array(
        [event for event in events if int(event[2]) in valid_codes], dtype=int
    )
    # Select exact event samples as well as codes in case future annotations use
    # another event with the same code that is not part of this collection.
    valid_pairs = {
        (record.event_sample, record.event_code) for record in records if record.valid
    }
    valid_event_rows = np.array(
        [event for event in valid_event_rows if (int(event[0]), int(event[2])) in valid_pairs],
        dtype=int,
    )
    if valid_event_rows.size == 0:
        raise RuntimeError("No valid events remain for epoch extraction.")
    selected_event_id = {
        semantic: code for semantic, code in event_id.items() if code in valid_codes
    }
    epochs = mne.Epochs(
        raw,
        valid_event_rows,
        event_id=selected_event_id,
        tmin=window.tmin_seconds,
        tmax=window.tmax_seconds,
        baseline=window.baseline,
        picks="eeg",
        preload=True,
        proj=False,
        reject=None,
        flat=None,
        detrend=None,
        reject_by_annotation=False,
        event_repeated="error",
        verbose="error",
    )
    expected = expected_epoch_samples(window, float(raw.info["sfreq"]))
    if epochs.get_data(copy=False).shape[2] != expected:
        raise RuntimeError("MNE epoch shape differs from inclusive endpoint calculation.")
    if epochs.baseline is not None:
        raise RuntimeError("Time-domain baseline correction was unexpectedly applied.")
    return epochs


def extract_run_epochs(
    subject: int,
    run: int,
    preprocessing_config: PreprocessingConfig,
    epoching_config: EpochingConfig,
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
) -> RunEpochs:
    """Preprocess one run, audit events, and extract task/rest collections."""
    preprocessing = preprocess_recording(
        subject, run, preprocessing_config, data_directory=data_directory
    )
    raw = preprocessing.filtered
    events, event_id, annotation_rows = annotations_to_events(
        raw, run, epoching_config
    )
    task_records = build_epoch_records(
        subject,
        run,
        raw,
        events,
        annotation_rows,
        ("T1", "T2"),
        epoching_config.task_epoch,
        epoching_config,
    )
    rest_records = build_epoch_records(
        subject,
        run,
        raw,
        events,
        annotation_rows,
        ("T0",),
        epoching_config.rest_epoch,
        epoching_config,
    )
    task_event_id = {
        epoching_config.task_semantics[label]: epoching_config.event_id[label]
        for label in ("T1", "T2")
    }
    rest_event_id = {
        epoching_config.rest_semantic: epoching_config.event_id["T0"]
    }
    task_epochs = _create_mne_epochs(
        raw, events, task_records, task_event_id, epoching_config.task_epoch
    )
    rest_epochs = _create_mne_epochs(
        raw, events, rest_records, rest_event_id, epoching_config.rest_epoch
    )
    if len(task_epochs) != sum(record.valid for record in task_records):
        raise RuntimeError("Task Epochs count differs from valid task records.")
    if len(rest_epochs) != sum(record.valid for record in rest_records):
        raise RuntimeError("Rest Epochs count differs from valid rest records.")
    return RunEpochs(
        preprocessing=preprocessing,
        task_epochs=task_epochs,
        rest_epochs=rest_epochs,
        annotation_rows=annotation_rows,
        task_records=task_records,
        rest_records=rest_records,
        event_id=event_id,
    )


def modified_robust_z(values: np.ndarray) -> np.ndarray:
    """Return deterministic modified robust z-scores, with zero-MAD safety."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


def calculate_trial_quality(
    epoch_data_volts: np.ndarray,
    times: np.ndarray,
    channel_names: Sequence[str],
    records: Sequence[EpochRecord],
    settings: TrialQualitySettings,
) -> list[dict[str, object]]:
    """Calculate compact exploratory task-trial metrics and robust flags."""
    if epoch_data_volts.shape[0] != len(records):
        raise ValueError("Epoch data and provenance record counts differ.")
    if epoch_data_volts.shape[1] != len(channel_names):
        raise ValueError("Epoch channel dimension and names differ.")
    if not np.isfinite(epoch_data_volts).all():
        raise RuntimeError("Epoch data contain NaN or infinity.")
    missing_frontal = sorted(set(settings.frontal_channels) - set(channel_names))
    if missing_frontal:
        raise RuntimeError(f"Missing configured frontal channels: {missing_frontal}.")

    data_uv = epoch_data_volts * 1_000_000
    task_mask = (times >= 0) & (times <= 4.0)
    precontext_mask = (times >= -2.0) & (times < 0)
    task_data = data_uv[:, :, task_mask]
    precontext_data = data_uv[:, :, precontext_mask]
    task_p2p = np.ptp(task_data, axis=2)
    full_p2p = np.ptp(data_uv, axis=2)
    precontext_p2p = np.ptp(precontext_data, axis=2)
    task_std = np.std(task_data, axis=2)
    task_steps = np.max(np.abs(np.diff(task_data, axis=2)), axis=(1, 2))
    frontal_indices = [channel_names.index(name) for name in settings.frontal_channels]

    metrics = {
        "max_full_epoch_p2p_uV": np.max(full_p2p, axis=1),
        "max_task_p2p_uV": np.max(task_p2p, axis=1),
        "median_task_p2p_uV": np.median(task_p2p, axis=1),
        "max_precontext_p2p_uV": np.max(precontext_p2p, axis=1),
        "frontal_max_task_p2p_uV": np.max(task_p2p[:, frontal_indices], axis=1),
        "max_task_abs_step_uV": task_steps,
        "minimum_task_channel_std_uV": np.min(task_std, axis=1),
        "median_task_channel_std_uV": np.median(task_std, axis=1),
    }
    robust_scores = {name: modified_robust_z(values) for name, values in metrics.items()}
    channel_p2p_scores = np.column_stack(
        [modified_robust_z(task_p2p[:, index]) for index in range(task_p2p.shape[1])]
    )
    threshold = settings.modified_robust_z_threshold
    rows: list[dict[str, object]] = []
    high_metrics = (
        "max_full_epoch_p2p_uV",
        "max_task_p2p_uV",
        "median_task_p2p_uV",
        "max_precontext_p2p_uV",
        "frontal_max_task_p2p_uV",
        "max_task_abs_step_uV",
        "median_task_channel_std_uV",
    )
    low_metrics = ("minimum_task_channel_std_uV",)

    for index, record in enumerate(records):
        evidence: list[str] = []
        for name in high_metrics:
            score = robust_scores[name][index]
            if score > threshold:
                evidence.append(f"high {name} (robust_z={score:.2f})")
        for name in low_metrics:
            score = robust_scores[name][index]
            if score < -threshold:
                evidence.append(f"low {name} (robust_z={score:.2f})")
        candidate_channels = [
            channel_names[channel_index]
            for channel_index in np.flatnonzero(channel_p2p_scores[index] > threshold)
        ]
        # Channel-specific scores are retained as supporting evidence, but do
        # not independently flag a trial: 64 simultaneous exploratory tests
        # would otherwise create a multiple-comparison candidate explosion.
        if candidate_channels and evidence:
            evidence.append(
                "channel-specific high task p2p: " + "/".join(candidate_channels)
            )
        row: dict[str, object] = {
            **record.to_dict(),
            **{name: float(values[index]) for name, values in metrics.items()},
            **{
                f"{name}_robust_z": float(values[index])
                for name, values in robust_scores.items()
            },
            "channel_specific_p2p_candidate_count": len(candidate_channels),
            "channel_specific_p2p_candidate_channels": "/".join(candidate_channels),
            "quality_status": "statistical candidate" if evidence else "not flagged",
            "quality_evidence": "; ".join(evidence),
            "confirmed_exclusion": False,
        }
        rows.append(row)
    return rows
