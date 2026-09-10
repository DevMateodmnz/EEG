"""Deterministic continuous-EEG preprocessing with immutable EDF inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import mne
import numpy as np
from mne.datasets import eegbci
from mne.filter import create_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "preprocessing.json"


@dataclass(frozen=True)
class FilterSettings:
    """Complete FIR settings needed to reproduce the temporal filter."""

    l_freq_hz: float
    h_freq_hz: float
    l_trans_bandwidth_hz: float
    h_trans_bandwidth_hz: float
    method: str
    fir_design: str
    fir_window: str
    filter_length_samples: int
    phase: str
    pad: str


@dataclass(frozen=True)
class PreprocessingConfig:
    """Versioned scientific preprocessing policy."""

    schema_version: int
    sampling_frequency_hz: float
    tail_policy: str
    reference: str
    excluded_reference_channels: tuple[str, ...]
    filter: FilterSettings
    notch_frequencies_hz: tuple[float, ...]


@dataclass
class ValidRecording:
    """A copied valid-data view plus immutable-source provenance."""

    raw: mne.io.BaseRaw
    source_path: Path
    source_sha256: str
    original_samples: int
    valid_samples: int
    excluded_tail_samples: int
    other_all_channel_zero_samples: int


@dataclass
class PreprocessedRecording:
    """Three controlled stages retained for validation and reporting."""

    stored_valid: mne.io.BaseRaw
    referenced: mne.io.BaseRaw
    filtered: mne.io.BaseRaw
    source_path: Path
    source_sha256: str
    original_samples: int
    valid_samples: int
    excluded_tail_samples: int
    excluded_reference_channels: tuple[str, ...]
    filter_coefficients: np.ndarray


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a source file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preprocessing_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> PreprocessingConfig:
    """Load and strictly validate the version-controlled JSON configuration."""
    with path.expanduser().resolve().open(encoding="utf-8") as config_file:
        payload: dict[str, Any] = json.load(config_file)

    filter_payload = payload["filter"]
    settings = FilterSettings(
        l_freq_hz=float(filter_payload["l_freq_hz"]),
        h_freq_hz=float(filter_payload["h_freq_hz"]),
        l_trans_bandwidth_hz=float(filter_payload["l_trans_bandwidth_hz"]),
        h_trans_bandwidth_hz=float(filter_payload["h_trans_bandwidth_hz"]),
        method=str(filter_payload["method"]),
        fir_design=str(filter_payload["fir_design"]),
        fir_window=str(filter_payload["fir_window"]),
        filter_length_samples=int(filter_payload["filter_length_samples"]),
        phase=str(filter_payload["phase"]),
        pad=str(filter_payload["pad"]),
    )
    config = PreprocessingConfig(
        schema_version=int(payload["schema_version"]),
        sampling_frequency_hz=float(payload["sampling_frequency_hz"]),
        tail_policy=str(payload["tail_policy"]),
        reference=str(payload["reference"]),
        excluded_reference_channels=tuple(payload["excluded_reference_channels"]),
        filter=settings,
        notch_frequencies_hz=tuple(float(value) for value in payload["notch_frequencies_hz"]),
    )
    validate_config(config)
    return config


def validate_config(config: PreprocessingConfig) -> None:
    """Reject incomplete or scientifically inconsistent production settings."""
    if config.schema_version != 1:
        raise ValueError(f"Unsupported preprocessing schema: {config.schema_version}.")
    if config.tail_policy != "exclude_trailing_all_channel_zeros":
        raise ValueError(f"Unsupported tail policy: {config.tail_policy}.")
    if config.reference != "average":
        raise ValueError("The current validated production reference must be average.")
    if config.notch_frequencies_hz:
        raise ValueError("The validated production configuration contains no notch.")
    settings = config.filter
    if settings.method != "fir" or settings.fir_design != "firwin":
        raise ValueError("The validated production filter is an FIR firwin design.")
    if settings.phase != "zero" or settings.pad != "reflect_limited":
        raise ValueError("Unexpected production phase or padding configuration.")
    if settings.filter_length_samples <= 1 or settings.filter_length_samples % 2 == 0:
        raise ValueError("The linear-phase FIR length must be an odd integer > 1.")
    if not 0 < settings.l_freq_hz < settings.h_freq_hz < config.sampling_frequency_hz / 2:
        raise ValueError("Filter passband edges are inconsistent with Nyquist.")


def find_valid_stop(data: np.ndarray) -> tuple[int, int, int]:
    """Detect an all-channel zero tail and any earlier all-channel zero samples."""
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("Expected a non-empty (channels, samples) EEG array.")
    all_channel_zero = np.all(data == 0, axis=0)
    if all_channel_zero[-1]:
        nonzero_indices = np.flatnonzero(~all_channel_zero)
        valid_stop = int(nonzero_indices[-1] + 1) if nonzero_indices.size else 0
    else:
        valid_stop = data.shape[1]
    tail_samples = int(data.shape[1] - valid_stop)
    earlier_zero_samples = int(np.count_nonzero(all_channel_zero[:valid_stop]))
    return valid_stop, tail_samples, earlier_zero_samples


def load_valid_recording(
    subject: int,
    run: int,
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
) -> ValidRecording:
    """Load an EDF and create a preloaded copy excluding its detected zero tail."""
    if not 1 <= subject <= 109:
        raise ValueError(f"Subject must be in 1-109; received {subject}.")
    if run not in (4, 6, 8, 10, 12, 14):
        raise ValueError("Supported motor-imagery acquisition runs are 4, 6, 8, 10, 12, and 14.")

    (downloaded_file,) = eegbci.load_data(
        subjects=subject,
        runs=run,
        path=data_directory,
        update_path=False,
    )
    source_path = Path(downloaded_file).resolve()
    source_digest = sha256_file(source_path)
    source_raw = mne.io.read_raw_edf(source_path, preload=False, verbose="error")
    eegbci.standardize(source_raw)
    original_samples = source_raw.n_times
    complete_data = source_raw.get_data()
    valid_stop, tail_samples, earlier_zero_samples = find_valid_stop(complete_data)
    if earlier_zero_samples:
        raise RuntimeError(
            f"Found {earlier_zero_samples} all-channel zero samples before the tail."
        )
    if valid_stop < 2:
        raise RuntimeError("Fewer than two valid samples remain after tail exclusion.")

    valid_raw = source_raw.copy().crop(tmax=source_raw.times[valid_stop - 1])
    valid_raw.load_data(verbose="error")
    if valid_raw.n_times != valid_stop:
        raise RuntimeError(
            f"Valid copy contains {valid_raw.n_times} samples; expected {valid_stop}."
        )
    if sha256_file(source_path) != source_digest:
        raise RuntimeError("Source EDF changed while the valid copy was being created.")
    return ValidRecording(
        raw=valid_raw,
        source_path=source_path,
        source_sha256=source_digest,
        original_samples=original_samples,
        valid_samples=valid_stop,
        excluded_tail_samples=tail_samples,
        other_all_channel_zero_samples=earlier_zero_samples,
    )


def apply_analysis_reference(
    raw: mne.io.BaseRaw,
    excluded_channels: tuple[str, ...] = (),
) -> mne.io.BaseRaw:
    """Apply the selected average reference to a copy and verify the equation."""
    missing = sorted(set(excluded_channels) - set(raw.ch_names))
    if missing:
        raise ValueError(f"Reference exclusions are absent from the recording: {missing}.")
    if raw.info["bads"]:
        raise RuntimeError("Unexpected pre-existing bad-channel marks require review.")

    original = raw.get_data().copy()
    included_indices = [
        index
        for index, channel in enumerate(raw.ch_names)
        if channel not in excluded_channels
    ]
    if not included_indices:
        raise ValueError("Average reference requires at least one included channel.")
    expected_reference = np.mean(original[included_indices], axis=0)

    referenced = raw.copy()
    referenced.info["bads"] = list(excluded_channels)
    referenced.set_eeg_reference(
        ref_channels="average", projection=False, verbose="error"
    )
    actual = referenced.get_data()

    # The validated production configuration currently has no exclusions. Keep
    # this strong whole-array equality check rather than silently assuming how a
    # future bad-channel policy should transform excluded channels.
    if excluded_channels:
        raise NotImplementedError(
            "A non-empty exclusion policy requires an explicit reviewed transform."
        )
    if not np.allclose(actual, original - expected_reference, rtol=1e-12, atol=1e-15):
        raise RuntimeError("Average reference differs from explicit subtraction.")
    if np.max(np.abs(np.mean(actual, axis=0))) > 1e-15:
        raise RuntimeError("Average-referenced channel mean is not numerically zero.")
    if not np.array_equal(raw.get_data(), original):
        raise RuntimeError("Average referencing modified its input Raw object.")
    return referenced


def design_production_filter(
    sampling_frequency: float,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Return the exact FIR coefficient vector specified by the configuration."""
    if not np.isclose(sampling_frequency, config.sampling_frequency_hz):
        raise ValueError(
            f"Expected {config.sampling_frequency_hz:g} Hz; received {sampling_frequency:g} Hz."
        )
    settings = config.filter
    coefficients = create_filter(
        None,
        sampling_frequency,
        l_freq=settings.l_freq_hz,
        h_freq=settings.h_freq_hz,
        filter_length=settings.filter_length_samples,
        l_trans_bandwidth=settings.l_trans_bandwidth_hz,
        h_trans_bandwidth=settings.h_trans_bandwidth_hz,
        method=settings.method,
        phase=settings.phase,
        fir_window=settings.fir_window,
        fir_design=settings.fir_design,
        verbose=False,
    )
    if coefficients.size != settings.filter_length_samples:
        raise RuntimeError("MNE produced an unexpected FIR coefficient count.")
    if not np.allclose(coefficients, coefficients[::-1], rtol=0, atol=1e-15):
        raise RuntimeError("Production FIR coefficients are not symmetric linear phase.")
    return coefficients


def apply_production_filter(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
) -> mne.io.BaseRaw:
    """Apply the configured zero-phase FIR to one continuous copied recording."""
    sampling_frequency = float(raw.info["sfreq"])
    design_production_filter(sampling_frequency, config)
    settings = config.filter
    snapshot = raw.get_data().copy()
    annotations_before = raw.annotations.copy()

    filtered = raw.copy().filter(
        l_freq=settings.l_freq_hz,
        h_freq=settings.h_freq_hz,
        picks="eeg",
        filter_length=settings.filter_length_samples,
        l_trans_bandwidth=settings.l_trans_bandwidth_hz,
        h_trans_bandwidth=settings.h_trans_bandwidth_hz,
        method=settings.method,
        phase=settings.phase,
        fir_window=settings.fir_window,
        fir_design=settings.fir_design,
        pad=settings.pad,
        skip_by_annotation=(),
        verbose="error",
    )
    output = filtered.get_data()
    if output.shape != snapshot.shape or not np.isfinite(output).all():
        raise RuntimeError("Filtered EEG has an invalid shape or non-finite values.")
    if not np.array_equal(raw.get_data(), snapshot):
        raise RuntimeError("Filtering modified its referenced input Raw object.")
    if annotations_before != filtered.annotations:
        raise RuntimeError("Filtering unexpectedly changed annotations.")
    return filtered


def preprocess_recording(
    subject: int,
    run: int,
    config: PreprocessingConfig,
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
) -> PreprocessedRecording:
    """Execute the approved crop → reference → continuous-filter pipeline."""
    valid = load_valid_recording(subject, run, data_directory=data_directory)
    if not np.isclose(valid.raw.info["sfreq"], config.sampling_frequency_hz):
        raise RuntimeError("Recording sampling frequency differs from configuration.")
    referenced = apply_analysis_reference(
        valid.raw, config.excluded_reference_channels
    )
    filtered = apply_production_filter(referenced, config)
    coefficients = design_production_filter(
        float(valid.raw.info["sfreq"]), config
    )
    if sha256_file(valid.source_path) != valid.source_sha256:
        raise RuntimeError("Source EDF changed during preprocessing.")
    return PreprocessedRecording(
        stored_valid=valid.raw,
        referenced=referenced,
        filtered=filtered,
        source_path=valid.source_path,
        source_sha256=valid.source_sha256,
        original_samples=valid.original_samples,
        valid_samples=valid.valid_samples,
        excluded_tail_samples=valid.excluded_tail_samples,
        excluded_reference_channels=config.excluded_reference_channels,
        filter_coefficients=coefficients,
    )
