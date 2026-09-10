"""Reusable paired-rest event-related spectral operations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
from mne.time_frequency import morlet, tfr_array_morlet
import numpy as np

from .epoching import EpochRecord, RunEpochs
from .provenance import canonical_source_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_RELATED_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "event_related_spectral.json"
)


@dataclass(frozen=True)
class FrequencyGrid:
    """Inclusive equally spaced frequency centers in hertz."""

    start: float
    stop: float
    step: float


@dataclass(frozen=True)
class WaveletSettings:
    """Explicit Morlet construction and output-sampling policy."""

    cycles_per_hz: float
    zero_mean: bool
    use_fft: bool
    decimation: int


@dataclass(frozen=True)
class FrequencyBand:
    """Inclusive named frequency-center interval."""

    name: str
    lower_hz: float
    upper_hz: float


@dataclass(frozen=True)
class EventRelatedSpectralConfig:
    """Versioned policy for paired-rest Morlet analysis."""

    schema_version: int
    method: str
    frequency_grid: FrequencyGrid
    wavelet: WaveletSettings
    task_analysis_interval_seconds: tuple[float, float]
    paired_rest_reference_interval_seconds: tuple[float, float]
    display_interval_seconds: tuple[float, float]
    normalization: str
    bands: tuple[FrequencyBand, ...]
    sensorimotor_channels: tuple[str, ...]
    task_window_sensitivity_seconds: tuple[tuple[float, float], ...]
    rest_window_sensitivity_seconds: tuple[tuple[float, float], ...]
    continuous_filter_half_support_seconds: float
    primary_trials: str
    qc_sensitivity: str
    persist_full_tfr_arrays: bool


@dataclass(frozen=True)
class TrialPair:
    """Trace one task epoch to its immediately preceding T0 epoch."""

    task_array_index: int
    rest_array_index: int
    task: EpochRecord
    rest: EpochRecord
    source_path: Path
    source_sha256: str


@dataclass
class SpectralEpochDataset:
    """Row-aligned task/rest arrays and provenance before the TFR."""

    task_data_volts: np.ndarray
    rest_data_volts: np.ndarray
    task_times: np.ndarray
    rest_times: np.ndarray
    channel_names: list[str]
    sampling_frequency_hz: float
    pairs: list[TrialPair]


@dataclass
class TimeFrequencyPower:
    """Trial-level Morlet power plus explicit coordinate vectors."""

    power_v2: np.ndarray
    frequencies_hz: np.ndarray
    times_seconds: np.ndarray
    n_cycles: np.ndarray
    wavelet_lengths_samples: np.ndarray


def _float_pair(values: Sequence[object], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    return float(values[0]), float(values[1])


def load_event_related_spectral_config(
    path: Path = DEFAULT_EVENT_RELATED_CONFIG_PATH,
) -> EventRelatedSpectralConfig:
    """Load and strictly validate the event-related spectral policy."""
    with path.expanduser().resolve().open(encoding="utf-8") as config_file:
        payload: dict[str, Any] = json.load(config_file)
    grid_payload = payload["frequency_grid_hz"]
    wavelet_payload = payload["wavelet"]
    config = EventRelatedSpectralConfig(
        schema_version=int(payload["schema_version"]),
        method=str(payload["method"]),
        frequency_grid=FrequencyGrid(
            start=float(grid_payload["start"]),
            stop=float(grid_payload["stop"]),
            step=float(grid_payload["step"]),
        ),
        wavelet=WaveletSettings(
            cycles_per_hz=float(wavelet_payload["cycles_per_hz"]),
            zero_mean=bool(wavelet_payload["zero_mean"]),
            use_fft=bool(wavelet_payload["use_fft"]),
            decimation=int(wavelet_payload["decimation"]),
        ),
        task_analysis_interval_seconds=_float_pair(
            payload["task_analysis_interval_seconds"], "task interval"
        ),
        paired_rest_reference_interval_seconds=_float_pair(
            payload["paired_rest_reference_interval_seconds"], "rest interval"
        ),
        display_interval_seconds=_float_pair(
            payload["display_interval_seconds"], "display interval"
        ),
        normalization=str(payload["normalization"]),
        bands=tuple(
            FrequencyBand(name, *_float_pair(values, f"{name} band"))
            for name, values in payload["bands_hz"].items()
        ),
        sensorimotor_channels=tuple(payload["sensorimotor_channels"]),
        task_window_sensitivity_seconds=tuple(
            _float_pair(values, "task sensitivity interval")
            for values in payload["task_window_sensitivity_seconds"]
        ),
        rest_window_sensitivity_seconds=tuple(
            _float_pair(values, "rest sensitivity interval")
            for values in payload["rest_window_sensitivity_seconds"]
        ),
        continuous_filter_half_support_seconds=float(
            payload["continuous_filter_half_support_seconds"]
        ),
        primary_trials=str(payload["primary_trials"]),
        qc_sensitivity=str(payload["qc_sensitivity"]),
        persist_full_tfr_arrays=bool(payload["persist_full_tfr_arrays"]),
    )
    validate_event_related_spectral_config(config)
    return config


def validate_event_related_spectral_config(
    config: EventRelatedSpectralConfig,
) -> None:
    """Reject ambiguous or inconsistent scientific settings."""
    if config.schema_version != 1:
        raise ValueError(f"Unsupported event-related schema: {config.schema_version}.")
    if config.method != "morlet":
        raise ValueError("The current event-related method must be Morlet wavelets.")
    grid = config.frequency_grid
    if not 0 < grid.start <= grid.stop or grid.step <= 0:
        raise ValueError("Frequency-grid bounds or step are invalid.")
    frequencies = frequency_vector(config)
    if not np.isclose(frequencies[-1], grid.stop):
        raise ValueError("Frequency step must land exactly on the configured stop.")
    if config.wavelet.cycles_per_hz <= 0 or config.wavelet.decimation < 1:
        raise ValueError("Wavelet cycles and decimation must be positive.")
    if config.normalization != "percent_change":
        raise ValueError("The current normalization must be paired percent change.")
    if config.task_analysis_interval_seconds != (1.0, 3.0):
        raise ValueError("The reviewed task interval must be +1 to +3 seconds.")
    if config.paired_rest_reference_interval_seconds != (1.1, 3.1):
        raise ValueError("The reviewed paired-rest interval must be +1.1 to +3.1 seconds.")
    for name, interval, bounds in (
        ("task", config.task_analysis_interval_seconds, (-2.0, 4.0)),
        ("rest", config.paired_rest_reference_interval_seconds, (0.0, 4.0)),
        ("display", config.display_interval_seconds, (-2.0, 4.0)),
    ):
        if not bounds[0] <= interval[0] < interval[1] <= bounds[1]:
            raise ValueError(f"Configured {name} interval is outside stored data.")
    required_bands = {
        "mu": (8.0, 13.0),
        "beta": (14.0, 30.0),
        "central_12_13": (12.0, 13.0),
    }
    actual_bands = {
        band.name: (band.lower_hz, band.upper_hz) for band in config.bands
    }
    if actual_bands != required_bands:
        raise ValueError("Unexpected project frequency-band definitions.")
    for band in config.bands:
        mask = band_frequency_mask(frequencies, band)
        if not mask.any():
            raise ValueError(f"Band {band.name} contains no configured frequency center.")
    if config.sensorimotor_channels != ("C3", "Cz", "C4"):
        raise ValueError("Primary sensorimotor channels must be C3/Cz/C4.")
    if config.continuous_filter_half_support_seconds <= 0:
        raise ValueError("Continuous-filter half-support must be positive.")
    if config.primary_trials != "all_retained":
        raise ValueError("Primary analysis must retain every epoch.")
    if config.persist_full_tfr_arrays:
        raise ValueError("Full TFR persistence is not approved for this milestone.")


def frequency_vector(config: EventRelatedSpectralConfig) -> np.ndarray:
    """Return inclusive frequency centers without floating endpoint drift."""
    grid = config.frequency_grid
    count = int(round((grid.stop - grid.start) / grid.step)) + 1
    frequencies = grid.start + np.arange(count, dtype=float) * grid.step
    if not np.isclose(frequencies[-1], grid.stop):
        raise ValueError("Frequency grid does not terminate at the configured stop.")
    return frequencies


def wavelet_cycles(
    frequencies_hz: np.ndarray, config: EventRelatedSpectralConfig
) -> np.ndarray:
    """Return the explicit frequency-dependent number of cycles."""
    return np.asarray(frequencies_hz, dtype=float) * config.wavelet.cycles_per_hz


def wavelet_lengths_samples(
    sampling_frequency_hz: float,
    frequencies_hz: np.ndarray,
    n_cycles: np.ndarray,
    *,
    zero_mean: bool,
) -> np.ndarray:
    """Ask MNE for actual discrete wavelets and return their sample lengths."""
    wavelets = morlet(
        sampling_frequency_hz,
        frequencies_hz,
        n_cycles=n_cycles,
        zero_mean=zero_mean,
    )
    return np.array([len(wavelet) for wavelet in wavelets], dtype=int)


def band_frequency_mask(
    frequencies_hz: np.ndarray, band: FrequencyBand
) -> np.ndarray:
    """Select inclusive frequency centers for one documented band."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    return (frequencies_hz >= band.lower_hz) & (frequencies_hz <= band.upper_hz)


def interval_mask(
    times_seconds: np.ndarray, interval_seconds: tuple[float, float]
) -> np.ndarray:
    """Select inclusive sampled time centers and fail on an empty interval."""
    times_seconds = np.asarray(times_seconds, dtype=float)
    mask = (times_seconds >= interval_seconds[0] - 1e-12) & (
        times_seconds <= interval_seconds[1] + 1e-12
    )
    if not mask.any():
        raise ValueError(f"No sampled time lies in interval {interval_seconds}.")
    return mask


def pair_task_with_preceding_rest(
    run_result: RunEpochs,
    *,
    task_index_offset: int = 0,
    rest_index_offset: int = 0,
) -> list[TrialPair]:
    """Pair every valid task with the immediately preceding valid T0 interval."""
    valid_task_array_index = {
        record.annotation_index: index
        for index, record in enumerate(record for record in run_result.task_records if record.valid)
    }
    rest_by_annotation = {
        record.annotation_index: (index, record)
        for index, record in enumerate(record for record in run_result.rest_records if record.valid)
    }
    pairs: list[TrialPair] = []
    all_rest_annotations = {
        record.annotation_index: record for record in run_result.rest_records
    }
    for task in run_result.task_records:
        if not task.valid:
            continue
        expected_rest_annotation = task.annotation_index - 1
        if expected_rest_annotation not in rest_by_annotation:
            if expected_rest_annotation in all_rest_annotations:
                # A valid task cannot support paired normalization when its
                # deterministic reference epoch crosses a recording boundary.
                continue
            raise RuntimeError(
                f"Run {task.run} task annotation {task.annotation_index} has no preceding T0."
            )
        local_rest_index, rest = rest_by_annotation[expected_rest_annotation]
        if rest.annotation != "T0" or task.previous_annotation != "T0":
            raise RuntimeError("Task/rest pairing does not follow measured T0→task order.")
        if rest.run_trial_index != task.run_trial_index:
            raise RuntimeError("Task/rest trial indices are not aligned within the run.")
        if rest.event_time_seconds >= task.event_time_seconds:
            raise RuntimeError("Paired T0 must begin before its following task onset.")
        pairs.append(
            TrialPair(
                task_array_index=(
                    task_index_offset + valid_task_array_index[task.annotation_index]
                ),
                rest_array_index=rest_index_offset + local_rest_index,
                task=task,
                rest=rest,
                source_path=run_result.preprocessing.source_path,
                source_sha256=run_result.preprocessing.source_sha256,
            )
        )
    return pairs


def assemble_spectral_epoch_dataset(
    run_results: Mapping[int, RunEpochs], run_order: Sequence[int]
) -> SpectralEpochDataset:
    """Combine run Epochs without losing row-aligned task/rest provenance."""
    if not run_order:
        raise ValueError("At least one run is required.")
    task_arrays: list[np.ndarray] = []
    rest_arrays: list[np.ndarray] = []
    pairs: list[TrialPair] = []
    task_offset = 0
    rest_offset = 0
    first = run_results[run_order[0]]
    channel_names = list(first.task_epochs.ch_names)
    task_times = first.task_epochs.times.copy()
    rest_times = first.rest_epochs.times.copy()
    sfreq = float(first.task_epochs.info["sfreq"])
    for run in run_order:
        result = run_results[run]
        if result.task_epochs.ch_names != channel_names:
            raise RuntimeError("Channel order differs across runs.")
        if not np.array_equal(result.task_epochs.times, task_times):
            raise RuntimeError("Task time vectors differ across runs.")
        if not np.array_equal(result.rest_epochs.times, rest_times):
            raise RuntimeError("Rest time vectors differ across runs.")
        all_task_data = result.task_epochs.get_data(copy=True)
        all_rest_data = result.rest_epochs.get_data(copy=True)
        local_pairs = pair_task_with_preceding_rest(result)
        task_arrays.append(
            all_task_data[[pair.task_array_index for pair in local_pairs]]
        )
        rest_arrays.append(
            all_rest_data[[pair.rest_array_index for pair in local_pairs]]
        )
        for local_index, pair in enumerate(local_pairs):
            pairs.append(
                TrialPair(
                    task_array_index=task_offset + local_index,
                    rest_array_index=rest_offset + local_index,
                    task=pair.task,
                    rest=pair.rest,
                    source_path=pair.source_path,
                    source_sha256=pair.source_sha256,
                )
            )
        task_offset += len(local_pairs)
        rest_offset += len(local_pairs)
    combined_task = np.concatenate(task_arrays, axis=0)
    combined_rest = np.concatenate(rest_arrays, axis=0)
    if len(pairs) != combined_task.shape[0] or len(pairs) != combined_rest.shape[0]:
        raise RuntimeError("Combined task/rest arrays and provenance are misaligned.")
    if not np.isfinite(combined_task).all() or not np.isfinite(combined_rest).all():
        raise RuntimeError("Combined spectral input contains NaN or infinity.")
    return SpectralEpochDataset(
        task_data_volts=combined_task,
        rest_data_volts=combined_rest,
        task_times=task_times,
        rest_times=rest_times,
        channel_names=channel_names,
        sampling_frequency_hz=sfreq,
        pairs=pairs,
    )


def compute_morlet_power(
    data_volts: np.ndarray,
    times_seconds: np.ndarray,
    sampling_frequency_hz: float,
    config: EventRelatedSpectralConfig,
) -> TimeFrequencyPower:
    """Compute deterministic single-trial power without condition averaging."""
    data_volts = np.asarray(data_volts, dtype=float)
    times_seconds = np.asarray(times_seconds, dtype=float)
    if data_volts.ndim != 3 or data_volts.shape[2] != times_seconds.size:
        raise ValueError("Expected data shaped (trials, channels, epoch times).")
    if not np.isfinite(data_volts).all():
        raise RuntimeError("Morlet input contains NaN or infinity.")
    if times_seconds.size > 1:
        measured_sfreq = 1.0 / float(np.median(np.diff(times_seconds)))
        if not np.isclose(measured_sfreq, sampling_frequency_hz):
            raise ValueError("Time vector and sampling frequency are inconsistent.")
    frequencies = frequency_vector(config)
    cycles = wavelet_cycles(frequencies, config)
    lengths = wavelet_lengths_samples(
        sampling_frequency_hz,
        frequencies,
        cycles,
        zero_mean=config.wavelet.zero_mean,
    )
    if lengths.max() > data_volts.shape[2]:
        raise ValueError("A configured wavelet is longer than the input epoch.")
    power = tfr_array_morlet(
        data_volts,
        sampling_frequency_hz,
        frequencies,
        n_cycles=cycles,
        zero_mean=config.wavelet.zero_mean,
        use_fft=config.wavelet.use_fft,
        decim=config.wavelet.decimation,
        output="power",
        n_jobs=1,
        verbose="error",
    )
    output_times = times_seconds[:: config.wavelet.decimation]
    expected_shape = (
        data_volts.shape[0],
        data_volts.shape[1],
        frequencies.size,
        output_times.size,
    )
    if power.shape != expected_shape or not np.isfinite(power).all():
        raise RuntimeError(f"Unexpected or non-finite Morlet output: {power.shape}.")
    if np.any(power < 0):
        raise RuntimeError("Morlet power contains a negative value.")
    return TimeFrequencyPower(power, frequencies, output_times, cycles, lengths)


def percent_power_change(
    task_power: np.ndarray, reference_power: np.ndarray
) -> np.ndarray:
    """Return 100 × (task − reference) / reference with strict denominator checks."""
    task_power = np.asarray(task_power, dtype=float)
    reference_power = np.asarray(reference_power, dtype=float)
    if np.any(reference_power <= 0) or not np.isfinite(reference_power).all():
        raise ValueError("Reference power must be finite and strictly positive.")
    change = 100.0 * (task_power - reference_power) / reference_power
    if not np.isfinite(change).all():
        raise RuntimeError("Percent-power normalization produced NaN or infinity.")
    return change


def paired_frequency_reference(
    rest_power: TimeFrequencyPower,
    reference_interval_seconds: tuple[float, float],
) -> np.ndarray:
    """Average each paired T0 over reference time, preserving frequency."""
    mask = interval_mask(rest_power.times_seconds, reference_interval_seconds)
    reference = np.mean(rest_power.power_v2[..., mask], axis=-1)
    if np.any(reference <= 0) or not np.isfinite(reference).all():
        raise RuntimeError("Paired frequency reference is invalid.")
    return reference


def normalize_task_tfr_percent(
    task_power: TimeFrequencyPower,
    rest_power: TimeFrequencyPower,
    reference_interval_seconds: tuple[float, float],
) -> np.ndarray:
    """Normalize every task time/frequency point to its paired T0 frequency power."""
    if task_power.power_v2.shape[:3] != rest_power.power_v2.shape[:3]:
        raise ValueError("Task/rest TFR trial, channel, or frequency dimensions differ.")
    if not np.array_equal(task_power.frequencies_hz, rest_power.frequencies_hz):
        raise ValueError("Task/rest TFR frequency vectors differ.")
    reference = paired_frequency_reference(rest_power, reference_interval_seconds)
    return percent_power_change(task_power.power_v2, reference[..., np.newaxis])


def trial_band_measurements(
    task_power: TimeFrequencyPower,
    rest_power: TimeFrequencyPower,
    dataset: SpectralEpochDataset,
    config: EventRelatedSpectralConfig,
    quality_by_identity: Mapping[tuple[int, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    """Create one traceable task/rest band-power record per trial and channel."""
    if task_power.power_v2.shape[0] != len(dataset.pairs):
        raise ValueError("Task TFR and trial-pair counts differ.")
    if rest_power.power_v2.shape[0] != len(dataset.pairs):
        raise ValueError("Rest TFR and trial-pair counts differ.")
    task_time_mask = interval_mask(
        task_power.times_seconds, config.task_analysis_interval_seconds
    )
    rest_time_mask = interval_mask(
        rest_power.times_seconds, config.paired_rest_reference_interval_seconds
    )
    rows: list[dict[str, object]] = []
    for band in config.bands:
        frequency_mask = band_frequency_mask(task_power.frequencies_hz, band)
        task_values = np.mean(
            task_power.power_v2[:, :, frequency_mask][:, :, :, task_time_mask],
            axis=(2, 3),
        )
        rest_values = np.mean(
            rest_power.power_v2[:, :, frequency_mask][:, :, :, rest_time_mask],
            axis=(2, 3),
        )
        changes = percent_power_change(task_values, rest_values)
        for trial_index, pair in enumerate(dataset.pairs):
            identity = (pair.task.run, pair.task.run_trial_index)
            quality = quality_by_identity[identity]
            for channel_index, channel in enumerate(dataset.channel_names):
                rows.append(
                    {
                        "subject": pair.task.subject,
                        "run": pair.task.run,
                        "run_trial_index": pair.task.run_trial_index,
                        "task_epoch_array_index": pair.task_array_index,
                        "rest_epoch_array_index": pair.rest_array_index,
                        "task_annotation_index": pair.task.annotation_index,
                        "rest_annotation_index": pair.rest.annotation_index,
                        "annotation": pair.task.annotation,
                        "semantic_condition": pair.task.semantic_condition,
                        "task_event_sample": pair.task.event_sample,
                        "task_event_time_seconds": pair.task.event_time_seconds,
                        "rest_event_sample": pair.rest.event_sample,
                        "rest_event_time_seconds": pair.rest.event_time_seconds,
                        "source_file": canonical_source_id(str(pair.source_path)),
                        "source_sha256": pair.source_sha256,
                        "quality_status": quality["quality_status"],
                        "quality_evidence": quality["quality_evidence"],
                        "confirmed_exclusion": quality["confirmed_exclusion"],
                        "channel": channel,
                        "frequency_band": band.name,
                        "band_lower_hz": band.lower_hz,
                        "band_upper_hz": band.upper_hz,
                        "frequency_center_count": int(frequency_mask.sum()),
                        "task_interval_start_seconds": config.task_analysis_interval_seconds[0],
                        "task_interval_stop_seconds": config.task_analysis_interval_seconds[1],
                        "rest_interval_start_seconds": config.paired_rest_reference_interval_seconds[0],
                        "rest_interval_stop_seconds": config.paired_rest_reference_interval_seconds[1],
                        "reference_mean_wavelet_power_v2": float(
                            rest_values[trial_index, channel_index]
                        ),
                        "task_mean_wavelet_power_v2": float(
                            task_values[trial_index, channel_index]
                        ),
                        "change_percent": float(changes[trial_index, channel_index]),
                        "primary_retained": True,
                    }
                )
    return rows


def band_percent_time_courses(
    task_power: TimeFrequencyPower,
    rest_power: TimeFrequencyPower,
    band: FrequencyBand,
    reference_interval_seconds: tuple[float, float],
) -> np.ndarray:
    """Return trial × channel × task-time band change against paired rest."""
    frequency_mask = band_frequency_mask(task_power.frequencies_hz, band)
    rest_time_mask = interval_mask(rest_power.times_seconds, reference_interval_seconds)
    reference = np.mean(
        rest_power.power_v2[:, :, frequency_mask][:, :, :, rest_time_mask],
        axis=(2, 3),
    )
    task_band_by_time = np.mean(
        task_power.power_v2[:, :, frequency_mask], axis=2
    )
    return percent_power_change(task_band_by_time, reference[..., np.newaxis])


def descriptive_statistics(values: np.ndarray) -> dict[str, float | int]:
    """Return deterministic trial-distribution summaries."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Summary values must be a non-empty finite vector.")
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        "trial_count": int(values.size),
        "mean_change_percent": float(np.mean(values)),
        "standard_deviation_change_percent": float(
            np.std(values, ddof=1) if values.size > 1 else 0.0
        ),
        "median_change_percent": float(median),
        "q25_change_percent": float(q25),
        "q75_change_percent": float(q75),
        "iqr_change_percentage_points": float(q75 - q25),
        "minimum_change_percent": float(np.min(values)),
        "maximum_change_percent": float(np.max(values)),
    }
