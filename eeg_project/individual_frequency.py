"""Leakage-safe resting central-frequency estimation for the IMF study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks, peak_widths, welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDIVIDUAL_FREQUENCY_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "individual_mu_frequency.json"
)


@dataclass(frozen=True)
class PeakEstimate:
    """One deterministic central alpha/mu-range peak decision."""

    peak_frequency_hz: float | None
    candidate_frequency_hz: float
    peak_prominence_db: float
    peak_width_hz: float | None
    peak_quality: str
    method_available: bool
    failure_reason: str


@dataclass(frozen=True)
class IndividualFrequencyStudyConfig:
    """Fields required to protect the staged individualized-frequency study."""

    schema_version: int
    study_stage: str
    study_start_git_commit: str
    individualized_task_outcomes_inspected_at_initial_freeze: bool
    development_subjects: tuple[int, ...]
    evaluation_subjects: tuple[int, ...]
    central_channels: tuple[str, ...]
    candidate_search_ranges_hz: tuple[tuple[float, float], ...]
    welch_window_seconds: float
    welch_frequency_resolution_hz: float
    candidate_prominence_thresholds_db: tuple[float, ...]
    candidate_half_widths_hz: tuple[float, ...]
    evaluation_frequency_step_hz: float
    runs: tuple[int, ...]
    training_runs_by_held_out: Mapping[int, tuple[int, ...]]
    frozen_historical_inputs: Mapping[str, Mapping[str, str]]
    evaluation_locked: bool
    task_outcomes_may_define_peak: bool
    boundary_maximum_is_valid: bool
    all_three_estimates_required: bool
    classification_performed: bool
    csp_performed: bool
    persist_full_tfr_arrays: bool


def _float_pair(values: Sequence[object], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    return float(values[0]), float(values[1])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_individual_frequency_config(
    path: Path = DEFAULT_INDIVIDUAL_FREQUENCY_CONFIG_PATH,
) -> IndividualFrequencyStudyConfig:
    """Load and strictly validate the staged IMF study policy."""
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    partition = payload["partition"]
    peak = payload["candidate_peak_method"]
    band = payload["candidate_individualized_band"]
    loro = payload["leave_one_run_out"]
    runs = tuple(int(run) for run in loro["runs"])
    training = {
        held_out: tuple(int(run) for run in loro[f"held_out_{held_out}_training_runs"])
        for held_out in runs
    }
    config = IndividualFrequencyStudyConfig(
        schema_version=int(payload["schema_version"]),
        study_stage=str(payload["study_stage"]),
        study_start_git_commit=str(payload["study_start_git_commit"]),
        individualized_task_outcomes_inspected_at_initial_freeze=bool(
            payload["individualized_task_outcomes_inspected_at_initial_freeze"]
        ),
        development_subjects=tuple(int(value) for value in partition["development_subjects"]),
        evaluation_subjects=tuple(int(value) for value in partition["evaluation_subjects"]),
        central_channels=tuple(peak["central_roi_candidates"][0].removeprefix("mean_log_psd_").split("_")),
        candidate_search_ranges_hz=tuple(
            _float_pair(values, "candidate search range")
            for values in peak["candidate_search_ranges_hz"]
        ),
        welch_window_seconds=float(peak["welch_window_seconds"]),
        welch_frequency_resolution_hz=float(peak["welch_frequency_resolution_hz"]),
        candidate_prominence_thresholds_db=tuple(
            float(value) for value in peak["candidate_prominence_thresholds_db"]
        ),
        candidate_half_widths_hz=tuple(
            float(value) for value in band["candidate_half_widths_hz"]
        ),
        evaluation_frequency_step_hz=float(band["evaluation_frequency_step_hz"]),
        runs=runs,
        training_runs_by_held_out=training,
        frozen_historical_inputs=payload["frozen_historical_inputs"],
        evaluation_locked=bool(
            partition["evaluation_individualized_task_outcomes_locked_until_final_method_freeze"]
        ),
        task_outcomes_may_define_peak=bool(peak["task_labels_or_task_outcomes_may_define_peak"]),
        boundary_maximum_is_valid=bool(peak["boundary_maximum_is_valid"]),
        all_three_estimates_required=bool(
            payload["candidate_no_peak_policy"]
            ["all_three_leave_one_run_out_estimates_required_for_primary_matched_comparison"]
        ),
        classification_performed=bool(payload["classification_performed"]),
        csp_performed=bool(payload["csp_performed"]),
        persist_full_tfr_arrays=bool(payload["persist_full_tfr_arrays"]),
    )
    validate_individual_frequency_config(config)
    return config


def validate_individual_frequency_config(config: IndividualFrequencyStudyConfig) -> None:
    """Reject partition drift, leakage, historical drift, and ambiguous units."""
    if config.schema_version != 1:
        raise ValueError(f"Unsupported IMF study schema: {config.schema_version}.")
    if config.study_start_git_commit != "d322c58fc6474b7970eb113fe0448c4c5177b157":
        raise ValueError("The IMF study must start from completed fixed-band commit d322c58.")
    if config.individualized_task_outcomes_inspected_at_initial_freeze:
        raise ValueError("The initial IMF freeze must precede individualized task outcomes.")
    eligible = set(range(2, 110)) - {88, 92, 100}
    development = set(config.development_subjects)
    evaluation = set(config.evaluation_subjects)
    if development != {subject for subject in eligible if subject % 2 == 0}:
        raise ValueError("IMF development subjects must be eligible even IDs.")
    if evaluation != {subject for subject in eligible if subject % 2 == 1}:
        raise ValueError("IMF evaluation subjects must be eligible odd IDs.")
    if development & evaluation or development | evaluation != eligible:
        raise ValueError("IMF partitions must be disjoint and exhaustive over eligible subjects.")
    if config.central_channels != ("C3", "Cz", "C4"):
        raise ValueError("Peak estimation ROI must remain mean log PSD over C3/Cz/C4.")
    if config.runs != (6, 10, 14):
        raise ValueError("Leave-one-run-out runs must remain 6, 10, and 14.")
    for held_out, training in config.training_runs_by_held_out.items():
        if set(training) != set(config.runs) - {held_out} or held_out in training:
            raise ValueError("A held-out run entered its own peak-estimation inputs.")
    if config.task_outcomes_may_define_peak:
        raise ValueError("Task labels or outcomes may never define individual frequency.")
    if config.boundary_maximum_is_valid:
        raise ValueError("A search-boundary maximum may not be called a valid IMF.")
    expected_resolution = 1.0 / config.welch_window_seconds
    if not np.isclose(config.welch_frequency_resolution_hz, expected_resolution):
        raise ValueError("Welch resolution must equal one divided by window duration.")
    if any(not 0 < low < high for low, high in config.candidate_search_ranges_hz):
        raise ValueError("Candidate search ranges must be positive and increasing.")
    if any(value <= 0 for value in config.candidate_prominence_thresholds_db):
        raise ValueError("Prominence thresholds must be positive dB values.")
    if config.candidate_half_widths_hz != (1.0,):
        raise ValueError("Only the pre-specified ±1 Hz candidate band is permitted.")
    if config.evaluation_frequency_step_hz != 0.5:
        raise ValueError("Individualized evaluation frequency step must remain 0.5 Hz.")
    if not config.all_three_estimates_required:
        raise ValueError("Primary matched comparison requires all three LORO estimates.")
    if config.classification_performed or config.csp_performed or config.persist_full_tfr_arrays:
        raise ValueError("CSP, classification, and full TFR persistence remain forbidden.")
    for name, record in config.frozen_historical_inputs.items():
        frozen_path = (PROJECT_ROOT / record["path"]).resolve()
        if not frozen_path.is_file() or _sha256(frozen_path) != record["sha256"]:
            raise RuntimeError(f"Frozen historical input drift: {name} ({frozen_path}).")


def rest_roi_log_welch(
    rest_data_volts: np.ndarray,
    channel_names: Sequence[str],
    sampling_frequency_hz: float,
    central_channels: Sequence[str],
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average single-epoch log PSD over a predefined central ROI.

    Input shape is ``(rest epochs, channels, samples)``. Exactly one complete Hann
    window is used from each rest epoch, so epoch boundaries are never concatenated.
    The output is ``(frequency centers,)`` in Hz and dB re 1 V²/Hz.
    """
    data = np.asarray(rest_data_volts, dtype=float)
    if data.ndim != 3 or data.shape[0] < 1:
        raise ValueError("Rest data must have shape (epochs, channels, samples).")
    if data.shape[1] != len(channel_names) or not np.isfinite(data).all():
        raise ValueError("Rest data/channel metadata are inconsistent or non-finite.")
    if sampling_frequency_hz <= 0 or window_seconds <= 0:
        raise ValueError("Sampling frequency and Welch window must be positive.")
    missing = sorted(set(central_channels) - set(channel_names))
    if missing:
        raise ValueError(f"Missing central ROI channels: {missing}.")
    n_per_segment = int(round(window_seconds * sampling_frequency_hz))
    if not np.isclose(n_per_segment / sampling_frequency_hz, window_seconds):
        raise ValueError("Welch window does not map to an integer sample count.")
    if data.shape[2] < n_per_segment:
        raise ValueError("Rest epoch is shorter than the configured Welch window.")
    indices = [list(channel_names).index(channel) for channel in central_channels]
    selected = data[:, indices, :n_per_segment]
    frequencies, power = welch(
        selected,
        fs=sampling_frequency_hz,
        window="hann",
        nperseg=n_per_segment,
        noverlap=0,
        nfft=n_per_segment,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
    )
    tiny = np.finfo(float).tiny
    log_power_db = 10.0 * np.log10(np.maximum(power, tiny))
    roi_log_power_db = np.mean(log_power_db, axis=(0, 1))
    expected_step = sampling_frequency_hz / n_per_segment
    if not np.allclose(np.diff(frequencies), expected_step):
        raise RuntimeError("Welch frequency grid differs from fs/nperseg.")
    if not np.isfinite(roi_log_power_db).all():
        raise RuntimeError("Central rest PSD contains NaN or infinity.")
    return frequencies, roi_log_power_db


def estimate_central_peak(
    frequencies_hz: np.ndarray,
    log_power_db: np.ndarray,
    search_range_hz: tuple[float, float],
    minimum_prominence_db: float,
) -> PeakEstimate:
    """Classify an interior peak, ambiguous boundary maximum, or absent peak."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    power = np.asarray(log_power_db, dtype=float)
    if frequencies.ndim != 1 or power.shape != frequencies.shape or frequencies.size < 3:
        raise ValueError("Peak estimation requires aligned one-dimensional vectors.")
    if not np.isfinite(frequencies).all() or not np.isfinite(power).all():
        raise ValueError("Peak-estimation inputs must be finite.")
    if not np.all(np.diff(frequencies) > 0):
        raise ValueError("Frequency centers must increase strictly.")
    low, high = search_range_hz
    mask = (frequencies >= low - 1e-12) & (frequencies <= high + 1e-12)
    search_frequencies = frequencies[mask]
    search_power = power[mask]
    if search_frequencies.size < 3 or not np.isclose(search_frequencies[0], low) or not np.isclose(search_frequencies[-1], high):
        raise ValueError("Search bounds must land on the supplied frequency grid.")
    if minimum_prominence_db <= 0:
        raise ValueError("Minimum prominence must be positive.")

    peak_indices, properties = find_peaks(
        search_power, prominence=minimum_prominence_db
    )
    if peak_indices.size:
        prominences = np.asarray(properties["prominences"], dtype=float)
        best_order = np.lexsort((search_frequencies[peak_indices], -prominences))
        selected_position = int(best_order[0])
        peak_index = int(peak_indices[selected_position])
        width_bins = float(
            peak_widths(search_power, [peak_index], rel_height=0.5)[0][0]
        )
        frequency_step = float(np.median(np.diff(search_frequencies)))
        return PeakEstimate(
            peak_frequency_hz=float(search_frequencies[peak_index]),
            candidate_frequency_hz=float(search_frequencies[peak_index]),
            peak_prominence_db=float(prominences[selected_position]),
            peak_width_hz=width_bins * frequency_step,
            peak_quality="well_defined_interior",
            method_available=True,
            failure_reason="",
        )

    maximum_index = int(np.argmax(search_power))
    candidate_frequency = float(search_frequencies[maximum_index])
    candidate_height_over_median = float(
        search_power[maximum_index] - np.median(search_power)
    )
    at_boundary = maximum_index in (0, search_power.size - 1)
    if at_boundary and candidate_height_over_median >= minimum_prominence_db:
        return PeakEstimate(
            peak_frequency_hz=None,
            candidate_frequency_hz=candidate_frequency,
            peak_prominence_db=candidate_height_over_median,
            peak_width_hz=None,
            peak_quality="boundary_candidate",
            method_available=False,
            failure_reason="search_boundary_maximum",
        )
    return PeakEstimate(
        peak_frequency_hz=None,
        candidate_frequency_hz=candidate_frequency,
        peak_prominence_db=max(0.0, candidate_height_over_median),
        peak_width_hz=None,
        peak_quality="poorly_defined_no_peak",
        method_available=False,
        failure_reason="no_interior_peak_meets_prominence",
    )


def training_runs_for_held_out(
    held_out_run: int, runs: Sequence[int]
) -> tuple[int, ...]:
    """Return the two estimation runs and prove the evaluation run is absent."""
    ordered = tuple(int(run) for run in runs)
    if ordered != (6, 10, 14) or held_out_run not in ordered:
        raise ValueError("Expected held-out run 6, 10, or 14.")
    training = tuple(run for run in ordered if run != held_out_run)
    if held_out_run in training or len(training) != 2:
        raise RuntimeError("Leave-one-run-out leakage guard failed.")
    return training


def individualized_frequency_vector(
    peak_frequency_hz: float,
    half_width_hz: float,
    step_hz: float,
) -> np.ndarray:
    """Construct an inclusive fixed-width individualized evaluation grid."""
    if peak_frequency_hz <= 0 or half_width_hz <= 0 or step_hz <= 0:
        raise ValueError("Peak, half-width, and frequency step must be positive.")
    count = int(round((2.0 * half_width_hz) / step_hz)) + 1
    frequencies = peak_frequency_hz - half_width_hz + np.arange(count) * step_hz
    if not np.isclose(frequencies[-1], peak_frequency_hz + half_width_hz):
        raise ValueError("Individualized frequency step does not land on band edge.")
    if frequencies[0] < 6.0 or frequencies[-1] > 35.0:
        raise ValueError("Individualized grid lies outside the frozen Morlet domain.")
    return frequencies.astype(float)

