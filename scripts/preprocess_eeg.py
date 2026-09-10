"""Validate and report the production continuous-EEG preprocessing pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.filter import create_filter, filter_data, notch_filter
from mne.time_frequency import psd_array_welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.preprocessing import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PreprocessedRecording,
    PreprocessingConfig,
    load_preprocessing_config,
    preprocess_recording,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_RUNS = (6, 10, 14)
WELCH_SEGMENT_SAMPLES = 480
WELCH_OVERLAP_SAMPLES = 240
WELCH_WINDOW = "hann"
REPORT_CHANNELS = ("Fp1", "C3", "Cz", "C4")
NORMAL_WINDOW_SECONDS = (8.0, 12.0)
TRANSIENT_WINDOW_SECONDS = (13.0, 17.0)
COLORS = {
    "stored_valid": "#777777",
    "average_reference": "#4c78a8",
    "filtered_1_40_hz": "#54a24b",
}


def parse_arguments() -> argparse.Namespace:
    """Parse reproducible report options."""
    parser = argparse.ArgumentParser(
        description="Apply and validate the approved continuous EEG preprocessing."
    )
    parser.add_argument("--subject", type=int, default=1, help="EEGBCI subject (1-109).")
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        choices=DEFAULT_RUNS,
        default=list(DEFAULT_RUNS),
        help="One or more current motor-imagery runs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Version-controlled preprocessing JSON configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for report CSV, JSON, and PNG artifacts.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject ambiguous or duplicated run configuration."""
    if not 1 <= arguments.subject <= 109:
        raise ValueError("Subject must be between 1 and 109.")
    if len(set(arguments.runs)) != len(arguments.runs):
        raise ValueError("Run numbers must not be duplicated.")


def compute_psd(data_volts: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the same explicit Welch PSD used in spectral characterization."""
    psd, frequencies = psd_array_welch(
        data_volts,
        sfreq,
        fmin=0,
        fmax=sfreq / 2,
        n_fft=WELCH_SEGMENT_SAMPLES,
        n_per_seg=WELCH_SEGMENT_SAMPLES,
        n_overlap=WELCH_OVERLAP_SAMPLES,
        average="mean",
        window=WELCH_WINDOW,
        remove_dc=True,
        output="power",
        verbose="error",
    )
    return psd * 1_000_000**2, frequencies


def frequency_mask(
    frequencies: np.ndarray, lower_hz: float, upper_hz: float
) -> np.ndarray:
    """Select an inclusive frequency interval."""
    tolerance = np.finfo(float).eps * max(1.0, upper_hz) * 16
    return (frequencies >= lower_hz - tolerance) & (
        frequencies <= upper_hz + tolerance
    )


def integrate_psd(
    psd: np.ndarray,
    frequencies: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> np.ndarray:
    """Integrate PSD over one closed band in µV²."""
    mask = frequency_mask(frequencies, lower_hz, upper_hz)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError(f"Insufficient bins in {lower_hz:g}-{upper_hz:g} Hz.")
    return np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)


def stage_arrays(
    result: PreprocessedRecording,
) -> dict[str, np.ndarray]:
    """Return consistently named processing-stage arrays in volts."""
    return {
        "stored_valid": result.stored_valid.get_data(),
        "average_reference": result.referenced.get_data(),
        "filtered_1_40_hz": result.filtered.get_data(),
    }


def processing_metric_rows(
    subject: int,
    run: int,
    result: PreprocessedRecording,
) -> list[dict[str, object]]:
    """Quantify interpretable time- and frequency-domain values per channel/stage."""
    sfreq = float(result.filtered.info["sfreq"])
    rows: list[dict[str, object]] = []
    for stage, data_volts in stage_arrays(result).items():
        data_uv = data_volts * 1_000_000
        psd, frequencies = compute_psd(data_volts, sfreq)
        low = integrate_psd(psd, frequencies, frequencies[1], 1.0)
        alpha_mu = integrate_psd(psd, frequencies, 8.0, 13.0)
        motor = integrate_psd(psd, frequencies, 8.0, 30.0)
        sixty_index = int(np.argmin(np.abs(frequencies - 60.0)))
        if not np.isclose(frequencies[sixty_index], 60.0):
            raise RuntimeError("Welch grid has no exact 60 Hz bin.")
        for index, channel in enumerate(result.filtered.ch_names):
            values = data_uv[index]
            rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "channel": channel,
                    "stage": stage,
                    "samples": values.size,
                    "duration_seconds": values.size / sfreq,
                    "sampling_frequency_hz": sfreq,
                    "mean_uV": float(np.mean(values)),
                    "standard_deviation_uV": float(np.std(values)),
                    "rms_uV": float(np.sqrt(np.mean(values**2))),
                    "peak_to_peak_uV": float(np.ptp(values)),
                    "power_0.333_1_hz_uV2": float(low[index]),
                    "power_8_13_hz_uV2": float(alpha_mu[index]),
                    "power_8_30_hz_uV2": float(motor[index]),
                    "psd_60_hz_uV2_per_hz": float(psd[index, sixty_index]),
                    "reference": (
                        "stored" if stage == "stored_valid" else "64_channel_average"
                    ),
                    "temporal_filter": (
                        "none" if stage != "filtered_1_40_hz" else "production_1_40_hz_fir"
                    ),
                }
            )
    return rows


def time_window_metric_rows(
    subject: int,
    run: int,
    result: PreprocessedRecording,
) -> list[dict[str, object]]:
    """Quantify the two controlled run-6 comparison windows."""
    sfreq = float(result.filtered.info["sfreq"])
    windows = {
        "relatively_lower_amplitude_8_12_s": NORMAL_WINDOW_SECONDS,
        "large_frontal_transient_13_17_s": TRANSIENT_WINDOW_SECONDS,
    }
    rows: list[dict[str, object]] = []
    for window_name, (start, stop) in windows.items():
        start_sample = int(round(start * sfreq))
        stop_sample = int(round(stop * sfreq))
        for stage, data_volts in stage_arrays(result).items():
            data_uv = data_volts[:, start_sample:stop_sample] * 1_000_000
            for index, channel in enumerate(result.filtered.ch_names):
                values = data_uv[index]
                rows.append(
                    {
                        "subject": subject,
                        "run": run,
                        "window": window_name,
                        "window_start_seconds": start,
                        "window_stop_seconds": stop,
                        "channel": channel,
                        "stage": stage,
                        "mean_uV": float(np.mean(values)),
                        "standard_deviation_uV": float(np.std(values)),
                        "rms_uV": float(np.sqrt(np.mean(values**2))),
                        "peak_to_peak_uV": float(np.ptp(values)),
                    }
                )
    return rows


def create_candidate_filter(
    name: str, sfreq: float
) -> tuple[np.ndarray, dict[str, object]]:
    """Create one serious filter candidate and its explicit design metadata."""
    candidates: dict[str, dict[str, object]] = {
        "highpass_0.5": {
            "l_freq": 0.5,
            "h_freq": None,
            "l_trans_bandwidth": 0.5,
            "h_trans_bandwidth": "auto",
            "filter_length": 1057,
        },
        "highpass_1": {
            "l_freq": 1.0,
            "h_freq": None,
            "l_trans_bandwidth": 1.0,
            "h_trans_bandwidth": "auto",
            "filter_length": 529,
        },
        "lowpass_40": {
            "l_freq": None,
            "h_freq": 40.0,
            "l_trans_bandwidth": "auto",
            "h_trans_bandwidth": 10.0,
            "filter_length": 53,
        },
        "lowpass_45": {
            "l_freq": None,
            "h_freq": 45.0,
            "l_trans_bandwidth": "auto",
            "h_trans_bandwidth": 11.25,
            "filter_length": 47,
        },
        "bandpass_1_40": {
            "l_freq": 1.0,
            "h_freq": 40.0,
            "l_trans_bandwidth": 1.0,
            "h_trans_bandwidth": 10.0,
            "filter_length": 529,
        },
        "bandpass_1_45": {
            "l_freq": 1.0,
            "h_freq": 45.0,
            "l_trans_bandwidth": 1.0,
            "h_trans_bandwidth": 11.25,
            "filter_length": 529,
        },
    }
    if name == "notch_60":
        notch_width = 60.0 / 200.0
        half_transition = 0.5
        # This reproduces the explicit band-stop edges used internally by
        # MNE notch_filter for notch_widths=None and trans_bandwidth=1 Hz.
        metadata = {
            "l_freq": 60.0 + notch_width / 2 + half_transition,
            "h_freq": 60.0 - notch_width / 2 - half_transition,
            "l_trans_bandwidth": half_transition,
            "h_trans_bandwidth": half_transition,
            "filter_length": 1057,
            "notch_center_hz": 60.0,
            "notch_width_hz": notch_width,
            "total_transition_bandwidth_hz": 1.0,
        }
    else:
        metadata = candidates[name].copy()

    coefficients = create_filter(
        None,
        sfreq,
        l_freq=metadata["l_freq"],
        h_freq=metadata["h_freq"],
        filter_length=metadata["filter_length"],
        l_trans_bandwidth=metadata["l_trans_bandwidth"],
        h_trans_bandwidth=metadata["h_trans_bandwidth"],
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        verbose=False,
    )
    metadata.update(
        {
            "candidate": name,
            "filter_length_samples": int(coefficients.size),
            "filter_length_seconds": float(coefficients.size / sfreq),
            "method": "fir",
            "phase": "zero",
            "fir_window": "hamming",
            "fir_design": "firwin",
        }
    )
    return coefficients, metadata


def filter_response_rows(sfreq: float) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    """Calculate dense responses and selected-frequency gains for candidates."""
    candidate_names = (
        "highpass_0.5",
        "highpass_1",
        "lowpass_40",
        "lowpass_45",
        "bandpass_1_40",
        "bandpass_1_45",
        "notch_60",
    )
    report_frequencies = (0.1, 0.25, 0.5, 1.0, 8.0, 12.0, 30.0, 40.0, 45.0, 50.0, 60.0, 65.0)
    dense_frequencies = np.fft.rfftfreq(262_144, d=1 / sfreq)
    rows: list[dict[str, object]] = []
    dense_responses: dict[str, np.ndarray] = {"frequencies": dense_frequencies}

    for name in candidate_names:
        coefficients, metadata = create_candidate_filter(name, sfreq)
        magnitude = np.abs(np.fft.rfft(coefficients, 262_144))
        dense_responses[name] = magnitude
        for frequency in report_frequencies:
            index = int(np.argmin(np.abs(dense_frequencies - frequency)))
            gain = float(magnitude[index])
            rows.append(
                {
                    **metadata,
                    "frequency_hz": frequency,
                    "amplitude_gain": gain,
                    "gain_db": float(20 * np.log10(max(gain, np.finfo(float).tiny))),
                }
            )
    return rows, dense_responses


def apply_candidate_filter(
    data: np.ndarray,
    sfreq: float,
    *,
    l_freq: float | None,
    h_freq: float,
    l_transition: float | str,
    h_transition: float,
    length: int,
) -> np.ndarray:
    """Apply one explicitly configured candidate to a copied array."""
    return filter_data(
        data,
        sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        filter_length=length,
        l_trans_bandwidth=l_transition,
        h_trans_bandwidth=h_transition,
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        pad="reflect_limited",
        copy=True,
        verbose="error",
    )


def candidate_eeg_rows(
    subject: int,
    run: int,
    result: PreprocessedRecording,
) -> list[dict[str, object]]:
    """Compare serious candidates using identical average-referenced samples."""
    sfreq = float(result.referenced.info["sfreq"])
    baseline = result.referenced.get_data()
    baseline_psd, frequencies = compute_psd(baseline, sfreq)
    low_baseline = integrate_psd(baseline_psd, frequencies, frequencies[1], 1.0)
    motor_baseline = integrate_psd(baseline_psd, frequencies, 8.0, 30.0)
    sixty_index = int(np.argmin(np.abs(frequencies - 60.0)))

    candidates = {
        "no_highpass_lowpass_40": apply_candidate_filter(
            baseline,
            sfreq,
            l_freq=None,
            h_freq=40.0,
            l_transition="auto",
            h_transition=10.0,
            length=53,
        ),
        "highpass_0.5_lowpass_40": apply_candidate_filter(
            baseline,
            sfreq,
            l_freq=0.5,
            h_freq=40.0,
            l_transition=0.5,
            h_transition=10.0,
            length=1057,
        ),
        "highpass_1_lowpass_40_production": result.filtered.get_data(),
        "highpass_1_lowpass_45": apply_candidate_filter(
            baseline,
            sfreq,
            l_freq=1.0,
            h_freq=45.0,
            l_transition=1.0,
            h_transition=11.25,
            length=529,
        ),
    }
    production = candidates["highpass_1_lowpass_40_production"]
    candidates["production_plus_60_hz_notch"] = notch_filter(
        production,
        sfreq,
        freqs=[60.0],
        filter_length=1057,
        notch_widths=None,
        trans_bandwidth=1.0,
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        pad="reflect_limited",
        copy=True,
        verbose="error",
    )

    production_psd, _ = compute_psd(production, sfreq)
    rows: list[dict[str, object]] = []
    for name, data in candidates.items():
        psd, _ = compute_psd(data, sfreq)
        low = integrate_psd(psd, frequencies, frequencies[1], 1.0)
        motor = integrate_psd(psd, frequencies, 8.0, 30.0)
        relative_rms = np.sqrt(np.mean((data - baseline) ** 2, axis=1)) / np.sqrt(
            np.mean(baseline**2, axis=1)
        )
        row: dict[str, object] = {
            "subject": subject,
            "run": run,
            "candidate": name,
            "median_very_low_power_ratio_to_referenced": float(np.median(low / low_baseline)),
            "median_motor_8_30_power_ratio_to_referenced": float(np.median(motor / motor_baseline)),
            "median_60_hz_attenuation_db_vs_referenced": float(
                np.median(10 * np.log10(psd[:, sixty_index] / baseline_psd[:, sixty_index]))
            ),
            "median_relative_rms_change_vs_referenced": float(np.median(relative_rms)),
        }
        if name == "production_plus_60_hz_notch":
            notch_rms = np.sqrt(np.mean((data - production) ** 2, axis=1)) / np.sqrt(
                np.mean(production**2, axis=1)
            )
            row["additional_60_hz_attenuation_db_vs_production"] = float(
                np.median(10 * np.log10(psd[:, sixty_index] / production_psd[:, sixty_index]))
            )
            row["median_relative_rms_change_vs_production"] = float(np.median(notch_rms))
        else:
            row["additional_60_hz_attenuation_db_vs_production"] = ""
            row["median_relative_rms_change_vs_production"] = ""
        rows.append(row)
    return rows


def write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    """Write one non-empty deterministic table."""
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_db(power: np.ndarray) -> np.ndarray:
    """Convert positive power to dB relative to 1 µV²/Hz."""
    return 10 * np.log10(np.maximum(power, np.finfo(float).tiny))


def plot_filter_responses(
    sfreq: float,
    responses: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Show candidate mathematical frequency and impulse responses."""
    frequencies = responses["frequencies"]
    figure, axes = plt.subplots(3, 1, figsize=(13, 13), constrained_layout=True)
    colors = {
        "highpass_0.5": "#f58518",
        "highpass_1": "#4c78a8",
        "lowpass_40": "#54a24b",
        "lowpass_45": "#e45756",
        "bandpass_1_40": "#4c78a8",
        "bandpass_1_45": "#b279a2",
        "notch_60": "#79706e",
    }
    for name in ("highpass_0.5", "highpass_1"):
        axes[0].plot(
            frequencies,
            20 * np.log10(np.maximum(responses[name], 1e-8)),
            color=colors[name],
            label=name.replace("_", " "),
        )
    axes[0].set_xlim(0, 5)
    axes[0].set_ylim(-80, 5)
    axes[0].axhline(-6.02, color="#777777", linestyle=":", label="−6.02 dB")
    axes[0].set_title("High-pass candidates: transition behavior")

    for name in ("lowpass_40", "lowpass_45", "notch_60"):
        axes[1].plot(
            frequencies,
            20 * np.log10(np.maximum(responses[name], 1e-8)),
            color=colors[name],
            label=name.replace("_", " "),
        )
    axes[1].set_xlim(25, 70)
    axes[1].set_ylim(-100, 5)
    axes[1].axvline(60, color="#d1495b", linestyle=":")
    axes[1].set_title("Low-pass candidates and direct 60 Hz notch")

    for name in ("highpass_0.5", "bandpass_1_40", "lowpass_40", "notch_60"):
        coefficients, _ = create_candidate_filter(name, sfreq)
        time = (np.arange(coefficients.size) - (coefficients.size - 1) / 2) / sfreq
        axes[2].plot(time, coefficients, color=colors[name], label=name.replace("_", " "))
    axes[2].set_xlim(-3.5, 3.5)
    axes[2].set_title("Centered FIR coefficient sequences (impulse responses)")
    axes[2].set_xlabel("Time relative to filter center (s)")

    for axis in axes:
        axis.set_ylabel("Gain (dB)" if axis is not axes[2] else "Coefficient")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("Candidate FIR filters designed before EEG application", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_time_comparison(
    result: PreprocessedRecording,
    start: float,
    stop: float,
    title: str,
    output_path: Path,
) -> None:
    """Compare stored, referenced, and filtered signals with matched scaling."""
    stages = {
        "Stored valid": result.stored_valid,
        "Average reference": result.referenced,
        "Average reference + 1–40 Hz FIR": result.filtered,
    }
    figure, axes = plt.subplots(
        len(REPORT_CHANNELS), len(stages), figsize=(16, 11), sharex=True,
        constrained_layout=True,
    )
    sfreq = float(result.filtered.info["sfreq"])
    start_sample = int(round(start * sfreq))
    stop_sample = int(round(stop * sfreq))
    times = result.filtered.times[start_sample:stop_sample]

    for row, channel in enumerate(REPORT_CHANNELS):
        channel_index = result.filtered.ch_names.index(channel)
        stage_values = [
            raw.get_data(picks=[channel])[0, start_sample:stop_sample] * 1_000_000
            for raw in stages.values()
        ]
        lower = min(float(np.min(values)) for values in stage_values)
        upper = max(float(np.max(values)) for values in stage_values)
        margin = max((upper - lower) * 0.05, 1.0)
        for column, ((stage_name, raw), values) in enumerate(
            zip(stages.items(), stage_values, strict=True)
        ):
            axis = axes[row, column]
            axis.plot(times, values, color=list(COLORS.values())[column], linewidth=0.9)
            for annotation in raw.annotations:
                annotation_start = float(annotation["onset"])
                annotation_stop = annotation_start + float(annotation["duration"])
                if annotation_stop >= start and annotation_start <= stop:
                    axis.axvspan(
                        max(start, annotation_start),
                        min(stop, annotation_stop),
                        color={"T0": "#dddddd", "T1": "#9ecae9", "T2": "#fdae6b"}.get(
                            str(annotation["description"]), "#eeeeee"
                        ),
                        alpha=0.13,
                    )
            axis.set_ylim(lower - margin, upper + margin)
            axis.grid(alpha=0.18)
            if row == 0:
                axis.set_title(stage_name)
            if column == 0:
                axis.set_ylabel(f"{channel}\nµV")
            if row == len(REPORT_CHANNELS) - 1:
                axis.set_xlabel("Time from recording start (s)")
    figure.suptitle(title + "\nIdentical channel scaling across processing stages", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_preprocessing_psd(
    subject: int,
    runs: Sequence[int],
    results: dict[int, PreprocessedRecording],
    output_path: Path,
) -> None:
    """Show the filter's measured spectral consequence for every run."""
    figure, axes = plt.subplots(
        len(runs), 1, figsize=(14, 4 * len(runs)), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, run in zip(axes_array, runs, strict=True):
        for stage, label in (
            ("average_reference", "Average reference, unfiltered"),
            ("filtered_1_40_hz", "Production 1–40 Hz FIR"),
        ):
            data = stage_arrays(results[run])[stage]
            psd, frequencies = compute_psd(data, 160.0)
            median = np.median(psd, axis=0)
            mask = (frequencies > 0) & (frequencies < 80)
            axis.plot(
                frequencies[mask],
                to_db(median[mask]),
                color=COLORS[stage],
                label=label,
            )
        axis.axvline(60, color="#d1495b", linestyle=":", label="Measured 60 Hz line")
        axis.set_ylabel(f"Run {run}\nPSD (dB re 1 µV²/Hz)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes_array[-1].set_xlabel("Frequency (Hz)")
    axes_array[-1].set_xlim(0.333333, 70)
    figure.suptitle(
        f"Subject {subject} — measured spectral effect of production preprocessing",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def synthetic_transient_metrics(
    config: PreprocessingConfig,
    output_path: Path,
) -> dict[str, object]:
    """Expose zero-phase pre/post-ringing with controlled impulse and step inputs."""
    sfreq = config.sampling_frequency_hz
    samples = int(20 * sfreq)
    center = samples // 2
    impulse = np.zeros(samples)
    impulse[center] = 1.0
    step = np.zeros(samples)
    step[center:] = 1.0
    settings = config.filter
    inputs = np.vstack([impulse, step])
    outputs = filter_data(
        inputs,
        sfreq,
        settings.l_freq_hz,
        settings.h_freq_hz,
        filter_length=settings.filter_length_samples,
        l_trans_bandwidth=settings.l_trans_bandwidth_hz,
        h_trans_bandwidth=settings.h_trans_bandwidth_hz,
        method=settings.method,
        phase=settings.phase,
        fir_window=settings.fir_window,
        fir_design=settings.fir_design,
        pad=settings.pad,
        verbose="error",
    )
    relative_time = (np.arange(samples) - center) / sfreq
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for axis, original, filtered, label in zip(
        axes, inputs, outputs, ("Unit impulse", "Unit step"), strict=True
    ):
        axis.plot(relative_time, original, color="#777777", label="Input", alpha=0.8)
        axis.plot(relative_time, filtered, color="#4c78a8", label="Filtered output")
        axis.axvline(0, color="#d1495b", linestyle=":")
        axis.set_xlim(-3.5, 3.5)
        axis.set_ylabel("Amplitude")
        axis.set_title(label)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[-1].set_xlabel("Time relative to abrupt change (s)")
    figure.suptitle(
        "Synthetic test — zero-phase filtering spreads abrupt structure in both directions",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    impulse_nonzero = np.flatnonzero(np.abs(outputs[0]) > np.max(np.abs(outputs[0])) * 1e-6)
    return {
        "synthetic_sampling_frequency_hz": sfreq,
        "synthetic_duration_seconds": samples / sfreq,
        "impulse_index": center,
        "impulse_response_first_relative_seconds_at_1e_6_peak": float(
            relative_time[impulse_nonzero[0]]
        ),
        "impulse_response_last_relative_seconds_at_1e_6_peak": float(
            relative_time[impulse_nonzero[-1]]
        ),
        "pre_ringing_present": bool(np.any(np.abs(outputs[0, :center]) > 1e-12)),
        "post_ringing_present": bool(np.any(np.abs(outputs[0, center + 1 :]) > 1e-12)),
    }


def artifact_token(subject: int, runs: Sequence[int]) -> str:
    """Return a stable subject/run prefix."""
    return f"subject{subject:02d}_runs" + "-".join(f"{run:02d}" for run in runs)


def display_path(path: Path) -> Path:
    """Prefer a project-relative display path."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main() -> None:
    """Execute preprocessing, candidate experiments, reports, and verification."""
    arguments = parse_arguments()
    validate_arguments(arguments)
    config_path = arguments.config.expanduser().resolve()
    config = load_preprocessing_config(config_path)
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    results: dict[int, PreprocessedRecording] = {}
    metric_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for run in arguments.runs:
        result = preprocess_recording(arguments.subject, run, config)
        results[run] = result
        metric_rows.extend(processing_metric_rows(arguments.subject, run, result))
        candidate_rows.extend(candidate_eeg_rows(arguments.subject, run, result))

    response_rows, dense_responses = filter_response_rows(
        config.sampling_frequency_hz
    )
    window_rows = (
        time_window_metric_rows(arguments.subject, 6, results[6])
        if 6 in results
        else []
    )
    prefix = artifact_token(arguments.subject, arguments.runs)
    metrics_path = output_directory / f"{prefix}_preprocessing_metrics.csv"
    candidates_path = output_directory / f"{prefix}_filter_candidate_comparison.csv"
    responses_path = output_directory / f"{prefix}_filter_response.csv"
    window_metrics_path = output_directory / f"{prefix}_time_window_metrics.csv"
    metadata_path = output_directory / f"{prefix}_preprocessing_metadata.json"
    response_figure_path = output_directory / f"{prefix}_filter_responses.png"
    psd_figure_path = output_directory / f"{prefix}_preprocessing_psd.png"
    synthetic_figure_path = output_directory / f"{prefix}_synthetic_transient_response.png"

    write_csv(metric_rows, metrics_path)
    write_csv(candidate_rows, candidates_path)
    write_csv(response_rows, responses_path)
    if window_rows:
        write_csv(window_rows, window_metrics_path)
    plot_filter_responses(config.sampling_frequency_hz, dense_responses, response_figure_path)
    plot_preprocessing_psd(
        arguments.subject, arguments.runs, results, psd_figure_path
    )
    synthetic = synthetic_transient_metrics(config, synthetic_figure_path)

    generated_paths = [
        metrics_path,
        candidates_path,
        responses_path,
        *([window_metrics_path] if window_rows else []),
        metadata_path,
        response_figure_path,
        psd_figure_path,
        synthetic_figure_path,
    ]
    if 6 in results:
        normal_path = output_directory / "subject01_run06_preprocessing_normal_008-012s.png"
        transient_path = output_directory / "subject01_run06_preprocessing_transient_013-017s.png"
        plot_time_comparison(
            results[6],
            *NORMAL_WINDOW_SECONDS,
            "Subject 1, run 6 — relatively lower-amplitude interval",
            normal_path,
        )
        plot_time_comparison(
            results[6],
            *TRANSIENT_WINDOW_SECONDS,
            "Subject 1, run 6 — large frontal transient interval",
            transient_path,
        )
        generated_paths.extend([normal_path, transient_path])

    metadata = {
        "schema_version": 1,
        "subject": arguments.subject,
        "runs": list(arguments.runs),
        "config_path": str(display_path(config_path)),
        "source_files": [
            {
                "run": run,
                "path": str(results[run].source_path.relative_to(PROJECT_ROOT)),
                "sha256": results[run].source_sha256,
                "original_samples": int(results[run].original_samples),
                "valid_samples": int(results[run].valid_samples),
                "excluded_tail_samples": int(results[run].excluded_tail_samples),
                "output_channel_count": len(results[run].filtered.ch_names),
                "output_samples": int(results[run].filtered.n_times),
                "annotation_count": len(results[run].filtered.annotations),
                "annotation_description_counts": {
                    description: int(
                        np.count_nonzero(
                            results[run].filtered.annotations.description == description
                        )
                    )
                    for description in sorted(
                        set(results[run].filtered.annotations.description)
                    )
                },
            }
            for run in arguments.runs
        ],
        "sampling_frequency_hz": config.sampling_frequency_hz,
        "output_channel_count": len(results[arguments.runs[0]].filtered.ch_names),
        "output_channel_names": list(results[arguments.runs[0]].filtered.ch_names),
        "valid_interval": "samples 0:19920; times 0.0 through 124.49375 s; 124.5 s of data",
        "pipeline_order": [
            "load immutable EDF",
            "exclude detected all-channel zero tail",
            "retain all channels after candidate review",
            "apply 64-channel average reference",
            "filter continuous valid EEG",
        ],
        "excluded_reference_channels": list(config.excluded_reference_channels),
        "reference": "64_channel_average",
        "filter": {
            **vars(config.filter),
            "filter_length_seconds": (
                config.filter.filter_length_samples / config.sampling_frequency_hz
            ),
            "notch_frequencies_hz": list(config.notch_frequencies_hz),
        },
        "welch_validation": {
            "segment_samples": WELCH_SEGMENT_SAMPLES,
            "segment_seconds": WELCH_SEGMENT_SAMPLES / config.sampling_frequency_hz,
            "overlap_samples": WELCH_OVERLAP_SAMPLES,
            "window": WELCH_WINDOW,
            "n_fft": WELCH_SEGMENT_SAMPLES,
        },
        "candidate_channel_decision": "No channels excluded from average reference at this stage.",
        "notch_decision": "No 50 or 60 Hz notch in production; 60 Hz notch was tested after low-pass and rejected as redundant.",
        "persistence_policy": "Regenerate continuous preprocessed EEG in memory; persist only small report/provenance artifacts.",
        "software": {
            "python": sys.version.split()[0],
            "mne": mne.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "synthetic_transient_test": synthetic,
        "report_artifacts": [str(path.relative_to(output_directory)) for path in generated_paths],
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    print("Controlled continuous-EEG preprocessing")
    print(f"Subject: {arguments.subject}; runs: {arguments.runs}")
    print("Order: immutable EDF → zero-tail exclusion → channel review → average reference → continuous FIR")
    print("Candidate-channel decision: no channels excluded from average reference")
    print(
        "Production FIR: 1-40 Hz passband, 1 Hz lower and 10 Hz upper transitions, "
        f"{config.filter.filter_length_samples} samples "
        f"({config.filter.filter_length_samples / config.sampling_frequency_hz:.5f} s), "
        "Hamming firwin, zero phase, reflect_limited padding"
    )
    print("Notch: none; 50 Hz unsupported and 60 Hz redundant after low-pass")
    print("No derived FIF saved; processing is regenerated from config and immutable EDF")

    for run in arguments.runs:
        result = results[run]
        rows = [row for row in candidate_rows if int(row["run"]) == run]
        production = next(
            row for row in rows if row["candidate"] == "highpass_1_lowpass_40_production"
        )
        notched = next(
            row for row in rows if row["candidate"] == "production_plus_60_hz_notch"
        )
        print(f"\nRun {run}: {result.source_path}")
        print(
            f"  Valid/excluded: {result.valid_samples}/{result.excluded_tail_samples} samples; "
            f"source SHA-256 {result.source_sha256}"
        )
        print(
            "  Median filter effects: very-low power ratio "
            f"{float(production['median_very_low_power_ratio_to_referenced']):.3f}; "
            "8-30 Hz preservation "
            f"{float(production['median_motor_8_30_power_ratio_to_referenced']):.5f}; "
            "60 Hz attenuation "
            f"{float(production['median_60_hz_attenuation_db_vs_referenced']):.2f} dB"
        )
        print(
            "  Added notch: relative RMS change vs production "
            f"{float(notched['median_relative_rms_change_vs_production']):.6f}"
        )

    print("\nGenerated report artifacts:")
    for path in generated_paths:
        print(f"  {display_path(path)}")


if __name__ == "__main__":
    main()
