"""Characterize raw EEG spectra before designing production filters.

The workflow explicitly excludes a detected all-channel zero tail, compares
stored and average-reference representations, and applies no temporal filter.
Downloaded EDF files remain immutable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci
from mne.time_frequency import psd_array_welch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
MOTOR_IMAGERY_RUNS = (6, 10, 14)
DEFAULT_RUNS = (6, 10, 14)
DEFAULT_SEGMENT_SECONDS = 3.0
DEFAULT_OVERLAP_FRACTION = 0.5
DEFAULT_LINE_FREQUENCIES = (50.0, 60.0)
WELCH_WINDOW = "hann"
SENSORIMOTOR_CHANNELS = ("C3", "Cz", "C4")
FRONTAL_REVIEW_GROUP = ("Fp1", "Fpz", "Fp2", "AF7", "AF3", "AF8")
REFERENCE_STATES = ("stored", "average")
TOTAL_POWER_RANGE_HZ = (1.0, 45.0)
EEG_BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha_mu", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma_descriptive", 30.0, 45.0),
)
COLORS = {
    "stored": "#222222",
    "average": "#4c78a8",
    "C3": "#4c78a8",
    "Cz": "#f58518",
    "C4": "#54a24b",
}


def parse_arguments() -> argparse.Namespace:
    """Parse reproducible spectral-analysis options."""
    parser = argparse.ArgumentParser(
        description=(
            "Characterize unfiltered EEGBCI spectra and propose evidence-based filters."
        )
    )
    parser.add_argument("--subject", type=int, default=1, help="EEGBCI subject (1-109).")
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        choices=MOTOR_IMAGERY_RUNS,
        default=list(DEFAULT_RUNS),
        help="One or more hands-versus-feet motor-imagery runs.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Welch segment duration in seconds.",
    )
    parser.add_argument(
        "--overlap-fraction",
        type=float,
        default=DEFAULT_OVERLAP_FRACTION,
        help="Fractional overlap between adjacent Welch segments.",
    )
    parser.add_argument(
        "--line-frequencies",
        type=float,
        nargs="+",
        default=list(DEFAULT_LINE_FREQUENCIES),
        help="Candidate narrow line frequencies to inspect in Hz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for report figures and CSV tables.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Validate configuration before loading or transforming data."""
    if not 1 <= arguments.subject <= 109:
        raise ValueError(f"Subject must be between 1 and 109; received {arguments.subject}.")
    if len(set(arguments.runs)) != len(arguments.runs):
        raise ValueError("Run numbers must not be duplicated.")
    if arguments.segment_seconds <= 0:
        raise ValueError("Welch segment duration must be greater than zero.")
    if not 0 <= arguments.overlap_fraction < 1:
        raise ValueError("Welch overlap fraction must be in [0, 1).")
    if len(set(arguments.line_frequencies)) != len(arguments.line_frequencies):
        raise ValueError("Line frequencies must not be duplicated.")


def load_recording(subject: int, run: int) -> tuple[mne.io.BaseRaw, Path]:
    """Load one raw EDF and standardize labels without signal preprocessing."""
    (downloaded_file,) = eegbci.load_data(
        subjects=subject,
        runs=run,
        path=DATA_DIRECTORY,
        update_path=False,
    )
    file_path = Path(downloaded_file)
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose="error")
    eegbci.standardize(raw)
    return raw, file_path


def find_trailing_all_channel_zeros(data: np.ndarray) -> tuple[int, int]:
    """Return the valid stop index and trailing all-channel zero count."""
    all_channel_zero = np.all(data == 0, axis=0)
    if not all_channel_zero[-1]:
        return data.shape[1], 0
    nonzero_indices = np.flatnonzero(~all_channel_zero)
    valid_stop = int(nonzero_indices[-1] + 1) if nonzero_indices.size else 0
    return valid_stop, int(data.shape[1] - valid_stop)


def create_reference_representations(
    raw: mne.io.BaseRaw, valid_stop: int
) -> tuple[mne.io.BaseRaw, dict[str, np.ndarray], np.ndarray]:
    """Create stored and average-reference arrays from one valid copied view."""
    if valid_stop < 2:
        raise RuntimeError("Fewer than two valid EEG samples remain.")
    valid_end_time = float(raw.times[valid_stop - 1])
    valid_raw = raw.copy().crop(tmax=valid_end_time).load_data(verbose="error")
    if valid_raw.n_times != valid_stop:
        raise RuntimeError(
            f"Valid crop contains {valid_raw.n_times} samples; expected {valid_stop}."
        )

    stored = valid_raw.get_data()
    stored_snapshot = stored.copy()
    reference_trace = np.mean(stored, axis=0)

    average_raw = valid_raw.copy()
    average_raw.set_eeg_reference(
        ref_channels="average", projection=False, verbose="error"
    )
    average = average_raw.get_data()

    if not np.array_equal(valid_raw.get_data(), stored_snapshot):
        raise RuntimeError("Stored-reference valid Raw object was modified.")
    if not np.allclose(
        average,
        stored - reference_trace,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise RuntimeError("MNE average reference does not match explicit subtraction.")
    if np.max(np.abs(np.mean(average, axis=0))) > 1e-15:
        raise RuntimeError("Average-reference channel mean is not numerically zero.")

    return valid_raw, {"stored": stored, "average": average}, reference_trace


def compute_welch_psd(
    data_volts: np.ndarray,
    sampling_frequency: float,
    segment_samples: int,
    overlap_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute explicit mean Welch PSD and convert V²/Hz to µV²/Hz."""
    psd_volts_squared, frequencies = psd_array_welch(
        data_volts,
        sampling_frequency,
        fmin=0.0,
        fmax=sampling_frequency / 2,
        n_fft=segment_samples,
        n_per_seg=segment_samples,
        n_overlap=overlap_samples,
        average="mean",
        window=WELCH_WINDOW,
        remove_dc=True,
        output="power",
        verbose="error",
    )
    return psd_volts_squared * 1_000_000**2, frequencies


def frequency_mask(
    frequencies: np.ndarray, lower_hz: float, upper_hz: float
) -> np.ndarray:
    """Select an inclusive frequency interval with floating-point tolerance."""
    tolerance = np.finfo(float).eps * max(1.0, abs(lower_hz), abs(upper_hz)) * 16
    return (frequencies >= lower_hz - tolerance) & (
        frequencies <= upper_hz + tolerance
    )


def integrate_power(
    psd_uV2_per_hz: np.ndarray,
    frequencies: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> np.ndarray:
    """Integrate PSD over a closed band, returning power in µV²."""
    mask = frequency_mask(frequencies, lower_hz, upper_hz)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Band {lower_hz:g}-{upper_hz:g} Hz has fewer than two bins.")
    return np.trapezoid(
        psd_uV2_per_hz[..., mask], frequencies[mask], axis=-1
    )


def to_db(power: np.ndarray) -> np.ndarray:
    """Convert positive power to dB relative to 1 µV² or 1 µV²/Hz."""
    if np.any(power <= 0) or not np.isfinite(power).all():
        raise RuntimeError("dB conversion requires finite positive power values.")
    return 10 * np.log10(power)


def verify_welch_configuration(
    stored_data: np.ndarray,
    psd_uV2_per_hz: np.ndarray,
    frequencies: np.ndarray,
    sampling_frequency: float,
    segment_samples: int,
    overlap_samples: int,
    expected_segments: int,
) -> None:
    """Perform independent grid, segment, averaging, and power sanity checks."""
    expected_frequencies = np.fft.rfftfreq(
        segment_samples, d=1 / sampling_frequency
    )
    if not np.array_equal(frequencies, expected_frequencies):
        raise RuntimeError("Welch frequency grid does not match NumPy's DFT grid.")
    if frequencies[-1] != sampling_frequency / 2:
        raise RuntimeError("Welch grid does not end exactly at Nyquist.")

    unaggregated, unaggregated_frequencies = psd_array_welch(
        stored_data[0],
        sampling_frequency,
        fmin=0.0,
        fmax=sampling_frequency / 2,
        n_fft=segment_samples,
        n_per_seg=segment_samples,
        n_overlap=overlap_samples,
        average=None,
        window=WELCH_WINDOW,
        remove_dc=True,
        output="power",
        verbose="error",
    )
    if unaggregated.shape != (frequencies.size, expected_segments):
        raise RuntimeError(
            f"Unexpected unaggregated Welch shape: {unaggregated.shape}."
        )
    if not np.array_equal(unaggregated_frequencies, frequencies):
        raise RuntimeError("Aggregated and unaggregated frequency grids differ.")
    if not np.allclose(
        np.mean(unaggregated, axis=-1) * 1_000_000**2,
        psd_uV2_per_hz[0],
        rtol=1e-12,
        atol=1e-18,
    ):
        raise RuntimeError("Mean Welch PSD does not equal the segment-periodogram mean.")

    sample_indices = np.arange(segment_samples * 4)
    synthetic = np.sin(
        2 * np.pi * 10.0 * sample_indices / sampling_frequency
    )
    synthetic_psd, synthetic_frequencies = psd_array_welch(
        synthetic,
        sampling_frequency,
        fmin=0.0,
        fmax=sampling_frequency / 2,
        n_fft=segment_samples,
        n_per_seg=segment_samples,
        n_overlap=overlap_samples,
        average="mean",
        window=WELCH_WINDOW,
        remove_dc=True,
        output="power",
        verbose="error",
    )
    peak_frequency = float(synthetic_frequencies[np.argmax(synthetic_psd)])
    integrated_power = float(np.trapezoid(synthetic_psd, synthetic_frequencies))
    if not np.isclose(peak_frequency, 10.0) or not np.isclose(
        integrated_power, 0.5, rtol=1e-10
    ):
        raise RuntimeError(
            "Synthetic 10 Hz sine sanity check failed: "
            f"peak={peak_frequency}, integrated power={integrated_power}."
        )


def channel_summary_rows(
    subject: int,
    run: int,
    channel_names: Sequence[str],
    reference_state: str,
    psd: np.ndarray,
    frequencies: np.ndarray,
    valid_samples: int,
    excluded_tail_samples: int,
    sampling_frequency: float,
    segment_samples: int,
    overlap_samples: int,
    segment_count: int,
) -> list[dict[str, object]]:
    """Create interpretable per-channel spectral summary rows."""
    first_positive_frequency = float(frequencies[1])
    highest_non_nyquist_frequency = float(frequencies[-2])
    very_low = integrate_power(psd, frequencies, first_positive_frequency, 1.0)
    total = integrate_power(psd, frequencies, *TOTAL_POWER_RANGE_HZ)
    high = integrate_power(
        psd, frequencies, 45.0, highest_non_nyquist_frequency
    )
    peak_mask = frequency_mask(frequencies, *TOTAL_POWER_RANGE_HZ)
    peak_indices = np.argmax(psd[:, peak_mask], axis=1)
    selected_frequencies = frequencies[peak_mask]
    selected_psd = psd[:, peak_mask]
    frequency_resolution = float(frequencies[1] - frequencies[0])

    rows: list[dict[str, object]] = []
    for index, channel in enumerate(channel_names):
        rows.append(
            {
                "subject": subject,
                "run": run,
                "channel": channel,
                "reference_state": reference_state,
                "valid_samples": valid_samples,
                "valid_duration_seconds": valid_samples / sampling_frequency,
                "excluded_tail_samples": excluded_tail_samples,
                "excluded_tail_seconds": excluded_tail_samples / sampling_frequency,
                "sampling_frequency_hz": sampling_frequency,
                "nyquist_hz": sampling_frequency / 2,
                "welch_segment_seconds": segment_samples / sampling_frequency,
                "welch_segment_samples": segment_samples,
                "welch_overlap_samples": overlap_samples,
                "welch_overlap_fraction": overlap_samples / segment_samples,
                "welch_window": WELCH_WINDOW,
                "welch_average": "mean",
                "remove_segment_dc": True,
                "n_fft": segment_samples,
                "zero_padding": False,
                "frequency_bin_spacing_hz": frequency_resolution,
                "welch_segment_count": segment_count,
                "very_low_lower_hz": first_positive_frequency,
                "very_low_upper_hz": 1.0,
                "very_low_power_uV2": float(very_low[index]),
                "total_power_1_45_hz_uV2": float(total[index]),
                "high_power_lower_hz": 45.0,
                "high_power_upper_hz": highest_non_nyquist_frequency,
                "high_power_uV2": float(high[index]),
                "peak_frequency_1_45_hz": float(selected_frequencies[peak_indices[index]]),
                "peak_psd_1_45_hz_uV2_per_hz": float(
                    selected_psd[index, peak_indices[index]]
                ),
            }
        )
    return rows


def band_power_rows(
    subject: int,
    run: int,
    channel_names: Sequence[str],
    reference_state: str,
    psd: np.ndarray,
    frequencies: np.ndarray,
    segment_seconds: float,
) -> list[dict[str, object]]:
    """Calculate absolute and explicitly denominator-defined relative power."""
    total = integrate_power(psd, frequencies, *TOTAL_POWER_RANGE_HZ)
    rows: list[dict[str, object]] = []
    summed_band_power = np.zeros_like(total)

    for band_name, lower_hz, upper_hz in EEG_BANDS:
        absolute = integrate_power(psd, frequencies, lower_hz, upper_hz)
        summed_band_power += absolute
        relative = absolute / total
        for index, channel in enumerate(channel_names):
            rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "channel": channel,
                    "reference_state": reference_state,
                    "band": band_name,
                    "band_lower_hz": lower_hz,
                    "band_upper_hz": upper_hz,
                    "absolute_band_power_uV2": float(absolute[index]),
                    "relative_band_power": float(relative[index]),
                    "relative_power_denominator": "total_power_1_45_hz",
                    "total_power_1_45_hz_uV2": float(total[index]),
                    "welch_segment_seconds": segment_seconds,
                    "psd_units_before_integration": "uV2_per_Hz",
                    "absolute_power_units": "uV2",
                }
            )

    if not np.allclose(summed_band_power, total, rtol=1e-12, atol=1e-9):
        maximum_error = float(np.max(np.abs(summed_band_power - total)))
        raise RuntimeError(f"Adjacent band powers do not sum to 1-45 Hz: {maximum_error}")
    return rows


def line_frequency_rows(
    subject: int,
    run: int,
    channel_names: Sequence[str],
    reference_state: str,
    psd: np.ndarray,
    frequencies: np.ndarray,
    line_frequency: float,
) -> list[dict[str, object]]:
    """Compare an exact candidate line bin with flanking baseline intervals."""
    line_index = int(np.argmin(np.abs(frequencies - line_frequency)))
    if not np.isclose(frequencies[line_index], line_frequency):
        raise RuntimeError(
            f"Configured grid has no exact {line_frequency:g} Hz frequency bin."
        )
    neighbor_mask = frequency_mask(
        frequencies, line_frequency - 3, line_frequency - 1
    ) | frequency_mask(frequencies, line_frequency + 1, line_frequency + 3)
    zoom_mask = frequency_mask(
        frequencies, line_frequency - 2, line_frequency + 2
    )
    neighbor_median = np.median(psd[:, neighbor_mask], axis=1)
    line_psd = psd[:, line_index]
    ratio = line_psd / neighbor_median
    zoom_psd = psd[:, zoom_mask]
    zoom_frequencies = frequencies[zoom_mask]
    zoom_peak_indices = np.argmax(zoom_psd, axis=1)

    rows: list[dict[str, object]] = []
    for index, channel in enumerate(channel_names):
        rows.append(
            {
                "subject": subject,
                "run": run,
                "channel": channel,
                "reference_state": reference_state,
                "line_frequency_hz": line_frequency,
                "line_bin_psd_uV2_per_hz": float(line_psd[index]),
                "neighbor_definition_hz": (
                    f"{line_frequency - 3:g}-{line_frequency - 1:g} and "
                    f"{line_frequency + 1:g}-{line_frequency + 3:g}"
                ),
                "neighbor_median_psd_uV2_per_hz": float(neighbor_median[index]),
                "line_to_neighbor_ratio": float(ratio[index]),
                "line_excess_db": float(to_db(ratio[index : index + 1])[0]),
                "local_peak_frequency_hz": float(
                    zoom_frequencies[zoom_peak_indices[index]]
                ),
                "local_peak_psd_uV2_per_hz": float(
                    zoom_psd[index, zoom_peak_indices[index]]
                ),
            }
        )
    return rows


def reference_effect_rows(
    subject: int,
    run: int,
    channel_names: Sequence[str],
    stored_data: np.ndarray,
    reference_trace: np.ndarray,
    stored_psd: np.ndarray,
    average_psd: np.ndarray,
    frequencies: np.ndarray,
) -> list[dict[str, object]]:
    """Quantify power changes and leave-one-channel-out reference sensitivity."""
    stored_total = integrate_power(
        stored_psd, frequencies, *TOTAL_POWER_RANGE_HZ
    )
    average_total = integrate_power(
        average_psd, frequencies, *TOTAL_POWER_RANGE_HZ
    )
    channel_count = len(channel_names)
    reference_std = float(np.std(reference_trace))
    rows: list[dict[str, object]] = []

    for index, channel in enumerate(channel_names):
        correlation = float(np.corrcoef(stored_data[index], reference_trace)[0, 1])
        reference_without_channel = (
            channel_count * reference_trace - stored_data[index]
        ) / (channel_count - 1)
        sensitivity = float(
            np.std(reference_trace - reference_without_channel) / reference_std
        )
        power_ratio = float(average_total[index] / stored_total[index])
        rows.append(
            {
                "subject": subject,
                "run": run,
                "channel": channel,
                "correlation_with_64_channel_mean_trace": correlation,
                "leave_one_out_reference_std_change_fraction": sensitivity,
                "stored_total_power_1_45_hz_uV2": float(stored_total[index]),
                "average_reference_total_power_1_45_hz_uV2": float(
                    average_total[index]
                ),
                "average_to_stored_power_ratio": power_ratio,
                "average_to_stored_power_change_db": float(
                    to_db(np.array([power_ratio]))[0]
                ),
            }
        )
    return rows


def psd_curve_rows(
    subject: int,
    run: int,
    channel_names: Sequence[str],
    reference_state: str,
    psd: np.ndarray,
    frequencies: np.ndarray,
) -> list[dict[str, object]]:
    """Export the robust channel aggregation and central-channel plot curves."""
    report_mask = (frequencies > 0) & (frequencies < frequencies[-1])
    central_indices = {
        channel: channel_names.index(channel) for channel in SENSORIMOTOR_CHANNELS
    }
    rows: list[dict[str, object]] = []
    for frequency_index in np.flatnonzero(report_mask):
        values = psd[:, frequency_index]
        row: dict[str, object] = {
            "subject": subject,
            "run": run,
            "reference_state": reference_state,
            "frequency_hz": float(frequencies[frequency_index]),
            "channel_median_psd_uV2_per_hz": float(np.median(values)),
            "channel_q25_psd_uV2_per_hz": float(np.percentile(values, 25)),
            "channel_q75_psd_uV2_per_hz": float(np.percentile(values, 75)),
        }
        for channel, channel_index in central_indices.items():
            row[f"{channel}_psd_uV2_per_hz"] = float(
                psd[channel_index, frequency_index]
            )
        rows.append(row)
    return rows


def write_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    """Write a non-empty deterministic dictionary table."""
    if not rows:
        raise ValueError(f"Cannot write an empty table: {output_path}")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_broad_psd(
    subject: int,
    runs: Sequence[int],
    psds: dict[int, dict[str, np.ndarray]],
    frequencies: np.ndarray,
    line_frequencies: Sequence[float],
    output_path: Path,
) -> None:
    """Plot robust whole-scalp PSD across the representable frequency range."""
    figure, axes = plt.subplots(
        len(runs), 1, figsize=(15, 4 * len(runs)), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    plot_mask = (frequencies >= frequencies[1]) & (frequencies < frequencies[-1])
    plot_frequencies = frequencies[plot_mask]

    for axis, run in zip(axes_array, runs, strict=True):
        for reference_state in REFERENCE_STATES:
            values = psds[run][reference_state][:, plot_mask]
            median = np.median(values, axis=0)
            q25 = np.percentile(values, 25, axis=0)
            q75 = np.percentile(values, 75, axis=0)
            color = COLORS[reference_state]
            label = "Stored reference" if reference_state == "stored" else "Average reference"
            axis.plot(plot_frequencies, to_db(median), color=color, label=label)
            axis.fill_between(
                plot_frequencies,
                to_db(q25),
                to_db(q75),
                color=color,
                alpha=0.12,
            )
        for line_frequency in line_frequencies:
            axis.axvline(
                line_frequency, color="#d1495b", linestyle=":", linewidth=1.2
            )
            axis.text(
                line_frequency + 0.8,
                0.95,
                f"{line_frequency:g} Hz",
                transform=axis.get_xaxis_transform(),
                color="#a12f40",
                va="top",
            )
        axis.set_ylabel(f"Run {run}\nPSD (dB re 1 µV²/Hz)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, loc="upper right")
    axes_array[-1].set_xlabel("Frequency (Hz)")
    axes_array[-1].set_xlim(float(frequencies[1]), float(frequencies[-2]))
    figure.suptitle(
        f"Subject {subject} — broad unfiltered Welch PSD with reference comparison\n"
        "Lines are the 64-channel median; shading is the interquartile range",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_sensorimotor_psd(
    subject: int,
    runs: Sequence[int],
    channel_names: Sequence[str],
    psds: dict[int, dict[str, np.ndarray]],
    frequencies: np.ndarray,
    output_path: Path,
) -> None:
    """Plot continuous average-reference PSD for C3, Cz, and C4."""
    figure, axes = plt.subplots(
        len(runs), 1, figsize=(14, 4 * len(runs)), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    plot_mask = frequency_mask(frequencies, 2.0, 35.0)

    for axis, run in zip(axes_array, runs, strict=True):
        axis.axvspan(8, 13, color="#72b7b2", alpha=0.12, label="8–13 Hz alpha/mu")
        axis.axvspan(13, 30, color="#eeca3b", alpha=0.10, label="13–30 Hz beta")
        for channel in SENSORIMOTOR_CHANNELS:
            channel_index = channel_names.index(channel)
            values = psds[run]["average"][channel_index, plot_mask]
            axis.plot(
                frequencies[plot_mask],
                to_db(values),
                color=COLORS[channel],
                label=channel,
                linewidth=1.4,
            )
        axis.set_ylabel(f"Run {run}\nPSD (dB re 1 µV²/Hz)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=5, loc="upper right")
    axes_array[-1].set_xlabel("Frequency (Hz)")
    axes_array[-1].set_xlim(2, 35)
    figure.suptitle(
        f"Subject {subject} — continuous average-reference sensorimotor spectra\n"
        "Descriptive whole-run PSD; not an event-related motor-imagery comparison",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_line_frequency_analysis(
    subject: int,
    runs: Sequence[int],
    psds: dict[int, dict[str, np.ndarray]],
    frequencies: np.ndarray,
    line_frequencies: Sequence[float],
    line_table: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    """Plot local PSD shape and per-channel excess at candidate line frequencies."""
    figure, axes = plt.subplots(
        1 + len(line_frequencies),
        1,
        figsize=(14, 4 + 4 * len(line_frequencies)),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    zoom_mask = frequency_mask(
        frequencies,
        min(line_frequencies) - 5,
        max(line_frequencies) + 5,
    )
    run_colors = ("#4c78a8", "#f58518", "#54a24b")

    for run, color in zip(runs, run_colors, strict=False):
        median = np.median(psds[run]["average"][:, zoom_mask], axis=0)
        axes_array[0].plot(
            frequencies[zoom_mask],
            to_db(median),
            color=color,
            label=f"Run {run}, average reference",
        )
    for line_frequency in line_frequencies:
        axes_array[0].axvline(line_frequency, color="#d1495b", linestyle=":")
    axes_array[0].set_ylabel("Channel-median PSD\n(dB re 1 µV²/Hz)")
    axes_array[0].grid(alpha=0.2)
    axes_array[0].legend(frameon=False)

    for axis, line_frequency in zip(
        axes_array[1:], line_frequencies, strict=True
    ):
        distributions: list[list[float]] = []
        labels: list[str] = []
        colors: list[str] = []
        for run, run_color in zip(runs, run_colors, strict=False):
            for reference_state in REFERENCE_STATES:
                values = [
                    float(row["line_excess_db"])
                    for row in line_table
                    if int(row["run"]) == run
                    and row["reference_state"] == reference_state
                    and float(row["line_frequency_hz"]) == line_frequency
                ]
                distributions.append(values)
                labels.append(
                    f"R{run}\n"
                    f"{'stored' if reference_state == 'stored' else 'average'}"
                )
                colors.append(run_color if reference_state == "average" else "#999999")
        boxplot = axis.boxplot(
            distributions, tick_labels=labels, patch_artist=True
        )
        for patch, color in zip(boxplot["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.axhline(0, color="#222222", linewidth=1)
        axis.axhline(
            3,
            color="#d1495b",
            linestyle="--",
            linewidth=1,
            label="2× power (3 dB)",
        )
        axis.set_ylabel(
            f"{line_frequency:g} Hz excess over\n"
            f"{line_frequency - 3:g}–{line_frequency - 1:g} and "
            f"{line_frequency + 1:g}–{line_frequency + 3:g} Hz (dB)"
        )
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle(
        f"Subject {subject} — quantitative narrow line-frequency inspection",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def format_runs(runs: Sequence[int]) -> str:
    """Create a stable run token for generated artifact names."""
    return "-".join(f"{run:02d}" for run in runs)


def display_path(path: Path) -> Path:
    """Prefer a project-relative path while supporting external output paths."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main() -> None:
    """Execute spectral characterization without applying production filters."""
    arguments = parse_arguments()
    validate_arguments(arguments)
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    raw_by_run: dict[int, mne.io.BaseRaw] = {}
    path_by_run: dict[int, Path] = {}
    valid_stop_by_run: dict[int, int] = {}
    tail_count_by_run: dict[int, int] = {}
    other_zero_count_by_run: dict[int, int] = {}
    representations: dict[int, dict[str, np.ndarray]] = {}
    reference_trace_by_run: dict[int, np.ndarray] = {}
    psds: dict[int, dict[str, np.ndarray]] = {}

    channel_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    line_frequency_table: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    common_frequencies: np.ndarray | None = None
    channel_names: list[str] | None = None

    for run in arguments.runs:
        raw, file_path = load_recording(arguments.subject, run)
        raw_by_run[run] = raw
        path_by_run[run] = file_path
        complete_data = raw.get_data()
        valid_stop, tail_count = find_trailing_all_channel_zeros(complete_data)
        other_zero_count = int(
            np.count_nonzero(np.all(complete_data[:, :valid_stop] == 0, axis=0))
        )
        if other_zero_count:
            raise RuntimeError(
                f"Run {run} has {other_zero_count} all-channel zero samples before the tail."
            )
        valid_raw, run_representations, reference_trace = (
            create_reference_representations(raw, valid_stop)
        )
        if set(valid_raw.info["bads"]):
            raise RuntimeError("This workflow does not silently exclude marked bad channels.")
        missing_central = sorted(set(SENSORIMOTOR_CHANNELS) - set(valid_raw.ch_names))
        if missing_central:
            raise RuntimeError(f"Missing sensorimotor channels: {missing_central}")

        raw_by_run[run] = valid_raw
        valid_stop_by_run[run] = valid_stop
        tail_count_by_run[run] = tail_count
        other_zero_count_by_run[run] = other_zero_count
        representations[run] = run_representations
        reference_trace_by_run[run] = reference_trace
        if channel_names is None:
            channel_names = list(valid_raw.ch_names)
        elif channel_names != valid_raw.ch_names:
            raise RuntimeError("Channel order differs across requested runs.")

    assert channel_names is not None
    sampling_frequency = float(raw_by_run[arguments.runs[0]].info["sfreq"])
    segment_samples = int(round(arguments.segment_seconds * sampling_frequency))
    overlap_samples = int(round(arguments.overlap_fraction * segment_samples))
    if segment_samples < 2 or overlap_samples >= segment_samples:
        raise ValueError("Configured Welch segment/overlap has invalid sample counts.")
    frequency_resolution = sampling_frequency / segment_samples
    nyquist = sampling_frequency / 2
    invalid_line_frequencies = [
        frequency
        for frequency in arguments.line_frequencies
        if frequency <= 0 or frequency >= nyquist
    ]
    if invalid_line_frequencies:
        raise ValueError(
            "Line frequencies must be greater than 0 Hz and below Nyquist; "
            f"received {invalid_line_frequencies}."
        )

    for run in arguments.runs:
        valid_samples = valid_stop_by_run[run]
        segment_step = segment_samples - overlap_samples
        segment_count = 1 + (valid_samples - segment_samples) // segment_step
        used_samples = segment_samples + (segment_count - 1) * segment_step
        if used_samples != valid_samples:
            raise RuntimeError(
                f"Run {run}: Welch windows use {used_samples}/{valid_samples} valid samples. "
                "Choose parameters that cover the valid interval exactly."
            )
        psds[run] = {}
        for reference_state in REFERENCE_STATES:
            psd, frequencies = compute_welch_psd(
                representations[run][reference_state],
                sampling_frequency,
                segment_samples,
                overlap_samples,
            )
            if psd.shape != (64, segment_samples // 2 + 1):
                raise RuntimeError(f"Unexpected run {run} PSD shape: {psd.shape}")
            if not np.isfinite(psd).all() or np.any(psd < 0):
                raise RuntimeError(f"Run {run} {reference_state} PSD is invalid.")
            if frequencies[-1] > nyquist:
                raise RuntimeError("PSD frequency grid exceeds Nyquist.")
            if not np.isclose(frequencies[1] - frequencies[0], frequency_resolution):
                raise RuntimeError("PSD bin spacing differs from fs/n_fft.")
            if common_frequencies is None:
                common_frequencies = frequencies
            elif not np.array_equal(common_frequencies, frequencies):
                raise RuntimeError("Frequency grids differ across run/reference analyses.")
            psds[run][reference_state] = psd

            channel_rows.extend(
                channel_summary_rows(
                    arguments.subject,
                    run,
                    channel_names,
                    reference_state,
                    psd,
                    frequencies,
                    valid_samples,
                    tail_count_by_run[run],
                    sampling_frequency,
                    segment_samples,
                    overlap_samples,
                    segment_count,
                )
            )
            band_rows.extend(
                band_power_rows(
                    arguments.subject,
                    run,
                    channel_names,
                    reference_state,
                    psd,
                    frequencies,
                    segment_samples / sampling_frequency,
                )
            )
            for line_frequency in arguments.line_frequencies:
                line_frequency_table.extend(
                    line_frequency_rows(
                        arguments.subject,
                        run,
                        channel_names,
                        reference_state,
                        psd,
                        frequencies,
                        line_frequency,
                    )
                )
            curve_rows.extend(
                psd_curve_rows(
                    arguments.subject,
                    run,
                    channel_names,
                    reference_state,
                    psd,
                    frequencies,
                )
            )

        verify_welch_configuration(
            representations[run]["stored"],
            psds[run]["stored"],
            frequencies,
            sampling_frequency,
            segment_samples,
            overlap_samples,
            segment_count,
        )
        reference_rows.extend(
            reference_effect_rows(
                arguments.subject,
                run,
                channel_names,
                representations[run]["stored"],
                reference_trace_by_run[run],
                psds[run]["stored"],
                psds[run]["average"],
                frequencies,
            )
        )

    assert common_frequencies is not None
    run_token = format_runs(arguments.runs)
    prefix = f"subject{arguments.subject:02d}_runs{run_token}"
    channel_csv_path = output_directory / f"{prefix}_channel_psd_summary.csv"
    band_csv_path = output_directory / f"{prefix}_band_power.csv"
    line_frequency_csv_path = (
        output_directory / f"{prefix}_line_frequency_analysis.csv"
    )
    reference_csv_path = output_directory / f"{prefix}_reference_effects.csv"
    curves_csv_path = output_directory / f"{prefix}_psd_curves.csv"
    broad_figure_path = output_directory / f"{prefix}_broad_psd.png"
    sensorimotor_figure_path = output_directory / f"{prefix}_sensorimotor_psd.png"
    line_frequency_figure_path = (
        output_directory / f"{prefix}_line_frequency_analysis.png"
    )

    write_csv(channel_rows, channel_csv_path)
    write_csv(band_rows, band_csv_path)
    write_csv(line_frequency_table, line_frequency_csv_path)
    write_csv(reference_rows, reference_csv_path)
    write_csv(curve_rows, curves_csv_path)
    plot_broad_psd(
        arguments.subject,
        arguments.runs,
        psds,
        common_frequencies,
        arguments.line_frequencies,
        broad_figure_path,
    )
    plot_sensorimotor_psd(
        arguments.subject,
        arguments.runs,
        channel_names,
        psds,
        common_frequencies,
        sensorimotor_figure_path,
    )
    plot_line_frequency_analysis(
        arguments.subject,
        arguments.runs,
        psds,
        common_frequencies,
        arguments.line_frequencies,
        line_frequency_table,
        line_frequency_figure_path,
    )

    print("Raw frequency-domain characterization")
    print(f"Subject: {arguments.subject}; runs: {arguments.runs}")
    print("No high-pass, low-pass, band-pass, or notch filter was applied.")
    print("Reference states: stored and 64-channel average-reference copy")
    print(f"Sampling frequency: {sampling_frequency:g} Hz; Nyquist: {nyquist:g} Hz")
    print(
        f"Welch: {segment_samples} samples ({segment_samples / sampling_frequency:g} s), "
        f"{overlap_samples} sample overlap ({overlap_samples / segment_samples:.0%}), "
        f"{WELCH_WINDOW} window, mean average, segment DC removed"
    )
    print(
        f"n_fft={segment_samples}; zero padding=False; "
        f"frequency-bin spacing={frequency_resolution:.6f} Hz"
    )

    for run in arguments.runs:
        print(f"\nRun {run}")
        print(f"  Raw file: {path_by_run[run]}")
        print(
            f"  Excluded trailing all-channel zeros: {tail_count_by_run[run]} samples "
            f"({tail_count_by_run[run] / sampling_frequency:g} s)"
        )
        print(f"  Other all-channel zero samples in valid view: {other_zero_count_by_run[run]}")
        print(
            f"  Valid data: {valid_stop_by_run[run]} samples "
            f"({valid_stop_by_run[run] / sampling_frequency:g} s), "
            f"{1 + (valid_stop_by_run[run] - segment_samples) // (segment_samples - overlap_samples)} "
            "Welch segments"
        )

        for reference_state in REFERENCE_STATES:
            for line_frequency in arguments.line_frequencies:
                rows = [
                    row
                    for row in line_frequency_table
                    if int(row["run"]) == run
                    and row["reference_state"] == reference_state
                    and float(row["line_frequency_hz"]) == line_frequency
                ]
                excess = np.array([float(row["line_excess_db"]) for row in rows])
                print(
                    f"  {reference_state} {line_frequency:g} Hz excess dB: "
                    f"median={np.median(excess):.2f}, "
                    f"Q75={np.percentile(excess, 75):.2f}, "
                    f"max={np.max(excess):.2f}, "
                    f"channels >3 dB={np.count_nonzero(excess > 3)}/64"
                )

        run_reference_rows = [
            row for row in reference_rows if int(row["run"]) == run
        ]
        power_changes = np.array(
            [float(row["average_to_stored_power_change_db"]) for row in run_reference_rows]
        )
        sensitivities = np.array(
            [
                float(row["leave_one_out_reference_std_change_fraction"])
                for row in run_reference_rows
            ]
        )
        maximum_index = int(np.argmax(sensitivities))
        frontal_indices = [channel_names.index(channel) for channel in FRONTAL_REVIEW_GROUP]
        stored = representations[run]["stored"]
        reference_trace = reference_trace_by_run[run]
        reference_without_frontal = (
            len(channel_names) * reference_trace - stored[frontal_indices].sum(axis=0)
        ) / (len(channel_names) - len(frontal_indices))
        frontal_sensitivity = float(
            np.std(reference_trace - reference_without_frontal)
            / np.std(reference_trace)
        )
        print(
            "  Average-reference 1-45 Hz power change: "
            f"channel median={np.median(power_changes):.2f} dB"
        )
        print(
            "  Largest single-channel leave-one-out reference sensitivity: "
            f"{channel_names[maximum_index]}={sensitivities[maximum_index]:.3f}"
        )
        print(
            "  Six-channel frontal-group leave-out reference sensitivity: "
            f"{frontal_sensitivity:.3f}"
        )

        run_band_rows = [
            row
            for row in band_rows
            if int(row["run"]) == run
            and row["reference_state"] == "average"
        ]
        print("  Average-reference median relative 1-45 Hz band powers:")
        for band_name, _, _ in EEG_BANDS:
            values = np.array(
                [
                    float(row["relative_band_power"])
                    for row in run_band_rows
                    if row["band"] == band_name
                ]
            )
            print(f"    {band_name}: {np.median(values):.3f}")

    print("\nEvidence supplied by this spectral milestone for filter validation")
    print("  High-pass: evaluate a 1 Hz design; substantial sub-1/delta power is present")
    print("  Low-pass: evaluate a 40 Hz design for the primary 8-30 Hz objective")
    print("  50 Hz notch: not justified by the measured 50 Hz excess")
    print(
        "  60 Hz notch: conditional; a strong line is measured, but a 40 Hz "
        "low-pass may already suppress it"
    )
    print("  Feature band: 8-30 Hz remains a later analysis choice, not raw preprocessing")
    print("  Production filters were deliberately not implemented here")

    print("\nGenerated report artifacts:")
    for generated_path in (
        channel_csv_path,
        band_csv_path,
        line_frequency_csv_path,
        reference_csv_path,
        curves_csv_path,
        broad_figure_path,
        sensorimotor_figure_path,
        line_frequency_figure_path,
    ):
        print(f"  {display_path(generated_path)}")


if __name__ == "__main__":
    main()
