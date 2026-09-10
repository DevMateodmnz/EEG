"""Outcome-free, participant-aware reliability operations for individual mu peaks.

This module deliberately accepts only T0/rest arrays. It delegates spectral and
peak decisions to the frozen v1.0 implementation instead of re-implementing them.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .individual_frequency import PeakEstimate, estimate_central_peak, rest_roi_log_welch


RunPeaks = Mapping[int, Sequence[tuple[int, float]]]


def estimate_run_rest_peak(
    rest_data_volts: np.ndarray,
    channel_names: Sequence[str],
    sampling_frequency_hz: float,
    central_channels: Sequence[str],
    window_seconds: float,
    search_range_hz: tuple[float, float],
    prominence_db: float,
) -> PeakEstimate:
    """Estimate one run's peak from its T0 rest epochs alone.

    No task array, label, or outcome is accepted by this API. The two called
    functions are the frozen v1.0 Welch and peak-decision implementations.
    """
    frequencies, log_power = rest_roi_log_welch(
        rest_data_volts, channel_names, sampling_frequency_hz, central_channels, window_seconds
    )
    return estimate_central_peak(frequencies, log_power, search_range_hz, prominence_db)


def pairwise_rows(values_by_subject: RunPeaks) -> list[dict[str, float | int]]:
    """Return deterministic unique unordered within-subject peak differences."""
    rows: list[dict[str, float | int]] = []
    for subject in sorted(values_by_subject):
        for (left_run, left), (right_run, right) in combinations(sorted(values_by_subject[subject]), 2):
            rows.append({
                "subject": subject, "run_a": left_run, "run_b": right_run,
                "peak_a_hz": float(left), "peak_b_hz": float(right),
                "abs_difference_hz": abs(float(left) - float(right)),
            })
    return rows


def subject_summary(values_by_subject: RunPeaks) -> list[dict[str, float | int | str]]:
    """Keep every eligible subject while leaving undefined stability metrics blank."""
    pairwise_by_subject: dict[int, list[float]] = {}
    for row in pairwise_rows(values_by_subject):
        pairwise_by_subject.setdefault(int(row["subject"]), []).append(float(row["abs_difference_hz"]))
    output: list[dict[str, float | int | str]] = []
    for subject in sorted(values_by_subject):
        values = np.array([value for _, value in sorted(values_by_subject[subject])], dtype=float)
        count = int(values.size)
        pairs = pairwise_by_subject.get(subject, [])
        output.append({
            "subject": subject, "valid_run_peak_count": count,
            "min_peak_hz": float(values.min()) if count else "",
            "max_peak_hz": float(values.max()) if count else "",
            "range_hz": float(np.ptp(values)) if count >= 2 else "",
            "mean_peak_hz": float(values.mean()) if count else "",
            "median_peak_hz": float(np.median(values)) if count else "",
            "std_peak_hz": float(values.std(ddof=1)) if count >= 2 else "",
            "mad_peak_hz": float(np.median(np.abs(values - np.median(values)))) if count >= 2 else "",
            "pairwise_difference_count": len(pairs),
            "median_pairwise_abs_difference_hz": float(np.median(pairs)) if pairs else "",
        })
    return output


def availability_counts(values_by_subject: RunPeaks, run_count: int) -> dict[str, int]:
    """Count subjects by number of available run peaks, including zero."""
    return {str(count): sum(len(values_by_subject[subject]) == count for subject in values_by_subject)
            for count in range(run_count + 1)}


def agreement_proportions(
    rows: Sequence[Mapping[str, float | int]], thresholds_hz: Sequence[float],
    grid_tolerance_hz: float = 1e-9,
) -> dict[str, float]:
    """Report exact-grid and tolerance-safe agreement proportions."""
    differences = np.array([float(row["abs_difference_hz"]) for row in rows], dtype=float)
    if not differences.size:
        return {"same_grid_bin": float("nan"), **{str(x): float("nan") for x in thresholds_hz}}
    result = {"same_grid_bin": float(np.mean(np.isclose(differences, 0.0, atol=grid_tolerance_hz)))}
    result.update({str(t): float(np.mean(differences <= t + grid_tolerance_hz)) for t in thresholds_hz})
    return result


def bootstrap_median_pair_difference_samples(
    rows: Sequence[Mapping[str, float | int]], resamples: int, seed: int
) -> np.ndarray:
    """Resample participants, retaining all each selected participant's pairs."""
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive.")
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["subject"]), []).append(float(row["abs_difference_hz"]))
    if not grouped:
        raise ValueError("At least one participant with a peak pair is required.")
    subjects = np.array(sorted(grouped), dtype=int)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = rng.choice(subjects, len(subjects), replace=True)
        pooled = [difference for subject in selected for difference in grouped[int(subject)]]
        samples[index] = float(np.median(pooled))
    return samples


def bootstrap_median_pair_difference(
    rows: Sequence[Mapping[str, float | int]], resamples: int, seed: int
) -> tuple[float, float]:
    """Return the deterministic participant-bootstrap percentile 95% interval."""
    samples = bootstrap_median_pair_difference_samples(rows, resamples, seed)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
