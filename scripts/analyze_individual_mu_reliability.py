"""Generate the frozen v1.1 independent-rest individual-mu reliability package."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_project.epoching import extract_run_epochs, load_epoching_config
from eeg_project.individual_frequency import load_individual_frequency_config
from eeg_project.individual_mu_reliability import (
    agreement_proportions,
    availability_counts,
    bootstrap_median_pair_difference,
    estimate_run_rest_peak,
    pairwise_rows,
    subject_summary,
)
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY, load_preprocessing_config


ASSETS = ROOT / "docs" / "assets"
PREFIX = "individual_mu_reliability_"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write schema-free empty artifact: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson interval for a binomial proportion."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson inputs must satisfy 0 <= successes <= total and total > 0.")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / denominator
    return max(0.0, float(center - half_width)), min(1.0, float(center + half_width))


def numeric_summary(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    if not data.size:
        return {}
    return {
        "count": int(data.size), "median_hz": float(np.median(data)),
        "iqr_hz": [float(np.quantile(data, 0.25)), float(np.quantile(data, 0.75))],
        "mean_hz": float(np.mean(data)), "sd_hz": float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
        "p90_hz": float(np.quantile(data, 0.90)), "min_hz": float(np.min(data)), "max_hz": float(np.max(data)),
    }


def make_figures(run_rows: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Make deterministic report figures strictly from the v1.1 tables."""
    primary = [row for row in run_rows if float(row["prominence_threshold_db"]) == 1.0 and row["valid"] is True]
    peaks = np.array([float(row["peak_frequency_hz"]) for row in primary])
    differences = np.array([float(row["abs_difference_hz"]) for row in pairs])
    plt.style.use("default")
    fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    axis.hist(peaks, bins=np.arange(6.875, 14.126, 0.25), color="#2166ac", edgecolor="white")
    axis.set(xlabel="Independent run-level peak (Hz)", ylabel="Run estimates", xlim=(7, 14))
    fig.savefig(ASSETS / f"{PREFIX}peak_distribution.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    axis.hist(differences, bins=np.arange(-0.125, 5.876, 0.25), color="#b2182b", edgecolor="white")
    axis.set(xlabel="Within-subject absolute peak difference (Hz)", ylabel="Run pairs", xlim=(0, 5.5))
    fig.savefig(ASSETS / f"{PREFIX}pairwise_disagreement.png", dpi=180)
    plt.close(fig)
    agreement = summary["agreement"]
    labels = ["same bin", "≤0.25", "≤0.50", "≤1.00"]
    values = [agreement["same_grid_bin"], agreement["0.25"], agreement["0.5"], agreement["1.0"]]
    fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    axis.bar(labels, values, color="#4d9221")
    axis.set(ylabel="Proportion of independent run pairs", ylim=(0, 1))
    fig.savefig(ASSETS / f"{PREFIX}agreement.png", dpi=180)
    plt.close(fig)


def main() -> None:
    config_path = ROOT / "config" / "individual_mu_reliability.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent_v1_0"]["config"]
    if sha256(parent_path) != config["parent_v1_0"]["sha256"]:
        raise RuntimeError("Frozen v1.0 individual-mu configuration has drifted.")
    individual = load_individual_frequency_config()
    preprocessing = load_preprocessing_config()
    epoching = load_epoching_config()
    subjects = sorted(set(individual.development_subjects) | set(individual.evaluation_subjects))
    thresholds = tuple(float(value) for value in config["robustness"]["prominence_db"])
    records: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    for subject in subjects:
        for run in individual.runs:
            result = extract_run_epochs(subject, run, preprocessing, epoching, data_directory=DEFAULT_DATA_DIRECTORY)
            rest_data = result.rest_epochs.get_data(copy=True)
            raw_hashes[str(result.preprocessing.source_path)] = result.preprocessing.source_sha256
            for threshold in thresholds:
                estimate = estimate_run_rest_peak(
                    rest_data, result.rest_epochs.ch_names, float(result.rest_epochs.info["sfreq"]),
                    individual.central_channels, individual.welch_window_seconds,
                    tuple(config["primary"]["search_range_hz"]), threshold,
                )
                records.append({
                    "subject": subject, "run": run, "prominence_threshold_db": threshold,
                    "peak_frequency_hz": "" if estimate.peak_frequency_hz is None else estimate.peak_frequency_hz,
                    "peak_prominence_db": estimate.peak_prominence_db, "valid": estimate.method_available,
                    "invalid_reason": estimate.failure_reason, "peak_quality": estimate.peak_quality,
                    "rest_epoch_count": rest_data.shape[0], "estimator": "v1_0_exact_single_run_rest",
                })
    ASSETS.mkdir(parents=True, exist_ok=True)
    write_csv(records, ASSETS / f"{PREFIX}run_peaks.csv")
    primary = [row for row in records if float(row["prominence_threshold_db"]) == float(config["primary"]["prominence_db"])]
    values = {subject: [] for subject in subjects}
    for row in primary:
        if row["valid"]:
            values[int(row["subject"])].append((int(row["run"]), float(row["peak_frequency_hz"])))
    pairs = pairwise_rows(values)
    summaries = subject_summary(values)
    write_csv(pairs, ASSETS / f"{PREFIX}pairwise_differences.csv")
    write_csv(summaries, ASSETS / f"{PREFIX}subject_summary.csv")
    availability = availability_counts(values, len(individual.runs))
    peak_values = [float(row["peak_frequency_hz"]) for row in primary if row["valid"]]
    differences = [float(row["abs_difference_hz"]) for row in pairs]
    fixed_low, fixed_high = tuple(float(value) for value in config["primary"]["fixed_comparator_range_hz"])
    summary: dict[str, Any] = {
        "schema_version": 2, "study": "v1.1_RELIABILITY", "classification": "RELIABILITY",
        "eligible_subject_count": len(subjects), "runs": list(individual.runs),
        "availability_counts": availability,
        "availability_percent": {key: value / len(subjects) for key, value in availability.items()},
        "availability_wilson_95": {key: list(wilson_interval(value, len(subjects))) for key, value in availability.items()},
        "per_run_available": {str(run): sum(row["valid"] for row in primary if row["run"] == run) for run in individual.runs},
        "pair_count": len(pairs), "pairwise": numeric_summary(differences),
        "agreement": agreement_proportions(pairs, config["primary"]["agreement_hz"]),
        "bootstrap_median_95": list(bootstrap_median_pair_difference(pairs, config["primary"]["bootstrap_resamples"], config["primary"]["bootstrap_seed"])),
        "peak_frequency_distribution": {**numeric_summary(peak_values), "fixed_12_13_hz": {
            "below": sum(value < fixed_low for value in peak_values), "inside": sum(fixed_low <= value <= fixed_high for value in peak_values), "above": sum(value > fixed_high for value in peak_values),
        }},
        "config_sha256": sha256(config_path), "parent_v1_0_config_sha256": config["parent_v1_0"]["sha256"],
    }
    sensitivity: list[dict[str, Any]] = []
    for threshold in thresholds:
        rows = [row for row in records if float(row["prominence_threshold_db"]) == threshold]
        threshold_values = {subject: [(int(row["run"]), float(row["peak_frequency_hz"])) for row in rows if row["subject"] == subject and row["valid"]] for subject in subjects}
        threshold_pairs = pairwise_rows(threshold_values)
        threshold_peaks = [float(row["peak_frequency_hz"]) for row in rows if row["valid"]]
        agreement = agreement_proportions(threshold_pairs, (0.5, 1.0))
        sensitivity.append({"classification": "ROBUSTNESS_DIAGNOSTIC", "prominence_threshold_db": threshold,
            "valid_run_peaks": len(threshold_peaks), "valid_subject_run_fraction": len(threshold_peaks) / (len(subjects) * len(individual.runs)),
            "peak_frequency": numeric_summary(threshold_peaks), "pairwise": numeric_summary([float(row["abs_difference_hz"]) for row in threshold_pairs]),
            "agreement_le_0_5": agreement["0.5"], "agreement_le_1_0": agreement["1.0"]})
    summary["prominence_sensitivity"] = sensitivity
    manifest = {
        "schema_version": 1, "study": "v1.1_RELIABILITY", "generation_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": summary["config_sha256"], "parent_v1_0_config_sha256": config["parent_v1_0"]["sha256"],
        "implementation_sha256": {
            "eeg_project/individual_frequency.py": sha256(ROOT / "eeg_project/individual_frequency.py"),
            "eeg_project/individual_mu_reliability.py": sha256(ROOT / "eeg_project/individual_mu_reliability.py"),
            "scripts/analyze_individual_mu_reliability.py": sha256(ROOT / "scripts/analyze_individual_mu_reliability.py"),
        },
        "full_cohort_raw_manifest": {"path": "docs/assets/subjects01-109_full_replication_metadata.json", "sha256": sha256(ASSETS / "subjects01-109_full_replication_metadata.json")},
        "raw_source_hashes": dict(sorted(raw_hashes.items())), "python": platform.python_version(),
        "packages": {"mne": mne.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": plt.matplotlib.__version__},
        "bootstrap": {"resamples": config["primary"]["bootstrap_resamples"], "seed": config["primary"]["bootstrap_seed"]},
    }
    write_csv(sensitivity, ASSETS / f"{PREFIX}prominence_sensitivity.csv")
    (ASSETS / f"{PREFIX}summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (ASSETS / f"{PREFIX}provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    make_figures(records, pairs, summary)


if __name__ == "__main__":
    main()
