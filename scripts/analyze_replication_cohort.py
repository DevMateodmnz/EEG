"""Audit and analyze the frozen subjects 1-20 EEG replication cohort."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.cohort import (  # noqa: E402
    DEFAULT_REPLICATION_CONFIG_PATH,
    audit_cohort,
    build_primary_replication_rows,
    central_peak_summary,
    cohort_feature_summaries,
    configuration_hashes,
    frozen_config_fingerprint,
    load_replication_cohort_config,
    subject_eligibility_rows,
    summarize_subject_features,
    summarize_subject_runs,
)
from eeg_project.epoching import (  # noqa: E402
    calculate_trial_quality,
    extract_run_epochs,
    load_epoching_config,
)
from eeg_project.preprocessing import (  # noqa: E402
    DEFAULT_DATA_DIRECTORY,
    load_preprocessing_config,
)
from eeg_project.time_frequency import (  # noqa: E402
    SpectralEpochDataset,
    assemble_spectral_epoch_dataset,
    compute_morlet_power,
    load_event_related_spectral_config,
    trial_band_measurements,
)
from eeg_project.trial_quality import (  # noqa: E402
    add_normalization_diagnostics,
    classify_metric_candidates,
    load_trial_quality_config,
    paired_side_quality_metrics,
    spearman_rho,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit subjects 1-20, then run frozen subject-level replication "
            "measurements without CSP, classification, or outcome-based exclusion."
        )
    )
    parser.add_argument(
        "--cohort-config", type=Path, default=DEFAULT_REPLICATION_CONFIG_PATH
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Download/load and audit the predefined files without spectral outcomes.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.expanduser().resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.expanduser().resolve())


def write_csv(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {output_path}.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def subject_qc_summary(
    subject: int,
    role: str,
    dataset,
    run_results,
    task_quality_rows,
    rest_candidate_count: int,
    audit_rows,
) -> dict[str, object]:
    condition_counts = Counter(pair.task.semantic_condition for pair in dataset.pairs)
    channel_names = sorted(
        {
            channel
            for row in audit_rows
            if int(row["subject"]) == subject
            for channel in str(row["channel_candidate_names"]).split("/")
            if channel
        }
    )
    subject_audit = [row for row in audit_rows if int(row["subject"]) == subject]
    return {
        "subject": subject,
        "subject_role": role,
        "runs_available": "/".join(str(run) for run in sorted(run_results)),
        "run_count": len(run_results),
        "task_trials": len(dataset.pairs),
        "fists_trials": condition_counts["both_fists_imagery"],
        "feet_trials": condition_counts["both_feet_imagery"],
        "qc_task_candidates": sum(
            row["quality_status"] == "statistical candidate"
            for row in task_quality_rows
        ),
        "qc_rest_candidates": rest_candidate_count,
        "channel_candidate_run_records": sum(
            int(row["channel_candidate_count"]) for row in subject_audit
        ),
        "unique_channel_candidate_count": len(channel_names),
        "unique_channel_candidate_names": "/".join(channel_names),
        "confirmed_channel_failures": sum(
            int(row["confirmed_flat_or_nonfinite_channel_count"])
            for row in subject_audit
        ),
        "deterministic_invalid_task_epochs": sum(
            not record.valid
            for result in run_results.values()
            for record in result.task_records
        ),
        "deterministic_invalid_rest_epochs": sum(
            not record.valid
            for result in run_results.values()
            for record in result.rest_records
        ),
        "permanent_artifact_trial_exclusions": 0,
        "processing_status": "processed_frozen_pipeline",
        "outcome_used_for_inclusion": False,
    }


def process_subject(
    subject: int,
    role: str,
    cohort_config,
    preprocessing_config,
    epoching_config,
    spectral_config,
    quality_config,
    data_directory: Path,
    audit_rows,
) -> dict[str, object]:
    run_results = {
        run: extract_run_epochs(
            subject,
            run,
            preprocessing_config,
            epoching_config,
            data_directory=data_directory,
        )
        for run in cohort_config.runs
    }
    dataset = assemble_spectral_epoch_dataset(run_results, cohort_config.runs)
    master_snapshot = dataset.task_data_volts.copy()
    task_quality_rows = calculate_trial_quality(
        dataset.task_data_volts,
        dataset.task_times,
        dataset.channel_names,
        [pair.task for pair in dataset.pairs],
        epoching_config.trial_quality,
    )
    quality_by_identity = {
        (int(row["run"]), int(row["run_trial_index"])): row
        for row in task_quality_rows
    }

    sensor_indices = [dataset.channel_names.index(channel) for channel in ("C3", "Cz", "C4")]
    sensor_dataset = SpectralEpochDataset(
        task_data_volts=dataset.task_data_volts[:, sensor_indices].copy(),
        rest_data_volts=dataset.rest_data_volts[:, sensor_indices].copy(),
        task_times=dataset.task_times.copy(),
        rest_times=dataset.rest_times.copy(),
        channel_names=["C3", "Cz", "C4"],
        sampling_frequency_hz=dataset.sampling_frequency_hz,
        pairs=dataset.pairs.copy(),
    )
    task_power = compute_morlet_power(
        sensor_dataset.task_data_volts,
        sensor_dataset.task_times,
        sensor_dataset.sampling_frequency_hz,
        spectral_config,
    )
    rest_power = compute_morlet_power(
        sensor_dataset.rest_data_volts,
        sensor_dataset.rest_times,
        sensor_dataset.sampling_frequency_hz,
        spectral_config,
    )
    trial_rows = trial_band_measurements(
        task_power,
        rest_power,
        sensor_dataset,
        spectral_config,
        quality_by_identity,
    )
    diagnostic_rows = add_normalization_diagnostics(
        trial_rows, quality_config.robust_z_threshold
    )
    for row in diagnostic_rows:
        row["subject_role"] = role
        row["source_file"] = display_path(Path(str(row["source_file"])))

    rest_metrics = paired_side_quality_metrics(
        dataset.rest_data_volts,
        dataset.rest_times,
        dataset.channel_names,
        epoching_config.trial_quality.frontal_channels,
        (0.0, 4.0),
    )
    _, _, rest_candidates = classify_metric_candidates(
        rest_metrics, quality_config.robust_z_threshold
    )
    feature_rows = summarize_subject_features(
        diagnostic_rows,
        subject,
        role,
        cohort_config.all_features,
        quality_config.trimmed_mean_fraction_each_tail,
    )
    run_rows = summarize_subject_runs(
        diagnostic_rows, subject, role, cohort_config.all_features
    )
    peak_row = central_peak_summary(
        subject,
        role,
        [
            (
                run,
                result.preprocessing.referenced.get_data(),
                result.preprocessing.referenced.ch_names,
                float(result.preprocessing.referenced.info["sfreq"]),
            )
            for run, result in sorted(run_results.items())
        ],
        cohort_config.central_peak_exploration,
    )
    qc_row = subject_qc_summary(
        subject,
        role,
        dataset,
        run_results,
        task_quality_rows,
        int(rest_candidates.sum()),
        audit_rows,
    )
    if not np.array_equal(dataset.task_data_volts, master_snapshot):
        raise RuntimeError(f"Subject {subject}: sensitivity analysis mutated master epochs.")
    if not all(row["primary_retained"] for row in diagnostic_rows):
        raise RuntimeError(f"Subject {subject}: a primary spectral trial was removed.")
    return {
        "trial_rows": diagnostic_rows,
        "feature_rows": feature_rows,
        "run_rows": run_rows,
        "peak_row": peak_row,
        "qc_row": qc_row,
        "task_tfr_shape": list(task_power.power_v2.shape),
        "rest_tfr_shape": list(rest_power.power_v2.shape),
    }


def peak_relationship_rows(peak_rows, subject_feature_rows, cohort_config):
    output: list[dict[str, object]] = []
    for condition, channel, band in cohort_config.primary_replication_features:
        features = {
            int(row["subject"]): float(row["median_change_percent"])
            for row in subject_feature_rows
            if row["subject_role"] == "replication"
            and row["semantic_condition"] == condition
            and row["channel"] == channel
            and row["frequency_band"] == band
        }
        eligible_peaks = [
            row
            for row in peak_rows
            if row["subject_role"] == "replication" and row["well_defined_peak"]
        ]
        frequencies = np.array([float(row["peak_frequency_hz"]) for row in eligible_peaks])
        distances = np.abs(frequencies - 12.5)
        values = np.array([features[int(row["subject"])] for row in eligible_peaks])
        output.append(
            {
                "semantic_condition": condition,
                "channel": channel,
                "frequency_band": band,
                "analysis_role": "exploratory_peak_relationship",
                "replication_subjects_with_well_defined_peak": len(eligible_peaks),
                "spearman_peak_frequency_vs_fixed_band_change": (
                    spearman_rho(frequencies, values) if len(values) >= 2 else ""
                ),
                "spearman_distance_from_12_5_hz_vs_fixed_band_change": (
                    spearman_rho(distances, values) if len(values) >= 2 else ""
                ),
                "primary_band_redefined": False,
            }
        )
    return output


def plot_primary_subjects(primary_rows, cohort_rows, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for axis, channel in zip(axes, ("C4", "C3"), strict=True):
        selected = sorted(
            [row for row in primary_rows if row["channel"] == channel],
            key=lambda row: int(row["subject"]),
        )
        development = [row for row in selected if row["subject_role"] == "development"]
        replication = [row for row in selected if row["subject_role"] == "replication"]
        axis.axhline(0, color="#555555", linewidth=0.9)
        axis.scatter(
            [int(row["subject"]) for row in replication],
            [float(row["median_change_percent"]) for row in replication],
            color="#4c78a8", s=48, label="replication subjects 2-20",
        )
        axis.scatter(
            [int(row["subject"]) for row in development],
            [float(row["median_change_percent"]) for row in development],
            color="#f58518", marker="*", s=150, label="development subject 1",
        )
        cohort = next(
            row for row in cohort_rows
            if row["channel"] == channel and row["frequency_band"] == "central_12_13"
        )
        axis.axhline(
            float(cohort["median_across_subject_medians_percent"]),
            color="#4c78a8", linestyle="--", linewidth=1,
            label="replication median",
        )
        axis.set_ylabel(f"{channel} fists 12-13 Hz\nsubject median (%)")
        axis.grid(alpha=0.17)
        axis.legend(frameon=False, ncol=3, fontsize=9)
    axes[-1].set_xlabel("Subject ID")
    axes[-1].set_xticks(range(1, 21))
    figure.suptitle("Frozen primary replication measurements — one point per subject", fontsize=14)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_c3_c4_pair(primary_rows, output_path: Path) -> None:
    lookup = {
        (int(row["subject"]), str(row["channel"])): float(row["median_change_percent"])
        for row in primary_rows
    }
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    replication = sorted(
        {
            int(row["subject"])
            for row in primary_rows
            if row["subject_role"] == "replication"
        }
    )
    axis.scatter(
        [lookup[(subject, "C3")] for subject in replication],
        [lookup[(subject, "C4")] for subject in replication],
        color="#4c78a8", s=55, label="replication",
    )
    axis.scatter(
        [lookup[(1, "C3")]], [lookup[(1, "C4")]],
        color="#f58518", marker="*", s=170, label="development subject 1",
    )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.axvline(0, color="#555555", linewidth=0.8)
    plotted_values = [
        lookup[(subject, channel)]
        for subject in (1, *replication)
        for channel in ("C3", "C4")
    ]
    diagonal_lower = min(0.0, *plotted_values) - 8.0
    diagonal_upper = max(0.0, *plotted_values) + 8.0
    axis.plot(
        [diagonal_lower, diagonal_upper],
        [diagonal_lower, diagonal_upper],
        color="#999999", linestyle=":", linewidth=0.8,
    )
    for subject in replication:
        axis.annotate(str(subject), (lookup[(subject, "C3")], lookup[(subject, "C4")]), fontsize=7)
    axis.set(
        xlabel="C3 fists 12-13 Hz subject median (%)",
        ylabel="C4 fists 12-13 Hz subject median (%)",
        title="C3 versus C4 subject-level replication",
    )
    axis.grid(alpha=0.15)
    axis.legend(frameon=False)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_narrow_vs_mu(subject_rows, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, channel in zip(axes, ("C3", "C4"), strict=True):
        lookup = {
            (int(row["subject"]), str(row["frequency_band"])): float(row["median_change_percent"])
            for row in subject_rows
            if row["semantic_condition"] == "both_fists_imagery"
            and row["channel"] == channel
            and row["frequency_band"] in {"central_12_13", "mu"}
        }
        replication_subjects = sorted(
            subject
            for subject, band in lookup
            if band == "central_12_13" and subject != 1
        )
        axis.scatter(
            [lookup[(subject, "central_12_13")] for subject in replication_subjects],
            [lookup[(subject, "mu")] for subject in replication_subjects],
            color="#4c78a8", alpha=0.8,
        )
        axis.scatter(
            [lookup[(1, "central_12_13")]], [lookup[(1, "mu")]],
            color="#f58518", marker="*", s=150,
        )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set(
            xlabel="Fixed 12-13 Hz median (%)",
            ylabel="Fixed 8-13 Hz mu median (%)",
            title=channel,
        )
        axis.grid(alpha=0.15)
    figure.suptitle("Narrow versus broad fixed-band fists responses")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_feet_variability(subject_rows, output_path: Path) -> None:
    measurements = (("C3", "central_12_13"), ("C4", "central_12_13"), ("C3", "mu"), ("C4", "mu"), ("Cz", "beta"))
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for index, (channel, band) in enumerate(measurements):
        selected = sorted(
            [
                row for row in subject_rows
                if row["subject_role"] == "replication"
                and row["semantic_condition"] == "both_feet_imagery"
                and row["channel"] == channel
                and row["frequency_band"] == band
            ],
            key=lambda row: int(row["subject"]),
        )
        offsets = np.linspace(-0.22, 0.22, len(selected))
        axis.scatter(
            index + offsets,
            [float(row["median_change_percent"]) for row in selected],
            color="#f58518", alpha=0.75, s=35,
        )
        axis.scatter(
            [index], [np.median([float(row["median_change_percent"]) for row in selected])],
            color="black", marker="_", s=180,
        )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks(range(len(measurements)), [f"{channel} {band}" for channel, band in measurements], rotation=20)
    axis.set_ylabel("Subject-level feet median change (%)")
    axis.set_title("Feet responses across replication subjects — points are people")
    axis.grid(axis="y", alpha=0.15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_peak_exploration(peak_rows, primary_rows, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    replication_peaks = [row for row in peak_rows if row["subject_role"] == "replication"]
    categories = ["8", "9", "10", "11", "12", "13", "poorly_defined"]
    counts = Counter(str(row["peak_category"]) for row in replication_peaks)
    axes[0].bar(categories, [counts[category] for category in categories], color="#72b7b2")
    axes[0].set(xlabel="Central 8-13 Hz peak category", ylabel="Replication subjects", title="Exploratory fixed-grid peaks")
    c4 = {
        int(row["subject"]): float(row["median_change_percent"])
        for row in primary_rows if row["channel"] == "C4"
    }
    well = [row for row in replication_peaks if row["well_defined_peak"]]
    axes[1].scatter(
        [abs(float(row["peak_frequency_hz"]) - 12.5) for row in well],
        [c4[int(row["subject"])] for row in well],
        color="#4c78a8", s=50,
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set(
        xlabel="Distance from 12.5 Hz (Hz)",
        ylabel="C4 fixed 12-13 Hz fists median (%)",
        title="Exploratory peak proximity; primary band unchanged",
    )
    for axis in axes:
        axis.grid(alpha=0.15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    cohort_config = load_replication_cohort_config(arguments.cohort_config)
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    data_directory = arguments.data_dir.expanduser().resolve()
    prefix = "subjects01-20_replication"
    audit_path = output_directory / f"{prefix}_cohort_audit.csv"
    eligibility_path = output_directory / f"{prefix}_subject_eligibility.csv"

    audit_rows = audit_cohort(cohort_config, data_directory=data_directory)
    for row in audit_rows:
        if row["source_file"]:
            row["source_file"] = display_path(Path(str(row["source_file"])))
    eligibility_rows = subject_eligibility_rows(audit_rows, cohort_config)
    write_csv(audit_rows, audit_path)
    write_csv(eligibility_rows, eligibility_path)
    compatible_runs = sum(bool(row["protocol_compatible"]) for row in audit_rows)
    eligible_subjects = [
        int(row["subject"])
        for row in eligibility_rows
        if row["technical_eligibility"] == "eligible"
    ]
    print(
        f"Technical audit: {compatible_runs}/{len(audit_rows)} compatible runs; "
        f"{len(eligible_subjects)}/{len(cohort_config.all_subjects)} eligible subjects."
    )
    if arguments.audit_only:
        print(f"Audit: {display_path(audit_path)}")
        return

    frozen_paths = {
        name: PROJECT_ROOT / path
        for name, path in cohort_config.frozen_configuration_paths.items()
    }
    preprocessing_config = load_preprocessing_config(frozen_paths["preprocessing"])
    epoching_config = load_epoching_config(frozen_paths["epoching"])
    spectral_config = load_event_related_spectral_config(
        frozen_paths["event_related_spectral"]
    )
    quality_config = load_trial_quality_config(frozen_paths["trial_quality"])

    all_trial_rows: list[dict[str, object]] = []
    all_subject_rows: list[dict[str, object]] = []
    all_run_rows: list[dict[str, object]] = []
    peak_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    shapes: dict[str, dict[str, list[int]]] = {}
    for subject in eligible_subjects:
        role = "development" if subject == cohort_config.development_subject else "replication"
        print(f"Processing subject {subject:02d}/20 ({role})...")
        result = process_subject(
            subject,
            role,
            cohort_config,
            preprocessing_config,
            epoching_config,
            spectral_config,
            quality_config,
            data_directory,
            audit_rows,
        )
        all_trial_rows.extend(result["trial_rows"])
        all_subject_rows.extend(result["feature_rows"])
        all_run_rows.extend(result["run_rows"])
        peak_rows.append(result["peak_row"])
        qc_rows.append(result["qc_row"])
        shapes[str(subject)] = {
            "task_tfr": result["task_tfr_shape"],
            "rest_tfr": result["rest_tfr_shape"],
        }

    primary_rows = build_primary_replication_rows(
        all_subject_rows, all_run_rows, cohort_config
    )
    cohort_rows = cohort_feature_summaries(
        all_subject_rows, primary_rows, cohort_config
    )
    relationship_rows = peak_relationship_rows(
        peak_rows, all_subject_rows, cohort_config
    )

    subject_one = {
        (row["channel"], row["frequency_band"]): float(row["median_change_percent"])
        for row in all_subject_rows
        if int(row["subject"]) == 1 and row["semantic_condition"] == "both_fists_imagery"
    }
    if not np.isclose(subject_one[("C3", "central_12_13")], -55.57490648782269):
        raise RuntimeError("Frozen subject-1 C3 result changed.")
    if not np.isclose(subject_one[("C4", "central_12_13")], -52.28373306890746):
        raise RuntimeError("Frozen subject-1 C4 result changed.")
    requested_replication = set(cohort_config.replication_subjects)
    accounted_replication = {
        int(row["subject"])
        for row in eligibility_rows
        if row["subject_role"] == "replication"
    }
    if requested_replication != accounted_replication:
        raise RuntimeError("Not every requested replication subject is accounted for.")

    paths = {
        "audit": audit_path,
        "eligibility": eligibility_path,
        "trial": output_directory / f"{prefix}_trial_spectral_measurements.csv",
        "subject": output_directory / f"{prefix}_subject_feature_summary.csv",
        "run": output_directory / f"{prefix}_subject_run_summary.csv",
        "qc": output_directory / f"{prefix}_subject_qc_summary.csv",
        "primary": output_directory / f"{prefix}_replication_primary_features.csv",
        "cohort": output_directory / f"{prefix}_cohort_feature_summary.csv",
        "peaks": output_directory / f"{prefix}_central_peak_summary.csv",
        "relationship": output_directory / f"{prefix}_peak_feature_relationship.csv",
        "metadata": output_directory / f"{prefix}_metadata.json",
        "primary_figure": output_directory / f"{prefix}_primary_subject_results.png",
        "paired_figure": output_directory / f"{prefix}_c3_c4_subject_scatter.png",
        "band_figure": output_directory / f"{prefix}_narrow_vs_mu.png",
        "feet_figure": output_directory / f"{prefix}_feet_variability.png",
        "peak_figure": output_directory / f"{prefix}_central_peak_exploration.png",
    }
    for key, rows in (
        ("trial", all_trial_rows),
        ("subject", all_subject_rows),
        ("run", all_run_rows),
        ("qc", qc_rows),
        ("primary", primary_rows),
        ("cohort", cohort_rows),
        ("peaks", peak_rows),
        ("relationship", relationship_rows),
    ):
        write_csv(rows, paths[key])
    plot_primary_subjects(primary_rows, cohort_rows, paths["primary_figure"])
    plot_c3_c4_pair(primary_rows, paths["paired_figure"])
    plot_narrow_vs_mu(all_subject_rows, paths["band_figure"])
    plot_feet_variability(all_subject_rows, paths["feet_figure"])
    plot_peak_exploration(peak_rows, primary_rows, paths["peak_figure"])

    metadata = {
        "schema_version": 1,
        "requested_subjects": list(cohort_config.all_subjects),
        "development_subject": cohort_config.development_subject,
        "requested_replication_subjects": list(cohort_config.replication_subjects),
        "technically_eligible_subjects": eligible_subjects,
        "technically_ineligible_subjects": [
            int(row["subject"])
            for row in eligibility_rows
            if row["technical_eligibility"] != "eligible"
        ],
        "eligible_replication_subject_count": sum(
            subject in cohort_config.replication_subjects for subject in eligible_subjects
        ),
        "total_analyzed_task_trials": sum(int(row["task_trials"]) for row in qc_rows),
        "replication_analyzed_task_trials": sum(
            int(row["task_trials"]) for row in qc_rows if row["subject_role"] == "replication"
        ),
        "configuration_hashes": configuration_hashes(cohort_config),
        "frozen_config_fingerprint": frozen_config_fingerprint(
            cohort_config, arguments.cohort_config
        ),
        "subject_tfr_shapes": shapes,
        "full_tfr_arrays_persisted": False,
        "cache_policy": cohort_config.cache_policy,
        "outcome_based_subject_exclusions": 0,
        "classification_performed": False,
        "csp_performed": False,
        "source_hashes": {
            f"subject{int(row['subject']):02d}_run{int(row['run']):02d}": row["source_sha256"]
            for row in audit_rows if row["source_sha256"]
        },
        "artifacts": {
            key: display_path(path) for key, path in paths.items() if key != "metadata"
        },
    }
    with paths["metadata"].open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    print("\nPrimary replication results (subjects 2-20 only):")
    for row in cohort_rows:
        if row["analysis_role"] != "primary":
            continue
        print(
            f"  {row['channel']} fists 12-13 Hz: "
            f"{row['negative_subject_count']}/{row['replication_subject_count']} negative; "
            f"median {float(row['median_across_subject_medians_percent']):.2f}%; "
            f"{row['replication_category']}"
        )
    print(f"Metadata: {display_path(paths['metadata'])}")


if __name__ == "__main__":
    main()
