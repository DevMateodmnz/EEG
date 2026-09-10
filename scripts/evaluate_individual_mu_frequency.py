"""Evaluate the frozen individualized-frequency method on held-out odd-ID subjects."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
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

from eeg_project.epoching import extract_run_epochs  # noqa: E402
from eeg_project.individual_frequency import (  # noqa: E402
    DEFAULT_INDIVIDUAL_FREQUENCY_CONFIG_PATH,
    estimate_central_peak,
    individualized_frequency_vector,
    load_individual_frequency_config,
    rest_roi_log_welch,
    training_runs_for_held_out,
)
from eeg_project.individual_frequency_evaluation import (  # noqa: E402
    compute_custom_morlet_power,
    individualized_trial_measurements,
    paired_group_summaries,
    peak_reliability_category,
    summarize_subject_methods,
)
from eeg_project.epoching import load_epoching_config  # noqa: E402
from eeg_project.preprocessing import (  # noqa: E402
    DEFAULT_DATA_DIRECTORY,
    load_preprocessing_config,
)
from eeg_project.time_frequency import (  # noqa: E402
    SpectralEpochDataset,
    assemble_spectral_epoch_dataset,
    load_event_related_spectral_config,
)
from eeg_project.trial_quality import (  # noqa: E402
    add_normalization_diagnostics,
    load_trial_quality_config,
    spearman_rho,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
REPORT_PREFIX = "individual_mu_heldout_evaluation"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final-frozen rest-only leave-one-run-out individualized "
            "central frequency method on odd-ID held-out participants."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_INDIVIDUAL_FREQUENCY_CONFIG_PATH
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty output: {path}.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixed_trial_lookup(evaluation_subjects: Sequence[int]) -> dict[tuple[int, int, int, str, str], dict[str, str]]:
    paths = (
        PROJECT_ROOT / "docs/assets/subjects01-20_replication_trial_spectral_measurements.csv",
        PROJECT_ROOT / "docs/assets/subjects01-109_full_replication_replication2_trial_spectral_measurements.csv",
    )
    allowed = set(evaluation_subjects)
    lookup: dict[tuple[int, int, int, str, str], dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            if (
                int(row["subject"]) not in allowed
                or row["semantic_condition"] != "both_fists_imagery"
                or row["channel"] not in ("C4", "C3")
                or row["frequency_band"] not in ("central_12_13", "mu")
            ):
                continue
            key = (
                int(row["subject"]),
                int(row["run"]),
                int(row["run_trial_index"]),
                row["channel"],
                row["frequency_band"],
            )
            if key in lookup:
                raise RuntimeError(f"Duplicate fixed trial row: {key}.")
            lookup[key] = row
    return lookup


def peak_table_row(
    subject: int,
    estimate_type: str,
    held_out_run: int | str,
    training_runs: Sequence[int],
    estimate,
    rest_epoch_count: int,
) -> dict[str, object]:
    return {
        "subject": subject,
        "study_role": "held_out_method_evaluation",
        "estimate_type": estimate_type,
        "held_out_run": held_out_run,
        "training_runs": "/".join(str(run) for run in training_runs),
        "peak_frequency_hz": "" if estimate.peak_frequency_hz is None else estimate.peak_frequency_hz,
        "candidate_frequency_hz": estimate.candidate_frequency_hz,
        "peak_prominence_db": estimate.peak_prominence_db,
        "peak_width_hz": "" if estimate.peak_width_hz is None else estimate.peak_width_hz,
        "peak_quality": estimate.peak_quality,
        "method_available": estimate.method_available,
        "failure_reason": estimate.failure_reason,
        "search_low_hz": 7.0,
        "search_high_hz": 14.0,
        "minimum_prominence_db": 1.0,
        "band_low_hz": "" if estimate.peak_frequency_hz is None else estimate.peak_frequency_hz - 1.0,
        "band_high_hz": "" if estimate.peak_frequency_hz is None else estimate.peak_frequency_hz + 1.0,
        "rest_epoch_count": rest_epoch_count,
        "central_roi": "C3/Cz/C4_mean_log_psd",
        "frequency_estimation_data": "T0_rest_only",
        "task_outcome_used_for_peak": False,
    }


def summarize_peak_subjects(
    peak_rows: Sequence[Mapping[str, object]], subjects: Sequence[int]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for subject in subjects:
        loro = [
            row for row in peak_rows
            if int(row["subject"]) == subject and row["estimate_type"] == "leave_one_run_out"
        ]
        independent = [
            row for row in peak_rows
            if int(row["subject"]) == subject and row["estimate_type"] == "independent_single_run"
        ]
        values = [float(row["peak_frequency_hz"]) for row in loro if row["method_available"]]
        independent_values = [
            float(row["peak_frequency_hz"]) for row in independent if row["method_available"]
        ]
        complete = len(values) == 3
        span = max(values) - min(values) if complete else None
        independent_span = (
            max(independent_values) - min(independent_values)
            if len(independent_values) == 3 else None
        )
        failures = [
            f"held_out_{row['held_out_run']}:{row['failure_reason']}"
            for row in loro if not row["method_available"]
        ]
        output.append(
            {
                "subject": subject,
                "study_role": "held_out_method_evaluation",
                "valid_leave_one_run_out_peak_count": len(values),
                "boundary_candidate_fold_count": sum(row["peak_quality"] == "boundary_candidate" for row in loro),
                "no_peak_fold_count": sum(row["peak_quality"] == "poorly_defined_no_peak" for row in loro),
                "method_available": complete,
                "failure_reason": ";".join(failures),
                "peak_frequency_median_hz": float(np.median(values)) if complete else "",
                "peak_frequency_minimum_hz": min(values) if complete else "",
                "peak_frequency_maximum_hz": max(values) if complete else "",
                "three_fold_peak_span_hz": span if span is not None else "",
                "peak_reliability": peak_reliability_category(span) if span is not None else "unavailable",
                "exact_three_fold_agreement": bool(complete and len(set(values)) == 1),
                "valid_independent_single_run_peak_count": len(independent_values),
                "independent_single_run_peak_span_hz": independent_span if independent_span is not None else "",
                "independent_single_run_reliability": (
                    peak_reliability_category(independent_span)
                    if independent_span is not None else "unavailable"
                ),
                "task_outcome_used_for_availability": False,
            }
        )
    return output


def join_fixed_comparators(
    individualized_rows: Sequence[Mapping[str, object]],
    fixed_lookup: Mapping[tuple[int, int, int, str, str], Mapping[str, str]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source in individualized_rows:
        if source["semantic_condition"] != "both_fists_imagery":
            continue
        identity = (
            int(source["subject"]), int(source["run"]),
            int(source["run_trial_index"]), str(source["channel"])
        )
        narrow = fixed_lookup.get((*identity, "central_12_13"))
        broad = fixed_lookup.get((*identity, "mu"))
        if narrow is None or broad is None:
            raise RuntimeError(f"Missing exact fixed comparator for {identity}.")
        if narrow["source_sha256"] != source["source_sha256"]:
            raise RuntimeError(f"Raw provenance drift for {identity}.")
        output.append(
            {
                **dict(source),
                "quality_status": narrow["quality_status"],
                "quality_evidence": narrow["quality_evidence"],
                "confirmed_exclusion": narrow["confirmed_exclusion"],
                "fixed_narrow_change_percent": float(narrow["change_percent"]),
                "fixed_narrow_change_db": float(narrow["change_db"]),
                "fixed_broad_change_percent": float(broad["change_percent"]),
                "fixed_broad_change_db": float(broad["change_db"]),
                "matched_trial_identity": True,
            }
        )
    return output


def normalization_diagnostics(
    rows: Sequence[Mapping[str, object]], robust_threshold: float
) -> list[dict[str, object]]:
    prepared = [
        {
            **dict(row),
            "frequency_band": "individualized_mu",
            "change_percent": row["individualized_change_percent"],
        }
        for row in rows
    ]
    diagnosed = add_normalization_diagnostics(prepared, robust_threshold)
    output: list[dict[str, object]] = []
    for row in diagnosed:
        updated = dict(row)
        updated["individualized_reference_log_power_robust_z"] = updated.pop(
            "reference_log_power_robust_z"
        )
        updated["individualized_task_log_power_robust_z"] = updated.pop(
            "task_log_power_robust_z"
        )
        updated["individualized_low_reference_power_candidate"] = updated.pop(
            "low_reference_power_candidate"
        )
        # Independently generated dB must match the shared normalization helper.
        helper_db = float(updated.pop("change_db"))
        if not np.isclose(helper_db, float(updated["individualized_change_db"])):
            raise RuntimeError("Individualized dB normalization implementations differ.")
        updated.pop("change_percent")
        updated.pop("frequency_band")
        output.append(updated)
    return output


def create_peak_behavior_figure(
    peak_subject_rows: Sequence[Mapping[str, object]],
    peak_rows: Sequence[Mapping[str, object]],
    spectrum_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    available = [row for row in peak_subject_rows if row["method_available"]]
    frequencies = np.array([float(row["peak_frequency_median_hz"]) for row in available])
    loro = [row for row in peak_rows if row["estimate_type"] == "leave_one_run_out"]
    valid = [row for row in loro if row["method_available"]]
    examples: list[tuple[str, Mapping[str, object]]] = []
    if valid:
        examples.append(("strong interior", max(valid, key=lambda row: float(row["peak_prominence_db"]))))
        examples.append(("weakest valid", min(valid, key=lambda row: float(row["peak_prominence_db"]))))
    for quality, label in (("boundary_candidate", "boundary candidate"), ("poorly_defined_no_peak", "no valid peak")):
        candidates = [row for row in loro if row["peak_quality"] == quality]
        if candidates:
            examples.append((label, candidates[0]))
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    bins = np.arange(6.875, 14.126, 0.25)
    axes[0].hist(frequencies, bins=bins, color="#2a9d8f", edgecolor="white")
    axes[0].axvline(12.5, color="#e76f51", linestyle="--", label="fixed midpoint 12.5 Hz")
    axes[0].set(title="Held-out participant median central peak", xlabel="Frequency (Hz)", ylabel="Participants")
    axes[0].legend(frameon=False)
    colors = ("#264653", "#2a9d8f", "#e9c46a", "#e76f51")
    for (label, example), color in zip(examples, colors, strict=False):
        selected = [
            row for row in spectrum_rows
            if int(row["subject"]) == int(example["subject"])
            and int(row["held_out_run"]) == int(example["held_out_run"])
        ]
        axes[1].plot(
            [float(row["frequency_hz"]) for row in selected],
            [float(row["mean_log_psd_db"]) for row in selected],
            color=color,
            label=f"S{int(example['subject']):03d} hold {example['held_out_run']}: {label}",
        )
        candidate = float(example["candidate_frequency_hz"])
        candidate_row = min(selected, key=lambda row: abs(float(row["frequency_hz"]) - candidate))
        axes[1].scatter(candidate, float(candidate_row["mean_log_psd_db"]), color=color, s=30)
    axes[1].axvspan(7, 14, color="grey", alpha=0.08)
    axes[1].set(title="Measured rest-spectrum examples", xlabel="Frequency (Hz)", ylabel="Mean log PSD (dB re 1 V²/Hz)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_reliability_figure(
    peak_rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    loro = [row for row in peak_rows if row["estimate_type"] == "leave_one_run_out"]
    complete_subjects = sorted(
        subject for subject in {int(row["subject"]) for row in loro}
        if sum(int(row["subject"]) == subject and bool(row["method_available"]) for row in loro) == 3
    )
    figure, axis = plt.subplots(figsize=(12, 5.4))
    for subject in complete_subjects:
        selected = sorted(
            [row for row in loro if int(row["subject"]) == subject],
            key=lambda row: int(row["held_out_run"]),
        )
        axis.plot(
            [6, 10, 14], [float(row["peak_frequency_hz"]) for row in selected],
            marker="o", markersize=2.5, linewidth=0.8, alpha=0.35, color="#2a9d8f"
        )
    axis.set(
        title="Leave-one-run-out central peak reliability",
        xlabel="Held-out run (peak estimated from the other two runs)",
        ylabel="Estimated peak frequency (Hz)",
        xticks=[6, 10, 14],
        ylim=(6.8, 14.2),
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_method_scatter(
    subject_rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, channel in zip(axes, ("C4", "C3"), strict=True):
        rows = [row for row in subject_rows if row["channel"] == channel]
        fixed = np.array([float(row["fixed_narrow_median_percent"]) for row in rows])
        individualized = np.array([float(row["individualized_median_percent"]) for row in rows])
        lower = min(float(fixed.min()), float(individualized.min())) - 5
        upper = max(float(fixed.max()), float(individualized.max())) + 5
        colors = [
            {"stable": "#2a9d8f", "moderate": "#e9c46a", "unstable": "#e76f51"}[str(row["peak_reliability"])]
            for row in rows
        ]
        axis.scatter(fixed, individualized, c=colors, alpha=0.8, edgecolor="white", linewidth=0.4)
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--")
        axis.axhline(0, color="grey", linewidth=0.8)
        axis.axvline(0, color="grey", linewidth=0.8)
        axis.set(
            title=channel,
            xlabel="Fixed 12–13 Hz subject median (%)",
            ylabel="Individualized subject median (%)",
            xlim=(lower, upper), ylim=(lower, upper),
        )
    figure.suptitle("Matched held-out fixed versus individualized measurements")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_benefit_figure(
    subject_rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, channel in zip(axes, ("C4", "C3"), strict=True):
        rows = [row for row in subject_rows if row["channel"] == channel]
        distance = np.array([float(row["peak_distance_from_12_5_hz"]) for row in rows])
        difference = np.array([
            float(row["individualized_minus_fixed_narrow_median_percent"]) for row in rows
        ])
        rho = spearman_rho(distance, difference)
        axis.scatter(distance, difference, color="#2a9d8f", alpha=0.75)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            title=f"{channel}: Spearman ρ={rho:.2f}",
            xlabel="|median peak − 12.5 Hz| (Hz)",
            ylabel="Individualized − fixed median (points)",
        )
    figure.suptitle("Exploratory benefit versus distance from the fixed band")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    config = load_individual_frequency_config(args.config)
    raw_config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config.study_stage != "final_method_frozen" or config.evaluation_locked:
        raise RuntimeError("Held-out evaluation requires the committed final method freeze.")
    if raw_config["held_out_evaluation_individualized_task_outcomes_inspected_at_final_freeze"]:
        raise RuntimeError("Final freeze chronology is invalid.")
    final = raw_config["final_method"]
    search_range = tuple(float(value) for value in final["search_range_hz"])
    prominence = float(final["minimum_prominence_db"])
    half_width = float(final["individualized_band_half_width_hz"])
    frequency_step = float(final["individualized_morlet_frequency_step_hz"])
    preprocessing = load_preprocessing_config()
    epoching = load_epoching_config()
    spectral = load_event_related_spectral_config()
    quality = load_trial_quality_config()
    fixed_lookup = load_fixed_trial_lookup(config.evaluation_subjects)

    peak_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    all_individualized_rows: list[dict[str, object]] = []
    for index, subject in enumerate(config.evaluation_subjects, start=1):
        run_results = {
            run: extract_run_epochs(
                subject, run, preprocessing, epoching, data_directory=args.data_dir
            )
            for run in config.runs
        }
        subject_loro: dict[int, object] = {}
        for run in config.runs:
            result = run_results[run]
            data = result.rest_epochs.get_data(copy=True)
            frequencies, log_power = rest_roi_log_welch(
                data, result.rest_epochs.ch_names, float(result.rest_epochs.info["sfreq"]),
                config.central_channels, config.welch_window_seconds,
            )
            estimate = estimate_central_peak(frequencies, log_power, search_range, prominence)
            peak_rows.append(peak_table_row(subject, "independent_single_run", "", (run,), estimate, data.shape[0]))
        for held_out_run in config.runs:
            training_runs = training_runs_for_held_out(held_out_run, config.runs)
            training_data = np.concatenate(
                [run_results[run].rest_epochs.get_data(copy=True) for run in training_runs], axis=0
            )
            reference = run_results[training_runs[0]].rest_epochs
            frequencies, log_power = rest_roi_log_welch(
                training_data, reference.ch_names, float(reference.info["sfreq"]),
                config.central_channels, config.welch_window_seconds,
            )
            estimate = estimate_central_peak(frequencies, log_power, search_range, prominence)
            subject_loro[held_out_run] = estimate
            peak_rows.append(peak_table_row(subject, "leave_one_run_out", held_out_run, training_runs, estimate, training_data.shape[0]))
            for frequency, power_db in zip(frequencies, log_power, strict=True):
                if 6.0 <= frequency <= 15.0:
                    spectrum_rows.append(
                        {
                            "subject": subject,
                            "held_out_run": held_out_run,
                            "training_runs": "/".join(str(run) for run in training_runs),
                            "frequency_hz": float(frequency),
                            "mean_log_psd_db": float(power_db),
                            "task_outcome_used": False,
                        }
                    )
        complete = all(subject_loro[run].method_available for run in config.runs)
        subject_rows: list[dict[str, object]] = []
        if complete:
            for held_out_run in config.runs:
                estimate = subject_loro[held_out_run]
                frequencies = individualized_frequency_vector(
                    float(estimate.peak_frequency_hz), half_width, frequency_step
                )
                dataset = assemble_spectral_epoch_dataset(
                    {held_out_run: run_results[held_out_run]}, (held_out_run,)
                )
                channel_indices = [dataset.channel_names.index(channel) for channel in ("C4", "C3")]
                selected = SpectralEpochDataset(
                    task_data_volts=dataset.task_data_volts[:, channel_indices].copy(),
                    rest_data_volts=dataset.rest_data_volts[:, channel_indices].copy(),
                    task_times=dataset.task_times.copy(), rest_times=dataset.rest_times.copy(),
                    channel_names=["C4", "C3"], sampling_frequency_hz=dataset.sampling_frequency_hz,
                    pairs=dataset.pairs.copy(),
                )
                task_snapshot = selected.task_data_volts.copy()
                rest_snapshot = selected.rest_data_volts.copy()
                task_power = compute_custom_morlet_power(
                    selected.task_data_volts, selected.task_times,
                    selected.sampling_frequency_hz, frequencies, spectral
                )
                rest_power = compute_custom_morlet_power(
                    selected.rest_data_volts, selected.rest_times,
                    selected.sampling_frequency_hz, frequencies, spectral
                )
                measured = individualized_trial_measurements(
                    task_power, rest_power, selected, spectral, float(estimate.peak_frequency_hz)
                )
                if not np.array_equal(task_snapshot, selected.task_data_volts) or not np.array_equal(rest_snapshot, selected.rest_data_volts):
                    raise RuntimeError("Individualized evaluation mutated source epochs.")
                subject_rows.extend(measured)
            joined = join_fixed_comparators(subject_rows, fixed_lookup)
            diagnosed = normalization_diagnostics(joined, quality.robust_z_threshold)
            all_individualized_rows.extend(diagnosed)
        print(
            f"Held-out evaluation: subject {subject:03d} ({index}/{len(config.evaluation_subjects)}); "
            f"complete peak method={complete}"
        )

    peak_subject_rows = summarize_peak_subjects(peak_rows, config.evaluation_subjects)
    peak_by_subject = {int(row["subject"]): row for row in peak_subject_rows}
    subject_rows = summarize_subject_methods(all_individualized_rows, peak_by_subject)
    group_rows = paired_group_summaries(subject_rows, len(config.evaluation_subjects))
    categories = {row["channel"]: row["method_category"] for row in group_rows}
    overall_category = (
        "clear improvement"
        if all(value == "clear improvement" for value in categories.values())
        else "mixed result"
        if all(value != "no improvement" for value in categories.values())
        and any(value in ("clear improvement", "mixed result") for value in categories.values())
        else "no improvement"
    )
    relationship_rows: list[dict[str, object]] = []
    for channel in ("C4", "C3"):
        rows = [row for row in subject_rows if row["channel"] == channel]
        distance = np.array([float(row["peak_distance_from_12_5_hz"]) for row in rows])
        benefit = np.array([
            float(row["individualized_minus_fixed_narrow_median_percent"]) for row in rows
        ])
        relationship_rows.append(
            {
                "channel": channel,
                "matched_subject_count": len(rows),
                "spearman_peak_distance_vs_individualized_minus_fixed_percent": spearman_rho(distance, benefit),
                "analysis_role": "secondary_exploratory",
            }
        )
    reliability_counts = Counter(
        str(row["peak_reliability"]) for row in peak_subject_rows if row["method_available"]
    )
    availability = sum(bool(row["method_available"]) for row in peak_subject_rows)
    metadata = {
        "schema_version": 1,
        "study_start_commit": raw_config["study_start_git_commit"],
        "final_method_freeze_commit": "9eb3989",
        "evaluation_subject_count": len(config.evaluation_subjects),
        "individualizable_subject_count": availability,
        "individualizable_fraction": availability / len(config.evaluation_subjects),
        "peak_reliability_counts": dict(reliability_counts),
        "overall_method_category": overall_category,
        "channel_method_categories": categories,
        "fixed_baseline_redefined": False,
        "outcome_used_for_peak_estimation": False,
        "held_out_run_used_for_its_peak_estimation": False,
        "full_tfr_arrays_persisted": False,
        "csp_performed": False,
        "classification_performed": False,
        "configuration_sha256": sha256(args.config.resolve()),
        "fixed_trial_artifact_sha256": {
            "subjects01_20": sha256(PROJECT_ROOT / "docs/assets/subjects01-20_replication_trial_spectral_measurements.csv"),
            "subjects21_109": sha256(PROJECT_ROOT / "docs/assets/subjects01-109_full_replication_replication2_trial_spectral_measurements.csv"),
        },
    }

    output = args.output_dir
    write_csv(peak_rows, output / f"{REPORT_PREFIX}_peak_estimates.csv")
    write_csv(peak_subject_rows, output / f"{REPORT_PREFIX}_peak_subject_summary.csv")
    write_csv(spectrum_rows, output / f"{REPORT_PREFIX}_rest_spectra.csv")
    write_csv(all_individualized_rows, output / f"{REPORT_PREFIX}_trial_comparison.csv")
    write_csv(subject_rows, output / f"{REPORT_PREFIX}_subject_comparison.csv")
    write_csv(group_rows, output / f"{REPORT_PREFIX}_group_comparison.csv")
    write_csv(relationship_rows, output / f"{REPORT_PREFIX}_peak_distance_relationship.csv")
    metadata_path = output / f"{REPORT_PREFIX}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    create_peak_behavior_figure(
        peak_subject_rows, peak_rows, spectrum_rows,
        output / f"{REPORT_PREFIX}_peak_behavior.png"
    )
    create_reliability_figure(peak_rows, output / f"{REPORT_PREFIX}_peak_reliability.png")
    create_method_scatter(subject_rows, output / f"{REPORT_PREFIX}_fixed_vs_individualized.png")
    create_benefit_figure(subject_rows, output / f"{REPORT_PREFIX}_benefit_vs_distance.png")

    print("\nHeld-out method evaluation summary:")
    print(f"  Coverage: {availability}/{len(config.evaluation_subjects)} ({availability / len(config.evaluation_subjects):.1%})")
    for row in group_rows:
        print(
            f"  {row['channel']}: fixed {row['fixed_narrow_negative_subject_count']}/{availability} "
            f"vs individualized {row['individualized_negative_subject_count']}/{availability}; "
            f"paired median difference {row['median_paired_individualized_minus_fixed_narrow_percent']:.2f} points; "
            f"{row['method_category']}"
        )
    print(f"  Overall frozen category: {overall_category}")


if __name__ == "__main__":
    main()
