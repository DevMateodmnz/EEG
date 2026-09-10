"""Run the frozen second EEG replication and combined subject-level report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    build_primary_replication_rows,
    cohort_feature_summaries,
)
from eeg_project.cohort_processing import display_path, process_subject  # noqa: E402
from eeg_project.epoching import load_epoching_config  # noqa: E402
from eeg_project.full_cohort import (  # noqa: E402
    DEFAULT_FULL_COHORT_CONFIG_PATH,
    audit_requested_subjects,
    checkpoint_digest,
    checkpoint_payload,
    cohort_label,
    eligibility_for_requested_subjects,
    full_cohort_fingerprint,
    load_frozen_replication_policy,
    load_full_cohort_config,
    load_subject_checkpoint,
    write_subject_checkpoint,
)
from eeg_project.preprocessing import (  # noqa: E402
    DEFAULT_DATA_DIRECTORY,
    load_preprocessing_config,
)
from eeg_project.time_frequency import load_event_related_spectral_config  # noqa: E402
from eeg_project.trial_quality import (  # noqa: E402
    load_trial_quality_config,
    spearman_rho,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "full_cohort_checkpoints"
HISTORICAL_PREFIX = "subjects01-20_replication"
REPORT_PREFIX = "subjects01-109_full_replication"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit subjects 21-109, analyze them as independent replication cohort 2, "
            "then calculate combined subjects 2-109 evidence without CSP/classification."
        )
    )
    parser.add_argument(
        "--full-config", type=Path, default=DEFAULT_FULL_COHORT_CONFIG_PATH
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit all subjects 21-109 without calculating spectral outcomes.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute every eligible cohort-2 subject instead of using valid checkpoints.",
    )
    parser.add_argument(
        "--verify-subjects",
        nargs="+",
        type=int,
        help="Recompute listed cohort-2 subjects and compare with existing checkpoints.",
    )
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {output_path}.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def source_hashes_for_subject(
    subject: int, audit_rows: Sequence[Mapping[str, object]]
) -> dict[str, str]:
    rows = sorted(
        [row for row in audit_rows if int(row["subject"]) == subject],
        key=lambda row: int(row["run"]),
    )
    return {
        f"run{int(row['run']):02d}": str(row["source_sha256"])
        for row in rows
        if row["source_sha256"]
    }


def load_scientific_configs(full_config):
    paths = {
        name: PROJECT_ROOT / record.path
        for name, record in full_config.frozen_methodology.items()
    }
    return (
        load_preprocessing_config(paths["preprocessing"]),
        load_epoching_config(paths["epoching"]),
        load_event_related_spectral_config(paths["event_related_spectral"]),
        load_trial_quality_config(paths["trial_quality"]),
    )


def historical_rows(output_directory: Path) -> dict[str, list[dict[str, str]]]:
    suffixes = {
        "subject_rows": "subject_feature_summary",
        "run_rows": "subject_run_summary",
        "primary_rows": "replication_primary_features",
        "qc_rows": "subject_qc_summary",
        "peak_rows": "central_peak_summary",
        "audit_rows": "cohort_audit",
        "cohort_rows": "cohort_feature_summary",
    }
    return {
        key: read_csv(output_directory / f"{HISTORICAL_PREFIX}_{suffix}.csv")
        for key, suffix in suffixes.items()
    }


def verify_historical_results(rows: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    subject_one = {
        str(row["channel"]): float(row["median_change_percent"])
        for row in rows["primary_rows"]
        if int(row["subject"]) == 1
    }
    if not np.isclose(subject_one["C4"], -52.28373306890746):
        raise RuntimeError("Historical subject-1 C4 result changed.")
    if not np.isclose(subject_one["C3"], -55.57490648782269):
        raise RuntimeError("Historical subject-1 C3 result changed.")
    historical_primary = {
        str(row["channel"]): row
        for row in rows["cohort_rows"]
        if row["analysis_role"] == "primary"
    }
    expected = {
        "C4": (17, -30.732830057741275),
        "C3": (16, -29.82874371941025),
    }
    for channel, (negative, median) in expected.items():
        row = historical_primary[channel]
        if int(row["negative_subject_count"]) != negative or not np.isclose(
            float(row["median_across_subject_medians_percent"]), median
        ):
            raise RuntimeError(f"Historical replication-1 {channel} result changed.")


def process_replication_2(
    eligible_subjects,
    audit_rows,
    full_config,
    replication_policy,
    scientific_configs,
    data_directory: Path,
    checkpoint_directory: Path,
    *,
    resume: bool,
) -> list[dict[str, object]]:
    fingerprint = full_cohort_fingerprint(full_config)
    preprocessing_config, epoching_config, spectral_config, quality_config = scientific_configs
    payloads: list[dict[str, object]] = []
    for index, subject in enumerate(eligible_subjects, start=1):
        hashes = source_hashes_for_subject(subject, audit_rows)
        checkpoint_path = checkpoint_directory / f"subject{subject:03d}.json"
        payload = (
            load_subject_checkpoint(
                checkpoint_path,
                subject,
                "replication_2",
                fingerprint,
                hashes,
            )
            if resume
            else None
        )
        if payload is None:
            print(
                f"Processing cohort-2 subject {subject:03d} "
                f"({index}/{len(eligible_subjects)})..."
            )
            result = process_subject(
                subject,
                "replication_2",
                replication_policy,
                preprocessing_config,
                epoching_config,
                spectral_config,
                quality_config,
                data_directory,
                audit_rows,
            )
            payload = checkpoint_payload(
                subject,
                "replication_2",
                result,
                fingerprint,
                hashes,
            )
            write_subject_checkpoint(checkpoint_path, payload)
        else:
            print(
                f"Reusing validated subject {subject:03d} checkpoint "
                f"({index}/{len(eligible_subjects)})."
            )
        payloads.append(payload)
    return payloads


def verify_checkpoint_subjects(
    subjects,
    audit_rows,
    full_config,
    replication_policy,
    scientific_configs,
    data_directory: Path,
    checkpoint_directory: Path,
) -> None:
    fingerprint = full_cohort_fingerprint(full_config)
    preprocessing_config, epoching_config, spectral_config, quality_config = scientific_configs
    allowed = set(full_config.replication_2_subjects)
    for subject in subjects:
        if subject not in allowed:
            raise ValueError(f"Verification subject {subject} is not in cohort 2.")
        hashes = source_hashes_for_subject(subject, audit_rows)
        path = checkpoint_directory / f"subject{subject:03d}.json"
        existing = load_subject_checkpoint(
            path, subject, "replication_2", fingerprint, hashes
        )
        if existing is None:
            raise RuntimeError(f"No valid checkpoint exists for subject {subject}.")
        regenerated_result = process_subject(
            subject,
            "replication_2",
            replication_policy,
            preprocessing_config,
            epoching_config,
            spectral_config,
            quality_config,
            data_directory,
            audit_rows,
        )
        regenerated = checkpoint_payload(
            subject,
            "replication_2",
            regenerated_result,
            fingerprint,
            hashes,
        )
        if checkpoint_digest(existing) != checkpoint_digest(regenerated):
            raise RuntimeError(f"Subject {subject} checkpoint is not deterministic.")
        print(f"Subject {subject}: deterministic checkpoint verified.")


def flatten_results(payloads: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    output = {
        "trial_rows": [],
        "subject_rows": [],
        "run_rows": [],
        "qc_rows": [],
        "peak_rows": [],
    }
    for payload in payloads:
        result = payload["result"]
        output["trial_rows"].extend(result["trial_rows"])
        output["subject_rows"].extend(result["feature_rows"])
        output["run_rows"].extend(result["run_rows"])
        output["qc_rows"].append(result["qc_row"])
        output["peak_rows"].append(result["peak_row"])
    return output


def add_cohort_name(rows, cohort: str):
    return [{"cohort": cohort, **dict(row)} for row in rows]


def primary_subject_table(
    full_config,
    eligibility_rows,
    primary_rows,
) -> list[dict[str, object]]:
    primary_lookup = {
        (int(row["subject"]), str(row["channel"])): row for row in primary_rows
    }
    eligibility_lookup = {
        int(row["subject"]): row for row in eligibility_rows
    }
    output: list[dict[str, object]] = []
    for subject in range(1, 110):
        cohort = cohort_label(subject, full_config)
        eligible = subject <= 20 or (
            subject in eligibility_lookup
            and eligibility_lookup[subject]["technical_eligibility"] == "eligible"
        )
        row: dict[str, object] = {
            "subject": subject,
            "cohort": cohort,
            "technical_eligibility": "eligible" if eligible else "technically_incompatible",
            "failure_reason": (
                "" if subject <= 20 else eligibility_lookup[subject]["failure_reason"]
            ),
            "subject_is_group_observational_unit": True,
        }
        for channel in ("C4", "C3"):
            feature = primary_lookup.get((subject, channel))
            prefix = channel.lower()
            row.update(
                {
                    f"{prefix}_fists_trial_count": feature["trial_count"] if feature else "",
                    f"{prefix}_negative_trial_count": feature["negative_trial_count"] if feature else "",
                    f"{prefix}_negative_trial_fraction": feature["negative_trial_fraction"] if feature else "",
                    f"{prefix}_median_percent_change": feature["median_change_percent"] if feature else "",
                    f"{prefix}_iqr_percentage_points": feature["iqr_change_percentage_points"] if feature else "",
                    f"{prefix}_run_6_median": feature["run_6_median_change_percent"] if feature else "",
                    f"{prefix}_run_10_median": feature["run_10_median_change_percent"] if feature else "",
                    f"{prefix}_run_14_median": feature["run_14_median_change_percent"] if feature else "",
                    f"{prefix}_supporting_run_count": feature["supporting_run_count"] if feature else "",
                    f"{prefix}_run_direction": feature["run_direction_consistency"] if feature else "",
                    f"{prefix}_qc_sensitivity_median": feature["exclude_qc_candidates_median_change_percent"] if feature else "",
                    f"{prefix}_qc_direction_preserved": feature["qc_sensitivity_preserves_primary_direction"] if feature else "",
                }
            )
        output.append(row)
    return output


def run_consistency_summaries(
    run_rows,
    subject_ids: Sequence[int],
    cohort: str,
    features,
) -> list[dict[str, object]]:
    included = set(subject_ids)
    output: list[dict[str, object]] = []
    for condition, channel, band in features:
        by_subject: dict[int, list[float]] = defaultdict(list)
        for row in run_rows:
            if (
                int(row["subject"]) in included
                and row["semantic_condition"] == condition
                and row["channel"] == channel
                and row["frequency_band"] == band
            ):
                by_subject[int(row["subject"])].append(
                    float(row["median_change_percent"])
                )
        if set(by_subject) != included or any(len(values) != 3 for values in by_subject.values()):
            raise RuntimeError(f"Incomplete run summaries for {cohort} {(condition, channel, band)}.")
        counts = Counter(sum(value < 0 for value in values) for values in by_subject.values())
        output.append(
            {
                "cohort": cohort,
                "semantic_condition": condition,
                "channel": channel,
                "frequency_band": band,
                "subject_count": len(included),
                "three_of_three_negative": counts[3],
                "two_of_three_negative": counts[2],
                "one_of_three_negative": counts[1],
                "zero_of_three_negative": counts[0],
                "fraction_with_at_least_two_negative_runs": (
                    counts[3] + counts[2]
                ) / len(included),
            }
        )
    return output


def trial_consistency_summaries(
    primary_rows,
    subject_ids: Sequence[int],
    cohort: str,
) -> list[dict[str, object]]:
    included = set(subject_ids)
    output = []
    for channel in ("C4", "C3"):
        fractions = np.array(
            [
                float(row["negative_trial_fraction"])
                for row in primary_rows
                if int(row["subject"]) in included and row["channel"] == channel
            ]
        )
        if fractions.size != len(included):
            raise RuntimeError(f"Incomplete trial consistency for {cohort} {channel}.")
        q25, median, q75 = np.percentile(fractions, [25, 50, 75])
        output.append(
            {
                "cohort": cohort,
                "channel": channel,
                "subject_count": fractions.size,
                "median_negative_trial_fraction": float(median),
                "q25_negative_trial_fraction": float(q25),
                "q75_negative_trial_fraction": float(q75),
                "minimum_negative_trial_fraction": float(fractions.min()),
                "maximum_negative_trial_fraction": float(fractions.max()),
            }
        )
    return output


def peak_distribution_rows(peak_rows, cohorts: Mapping[str, Sequence[int]]):
    output = []
    categories = ("8", "9", "10", "11", "12", "13", "poorly_defined")
    for cohort, subjects in cohorts.items():
        included = set(subjects)
        selected = [row for row in peak_rows if int(row["subject"]) in included]
        if len(selected) != len(included):
            raise RuntimeError(f"Incomplete central-peak rows for {cohort}.")
        counts = Counter(str(row["peak_category"]) for row in selected)
        for category in categories:
            output.append(
                {
                    "cohort": cohort,
                    "peak_category": category,
                    "subject_count": counts[category],
                    "cohort_subject_count": len(included),
                    "fraction": counts[category] / len(included),
                    "peak_procedure_redefined": False,
                }
            )
    return output


def peak_relationship_rows(
    peak_rows,
    subject_rows,
    cohorts: Mapping[str, Sequence[int]],
    replication_policy,
):
    output = []
    for cohort, subjects in cohorts.items():
        included = set(subjects)
        for condition, channel, band in replication_policy.primary_replication_features:
            feature_values = {
                int(row["subject"]): float(row["median_change_percent"])
                for row in subject_rows
                if int(row["subject"]) in included
                and row["semantic_condition"] == condition
                and row["channel"] == channel
                and row["frequency_band"] == band
            }
            peaks = [
                row
                for row in peak_rows
                if int(row["subject"]) in included
                and str(row["well_defined_peak"]).lower() == "true"
            ]
            frequencies = np.array([float(row["peak_frequency_hz"]) for row in peaks])
            changes = np.array([feature_values[int(row["subject"])] for row in peaks])
            distances = np.abs(frequencies - 12.5)
            output.append(
                {
                    "cohort": cohort,
                    "channel": channel,
                    "subjects_with_well_defined_peak": len(peaks),
                    "spearman_peak_frequency_vs_change": (
                        spearman_rho(frequencies, changes) if len(peaks) >= 2 else ""
                    ),
                    "spearman_distance_from_12_5_hz_vs_change": (
                        spearman_rho(distances, changes) if len(peaks) >= 2 else ""
                    ),
                    "primary_band_redefined": False,
                    "analysis_role": "exploratory",
                }
            )
    return output


def qc_relationship_rows(qc_rows, primary_rows, cohorts):
    output = []
    for cohort, subjects in cohorts.items():
        included = set(subjects)
        burden = {
            int(row["subject"]): (
                int(row["qc_task_candidates"])
                + int(row["qc_rest_candidates"])
                + int(row["channel_candidate_run_records"])
            )
            for row in qc_rows
            if int(row["subject"]) in included
        }
        for channel in ("C4", "C3"):
            values = {
                int(row["subject"]): abs(float(row["median_change_percent"]))
                for row in primary_rows
                if int(row["subject"]) in included and row["channel"] == channel
            }
            ids = sorted(included)
            if set(burden) != included or set(values) != included:
                raise RuntimeError(f"Incomplete QC relationship rows for {cohort}.")
            output.append(
                {
                    "cohort": cohort,
                    "channel": channel,
                    "subject_count": len(ids),
                    "spearman_candidate_burden_vs_absolute_primary_change": spearman_rho(
                        np.array([burden[subject] for subject in ids], dtype=float),
                        np.array([values[subject] for subject in ids]),
                    ),
                    "analysis_role": "exploratory_no_exclusion",
                }
            )
    return output


def plot_primary_subjects(primary_rows, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    colors = {"replication_1": "#4c78a8", "replication_2": "#72b7b2"}
    for axis, channel in zip(axes, ("C4", "C3"), strict=True):
        axis.axhline(0, color="#555555", linewidth=0.9)
        for cohort in ("replication_1", "replication_2"):
            selected = sorted(
                [
                    row for row in primary_rows
                    if row["subject_role"] == cohort and row["channel"] == channel
                ],
                key=lambda row: int(row["subject"]),
            )
            axis.scatter(
                [int(row["subject"]) for row in selected],
                [float(row["median_change_percent"]) for row in selected],
                color=colors[cohort], s=24, alpha=0.8,
                label=cohort.replace("_", " "),
            )
        development = next(
            row for row in primary_rows
            if int(row["subject"]) == 1 and row["channel"] == channel
        )
        axis.scatter(
            [1], [float(development["median_change_percent"])],
            color="#f58518", marker="*", s=150, label="development subject 1",
        )
        axis.set_ylabel(f"{channel} fists 12–13 Hz\nsubject median (%)")
        axis.grid(alpha=0.15)
        axis.legend(frameon=False, ncol=3, fontsize=9)
    axes[-1].set_xlabel("Subject ID")
    figure.suptitle("Frozen primary measurements — one value per person")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_cohort_comparison(primary_rows, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    rng = np.random.default_rng(20260829)
    for axis, channel in zip(axes, ("C4", "C3"), strict=True):
        groups = []
        for cohort in ("replication_1", "replication_2"):
            groups.append(
                np.array(
                    [
                        float(row["median_change_percent"])
                        for row in primary_rows
                        if row["subject_role"] == cohort and row["channel"] == channel
                    ]
                )
            )
        axis.boxplot(groups, tick_labels=["replication 1", "replication 2"], showfliers=False)
        for index, values in enumerate(groups, start=1):
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            axis.scatter(index + jitter, values, s=18, alpha=0.6, color=("#4c78a8", "#72b7b2")[index - 1])
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set(title=channel, ylabel="Subject median change (%)")
        axis.grid(axis="y", alpha=0.15)
    figure.suptitle("Independent replication-cohort comparison")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_run_consistency(rows, output_path: Path) -> None:
    selected = [
        row for row in rows
        if row["cohort"] in {"replication_1", "replication_2"}
        and row["semantic_condition"] == "both_fists_imagery"
        and row["frequency_band"] == "central_12_13"
        and row["channel"] in {"C4", "C3"}
    ]
    labels = [f"{row['cohort'].replace('_', ' ')}\n{row['channel']}" for row in selected]
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bottom = np.zeros(len(selected))
    for field, label, color in (
        ("three_of_three_negative", "3/3 negative", "#2a9d8f"),
        ("two_of_three_negative", "2/3 negative", "#8ab17d"),
        ("one_of_three_negative", "1/3 negative", "#f4a261"),
        ("zero_of_three_negative", "0/3 negative", "#e76f51"),
    ):
        values = np.array([int(row[field]) for row in selected])
        axis.bar(labels, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_ylabel("Subjects")
    axis.set_title("Within-subject run-direction consistency")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_narrow_mu(subject_rows, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis, channel in zip(axes, ("C4", "C3"), strict=True):
        lookup = {
            (int(row["subject"]), str(row["frequency_band"])): float(row["median_change_percent"])
            for row in subject_rows
            if row["semantic_condition"] == "both_fists_imagery"
            and row["channel"] == channel
            and row["frequency_band"] in {"central_12_13", "mu"}
        }
        for cohort, color in (("replication_1", "#4c78a8"), ("replication_2", "#72b7b2")):
            subjects = sorted(
                subject for subject in range(2, 110)
                if cohort_label_from_id(subject) == cohort and (subject, "mu") in lookup
            )
            axis.scatter(
                [lookup[(subject, "central_12_13")] for subject in subjects],
                [lookup[(subject, "mu")] for subject in subjects],
                color=color, alpha=0.65, s=22, label=cohort.replace("_", " "),
            )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set(
            title=channel,
            xlabel="Fixed 12–13 Hz median (%)",
            ylabel="Fixed 8–13 Hz mu median (%)",
        )
        axis.grid(alpha=0.15)
        axis.legend(frameon=False)
    figure.suptitle("Narrow versus broad fists responses")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def cohort_label_from_id(subject: int) -> str:
    if subject == 1:
        return "development"
    return "replication_1" if subject <= 20 else "replication_2"


def plot_peak_evidence(peak_rows, primary_rows, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    categories = ("8", "9", "10", "11", "12", "13", "poorly_defined")
    positions = np.arange(len(categories))
    width = 0.38
    for offset, cohort, color in (
        (-width / 2, "replication_1", "#4c78a8"),
        (width / 2, "replication_2", "#72b7b2"),
    ):
        selected = [row for row in peak_rows if row["subject_role"] == cohort]
        counts = Counter(str(row["peak_category"]) for row in selected)
        axes[0].bar(
            positions + offset,
            [counts[category] for category in categories],
            width=width,
            color=color,
            label=cohort.replace("_", " "),
        )
    axes[0].set_xticks(positions, categories)
    axes[0].set(xlabel="Central peak category", ylabel="Subjects", title="Frozen peak procedure")
    axes[0].legend(frameon=False)
    c4 = {
        int(row["subject"]): float(row["median_change_percent"])
        for row in primary_rows if row["channel"] == "C4"
    }
    for cohort, color in (("replication_1", "#4c78a8"), ("replication_2", "#72b7b2")):
        selected = [
            row for row in peak_rows
            if row["subject_role"] == cohort
            and str(row["well_defined_peak"]).lower() == "true"
        ]
        axes[1].scatter(
            [abs(float(row["peak_frequency_hz"]) - 12.5) for row in selected],
            [c4[int(row["subject"])] for row in selected],
            color=color, alpha=0.65, s=25, label=cohort.replace("_", " "),
        )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set(
        xlabel="Distance from 12.5 Hz (Hz)",
        ylabel="C4 fixed-band median (%)",
        title="Exploratory peak-distance relationship",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def normalize_historical_rows(rows, full_config, historical_audit):
    audit_by_subject = defaultdict(list)
    for row in historical_audit:
        audit_by_subject[int(row["subject"])].append(row)
    for collection_name in ("subject_rows", "run_rows", "primary_rows", "qc_rows", "peak_rows"):
        for row in rows[collection_name]:
            subject = int(row["subject"])
            row["subject_role"] = cohort_label(subject, full_config)
            if collection_name == "qc_rows":
                audits = audit_by_subject[subject]
                row.setdefault("rest_pairs", row["task_trials"])
                row.setdefault(
                    "invalid_tail_runs",
                    sum(int(item["trailing_zero_samples"]) > 0 for item in audits),
                )
                row.setdefault(
                    "trailing_zero_samples_total",
                    sum(int(item["trailing_zero_samples"]) for item in audits),
                )


def main() -> None:
    arguments = parse_arguments()
    full_config = load_full_cohort_config(arguments.full_config)
    replication_policy = load_frozen_replication_policy(full_config)
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    data_directory = arguments.data_dir.expanduser().resolve()
    checkpoint_directory = arguments.checkpoint_dir.expanduser().resolve()

    audit_path = output_directory / f"{REPORT_PREFIX}_replication2_audit.csv"
    eligibility_path = output_directory / f"{REPORT_PREFIX}_replication2_eligibility.csv"
    audit_rows = audit_requested_subjects(
        full_config.replication_2_subjects,
        full_config.runs,
        replication_policy,
        data_directory=data_directory,
    )
    for row in audit_rows:
        if row["source_file"]:
            row["source_file"] = display_path(Path(str(row["source_file"])))
        row["cohort"] = "replication_2"
        row["task_counterbalance_pattern"] = (
            f"{row['T1_count']}/{row['T2_count']}"
            if row["T1_count"] != "" and row["T2_count"] != ""
            else ""
        )
        row["outcome_used_for_eligibility"] = False
    eligibility_rows = eligibility_for_requested_subjects(
        audit_rows,
        full_config.replication_2_subjects,
        full_config,
    )
    write_csv(audit_rows, audit_path)
    write_csv(eligibility_rows, eligibility_path)
    compatible_runs = sum(bool(row["protocol_compatible"]) for row in audit_rows)
    eligible_subjects = [
        int(row["subject"])
        for row in eligibility_rows
        if row["technical_eligibility"] == "eligible"
    ]
    print(
        f"Cohort-2 technical audit: {compatible_runs}/{len(audit_rows)} compatible runs; "
        f"{len(eligible_subjects)}/{len(full_config.replication_2_subjects)} eligible subjects."
    )
    if arguments.audit_only:
        print(f"Audit: {display_path(audit_path)}")
        return

    scientific_configs = load_scientific_configs(full_config)
    if arguments.verify_subjects:
        verify_checkpoint_subjects(
            arguments.verify_subjects,
            audit_rows,
            full_config,
            replication_policy,
            scientific_configs,
            data_directory,
            checkpoint_directory,
        )
        return

    payloads = process_replication_2(
        eligible_subjects,
        audit_rows,
        full_config,
        replication_policy,
        scientific_configs,
        data_directory,
        checkpoint_directory,
        resume=not arguments.no_resume,
    )
    current = flatten_results(payloads)
    current_primary = build_primary_replication_rows(
        current["subject_rows"], current["run_rows"], replication_policy
    )

    historical = historical_rows(output_directory)
    verify_historical_results(historical)
    normalize_historical_rows(historical, full_config, historical["audit_rows"])

    all_subject_rows = [*historical["subject_rows"], *current["subject_rows"]]
    all_run_rows = [*historical["run_rows"], *current["run_rows"]]
    all_primary_rows = [*historical["primary_rows"], *current_primary]
    all_qc_rows = [*historical["qc_rows"], *current["qc_rows"]]
    all_peak_rows = [*historical["peak_rows"], *current["peak_rows"]]
    cohort_ids = {
        "replication_1": full_config.replication_1_subjects,
        "replication_2": tuple(eligible_subjects),
        "combined_replication": (
            *full_config.replication_1_subjects,
            *tuple(eligible_subjects),
        ),
    }

    comparison_rows: list[dict[str, object]] = []
    for cohort, subjects in cohort_ids.items():
        summaries = cohort_feature_summaries(
            all_subject_rows,
            all_primary_rows,
            replication_policy,
            included_subject_ids=subjects,
        )
        comparison_rows.extend(add_cohort_name(summaries, cohort))

    replication_1_primary = {
        row["channel"]: row
        for row in comparison_rows
        if row["cohort"] == "replication_1" and row["analysis_role"] == "primary"
    }
    if not (
        int(replication_1_primary["C4"]["negative_subject_count"]) == 17
        and int(replication_1_primary["C3"]["negative_subject_count"]) == 16
        and np.isclose(
            float(replication_1_primary["C4"]["median_across_subject_medians_percent"]),
            -30.732830057741275,
        )
        and np.isclose(
            float(replication_1_primary["C3"]["median_across_subject_medians_percent"]),
            -29.82874371941025,
        )
    ):
        raise RuntimeError("Replication cohort 1 did not reproduce exactly.")

    run_rows = []
    trial_consistency_rows = []
    for cohort, subjects in cohort_ids.items():
        run_rows.extend(
            run_consistency_summaries(
                all_run_rows,
                subjects,
                cohort,
                replication_policy.all_features,
            )
        )
        trial_consistency_rows.extend(
            trial_consistency_summaries(all_primary_rows, subjects, cohort)
        )
    peak_cohorts = {
        "replication_1": cohort_ids["replication_1"],
        "replication_2": cohort_ids["replication_2"],
        "combined_replication": cohort_ids["combined_replication"],
    }
    peak_distributions = peak_distribution_rows(all_peak_rows, peak_cohorts)
    peak_relationships = peak_relationship_rows(
        all_peak_rows, all_subject_rows, peak_cohorts, replication_policy
    )
    qc_relationships = qc_relationship_rows(
        all_qc_rows,
        all_primary_rows,
        {
            "replication_2": cohort_ids["replication_2"],
            "combined_replication": cohort_ids["combined_replication"],
        },
    )
    subject_primary = primary_subject_table(
        full_config, eligibility_rows, all_primary_rows
    )

    paths = {
        "audit": audit_path,
        "eligibility": eligibility_path,
        "replication2_trial": output_directory / f"{REPORT_PREFIX}_replication2_trial_spectral_measurements.csv",
        "subject_feature": output_directory / f"{REPORT_PREFIX}_subject_feature_summary.csv",
        "subject_run": output_directory / f"{REPORT_PREFIX}_subject_run_summary.csv",
        "subject_primary": output_directory / f"{REPORT_PREFIX}_subject_primary_table.csv",
        "subject_qc": output_directory / f"{REPORT_PREFIX}_subject_qc_summary.csv",
        "central_peak": output_directory / f"{REPORT_PREFIX}_central_peak_summary.csv",
        "cohort_comparison": output_directory / f"{REPORT_PREFIX}_cohort_comparison.csv",
        "run_consistency": output_directory / f"{REPORT_PREFIX}_run_consistency_summary.csv",
        "trial_consistency": output_directory / f"{REPORT_PREFIX}_trial_consistency_summary.csv",
        "peak_distribution": output_directory / f"{REPORT_PREFIX}_peak_distribution.csv",
        "peak_relationship": output_directory / f"{REPORT_PREFIX}_peak_relationship.csv",
        "qc_relationship": output_directory / f"{REPORT_PREFIX}_qc_relationship.csv",
        "metadata": output_directory / f"{REPORT_PREFIX}_metadata.json",
        "primary_figure": output_directory / f"{REPORT_PREFIX}_primary_subjects.png",
        "comparison_figure": output_directory / f"{REPORT_PREFIX}_cohort_comparison.png",
        "run_figure": output_directory / f"{REPORT_PREFIX}_run_consistency.png",
        "band_figure": output_directory / f"{REPORT_PREFIX}_narrow_vs_mu.png",
        "peak_figure": output_directory / f"{REPORT_PREFIX}_central_peak_evidence.png",
    }
    for key, rows_to_write in (
        ("replication2_trial", current["trial_rows"]),
        ("subject_feature", all_subject_rows),
        ("subject_run", all_run_rows),
        ("subject_primary", subject_primary),
        ("subject_qc", all_qc_rows),
        ("central_peak", all_peak_rows),
        ("cohort_comparison", comparison_rows),
        ("run_consistency", run_rows),
        ("trial_consistency", trial_consistency_rows),
        ("peak_distribution", peak_distributions),
        ("peak_relationship", peak_relationships),
        ("qc_relationship", qc_relationships),
    ):
        write_csv(rows_to_write, paths[key])

    plot_primary_subjects(all_primary_rows, paths["primary_figure"])
    plot_cohort_comparison(all_primary_rows, paths["comparison_figure"])
    plot_run_consistency(run_rows, paths["run_figure"])
    plot_narrow_mu(all_subject_rows, paths["band_figure"])
    plot_peak_evidence(all_peak_rows, all_primary_rows, paths["peak_figure"])

    replication_2_qc = [
        row for row in current["qc_rows"] if int(row["subject"]) in set(eligible_subjects)
    ]
    historical_hashes = {
        f"subject{int(row['subject']):03d}_run{int(row['run']):02d}": row["source_sha256"]
        for row in historical["audit_rows"]
    }
    current_hashes = {
        f"subject{int(row['subject']):03d}_run{int(row['run']):02d}": row["source_sha256"]
        for row in audit_rows
        if row["source_sha256"]
    }
    metadata = {
        "schema_version": 1,
        "methodology_frozen_at_git_commit": full_config.methodology_frozen_at_git_commit,
        "full_cohort_fingerprint": full_cohort_fingerprint(full_config),
        "cohort_definitions": {
            "development": list(full_config.development_subjects),
            "replication_1": list(full_config.replication_1_subjects),
            "replication_2_requested": list(full_config.replication_2_subjects),
            "replication_2_eligible": eligible_subjects,
            "combined_replication_eligible": list(cohort_ids["combined_replication"]),
        },
        "replication_2_requested_run_count": len(audit_rows),
        "replication_2_usable_run_count": compatible_runs,
        "replication_2_task_trials": sum(int(row["task_trials"]) for row in replication_2_qc),
        "replication_2_fists_trials": sum(int(row["fists_trials"]) for row in replication_2_qc),
        "replication_2_feet_trials": sum(int(row["feet_trials"]) for row in replication_2_qc),
        "outcome_based_exclusions": 0,
        "full_tfr_arrays_persisted": False,
        "classification_performed": False,
        "csp_performed": False,
        "checkpoint_directory": display_path(checkpoint_directory),
        "checkpoint_count": len(payloads),
        "source_hashes": {**historical_hashes, **current_hashes},
        "artifacts": {
            key: display_path(path) for key, path in paths.items() if key != "metadata"
        },
    }
    with paths["metadata"].open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print("\nIndependent replication cohort 2 (subjects 21-109 only):")
    for row in comparison_rows:
        if row["cohort"] == "replication_2" and row["analysis_role"] == "primary":
            print(
                f"  {row['channel']}: {row['negative_subject_count']}/"
                f"{row['replication_subject_count']} negative; median "
                f"{float(row['median_across_subject_medians_percent']):.2f}%; "
                f"{row['replication_category']}"
            )
    print("Combined replication evidence (subjects 2-109, subject 1 excluded):")
    for row in comparison_rows:
        if row["cohort"] == "combined_replication" and row["analysis_role"] == "primary":
            print(
                f"  {row['channel']}: {row['negative_subject_count']}/"
                f"{row['replication_subject_count']} negative; median "
                f"{float(row['median_across_subject_medians_percent']):.2f}%"
            )
    print(f"Metadata: {display_path(paths['metadata'])}")


if __name__ == "__main__":
    main()
