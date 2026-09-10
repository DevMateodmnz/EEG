"""Final calibration evaluation summaries and post-primary diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, spearmanr

from .cross_subject_decoding import file_sha256
from .decoding import PROJECT_ROOT, RUNS
from .target_calibration import (
    CALIBRATION_SIZES,
    DEFAULT_CALIBRATION_CONFIG_PATH,
    load_calibration_config,
)


DEFAULT_FINAL_CALIBRATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "minimal_target_calibration_final.json"
)


def load_final_calibration_config(
    path: Path = DEFAULT_FINAL_CALIBRATION_CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the final freeze and validate its initial policy/development evidence."""
    resolved = path.expanduser().resolve()
    final: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    if final["study_stage"] != "final_minimal_target_calibration_method_frozen":
        raise RuntimeError("Evaluation requires the final calibration-method freeze.")
    if final["evaluation_calibration_outcomes_inspected_at_final_freeze"]:
        raise RuntimeError("Final chronology says calibration outcomes were already open.")
    if final["evaluation_subject_EEG_loaded_for_calibration_at_final_freeze"]:
        raise RuntimeError("Final chronology says evaluation EEG was already loaded.")
    if not final["evaluation_unlocked_after_this_freeze_commit"]:
        raise RuntimeError("The final calibration freeze does not unlock evaluation.")
    initial_record = final["initial_policy"]
    initial_path = PROJECT_ROOT / initial_record["path"]
    if file_sha256(initial_path) != initial_record["sha256"]:
        raise RuntimeError("Initial calibration policy drifted after its freeze.")
    initial = load_calibration_config(initial_path)
    core_record = final["frozen_calibration_implementation"]
    if file_sha256(PROJECT_ROOT / core_record["path"]) != core_record["sha256"]:
        raise RuntimeError("Frozen calibration implementation drifted.")
    for record in final["development_evidence"]["artifacts"].values():
        if file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"Calibration development artifact drift: {record['path']}.")
    cohorts = final["cohorts"]
    if cohorts["source_training_subjects"] != initial["cohorts"]["development_subjects"]:
        raise RuntimeError("Final source-training cohort differs from initial policy.")
    for key in (
        "evaluation_requested_subjects",
        "evaluation_eligible_subjects",
        "technically_incompatible_subjects",
    ):
        if cohorts[key] != initial["cohorts"][key]:
            raise RuntimeError(f"Final calibration cohort field {key} drifted.")
    if final["selected_calibration_method"]["name"] != "threshold":
        raise RuntimeError("Unexpected final calibration method.")
    if final["calibration_curve"]["ordered_total_trials"] != list(CALIBRATION_SIZES):
        raise RuntimeError("Final calibration curve differs from initial policy.")
    if final["convincing_benefit_criteria"] != initial["convincing_benefit_criteria"]:
        raise RuntimeError("Final calibration success criteria drifted.")
    return final, initial


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a tracked CSV artifact."""
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def matched_gain_rows(
    participant_rows: Sequence[Mapping[str, object]],
    *,
    method: str,
) -> list[dict[str, object]]:
    """Match every calibrated participant score to size-zero on identical scenarios."""
    primary = [
        row
        for row in participant_rows
        if row["method"] == method and not bool(row["qc_sensitivity"])
    ]
    zero = {
        int(row["subject"]): float(row["primary_mean_test_run_balanced_accuracy"])
        for row in primary
        if int(row["calibration_size"]) == 0
    }
    output: list[dict[str, object]] = []
    for row in primary:
        subject = int(row["subject"])
        size = int(row["calibration_size"])
        calibrated = float(row["primary_mean_test_run_balanced_accuracy"])
        output.append(
            {
                "subject": subject,
                "method": method,
                "calibration_size": size,
                "zero_balanced_accuracy": zero[subject],
                "calibrated_balanced_accuracy": calibrated,
                "gain_over_matched_zero": calibrated - zero[subject],
            }
        )
    return output


def calibration_run_summary_rows(
    test_run_rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    expected_subject_count: int,
) -> list[dict[str, object]]:
    """Describe whether the identity of the one calibration run changes performance."""
    primary = [
        row
        for row in test_run_rows
        if row["method"] == method and not bool(row["qc_sensitivity"])
    ]
    output: list[dict[str, object]] = []
    for size in CALIBRATION_SIZES:
        by_run: dict[int, dict[int, float]] = {}
        for calibration_run in RUNS:
            participant_scores: dict[int, float] = {}
            for subject in sorted({int(row["subject"]) for row in primary}):
                selected = [
                    float(row["balanced_accuracy"])
                    for row in primary
                    if int(row["subject"]) == subject
                    and int(row["calibration_size"]) == size
                    and int(row["calibration_run"]) == calibration_run
                ]
                if len(selected) != 2:
                    raise RuntimeError("Calibration-run summary lacks two test runs.")
                participant_scores[subject] = float(np.mean(selected))
            if len(participant_scores) != expected_subject_count:
                raise RuntimeError("Calibration-run summary lacks participants.")
            by_run[calibration_run] = participant_scores
        subjects = sorted(by_run[RUNS[0]])
        arrays = [np.asarray([by_run[run][subject] for subject in subjects]) for run in RUNS]
        if all(np.array_equal(arrays[0], values) for values in arrays[1:]):
            friedman_statistic = 0.0
            friedman_p = 1.0
        else:
            friedman = friedmanchisquare(*arrays)
            friedman_statistic = float(friedman.statistic)
            friedman_p = float(friedman.pvalue)
        for run, values in zip(RUNS, arrays, strict=True):
            output.append(
                {
                    "method": method,
                    "calibration_size": size,
                    "calibration_run": run,
                    "participant_count": expected_subject_count,
                    "median_balanced_accuracy": float(np.median(values)),
                    "q25_balanced_accuracy": float(np.percentile(values, 25)),
                    "q75_balanced_accuracy": float(np.percentile(values, 75)),
                    "minimum_balanced_accuracy": float(np.min(values)),
                    "maximum_balanced_accuracy": float(np.max(values)),
                    "friedman_statistic_across_calibration_runs": friedman_statistic,
                    "friedman_two_sided_p_across_calibration_runs": friedman_p,
                    "analysis_role": "post_primary_exploratory_unadjusted",
                }
            )
    return output


def subgroup_gain_rows(
    gain_rows: Sequence[Mapping[str, object]],
    historical_within_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Summarize poor zero-shot and historically persistent low performers."""
    within = {
        int(row["subject"]): float(row["primary_mean_fold_balanced_accuracy"])
        for row in historical_within_rows
        if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"
    }
    output: list[dict[str, object]] = []
    for size in CALIBRATION_SIZES[1:]:
        selected_size = [row for row in gain_rows if int(row["calibration_size"]) == size]
        groups = {
            "all_evaluation_participants": selected_size,
            "poor_zero_shot_at_or_below_0_50": [
                row for row in selected_size if float(row["zero_balanced_accuracy"]) <= 0.5
            ],
            "persistent_low_within_subject_at_or_below_0_50": [
                row for row in selected_size if within[int(row["subject"])] <= 0.5
            ],
        }
        for name, rows in groups.items():
            gains = np.asarray([float(row["gain_over_matched_zero"]) for row in rows])
            scores = np.asarray([float(row["calibrated_balanced_accuracy"]) for row in rows])
            if gains.size == 0:
                continue
            output.append(
                {
                    "subgroup": name,
                    "calibration_size": size,
                    "participant_count": int(gains.size),
                    "median_calibrated_balanced_accuracy": float(np.median(scores)),
                    "median_gain_over_matched_zero": float(np.median(gains)),
                    "q25_gain_over_matched_zero": float(np.percentile(gains, 25)),
                    "q75_gain_over_matched_zero": float(np.percentile(gains, 75)),
                    "improved_count": int(np.sum(gains > 0)),
                    "worsened_count": int(np.sum(gains < 0)),
                    "tie_count": int(np.sum(gains == 0)),
                    "improved_fraction": float(np.mean(gains > 0)),
                    "analysis_role": "post_primary_exploratory",
                }
            )
    return output


def _spearman_row(
    relationship: str,
    size: int,
    x: Sequence[float],
    y: Sequence[float],
) -> dict[str, object]:
    statistic = spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return {
        "relationship": relationship,
        "calibration_size": size,
        "participant_count": len(x),
        "spearman_rho": float(statistic.statistic),
        "two_sided_unadjusted_p": float(statistic.pvalue),
        "analysis_role": "post_primary_exploratory_unadjusted",
    }


def exploratory_relationship_rows(
    gain_rows: Sequence[Mapping[str, object]],
    physiology_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Calculate only the pre-listed zero-shot and ERD gain relationships."""
    physiology = {int(row["subject"]): row for row in physiology_rows}
    output: list[dict[str, object]] = []
    for size in CALIBRATION_SIZES[1:]:
        selected = sorted(
            (row for row in gain_rows if int(row["calibration_size"]) == size),
            key=lambda row: int(row["subject"]),
        )
        subjects = [int(row["subject"]) for row in selected]
        gains = [float(row["gain_over_matched_zero"]) for row in selected]
        output.append(
            _spearman_row(
                "historical_zero_shot_CSP_accuracy_vs_calibration_gain",
                size,
                [float(row["zero_balanced_accuracy"]) for row in selected],
                gains,
            )
        )
        for channel, field in (
            ("C3", "c3_fixed_fists_erd_median_percent"),
            ("C4", "c4_fixed_fists_erd_median_percent"),
        ):
            output.append(
                _spearman_row(
                    f"historical_{channel}_fists_12_13_hz_ERD_vs_calibration_gain",
                    size,
                    [float(physiology[subject][field]) for subject in subjects],
                    gains,
                )
            )
    return output


def incremental_gain_rows(
    participant_rows: Sequence[Mapping[str, object]],
    *,
    method: str,
) -> list[dict[str, object]]:
    """Describe added value from each successive nested calibration increment."""
    lookup = {
        (int(row["subject"]), int(row["calibration_size"])): float(
            row["primary_mean_test_run_balanced_accuracy"]
        )
        for row in participant_rows
        if row["method"] == method and not bool(row["qc_sensitivity"])
    }
    subjects = sorted({subject for subject, _ in lookup})
    output: list[dict[str, object]] = []
    for previous, current in zip(CALIBRATION_SIZES[:-1], CALIBRATION_SIZES[1:], strict=True):
        differences = np.asarray(
            [lookup[(subject, current)] - lookup[(subject, previous)] for subject in subjects]
        )
        output.append(
            {
                "previous_calibration_size": previous,
                "current_calibration_size": current,
                "added_labeled_trials": current - previous,
                "participant_count": len(subjects),
                "median_incremental_gain": float(np.median(differences)),
                "q25_incremental_gain": float(np.percentile(differences, 25)),
                "q75_incremental_gain": float(np.percentile(differences, 75)),
                "improved_count": int(np.sum(differences > 0)),
                "worsened_count": int(np.sum(differences < 0)),
                "tie_count": int(np.sum(differences == 0)),
                "improved_fraction": float(np.mean(differences > 0)),
                "analysis_role": "post_primary_exploratory_saturation",
            }
        )
    return output


def historical_within_comparison_rows(
    participant_rows: Sequence[Mapping[str, object]],
    historical_within_rows: Sequence[Mapping[str, str]],
    *,
    method: str,
) -> list[dict[str, object]]:
    """Compare curve points descriptively with the non-identical within-subject study."""
    within = {
        int(row["subject"]): float(row["primary_mean_fold_balanced_accuracy"])
        for row in historical_within_rows
        if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"
    }
    output: list[dict[str, object]] = []
    for size in CALIBRATION_SIZES:
        selected = sorted(
            (
                row
                for row in participant_rows
                if row["method"] == method
                and int(row["calibration_size"]) == size
                and not bool(row["qc_sensitivity"])
            ),
            key=lambda row: int(row["subject"]),
        )
        calibrated = np.asarray(
            [float(row["primary_mean_test_run_balanced_accuracy"]) for row in selected]
        )
        historical = np.asarray([within[int(row["subject"])] for row in selected])
        difference = calibrated - historical
        output.append(
            {
                "calibration_size": size,
                "participant_count": len(selected),
                "median_historical_within_subject_balanced_accuracy": float(
                    np.median(historical)
                ),
                "median_calibrated_balanced_accuracy": float(np.median(calibrated)),
                "difference_of_medians_calibrated_minus_within": float(
                    np.median(calibrated) - np.median(historical)
                ),
                "median_paired_calibrated_minus_within": float(np.median(difference)),
                "calibrated_better_count": int(np.sum(difference > 0)),
                "within_better_count": int(np.sum(difference < 0)),
                "tie_count": int(np.sum(difference == 0)),
                "analysis_role": "post_primary_descriptive_nonidentical_designs",
            }
        )
    return output
