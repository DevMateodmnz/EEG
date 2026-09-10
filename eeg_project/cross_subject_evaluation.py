"""Final zero-shot cross-subject evaluation summaries and historical comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .cross_subject_decoding import (
    DEFAULT_CROSS_SUBJECT_CONFIG_PATH,
    MODEL_NAMES,
    file_sha256,
    load_cross_subject_config,
)
from .decoding import PROJECT_ROOT, RUNS, paired_wilcoxon


DEFAULT_FINAL_CROSS_SUBJECT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "cross_subject_decoding_final.json"
)


def load_final_cross_subject_config(
    path: Path = DEFAULT_FINAL_CROSS_SUBJECT_CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the final freeze and validate its initial policy and development evidence."""
    resolved = path.expanduser().resolve()
    final: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    if final["study_stage"] != "final_cross_subject_method_frozen":
        raise RuntimeError("Evaluation requires the final cross-subject method freeze.")
    if final["evaluation_classifier_outcomes_inspected_at_final_freeze"]:
        raise RuntimeError("Final chronology says evaluation outcomes were already inspected.")
    if final["evaluation_subject_EEG_loaded_at_final_freeze"]:
        raise RuntimeError("Final chronology says evaluation EEG was already loaded.")
    if not final["evaluation_unlocked_after_this_freeze_commit"]:
        raise RuntimeError("The final cross-subject configuration does not unlock evaluation.")
    initial_record = final["initial_policy"]
    initial_path = PROJECT_ROOT / initial_record["path"]
    if file_sha256(initial_path) != initial_record["sha256"]:
        raise RuntimeError("Initial cross-subject policy drifted after its freeze.")
    initial = load_cross_subject_config(initial_path)
    core_record = final["frozen_method_implementation"]
    core_path = PROJECT_ROOT / core_record["path"]
    if file_sha256(core_path) != core_record["sha256"]:
        raise RuntimeError("Frozen cross-subject fitting implementation drifted.")
    for record in final["development_evidence"]["artifacts"].values():
        artifact = PROJECT_ROOT / record["path"]
        if file_sha256(artifact) != record["sha256"]:
            raise RuntimeError(f"Cross-subject development artifact drift: {artifact}.")
    if final["cohorts"]["training_subjects"] != initial["cohorts"]["development_subjects"]:
        raise RuntimeError("Final training subjects differ from the initial policy.")
    for key in (
        "evaluation_requested_subjects",
        "evaluation_eligible_subjects",
        "technically_incompatible_subjects",
    ):
        if final["cohorts"][key] != initial["cohorts"][key]:
            raise RuntimeError(f"Final cross-subject cohort field {key} drifted.")
    if tuple(final["selected_models"]) != MODEL_NAMES:
        raise RuntimeError("Final cross-subject models differ from the inherited pair.")
    for key in ("metrics", "evaluation_success_criteria", "paired_model_criteria", "group_inference"):
        if final[key] != initial[key]:
            raise RuntimeError(f"Final cross-subject {key} differs from the initial freeze.")
    return final, initial


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a tracked CSV table."""
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def evaluation_run_summary_rows(
    run_rows: Sequence[Mapping[str, object]],
    *,
    expected_subject_count: int,
) -> list[dict[str, object]]:
    """Summarize matched participant run scores for both frozen models."""
    output: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        for run in RUNS:
            selected = [
                row
                for row in run_rows
                if row["model"] == model and int(row["test_run"]) == run
            ]
            if (
                len(selected) != expected_subject_count
                or len({int(row["subject"]) for row in selected}) != expected_subject_count
            ):
                raise RuntimeError("Run summary lacks one row per evaluation participant.")
            scores = np.asarray([float(row["balanced_accuracy"]) for row in selected])
            output.append(
                {
                    "model": model,
                    "run": run,
                    "participant_count": expected_subject_count,
                    "median_balanced_accuracy": float(np.median(scores)),
                    "q25_balanced_accuracy": float(np.percentile(scores, 25)),
                    "q75_balanced_accuracy": float(np.percentile(scores, 75)),
                    "minimum_balanced_accuracy": float(np.min(scores)),
                    "maximum_balanced_accuracy": float(np.max(scores)),
                    "above_0_5_count": int(np.sum(scores > 0.5)),
                    "median_fists_recall": float(
                        np.median([float(row["fists_recall"]) for row in selected])
                    ),
                    "median_feet_recall": float(
                        np.median([float(row["feet_recall"]) for row in selected])
                    ),
                }
            )
    return output


def aggregate_confusion_rows(
    subject_rows: Sequence[Mapping[str, object]],
    *,
    expected_subject_count: int,
) -> list[dict[str, object]]:
    """Pool trial confusion counts only after participant-level scores are finalized."""
    fields = (
        "true_fists_predicted_fists",
        "true_fists_predicted_feet",
        "true_feet_predicted_fists",
        "true_feet_predicted_feet",
    )
    output: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        selected = [row for row in subject_rows if row["model"] == model]
        if len(selected) != expected_subject_count:
            raise RuntimeError("Confusion aggregation lacks participant rows.")
        counts = {field: sum(int(row[field]) for row in selected) for field in fields}
        fists_total = counts[fields[0]] + counts[fields[1]]
        feet_total = counts[fields[2]] + counts[fields[3]]
        output.append(
            {
                "model": model,
                **counts,
                "fists_recall": counts[fields[0]] / fists_total,
                "feet_recall": counts[fields[3]] / feet_total,
                "trial_count": fists_total + feet_total,
            }
        )
    return output


def within_cross_comparison_rows(
    cross_subject_rows: Sequence[Mapping[str, object]],
    historical_within_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Join matched historical within-subject and finalized cross-subject scores."""
    historical = {
        (int(row["subject"]), row["model"]): float(
            row["primary_mean_fold_balanced_accuracy"]
        )
        for row in historical_within_rows
        if row["qc_sensitivity"] == "False" and row["model"] in MODEL_NAMES
    }
    output: list[dict[str, object]] = []
    for row in cross_subject_rows:
        subject = int(row["subject"])
        model = str(row["model"])
        key = (subject, model)
        if key not in historical:
            raise RuntimeError(f"Missing historical within-subject comparator for {key}.")
        within = historical[key]
        cross = float(row["primary_mean_run_balanced_accuracy"])
        output.append(
            {
                "subject": subject,
                "model": model,
                "within_subject_balanced_accuracy": within,
                "cross_subject_balanced_accuracy": cross,
                "cross_minus_within_balanced_accuracy": cross - within,
                "within_minus_cross_balanced_accuracy": within - cross,
                "analysis_role": "post_primary_exploratory_matched_comparison",
            }
        )
    return output


def within_cross_summary_rows(
    comparison_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarize the participant-matched cost of removing personal training."""
    output: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        selected = [row for row in comparison_rows if row["model"] == model]
        if len(selected) != 86:
            raise RuntimeError("Within-versus-cross comparison must contain 86 participants.")
        within = np.asarray([float(row["within_subject_balanced_accuracy"]) for row in selected])
        cross = np.asarray([float(row["cross_subject_balanced_accuracy"]) for row in selected])
        differences = cross - within
        output.append(
            {
                "model": model,
                "participant_count": len(selected),
                "median_within_subject_balanced_accuracy": float(np.median(within)),
                "median_cross_subject_balanced_accuracy": float(np.median(cross)),
                "median_cross_minus_within_balanced_accuracy": float(np.median(differences)),
                "q25_cross_minus_within_balanced_accuracy": float(
                    np.percentile(differences, 25)
                ),
                "q75_cross_minus_within_balanced_accuracy": float(
                    np.percentile(differences, 75)
                ),
                "cross_better_count": int(np.sum(differences > 0)),
                "within_better_count": int(np.sum(differences < 0)),
                "tie_count": int(np.sum(differences == 0)),
                "paired_wilcoxon_two_sided_p": paired_wilcoxon(cross, within),
                "analysis_role": "post_primary_exploratory_matched_comparison",
            }
        )
    return output


def _spearman_row(
    model: str,
    relationship: str,
    x: Sequence[float],
    y: Sequence[float],
) -> dict[str, object]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.size != y_values.size or x_values.size < 3:
        raise RuntimeError(f"Invalid relationship inputs for {relationship}.")
    statistic = spearmanr(x_values, y_values)
    return {
        "model": model,
        "relationship": relationship,
        "participant_count": int(x_values.size),
        "spearman_rho": float(statistic.statistic),
        "two_sided_unadjusted_p": float(statistic.pvalue),
        "analysis_role": "post_primary_exploratory_unadjusted",
    }


def exploratory_relationship_rows(
    cross_subject_rows: Sequence[Mapping[str, object]],
    physiology_rows: Sequence[Mapping[str, str]],
    reliability_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Calculate only the few post-primary relationships named in the freeze."""
    physiology = {int(row["subject"]): row for row in physiology_rows}
    reliability = {int(row["subject"]): row for row in reliability_rows}
    output: list[dict[str, object]] = []
    for model in MODEL_NAMES:
        selected = sorted(
            (row for row in cross_subject_rows if row["model"] == model),
            key=lambda row: int(row["subject"]),
        )
        if len(selected) != 86:
            raise RuntimeError("Exploratory join must retain all 86 evaluation participants.")
        subjects = [int(row["subject"]) for row in selected]
        accuracy = [float(row["primary_mean_run_balanced_accuracy"]) for row in selected]
        for channel, field in (
            ("C3", "c3_fixed_fists_erd_median_percent"),
            ("C4", "c4_fixed_fists_erd_median_percent"),
        ):
            output.append(
                _spearman_row(
                    model,
                    f"historical_{channel}_fists_12_13_hz_ERD_vs_cross_accuracy",
                    [float(physiology[subject][field]) for subject in subjects],
                    accuracy,
                )
            )
        output.extend(
            (
                _spearman_row(
                    model,
                    "historical_within_subject_accuracy_vs_cross_accuracy",
                    [
                        float(reliability[subject]["historical_primary_mean_balanced_accuracy"])
                        for subject in subjects
                    ],
                    accuracy,
                ),
                _spearman_row(
                    model,
                    "historical_within_subject_run_range_vs_cross_accuracy",
                    [
                        float(reliability[subject]["run_balanced_accuracy_range"])
                        for subject in subjects
                    ],
                    accuracy,
                ),
                _spearman_row(
                    model,
                    "historical_within_subject_runs_above_chance_vs_cross_accuracy",
                    [
                        float(reliability[subject]["runs_above_0_5_count"])
                        for subject in subjects
                    ],
                    accuracy,
                ),
            )
        )
    return output
