"""Final spatial-personalization summaries and post-primary diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, spearmanr

from .cross_subject_decoding import file_sha256
from .decoding import PROJECT_ROOT, RUNS
from .spatial_personalization import (
    A_METHOD,
    load_spatial_personalization_config,
    validate_spatial_personalization_config,
)


DEFAULT_FINAL_SPATIAL_PERSONALIZATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "spatial_personalization_final.json"
)
FINAL_SPATIAL_PERSONALIZATION_FREEZE_COMMIT = "5d43e46"


def load_final_spatial_personalization_config(
    path: Path = DEFAULT_FINAL_SPATIAL_PERSONALIZATION_CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the final method and validate every frozen development dependency."""
    resolved = path.expanduser().resolve()
    final: dict[str, Any] = json.loads(resolved.read_text(encoding="utf-8"))
    if final["schema_version"] != 1 or final["study_stage"] != "final_spatial_personalization_method_frozen":
        raise ValueError("Expected the final spatial-personalization freeze.")
    initial_path = PROJECT_ROOT / final["initial_policy"]["path"]
    if file_sha256(initial_path) != final["initial_policy"]["sha256"]:
        raise RuntimeError("Initial spatial-personalization policy drifted.")
    initial = load_spatial_personalization_config(initial_path)
    validate_spatial_personalization_config(initial)
    implementation = PROJECT_ROOT / final["frozen_implementation"]["path"]
    if file_sha256(implementation) != final["frozen_implementation"]["sha256"]:
        raise RuntimeError("Frozen spatial-personalization implementation drifted.")
    for record in final["development_evidence"]["artifacts"].values():
        artifact = PROJECT_ROOT / record["path"]
        if file_sha256(artifact) != record["sha256"]:
            raise RuntimeError(f"Spatial-personalization development drift: {artifact}.")
    if final["cohorts"]["evaluation_eligible_subjects"] != initial["cohorts"][
        "evaluation_eligible_subjects"
    ]:
        raise RuntimeError("Final spatial-personalization cohort drifted.")
    return final, initial


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one tracked historical table."""
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def calibration_run_summary_rows(
    test_run_rows: Sequence[Mapping[str, object]],
    methods: Sequence[str],
    *,
    expected_subject_count: int,
) -> list[dict[str, object]]:
    """Summarize each calibration source run using one value per participant."""
    primary = [row for row in test_run_rows if not bool(row["qc_sensitivity"])]
    output: list[dict[str, object]] = []
    for method in methods:
        subject_scores: dict[int, dict[int, float]] = {}
        for subject in sorted({int(row["subject"]) for row in primary if row["method"] == method}):
            by_run: dict[int, float] = {}
            for calibration_run in RUNS:
                values = [
                    float(row["balanced_accuracy"])
                    for row in primary
                    if row["method"] == method
                    and int(row["subject"]) == subject
                    and int(row["calibration_run"]) == calibration_run
                ]
                if len(values) != 2:
                    raise RuntimeError("Calibration-run summary lacks two test runs.")
                by_run[calibration_run] = float(np.mean(values))
            subject_scores[subject] = by_run
        if len(subject_scores) != expected_subject_count:
            raise RuntimeError("Calibration-run summary lacks participant coverage.")
        arrays = [
            np.asarray([subject_scores[subject][run] for subject in sorted(subject_scores)])
            for run in RUNS
        ]
        statistic, p_value = friedmanchisquare(*arrays)
        for run, values in zip(RUNS, arrays, strict=True):
            output.append(
                {
                    "method": method,
                    "calibration_run": run,
                    "participant_count": expected_subject_count,
                    "median_balanced_accuracy": float(np.median(values)),
                    "q25_balanced_accuracy": float(np.percentile(values, 25)),
                    "q75_balanced_accuracy": float(np.percentile(values, 75)),
                    "minimum_balanced_accuracy": float(np.min(values)),
                    "maximum_balanced_accuracy": float(np.max(values)),
                    "friedman_statistic_across_calibration_runs": float(statistic),
                    "friedman_two_sided_p_across_calibration_runs": float(p_value),
                    "analysis_role": "post_primary_exploratory_unadjusted",
                }
            )
    return output


def historical_context_rows(
    participant_rows: Sequence[Mapping[str, object]],
    threshold_rows: Sequence[Mapping[str, object]],
    within_rows: Sequence[Mapping[str, object]],
    methods: Sequence[str],
) -> list[dict[str, object]]:
    """Compare current scores with same-geometry threshold and descriptive within history."""
    current = {
        (int(row["subject"]), str(row["method"])): float(
            row["primary_mean_scenario_test_run_balanced_accuracy"]
        )
        for row in participant_rows
        if not bool(row["qc_sensitivity"])
    }
    threshold = {
        int(row["subject"]): float(row["primary_mean_test_run_balanced_accuracy"])
        for row in threshold_rows
        if row["method"] == "threshold"
        and int(row["calibration_size"]) == 14
        and row["qc_sensitivity"] == "False"
    }
    within = {
        int(row["subject"]): float(row["primary_mean_fold_balanced_accuracy"])
        for row in within_rows
        if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"
    }
    if set(threshold) != set(within):
        raise RuntimeError("Historical threshold and within-subject cohorts differ.")
    output: list[dict[str, object]] = []
    for method in methods:
        values = np.asarray([current[(subject, method)] for subject in sorted(threshold)])
        for reference_name, reference, geometry in (
            ("historical_14_label_threshold", threshold, "same_calibration_test_geometry"),
            ("historical_within_subject_CSP", within, "descriptive_different_training_geometry"),
        ):
            reference_values = np.asarray([reference[subject] for subject in sorted(reference)])
            differences = values - reference_values
            output.append(
                {
                    "method": method,
                    "reference": reference_name,
                    "participant_count": len(values),
                    "method_median_balanced_accuracy": float(np.median(values)),
                    "reference_median_balanced_accuracy": float(np.median(reference_values)),
                    "difference_of_medians": float(np.median(values) - np.median(reference_values)),
                    "median_paired_method_minus_reference": float(np.median(differences)),
                    "q25_paired_method_minus_reference": float(np.percentile(differences, 25)),
                    "q75_paired_method_minus_reference": float(np.percentile(differences, 75)),
                    "method_better_count": int(np.sum(differences > 0)),
                    "reference_better_count": int(np.sum(differences < 0)),
                    "tie_count": int(np.sum(differences == 0)),
                    "comparison_geometry": geometry,
                    "analysis_role": "post_primary_historical_context",
                }
            )
    return output


def poor_zero_shot_subgroup_row(
    paired_c_minus_b: Sequence[Mapping[str, object]],
    participant_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Describe C-minus-B among pre-defined poor zero-shot participants."""
    zero = {
        int(row["subject"]): float(row["primary_mean_scenario_test_run_balanced_accuracy"])
        for row in participant_rows
        if row["method"] == A_METHOD and not bool(row["qc_sensitivity"])
    }
    selected = [
        row
        for row in paired_c_minus_b
        if zero[int(row["subject"])] <= 0.50
    ]
    gains = np.asarray([float(row["candidate_minus_reference"]) for row in selected])
    candidate = np.asarray([float(row["candidate_balanced_accuracy"]) for row in selected])
    return {
        "subgroup": "historical_zero_shot_CSP_at_or_below_0_50",
        "participant_count": len(selected),
        "median_C_minus_B": float(np.median(gains)),
        "q25_C_minus_B": float(np.percentile(gains, 25)),
        "q75_C_minus_B": float(np.percentile(gains, 75)),
        "C_better_count": int(np.sum(gains > 0)),
        "B_better_count": int(np.sum(gains < 0)),
        "tie_count": int(np.sum(gains == 0)),
        "median_C_balanced_accuracy": float(np.median(candidate)),
        "analysis_role": "post_primary_exploratory_subgroup",
        "regression_to_mean_caution": True,
    }


def _spearman_row(
    name: str,
    predictor: Sequence[float],
    outcome: Sequence[float],
) -> dict[str, object]:
    x = np.asarray(predictor, dtype=float)
    y = np.asarray(outcome, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 3:
        rho, p_value = np.nan, np.nan
    else:
        rho, p_value = spearmanr(x[valid], y[valid])
    return {
        "relationship": name,
        "participant_count": int(np.sum(valid)),
        "spearman_rho": float(rho),
        "two_sided_unadjusted_p": float(p_value),
        "analysis_role": "post_primary_exploratory_unadjusted",
    }


def exploratory_relationship_rows(
    paired_c_minus_b: Sequence[Mapping[str, object]],
    participant_rows: Sequence[Mapping[str, object]],
    physiology_rows: Sequence[Mapping[str, object]],
    within_rows: Sequence[Mapping[str, object]],
    stability_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return only pre-listed, post-primary relationships with C-minus-B gain."""
    gains = {
        int(row["subject"]): float(row["candidate_minus_reference"])
        for row in paired_c_minus_b
    }
    zero = {
        int(row["subject"]): float(row["primary_mean_scenario_test_run_balanced_accuracy"])
        for row in participant_rows
        if row["method"] == A_METHOD and not bool(row["qc_sensitivity"])
    }
    physiology = {int(row["subject"]): row for row in physiology_rows}
    within = {
        int(row["subject"]): float(row["primary_mean_fold_balanced_accuracy"])
        for row in within_rows
        if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"
    }
    stability = {
        int(row["subject"]): float(row["mean_pairwise_subspace_similarity"])
        for row in stability_rows
    }
    subjects = sorted(gains)

    def historical_value(subject: int, field: str) -> float:
        text = physiology[subject].get(field, "")
        return float(text) if text not in ("", None) else float("nan")

    outcome = [gains[subject] for subject in subjects]
    return [
        _spearman_row("zero_shot_A_vs_C_minus_B", [zero[s] for s in subjects], outcome),
        _spearman_row(
            "historical_C3_fixed_fists_ERD_vs_C_minus_B",
            [historical_value(s, "c3_fixed_fists_erd_median_percent") for s in subjects],
            outcome,
        ),
        _spearman_row(
            "historical_C4_fixed_fists_ERD_vs_C_minus_B",
            [historical_value(s, "c4_fixed_fists_erd_median_percent") for s in subjects],
            outcome,
        ),
        _spearman_row(
            "historical_within_subject_CSP_vs_C_minus_B",
            [within[s] for s in subjects],
            outcome,
        ),
        _spearman_row(
            "target_CSP_subspace_stability_vs_C_minus_B",
            [stability[s] for s in subjects],
            outcome,
        ),
    ]
