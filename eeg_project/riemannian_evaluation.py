"""Frozen Riemannian evaluation configuration and descriptive helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .decoding import PROJECT_ROOT, RUNS
from .riemannian_decoding import file_sha256, load_riemannian_config


DEFAULT_FINAL_RIEMANNIAN_CONFIG_PATH = PROJECT_ROOT / "config" / "riemannian_decoding_final.json"
FINAL_RIEMANNIAN_FREEZE_COMMIT = "c2f132f"


def load_final_riemannian_config(path: Path = DEFAULT_FINAL_RIEMANNIAN_CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the final method and every pre-evaluation dependency."""
    final: dict[str, Any] = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if final["study_stage"] != "final_riemannian_method_frozen":
        raise RuntimeError("Evaluation requires the final Riemannian method freeze.")
    if final["evaluation_classifier_outcomes_inspected_at_final_freeze"] or final["evaluation_subject_EEG_loaded_at_final_freeze"]:
        raise RuntimeError("The final Riemannian chronology is already open.")
    if not final["evaluation_unlocked_after_this_freeze_commit"]:
        raise RuntimeError("The final Riemannian configuration does not unlock evaluation.")
    initial_record = final["initial_policy"]
    initial_path = PROJECT_ROOT / initial_record["path"]
    if file_sha256(initial_path) != initial_record["sha256"]:
        raise RuntimeError("Initial Riemannian policy drifted.")
    initial = load_riemannian_config(initial_path)
    core = final["frozen_method_implementation"]
    if file_sha256(PROJECT_ROOT / core["path"]) != core["sha256"]:
        raise RuntimeError("Frozen Riemannian implementation drifted.")
    for record in final["development_evidence"]["artifacts"].values():
        if file_sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"Riemannian development artifact drift: {record['path']}.")
    if final["cohorts"]["evaluation_eligible_subjects"] != initial["cohorts"]["evaluation_eligible_subjects"]:
        raise RuntimeError("Final Riemannian evaluation cohort drifted.")
    if final["selected_method"]["classifier"]["class"] != "LinearDiscriminantAnalysis":
        raise RuntimeError("Unexpected Riemannian final classifier.")
    return final, initial


def run_summary_rows(run_rows: Sequence[Mapping[str, object]], *, expected_subject_count: int) -> list[dict[str, object]]:
    """Summarize primary run scores with people as the observational unit."""
    primary = [row for row in run_rows if not bool(row["qc_sensitivity"])]
    output: list[dict[str, object]] = []
    for run in RUNS:
        values = np.asarray([float(row["balanced_accuracy"]) for row in primary if int(row["test_run"]) == run])
        if values.size != expected_subject_count:
            raise RuntimeError("Riemannian run summary lacks one score per participant.")
        output.append({
            "model": "riemannian_ledoit_wolf_tangent_lda", "test_run": run,
            "participant_count": expected_subject_count, "median_balanced_accuracy": float(np.median(values)),
            "q25_balanced_accuracy": float(np.percentile(values, 25)), "q75_balanced_accuracy": float(np.percentile(values, 75)),
        })
    return output


def exploratory_variability_rows(riemannian_rows: Sequence[Mapping[str, object]], csp_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Post-primary matched variability and poor-CSP descriptive diagnostics."""
    riem = {int(row["subject"]): row for row in riemannian_rows if not bool(row["qc_sensitivity"])}
    csp = {int(row["subject"]): row for row in csp_rows if row["model"] == "csp_4_empirical"}
    subjects = sorted(riem)
    gains = np.asarray([float(riem[s]["primary_mean_run_balanced_accuracy"]) - float(csp[s]["primary_mean_run_balanced_accuracy"]) for s in subjects])
    poor = np.asarray([s for s in subjects if float(csp[s]["primary_mean_run_balanced_accuracy"]) <= 0.5])
    poor_gains = np.asarray([gains[subjects.index(int(s))] for s in poor])
    riem_range = np.asarray([float(riem[s]["participant_run_range"]) for s in subjects])
    csp_range = np.asarray([max(float(csp[s][f"balanced_accuracy_run_{run}"]) for run in RUNS) - min(float(csp[s][f"balanced_accuracy_run_{run}"]) for run in RUNS) for s in subjects])
    rho, p = spearmanr(csp_range, gains)
    return [
        {"analysis": "poor_historical_CSP_subjects", "participant_count": int(poor.size), "median_Riemannian_minus_CSP": float(np.median(poor_gains)), "Riemannian_better_count": int(np.sum(poor_gains > 0)), "CSP_better_count": int(np.sum(poor_gains < 0)), "analysis_role": "post_primary_exploratory"},
        {"analysis": "participant_run_range", "participant_count": len(subjects), "median_Riemannian_run_range": float(np.median(riem_range)), "median_CSP_run_range": float(np.median(csp_range)), "median_Riemannian_minus_CSP_run_range": float(np.median(riem_range - csp_range)), "Spearman_CSP_range_vs_gain_rho": float(rho), "Spearman_CSP_range_vs_gain_p": float(p), "analysis_role": "post_primary_exploratory_unadjusted"},
    ]
