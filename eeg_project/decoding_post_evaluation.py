"""Post-evaluation historical relationships that cannot affect decoder fitting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .decoding import PROJECT_ROOT


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def _spearman_row(
    relationship: str,
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, object]:
    selected = [row for row in rows if row[field] != ""]
    x = np.array([float(row[field]) for row in selected])
    y = np.array([float(row["csp_balanced_accuracy"]) for row in selected])
    statistic, p_value = spearmanr(x, y)
    return {
        "relationship": relationship,
        "subject_count": len(selected),
        "spearman_rho": float(statistic),
        "two_sided_unadjusted_p": float(p_value),
        "analysis_role": "post_evaluation_exploratory_only",
    }


def safe_exploratory_relationship_rows(
    csp_subject_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Join historical measures while retaining unavailable peaks as missing."""
    score_lookup = {
        int(row["subject"]): float(row["primary_mean_fold_balanced_accuracy"])
        for row in csp_subject_rows
        if not row["qc_sensitivity"]
    }
    primary = {
        int(row["subject"]): row
        for row in _read_csv(
            PROJECT_ROOT / "docs/assets/subjects01-109_full_replication_subject_primary_table.csv"
        )
        if row["technical_eligibility"] == "eligible"
    }
    peaks = {
        int(row["subject"]): row
        for row in _read_csv(
            PROJECT_ROOT / "docs/assets/subjects01-109_full_replication_central_peak_summary.csv"
        )
    }
    imf = {
        int(row["subject"]): row
        for row in _read_csv(
            PROJECT_ROOT / "docs/assets/individual_mu_heldout_evaluation_peak_subject_summary.csv"
        )
    }
    joined: list[dict[str, object]] = []
    for subject, score in sorted(score_lookup.items()):
        historical = primary[subject]
        peak = peaks[subject]
        imf_row = imf.get(subject)
        peak_available = peak["peak_frequency_hz"] != ""
        imf_available = imf_row is not None and imf_row["method_available"] == "True"
        joined.append(
            {
                "subject": subject,
                "csp_balanced_accuracy": score,
                "c4_fixed_fists_erd_median_percent": float(historical["c4_median_percent_change"]),
                "c3_fixed_fists_erd_median_percent": float(historical["c3_median_percent_change"]),
                "historical_central_peak_available": peak_available,
                "historical_central_peak_frequency_hz": (
                    float(peak["peak_frequency_hz"]) if peak_available else ""
                ),
                "historical_central_peak_well_defined": peak["well_defined_peak"] == "True",
                "imf_study_subject": imf_row is not None,
                "imf_method_available": imf_available,
                "imf_peak_frequency_median_hz": (
                    float(imf_row["peak_frequency_median_hz"]) if imf_available else ""
                ),
                "imf_three_fold_peak_span_hz": (
                    float(imf_row["three_fold_peak_span_hz"]) if imf_available else ""
                ),
                "analysis_role": "post_evaluation_exploratory_only",
            }
        )
    relationships = [
        _spearman_row("C4_fixed_ERD", joined, "c4_fixed_fists_erd_median_percent"),
        _spearman_row("C3_fixed_ERD", joined, "c3_fixed_fists_erd_median_percent"),
        _spearman_row(
            "historical_central_peak_frequency",
            joined,
            "historical_central_peak_frequency_hz",
        ),
        _spearman_row("IMF_peak_frequency", joined, "imf_peak_frequency_median_hz"),
        _spearman_row(
            "IMF_peak_span_reliability", joined, "imf_three_fold_peak_span_hz"
        ),
    ]
    imf_overlap = [row for row in joined if row["imf_study_subject"]]
    available_scores = [
        float(row["csp_balanced_accuracy"])
        for row in imf_overlap
        if row["imf_method_available"]
    ]
    unavailable_scores = [
        float(row["csp_balanced_accuracy"])
        for row in imf_overlap
        if not row["imf_method_available"]
    ]
    relationships.append(
        {
            "relationship": "IMF_availability_group_medians",
            "subject_count": len(imf_overlap),
            "spearman_rho": "",
            "two_sided_unadjusted_p": "",
            "available_count": len(available_scores),
            "available_median_balanced_accuracy": float(np.median(available_scores)),
            "unavailable_count": len(unavailable_scores),
            "unavailable_median_balanced_accuracy": (
                float(np.median(unavailable_scores)) if unavailable_scores else ""
            ),
            "analysis_role": "post_evaluation_exploratory_only",
        }
    )
    return joined, relationships
