"""Tests for final-frozen decoder group evaluation operations."""

from __future__ import annotations

import unittest

import numpy as np

from eeg_project.decoding import load_subject_decoding_data
from eeg_project.decoding_evaluation import (
    aggregate_confusions,
    classify_csp_added_value,
    classify_decoding_signal,
    load_final_decoding_config,
    load_evaluation_subject_data,
    model_comparison_summary,
    paired_model_rows,
    summarize_model_group,
)


def subject_row(subject: int, model: str, qc: bool, score: float) -> dict[str, object]:
    return {
        "subject": subject,
        "model": model,
        "qc_sensitivity": qc,
        "primary_mean_fold_balanced_accuracy": score,
        "folds_above_chance_count": 3 if score > 0.5 else 0,
        "fold_balanced_accuracy_range": 0.1,
        "out_of_fold_fists_recall": score,
        "out_of_fold_feet_recall": score,
        "qc_candidate_count": 1,
    }


class DecodingEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.final, cls.initial = load_final_decoding_config()

    def test_final_loader_keeps_evaluation_cohort_and_frozen_models(self) -> None:
        self.assertEqual(
            len(self.final["cohorts"]["decoder_evaluation_eligible_subjects"]), 86
        )
        self.assertEqual(
            self.final["selected_spectral_baseline"]["candidate_name"],
            "spectral_baseline_6",
        )
        self.assertEqual(
            self.final["selected_csp_decoder"]["candidate_name"], "csp_4_empirical"
        )

    def test_corrected_loader_matches_core_for_standard_runs_and_accepts_7_7(self) -> None:
        frozen = load_subject_decoding_data(21, self.initial)
        corrected = load_evaluation_subject_data(21, self.initial)
        np.testing.assert_array_equal(
            frozen.fixed_task_data_volts, corrected.fixed_task_data_volts
        )
        np.testing.assert_array_equal(
            frozen.csp_task_data_volts, corrected.csp_task_data_volts
        )
        np.testing.assert_array_equal(frozen.labels, corrected.labels)
        np.testing.assert_array_equal(frozen.runs, corrected.runs)
        boundary = load_evaluation_subject_data(34, self.initial)
        self.assertEqual(boundary.labels.size, 42)
        for run in (6, 10, 14):
            labels = boundary.labels[boundary.runs == run]
            self.assertEqual(int(np.sum(labels == 0)), 7)
            self.assertEqual(int(np.sum(labels == 1)), 7)

    def test_paired_model_rows_require_exact_primary_and_qc_subject_pairs(self) -> None:
        rows = []
        for subject in (21, 22, 23):
            rows.extend(
                [
                    subject_row(subject, "baseline", False, 0.55),
                    subject_row(subject, "csp", False, 0.65),
                    subject_row(subject, "baseline", True, 0.54),
                    subject_row(subject, "csp", True, 0.64),
                ]
            )
        paired = paired_model_rows(rows, "baseline", "csp")
        self.assertEqual(len(paired), 3)
        self.assertTrue(all(row["matched_subject_and_trials"] for row in paired))
        self.assertTrue(
            all(abs(float(row["csp_minus_baseline_balanced_accuracy"]) - 0.1) < 1e-12 for row in paired)
        )
        comparison = model_comparison_summary(paired)
        self.assertAlmostEqual(
            comparison["median_csp_minus_baseline_balanced_accuracy"], 0.1
        )
        self.assertEqual(comparison["csp_better_count"], 3)

    def test_group_summary_uses_one_subject_score_and_fixed_seed_ci(self) -> None:
        rows = []
        for subject, score in zip(range(21, 27), (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)):
            rows.append(subject_row(subject, "csp", False, score))
            rows.append(subject_row(subject, "csp", True, score - 0.01))
        first = summarize_model_group(rows, "csp", self.final)
        second = summarize_model_group(rows, "csp", self.final)
        self.assertEqual(first, second)
        self.assertEqual(first["evaluation_subject_count"], 6)
        self.assertAlmostEqual(first["median_subject_balanced_accuracy"], 0.675)
        self.assertEqual(first["group_unit"], "participant")

    def test_frozen_decoding_and_added_value_categories(self) -> None:
        csp = {
            "median_subject_balanced_accuracy": 0.65,
            "fraction_subjects_above_0_5": 0.75,
            "fraction_subjects_with_at_least_two_folds_above_0_5": 0.7,
            "median_fists_recall": 0.64,
            "median_feet_recall": 0.62,
            "absolute_median_class_recall_difference": 0.02,
            "qc_cohort_median_minus_primary_cohort_median": -0.01,
            "leave_one_subject_out_minimum_cohort_median": 0.64,
        }
        self.assertEqual(
            classify_decoding_signal(csp, 0.04, self.final), "useful decoding signal"
        )
        comparison = {
            "median_csp_minus_baseline_balanced_accuracy": 0.04,
            "fraction_csp_better": 0.7,
            "qc_median_csp_minus_baseline_balanced_accuracy": 0.03,
        }
        self.assertEqual(
            classify_csp_added_value(comparison, self.final), "clear added value"
        )

    def test_confusion_totals_equal_prediction_totals(self) -> None:
        rows = []
        for model in ("baseline", "csp"):
            for qc in (False, True):
                for true, predicted in (
                    ("both_fists_imagery", "both_fists_imagery"),
                    ("both_fists_imagery", "both_feet_imagery"),
                    ("both_feet_imagery", "both_fists_imagery"),
                    ("both_feet_imagery", "both_feet_imagery"),
                ):
                    rows.append(
                        {
                            "model": model,
                            "qc_sensitivity": qc,
                            "true_condition": true,
                            "predicted_condition": predicted,
                        }
                    )
        summaries = aggregate_confusions(rows)
        self.assertEqual(len(summaries), 4)
        self.assertTrue(all(row["trial_count"] == 4 for row in summaries))
        self.assertTrue(all(row["fists_recall"] == 0.5 for row in summaries))
        self.assertTrue(all(row["feet_recall"] == 0.5 for row in summaries))


if __name__ == "__main__":
    unittest.main()
