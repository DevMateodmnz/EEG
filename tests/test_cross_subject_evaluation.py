"""Pre-evaluation tests for frozen cross-subject summaries and comparisons."""

from __future__ import annotations

import unittest

from eeg_project.cross_subject_decoding import (
    MODEL_NAMES,
    classify_paired_transfer,
    classify_transfer,
)
from eeg_project.cross_subject_evaluation import (
    aggregate_confusion_rows,
    evaluation_run_summary_rows,
    load_final_cross_subject_config,
    within_cross_comparison_rows,
    within_cross_summary_rows,
)


def cross_subject_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(MODEL_NAMES):
        for subject in range(21, 107):
            score = 0.55 + model_index * 0.04 + (subject % 7) / 1000
            rows.append(
                {
                    "subject": subject,
                    "model": model,
                    "primary_mean_run_balanced_accuracy": score,
                    "fists_recall": score,
                    "feet_recall": score,
                    "group_observational_unit": "participant",
                    "true_fists_predicted_fists": 10,
                    "true_fists_predicted_feet": 5,
                    "true_feet_predicted_fists": 5,
                    "true_feet_predicted_feet": 10,
                }
            )
    return rows


class CrossSubjectEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.final, cls.initial = load_final_cross_subject_config()

    def test_final_loader_preserves_frozen_cohorts_models_and_no_adaptation(self) -> None:
        self.assertEqual(self.final["cohorts"]["training_subjects"], list(range(1, 21)))
        self.assertEqual(
            self.final["cohorts"]["evaluation_eligible_subjects"],
            [subject for subject in range(21, 110) if subject not in {88, 92, 100}],
        )
        self.assertEqual(tuple(self.final["selected_models"]), MODEL_NAMES)
        self.assertFalse(
            self.final["zero_shot_evaluation"]["target_subject_unlabeled_adaptation"]
        )

    def test_run_and_confusion_summaries_require_participant_rows(self) -> None:
        run_rows = []
        for model in MODEL_NAMES:
            for subject in range(21, 107):
                for run in (6, 10, 14):
                    run_rows.append(
                        {
                            "subject": subject,
                            "model": model,
                            "test_run": run,
                            "balanced_accuracy": 0.6,
                            "fists_recall": 0.6,
                            "feet_recall": 0.6,
                        }
                    )
        summaries = evaluation_run_summary_rows(run_rows, expected_subject_count=86)
        self.assertEqual(len(summaries), 6)
        self.assertTrue(all(row["participant_count"] == 86 for row in summaries))
        confusions = aggregate_confusion_rows(
            cross_subject_rows(), expected_subject_count=86
        )
        self.assertEqual(len(confusions), 2)
        self.assertTrue(all(row["trial_count"] == 86 * 30 for row in confusions))

    def test_within_cross_comparison_is_exactly_matched(self) -> None:
        historical = []
        for model_index, model in enumerate(MODEL_NAMES):
            for subject in range(21, 107):
                historical.append(
                    {
                        "subject": str(subject),
                        "model": model,
                        "qc_sensitivity": "False",
                        "primary_mean_fold_balanced_accuracy": str(
                            0.65 + model_index * 0.03
                        ),
                    }
                )
        comparisons = within_cross_comparison_rows(cross_subject_rows(), historical)
        self.assertEqual(len(comparisons), 172)
        self.assertEqual(
            len({(row["subject"], row["model"]) for row in comparisons}), 172
        )
        summaries = within_cross_summary_rows(comparisons)
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(row["participant_count"] == 86 for row in summaries))
        self.assertTrue(
            all(row["median_cross_minus_within_balanced_accuracy"] < 0 for row in summaries)
        )

    def test_transfer_categories_apply_only_frozen_thresholds(self) -> None:
        clear = {
            "median_balanced_accuracy": 0.61,
            "above_0_5_fraction": 0.7,
            "median_fists_recall": 0.6,
            "median_feet_recall": 0.61,
        }
        no_transfer = {**clear, "median_balanced_accuracy": 0.51}
        self.assertEqual(classify_transfer(clear, self.initial), "clear zero-shot transfer")
        self.assertEqual(
            classify_transfer(no_transfer, self.initial),
            "no convincing zero-shot transfer",
        )
        comparison = {
            "median_CSP_minus_spectral": -0.04,
            "CSP_better_fraction": 0.25,
            "spectral_better_fraction": 0.7,
        }
        self.assertEqual(
            classify_paired_transfer(comparison, self.initial),
            "spectral transfers better",
        )


if __name__ == "__main__":
    unittest.main()
