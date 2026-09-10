"""Scientific invariants for cohort audit and subject-level replication."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from eeg_project.cohort import (
    audit_recording,
    build_primary_replication_rows,
    cohort_feature_summaries,
    exact_sign_test_one_sided,
    holm_adjust,
    load_replication_cohort_config,
    replication_category,
    run_direction_category,
    subject_eligibility_rows,
    summarize_subject_features,
    wilson_interval,
)


class CohortUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_replication_cohort_config()

    def test_frozen_cohort_and_primary_hypotheses(self) -> None:
        self.assertEqual(self.config.development_subject, 1)
        self.assertEqual(self.config.replication_subjects, tuple(range(2, 21)))
        self.assertEqual(self.config.runs, (6, 10, 14))
        self.assertEqual(
            self.config.primary_replication_features,
            (
                ("both_fists_imagery", "C4", "central_12_13"),
                ("both_fists_imagery", "C3", "central_12_13"),
            ),
        )
        self.assertTrue(self.config.technical_eligibility.outcome_based_exclusion_forbidden)

    def test_subject_aggregation_uses_trials_within_one_person(self) -> None:
        rows = [
            {
                "semantic_condition": "both_fists_imagery",
                "channel": "C4",
                "frequency_band": "central_12_13",
                "change_percent": value,
                "change_db": db,
                "quality_status": status,
            }
            for value, db, status in (
                (-50.0, -3.01, "not flagged"),
                (-25.0, -1.25, "statistical candidate"),
                (10.0, 0.41, "not flagged"),
            )
        ]
        (summary,) = summarize_subject_features(
            rows,
            2,
            "replication",
            (("both_fists_imagery", "C4", "central_12_13"),),
            0.2,
        )
        self.assertEqual(summary["trial_count"], 3)
        self.assertEqual(summary["median_change_percent"], -25.0)
        self.assertEqual(summary["negative_trial_count"], 2)
        self.assertAlmostEqual(summary["negative_trial_fraction"], 2 / 3)
        self.assertEqual(summary["qc_candidate_trial_count"], 1)
        self.assertEqual(summary["exclude_qc_candidates_median_change_percent"], -20.0)

    def test_replication_aggregation_excludes_development_and_keeps_reversal(self) -> None:
        small_inference = replace(
            self.config.group_inference,
            bootstrap_subject_median_resamples=100,
        )
        config = replace(
            self.config,
            secondary_features=(),
            replication_subjects=(2, 3, 4),
            group_inference=small_inference,
        )
        subject_rows = []
        primary_rows = []
        values = {
            "C4": {1: -99.0, 2: -10.0, 3: -20.0, 4: 5.0},
            "C3": {1: -99.0, 2: 10.0, 3: 20.0, 4: -5.0},
        }
        for channel, subject_values in values.items():
            for subject, value in subject_values.items():
                row = {
                    "subject": subject,
                    "subject_role": "development" if subject == 1 else "replication",
                    "semantic_condition": "both_fists_imagery",
                    "channel": channel,
                    "frequency_band": "central_12_13",
                    "median_change_percent": value,
                    "exclude_qc_candidates_median_change_percent": value,
                }
                subject_rows.append(row)
                primary_rows.append({**row, "supporting_run_count": 3 if value < 0 else 0})
        summaries = cohort_feature_summaries(subject_rows, primary_rows, config)
        c4 = next(row for row in summaries if row["channel"] == "C4")
        c3 = next(row for row in summaries if row["channel"] == "C3")
        self.assertEqual(c4["replication_subject_count"], 3)
        self.assertEqual(c4["negative_subject_count"], 2)
        self.assertEqual(c4["median_across_subject_medians_percent"], -10.0)
        self.assertEqual(c4["replication_category"], "strong replication")
        self.assertEqual(c3["negative_subject_count"], 1)
        self.assertEqual(c3["median_across_subject_medians_percent"], 10.0)
        self.assertEqual(c3["replication_category"], "failed replication")
        # Subject 4's reversed C4 result remains a contributing person.
        self.assertEqual(c4["maximum_subject_median_percent"], 5.0)

    def test_technical_failure_is_documented_without_outcome_input(self) -> None:
        config = replace(self.config, replication_subjects=(2,))
        rows = []
        for subject in (1, 2):
            for run in (6, 10, 14):
                compatible = not (subject == 2 and run == 10)
                rows.append(
                    {
                        "subject": subject,
                        "run": run,
                        "protocol_compatible": compatible,
                        "failure_reason": "corrupt_file" if not compatible else "",
                    }
                )
        eligibility = subject_eligibility_rows(rows, config)
        subject_two = next(row for row in eligibility if row["subject"] == 2)
        self.assertEqual(subject_two["technical_eligibility"], "technically_incompatible")
        self.assertIn("run10:corrupt_file", subject_two["failure_reason"])
        self.assertFalse(subject_two["outcome_used_for_eligibility"])

    def test_run_sign_inference_and_replication_rules(self) -> None:
        self.assertEqual(run_direction_category([-1, -2, 3], "negative"), "2/3_negative")
        self.assertAlmostEqual(exact_sign_test_one_sided(3, 3), 0.125)
        lower, upper = wilson_interval(8, 10)
        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 0.8)
        np.testing.assert_allclose(holm_adjust([0.01, 0.04]), [0.02, 0.04])
        category, passed = replication_category(
            0.8, -20.0, 0.7, True, True, self.config.replication_success_criteria
        )
        self.assertEqual(category, "strong replication")
        self.assertTrue(all(passed.values()))


class RealDataCohortAuditIntegrationTests(unittest.TestCase):
    def test_subject_two_reversed_class_count_is_technically_valid(self) -> None:
        config = load_replication_cohort_config()
        row = audit_recording(2, 6, config)
        self.assertTrue(row["protocol_compatible"])
        self.assertEqual(row["sampling_frequency_hz"], 160.0)
        self.assertEqual(row["channel_count"], 64)
        self.assertEqual((row["T0_count"], row["T1_count"], row["T2_count"]), (15, 8, 7))
        self.assertEqual(row["trailing_zero_samples"], 0)
        self.assertEqual(row["confirmed_flat_or_nonfinite_channel_count"], 0)


if __name__ == "__main__":
    unittest.main()
