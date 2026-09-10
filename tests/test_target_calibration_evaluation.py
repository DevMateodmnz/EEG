"""Pre-outcome tests for final calibration summaries and checkpoints."""

from __future__ import annotations

import unittest

import numpy as np

from eeg_project.decoding import TrialIdentity
from eeg_project.target_calibration import TargetSourceFeatures, convincing_benefit
from eeg_project.target_calibration_evaluation import (
    calibration_run_summary_rows,
    exploratory_relationship_rows,
    historical_within_comparison_rows,
    incremental_gain_rows,
    load_final_calibration_config,
    matched_gain_rows,
    subgroup_gain_rows,
)
from scripts.evaluate_target_calibration import (
    deserialize_target_features,
    serialize_target_features,
)


def participant_rows() -> list[dict[str, object]]:
    rows = []
    for subject in range(21, 25):
        for size in (0, 4, 8, 12, 14):
            for qc in (False, True):
                score = (
                    0.50
                    + (subject - 21) * 0.02
                    + size * 0.002 * (subject - 20) / 4
                )
                rows.append(
                    {
                        "subject": subject,
                        "method": "threshold",
                        "calibration_size": size,
                        "qc_sensitivity": qc,
                        "primary_mean_test_run_balanced_accuracy": score
                        + (0.002 if qc else 0.0),
                        "fists_recall": score,
                        "feet_recall": score,
                    }
                )
    return rows


class TargetCalibrationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.final, cls.initial = load_final_calibration_config()

    def test_final_loader_preserves_selected_threshold_and_evaluation_cohort(self) -> None:
        self.assertEqual(self.final["selected_calibration_method"]["name"], "threshold")
        self.assertEqual(
            self.final["cohorts"]["evaluation_eligible_subjects"],
            [subject for subject in range(21, 110) if subject not in {88, 92, 100}],
        )
        self.assertFalse(self.final["target_operations"]["target_CSP_refit"])
        self.assertFalse(self.final["target_operations"]["target_scaler_refit"])

    def test_matched_gains_incremental_and_within_comparisons_are_participant_level(self) -> None:
        participants = participant_rows()
        gains = matched_gain_rows(participants, method="threshold")
        self.assertEqual(len(gains), 20)
        self.assertEqual(
            len({(row["subject"], row["calibration_size"]) for row in gains}), 20
        )
        increments = incremental_gain_rows(participants, method="threshold")
        self.assertEqual(len(increments), 4)
        historical = [
            {
                "subject": str(subject),
                "model": "csp_4_empirical",
                "qc_sensitivity": "False",
                "primary_mean_fold_balanced_accuracy": "0.65",
            }
            for subject in range(21, 25)
        ]
        within = historical_within_comparison_rows(
            participants, historical, method="threshold"
        )
        self.assertEqual(len(within), 5)
        subgroups = subgroup_gain_rows(gains, historical)
        self.assertTrue(
            all(row["analysis_role"] == "post_primary_exploratory" for row in subgroups)
        )

    def test_calibration_run_summaries_use_two_test_runs_then_one_subject_value(self) -> None:
        rows = []
        for subject in range(21, 25):
            for size in (0, 4, 8, 12, 14):
                for calibration_run in (6, 10, 14):
                    for test_run in (run for run in (6, 10, 14) if run != calibration_run):
                        rows.append(
                            {
                                "subject": subject,
                                "method": "threshold",
                                "calibration_size": size,
                                "calibration_run": calibration_run,
                                "test_run": test_run,
                                "qc_sensitivity": False,
                                "balanced_accuracy": 0.5
                                + subject / 1000
                                + calibration_run / 10000
                                + test_run / 100000,
                            }
                        )
        summary = calibration_run_summary_rows(
            rows, method="threshold", expected_subject_count=4
        )
        self.assertEqual(len(summary), 15)
        self.assertTrue(all(row["participant_count"] == 4 for row in summary))

    def test_exploratory_ERD_relationships_cannot_select_the_method(self) -> None:
        gains = matched_gain_rows(participant_rows(), method="threshold")
        physiology = [
            {
                "subject": str(subject),
                "c3_fixed_fists_erd_median_percent": str(-subject),
                "c4_fixed_fists_erd_median_percent": str(-subject * 1.1),
            }
            for subject in range(21, 25)
        ]
        relationships = exploratory_relationship_rows(gains, physiology)
        self.assertEqual(len(relationships), 12)
        self.assertTrue(
            all(
                row["analysis_role"] == "post_primary_exploratory_unadjusted"
                for row in relationships
            )
        )

    def test_checkpoint_feature_serialization_round_trip_is_exact(self) -> None:
        runs = (6, 6, 10, 10, 14, 14)
        identities = tuple(
            TrialIdentity(
                21,
                runs[index],
                index % 2,
                index * 2 + 1,
                "both_fists_imagery" if index % 2 == 0 else "both_feet_imagery",
                "source.edf",
                "source-hash",
            )
            for index in range(6)
        )
        source = TargetSourceFeatures(
            21,
            np.arange(24, dtype=float).reshape(6, 4),
            np.asarray([-1.0, 1.0, -0.5, 0.5, -0.2, 0.2]),
            np.asarray([0, 1, 0, 1, 0, 1]),
            np.asarray([0, 1, 0, 1, 0, 1]),
            np.asarray(runs),
            identities,
            np.asarray([False, True, False, False, False, True]),
        )
        payload = serialize_target_features(
            source, "final", "calibration", "cross", {"train.edf": "hash"}
        )
        restored = deserialize_target_features(payload)
        self.assertTrue(np.array_equal(restored.features, source.features))
        self.assertTrue(
            np.array_equal(restored.source_decision_scores, source.source_decision_scores)
        )
        self.assertEqual(
            [identity.key for identity in restored.identities],
            [identity.key for identity in source.identities],
        )

    def test_convincing_benefit_uses_every_frozen_criterion(self) -> None:
        passing = {
            "calibration_size": 4,
            "median_balanced_accuracy": 0.61,
            "median_gain_over_matched_zero": 0.04,
            "improved_fraction": 0.65,
            "median_fists_recall": 0.60,
            "median_feet_recall": 0.58,
            "median_paired_QC_minus_primary_balanced_accuracy": -0.01,
            "one_sided_exact_sign_p": 0.01,
        }
        self.assertTrue(convincing_benefit(passing, self.initial))
        self.assertFalse(
            convincing_benefit(
                {**passing, "median_gain_over_matched_zero": 0.029}, self.initial
            )
        )


if __name__ == "__main__":
    unittest.main()
