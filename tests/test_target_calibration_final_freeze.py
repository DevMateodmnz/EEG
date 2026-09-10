"""Immutable guards for the final target-calibration method freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = PROJECT_ROOT / "config" / "minimal_target_calibration_final.json"


class TargetCalibrationFinalFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(FINAL_PATH.read_text(encoding="utf-8"))

    def test_chronology_precedes_evaluation_calibration_outcomes(self) -> None:
        self.assertEqual(
            self.payload["study_stage"],
            "final_minimal_target_calibration_method_frozen",
        )
        self.assertEqual(self.payload["initial_calibration_freeze_git_commit"], "d4cff9a")
        self.assertEqual(self.payload["calibration_development_git_commit"], "b918cb4")
        self.assertFalse(
            self.payload["evaluation_calibration_outcomes_inspected_at_final_freeze"]
        )
        self.assertFalse(
            self.payload["evaluation_subject_EEG_loaded_for_calibration_at_final_freeze"]
        )
        self.assertTrue(self.payload["evaluation_unlocked_after_this_freeze_commit"])

    def test_exact_cohorts_selected_method_and_curve_are_frozen(self) -> None:
        cohorts = self.payload["cohorts"]
        self.assertEqual(cohorts["source_training_subjects"], list(range(1, 21)))
        self.assertEqual(
            cohorts["evaluation_eligible_subjects"],
            [subject for subject in range(21, 110) if subject not in {88, 92, 100}],
        )
        self.assertEqual(self.payload["selected_calibration_method"]["name"], "threshold")
        self.assertEqual(
            self.payload["calibration_curve"]["ordered_total_trials"],
            [0, 4, 8, 12, 14],
        )
        self.assertTrue(self.payload["calibration_curve"]["nested"])
        self.assertTrue(self.payload["calibration_curve"]["balanced"])

    def test_source_spatial_representation_and_scaler_remain_target_free(self) -> None:
        source = self.payload["source_model"]
        self.assertEqual(source["name"], "csp_4_empirical")
        self.assertEqual(source["CSP_components"], 4)
        self.assertEqual(source["expected_training_trial_count"], 900)
        self.assertFalse(source["source_CSP_fit_target_EEG"])
        self.assertFalse(source["source_scaler_fit_target_EEG"])
        target = self.payload["target_operations"]
        for field in (
            "test_run_labels_used",
            "test_run_EEG_used_for_calibration",
            "unlabeled_test_EEG_used_for_adaptation",
            "target_CSP_refit",
            "target_scaler_refit",
            "target_covariance_alignment",
            "target_specific_preprocessing",
        ):
            self.assertFalse(target[field])

    def test_primary_aggregation_comparisons_and_success_rules_are_frozen(self) -> None:
        primary = self.payload["primary_evaluation"]
        self.assertEqual(primary["group_observational_unit"], "participant")
        self.assertTrue(primary["matched_zero_comparison"])
        self.assertEqual(
            self.payload["primary_comparisons"]["sizes_vs_matched_zero"],
            [4, 8, 12, 14],
        )
        criteria = self.payload["convincing_benefit_criteria"]
        self.assertEqual(criteria["minimum_median_balanced_accuracy"], 0.60)
        self.assertEqual(criteria["minimum_median_gain_over_matched_zero"], 0.03)
        self.assertEqual(criteria["minimum_fraction_subjects_improved"], 0.60)

    def test_initial_core_and_development_artifacts_are_byte_exact(self) -> None:
        records = [
            self.payload["initial_policy"],
            self.payload["frozen_calibration_implementation"],
            *self.payload["development_evidence"]["artifacts"].values(),
        ]
        for record in records:
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], path
            )
        self.assertFalse(self.payload["development_evidence"]["evaluation_outcomes_used"])


if __name__ == "__main__":
    unittest.main()
