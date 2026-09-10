"""Immutable guards for the initial spatial-personalization protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "spatial_personalization.json"


class SpatialPersonalizationFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_new_target_csp_outcomes_are_locked(self) -> None:
        self.assertEqual(
            self.payload["study_stage"],
            "initial_spatial_personalization_methodology_frozen",
        )
        self.assertEqual(self.payload["parent_completed_milestone_git_commit"], "5d11ed1")
        self.assertTrue(self.payload["evaluation_locked"])
        self.assertFalse(self.payload["development_outcomes_inspected_at_freeze"])
        self.assertFalse(self.payload["evaluation_target_CSP_outcomes_inspected_at_freeze"])
        self.assertFalse(self.payload["evaluation_target_CSP_fitted_at_freeze"])
        self.assertTrue(self.payload["evaluation_EEG_is_not_broadly_untouched"])

    def test_cohorts_and_complete_run_geometry_are_exact(self) -> None:
        cohorts = self.payload["cohorts"]
        self.assertEqual(cohorts["development_subjects"], list(range(1, 21)))
        self.assertEqual(
            cohorts["evaluation_eligible_subjects"],
            [subject for subject in range(21, 110) if subject not in {88, 92, 100}],
        )
        geometry = self.payload["calibration_geometry"]
        self.assertEqual(geometry["total_calibration_trials"], 14)
        self.assertEqual(geometry["trials_per_class"], 7)
        self.assertTrue(geometry["same_test_trials_required_for_all_methods"])
        self.assertFalse(geometry["test_labels_enter_fitting"])
        self.assertFalse(geometry["unlabeled_test_EEG_enters_adaptation"])
        self.assertEqual(
            {row["calibration_run"]: set(row["test_runs"]) for row in geometry["scenarios"]},
            {6: {10, 14}, 10: {6, 14}, 14: {6, 10}},
        )

    def test_candidate_family_is_tiny_and_representation_is_fixed(self) -> None:
        fixed = self.payload["fixed_signal_representation"]
        self.assertEqual(fixed["passband_hz"], [8.0, 30.0])
        self.assertEqual(fixed["task_interval_seconds"], [1.0, 3.0])
        self.assertEqual(fixed["target_CSP_components"], 4)
        self.assertEqual(
            set(self.payload["representation_B_source_CSP_target_classifier"]["development_scaler_candidates"]),
            {"retain_source_scaler", "fit_target_scaler"},
        )
        self.assertEqual(
            set(self.payload["representation_C_target_CSP_target_classifier"]["covariance_candidates"]),
            {"empirical", "ledoit_wolf"},
        )

    def test_primary_comparisons_gate_and_stability_are_frozen(self) -> None:
        self.assertEqual(
            self.payload["primary_comparisons"], ["B_minus_A", "C_minus_B", "C_minus_A"]
        )
        criteria = self.payload["convincing_paired_gain_criteria"]
        self.assertEqual(criteria["minimum_median_paired_gain"], 0.03)
        self.assertEqual(criteria["minimum_fraction_participants_improved"], 0.60)
        self.assertEqual(criteria["minimum_QC_median_paired_gain"], 0.02)
        stability = self.payload["CSP_stability"]
        self.assertTrue(stability["sign_order_and_within_subspace_rotation_invariant"])
        self.assertEqual(stability["descriptive_categories"]["high_at_or_above"], 0.75)

    def test_completed_historical_evidence_is_byte_exact(self) -> None:
        for record in self.payload["historical_inputs"].values():
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], path)


if __name__ == "__main__":
    unittest.main()
