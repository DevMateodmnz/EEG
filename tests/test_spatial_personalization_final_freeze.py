"""Immutable guards for the final spatial-personalization method freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = PROJECT_ROOT / "config" / "spatial_personalization_final.json"


class SpatialPersonalizationFinalFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(FINAL_PATH.read_text(encoding="utf-8"))

    def test_chronology_precedes_evaluation_target_csp(self) -> None:
        self.assertEqual(self.payload["study_stage"], "final_spatial_personalization_method_frozen")
        self.assertEqual(self.payload["initial_freeze_git_commit"], "4ceea62")
        self.assertEqual(self.payload["implementation_git_commit"], "fb9f55d")
        self.assertEqual(self.payload["development_git_commit"], "51fab24")
        self.assertFalse(self.payload["evaluation_target_CSP_outcomes_inspected_at_final_freeze"])
        self.assertFalse(self.payload["evaluation_target_CSP_fitted_at_final_freeze"])
        self.assertTrue(self.payload["evaluation_unlocked_after_this_freeze_commit"])

    def test_selected_A_B_C_are_exact(self) -> None:
        self.assertEqual(self.payload["representation_A"]["method"], "source_zero_shot")
        self.assertFalse(self.payload["representation_A"]["target_fitting"])
        self.assertEqual(
            self.payload["representation_B"]["method"],
            "source_csp_target_lda_source_scaler",
        )
        self.assertFalse(self.payload["representation_B"]["target_scaler_fit"])
        self.assertEqual(self.payload["representation_C"]["method"], "target_csp_ledoit_wolf")
        self.assertEqual(self.payload["representation_C"]["target_CSP_components"], 4)
        self.assertEqual(self.payload["representation_C"]["target_CSP_covariance"], "ledoit_wolf")

    def test_geometry_leakage_and_primary_comparisons_are_frozen(self) -> None:
        geometry = self.payload["calibration_geometry"]
        self.assertEqual(geometry["total_calibration_trials"], 14)
        self.assertEqual(geometry["trials_per_class"], 7)
        self.assertTrue(geometry["same_test_trials_required_for_A_B_C"])
        self.assertTrue(all(not value for value in self.payload["forbidden_target_operations"].values()))
        self.assertEqual(
            [row["name"] for row in self.payload["primary_comparisons"]],
            ["B_minus_A", "C_minus_B", "C_minus_A"],
        )
        self.assertEqual(
            self.payload["participant_aggregation"]["group_observational_unit"],
            "participant",
        )

    def test_practical_gate_qc_and_stability_are_frozen(self) -> None:
        criteria = self.payload["convincing_paired_gain_criteria"]
        self.assertEqual(criteria["minimum_median_paired_gain"], 0.03)
        self.assertEqual(criteria["minimum_fraction_participants_improved"], 0.60)
        self.assertEqual(criteria["minimum_QC_median_paired_gain"], 0.02)
        self.assertTrue(self.payload["QC_sensitivity"]["calibration_trials_unchanged"])
        self.assertTrue(
            self.payload["CSP_stability"]["sign_order_and_within_subspace_rotation_invariant"]
        )

    def test_initial_core_and_development_artifacts_are_byte_exact(self) -> None:
        records = [
            self.payload["initial_policy"],
            self.payload["frozen_implementation"],
            *self.payload["development_evidence"]["artifacts"].values(),
        ]
        for record in records:
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], path)
        self.assertFalse(self.payload["development_evidence"]["evaluation_target_CSP_outcomes_used"])


if __name__ == "__main__":
    unittest.main()
