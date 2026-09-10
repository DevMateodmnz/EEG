"""Final Riemannian method guards written before held-out evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = PROJECT_ROOT / "config" / "riemannian_decoding_final.json"


class RiemannianFinalFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(FINAL_PATH.read_text(encoding="utf-8"))

    def test_chronology_and_cohorts_precede_evaluation(self) -> None:
        self.assertEqual(self.payload["study_stage"], "final_riemannian_method_frozen")
        self.assertEqual(self.payload["initial_riemannian_freeze_git_commit"], "131e601")
        self.assertEqual(self.payload["riemannian_implementation_git_commit"], "edcea46")
        self.assertEqual(self.payload["riemannian_development_git_commit"], "8adad9c")
        self.assertFalse(self.payload["evaluation_classifier_outcomes_inspected_at_final_freeze"])
        self.assertFalse(self.payload["evaluation_subject_EEG_loaded_at_final_freeze"])
        self.assertTrue(self.payload["evaluation_unlocked_after_this_freeze_commit"])
        cohorts = self.payload["cohorts"]
        self.assertEqual(cohorts["training_subjects"], list(range(1, 21)))
        self.assertEqual(cohorts["evaluation_eligible_subjects"], [subject for subject in range(21, 110) if subject not in {88, 92, 100}])

    def test_exact_method_and_all_target_adaptation_routes_are_frozen_off(self) -> None:
        method = self.payload["selected_method"]
        self.assertEqual(method["covariance_estimator"], "LedoitWolf_assume_centered_false_trace_normalized_per_trial")
        self.assertEqual(method["reference_mean"], "affine_invariant_Riemannian_Karcher_mean_training_trials_only")
        self.assertEqual(method["tangent_feature_count"], 2080)
        self.assertFalse(method["tangent_space_update"])
        self.assertEqual(method["classifier"]["class"], "LinearDiscriminantAnalysis")
        self.assertEqual(method["classifier"]["shrinkage"], "auto")
        zero_shot = self.payload["zero_shot_evaluation"]
        for key, value in zero_shot.items():
            if key.startswith("target_subject") or key.startswith("evaluation_subject"):
                self.assertFalse(value)

    def test_metrics_gate_and_development_selection_are_exact(self) -> None:
        self.assertEqual(self.payload["metrics"]["group_observational_unit"], "participant")
        self.assertEqual(self.payload["metrics"]["primary"], "unweighted_mean_of_three_run_balanced_accuracies_per_evaluation_subject")
        clear = self.payload["success_criteria"]["clear_representation_improvement"]
        self.assertEqual(clear["minimum_median_Riemannian_minus_CSP"], 0.03)
        development = self.payload["development_evidence"]
        self.assertEqual(development["selected_classifier"], "shrinkage_lda")
        self.assertAlmostEqual(development["median_logistic_minus_lda"], 0.011904761904761918)
        self.assertEqual(development["logistic_better_fraction"], 0.5)
        self.assertFalse(development["evaluation_outcome_used"])

    def test_initial_core_and_development_artifacts_are_byte_exact(self) -> None:
        records = [self.payload["initial_policy"], self.payload["frozen_method_implementation"], *self.payload["development_evidence"]["artifacts"].values()]
        for record in records:
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], path)


if __name__ == "__main__":
    unittest.main()
