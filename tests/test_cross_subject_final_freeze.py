"""Immutable cross-subject final-method guards written before evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = PROJECT_ROOT / "config" / "cross_subject_decoding_final.json"


class CrossSubjectFinalFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(FINAL_PATH.read_text(encoding="utf-8"))

    def test_chronology_precedes_evaluation_outcomes(self) -> None:
        self.assertEqual(self.payload["study_stage"], "final_cross_subject_method_frozen")
        self.assertEqual(self.payload["initial_cross_subject_freeze_git_commit"], "74b49d0")
        self.assertEqual(self.payload["cross_subject_implementation_git_commit"], "58e87e1")
        self.assertEqual(self.payload["cross_subject_development_git_commit"], "5da3b37")
        self.assertFalse(
            self.payload["evaluation_classifier_outcomes_inspected_at_final_freeze"]
        )
        self.assertFalse(self.payload["evaluation_subject_EEG_loaded_at_final_freeze"])
        self.assertTrue(self.payload["evaluation_unlocked_after_this_freeze_commit"])

    def test_cohorts_are_exact_and_disjoint(self) -> None:
        cohorts = self.payload["cohorts"]
        self.assertEqual(cohorts["training_subjects"], list(range(1, 21)))
        self.assertEqual(cohorts["evaluation_requested_subjects"], list(range(21, 110)))
        self.assertEqual(cohorts["technically_incompatible_subjects"], [88, 92, 100])
        self.assertEqual(
            cohorts["evaluation_eligible_subjects"],
            [subject for subject in range(21, 110) if subject not in {88, 92, 100}],
        )
        self.assertTrue(
            set(cohorts["training_subjects"]).isdisjoint(cohorts["evaluation_eligible_subjects"])
        )

    def test_exact_inherited_models_and_global_training_are_frozen(self) -> None:
        models = self.payload["selected_models"]
        self.assertEqual(set(models), {"spectral_baseline_6", "csp_4_empirical"})
        self.assertEqual(models["spectral_baseline_6"]["feature_count"], 6)
        self.assertEqual(models["csp_4_empirical"]["n_components"], 4)
        self.assertIsNone(models["csp_4_empirical"]["covariance_regularization"])
        training = self.payload["final_training"]
        self.assertTrue(training["fit_each_model_once"])
        self.assertEqual(training["training_subject_count"], 20)
        self.assertEqual(training["expected_training_trial_count"], 900)
        self.assertTrue(training["CSP_fit_training_subjects_only"])
        self.assertTrue(training["scaler_fit_training_subjects_only"])
        self.assertTrue(training["LDA_fit_training_subjects_only"])

    def test_no_target_adaptation_route_is_enabled(self) -> None:
        zero_shot = self.payload["zero_shot_evaluation"]
        self.assertFalse(zero_shot["evaluation_subject_EEG_in_training"])
        self.assertFalse(zero_shot["evaluation_subject_labels_in_training"])
        for key in (
            "target_subject_centering",
            "target_subject_scaler_fit",
            "target_subject_covariance_alignment",
            "target_subject_CSP_adaptation",
            "target_subject_transductive_normalization",
            "target_subject_unlabeled_adaptation",
        ):
            self.assertFalse(zero_shot[key])

    def test_metrics_success_rules_and_persistence_are_frozen(self) -> None:
        self.assertEqual(
            self.payload["metrics"]["primary"],
            "unweighted_mean_of_three_run_balanced_accuracies_per_evaluation_subject",
        )
        self.assertEqual(self.payload["metrics"]["group_observational_unit"], "participant")
        self.assertIn("clear_zero_shot_transfer", self.payload["evaluation_success_criteria"])
        self.assertIn("modest_zero_shot_transfer", self.payload["evaluation_success_criteria"])
        persistence = self.payload["persistence"]
        self.assertFalse(persistence["target_subject_adaptation_performed"])
        self.assertFalse(persistence["new_classifier_family_performed"])
        self.assertFalse(persistence["deep_learning_performed"])

    def test_initial_implementation_and_development_evidence_are_byte_exact(self) -> None:
        records = [
            self.payload["initial_policy"],
            self.payload["frozen_method_implementation"],
            *self.payload["development_evidence"]["artifacts"].values(),
        ]
        for record in records:
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], path)
        self.assertFalse(self.payload["development_evidence"]["evaluation_outcome_used"])


if __name__ == "__main__":
    unittest.main()
