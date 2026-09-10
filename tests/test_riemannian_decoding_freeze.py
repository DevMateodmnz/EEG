"""Immutable pre-development guards for the Riemannian decoding study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "riemannian_decoding.json"


class RiemannianDecodingFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_cohorts_are_exact_disjoint_and_evaluation_is_locked(self) -> None:
        cohorts = self.config["cohorts"]
        self.assertEqual(cohorts["development_subjects"], list(range(1, 21)))
        self.assertEqual(cohorts["evaluation_requested_subjects"], list(range(21, 110)))
        self.assertEqual(cohorts["technically_incompatible_subjects"], [88, 92, 100])
        self.assertEqual(
            cohorts["evaluation_eligible_subjects"],
            [subject for subject in range(21, 110) if subject not in {88, 92, 100}],
        )
        self.assertTrue(
            set(cohorts["development_subjects"]).isdisjoint(
                cohorts["evaluation_eligible_subjects"]
            )
        )
        self.assertTrue(self.config["evaluation_locked"])
        self.assertFalse(self.config["development_classifier_outcomes_inspected_at_freeze"])
        self.assertFalse(self.config["evaluation_classifier_outcomes_inspected_at_freeze"])
        self.assertFalse(self.config["evaluation_subject_EEG_loaded_at_freeze"])

    def test_full_covariance_geometry_and_tiny_candidate_set_are_frozen(self) -> None:
        representation = self.config["frozen_EEG_representation"]
        self.assertEqual(representation["passband_hz"], [8.0, 30.0])
        self.assertEqual(representation["target_interval_seconds"], [1.0, 3.0])
        self.assertEqual(representation["channels"], 64)
        self.assertEqual(representation["samples_per_trial"], 320)
        covariance = self.config["covariance"]
        self.assertEqual(covariance["estimator"], "sklearn.covariance.LedoitWolf")
        self.assertTrue(covariance["trace_normalization"])
        self.assertTrue(covariance["must_be_symmetric_positive_definite"])
        tangent = self.config["tangent_space"]
        self.assertEqual(tangent["reference_mean_metric"], "riemann")
        self.assertEqual(tangent["mapping_metric"], "riemann")
        self.assertEqual(tangent["feature_count"], 2080)
        self.assertFalse(tangent["tsupdate"])
        self.assertEqual(set(self.config["candidate_linear_classifiers"]), {"shrinkage_lda", "l2_logistic_regression"})
        policy = self.config["candidate_policy"]
        self.assertTrue(policy["covariance_estimator_fixed_before_development"])
        self.assertTrue(policy["tangent_space_method_fixed_before_development"])
        self.assertTrue(policy["SVM_forbidden"])
        self.assertTrue(policy["neural_network_forbidden"])

    def test_all_learned_stages_forbid_target_participant_input(self) -> None:
        guards = self.config["zero_shot_leakage_guards"]
        for key in (
            "reference_mean_fit_training_subjects_only",
            "tangent_space_basis_fit_training_subjects_only",
            "scaler_fit_training_subjects_only",
            "classifier_fit_training_subjects_only",
        ):
            self.assertTrue(guards[key])
        for key in (
            "target_subject_feature_selection",
            "target_subject_hyperparameter_selection",
            "target_subject_centering",
            "target_subject_scaling",
            "target_subject_covariance_alignment",
            "target_subject_transductive_normalization",
            "target_subject_unlabeled_adaptation",
            "target_tangent_space_update",
        ):
            self.assertFalse(guards[key])

    def test_participant_metric_and_practical_gate_are_predefined(self) -> None:
        self.assertEqual(
            self.config["metrics"]["primary"],
            "unweighted_mean_of_three_run_balanced_accuracies_per_evaluation_subject",
        )
        self.assertEqual(self.config["metrics"]["group_observational_unit"], "participant")
        clear = self.config["success_criteria"]["clear_representation_improvement"]
        self.assertEqual(clear["minimum_median_Riemannian_minus_CSP"], 0.03)
        self.assertEqual(clear["minimum_fraction_Riemannian_better_than_CSP"], 0.6)
        self.assertEqual(clear["maximum_one_sided_exact_sign_p"], 0.05)
        self.assertEqual(self.config["success_criteria"]["otherwise"], "no_improvement")

    def test_historical_inputs_are_byte_exact(self) -> None:
        for record in self.config["historical_inputs"].values():
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], path)


if __name__ == "__main__":
    unittest.main()
