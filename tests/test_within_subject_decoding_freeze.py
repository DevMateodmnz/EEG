"""Immutable initial guards for the within-subject decoder study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "within_subject_decoding.json"


class WithinSubjectDecodingInitialFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_study_starts_after_completed_individual_frequency_milestone(self) -> None:
        self.assertEqual(
            self.payload["study_start_git_commit"],
            "4dcbc16ce12703db03569f407c7bf75a851b22ee",
        )
        self.assertEqual(self.payload["study_stage"], "initial_methodology_frozen")
        self.assertTrue(self.payload["evaluation_locked"])
        self.assertFalse(
            self.payload["decoder_development_outcomes_inspected_at_initial_freeze"]
        )
        self.assertFalse(
            self.payload["evaluation_classifier_outcomes_inspected_at_initial_freeze"]
        )

    def test_cohorts_are_exact_disjoint_and_outcome_independent(self) -> None:
        cohorts = self.payload["cohorts"]
        development = cohorts["decoder_development_subjects"]
        requested = cohorts["decoder_evaluation_requested_subjects"]
        eligible = cohorts["decoder_evaluation_eligible_subjects"]
        incompatible = cohorts["technically_incompatible_subjects"]
        self.assertEqual(development, list(range(1, 21)))
        self.assertEqual(requested, list(range(21, 110)))
        self.assertEqual(incompatible, [88, 92, 100])
        self.assertEqual(eligible, [subject for subject in requested if subject not in incompatible])
        self.assertEqual(len(eligible), 86)
        self.assertTrue(set(development).isdisjoint(requested))

    def test_target_and_leave_one_run_out_policy_are_exact(self) -> None:
        target = self.payload["prediction_target"]
        self.assertEqual(
            target["classes"],
            {"both_fists_imagery": 0, "both_feet_imagery": 1},
        )
        self.assertEqual(target["excluded_class"], "T0_rest")
        self.assertEqual(target["task_interval_seconds"], [1.0, 3.0])
        cross_validation = self.payload["cross_validation"]
        self.assertEqual(cross_validation["method"], "leave_one_run_out")
        self.assertTrue(cross_validation["random_trial_split_forbidden"])
        self.assertTrue(cross_validation["supervised_transforms_fit_on_training_fold_only"])
        for fold in cross_validation["folds"]:
            self.assertNotIn(fold["test_run"], fold["training_runs"])
            self.assertEqual(
                set(fold["training_runs"]), {6, 10, 14} - {fold["test_run"]}
            )

    def test_candidate_family_is_small_and_fixed(self) -> None:
        candidates = self.payload["candidate_models"]
        self.assertEqual(
            set(candidates),
            {
                "spectral_baseline_6",
                "spectral_baseline_9",
                "csp_4_empirical",
                "csp_4_ledoit_wolf",
                "csp_6_ledoit_wolf",
            },
        )
        self.assertEqual(sum(name.startswith("spectral_") for name in candidates), 2)
        self.assertEqual(sum(name.startswith("csp_") for name in candidates), 3)
        for name, candidate in candidates.items():
            if name.startswith("csp_"):
                self.assertEqual(candidate["frequency_band_hz"], [8.0, 30.0])
                self.assertIn(candidate["csp_components"], (4, 6))

    def test_primary_metric_and_subject_aggregation_are_predefined(self) -> None:
        metrics = self.payload["metrics"]
        self.assertEqual(metrics["primary"], "balanced_accuracy")
        self.assertEqual(metrics["chance_balanced_accuracy"], 0.5)
        self.assertEqual(
            metrics["subject_aggregation"],
            "unweighted_mean_of_three_fold_balanced_accuracies",
        )
        self.assertEqual(metrics["group_observational_unit"], "participant")
        self.assertIn("useful_decoding_signal", self.payload["evaluation_success_criteria"])
        self.assertIn("mixed_decoding_evidence", self.payload["evaluation_success_criteria"])

    def test_historical_inputs_are_byte_exact(self) -> None:
        for record in self.payload["historical_files"].values():
            path = PROJECT_ROOT / record["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, record["sha256"], path)

    def test_persistence_and_qc_guards_are_frozen(self) -> None:
        persistence = self.payload["persistence"]
        self.assertTrue(persistence["preserve_every_out_of_fold_prediction"])
        self.assertFalse(persistence["persist_full_feature_arrays"])
        self.assertFalse(persistence["persist_fitted_models"])
        quality = self.payload["quality_policy"]
        self.assertEqual(quality["primary"], "all_technically_valid_task_trials_retained")
        self.assertTrue(quality["misclassification_never_justifies_trial_deletion"])
        self.assertFalse(quality["master_epochs_mutated"])


if __name__ == "__main__":
    unittest.main()
