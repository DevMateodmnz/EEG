"""Immutable final decoder-method guards written before evaluation outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = PROJECT_ROOT / "config" / "within_subject_decoding_final.json"


class DecoderFinalFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(FINAL_PATH.read_text(encoding="utf-8"))

    def test_final_method_is_frozen_before_evaluation_outcomes(self) -> None:
        self.assertEqual(self.payload["study_stage"], "final_decoder_method_frozen")
        self.assertFalse(
            self.payload["evaluation_classifier_outcomes_inspected_at_final_freeze"]
        )
        self.assertTrue(self.payload["evaluation_unlocked_after_this_freeze_commit"])
        self.assertEqual(self.payload["initial_decoder_freeze_git_commit"], "14e32ae")
        self.assertEqual(self.payload["decoder_implementation_git_commit"], "49a51bc")
        self.assertEqual(self.payload["decoder_development_git_commit"], "bf27724")

    def test_final_cohorts_preserve_subject_level_separation(self) -> None:
        cohorts = self.payload["cohorts"]
        self.assertEqual(cohorts["decoder_development_subjects"], list(range(1, 21)))
        self.assertEqual(cohorts["decoder_evaluation_requested_subjects"], list(range(21, 110)))
        self.assertEqual(cohorts["technically_incompatible_subjects"], [88, 92, 100])
        self.assertEqual(
            cohorts["decoder_evaluation_eligible_subjects"],
            [subject for subject in range(21, 110) if subject not in {88, 92, 100}],
        )

    def test_selected_baseline_and_csp_are_exact(self) -> None:
        baseline = self.payload["selected_spectral_baseline"]
        self.assertEqual(baseline["candidate_name"], "spectral_baseline_6")
        self.assertEqual(baseline["channels"], ["C3", "Cz", "C4"])
        self.assertEqual(baseline["bands_hz"], {"mu": [8.0, 13.0], "beta": [14.0, 30.0]})
        self.assertEqual(baseline["feature_count"], 6)
        csp = self.payload["selected_csp_decoder"]
        self.assertEqual(csp["candidate_name"], "csp_4_empirical")
        self.assertEqual(csp["feature_filter"]["passband_hz"], [8.0, 30.0])
        self.assertEqual(csp["n_components"], 4)
        self.assertIsNone(csp["covariance_regularization"])
        self.assertTrue(csp["log"])

    def test_every_supervised_stage_is_training_fold_only(self) -> None:
        cross_validation = self.payload["cross_validation"]
        self.assertTrue(cross_validation["random_trial_split_forbidden"])
        self.assertTrue(cross_validation["subject_isolation_required"])
        self.assertTrue(cross_validation["fit_CSP_scaler_and_LDA_on_training_runs_only"])
        self.assertTrue(cross_validation["each_trial_predicted_exactly_once_out_of_run"])
        for fold in cross_validation["folds"]:
            self.assertEqual(
                set(fold["training_runs"]), {6, 10, 14} - {fold["test_run"]}
            )
        self.assertTrue(self.payload["shared_classifier"]["scaler"]["fit_training_fold_only"])
        self.assertTrue(self.payload["shared_classifier"]["lda"]["fit_training_fold_only"])

    def test_development_evidence_and_core_method_hashes_are_exact(self) -> None:
        records = [self.payload["initial_policy"]]
        records.extend(self.payload["development_evidence"]["artifacts"].values())
        records.extend(self.payload["frozen_method_implementation"].values())
        for record in records:
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"], path)
        self.assertFalse(self.payload["development_evidence"]["evaluation_subject_outcome_used"])

    def test_metrics_qc_success_and_persistence_are_frozen(self) -> None:
        self.assertEqual(
            self.payload["metrics"]["primary"],
            "unweighted_mean_of_three_held_out_run_balanced_accuracies_per_subject",
        )
        self.assertEqual(self.payload["metrics"]["group_observational_unit"], "participant")
        self.assertIn("useful_decoding_signal", self.payload["evaluation_success_criteria"])
        self.assertIn("mixed_decoding_evidence", self.payload["evaluation_success_criteria"])
        self.assertTrue(self.payload["quality_policy"]["misclassified_trials_never_deleted"])
        persistence = self.payload["persistence"]
        self.assertFalse(persistence["persist_full_feature_arrays"])
        self.assertFalse(persistence["persist_fitted_models"])
        self.assertFalse(persistence["cross_subject_classifier_performed"])
        self.assertFalse(persistence["deep_learning_performed"])


if __name__ == "__main__":
    unittest.main()
