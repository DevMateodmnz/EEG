"""Regression guards for completed Riemannian held-out artifacts."""
from __future__ import annotations
import csv, hashlib, json, unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT_ROOT / "docs" / "assets"
PREFIX = "riemannian_decoder_evaluation_"

class RiemannianEvaluationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((ASSETS / f"{PREFIX}metadata.json").read_text())
    @staticmethod
    def rows(suffix):
        with (ASSETS / f"{PREFIX}{suffix}").open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    def test_all_subjects_checkpoints_and_predictions_are_complete(self):
        self.assertEqual(self.metadata["evaluation_subject_count"], 86)
        self.assertEqual(self.metadata["checkpoint_count"], 86)
        self.assertEqual(self.metadata["evaluation_trial_count"], 3840)
        predictions = self.rows("predictions.csv")
        self.assertEqual(len(predictions), 3840)
        counts = Counter((r["subject"], r["trial_key"]) for r in predictions)
        self.assertEqual(set(counts.values()), {1})
        self.assertTrue(all(r["target_adaptation_used"] == "False" for r in predictions))
    def test_fit_audit_proves_no_target_EEG_enters_learned_stages(self):
        audits = self.rows("fit_audit.csv")
        self.assertEqual(len(audits), 86)
        for row in audits:
            self.assertEqual(row["training_subject_count"], "20")
            self.assertEqual(row["training_trial_count"], "900")
            self.assertEqual(row["target_subject_absent_from_training"], "True")
            for field in ("test_EEG_used_for_reference_fit", "test_EEG_used_for_tangent_basis_fit", "test_EEG_used_for_scaler_fit", "test_EEG_used_for_classifier_fit", "test_labels_used_for_fitting", "target_unlabeled_adaptation_used"):
                self.assertEqual(row[field], "False")
    def test_historical_results_and_artifact_hashes_are_unchanged(self):
        self.assertFalse(self.metadata["historical_CSP_outputs_modified"])
        self.assertFalse(self.metadata["historical_spectral_outputs_modified"])
        self.assertEqual(len(self.metadata["training_source_hashes"]), 60)
        self.assertEqual(len(self.metadata["evaluation_source_hashes"]), 258)
        for name, digest in self.metadata["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((ASSETS / name).read_bytes()).hexdigest(), digest)
    def test_frozen_no_improvement_result_is_exact(self):
        group = self.rows("group_summary.csv")[0]
        paired = self.rows("paired_summary.csv")[0]
        self.assertAlmostEqual(float(group["median_balanced_accuracy"]), 0.5729166666666667)
        self.assertAlmostEqual(float(paired["median_paired_difference"]), -0.023809523809523836)
        self.assertEqual(self.metadata["interpretation_category"], "no improvement")

if __name__ == "__main__": unittest.main()
