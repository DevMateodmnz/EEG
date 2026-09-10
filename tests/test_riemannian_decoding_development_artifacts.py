"""Regression guards for completed Riemannian LOSO development evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT_ROOT / "docs" / "assets"
PREFIX = "riemannian_decoder_development_"


def rows(suffix: str) -> list[dict[str, str]]:
    with (ASSETS / f"{PREFIX}{suffix}").open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


class RiemannianDevelopmentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((PROJECT_ROOT / "config" / "riemannian_decoding.json").read_text())
        cls.metadata = json.loads((ASSETS / f"{PREFIX}metadata.json").read_text())

    def test_development_cohort_is_exact_and_evaluation_is_unopened(self) -> None:
        self.assertEqual(self.metadata["development_subjects"], list(range(1, 21)))
        self.assertEqual(self.metadata["development_fold_count"], 20)
        self.assertEqual(self.metadata["training_subject_count_per_fold"], 19)
        self.assertFalse(self.metadata["evaluation_subject_classifier_outcomes_calculated"])
        self.assertFalse(self.metadata["evaluation_subject_EEG_loaded"])
        self.assertFalse(self.metadata["target_adaptation_used"])
        self.assertEqual(len(self.metadata["source_hashes"]), 60)

    def test_each_development_trial_is_predicted_once_per_candidate(self) -> None:
        predictions = rows("predictions.csv")
        self.assertEqual(len(predictions), 20 * 45 * 2)
        keys = {(row["subject"], row["model"], row["trial_key"]) for row in predictions}
        self.assertEqual(len(keys), len(predictions))
        self.assertEqual({row["model"] for row in predictions}, {"shrinkage_lda", "l2_logistic_regression"})
        self.assertTrue(all(row["zero_shot"] == "True" for row in predictions))
        self.assertTrue(all(row["target_adaptation_used"] == "False" for row in predictions))

    def test_fit_audit_proves_all_learned_stages_exclude_held_out_person(self) -> None:
        audits = rows("fit_audit.csv")
        self.assertEqual(len(audits), 40)
        for audit in audits:
            target = int(audit["target_subject"])
            training = {int(subject) for subject in audit["training_subjects"].split("/")}
            self.assertEqual(len(training), 19)
            self.assertNotIn(target, training)
            self.assertEqual(audit["training_trial_count"], "855")
            self.assertEqual(audit["reference_mean_fit_subjects"], audit["training_subjects"])
            self.assertEqual(audit["tangent_basis_fit_subjects"], audit["training_subjects"])
            self.assertEqual(audit["scaler_fit_subjects"], audit["training_subjects"])
            self.assertEqual(audit["classifier_fit_subjects"], audit["training_subjects"])
            for field in (
                "test_EEG_used_for_reference_fit", "test_EEG_used_for_tangent_basis_fit",
                "test_EEG_used_for_scaler_fit", "test_EEG_used_for_classifier_fit",
                "test_labels_used_for_fitting", "target_unlabeled_adaptation_used",
            ):
                self.assertEqual(audit[field], "False")

    def test_subject_aggregation_and_global_classifier_choice_are_exact(self) -> None:
        subject_rows = rows("subject_scores.csv")
        primary = [row for row in subject_rows if row["qc_sensitivity"] == "False"]
        self.assertEqual(len(primary), 40)
        self.assertEqual(
            {(row["subject"], row["model"]) for row in primary},
            {(str(subject), model) for subject in range(1, 21) for model in ("shrinkage_lda", "l2_logistic_regression")},
        )
        selection = rows("method_selection.csv")
        self.assertEqual(len(selection), 1)
        row = selection[0]
        self.assertEqual(row["selected_classifier"], "shrinkage_lda")
        self.assertAlmostEqual(float(row["median_logistic_minus_lda"]), 0.011904761904761918)
        self.assertEqual(row["logistic_better_count"], "10")
        self.assertEqual(row["lda_better_count"], "9")
        self.assertEqual(row["tie_count"], "1")
        self.assertEqual(row["logistic_better_fraction"], "0.5")
        self.assertEqual(row["evaluation_outcome_used"], "False")

    def test_report_artifacts_match_recorded_hashes(self) -> None:
        for name, expected in self.metadata["artifact_sha256"].items():
            observed = hashlib.sha256((ASSETS / name).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, name)


if __name__ == "__main__":
    unittest.main()
