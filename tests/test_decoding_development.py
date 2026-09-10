"""Regression guards for decoder development outputs before final freeze."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT_ROOT / "docs" / "assets"
PREFIX = "within_subject_decoder_development_"


def read_csv(suffix: str) -> list[dict[str, str]]:
    with (ASSETS / f"{PREFIX}{suffix}").open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


class DecoderDevelopmentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = json.loads(
            (ASSETS / f"{PREFIX}metadata.json").read_text(encoding="utf-8")
        )

    def test_development_never_opens_evaluation_classifier_outcomes(self) -> None:
        self.assertEqual(self.metadata["development_subjects"], list(range(1, 21)))
        self.assertFalse(self.metadata["evaluation_subject_classifier_outcomes_calculated"])
        self.assertEqual(self.metadata["study_stage"], "decoder_method_development_only")
        self.assertEqual(
            self.metadata["configuration_sha256"],
            hashlib.sha256(
                (PROJECT_ROOT / "config/within_subject_decoding.json").read_bytes()
            ).hexdigest(),
        )

    def test_candidate_outputs_use_one_complete_score_per_development_subject(self) -> None:
        subject_rows = read_csv("subject_scores.csv")
        primary = [row for row in subject_rows if row["qc_sensitivity"] == "False"]
        sensitivity = [row for row in subject_rows if row["qc_sensitivity"] == "True"]
        models = {row["model"] for row in primary}
        self.assertEqual(len(models), 5)
        self.assertEqual(len(primary), 100)
        self.assertEqual(len(sensitivity), 100)
        for model in models:
            rows = [row for row in primary if row["model"] == model]
            self.assertEqual({int(row["subject"]) for row in rows}, set(range(1, 21)))
            self.assertTrue(all(row["included_trial_count"] == "45" for row in rows))
            self.assertTrue(all(row["group_observational_unit"] == "participant" for row in rows))

    def test_fold_outputs_are_run_held_out_and_complete(self) -> None:
        rows = read_csv("fold_scores.csv")
        self.assertEqual(len(rows), 20 * 5 * 2 * 3)
        for row in rows:
            held_out = int(row["test_run"])
            training = {int(run) for run in row["training_runs"].split("/")}
            self.assertEqual(training, {6, 10, 14} - {held_out})
            self.assertGreater(int(row["trial_count"]), 0)
            self.assertLessEqual(int(row["trial_count"]), 15)
            if row["qc_sensitivity"] == "False":
                self.assertEqual(int(row["trial_count"]), 15)

    def test_frozen_selection_rule_yields_exact_family_choices(self) -> None:
        rows = read_csv("candidate_summary.csv")
        selected = {
            row["model"] for row in rows if row["selected_final_family_candidate"] == "True"
        }
        self.assertEqual(selected, {"spectral_baseline_6", "csp_4_empirical"})
        self.assertEqual(self.metadata["selected_spectral_baseline"], "spectral_baseline_6")
        self.assertEqual(self.metadata["selected_csp_model"], "csp_4_empirical")
        self.assertEqual(self.metadata["representative_csp_pattern_subject"], 2)

    def test_development_artifact_hashes_are_exact(self) -> None:
        for name, expected in self.metadata["artifacts"].items():
            path = ASSETS / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        self.assertFalse(self.metadata["full_feature_arrays_persisted"])
        self.assertFalse(self.metadata["fitted_models_persisted"])


if __name__ == "__main__":
    unittest.main()
