"""Regression guards for cross-subject LOSO development artifacts."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import unittest

from eeg_project.cross_subject_decoding import file_sha256, load_cross_subject_config
from eeg_project.decoding import PROJECT_ROOT


class CrossSubjectDevelopmentArtifactTests(unittest.TestCase):
    ASSETS = PROJECT_ROOT / "docs" / "assets"
    PREFIX = "cross_subject_decoder_development_"

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_cross_subject_config()
        metadata_path = cls.ASSETS / f"{cls.PREFIX}metadata.json"
        if not metadata_path.exists():
            raise unittest.SkipTest("Cross-subject development artifacts are absent.")
        cls.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @classmethod
    def read_rows(cls, suffix: str) -> list[dict[str, str]]:
        with (cls.ASSETS / f"{cls.PREFIX}{suffix}").open(
            newline="", encoding="utf-8"
        ) as input_file:
            return list(csv.DictReader(input_file))

    def test_development_keeps_evaluation_completely_unopened(self) -> None:
        self.assertEqual(self.metadata["development_subjects"], list(range(1, 21)))
        self.assertEqual(self.metadata["development_fold_count"], 20)
        self.assertFalse(self.metadata["evaluation_subject_EEG_loaded"])
        self.assertFalse(
            self.metadata["evaluation_subject_classifier_outcomes_calculated"]
        )
        self.assertFalse(self.metadata["target_adaptation_used"])
        self.assertTrue(self.metadata["every_learned_stage_training_subjects_only"])
        self.assertEqual(len(self.metadata["source_hashes"]), 60)
        self.assertEqual(
            {
                int(Path(path).parent.name.removeprefix("S"))
                for path in self.metadata["source_hashes"]
            },
            set(range(1, 21)),
        )

    def test_every_development_trial_is_predicted_once_per_model(self) -> None:
        rows = self.read_rows("predictions.csv")
        self.assertEqual(len(rows), 1800)
        self.assertEqual({row["model"] for row in rows}, set(self.metadata["models"]))
        counts = Counter((row["model"], row["trial_key"]) for row in rows)
        self.assertEqual(len(counts), 1800)
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(len({row["trial_key"] for row in rows}), 900)
        for row in rows:
            subject = int(row["subject"])
            training = {int(value) for value in row["training_subjects"].split("/")}
            self.assertNotIn(subject, training)
            self.assertEqual(training, set(range(1, 21)) - {subject})
            self.assertEqual(row["zero_shot"], "True")
            self.assertEqual(row["target_adaptation_used"], "False")

    def test_fit_audit_excludes_target_from_CSP_scaler_and_LDA(self) -> None:
        rows = self.read_rows("fit_audit.csv")
        self.assertEqual(len(rows), 40)
        for row in rows:
            target = int(row["target_subject"])
            expected = set(range(1, 21)) - {target}
            self.assertEqual(
                {int(value) for value in row["training_subjects"].split("/")},
                expected,
            )
            self.assertEqual(int(row["training_subject_count"]), 19)
            self.assertEqual(int(row["training_trial_count"]), 855)
            self.assertEqual(
                {int(value) for value in row["scaler_fit_subjects"].split("/")},
                expected,
            )
            self.assertEqual(
                {int(value) for value in row["lda_fit_subjects"].split("/")},
                expected,
            )
            if row["model"] == "csp_4_empirical":
                self.assertEqual(
                    {int(value) for value in row["csp_fit_subjects"].split("/")},
                    expected,
                )
            else:
                self.assertEqual(row["csp_fit_subjects"], "not_applicable")
            self.assertEqual(row["target_subject_absent_from_training"], "True")
            for field in (
                "target_centering_used",
                "target_scaling_fit_used",
                "target_covariance_alignment_used",
                "target_CSP_adaptation_used",
                "target_transductive_normalization_used",
                "target_unlabeled_adaptation_used",
            ):
                self.assertEqual(row[field], "False")

    def test_group_summaries_use_one_value_per_held_out_participant(self) -> None:
        subject_rows = self.read_rows("subject_scores.csv")
        self.assertEqual(len(subject_rows), 40)
        for model in self.metadata["models"]:
            selected = [row for row in subject_rows if row["model"] == model]
            self.assertEqual(len(selected), 20)
            self.assertEqual({int(row["subject"]) for row in selected}, set(range(1, 21)))
            self.assertTrue(
                all(row["group_observational_unit"] == "participant" for row in selected)
            )
            self.assertTrue(all(row["trial_count"] == "45" for row in selected))
        summaries = self.read_rows("model_summary.csv")
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(row["participant_count"] == "20" for row in summaries))

    def test_confusion_and_run_totals_match_predictions(self) -> None:
        predictions = self.read_rows("predictions.csv")
        runs = self.read_rows("run_scores.csv")
        self.assertEqual(len(runs), 120)
        self.assertEqual(sum(int(row["trial_count"]) for row in runs), len(predictions))
        for row in runs:
            total = sum(
                int(row[field])
                for field in (
                    "true_fists_predicted_fists",
                    "true_fists_predicted_feet",
                    "true_feet_predicted_fists",
                    "true_feet_predicted_feet",
                )
            )
            self.assertEqual(total, int(row["trial_count"]))

    def test_historical_files_raw_sources_and_artifact_hashes_are_exact(self) -> None:
        for record in self.config["historical_inputs"].values():
            self.assertEqual(file_sha256(PROJECT_ROOT / record["path"]), record["sha256"])
        for path, expected in self.metadata["source_hashes"].items():
            self.assertEqual(file_sha256(Path(path)), expected)
        self.assertEqual(len(self.metadata["artifact_sha256"]), 7)
        for name, expected in self.metadata["artifact_sha256"].items():
            self.assertEqual(file_sha256(self.ASSETS / name), expected)


if __name__ == "__main__":
    unittest.main()
