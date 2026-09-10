"""Regression guards for the completed zero-shot cross-subject evaluation."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import re
import unittest

from eeg_project.cross_subject_decoding import file_sha256
from eeg_project.cross_subject_evaluation import load_final_cross_subject_config
from eeg_project.decoding import PROJECT_ROOT


class CrossSubjectEvaluationArtifactTests(unittest.TestCase):
    ASSETS = PROJECT_ROOT / "docs" / "assets"
    CHECKPOINTS = PROJECT_ROOT / "outputs" / "cross_subject_decoder_checkpoints"
    PREFIX = "cross_subject_decoder_evaluation_"
    TRAINING_SUBJECTS = set(range(1, 21))

    @classmethod
    def setUpClass(cls) -> None:
        cls.final, cls.initial = load_final_cross_subject_config()
        metadata_path = cls.ASSETS / f"{cls.PREFIX}metadata.json"
        if not metadata_path.exists():
            raise unittest.SkipTest("Cross-subject evaluation artifacts are absent.")
        cls.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @classmethod
    def read_rows(cls, suffix: str) -> list[dict[str, str]]:
        with (cls.ASSETS / f"{cls.PREFIX}{suffix}").open(
            newline="", encoding="utf-8"
        ) as input_file:
            return list(csv.DictReader(input_file))

    def test_all_86_eligible_subjects_have_validated_checkpoints(self) -> None:
        subjects = self.final["cohorts"]["evaluation_eligible_subjects"]
        self.assertEqual(len(subjects), 86)
        self.assertEqual(set(subjects), set(range(21, 110)) - {88, 92, 100})
        self.assertEqual(self.metadata["evaluation_subject_count"], 86)
        self.assertEqual(self.metadata["checkpoint_count"], 86)
        self.assertEqual(self.metadata["checkpoint_reuse_count_this_run"], 86)

        observed: list[int] = []
        for subject in subjects:
            checkpoint = self.CHECKPOINTS / f"subject_{subject:03d}.json"
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["subject"], subject)
            self.assertEqual(
                payload["final_config_sha256"], self.metadata["final_config_sha256"]
            )
            self.assertEqual(
                payload["frozen_core_sha256"], self.metadata["frozen_core_sha256"]
            )
            self.assertEqual(
                payload["training_source_hashes"],
                self.metadata["training_source_hashes"],
            )
            self.assertEqual(len(payload["target_source_hashes"]), 3)
            for path, expected in payload["target_source_hashes"].items():
                self.assertEqual(self.metadata["evaluation_source_hashes"][path], expected)
            self.assertFalse(payload["target_adaptation_used"])
            observed.append(subject)
        self.assertEqual(observed, subjects)

    def test_every_target_trial_has_one_zero_shot_prediction_per_model(self) -> None:
        rows = self.read_rows("predictions.csv")
        self.assertEqual(len(rows), 7680)
        counts = Counter((row["model"], row["trial_key"]) for row in rows)
        self.assertEqual(len(counts), 7680)
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(len({row["trial_key"] for row in rows}), 3840)
        self.assertEqual(
            {int(row["subject"]) for row in rows},
            set(self.final["cohorts"]["evaluation_eligible_subjects"]),
        )
        for row in rows:
            self.assertEqual(
                {int(value) for value in row["training_subjects"].split("/")},
                self.TRAINING_SUBJECTS,
            )
            self.assertNotIn(int(row["subject"]), self.TRAINING_SUBJECTS)
            self.assertEqual(row["zero_shot"], "True")
            self.assertEqual(row["target_adaptation_used"], "False")
            self.assertIn(f"R{int(row['run']):02d}", row["trial_key"])

    def test_fit_audit_proves_no_target_csp_scaler_or_lda_fit(self) -> None:
        rows = self.read_rows("fit_audit.csv")
        self.assertEqual(len(rows), 172)
        for row in rows:
            self.assertEqual(int(row["training_subject_count"]), 20)
            self.assertEqual(int(row["training_trial_count"]), 900)
            self.assertEqual(
                {int(value) for value in row["training_subjects"].split("/")},
                self.TRAINING_SUBJECTS,
            )
            self.assertEqual(
                {int(value) for value in row["scaler_fit_subjects"].split("/")},
                self.TRAINING_SUBJECTS,
            )
            self.assertEqual(
                {int(value) for value in row["lda_fit_subjects"].split("/")},
                self.TRAINING_SUBJECTS,
            )
            if row["model"] == "csp_4_empirical":
                self.assertEqual(
                    {int(value) for value in row["csp_fit_subjects"].split("/")},
                    self.TRAINING_SUBJECTS,
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

    def test_participants_confusions_and_frozen_decisions_are_exact(self) -> None:
        subjects = self.read_rows("subject_scores.csv")
        self.assertEqual(len(subjects), 172)
        self.assertTrue(
            all(row["group_observational_unit"] == "participant" for row in subjects)
        )
        self.assertEqual({int(row["trial_count"]) for row in subjects}, {42, 45})
        self.assertEqual(sum(int(row["trial_count"]) for row in subjects), 7680)
        summaries = self.read_rows("group_summary.csv")
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(row["participant_count"] == "86" for row in summaries))
        confusion = self.read_rows("confusion_matrices.csv")
        self.assertEqual(len(confusion), 2)
        self.assertTrue(all(row["trial_count"] == "3840" for row in confusion))
        self.assertEqual(
            self.metadata["transfer_categories"],
            {
                "spectral_baseline_6": "no convincing zero-shot transfer",
                "csp_4_empirical": "no convincing zero-shot transfer",
            },
        )
        self.assertEqual(
            self.metadata["paired_transfer_category"], "CSP transfers better"
        )

    def test_sources_historical_inputs_and_report_hashes_are_unchanged(self) -> None:
        self.assertEqual(len(self.metadata["training_source_hashes"]), 60)
        self.assertEqual(len(self.metadata["evaluation_source_hashes"]), 258)
        self.assertEqual(len(self.metadata["source_hashes"]), 318)
        for path, expected in self.metadata["source_hashes"].items():
            self.assertEqual(file_sha256(Path(path)), expected)
        for record in self.initial["historical_inputs"].values():
            self.assertEqual(
                file_sha256(PROJECT_ROOT / record["path"]), record["sha256"]
            )
        self.assertFalse(
            self.metadata["historical_within_subject_predictions_modified"]
        )
        self.assertEqual(len(self.metadata["artifact_sha256"]), 16)
        for name, expected in self.metadata["artifact_sha256"].items():
            self.assertEqual(file_sha256(self.ASSETS / name), expected)

    def test_metadata_records_strict_zero_shot_boundaries(self) -> None:
        self.assertFalse(self.metadata["target_adaptation_used"])
        self.assertFalse(
            self.metadata["target_subject_EEG_used_for_CSP_scaler_or_LDA_fit"]
        )
        self.assertTrue(self.metadata["one_primary_value_per_evaluation_participant"])
        self.assertFalse(self.metadata["new_classifier_family_performed"])
        self.assertFalse(self.metadata["deep_learning_performed"])

    def test_updated_documentation_local_links_resolve(self) -> None:
        documents = (
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "cross_subject_decoding.md",
            PROJECT_ROOT / "docs" / "decoding_reliability.md",
            PROJECT_ROOT / "docs" / "reproducibility.md",
        )
        missing: list[str] = []
        for document in documents:
            text = document.read_text(encoding="utf-8")
            for raw_target in re.findall(r"\]\(([^)]+)\)", text):
                target = raw_target.strip("<>").split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                if not (document.parent / target).resolve().exists():
                    missing.append(f"{document.relative_to(PROJECT_ROOT)} -> {target}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
