"""Regression checks for completed decoder reliability report artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import unittest

import numpy as np

from eeg_project.decoding import PROJECT_ROOT
from eeg_project.decoding_reliability import file_sha256, load_reliability_config
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY
from eeg_project.provenance import resolve_source_id


class DecodingReliabilityArtifactTests(unittest.TestCase):
    ASSETS = PROJECT_ROOT / "docs" / "assets"
    CHECKPOINTS = PROJECT_ROOT / "outputs" / "decoding_reliability_checkpoints"
    PREFIX = "decoding_reliability_"

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_reliability_config()
        metadata_path = cls.ASSETS / f"{cls.PREFIX}metadata.json"
        if not metadata_path.exists():
            raise unittest.SkipTest("Reliability report artifacts are absent.")
        cls.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @classmethod
    def read_rows(cls, suffix: str) -> list[dict[str, str]]:
        with (cls.ASSETS / f"{cls.PREFIX}{suffix}").open(
            newline="", encoding="utf-8"
        ) as input_file:
            return list(csv.DictReader(input_file))

    def test_all_subjects_runs_and_historical_scores_are_preserved(self) -> None:
        reliability = self.read_rows("subject_reliability.csv")
        self.assertEqual(len(reliability), 86)
        historical_path = (
            self.ASSETS / "within_subject_decoder_evaluation_subject_scores.csv"
        )
        with historical_path.open(newline="", encoding="utf-8") as input_file:
            historical_rows = list(csv.DictReader(input_file))
        historical = {
            int(row["subject"]): float(row["primary_mean_fold_balanced_accuracy"])
            for row in historical_rows
            if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"
        }
        self.assertEqual({int(row["subject"]) for row in reliability}, set(historical))
        for row in reliability:
            subject = int(row["subject"])
            self.assertEqual(
                float(row["historical_primary_mean_balanced_accuracy"]), historical[subject]
            )
            run_scores = [
                float(row[f"balanced_accuracy_run_{run}"]) for run in (6, 10, 14)
            ]
            self.assertAlmostEqual(np.mean(run_scores), historical[subject])
        categories = self.read_rows("category_summary.csv")[:4]
        self.assertEqual(
            [int(row["participant_count"]) for row in categories], [55, 15, 11, 5]
        )

    def test_run_summaries_reproduce_from_258_historical_fold_rows(self) -> None:
        summaries = {int(row["test_run"]): row for row in self.read_rows("run_summary.csv")}
        historical_path = self.ASSETS / "within_subject_decoder_evaluation_fold_scores.csv"
        with historical_path.open(newline="", encoding="utf-8") as input_file:
            folds = [
                row
                for row in csv.DictReader(input_file)
                if row["model"] == "csp_4_empirical"
                and row["qc_sensitivity"] == "False"
            ]
        self.assertEqual(len(folds), 258)
        for run in (6, 10, 14):
            selected = [float(row["balanced_accuracy"]) for row in folds if int(row["test_run"]) == run]
            self.assertEqual(len(selected), 86)
            self.assertEqual(float(summaries[run]["median_balanced_accuracy"]), np.median(selected))
            self.assertEqual(float(summaries[run]["minimum_balanced_accuracy"]), np.min(selected))
            self.assertEqual(float(summaries[run]["maximum_balanced_accuracy"]), np.max(selected))

    def test_run_shift_rows_and_checkpoints_are_label_independent_and_exact(self) -> None:
        rows = self.read_rows("run_shift_folds.csv")
        self.assertEqual(len(rows), 258)
        self.assertEqual({int(row["subject"]) for row in rows}, {
            subject for subject in range(21, 110) if subject not in {88, 92, 100}
        })
        for row in rows:
            test_run = int(row["test_run"])
            training_runs = {int(run) for run in row["training_runs"].split("/")}
            self.assertEqual(training_runs, {6, 10, 14} - {test_run})
            self.assertEqual(row["labels_used_for_distance"], "False")
            self.assertTrue(np.isfinite(float(row["affine_invariant_covariance_distance"])))
        expected_config = self.metadata["reliability_config_sha256"]
        evaluation_metadata = json.loads(
            (self.ASSETS / "within_subject_decoder_evaluation_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoints = sorted(self.CHECKPOINTS.glob("subject_*.json"))
        self.assertEqual(len(checkpoints), 86)
        for path in checkpoints:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["reliability_config_sha256"], expected_config)
            self.assertFalse(payload["class_labels_used"])
            self.assertFalse(payload["classifier_refit"])
            for source_name, source_hash in payload["source_hashes"].items():
                self.assertEqual(evaluation_metadata["source_hashes"][source_name], source_hash)

    def test_historical_predictions_eeg_artifacts_and_raw_edfs_are_unchanged(self) -> None:
        for record in self.config["historical_inputs"].values():
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(file_sha256(path), record["sha256"])
        self.assertEqual(
            self.metadata["historical_prediction_sha256"],
            self.config["historical_inputs"]["predictions"]["sha256"],
        )
        self.assertEqual(len(self.metadata["source_hashes"]), 258)
        for source_name, expected in self.metadata["source_hashes"].items():
            self.assertEqual(file_sha256(resolve_source_id(source_name, DEFAULT_DATA_DIRECTORY)), expected)
        self.assertFalse(self.metadata["historical_predictions_modified"])
        self.assertFalse(self.metadata["classifier_refit"])
        self.assertFalse(self.metadata["run_shift_class_labels_used"])

    def test_report_artifacts_match_recorded_hashes(self) -> None:
        self.assertEqual(len(self.metadata["artifact_sha256"]), 15)
        for name, expected in self.metadata["artifact_sha256"].items():
            self.assertEqual(file_sha256(self.ASSETS / name), expected)

    def test_reliability_documentation_links_resolve(self) -> None:
        documents = (
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "within_subject_decoding.md",
            PROJECT_ROOT / "docs" / "decoding_reliability.md",
        )
        missing: list[str] = []
        for document in documents:
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                if target.startswith(("http://", "https://", "#")):
                    continue
                local_target = target.split("#", maxsplit=1)[0]
                if not (document.parent / local_target).resolve().exists():
                    missing.append(f"{document.relative_to(PROJECT_ROOT)} -> {target}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
