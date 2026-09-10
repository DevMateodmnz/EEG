"""Regression guards for the completed target-calibration evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import unittest

from eeg_project.cross_subject_decoding import file_sha256
from eeg_project.decoding import PROJECT_ROOT
from eeg_project.target_calibration_evaluation import load_final_calibration_config


class TargetCalibrationEvaluationArtifactTests(unittest.TestCase):
    ASSETS = PROJECT_ROOT / "docs" / "assets"
    CHECKPOINTS = PROJECT_ROOT / "outputs" / "target_calibration_checkpoints"
    PREFIX = "target_calibration_evaluation_"
    TRAINING_SUBJECTS = set(range(1, 21))
    SIZES = (0, 4, 8, 12, 14)

    @classmethod
    def setUpClass(cls) -> None:
        cls.final, cls.initial = load_final_calibration_config()
        metadata_path = cls.ASSETS / f"{cls.PREFIX}metadata.json"
        if not metadata_path.exists():
            raise unittest.SkipTest("Target-calibration evaluation artifacts are absent.")
        cls.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @classmethod
    def read_rows(cls, suffix: str) -> list[dict[str, str]]:
        with (cls.ASSETS / f"{cls.PREFIX}{suffix}").open(
            newline="", encoding="utf-8"
        ) as input_file:
            return list(csv.DictReader(input_file))

    def test_all_86_subject_checkpoints_match_frozen_configs_and_sources(self) -> None:
        subjects = self.final["cohorts"]["evaluation_eligible_subjects"]
        self.assertEqual(len(subjects), 86)
        self.assertEqual(set(subjects), set(range(21, 110)) - {88, 92, 100})
        self.assertEqual(self.metadata["evaluation_subject_count"], 86)
        self.assertEqual(self.metadata["checkpoint_count"], 86)
        self.assertEqual(self.metadata["checkpoint_reuse_count_this_run"], 86)

        checkpoint_paths = sorted(self.CHECKPOINTS.glob("subject_*.json"))
        self.assertEqual(len(checkpoint_paths), 86)
        for subject, checkpoint_path in zip(subjects, checkpoint_paths, strict=True):
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["subject"], subject)
            for field in (
                "final_config_sha256",
                "frozen_calibration_core_sha256",
                "frozen_cross_subject_core_sha256",
                "training_source_hashes",
            ):
                self.assertEqual(payload[field], self.metadata[field])
            self.assertEqual(len(payload["target_source_hashes"]), 3)
            for path, expected in payload["target_source_hashes"].items():
                self.assertEqual(self.metadata["evaluation_source_hashes"][path], expected)
            self.assertFalse(payload["source_CSP_or_scaler_fit_target_EEG"])
            self.assertIn(len(payload["target_feature_rows"]), {42, 45})

    def test_every_trial_has_exactly_two_test_predictions_per_size(self) -> None:
        rows = self.read_rows("predictions.csv")
        self.assertEqual(len(rows), 38400)
        counts = Counter(
            (int(row["subject"]), int(row["calibration_size"]), row["trial_key"])
            for row in rows
        )
        self.assertEqual(len(counts), 3840 * len(self.SIZES))
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(len({row["trial_key"] for row in rows}), 3840)
        self.assertEqual(
            {int(row["subject"]) for row in rows},
            set(self.final["cohorts"]["evaluation_eligible_subjects"]),
        )
        for row in rows:
            self.assertNotEqual(int(row["calibration_run"]), int(row["test_run"]))
            self.assertEqual(row["test_run_used_for_calibration"], "False")
            self.assertEqual(row["source_CSP_frozen"], "True")
            self.assertEqual(row["source_scaler_frozen"], "True")
            self.assertEqual(
                {int(value) for value in row["training_subjects"].split("/")},
                self.TRAINING_SUBJECTS,
            )

    def test_fit_audit_proves_balanced_nested_calibration_without_leakage(self) -> None:
        rows = self.read_rows("fit_audit.csv")
        self.assertEqual(len(rows), 86 * len(self.SIZES) * 3)
        grouped: dict[tuple[int, int], dict[int, set[str]]] = defaultdict(dict)
        for row in rows:
            subject = int(row["subject"])
            size = int(row["calibration_size"])
            calibration_run = int(row["calibration_run"])
            calibration_keys = set(filter(None, row["calibration_trial_keys"].split("/")))
            test_keys = set(filter(None, row["test_trial_keys"].split("/")))
            grouped[(subject, calibration_run)][size] = calibration_keys

            self.assertEqual(len(calibration_keys), size)
            self.assertEqual(int(row["calibration_trial_count"]), size)
            self.assertEqual(int(row["calibration_fists_count"]), size // 2)
            self.assertEqual(int(row["calibration_feet_count"]), size // 2)
            self.assertTrue(calibration_keys.isdisjoint(test_keys))
            self.assertEqual(row["test_runs_absent_from_calibration"], "True")
            self.assertEqual(
                {int(value) for value in row["training_subjects"].split("/")},
                self.TRAINING_SUBJECTS,
            )
            for field in (
                "target_EEG_used_for_source_CSP_fit",
                "target_EEG_used_for_source_scaler_fit",
                "unlabeled_test_EEG_used_for_adaptation",
                "target_CSP_refit",
                "target_scaler_refit",
            ):
                self.assertEqual(row[field], "False")
            self.assertEqual(
                row["decision_layer_fitted_from_calibration_only"],
                "False" if size == 0 else "True",
            )

        self.assertEqual(len(grouped), 86 * 3)
        for sets_by_size in grouped.values():
            self.assertEqual(set(sets_by_size), set(self.SIZES))
            for previous, current in zip(self.SIZES[:-1], self.SIZES[1:], strict=True):
                self.assertTrue(sets_by_size[previous] < sets_by_size[current])

    def test_participant_curve_and_zero_shot_regression_are_exact(self) -> None:
        participants = self.read_rows("participant_scores.csv")
        self.assertEqual(len(participants), 86 * len(self.SIZES) * 2)
        keys = Counter(
            (row["subject"], row["calibration_size"], row["qc_sensitivity"])
            for row in participants
        )
        self.assertEqual(set(keys.values()), {1})
        self.assertTrue(
            all(row["group_observational_unit"] == "participant" for row in participants)
        )

        historical = {}
        with (
            self.ASSETS / "cross_subject_decoder_evaluation_subject_scores.csv"
        ).open(newline="", encoding="utf-8") as input_file:
            for row in csv.DictReader(input_file):
                if row["model"] == "csp_4_empirical":
                    historical[int(row["subject"])] = float(
                        row["primary_mean_run_balanced_accuracy"]
                    )
        zero = {
            int(row["subject"]): float(
                row["primary_mean_test_run_balanced_accuracy"]
            )
            for row in participants
            if row["calibration_size"] == "0" and row["qc_sensitivity"] == "False"
        }
        self.assertEqual(zero, historical)

        curve = self.read_rows("curve_summary.csv")
        self.assertEqual(len(curve), len(self.SIZES))
        self.assertEqual([int(row["calibration_size"]) for row in curve], list(self.SIZES))
        self.assertTrue(all(row["participant_count"] == "86" for row in curve))
        self.assertTrue(all(row["convincing_benefit"] == "False" for row in curve))
        self.assertIsNone(self.metadata["smallest_convincing_calibration_size"])

    def test_frozen_sources_history_and_report_hashes_are_unchanged(self) -> None:
        self.assertEqual(len(self.metadata["training_source_hashes"]), 60)
        self.assertEqual(len(self.metadata["evaluation_source_hashes"]), 258)
        self.assertEqual(len(self.metadata["source_hashes"]), 318)
        for path, expected in self.metadata["source_hashes"].items():
            self.assertEqual(file_sha256(Path(path)), expected)
        for record in self.initial["historical_inputs"].values():
            self.assertEqual(file_sha256(PROJECT_ROOT / record["path"]), record["sha256"])
        self.assertFalse(self.metadata["zero_shot_historical_outputs_modified"])
        self.assertFalse(self.metadata["within_subject_historical_outputs_modified"])
        self.assertEqual(len(self.metadata["artifact_sha256"]), 14)
        for name, expected in self.metadata["artifact_sha256"].items():
            self.assertEqual(file_sha256(self.ASSETS / name), expected)

    def test_metadata_records_strict_frozen_source_boundaries(self) -> None:
        self.assertEqual(self.metadata["selected_method"], "threshold")
        self.assertEqual(self.metadata["calibration_sizes"], list(self.SIZES))
        self.assertEqual(self.metadata["source_training_subject_count"], 20)
        self.assertEqual(self.metadata["source_training_trial_count"], 900)
        self.assertFalse(self.metadata["source_CSP_or_scaler_fit_target_EEG"])
        self.assertFalse(self.metadata["unlabeled_test_EEG_used_for_adaptation"])
        self.assertTrue(self.metadata["one_value_per_participant_per_size"])
        self.assertFalse(self.metadata["new_classifier_family_performed"])
        self.assertFalse(self.metadata["deep_learning_performed"])

    def test_updated_documentation_local_links_resolve(self) -> None:
        documents = (
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "cross_subject_decoding.md",
            PROJECT_ROOT / "docs" / "minimal_target_calibration.md",
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
