"""Regression guards for minimal-calibration development evidence."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import unittest

from eeg_project.cross_subject_decoding import file_sha256
from eeg_project.decoding import PROJECT_ROOT
from eeg_project.target_calibration import CALIBRATION_METHODS, CALIBRATION_SIZES


class TargetCalibrationDevelopmentArtifactTests(unittest.TestCase):
    ASSETS = PROJECT_ROOT / "docs" / "assets"
    PREFIX = "target_calibration_development_"

    @classmethod
    def setUpClass(cls) -> None:
        metadata_path = cls.ASSETS / f"{cls.PREFIX}metadata.json"
        if not metadata_path.exists():
            raise unittest.SkipTest("Calibration development artifacts are absent.")
        cls.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cls.config = json.loads(
            (PROJECT_ROOT / "config" / "minimal_target_calibration.json").read_text(
                encoding="utf-8"
            )
        )

    @classmethod
    def read_rows(cls, suffix: str) -> list[dict[str, str]]:
        with (cls.ASSETS / f"{cls.PREFIX}{suffix}").open(
            newline="", encoding="utf-8"
        ) as input_file:
            return list(csv.DictReader(input_file))

    def test_development_keeps_evaluation_locked_and_uses_exact_cohort(self) -> None:
        self.assertEqual(self.metadata["development_subjects"], list(range(1, 21)))
        self.assertEqual(self.metadata["development_subject_count"], 20)
        self.assertFalse(self.metadata["evaluation_subject_EEG_loaded"])
        self.assertFalse(self.metadata["evaluation_calibration_outcomes_calculated"])
        self.assertFalse(self.metadata["target_CSP_or_scaler_refit"])
        self.assertEqual(self.metadata["selected_method"], "threshold")
        self.assertEqual(len(self.metadata["source_hashes"]), 60)

    def test_every_prediction_has_disjoint_calibration_and_test_provenance(self) -> None:
        rows = self.read_rows("predictions.csv")
        self.assertEqual(len(rows), 18000)
        counts = Counter(
            (
                row["subject"],
                row["method"],
                row["calibration_size"],
                row["calibration_run"],
                row["trial_key"],
            )
            for row in rows
        )
        self.assertEqual(len(counts), len(rows))
        self.assertEqual(set(counts.values()), {1})
        for row in rows:
            self.assertNotEqual(row["calibration_run"], row["test_run"])
            self.assertEqual(row["test_run_used_for_calibration"], "False")
            self.assertEqual(row["source_CSP_frozen"], "True")
            self.assertEqual(row["source_scaler_frozen"], "True")
            calibration = set(filter(None, row["calibration_trial_keys"].split("/")))
            self.assertNotIn(row["trial_key"], calibration)

    def test_audits_prove_nested_balanced_sets_and_source_isolation(self) -> None:
        rows = self.read_rows("fit_audit.csv")
        self.assertEqual(len(rows), 600)
        grouped: dict[tuple[str, str, str], dict[int, set[str]]] = {}
        for row in rows:
            subject = int(row["subject"])
            training = {int(value) for value in row["training_subjects"].split("/")}
            self.assertEqual(training, set(range(1, 21)) - {subject})
            self.assertEqual(row["source_CSP_fit_subjects"], row["training_subjects"])
            self.assertEqual(row["source_scaler_fit_subjects"], row["training_subjects"])
            self.assertEqual(row["test_runs_absent_from_calibration"], "True")
            self.assertEqual(row["target_EEG_used_for_source_CSP_fit"], "False")
            self.assertEqual(row["target_EEG_used_for_source_scaler_fit"], "False")
            self.assertEqual(row["unlabeled_test_EEG_used_for_adaptation"], "False")
            self.assertEqual(row["target_CSP_refit"], "False")
            self.assertEqual(row["target_scaler_refit"], "False")
            size = int(row["calibration_size"])
            self.assertEqual(int(row["calibration_trial_count"]), size)
            self.assertEqual(int(row["calibration_fists_count"]), size // 2)
            self.assertEqual(int(row["calibration_feet_count"]), size // 2)
            key = (row["subject"], row["method"], row["calibration_run"])
            grouped.setdefault(key, {})[size] = set(
                filter(None, row["calibration_trial_keys"].split("/"))
            )
        for by_size in grouped.values():
            self.assertEqual(set(by_size), set(CALIBRATION_SIZES))
            previous: set[str] = set()
            for size in CALIBRATION_SIZES:
                self.assertTrue(previous.issubset(by_size[size]))
                previous = by_size[size]

    def test_participant_aggregation_and_zero_shot_reproduction_are_exact(self) -> None:
        rows = self.read_rows("participant_scores.csv")
        self.assertEqual(len(rows), 20 * 2 * 5 * 2)
        keys = Counter(
            (
                row["subject"],
                row["method"],
                row["calibration_size"],
                row["qc_sensitivity"],
            )
            for row in rows
        )
        self.assertEqual(set(keys.values()), {1})
        self.assertTrue(
            all(row["group_observational_unit"] == "participant" for row in rows)
        )
        historical = {
            int(row["subject"]): row["primary_mean_run_balanced_accuracy"]
            for row in _read_csv(
                self.ASSETS / "cross_subject_decoder_development_subject_scores.csv"
            )
            if row["model"] == "csp_4_empirical"
        }
        zero = {
            int(row["subject"]): row["primary_mean_test_run_balanced_accuracy"]
            for row in rows
            if row["method"] == "threshold"
            and row["calibration_size"] == "0"
            and row["qc_sensitivity"] == "False"
        }
        self.assertEqual(zero, historical)

    def test_curve_methods_selection_sources_and_artifact_hashes_are_exact(self) -> None:
        curve = self.read_rows("curve_summary.csv")
        self.assertEqual(len(curve), 10)
        self.assertEqual({row["method"] for row in curve}, set(CALIBRATION_METHODS))
        selection = self.read_rows("method_selection.csv")
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection[0]["selected_method"], "threshold")
        for path, expected in self.metadata["source_hashes"].items():
            self.assertEqual(file_sha256(Path(path)), expected)
        for record in self.config["historical_inputs"].values():
            self.assertEqual(
                file_sha256(PROJECT_ROOT / record["path"]), record["sha256"]
            )
        self.assertEqual(len(self.metadata["artifact_sha256"]), 7)
        for name, expected in self.metadata["artifact_sha256"].items():
            self.assertEqual(file_sha256(self.ASSETS / name), expected)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


if __name__ == "__main__":
    unittest.main()
