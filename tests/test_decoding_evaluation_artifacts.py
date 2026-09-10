"""Regression checks for the completed held-out decoder evaluation."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import unittest

from eeg_project.decoding import PROJECT_ROOT
from eeg_project.decoding_evaluation import file_sha256, load_final_decoding_config
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY
from eeg_project.provenance import resolve_source_id


class DecodingEvaluationArtifactTests(unittest.TestCase):
    ASSETS = PROJECT_ROOT / "docs" / "assets"
    CHECKPOINTS = PROJECT_ROOT / "outputs" / "within_subject_decoder_checkpoints"
    PREFIX = "within_subject_decoder_evaluation_"

    @classmethod
    def setUpClass(cls) -> None:
        cls.final, _ = load_final_decoding_config()
        metadata_path = cls.ASSETS / f"{cls.PREFIX}metadata.json"
        if not metadata_path.exists():
            raise unittest.SkipTest("Final decoder evaluation artifacts are absent.")
        cls.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @classmethod
    def read_rows(cls, suffix: str) -> list[dict[str, str]]:
        with (cls.ASSETS / f"{cls.PREFIX}{suffix}").open(
            newline="", encoding="utf-8"
        ) as input_file:
            return list(csv.DictReader(input_file))

    def test_all_eligible_subjects_and_checkpoints_are_exact(self) -> None:
        subjects = self.final["cohorts"]["decoder_evaluation_eligible_subjects"]
        self.assertEqual(len(subjects), 86)
        self.assertEqual(self.metadata["technically_eligible_evaluation_subject_count"], 86)
        self.assertEqual(self.metadata["checkpoint_count"], 86)
        subject_rows = self.read_rows("subject_scores.csv")
        self.assertEqual(len(subject_rows), 86 * 2 * 2)
        self.assertEqual({int(row["subject"]) for row in subject_rows}, set(subjects))
        expected_config = self.metadata["final_config_sha256"]
        expected_core = self.metadata["frozen_core_sha256"]
        observed: list[int] = []
        for subject in subjects:
            path = self.CHECKPOINTS / f"subject_{subject:03d}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["subject"], subject)
            self.assertEqual(payload["final_config_sha256"], expected_config)
            self.assertEqual(payload["frozen_core_sha256"], expected_core)
            self.assertEqual(len(payload["source_hashes"]), 3)
            for source_name, source_hash in payload["source_hashes"].items():
                self.assertEqual(self.metadata["source_hashes"][source_name], source_hash)
            observed.append(subject)
        self.assertEqual(observed, subjects)

    def test_report_artifacts_match_recorded_hashes(self) -> None:
        self.assertEqual(len(self.metadata["artifact_sha256"]), 16)
        for name, expected in self.metadata["artifact_sha256"].items():
            self.assertEqual(file_sha256(self.ASSETS / name), expected)

    def test_every_primary_trial_has_one_out_of_run_prediction_per_model(self) -> None:
        rows = [
            row
            for row in self.read_rows("predictions.csv")
            if row["qc_sensitivity"] == "False"
        ]
        self.assertEqual(len(rows), 7680)
        self.assertEqual(
            {row["model"] for row in rows},
            {"spectral_baseline_6", "csp_4_empirical"},
        )
        counts = Counter((row["model"], row["trial_key"]) for row in rows)
        self.assertEqual(len(counts), 7680)
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(len({row["trial_key"] for row in rows}), 3840)
        for row in rows:
            self.assertEqual(row["run"], row["test_run"])
            self.assertNotIn(row["test_run"], row["training_runs"].split("/"))

    def test_fit_audit_proves_no_csp_scaler_or_lda_leakage(self) -> None:
        rows = self.read_rows("fit_audit.csv")
        self.assertEqual(len(rows), 1032)
        for row in rows:
            train = set(row["train_trial_keys"].split("/"))
            test = set(row["test_trial_keys"].split("/"))
            scaler = set(row["scaler_fit_trial_keys"].split("/"))
            lda = set(row["lda_fit_trial_keys"].split("/"))
            self.assertTrue(train.isdisjoint(test))
            self.assertEqual(scaler, train)
            self.assertEqual(lda, train)
            if row["model"] == "csp_4_empirical":
                self.assertEqual(set(row["csp_fit_trial_keys"].split("/")), train)
            else:
                self.assertEqual(row["csp_fit_trial_keys"], "not_applicable")
            self.assertEqual(row["test_run_absent_from_all_fit_stages"], "True")
            self.assertEqual(row["subject_isolation_preserved"], "True")

    def test_source_hashes_and_historical_eeg_artifacts_remain_unchanged(self) -> None:
        self.assertEqual(len(self.metadata["source_hashes"]), 258)
        for source_name, expected in self.metadata["source_hashes"].items():
            self.assertEqual(file_sha256(resolve_source_id(source_name, DEFAULT_DATA_DIRECTORY)), expected)
        imf_metadata = json.loads(
            (self.ASSETS / "individual_mu_heldout_evaluation_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        historical = {
            "subjects01_20": self.ASSETS
            / "subjects01-20_replication_trial_spectral_measurements.csv",
            "subjects21_109": self.ASSETS
            / "subjects01-109_full_replication_replication2_trial_spectral_measurements.csv",
        }
        for name, path in historical.items():
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                imf_metadata["fixed_trial_artifact_sha256"][name],
            )

    def test_missing_historical_peaks_remain_legitimate_missing_values(self) -> None:
        rows = self.read_rows("exploratory_subject_context.csv")
        self.assertEqual(len(rows), 86)
        unavailable = [
            row
            for row in rows
            if row["historical_central_peak_available"] == "False"
        ]
        self.assertEqual(len(unavailable), 17)
        self.assertTrue(
            all(row["historical_central_peak_frequency_hz"] == "" for row in unavailable)
        )
        available = [
            row
            for row in rows
            if row["historical_central_peak_available"] == "True"
        ]
        self.assertEqual(len(available), 69)
        self.assertTrue(
            all(row["historical_central_peak_frequency_hz"] != "" for row in available)
        )
        imf_missing = [
            row
            for row in rows
            if row["imf_study_subject"] == "True"
            and row["imf_method_available"] == "False"
        ]
        self.assertEqual(len(imf_missing), 2)
        self.assertTrue(
            all(row["imf_peak_frequency_median_hz"] == "" for row in imf_missing)
        )

    def test_confusions_and_frozen_decision_match_predictions(self) -> None:
        confusion = self.read_rows("confusion_matrices.csv")
        self.assertEqual(
            sum(
                int(row["trial_count"])
                for row in confusion
                if row["qc_sensitivity"] == "False"
            ),
            7680,
        )
        self.assertEqual(self.metadata["decoding_category"], "useful decoding signal")
        self.assertEqual(
            self.metadata["csp_added_value_category"], "clear added value"
        )
        self.assertTrue(self.metadata["every_trial_predicted_once_out_of_run"])
        self.assertTrue(self.metadata["all_fit_audits_exclude_test_run"])
        self.assertTrue(self.metadata["all_fit_audits_preserve_subject_isolation"])


if __name__ == "__main__":
    unittest.main()
