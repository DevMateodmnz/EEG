"""Regression guards for completed spatial-personalization evaluation artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import unittest

import numpy as np

from eeg_project.cross_subject_decoding import file_sha256
from eeg_project.decoding import PROJECT_ROOT
from eeg_project.spatial_personalization import A_METHOD
from eeg_project.spatial_personalization_evaluation import (
    load_final_spatial_personalization_config,
)


class SpatialPersonalizationEvaluationArtifactTests(unittest.TestCase):
    ASSETS = PROJECT_ROOT / "docs" / "assets"
    RAW_CHECKPOINTS = PROJECT_ROOT / "outputs" / "spatial_personalization_checkpoints"
    SOURCE_CHECKPOINTS = PROJECT_ROOT / "outputs" / "target_calibration_checkpoints"
    PREFIX = "spatial_personalization_evaluation_"
    METHODS = (
        A_METHOD,
        "source_csp_target_lda_source_scaler",
        "target_csp_ledoit_wolf",
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.final, cls.initial = load_final_spatial_personalization_config()
        cls.metadata = json.loads(
            (cls.ASSETS / f"{cls.PREFIX}metadata.json").read_text(encoding="utf-8")
        )

    @classmethod
    def read_rows(cls, suffix: str) -> list[dict[str, str]]:
        with (cls.ASSETS / f"{cls.PREFIX}{suffix}").open(
            newline="", encoding="utf-8"
        ) as input_file:
            return list(csv.DictReader(input_file))

    def test_all_86_raw_and_source_feature_checkpoints_are_hash_validated(self) -> None:
        subjects = self.final["cohorts"]["evaluation_eligible_subjects"]
        self.assertEqual(len(subjects), 86)
        self.assertEqual(self.metadata["evaluation_subject_count"], 86)
        self.assertEqual(self.metadata["raw_checkpoint_count"], 86)
        self.assertEqual(self.metadata["raw_checkpoint_reuse_count_this_run"], 86)
        self.assertEqual(self.metadata["source_feature_checkpoint_count"], 86)
        self.assertEqual(
            len(list(self.RAW_CHECKPOINTS.glob("subject_*.json"))), 86
        )
        self.assertEqual(
            len(list(self.RAW_CHECKPOINTS.glob("subject_*_csp.npy"))), 86
        )
        for subject in subjects:
            metadata_path = self.RAW_CHECKPOINTS / f"subject_{subject:03d}.json"
            array_path = self.RAW_CHECKPOINTS / f"subject_{subject:03d}_csp.npy"
            source_path = self.SOURCE_CHECKPOINTS / f"subject_{subject:03d}.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["subject"], subject)
            self.assertEqual(payload["final_config_sha256"], self.metadata["final_config_sha256"])
            self.assertEqual(payload["frozen_core_sha256"], self.metadata["frozen_core_sha256"])
            self.assertEqual(payload["source_feature_checkpoint_sha256"], file_sha256(source_path))
            self.assertEqual(payload["array_sha256"], file_sha256(array_path))
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            self.assertIn(list(array.shape), ([42, 64, 320], [45, 64, 320]))
            self.assertEqual(str(array.dtype), "float64")
            self.assertFalse(payload["test_EEG_used_for_fitting_at_checkpoint_creation"])
            self.assertEqual(len(payload["source_hashes"]), 3)
            for path, expected in payload["source_hashes"].items():
                self.assertEqual(file_sha256(Path(path)), expected)

    def test_every_trial_has_two_identical_test_contexts_per_method(self) -> None:
        rows = self.read_rows("predictions.csv")
        self.assertEqual(len(rows), 3840 * len(self.METHODS) * 2)
        counts = Counter(
            (int(row["subject"]), row["method"], row["trial_key"]) for row in rows
        )
        self.assertEqual(len(counts), 3840 * len(self.METHODS))
        self.assertEqual(set(counts.values()), {2})
        contexts: dict[tuple[int, str], set[tuple[int, str]]] = defaultdict(set)
        for row in rows:
            contexts[(int(row["subject"]), row["method"])].add(
                (int(row["calibration_run"]), row["trial_key"])
            )
            self.assertNotEqual(int(row["calibration_run"]), int(row["test_run"]))
            self.assertEqual(row["test_run_used_for_fitting"], "False")
        for subject in self.final["cohorts"]["evaluation_eligible_subjects"]:
            reference = contexts[(subject, A_METHOD)]
            self.assertTrue(all(contexts[(subject, method)] == reference for method in self.METHODS))

    def test_fit_audit_proves_exact_A_B_C_learning_boundaries(self) -> None:
        rows = self.read_rows("fit_audit.csv")
        self.assertEqual(len(rows), 86 * len(self.METHODS) * 3)
        geometry: dict[tuple[int, int], dict[str, tuple[set[str], set[str]]]] = defaultdict(dict)
        for row in rows:
            subject = int(row["subject"])
            calibration_run = int(row["calibration_run"])
            method = row["method"]
            calibration = set(row["calibration_trial_keys"].split("/"))
            test = set(row["test_trial_keys"].split("/"))
            geometry[(subject, calibration_run)][method] = (calibration, test)
            self.assertEqual(len(calibration), 14)
            self.assertEqual(int(row["calibration_fists_count"]), 7)
            self.assertEqual(int(row["calibration_feet_count"]), 7)
            self.assertTrue(calibration.isdisjoint(test))
            for field in (
                "source_CSP_fit_target_EEG",
                "test_EEG_used_for_CSP_fit",
                "test_EEG_used_for_scaler_fit",
                "test_EEG_used_for_LDA_fit",
                "test_labels_used_for_fitting",
                "unlabeled_test_EEG_used_for_adaptation",
                "test_covariance_alignment",
            ):
                self.assertEqual(row[field], "False")
            if method == A_METHOD:
                for field in (
                    "target_CSP_fit_trial_keys",
                    "target_scaler_fit_trial_keys",
                    "target_LDA_fit_trial_keys",
                ):
                    self.assertEqual(row[field], "not_applicable")
            elif method == self.METHODS[1]:
                self.assertEqual(row["target_CSP_fit_trial_keys"], "not_applicable")
                self.assertEqual(row["target_scaler_fit_trial_keys"], "not_applicable")
                self.assertEqual(set(row["target_LDA_fit_trial_keys"].split("/")), calibration)
            else:
                for field in (
                    "target_CSP_fit_trial_keys",
                    "target_scaler_fit_trial_keys",
                    "target_LDA_fit_trial_keys",
                ):
                    self.assertEqual(set(row[field].split("/")), calibration)
        for methods in geometry.values():
            self.assertEqual(set(methods), set(self.METHODS))
            self.assertEqual(len({frozenset(value[0]) for value in methods.values()}), 1)
            self.assertEqual(len({frozenset(value[1]) for value in methods.values()}), 1)

    def test_participant_primary_results_and_historical_A_are_exact(self) -> None:
        rows = self.read_rows("participant_scores.csv")
        self.assertEqual(len(rows), 86 * len(self.METHODS) * 2)
        keys = Counter((row["subject"], row["method"], row["qc_sensitivity"]) for row in rows)
        self.assertEqual(set(keys.values()), {1})
        historical = {}
        with (
            self.ASSETS / "cross_subject_decoder_evaluation_subject_scores.csv"
        ).open(newline="", encoding="utf-8") as input_file:
            for row in csv.DictReader(input_file):
                if row["model"] == "csp_4_empirical":
                    historical[int(row["subject"])] = float(
                        row["primary_mean_run_balanced_accuracy"]
                    )
        observed = {
            int(row["subject"]): float(row["primary_mean_scenario_test_run_balanced_accuracy"])
            for row in rows
            if row["method"] == A_METHOD and row["qc_sensitivity"] == "False"
        }
        self.assertEqual(observed, historical)

        groups = self.read_rows("group_summary.csv")
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(row["participant_count"] == "86" for row in groups))
        paired = self.read_rows("paired_summary.csv")
        self.assertEqual([row["comparison_name"] for row in paired], ["B_minus_A", "C_minus_B", "C_minus_A"])
        self.assertTrue(all(row["convincing_paired_gain"] == "False" for row in paired))
        self.assertEqual(self.metadata["interpretation_category"], "one-run personalization is insufficient")

    def test_target_csp_stability_and_qc_are_complete(self) -> None:
        stability = self.read_rows("csp_stability.csv")
        self.assertEqual(len(stability), 86)
        self.assertTrue(all(row["sign_order_and_rotation_invariant"] == "True" for row in stability))
        self.assertTrue(
            all(0.0 <= float(row["mean_pairwise_subspace_similarity"]) < 0.50 for row in stability)
        )
        summary = self.read_rows("csp_stability_summary.csv")
        self.assertEqual(summary[0]["low_below_0_50_count"], "86")
        groups = self.read_rows("group_summary.csv")
        self.assertTrue(
            all(
                abs(float(row["median_paired_QC_minus_primary_balanced_accuracy"])) < 0.002
                for row in groups
            )
        )

    def test_historical_inputs_sources_and_report_hashes_are_unchanged(self) -> None:
        for record in self.initial["historical_inputs"].values():
            self.assertEqual(file_sha256(PROJECT_ROOT / record["path"]), record["sha256"])
        self.assertEqual(len(self.metadata["training_source_hashes"]), 60)
        self.assertEqual(len(self.metadata["evaluation_source_hashes"]), 258)
        for path, expected in self.metadata["training_source_hashes"].items():
            self.assertEqual(file_sha256(Path(path)), expected)
        self.assertFalse(self.metadata["historical_zero_shot_outputs_modified"])
        self.assertFalse(self.metadata["historical_calibration_outputs_modified"])
        self.assertFalse(self.metadata["historical_within_subject_outputs_modified"])
        self.assertEqual(len(self.metadata["artifact_sha256"]), 16)
        for name, expected in self.metadata["artifact_sha256"].items():
            self.assertEqual(file_sha256(self.ASSETS / name), expected)

    def test_metadata_records_no_source_refit_or_test_adaptation(self) -> None:
        self.assertFalse(self.metadata["source_model_refitted"])
        self.assertFalse(self.metadata["source_CSP_or_scaler_refitted"])
        self.assertEqual(self.metadata["target_CSP_fit_count"], 258)
        self.assertTrue(self.metadata["target_CSP_fit_calibration_only"])
        self.assertFalse(self.metadata["test_EEG_used_for_any_fit"])
        self.assertFalse(self.metadata["unlabeled_test_EEG_used_for_adaptation"])
        self.assertTrue(self.metadata["same_test_trials_compared_across_A_B_C"])
        self.assertTrue(self.metadata["one_value_per_participant_per_method"])

    def test_updated_documentation_local_links_resolve(self) -> None:
        documents = (
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "spatial_personalization.md",
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
