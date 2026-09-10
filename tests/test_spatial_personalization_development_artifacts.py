"""Regression guards for spatial-personalization development evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import unittest

from eeg_project.cross_subject_decoding import file_sha256
from eeg_project.decoding import PROJECT_ROOT
from eeg_project.spatial_personalization import (
    A_METHOD,
    B_METHODS,
    C_METHODS,
    DEVELOPMENT_METHODS,
    load_spatial_personalization_config,
)


class SpatialPersonalizationDevelopmentArtifactTests(unittest.TestCase):
    ASSETS = PROJECT_ROOT / "docs" / "assets"
    PREFIX = "spatial_personalization_development_"

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_spatial_personalization_config()
        cls.metadata = json.loads(
            (cls.ASSETS / f"{cls.PREFIX}metadata.json").read_text(encoding="utf-8")
        )

    @classmethod
    def read_rows(cls, suffix: str) -> list[dict[str, str]]:
        with (cls.ASSETS / f"{cls.PREFIX}{suffix}").open(
            newline="", encoding="utf-8"
        ) as input_file:
            return list(csv.DictReader(input_file))

    def test_development_cohort_is_exact_and_evaluation_remains_locked(self) -> None:
        self.assertEqual(self.metadata["development_subjects"], list(range(1, 21)))
        self.assertEqual(self.metadata["development_subject_count"], 20)
        self.assertFalse(self.metadata["evaluation_target_CSP_fitted"])
        self.assertFalse(self.metadata["evaluation_target_CSP_outcomes_calculated"])
        self.assertTrue(self.metadata["same_test_trials_compared_across_methods"])
        self.assertFalse(self.metadata["test_EEG_used_for_any_fit"])

    def test_every_method_uses_identical_test_contexts(self) -> None:
        rows = self.read_rows("predictions.csv")
        self.assertEqual(len(rows), 20 * len(DEVELOPMENT_METHODS) * 3 * 30)
        counts = Counter(
            (row["subject"], row["method"], row["calibration_run"], row["trial_key"])
            for row in rows
        )
        self.assertEqual(set(counts.values()), {1})
        contexts: dict[tuple[int, str], set[tuple[int, str]]] = defaultdict(set)
        for row in rows:
            contexts[(int(row["subject"]), row["method"])].add(
                (int(row["calibration_run"]), row["trial_key"])
            )
            self.assertNotEqual(int(row["calibration_run"]), int(row["test_run"]))
            self.assertEqual(row["test_run_used_for_fitting"], "False")
        for subject in range(1, 21):
            reference = contexts[(subject, A_METHOD)]
            self.assertTrue(
                all(contexts[(subject, method)] == reference for method in DEVELOPMENT_METHODS)
            )

    def test_fit_audit_proves_exact_calibration_only_learning(self) -> None:
        rows = self.read_rows("fit_audit.csv")
        self.assertEqual(len(rows), 20 * len(DEVELOPMENT_METHODS) * 3)
        for row in rows:
            calibration = set(row["calibration_trial_keys"].split("/"))
            test = set(row["test_trial_keys"].split("/"))
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
            method = row["method"]
            if method in C_METHODS:
                self.assertEqual(set(row["target_CSP_fit_trial_keys"].split("/")), calibration)
            else:
                self.assertEqual(row["target_CSP_fit_trial_keys"], "not_applicable")
            if method == A_METHOD:
                self.assertEqual(row["target_LDA_fit_trial_keys"], "not_applicable")
            else:
                self.assertEqual(set(row["target_LDA_fit_trial_keys"].split("/")), calibration)

    def test_participant_aggregation_and_zero_shot_reference_are_exact(self) -> None:
        rows = self.read_rows("participant_scores.csv")
        self.assertEqual(len(rows), 20 * len(DEVELOPMENT_METHODS) * 2)
        keys = Counter((row["subject"], row["method"], row["qc_sensitivity"]) for row in rows)
        self.assertEqual(set(keys.values()), {1})
        historical = {
            int(row["subject"]): float(row["primary_mean_run_balanced_accuracy"])
            for row in self._read_external("cross_subject_decoder_development_subject_scores.csv")
            if row["model"] == "csp_4_empirical"
        }
        observed = {
            int(row["subject"]): float(row["primary_mean_scenario_test_run_balanced_accuracy"])
            for row in rows
            if row["method"] == A_METHOD and row["qc_sensitivity"] == "False"
        }
        self.assertEqual(observed, historical)

    def _read_external(self, name: str) -> list[dict[str, str]]:
        with (self.ASSETS / name).open(newline="", encoding="utf-8") as input_file:
            return list(csv.DictReader(input_file))

    def test_frozen_selection_and_subspace_stability_are_complete(self) -> None:
        selection = self.read_rows("method_selection.csv")
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection[0]["selected_B_method"], B_METHODS[0])
        self.assertEqual(selection[0]["selected_C_method"], C_METHODS[1])
        self.assertEqual(selection[0]["B_tie_count"], "20")
        stability = self.read_rows("csp_stability.csv")
        self.assertEqual(len(stability), 40)
        self.assertEqual(
            Counter(row["method"] for row in stability),
            Counter({C_METHODS[0]: 20, C_METHODS[1]: 20}),
        )
        self.assertTrue(all(row["sign_order_and_rotation_invariant"] == "True" for row in stability))
        self.assertTrue(
            all(0.0 <= float(row["mean_pairwise_subspace_similarity"]) <= 1.0 for row in stability)
        )

    def test_sources_and_artifact_hashes_are_exact(self) -> None:
        self.assertEqual(len(self.metadata["source_hashes"]), 60)
        for path, expected in self.metadata["source_hashes"].items():
            self.assertEqual(file_sha256(Path(path)), expected)
        self.assertEqual(len(self.metadata["artifact_sha256"]), 8)
        for name, expected in self.metadata["artifact_sha256"].items():
            self.assertEqual(file_sha256(self.ASSETS / name), expected)


if __name__ == "__main__":
    unittest.main()
