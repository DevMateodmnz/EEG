"""Integrity checks for the completed frozen unilateral generalization study."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

from eeg_project.unilateral_generalization import load_study_config
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY
from eeg_project.provenance import resolve_source_id


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
PREFIX = "unilateral_motor_imagery_generalization_"


class UnilateralGeneralizationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_study_config()
        cls.summary = json.loads((ASSETS / f"{PREFIX}summary.json").read_text())
        cls.provenance = json.loads((ASSETS / f"{PREFIX}provenance.json").read_text())

    @staticmethod
    def rows(suffix: str) -> list[dict[str, str]]:
        with (ASSETS / f"{PREFIX}{suffix}.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_matched_cohort_and_verified_source_manifest_are_complete(self) -> None:
        final = json.loads((ROOT / self.config["parent_bilateral_decoder"]["config"]).read_text())
        subjects = final["cohorts"]["decoder_evaluation_eligible_subjects"]
        self.assertEqual(len(subjects), 86)
        self.assertEqual(self.summary["unilateral_eligible_subject_count"], 86)
        self.assertEqual(self.summary["matched_bilateral_subject_count"], 86)
        self.assertEqual(len(self.provenance["raw_source_hashes"]), 258)
        self.assertEqual(self.provenance["config_sha256"], hashlib.sha256((ROOT / "config/unilateral_motor_imagery_generalization.json").read_bytes()).hexdigest())
        for name, digest in self.provenance["raw_source_hashes"].items():
            self.assertEqual(hashlib.sha256(resolve_source_id(name, DEFAULT_DATA_DIRECTORY).read_bytes()).hexdigest(), digest)

    def test_predictions_are_unique_contextual_and_out_of_run(self) -> None:
        rows = [row for row in self.rows("predictions") if row["qc_sensitivity"] == "False"]
        self.assertEqual(len(rows), 7674)
        self.assertEqual({row["run"] for row in rows}, {"4", "8", "12"})
        self.assertEqual({row["true_condition"] for row in rows}, {"left_fist_imagery", "right_fist_imagery"})
        counts = Counter((row["model"], row["trial_key"]) for row in rows)
        self.assertEqual(len(counts), len(rows))
        self.assertEqual(set(counts.values()), {1})
        for row in rows:
            self.assertEqual(row["run"], row["test_run"])
            self.assertNotIn(row["test_run"], row["training_runs"].split("/"))

    def test_fit_audits_and_metrics_prove_no_leakage(self) -> None:
        folds = self.rows("fold_scores")
        self.assertEqual(len(folds), 86 * 3 * 2 * 2)
        self.assertTrue(all(0 <= float(row["balanced_accuracy"]) <= 1 for row in folds))
        self.assertTrue(all(int(row["trial_count"]) == sum(int(row[key]) for key in ("true_left_predicted_left", "true_left_predicted_right", "true_right_predicted_left", "true_right_predicted_right")) for row in folds))
        for row in self.rows("fit_audit"):
            train, test = set(row["train_trial_keys"].split("/")), set(row["test_trial_keys"].split("/"))
            self.assertTrue(train.isdisjoint(test))
            self.assertEqual(set(row["scaler_fit_trial_keys"].split("/")), train)
            self.assertEqual(set(row["lda_fit_trial_keys"].split("/")), train)
            if row["model"] == "csp_4_empirical":
                self.assertEqual(set(row["csp_fit_trial_keys"].split("/")), train)
            self.assertEqual(row["test_run_absent_from_all_fit_stages"], "True")

    def test_parent_artifacts_and_staging_boundary_are_preserved(self) -> None:
        parent = self.config["parent_bilateral_decoder"]
        self.assertEqual(hashlib.sha256((ROOT / parent["config"]).read_bytes()).hexdigest(), parent["sha256"])
        self.assertEqual(hashlib.sha256((ROOT / parent["metadata"]).read_bytes()).hexdigest(), parent["metadata_sha256"])
        staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
        self.assertNotIn("AGENTS.md", staged)


if __name__ == "__main__":
    unittest.main()
