"""Tests for post-hoc decoder reliability and label-independent run shift."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from eeg_project.decoding import PROJECT_ROOT
from eeg_project.decoding_reliability import (
    EVALUATION_PREFIX,
    affine_invariant_spd_distance,
    load_reliability_config,
    participant_centered_correlation,
    reliability_category_rows,
    subject_reliability_rows,
    subject_run_shift_rows,
    within_subject_permutation_p,
)


class DecodingReliabilityTests(unittest.TestCase):
    ASSETS = PROJECT_ROOT / "docs" / "assets"

    @classmethod
    def read_rows(cls, suffix: str) -> list[dict[str, str]]:
        with (cls.ASSETS / f"{EVALUATION_PREFIX}{suffix}").open(
            newline="", encoding="utf-8"
        ) as input_file:
            return list(csv.DictReader(input_file))

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_reliability_config()
        cls.folds = cls.read_rows("fold_scores.csv")
        cls.subjects = cls.read_rows("subject_scores.csv")
        cls.predictions = cls.read_rows("predictions.csv")

    def test_diagnostic_freeze_preserves_completed_decoder(self) -> None:
        self.assertEqual(
            self.config["parent_completed_milestone_git_commit"], "4a39b4f"
        )
        self.assertTrue(self.config["historical_decoder_result_may_not_change"])
        self.assertFalse(self.config["new_run_shift_results_inspected_at_freeze"])
        boundaries = self.config["immutable_boundaries"]
        self.assertFalse(boundaries["classifier_refit_planned"])
        self.assertFalse(boundaries["cross_subject_decoding"])
        self.assertFalse(boundaries["new_classifier"])
        self.assertFalse(boundaries["hyperparameter_search"])
        self.assertFalse(boundaries["deep_learning"])
        for record in self.config["historical_inputs"].values():
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])

    def test_reliability_reproduces_all_historical_subject_and_fold_scores(self) -> None:
        rows = subject_reliability_rows(self.folds, self.subjects, self.predictions)
        self.assertEqual(len(rows), 86)
        historical = {
            int(row["subject"]): float(row["primary_mean_fold_balanced_accuracy"])
            for row in self.subjects
            if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"
        }
        self.assertEqual({int(row["subject"]) for row in rows}, set(historical))
        for row in rows:
            subject = int(row["subject"])
            self.assertEqual(row["historical_primary_mean_balanced_accuracy"], historical[subject])
            fold_values = np.array(
                [
                    float(row["balanced_accuracy_run_6"]),
                    float(row["balanced_accuracy_run_10"]),
                    float(row["balanced_accuracy_run_14"]),
                ]
            )
            self.assertAlmostEqual(float(np.mean(fold_values)), historical[subject])
            self.assertAlmostEqual(float(np.ptp(fold_values)), row["run_balanced_accuracy_range"])
        categories = reliability_category_rows(rows)
        self.assertEqual(sum(int(row["participant_count"]) for row in categories[:4]), 86)

    def test_primary_run_labels_trial_identities_and_confusions_remain_exact(self) -> None:
        primary_predictions = [
            row
            for row in self.predictions
            if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"
        ]
        self.assertEqual(len(primary_predictions), 3840)
        self.assertEqual(
            len({row["trial_key"] for row in primary_predictions}), 3840
        )
        self.assertTrue(
            all(row["run"] == row["test_run"] for row in primary_predictions)
        )
        primary_folds = [
            row
            for row in self.folds
            if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"
        ]
        self.assertEqual(sum(int(row["trial_count"]) for row in primary_folds), 3840)
        self.assertEqual(
            sum(
                int(row[field])
                for row in primary_folds
                for field in (
                    "true_fists_predicted_fists",
                    "true_fists_predicted_feet",
                    "true_feet_predicted_fists",
                    "true_feet_predicted_feet",
                )
            ),
            3840,
        )

    def test_affine_invariant_distance_is_symmetric_and_congruence_invariant(self) -> None:
        first = np.array([[2.0, 0.3], [0.3, 1.0]])
        second = np.array([[1.2, -0.1], [-0.1, 1.8]])
        transform = np.array([[1.0, 0.4], [-0.2, 1.3]])
        distance = affine_invariant_spd_distance(first, second)
        self.assertGreater(distance, 0.0)
        self.assertAlmostEqual(distance, affine_invariant_spd_distance(second, first))
        self.assertAlmostEqual(
            distance,
            affine_invariant_spd_distance(
                transform @ first @ transform.T,
                transform @ second @ transform.T,
            ),
        )
        self.assertAlmostEqual(affine_invariant_spd_distance(first, first), 0.0)

    def test_run_shift_api_cannot_receive_class_labels(self) -> None:
        parameters = inspect.signature(subject_run_shift_rows).parameters
        self.assertNotIn("labels", parameters)
        rng = np.random.default_rng(8)
        data = rng.normal(size=(18, 4, 80))
        runs = np.repeat(np.array([6, 10, 14]), 6)
        data[runs == 10, 0] *= 2.0
        rows = subject_run_shift_rows(1, data, runs, {"source": "hash"})
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["labels_used_for_distance"] is False for row in rows))
        self.assertTrue(
            all(np.isfinite(row["affine_invariant_covariance_distance"]) for row in rows)
        )

    def test_within_subject_permutation_is_deterministic(self) -> None:
        subjects = np.repeat(np.arange(6), 3)
        x = np.tile(np.array([0.0, 1.0, 2.0]), 6)
        y = -x + np.repeat(np.arange(6), 3)
        first = within_subject_permutation_p(x, y, subjects, resamples=500, seed=12)
        second = within_subject_permutation_p(x, y, subjects, resamples=500, seed=12)
        self.assertEqual(first, second)
        self.assertAlmostEqual(participant_centered_correlation(x, y, subjects), -1.0)


if __name__ == "__main__":
    unittest.main()
