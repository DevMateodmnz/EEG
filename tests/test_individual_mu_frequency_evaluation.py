"""Tests for held-out individualized-frequency comparison operations."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from eeg_project.individual_frequency_evaluation import (
    classify_method_comparison,
    compute_custom_morlet_power,
    paired_group_summaries,
    peak_reliability_category,
)
from eeg_project.time_frequency import load_event_related_spectral_config
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY
from eeg_project.provenance import resolve_source_id


class IndividualFrequencyEvaluationTests(unittest.TestCase):
    ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"

    def test_peak_reliability_boundaries_are_frozen(self) -> None:
        self.assertEqual(peak_reliability_category(0.5), "stable")
        self.assertEqual(peak_reliability_category(0.75), "moderate")
        self.assertEqual(peak_reliability_category(1.0), "moderate")
        self.assertEqual(peak_reliability_category(1.25), "unstable")

    def test_custom_morlet_uses_exact_five_center_grid(self) -> None:
        config = load_event_related_spectral_config()
        sampling_frequency = 160.0
        times = np.arange(961) / sampling_frequency - 2.0
        data = np.zeros((2, 2, times.size))
        frequencies = np.array([9.0, 9.5, 10.0, 10.5, 11.0])
        power = compute_custom_morlet_power(
            data, times, sampling_frequency, frequencies, config
        )
        self.assertEqual(power.power_v2.shape, (2, 2, 5, 241))
        np.testing.assert_array_equal(power.frequencies_hz, frequencies)

    def test_pre_registered_method_categories(self) -> None:
        self.assertEqual(
            classify_method_comparison(0.9, 0.06, 0.07, -2, -1, -0.1, -0.5),
            "clear improvement",
        )
        self.assertEqual(
            classify_method_comparison(0.7, 0.00, -0.04, -2, -1, -0.1, 1),
            "mixed result",
        )
        self.assertEqual(
            classify_method_comparison(0.9, -0.10, -0.10, 2, 1, 0.1, 2),
            "no improvement",
        )

    def test_group_comparison_uses_one_paired_row_per_subject(self) -> None:
        rows = []
        for channel in ("C4", "C3"):
            for subject, fixed, individualized in (
                (3, -10.0, -20.0), (5, 5.0, -5.0), (7, -4.0, -6.0)
            ):
                rows.append(
                    {
                        "subject": subject,
                        "channel": channel,
                        "fixed_narrow_median_percent": fixed,
                        "individualized_median_percent": individualized,
                        "fixed_broad_median_percent": fixed - 1,
                        "individualized_minus_fixed_narrow_median_db": -0.2,
                        "qc_paired_difference_percent": -1.0,
                        "fixed_narrow_negative_run_count": 2,
                        "individualized_negative_run_count": 3,
                        "fixed_narrow_run_range_percentage_points": 4.0,
                        "individualized_run_range_percentage_points": 3.0,
                    }
                )
        summaries = paired_group_summaries(rows, eligible_evaluation_count=4)
        self.assertEqual({row["channel"] for row in summaries}, {"C4", "C3"})
        self.assertTrue(all(row["matched_individualizable_subject_count"] == 3 for row in summaries))
        self.assertTrue(all(row["coverage_fraction"] == 0.75 for row in summaries))
        self.assertTrue(all(row["median_paired_individualized_minus_fixed_narrow_percent"] < 0 for row in summaries))

    def test_held_out_artifacts_preserve_leakage_barriers(self) -> None:
        peak_path = self.ASSETS / "individual_mu_heldout_evaluation_peak_estimates.csv"
        if not peak_path.exists():
            self.skipTest("Held-out artifacts have not been generated yet.")
        with peak_path.open(newline="", encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))
        loro = [row for row in rows if row["estimate_type"] == "leave_one_run_out"]
        self.assertEqual(len(loro), 54 * 3)
        for row in loro:
            held_out = int(row["held_out_run"])
            training = {int(value) for value in row["training_runs"].split("/")}
            self.assertNotIn(held_out, training)
            self.assertEqual(training, {6, 10, 14} - {held_out})
            self.assertEqual(row["frequency_estimation_data"], "T0_rest_only")
            self.assertEqual(row["task_outcome_used_for_peak"], "False")

    def test_held_out_comparison_is_exactly_matched_and_subject_level(self) -> None:
        trial_path = self.ASSETS / "individual_mu_heldout_evaluation_trial_comparison.csv"
        group_path = self.ASSETS / "individual_mu_heldout_evaluation_group_comparison.csv"
        if not trial_path.exists() or not group_path.exists():
            self.skipTest("Held-out artifacts have not been generated yet.")
        with trial_path.open(newline="", encoding="utf-8") as input_file:
            trials = list(csv.DictReader(input_file))
        identities = [
            (row["subject"], row["run"], row["run_trial_index"], row["channel"])
            for row in trials
        ]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertTrue(all(row["matched_trial_identity"] == "True" for row in trials))
        self.assertTrue(all(row["semantic_condition"] == "both_fists_imagery" for row in trials))
        self.assertTrue(all(row["channel"] in {"C4", "C3"} for row in trials))
        with group_path.open(newline="", encoding="utf-8") as input_file:
            groups = list(csv.DictReader(input_file))
        self.assertEqual({row["channel"] for row in groups}, {"C4", "C3"})
        self.assertTrue(all(row["matched_individualizable_subject_count"] == "50" for row in groups))
        self.assertTrue(all(row["eligible_evaluation_subject_count"] == "54" for row in groups))

    def test_historical_fixed_artifacts_and_raw_sources_have_not_changed(self) -> None:
        metadata_path = self.ASSETS / "individual_mu_heldout_evaluation_metadata.json"
        trial_path = self.ASSETS / "individual_mu_heldout_evaluation_trial_comparison.csv"
        if not metadata_path.exists() or not trial_path.exists():
            self.skipTest("Held-out artifacts have not been generated yet.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        historical = {
            "subjects01_20": self.ASSETS / "subjects01-20_replication_trial_spectral_measurements.csv",
            "subjects21_109": self.ASSETS / "subjects01-109_full_replication_replication2_trial_spectral_measurements.csv",
        }
        for name, path in historical.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, metadata["fixed_trial_artifact_sha256"][name])
        with trial_path.open(newline="", encoding="utf-8") as input_file:
            trials = list(csv.DictReader(input_file))
        source_hashes = {(row["source_file"], row["source_sha256"]) for row in trials}
        for source_name, expected_hash in source_hashes:
            source = resolve_source_id(source_name, DEFAULT_DATA_DIRECTORY)
            self.assertTrue(source.is_file())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected_hash)

    def test_metadata_records_scope_and_negative_decision(self) -> None:
        metadata_path = self.ASSETS / "individual_mu_heldout_evaluation_metadata.json"
        if not metadata_path.exists():
            self.skipTest("Held-out artifacts have not been generated yet.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertFalse(metadata["outcome_used_for_peak_estimation"])
        self.assertFalse(metadata["held_out_run_used_for_its_peak_estimation"])
        self.assertFalse(metadata["full_tfr_arrays_persisted"])
        self.assertFalse(metadata["csp_performed"])
        self.assertFalse(metadata["classification_performed"])
        self.assertEqual(metadata["overall_method_category"], "no improvement")


if __name__ == "__main__":
    unittest.main()
