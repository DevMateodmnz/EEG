"""Pre-outcome partition and guard tests for the individualized-frequency study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from eeg_project.individual_frequency import (
    estimate_central_peak,
    individualized_frequency_vector,
    load_individual_frequency_config,
    rest_roi_log_welch,
    training_runs_for_held_out,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/individual_mu_frequency.json"


class IndividualMuStudyFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_initial_freeze_starts_from_completed_full_cohort(self) -> None:
        self.assertEqual(
            self.payload["study_start_git_commit"],
            "d322c58fc6474b7970eb113fe0448c4c5177b157",
        )
        self.assertFalse(
            self.payload["individualized_task_outcomes_inspected_at_initial_freeze"]
        )
        self.assertEqual(self.payload["study_stage"], "final_method_frozen")
        self.assertFalse(
            self.payload[
                "held_out_evaluation_individualized_task_outcomes_inspected_at_final_freeze"
            ]
        )

    def test_partition_is_exact_disjoint_and_outcome_independent(self) -> None:
        partition = self.payload["partition"]
        eligible = set(range(2, 110)) - {88, 92, 100}
        development = set(partition["development_subjects"])
        evaluation = set(partition["evaluation_subjects"])
        self.assertEqual(development | evaluation, eligible)
        self.assertFalse(development & evaluation)
        self.assertEqual(development, {subject for subject in eligible if subject % 2 == 0})
        self.assertEqual(evaluation, {subject for subject in eligible if subject % 2 == 1})
        self.assertEqual(len(development), 51)
        self.assertEqual(len(evaluation), 54)
        self.assertTrue(partition["outcome_independent"])
        self.assertFalse(
            partition["evaluation_individualized_task_outcomes_locked_until_final_method_freeze"]
        )

    def test_historical_configuration_hashes_are_frozen(self) -> None:
        for record in self.payload["frozen_historical_inputs"].values():
            path = ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])

    def test_peak_estimation_and_evaluation_guards(self) -> None:
        peak = self.payload["candidate_peak_method"]
        self.assertEqual(peak["frequency_estimation_data"], "T0_rest_only")
        self.assertTrue(peak["held_out_run_samples_forbidden_from_its_peak_estimate"])
        self.assertFalse(peak["task_labels_or_task_outcomes_may_define_peak"])
        self.assertFalse(peak["boundary_maximum_is_valid"])
        self.assertIn("fists_ERD_magnitude", peak["development_selection_may_not_use"])
        self.assertTrue(
            self.payload["candidate_no_peak_policy"]
            ["all_three_leave_one_run_out_estimates_required_for_primary_matched_comparison"]
        )

    def test_final_method_and_success_rules_are_frozen_before_evaluation(self) -> None:
        method = self.payload["final_method"]
        self.assertEqual(method["frequency_estimation_data"], "T0_rest_only_from_training_runs")
        self.assertEqual(method["central_roi"], ["C3", "Cz", "C4"])
        self.assertEqual(method["search_range_hz"], [7.0, 14.0])
        self.assertEqual(method["minimum_prominence_db"], 1.0)
        self.assertEqual(method["individualized_band_half_width_hz"], 1.0)
        self.assertEqual(method["individualized_morlet_frequency_step_hz"], 0.5)
        self.assertTrue(method["leave_one_run_out"])
        self.assertIn("clear_improvement", self.payload["primary_method_success_criteria_per_channel"])
        self.assertEqual(
            self.payload["paired_inference"]["multiple_comparison_adjustment"],
            "Holm_for_C4_and_C3",
        )

    def test_final_method_implementation_and_development_hashes_are_exact(self) -> None:
        records = [
            *self.payload["method_implementation"].values(),
            self.payload["method_development_evidence"]["candidate_table"],
            self.payload["method_development_evidence"]["candidate_summary"],
        ]
        for record in records:
            path = ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])

    def test_fixed_baselines_and_deferred_models_remain_unchanged(self) -> None:
        fixed = self.payload["validated_comparators"]
        self.assertEqual(fixed["primary_fixed_narrow"]["frequency_centers_hz"], [12.0, 13.0])
        self.assertFalse(fixed["may_be_redefined"])
        self.assertFalse(self.payload["classification_performed"])
        self.assertFalse(self.payload["csp_performed"])
        self.assertFalse(self.payload["persist_full_tfr_arrays"])


class IndividualFrequencyEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_individual_frequency_config()

    def test_known_ten_hz_rest_signal_recovers_ten_hz(self) -> None:
        sampling_frequency = 160.0
        times = np.arange(640) / sampling_frequency
        rng = np.random.default_rng(20260829)
        data = np.stack(
            [
                np.stack(
                    [
                        8e-6 * np.sin(2 * np.pi * 10.0 * times)
                        + rng.normal(0.0, 0.3e-6, times.size)
                        for _ in range(3)
                    ]
                )
                for _ in range(12)
            ]
        )
        frequencies, power = rest_roi_log_welch(
            data, ["C3", "Cz", "C4"], sampling_frequency,
            self.config.central_channels, 4.0
        )
        estimate = estimate_central_peak(frequencies, power, (7.0, 14.0), 1.0)
        self.assertTrue(estimate.method_available)
        self.assertEqual(estimate.peak_quality, "well_defined_interior")
        self.assertAlmostEqual(estimate.peak_frequency_hz, 10.0)

    def test_search_boundary_peak_is_not_valid(self) -> None:
        frequencies = np.arange(6.0, 15.25, 0.25)
        power = -20.0 - 3.0 * np.abs(frequencies - 7.0)
        estimate = estimate_central_peak(frequencies, power, (7.0, 14.0), 1.0)
        self.assertFalse(estimate.method_available)
        self.assertEqual(estimate.peak_quality, "boundary_candidate")
        self.assertEqual(estimate.candidate_frequency_hz, 7.0)

    def test_flat_spectrum_has_no_peak(self) -> None:
        frequencies = np.arange(6.0, 15.25, 0.25)
        estimate = estimate_central_peak(
            frequencies, np.zeros_like(frequencies), (7.0, 14.0), 1.0
        )
        self.assertFalse(estimate.method_available)
        self.assertEqual(estimate.peak_quality, "poorly_defined_no_peak")

    def test_leave_one_run_out_never_contains_held_out_run(self) -> None:
        for run in (6, 10, 14):
            training = training_runs_for_held_out(run, (6, 10, 14))
            self.assertNotIn(run, training)
            self.assertEqual(set(training), {6, 10, 14} - {run})

    def test_band_construction_is_exact(self) -> None:
        np.testing.assert_array_equal(
            individualized_frequency_vector(10.25, 1.0, 0.5),
            np.array([9.25, 9.75, 10.25, 10.75, 11.25]),
        )

    def test_welch_grid_has_hertz_units_and_expected_resolution(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.normal(size=(4, 3, 641)) * 1e-6
        frequencies, _ = rest_roi_log_welch(
            data, ["C3", "Cz", "C4"], 160.0,
            self.config.central_channels, 4.0
        )
        self.assertAlmostEqual(frequencies[1] - frequencies[0], 0.25)
        self.assertAlmostEqual(frequencies[-1], 80.0)


if __name__ == "__main__":
    unittest.main()
