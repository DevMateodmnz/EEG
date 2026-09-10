"""Scientific invariants for artifact policy and feature stability."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

import numpy as np

from eeg_project.epoching import (
    calculate_trial_quality,
    extract_run_epochs,
    load_epoching_config,
)
from eeg_project.preprocessing import load_preprocessing_config
from eeg_project.time_frequency import assemble_spectral_epoch_dataset
from eeg_project.trial_quality import (
    QUALITY_METRICS,
    classify_metric_candidates,
    db_power_ratio,
    grade_stability,
    leave_one_out_median_influence,
    load_trial_quality_config,
    spearman_rho,
    systematic_matched_controls,
    trimmed_mean,
)


class TrialQualityUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_trial_quality_config()

    def test_db_ratio_and_trimmed_mean_known_values(self) -> None:
        result = db_power_ratio(np.array([0.5, 1.0, 2.0]), np.ones(3))
        np.testing.assert_allclose(result, [-3.01029995664, 0.0, 3.01029995664])
        self.assertEqual(trimmed_mean(np.array([-100, 1, 2, 3, 100]), 0.2), 2.0)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            db_power_ratio(np.array([0.0]), np.array([1.0]))

    def test_label_free_robust_outlier_flags_expected_row(self) -> None:
        metrics = {
            name: np.array([1.0, 1.1, 0.9, 1.05, 20.0])
            if name == "max_p2p_uV"
            else np.array([1.0, 1.1, 0.9, 1.05, 1.0])
            for name in QUALITY_METRICS
        }
        _, evidence, candidates = classify_metric_candidates(metrics, 3.5)
        np.testing.assert_array_equal(candidates, [False, False, False, False, True])
        self.assertEqual(evidence[-1], ("high max_p2p_uV",))

    def test_influence_and_rank_correlation_are_deterministic(self) -> None:
        rows = leave_one_out_median_influence(
            np.array([1.0, 2.0, 100.0]), [(6, 0), (6, 1), (6, 2)]
        )
        self.assertEqual(rows[-1]["median_without_trial"], 1.5)
        self.assertEqual(rows[-1]["median_shift"], -0.5)
        self.assertEqual(spearman_rho(np.array([3, 1, 2]), np.array([30, 10, 20])), 1.0)

    def test_predefined_stability_grades_stable_mixed_unstable(self) -> None:
        names = [f"criterion_{index}" for index in range(7)]
        stable = {name: True for name in names}
        mixed = {name: index < 5 for index, name in enumerate(names)}
        unstable = {name: index < 4 for index, name in enumerate(names)}
        self.assertEqual(grade_stability(stable, self.config), (7, "high"))
        self.assertEqual(grade_stability(mixed, self.config), (5, "moderate"))
        self.assertEqual(grade_stability(unstable, self.config), (4, "low"))
        stricter = replace(self.config, high_minimum_passed_criteria=8)
        self.assertEqual(grade_stability(stable, stricter), (7, "moderate"))


class RealDataTrialQualityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        preprocessing = load_preprocessing_config()
        epoching = load_epoching_config()
        cls.results = {
            run: extract_run_epochs(1, run, preprocessing, epoching)
            for run in (6, 10, 14)
        }
        cls.dataset = assemble_spectral_epoch_dataset(cls.results, (6, 10, 14))
        cls.snapshot = cls.dataset.task_data_volts.copy()
        cls.quality_rows = calculate_trial_quality(
            cls.dataset.task_data_volts,
            cls.dataset.task_times,
            cls.dataset.channel_names,
            [pair.task for pair in cls.dataset.pairs],
            epoching.trial_quality,
        )

    def test_master_trials_pairing_hashes_and_candidate_identities_remain_exact(self) -> None:
        self.assertEqual(self.dataset.task_data_volts.shape, (45, 64, 961))
        self.assertEqual(self.dataset.rest_data_volts.shape, (45, 64, 641))
        self.assertEqual(
            Counter(pair.task.semantic_condition for pair in self.dataset.pairs),
            {"both_fists_imagery": 21, "both_feet_imagery": 24},
        )
        expected_candidates = {(6, 1), (6, 7), (6, 8), (6, 12), (10, 8), (14, 9)}
        actual_candidates = {
            (int(row["run"]), int(row["run_trial_index"]))
            for row in self.quality_rows
            if row["quality_status"] == "statistical candidate"
        }
        self.assertEqual(actual_candidates, expected_candidates)
        self.assertEqual(
            {pair.source_sha256 for pair in self.dataset.pairs},
            {
                "5369364f2c4e81ca141679d6dd2ba6ece61c7eb53d7fae31241b308876e1b6b3",
                "20de1c7746c2349d16bda5e9f1b0ac7b7ad1581102a2e30dd2ac422696f62fb1",
                "2110c48e3106898e3dbca47e39b330637afd3d3b8bc2da3ba1e44f4ac1118137",
            },
        )
        np.testing.assert_array_equal(self.dataset.task_data_volts, self.snapshot)
        self.assertTrue(
            all(pair.rest.annotation_index + 1 == pair.task.annotation_index for pair in self.dataset.pairs)
        )

    def test_systematic_controls_are_same_run_condition_and_not_flagged(self) -> None:
        matches = systematic_matched_controls(
            self.quality_rows,
            (
                "max_task_p2p_uV",
                "median_task_p2p_uV",
                "frontal_max_task_p2p_uV",
                "max_task_abs_step_uV",
                "minimum_task_channel_std_uV",
                "median_task_channel_std_uV",
            ),
        )
        lookup = {
            (int(row["run"]), int(row["run_trial_index"])): row
            for row in self.quality_rows
        }
        self.assertEqual(len(matches), 6)
        for candidate, matched in matches.items():
            self.assertEqual(candidate[0], matched[0])
            self.assertEqual(
                lookup[candidate]["semantic_condition"],
                lookup[matched]["semantic_condition"],
            )
            self.assertEqual(lookup[matched]["quality_status"], "not flagged")


if __name__ == "__main__":
    unittest.main()
