"""RED-first invariants for frozen-architecture unilateral generalization."""

from __future__ import annotations

import unittest

import numpy as np

from eeg_project.unilateral_generalization import (
    UNILATERAL_LABELS,
    UNILATERAL_RUNS,
    leave_one_run_out_folds,
    task_definition_for_run,
    validate_unilateral_class_balance,
    validate_source_identity,
    load_study_config,
    load_parent_decoder_config,
)


class UnilateralTaskTests(unittest.TestCase):
    def test_contextual_unilateral_mapping_is_exact(self) -> None:
        for run in UNILATERAL_RUNS:
            task = task_definition_for_run(run)
            self.assertEqual(task.labels, UNILATERAL_LABELS)
            self.assertEqual(task.annotation_to_label, {"T1": 0, "T2": 1})
        self.assertEqual(task_definition_for_run(6).identifier, "bilateral_historical_only")

    def test_physical_and_unknown_runs_are_rejected(self) -> None:
        for run in (3, 5, 7, 9, 11, 13, 99):
            with self.assertRaises(ValueError):
                task_definition_for_run(run)

    def test_run_holdout_is_complete_deterministic_and_disjoint(self) -> None:
        runs = np.array([4] * 15 + [8] * 15 + [12] * 15)
        folds = leave_one_run_out_folds(runs)
        self.assertEqual([fold.test_run for fold in folds], [4, 8, 12])
        self.assertEqual([fold.training_runs for fold in folds], [(8, 12), (4, 12), (4, 8)])
        self.assertTrue(all(not set(fold.train_indices) & set(fold.test_indices) for fold in folds))
        self.assertTrue(all(set(runs[fold.test_indices]) == {fold.test_run} for fold in folds))

    def test_source_validation_never_accepts_wrong_identity(self) -> None:
        with self.assertRaises(ValueError):
            validate_source_identity(21, 6, "/tmp/S021R06.edf")
        self.assertEqual(validate_source_identity(21, 4, "/tmp/S021R04.edf"), "S021R04.edf")

    def test_study_reuses_exact_parent_architecture_without_unilateral_defaults(self) -> None:
        study = load_study_config()
        parent = load_parent_decoder_config(study)
        self.assertEqual(study["classification"], "GENERALIZATION_REPLICATION")
        self.assertEqual(study["frozen_architecture"]["parent_models"], ["spectral_baseline_6", "csp_4_empirical"])
        self.assertEqual(parent["candidate_models"]["csp_4_empirical"]["frequency_band_hz"], [8.0, 30.0])
        self.assertEqual(parent["candidate_models"]["csp_4_empirical"]["csp_components"], 4)
        self.assertEqual(parent["prediction_target"]["task_interval_seconds"], [1.0, 3.0])

    def test_technical_retention_allows_only_verified_target_counts(self) -> None:
        validate_unilateral_class_balance(np.array([0] * 7 + [1] * 8))
        validate_unilateral_class_balance(np.array([0] * 7 + [1] * 7))
        validate_unilateral_class_balance(np.array([0] * 7 + [1] * 6))
        validate_unilateral_class_balance(np.array([0] * 6 + [1] * 6))
        with self.assertRaisesRegex(RuntimeError, "class balance"):
            validate_unilateral_class_balance(np.array([0] * 6 + [1] * 8))


if __name__ == "__main__":
    unittest.main()
