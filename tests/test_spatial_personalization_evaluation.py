"""Pre-outcome tests for final spatial-personalization reporting and checkpoints."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from eeg_project.decoding import SubjectDecodingData
from eeg_project.spatial_personalization import (
    A_METHOD,
    classify_personalization_geometry,
)
from eeg_project.spatial_personalization_evaluation import (
    calibration_run_summary_rows,
    historical_context_rows,
    load_final_spatial_personalization_config,
)
from scripts.evaluate_spatial_personalization import (
    raw_checkpoint_metadata,
    valid_raw_checkpoint,
)
from tests.test_spatial_personalization import synthetic_inputs


class SpatialPersonalizationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.final, cls.initial = load_final_spatial_personalization_config()

    def test_final_loader_preserves_selected_methods_and_cohort(self) -> None:
        self.assertEqual(self.final["representation_A"]["method"], A_METHOD)
        self.assertEqual(
            self.final["representation_B"]["method"],
            "source_csp_target_lda_source_scaler",
        )
        self.assertEqual(
            self.final["representation_C"]["method"], "target_csp_ledoit_wolf"
        )
        self.assertEqual(
            self.final["cohorts"]["evaluation_eligible_subjects"],
            [subject for subject in range(21, 110) if subject not in {88, 92, 100}],
        )

    def test_calibration_run_summary_keeps_participants_as_units(self) -> None:
        rows: list[dict[str, object]] = []
        methods = [A_METHOD, "B", "C"]
        for subject in range(21, 25):
            for method_index, method in enumerate(methods):
                for calibration_run in (6, 10, 14):
                    for test_run in (run for run in (6, 10, 14) if run != calibration_run):
                        rows.append(
                            {
                                "subject": subject,
                                "method": method,
                                "calibration_run": calibration_run,
                                "test_run": test_run,
                                "qc_sensitivity": False,
                                "balanced_accuracy": (
                                    0.50
                                    + method_index * 0.03
                                    + subject / 1000
                                    + calibration_run / 10000
                                ),
                            }
                        )
        summary = calibration_run_summary_rows(rows, methods, expected_subject_count=4)
        self.assertEqual(len(summary), 9)
        self.assertTrue(all(row["participant_count"] == 4 for row in summary))

    def test_historical_context_marks_incompatible_training_geometry(self) -> None:
        methods = [A_METHOD, "B", "C"]
        participants = [
            {
                "subject": subject,
                "method": method,
                "qc_sensitivity": False,
                "primary_mean_scenario_test_run_balanced_accuracy": 0.55,
            }
            for subject in range(21, 25)
            for method in methods
        ]
        threshold = [
            {
                "subject": str(subject),
                "method": "threshold",
                "calibration_size": "14",
                "qc_sensitivity": "False",
                "primary_mean_test_run_balanced_accuracy": "0.60",
            }
            for subject in range(21, 25)
        ]
        within = [
            {
                "subject": str(subject),
                "model": "csp_4_empirical",
                "qc_sensitivity": "False",
                "primary_mean_fold_balanced_accuracy": "0.65",
            }
            for subject in range(21, 25)
        ]
        rows = historical_context_rows(participants, threshold, within, methods)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["comparison_geometry"] for row in rows},
            {"same_calibration_test_geometry", "descriptive_different_training_geometry"},
        )

    def test_interpretation_gate_uses_only_frozen_boolean_transitions(self) -> None:
        convincing = {"convincing_paired_gain": True, "median_paired_difference": 0.04}
        not_convincing = {"convincing_paired_gain": False, "median_paired_difference": 0.01}
        self.assertEqual(
            classify_personalization_geometry(convincing, not_convincing, convincing),
            "decision-boundary bottleneck",
        )
        self.assertEqual(
            classify_personalization_geometry(not_convincing, convincing, convincing),
            "spatial-representation bottleneck",
        )
        self.assertEqual(
            classify_personalization_geometry(convincing, convincing, convincing),
            "both decision layer and spatial representation matter",
        )
        self.assertEqual(
            classify_personalization_geometry(not_convincing, not_convincing, not_convincing),
            "one-run personalization is insufficient",
        )

    def test_raw_target_tensor_checkpoint_round_trip_is_hash_validated(self) -> None:
        dataset, source = synthetic_inputs()
        full_array = np.repeat(
            np.repeat(dataset.csp_task_data_volts[:, :1, :1], 64, axis=1),
            320,
            axis=2,
        )
        dataset = SubjectDecodingData(
            dataset.subject,
            np.empty((dataset.labels.size, 0, 0)),
            full_array,
            dataset.labels,
            dataset.runs,
            dataset.identities,
            dataset.qc_candidates,
            [f"EEG{index:02d}" for index in range(64)],
            160.0,
            1.0,
            3.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "subject_021.json"
            array_path = root / "subject_021_csp.npy"
            raw_checkpoint_metadata(
                dataset,
                metadata_path,
                array_path,
                final_hash="final",
                frozen_core_hash="core",
                source_checkpoint_sha256="source",
            )
            restored = valid_raw_checkpoint(
                metadata_path,
                array_path,
                21,
                source,
                final_hash="final",
                frozen_core_hash="core",
                source_checkpoint_sha256="source",
            )
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertTrue(np.array_equal(restored[1], full_array))
            invalid = valid_raw_checkpoint(
                metadata_path,
                array_path,
                21,
                source,
                final_hash="changed",
                frozen_core_hash="core",
                source_checkpoint_sha256="source",
            )
            self.assertIsNone(invalid)


if __name__ == "__main__":
    unittest.main()
