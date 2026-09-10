"""Scientific invariants for event conversion, epoching, and trial QC."""

from __future__ import annotations

from collections import Counter
import unittest

import mne
import numpy as np

from eeg_project.epoching import (
    annotations_to_events,
    build_epoch_records,
    calculate_trial_quality,
    expected_epoch_samples,
    extract_run_epochs,
    load_epoching_config,
    modified_robust_z,
    semantic_mapping_for_run,
)
from eeg_project.preprocessing import load_preprocessing_config


class EpochingUnitTests(unittest.TestCase):
    """Fast checks that isolate timing and protocol-policy logic."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_epoching_config()

    def test_run_family_semantics_are_explicit(self) -> None:
        expected = {
            "T0": "rest",
            "T1": "both_fists_imagery",
            "T2": "both_feet_imagery",
        }
        for run in (6, 10, 14):
            self.assertEqual(semantic_mapping_for_run(run, self.config), expected)
        with self.assertRaisesRegex(ValueError, "not supported"):
            semantic_mapping_for_run(4, self.config)

    def test_inclusive_epoch_sample_counts(self) -> None:
        self.assertEqual(expected_epoch_samples(self.config.task_epoch, 160.0), 961)
        self.assertEqual(expected_epoch_samples(self.config.rest_epoch, 160.0), 641)

    def test_annotation_event_samples_match_independent_rounding(self) -> None:
        info = mne.create_info(["C3", "C4"], sfreq=160.0, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((2, 2_000)), info, verbose="error")
        raw.set_annotations(
            mne.Annotations(
                onset=[0.0, 1.234, 2.5],
                duration=[1.234, 1.266, 1.0],
                description=["T0", "T1", "T2"],
                orig_time=raw.info["meas_date"],
            )
        )

        events, event_id, rows = annotations_to_events(raw, 6, self.config)

        self.assertEqual(event_id, self.config.event_id)
        self.assertEqual(events[:, 0].tolist(), [0, 197, 400])
        self.assertEqual([row["manual_event_sample"] for row in rows], [0, 197, 400])
        self.assertEqual(rows[1]["previous_annotation"], "T0")
        self.assertEqual(rows[1]["next_annotation"], "T2")

    def test_boundary_audit_rejects_crossing_without_artifact_judgment(self) -> None:
        info = mne.create_info(["C3"], sfreq=160.0, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((1, 1_000)), info, verbose="error")
        events = np.array([[100, 0, self.config.event_id["T1"]]], dtype=int)
        rows = [
            {
                "annotation_index": 0,
                "description": "T1",
                "duration_seconds": 4.1,
                "previous_annotation": "T0",
                "next_annotation": "T0",
            }
        ]

        (record,) = build_epoch_records(
            1, 6, raw, events, rows, ("T1",), self.config.task_epoch, self.config
        )

        self.assertFalse(record.valid)
        self.assertEqual(record.exclusion_reason, "epoch_crosses_valid_recording_boundary")
        self.assertEqual(record.n_samples, 961)

    def test_modified_robust_z_is_safe_for_zero_mad(self) -> None:
        np.testing.assert_array_equal(modified_robust_z(np.ones(5)), np.zeros(5))


class RealDataEpochingIntegrationTests(unittest.TestCase):
    """Checks against the immutable local PhysioNet subject-1 recordings."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.preprocessing_config = load_preprocessing_config()
        cls.epoching_config = load_epoching_config()
        cls.results = {
            run: extract_run_epochs(
                1, run, cls.preprocessing_config, cls.epoching_config
            )
            for run in (6, 10, 14)
        }
        cls.task_records = [
            record
            for run in (6, 10, 14)
            for record in cls.results[run].task_records
            if record.valid
        ]
        cls.task_data = np.concatenate(
            [cls.results[run].task_epochs.get_data(copy=True) for run in (6, 10, 14)],
            axis=0,
        )
        cls.quality_rows = calculate_trial_quality(
            cls.task_data,
            cls.results[6].task_epochs.times,
            cls.results[6].task_epochs.ch_names,
            cls.task_records,
            cls.epoching_config.trial_quality,
        )

    def test_real_annotation_counts_and_contiguous_alternation(self) -> None:
        for run, result in self.results.items():
            counts = Counter(row["description"] for row in result.annotation_rows)
            self.assertEqual(counts, {"T0": 15, "T1": 7, "T2": 8})
            self.assertEqual(len(result.annotation_rows), 30)
            self.assertTrue(all(row["inside_valid_duration"] for row in result.annotation_rows))
            self.assertFalse(any(row["overlaps_next"] for row in result.annotation_rows))
            finite_gaps = [
                float(row["gap_to_next_seconds"])
                for row in result.annotation_rows
                if row["gap_to_next_seconds"] != ""
            ]
            np.testing.assert_allclose(finite_gaps, 0.0, rtol=0.0, atol=2e-14)
            self.assertTrue(
                all(
                    row["description"] == ("T0" if index % 2 == 0 else row["description"])
                    and (index % 2 == 0 or row["description"] in {"T1", "T2"})
                    for index, row in enumerate(result.annotation_rows)
                ),
                msg=f"Run {run} annotations do not alternate T0/task.",
            )

    def test_real_epoch_counts_shapes_times_and_class_balance(self) -> None:
        for result in self.results.values():
            self.assertEqual(result.task_epochs.get_data(copy=False).shape, (15, 64, 961))
            self.assertEqual(result.rest_epochs.get_data(copy=False).shape, (15, 64, 641))
            self.assertEqual(float(result.task_epochs.times[0]), -2.0)
            self.assertEqual(float(result.task_epochs.times[-1]), 4.0)
            self.assertEqual(float(result.task_epochs.times[320]), 0.0)
            self.assertIsNone(result.task_epochs.baseline)
            self.assertIsNone(result.rest_epochs.baseline)
        self.assertEqual(self.task_data.shape, (45, 64, 961))
        self.assertEqual(
            Counter(record.semantic_condition for record in self.task_records),
            {"both_fists_imagery": 21, "both_feet_imagery": 24},
        )

    def test_exact_first_middle_final_task_timing_and_boundaries(self) -> None:
        for run, result in self.results.items():
            records = result.task_records
            for index, expected_event in ((0, 672), (7, 9_968), (14, 19_264)):
                record = records[index]
                self.assertEqual(record.event_sample, expected_event)
                self.assertAlmostEqual(record.event_time_seconds, expected_event / 160.0)
                self.assertEqual(record.epoch_start_sample, expected_event - 320)
                self.assertEqual(record.epoch_stop_sample_inclusive, expected_event + 640)
                self.assertEqual(record.n_samples, 961)
                self.assertTrue(record.valid, msg=f"Run {run}, task {index} crossed a boundary.")
            self.assertLessEqual(records[-1].epoch_stop_sample_inclusive, 19_919)

    def test_preprocessing_provenance_and_finite_data_are_preserved(self) -> None:
        expected_hashes = {
            6: "5369364f2c4e81ca141679d6dd2ba6ece61c7eb53d7fae31241b308876e1b6b3",
            10: "20de1c7746c2349d16bda5e9f1b0ac7b7ad1581102a2e30dd2ac422696f62fb1",
            14: "2110c48e3106898e3dbca47e39b330637afd3d3b8bc2da3ba1e44f4ac1118137",
        }
        self.assertTrue(np.isfinite(self.task_data).all())
        for run, result in self.results.items():
            self.assertEqual(result.preprocessing.source_sha256, expected_hashes[run])
            self.assertEqual(result.preprocessing.valid_samples, 19_920)
            self.assertEqual(result.preprocessing.excluded_tail_samples, 80)
            self.assertEqual(
                result.preprocessing.stored_valid.annotations,
                result.preprocessing.filtered.annotations,
            )
            self.assertTrue(
                all(record.epoch_stop_sample_inclusive < 19_920 for record in result.task_records)
            )

    def test_trial_quality_flags_candidates_but_rejects_nothing(self) -> None:
        statuses = Counter(row["quality_status"] for row in self.quality_rows)
        self.assertEqual(statuses, {"not flagged": 39, "statistical candidate": 6})
        self.assertTrue(all(row["confirmed_exclusion"] is False for row in self.quality_rows))
        candidate_runs = Counter(
            row["run"]
            for row in self.quality_rows
            if row["quality_status"] == "statistical candidate"
        )
        self.assertEqual(candidate_runs, {6: 4, 10: 1, 14: 1})

    def test_run_six_extraction_is_deterministic(self) -> None:
        repeated = extract_run_epochs(
            1, 6, self.preprocessing_config, self.epoching_config
        )
        np.testing.assert_array_equal(
            repeated.task_epochs.get_data(copy=False),
            self.results[6].task_epochs.get_data(copy=False),
        )
        self.assertEqual(repeated.task_records, self.results[6].task_records)


if __name__ == "__main__":
    unittest.main()
