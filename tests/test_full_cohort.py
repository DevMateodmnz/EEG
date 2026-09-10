"""Pre-outcome invariants for full-cohort partitioning and method freeze."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import csv

from dataclasses import replace

from eeg_project.cohort import cohort_feature_summaries
from eeg_project.cohort_processing import process_subject
from eeg_project.epoching import load_epoching_config
from eeg_project.full_cohort import (
    checkpoint_payload,
    cohort_label,
    eligibility_for_requested_subjects,
    full_cohort_fingerprint,
    load_subject_checkpoint,
    load_frozen_replication_policy,
    load_full_cohort_config,
    write_subject_checkpoint,
)
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY, load_preprocessing_config
from eeg_project.time_frequency import load_event_related_spectral_config
from eeg_project.trial_quality import load_trial_quality_config


class FullCohortFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_full_cohort_config()
        cls.policy = load_frozen_replication_policy(cls.config)

    def test_chronological_cohort_partition_is_exact_and_disjoint(self) -> None:
        self.assertEqual(self.config.development_subjects, (1,))
        self.assertEqual(self.config.replication_1_subjects, tuple(range(2, 21)))
        self.assertEqual(self.config.replication_2_subjects, tuple(range(21, 110)))
        self.assertEqual(self.config.combined_replication_subjects, tuple(range(2, 110)))
        self.assertEqual(
            set(self.config.development_subjects)
            | set(self.config.combined_replication_subjects),
            set(range(1, 110)),
        )
        self.assertFalse(
            set(self.config.replication_1_subjects)
            & set(self.config.replication_2_subjects)
        )

    def test_permanent_labels_do_not_make_old_data_new_again(self) -> None:
        self.assertEqual(cohort_label(1, self.config), "development")
        self.assertEqual(cohort_label(2, self.config), "replication_1")
        self.assertEqual(cohort_label(20, self.config), "replication_1")
        self.assertEqual(cohort_label(21, self.config), "replication_2")
        self.assertEqual(cohort_label(109, self.config), "replication_2")

    def test_primary_features_and_success_rules_reuse_frozen_policy(self) -> None:
        self.assertEqual(
            self.policy.primary_replication_features,
            (
                ("both_fists_imagery", "C4", "central_12_13"),
                ("both_fists_imagery", "C3", "central_12_13"),
            ),
        )
        criteria = self.policy.replication_success_criteria
        self.assertEqual(criteria.predicted_direction, "negative")
        self.assertEqual(criteria.strong_minimum_subject_fraction_in_direction, 2 / 3)
        self.assertTrue(criteria.strong_requires_qc_sensitivity_direction)
        self.assertTrue(criteria.strong_requires_leave_one_subject_out_direction)

    def test_freeze_precedes_outcomes_and_forbids_parameter_refinement(self) -> None:
        self.assertEqual(
            self.config.methodology_frozen_at_git_commit,
            "395df8ebc79b1658ff91ed96c90b41fd954d037e",
        )
        self.assertFalse(self.config.outcomes_inspected_at_freeze)
        self.assertTrue(self.config.cohort_2_separate_before_combining)
        self.assertTrue(self.config.full_cohort_parameter_refinement_forbidden)
        self.assertFalse(self.config.persist_full_tfr_arrays)

    def test_missing_run_is_explicit_and_positive_outcome_is_not_eligibility_input(self) -> None:
        rows = [
            {
                "subject": subject,
                "run": run,
                "protocol_compatible": True,
                "failure_reason": "",
            }
            for subject in (21, 22)
            for run in ((6, 10, 14) if subject == 21 else (6, 14))
        ]
        eligibility = eligibility_for_requested_subjects(
            rows, (21, 22), self.config
        )
        eligible, missing = eligibility
        self.assertEqual(eligible["technical_eligibility"], "eligible")
        self.assertFalse(eligible["outcome_used_for_eligibility"])
        self.assertEqual(missing["technical_eligibility"], "technically_incompatible")
        self.assertIn("run10:missing_audit_row", missing["failure_reason"])

    def test_cohort_statistics_isolate_replication_2_and_combined_subjects(self) -> None:
        policy = replace(
            self.policy,
            secondary_features=(),
            group_inference=replace(
                self.policy.group_inference,
                bootstrap_subject_median_resamples=100,
            ),
        )
        subject_rows = []
        primary_rows = []
        for channel in ("C4", "C3"):
            for subject, value in ((1, -99.0), (2, -20.0), (20, -10.0), (21, 5.0), (109, 15.0)):
                row = {
                    "subject": subject,
                    "subject_role": cohort_label(subject, self.config),
                    "semantic_condition": "both_fists_imagery",
                    "channel": channel,
                    "frequency_band": "central_12_13",
                    "median_change_percent": value,
                    "exclude_qc_candidates_median_change_percent": value,
                }
                subject_rows.append(row)
                primary_rows.append({**row, "supporting_run_count": 3 if value < 0 else 0})
        cohort_2 = cohort_feature_summaries(
            subject_rows, primary_rows, policy, included_subject_ids=(21, 109)
        )
        combined = cohort_feature_summaries(
            subject_rows, primary_rows, policy, included_subject_ids=(2, 20, 21, 109)
        )
        self.assertTrue(all(row["negative_subject_count"] == 0 for row in cohort_2))
        self.assertTrue(all(row["replication_subject_count"] == 2 for row in cohort_2))
        self.assertTrue(all(row["negative_subject_count"] == 2 for row in combined))
        self.assertTrue(all(row["replication_subject_count"] == 4 for row in combined))

    def test_checkpoint_rejects_configuration_or_source_drift(self) -> None:
        fingerprint = full_cohort_fingerprint(self.config)
        result = {
            "trial_rows": [],
            "feature_rows": [],
            "run_rows": [],
            "peak_row": {},
            "qc_row": {},
            "task_tfr_shape": [45, 3, 30, 241],
            "rest_tfr_shape": [45, 3, 30, 161],
        }
        payload = checkpoint_payload(
            21, "replication_2", result, fingerprint, {"run06": "abc"}
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "subject021.json"
            write_subject_checkpoint(path, payload)
            self.assertIsNotNone(
                load_subject_checkpoint(
                    path, 21, "replication_2", fingerprint, {"run06": "abc"}
                )
            )
            self.assertIsNone(
                load_subject_checkpoint(
                    path, 21, "replication_2", "changed", {"run06": "abc"}
                )
            )
            self.assertIsNone(
                load_subject_checkpoint(
                    path, 21, "replication_2", fingerprint, {"run06": "changed"}
                )
            )


class HistoricalPipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full_config = load_full_cohort_config()
        cls.policy = load_frozen_replication_policy(cls.full_config)
        paths = {
            name: Path(__file__).resolve().parents[1] / record.path
            for name, record in cls.full_config.frozen_methodology.items()
        }
        cls.scientific_configs = (
            load_preprocessing_config(paths["preprocessing"]),
            load_epoching_config(paths["epoching"]),
            load_event_related_spectral_config(paths["event_related_spectral"]),
            load_trial_quality_config(paths["trial_quality"]),
        )
        audit_path = (
            Path(__file__).resolve().parents[1]
            / "docs/assets/subjects01-20_replication_cohort_audit.csv"
        )
        with audit_path.open(newline="", encoding="utf-8") as input_file:
            cls.audit_rows = list(csv.DictReader(input_file))

    def _primary(self, subject: int, role: str) -> dict[str, float]:
        result = process_subject(
            subject,
            role,
            self.policy,
            *self.scientific_configs,
            DEFAULT_DATA_DIRECTORY,
            self.audit_rows,
        )
        return {
            str(row["channel"]): float(row["median_change_percent"])
            for row in result["feature_rows"]
            if row["semantic_condition"] == "both_fists_imagery"
            and row["frequency_band"] == "central_12_13"
        }

    def test_subject_one_remains_numerically_exact(self) -> None:
        values = self._primary(1, "development")
        self.assertAlmostEqual(values["C4"], -52.28373306890746)
        self.assertAlmostEqual(values["C3"], -55.57490648782269)

    def test_known_replication_one_subject_remains_numerically_exact(self) -> None:
        values = self._primary(2, "replication_1")
        historical_path = (
            Path(__file__).resolve().parents[1]
            / "docs/assets/subjects01-20_replication_subject_feature_summary.csv"
        )
        with historical_path.open(newline="", encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))
        expected = {
            row["channel"]: float(row["median_change_percent"])
            for row in rows
            if int(row["subject"]) == 2
            and row["semantic_condition"] == "both_fists_imagery"
            and row["frequency_band"] == "central_12_13"
        }
        self.assertEqual(values, expected)


if __name__ == "__main__":
    unittest.main()
