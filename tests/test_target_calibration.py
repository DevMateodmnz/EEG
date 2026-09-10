"""Pre-evaluation guards for minimal supervised target calibration."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from eeg_project.cross_subject_decoding import FittedCrossSubjectModel
from eeg_project.decoding import SubjectDecodingData, TrialIdentity
from eeg_project.target_calibration import (
    CALIBRATION_METHODS,
    CALIBRATION_SIZES,
    calibration_curve_summary,
    calibration_indices,
    convincing_benefit,
    development_method_comparison,
    evaluate_calibration_scenario,
    fit_target_lda,
    load_calibration_config,
    participant_calibration_rows,
    transform_target_with_source,
    validate_nested_calibration_subsets,
)


class IdentityCSP:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return values[:, :, 0]


class IdentityScaler:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return values


class FirstFeatureLDA:
    def decision_function(self, values: np.ndarray) -> np.ndarray:
        return values[:, 0]

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (values[:, 0] >= 0).astype(int)


def synthetic_target(subject: int = 21) -> SubjectDecodingData:
    labels: list[int] = []
    runs: list[int] = []
    identities: list[TrialIdentity] = []
    features: list[np.ndarray] = []
    qc: list[bool] = []
    for run in (6, 10, 14):
        run_labels = [0, 1] * 7 + [1]
        for trial, label in enumerate(run_labels):
            labels.append(label)
            runs.append(run)
            identities.append(
                TrialIdentity(
                    subject,
                    run,
                    trial,
                    trial * 2 + 1,
                    "both_fists_imagery" if label == 0 else "both_feet_imagery",
                    f"S{subject:03d}R{run:02d}.edf",
                    f"hash-{subject}-{run}",
                )
            )
            sign = -1.0 if label == 0 else 1.0
            features.append(np.asarray([[sign], [sign * 0.5], [0.1], [-0.1]]))
            qc.append(trial == 0)
    array = np.stack(features)
    return SubjectDecodingData(
        subject,
        array.copy(),
        array.copy(),
        np.asarray(labels),
        np.asarray(runs),
        identities,
        np.asarray(qc),
        ["a", "b", "c", "d"],
        160.0,
        1.0,
        3.0,
    )


def fitted_source() -> FittedCrossSubjectModel:
    pipeline = SimpleNamespace(
        named_steps={
            "csp": IdentityCSP(),
            "scaler": IdentityScaler(),
            "lda": FirstFeatureLDA(),
        }
    )
    return FittedCrossSubjectModel(
        "csp_4_empirical",
        pipeline,
        tuple(range(1, 21)),
        900,
        {subject: 45 for subject in range(1, 21)},
        {},
        tuple(f"CSP{index}_log_average_power" for index in range(1, 5)),
    )


class TargetCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_calibration_config()
        cls.target = synthetic_target()
        cls.source = transform_target_with_source(fitted_source(), cls.target)

    def test_initial_freeze_preserves_cohorts_curve_and_evaluation_lock(self) -> None:
        self.assertEqual(
            self.config["parent_completed_milestone_git_commit"], "6dca561"
        )
        self.assertTrue(self.config["evaluation_locked"])
        self.assertFalse(
            self.config["evaluation_calibration_outcomes_inspected_at_freeze"]
        )
        self.assertEqual(
            self.config["cohorts"]["development_subjects"], list(range(1, 21))
        )
        self.assertEqual(
            self.config["cohorts"]["evaluation_eligible_subjects"],
            [subject for subject in range(21, 110) if subject not in {88, 92, 100}],
        )
        self.assertEqual(
            tuple(self.config["calibration_sizes"]["ordered_total_trials"]),
            CALIBRATION_SIZES,
        )

    def test_calibration_subsets_are_balanced_chronological_and_nested(self) -> None:
        validate_nested_calibration_subsets(self.target)
        for run in (6, 10, 14):
            previous: set[int] = set()
            for size in CALIBRATION_SIZES:
                indices = calibration_indices(self.target, run, size)
                self.assertEqual(indices.size, size)
                self.assertTrue(previous.issubset(set(indices)))
                self.assertTrue(all(self.target.runs[index] == run for index in indices))
                if size:
                    self.assertEqual(
                        np.bincount(self.target.labels[indices], minlength=2).tolist(),
                        [size // 2, size // 2],
                    )
                    annotations = [
                        self.target.identities[index].annotation_index for index in indices
                    ]
                    self.assertEqual(annotations, sorted(annotations))
                previous = set(indices)

    def test_source_transform_never_refits_target_csp_or_scaler(self) -> None:
        self.assertEqual(self.source.features.shape, (45, 4))
        self.assertTrue(np.array_equal(self.source.labels, self.target.labels))
        with self.assertRaises(ValueError):
            transform_target_with_source(
                FittedCrossSubjectModel(
                    **{
                        **fitted_source().__dict__,
                        "training_subjects": tuple(range(1, 22)),
                    }
                ),
                self.target,
            )

    def test_test_runs_never_enter_calibration_or_decision_fit(self) -> None:
        result = evaluate_calibration_scenario(
            self.source, tuple(range(1, 21)), 6, 8, "threshold"
        )
        self.assertEqual({row["test_run"] for row in result.prediction_rows}, {10, 14})
        calibration = set(result.audit_row["calibration_trial_keys"].split("/"))
        test = set(result.audit_row["test_trial_keys"].split("/"))
        self.assertTrue(calibration.isdisjoint(test))
        self.assertTrue(result.audit_row["test_runs_absent_from_calibration"])
        self.assertFalse(result.audit_row["target_CSP_refit"])
        self.assertFalse(result.audit_row["target_scaler_refit"])
        self.assertFalse(result.audit_row["unlabeled_test_EEG_used_for_adaptation"])

    def test_four_trial_target_lda_is_supported_without_a_fallback(self) -> None:
        indices = calibration_indices(self.target, 6, 4)
        lda = fit_target_lda(self.source.features[indices], self.source.labels[indices])
        self.assertEqual(set(lda.classes_), {0, 1})
        result = evaluate_calibration_scenario(
            self.source, tuple(range(1, 21)), 6, 4, "target_lda"
        )
        self.assertEqual(result.audit_row["target_LDA_training_trial_count"], 4)
        self.assertEqual(result.audit_row["calibration_fists_count"], 2)
        self.assertEqual(result.audit_row["calibration_feet_count"], 2)

    def test_zero_trials_exactly_reproduce_source_predictions(self) -> None:
        for method in CALIBRATION_METHODS:
            result = evaluate_calibration_scenario(
                self.source, tuple(range(1, 21)), 6, 0, method
            )
            observed = np.asarray(
                [int(row["predicted_label"]) for row in result.prediction_rows]
            )
            expected = self.source.source_predictions[np.isin(self.source.runs, [10, 14])]
            self.assertTrue(np.array_equal(observed, expected))

    def test_participant_aggregation_and_selection_are_subject_level(self) -> None:
        predictions = []
        test_runs = []
        for method in CALIBRATION_METHODS:
            for size in CALIBRATION_SIZES:
                for run in (6, 10, 14):
                    result = evaluate_calibration_scenario(
                        self.source, tuple(range(1, 21)), run, size, method
                    )
                    predictions.extend(result.prediction_rows)
                    test_runs.extend(result.test_run_rows)
        participants = participant_calibration_rows(
            test_runs,
            predictions,
            expected_subjects=[21],
            methods=CALIBRATION_METHODS,
        )
        self.assertEqual(len(participants), 20)
        self.assertTrue(
            all(row["group_observational_unit"] == "participant" for row in participants)
        )
        paired, summary = development_method_comparison(participants, self.config)
        self.assertEqual(len(paired), 1)
        self.assertIn(summary["selected_method"], CALIBRATION_METHODS)
        two_subject_rows = participants + [
            {**row, "subject": 22} for row in participants
        ]
        curve = calibration_curve_summary(
            two_subject_rows,
            "threshold",
            self.config,
            expected_subject_count=2,
        )
        self.assertEqual([row["calibration_size"] for row in curve], list(CALIBRATION_SIZES))
        self.assertFalse(convincing_benefit(curve[0], self.config))


if __name__ == "__main__":
    unittest.main()
