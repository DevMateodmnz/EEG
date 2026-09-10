"""Numerical and leakage guards for Riemannian zero-shot decoding."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from eeg_project.decoding import RUNS, SubjectDecodingData, TrialIdentity
from eeg_project.riemannian_decoding import (
    covariance_data_from_dataset,
    fit_riemannian_model,
    leave_one_subject_out_folds,
    load_riemannian_config,
    predict_riemannian_target,
    summarize_riemannian_group,
    trial_covariances,
)


def synthetic_subject(subject: int, *, scale: float = 1.0) -> SubjectDecodingData:
    """Make one provenance-complete participant with a covariance-only class signal."""
    rng = np.random.default_rng(8000 + subject)
    arrays: list[np.ndarray] = []
    labels: list[int] = []
    runs: list[int] = []
    identities: list[TrialIdentity] = []
    for run in RUNS:
        for trial in range(10):
            label = trial % 2
            epoch = rng.normal(size=(64, 320))
            epoch[label] *= 2.2
            arrays.append(epoch * scale * 1e-6)
            labels.append(label)
            runs.append(run)
            condition = "both_fists_imagery" if label == 0 else "both_feet_imagery"
            identities.append(
                TrialIdentity(
                    subject, run, trial, trial * 2 + 1, condition,
                    f"S{subject:03d}R{run:02d}.edf", f"hash-{subject}-{run}",
                )
            )
    data = np.stack(arrays)
    return SubjectDecodingData(
        subject, data.copy(), data.copy(), np.asarray(labels), np.asarray(runs),
        identities, np.zeros(len(labels), dtype=bool), [f"EEG{index:02d}" for index in range(64)],
        160.0, 1.0, 3.0,
    )


class RiemannianDecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_riemannian_config()

    def test_covariances_are_symmetric_trace_normalized_and_positive_definite(self) -> None:
        values = np.random.default_rng(11).normal(size=(3, 64, 320)) * 1e-6
        covariances = trial_covariances(values)
        self.assertEqual(covariances.shape, (3, 64, 64))
        np.testing.assert_allclose(covariances, np.swapaxes(covariances, 1, 2), atol=1e-15)
        np.testing.assert_allclose(np.trace(covariances, axis1=1, axis2=2), np.ones(3))
        self.assertTrue(all(np.min(np.linalg.eigvalsh(matrix)) > 0.0 for matrix in covariances))

    def test_development_loso_is_exactly_participant_isolated(self) -> None:
        folds = leave_one_subject_out_folds(range(1, 21))
        self.assertEqual(len(folds), 20)
        for fold in folds:
            self.assertEqual(len(fold.training_subjects), 19)
            self.assertNotIn(fold.test_subject, fold.training_subjects)

    def test_reference_scaler_and_classifier_use_training_people_only(self) -> None:
        training = [covariance_data_from_dataset(synthetic_subject(subject)) for subject in range(1, 5)]
        target = covariance_data_from_dataset(synthetic_subject(20, scale=40.0))
        fitted = fit_riemannian_model(training, "shrinkage_lda", self.config)
        self.assertEqual(fitted.training_subjects, (1, 2, 3, 4))
        self.assertNotIn(target.subject, fitted.training_subjects)
        self.assertEqual(fitted.pipeline.named_steps["tangent"].reference_.shape, (64, 64))
        self.assertEqual(
            int(np.max(np.atleast_1d(fitted.pipeline.named_steps["scaler"].n_samples_seen_))),
            sum(dataset.labels.size for dataset in training),
        )
        result = predict_riemannian_target(fitted, target)
        audit = result.fit_audit_row
        self.assertEqual(audit["reference_mean_fit_subjects"], "1/2/3/4")
        self.assertEqual(audit["tangent_basis_fit_subjects"], "1/2/3/4")
        self.assertEqual(audit["scaler_fit_subjects"], "1/2/3/4")
        self.assertEqual(audit["classifier_fit_subjects"], "1/2/3/4")
        self.assertTrue(audit["target_subject_absent_from_training"])
        for field in (
            "test_EEG_used_for_reference_fit", "test_EEG_used_for_tangent_basis_fit",
            "test_EEG_used_for_scaler_fit", "test_EEG_used_for_classifier_fit",
            "test_labels_used_for_fitting", "target_unlabeled_adaptation_used",
        ):
            self.assertFalse(audit[field])

    def test_target_is_predicted_exactly_once_and_aggregation_is_participant_level(self) -> None:
        training = [covariance_data_from_dataset(synthetic_subject(subject)) for subject in range(1, 5)]
        target = covariance_data_from_dataset(synthetic_subject(20))
        fitted = fit_riemannian_model(training, "l2_logistic_regression", self.config)
        result = predict_riemannian_target(fitted, target)
        self.assertEqual(len(result.prediction_rows), target.labels.size)
        self.assertEqual(len({row["trial_key"] for row in result.prediction_rows}), target.labels.size)
        primary = [row for row in result.subject_rows if not row["qc_sensitivity"]]
        self.assertEqual(len(primary), 1)
        self.assertAlmostEqual(
            primary[0]["primary_mean_run_balanced_accuracy"],
            np.mean([row["balanced_accuracy"] for row in result.run_rows if not row["qc_sensitivity"]]),
        )
        second_participant = {**primary[0], "subject": 21}
        summary = summarize_riemannian_group(
            primary + [second_participant], "l2_logistic_regression", self.config, expected_subject_count=2
        )
        self.assertEqual(summary["participant_count"], 2)
        self.assertEqual(summary["group_observational_unit"], "participant")

    def test_target_participant_in_training_is_rejected_and_prediction_has_no_adaptation_api(self) -> None:
        training = [covariance_data_from_dataset(synthetic_subject(subject)) for subject in range(1, 5)]
        fitted = fit_riemannian_model(training, "shrinkage_lda", self.config)
        with self.assertRaisesRegex(ValueError, "Target participant"):
            predict_riemannian_target(fitted, training[0])
        parameters = inspect.signature(predict_riemannian_target).parameters
        self.assertNotIn("adapt", parameters)
        self.assertNotIn("target_scaler", parameters)


if __name__ == "__main__":
    unittest.main()
