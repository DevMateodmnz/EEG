"""Unit and leakage tests for zero-shot cross-subject decoding."""

from __future__ import annotations

import inspect
import unittest

import mne
import numpy as np

from eeg_project.cross_subject_decoding import (
    MODEL_NAMES,
    fit_cross_subject_model,
    leave_one_subject_out_folds,
    load_cross_subject_config,
    predict_cross_subject_target,
    summarize_cross_subject_group,
)
from eeg_project.decoding import (
    RUNS,
    SubjectDecodingData,
    TrialIdentity,
    load_decoding_config,
    spectral_log_psd_features,
)


def synthetic_subject(subject: int, *, scale: float = 1.0) -> SubjectDecodingData:
    """Create one participant with stable provenance and a class covariance signal."""
    rng = np.random.default_rng(1000 + subject)
    channels = ["C3", "Cz", "C4", "P3", "Pz", "P4"]
    arrays: list[np.ndarray] = []
    labels: list[int] = []
    runs: list[int] = []
    identities: list[TrialIdentity] = []
    for run in RUNS:
        for trial in range(10):
            label = trial % 2
            epoch = rng.normal(size=(len(channels), 320))
            epoch[label] *= 2.5
            arrays.append(epoch * scale * 1e-6)
            labels.append(label)
            runs.append(run)
            condition = "both_fists_imagery" if label == 0 else "both_feet_imagery"
            identities.append(
                TrialIdentity(
                    subject,
                    run,
                    trial,
                    trial * 2 + 1,
                    condition,
                    f"S{subject:03d}R{run:02d}.edf",
                    f"hash-{subject}-{run}",
                )
            )
    data = np.stack(arrays)
    return SubjectDecodingData(
        subject=subject,
        fixed_task_data_volts=data.copy(),
        csp_task_data_volts=data.copy(),
        labels=np.asarray(labels),
        runs=np.asarray(runs),
        identities=identities,
        qc_candidates=np.zeros(len(labels), dtype=bool),
        channel_names=channels,
        sampling_frequency_hz=160.0,
        interval_start_seconds=1.0,
        interval_stop_seconds=3.0,
    )


class CrossSubjectDecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mne.set_log_level("ERROR")
        cls.cross_config = load_cross_subject_config()
        cls.within_config = load_decoding_config()

    def test_initial_freeze_keeps_evaluation_locked_and_methods_small(self) -> None:
        self.assertEqual(
            self.cross_config["study_stage"],
            "initial_cross_subject_methodology_frozen",
        )
        self.assertTrue(self.cross_config["evaluation_locked"])
        self.assertFalse(
            self.cross_config["evaluation_classifier_outcomes_inspected_at_freeze"]
        )
        self.assertEqual(tuple(self.cross_config["candidate_models"]), MODEL_NAMES)
        self.assertTrue(
            self.cross_config["candidate_policy"]["both_families_advance_to_evaluation"]
        )

    def test_development_loso_excludes_exactly_one_participant(self) -> None:
        folds = leave_one_subject_out_folds(range(1, 21))
        self.assertEqual(len(folds), 20)
        self.assertEqual({fold.test_subject for fold in folds}, set(range(1, 21)))
        for fold in folds:
            self.assertEqual(len(fold.training_subjects), 19)
            self.assertNotIn(fold.test_subject, fold.training_subjects)
            self.assertEqual(
                set(fold.training_subjects), set(range(1, 21)) - {fold.test_subject}
            )

    def test_scaler_and_lda_fit_training_participants_only(self) -> None:
        training = [synthetic_subject(subject) for subject in range(1, 5)]
        target = synthetic_subject(20, scale=50.0)
        fitted = fit_cross_subject_model(
            training, "spectral_baseline_6", self.within_config
        )
        training_features = np.concatenate(
            [
                spectral_log_psd_features(
                    dataset,
                    self.within_config["candidate_models"]["spectral_baseline_6"],
                    self.within_config["fixed_feature_methods"]["spectral"],
                )[0]
                for dataset in training
            ]
        )
        np.testing.assert_allclose(
            fitted.pipeline.named_steps["scaler"].mean_,
            np.mean(training_features, axis=0),
        )
        self.assertEqual(fitted.training_subjects, (1, 2, 3, 4))
        self.assertNotIn(target.subject, fitted.training_subjects)
        result = predict_cross_subject_target(fitted, target, self.within_config)
        self.assertTrue(result.fit_audit_row["target_subject_absent_from_training"])
        self.assertEqual(result.fit_audit_row["scaler_fit_subjects"], "1/2/3/4")
        self.assertEqual(result.fit_audit_row["lda_fit_subjects"], "1/2/3/4")
        self.assertFalse(result.fit_audit_row["target_scaling_fit_used"])
        self.assertFalse(result.fit_audit_row["target_unlabeled_adaptation_used"])

    def test_csp_sees_training_participants_only_and_target_is_predicted_once(self) -> None:
        training = [synthetic_subject(subject) for subject in range(1, 5)]
        target = synthetic_subject(20)
        fitted = fit_cross_subject_model(training, "csp_4_empirical", self.within_config)
        result = predict_cross_subject_target(fitted, target, self.within_config)
        self.assertEqual(result.fit_audit_row["csp_fit_subjects"], "1/2/3/4")
        self.assertFalse(result.fit_audit_row["target_CSP_adaptation_used"])
        self.assertEqual(len(result.prediction_rows), target.labels.size)
        self.assertEqual(
            len({row["trial_key"] for row in result.prediction_rows}), target.labels.size
        )
        self.assertTrue(all(row["zero_shot"] for row in result.prediction_rows))
        self.assertTrue(
            all(not row["target_adaptation_used"] for row in result.prediction_rows)
        )
        self.assertEqual(len(result.run_rows), 3)
        self.assertAlmostEqual(
            result.subject_row["primary_mean_run_balanced_accuracy"],
            np.mean([row["balanced_accuracy"] for row in result.run_rows]),
        )

    def test_target_in_training_is_rejected(self) -> None:
        training = [synthetic_subject(subject) for subject in range(1, 5)]
        fitted = fit_cross_subject_model(
            training, "spectral_baseline_6", self.within_config
        )
        with self.assertRaisesRegex(ValueError, "Target participant"):
            predict_cross_subject_target(fitted, training[0], self.within_config)

    def test_group_summary_uses_participants_not_pooled_trials(self) -> None:
        rows = []
        for model_index, model in enumerate(MODEL_NAMES):
            for subject in range(1, 21):
                score = 0.51 + model_index * 0.02 + subject / 1000
                rows.append(
                    {
                        "subject": subject,
                        "model": model,
                        "primary_mean_run_balanced_accuracy": score,
                        "fists_recall": score,
                        "feet_recall": score,
                        "group_observational_unit": "participant",
                        "trial_count": 10000 if subject == 1 else 1,
                    }
                )
        summary = summarize_cross_subject_group(
            rows,
            "spectral_baseline_6",
            self.cross_config,
            expected_subject_count=20,
        )
        self.assertEqual(summary["participant_count"], 20)
        self.assertAlmostEqual(summary["median_balanced_accuracy"], 0.5205)
        with self.assertRaisesRegex(RuntimeError, "one row per participant"):
            summarize_cross_subject_group(
                rows[:-1],
                "csp_4_empirical",
                self.cross_config,
                expected_subject_count=20,
            )

    def test_prediction_api_has_no_adaptation_arguments(self) -> None:
        parameters = inspect.signature(predict_cross_subject_target).parameters
        self.assertNotIn("adapt", parameters)
        self.assertNotIn("target_scaler", parameters)
        guards = self.cross_config["zero_shot_leakage_guards"]
        self.assertFalse(guards["target_subject_centering"])
        self.assertFalse(guards["target_subject_covariance_alignment"])
        self.assertFalse(guards["target_subject_transductive_normalization"])


if __name__ == "__main__":
    unittest.main()
