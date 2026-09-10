"""Unit, synthetic, and leakage tests for within-subject decoding."""

from __future__ import annotations

import unittest

import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_project.decoding import (
    RUNS,
    SubjectDecodingData,
    TrialIdentity,
    build_candidate_pipeline,
    evaluate_candidate,
    leave_one_run_out_folds,
    load_decoding_config,
    majority_baseline_metrics,
    spectral_log_psd_features,
)


def synthetic_dataset(*, signal: bool, seed: int = 7) -> SubjectDecodingData:
    """Construct three synthetic runs with optional class covariance signal."""
    rng = np.random.default_rng(seed)
    sfreq = 160.0
    channels = ["C3", "Cz", "C4", "P3", "Pz", "P4"]
    data: list[np.ndarray] = []
    labels: list[int] = []
    runs: list[int] = []
    identities: list[TrialIdentity] = []
    for run in RUNS:
        for trial in range(20):
            label = trial % 2
            epoch = rng.normal(size=(len(channels), 320))
            if signal:
                epoch[label] *= 4.0
            data.append(epoch * 1e-6)
            labels.append(label)
            runs.append(run)
            condition = "both_fists_imagery" if label == 0 else "both_feet_imagery"
            identities.append(
                TrialIdentity(1, run, trial, trial * 2 + 1, condition, f"run{run}.edf", f"hash{run}")
            )
    array = np.stack(data)
    return SubjectDecodingData(
        subject=1,
        fixed_task_data_volts=array.copy(),
        csp_task_data_volts=array.copy(),
        labels=np.asarray(labels),
        runs=np.asarray(runs),
        identities=identities,
        qc_candidates=np.zeros(len(labels), dtype=bool),
        channel_names=channels,
        sampling_frequency_hz=sfreq,
        interval_start_seconds=1.0,
        interval_stop_seconds=3.0,
    )


class DecodingUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mne.set_log_level("ERROR")
        cls.config = load_decoding_config()

    def test_leave_one_run_out_folds_are_exact_and_unique(self) -> None:
        runs = np.repeat(np.array(RUNS), 15)
        folds = leave_one_run_out_folds(runs)
        predicted: list[int] = []
        for fold in folds:
            self.assertNotIn(fold.test_run, set(runs[fold.train_indices]))
            self.assertEqual(set(runs[fold.test_indices]), {fold.test_run})
            self.assertEqual(set(runs[fold.train_indices]), set(RUNS) - {fold.test_run})
            predicted.extend(fold.test_indices.tolist())
        self.assertEqual(sorted(predicted), list(range(45)))
        self.assertEqual(len(predicted), len(set(predicted)))

    def test_majority_baseline_has_half_balanced_accuracy(self) -> None:
        result = majority_baseline_metrics(np.array([0] * 7 + [1] * 8))
        self.assertAlmostEqual(result["accuracy"], 8 / 15)
        self.assertEqual(result["balanced_accuracy"], 0.5)

    def test_spectral_features_have_frozen_shape_and_frequency_content(self) -> None:
        dataset = synthetic_dataset(signal=True)
        candidate = self.config["candidate_models"]["spectral_baseline_6"]
        features, names = spectral_log_psd_features(
            dataset, candidate, self.config["fixed_feature_methods"]["spectral"]
        )
        self.assertEqual(features.shape, (60, 6))
        self.assertEqual(len(names), 6)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(names[0], "C3_mu_mean_log_psd_db")

    def test_fold_fit_audit_proves_csp_scaler_and_lda_exclude_test_run(self) -> None:
        dataset = synthetic_dataset(signal=True)
        result = evaluate_candidate(dataset, "csp_4_ledoit_wolf", self.config)
        self.assertEqual(len(result.prediction_rows), 60)
        self.assertEqual(len({row["trial_key"] for row in result.prediction_rows}), 60)
        for audit in result.fit_audit_rows:
            test_run = int(audit["test_run"])
            for field in ("csp_fit_trial_keys", "scaler_fit_trial_keys", "lda_fit_trial_keys"):
                keys = str(audit[field]).split("/")
                self.assertFalse(any(f"_R{test_run:02d}_" in key for key in keys))
            self.assertTrue(audit["test_run_absent_from_all_fit_stages"])
            self.assertTrue(audit["subject_isolation_preserved"])

    def test_scaler_statistics_are_training_fold_only(self) -> None:
        dataset = synthetic_dataset(signal=True)
        features, _ = spectral_log_psd_features(
            dataset,
            self.config["candidate_models"]["spectral_baseline_6"],
            self.config["fixed_feature_methods"]["spectral"],
        )
        fitted = []

        def factory(name, config):
            pipeline = build_candidate_pipeline(name, config)
            fitted.append(pipeline)
            return pipeline

        evaluate_candidate(
            dataset, "spectral_baseline_6", self.config, pipeline_factory=factory
        )
        folds = leave_one_run_out_folds(dataset.runs)
        for pipeline, fold in zip(fitted, folds, strict=True):
            np.testing.assert_allclose(
                pipeline.named_steps["scaler"].mean_,
                np.mean(features[fold.train_indices], axis=0),
            )
            self.assertEqual(set(pipeline.named_steps["lda"].classes_), {0, 1})


class SyntheticDecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mne.set_log_level("ERROR")
        cls.config = load_decoding_config()

    def test_clear_spatial_covariance_signal_decodes_above_chance(self) -> None:
        result = evaluate_candidate(
            synthetic_dataset(signal=True), "csp_4_ledoit_wolf", self.config
        )
        self.assertGreater(
            float(result.subject_row["primary_mean_fold_balanced_accuracy"]), 0.9
        )

    def test_identical_random_distributions_remain_near_chance(self) -> None:
        result = evaluate_candidate(
            synthetic_dataset(signal=False, seed=17), "csp_4_ledoit_wolf", self.config
        )
        score = float(result.subject_row["primary_mean_fold_balanced_accuracy"])
        self.assertGreater(score, 0.25)
        self.assertLess(score, 0.75)

    def test_full_supervised_pipeline_loses_signal_after_training_label_permutation(self) -> None:
        dataset = synthetic_dataset(signal=True, seed=21)
        true_result = evaluate_candidate(dataset, "csp_4_ledoit_wolf", self.config)
        rng = np.random.default_rng(20260829)

        class PermutedTrainingPipeline:
            def __init__(self, name, config):
                self.pipeline = build_candidate_pipeline(name, config)

            def fit(self, x, y):
                self.pipeline.fit(x, rng.permutation(y))
                return self

            def predict(self, x):
                return self.pipeline.predict(x)

            def decision_function(self, x):
                return self.pipeline.decision_function(x)

        permuted = evaluate_candidate(
            dataset,
            "csp_4_ledoit_wolf",
            self.config,
            pipeline_factory=lambda name, config: PermutedTrainingPipeline(name, config),
        )
        self.assertGreater(
            float(true_result.subject_row["primary_mean_fold_balanced_accuracy"]), 0.9
        )
        self.assertLess(
            float(permuted.subject_row["primary_mean_fold_balanced_accuracy"]), 0.7
        )

    def test_random_trial_split_hides_run_specific_feature_failure(self) -> None:
        rng = np.random.default_rng(11)
        labels = np.tile(np.repeat([0, 1], 15), 3)
        runs = np.repeat(np.array(RUNS), 30)
        features = np.zeros((90, 3))
        for run_index, run in enumerate(RUNS):
            selected = runs == run
            features[selected, run_index] = (
                np.where(labels[selected] == 0, -2.0, 2.0)
                + rng.normal(scale=0.1, size=np.sum(selected))
            )
        estimator = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]),
        )
        random_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=5)
        random_score = float(
            np.mean(cross_val_score(estimator, features, labels, cv=random_cv, scoring="balanced_accuracy"))
        )
        run_scores = []
        for fold in leave_one_run_out_folds(runs):
            estimator.fit(features[fold.train_indices], labels[fold.train_indices])
            predictions = estimator.predict(features[fold.test_indices])
            true = labels[fold.test_indices]
            fists = np.mean(predictions[true == 0] == 0)
            feet = np.mean(predictions[true == 1] == 1)
            run_scores.append((fists + feet) / 2)
        self.assertGreater(random_score, 0.9)
        self.assertLess(float(np.mean(run_scores)), 0.6)


if __name__ == "__main__":
    unittest.main()
