"""Leakage, aggregation, selection, and CSP-stability tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from eeg_project.decoding import SubjectDecodingData, TrialIdentity
from eeg_project.spatial_personalization import (
    A_METHOD,
    B_METHODS,
    C_METHODS,
    DEVELOPMENT_METHODS,
    complete_run_calibration_indices,
    csp_subspace_similarity,
    evaluate_personalization_scenario,
    load_spatial_personalization_config,
    participant_csp_stability_row,
    participant_personalization_rows,
    select_development_methods,
)
from eeg_project.target_calibration import TargetSourceFeatures


def synthetic_inputs(subject: int = 21) -> tuple[SubjectDecodingData, TargetSourceFeatures]:
    labels: list[int] = []
    runs: list[int] = []
    identities: list[TrialIdentity] = []
    raw_trials: list[np.ndarray] = []
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
            raw_trials.append(
                np.asarray(
                    [
                        np.full(8, sign),
                        np.full(8, sign * 0.5),
                        np.full(8, -sign * 0.25),
                        np.full(8, sign * 0.1),
                    ]
                )
            )
            features.append(np.asarray([sign, sign * 0.5, -sign * 0.25, sign * 0.1]))
            qc.append(trial == 0)
    label_array = np.asarray(labels)
    run_array = np.asarray(runs)
    qc_array = np.asarray(qc)
    raw_array = np.asarray(raw_trials)
    feature_array = np.asarray(features)
    dataset = SubjectDecodingData(
        subject,
        raw_array.copy(),
        raw_array.copy(),
        label_array,
        run_array,
        identities,
        qc_array,
        ["a", "b", "c", "d"],
        160.0,
        1.0,
        3.0,
    )
    source = TargetSourceFeatures(
        subject,
        feature_array,
        feature_array[:, 0],
        (feature_array[:, 0] >= 0).astype(int),
        label_array.copy(),
        run_array.copy(),
        tuple(identities),
        qc_array.copy(),
    )
    return dataset, source


class RecordingModel:
    def __init__(self, *, csp: bool) -> None:
        self.csp = csp
        self.fit_values: np.ndarray | None = None
        self.fit_labels: np.ndarray | None = None
        self.named_steps = (
            {"csp": SimpleNamespace(filters_=np.eye(4))} if csp else {}
        )

    def fit(self, values: np.ndarray, labels: np.ndarray) -> "RecordingModel":
        self.fit_values = np.asarray(values).copy()
        self.fit_labels = np.asarray(labels).copy()
        return self

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        return array[:, 0] if array.ndim == 2 else np.mean(array[:, 0], axis=1)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (self.decision_function(values) >= 0).astype(int)


class SpatialPersonalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_spatial_personalization_config()
        cls.dataset, cls.source = synthetic_inputs()

    def test_complete_calibration_run_is_balanced_chronological_and_disjoint(self) -> None:
        for run in (6, 10, 14):
            indices = complete_run_calibration_indices(self.dataset, run)
            self.assertEqual(indices.size, 14)
            self.assertEqual(np.bincount(self.dataset.labels[indices]).tolist(), [7, 7])
            self.assertTrue(all(self.dataset.runs[index] == run for index in indices))
            annotations = [self.dataset.identities[index].annotation_index for index in indices]
            self.assertEqual(annotations, sorted(annotations))

    def test_A_uses_exact_source_predictions_and_performs_no_target_fit(self) -> None:
        def forbidden_factory(method: str, config: object) -> object:
            raise AssertionError("A must not construct a target-fitted model")

        result = evaluate_personalization_scenario(
            self.dataset,
            self.source,
            tuple(range(1, 21)),
            6,
            A_METHOD,
            self.config,
            pipeline_factory=forbidden_factory,
        )
        expected_indices = np.flatnonzero(np.isin(self.dataset.runs, [10, 14]))
        self.assertEqual(
            [row["predicted_label"] for row in result.prediction_rows],
            self.source.source_predictions[expected_indices].tolist(),
        )
        self.assertEqual(result.audit_row["target_CSP_fit_trial_keys"], "not_applicable")
        self.assertEqual(result.audit_row["target_LDA_fit_trial_keys"], "not_applicable")

    def test_B_never_refits_CSP_and_all_fits_use_calibration_only(self) -> None:
        created: list[RecordingModel] = []

        def factory(method: str, config: object) -> RecordingModel:
            model = RecordingModel(csp=False)
            created.append(model)
            return model

        for method in B_METHODS:
            result = evaluate_personalization_scenario(
                self.dataset,
                self.source,
                tuple(range(1, 21)),
                6,
                method,
                self.config,
                pipeline_factory=factory,
            )
            calibration = complete_run_calibration_indices(self.dataset, 6)
            self.assertTrue(np.array_equal(created[-1].fit_values, self.source.features[calibration]))
            self.assertEqual(result.audit_row["target_CSP_fit_trial_keys"], "not_applicable")
            self.assertEqual(result.audit_row["target_LDA_fit_trial_keys"], result.audit_row["calibration_trial_keys"])
            self.assertFalse(result.audit_row["test_EEG_used_for_LDA_fit"])

    def test_C_fits_CSP_scaler_and_LDA_only_on_calibration_run(self) -> None:
        created: list[RecordingModel] = []

        def factory(method: str, config: object) -> RecordingModel:
            model = RecordingModel(csp=True)
            created.append(model)
            return model

        result = evaluate_personalization_scenario(
            self.dataset,
            self.source,
            tuple(range(1, 21)),
            10,
            C_METHODS[0],
            self.config,
            pipeline_factory=factory,
        )
        calibration = complete_run_calibration_indices(self.dataset, 10)
        self.assertTrue(
            np.array_equal(created[0].fit_values, self.dataset.csp_task_data_volts[calibration])
        )
        fit_keys = result.audit_row["calibration_trial_keys"]
        self.assertEqual(result.audit_row["target_CSP_fit_trial_keys"], fit_keys)
        self.assertEqual(result.audit_row["target_scaler_fit_trial_keys"], fit_keys)
        self.assertEqual(result.audit_row["target_LDA_fit_trial_keys"], fit_keys)
        test_keys = set(result.audit_row["test_trial_keys"].split("/"))
        self.assertTrue(set(fit_keys.split("/")).isdisjoint(test_keys))
        self.assertEqual(result.csp_filter_subspace.shape, (4, 4))

    def test_all_methods_compare_identical_test_trials_and_aggregate_one_subject(self) -> None:
        predictions: list[dict[str, object]] = []
        scores: list[dict[str, object]] = []
        for method in DEVELOPMENT_METHODS:
            for calibration_run in (6, 10, 14):
                result = evaluate_personalization_scenario(
                    self.dataset,
                    self.source,
                    tuple(range(1, 21)),
                    calibration_run,
                    method,
                    self.config,
                    pipeline_factory=lambda method, config: RecordingModel(csp=method in C_METHODS),
                )
                predictions.extend(result.prediction_rows)
                scores.extend(result.test_run_rows)
        contexts: dict[str, set[tuple[int, str]]] = {}
        for method in DEVELOPMENT_METHODS:
            contexts[method] = {
                (int(row["calibration_run"]), str(row["trial_key"]))
                for row in predictions
                if row["method"] == method
            }
        self.assertTrue(all(value == contexts[A_METHOD] for value in contexts.values()))
        participants = participant_personalization_rows(
            scores,
            predictions,
            expected_subjects=[21],
            methods=DEVELOPMENT_METHODS,
        )
        self.assertEqual(len(participants), len(DEVELOPMENT_METHODS) * 2)
        self.assertTrue(all(row["group_observational_unit"] == "participant" for row in participants))

    def test_development_selection_is_global_and_uses_no_evaluation_outcome(self) -> None:
        rows: list[dict[str, object]] = []
        for subject in range(1, 21):
            values = {
                A_METHOD: 0.50,
                B_METHODS[0]: 0.55,
                B_METHODS[1]: 0.57,
                C_METHODS[0]: 0.56,
                C_METHODS[1]: 0.565,
            }
            for method, score in values.items():
                for qc in (False, True):
                    rows.append(
                        {
                            "subject": subject,
                            "method": method,
                            "qc_sensitivity": qc,
                            "primary_mean_scenario_test_run_balanced_accuracy": score,
                        }
                    )
        paired, summary = select_development_methods(rows, self.config)
        self.assertEqual(len(paired), 40)
        self.assertEqual(summary["selected_B_method"], B_METHODS[1])
        self.assertEqual(summary["selected_C_method"], C_METHODS[1])
        self.assertFalse(summary["selection_used_evaluation_target_CSP_outcomes"])

    def test_subspace_stability_ignores_sign_order_and_basis_rotation(self) -> None:
        rng = np.random.default_rng(4)
        base, _ = np.linalg.qr(rng.normal(size=(12, 4)))
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        equivalent = base @ rotation
        self.assertAlmostEqual(csp_subspace_similarity(base, equivalent), 1.0)
        complement, _ = np.linalg.qr(
            rng.normal(size=(12, 4)) - base @ (base.T @ rng.normal(size=(12, 4)))
        )
        # Construct an exact orthogonal complement for a stable zero-similarity check.
        projection = np.eye(12) - base @ base.T
        complement, _ = np.linalg.qr(projection @ rng.normal(size=(12, 4)))
        self.assertAlmostEqual(csp_subspace_similarity(base, complement), 0.0, places=12)
        row = participant_csp_stability_row(
            21, C_METHODS[0], {6: base, 10: equivalent, 14: base[:, ::-1]}
        )
        self.assertAlmostEqual(row["mean_pairwise_subspace_similarity"], 1.0)
        self.assertTrue(row["sign_order_and_rotation_invariant"])


if __name__ == "__main__":
    unittest.main()
