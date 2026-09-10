"""RED-first guards for Lee2019 external cross-session replication."""

from __future__ import annotations

import unittest
import json
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from eeg_project.external_cross_session_lee2019 import (
    ExternalSubjectData,
    ExternalTrialIdentity,
    LEE_LABELS,
    SESSION_IDS,
    SessionTrials,
    cross_session_folds,
    epoch_sample_bounds,
    evaluate_subject,
    load_study_config,
    lee2019_nemar_provider,
    lee2019_nemar_sources,
    participant_mean_score,
    validate_channel_names,
    validate_fit_audits,
    validate_formal_run,
    validate_labels,
    validate_source_only_fit,
)
from eeg_project.unilateral_generalization import load_parent_decoder_config


class ExternalLee2019FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_study_config()

    def test_identity_sessions_labels_and_online_policy_are_exact(self) -> None:
        dataset = self.config["external_dataset"]
        self.assertEqual(self.config["classification"], "EXTERNAL_CROSS_SESSION_REPLICATION")
        self.assertEqual(dataset["adapter"], "moabb.datasets.Lee2019_MI")
        self.assertEqual(dataset["candidate_subjects"], list(range(1, 55)))
        self.assertEqual(tuple(dataset["sessions"]), SESSION_IDS)
        self.assertEqual(dataset["formal_run"], "1train")
        self.assertTrue(dataset["online_test_run_forbidden"])
        self.assertEqual(dataset["download_provider"], "nemar")
        self.assertEqual(dataset["nemar_deposit"], "nm000338")
        self.assertEqual(LEE_LABELS, {"left_hand": 0, "right_hand": 1})

    def test_only_labelled_training_run_is_accepted(self) -> None:
        validate_formal_run("1train")
        for invalid in ("4test", "0preTrainRest", "ERP", "SSVEP"):
            with self.assertRaises(ValueError):
                validate_formal_run(invalid)

    def test_exactly_two_disjoint_session_directions_exist(self) -> None:
        folds = cross_session_folds()
        self.assertEqual([(fold.train_session, fold.test_session) for fold in folds], [("1", "2"), ("2", "1")])
        self.assertTrue(all(fold.train_session != fold.test_session for fold in folds))

    def test_epoch_is_two_seconds_after_actual_cue_at_native_rate(self) -> None:
        self.assertEqual(epoch_sample_bounds(1000.0), (1000, 3000))
        self.assertEqual(epoch_sample_bounds(500.0), (500, 1500))
        self.assertEqual(self.config["dataset_adapter"]["task_interval_seconds_after_cue"], [1.0, 3.0])

    def test_only_eeg_and_frozen_baseline_channels_are_valid(self) -> None:
        validate_channel_names(["C3", "Cz", "C4", "FC1"], required=("C3", "Cz", "C4"))
        with self.assertRaisesRegex(ValueError, "Cz"):
            validate_channel_names(["C3", "C4"], required=("C3", "Cz", "C4"))

    def test_class_mapping_and_balance_are_deterministic(self) -> None:
        self.assertTrue(np.array_equal(validate_labels(["left_hand"] * 50 + ["right_hand"] * 50), np.array([0] * 50 + [1] * 50)))
        with self.assertRaises(ValueError):
            validate_labels(["left_hand"] * 100)
        with self.assertRaises(ValueError):
            validate_labels(["online_test"] * 2)

    def test_fit_audit_rejects_target_or_overlap_and_participant_mean_is_correct(self) -> None:
        validate_source_only_fit("1", "2", ["S01_S1_T001"], ["S01_S2_T001"])
        with self.assertRaises(ValueError):
            validate_source_only_fit("1", "1", ["S01_S1_T001"], ["S01_S1_T001"])
        self.assertEqual(participant_mean_score([0.4, 0.6]), 0.5)
        with self.assertRaises(ValueError):
            participant_mean_score([0.6])

    def test_synthetic_cross_session_pipeline_has_two_source_only_directions(self) -> None:
        rng = np.random.default_rng(3)
        labels = np.array([0, 1] * 4)
        channels = ["C3", "Cz", "C4"] + [f"X{index:02d}" for index in range(59)]
        sessions = {}
        for session in SESSION_IDS:
            data = rng.normal(size=(8, 62, 40)) * 1e-6
            data[labels == 1, :4] += 0.5e-6
            identities = [ExternalTrialIdentity(1, session, index, f"/tmp/s{session}.mat", "a" * 64) for index in range(8)]
            sessions[session] = SessionTrials(data, data.copy(), labels, identities, channels, 20.0)
        subject = ExternalSubjectData(1, sessions, [])
        unilateral = json.loads((Path(__file__).resolve().parents[1] / "config/unilateral_motor_imagery_generalization.json").read_text())
        parent = load_parent_decoder_config(unilateral)
        predictions, directions, audits, summary = evaluate_subject(subject, parent, self.config, "csp_4_empirical")
        self.assertEqual(len(predictions), 16)
        self.assertEqual(len(directions), len(audits), 2)
        self.assertEqual(summary["group_observational_unit"], "participant")
        for audit in audits:
            self.assertEqual(audit["source_session_only"], True)
            self.assertTrue(set(audit["train_trial_keys"].split("/")).isdisjoint(audit["test_trial_keys"].split("/")))

    def test_nemar_source_manifest_requires_complete_hash_valid_original_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_root = root / "NEMAR" / "nm000338" / "sourcedata"
            records = []
            for session, content in (("session1", b"session one"), ("session2", b"session two")):
                filename = f"{session}/s1/test_{session}.mat"
                source = manifest_root / filename
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(content)
                records.append({"file": filename, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "subject": "1"})
            (manifest_root / "sourcedata_provenance.json").write_text(json.dumps({"files": records}), encoding="utf-8")
            validated = lee2019_nemar_sources(1, root)
            self.assertEqual([record["session"] for record in validated], ["1", "2"])
            self.assertEqual([record["provider"] for record in validated], ["MOABB_NEMAR_sourcedata"] * 2)
            self.assertTrue(all(not Path(str(record["source_file"])).is_absolute() for record in validated))
            (manifest_root / "session2" / "s1" / "test_session2.mat").unlink()
            with self.assertRaises(FileNotFoundError):
                lee2019_nemar_sources(1, root)

    def test_nemar_provider_context_is_local_to_the_command(self) -> None:
        original = os.environ.get("MOABB_DOWNLOAD_PROVIDER")
        with tempfile.TemporaryDirectory() as temporary:
            with lee2019_nemar_provider(Path(temporary)):
                self.assertEqual(os.environ["MOABB_DOWNLOAD_PROVIDER"], "nemar")
                self.assertEqual(os.environ["MNE_DATASETS_LEE2019-MI_PATH"], str(Path(temporary).resolve()))
        self.assertEqual(os.environ.get("MOABB_DOWNLOAD_PROVIDER"), original)

    def test_fit_audit_validator_rejects_any_target_fit_provenance(self) -> None:
        audit = {
            "subject": 1, "model": "csp_4_empirical", "train_session": "1", "test_session": "2",
            "train_trial_keys": "S01_S1_T000/S01_S1_T001", "test_trial_keys": "S01_S2_T000/S01_S2_T001",
            "csp_fit_trial_keys": "S01_S1_T000/S01_S1_T001", "scaler_fit_trial_keys": "S01_S1_T000/S01_S1_T001",
            "lda_fit_trial_keys": "S01_S1_T000/S01_S1_T001", "source_session_only": True,
        }
        validate_fit_audits([audit])
        audit["lda_fit_trial_keys"] = "S01_S2_T000/S01_S2_T001"
        with self.assertRaises(ValueError):
            validate_fit_audits([audit])


if __name__ == "__main__":
    unittest.main()
