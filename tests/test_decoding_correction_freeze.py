"""Guards for the outcome-independent retained-trial validation correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH = PROJECT_ROOT / "config" / "within_subject_decoding_correction.json"


class DecoderCorrectionFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(PATH.read_text(encoding="utf-8"))

    def test_correction_precedes_restarted_evaluation(self) -> None:
        self.assertEqual(
            self.payload["study_stage"], "technical_validation_correction_frozen"
        )
        attempt = self.payload["evaluation_attempt_before_correction"]
        self.assertEqual(attempt["subjects_with_compact_checkpoints"], list(range(21, 34)))
        self.assertFalse(attempt["cohort_summary_calculated"])
        self.assertFalse(attempt["subject_accuracy_inspected_for_method_revision"])
        self.assertEqual(attempt["failure_subject"], 34)
        self.assertFalse(
            self.payload["evaluation_classifier_outcomes_inspected_at_correction_freeze"]
        )

    def test_only_retained_trial_count_validation_changes(self) -> None:
        correction = self.payload["correction"]
        self.assertEqual(
            correction["historically_verified_42_trial_evaluation_subjects"],
            [34, 37, 41, 51, 64, 72, 73, 74, 76, 102],
        )
        self.assertEqual(correction["ordinary_45_trial_evaluation_subject_count"], 76)
        self.assertEqual(correction["retained_42_trial_evaluation_subject_count"], 10)
        for key, value in correction.items():
            if key.endswith("_changed"):
                self.assertFalse(value, key)
        self.assertFalse(self.payload["model_or_metric_setting_changed"])
        self.assertFalse(self.payload["evaluation_outcomes_used_for_correction"])

    def test_parent_policy_and_corrected_code_are_byte_exact(self) -> None:
        for key in (
            "parent_final_policy",
            "corrected_evaluation_implementation",
            "correction_tests",
        ):
            record = self.payload[key]
            path = PROJECT_ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
