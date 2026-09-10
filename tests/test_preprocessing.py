"""Focused numerical tests for reusable preprocessing operations."""

from __future__ import annotations

import unittest

import mne
import numpy as np

from eeg_project.preprocessing import (
    apply_analysis_reference,
    apply_production_filter,
    design_production_filter,
    find_valid_stop,
    load_preprocessing_config,
)


class PreprocessingTests(unittest.TestCase):
    """Check scientific invariants without requiring downloaded EEG files."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_preprocessing_config()

    def test_configuration_and_filter_response(self) -> None:
        coefficients = design_production_filter(160.0, self.config)
        self.assertEqual(coefficients.size, 529)
        np.testing.assert_allclose(coefficients, coefficients[::-1], rtol=0, atol=1e-15)
        self.assertEqual(self.config.notch_frequencies_hz, ())

        frequencies = np.fft.rfftfreq(262_144, d=1 / 160.0)
        response = np.abs(np.fft.rfft(coefficients, 262_144))

        def gain_db(frequency: float) -> float:
            index = int(np.argmin(np.abs(frequencies - frequency)))
            return float(20 * np.log10(response[index]))

        self.assertLess(gain_db(0.25), -15)
        self.assertGreater(gain_db(8), -0.1)
        self.assertGreater(gain_db(30), -0.1)
        self.assertLess(gain_db(50), -40)
        self.assertLess(gain_db(60), -55)

    def test_zero_tail_detection(self) -> None:
        data = np.ones((4, 100))
        data[:, 92:] = 0
        valid_stop, tail_samples, earlier_zeros = find_valid_stop(data)
        self.assertEqual((valid_stop, tail_samples, earlier_zeros), (92, 8, 0))
        data[:, 40] = 0
        self.assertEqual(find_valid_stop(data), (92, 8, 1))

        no_tail = np.arange(400, dtype=float).reshape(4, 100) + 1
        self.assertEqual(find_valid_stop(no_tail), (100, 0, 0))

    def test_reference_filter_determinism_and_commutation(self) -> None:
        sfreq = 160.0
        samples = np.arange(3200)
        data = np.vstack(
            [
                np.sin(2 * np.pi * frequency * samples / sfreq)
                + 0.25 * np.sin(2 * np.pi * 60 * samples / sfreq)
                for frequency in (8, 10, 12, 20)
            ]
        ) * 1e-6
        info = mne.create_info(["A", "B", "C", "D"], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose="error")
        raw.set_annotations(mne.Annotations([5.0], [2.0], ["test"]))

        referenced = apply_analysis_reference(raw)
        first = apply_production_filter(referenced, self.config)
        second = apply_production_filter(referenced, self.config)
        np.testing.assert_array_equal(first.get_data(), second.get_data())
        self.assertEqual(first.annotations, raw.annotations)
        self.assertEqual(first.n_times, raw.n_times)
        self.assertEqual(first.info["sfreq"], raw.info["sfreq"])

        filtered_first = apply_production_filter(raw, self.config)
        reference_second = apply_analysis_reference(filtered_first)
        np.testing.assert_allclose(
            first.get_data(), reference_second.get_data(), rtol=1e-12, atol=1e-15
        )


if __name__ == "__main__":
    unittest.main()
