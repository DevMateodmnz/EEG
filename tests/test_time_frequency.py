"""Scientific invariants for paired-rest event-related spectral analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

import numpy as np

from eeg_project.epoching import (
    RunEpochs,
    calculate_trial_quality,
    extract_run_epochs,
    load_epoching_config,
)
from eeg_project.preprocessing import load_preprocessing_config
from eeg_project.time_frequency import (
    SpectralEpochDataset,
    assemble_spectral_epoch_dataset,
    band_frequency_mask,
    compute_morlet_power,
    frequency_vector,
    load_event_related_spectral_config,
    normalize_task_tfr_percent,
    pair_task_with_preceding_rest,
    percent_power_change,
    trial_band_measurements,
    wavelet_cycles,
    wavelet_lengths_samples,
)


class TimeFrequencyUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_event_related_spectral_config()

    def test_frequency_bands_cycles_and_wavelet_support(self) -> None:
        frequencies = frequency_vector(self.config)
        np.testing.assert_array_equal(frequencies, np.arange(6.0, 36.0))
        cycles = wavelet_cycles(frequencies, self.config)
        np.testing.assert_array_equal(cycles, frequencies / 2.0)
        lengths = wavelet_lengths_samples(
            160.0, frequencies, cycles, zero_mean=True
        )
        np.testing.assert_array_equal(lengths, np.full(30, 127))
        bands = {band.name: band for band in self.config.bands}
        self.assertEqual(frequencies[band_frequency_mask(frequencies, bands["mu"])].tolist(),
                         [8, 9, 10, 11, 12, 13])
        self.assertEqual(
            frequencies[band_frequency_mask(frequencies, bands["beta"])].tolist(),
            list(range(14, 31)),
        )
        self.assertFalse(
            np.any(
                band_frequency_mask(frequencies, bands["mu"])
                & band_frequency_mask(frequencies, bands["beta"])
            )
        )

    def test_percent_change_equation(self) -> None:
        reference = np.array([2.0, 4.0, 8.0])
        task = np.array([1.0, 4.0, 12.0])
        np.testing.assert_array_equal(
            percent_power_change(task, reference), np.array([-50.0, 0.0, 50.0])
        )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            percent_power_change(np.ones(2), np.array([1.0, 0.0]))

    def test_synthetic_ten_hz_localization_and_power_reduction(self) -> None:
        task_times = np.arange(-2.0, 4.0 + 1 / 160, 1 / 160)
        rest_times = np.arange(0.0, 4.0 + 1 / 160, 1 / 160)
        task = (0.5 * np.sin(2 * np.pi * 10 * task_times))[None, None, :]
        rest = np.sin(2 * np.pi * 10 * rest_times)[None, None, :]
        task_power = compute_morlet_power(task, task_times, 160.0, self.config)
        rest_power = compute_morlet_power(rest, rest_times, 160.0, self.config)
        task_mask = (task_power.times_seconds >= 1) & (task_power.times_seconds <= 3)
        rest_mask = (rest_power.times_seconds >= 1.1) & (rest_power.times_seconds <= 3.1)
        spectrum = task_power.power_v2[0, 0][:, task_mask].mean(axis=1)
        self.assertEqual(float(task_power.frequencies_hz[np.argmax(spectrum)]), 10.0)
        task_mean = task_power.power_v2[0, 0, :, task_mask].mean()
        rest_mean = rest_power.power_v2[0, 0, :, rest_mask].mean()
        self.assertAlmostEqual(
            float(percent_power_change(task_mean, rest_mean)), -75.0, delta=0.5
        )


class RealDataTimeFrequencyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preprocessing_config = load_preprocessing_config()
        cls.epoching_config = load_epoching_config()
        cls.spectral_config = load_event_related_spectral_config()
        cls.results = {
            run: extract_run_epochs(
                1, run, cls.preprocessing_config, cls.epoching_config
            )
            for run in (6, 10, 14)
        }
        cls.dataset = assemble_spectral_epoch_dataset(cls.results, (6, 10, 14))
        cls.sensor_indices = [cls.dataset.channel_names.index(c) for c in ("C3", "Cz", "C4")]
        cls.sensor_dataset = SpectralEpochDataset(
            task_data_volts=cls.dataset.task_data_volts[:, cls.sensor_indices].copy(),
            rest_data_volts=cls.dataset.rest_data_volts[:, cls.sensor_indices].copy(),
            task_times=cls.dataset.task_times.copy(),
            rest_times=cls.dataset.rest_times.copy(),
            channel_names=["C3", "Cz", "C4"],
            sampling_frequency_hz=cls.dataset.sampling_frequency_hz,
            pairs=cls.dataset.pairs.copy(),
        )
        cls.task_power = compute_morlet_power(
            cls.sensor_dataset.task_data_volts,
            cls.sensor_dataset.task_times,
            160.0,
            cls.spectral_config,
        )
        cls.rest_power = compute_morlet_power(
            cls.sensor_dataset.rest_data_volts,
            cls.sensor_dataset.rest_times,
            160.0,
            cls.spectral_config,
        )
        records = [pair.task for pair in cls.dataset.pairs]
        cls.quality_rows = calculate_trial_quality(
            cls.dataset.task_data_volts,
            cls.dataset.task_times,
            cls.dataset.channel_names,
            records,
            cls.epoching_config.trial_quality,
        )
        cls.quality_by_identity = {
            (int(row["run"]), int(row["run_trial_index"])): row
            for row in cls.quality_rows
        }

    def test_task_rest_pairing_is_exact(self) -> None:
        self.assertEqual(len(self.dataset.pairs), 45)
        for run, result in self.results.items():
            pairs = pair_task_with_preceding_rest(result)
            self.assertEqual(len(pairs), 15)
            for pair in pairs:
                self.assertEqual(pair.rest.annotation_index + 1, pair.task.annotation_index)
                self.assertEqual(pair.rest.run_trial_index, pair.task.run_trial_index)
                self.assertAlmostEqual(
                    pair.rest.event_time_seconds + 4.2,
                    pair.task.event_time_seconds,
                    places=10,
                    msg=f"Run {run} has a mispaired rest interval.",
                )

    def test_real_tfr_coordinates_dimensions_and_finiteness(self) -> None:
        self.assertEqual(self.task_power.power_v2.shape, (45, 3, 30, 241))
        self.assertEqual(self.rest_power.power_v2.shape, (45, 3, 30, 161))
        self.assertEqual(self.task_power.times_seconds[0], -2.0)
        self.assertEqual(self.task_power.times_seconds[-1], 4.0)
        self.assertAlmostEqual(float(np.diff(self.task_power.times_seconds).mean()), 0.025)
        self.assertTrue(np.isfinite(self.task_power.power_v2).all())
        self.assertTrue(np.isfinite(self.rest_power.power_v2).all())
        self.assertTrue(np.all(self.task_power.power_v2 >= 0))

    def test_trial_rows_preserve_labels_runs_qc_and_normalization(self) -> None:
        rows = trial_band_measurements(
            self.task_power,
            self.rest_power,
            self.sensor_dataset,
            self.spectral_config,
            self.quality_by_identity,
        )
        self.assertEqual(len(rows), 45 * 3 * 3)
        identities = {
            (row["run"], row["run_trial_index"], row["semantic_condition"])
            for row in rows
        }
        self.assertEqual(len(identities), 45)
        condition_counts = Counter(pair.task.semantic_condition for pair in self.dataset.pairs)
        self.assertEqual(condition_counts, {"both_fists_imagery": 21, "both_feet_imagery": 24})
        candidate_identities = {
            (row["run"], row["run_trial_index"])
            for row in rows if row["quality_status"] == "statistical candidate"
        }
        self.assertEqual(len(candidate_identities), 6)
        for row in rows[::67]:
            expected = 100 * (
                float(row["task_mean_wavelet_power_v2"])
                - float(row["reference_mean_wavelet_power_v2"])
            ) / float(row["reference_mean_wavelet_power_v2"])
            self.assertAlmostEqual(float(row["change_percent"]), expected, places=12)

    def test_qc_sensitivity_does_not_mutate_primary_epochs(self) -> None:
        snapshot = self.sensor_dataset.task_data_volts.copy()
        normalized = normalize_task_tfr_percent(
            self.task_power,
            self.rest_power,
            self.spectral_config.paired_rest_reference_interval_seconds,
        )
        candidate_mask = np.array(
            [row["quality_status"] == "statistical candidate" for row in self.quality_rows]
        )
        sensitivity_copy = normalized[~candidate_mask].copy()
        self.assertEqual(sensitivity_copy.shape[0], 39)
        np.testing.assert_array_equal(self.sensor_dataset.task_data_volts, snapshot)
        self.assertEqual(self.sensor_dataset.task_data_volts.shape[0], 45)

    def test_deterministically_invalid_task_is_excluded_by_provenance(self) -> None:
        original = self.results[6]
        invalid_index = 3
        task_records = [
            replace(
                record,
                valid=False,
                exclusion_reason="synthetic_boundary_test",
            )
            if index == invalid_index
            else record
            for index, record in enumerate(original.task_records)
        ]
        retained_indices = [index for index in range(15) if index != invalid_index]
        modified = RunEpochs(
            preprocessing=original.preprocessing,
            task_epochs=original.task_epochs[retained_indices],
            rest_epochs=original.rest_epochs,
            annotation_rows=original.annotation_rows,
            task_records=task_records,
            rest_records=original.rest_records,
            event_id=original.event_id,
        )

        dataset = assemble_spectral_epoch_dataset({6: modified}, (6,))

        self.assertEqual(dataset.task_data_volts.shape, (14, 64, 961))
        self.assertEqual(dataset.rest_data_volts.shape, (14, 64, 641))
        self.assertNotIn(invalid_index, [pair.task.run_trial_index for pair in dataset.pairs])
        self.assertEqual(
            [pair.task_array_index for pair in dataset.pairs], list(range(14))
        )
        self.assertEqual(
            [pair.rest_array_index for pair in dataset.pairs], list(range(14))
        )

    def test_task_with_deterministically_invalid_rest_is_excluded(self) -> None:
        original = self.results[6]
        invalid_index = 3
        rest_records = [
            replace(
                record,
                valid=False,
                exclusion_reason="synthetic_boundary_test",
            )
            if index == invalid_index
            else record
            for index, record in enumerate(original.rest_records)
        ]
        retained_indices = [index for index in range(15) if index != invalid_index]
        modified = RunEpochs(
            preprocessing=original.preprocessing,
            task_epochs=original.task_epochs,
            rest_epochs=original.rest_epochs[retained_indices],
            annotation_rows=original.annotation_rows,
            task_records=original.task_records,
            rest_records=rest_records,
            event_id=original.event_id,
        )

        dataset = assemble_spectral_epoch_dataset({6: modified}, (6,))

        self.assertEqual(dataset.task_data_volts.shape, (14, 64, 961))
        self.assertEqual(dataset.rest_data_volts.shape, (14, 64, 641))
        self.assertNotIn(
            invalid_index, [pair.task.run_trial_index for pair in dataset.pairs]
        )
        self.assertEqual(
            [pair.task_array_index for pair in dataset.pairs], list(range(14))
        )
        self.assertEqual(
            [pair.rest_array_index for pair in dataset.pairs], list(range(14))
        )

    def test_pairing_uses_annotation_order_not_rounded_duration_equality(self) -> None:
        original = self.results[6]
        rest_records = [
            replace(
                record,
                annotation_duration_seconds=(
                    record.annotation_duration_seconds + 0.05
                    if index == 3
                    else record.annotation_duration_seconds
                ),
            )
            for index, record in enumerate(original.rest_records)
        ]
        modified = RunEpochs(
            preprocessing=original.preprocessing,
            task_epochs=original.task_epochs,
            rest_epochs=original.rest_epochs,
            annotation_rows=original.annotation_rows,
            task_records=original.task_records,
            rest_records=rest_records,
            event_id=original.event_id,
        )

        dataset = assemble_spectral_epoch_dataset({6: modified}, (6,))

        self.assertEqual(len(dataset.pairs), 15)
        self.assertEqual(dataset.pairs[3].rest.annotation_index + 1, dataset.pairs[3].task.annotation_index)


if __name__ == "__main__":
    unittest.main()
