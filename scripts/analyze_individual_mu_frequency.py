"""Develop and later evaluate the pre-registered individualized-frequency method."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.epoching import extract_run_epochs, load_epoching_config  # noqa: E402
from eeg_project.individual_frequency import (  # noqa: E402
    DEFAULT_INDIVIDUAL_FREQUENCY_CONFIG_PATH,
    estimate_central_peak,
    load_individual_frequency_config,
    rest_roi_log_welch,
    training_runs_for_held_out,
)
from eeg_project.preprocessing import (  # noqa: E402
    DEFAULT_DATA_DIRECTORY,
    load_preprocessing_config,
)


DEFAULT_DEVELOPMENT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "imf_method_development"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Develop the rest-only central peak method without opening held-out "
            "individualized task outcomes."
        )
    )
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument(
        "--develop-method",
        action="store_true",
        help="Analyze rest spectra for the frozen even-ID method-development subjects only.",
    )
    stage.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate odd-ID task outcomes only after a final method freeze permits it.",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_INDIVIDUAL_FREQUENCY_CONFIG_PATH
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DEVELOPMENT_OUTPUT_DIRECTORY
    )
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def method_name(search_range: tuple[float, float], prominence: float) -> str:
    return f"search_{search_range[0]:g}_{search_range[1]:g}Hz_prom_{prominence:g}dB"


def peak_row(
    subject: int,
    estimate_type: str,
    held_out_run: int | str,
    training_runs: Sequence[int],
    search_range: tuple[float, float],
    prominence: float,
    estimate,
    rest_epoch_count: int,
) -> dict[str, object]:
    return {
        "subject": subject,
        "study_role": "method_development",
        "estimate_type": estimate_type,
        "held_out_run": held_out_run,
        "training_runs": "/".join(str(run) for run in training_runs),
        "method": method_name(search_range, prominence),
        "search_low_hz": search_range[0],
        "search_high_hz": search_range[1],
        "minimum_prominence_db": prominence,
        "peak_frequency_hz": "" if estimate.peak_frequency_hz is None else estimate.peak_frequency_hz,
        "candidate_frequency_hz": estimate.candidate_frequency_hz,
        "peak_prominence_db": estimate.peak_prominence_db,
        "peak_width_hz": "" if estimate.peak_width_hz is None else estimate.peak_width_hz,
        "peak_quality": estimate.peak_quality,
        "method_available": estimate.method_available,
        "failure_reason": estimate.failure_reason,
        "rest_epoch_count": rest_epoch_count,
        "central_roi": "C3/Cz/C4_mean_log_psd",
        "frequency_estimation_data": "T0_rest_only",
        "task_outcome_used": False,
    }


def summarize_candidates(rows: Sequence[Mapping[str, object]], subjects: Sequence[int]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [
            row for row in rows
            if row["method"] == method and row["estimate_type"] == "leave_one_run_out"
        ]
        quality = Counter(str(row["peak_quality"]) for row in selected)
        complete_subjects: list[int] = []
        ranges: list[float] = []
        pairwise_differences: list[float] = []
        exact_agreement = 0
        for subject in subjects:
            subject_rows = [row for row in selected if int(row["subject"]) == subject]
            values = [float(row["peak_frequency_hz"]) for row in subject_rows if bool(row["method_available"])]
            if len(values) == 3:
                complete_subjects.append(subject)
                ranges.append(max(values) - min(values))
                pairwise_differences.extend(
                    abs(values[left] - values[right])
                    for left, right in ((0, 1), (0, 2), (1, 2))
                )
                exact_agreement += int(len(set(values)) == 1)
        first = selected[0]
        summaries.append(
            {
                "method": method,
                "search_low_hz": first["search_low_hz"],
                "search_high_hz": first["search_high_hz"],
                "minimum_prominence_db": first["minimum_prominence_db"],
                "development_subject_count": len(subjects),
                "fold_count": len(selected),
                "well_defined_fold_count": quality["well_defined_interior"],
                "boundary_fold_count": quality["boundary_candidate"],
                "no_peak_fold_count": quality["poorly_defined_no_peak"],
                "fold_coverage_fraction": quality["well_defined_interior"] / len(selected),
                "subjects_with_all_three_estimates": len(complete_subjects),
                "complete_subject_coverage_fraction": len(complete_subjects) / len(subjects),
                "complete_subjects_exact_three_fold_agreement": exact_agreement,
                "median_within_subject_peak_range_hz": (
                    float(np.median(ranges)) if ranges else ""
                ),
                "median_pairwise_fold_peak_difference_hz": (
                    float(np.median(pairwise_differences)) if pairwise_differences else ""
                ),
                "task_outcome_used_for_method_selection": False,
            }
        )
    return summaries


def develop_method(args: argparse.Namespace) -> None:
    config = load_individual_frequency_config(args.config)
    if config.study_stage != "partition_and_candidate_method_freeze":
        raise RuntimeError("Development command requires the initial candidate-method freeze.")
    preprocessing = load_preprocessing_config()
    epoching = load_epoching_config()
    rows: list[dict[str, object]] = []
    for index, subject in enumerate(config.development_subjects, start=1):
        run_results = {
            run: extract_run_epochs(
                subject,
                run,
                preprocessing,
                epoching,
                data_directory=args.data_dir,
            )
            for run in config.runs
        }
        spectra: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, int]] = {}
        for run in config.runs:
            result = run_results[run]
            data = result.rest_epochs.get_data(copy=True)
            frequencies, log_power = rest_roi_log_welch(
                data,
                result.rest_epochs.ch_names,
                float(result.rest_epochs.info["sfreq"]),
                config.central_channels,
                config.welch_window_seconds,
            )
            spectra[(run,)] = (frequencies, log_power, data.shape[0])
        for held_out_run in config.runs:
            training_runs = training_runs_for_held_out(held_out_run, config.runs)
            training_data = np.concatenate(
                [run_results[run].rest_epochs.get_data(copy=True) for run in training_runs],
                axis=0,
            )
            reference_epochs = run_results[training_runs[0]].rest_epochs
            frequencies, log_power = rest_roi_log_welch(
                training_data,
                reference_epochs.ch_names,
                float(reference_epochs.info["sfreq"]),
                config.central_channels,
                config.welch_window_seconds,
            )
            spectra[training_runs] = (frequencies, log_power, training_data.shape[0])
        for search_range in config.candidate_search_ranges_hz:
            for prominence in config.candidate_prominence_thresholds_db:
                for run in config.runs:
                    frequencies, log_power, count = spectra[(run,)]
                    estimate = estimate_central_peak(
                        frequencies, log_power, search_range, prominence
                    )
                    rows.append(
                        peak_row(
                            subject, "independent_single_run", "", (run,), search_range,
                            prominence, estimate, count
                        )
                    )
                for held_out_run in config.runs:
                    training_runs = training_runs_for_held_out(held_out_run, config.runs)
                    frequencies, log_power, count = spectra[training_runs]
                    estimate = estimate_central_peak(
                        frequencies, log_power, search_range, prominence
                    )
                    rows.append(
                        peak_row(
                            subject, "leave_one_run_out", held_out_run, training_runs,
                            search_range, prominence, estimate, count
                        )
                    )
        print(f"Development rest spectra: subject {subject:03d} ({index}/{len(config.development_subjects)})")
    summary_rows = summarize_candidates(rows, config.development_subjects)
    write_csv(rows, args.output_dir / "development_peak_candidates.csv")
    write_csv(summary_rows, args.output_dir / "development_candidate_summary.csv")
    print("\nCandidate summary (rest spectra only; no individualized task outcome):")
    for row in summary_rows:
        print(
            f"  {row['method']}: folds {row['well_defined_fold_count']}/{row['fold_count']}; "
            f"complete subjects {row['subjects_with_all_three_estimates']}/{row['development_subject_count']}; "
            f"median fold range {row['median_within_subject_peak_range_hz']} Hz"
        )


def main() -> None:
    args = parse_arguments()
    config = load_individual_frequency_config(args.config)
    if args.evaluate:
        if config.evaluation_locked or config.study_stage != "final_method_frozen":
            raise RuntimeError(
                "Held-out individualized task evaluation is locked until a final method-freeze commit."
            )
        raise RuntimeError("Evaluation implementation is intentionally unavailable before final freeze.")
    develop_method(args)


if __name__ == "__main__":
    main()
