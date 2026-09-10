"""Develop one minimal target-calibration method using subjects 1–20 only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import mne


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.cross_subject_decoding import (  # noqa: E402
    file_sha256,
    fit_cross_subject_model,
    load_cross_subject_data,
)
from eeg_project.decoding import RUNS, load_decoding_config  # noqa: E402
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY  # noqa: E402
from eeg_project.target_calibration import (  # noqa: E402
    CALIBRATION_METHODS,
    CALIBRATION_SIZES,
    DEFAULT_CALIBRATION_CONFIG_PATH,
    INITIAL_CALIBRATION_FREEZE_COMMIT,
    calibration_curve_summary,
    development_method_comparison,
    evaluate_calibration_scenario,
    load_calibration_config,
    participant_calibration_rows,
    transform_target_with_source,
    validate_nested_calibration_subsets,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
PREFIX = "target_calibration_development_"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two frozen decision-layer calibration methods on subjects 1–20."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CALIBRATION_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty calibration artifact: {path}.")
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


def verify_initial_freeze_is_ancestor() -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", INITIAL_CALIBRATION_FREEZE_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("The initial calibration freeze is not an ancestor of HEAD.")


def main() -> None:
    args = parse_arguments()
    mne.set_log_level("ERROR")
    verify_initial_freeze_is_ancestor()
    config = load_calibration_config(args.config)
    if not config["evaluation_locked"]:
        raise RuntimeError("Development requires evaluation calibration outcomes locked.")
    within_config = load_decoding_config()
    cross_config = json.loads(
        (PROJECT_ROOT / "config" / "cross_subject_decoding.json").read_text(
            encoding="utf-8"
        )
    )
    subjects = list(config["cohorts"]["development_subjects"])
    datasets = {
        subject: load_cross_subject_data(
            subject, cross_config, data_directory=args.data_dir
        )
        for subject in subjects
    }
    if any(dataset.labels.size != 45 for dataset in datasets.values()):
        raise RuntimeError("Calibration development requires 45 trials per participant.")
    for dataset in datasets.values():
        validate_nested_calibration_subsets(dataset)

    prediction_rows: list[dict[str, object]] = []
    test_run_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for fold_index, target_subject in enumerate(subjects, start=1):
        training_subjects = tuple(
            subject for subject in subjects if subject != target_subject
        )
        fitted = fit_cross_subject_model(
            [datasets[subject] for subject in training_subjects],
            "csp_4_empirical",
            within_config,
        )
        source = transform_target_with_source(fitted, datasets[target_subject])
        for method in CALIBRATION_METHODS:
            for size in CALIBRATION_SIZES:
                for calibration_run in RUNS:
                    result = evaluate_calibration_scenario(
                        source,
                        training_subjects,
                        calibration_run,
                        size,
                        method,
                    )
                    prefix = {"development_fold": fold_index}
                    prediction_rows.extend(
                        {**prefix, **row} for row in result.prediction_rows
                    )
                    test_run_rows.extend(
                        {**prefix, **row} for row in result.test_run_rows
                    )
                    audit_rows.append({**prefix, **result.audit_row})
        print(
            f"Calibration development: held out S{target_subject:03d} "
            f"({fold_index}/{len(subjects)})"
        )

    participant_rows = participant_calibration_rows(
        test_run_rows,
        prediction_rows,
        expected_subjects=subjects,
        methods=CALIBRATION_METHODS,
    )
    curve_rows = []
    for method in CALIBRATION_METHODS:
        curve_rows.extend(
            calibration_curve_summary(
                participant_rows,
                method,
                config,
                expected_subject_count=20,
            )
        )
    paired_rows, selection_summary = development_method_comparison(
        participant_rows, config
    )

    historical = {
        int(row["subject"]): float(row["primary_mean_run_balanced_accuracy"])
        for row in _read_csv(
            args.output_dir / "cross_subject_decoder_development_subject_scores.csv"
        )
        if row["model"] == "csp_4_empirical"
    }
    observed_zero = {
        int(row["subject"]): float(row["primary_mean_test_run_balanced_accuracy"])
        for row in participant_rows
        if row["method"] == "threshold"
        and int(row["calibration_size"]) == 0
        and not bool(row["qc_sensitivity"])
    }
    if historical != observed_zero:
        raise RuntimeError("Size-zero development does not reproduce frozen zero-shot CSP.")

    artifact_rows = {
        "predictions.csv": prediction_rows,
        "test_run_scores.csv": test_run_rows,
        "participant_scores.csv": participant_rows,
        "fit_audit.csv": audit_rows,
        "curve_summary.csv": curve_rows,
        "method_comparison_subjects.csv": paired_rows,
        "method_selection.csv": [selection_summary],
    }
    artifact_hashes: dict[str, str] = {}
    for suffix, rows in artifact_rows.items():
        path = args.output_dir / f"{PREFIX}{suffix}"
        write_csv(rows, path)
        artifact_hashes[path.name] = file_sha256(path)

    source_hashes = {
        identity.source_file: identity.source_sha256
        for dataset in datasets.values()
        for identity in dataset.identities
    }
    metadata = {
        "schema_version": 1,
        "study_stage": "minimal_target_calibration_development_complete",
        "initial_calibration_freeze_git_commit": INITIAL_CALIBRATION_FREEZE_COMMIT,
        "calibration_config_sha256": file_sha256(args.config.resolve()),
        "calibration_implementation_sha256": file_sha256(
            PROJECT_ROOT / "eeg_project" / "target_calibration.py"
        ),
        "development_subjects": subjects,
        "development_subject_count": 20,
        "source_training_subject_count_per_fold": 19,
        "source_training_trial_count_per_fold": 855,
        "calibration_sizes": list(CALIBRATION_SIZES),
        "candidate_methods": list(CALIBRATION_METHODS),
        "selected_method": selection_summary["selected_method"],
        "evaluation_subject_EEG_loaded": False,
        "evaluation_calibration_outcomes_calculated": False,
        "target_CSP_or_scaler_refit": False,
        "one_value_per_participant_per_method_and_size": True,
        "source_hashes": dict(sorted(source_hashes.items())),
        "artifact_sha256": artifact_hashes,
        "full_feature_arrays_persisted": False,
        "fitted_models_persisted": False,
    }
    metadata_path = args.output_dir / f"{PREFIX}metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nCalibration development summary:")
    for method in CALIBRATION_METHODS:
        print(f"  {method}:")
        for row in curve_rows:
            if row["method"] == method:
                print(
                    f"    n={row['calibration_size']:>2}: "
                    f"median={row['median_balanced_accuracy']:.3f}; "
                    f"gain={row['median_gain_over_matched_zero']:+.3f}"
                )
    print(
        "  target-LDA minus threshold utility median="
        f"{selection_summary['median_target_LDA_minus_threshold_utility']:+.3f}; "
        f"selected={selection_summary['selected_method']}"
    )
    print("  Evaluation calibration outcomes remain locked.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


if __name__ == "__main__":
    main()
