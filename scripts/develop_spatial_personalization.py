"""Develop decision-layer and target-CSP personalization on subjects 1–20."""

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
    load_cross_subject_config,
    load_cross_subject_data,
)
from eeg_project.decoding import RUNS, load_decoding_config  # noqa: E402
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY  # noqa: E402
from eeg_project.spatial_personalization import (  # noqa: E402
    A_METHOD,
    C_METHODS,
    DEVELOPMENT_METHODS,
    DEFAULT_SPATIAL_PERSONALIZATION_CONFIG_PATH,
    INITIAL_SPATIAL_PERSONALIZATION_FREEZE_COMMIT,
    evaluate_personalization_scenario,
    load_spatial_personalization_config,
    participant_csp_stability_row,
    participant_personalization_rows,
    personalization_group_summary,
    select_development_methods,
)
from eeg_project.target_calibration import (  # noqa: E402
    transform_target_with_source,
    validate_nested_calibration_subsets,
)


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
PREFIX = "spatial_personalization_development_"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop frozen decision-layer and target-CSP personalization."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_SPATIAL_PERSONALIZATION_CONFIG_PATH
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty personalization artifact: {path}.")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def verify_initial_freeze_is_ancestor() -> None:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            INITIAL_SPATIAL_PERSONALIZATION_FREEZE_COMMIT,
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("The initial spatial-personalization freeze is not an ancestor.")


def main() -> None:
    args = parse_arguments()
    mne.set_log_level("ERROR")
    verify_initial_freeze_is_ancestor()
    config = load_spatial_personalization_config(args.config)
    if not config["evaluation_locked"]:
        raise RuntimeError("Development requires evaluation target-CSP outcomes locked.")
    cross_config = load_cross_subject_config()
    within_config = load_decoding_config()
    subjects = list(config["cohorts"]["development_subjects"])
    datasets = {
        subject: load_cross_subject_data(
            subject, cross_config, data_directory=args.data_dir
        )
        for subject in subjects
    }
    if any(dataset.labels.size != 45 for dataset in datasets.values()):
        raise RuntimeError("Personalization development requires 45 trials per subject.")
    for dataset in datasets.values():
        validate_nested_calibration_subsets(dataset)

    prediction_rows: list[dict[str, object]] = []
    test_run_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
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
        subspaces: dict[str, dict[int, object]] = {method: {} for method in C_METHODS}
        for method in DEVELOPMENT_METHODS:
            for calibration_run in RUNS:
                result = evaluate_personalization_scenario(
                    datasets[target_subject],
                    source,
                    training_subjects,
                    calibration_run,
                    method,
                    config,
                )
                prefix = {"development_fold": fold_index}
                prediction_rows.extend({**prefix, **row} for row in result.prediction_rows)
                test_run_rows.extend({**prefix, **row} for row in result.test_run_rows)
                audit_rows.append({**prefix, **result.audit_row})
                if result.csp_filter_subspace is not None:
                    subspaces[method][calibration_run] = result.csp_filter_subspace
        for method in C_METHODS:
            stability_rows.append(
                participant_csp_stability_row(
                    target_subject, method, subspaces[method]
                )
            )
        print(
            f"Spatial-personalization development: held out S{target_subject:03d} "
            f"({fold_index}/{len(subjects)})"
        )

    participant_rows = participant_personalization_rows(
        test_run_rows,
        prediction_rows,
        expected_subjects=subjects,
        methods=DEVELOPMENT_METHODS,
    )
    group_rows = personalization_group_summary(
        participant_rows,
        DEVELOPMENT_METHODS,
        config,
        expected_subject_count=20,
    )
    selection_subject_rows, selection_summary = select_development_methods(
        participant_rows, config
    )

    historical = {
        int(row["subject"]): float(row["primary_mean_run_balanced_accuracy"])
        for row in read_csv(
            args.output_dir / "cross_subject_decoder_development_subject_scores.csv"
        )
        if row["model"] == "csp_4_empirical"
    }
    observed_a = {
        int(row["subject"]): float(
            row["primary_mean_scenario_test_run_balanced_accuracy"]
        )
        for row in participant_rows
        if row["method"] == A_METHOD and not bool(row["qc_sensitivity"])
    }
    if historical != observed_a:
        raise RuntimeError("Development A does not reproduce frozen zero-shot CSP.")

    artifact_rows = {
        "predictions.csv": prediction_rows,
        "test_run_scores.csv": test_run_rows,
        "participant_scores.csv": participant_rows,
        "fit_audit.csv": audit_rows,
        "group_summary.csv": group_rows,
        "method_selection_subjects.csv": selection_subject_rows,
        "method_selection.csv": [selection_summary],
        "csp_stability.csv": stability_rows,
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
        "study_stage": "spatial_personalization_development_complete",
        "initial_freeze_git_commit": INITIAL_SPATIAL_PERSONALIZATION_FREEZE_COMMIT,
        "config_sha256": file_sha256(args.config.resolve()),
        "implementation_sha256": file_sha256(
            PROJECT_ROOT / "eeg_project" / "spatial_personalization.py"
        ),
        "development_subjects": subjects,
        "development_subject_count": 20,
        "source_training_subject_count_per_fold": 19,
        "source_training_trial_count_per_fold": 855,
        "calibration_trial_count": 14,
        "candidate_methods": list(DEVELOPMENT_METHODS),
        "selected_B_method": selection_summary["selected_B_method"],
        "selected_C_method": selection_summary["selected_C_method"],
        "evaluation_target_CSP_fitted": False,
        "evaluation_target_CSP_outcomes_calculated": False,
        "one_value_per_participant_per_method": True,
        "same_test_trials_compared_across_methods": True,
        "test_EEG_used_for_any_fit": False,
        "source_hashes": dict(sorted(source_hashes.items())),
        "artifact_sha256": artifact_hashes,
        "fitted_models_persisted": False,
    }
    metadata_path = args.output_dir / f"{PREFIX}metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nSpatial-personalization development summary:")
    for row in group_rows:
        print(
            f"  {row['method']}: median={row['median_balanced_accuracy']:.3f}; "
            f"IQR={row['q25_balanced_accuracy']:.3f}–{row['q75_balanced_accuracy']:.3f}"
        )
    print(f"  Selected B: {selection_summary['selected_B_method']}")
    print(f"  Selected C: {selection_summary['selected_C_method']}")


if __name__ == "__main__":
    main()
