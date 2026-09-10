"""Run participant-isolated cross-subject development on subjects 1–20 only."""

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
    DEFAULT_CROSS_SUBJECT_CONFIG_PATH,
    INITIAL_CROSS_SUBJECT_FREEZE_COMMIT,
    MODEL_NAMES,
    file_sha256,
    fit_cross_subject_model,
    leave_one_subject_out_folds,
    load_cross_subject_config,
    load_cross_subject_data,
    paired_cross_subject_models,
    predict_cross_subject_target,
    summarize_cross_subject_group,
)
from eeg_project.decoding import load_decoding_config  # noqa: E402
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY  # noqa: E402


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
PREFIX = "cross_subject_decoder_development_"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop the frozen zero-shot decoder with subjects 1–20 LOSO only."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CROSS_SUBJECT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write deterministic field-order CSV."""
    if not rows:
        raise ValueError(f"Cannot write an empty development artifact: {path}.")
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
    """Refuse development without the committed pre-outcome protocol."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", INITIAL_CROSS_SUBJECT_FREEZE_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("The initial cross-subject freeze is not an ancestor of HEAD.")


def main() -> None:
    args = parse_arguments()
    mne.set_log_level("ERROR")
    verify_initial_freeze_is_ancestor()
    cross_config = load_cross_subject_config(args.config)
    if not cross_config["evaluation_locked"]:
        raise RuntimeError("Development requires the evaluation cohort to remain locked.")
    subjects = list(cross_config["cohorts"]["development_subjects"])
    folds = leave_one_subject_out_folds(subjects)
    within_config = load_decoding_config()
    datasets = {
        subject: load_cross_subject_data(
            subject, cross_config, data_directory=args.data_dir
        )
        for subject in subjects
    }
    if any(dataset.labels.size != 45 for dataset in datasets.values()):
        raise RuntimeError("Frozen equal-weight development expects 45 trials per participant.")

    prediction_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds, start=1):
        training = [datasets[subject] for subject in fold.training_subjects]
        target = datasets[fold.test_subject]
        for model_name in MODEL_NAMES:
            fitted = fit_cross_subject_model(training, model_name, within_config)
            result = predict_cross_subject_target(fitted, target, within_config)
            prediction_rows.extend(
                {"development_fold": fold_index, **row} for row in result.prediction_rows
            )
            run_rows.extend(
                {"development_fold": fold_index, **row} for row in result.run_rows
            )
            subject_rows.append(
                {"development_fold": fold_index, **result.subject_row}
            )
            audit_rows.append(
                {"development_fold": fold_index, **result.fit_audit_row}
            )
        print(
            f"Cross-subject development: held out S{fold.test_subject:03d} "
            f"({fold_index}/{len(folds)})"
        )

    group_rows = [
        summarize_cross_subject_group(
            subject_rows,
            model_name,
            cross_config,
            expected_subject_count=20,
        )
        for model_name in MODEL_NAMES
    ]
    paired_rows, paired_summary = paired_cross_subject_models(subject_rows)

    output = args.output_dir
    artifact_rows = {
        "predictions.csv": prediction_rows,
        "run_scores.csv": run_rows,
        "subject_scores.csv": subject_rows,
        "fit_audit.csv": audit_rows,
        "model_summary.csv": group_rows,
        "paired_subjects.csv": paired_rows,
        "paired_summary.csv": [paired_summary],
    }
    artifact_hashes: dict[str, str] = {}
    for suffix, rows in artifact_rows.items():
        path = output / f"{PREFIX}{suffix}"
        write_csv(rows, path)
        artifact_hashes[path.name] = file_sha256(path)

    source_hashes: dict[str, str] = {}
    for dataset in datasets.values():
        for identity in dataset.identities:
            source_hashes[identity.source_file] = identity.source_sha256
    metadata = {
        "schema_version": 1,
        "study_stage": "cross_subject_development_LOSO_complete",
        "initial_cross_subject_freeze_git_commit": INITIAL_CROSS_SUBJECT_FREEZE_COMMIT,
        "cross_subject_config_sha256": file_sha256(args.config.resolve()),
        "cross_subject_implementation_sha256": file_sha256(
            PROJECT_ROOT / "eeg_project" / "cross_subject_decoding.py"
        ),
        "development_subjects": subjects,
        "development_fold_count": len(folds),
        "training_subject_count_per_fold": 19,
        "models": list(MODEL_NAMES),
        "evaluation_subject_classifier_outcomes_calculated": False,
        "evaluation_subject_EEG_loaded": False,
        "target_adaptation_used": False,
        "every_learned_stage_training_subjects_only": True,
        "one_primary_value_per_held_out_participant": True,
        "primary_prediction_count_all_models": len(prediction_rows),
        "source_hashes": dict(sorted(source_hashes.items())),
        "artifact_sha256": artifact_hashes,
        "full_feature_arrays_persisted": False,
        "fitted_models_persisted": False,
    }
    metadata_path = output / f"{PREFIX}metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nDevelopment LOSO summary:")
    for row in group_rows:
        print(
            f"  {row['model']}: median={row['median_balanced_accuracy']:.3f}; "
            f"IQR={row['q25_balanced_accuracy']:.3f}–{row['q75_balanced_accuracy']:.3f}; "
            f">0.5={row['above_0_5_fraction']:.1%}"
        )
    print(
        "  CSP minus spectral median="
        f"{paired_summary['median_CSP_minus_spectral']:+.3f}"
    )
    print("  Evaluation participant classifier outcomes remain locked.")


if __name__ == "__main__":
    main()
