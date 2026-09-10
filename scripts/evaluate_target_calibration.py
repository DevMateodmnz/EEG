"""Evaluate the final-frozen minimal target-calibration curve once."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.cross_subject_decoding import (  # noqa: E402
    file_sha256,
    fit_cross_subject_model,
    load_cross_subject_config,
    load_cross_subject_data,
)
from eeg_project.decoding import RUNS, TrialIdentity, load_decoding_config  # noqa: E402
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY  # noqa: E402
from eeg_project.target_calibration import (  # noqa: E402
    CALIBRATION_SIZES,
    TargetSourceFeatures,
    calibration_curve_summary,
    convincing_benefit,
    evaluate_calibration_scenario,
    participant_calibration_rows,
    transform_target_with_source,
    validate_nested_calibration_subsets,
)
from eeg_project.target_calibration_evaluation import (  # noqa: E402
    DEFAULT_FINAL_CALIBRATION_CONFIG_PATH,
    calibration_run_summary_rows,
    exploratory_relationship_rows,
    historical_within_comparison_rows,
    incremental_gain_rows,
    load_final_calibration_config,
    matched_gain_rows,
    read_csv,
    subgroup_gain_rows,
)


FINAL_FREEZE_COMMIT = "80e5ebc"
PREFIX = "target_calibration_evaluation_"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "target_calibration_checkpoints"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen threshold calibration on subjects 21–109."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_FINAL_CALIBRATION_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty calibration evaluation artifact: {path}.")
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


def verify_final_freeze_is_ancestor() -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FINAL_FREEZE_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("The final calibration freeze is not an ancestor of HEAD.")


def checkpoint_path(directory: Path, subject: int) -> Path:
    return directory / f"subject_{subject:03d}.json"


def valid_checkpoint(
    path: Path,
    subject: int,
    final_config_hash: str,
    frozen_calibration_core_hash: str,
    frozen_cross_core_hash: str,
    training_source_hashes: Mapping[str, str],
) -> dict[str, object] | None:
    """Validate cached frozen-source target features and all source hashes."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != 1
        or payload.get("complete") is not True
        or payload.get("subject") != subject
        or payload.get("final_config_sha256") != final_config_hash
        or payload.get("frozen_calibration_core_sha256")
        != frozen_calibration_core_hash
        or payload.get("frozen_cross_subject_core_sha256") != frozen_cross_core_hash
        or payload.get("training_source_hashes") != dict(training_source_hashes)
        or payload.get("source_CSP_or_scaler_fit_target_EEG") is not False
        or len(payload.get("target_feature_rows", [])) not in {42, 45}
    ):
        return None
    target_hashes = payload.get("target_source_hashes", {})
    if len(target_hashes) != 3:
        return None
    for source_name, expected in target_hashes.items():
        source = Path(source_name)
        if not source.is_file() or file_sha256(source) != expected:
            return None
    return payload


def serialize_target_features(
    source: TargetSourceFeatures,
    final_config_hash: str,
    frozen_calibration_core_hash: str,
    frozen_cross_core_hash: str,
    training_source_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Persist only compact source-transformed target rows, labels, and provenance."""
    rows = []
    for index, identity in enumerate(source.identities):
        rows.append(
            {
                "subject": source.subject,
                "run": int(source.runs[index]),
                "run_trial_index": identity.run_trial_index,
                "annotation_index": identity.annotation_index,
                "semantic_condition": identity.semantic_condition,
                "trial_key": identity.key,
                "label": int(source.labels[index]),
                "qc_candidate": bool(source.qc_candidates[index]),
                "source_scaled_CSP_features": source.features[index].tolist(),
                "source_decision_score_feet": float(source.source_decision_scores[index]),
                "source_prediction": int(source.source_predictions[index]),
                "source_file": identity.source_file,
                "source_sha256": identity.source_sha256,
            }
        )
    target_hashes = {
        identity.source_file: identity.source_sha256 for identity in source.identities
    }
    return {
        "schema_version": 1,
        "complete": True,
        "subject": source.subject,
        "final_config_sha256": final_config_hash,
        "frozen_calibration_core_sha256": frozen_calibration_core_hash,
        "frozen_cross_subject_core_sha256": frozen_cross_core_hash,
        "training_source_hashes": dict(training_source_hashes),
        "target_source_hashes": target_hashes,
        "source_CSP_or_scaler_fit_target_EEG": False,
        "target_feature_rows": rows,
    }


def deserialize_target_features(payload: Mapping[str, object]) -> TargetSourceFeatures:
    """Reconstruct the exact compact target feature bundle from a checkpoint."""
    subject = int(payload["subject"])
    rows = list(payload["target_feature_rows"])
    identities = tuple(
        TrialIdentity(
            subject,
            int(row["run"]),
            int(row["run_trial_index"]),
            int(row["annotation_index"]),
            str(row["semantic_condition"]),
            str(row["source_file"]),
            str(row["source_sha256"]),
        )
        for row in rows
    )
    source = TargetSourceFeatures(
        subject,
        np.asarray([row["source_scaled_CSP_features"] for row in rows], dtype=float),
        np.asarray([row["source_decision_score_feet"] for row in rows], dtype=float),
        np.asarray([row["source_prediction"] for row in rows], dtype=int),
        np.asarray([row["label"] for row in rows], dtype=int),
        np.asarray([row["run"] for row in rows], dtype=int),
        identities,
        np.asarray([row["qc_candidate"] for row in rows], dtype=bool),
    )
    if (
        source.features.shape != (len(rows), 4)
        or len({identity.key for identity in identities}) != len(rows)
        or set(np.unique(source.runs)) != set(RUNS)
        or set(np.unique(source.labels)) != {0, 1}
    ):
        raise RuntimeError("Calibration checkpoint target features are invalid.")
    return source


def create_curve_figure(
    curve_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    sizes = np.asarray([int(row["calibration_size"]) for row in curve_rows])
    medians = np.asarray([float(row["median_balanced_accuracy"]) for row in curve_rows])
    lower = np.asarray([float(row["bootstrap_median_ci_lower"]) for row in curve_rows])
    upper = np.asarray([float(row["bootstrap_median_ci_upper"]) for row in curve_rows])
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    axis.plot(sizes, medians, marker="o", color="#e76f51", linewidth=2)
    axis.fill_between(sizes, lower, upper, color="#e76f51", alpha=0.18)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.axhline(
        0.6577380952380952,
        color="#457b9d",
        linestyle=":",
        linewidth=1.5,
        label="Historical within-subject median",
    )
    axis.set(
        xlabel="Labeled target calibration trials",
        ylabel="Participant balanced accuracy",
        xticks=list(CALIBRATION_SIZES),
        ylim=(0.45, 0.72),
        title="Frozen threshold-calibration curve",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_gain_figure(
    gain_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    values = [
        np.asarray(
            [
                float(row["gain_over_matched_zero"])
                for row in gain_rows
                if int(row["calibration_size"]) == size
            ]
        )
        for size in CALIBRATION_SIZES[1:]
    ]
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    axis.boxplot(values, tick_labels=[str(size) for size in CALIBRATION_SIZES[1:]], showfliers=False)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Labeled target calibration trials",
        ylabel="Matched gain over zero-shot balanced accuracy",
        title="Participant-matched calibration gains",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_calibration_run_figure(
    run_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    colors = ("#2a9d8f", "#e76f51", "#457b9d")
    for run, color in zip(RUNS, colors, strict=True):
        selected = [row for row in run_rows if int(row["calibration_run"]) == run]
        axis.plot(
            [int(row["calibration_size"]) for row in selected],
            [float(row["median_balanced_accuracy"]) for row in selected],
            marker="o",
            color=color,
            label=f"Calibration run {run}",
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Labeled target calibration trials",
        ylabel="Median balanced accuracy",
        xticks=list(CALIBRATION_SIZES),
        title="Calibration-run sensitivity",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    mne.set_log_level("ERROR")
    verify_final_freeze_is_ancestor()
    final, calibration_config = load_final_calibration_config(args.config)
    cross_config = load_cross_subject_config()
    within_config = load_decoding_config()
    final_hash = file_sha256(args.config.resolve())
    frozen_calibration_core_hash = final["frozen_calibration_implementation"]["sha256"]
    # The actual frozen cross-subject core hash is preserved by the final zero-shot freeze.
    cross_final = json.loads(
        (PROJECT_ROOT / "config" / "cross_subject_decoding_final.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_cross_core_hash = cross_final["frozen_method_implementation"]["sha256"]
    development_metadata = json.loads(
        (
            args.output_dir / "target_calibration_development_metadata.json"
        ).read_text(encoding="utf-8")
    )
    training_source_hashes = development_metadata["source_hashes"]
    for source_name, expected in training_source_hashes.items():
        if file_sha256(Path(source_name)) != expected:
            raise RuntimeError(f"Calibration source EDF drift: {source_name}.")

    subjects = list(final["cohorts"]["evaluation_eligible_subjects"])
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[int, dict[str, object]] = {}
    missing: list[int] = []
    for subject in subjects:
        payload = valid_checkpoint(
            checkpoint_path(args.checkpoint_dir, subject),
            subject,
            final_hash,
            frozen_calibration_core_hash,
            frozen_cross_core_hash,
            training_source_hashes,
        )
        if payload is None:
            missing.append(subject)
        else:
            checkpoints[subject] = payload

    fitted = None
    if missing:
        training = [
            load_cross_subject_data(
                subject, cross_config, data_directory=args.data_dir
            )
            for subject in final["cohorts"]["source_training_subjects"]
        ]
        fitted = fit_cross_subject_model(
            training, "csp_4_empirical", within_config
        )
        if fitted.training_trial_count != 900:
            raise RuntimeError("Final calibration source model requires 900 trials.")
        if fitted.training_source_hashes != training_source_hashes:
            raise RuntimeError("Final calibration source hashes differ from development.")

    for index, subject in enumerate(subjects, start=1):
        if subject in checkpoints:
            print(f"Target calibration: reused S{subject:03d} ({index}/{len(subjects)})")
            continue
        target = load_cross_subject_data(
            subject, cross_config, data_directory=args.data_dir
        )
        validate_nested_calibration_subsets(target)
        source = transform_target_with_source(fitted, target)
        payload = serialize_target_features(
            source,
            final_hash,
            frozen_calibration_core_hash,
            frozen_cross_core_hash,
            training_source_hashes,
        )
        checkpoint_path(args.checkpoint_dir, subject).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        checkpoints[subject] = payload
        print(f"Target calibration: calculated S{subject:03d} ({index}/{len(subjects)})")

    method = final["selected_calibration_method"]["name"]
    prediction_rows: list[dict[str, object]] = []
    test_run_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    evaluation_source_hashes: dict[str, str] = {}
    training_subjects = tuple(final["cohorts"]["source_training_subjects"])
    for subject in subjects:
        payload = checkpoints[subject]
        evaluation_source_hashes.update(payload["target_source_hashes"])
        source = deserialize_target_features(payload)
        for size in CALIBRATION_SIZES:
            for calibration_run in RUNS:
                result = evaluate_calibration_scenario(
                    source,
                    training_subjects,
                    calibration_run,
                    size,
                    method,
                )
                prediction_rows.extend(result.prediction_rows)
                test_run_rows.extend(result.test_run_rows)
                audit_rows.append(result.audit_row)

    participant_rows = participant_calibration_rows(
        test_run_rows,
        prediction_rows,
        expected_subjects=subjects,
        methods=[method],
    )
    curve_rows = calibration_curve_summary(
        participant_rows,
        method,
        calibration_config,
        expected_subject_count=86,
    )
    for row in curve_rows:
        row["convincing_benefit"] = convincing_benefit(row, calibration_config)
    convincing_sizes = [
        int(row["calibration_size"])
        for row in curve_rows
        if bool(row["convincing_benefit"])
    ]
    smallest_convincing = min(convincing_sizes) if convincing_sizes else None

    historical_cross = read_csv(
        args.output_dir / "cross_subject_decoder_evaluation_subject_scores.csv"
    )
    frozen_zero = {
        int(row["subject"]): float(row["primary_mean_run_balanced_accuracy"])
        for row in historical_cross
        if row["model"] == "csp_4_empirical"
    }
    observed_zero = {
        int(row["subject"]): float(row["primary_mean_test_run_balanced_accuracy"])
        for row in participant_rows
        if int(row["calibration_size"]) == 0 and not bool(row["qc_sensitivity"])
    }
    if frozen_zero != observed_zero:
        raise RuntimeError("Evaluation size zero does not reproduce frozen zero-shot CSP.")

    gain_rows = matched_gain_rows(participant_rows, method=method)
    historical_within = read_csv(
        args.output_dir / "within_subject_decoder_evaluation_subject_scores.csv"
    )
    run_summary = calibration_run_summary_rows(
        test_run_rows, method=method, expected_subject_count=86
    )
    subgroup_rows = subgroup_gain_rows(gain_rows, historical_within)
    physiology = read_csv(
        args.output_dir / "within_subject_decoder_evaluation_exploratory_subject_context.csv"
    )
    relationship_rows = exploratory_relationship_rows(gain_rows, physiology)
    incremental_rows = incremental_gain_rows(participant_rows, method=method)
    within_rows = historical_within_comparison_rows(
        participant_rows, historical_within, method=method
    )

    artifact_rows = {
        "predictions.csv": prediction_rows,
        "test_run_scores.csv": test_run_rows,
        "participant_scores.csv": participant_rows,
        "fit_audit.csv": audit_rows,
        "curve_summary.csv": curve_rows,
        "matched_gain_subjects.csv": gain_rows,
        "calibration_run_summary.csv": run_summary,
        "subgroup_gains.csv": subgroup_rows,
        "exploratory_relationships.csv": relationship_rows,
        "incremental_gains.csv": incremental_rows,
        "historical_within_summary.csv": within_rows,
    }
    artifact_hashes: dict[str, str] = {}
    for suffix, rows in artifact_rows.items():
        path = args.output_dir / f"{PREFIX}{suffix}"
        write_csv(rows, path)
        artifact_hashes[path.name] = file_sha256(path)

    figures = {
        "curve.png": lambda path: create_curve_figure(curve_rows, path),
        "matched_gains.png": lambda path: create_gain_figure(gain_rows, path),
        "calibration_runs.png": lambda path: create_calibration_run_figure(run_summary, path),
    }
    for suffix, creator in figures.items():
        path = args.output_dir / f"{PREFIX}{suffix}"
        creator(path)
        artifact_hashes[path.name] = file_sha256(path)

    all_source_hashes = {**training_source_hashes, **evaluation_source_hashes}
    metadata = {
        "schema_version": 1,
        "study_stage": "minimal_target_calibration_evaluation_complete",
        "final_freeze_git_commit": FINAL_FREEZE_COMMIT,
        "final_config_sha256": final_hash,
        "frozen_calibration_core_sha256": frozen_calibration_core_hash,
        "frozen_cross_subject_core_sha256": frozen_cross_core_hash,
        "selected_method": method,
        "calibration_sizes": list(CALIBRATION_SIZES),
        "source_training_subject_count": 20,
        "source_training_trial_count": 900,
        "evaluation_subject_count": 86,
        "evaluation_unique_trial_count": len(frozen_zero) and len(
            {row["trial_key"] for row in prediction_rows}
        ),
        "prediction_appearance_count": len(prediction_rows),
        "checkpoint_count": len(checkpoints),
        "checkpoint_reuse_count_this_run": len(subjects) - len(missing),
        "smallest_convincing_calibration_size": smallest_convincing,
        "source_CSP_or_scaler_fit_target_EEG": False,
        "unlabeled_test_EEG_used_for_adaptation": False,
        "zero_shot_historical_outputs_modified": False,
        "within_subject_historical_outputs_modified": False,
        "one_value_per_participant_per_size": True,
        "new_classifier_family_performed": False,
        "deep_learning_performed": False,
        "training_source_hashes": training_source_hashes,
        "evaluation_source_hashes": evaluation_source_hashes,
        "source_hashes": dict(sorted(all_source_hashes.items())),
        "artifact_sha256": artifact_hashes,
        "fitted_models_persisted": False,
    }
    metadata_path = args.output_dir / f"{PREFIX}metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nFinal target-calibration curve:")
    for row in curve_rows:
        print(
            f"  n={row['calibration_size']:>2}: "
            f"median={row['median_balanced_accuracy']:.3f}; "
            f"gain={row['median_gain_over_matched_zero']:+.3f}; "
            f"improved={row['improved_fraction']:.1%}; "
            f"convincing={row['convincing_benefit']}"
        )
    print(f"  Smallest convincing calibration size: {smallest_convincing}")


if __name__ == "__main__":
    main()
