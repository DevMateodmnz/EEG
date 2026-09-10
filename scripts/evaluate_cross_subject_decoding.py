"""Run the final-frozen zero-shot cross-subject evaluation once."""

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
    MODEL_NAMES,
    classify_paired_transfer,
    classify_transfer,
    file_sha256,
    fit_cross_subject_model,
    load_cross_subject_data,
    paired_cross_subject_models,
    predict_cross_subject_target,
    summarize_cross_subject_group,
)
from eeg_project.cross_subject_evaluation import (  # noqa: E402
    DEFAULT_FINAL_CROSS_SUBJECT_CONFIG_PATH,
    aggregate_confusion_rows,
    evaluation_run_summary_rows,
    exploratory_relationship_rows,
    load_final_cross_subject_config,
    read_csv,
    within_cross_comparison_rows,
    within_cross_summary_rows,
)
from eeg_project.decoding import load_decoding_config  # noqa: E402
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY  # noqa: E402


FINAL_FREEZE_COMMIT = "f66b299"
PREFIX = "cross_subject_decoder_evaluation_"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "cross_subject_decoder_checkpoints"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen zero-shot models on unseen subjects 21–109."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_FINAL_CROSS_SUBJECT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write stable field-order CSV."""
    if not rows:
        raise ValueError(f"Cannot write empty evaluation artifact: {path}.")
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
    """Refuse evaluation unless the final method freeze is committed."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FINAL_FREEZE_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("The final cross-subject freeze is not an ancestor of HEAD.")


def checkpoint_path(directory: Path, subject: int) -> Path:
    return directory / f"subject_{subject:03d}.json"


def valid_checkpoint(
    path: Path,
    subject: int,
    final_config_hash: str,
    frozen_core_hash: str,
    training_source_hashes: Mapping[str, str],
) -> dict[str, object] | None:
    """Reuse a complete checkpoint only when method and every source still match."""
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
        or payload.get("frozen_core_sha256") != frozen_core_hash
        or payload.get("training_source_hashes") != dict(training_source_hashes)
        or payload.get("target_adaptation_used") is not False
        or len(payload.get("model_results", [])) != 2
    ):
        return None
    for source_name, expected in payload.get("target_source_hashes", {}).items():
        source = Path(source_name)
        if not source.is_file() or file_sha256(source) != expected:
            return None
    return payload


def calculate_checkpoint(
    subject: int,
    fitted_models: Mapping[str, object],
    within_config: Mapping[str, object],
    cross_config: Mapping[str, object],
    data_directory: Path,
    final_config_hash: str,
    frozen_core_hash: str,
    training_source_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Predict both already-fitted models for one unseen participant."""
    target = load_cross_subject_data(subject, cross_config, data_directory=data_directory)
    model_results: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        result = predict_cross_subject_target(
            fitted_models[model_name], target, within_config
        )
        model_results.append(
            {
                "model": model_name,
                "prediction_rows": result.prediction_rows,
                "run_rows": result.run_rows,
                "subject_row": result.subject_row,
                "fit_audit_row": result.fit_audit_row,
            }
        )
    target_source_hashes = {
        identity.source_file: identity.source_sha256 for identity in target.identities
    }
    return {
        "schema_version": 1,
        "complete": True,
        "subject": subject,
        "final_config_sha256": final_config_hash,
        "frozen_core_sha256": frozen_core_hash,
        "training_source_hashes": dict(training_source_hashes),
        "target_source_hashes": target_source_hashes,
        "target_adaptation_used": False,
        "model_results": model_results,
    }


def create_distribution_figure(
    subject_rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    rng = np.random.default_rng(20260831)
    colors = ("#2a9d8f", "#e76f51")
    values = []
    for index, (model, color) in enumerate(zip(MODEL_NAMES, colors, strict=True), start=1):
        scores = np.asarray(
            [
                float(row["primary_mean_run_balanced_accuracy"])
                for row in subject_rows
                if row["model"] == model
            ]
        )
        values.append(scores)
        axis.scatter(
            index + rng.uniform(-0.09, 0.09, size=scores.size),
            scores,
            color=color,
            alpha=0.62,
            s=24,
        )
    axis.boxplot(values, tick_labels=["Spectral-LDA", "CSP+LDA"], showfliers=False)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set(
        ylabel="Zero-shot balanced accuracy",
        ylim=(0.05, 1.02),
        title="Unseen-participant decoding performance",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_paired_figure(
    paired_rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    spectral = np.asarray([float(row["spectral_balanced_accuracy"]) for row in paired_rows])
    csp = np.asarray([float(row["csp_balanced_accuracy"]) for row in paired_rows])
    figure, axis = plt.subplots(figsize=(6.2, 6.0))
    axis.scatter(spectral, csp, color="#457b9d", alpha=0.68, s=30)
    axis.plot((0.2, 0.9), (0.2, 0.9), color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Spectral-LDA balanced accuracy",
        ylabel="CSP+LDA balanced accuracy",
        xlim=(0.2, 0.9),
        ylim=(0.2, 0.9),
        aspect="equal",
        title="Matched zero-shot model comparison",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_within_cross_figure(
    comparison_rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharex=True, sharey=True)
    labels = ("Spectral-LDA", "CSP+LDA")
    colors = ("#2a9d8f", "#e76f51")
    for axis, model, label, color in zip(axes, MODEL_NAMES, labels, colors, strict=True):
        selected = [row for row in comparison_rows if row["model"] == model]
        within = [float(row["within_subject_balanced_accuracy"]) for row in selected]
        cross = [float(row["cross_subject_balanced_accuracy"]) for row in selected]
        axis.scatter(within, cross, color=color, alpha=0.62, s=27)
        axis.plot((0.2, 1.0), (0.2, 1.0), color="black", linestyle="--", linewidth=1)
        axis.set(title=label, xlabel="Within-subject balanced accuracy")
    axes[0].set_ylabel("Zero-shot cross-subject balanced accuracy")
    axes[0].set_xlim(0.2, 1.0)
    axes[0].set_ylim(0.2, 1.0)
    figure.suptitle("Removing participant-specific training changes performance")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_run_figure(run_rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    labels = ("Spectral-LDA", "CSP+LDA")
    colors = ("#2a9d8f", "#e76f51", "#457b9d")
    rng = np.random.default_rng(20260831)
    for axis, model, label in zip(axes, MODEL_NAMES, labels, strict=True):
        values = []
        for index, (run, color) in enumerate(zip((6, 10, 14), colors, strict=True), start=1):
            scores = np.asarray(
                [
                    float(row["balanced_accuracy"])
                    for row in run_rows
                    if row["model"] == model and int(row["test_run"]) == run
                ]
            )
            values.append(scores)
            axis.scatter(
                index + rng.uniform(-0.08, 0.08, size=scores.size),
                scores,
                color=color,
                alpha=0.42,
                s=18,
            )
        axis.boxplot(values, tick_labels=["Run 6", "Run 10", "Run 14"], showfliers=False)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set(title=label, xlabel="Target run", ylim=(0.05, 1.02))
    axes[0].set_ylabel("Zero-shot balanced accuracy")
    figure.suptitle("Run-specific performance in unseen participants")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    mne.set_log_level("ERROR")
    verify_final_freeze_is_ancestor()
    final, cross_config = load_final_cross_subject_config(args.config)
    within_config = load_decoding_config()
    final_hash = file_sha256(args.config.resolve())
    frozen_core_hash = final["frozen_method_implementation"]["sha256"]
    development_metadata = json.loads(
        (
            PROJECT_ROOT
            / "docs"
            / "assets"
            / "cross_subject_decoder_development_metadata.json"
        ).read_text(encoding="utf-8")
    )
    training_source_hashes = development_metadata["source_hashes"]
    for source_name, expected in training_source_hashes.items():
        if file_sha256(Path(source_name)) != expected:
            raise RuntimeError(f"Training EDF drift before final evaluation: {source_name}.")

    subjects = list(final["cohorts"]["evaluation_eligible_subjects"])
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[int, dict[str, object]] = {}
    missing: list[int] = []
    for subject in subjects:
        payload = valid_checkpoint(
            checkpoint_path(args.checkpoint_dir, subject),
            subject,
            final_hash,
            frozen_core_hash,
            training_source_hashes,
        )
        if payload is None:
            missing.append(subject)
        else:
            checkpoints[subject] = payload

    fitted_models: dict[str, object] = {}
    if missing:
        training_datasets = [
            load_cross_subject_data(
                subject, cross_config, data_directory=args.data_dir
            )
            for subject in final["cohorts"]["training_subjects"]
        ]
        if sum(dataset.labels.size for dataset in training_datasets) != 900:
            raise RuntimeError("Final cross-subject training must contain exactly 900 trials.")
        fitted_models = {
            model: fit_cross_subject_model(training_datasets, model, within_config)
            for model in MODEL_NAMES
        }

    for subject_index, subject in enumerate(subjects, start=1):
        if subject in checkpoints:
            print(
                f"Cross-subject evaluation: reused S{subject:03d} "
                f"({subject_index}/{len(subjects)})"
            )
            continue
        payload = calculate_checkpoint(
            subject,
            fitted_models,
            within_config,
            cross_config,
            args.data_dir,
            final_hash,
            frozen_core_hash,
            training_source_hashes,
        )
        path = checkpoint_path(args.checkpoint_dir, subject)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        checkpoints[subject] = payload
        print(
            f"Cross-subject evaluation: calculated S{subject:03d} "
            f"({subject_index}/{len(subjects)})"
        )

    prediction_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    evaluation_source_hashes: dict[str, str] = {}
    for subject in subjects:
        payload = checkpoints[subject]
        evaluation_source_hashes.update(payload["target_source_hashes"])
        for result in payload["model_results"]:
            prediction_rows.extend(result["prediction_rows"])
            run_rows.extend(result["run_rows"])
            subject_rows.append(result["subject_row"])
            audit_rows.append(result["fit_audit_row"])

    # Primary result is finalized before any historical decoder or physiology join.
    group_rows = [
        summarize_cross_subject_group(
            subject_rows,
            model,
            cross_config,
            expected_subject_count=86,
        )
        for model in MODEL_NAMES
    ]
    paired_rows, paired_summary = paired_cross_subject_models(subject_rows)
    run_summary = evaluation_run_summary_rows(run_rows, expected_subject_count=86)
    confusion_rows = aggregate_confusion_rows(subject_rows, expected_subject_count=86)
    transfer_categories = {
        row["model"]: classify_transfer(row, cross_config) for row in group_rows
    }
    paired_category = classify_paired_transfer(paired_summary, cross_config)

    historical_within = read_csv(
        args.output_dir / "within_subject_decoder_evaluation_subject_scores.csv"
    )
    comparison_rows = within_cross_comparison_rows(subject_rows, historical_within)
    comparison_summary = within_cross_summary_rows(comparison_rows)
    physiology = read_csv(
        args.output_dir / "within_subject_decoder_evaluation_exploratory_subject_context.csv"
    )
    reliability = read_csv(
        args.output_dir / "decoding_reliability_subject_reliability.csv"
    )
    relationship_rows = exploratory_relationship_rows(
        subject_rows, physiology, reliability
    )

    output = args.output_dir
    artifact_rows = {
        "predictions.csv": prediction_rows,
        "run_scores.csv": run_rows,
        "subject_scores.csv": subject_rows,
        "fit_audit.csv": audit_rows,
        "group_summary.csv": group_rows,
        "paired_subjects.csv": paired_rows,
        "paired_summary.csv": [{**paired_summary, "transfer_category": paired_category}],
        "run_summary.csv": run_summary,
        "confusion_matrices.csv": confusion_rows,
        "within_cross_subjects.csv": comparison_rows,
        "within_cross_summary.csv": comparison_summary,
        "exploratory_relationships.csv": relationship_rows,
    }
    artifact_hashes: dict[str, str] = {}
    for suffix, rows in artifact_rows.items():
        path = output / f"{PREFIX}{suffix}"
        write_csv(rows, path)
        artifact_hashes[path.name] = file_sha256(path)

    figures = {
        "score_distribution.png": lambda path: create_distribution_figure(subject_rows, path),
        "paired_models.png": lambda path: create_paired_figure(paired_rows, path),
        "within_vs_cross.png": lambda path: create_within_cross_figure(comparison_rows, path),
        "run_performance.png": lambda path: create_run_figure(run_rows, path),
    }
    for suffix, creator in figures.items():
        path = output / f"{PREFIX}{suffix}"
        creator(path)
        artifact_hashes[path.name] = file_sha256(path)

    all_source_hashes = {**training_source_hashes, **evaluation_source_hashes}
    metadata = {
        "schema_version": 1,
        "study_stage": "final_zero_shot_cross_subject_evaluation_complete",
        "final_freeze_git_commit": FINAL_FREEZE_COMMIT,
        "final_config_sha256": final_hash,
        "frozen_core_sha256": frozen_core_hash,
        "development_training_subject_count": 20,
        "development_training_trial_count": 900,
        "evaluation_subject_count": 86,
        "evaluation_trial_count_per_model": len(prediction_rows) // len(MODEL_NAMES),
        "checkpoint_count": len(checkpoints),
        "checkpoint_reuse_count_this_run": len(subjects) - len(missing),
        "models": list(MODEL_NAMES),
        "transfer_categories": transfer_categories,
        "paired_transfer_category": paired_category,
        "target_adaptation_used": False,
        "target_subject_EEG_used_for_CSP_scaler_or_LDA_fit": False,
        "one_primary_value_per_evaluation_participant": True,
        "historical_within_subject_predictions_modified": False,
        "new_classifier_family_performed": False,
        "deep_learning_performed": False,
        "training_source_hashes": training_source_hashes,
        "evaluation_source_hashes": evaluation_source_hashes,
        "source_hashes": dict(sorted(all_source_hashes.items())),
        "artifact_sha256": artifact_hashes,
        "full_feature_arrays_persisted": False,
        "fitted_models_persisted": False,
    }
    metadata_path = output / f"{PREFIX}metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nFinal zero-shot evaluation summary:")
    for row in group_rows:
        print(
            f"  {row['model']}: median={row['median_balanced_accuracy']:.3f}; "
            f"IQR={row['q25_balanced_accuracy']:.3f}–{row['q75_balanced_accuracy']:.3f}; "
            f">0.5={row['above_0_5_fraction']:.1%}; "
            f"category={transfer_categories[row['model']]}"
        )
    print(
        f"  CSP minus spectral median={paired_summary['median_CSP_minus_spectral']:+.3f}; "
        f"category={paired_category}"
    )
    print("  No evaluation-participant adaptation was used.")


if __name__ == "__main__":
    main()
