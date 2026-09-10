"""Evaluate frozen A/B/C personalization on the 86 eligible participants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.cross_subject_decoding import (  # noqa: E402
    file_sha256,
    load_cross_subject_config,
    load_cross_subject_data,
)
from eeg_project.decoding import RUNS, SubjectDecodingData, TrialIdentity  # noqa: E402
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY  # noqa: E402
from eeg_project.spatial_personalization import (  # noqa: E402
    A_METHOD,
    evaluate_personalization_scenario,
    paired_personalization_comparison,
    participant_csp_stability_row,
    participant_personalization_rows,
    personalization_group_summary,
    classify_personalization_geometry,
)
from eeg_project.spatial_personalization_evaluation import (  # noqa: E402
    DEFAULT_FINAL_SPATIAL_PERSONALIZATION_CONFIG_PATH,
    FINAL_SPATIAL_PERSONALIZATION_FREEZE_COMMIT,
    calibration_run_summary_rows,
    exploratory_relationship_rows,
    historical_context_rows,
    load_final_spatial_personalization_config,
    poor_zero_shot_subgroup_row,
    read_csv,
)
from eeg_project.target_calibration import TargetSourceFeatures  # noqa: E402
from scripts.evaluate_target_calibration import deserialize_target_features  # noqa: E402


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_SOURCE_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "target_calibration_checkpoints"
DEFAULT_RAW_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "spatial_personalization_checkpoints"
PREFIX = "spatial_personalization_evaluation_"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen target-CSP personalization.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_FINAL_SPATIAL_PERSONALIZATION_CONFIG_PATH
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--source-checkpoint-dir", type=Path, default=DEFAULT_SOURCE_CHECKPOINT_DIRECTORY
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_RAW_CHECKPOINT_DIRECTORY
    )
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty spatial-personalization artifact: {path}.")
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
        [
            "git",
            "merge-base",
            "--is-ancestor",
            FINAL_SPATIAL_PERSONALIZATION_FREEZE_COMMIT,
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("The final spatial-personalization freeze is not an ancestor.")


def source_checkpoint_path(directory: Path, subject: int) -> Path:
    return directory / f"subject_{subject:03d}.json"


def raw_checkpoint_paths(directory: Path, subject: int) -> tuple[Path, Path]:
    return directory / f"subject_{subject:03d}.json", directory / f"subject_{subject:03d}_csp.npy"


def load_validated_source_features(
    directory: Path,
    subject: int,
    calibration_metadata: Mapping[str, object],
) -> tuple[TargetSourceFeatures, dict[str, object], str]:
    """Reuse and revalidate the completed source-CSP target feature checkpoint."""
    path = source_checkpoint_path(directory, subject)
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "final_config_sha256": calibration_metadata["final_config_sha256"],
        "frozen_calibration_core_sha256": calibration_metadata[
            "frozen_calibration_core_sha256"
        ],
        "frozen_cross_subject_core_sha256": calibration_metadata[
            "frozen_cross_subject_core_sha256"
        ],
        "training_source_hashes": calibration_metadata["training_source_hashes"],
    }
    if not payload.get("complete") or int(payload.get("subject", -1)) != subject:
        raise RuntimeError(f"Invalid source-feature checkpoint for S{subject:03d}.")
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"Source-feature checkpoint {field} drift for S{subject:03d}.")
    if payload.get("source_CSP_or_scaler_fit_target_EEG"):
        raise RuntimeError("Source-feature checkpoint reports target source-model fitting.")
    target_hashes = payload.get("target_source_hashes")
    if not isinstance(target_hashes, dict) or len(target_hashes) != 3:
        raise RuntimeError("Source-feature checkpoint target hashes are incomplete.")
    for source_name, expected_hash in target_hashes.items():
        if file_sha256(Path(source_name)) != expected_hash:
            raise RuntimeError(f"Target source EDF drift: {source_name}.")
    source = deserialize_target_features(payload)
    return source, payload, file_sha256(path)


def raw_checkpoint_metadata(
    dataset: SubjectDecodingData,
    metadata_path: Path,
    array_path: Path,
    *,
    final_hash: str,
    frozen_core_hash: str,
    source_checkpoint_sha256: str,
) -> dict[str, object]:
    """Persist only the target 8–30 Hz trial tensor needed for target CSP."""
    np.save(array_path, dataset.csp_task_data_volts, allow_pickle=False)
    source_hashes = {
        identity.source_file: identity.source_sha256 for identity in dataset.identities
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "complete": True,
        "subject": dataset.subject,
        "final_config_sha256": final_hash,
        "frozen_core_sha256": frozen_core_hash,
        "source_feature_checkpoint_sha256": source_checkpoint_sha256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "array_file": array_path.name,
        "array_sha256": file_sha256(array_path),
        "array_shape": list(dataset.csp_task_data_volts.shape),
        "array_dtype": str(dataset.csp_task_data_volts.dtype),
        "test_EEG_used_for_fitting_at_checkpoint_creation": False,
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def valid_raw_checkpoint(
    metadata_path: Path,
    array_path: Path,
    subject: int,
    source: TargetSourceFeatures,
    *,
    final_hash: str,
    frozen_core_hash: str,
    source_checkpoint_sha256: str,
) -> tuple[dict[str, object], np.ndarray] | None:
    """Return a source/config/hash-validated target tensor checkpoint."""
    if not metadata_path.exists() or not array_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not payload["complete"]
            or payload["subject"] != subject
            or payload["final_config_sha256"] != final_hash
            or payload["frozen_core_sha256"] != frozen_core_hash
            or payload["source_feature_checkpoint_sha256"] != source_checkpoint_sha256
            or payload["array_file"] != array_path.name
            or payload["array_sha256"] != file_sha256(array_path)
            or payload["source_hashes"]
            != {
                identity.source_file: identity.source_sha256
                for identity in source.identities
            }
        ):
            return None
        array = np.load(array_path, allow_pickle=False)
        if (
            list(array.shape) != payload["array_shape"]
            or str(array.dtype) != payload["array_dtype"]
            or array.shape[0] != source.labels.size
            or array.shape[1:] != (64, 320)
            or not np.isfinite(array).all()
        ):
            return None
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    return payload, array


def dataset_from_checkpoint(source: TargetSourceFeatures, array: np.ndarray) -> SubjectDecodingData:
    """Reconstruct the aligned target object without re-running preprocessing."""
    return SubjectDecodingData(
        source.subject,
        np.empty((source.labels.size, 0, 0)),
        np.asarray(array),
        source.labels.copy(),
        source.runs.copy(),
        [identity for identity in source.identities if isinstance(identity, TrialIdentity)],
        source.qc_candidates.copy(),
        [f"EEG{index:02d}" for index in range(array.shape[1])],
        160.0,
        1.0,
        3.0,
    )


def create_method_figure(
    participant_rows: Sequence[Mapping[str, object]], methods: Sequence[str], path: Path
) -> None:
    values = [
        [
            float(row["primary_mean_scenario_test_run_balanced_accuracy"])
            for row in participant_rows
            if row["method"] == method and not bool(row["qc_sensitivity"])
        ]
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.boxplot(values, tick_labels=["A: source", "B: target LDA", "C: target CSP"], showfliers=False)
    axis.axhline(0.5, color="0.45", linestyle="--", linewidth=1)
    for index, method_values in enumerate(values, start=1):
        axis.scatter(np.full(len(method_values), index), method_values, s=13, alpha=0.45)
    axis.set_ylabel("Participant balanced accuracy")
    axis.set_ylim(0.3, 1.02)
    axis.set_title("Frozen spatial-personalization comparison")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_paired_gain_figure(
    paired_rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    comparisons = []
    for candidate, reference, label in (
        ("source_csp_target_lda_source_scaler", A_METHOD, "B − A"),
        ("target_csp_ledoit_wolf", "source_csp_target_lda_source_scaler", "C − B"),
        ("target_csp_ledoit_wolf", A_METHOD, "C − A"),
    ):
        comparisons.append(
            (
                label,
                [
                    float(row["candidate_minus_reference"])
                    for row in paired_rows
                    if row["candidate_method"] == candidate
                    and row["reference_method"] == reference
                ],
            )
        )
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    values = [values for _, values in comparisons]
    axis.boxplot(values, tick_labels=[label for label, _ in comparisons], showfliers=False)
    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    for index, method_values in enumerate(values, start=1):
        axis.scatter(np.full(len(method_values), index), method_values, s=13, alpha=0.45)
    axis.set_ylabel("Participant-matched balanced-accuracy gain")
    axis.set_title("Where does personalization add value?")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_stability_figure(
    stability_rows: Sequence[Mapping[str, object]],
    paired_c_minus_b: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    gain = {int(row["subject"]): float(row["candidate_minus_reference"]) for row in paired_c_minus_b}
    x = np.asarray([float(row["mean_pairwise_subspace_similarity"]) for row in stability_rows])
    y = np.asarray([gain[int(row["subject"])] for row in stability_rows])
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.scatter(x, y, s=24, alpha=0.65)
    axis.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    axis.axvline(0.5, color="0.7", linestyle=":", linewidth=1)
    axis.set_xlabel("Target-CSP subspace similarity (0–1)")
    axis.set_ylabel("C − B balanced-accuracy gain")
    axis.set_title("Run-to-run target-CSP stability")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    mne.set_log_level("ERROR")
    verify_final_freeze_is_ancestor()
    final, initial = load_final_spatial_personalization_config(args.config)
    final_hash = file_sha256(args.config.resolve())
    frozen_core_hash = final["frozen_implementation"]["sha256"]
    methods = [
        final["representation_A"]["method"],
        final["representation_B"]["method"],
        final["representation_C"]["method"],
    ]
    subjects = list(final["cohorts"]["evaluation_eligible_subjects"])
    training_subjects = tuple(final["cohorts"]["source_training_subjects"])

    calibration_metadata = json.loads(
        (args.output_dir / "target_calibration_evaluation_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    for path, expected in calibration_metadata["training_source_hashes"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"Source training EDF drift: {path}.")

    cross_config = load_cross_subject_config()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, object]] = []
    test_run_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    evaluation_source_hashes: dict[str, str] = {}
    checkpoint_reuse_count = 0

    for index, subject in enumerate(subjects, start=1):
        source, source_payload, source_checkpoint_sha = load_validated_source_features(
            args.source_checkpoint_dir, subject, calibration_metadata
        )
        evaluation_source_hashes.update(source_payload["target_source_hashes"])
        metadata_path, array_path = raw_checkpoint_paths(args.checkpoint_dir, subject)
        checkpoint = valid_raw_checkpoint(
            metadata_path,
            array_path,
            subject,
            source,
            final_hash=final_hash,
            frozen_core_hash=frozen_core_hash,
            source_checkpoint_sha256=source_checkpoint_sha,
        )
        if checkpoint is None:
            dataset = load_cross_subject_data(
                subject, cross_config, data_directory=args.data_dir
            )
            if [identity.key for identity in dataset.identities] != [
                identity.key for identity in source.identities
            ]:
                raise RuntimeError("New raw checkpoint trial identities differ from source features.")
            raw_checkpoint_metadata(
                dataset,
                metadata_path,
                array_path,
                final_hash=final_hash,
                frozen_core_hash=frozen_core_hash,
                source_checkpoint_sha256=source_checkpoint_sha,
            )
            status = "calculated"
        else:
            _, array = checkpoint
            dataset = dataset_from_checkpoint(source, array)
            checkpoint_reuse_count += 1
            status = "reused"

        subspaces: dict[int, np.ndarray] = {}
        for method in methods:
            for calibration_run in RUNS:
                result = evaluate_personalization_scenario(
                    dataset,
                    source,
                    training_subjects,
                    calibration_run,
                    method,
                    initial,
                )
                prediction_rows.extend(result.prediction_rows)
                test_run_rows.extend(result.test_run_rows)
                audit_rows.append(result.audit_row)
                if method == final["representation_C"]["method"]:
                    assert result.csp_filter_subspace is not None
                    subspaces[calibration_run] = result.csp_filter_subspace
        stability_rows.append(
            participant_csp_stability_row(
                subject, final["representation_C"]["method"], subspaces
            )
        )
        print(f"Spatial personalization: {status} S{subject:03d} ({index}/{len(subjects)})")

    participant_rows = participant_personalization_rows(
        test_run_rows,
        prediction_rows,
        expected_subjects=subjects,
        methods=methods,
    )
    group_rows = personalization_group_summary(
        participant_rows, methods, initial, expected_subject_count=86
    )
    all_paired_rows: list[dict[str, object]] = []
    paired_summaries: list[dict[str, object]] = []
    paired_by_name: dict[str, tuple[list[dict[str, object]], dict[str, object]]] = {}
    for record in final["primary_comparisons"]:
        rows, summary = paired_personalization_comparison(
            participant_rows,
            record["candidate"],
            record["reference"],
            initial,
            expected_subject_count=86,
        )
        summary["comparison_name"] = record["name"]
        all_paired_rows.extend({"comparison_name": record["name"], **row} for row in rows)
        paired_summaries.append(summary)
        paired_by_name[record["name"]] = (rows, summary)
    interpretation = classify_personalization_geometry(
        paired_by_name["B_minus_A"][1],
        paired_by_name["C_minus_B"][1],
        paired_by_name["C_minus_A"][1],
    )

    historical_cross = read_csv(
        args.output_dir / "cross_subject_decoder_evaluation_subject_scores.csv"
    )
    frozen_a = {
        int(row["subject"]): float(row["primary_mean_run_balanced_accuracy"])
        for row in historical_cross
        if row["model"] == "csp_4_empirical"
    }
    observed_a = {
        int(row["subject"]): float(row["primary_mean_scenario_test_run_balanced_accuracy"])
        for row in participant_rows
        if row["method"] == A_METHOD and not bool(row["qc_sensitivity"])
    }
    if observed_a != frozen_a:
        raise RuntimeError("Evaluation A does not reproduce frozen zero-shot CSP.")

    threshold_rows = read_csv(
        args.output_dir / "target_calibration_evaluation_participant_scores.csv"
    )
    within_rows = read_csv(
        args.output_dir / "within_subject_decoder_evaluation_subject_scores.csv"
    )
    physiology_rows = read_csv(
        args.output_dir / "within_subject_decoder_evaluation_exploratory_subject_context.csv"
    )
    run_rows = calibration_run_summary_rows(
        test_run_rows, methods, expected_subject_count=86
    )
    context_rows = historical_context_rows(
        participant_rows, threshold_rows, within_rows, methods
    )
    poor_row = poor_zero_shot_subgroup_row(
        paired_by_name["C_minus_B"][0], participant_rows
    )
    relationship_rows = exploratory_relationship_rows(
        paired_by_name["C_minus_B"][0],
        participant_rows,
        physiology_rows,
        within_rows,
        stability_rows,
    )
    similarities = np.asarray(
        [float(row["mean_pairwise_subspace_similarity"]) for row in stability_rows]
    )
    stability_summary = {
        "method": final["representation_C"]["method"],
        "participant_count": len(similarities),
        "median_mean_pairwise_subspace_similarity": float(np.median(similarities)),
        "q25_mean_pairwise_subspace_similarity": float(np.percentile(similarities, 25)),
        "q75_mean_pairwise_subspace_similarity": float(np.percentile(similarities, 75)),
        "low_below_0_50_count": int(np.sum(similarities < 0.50)),
        "moderate_0_50_to_below_0_75_count": int(
            np.sum((similarities >= 0.50) & (similarities < 0.75))
        ),
        "high_at_or_above_0_75_count": int(np.sum(similarities >= 0.75)),
        "sign_order_and_rotation_invariant": True,
        "analysis_role": "post_primary_exploratory_stability",
    }

    artifact_rows = {
        "predictions.csv": prediction_rows,
        "test_run_scores.csv": test_run_rows,
        "participant_scores.csv": participant_rows,
        "fit_audit.csv": audit_rows,
        "group_summary.csv": group_rows,
        "paired_comparison_subjects.csv": all_paired_rows,
        "paired_summary.csv": paired_summaries,
        "calibration_run_summary.csv": run_rows,
        "csp_stability.csv": stability_rows,
        "csp_stability_summary.csv": [stability_summary],
        "poor_zero_shot_subgroup.csv": [poor_row],
        "exploratory_relationships.csv": relationship_rows,
        "historical_context.csv": context_rows,
    }
    artifact_hashes: dict[str, str] = {}
    for suffix, rows in artifact_rows.items():
        path = args.output_dir / f"{PREFIX}{suffix}"
        write_csv(rows, path)
        artifact_hashes[path.name] = file_sha256(path)

    figures = {
        "method_scores.png": lambda path: create_method_figure(participant_rows, methods, path),
        "paired_gains.png": lambda path: create_paired_gain_figure(all_paired_rows, path),
        "csp_stability.png": lambda path: create_stability_figure(
            stability_rows, paired_by_name["C_minus_B"][0], path
        ),
    }
    for suffix, creator in figures.items():
        path = args.output_dir / f"{PREFIX}{suffix}"
        creator(path)
        artifact_hashes[path.name] = file_sha256(path)

    metadata = {
        "schema_version": 1,
        "study_stage": "spatial_personalization_evaluation_complete",
        "final_freeze_git_commit": FINAL_SPATIAL_PERSONALIZATION_FREEZE_COMMIT,
        "final_config_sha256": final_hash,
        "frozen_core_sha256": frozen_core_hash,
        "methods": methods,
        "interpretation_category": interpretation,
        "evaluation_subject_count": 86,
        "evaluation_unique_trial_count": len({row["trial_key"] for row in prediction_rows}),
        "prediction_appearance_count": len(prediction_rows),
        "raw_checkpoint_count": len(subjects),
        "raw_checkpoint_reuse_count_this_run": checkpoint_reuse_count,
        "source_feature_checkpoint_count": len(subjects),
        "source_model_refitted": False,
        "source_CSP_or_scaler_refitted": False,
        "target_CSP_fit_count": 86 * 3,
        "target_CSP_fit_calibration_only": True,
        "test_EEG_used_for_any_fit": False,
        "unlabeled_test_EEG_used_for_adaptation": False,
        "same_test_trials_compared_across_A_B_C": True,
        "one_value_per_participant_per_method": True,
        "historical_zero_shot_outputs_modified": False,
        "historical_calibration_outputs_modified": False,
        "historical_within_subject_outputs_modified": False,
        "new_classifier_family_performed": False,
        "deep_learning_performed": False,
        "training_source_hashes": calibration_metadata["training_source_hashes"],
        "evaluation_source_hashes": dict(sorted(evaluation_source_hashes.items())),
        "artifact_sha256": artifact_hashes,
        "fitted_models_persisted": False,
    }
    metadata_path = args.output_dir / f"{PREFIX}metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nFinal spatial-personalization summary:")
    for row in group_rows:
        print(
            f"  {row['method']}: median={row['median_balanced_accuracy']:.3f}; "
            f"IQR={row['q25_balanced_accuracy']:.3f}–{row['q75_balanced_accuracy']:.3f}"
        )
    for row in paired_summaries:
        print(
            f"  {row['comparison_name']}: median={row['median_paired_difference']:+.3f}; "
            f"improved={row['improved_fraction']:.1%}; "
            f"convincing={row['convincing_paired_gain']}"
        )
    print(f"  Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
