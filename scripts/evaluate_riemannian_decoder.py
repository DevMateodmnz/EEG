"""Evaluate the final-frozen Riemannian tangent decoder on 86 unseen people."""

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

from eeg_project.cross_subject_decoding import file_sha256, load_cross_subject_config, load_cross_subject_data  # noqa: E402
from eeg_project.riemannian_decoding import (  # noqa: E402
    SubjectCovarianceData, covariance_data_from_dataset, fit_riemannian_model,
    paired_riemannian_comparison, predict_riemannian_target, summarize_riemannian_group,
    classify_riemannian_result,
)
from eeg_project.riemannian_evaluation import (  # noqa: E402
    DEFAULT_FINAL_RIEMANNIAN_CONFIG_PATH, FINAL_RIEMANNIAN_FREEZE_COMMIT,
    exploratory_variability_rows, load_final_riemannian_config, run_summary_rows,
)
from scripts.evaluate_spatial_personalization import (  # noqa: E402
    dataset_from_checkpoint, load_validated_source_features,
)

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_SOURCE_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "target_calibration_checkpoints"
DEFAULT_RAW_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "spatial_personalization_checkpoints"
DEFAULT_COVARIANCE_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "riemannian_decoder_checkpoints"
PREFIX = "riemannian_decoder_evaluation_"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the final-frozen Riemannian zero-shot decoder.")
    parser.add_argument("--config", type=Path, default=DEFAULT_FINAL_RIEMANNIAN_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--source-checkpoint-dir", type=Path, default=DEFAULT_SOURCE_CHECKPOINT_DIRECTORY)
    parser.add_argument("--raw-checkpoint-dir", type=Path, default=DEFAULT_RAW_CHECKPOINT_DIRECTORY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_COVARIANCE_CHECKPOINT_DIRECTORY)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty Riemannian artifact {path}.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def verify_final_freeze_is_ancestor() -> None:
    if subprocess.run(["git", "merge-base", "--is-ancestor", FINAL_RIEMANNIAN_FREEZE_COMMIT, "HEAD"], cwd=PROJECT_ROOT, check=False).returncode:
        raise RuntimeError("The final Riemannian freeze is not an ancestor of HEAD.")


def covariance_checkpoint_paths(directory: Path, subject: int) -> tuple[Path, Path]:
    return directory / f"subject_{subject:03d}.json", directory / f"subject_{subject:03d}_covariances.npy"


def checkpoint_data(subject: int, source: object, raw_directory: Path, covariance_directory: Path, final_hash: str, core_hash: str) -> tuple[SubjectCovarianceData, bool]:
    """Reuse a hash-validated compact SPD checkpoint or derive it from prior raw tensors."""
    metadata_path, array_path = covariance_checkpoint_paths(covariance_directory, subject)
    raw_metadata = raw_directory / f"subject_{subject:03d}.json"
    raw_array = raw_directory / f"subject_{subject:03d}_csp.npy"
    raw_payload = json.loads(raw_metadata.read_text(encoding="utf-8"))
    if raw_payload.get("array_sha256") != file_sha256(raw_array):
        raise RuntimeError(f"Spatial raw checkpoint drift for S{subject:03d}.")
    source_hash = file_sha256(DEFAULT_SOURCE_CHECKPOINT_DIRECTORY / f"subject_{subject:03d}.json")
    expected = {"final_config_sha256": final_hash, "frozen_core_sha256": core_hash, "spatial_raw_array_sha256": raw_payload["array_sha256"], "source_feature_checkpoint_sha256": source_hash}
    if metadata_path.exists() and array_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("complete") and payload.get("subject") == subject and all(payload.get(key) == value for key, value in expected.items()) and payload.get("array_sha256") == file_sha256(array_path):
            covariances = np.load(array_path, allow_pickle=False)
            if covariances.shape[1:] == (64, 64) and np.isfinite(covariances).all():
                return SubjectCovarianceData(subject, covariances, source.labels.copy(), source.runs.copy(), list(source.identities), source.qc_candidates.copy()), True
    dataset = dataset_from_checkpoint(source, np.load(raw_array, allow_pickle=False))
    converted = covariance_data_from_dataset(dataset)
    np.save(array_path, converted.covariances, allow_pickle=False)
    payload = {"schema_version": 1, "complete": True, "subject": subject, **expected, "array_file": array_path.name, "array_sha256": file_sha256(array_path), "array_shape": list(converted.covariances.shape), "array_dtype": str(converted.covariances.dtype), "test_EEG_used_for_any_fit_at_checkpoint_creation": False, "source_hashes": {identity.source_file: identity.source_sha256 for identity in converted.identities}}
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return converted, False


def historical_group(rows: list[dict[str, str]], model: str) -> dict[str, object]:
    selected = [row for row in rows if row["model"] == model]
    scores = np.asarray([float(row["primary_mean_run_balanced_accuracy"]) for row in selected])
    ranges = np.asarray([max(float(row[f"balanced_accuracy_run_{run}"]) for run in (6, 10, 14)) - min(float(row[f"balanced_accuracy_run_{run}"]) for run in (6, 10, 14)) for row in selected])
    return {"model": model, "participant_count": len(selected), "median_balanced_accuracy": float(np.median(scores)), "median_participant_run_range": float(np.median(ranges))}


def main() -> None:
    args = parse_arguments(); mne.set_log_level("ERROR"); verify_final_freeze_is_ancestor()
    final, initial = load_final_riemannian_config(args.config); final_hash = file_sha256(args.config.resolve())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cross = load_cross_subject_config()
    training = [covariance_data_from_dataset(load_cross_subject_data(subject, cross)) for subject in final["cohorts"]["training_subjects"]]
    fitted = fit_riemannian_model(training, "shrinkage_lda", initial)
    if fitted.training_trial_count != 900:
        raise RuntimeError("Final Riemannian training count drifted.")
    calibration_metadata = json.loads((args.output_dir / "target_calibration_evaluation_metadata.json").read_text())
    prediction_rows: list[dict[str, object]] = []; run_rows: list[dict[str, object]] = []; subject_rows: list[dict[str, object]] = []; audits: list[dict[str, object]] = []
    reuse_count = 0; evaluation_hashes: dict[str, str] = {}
    subjects = list(final["cohorts"]["evaluation_eligible_subjects"])
    for index, subject in enumerate(subjects, start=1):
        source, source_payload, _ = load_validated_source_features(args.source_checkpoint_dir, subject, calibration_metadata)
        converted, reused = checkpoint_data(subject, source, args.raw_checkpoint_dir, args.checkpoint_dir, final_hash, final["frozen_method_implementation"]["sha256"])
        reuse_count += int(reused); evaluation_hashes.update(source_payload["target_source_hashes"])
        result = predict_riemannian_target(fitted, converted)
        prediction_rows.extend(result.prediction_rows); run_rows.extend(result.run_rows); subject_rows.extend(result.subject_rows); audits.append(result.fit_audit_row)
        print(f"Riemannian evaluation: {'reused' if reused else 'calculated'} S{subject:03d} ({index}/{len(subjects)})")
    group_primary = summarize_riemannian_group(subject_rows, "shrinkage_lda", initial, expected_subject_count=86)
    group_qc = summarize_riemannian_group(subject_rows, "shrinkage_lda", initial, expected_subject_count=86, qc_sensitivity=True)
    import csv as _csv
    with (args.output_dir / "cross_subject_decoder_evaluation_subject_scores.csv").open(newline="", encoding="utf-8") as stream: historical = list(_csv.DictReader(stream))
    csp_group = historical_group(historical, "csp_4_empirical")
    spectral_group = historical_group(historical, "spectral_baseline_6")
    primary_rows = [row for row in subject_rows if not bool(row["qc_sensitivity"])]
    qc_rows = [row for row in subject_rows if bool(row["qc_sensitivity"])]
    csp_pairs, csp_summary = paired_riemannian_comparison(primary_rows, historical, historical_model="csp_4_empirical", expected_subject_count=86, qc_riemannian_rows=qc_rows)
    spectral_pairs, spectral_summary = paired_riemannian_comparison(primary_rows, historical, historical_model="spectral_baseline_6", expected_subject_count=86)
    outcome = classify_riemannian_result(group_primary, csp_summary, csp_group, initial)
    run_summary = run_summary_rows(run_rows, expected_subject_count=86)
    diagnostics = exploratory_variability_rows(subject_rows, historical)
    artifacts = {"predictions.csv": prediction_rows, "run_scores.csv": run_rows, "subject_scores.csv": subject_rows, "fit_audit.csv": audits, "group_summary.csv": [group_primary, group_qc], "paired_csp_subjects.csv": csp_pairs, "paired_spectral_subjects.csv": spectral_pairs, "paired_summary.csv": [csp_summary, spectral_summary], "run_summary.csv": run_summary, "exploratory_diagnostics.csv": diagnostics}
    hashes: dict[str, str] = {}
    for suffix, rows in artifacts.items():
        path = args.output_dir / f"{PREFIX}{suffix}"; write_csv(rows, path); hashes[path.name] = file_sha256(path)
    figure, axis = plt.subplots(figsize=(6.2, 4.4)); axis.hist([float(r["primary_mean_run_balanced_accuracy"]) for r in primary_rows], bins=14, color="#4477aa", alpha=.85); axis.axvline(.5, color="0.3", linestyle="--"); axis.set(xlabel="Participant balanced accuracy", ylabel="Participants", title="Riemannian zero-shot decoding"); axis.spines[["top", "right"]].set_visible(False); figure.tight_layout(); figure_path = args.output_dir / f"{PREFIX}score_distribution.png"; figure.savefig(figure_path, dpi=180); plt.close(figure); hashes[figure_path.name] = file_sha256(figure_path)
    metadata = {"schema_version": 1, "study_stage": "riemannian_final_evaluation_complete", "final_config_sha256": final_hash, "frozen_core_sha256": final["frozen_method_implementation"]["sha256"], "evaluation_subject_count": 86, "evaluation_trial_count": len(prediction_rows), "checkpoint_count": 86, "checkpoint_reuse_count_this_run": reuse_count, "training_trial_count": 900, "training_source_hashes": fitted.training_source_hashes, "evaluation_source_hashes": dict(sorted(evaluation_hashes.items())), "source_model_refitted": False, "target_reference_fit": False, "test_EEG_used_for_any_fit": False, "target_tangent_space_update": False, "historical_CSP_outputs_modified": False, "historical_spectral_outputs_modified": False, "interpretation_category": outcome, "artifact_sha256": hashes}
    (args.output_dir / f"{PREFIX}metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nRiemannian evaluation: median={group_primary['median_balanced_accuracy']:.3f}; CSP gain={csp_summary['median_paired_difference']:+.3f}; {outcome}.")

if __name__ == "__main__": main()
