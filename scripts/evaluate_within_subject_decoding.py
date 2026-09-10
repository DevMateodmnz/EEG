"""Evaluate final-frozen within-subject decoders on subjects 21–109."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eeg_project.decoding import evaluate_candidate  # noqa: E402
from eeg_project.decoding_evaluation import (  # noqa: E402
    DEFAULT_ACTIVE_DECODING_CONFIG_PATH,
    aggregate_confusions,
    classify_csp_added_value,
    classify_decoding_signal,
    error_analysis_rows,
    file_sha256,
    load_final_decoding_config,
    load_evaluation_subject_data,
    model_comparison_summary,
    paired_model_rows,
    summarize_model_group,
)
from eeg_project.decoding_post_evaluation import (  # noqa: E402
    safe_exploratory_relationship_rows,
)
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY  # noqa: E402


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "within_subject_decoder_checkpoints"
PREFIX = "within_subject_decoder_evaluation"
FINAL_FREEZE_COMMIT = "7bbd249"
ACTIVE_CORRECTION_FREEZE_COMMIT = "be524e0"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two final-frozen within-subject decoders out of run."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_ACTIVE_DECODING_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write deterministic compact evaluation tables."""
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
    """Refuse evaluation if Git chronology lacks the committed final freeze."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ACTIVE_CORRECTION_FREEZE_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Final decoder freeze is not an ancestor of the current checkout.")


def checkpoint_path(directory: Path, subject: int) -> Path:
    return directory / f"subject_{subject:03d}.json"


def valid_checkpoint(
    path: Path,
    subject: int,
    final_hash: str,
    core_hash: str,
) -> dict[str, object] | None:
    """Load a compact result only if method and every EDF hash remain exact."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        payload.get("schema_version") != 1
        or payload.get("complete") is not True
        or payload.get("subject") != subject
        or payload.get("final_config_sha256") != final_hash
        or payload.get("frozen_core_sha256") != core_hash
    ):
        return None
    for source, expected in payload.get("source_hashes", {}).items():
        path = Path(source)
        if not path.is_file() or file_sha256(path) != expected:
            return None
    return payload


def calculate_subject(
    subject: int,
    initial: Mapping[str, object],
    model_names: Sequence[str],
    data_directory: Path,
    final_hash: str,
    core_hash: str,
) -> dict[str, object]:
    """Fit both models in primary and copied-QC folds for one participant."""
    dataset = load_evaluation_subject_data(subject, initial, data_directory=data_directory)
    predictions: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    subjects: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for model in model_names:
        for qc_excluded in (False, True):
            result = evaluate_candidate(
                dataset, model, initial, qc_excluded=qc_excluded
            )
            predictions.extend(result.prediction_rows)
            folds.extend(result.fold_rows)
            subjects.append(result.subject_row)
            audits.extend(result.fit_audit_rows)
    source_hashes = {
        identity.source_file: identity.source_sha256 for identity in dataset.identities
    }
    return {
        "schema_version": 1,
        "complete": True,
        "subject": subject,
        "final_config_sha256": final_hash,
        "frozen_core_sha256": core_hash,
        "source_hashes": source_hashes,
        "prediction_rows": predictions,
        "fold_rows": folds,
        "subject_rows": subjects,
        "fit_audit_rows": audits,
    }


def create_distribution_figure(
    subject_rows: Sequence[Mapping[str, object]], model_names: Sequence[str], path: Path
) -> None:
    primary = [row for row in subject_rows if not row["qc_sensitivity"]]
    values = [
        np.array(
            [float(row["primary_mean_fold_balanced_accuracy"]) for row in primary if row["model"] == model]
        )
        for model in model_names
    ]
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.boxplot(values, tick_labels=["Spectral + LDA", "CSP + LDA"], showfliers=False)
    rng = np.random.default_rng(20260829)
    for index, scores in enumerate(values, start=1):
        jitter = rng.uniform(-0.10, 0.10, size=scores.size)
        axis.scatter(index + jitter, scores, alpha=0.58, s=25)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance balanced accuracy")
    axis.axhline(0.6, color="grey", linestyle=":", linewidth=1, label="0.60 descriptive threshold")
    axis.set(ylabel="Mean held-out-run balanced accuracy", ylim=(0.15, 1.02), title="Held-out within-subject decoding distribution")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_paired_figure(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    baseline = np.array([float(row["baseline_balanced_accuracy"]) for row in rows])
    csp = np.array([float(row["csp_balanced_accuracy"]) for row in rows])
    figure, axis = plt.subplots(figsize=(6.4, 6.0))
    axis.scatter(baseline, csp, color="#2a9d8f", alpha=0.72, edgecolor="white", linewidth=0.4)
    limits = (min(baseline.min(), csp.min()) - 0.03, max(baseline.max(), csp.max()) + 0.03)
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axis.axhline(0.5, color="grey", linewidth=0.8)
    axis.axvline(0.5, color="grey", linewidth=0.8)
    axis.set(
        xlim=limits,
        ylim=limits,
        xlabel="Spectral baseline balanced accuracy",
        ylabel="CSP balanced accuracy",
        title="Matched participant model comparison",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_run_heatmap(
    subject_rows: Sequence[Mapping[str, object]], csp_name: str, path: Path
) -> None:
    rows = [row for row in subject_rows if row["model"] == csp_name and not row["qc_sensitivity"]]
    rows.sort(key=lambda row: float(row["primary_mean_fold_balanced_accuracy"]), reverse=True)
    matrix = np.array(
        [
            [
                float(row["fold_balanced_accuracy_run_6"]),
                float(row["fold_balanced_accuracy_run_10"]),
                float(row["fold_balanced_accuracy_run_14"]),
            ]
            for row in rows
        ]
    )
    figure, axis = plt.subplots(figsize=(6.2, 11.5))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlBu")
    axis.set(
        xticks=[0, 1, 2],
        xticklabels=["Run 6", "Run 10", "Run 14"],
        yticks=np.arange(len(rows))[::5],
        yticklabels=[f"S{int(rows[index]['subject']):03d}" for index in range(0, len(rows), 5)],
        xlabel="Held-out run",
        ylabel="Participants sorted by mean score",
        title="CSP performance varies across people and runs",
    )
    figure.colorbar(image, ax=axis, label="Balanced accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_confusion_figure(
    rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    primary = [row for row in rows if not row["qc_sensitivity"]]
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), constrained_layout=True)
    for axis, row in zip(axes, primary, strict=True):
        matrix = np.array(
            [
                [row["true_fists_predicted_fists"], row["true_fists_predicted_feet"]],
                [row["true_feet_predicted_fists"], row["true_feet_predicted_feet"]],
            ],
            dtype=float,
        )
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for y in range(2):
            for x in range(2):
                axis.text(x, y, f"{int(matrix[y, x])}\n{normalized[y, x]:.1%}", ha="center", va="center")
        axis.set(
            xticks=[0, 1], xticklabels=["Fists", "Feet"],
            yticks=[0, 1], yticklabels=["Fists", "Feet"],
            xlabel="Predicted", ylabel="True", title=str(row["model"]),
        )
    figure.colorbar(
        image, ax=axes, label="Row-normalized proportion", shrink=0.82, pad=0.03
    )
    figure.suptitle("Cohort confusion matrices — primary retained trials")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_exploratory_figure(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, field, title in (
        (axes[0], "c4_fixed_fists_erd_median_percent", "Historical C4 fixed ERD"),
        (axes[1], "c3_fixed_fists_erd_median_percent", "Historical C3 fixed ERD"),
    ):
        x = np.array([float(row[field]) for row in rows])
        y = np.array([float(row["csp_balanced_accuracy"]) for row in rows])
        rho = float(spearmanr(x, y).statistic)
        axis.scatter(x, y, color="#2a9d8f", alpha=0.7)
        axis.axhline(0.5, color="black", linewidth=0.8)
        axis.set(xlabel="Fists task vs rest median (%)", ylabel="CSP balanced accuracy", title=f"{title}: ρ={rho:.2f}")
    figure.suptitle("Post-evaluation exploratory physiology–decoding relationships")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    mne.set_log_level("ERROR")
    verify_final_freeze_is_ancestor()
    final, initial = load_final_decoding_config(args.config)
    final_hash = file_sha256(args.config.resolve())
    core_hash = final["frozen_method_implementation"]["core"]["sha256"]
    subjects = list(final["cohorts"]["decoder_evaluation_eligible_subjects"])
    model_names = (
        final["selected_spectral_baseline"]["candidate_name"],
        final["selected_csp_decoder"]["candidate_name"],
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    checkpoint_reuse_count = 0
    for index, subject in enumerate(subjects, start=1):
        path = checkpoint_path(args.checkpoint_dir, subject)
        payload = valid_checkpoint(path, subject, final_hash, core_hash)
        reused = payload is not None
        if payload is None:
            payload = calculate_subject(
                subject, initial, model_names, args.data_dir, final_hash, core_hash
            )
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        else:
            checkpoint_reuse_count += 1
        prediction_rows.extend(payload["prediction_rows"])
        fold_rows.extend(payload["fold_rows"])
        subject_rows.extend(payload["subject_rows"])
        audit_rows.extend(payload["fit_audit_rows"])
        print(
            f"Decoder evaluation: subject {subject:03d} ({index}/{len(subjects)}); "
            f"checkpoint={'reused' if reused else 'computed'}"
        )
    baseline_name, csp_name = model_names
    paired_rows = paired_model_rows(subject_rows, baseline_name, csp_name)
    comparison = model_comparison_summary(paired_rows)
    comparison["csp_added_value_category"] = classify_csp_added_value(comparison, final)
    group_rows = [
        summarize_model_group(subject_rows, model, final) for model in model_names
    ]
    csp_group = next(row for row in group_rows if row["model"] == csp_name)
    decoding_category = classify_decoding_signal(
        csp_group,
        float(comparison["median_csp_minus_baseline_balanced_accuracy"]),
        final,
    )
    for row in group_rows:
        row["frozen_decoding_category"] = decoding_category if row["model"] == csp_name else "secondary baseline"
    confusion_rows = aggregate_confusions(prediction_rows)
    errors = error_analysis_rows(prediction_rows, fold_rows, subject_rows)
    csp_subject_rows = [row for row in subject_rows if row["model"] == csp_name]
    exploratory_subjects, relationships = safe_exploratory_relationship_rows(
        csp_subject_rows
    )
    output = args.output_dir
    artifacts: dict[str, Path] = {
        "predictions": output / f"{PREFIX}_predictions.csv",
        "fold_scores": output / f"{PREFIX}_fold_scores.csv",
        "subject_scores": output / f"{PREFIX}_subject_scores.csv",
        "fit_audit": output / f"{PREFIX}_fit_audit.csv",
        "group_summary": output / f"{PREFIX}_group_summary.csv",
        "paired_models": output / f"{PREFIX}_paired_model_comparison.csv",
        "model_comparison_summary": output / f"{PREFIX}_model_comparison_summary.csv",
        "confusion": output / f"{PREFIX}_confusion_matrices.csv",
        "error_analysis": output / f"{PREFIX}_error_analysis.csv",
        "exploratory_subjects": output / f"{PREFIX}_exploratory_subject_context.csv",
        "exploratory_relationships": output / f"{PREFIX}_exploratory_relationships.csv",
    }
    for key, rows in (
        ("predictions", prediction_rows), ("fold_scores", fold_rows),
        ("subject_scores", subject_rows), ("fit_audit", audit_rows),
        ("group_summary", group_rows), ("paired_models", paired_rows),
        ("model_comparison_summary", [comparison]), ("confusion", confusion_rows),
        ("error_analysis", errors), ("exploratory_subjects", exploratory_subjects),
        ("exploratory_relationships", relationships),
    ):
        write_csv(rows, artifacts[key])
    figure_paths = {
        "distribution_figure": output / f"{PREFIX}_score_distribution.png",
        "paired_figure": output / f"{PREFIX}_paired_models.png",
        "run_heatmap": output / f"{PREFIX}_run_heatmap.png",
        "confusion_figure": output / f"{PREFIX}_confusion_matrices.png",
        "exploratory_figure": output / f"{PREFIX}_exploratory_erd_relationships.png",
    }
    create_distribution_figure(subject_rows, model_names, figure_paths["distribution_figure"])
    create_paired_figure(paired_rows, figure_paths["paired_figure"])
    create_run_heatmap(subject_rows, csp_name, figure_paths["run_heatmap"])
    create_confusion_figure(confusion_rows, figure_paths["confusion_figure"])
    create_exploratory_figure(exploratory_subjects, figure_paths["exploratory_figure"])
    primary_predictions = [row for row in prediction_rows if not row["qc_sensitivity"]]
    source_hashes = {
        str(row["source_file"]): str(row["source_sha256"]) for row in primary_predictions
    }
    metadata = {
        "schema_version": 1,
        "study_start_git_commit": final["study_start_git_commit"],
        "final_decoder_freeze_git_commit": FINAL_FREEZE_COMMIT,
        "active_technical_correction_git_commit": ACTIVE_CORRECTION_FREEZE_COMMIT,
        "final_config_sha256": final_hash,
        "frozen_core_sha256": core_hash,
        "requested_evaluation_subject_count": len(final["cohorts"]["decoder_evaluation_requested_subjects"]),
        "technically_eligible_evaluation_subject_count": len(subjects),
        "technically_incompatible_subjects": final["cohorts"]["technically_incompatible_subjects"],
        "primary_task_trial_count_per_model": len(primary_predictions) // len(model_names),
        "primary_prediction_count_all_models": len(primary_predictions),
        "qc_prediction_count_all_models": len(prediction_rows) - len(primary_predictions),
        "source_edf_count": len(source_hashes),
        "source_hashes": source_hashes,
        "checkpoint_count": len(subjects),
        "checkpoint_reuse_count_this_run": checkpoint_reuse_count,
        "selected_models": list(model_names),
        "decoding_category": decoding_category,
        "csp_added_value_category": comparison["csp_added_value_category"],
        "every_trial_predicted_once_out_of_run": True,
        "all_fit_audits_exclude_test_run": all(row["test_run_absent_from_all_fit_stages"] for row in audit_rows),
        "all_fit_audits_preserve_subject_isolation": all(row["subject_isolation_preserved"] for row in audit_rows),
        "evaluation_outcomes_used_for_hyperparameters": False,
        "primary_and_qc_predictions_separate": True,
        "full_feature_arrays_persisted": False,
        "fitted_models_persisted": False,
        "cross_subject_classifier_performed": False,
        "deep_learning_performed": False,
        "artifact_sha256": {
            path.name: file_sha256(path) for path in [*artifacts.values(), *figure_paths.values()]
        },
    }
    metadata_path = output / f"{PREFIX}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nHeld-out decoder evaluation summary:")
    for row in group_rows:
        print(
            f"  {row['model']}: median={row['median_subject_balanced_accuracy']:.3f}; "
            f"IQR={row['q25_subject_balanced_accuracy']:.3f}–{row['q75_subject_balanced_accuracy']:.3f}; "
            f">0.5={row['fraction_subjects_above_0_5']:.1%}"
        )
    print(
        f"  CSP minus baseline median={comparison['median_csp_minus_baseline_balanced_accuracy']:.3f}; "
        f"{comparison['csp_added_value_category']}"
    )
    print(f"  Frozen decoding category: {decoding_category}")


if __name__ == "__main__":
    main()
