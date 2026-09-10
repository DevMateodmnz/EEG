"""Explain run and participant heterogeneity in the completed CSP decoder."""

from __future__ import annotations

import argparse
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

from eeg_project.decoding_evaluation import (  # noqa: E402
    load_evaluation_subject_data,
    load_final_decoding_config,
)
from eeg_project.decoding_reliability import (  # noqa: E402
    DEFAULT_RELIABILITY_CONFIG_PATH,
    EVALUATION_PREFIX,
    class_recall_rows,
    file_sha256,
    historical_relationship_rows,
    load_reliability_config,
    quality_relationship_rows,
    read_csv,
    reliability_category_rows,
    run_shift_run_summary_rows,
    run_shift_summaries,
    run_summary_and_comparisons,
    subject_reliability_rows,
    subject_run_shift_rows,
    write_csv,
)
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY  # noqa: E402


PREFIX = "decoding_reliability_"
FREEZE_COMMIT = "68a75a6"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEFAULT_CHECKPOINT_DIRECTORY = PROJECT_ROOT / "outputs" / "decoding_reliability_checkpoints"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build post-hoc reliability diagnostics without changing the decoder."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_RELIABILITY_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY)
    return parser.parse_args()


def verify_freeze_is_ancestor() -> None:
    """Refuse new run-shift calculation before the diagnostic freeze is committed."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Reliability diagnostic freeze is not an ancestor of HEAD.")


def checkpoint_path(directory: Path, subject: int) -> Path:
    return directory / f"subject_{subject:03d}.json"


def valid_shift_checkpoint(
    path: Path, subject: int, config_hash: str
) -> dict[str, object] | None:
    """Reuse a shift checkpoint only when policy and all raw EDF bytes still match."""
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
        or payload.get("reliability_config_sha256") != config_hash
        or len(payload.get("run_shift_rows", [])) != 3
    ):
        return None
    for source_name, expected in payload.get("source_hashes", {}).items():
        source = Path(source_name)
        if not source.is_file() or file_sha256(source) != expected:
            return None
    if any(row.get("labels_used_for_distance") is not False for row in payload["run_shift_rows"]):
        return None
    return payload


def calculate_shift_checkpoint(
    subject: int,
    initial_config: Mapping[str, object],
    data_directory: Path,
    config_hash: str,
) -> dict[str, object]:
    """Load the existing 8–30 Hz representation without fitting a classifier."""
    dataset = load_evaluation_subject_data(
        subject, initial_config, data_directory=data_directory
    )
    source_hashes = {
        identity.source_file: identity.source_sha256 for identity in dataset.identities
    }
    rows = subject_run_shift_rows(
        subject, dataset.csp_task_data_volts, dataset.runs, source_hashes
    )
    return {
        "schema_version": 1,
        "complete": True,
        "subject": subject,
        "reliability_config_sha256": config_hash,
        "source_hashes": source_hashes,
        "run_shift_rows": rows,
        "class_labels_used": False,
        "classifier_refit": False,
    }


def create_run_distribution_figure(
    reliability: Sequence[Mapping[str, object]], path: Path
) -> None:
    values = np.array(
        [
            [float(row[f"balanced_accuracy_run_{run}"]) for run in (6, 10, 14)]
            for row in reliability
        ]
    )
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    for row in values:
        axis.plot((1, 2, 3), row, color="0.5", alpha=0.08, linewidth=0.8)
    axis.boxplot(values, tick_labels=["Run 6", "Run 10", "Run 14"], showfliers=False)
    rng = np.random.default_rng(20260830)
    colors = ("#2a9d8f", "#e76f51", "#457b9d")
    for index, (scores, color) in enumerate(zip(values.T, colors, strict=True), start=1):
        axis.scatter(
            index + rng.uniform(-0.09, 0.09, size=scores.size),
            scores,
            color=color,
            alpha=0.58,
            s=22,
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set(
        ylabel="Held-out balanced accuracy",
        ylim=(0.05, 1.03),
        title="Matched held-out-run decoding distributions",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_subject_reliability_figure(
    reliability: Sequence[Mapping[str, object]], path: Path
) -> None:
    rows = sorted(
        reliability,
        key=lambda row: float(row["historical_primary_mean_balanced_accuracy"]),
        reverse=True,
    )
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(12.0, 5.8))
    colors = {6: "#2a9d8f", 10: "#e76f51", 14: "#457b9d"}
    for run in (6, 10, 14):
        axis.scatter(
            x,
            [float(row[f"balanced_accuracy_run_{run}"]) for row in rows],
            s=17,
            alpha=0.75,
            color=colors[run],
            label=f"Run {run}",
        )
    axis.plot(
        x,
        [float(row["historical_primary_mean_balanced_accuracy"]) for row in rows],
        color="black",
        linewidth=1.1,
        label="Three-run mean",
    )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    tick_indices = np.arange(0, len(rows), 8)
    axis.set(
        xticks=tick_indices,
        xticklabels=[f"S{int(rows[index]['subject']):03d}" for index in tick_indices],
        xlabel="Participants sorted by fixed three-run mean",
        ylabel="Balanced accuracy",
        ylim=(0.05, 1.03),
        title="Three held-out runs reveal participant-level reliability",
    )
    axis.legend(frameon=False, ncols=4)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_category_figure(
    categories: Sequence[Mapping[str, object]], path: Path
) -> None:
    rows = list(categories[:4])
    labels = [str(row["category"]).replace("_runs_above_0_5", "") for row in rows]
    counts = [int(row["participant_count"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar(labels, counts, color="#2a9d8f")
    axis.bar_label(bars)
    axis.set(
        xlabel="Held-out runs above 0.50",
        ylabel="Participants",
        ylim=(0, max(counts) * 1.15),
        title="Run-consistency categories are descriptive, not biological labels",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_shift_figure(
    shift_rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    subjects = np.array([int(row["subject"]) for row in shift_rows])
    distances = np.array(
        [float(row["affine_invariant_covariance_distance"]) for row in shift_rows]
    )
    accuracies = np.array([float(row["held_out_balanced_accuracy"]) for row in shift_rows])
    centered_distance = distances.copy()
    centered_accuracy = accuracies.copy()
    for subject in np.unique(subjects):
        selected = subjects == subject
        centered_distance[selected] -= np.mean(centered_distance[selected])
        centered_accuracy[selected] -= np.mean(centered_accuracy[selected])
    figure, axis = plt.subplots(figsize=(7.6, 5.4))
    colors = {6: "#2a9d8f", 10: "#e76f51", 14: "#457b9d"}
    for run in (6, 10, 14):
        selected = np.array([int(row["test_run"]) == run for row in shift_rows])
        axis.scatter(
            centered_distance[selected],
            centered_accuracy[selected],
            color=colors[run],
            alpha=0.65,
            s=27,
            label=f"Run {run}",
        )
    coefficient = np.polyfit(centered_distance, centered_accuracy, deg=1)
    x_line = np.linspace(centered_distance.min(), centered_distance.max(), 100)
    axis.plot(x_line, np.polyval(coefficient, x_line), color="black", linewidth=1.1)
    axis.axhline(0, color="0.5", linewidth=0.8)
    axis.axvline(0, color="0.5", linewidth=0.8)
    axis.set(
        xlabel="Run-shift distance minus participant mean",
        ylabel="Held-out accuracy minus participant mean",
        title="Within-participant covariance shift versus held-out accuracy",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_recall_figure(
    reliability: Sequence[Mapping[str, object]], path: Path
) -> None:
    fists = np.array([float(row["out_of_fold_fists_recall"]) for row in reliability])
    feet = np.array([float(row["out_of_fold_feet_recall"]) for row in reliability])
    figure, axis = plt.subplots(figsize=(7.0, 5.2))
    for first, second in zip(fists, feet, strict=True):
        axis.plot((1, 2), (first, second), color="0.55", alpha=0.18, linewidth=0.8)
    axis.boxplot((fists, feet), tick_labels=["Fists", "Feet"], showfliers=False)
    rng = np.random.default_rng(20260830)
    axis.scatter(1 + rng.uniform(-0.07, 0.07, fists.size), fists, alpha=0.55, s=22)
    axis.scatter(2 + rng.uniform(-0.07, 0.07, feet.size), feet, alpha=0.55, s=22)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    axis.set(
        ylabel="Participant out-of-fold recall",
        ylim=(-0.03, 1.03),
        title="Matched class recall across all held-out predictions",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    mne.set_log_level("ERROR")
    verify_freeze_is_ancestor()
    config = load_reliability_config(args.config)
    final, initial = load_final_decoding_config()
    subjects = list(final["cohorts"]["decoder_evaluation_eligible_subjects"])
    if len(subjects) != config["cohort"]["eligible_subject_count"]:
        raise RuntimeError("Reliability and decoder evaluation cohorts differ.")
    assets = PROJECT_ROOT / "docs" / "assets"
    predictions = read_csv(assets / f"{EVALUATION_PREFIX}predictions.csv")
    folds = read_csv(assets / f"{EVALUATION_PREFIX}fold_scores.csv")
    historical_subjects = read_csv(assets / f"{EVALUATION_PREFIX}subject_scores.csv")
    historical_context = read_csv(
        assets / f"{EVALUATION_PREFIX}exploratory_subject_context.csv"
    )
    reliability = subject_reliability_rows(folds, historical_subjects, predictions)
    categories = reliability_category_rows(reliability)
    run_summary, run_comparisons = run_summary_and_comparisons(folds)
    recalls = class_recall_rows(folds, reliability)
    quality = quality_relationship_rows(reliability, folds)
    historical = historical_relationship_rows(reliability, historical_context)

    config_hash = file_sha256(args.config.resolve())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shift_rows: list[dict[str, object]] = []
    reused_count = 0
    source_hashes: dict[str, str] = {}
    for index, subject in enumerate(subjects, start=1):
        path = checkpoint_path(args.checkpoint_dir, subject)
        checkpoint = valid_shift_checkpoint(path, subject, config_hash)
        reused = checkpoint is not None
        if checkpoint is None:
            checkpoint = calculate_shift_checkpoint(
                subject, initial, args.data_dir, config_hash
            )
            path.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        else:
            reused_count += 1
        shift_rows.extend(checkpoint["run_shift_rows"])
        source_hashes.update(checkpoint["source_hashes"])
        print(
            f"Reliability shift: subject {subject:03d} ({index}/{len(subjects)}); "
            f"checkpoint={'reused' if reused else 'computed'}"
        )
    joined_shift, shift_relationships = run_shift_summaries(shift_rows, folds, config)
    shift_run_summary = run_shift_run_summary_rows(joined_shift)

    output = args.output_dir
    artifacts: dict[str, Path] = {
        "subject_reliability": output / f"{PREFIX}subject_reliability.csv",
        "category_summary": output / f"{PREFIX}category_summary.csv",
        "run_summary": output / f"{PREFIX}run_summary.csv",
        "run_comparisons": output / f"{PREFIX}run_comparisons.csv",
        "class_recalls": output / f"{PREFIX}class_recalls.csv",
        "quality_relationships": output / f"{PREFIX}quality_relationships.csv",
        "run_shift_folds": output / f"{PREFIX}run_shift_folds.csv",
        "run_shift_run_summary": output / f"{PREFIX}run_shift_run_summary.csv",
        "run_shift_relationships": output / f"{PREFIX}run_shift_relationships.csv",
        "historical_relationships": output / f"{PREFIX}historical_relationships.csv",
    }
    for key, rows in (
        ("subject_reliability", reliability),
        ("category_summary", categories),
        ("run_summary", run_summary),
        ("run_comparisons", run_comparisons),
        ("class_recalls", recalls),
        ("quality_relationships", quality),
        ("run_shift_folds", joined_shift),
        ("run_shift_run_summary", shift_run_summary),
        ("run_shift_relationships", shift_relationships),
        ("historical_relationships", historical),
    ):
        write_csv(rows, artifacts[key])
    figures = {
        "run_distribution": output / f"{PREFIX}held_out_runs.png",
        "subject_reliability_figure": output / f"{PREFIX}subject_three_runs.png",
        "category_figure": output / f"{PREFIX}above_chance_run_counts.png",
        "shift_figure": output / f"{PREFIX}run_shift_vs_accuracy.png",
        "recall_figure": output / f"{PREFIX}class_recall.png",
    }
    create_run_distribution_figure(reliability, figures["run_distribution"])
    create_subject_reliability_figure(reliability, figures["subject_reliability_figure"])
    create_category_figure(categories, figures["category_figure"])
    create_shift_figure(joined_shift, figures["shift_figure"])
    create_recall_figure(reliability, figures["recall_figure"])

    evaluation_metadata_path = assets / f"{EVALUATION_PREFIX}metadata.json"
    evaluation_metadata = json.loads(evaluation_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 1,
        "analysis_role": "post_hoc_explanation_of_completed_held_out_study",
        "reliability_freeze_git_commit": FREEZE_COMMIT,
        "reliability_config_sha256": config_hash,
        "parent_completed_decoder_commit": config["parent_completed_milestone_git_commit"],
        "participant_count": len(reliability),
        "run_shift_fold_count": len(joined_shift),
        "run_shift_checkpoint_count": len(subjects),
        "run_shift_checkpoint_reuse_count_this_run": reused_count,
        "source_edf_count": len(source_hashes),
        "source_hashes": source_hashes,
        "historical_prediction_sha256": file_sha256(
            assets / f"{EVALUATION_PREFIX}predictions.csv"
        ),
        "historical_fold_score_sha256": file_sha256(
            assets / f"{EVALUATION_PREFIX}fold_scores.csv"
        ),
        "historical_subject_score_sha256": file_sha256(
            assets / f"{EVALUATION_PREFIX}subject_scores.csv"
        ),
        "historical_decoder_median_balanced_accuracy": next(
            row["median_subject_balanced_accuracy"]
            for row in read_csv(assets / f"{EVALUATION_PREFIX}group_summary.csv")
            if row["model"] == "csp_4_empirical"
        ),
        "historical_decoder_category": evaluation_metadata["decoding_category"],
        "historical_predictions_modified": False,
        "classifier_refit": False,
        "classifier_hyperparameters_changed": False,
        "run_shift_class_labels_used": False,
        "cross_subject_decoding_performed": False,
        "deep_learning_performed": False,
        "csp_stability_status": config["csp_stability"]["status"],
        "artifact_sha256": {
            path.name: file_sha256(path) for path in [*artifacts.values(), *figures.values()]
        },
    }
    metadata_path = output / f"{PREFIX}metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nReliability summary:")
    for row in run_summary:
        print(
            f"  run {row['test_run']}: median={row['median_balanced_accuracy']:.3f}; "
            f"IQR={row['q25_balanced_accuracy']:.3f}–{row['q75_balanced_accuracy']:.3f}"
        )
    print(
        "  above-chance runs: "
        + ", ".join(
            f"{row['category'].split('_')[0]}/3={row['participant_count']}"
            for row in categories[:4]
        )
    )
    print(
        f"  run shift centered r={shift_relationships[0]['statistic']:.3f}; "
        f"permutation p={shift_relationships[0]['p_value']:.4g}"
    )


if __name__ == "__main__":
    main()
