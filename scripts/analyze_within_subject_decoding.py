"""Develop and later evaluate frozen within-subject motor-imagery decoders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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

from eeg_project.decoding import (  # noqa: E402
    DEFAULT_DECODING_CONFIG_PATH,
    build_candidate_pipeline,
    evaluate_candidate,
    leave_one_run_out_folds,
    load_decoding_config,
    load_subject_decoding_data,
    select_development_model,
    sha256,
    summarize_development_candidates,
)
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY  # noqa: E402


DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "assets"
DEVELOPMENT_PREFIX = "within_subject_decoder_development"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run explicitly separated within-subject decoder study stages."
    )
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--develop", action="store_true")
    stage.add_argument("--evaluate", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_DECODING_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write stable field-order CSV without an index."""
    if not rows:
        raise ValueError(f"Cannot write empty decoder artifact: {path}.")
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


def create_representative_csp_pattern_figure(
    subject_rows: Sequence[Mapping[str, object]],
    selected_csp: str,
    config: Mapping[str, object],
    data_directory: Path,
    path: Path,
) -> tuple[int, str]:
    """Plot fixed-fold patterns for the subject nearest the development median."""
    rows = [
        row
        for row in subject_rows
        if row["model"] == selected_csp and not bool(row["qc_sensitivity"])
    ]
    median_score = float(
        np.median([float(row["primary_mean_fold_balanced_accuracy"]) for row in rows])
    )
    representative = min(
        rows,
        key=lambda row: (
            abs(float(row["primary_mean_fold_balanced_accuracy"]) - median_score),
            int(row["subject"]),
        ),
    )
    subject = int(representative["subject"])
    dataset = load_subject_decoding_data(subject, config, data_directory=data_directory)
    fold = leave_one_run_out_folds(dataset.runs)[0]
    pipeline = build_candidate_pipeline(selected_csp, config)
    pipeline.fit(
        dataset.csp_task_data_volts[fold.train_indices], dataset.labels[fold.train_indices]
    )
    csp = pipeline.named_steps["csp"]
    info = mne.create_info(
        dataset.channel_names, dataset.sampling_frequency_hz, ch_types="eeg"
    )
    info.set_montage("standard_1005", match_case=False, on_missing="raise")
    figure = csp.plot_patterns(
        info,
        components=np.arange(int(config["candidate_models"][selected_csp]["csp_components"])),
        ch_type="eeg",
        show=False,
    )
    figure.suptitle(
        f"Development-representative CSP sensor patterns — S{subject:03d}, "
        "train runs 10+14, hold run 6",
        fontsize=12,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return subject, "closest_to_development_cohort_median_then_lowest_subject_id"


def run_development(args: argparse.Namespace, config: Mapping[str, object]) -> None:
    """Calculate candidate outcomes for subjects 1–20 and nobody else."""
    if config["study_stage"] != "initial_methodology_frozen":
        raise RuntimeError("Development requires the immutable initial methodology freeze.")
    if not config["evaluation_locked"]:
        raise RuntimeError("Evaluation must remain locked during decoder development.")
    subjects = list(config["cohorts"]["decoder_development_subjects"])
    if subjects != list(range(1, 21)):
        raise RuntimeError("Development CLI refuses a non-frozen participant set.")
    candidate_names = list(config["candidate_models"])
    subject_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for subject_index, subject in enumerate(subjects, start=1):
        dataset = load_subject_decoding_data(
            subject, config, data_directory=args.data_dir
        )
        for candidate_name in candidate_names:
            primary = evaluate_candidate(dataset, candidate_name, config)
            sensitivity = evaluate_candidate(
                dataset, candidate_name, config, qc_excluded=True
            )
            subject_rows.extend((primary.subject_row, sensitivity.subject_row))
            fold_rows.extend(primary.fold_rows)
            fold_rows.extend(sensitivity.fold_rows)
        print(
            f"Decoder development: subject {subject:03d} "
            f"({subject_index}/{len(subjects)}); candidates={len(candidate_names)}"
        )
    summary_rows = summarize_development_candidates(subject_rows)
    qc_lookup = {
        str(row["model"]): row
        for row in summarize_development_candidates(
            subject_rows, qc_sensitivity=True
        )
    }
    if not qc_lookup:
        raise RuntimeError("QC development summaries were not produced.")
    selected_baseline = select_development_model(
        summary_rows, "spectral_", config
    )
    selected_csp = select_development_model(summary_rows, "csp_", config)
    augmented: list[dict[str, object]] = []
    for row in summary_rows:
        qc_row = qc_lookup[str(row["model"])]
        augmented.append(
            {
                **row,
                "qc_excluded_median_subject_balanced_accuracy": qc_row[
                    "median_subject_balanced_accuracy"
                ],
                "qc_minus_primary_median_balanced_accuracy": float(
                    qc_row["median_subject_balanced_accuracy"]
                )
                - float(row["median_subject_balanced_accuracy"]),
                "selected_final_family_candidate": str(row["model"])
                in {selected_baseline, selected_csp},
            }
        )
    output = args.output_dir
    subject_path = output / f"{DEVELOPMENT_PREFIX}_subject_scores.csv"
    fold_path = output / f"{DEVELOPMENT_PREFIX}_fold_scores.csv"
    summary_path = output / f"{DEVELOPMENT_PREFIX}_candidate_summary.csv"
    write_csv(subject_rows, subject_path)
    write_csv(fold_rows, fold_path)
    write_csv(augmented, summary_path)
    pattern_path = output / f"{DEVELOPMENT_PREFIX}_representative_csp_patterns.png"
    representative_subject, representative_rule = create_representative_csp_pattern_figure(
        subject_rows, selected_csp, config, args.data_dir, pattern_path
    )
    metadata = {
        "schema_version": 1,
        "study_stage": "decoder_method_development_only",
        "configuration_sha256": sha256(args.config.resolve()),
        "development_subjects": subjects,
        "evaluation_subject_classifier_outcomes_calculated": False,
        "candidate_count": len(candidate_names),
        "selected_spectral_baseline": selected_baseline,
        "selected_csp_model": selected_csp,
        "representative_csp_pattern_subject": representative_subject,
        "representative_csp_pattern_selection_rule": representative_rule,
        "representative_csp_pattern_test_run": 6,
        "selection_rule": config["development_selection"],
        "artifacts": {
            subject_path.name: sha256(subject_path),
            fold_path.name: sha256(fold_path),
            summary_path.name: sha256(summary_path),
            pattern_path.name: sha256(pattern_path),
        },
        "full_feature_arrays_persisted": False,
        "fitted_models_persisted": False,
    }
    metadata_path = output / f"{DEVELOPMENT_PREFIX}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nDevelopment candidate summary:")
    for row in augmented:
        print(
            f"  {row['model']}: median={row['median_subject_balanced_accuracy']:.3f}; "
            f"IQR={row['q25_subject_balanced_accuracy']:.3f}–"
            f"{row['q75_subject_balanced_accuracy']:.3f}; "
            f">0.5={row['fraction_subjects_above_0_5']:.1%}"
        )
    print(f"  Selected spectral baseline: {selected_baseline}")
    print(f"  Selected CSP model: {selected_csp}")


def main() -> None:
    args = parse_arguments()
    mne.set_log_level("ERROR")
    config = load_decoding_config(args.config)
    if args.develop:
        run_development(args, config)
        return
    if config["evaluation_locked"]:
        raise RuntimeError(
            "Evaluation remains locked. Commit a final decoder-method freeze first."
        )
    raise RuntimeError("Evaluation implementation is intentionally unavailable before final freeze.")


if __name__ == "__main__":
    main()
