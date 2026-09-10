"""Post-hoc reliability diagnostics for the completed within-subject decoder."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import eigvalsh
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from sklearn.covariance import LedoitWolf

from .decoding import PROJECT_ROOT, RUNS


DEFAULT_RELIABILITY_CONFIG_PATH = PROJECT_ROOT / "config" / "decoding_reliability.json"
EVALUATION_PREFIX = "within_subject_decoder_evaluation_"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a project CSV as string-valued dictionaries."""
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Write deterministic CSV rows while preserving first-seen field order."""
    if not rows:
        raise ValueError(f"Cannot write empty reliability artifact: {path}.")
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


def load_reliability_config(
    path: Path = DEFAULT_RELIABILITY_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the diagnostic freeze and verify every historical input byte."""
    config_path = path.expanduser().resolve()
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    if config["study_stage"] != "post_evaluation_reliability_diagnostics_frozen":
        raise RuntimeError("Reliability analysis requires its frozen diagnostic policy.")
    if not config["historical_decoder_result_may_not_change"]:
        raise RuntimeError("Historical decoder results must remain immutable.")
    for record in config["historical_inputs"].values():
        historical_path = PROJECT_ROOT / record["path"]
        if file_sha256(historical_path) != record["sha256"]:
            raise RuntimeError(f"Historical reliability input drift: {historical_path}.")
    return config


def primary_csp_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    """Select only the completed primary CSP evaluation rows."""
    return [
        row
        for row in rows
        if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"
    ]


def subject_reliability_rows(
    fold_rows: Sequence[Mapping[str, str]],
    subject_rows: Sequence[Mapping[str, str]],
    prediction_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Summarize the three fixed held-out runs for every evaluation participant."""
    primary_folds = primary_csp_rows(fold_rows)
    primary_subjects = primary_csp_rows(subject_rows)
    primary_predictions = primary_csp_rows(prediction_rows)
    qc_subjects = {
        int(row["subject"]): row
        for row in subject_rows
        if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "True"
    }
    folds_by_subject: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in primary_folds:
        folds_by_subject[int(row["subject"])].append(row)
    predictions_by_subject: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    for row in primary_predictions:
        predictions_by_subject[int(row["subject"])].append(row)
    output: list[dict[str, object]] = []
    for subject_row in sorted(primary_subjects, key=lambda row: int(row["subject"])):
        subject = int(subject_row["subject"])
        folds = sorted(folds_by_subject[subject], key=lambda row: int(row["test_run"]))
        if [int(row["test_run"]) for row in folds] != list(RUNS):
            raise RuntimeError(f"Subject {subject} does not have the three frozen runs.")
        scores = np.array([float(row["balanced_accuracy"]) for row in folds])
        above = int(np.sum(scores > 0.5))
        trials = predictions_by_subject[subject]
        if len({row["trial_key"] for row in trials}) != len(trials):
            raise RuntimeError(f"Subject {subject} primary predictions are not unique.")
        qc_count = sum(row["qc_candidate"] == "True" for row in trials)
        qc_row = qc_subjects[subject]
        by_run = {int(row["test_run"]): row for row in folds}
        qc_by_run = {
            run: sum(
                row["qc_candidate"] == "True" for row in trials if int(row["run"]) == run
            )
            for run in RUNS
        }
        trial_count_by_run = {
            run: sum(int(row["run"]) == run for row in trials) for run in RUNS
        }
        stored_mean = float(subject_row["primary_mean_fold_balanced_accuracy"])
        if not np.isclose(stored_mean, np.mean(scores), atol=1e-15):
            raise RuntimeError(f"Subject {subject} historical score changed.")
        output.append(
            {
                "subject": subject,
                "historical_primary_mean_balanced_accuracy": stored_mean,
                "three_run_median_balanced_accuracy": float(np.median(scores)),
                "minimum_run_balanced_accuracy": float(np.min(scores)),
                "maximum_run_balanced_accuracy": float(np.max(scores)),
                "run_balanced_accuracy_range": float(np.ptp(scores)),
                "run_balanced_accuracy_population_sd": float(np.std(scores, ddof=0)),
                "runs_above_0_5_count": above,
                "above_chance_run_category": f"{above}_of_3",
                "all_runs_at_or_above_0_6": bool(np.all(scores >= 0.6)),
                "run_range_at_or_above_0_25": bool(np.ptp(scores) >= 0.25),
                "one_bad_run_plus_two_good_runs": bool(stored_mean <= 0.5 and above == 2),
                "balanced_accuracy_run_6": float(by_run[6]["balanced_accuracy"]),
                "balanced_accuracy_run_10": float(by_run[10]["balanced_accuracy"]),
                "balanced_accuracy_run_14": float(by_run[14]["balanced_accuracy"]),
                "fists_recall_run_6": float(by_run[6]["fists_recall"]),
                "fists_recall_run_10": float(by_run[10]["fists_recall"]),
                "fists_recall_run_14": float(by_run[14]["fists_recall"]),
                "feet_recall_run_6": float(by_run[6]["feet_recall"]),
                "feet_recall_run_10": float(by_run[10]["feet_recall"]),
                "feet_recall_run_14": float(by_run[14]["feet_recall"]),
                "out_of_fold_fists_recall": float(subject_row["out_of_fold_fists_recall"]),
                "out_of_fold_feet_recall": float(subject_row["out_of_fold_feet_recall"]),
                "feet_minus_fists_recall": float(subject_row["out_of_fold_feet_recall"])
                - float(subject_row["out_of_fold_fists_recall"]),
                "task_qc_candidate_count": qc_count,
                "task_trial_count": len(trials),
                "task_qc_candidate_fraction": qc_count / len(trials),
                "task_qc_fraction_run_6": qc_by_run[6] / trial_count_by_run[6],
                "task_qc_fraction_run_10": qc_by_run[10] / trial_count_by_run[10],
                "task_qc_fraction_run_14": qc_by_run[14] / trial_count_by_run[14],
                "qc_sensitivity_mean_balanced_accuracy": float(
                    qc_row["primary_mean_fold_balanced_accuracy"]
                ),
                "qc_minus_primary_balanced_accuracy": float(
                    qc_row["primary_mean_fold_balanced_accuracy"]
                )
                - stored_mean,
                "analysis_role": "post_hoc_descriptive_reliability",
            }
        )
    return output


def reliability_category_rows(
    subject_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Count the frozen 3/3 through 0/3 reliability categories."""
    counter = Counter(int(row["runs_above_0_5_count"]) for row in subject_rows)
    output = [
        {
            "category": f"{count}_of_3_runs_above_0_5",
            "participant_count": counter[count],
            "participant_fraction": counter[count] / len(subject_rows),
            "interpretation": "descriptive_reliability_category_not_biological_label",
        }
        for count in (3, 2, 1, 0)
    ]
    ranges = np.array([float(row["run_balanced_accuracy_range"]) for row in subject_rows])
    sds = np.array(
        [float(row["run_balanced_accuracy_population_sd"]) for row in subject_rows]
    )
    output.extend(
        [
            {
                "category": "run_range_distribution",
                "participant_count": len(subject_rows),
                "participant_fraction": 1.0,
                "median": float(np.median(ranges)),
                "q25": float(np.quantile(ranges, 0.25)),
                "q75": float(np.quantile(ranges, 0.75)),
                "minimum": float(np.min(ranges)),
                "maximum": float(np.max(ranges)),
                "interpretation": "balanced_accuracy_points",
            },
            {
                "category": "run_population_sd_distribution",
                "participant_count": len(subject_rows),
                "participant_fraction": 1.0,
                "median": float(np.median(sds)),
                "q25": float(np.quantile(sds, 0.25)),
                "q75": float(np.quantile(sds, 0.75)),
                "minimum": float(np.min(sds)),
                "maximum": float(np.max(sds)),
                "interpretation": "balanced_accuracy_points",
            },
            {
                "category": "run_range_at_or_above_0_25",
                "participant_count": sum(
                    bool(row["run_range_at_or_above_0_25"]) for row in subject_rows
                ),
                "participant_fraction": sum(
                    bool(row["run_range_at_or_above_0_25"]) for row in subject_rows
                )
                / len(subject_rows),
                "interpretation": "predefined_descriptive_high_spread_threshold",
            },
            {
                "category": "all_three_runs_at_or_above_0_6",
                "participant_count": sum(
                    bool(row["all_runs_at_or_above_0_6"]) for row in subject_rows
                ),
                "participant_fraction": sum(
                    bool(row["all_runs_at_or_above_0_6"]) for row in subject_rows
                )
                / len(subject_rows),
                "interpretation": "predefined_descriptive_consistently_high_threshold",
            },
            {
                "category": "mean_at_or_below_0_5_with_two_runs_above_0_5",
                "participant_count": sum(
                    bool(row["one_bad_run_plus_two_good_runs"]) for row in subject_rows
                ),
                "participant_fraction": sum(
                    bool(row["one_bad_run_plus_two_good_runs"]) for row in subject_rows
                )
                / len(subject_rows),
                "interpretation": "one_bad_run_explanation_for_low_overall_score",
            },
        ]
    )
    low_rows = [
        row
        for row in subject_rows
        if float(row["historical_primary_mean_balanced_accuracy"]) <= 0.5
    ]
    for count in (0, 1, 2, 3):
        selected_count = sum(
            int(row["runs_above_0_5_count"]) == count for row in low_rows
        )
        output.append(
            {
                "category": f"mean_at_or_below_0_5_with_{count}_of_3_runs_above_0_5",
                "participant_count": selected_count,
                "participant_fraction": selected_count / len(low_rows),
                "denominator": len(low_rows),
                "interpretation": "pattern_within_low_overall_subjects",
            }
        )
    return output


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def run_summary_and_comparisons(
    fold_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize matched run performance and apply the frozen small test family."""
    primary = primary_csp_rows(fold_rows)
    lookup = {
        (int(row["subject"]), int(row["test_run"])): row for row in primary
    }
    subjects = sorted({int(row["subject"]) for row in primary})
    arrays = {
        run: np.array([float(lookup[(subject, run)]["balanced_accuracy"]) for subject in subjects])
        for run in RUNS
    }
    summaries: list[dict[str, object]] = []
    for run in RUNS:
        selected = [lookup[(subject, run)] for subject in subjects]
        values = arrays[run]
        fists_correct = sum(int(row["true_fists_predicted_fists"]) for row in selected)
        fists_total = fists_correct + sum(
            int(row["true_fists_predicted_feet"]) for row in selected
        )
        feet_correct = sum(int(row["true_feet_predicted_feet"]) for row in selected)
        feet_total = feet_correct + sum(
            int(row["true_feet_predicted_fists"]) for row in selected
        )
        summaries.append(
            {
                "test_run": run,
                "participant_count": len(subjects),
                "median_balanced_accuracy": float(np.median(values)),
                "q25_balanced_accuracy": float(np.quantile(values, 0.25)),
                "q75_balanced_accuracy": float(np.quantile(values, 0.75)),
                "minimum_balanced_accuracy": float(np.min(values)),
                "maximum_balanced_accuracy": float(np.max(values)),
                "median_participant_fists_recall": float(
                    np.median([float(row["fists_recall"]) for row in selected])
                ),
                "median_participant_feet_recall": float(
                    np.median([float(row["feet_recall"]) for row in selected])
                ),
                "pooled_fists_recall": fists_correct / fists_total,
                "pooled_feet_recall": feet_correct / feet_total,
                "analysis_role": "post_hoc_matched_run_diagnostic",
            }
        )
    omnibus = friedmanchisquare(*(arrays[run] for run in RUNS))
    comparisons: list[dict[str, object]] = [
        {
            "comparison": "runs_6_10_14_omnibus",
            "test": "friedman_two_sided_matched_subjects",
            "participant_count": len(subjects),
            "statistic": float(omnibus.statistic),
            "unadjusted_p": float(omnibus.pvalue),
            "holm_adjusted_p": "",
            "median_paired_difference_first_minus_second": "",
        }
    ]
    pair_results: list[tuple[int, int, float, float, float]] = []
    for first, second in ((6, 10), (6, 14), (10, 14)):
        result = wilcoxon(arrays[first], arrays[second], alternative="two-sided")
        pair_results.append(
            (
                first,
                second,
                float(result.statistic),
                float(result.pvalue),
                float(np.median(arrays[first] - arrays[second])),
            )
        )
    adjusted = _holm_adjust([row[3] for row in pair_results])
    for (first, second, statistic, p_value, difference), adjusted_p in zip(
        pair_results, adjusted, strict=True
    ):
        comparisons.append(
            {
                "comparison": f"run_{first}_vs_run_{second}",
                "test": "wilcoxon_two_sided_matched_subjects",
                "participant_count": len(subjects),
                "statistic": statistic,
                "unadjusted_p": p_value,
                "holm_adjusted_p": adjusted_p,
                "median_paired_difference_first_minus_second": difference,
            }
        )
    run_10_relative = arrays[10] - (arrays[6] + arrays[14]) / 2.0
    comparisons.append(
        {
            "comparison": "run_10_vs_mean_of_runs_6_and_14",
            "test": "descriptive_matched_profile",
            "participant_count": len(subjects),
            "statistic": "",
            "unadjusted_p": "",
            "holm_adjusted_p": "",
            "median_paired_difference_first_minus_second": float(
                np.median(run_10_relative)
            ),
            "q25_paired_difference": float(np.quantile(run_10_relative, 0.25)),
            "q75_paired_difference": float(np.quantile(run_10_relative, 0.75)),
            "run_10_lower_count": int(np.sum(run_10_relative < 0)),
            "run_10_higher_count": int(np.sum(run_10_relative > 0)),
            "tie_count": int(np.sum(run_10_relative == 0)),
            "run_10_strictly_lowest_of_three_count": int(
                np.sum((arrays[10] < arrays[6]) & (arrays[10] < arrays[14]))
            ),
        }
    )
    return summaries, comparisons


def class_recall_rows(
    fold_rows: Sequence[Mapping[str, str]],
    subject_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Describe fists/feet recall without changing class weighting."""
    fists = np.array([float(row["out_of_fold_fists_recall"]) for row in subject_rows])
    feet = np.array([float(row["out_of_fold_feet_recall"]) for row in subject_rows])
    test = wilcoxon(fists, feet, alternative="two-sided")
    output: list[dict[str, object]] = [
        {
            "scope": "all_runs_subject_out_of_fold",
            "participant_count": len(subject_rows),
            "median_fists_recall": float(np.median(fists)),
            "median_feet_recall": float(np.median(feet)),
            "median_feet_minus_fists_recall": float(np.median(feet - fists)),
            "fists_higher_count": int(np.sum(fists > feet)),
            "feet_higher_count": int(np.sum(feet > fists)),
            "tie_count": int(np.sum(feet == fists)),
            "paired_test": "wilcoxon_two_sided_subject_recalls",
            "paired_test_statistic": float(test.statistic),
            "paired_test_p": float(test.pvalue),
            "analysis_role": "post_hoc_class_asymmetry_diagnostic",
        }
    ]
    primary = primary_csp_rows(fold_rows)
    for run in RUNS:
        selected = [row for row in primary if int(row["test_run"]) == run]
        run_fists = np.array([float(row["fists_recall"]) for row in selected])
        run_feet = np.array([float(row["feet_recall"]) for row in selected])
        fists_correct = sum(int(row["true_fists_predicted_fists"]) for row in selected)
        fists_total = fists_correct + sum(
            int(row["true_fists_predicted_feet"]) for row in selected
        )
        feet_correct = sum(int(row["true_feet_predicted_feet"]) for row in selected)
        feet_total = feet_correct + sum(
            int(row["true_feet_predicted_fists"]) for row in selected
        )
        output.append(
            {
                "scope": f"held_out_run_{run}",
                "participant_count": len(selected),
                "median_fists_recall": float(np.median(run_fists)),
                "median_feet_recall": float(np.median(run_feet)),
                "median_feet_minus_fists_recall": float(np.median(run_feet - run_fists)),
                "pooled_fists_recall": fists_correct / fists_total,
                "pooled_feet_recall": feet_correct / feet_total,
                "fists_higher_count": int(np.sum(run_fists > run_feet)),
                "feet_higher_count": int(np.sum(run_feet > run_fists)),
                "tie_count": int(np.sum(run_feet == run_fists)),
                "paired_test": "descriptive_only",
                "paired_test_statistic": "",
                "paired_test_p": "",
                "analysis_role": "post_hoc_class_asymmetry_diagnostic",
            }
        )
    low = [
        row
        for row in subject_rows
        if float(row["historical_primary_mean_balanced_accuracy"]) <= 0.5
    ]
    low_fists = np.array([float(row["out_of_fold_fists_recall"]) for row in low])
    low_feet = np.array([float(row["out_of_fold_feet_recall"]) for row in low])
    output.append(
        {
            "scope": "subjects_with_mean_balanced_accuracy_at_or_below_0_5",
            "participant_count": len(low),
            "median_fists_recall": float(np.median(low_fists)),
            "median_feet_recall": float(np.median(low_feet)),
            "median_feet_minus_fists_recall": float(np.median(low_feet - low_fists)),
            "both_class_recalls_at_or_below_0_5_count": int(
                np.sum((low_fists <= 0.5) & (low_feet <= 0.5))
            ),
            "only_fists_recall_at_or_below_0_5_count": int(
                np.sum((low_fists <= 0.5) & (low_feet > 0.5))
            ),
            "only_feet_recall_at_or_below_0_5_count": int(
                np.sum((low_feet <= 0.5) & (low_fists > 0.5))
            ),
            "paired_test": "descriptive_only",
            "paired_test_statistic": "",
            "paired_test_p": "",
            "analysis_role": "post_hoc_poor_subject_class_pattern",
        }
    )
    return output


def participant_centered_correlation(
    x: Sequence[float], y: Sequence[float], subjects: Sequence[int]
) -> float:
    """Pearson correlation after removing each participant's three-fold mean."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    subject_array = np.asarray(subjects, dtype=int)
    centered_x = x_array.copy()
    centered_y = y_array.copy()
    for subject in np.unique(subject_array):
        selected = subject_array == subject
        centered_x[selected] -= np.mean(centered_x[selected])
        centered_y[selected] -= np.mean(centered_y[selected])
    denominator = np.sqrt(np.sum(centered_x**2) * np.sum(centered_y**2))
    if denominator == 0:
        raise ValueError("Participant-centered correlation has zero variance.")
    return float(np.sum(centered_x * centered_y) / denominator)


def within_subject_permutation_p(
    x: Sequence[float],
    y: Sequence[float],
    subjects: Sequence[int],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Test a centered association by permuting outcomes within participant."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    subject_array = np.asarray(subjects, dtype=int)
    observed = participant_centered_correlation(x_array, y_array, subject_array)
    indices = [np.flatnonzero(subject_array == subject) for subject in np.unique(subject_array)]
    if any(index.size != 3 for index in indices):
        raise ValueError("Within-subject permutation requires exactly three folds.")
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(resamples):
        permuted = y_array.copy()
        for index in indices:
            permuted[index] = permuted[rng.permutation(index)]
        statistic = participant_centered_correlation(x_array, permuted, subject_array)
        extreme += abs(statistic) >= abs(observed)
    return observed, (extreme + 1) / (resamples + 1)


def quality_relationship_rows(
    reliability_rows: Sequence[Mapping[str, object]],
    fold_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Relate existing label-independent QC burden to fixed decoder outcomes."""
    qc = np.array([float(row["task_qc_candidate_fraction"]) for row in reliability_rows])
    accuracy = np.array(
        [float(row["historical_primary_mean_balanced_accuracy"]) for row in reliability_rows]
    )
    ranges = np.array([float(row["run_balanced_accuracy_range"]) for row in reliability_rows])
    delta = np.array(
        [float(row["qc_minus_primary_balanced_accuracy"]) for row in reliability_rows]
    )
    qc_accuracy = spearmanr(qc, accuracy)
    qc_range = spearmanr(qc, ranges)
    output: list[dict[str, object]] = [
        {
            "relationship": "subject_task_qc_fraction_vs_mean_balanced_accuracy",
            "n": len(reliability_rows),
            "statistic": float(qc_accuracy.statistic),
            "p_value": float(qc_accuracy.pvalue),
            "median_effect_or_difference": "",
            "analysis_role": "post_hoc_qc_diagnostic",
        },
        {
            "relationship": "subject_task_qc_fraction_vs_run_accuracy_range",
            "n": len(reliability_rows),
            "statistic": float(qc_range.statistic),
            "p_value": float(qc_range.pvalue),
            "median_effect_or_difference": "",
            "analysis_role": "post_hoc_qc_diagnostic",
        },
        {
            "relationship": "qc_sensitivity_minus_primary_subject_accuracy",
            "n": len(reliability_rows),
            "statistic": "",
            "p_value": "",
            "median_effect_or_difference": float(np.median(delta)),
            "q25": float(np.quantile(delta, 0.25)),
            "q75": float(np.quantile(delta, 0.75)),
            "analysis_role": "already_generated_qc_sensitivity",
        },
    ]
    fold_qc: list[float] = []
    fold_accuracy: list[float] = []
    fold_subjects: list[int] = []
    reliability_lookup = {int(row["subject"]): row for row in reliability_rows}
    for row in primary_csp_rows(fold_rows):
        subject = int(row["subject"])
        run = int(row["test_run"])
        fold_qc.append(float(reliability_lookup[subject][f"task_qc_fraction_run_{run}"]))
        fold_accuracy.append(float(row["balanced_accuracy"]))
        fold_subjects.append(subject)
    output.append(
        {
            "relationship": "within_subject_run_qc_fraction_vs_fold_accuracy",
            "n": len(fold_qc),
            "statistic": participant_centered_correlation(
                fold_qc, fold_accuracy, fold_subjects
            ),
            "p_value": "",
            "median_effect_or_difference": "",
            "analysis_role": "post_hoc_descriptive_participant_centered_correlation",
        }
    )
    primary = primary_csp_rows(fold_rows)
    qc_folds = {
        (int(row["subject"]), int(row["test_run"])): row
        for row in fold_rows
        if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "True"
    }
    for run in RUNS:
        differences = np.array(
            [
                float(qc_folds[(int(row["subject"]), run)]["balanced_accuracy"])
                - float(row["balanced_accuracy"])
                for row in primary
                if int(row["test_run"]) == run
            ]
        )
        output.append(
            {
                "relationship": f"qc_sensitivity_minus_primary_run_{run}",
                "n": differences.size,
                "statistic": "",
                "p_value": "",
                "median_effect_or_difference": float(np.median(differences)),
                "q25": float(np.quantile(differences, 0.25)),
                "q75": float(np.quantile(differences, 0.75)),
                "analysis_role": "already_generated_qc_sensitivity",
            }
        )
    return output


def regularized_sensor_covariance(trials: np.ndarray) -> np.ndarray:
    """Estimate one label-blind SPD covariance from band-limited task trials.

    Parameters
    ----------
    trials
        Array shaped ``(trials, channels, samples)``. No class-label argument exists.
    """
    array = np.asarray(trials, dtype=float)
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 2 or array.shape[2] < 2:
        raise ValueError("Run-shift trials must have shape (trials, channels, samples).")
    if not np.isfinite(array).all():
        raise ValueError("Run-shift trials must be finite.")
    centered = array - np.mean(array, axis=2, keepdims=True)
    observations = centered.transpose(0, 2, 1).reshape(-1, array.shape[1])
    covariance = LedoitWolf(assume_centered=True).fit(observations).covariance_
    covariance = (covariance + covariance.T) / 2.0
    if np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise RuntimeError("Regularized run covariance is not positive definite.")
    return covariance


def affine_invariant_spd_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the affine-invariant Riemannian distance between two SPD matrices."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape or first_array.ndim != 2:
        raise ValueError("SPD covariance matrices must have the same square shape.")
    eigenvalues = eigvalsh(first_array, second_array, check_finite=True)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError("Generalized covariance eigenvalues must be positive and finite.")
    return float(np.sqrt(np.sum(np.log(eigenvalues) ** 2)))


def subject_run_shift_rows(
    subject: int,
    csp_task_data_volts: np.ndarray,
    runs: np.ndarray,
    source_hashes: Mapping[str, str],
) -> list[dict[str, object]]:
    """Measure class-blind covariance shift for each held-out run."""
    data = np.asarray(csp_task_data_volts, dtype=float)
    run_array = np.asarray(runs, dtype=int)
    if data.shape[0] != run_array.size or set(np.unique(run_array)) != set(RUNS):
        raise ValueError("Run-shift data and frozen run labels are inconsistent.")
    output: list[dict[str, object]] = []
    for test_run in RUNS:
        test_mask = run_array == test_run
        train_mask = ~test_mask
        train_covariance = regularized_sensor_covariance(data[train_mask])
        test_covariance = regularized_sensor_covariance(data[test_mask])
        training_runs = sorted(set(run_array[train_mask].tolist()))
        output.append(
            {
                "subject": subject,
                "test_run": test_run,
                "training_runs": "/".join(str(run) for run in training_runs),
                "affine_invariant_covariance_distance": affine_invariant_spd_distance(
                    train_covariance, test_covariance
                ),
                "train_trial_count": int(np.sum(train_mask)),
                "test_trial_count": int(np.sum(test_mask)),
                "train_observation_count": int(np.sum(train_mask) * data.shape[2]),
                "test_observation_count": int(np.sum(test_mask) * data.shape[2]),
                "channel_count": data.shape[1],
                "labels_used_for_distance": False,
                "covariance_estimator": "LedoitWolf_assume_centered_after_trial_demeaning",
                "trace_normalized": False,
                "source_edf_count": len(source_hashes),
                "analysis_role": "post_hoc_label_independent_run_shift",
            }
        )
    return output


def run_shift_summaries(
    shift_rows: Sequence[Mapping[str, object]],
    fold_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Join fixed accuracies to shift distances and summarize their association."""
    accuracy_lookup = {
        (int(row["subject"]), int(row["test_run"])): float(row["balanced_accuracy"])
        for row in primary_csp_rows(fold_rows)
    }
    joined: list[dict[str, object]] = []
    for row in sorted(shift_rows, key=lambda item: (int(item["subject"]), int(item["test_run"]))):
        key = (int(row["subject"]), int(row["test_run"]))
        joined.append({**row, "held_out_balanced_accuracy": accuracy_lookup[key]})
    x = [float(row["affine_invariant_covariance_distance"]) for row in joined]
    y = [float(row["held_out_balanced_accuracy"]) for row in joined]
    subjects = [int(row["subject"]) for row in joined]
    policy = config["label_independent_run_shift"]
    centered_r, permutation_p = within_subject_permutation_p(
        x,
        y,
        subjects,
        resamples=10000,
        seed=int(policy["permutation_seed"]),
    )
    subject_mean_x: list[float] = []
    subject_mean_y: list[float] = []
    for subject in sorted(set(subjects)):
        selected = [row for row in joined if int(row["subject"]) == subject]
        subject_mean_x.append(
            float(np.mean([float(row["affine_invariant_covariance_distance"]) for row in selected]))
        )
        subject_mean_y.append(
            float(np.mean([float(row["held_out_balanced_accuracy"]) for row in selected]))
        )
    between = spearmanr(subject_mean_x, subject_mean_y)
    relationships = [
        {
            "relationship": "within_subject_run_shift_vs_fold_accuracy",
            "n": len(joined),
            "independent_unit_count": len(set(subjects)),
            "statistic": centered_r,
            "statistic_type": "participant_centered_Pearson_r",
            "p_value": permutation_p,
            "p_value_method": "10000_accuracy_permutations_within_subject",
            "analysis_role": "post_hoc_exploratory_noncausal",
        },
        {
            "relationship": "subject_mean_run_shift_vs_subject_mean_accuracy",
            "n": len(subject_mean_x),
            "independent_unit_count": len(subject_mean_x),
            "statistic": float(between.statistic),
            "statistic_type": "Spearman_rho",
            "p_value": float(between.pvalue),
            "p_value_method": "two_sided_asymptotic",
            "analysis_role": "post_hoc_exploratory_noncausal",
        },
    ]
    return joined, relationships


def run_shift_run_summary_rows(
    joined_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Describe label-independent shift separately for each held-out run."""
    output: list[dict[str, object]] = []
    for run in RUNS:
        values = np.array(
            [
                float(row["affine_invariant_covariance_distance"])
                for row in joined_rows
                if int(row["test_run"]) == run
            ]
        )
        output.append(
            {
                "test_run": run,
                "participant_count": values.size,
                "median_distance": float(np.median(values)),
                "q25_distance": float(np.quantile(values, 0.25)),
                "q75_distance": float(np.quantile(values, 0.75)),
                "minimum_distance": float(np.min(values)),
                "maximum_distance": float(np.max(values)),
                "analysis_role": "post_hoc_label_independent_run_shift",
            }
        )
    return output


def historical_relationship_rows(
    reliability_rows: Sequence[Mapping[str, object]],
    context_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Evaluate only the small historical relationship set frozen for this study."""
    accuracy = {
        int(row["subject"]): float(row["historical_primary_mean_balanced_accuracy"])
        for row in reliability_rows
    }
    context = {int(row["subject"]): row for row in context_rows}
    if set(accuracy) != set(context):
        raise RuntimeError("Historical context and reliability subjects differ.")

    def correlation(name: str, field: str) -> dict[str, object]:
        selected = [
            subject for subject in sorted(accuracy) if context[subject][field] != ""
        ]
        result = spearmanr(
            [float(context[subject][field]) for subject in selected],
            [accuracy[subject] for subject in selected],
        )
        return {
            "relationship": name,
            "available_subject_count": len(selected),
            "statistic": float(result.statistic),
            "statistic_type": "Spearman_rho",
            "two_sided_unadjusted_p": float(result.pvalue),
            "available_group_median_accuracy": "",
            "unavailable_subject_count": "",
            "unavailable_group_median_accuracy": "",
            "analysis_role": "predefined_post_evaluation_exploratory_only",
        }

    output = [
        correlation("C3_fixed_fists_12_13_hz_ERD", "c3_fixed_fists_erd_median_percent"),
        correlation("C4_fixed_fists_12_13_hz_ERD", "c4_fixed_fists_erd_median_percent"),
        correlation(
            "historical_central_peak_frequency",
            "historical_central_peak_frequency_hz",
        ),
        correlation("held_out_IMF_peak_span", "imf_three_fold_peak_span_hz"),
    ]
    available = [
        subject
        for subject in sorted(accuracy)
        if context[subject]["historical_central_peak_available"] == "True"
    ]
    unavailable = sorted(set(accuracy) - set(available))
    output.append(
        {
            "relationship": "historical_central_peak_availability",
            "available_subject_count": len(available),
            "statistic": "",
            "statistic_type": "descriptive_group_medians",
            "two_sided_unadjusted_p": "",
            "available_group_median_accuracy": float(
                np.median([accuracy[subject] for subject in available])
            ),
            "unavailable_subject_count": len(unavailable),
            "unavailable_group_median_accuracy": float(
                np.median([accuracy[subject] for subject in unavailable])
            ),
            "analysis_role": "predefined_post_evaluation_exploratory_only",
        }
    )
    return output
