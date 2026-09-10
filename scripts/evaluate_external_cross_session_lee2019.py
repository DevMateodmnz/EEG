"""Evaluate the frozen participant-specific decoder across Lee2019 sessions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_project.external_cross_session_lee2019 import (bootstrap_median, evaluate_subject, load_study_config, load_subject_data, paired_summary, sign_test_above_chance, sha256_file, validate_fit_audits)
from eeg_project.unilateral_generalization import load_parent_decoder_config


PREFIX = "external_cross_session_lee2019_"
ASSETS = ROOT / "docs" / "assets"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty artifact: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def summary_rows(rows: Sequence[Mapping[str, object]], model: str, seed: int, resamples: int) -> dict[str, object]:
    values = np.asarray([float(row["primary_mean_cross_session_balanced_accuracy"]) for row in rows if row["model"] == model], dtype=float)
    return {"model": model, "participant_count": int(values.size), "directional_evaluations": int(values.size * 2), "median_balanced_accuracy": float(np.median(values)), "iqr_balanced_accuracy": [float(np.quantile(values, .25)), float(np.quantile(values, .75))], "mean_balanced_accuracy": float(np.mean(values)), "sd_balanced_accuracy": float(np.std(values, ddof=1)), "range_balanced_accuracy": [float(np.min(values)), float(np.max(values))], "bootstrap_median_95": list(bootstrap_median(values, resamples, seed)), "chance_test": sign_test_above_chance(values), "performance_bins": {"lt_0_5": int(np.sum(values < .5)), "0_5_to_lt_0_6": int(np.sum((values >= .5) & (values < .6))), "0_6_to_lt_0_7": int(np.sum((values >= .6) & (values < .7))), "0_7_to_lt_0_8": int(np.sum((values >= .7) & (values < .8))), "ge_0_8": int(np.sum(values >= .8))}}


def direction_summary(rows: Sequence[Mapping[str, object]], model: str, train_session: str) -> dict[str, object]:
    values = np.asarray([float(row["balanced_accuracy"]) for row in rows if row["model"] == model and row["train_session"] == train_session], dtype=float)
    if not values.size:
        raise ValueError("A frozen direction has no scores.")
    return {"model": model, "train_session": train_session, "test_session": "2" if train_session == "1" else "1", "participant_count": int(values.size), "median_balanced_accuracy": float(np.median(values)), "iqr_balanced_accuracy": [float(np.quantile(values, .25)), float(np.quantile(values, .75))], "mean_balanced_accuracy": float(np.mean(values)), "sd_balanced_accuracy": float(np.std(values, ddof=1)), "range_balanced_accuracy": [float(np.min(values)), float(np.max(values))]}


def holm_two(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values); adjusted = [0.0] * len(p_values); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * float(p_values[index])))
        adjusted[int(index)] = running
    return adjusted


def heterogeneity(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    csp = [row for row in rows if row["model"] == "csp_4_empirical"]
    both = sum(float(row["session1_to_session2_balanced_accuracy"]) > .5 and float(row["session2_to_session1_balanced_accuracy"]) > .5 for row in csp)
    neither = sum(float(row["session1_to_session2_balanced_accuracy"]) <= .5 and float(row["session2_to_session1_balanced_accuracy"]) <= .5 for row in csp)
    return {"above_chance_both_directions": int(both), "above_chance_exactly_one_direction": int(len(csp) - both - neither), "above_chance_neither_direction": int(neither)}


def validate_evaluation_rows(predictions: Sequence[Mapping[str, object]], directions: Sequence[Mapping[str, object]], participants: Sequence[Mapping[str, object]], audits: Sequence[Mapping[str, object]]) -> None:
    """Validate completed held-out records before any cohort inference is written."""
    validate_fit_audits(audits)
    expected = {(str(subject), model, source, target) for subject in {int(row["subject"]) for row in participants} for model in ("spectral_baseline_6", "csp_4_empirical") for source, target in (("1", "2"), ("2", "1"))}
    observed = {(str(row["subject"]), str(row["model"]), str(row["train_session"]), str(row["test_session"])) for row in directions}
    if observed != expected or len(directions) != len(expected):
        raise RuntimeError("External directional evaluation schema is incomplete or duplicated.")
    prediction_keys: set[tuple[object, ...]] = set()
    for row in predictions:
        if row["true_label"] not in (0, 1) or row["predicted_label"] not in (0, 1):
            raise RuntimeError("External predictions contain a non-frozen class.")
        key = (row["subject"], row["model"], row["train_session"], row["test_session"], row["trial_key"])
        if key in prediction_keys:
            raise RuntimeError("An external retained trial has duplicate held-out predictions.")
        prediction_keys.add(key)
    for row in participants:
        scores = [float(item["balanced_accuracy"]) for item in directions if item["subject"] == row["subject"] and item["model"] == row["model"]]
        if len(scores) != 2 or not np.isclose(float(row["primary_mean_cross_session_balanced_accuracy"]), np.mean(scores)):
            raise RuntimeError("Participant primary score is not the mean of its two frozen directions.")


def physionet_unilateral_scores() -> np.ndarray:
    path = ASSETS / "unilateral_motor_imagery_generalization_subject_scores.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scores = np.asarray([float(row["primary_mean_fold_balanced_accuracy"]) for row in rows if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"], dtype=float)
    if scores.size != 86:
        raise RuntimeError("Frozen PhysioNet unilateral CSP context table is incomplete.")
    return scores


def figures(subjects: Sequence[Mapping[str, object]], output: Path) -> None:
    csp=np.array([float(row["primary_mean_cross_session_balanced_accuracy"]) for row in subjects if row["model"]=="csp_4_empirical"]); baseline=np.array([float(row["primary_mean_cross_session_balanced_accuracy"]) for row in subjects if row["model"]=="spectral_baseline_6"])
    fig,ax=plt.subplots(figsize=(7,4.5));ax.boxplot([baseline,csp],tick_labels=["Spectral + LDA","CSP + LDA"],showfliers=False);rng=np.random.default_rng(20260909)
    for n,values in enumerate((baseline,csp),1): ax.scatter(n+rng.uniform(-.09,.09,values.size),values,s=16,alpha=.6)
    ax.axhline(.5,color="black",linestyle="--",linewidth=1);ax.set(ylabel="Mean cross-session balanced accuracy",ylim=(0,1.02));fig.tight_layout();fig.savefig(output/f"{PREFIX}score_distribution.png",dpi=180);plt.close(fig)
    for name,left,right,xlabel,ylabel in (("paired_baseline",baseline,csp,"Spectral + LDA","CSP + LDA"),("direction",np.array([float(row["session1_to_session2_balanced_accuracy"]) for row in subjects if row["model"]=="csp_4_empirical"]),np.array([float(row["session2_to_session1_balanced_accuracy"]) for row in subjects if row["model"]=="csp_4_empirical"]),"Session 1 → 2 CSP","Session 2 → 1 CSP")):
        fig,ax=plt.subplots(figsize=(5.2,5));ax.scatter(left,right,s=20,alpha=.65);limits=(min(left.min(),right.min())-.03,max(left.max(),right.max())+.03);ax.plot(limits,limits,"k--",linewidth=1);ax.set(xlabel=xlabel,ylabel=ylabel,xlim=limits,ylim=limits);fig.tight_layout();fig.savefig(output/f"{PREFIX}{name}.png",dpi=180);plt.close(fig)
    direction_12=np.array([float(row["session1_to_session2_balanced_accuracy"]) for row in subjects if row["model"]=="csp_4_empirical"]);direction_21=np.array([float(row["session2_to_session1_balanced_accuracy"]) for row in subjects if row["model"]=="csp_4_empirical"])
    fig,ax=plt.subplots(figsize=(6.2,4.8));colors=np.where((direction_12>.5)&(direction_21>.5),"#1b9e77",np.where((direction_12>.5)|(direction_21>.5),"#d95f02","#7570b3"));ax.scatter(direction_12,direction_21,s=28,c=colors,alpha=.75);ax.axvline(.5,color="black",linestyle="--",linewidth=1);ax.axhline(.5,color="black",linestyle="--",linewidth=1);ax.set(xlabel="Session 1 → 2 CSP balanced accuracy",ylabel="Session 2 → 1 CSP balanced accuracy",xlim=(0,1),ylim=(0,1));fig.tight_layout();fig.savefig(output/f"{PREFIX}heterogeneity_direction_matrix.png",dpi=180);plt.close(fig)
    historical=physionet_unilateral_scores();fig,ax=plt.subplots(figsize=(6.8,4.6));ax.boxplot([historical,csp],tick_labels=["PhysioNet\nwithin-cohort cross-run","Lee2019\nexternal cross-session"],showfliers=False);rng=np.random.default_rng(20260909);ax.scatter(1+rng.uniform(-.09,.09,historical.size),historical,s=12,alpha=.45);ax.scatter(2+rng.uniform(-.09,.09,csp.size),csp,s=14,alpha=.55);ax.axhline(.5,color="black",linestyle="--",linewidth=1);ax.set(ylabel="Participant balanced accuracy",ylim=(0,1.02));fig.tight_layout();fig.savefig(output/f"{PREFIX}physionet_context.png",dpi=180);plt.close(fig)


def main() -> None:
    arguments=args(); config=load_study_config(); unilateral=json.loads((ROOT / config["parent_unilateral_study"]["config"]).read_text()); parent=load_parent_decoder_config(unilateral)
    candidates=[1,2] if arguments.smoke else config["external_dataset"]["candidate_subjects"]
    output=(ROOT / "outputs" / "external_cross_session_lee2019" / "smoke") if arguments.smoke else arguments.output_dir; output.mkdir(parents=True,exist_ok=True)
    predictions=[]; directions=[]; audits=[]; participants=[]; sources=[]; exclusions=[]; eligibility=[]
    for subject in candidates:
        try:
            data=load_subject_data(subject,config,arguments.data_dir)
            sources.extend(data.source_manifest)
            eligibility.append({"subject": subject, "technical_eligibility": "eligible", "reason": "all_frozen_source_and_adapter_checks_passed"})
            for model in config["inherited_scientific_config"]["models"]:
                prediction, direction, audit, participant=evaluate_subject(data,parent,config,model)
                predictions.extend(prediction);directions.extend(direction);audits.extend(audit);participants.append(participant)
            print(f"completed subject {subject}", flush=True)
        except Exception as error:
            exclusion={"subject":subject,"technical_eligibility":"excluded","reason":"unreadable_authoritative_source_or_adapter_validation","detail":str(error)}
            exclusions.append(exclusion); eligibility.append(exclusion)
            print(f"excluded subject {subject}: {error}", flush=True)
    if not participants:
        raise RuntimeError("No Lee2019 participant passed technical validation.")
    validate_evaluation_rows(predictions,directions,participants,audits)
    write_csv(sources,output/f"{PREFIX}source_manifest.csv");write_csv(predictions,output/f"{PREFIX}predictions.csv");write_csv(directions,output/f"{PREFIX}direction_scores.csv");write_csv(audits,output/f"{PREFIX}fit_audit.csv");write_csv(participants,output/f"{PREFIX}participant_scores.csv");write_csv(eligibility,output/f"{PREFIX}technical_eligibility.csv")
    if exclusions: write_csv(exclusions,output/f"{PREFIX}technical_exclusions.csv")
    if arguments.smoke:return
    seed=int(config["inference"]["bootstrap_seed"]);resamples=int(config["inference"]["bootstrap_resamples"]);models=config["inherited_scientific_config"]["models"]
    summaries=[summary_rows(participants,model,seed,resamples) for model in models]
    baseline={int(row["subject"]):float(row["primary_mean_cross_session_balanced_accuracy"]) for row in participants if row["model"]=="spectral_baseline_6"};csp={int(row["subject"]):float(row["primary_mean_cross_session_balanced_accuracy"]) for row in participants if row["model"]=="csp_4_empirical"};common=sorted(set(baseline)&set(csp))
    direction_12=[float(row["balanced_accuracy"]) for row in directions if row["model"]=="csp_4_empirical" and row["train_session"]=="1"];direction_21=[float(row["balanced_accuracy"]) for row in directions if row["model"]=="csp_4_empirical" and row["train_session"]=="2"]
    comparison=paired_summary([csp[s] for s in common],[baseline[s] for s in common],seed,resamples);direction_test=paired_summary(direction_12,direction_21,seed,resamples)
    adjusted=holm_two([float(comparison["wilcoxon_p"]),float(direction_test["wilcoxon_p"])]);comparison["holm_adjusted_p"]=adjusted[0];direction_test["holm_adjusted_p"]=adjusted[1]
    direction_summaries=[direction_summary(directions,model,source) for model in models for source in ("1","2")]
    summary={"schema_version":1,"study_id":config["study_id"],"candidate_subject_count":len(candidates),"eligible_subject_count":len(common),"technical_exclusion_count":len(exclusions),"model_summaries":summaries,"direction_summaries":direction_summaries,"csp_heterogeneity":heterogeneity(participants),"comparisons":{"csp_minus_baseline":comparison,"session1_to_2_minus_session2_to_1":direction_test},"config_sha256":sha256_file(ROOT/"config/external_cross_session_lee2019.json"),"parent_unilateral":config["parent_unilateral_study"],"physionet_unilateral_csp_median_balanced_accuracy":0.5491071428571428}
    (output/f"{PREFIX}summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    provenance={"schema_version":1,"study_id":config["study_id"],"config_sha256":summary["config_sha256"],"parent_unilateral":config["parent_unilateral_study"],"source_manifest_sha256":sha256_file(output/f"{PREFIX}source_manifest.csv"),"python":platform.python_version(),"packages":{"mne":mne.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__,"moabb":"1.7.1"},"seed":seed}
    (output/f"{PREFIX}provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    write_csv([{"dataset":"PhysioNet EEGMMIDB","design":"within-cohort unilateral cross-run","participant_count":86,"csp_median_balanced_accuracy":float(np.median(physionet_unilateral_scores()))},{"dataset":"Lee2019/OpenBMI","design":"external participant-specific cross-session","participant_count":len(common),"csp_median_balanced_accuracy":float(np.median([csp[s] for s in common]))}],output/f"{PREFIX}cross_dataset_context.csv")
    figures(participants,output)


if __name__ == "__main__": main()
