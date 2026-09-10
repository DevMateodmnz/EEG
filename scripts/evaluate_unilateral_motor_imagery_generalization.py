"""Run the frozen-architecture left-versus-right imagery generalization study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY
from eeg_project.unilateral_generalization import (
    acquire_unilateral_edfs, bootstrap_median, evaluate_unilateral, file_sha256,
    load_parent_decoder_config, load_study_config, load_unilateral_dataset,
    paired_summary, sign_test_above_chance,
)


ASSETS = ROOT / "docs" / "assets"
PREFIX = "unilateral_motor_imagery_generalization"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run only frozen smoke subject 1 to outputs/.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    return parser.parse_args()


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows: raise ValueError(f"Cannot serialize empty artifact: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def parent_subject_scores() -> dict[int, float]:
    path = ASSETS / "within_subject_decoder_evaluation_subject_scores.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {int(row["subject"]): float(row["primary_mean_fold_balanced_accuracy"]) for row in rows if row["model"] == "csp_4_empirical" and row["qc_sensitivity"] == "False"}
    if len(output) != 86: raise RuntimeError("Frozen bilateral CSP subject table is incomplete.")
    return output


def model_summary(rows: Sequence[Mapping[str, object]], model: str, seed: int, resamples: int) -> dict[str, object]:
    scores = np.array([float(row["primary_mean_fold_balanced_accuracy"]) for row in rows if row["model"] == model and not row["qc_sensitivity"]])
    if not scores.size: raise RuntimeError(f"No primary rows for {model}.")
    return {"model": model, "participant_count": int(scores.size), "fold_count": int(scores.size * 3), "median_balanced_accuracy": float(np.median(scores)), "iqr_balanced_accuracy": [float(np.quantile(scores,.25)), float(np.quantile(scores,.75))], "mean_balanced_accuracy": float(np.mean(scores)), "sd_balanced_accuracy": float(np.std(scores,ddof=1)), "range_balanced_accuracy": [float(np.min(scores)),float(np.max(scores))], "bootstrap_median_95": list(bootstrap_median(scores,resamples,seed)), "performance_bins": {"lt_0_5":int(np.sum(scores<.5)), "0_5_to_lt_0_6":int(np.sum((scores>=.5)&(scores<.6))), "0_6_to_lt_0_7":int(np.sum((scores>=.6)&(scores<.7))), "0_7_to_lt_0_8":int(np.sum((scores>=.7)&(scores<.8))), "ge_0_8":int(np.sum(scores>=.8))}, "chance_test": sign_test_above_chance(scores)}


def holm_two(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values); adjusted=[0.0]*len(p_values); running=0.0
    for rank,index in enumerate(order):
        running=max(running, min(1.0, (len(p_values)-rank)*p_values[index])); adjusted[int(index)]=running
    return adjusted


def figures(subject_rows: Sequence[Mapping[str, object]], comparison_rows: Sequence[Mapping[str, object]], output: Path) -> None:
    primary=[row for row in subject_rows if not row["qc_sensitivity"]]
    by_model=[[float(row["primary_mean_fold_balanced_accuracy"]) for row in primary if row["model"]==model] for model in ("spectral_baseline_6","csp_4_empirical")]
    fig, ax=plt.subplots(figsize=(7,4.6)); ax.boxplot(by_model,tick_labels=["Spectral + LDA","CSP + LDA"],showfliers=False); rng=np.random.default_rng(20260908)
    for index,values in enumerate(by_model,1): ax.scatter(index+rng.uniform(-.09,.09,len(values)),values,s=15,alpha=.55)
    ax.axhline(.5,color="black",linestyle="--",linewidth=1); ax.set(ylabel="Mean held-out-run balanced accuracy",ylim=(.1,1.02)); fig.tight_layout();fig.savefig(output/f"{PREFIX}_score_distribution.png",dpi=180);plt.close(fig)
    baseline=np.array([float(row["baseline_balanced_accuracy"]) for row in comparison_rows]);csp=np.array([float(row["unilateral_csp_balanced_accuracy"]) for row in comparison_rows]);bilateral=np.array([float(row["bilateral_csp_balanced_accuracy"]) for row in comparison_rows])
    for left,right,name,xlabel,ylabel in ((baseline,csp,"paired_baseline","Spectral + LDA","CSP + LDA"),(bilateral,csp,"paired_bilateral","Bilateral CSP + LDA","Unilateral CSP + LDA")):
        fig,ax=plt.subplots(figsize=(5.4,5.0)); ax.scatter(left,right,s=20,alpha=.65); limits=(min(left.min(),right.min())-.03,max(left.max(),right.max())+.03);ax.plot(limits,limits,"k--",linewidth=1);ax.set(xlabel=xlabel,ylabel=ylabel,xlim=limits,ylim=limits);fig.tight_layout();fig.savefig(output/f"{PREFIX}_{name}.png",dpi=180);plt.close(fig)
    csp_rows=[row for row in primary if row["model"]=="csp_4_empirical"]; ranges=[float(row["fold_balanced_accuracy_range"]) for row in csp_rows]
    fig,ax=plt.subplots(figsize=(6.6,4.2)); ax.hist(ranges,bins=np.linspace(0,1,11),color="#4575b4",edgecolor="white");ax.set(xlabel="Participant held-out-run balanced-accuracy range",ylabel="Participants",xlim=(0,1));fig.tight_layout();fig.savefig(output/f"{PREFIX}_fold_ranges.png",dpi=180);plt.close(fig)


def main() -> None:
    args=arguments(); config=load_study_config(); parent=load_parent_decoder_config(config)
    final=json.loads((ROOT/config["parent_bilateral_decoder"]["config"]).read_text())
    subjects=[config["cohort"]["smoke_subject"]] if args.smoke else list(final["cohorts"]["decoder_evaluation_eligible_subjects"])
    output=(ROOT/"outputs"/PREFIX/"smoke") if args.smoke else args.output_dir; output.mkdir(parents=True,exist_ok=True)
    raw_hashes=acquire_unilateral_edfs(subjects,args.data_dir)
    predictions=[]; folds=[]; audits=[]; subject_rows=[]
    for subject in subjects:
        dataset=load_unilateral_dataset(subject,parent,args.data_dir)
        for model in config["frozen_architecture"]["parent_models"]:
            for qc in (False,True):
                prediction,fold,audit,summary=evaluate_unilateral(dataset,parent,model,qc)
                predictions.extend(prediction);folds.extend(fold);audits.extend(audit);subject_rows.append(summary)
        print(f"completed subject {subject}")
    write_csv(predictions,output/f"{PREFIX}_predictions.csv");write_csv(folds,output/f"{PREFIX}_fold_scores.csv");write_csv(audits,output/f"{PREFIX}_fit_audit.csv");write_csv(subject_rows,output/f"{PREFIX}_subject_scores.csv")
    if args.smoke: return
    seed=int(config["inference"]["bootstrap_seed"]);resamples=int(config["inference"]["bootstrap_resamples"])
    summaries=[model_summary(subject_rows,model,seed,resamples) for model in config["frozen_architecture"]["parent_models"]]
    primary=[row for row in subject_rows if not row["qc_sensitivity"]]; baseline={int(row["subject"]):float(row["primary_mean_fold_balanced_accuracy"]) for row in primary if row["model"]=="spectral_baseline_6"};csp={int(row["subject"]):float(row["primary_mean_fold_balanced_accuracy"]) for row in primary if row["model"]=="csp_4_empirical"};bilateral=parent_subject_scores(); matched=sorted(set(baseline)&set(csp)&set(bilateral))
    comparisons=[{"subject":s,"baseline_balanced_accuracy":baseline[s],"unilateral_csp_balanced_accuracy":csp[s],"bilateral_csp_balanced_accuracy":bilateral[s]} for s in matched]
    baseline_test=paired_summary([csp[s] for s in matched],[baseline[s] for s in matched],seed,resamples); task_test=paired_summary([csp[s] for s in matched],[bilateral[s] for s in matched],seed,resamples); adjusted=holm_two([float(baseline_test["wilcoxon_p"]),float(task_test["wilcoxon_p"])]);baseline_test["holm_adjusted_p"]=adjusted[0];task_test["holm_adjusted_p"]=adjusted[1]
    write_csv(comparisons,output/f"{PREFIX}_matched_comparisons.csv")
    summary={"schema_version":1,"study_id":config["study_id"],"classification":config["classification"],"unilateral_eligible_subject_count":len(subjects),"matched_bilateral_subject_count":len(matched),"model_summaries":summaries,"comparisons":{"csp_minus_baseline":baseline_test,"unilateral_minus_bilateral_csp":task_test},"median_csp_fold_range":float(np.median([float(row["fold_balanced_accuracy_range"]) for row in primary if row["model"]=="csp_4_empirical"])),"config_sha256":file_sha256(ROOT/"config/unilateral_motor_imagery_generalization.json"),"parent_bilateral_config_sha256":config["parent_bilateral_decoder"]["sha256"]}
    (output/f"{PREFIX}_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    provenance={"schema_version":1,"study_id":config["study_id"],"config_sha256":summary["config_sha256"],"parent_bilateral":config["parent_bilateral_decoder"],"raw_source_hashes":raw_hashes,"python":platform.python_version(),"packages":{"mne":mne.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__},"bootstrap":{"seed":seed,"resamples":resamples},"raw_data":config["raw_data"]}
    (output/f"{PREFIX}_provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    figures(subject_rows,comparisons,output)


if __name__=="__main__": main()
