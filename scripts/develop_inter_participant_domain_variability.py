"""Create the development-only tangent reference and source geometry."""
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path
import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from eeg_project.cross_subject_decoding import file_sha256, load_cross_subject_config, load_cross_subject_data
from eeg_project.inter_participant_domain_variability import fit_development_tangent, participant_geometry, subject_run_geometry
from eeg_project.riemannian_decoding import covariance_data_from_dataset

ASSETS = PROJECT_ROOT / "docs" / "assets"; PREFIX = "inter_participant_domain_variability_development_"

def write_csv(rows, path):
    fields = list(rows[0]);
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)

def main():
    if subprocess.run(["git", "merge-base", "--is-ancestor", "07af058", "HEAD"], cwd=PROJECT_ROOT).returncode: raise RuntimeError("Initial geometry freeze is not in history.")
    mne.set_log_level("ERROR"); config = load_cross_subject_config()
    datasets = [covariance_data_from_dataset(load_cross_subject_data(subject, config)) for subject in range(1, 21)]
    tangent = fit_development_tangent(datasets)
    reference_path = ASSETS / f"{PREFIX}tangent_reference.npy"; np.save(reference_path, tangent.reference_, allow_pickle=False)
    participant_rows=[]; run_rows=[]; source_midpoints=[]; source_directions=[]
    for data in datasets:
        runs = subject_run_geometry(data, tangent)
        midpoint, direction = participant_geometry(runs); source_midpoints.append(midpoint); source_directions.append(direction)
        participant_rows.append({"subject": data.subject, "midpoint_norm": float(np.linalg.norm(midpoint)), "task_direction_magnitude": float(np.linalg.norm(direction))})
        for row in runs: run_rows.append({"subject":row.subject,"run":row.run,"fists_trials":row.fists_trials,"feet_trials":row.feet_trials,"midpoint_norm":float(np.linalg.norm(row.midpoint)),"task_direction_magnitude":float(np.linalg.norm(row.direction))})
    source = np.stack([np.mean(source_midpoints, axis=0), np.mean(source_directions, axis=0)])
    source_path = ASSETS / f"{PREFIX}source_vectors.npy"; np.save(source_path, source, allow_pickle=False)
    write_csv(participant_rows, ASSETS / f"{PREFIX}participant_geometry.csv"); write_csv(run_rows, ASSETS / f"{PREFIX}run_geometry.csv")
    source_hashes={identity.source_file: identity.source_sha256 for data in datasets for identity in data.identities}
    artifacts={p.name:file_sha256(p) for p in (reference_path,source_path,ASSETS/f"{PREFIX}participant_geometry.csv",ASSETS/f"{PREFIX}run_geometry.csv")}
    payload={"schema_version":1,"stage":"development_geometry_complete","subjects":list(range(1,21)),"trial_count":900,"reference_fit_evaluation_subjects_used":False,"source_summary":"equal_participant_mean_of_three_run_vectors","source_hashes":dict(sorted(source_hashes.items())),"artifact_sha256":artifacts}
    (ASSETS/f"{PREFIX}metadata.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print("Development tangent reference fitted from 20 subjects / 900 covariances.")
if __name__ == "__main__": main()
